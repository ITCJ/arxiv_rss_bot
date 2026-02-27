# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-27 06:36:00 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference](https://arxiv.org/abs/2602.22868)

**Authors**: Yushi Ye, Feng Hong, Huangjie Zheng, Xu Chen, Zhiyong Chen, Yanfeng Wang, Jiangchao Yao  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2602.22868v1  

#### Abstract
Diffusion Large Language Models (DLLMs) promise fast non-autoregressive inference but suffer a severe quality-speed trade-off in parallel decoding. This stems from the ''combinatorial contradiction'' phenomenon, where parallel tokens form semantically inconsistent combinations. We address this by in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Rejection Mixing: Fast Semantic Propagation of Mask Tokens for Efficient DLLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
该论文针对 **Diffusion Large Language Models (DLLMs)** 在并行解码时面临的严重 **质量-速度权衡**（quality-speed trade-off）问题。  
其根本原因被归结为 **“组合矛盾”（combinatorial contradiction）**：在同一个解码步骤中，并行生成的多个 token 缺乏相互协调，导致语义上不一致的结果（例如，“high house” 而非 “full house”）。

传统方法由于采用纯离散（discrete）的解码方式，在多 token 同时采样时无法感知彼此依赖关系，从而引发错误。

---

### ✅ 提出了什么新方法或新思路
作者提出 **ReMix（Rejection Mixing）**，一种无需重新训练的高效解码框架，核心思想是：

- 引入一个 **连续混合状态（Continuous Mixing State, C）**，作为从 `[MASK]` 到最终语义 token 的中间过渡。
- 允许未确定的 token 在连续空间中通过 **Mixing Rule** 迭代更新其嵌入表示，保留上下文依赖信息。
- 设计 **Rejection Rule**：当某位置的输出分布不稳定（如前后步差异过大），则将其重置回 `[MASK]` 状态以避免错误传播。

这一机制实现了：
- **连续空间中的语义精炼**（Semantic Propagation）
- **动态修正不确定性**
- **提升并行解码的一致性和稳定性**

---

### ✅ 相比现有方法的优势
| 对比维度 | ReMix 优势 |
|--------|-----------|
| **是否需要训练** | ❌ 无需额外训练（training-free），可直接应用于已有 DLLM |
| **效率提升** | 实现 **2–8× 推理加速**，显著减少 decoding steps 和 latency |
| **生成质量** | 不仅无损，反而在多数任务上 **提升准确率**（accuracy gain） |
| **通用性** | 支持语言与多模态任务（LLaDA、MMaDA），适用于不同 generation/block 长度 |
| **实现复杂度** | 轻量级模块，计算开销低（仅增加约 9.12% runtime） |

相比 WINO、Fast-dLLM、APD 等方法，ReMix 在保持高 throughput 的同时实现了更优的 accuracy-throughput trade-off。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### **语言领域（基于 LLaDA-8B-Instruct）**
- 数学推理：GSM8K, MATH-500
- 代码生成：HumanEval, MBPP
- 逻辑推理：Countdown, Sudoku
- 常识推理：ARC-E, ARC-C

#### **多模态领域（基于 MMaDA-8B-MixCoT）**
- 图像描述：Flickr30k-lite
- 图表理解：AI2D-lite
- 数学视觉推理：MathVision-mini, MathVista-mini
- 多学科理解：MMMU-val, ScienceQA-IMG

---

### ⚙️ 实验设置和评估指标

| 项目 | 设置说明 |
|------|---------|
| **评估模式** | Zero-shot（除 Sudoku 为 4-shot） |
| **评估指标** | - 准确率（Accuracy）用于大多数任务<br>- CIDEr 用于 Flickr30k<br>- 使用 GPT-4o-mini 作为评判模型对多模态答案打分 |
| **生成参数** | - Generation length: 默认 256<br>- Block length: 128（半自回归策略）<br>- Confidence threshold $ \tau_{\text{conf}} = 0.8 $<br>- Mixing ratio $ \beta \in \{0.4, 0.5, 0.6\} $<br>- Rejection threshold $ \tau_{\text{rej}} \in [0.1, 0.4] $ |
| **硬件平台** | 8× NVIDIA A100 / RTX 3090 GPU |

---

### 🔁 基线方法对比
- **LLaDA / MMaDA**：标准的 diffusion-based LLM 解码流程
- **WINO**：支持可撤销解码（revocable decoding）
- **Fast-dLLM**：结合 KV Cache 加速的并行解码
- **Learn2PD**：使用轻量过滤器预测收敛 token

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

| 指标 | 表现 |
|------|------|
| **推理加速倍数** | **2.4× ~ 8.5×**（平均 3–5×） |
| **解码步数减少** | 从 256 步降至 **30~90 步**（最高减少 225 步） |
| **延迟降低** | 单次生成延迟下降 **7–13 秒** |
| **准确率变化** | 多数任务实现正向增益，最高提升 **+14.05%（ARC-C）** |

#### 示例任务表现：

| 任务 | 方法 | Accuracy | Steps | Speedup |
|------|------|----------|-------|---------|
| **GSM8K** | LLaDA | 73.01% | 256 | 1.00× |
| | ReMix | **75.66% (+2.65%)** | **51.55** | **4.63×** |
| **ARC-C** | MMaDA | 52.17% | 256 | 1.00× |
| | ReMix | **66.22% (+14.05%)** | **61.30** | **3.92×** |
| **Flickr30k-lite** | MMaDA | 57.52 (CIDEr) | 256 | 1.00× |
| | ReMix | **59.59 (+2.07)** | **30.13** | **7.52×** |

> 💡 **亮点**：ReMix 在大幅提升速度的同时，**没有牺牲质量，反而提升了输出准确性**。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）消融 Mixing Module（禁用连续状态）
- 将 $ \beta = 0 $，退化为 confidence-aware 并行解码
- 结果显示：**准确率明显下降**，证明连续状态对一致性建模至关重要

#### （2）调参分析（$ \beta $ 与 $ \tau_{\text{rej}} $）
- 中等 $ \beta $（0.5~0.6）效果最佳；过高会导致过拟合噪声
- $ \tau_{\text{rej}} $ 过小会频繁重置，过大则容忍不稳定输出
- 总体对超参鲁棒性强，大部分配置下均优于 baseline

#### （3）不同生成长度测试（Tab. 3）
- 随着 generation length 增加（128 → 512），ReMix 的加速比进一步提升（最高达 **6.85×**）
- 准确率增益也更显著（如 GSM8K 上 +2.95%）

#### （4）全扩散解码场景（Fully Diffusion-based）
- 当 block length = generation length 时，LLaDA 性能严重下降
- ReMix 成功补偿损失，仍保持高速高质量输出

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **“组合矛盾”是限制 DLLM 并行解码性能的根本瓶颈**，源于离散采样的局部独立性。
2. **引入连续中间状态（Continuous Mixing State）能有效缓解该问题**，允许 token 在离散化前进行语义协调。
3. **Rejection Mechanism 显著增强了解码稳定性**，防止低置信度预测引发连锁错误。
4. **ReMix 是一种即插即用、无需训练的高效方案**，在语言与多模态任务上均取得显著收益。
5. **速度与质量不再对立**：ReMix 打破了传统“越快越差”的权衡，实现 **加速且提效**。

---

### ⚠️ 方法的局限性
- **依赖预设阈值**：$ \tau_{\text{conf}}, \tau_{\text{rej}} $ 需手动调节，在不同任务间可能需微调。
- **理论解释有限**：虽然实验证明有效，但连续状态如何精确捕捉语义依赖尚缺乏形式化建模。
- **极端长序列适应性待验证**：当前实验集中在 ≤512 长度，超长文本下的表现未知。

---

### 🔮 未来工作方向
1. **自动化超参调节机制**：设计 adaptive $ \beta $ 或 $ \tau_{\text{rej}} $ 策略。
2. **将 ReMix 思想扩展至其他生成范式**：如 AR 模型中的 speculative decoding。
3. **构建端到端联合优化版本**：将 continuous state 学习纳入训练过程，进一步释放潜力。
4. **探索更多混合策略**：如结合 diffusion 与 AR 的 hybrid generation pipeline。

---

> 🔗 **开源地址**：[https://github.com/Serpientw/ReMix-DLLM](https://github.com/Serpientw/ReMix-DLLM)

</details>

---

### 2. [RLHFless: Serverless Computing for Efficient RLHF](https://arxiv.org/abs/2602.22718)

**Authors**: Rui Wei, Hanfei Yu, Shubham Jain, Yogarajan Sivakumar, Devesh Tiwari, Jian Li, Seung-Jong Park, Hao Wang  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2602.22718v1  

#### Abstract
Reinforcement Learning from Human Feedback (RLHF) has been widely applied to Large Language Model (LLM) post-training to align model outputs with human preferences. Recent models, such as DeepSeek-R1, have also shown RLHF's potential to improve LLM reasoning on complex tasks. In RL, inference and tr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：RLHFless: Serverless Computing for Efficient RLHF**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **Reinforcement Learning from Human Feedback (RLHF)** 框架大多基于 **serverful infrastructure**（有服务器架构），存在以下关键问题：
- **资源利用率低**：在同步训练流程中，各阶段（生成、准备、学习）之间存在依赖关系，导致大量 GPU 资源处于空闲状态。
- **静态资源配置无法适应动态负载**：随着训练进行，模型生成的响应长度不断变化（如数学推理任务中答案逐渐变长），固定数量的采样 actor 导致资源浪费或瓶颈。
- **重复计算开销大**：多个响应共享相同前缀时，每个 actor 都会独立执行 `prefill` 阶段，造成冗余的 KV cache 计算。

这些问题导致训练成本高、效率低下，尤其在大规模 LLM 场景下更加严重。

---

### **提出了什么新方法或新思路**
本文提出 **RLHFless** —— **首个基于 serverless 架构的可扩展同步 RLHF 训练框架**，其核心创新包括：

#### ✅ **1. Deduplicated Prefill（去重预填充）**
- 分析提示集中的共享前缀，在一个专用的 **prefill actor** 中统一计算一次 KV cache，并供所有 decode actors 复用。
- 显著减少重复的 `prefill` 计算，降低总计算量。

#### ✅ **2. Cost-Aware Actor Scaling（成本感知的 actor 扩缩容）**
- 动态调整每步训练中使用的 decode actor 数量。
- 基于历史响应长度预测当前步骤的工作负载，结合执行时间与成本模型，寻找“甜点”——即在最小化总成本的同时最大化训练速度。

#### ✅ **3. Prompt Assignment with Cut-and-Migrate（基于长度分组的任务分配 + 运行时迁移）**
- 利用历史数据预测每个 prompt 的输出长度，并将相似长度的 prompts 分配到同一个 actor 中，减少长尾效应带来的资源闲置。
- 引入 **cut-and-migrate 回退机制**：当某个 actor 因低估而处理超长响应时，将其未完成序列迁移到其他已完成任务的 actor 上继续执行，进一步提升资源利用率。

#### ✅ **4. Locality-Aware Actor Placement（局部性感知的部署策略）**
- 将 prefill actor 与 learner 放置在同一节点以减少模型同步延迟。
- 优先将重负载的 decode actor 部署在靠近 learner 的物理节点上，实现通信与计算的重叠，隐藏传输延迟。

---

### **相比现有方法的优势**
| 维度 | 现有方法（如 VERL） | RLHFless |
|------|---------------------|----------|
| 架构 | Serverful（静态资源） | **Serverless**（按需弹性伸缩） |
| 资源利用 | 存在显著空闲期 | **细粒度释放空闲组件** |
| 响应长度变化适应能力 | 固定 actor 数量 | **动态扩缩容** |
| KV cache 复用 | 有限复用（批内） | **跨 actor 全局去重 prefill** |
| 负载均衡 | 无显式优化 | **长度感知分组 + 运行时迁移** |

> ✅ 总体优势：**更低成本 + 更高速度 + 更好适应动态负载**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GSM8K**：小学数学应用题，用于测试数学推理能力。
- **GPQA**：研究生级别科学问答，考察深度知识理解。
- **LiveCodeBench**：编程任务基准，评估代码生成能力。

### **模型配置**
- 主要使用 **Qwen2.5-3B / 7B** 模型进行实机测试。
- 在大规模模拟中使用 **Llama2-70B** 和 **DeepSeek-7B**。
- 算法支持 **PPO** 和 **GRPO**（sampling-heavy 类型）。

### **硬件环境**
- 物理集群：两个 AWS EC2 `g6e.48xlarge` 实例，每台含 **8×NVIDIA L40S GPU**（共 16 GPUs）。
- 大规模仿真：基于 **Vidur** 模拟器构建最多 20 节点、每节点 8×H100 GPU 的集群。

### **评估指标**
- **Per-step execution time**（单步耗时）
- **GPU×second 成本**（衡量资源消耗）
- **端到端训练加速比**
- **成本降幅**

### **基线方法对比**
- **VERL**：主流开源 RLHF 框架，作为主要 baseline。
- **RLHFuse**：采用全局 cut-and-migrate 的优化系统，用于对比 prompt assignment 效果。
- **Oracle**：理想情况下的最优调度器（假设完全准确的长度预测），用于上限分析。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | 结果 |
|------|------|
| **最大加速比** | **1.35×** 端到端训练加速 |
| **最高成本降低** | **44.8%** 的 GPU×second 成本下降 |
| **大尺度模拟表现** | 平均 **1.23× 加速**，**38.7% 成本降低** |

> 💡 在 GRPO 上收益高于 PPO，因其为 **sampling-heavy** 算法，优化空间更大。

---

### **与基线方法的对比结果**
- 相比 **VERL**：
  - 显著缩短生成阶段等待时间。
  - 减少因长尾响应造成的资源锁定。
  - 成本下降超过 40%，且不牺牲训练质量。
- 相比 **RLHFuse**：
  - 通过 **prompt ranking + 分组分配**，避免粗粒度迁移开销。
  - 实现 **12.8% 的额外成本节省**。

---

### **消融实验结果（Ablation Study）**
在 **GRPO + Qwen2.5-3B + GSM8K** 设置下进行：

| 变体 | 相对改进说明 |
|------|--------------|
| **w/o DP+AS+PA**（全关闭） | 基准版本，等同于原始 VERL 行为 |
| **+ Deduplicated Prefill** | 减少约 22% 的冗余 prefill 开销（见 Fig. 3(c)） |
| **+ Prompt Assignment** | 降低 actor 内部负载不平衡，早终止更多轻量 actor |
| **+ Actor Scaling** | 动态调节资源匹配负载波动，找到 cost-speed 最优平衡点 |

> 🔍 结论：三项技术 **协同增效**，共同贡献最终性能提升。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Serverless 架构非常适合 RLHF 的动态特性**：
   - 各阶段间存在天然空闲窗口，适合按需启动/销毁函数。
   - 动态响应长度变化可通过弹性扩缩容有效应对。

2. **去重 prefill 是高效利用的关键**：
   - 多个响应共享前缀是普遍现象，集中 prefill 可大幅削减计算总量。

3. **无需复杂预测模型也能实现良好调度**：
   - 利用 **历史响应长度 + EWMA 平滑估计** 即可获得稳定排序，足以支撑有效的 prompt assignment。
   - 即使预测不准，**cut-and-migrate** 机制也能纠正错误，保证鲁棒性。

4. **动态扩缩容可在高吞吐与低成本间取得平衡**：
   - 并非越多 actor 越好；RLHFless 能自动识别“甜点”，实现 **更高并行度 ≠ 更高成本**。

---

### **方法的局限性**
- 当前聚焦于 **synchronous RLHF**，尚未直接支持异步训练（尽管设计正交，可扩展）。
- 依赖历史训练数据进行长度预测，在 **全新 prompt 分布** 下可能效果下降。
- serverless 冷启动问题虽通过预热缓解，但在极端低延迟场景仍需关注。

---

### **未来工作方向**
- 将 RLHFless 的设计理念扩展至 **asynchronous RLHF 系统**。
- 探索 **multi-round RLHF** 中的长期资源规划。
- 结合 **proxy model 或轻量预测头** 提升首次训练轮次的长度预测精度。
- 支持更多类型的 LLM alignment 方法（如 DPO、RAFT）。

---

> 📌 **一句话总结**：  
> **RLHFless 通过 serverless 架构实现了对 RLHF 动态资源需求的精细适配，结合去重 prefill、智能扩缩容与长度感知任务调度，在真实与模拟环境中均实现了高达 1.35× 加速与近半成本下降，为高效大模型对齐训练提供了新范式。**

</details>

---

### 3. [Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation](https://arxiv.org/abs/2602.23188)

**Authors**: Isma\"el Zighed, Andrea N\'ovoa, Luca Magri, Taraneh Sayadi  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2602.23188v1  

#### Abstract
We propose an efficient retraining strategy for a parameterized Reduced Order Model (ROM) that attains accuracy comparable to full retraining while requiring only a fraction of the computational time and relying solely on sparse observations of the full system. The architecture employs an encode-pro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Real-Time Adaptation of ROMs for Unsteady Flows Using Data Assimilation**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
该论文针对**非定常流体系统**（unsteady flows）中的**参数化 Reduced Order Model (ROM)** 在**跨参数域外推时预测精度下降**的问题。传统方法在模型泛化失败后需要重新训练整个模型，这通常依赖于高保真、全状态、密集采样的数据，计算成本高昂且不适用于实时场景。

此外，实际应用中观测数据往往是**稀疏的**（sparse observations），如何利用少量传感器数据实现高效、准确的模型自适应成为一大挑战。

---

### ✅ **提出了什么新方法或新思路**
提出了一种**高效的实时自适应策略**，将**数据同化**（Data Assimilation, DA）与**概率性 ROM 架构**紧密结合，实现仅用极少量稀疏观测数据即可完成模型微调。

核心思想包括：

- **选择性重训练（Selective Retraining）**：  
  发现模型误差主要来源于**潜在流形**（latent manifold）的几何失配，而非潜在空间动力学（latent dynamics）的失效。因此，只需重新训练 **VAE 编码器-解码器部分**，而冻结 **Transformer 动力学模块**，即可达到接近全模型重训练的性能。

- **基于 EnKF 的数据同化框架**：  
  利用 ROM 输出的**随机轨迹集合**（stochastic ensemble）作为先验，在仅有 **1% 空间维度的稀疏观测**下，通过 **Ensemble Kalman Filter (EnKF)** 融合观测数据，重建高质量的全状态轨迹用于微调。

- **轻量化微调流程**：  
  将 DA 生成的数据用于仅对 VAE 进行微调，显著降低计算开销，支持**近实时甚至秒级响应**。

---

### ✅ **相比现有方法的优势**
| 维度 | 本文方法 | 传统方法 |
|------|---------|--------|
| **数据需求** | 仅需 ~1% 稀疏观测（如 64 个传感器） | 需要全状态、高保真模拟数据 |
| **计算成本** | 微调时间从 2 小时降至 **~15 分钟**（第一矩收敛仅需数秒） | 完整重训练耗时长达 **10 小时以上** |
| **适用性** | 支持在线、实时适应新参数区域 | 多为离线训练，难以部署于动态环境 |
| **不确定性建模** | 内生支持 UQ（Uncertainty Quantification），EnKF 自然兼容 | 多为确定性模型，难以处理噪声与不确定性 |

---

## 2. **核心实验方法和设置**

### ✅ **使用的数据集**
- **物理系统**：二维不可压缩 Navier-Stokes 方程描述的绕障碍物流动（obstacle flow）
- **求解器**：Immersed Boundary Method Solver (**ibmos**) [30]
- **参数范围**：雷诺数 $ \text{Re} \in [80, 140] $
- **训练配置**：
  - 初始训练在 **Re = 90 和 120**
  - 测试/验证涵盖 **Re = {80, 90, ..., 140}**（共7个工况）
- **空间分辨率**：$131 \times 100$ 网格 → 总自由度 13,100（每点含 $u,v$ 速度分量）
- **降维后输入维度**：6550（经下采样至 3275 × 2）

---

### ✅ **实验设置和评估指标**

#### 🧪 **模型架构**
采用 **encode-process-decode** 结构：
- **Encoder/Decoder**：Variational Autoencoder (VAE)，学习低维潜在空间（latent space dim = 4）
- **Processor**：Transformer 网络，建模潜在空间的时间演化，并通过 cross-attention 引入参数 $ \text{Re} $
- **输出形式**：概率性预测（ensemble），提供均值与方差（UQ）

#### 📊 **评估指标**
1. **Kinetic Energy Signal**：聚合 $u,v$ 成单一物理量进行可视化比较。
2. **2-Wasserstein Distance**（能量距离）：
   - 衡量预测与真实轨迹的概率分布差异
   - 对相位偏移鲁棒，优于 MSE 或 MAE
3. **Relative L1/L2 Reconstruction Error**：衡量 VAE 重构能力
4. **Uncertainty Quantification (UQ)**：通过 ensemble variance 估计模型置信度

#### 🔁 **微调策略对比**
| 方法 | 描述 |
|------|------|
| **Full Model Retraining** | 使用完整高保真数据重新训练整个模型（作为“黄金标准”） |
| **VAE-only Retraining** | 仅重训 VAE，冻结 Transformer |
| **VAE + DA Retraining** | 使用 EnKF 同化稀疏观测生成的分析数据来微调 VAE |

---

### ✅ **基线方法对比**
- **经典 ROM 微调方法**：需运行新参数下的完整仿真并全模型再训练
- **Deterministic Surrogate Models**：缺乏不确定性建模，无法有效融合稀疏观测
- **纯机器学习外推**：无数据同化机制，面对 out-of-distribution 参数表现差

---

## 3. **主要实验结果和性能指标**

### ✅ **关键性能数据**

| 指标 | 数值/效果 |
|------|--------|
| **稀疏观测比例** | 64 个传感器 → 占总空间维度 **1%**（6550 → 64） |
| **EnKF 后误差降低** | U 分量误差 ↓89.4%，V 分量误差 ↓96.1%，总体 ↓**93.8%** |
| **能量距离减少** | 在 Re=140 上，相比原始模型 ↓**70%** |
| **微调时间** | 第一矩收敛：<1分钟；第二矩稳定：约 **15分钟**（vs 全训练 10h） |
| **数据量节省** | 微调所需新数据维度从 6550 降至 **64**（压缩 99%） |

---

### ✅ **与基线方法的对比结果**

| 方法 | 能量距离（Re=140） | 数据需求 | 时间成本 |
|------|------------------|----------|----------|
| 原始模型（未微调） | 0.0050 | — | — |
| Full Model Retraining | 0.0036 | 6550-dim | ~10h |
| VAE-only Retraining | 0.0036 | 6550-dim | <1h |
| **VAE + DA Retraining** | **0.0040** | **64-dim (1%)** | **~15min** |

> ⚠️ 可见：**VAE+DA 方法以 1% 数据 + 1/40 时间，达到了接近全重训练的性能**

---

### ✅ **消融实验结果**

#### 🔹 **潜在流形 vs 潜在动力学分析**
- **假设**：误差主要来自 latent manifold 几何失配，而非 dynamics 错误
- **验证方式**：
  - 对比不同 Re 下的 latent trajectory 及其 2D embedding（Spectral, Isomap, Hessian）
- **发现**：
  - 未微调模型在 Re=140 下 latent orbit 半径过大 → limit cycle 幅值偏差
  - 仅重训 VAE 即可修复 manifold 拓扑结构，使轨迹与真实高度重合

#### 🔹 **VAE 重构误差对比（Table 1 & 2）**

| 方法 | Relative L1 Error | Relative L2 Error |
|------|------------------|------------------|
| Pre-retraining | 2.53% | 3.11% |
| Post-full retraining | 0.19% | 0.35% |
| **Post-VAE-only retraining** | **0.19%** | **0.37%** |

> ✅ 表明：**只重训 VAE 即可实现与全模型重训相当的重构精度**

#### 🔹 **是否需考虑 EnKF 的二阶矩？**
- 分析表明：VAE 自动学习到的 encoder variance $ \text{diag}(A_g) $ 与 EnKF 分析 ensemble 的经验方差 $ \text{diag}(\Sigma_a) $ 高度一致（图17）
- 因此无需在损失函数中显式加入协方差惩罚项（如加权 loss Eq.6），简化训练流程

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **主导误差源是 latent manifold 的几何失配**，而非 dynamics 学习错误 → 支持“冻结 Transformer，仅调 VAE”的轻量化策略。
2. **EnKF 可有效融合稀疏观测与 ROM 先验**，即使只有 1% 观测也能重建高质量全状态轨迹。
3. **VAE 的变分结构天然适配 EnKF 输出的 Gaussian ensemble**，能自动捕捉同化后的不确定性。
4. **第一矩（均值）收敛极快**（<30秒），适合实时控制；第二矩（方差）稍慢但仍可在 15 分钟内完成。
5. **该框架实现了性能、效率与数据成本之间的优良平衡**，为工程级实时流场建模提供了可行路径。

---

### ⚠️ **方法的局限性**
1. **适用于平滑参数变化**，若发生**突变型分岔**（qualitative bifurcation），latent dynamics 也可能改变，此时仅调 VAE 不足。
2. **依赖 VAE-Gaussian 假设**，若 decoder 扰乱 Gaussianity，则 EnKF 效果可能下降（尽管 KS 检验支持该假设）。
3. **EnKF 引入噪声可能导致预测略显“模糊”**，虽不影响准确性，但可能影响下游任务（如梯度优化）。
4. **传感器布局依赖模态信息最大化原则**，需预先了解系统特征结构（如 POD/SVD modes）。

---

### 🔮 **未来工作方向**
1. **扩展至三维湍流或多物理场耦合系统**
2. **结合 smoother（如 RTS Smoother）对同化数据去噪，提升微调稳定性**
3. **发展自适应传感器部署策略**，实现闭环主动感知
4. **探索更复杂的 DA-ML 联合损失函数**，进一步融合 aleatoric 与 epistemic uncertainty
5. **硬件集成与边缘部署**，推动该方法在风洞实验、飞行控制等实时场景的应用

---

## ✅ **总结一句话**
> 本论文提出一种**基于 EnKF 数据同化的轻量级 ROM 实时自适应方法**，通过**仅微调 VAE 编解码器**，利用**1% 稀疏观测**即可在**15 分钟内**将预测误差降低 **70%**，实现了高精度、低数据依赖、快速响应的动态模型更新，为复杂系统的在线建模与控制开辟了新路径。

</details>

---

### 4. [Search-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training](https://arxiv.org/abs/2602.22576)

**Authors**: Tianle Xia, Ming Xu, Lingxiang Hu, Yiding Sun, Wenwei Li, Linfang Shang, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.22576v1  

#### Abstract
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, yet traditional single-round retrieval struggles with complex multi-step reasoning. Agentic RAG addresses this by enabling LLMs to dynamically decide when and what to retrieve, but current...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SEARCH-P1: Path-Centric Reward Shaping for Stable and Efficient Agentic RAG Training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **Retrieval-Augmented Generation (RAG)** 多采用单轮检索，在面对需要**多步推理**的复杂查询时表现不佳。虽然 **Agentic RAG** 允许 LLM 动态决定何时、如何检索，但现有的基于 **Reinforcement Learning (RL)** 的训练方法存在以下三大缺陷：
1. **Reward Sparsity**：仅依赖最终答案是否正确作为奖励信号（outcome-based reward），忽略了中间推理路径的质量。
2. **Sample Inefficiency**：失败样本（incorrect answer）获得零奖励，无法提供有效学习信号。
3. **Slow Convergence**：由于奖励信号稀疏且二元化，训练梯度弱，收敛速度慢。

---

### 🚀 提出的新方法：SEARCH-P1
为解决上述问题，作者提出 **SEARCH-P1**，一种**以路径为中心的奖励塑造框架**（path-centric reward shaping），其核心创新包括：

#### （1）**Dual-Track Path Scoring（双轨路径评分）**
- **Track A: Self-Consistency（自洽性）**  
  评估模型是否忠实执行了自己在推理中声明的计划（`planner`）。通过比较“计划步骤”与“实际执行步骤”的匹配度打分。
- **Track B: Reference-Alignment（参考对齐）**  
  利用离线生成的**参考规划器**（reference planner）构建最优推理路径，衡量模型轨迹对关键步骤的覆盖程度（order-agnostic 匹配）。
- 最终取两条轨道中的**最高分**作为路径奖励，避免因参考路径次优而压制模型探索更优策略的能力。

#### （2）**Soft Outcome Scoring（软结果评分）**
即使最终答案错误，只要推理过程合理，仍给予部分奖励：
```math
R_{\text{outcome}} = \alpha \cdot r_{\text{acc}} + (1-\alpha) \cdot r_{\text{reason}}, \quad \alpha=0.8
```
其中 $ r_{\text{acc}} $ 衡量答案部分正确性，$ r_{\text{reason}} $ 评估推理质量，从而将“失败样本”转化为有用的学习信号。

#### （3）**Path-Centric Reward Framework**
综合三项奖励构成总奖励函数：
```math
R_{\text{total}} = \lambda_p \cdot R_{\text{path}} + \lambda_a \cdot R_{\text{outcome}} + \lambda_f \cdot R_{\text{format}}
```
实现了从“只看结果”到“关注全过程”的范式转变。

---

### 🔍 相比现有方法的优势
| 特性 | Search-R1 / HiPRAG | SEARCH-P1 |
|------|---------------------|-----------|
| 奖励信号密度 | 稀疏（binary outcome） | 密集（path-level + soft outcome） |
| 样本利用率 | 低（失败样本无收益） | 高（失败样本仍有学习价值） |
| 收敛速度 | 慢（>150 steps） | 快（~60 steps 达标） |
| 推理效率 | 成功/失败路径差异大 | 更稳定一致 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **公开 QA 基准（7个）**：
  - **General QA**：NQ, TriviaQA, PopQA
  - **Multi-Hop QA**：HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle
- **内部工业数据集（1个）**：
  - **AD-QA**：来自腾讯广告业务的真实多跳问答数据集（1,000 条测试实例），涵盖广告投放、受众定向等复杂场景。

> 所有模型在 **NQ + HotpotQA 的合并训练集**上进行训练，其余为 out-of-domain 测试。

---

### ⚙️ 实验设置
- **模型**：
  - 主要使用 `Qwen2.5-7B-Instruct` 和 `Qwen2.5-3B-Instruct`（简称 7B / 3B）
  - 对比也包含 `Llama-3.2-3B-Instruct`
- **检索器**：
  - 使用 **E5** 检索模型，每次返回 top-3 文档
  - 知识源为 **2018 Wikipedia dump**
- **训练算法**：
  - 采用 **GRPO**（Generalized Reward Policy Optimization）
  - 参考路径由高能力 LLM（HY 2.0-Instruct）离线生成并缓存
- **超参数**：
  - 默认权重：$\lambda_f=0.1$, $\lambda_p=0.3$, $\lambda_a=0.6$
  - 最大动作预算（action budget）：4 次搜索-推理循环

---

### 🎯 评估指标
- **主指标**：**Accuracy (ACC%)** —— 是否生成了正确的答案
- **辅助分析指标**：
  - 路径一致性得分（Self-Consistency）
  - 参考路径覆盖率（Reference Coverage）
  - 平均交互轮数（Turns）
  - 训练收敛速度

---

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **Non-Retrieval** | Direct Inference, Chain-of-Thought (CoT) |
| **Standard RAG** | 单轮检索后生成 |
| **Prompt-Based Agentic RAG** | IRCoT, Search-o1 |
| **RL-Based Agentic RAG** | **Search-R1**, HiPRAG（复现） |

> 所有 RL 方法共享相同训练配置，仅奖励函数不同。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Table 1）

| 方法 | Avg. ACC (7B) | AD-QA (7B) | Avg. ACC (3B) | AD-QA (3B) |
|------|---------------|------------|----------------|-------------|
| Search-R1 | 39.6 | 65.6 | 33.6 | 58.3 |
| HiPRAG | 42.9 | 75.6 | 36.6 | 70.2 |
| **SEARCH-P1 (Ours)** | **47.3** | **86.2** | **41.5** | **79.5** |

> ✅ **平均准确率提升 +7.7 pts（7B）**，在 AD-QA 上高达 **+20.6 pts**

---

### 🔁 与基线方法对比结果
- 在所有数据集上均显著优于各类基线，尤其在复杂任务中优势明显：
  - **Multi-Hop QA 平均提升 +7.7 pts**
  - **AD-QA 提升达 +20.6 pts（7B）**
- 小模型（3B）也能取得 **+7.9 pts** 提升，说明方法对模型规模不敏感。
- 图表显示 SEARCH-P1 在 **Qwen2.5-7B 和 3B** 上均实现全面领先（见 Figure 1）。

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）路径奖励组件消融（Table 2 & 8）
| 方法 | Avg. ACC (7B) | 下降幅度 |
|------|----------------|---------|
| Full SEARCH-P1 | 47.3 | — |
| w/o Reference-Alignment | 42.0 | ↓5.3 pts |
| w/o Self-Consistency | 44.2 | ↓3.1 pts |
| Search-R1 | 39.6 | ↓7.7 pts |

> 结论：两个路径评分轨道互补，**Reference-Alignment 对多跳任务更重要**，**Self-Consistency 更利于通用 QA**

#### （2）Soft Outcome Scoring 效果（Figure 4 & Table 10）
| 任务类型 | 提升幅度 |
|--------|----------|
| General QA | +1.1 ~ +1.5 pts |
| Multi-Hop QA | +3.0 ~ +3.7 pts |
| AD-QA | **+8.8 ~ +11.0 pts** |

> 显示越复杂的任务，软评分带来的增益越大。

#### （3）Format Reward 影响（Figure 3）
- “Soft Format” 设计（允许部分格式合规即给分）相比“Strict Format”（格式错则零分）显著加快收敛。
- 加入 format reward 后，输出合规率从 ~75% 提升至 **>95%**（Table 6）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **路径级奖励显著提升训练效率与性能**  
   通过引入 dual-track path scoring，提供了密集、细粒度的监督信号，解决了 RL 中 reward sparsity 问题。

2. **软奖励机制极大提高样本利用率**  
   即使最终答案错误，高质量的推理路径仍可获得正反馈，尤其在复杂任务中效果显著。

3. **方法具有良好的泛化性和鲁棒性**  
   - 在不同模型（Qwen / Llama）、不同 RL 算法（GRPO / PPO）下均能带来稳定增益（Table 3, 9）
   - 对 LLM evaluator 不敏感：即使使用较小的 Qwen3-8B 作为 evaluator，核心 step coverage 判断依然可靠（human agreement >88%）

4. **工业场景适用性强**  
   在真实广告 QA 场景（AD-QA）中表现突出，验证了其在企业级知识系统中的实用价值。

---

### ⚠️ 局限性
1. **依赖外部 LLM evaluator 进行训练期评分**  
   虽然推理时不需调用，但在训练阶段增加了计算开销（尽管是离线缓存）。
2. **参考路径生成成本较高**  
   每条训练样本需约 1.91 次 LLM 调用生成 reference planner（一次性成本）。
3. **对 planner 模块的设计敏感**  
   若模型未能显式声明合理的 plan，会影响 self-consistency 评分的有效性。

---

### 🔮 未来工作方向
1. **自动化参考路径生成优化**  
   探索更高效、低成本的方式生成 reference planners，如利用小模型蒸馏。
2. **动态调整奖励权重**  
   当前 $\lambda_p$, $\lambda_a$ 为固定值，未来可尝试动态平衡 accuracy 与 reasoning quality。
3. **扩展至其他 agentic 任务**  
   如代码生成、数学推理、工具调用等同样涉及多步决策的任务。
4. **减少对外部 evaluator 的依赖**  
   探索 self-evaluation 或 consistency-based 自动评分机制。

---

## 总结
**SEARCH-P1** 是一项针对 **Agentic RAG 训练稳定性与效率**的重要改进，通过**路径中心化的奖励设计**，实现了从“只看结果”到“全过程指导”的跃迁。其实验充分、设计精巧，在学术与工业场景中均展现出强大潜力，为下一代智能代理系统的训练提供了新范式。

</details>

---

### 5. [CCCL: Node-Spanning GPU Collectives with CXL Memory Pooling](https://arxiv.org/abs/2602.22457)

**Authors**: Dong Xu (UC Merced), Han Meng (UC Merced), Xinyu Chen (Zhejinag University), Dengcheng Zhu (Bytedance and), Wei Tang (Bytedance and), Fei Liu (Bytedance and), Liguang Xie (Bytedance and), Wu Xiang (Bytedance and), Rui Shi (Bytedance and), Yue Li (Bytedance and), Henry Hu (Bytedance and), Hui Zhang (Bytedance and), Jianping Jiang (Xconn-tech), Dong Li (UC Merced)  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.22457v1  

#### Abstract
Large language models (LLMs) training or inference across multiple nodes introduces significant pressure on GPU memory and interconnect bandwidth. The Compute Express Link (CXL) shared memory pool offers a scalable solution by enabling memory sharing across nodes, reducing over-provisioning and impr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CCCL: Node-Spanning GPU Collectives with CXL Memory Pooling

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模 **Large Language Models (LLMs)** 的训练与推理中，跨节点的 GPU 集体通信（collective communication）对内存容量和互连带宽提出了极高要求。传统的基于 **RDMA** 和 **InfiniBand** 的通信方案存在以下问题：
- **高成本**：高性能网络设备（如 200 Gbps InfiniBand）价格昂贵；
- **复杂软件栈**：需协调 NCCL、MPI 和框架后端，增加系统复杂性；
- **资源利用率低**：难以实现高效的内存共享与负载均衡。

此外，虽然 **CXL shared memory pool** 已被用于 KV Cache 存储等场景（如 Beluga、TraCT），但尚未被系统性地用于支持高性能的跨节点 GPU 集体通信。

### 提出的新方法与新思路
本文提出 **CCCL** —— 一种构建于 **CXL shared memory pool** 上的新型集体通信库，首次将 CXL 内存池作为跨节点 GPU 通信的基础设施，而非仅用作存储扩展。

#### 核心创新点：
1. **首个利用 CXL 内存池进行跨节点 GPU 集体通信的工作**
   - 将 CXL 共享内存视为“通信 fabric”，通过 `load/store` 语义替代传统 RDMA 消息传递机制，简化通信协议栈。

2. **设计了三项关键技术应对挑战**
   - **软件层数据交错（Software-level Interleaving）**
     - 在缺乏硬件级 cache-line interleaving 支持的情况下，采用模型引导的预分配策略，在多个 CXL 设备间均匀分布数据访问，提升聚合带宽利用率。
   - **细粒度异步重叠（Fine-grained Asynchronous Overlapping）**
     - 将通信数据分块，并引入 **doorbell 机制** 实现生产者-消费者同步，允许写入与读取操作异步并发执行，减少空闲时间。
   - **轻量级内存锁机制（Doorbell-based Lightweight Locking）**
     - 提出基于索引计算的 **doorbell 同步机制**，避免复杂的元数据管理，显著降低跨节点同步开销。

3. **完全绕过 RDMA 协议栈**
   - 利用 CXL.mem 协议提供的直接内存访问能力，实现 GPU ↔ CXL memory pool 的高效 DMA 传输，无需依赖传统网络协议。

### 相比现有方法的优势
| 维度 | 传统 RDMA/InfiniBand 方案 | CCCL |
|------|---------------------------|------|
| **通信范式** | 消息传递（message-passing） | 内存语义（load/store） |
| **硬件成本** | 高（$16K/switch） | 低（$5.8K/switch） |
| **软件复杂度** | 高（需 NCCL+RDMA+驱动协同） | 低（纯内存映射+CUDA） |
| **可扩展性** | 受限于拓扑与带宽竞争 | 更好利用聚合带宽 |
| **能效与资源利用率** | 中等 | 更高（减少 GPU 计算资源占用） |

---

## 2. 核心实验方法和设置

### 实验平台硬件配置
- **节点数**：3 个服务器节点
- **每节点配置**：
  - CPU：Intel Xeon 6960P（72 核）
  - 主存：256 GB DRAM
  - GPU：NVIDIA H100（80 GB HBM），通过 PCIe Gen5×16 连接
- **CXL 架构**：
  - 使用 **TITAN-II CXL switch**（支持 CXL 2.0）
  - 连接 **6 块 Micron CZ120 CXL Type-3 内存卡**，每块 128 GB，共 **768 GB CXL shared memory pool**
  - 每张卡通过 Gen5×8 接口接入 switch
- **操作系统与软件栈**：
  - OS：Linux 6.2.6
  - CUDA：12.8
  - 基准测试工具：`nccl-tests v2.17.8`

> ⚠️ 注：未启用 DDIO，且 CXL 地址空间设为非缓存区域以确保一致性。

### 基线方法对比
- **Baseline**：基于 **200 Gbps InfiniBand** 的标准 NCCL 实现（RDMA-based）
- **对比版本**：
  1. **CCCL-ALL**：完整功能版（含 bandwidth aggregation + asynchronous overlapping + interleaving）
  2. **CCCL-Aggregate**：仅带宽聚合，无异步重叠
  3. **CCCL-Naive**：顺序内存分配，无优化

### 评估指标
- **端到端延迟（End-to-end latency）**
- **吞吐率 / 带宽加速比（Speedup over baseline）**
- **消融分析（Ablation study）**：验证各组件贡献
- **可扩展性分析**：模拟从 3 → 6 → 12 节点的性能变化
- **真实应用测试**：LLM 训练场景下的性能与成本效益

### 测试的 Collective Primitives
来自 NCCL 的典型操作（见 Table 2）：
| Primitive | 类型 | 描述 |
|----------|------|------|
| `AllReduce` | N-to-N | 所有 rank 参与归约并接收结果 |
| `Broadcast` | 1-to-N | 根节点广播数据 |
| `AllGather` | N-to-N | 每个 rank 收集所有其他 rank 数据 |
| `ReduceScatter` | N-to-1 → 分散 | 归约后分散部分结果 |
| `Gather` / `Scatter` | N-to-1 / 1-to-N | 集中或分发数据 |
| `AlltoAll` | N-to-N | 全互换通信 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（平均加速比 vs. InfiniBand）
> 数据来自 Figure 9 和 Abstract，消息大小范围：1MB ~ 4GB

| Collective Operation | CCCL 加速比（vs. IB） |
|------------------------|-------------------------|
| `AllGather`           | **1.34×**                |
| `Broadcast`           | **1.84×**                |
| `Gather`              | **1.94×**                |
| `Scatter`             | **1.04×**                |
| `AllReduce`           | **1.50×**                |
| `ReduceScatter`       | **1.43×**                |
| `Reduce`              | **1.70×**                |
| `AlltoAll`            | **1.53×**                |

> ✅ 多数操作实现 **1.3–1.9× 性能提升**

### 与基线方法对比结果
- 在大多数 collective 操作中，**CCCL-ALL 显著优于 InfiniBand 基线**，尤其在 `Broadcast`, `Gather`, `Reduce` 等 1-to-N/N-to-1 模式下表现突出。
- **成本优势巨大**：
  - 硬件成本节省 **2.75×**（CXL switch $5.8K vs. IB switch $16K）
  - 更适合部署于成本敏感型数据中心

### 消融实验结果（Ablation Study）
#### （1）不同 CCCL 版本对比（Figure 9）
| 对比项 | 性能差异 |
|-------|----------|
| `CCCL-Naive` vs. `CCCL-ALL` | 最多快 **8×**（如 Gather） |
| `CCCL-Aggregate` vs. `CCCL-ALL` | 异步重叠带来额外 **~2× 提升** |
| 结论 | **interleaving + async overlapping 是性能关键** |

#### （2）分块数量敏感性分析（Figure 11）
- 使用 `AllGather` 测试不同数据分块数（chunk 数量）的影响
- 发现：
  - 小消息（256MB）对 chunk size 更敏感，最优配置可达 **20% 差异**
  - 大消息（1GB）也受益于合理分块（避免单 chunk 同步阻塞）
- 结论：**细粒度分块有助于提高通信并发性和重叠效率**

#### （3）可扩展性评估（Figure 10）
| 操作 | 3→6 节点 | 3→12 节点 |
|------|----------|-----------|
| `AllReduce` | 延迟 ↑ 2.1–3.0× | 延迟 ↑ 8.7–12.2× |
| `Broadcast` | 延迟 ↑ ~1.3× | 延迟 ↑ ~2.5× |
| `AlltoAll` | 延迟 ↑ ~1.6× | 延迟 ↑ ~3.6× |

> ❗ 随着节点增多，CXL 内存池内设备数量固定（6 块），导致设备级争用加剧，成为瓶颈  
> ✅ 但仍优于或接近 IB 表现，说明 CCCL 在小规模集群中有良好适应性

---

## 4. 关键结论和发现

### 主要发现
1. **CXL shared memory pool 可有效支撑高性能 GPU 集体通信**
   - 首次证明 CXL 不仅可用于存储扩展，还可作为 **高性能通信 fabric** 替代 RDMA。

2. **CCCL 显著提升通信效率并降低成本**
   - 平均 **1.3–1.9× 加速比**，最高达 **1.94×（Gather）**
   - 硬件成本降低 **2.75×**，极具经济优势

3. **软件层优化至关重要**
   - 缺乏硬件 interleaving 时，**软件控制的数据分布 + doorbell 同步 + 异步流** 是性能保障的关键。

4. **适用于 LLM 等实际负载**
   - 在 **Llama-3-8B + FSDP** 训练任务中实现 **1.11× 端到端加速**

### 方法的局限性
1. **受限于当前 CXL 生态成熟度**
   - 当前 CXL 2.0 支持有限，尚无广泛商用产品；实验依赖特定厂商硬件（Xconn, Micron）

2. **可扩展性受制于 CXL switch 与 memory modules 数量**
   - 固定 6 块内存模块下，超过 6–12 节点后出现严重争用，限制横向扩展能力

3. **缺乏跨节点 cache coherence 支持**
   - 必须自行实现同步机制（如 doorbell），增加了编程负担

4. **GPU DMA 引擎限制**
   - 当前 GPU 仅有一个方向的 DMA engine，无法充分利用多 CXL 设备并行性（Observation 1）

### 未来工作方向
1. **结合 CXL 3.0 新特性**
   - 利用更强的 switching capability 和 enhanced coherence 支持更大规模部署

2. **探索更智能的资源调度算法**
   - 动态调整数据映射策略、chunk size、doorbell 分配以适应不同 workload

3. **集成进主流深度学习框架**
   - 将 CCCL 无缝对接 PyTorch/FSDP/TensorFlow，推动落地应用

4. **支持更多 accelerator 类型**
   - 扩展至 TPU、IPU 或其他 AI chip，打造通用 CXL-based collective 通信标准

5. **研究 disaggregated memory 场景下的容错机制**
   - 如何处理 CXL memory module 故障、热插拔等问题

---

> 📌 **总结一句话**：  
> **CCCL 开创性地将 CXL shared memory pool 用于跨节点 GPU 集体通信，摆脱 RDMA 依赖，实现了更高性能、更低成本的通信范式，为下一代 memory-centric AI 架构提供了重要实践路径。**

</details>

---

### 6. [AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning](https://arxiv.org/abs/2602.22268)

**Authors**: Changhai Zhou, Shiyang Zhang, Yuhua Zhou, Qian Qiao, Jun Gao, Cheng Jin, Kaizhou Qin, Weizhong Zhang  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.22268v1  

#### Abstract
Quantization followed by parameter-efficient fine-tuning has emerged as a promising paradigm for downstream adaptation under tight GPU memory constraints. However, this sequential pipeline fails to leverage the intricate interaction between quantization bit-width and LoRA rank. Specifically, a caref...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前主流的 **LLM 下游任务适配范式** 是“先量化后微调”（quantize-then-finetune），即：
1. 将预训练模型进行 **Mixed-Precision Quantization**（如 4-bit）以压缩内存；
2. 冻结量化后的主干，在其上使用 **Parameter-Efficient Fine-Tuning (PEFT)** 技术（如 LoRA）进行微调。

然而，这种**顺序优化**方式存在根本缺陷：
- 量化位宽（bit-width）和 LoRA 秩（rank）是**强耦合**的：低精度引入的噪声可通过高秩适配器补偿；
- 传统方法将二者独立优化，导致资源分配不均，无法在固定内存预算下实现最优性能。

### **提出的新方法与新思路**
作者提出 **AutoQRA**（Automated Quantization-Rank Allocation），一种**联合优化框架**，同时为每一层分配最优的量化位宽 $q_l$ 和 LoRA 秩 $r_l$。

#### **核心创新点：**
- **联合搜索空间建模**：将每层的 $(q_l, r_l)$ 组合作为一个联合决策变量，在严格内存约束下最大化下游任务性能。
- **两阶段粗到精（coarse-to-fine）搜索策略**：
  1. **Phase I：全局多保真度进化搜索（Global Multi-Fidelity Evolutionary Search）**
     - 利用 **重要性先验（importance priors）** 进行种群初始化；
     - 采用 **重要性引导变异（importance-guided mutation）** 聚焦敏感层；
     - 引入 **学习型代理模型（surrogate model）** 加速候选配置筛选；
     - 使用 **NSGA-II + 多保真度评估** 构建近似 Pareto 前沿。
  2. **Phase II：局部信任域贝叶斯优化（Local Trust-Region Bayesian Optimization, TuRBO）**
     - 在 Phase I 找到的优质区域中，使用 **Expected Improvement (EI)** 准则进一步精细化搜索；
     - 支持用户偏好（如更侧重性能或更低内存）的标量化目标函数。

### **相比现有方法的优势**
| 对比维度 | 传统方法（如 QLoRA, AdaLoRA） | AutoQRA |
|--------|-------------------------------|---------|
| 优化方式 | 分离式（先定 bit，再定 rank） | 联合优化（joint bit & rank allocation） |
| 补偿机制 | 忽略量化噪声可被 adapter 学习补偿 | 显式建模“低比特 + 高秩”补偿模式 |
| 搜索效率 | 依赖启发式规则或静态代理 | 多保真 + Surrogate + TuRBO，样本高效 |
| 性能上限 | 受限于次优组合 | 接近全精度 LoRA 微调性能 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **微调数据集**：
  - `Alpaca-52k`：指令跟随数据集。
  - `HC3`：人类与 ChatGPT 回答对比数据集。
- **评估数据集（zero/few-shot）**：
  - `BoolQ`, `PIQA`, `HellaSwag`, `WinoGrande`, `ARC-E/ARC-C`, `OpenBookQA`, `MMLU`, `GSM8K`

### **实验设置与评估指标**
- **模型主干（Backbones）**：
  - `LLaMA-3.1-8B`, `LLaMA-3.2-3B`, `Qwen-2.5-7B`, `Qwen-2.5-3B`
- **内存预算**：严格控制总内存不超过给定上限 $B_{\text{max}}$
- **评估指标**：
  - 主要指标：**任务平均准确率（task-average accuracy %）**
  - 辅助指标：
    - 平均量化位宽（AvgBit）
    - 平均 LoRA 秩（AvgRank）
    - 总内存占用（Mem, GB）

### **基线方法对比**
| 方法 | 简介 |
|------|------|
| **LoRA (FP16)** | 全精度主干 + LoRA 微调，性能上限基准 |
| **QLoRA (4-bit)** | 4-bit 量化主干 + 固定秩 LoRA |
| **AdaLoRA** | 自适应调整 LoRA 秩，但量化固定 |
| **LoftQ / LQ-LoRA** | 联合考虑量化与 LoRA 初始化，但仍解耦优化 |
| **AMQ+LoRA / AMQ+AdaLoRA** | 先用 AMQ 做混合精度量化，再接 LoRA，代表“解耦式”联合流程 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| 方法 | 模型 | Avg Acc (%) | AvgBit | AvgRank | Mem (GB) |
|------|------|-------------|--------|---------|----------|
| LoRA (FP16) | LLaMA-3.1-8B | 69.94 | 16.00 | 16.00 | 20.50 |
| QLoRA (4-bit) | LLaMA-3.1-8B | 67.45 | 4.00 | 16.00 | 15.22 |
| **AutoQRA (≤4bit)** | LLaMA-3.1-8B | **69.83** | **3.75** | **10.50** | **13.08** |
| **AutoQRA (Optimal)** | LLaMA-3.1-8B | **70.45** | 5.25 | 12.25 | 17.32 |

> ✅ **AutoQRA (≤4bit)** 在仅 **3.75-bit** 平均精度、**13.08GB** 内存下，达到 **69.83%** 准确率，接近全精度 LoRA（69.94%），远超 QLoRA（67.45%）。

> ✅ **AutoQRA (Optimal)** 更是以 **70.45%** 超越全精度 LoRA，且平均位宽仍仅为 **5.25-bit**。

### **与基线方法的对比结果**
- 在所有四个 backbone 上，**AutoQRA (≤4bit)** 均为 **≤4-bit 方法中的最佳表现者**；
- 相比统一 4-bit 方法（QLoRA/AdaLoRA/LoftQ），**内存减少 12–22%**，同时性能显著提升；
- 即使与全精度 LoRA 相比，AutoQRA 在更低内存下实现了**相当甚至更高**的性能；
- **Decoupled 方法（如 AMQ+LoRA）始终次优**，验证了联合优化的必要性。

### **消融实验结果**
- **移除 Warm-start / Importance Prior**：收敛变慢，初期探索效率下降；
- **禁用 Phase I（仅用 BO）**：难以覆盖全局空间，易陷入局部最优；
- **禁用 Phase II（仅用 EA）**：缺乏精细搜索能力，最终性能受限；
- **仅优化 bit 或 rank 单一维度**：性能明显低于联合优化；
- **移除 Multi-fidelity 或 Surrogate Screening**：搜索成本大幅上升，效率降低。

> 🔍 **完整系统在准确性-内存权衡上全面占优**，尤其在全局探索与可行性保障方面最为关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **量化位宽与 LoRA 秩存在强交互关系**：
   - 低精度层常被分配更高秩，形成“**补偿模式（compensation pattern）**”；
   - 图 3 显示：多个模型中均出现 **负相关趋势**（低 $q_l$ ↔ 高 $r_l$）；
   - 这表明：**adapter 容量可用于主动补偿量化噪声**。

2. **静态代理指标不可靠**：
   - 如 Perplexity、Reconstruction Loss 等与最终微调性能相关性弱（图 1b，$\rho=0.46$）；
   - 必须通过实际微调来评估配置质量，支持黑箱优化路径。

3. **AutoQRA 实现“类全精度性能 + 类 4-bit 内存”**：
   - 在 ≤4-bit 设置下，性能逼近甚至超越 FP16 LoRA；
   - 内存开销低于标准 4-bit 方法，真正实现“**高性能、低内存**”双重优势。

### **方法的局限性**
- **搜索过程仍较昂贵**：虽然设计为离线运行，但一次搜索需数十次 HF 微调；
- **依赖实现细节**：如 REPAIR 算子、Surrogate 设计等对结果有影响；
- **泛化性待验证**：是否适用于非 Transformer 架构或其他 PEFT 方法（如 IA³）尚不明确。

### **未来工作方向**
- **在线自适应搜索**：在部署时动态调整 bit/rank 配置；
- **扩展至其他压缩技术**：如剪枝（pruning）、稀疏化（sparsity）联合优化；
- **轻量化搜索器设计**：构建小型代理网络直接预测最优配置，避免重复搜索；
- **理论分析补偿机制**：从优化动力学角度解释为何某些层适合“低比特+高秩”。

---

> 📌 **总结一句话**：  
> **AutoQRA 首次实现了 LLM 量化与适配器的端到端联合自动化优化，在极低内存下逼近全精度微调性能，揭示了“量化噪声可被学习补偿”的核心机制，为高效 LLM 微调树立了新标杆。**

</details>

---

### 7. [Accelerating LLM Pre-Training through Flat-Direction Dynamics Enhancement](https://arxiv.org/abs/2602.22681)

**Authors**: Shuchen Zhu, Rizhen Hu, Mingze Wang, Mou Sun, Xue Wang, Kun Yuan, Zaiwen Wen  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.22681v1  

#### Abstract
Pre-training Large Language Models requires immense computational resources, making optimizer efficiency essential. The optimization landscape is highly anisotropic, with loss reduction driven predominantly by progress along flat directions. While matrix-based optimizers such as Muon and SOAP levera...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerating LLM Pre-Training through Flat-Direction Dynamics Enhancement

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在大语言模型（LLM）预训练中，优化过程面临**高度各向异性（anisotropic）的损失景观（loss landscape）**。该景观具有以下特征：

- **平坦方向（flat directions）**：Hessian矩阵的特征值接近零或为负，参数更新缓慢，但主导了最终的损失下降。
- **尖锐方向（sharp directions）**：特征值大且稀疏，参数快速振荡，对稳定性重要但对损失下降贡献小。

现有的自适应优化器（如 **AdamW**、**Muon**、**SOAP**）虽然通过引入矩阵级预条件器（preconditioner）提升了性能，但仍存在两个关键缺陷：

1. **更新幅度趋于各向同性（isotropic）**：即在平坦和尖锐方向上的更新步长相近，导致在平坦方向上进展太慢，在尖锐方向上可能过于激进。
2. **动量机制未能有效利用二阶信息**：传统动量（如EMA）本质上是各向同性的阻尼机制，无法针对不同曲率方向进行差异化加速。

这些问题限制了优化效率，尤其是在大规模、长周期的LLM预训练中。

---

### ✅ 提出了什么新方法或新思路

作者提出了 **LITE**（**acceLerating adaptIve opTimizers in LLM prE-training**），一种通用的加速策略，其核心思想是：

> **增强平坦方向上的训练动力学，同时保持尖锐方向的稳定性。**

具体实现基于一个统一的理论框架：

#### 📌 统一的 Riemannian ODE 框架

作者提出了一种连续时间下的 **Riemannian Inertial System with Hessian Damping (RISHD)** 框架，将主流自适应优化器（如 AdamW、Muon、SOAP）及其 Nesterov 加速变体统一建模为黎曼流形上的惯性系统：

$$
\nabla_{\dot{w}} \dot{w} + \alpha \dot{w} + \beta \text{Hess}_f(w)\dot{w} + \gamma \text{grad}_f(w) = 0
$$

其中：
- **预条件器（Preconditioner）** 构造了一个由 $ F(w) $ 定义的黎曼度量，缓解了损失景观的病态性（ill-conditioning）。
- **动量（Momentum）** 在该框架下被解释为一种“黎曼阻尼”（Riemannian damping），有助于收敛。

这一框架首次从几何角度揭示了**预条件器与动量之间的协同作用机制**。

#### 📌 LITE 加速策略

基于上述分析，LITE 在离散更新规则中显式地：

- 在**平坦子空间**中使用更大的 **学习率放大因子 $ x \geq 1 $** 和更大的 **Hessian 阻尼系数 $ \beta_2 $**。
- 在**尖锐子空间**中保持原始超参数不变以维持稳定。

这使得：
- 动量在平坦方向上积累更快；
- 更新步长在平坦方向上更大；
- 整体训练动态沿主导损失下降的方向被显著加速。

---

### ✅ 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **理论深度** | 首次建立统一的 Riemannian ODE 框架，揭示了预条件器与动量的协同机制。 |
| **方法普适性** | LITE 是一种通用策略，可无缝集成到 Muon、SOAP 等先进优化器中。 |
| **加速效果** | 显著提升训练速度，在多种设置下实现约 **2× 的长周期训练加速**。 |
| **无需额外状态** | 利用已有预条件器（如 $ G^\top G $）估计平坦方向，不增加额外内存开销。 |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集

- **C4**：用于中小规模实验，采用 T5 tokenizer。
- **The Pile**：用于大规模预训练任务，采用 LLaMA-2 tokenizer。

---

### ✅ 实验设置和评估指标

#### 模型架构与规模
- **Dense 模型**：LLaMA 系列，参数量从 **130M 到 1.3B**。
- **MoE 模型**：QwenMoE-1B（10亿参数，激活约297M）。

#### 优化器配置
- **基线**：Muon、SOAP（已调优至最优性能）。
- **LITE 变体**：
  - **MUON-LITE** / **SOAP-LITE**
  - 包含消融版本：仅放大学习率（LITE-L）、仅增大阻尼系数（LITE-H）

#### 学习率调度
- **cos**：线性 warmup 后接余弦衰减。
- **wsd**（warmup-stable-decay）：warmup + 稳定期 + 线性衰减。

#### 评估指标
- 主要指标：**训练损失（train loss）随迭代次数的变化曲线**。
- 下游任务：在 LLaMA-1.3B 上进行 **zero-shot 评测**（使用 `lm-evaluation-harness`），涵盖 MMLU、ARC、BoolQ 等8项任务。

#### 超参数搜索
- 在 LLaMA-0.25B 上进行网格搜索确定最优 $ x, \beta_1, \beta_2 $，并在其他规模上复用。

---

### ✅ 基线方法对比

| 优化器 | 类型 | 是否使用 LITE |
|--------|------|----------------|
| Muon | Matrix-based Preconditioner + Nesterov | ❌ |
| SOAP | Matrix-based Preconditioner + Adam-style | ❌ |
| **MUON-LITE** | Muon + LITE | ✅ |
| **SOAP-LITE** | SOAP + LITE | ✅ |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 🔹 训练损失下降速度
- 在所有实验设置下（不同模型大小、数据集、LR schedule），**MUON-LITE 和 SOAP-LITE 均显著优于基线**。
- 在 **LLaMA-1.3B on Pile** 上，达到相同损失所需的训练 token 数减少近一半。
- **长期训练（200× 参数量 token 预算）下，MUON-LITE 实现约 2× 的加速**（见 Figure 1）。

#### 🔹 缩放律（Scaling Laws）
- 图 1（右）显示，MUON-LITE 在不同模型尺寸下的缩放行为更优，表明其具备良好的可扩展性。

#### 🔹 下游任务表现（Zero-Shot）
| 方法 | MMLU↑ | ARC-C↑ | BoolQ↑ | AVG↑ |
|------|-------|--------|--------|------|
| Muon | 23.76 | 22.10 | 52.45 | 43.44 |
| **MUON-LITE** | **25.15** (+1.39) | **23.89** (+1.79) | **60.52** (+7.97) | **45.49** |

> ✅ 表明 LITE 不仅降低训练损失，还提升了模型泛化能力。

---

### ✅ 与基线方法的对比结果

| 对比维度 | 结果 |
|---------|------|
| **收敛速度** | LITE 在所有设置下均更快降低训练损失。 |
| **终端损失** | LITE 达到更低的最终损失。 |
| **稳定性** | 通过区分处理平坦/尖锐方向，避免了不稳定风险。 |
| **通用性** | 在 Dense 和 MoE 架构上均有效。 |

---

### ✅ 消融实验结果

#### 🔹 LITE-L vs LITE-H vs Full LITE
- **LITE-L**（仅放大学习率）：有一定提升，但不如完整版。
- **LITE-H**（仅增大 Hessian 阻尼）：也有改进，尤其在早期阶段。
- **Full LITE**（两者结合）：效果最佳，验证了**同时调整学习率和动量机制**的重要性。

#### 🔹 尖锐方向错误调参的危害
- 若在**尖锐方向**也应用较大的 $ \beta $ 或 $ x $，会导致训练不稳定甚至性能劣于基线。
- 例如，在 Muon 中将 $ \beta=0.5 $ 应用于所有方向时，终端损失反而更高（2.113 vs 基线 2.110）。

#### 🔹 投影估计的有效性
- 实验验证了 Muon 和 SOAP 所使用的预条件器（如 $ G^\top G $）能很好地覆盖 Hessian 的主特征空间，从而可靠地估计平坦/尖锐子空间。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **平坦方向主导损失下降，但进展缓慢**，是 LLM 预训练的瓶颈。
2. **现有优化器的更新幅度趋于各向同性**，无法针对性加速平坦方向。
3. **通过 Riemannian ODE 框架可统一理解预条件器与动量的协同作用**：前者构建良好几何，后者提供方向感知的阻尼。
4. **LITE 通过在平坦方向上增大 Hessian 阻尼和学习率，显著加速训练动态**，且不牺牲稳定性。
5. **LITE 具有强通用性和可扩展性**，适用于多种优化器、架构、规模和数据集。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **依赖预条件器质量** | 平坦方向的识别依赖于预条件器（如 $ G^\top G $）对 Hessian 的近似精度。若近似差，投影可能不准。 |
| **超参数需手动设定** | 尽管在多个尺度上可复用，但 $ x, \beta_2, d_s $ 等仍需通过网格搜索确定。 |
| **未完全自动化** | 当前为静态配置，缺乏在线自适应调整机制。 |

---

### 🔮 未来工作方向

1. **自适应调参机制**：设计动态调整 $ x, \beta_2 $ 的策略，根据训练阶段自动优化。
2. **扩展至更多优化器**：将 LITE 应用于 AdEMAMix、Sophia 等新兴优化器。
3. **理论深化**：进一步分析 LITE 在非凸随机优化下的收敛速率。
4. **硬件友好实现**：探索 Kernel Fusion 等系统级优化，进一步降低 NS iteration 开销。

---

> **代码开源地址**：[https://github.com/SHUCHENZHU/LITE](https://github.com/SHUCHENZHU/LITE)

</details>

---

### 8. [Efficient Continual Learning in Language Models via Thalamically Routed Cortical Columns](https://arxiv.org/abs/2602.22479)

**Authors**: Afshin Khadangi  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.22479v1  

#### Abstract
Continual learning is a core requirement for deployed language models, yet standard training and fine-tuning pipelines remain brittle under non-stationary data. Online updates often induce catastrophic forgetting, while methods that improve stability frequently increase latency, memory footprint, or...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient Continual Learning in Language Models via Thalamically Routed Cortical Columns

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大型语言模型（LLMs）在部署后面临**持续学习**（continual learning）的挑战，即在非平稳数据流中快速适应新分布的同时，避免对已学知识的**灾难性遗忘**（catastrophic forgetting）。传统的微调方法（如 fine-tuning、adapter、LoRA）虽然降低了计算成本，但在长期序列更新中仍存在参数干扰和遗忘问题，且缺乏对**在线适应机制**的原生支持。

此外，现有方法通常将适应视为训练后的“附加”过程，而非模型架构的一部分，导致难以规模化、稳定化和公平比较。

### 提出的新方法：TRC²（Thalamically Routed Cortical Columns）
TRC² 是一种**专为持续学习设计的 decoder-only 架构**，其核心思想是将**稳定性**（stability）与**可塑性**（plasticity）在架构层面分离：

- **稀疏路由**（Sparse Routing）：通过类丘脑（thalamic）路由器，每 token 仅激活 $k$ 个皮层柱（cortical columns），实现局部计算，减少全局扰动。
- **快速修正路径**（Fast Corrective Pathway）：引入类小脑（cerebellar）低秩修正模块，支持基于部署数据的快速在线调整，而不重写主干慢速参数。
- **生物启发机制整合**：
  - 类皮层预测编码（predictive coding）
  - 兴奋-抑制门控（EI gating）
  - 联想记忆（associative memory，基于 Modern Hopfield）
  - 反馈路由优化（cortico-thalamic feedback）
  - 神经调节控制器（neuromodulator controller）

该架构实现了**块级稀疏性**（block-wise sparsity）和**chunk-parallel**执行，兼顾效率与灵活性。

### 相比现有方法的优势
| 维度 | TRC² 的优势 |
|------|-------------|
| **遗忘控制** | 将干扰控制内建于前向计算图中（通过路由、抑制、快权重），而非依赖外部正则化 |
| **在线适应能力** | 支持无需反向传播的快速在线更新（via fast-weight corrector） |
| **扩展性** | chunk-parallel 设计支持高效 kernel 实现，适合现代加速器 |
| **可分析性** | 各子系统可独立开关，便于消融研究和模块化设计 |

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据**：
  - `C4`（streaming 模式）：模拟非平稳部署环境下的文本演化
- **验证/评估数据**（作为任务流）：
  - `C4`（validation）
  - `WikiText-103-v1`：检测过拟合与泛化稳定性
  - `LAMBADA`：测试长程上下文预测能力

> 所有评估被视为一个**无明确任务边界的流式任务序列**，用于衡量持续学习中的遗忘与迁移。

### 实验设置
- **硬件**：单节点 4×NVIDIA V100（32GB），fp16 混合精度
- **批大小**：每卡 batch=8，梯度累积=4，全局 batch=128 sequences（131,072 tokens/step）
- **序列长度**：1024
- **优化器**：AdamW，lr=2e-4，cosine decay，warmup=1000 steps
- **总训练量**：约 2.88B tokens（22,000 optimizer steps）

### 评估指标
| 指标类型 | 具体指标 |
|--------|---------|
| **基础性能** | Perplexity（PPL）、BLEU 分数 |
| **效率** | Tokens/s（吞吐量）、Peak Memory（峰值内存）、Mem×Hour×GPU |
| **持续学习能力** | **Proxy Forgetting**：<br>• 对于 PPL：当前值 − 历史最优值（clip at 0）<br>• 对于 BLEU/Accuracy：历史最优值 − 当前值（clip at 0）<br>→ 报告平均遗忘（Avg Forgetting）及归一化 AUC |
| **其他** | Token Accuracy、Exact Match、ROUGE、chrF |

### 基线方法对比
- **Transformer**：标准 dense attention 架构
- **Mamba**：基于 SSM 的高效序列建模架构
- **TRC²**：本文提出的方法（参数量相近，约 169M）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Table 2）

#### 表 1：基础建模性能与效率对比

| Model | Params | PPL (C4) ↓ | PPL (Wiki) ↓ | PPL (LAM) ↓ | BLEU (C4) ↑ | BLEU (Wiki) ↑ | BLEU (LAM) ↑ | Tokens/s ↑ | Mem×Hour×GPU ↓ |
|-------|--------|------------|-------------|-------------|--------------|----------------|----------------|---------------|------------------|
| Transformer | 162M | 60.70 | 215.18 | 105.72 | 8.12 | 8.23 | 5.09 | ~127,000 | 118 GB·h |
| Mamba | 176M | 70.45 | 357.67 | 116.73 | 6.90 | 2.87 | 3.97 | ~108,000 | 178 GB·h |
| **TRC² (Ours)** | **169M** | **2.00** | **2.56** | **2.02** | **71.66** | **66.57** | **70.07** | **~57,000** | **268 GB·h** |

> 💡 **解读**：TRC² 在语言建模质量上显著优于基线（PPL 极低，BLEU 极高），但以**更高内存消耗和更低吞吐**为代价。

#### 表 2：持续学习表现（遗忘程度）

| Model | Avg Forgetting (Last Step) ↓ | Avg Forgetting (Normalized AUC) ↓ |
|-------|-------------------------------|------------------------------------|
| | PPL | tokenacc | BLEU | PPL | tokenacc | BLEU |
| Transformer | 0.0000 | 0.0014 | 0.3757 | 0.0669 | 0.0008 | 0.1684 |
| Mamba | 0.0000 | 0.0006 | 0.0900 | 0.3371 | 0.0011 | 0.1957 |
| **TRC² (Ours)** | **0.0110** | **0.0010** | **0.0435** | **0.0018** | **0.0008** | **0.0981** |

> ✅ **关键发现**：
> - TRC² 的 **normalized AUC 遗忘面积远低于所有基线**，说明其在整个训练流中保持历史性能的能力最强。
> - 尽管最后一步的 PPL 遗忘略高（0.0110 vs 0），但这表明模型仍在学习，而非“冻结”。

### 消融实验结果（文中提及，未列详细表格）
论文强调其架构支持**干净的消融研究**（clean ablations），各模块可独立启用/禁用。关键发现包括：
- **Topology-aware routing**：鼓励时间连续性，减少参数干扰。
- **EI Gating**：有效抑制不稳定激活传播。
- **Fast-weight corrector**：提供快速适应通道，不破坏慢速结构。
- **Routing refinement**（cortico-thalamic feedback）：提升路由质量，增强一致性。

---

## 4. 关键结论和发现

### 主要结论
1. **持续学习应作为架构属性**：TRC² 成功将“稳定性-可塑性权衡”从训练策略上升为**架构原生能力**，通过稀疏路由与快慢路径分离实现。
2. **低遗忘 ≠ 不学习**：TRC² 在保持历史性能方面优于基线（AUC 更优），同时仍能响应新数据变化，避免“冻结”陷阱。
3. **生物启发机制具有工程价值**：丘脑路由、皮层预测、小脑快权重等神经科学概念可转化为有效的深度学习组件。
4. **效率与性能的权衡**：TRC² 牺牲了吞吐量（~57k vs ~127k tokens/s）换取更强的持续学习能力，未来可通过 kernel 优化改善。

### 局限性
- **计算开销大**：由于 dense token-to-column projection 和双遍 cortex 执行，内存和延迟较高。
- **chunk-level 路由限制细粒度信号**：可能丢失 token 级动态。
- **路由器在剧烈分布偏移下可能变脆**：尚未测试极端 non-stationary 场景。
- **当前评估仍为中小规模**：需扩展至更大模型和更长上下文验证普适性。

### 未来工作方向
- 探索更大规模（larger scale）和更长上下文（longer context）下的表现。
- 提升路由器在强非平稳流中的鲁棒性（router robustness）。
- 将**纠正路径与部署约束结合**：使适应过程具备**可解释性、可逆性、边界可控性**，尤其应对噪声或对抗性输入。
- 进一步优化实现（kernel fusion、memory layout），缩小与 dense 模型的效率差距。

--- 

> 📌 **一句话总结**：  
> TRC² 提出了一种将**持续学习能力内建于架构之中**的新范式，通过类脑的稀疏路由、快慢路径分离与反馈机制，在显著降低累积遗忘的同时维持强大的语言建模能力，为构建真正“终身学习”的语言系统提供了新路径。

</details>

---

### 9. [Multilingual Safety Alignment Via Sparse Weight Editing](https://arxiv.org/abs/2602.22554)

**Authors**: Jiaming Liang, Zhaoxin Wang, Handing Wang  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.22554v1  

#### Abstract
Large Language Models (LLMs) exhibit significant safety disparities across languages, with low-resource languages (LRLs) often bypassing safety guardrails established for high-resource languages (HRLs) like English. Existing solutions, such as multilingual supervised fine-tuning (SFT) or Reinforceme...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Multilingual Safety Alignment Via Sparse Weight Editing*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在高资源语言（HRLs，如英语）中通常具备较强的安全对齐能力，但在低资源语言（LRLs）中往往存在显著的安全差距（safety disparities）。这种差距源于以下原因：
- 多语言安全数据稀缺，难以进行有效的多语言监督微调（SFT）或基于人类反馈的强化学习（RLHF）。
- 现有方法依赖昂贵的数据收集或复杂的训练流程，且跨语言迁移效果受限于翻译质量或参数更新开销。

该论文旨在解决**如何在不重新训练模型的前提下，将英语等高资源语言中的安全能力高效迁移到低资源语言中**的问题。

---

### 提出了什么新方法或新思路
作者提出了一种名为 **SPARSE WEIGHT EDITING** 的**无需训练（training-free）** 的安全对齐框架，其核心思想是：
- 基于“**稀疏安全神经元假设**”（Sparse Safety Localization），即安全能力集中在少量特定的“Safety Neurons”中。
- 将跨语言安全对齐建模为一个**受约束的线性变换问题**：通过计算一个稀疏权重扰动 $ \Delta W $，将 LRL 中有害输入的表示映射到 HRL（如英语）中已对齐的安全子空间。
- 引入**零空间投影约束**（null-space projection constraint），确保修改仅影响安全相关方向，而不损害通用能力（utility）。

该方法的关键创新在于：
- **闭式解（closed-form solution）**：通过数学推导直接求解最优 $ \Delta W $，避免迭代优化。
- **轻量级插件式干预**：只需少量锚点样本即可完成一次前向计算，适用于不同架构模型作为后处理模块。

---

### 相比现有方法的优势
| 方法类型 | 局限性 | 本文优势 |
|--------|------|--------|
| SFT / RLHF | 需要大量标注数据，训练成本高 | **无需训练**，节省计算资源 |
| Translation-based pipelines | 存在语义失真、推理延迟 | **无需翻译**，直接操作内部表示 |
| Task Arithmetic (e.g., RESTA) | 缺乏细粒度控制，易破坏通用能力 | **可解释性强**，精准编辑安全子空间 |
| Adapter-based transfer | 仍需微调，引入额外参数 | **参数不变**，仅做一次性权重编辑 |

> ✅ 总结：本方法实现了**数据高效、计算高效、即插即用**的多语言安全增强。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **MULTI-STRONGREJECT**：作者构建的多语言安全评测基准，由英文 `walledai/StrongREJECT` 经 `tencent/Hunyuan-MT-7B` 翻译而来，覆盖 8 种语言：
  - 英语（En）、中文（Zh）、越南语（Vi）、日语（Ja）、泰语（Th）、印尼语（Id）、孟加拉语（Bn）、希伯来语（He）
- 每种语言包含 313 条恶意查询，用于测试 jailbreak 攻击成功率。
- 安全神经元识别阶段使用翻译后的 `HarmfulQA`, `CatHarmfulQA`, `LLM-LAT` 和 `NaturalReasoning` 构造探针数据集。

---

### 实验设置和评估指标

#### 模型
在多个主流 LLM 家族上验证泛化性：
- **Llama-3.2**（1B, 3B）
- **Qwen2 / Qwen2.5**（0.5B ~ 7B）

#### 评估协议
- **严格零样本（strict zero-shot）**：对齐所用锚点数据与测试集完全隔离，无重叠。
- 所有测试均在未见过的目标语言提示上进行。

#### 评估指标
| 类别 | 指标 | 描述 |
|-----|------|------|
| **Safety** | Attack Success Rate (**ASR**) ↓ | 被判定为 unsafe 的响应比例（越低越好） |
| **Utility** | MGSM ↑<br>M-MMLU ↑ | 多语言数学推理与常识理解准确率（越高越好） |

---

### 基线方法对比
- **None**：原始未对齐模型
- **OUR**：本文提出的 SPARSE WEIGHT EDITING 方法
- **MPO**（Zhao et al., 2025b）：基于奖励间隙优化的多语言安全对齐方法（代表性的 SFT 类基线）
- **MPO + OUR**：组合方法，检验兼容性

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & D）

#### 在 Qwen2-0.5B 上的表现（典型小模型）
| 方法 | 平均 ASR ↓ | MGSM ↑ | M-MMLU ↑ |
|------|-----------|--------|----------|
| None | 28.27% | 7.75% | 32.71% |
| OUR | 15.42% (-12.85pp) | 5.27% | 31.01% |
| MPO | 19.52% (-8.75pp) | 4.80% | 32.69% |
| MPO+OUR | **12.53% (-15.74pp)** | 4.36% | 32.39% |

> 💡 观察：OUR 单独使用即优于 MPO；联合使用进一步提升安全性。

#### 在 Qwen2.5-7B 上的表现（大模型）
| 方法 | 平均 ASR ↓ | MGSM ↑ | M-MMLU ↑ |
|------|-----------|--------|----------|
| None | 6.80% | 32.00% | 49.37% |
| OUR | 4.68% (-2.12pp) | 31.56% | 49.19% |
| MPO | 5.00% (-1.80pp) | 38.36% | 47.16% |
| MPO+OUR | **3.52% (-3.28pp)** | 38.65% | 47.72% |

> ✅ 即使在较大模型上，也能带来稳定增益，且不影响 utility。

---

### 与基线方法的对比结果
- **OUR 显著降低 ASR**：在所有模型规模和语言中一致下降，尤其在 LRLs（如 Bn, He, Th）中改善最明显。
- **MPO+OUR 表现最佳**：表明本文方法可作为现有对齐技术的**互补插件**，而非替代品。
- **utility 几乎无损**：MGSM 和 M-MMLU 变化极小，说明通用能力得以保留。

---

### 消融实验结果（Ablation Study）

#### （1）安全神经元识别方式的影响（Table 2）
| 方法 | ASR ↓ | MGSM ↑ | M-MMLU ↑ |
|------|-------|--------|---------|
| None | 28.27% | 18.58% | 26.54% |
| Other (NeuroStrike probe-based) | 14.93% | 17.71% | 27.14% |
| MPO+Other | **12.53%** | 19.53% | 26.54% |

> 🔍 发现：即使更换神经元选择策略（如使用 probe classifier），性能依然良好 → 方法对 neuron selection 不敏感。

#### （2）锚点数据选择的影响（Table 3）
| 锚点配置 | ASR ↓ | MGSM ↑ | M-MMLU ↑ |
|--------|-------|--------|---------|
| UtilityAnchor only | 68.57% | **0.11%** ❌ | 24.21% |
| Regular only | 17.25% | 11.02% | 26.02% |
| Both (ours) | **17.53%** | 18.36% | 27.22% ✅ |

> ⚠️ 结论：必须平衡 harmful 与 harmless 锚点，否则会导致严重 utility 退化。

#### （3）低秩约束中的秩 $ r $ 影响（Table 4）
| Rank $ r $ | ASR ↓ | MGSM ↑ | M-MMLU ↑ |
|------------|--------|--------|----------|
| 4 | 15.42% | 18.33% | 27.19% |
| 8 | 15.54% | 17.96% | 27.19% |
| 16 | 15.34% | 18.18% | 27.18% |
| ... | ... | ... | ... |
| 512 | 16.17% | 18.11% | 27.19% |

> 📌 发现：只要 $ r \geq 4 $，性能就趋于饱和 → 安全更新存在于**极低维子空间**，支持低秩设计的有效性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **安全能力具有跨语言可迁移性**：英语中的“Safety Neurons”在一定程度上也参与其他语言的安全判断。
2. **表示错位是根本瓶颈**：简单放大激活无效，因 LRL 的有害表示方向偏离 HRL 的安全子空间。
3. **稀疏权重编辑可行且高效**：仅修改 <1% 参数即可实现显著安全提升。
4. **闭式解实用性强**：无需训练，单次计算即可部署，适合工业级应用。

---

### 方法的局限性
- 依赖高质量的锚点数据构造 $ Y_{\text{target}} $ 和 $ X_{\text{safe}} $。
- 当前仅作用于单层 MLP 的 safety neurons，未考虑多层协同机制。
- 对极端对抗性攻击（如隐写 jailbreak）的鲁棒性尚未验证。
- 假设 safety neurons 是静态固定的，可能忽略动态上下文依赖。

---

### 未来工作方向（原文建议）
1. **自动化超参数选择**：自适应确定 rank $ r $ 和正则化系数。
2. **扩展至多层 hierarchical editing**：构建层次化安全子空间以应对更复杂攻击。
3. **结合更强的多语言 evaluator**：提升评估信度。
4. **探索更精细的 anchor 构造策略**：例如基于语义聚类选取代表性样本。
5. **研究 safety direction transferability theory**：建立跨语言安全对齐的理论基础。

---

## 总结

✅ **SPARSE WEIGHT EDITING** 是一种新颖、高效、即插即用的多语言安全对齐方法：
- 利用稀疏神经元假设，将安全迁移转化为**带约束的低秩权重编辑问题**；
- 推导出**闭式解**，实现无需训练的一次性更新；
- 在多种模型和语言上显著降低 ASR，同时保持 utility 稳定；
- 可与现有方法（如 MPO）组合，形成更强的安全防护体系。

📌 该工作推动了从“训练驱动”向“编辑驱动”的安全对齐范式转变，为低成本、高效率的全球化 AI 安全部署提供了新路径。

</details>

---

### 10. [Sustainable LLM Inference using Context-Aware Model Switching](https://arxiv.org/abs/2602.22261)

**Authors**: Yuvarani, Akashdeep Singh, Zahra Fathanah, Salsabila Harlen, Syeikha Syafura Al-Zahra binti Zahari, Hema Subramaniam  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.22261v1  

#### Abstract
Large language models have become central to many AI applications, but their growing energy consumption raises serious sustainability concerns. A key limitation in current AI deployments is the reliance on a one-size-fits-all inference strategy where most systems route every request to the same larg...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sustainable LLM Inference using Context-Aware Model Switching

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大多数AI系统在处理用户请求时采用“**one-size-fits-all**”的推理策略，即无论查询复杂度如何，所有请求都路由到同一个大型语言模型（LLM）。这种做法导致了严重的**能源浪费**，尤其是在处理简单任务（如问候语、事实查询）时仍消耗大量计算资源。

此外，随着LLM部署规模扩大至每日数百万次交互，其**inference阶段的能耗和碳排放**已成为不可忽视的环境问题，而现有研究多关注训练成本，对推理效率优化的研究尚不充分。

---

### 提出的新方法与新思路
本文提出了一种**Context-Aware Model Switching**（基于上下文感知的模型切换）架构，实现动态选择适合查询复杂度的语言模型。该系统结合了多层次、混合式的路由机制：

- **Level 1: 缓存层（LRU Cache）**  
  对重复查询直接返回缓存结果，延迟低于0.1ms。
  
- **Level 2: 规则基础分类器（Rule-Based Classifier）**  
  使用96个预编译的正则表达式和关键词哈希集进行模式匹配，识别编程语法、数学符号等结构特征，分类延迟为0.1~1.0ms。

- **Level 3: 机器学习语义分类器（Semantic ML Classifier）**  
  利用`all-MiniLM-L6-v2`生成句子嵌入，并通过余弦相似度与预定义任务向量比较，判断查询意图。

- **User-Adaptive Component**  
  引入会话级自适应机制，根据历史交互调整复杂度阈值，提升个性化路由准确性。

最终基于综合复杂度评分（0–100），将请求分发至不同规模的模型：
- **Small**: Gemma3 1B（简单任务）
- **Medium**: Gemma3 4B（中等推理）
- **Large**: Qwen3 4B（复杂推理/代码生成）

---

### 相比现有方法的优势
| 方法 | 局限性 | 本工作的改进 |
|------|--------|-------------|
| **Cascade-based Routing**（如 FrugalGPT） | 顺序执行多个模型，引入额外延迟 | 单次决策路径，避免串行调用 |
| **Learned Routing**（如 RouteLLM） | 依赖偏好数据，跨域迁移能力弱 | 混合确定性+学习方法，增强泛化性 |
| **纯压缩/剪枝技术** | 性能损失大，需重新训练 | 不修改模型本身，仅优化调度逻辑 |
| **云端集中部署方案** | 高网络开销，隐私风险 | 完全本地化、开源模型部署，低延迟高可控 |

> ✅ **首次在完全本地、开源环境中验证混合路由策略的有效性**

---

## 2. 核心实验方法和设置

### 数据集
构建了一个包含 **150个提示词（prompts）** 的标准化评估数据集，均匀分布于三类复杂度：
- **Simple（50条）**：问候、单句事实问答、常识检索
- **Medium（50条）**：基本推理、多句解释、信息整合
- **Complex（50条）**：多步推理、结构化输出、代码编写

所有提示经过人工校验以确保类别清晰、无歧义。

---

### 实验设置
- **硬件平台**：
  - CPU: AMD Ryzen 7 5800H (8核16线程)
  - GPU: NVIDIA GeForce GTX 1650 Ti (4GB VRAM)
  - 内存: 32GB DDR4
  - OS: Windows 11 (64位)
- **软件栈**：
  - Python 3.10 + Ollama + PyTorch (CUDA 11.8)
  - 模型运行时：Ollama（支持本地模型加载与管理）
- **能量测量**：
  - 使用 **NVML GPU power telemetry** 获取实时功耗（W）
  - 每次推理的能量消耗 = 平均GPU功率 × 推理时间（秒），单位为 **Joules**
  - 碳排放估算：采用IEA全球平均碳强度 **475 gCO₂e/kWh**

---

### 评估指标
| 类别 | 指标 |
|------|------|
| **Efficiency（效率）** | - End-to-end latency（端到端延迟）<br>- Throughput (tokens/sec)<br>- Energy consumption per query (kJ)<br>- Estimated CO₂ emissions (gCO₂e) |
| **Effectiveness（有效性）** | - Routing Accuracy<br>- BERTScore F1（衡量响应质量保留率） |

---

### 基线方法对比
- **Baseline**：所有查询一律使用最大模型 **Qwen3 4B**，且常驻内存
- **Proposed Method**：提出的三层混合路由 + 动态模型加载（keep_alive: 0）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（vs. Baseline）

| 指标 | Baseline | Proposed | 提升幅度 |
|------|---------|----------|----------|
| **平均响应延迟** | 13.8 秒 | 3.5 秒 | ↓ **68%** |
| **吞吐量** | 25.4 tokens/s | 61.3 tokens/s | ↑ **141%** |
| **总能耗（150 queries）** | 84.2 kJ | 22.0 kJ | ↓ **67.5%** |
| **碳排放** | ~11.1 gCO₂e | ~2.9 gCO₂e | ↓ **67.5%** |
| **输出质量保留率（BERTScore F1）** | — | **93.6%** | 接近基准水平 |

> 🔍 能耗降低主要来自对简单/中等查询的高效分流；复杂查询虽仍需大模型，但避免了冗余执行。

---

### 分类性能（Routing Accuracy）
| 查询类型 | Recall | Precision | F1 Score |
|--------|--------|---------|----------|
| Simple | **98%** | 85% | 91% |
| Medium | 76% | 78% | 77% |
| Complex | 52% | **96.3%** | 68% |

- **Simple queries**：高召回表明系统能有效识别并快速处理简单任务
- **Complex queries**：低召回但高精度 → 采取保守升级策略，优先保障质量而非极致节能

---

### 消融实验与定性观察（Ablation & Stress Test）
- **缓存命中率随会话增长上升**，进一步降低平均开销
- **stress test** 显示系统在持续负载下稳定运行，未出现崩溃或路由失败
- **user-adaptive component** 在长对话中逐步调整阈值，提升了对特定领域（如技术问题）的路由准确率
- 尽管未提供完整消融量化（如移除某一层的影响），但从设计上看，**fast-path-first原则显著减少了昂贵语义分析的调用频率**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **模型切换可大幅降低能耗而不牺牲响应质量**：在保持 **93.6% BERTScore F1** 的前提下，实现 **67.5% 的能耗下降** 和 **68% 的延迟减少**。
2. ✅ **智能资源分配优于统一调度**：将计算资源按需分配给不同复杂度的任务，是可持续AI的关键路径。
3. ✅ **混合路由优于单一策略**：结合规则、缓存与轻量ML的方法，在效率、可解释性和泛化之间取得良好平衡。
4. ✅ **本地化部署同样可行**：整个系统可在消费级GPU上运行，无需专用硬件或云服务，具备广泛应用潜力。

---

### 方法的局限性
1. **评估范围有限**：
   - 仅测试了**对话式工作负载**，未涵盖摘要、翻译、检索等其他任务类型
   - 实验基于**单主机部署**，未考虑高并发或多节点场景下的扩展性
2. **质量评估依赖自动化指标**：
   - 使用 **BERTScore F1** 替代人类评估，可能忽略细微语义差异或创造性表达的质量变化
3. **stress test 结果为定性描述**，缺乏详细的量化分析（如P99延迟、QPS峰值）
4. **adaptive component 尚未长期验证**，其收敛速度和稳定性有待进一步研究

---

### 未来工作方向
1. **扩展至更多任务类型和模型族**（如视觉-语言模型、边缘设备部署）
2. **引入分布式架构支持**，探索在Kubernetes或微服务环境中的部署模式
3. **开发领域专用分类器**，通过fine-tuning提升特定垂直领域的路由精度
4. **结合carbon-aware scheduling**，根据电网碳强度动态调整模型选择策略
5. **集成更先进的early exit或sparse activation机制**，进一步提升细粒度控制能力

---

## 总结
> 🌱 **Sustainability ≠ Sacrifice**  
本文证明：通过合理的系统设计，可以在不显著牺牲性能的前提下，大幅提升LLM推理的能效。**Context-Aware Model Switching** 是迈向绿色AI的重要一步，尤其适用于高流量、异构查询的真实应用场景。

该工作不仅提出了一个高效的解决方案，还提供了可复现的本地化实验框架，为后续研究开辟了新的方向。

</details>

---

### 11. [Coarse-to-Fine Learning of Dynamic Causal Structures](https://arxiv.org/abs/2602.22532)

**Authors**: Dezhi Yang, Qiaoyu Tan, Carlotta Domeniconi, Jun Wang, Lizhen Cui, Guoxian Yu  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.22532v1  

#### Abstract
Learning the dynamic causal structure of time series is a challenging problem. Most existing approaches rely on distributional or structural invariance to uncover underlying causal dynamics, assuming stationary or partially stationary causality. However, these assumptions often conflict with the com...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Coarse-to-Fine Learning of Dynamic Causal Structures**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文聚焦于**动态因果结构学习**（dynamic causal structure learning）这一挑战性任务。传统方法通常假设时间序列的因果关系是静态的或部分动态的（如仅瞬时因果变化），难以捕捉现实中普遍存在的**完全动态因果关系**（fully dynamic causality），即**瞬时因果**（instantaneous）和**滞后因果**（lagged）关系均随时间连续演变。

这种动态性在交通、医疗、气象等领域尤为常见（如疾病传播路径随季节变化）。现有方法在处理此类复杂动态信号时面临两大挑战：
1. **效率与稳定性差**：频繁更新每个时间步的因果图导致模型昂贵且难以收敛。
2. **循环约束失效**：传统的 acyclic constraint（如 `hexp`, `hpoly`）在动态设置下易出现梯度爆炸或消失，影响优化稳定性。

---

### **提出的新方法：DyCausal**
作者提出了 **DyCausal** —— 一种**从粗到细**（coarse-to-fine）学习动态因果结构的框架，其核心思想是：
- 利用**滑动卷积窗口**（sliding convolutional window）在粗粒度时间窗口上编码因果模式；
- 再通过**线性插值**（linear interpolation）恢复每个时间步的精细因果图。

#### **关键技术与创新**
1. **Coarse-to-Fine 编码机制**
   - 在长度为 $K$、步长为 $S$ 的滑动窗口上使用 CNN 编码粗粒度因果状态 $Z_t$；
   - 通过并行解码策略生成粗粒度因果矩阵 $W_t$；
   - 对相邻 $W_t$ 进行**线性插值**，近似得到所有时间步的精细因果图，显著减少需直接学习的参数数量。

2. **基于矩阵范数缩放的 acyclic constraint：`hnorm`**
   - 改进传统的 log-det 约束 $h_{\text{log}} = -\log \det(\alpha I - W^\top W)$；
   - 引入 **1-范数归一化**：  
     $$
     h_{\text{norm}} = -\log \det\left(\alpha I - \frac{W^\top W}{\|W^\top W\|_1}\right) + d \log \alpha
     $$
   - 优势：
     - 始终可导且在优化空间内（避免因谱半径超限而需重训练）；
     - 满足 **E-stable**, **V-stable**, **D-stable** 三大稳定性准则；
     - 提升训练效率与鲁棒性。

3. **灵活适配多种模型**
   - 可扩展至非线性 SEM 和 ODE 模型；
   - 在 ODE 设置下可移除 acyclic constraint（因其不涉及瞬时循环）。

---

### **相比现有方法的优势**
| 维度 | DyCausal | 传统方法（如 DYNO, NTS-NO, DyCAST） |
|------|---------|-------------------------------|
| 因果动态性 | ✅ 完全动态（瞬时 + 滞后） | ❌ 静态或部分动态 |
| 效率 | ✅ 仅学习少量粗粒度矩阵 + 插值 | ❌ 每步独立学习，计算开销大 |
| 稳定性 | ✅ `hnorm` 约束稳定可导 | ❌ `hlog` 易出界，需调参重训 |
| 准确性 | ✅ 更高 TPR/F1，更低 SHD | ❌ 在动态场景下性能下降明显 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### **合成数据（Synthetic Data）**
- **生成方式**：
  - 使用 ER 模型生成随机 DAG 作为基础因果图；
  - 权重设为时间索引的正弦/余弦函数以模拟动态变化；
  - 采用三种生成模型：
    - **线性 SEM**：$X_t = W_t [X_t, X_{t-1}] + \epsilon_t$
    - **非线性 SEM**：多层感知机（MLP）建模非线性关系
    - **ODE 模型**：Lorenz system 模拟连续动力学
- **变量规模**：$d = 10, 20, 50, 80, 100, 200, 300$
- **时间长度**：$T = 10, 30, 50, 70, 90$

#### **真实世界数据（Real-world Datasets）**
| 数据集 | 描述 |
|-------|------|
| **CausalTime** | 包含天气、交通、医疗三个场景的时间序列 |
| **NetSim** | fMRI 血氧水平依赖信号（脑区连接） |
| **DREAM-3** | 基因表达水平时间序列（100基因） |
| **Phoenix** | 大规模基因调控网络（11,165基因，筛选34个TF） |
| **CausalRivers** | 河流流量与因果图（Random 5子集） |

---

### **实验设置与评估指标**

#### **评估指标**
- **TPR**（True Positive Rate）：正确识别的边比例
- **F1**：精确率与召回率的调和平均
- **SHD**（Structural Hamming Distance）：估计图与真实图之间的差异（越低越好）
- **AUROC**：用于无真实图的真实数据集（如 Phoenix）

#### **基线方法对比**
| 方法 | 类型 | 是否支持动态 |
|------|------|-------------|
| **DYNO** (DYNOTEARS) | Score-based + NOTEARS | ❌ 静态 |
| **NTS-NO** (NTS-NOTEARS) | 非参数 DBN + NOTEARS | ❌ 静态 |
| **DyCAST** | 动态 ODE 建模 | ⭕️ 部分动态（仅瞬时） |
| **CUTS+** | 高维稀疏数据专用 | ❌ 静态摘要因果 |
| **JRNGC** | Jacobian-based GC | ❌ 静态 |
| **PCMCI**, **SVAM**, **TECDI**, **NGM** | 其他代表性方法 | 多为静态 |

#### **超参数设置**
- **DyCausal**：
  - 卷积窗口大小 $K=2$（动态）、$K=10$（静态）
  - 步长 $S=4$（动态）、$S=5$（静态）
  - 学习率 $lr=0.005$，稀疏系数 $\beta=0.05$，剪枝阈值 $\delta=0.3$
- 所有方法在相同服务器运行（Ubuntu + RTX 3090），确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **合成数据结果（动态因果）**
| 时间点 | 方法 | TPR (%) | F1 (%) | SHD ↓ |
|--------|------|--------|--------|-------|
| $t=1$ | DyCausal | **89.54±6.98** | **91.43±5.60** | **7.40±5.39** |
|        | DyCAST | 25.98±10.33 | 29.51±10.67 | 45.43±7.21 |
|        | DYNO | 3.59±3.18 | 3.52±2.90 | 78.40±10.75 |
| $t=25$ | DyCausal | **90.93±6.01** | **94.38±3.90** | **4.10±2.88** |
|        | DyCAST | 50.88±11.29 | 61.12±10.91 | 23.00±7.63 |
| $t=50$ | DyCausal | **87.29±6.60** | **90.38±5.99** | **8.20±6.06** |

> ✅ **DyCausal 在所有时间点显著优于基线**，尤其在序列两端表现更稳健。

#### **大规模图扩展性测试（$d=300$）**
- TPR > 0.78，F1 > 0.82，证明其良好的**scalability**。

#### **真实世界数据（CausalTime）**
| 方法 | Traffic (F1) | AQI (F1) | Medical (F1) |
|------|--------------|----------|---------------|
| **DyCausal** | **54.81±2.51** | **63.16±1.23** | **56.86±1.35** |
| CUTS+ | 45.21±1.74 | 58.20±2.44 | 30.23±1.85 |
| JRNGC | 52.04±2.41 | 55.36±1.61 | 52.05±0.75 |

> ✅ DyCausal 在医疗数据上远超第二名，说明其对复杂动态系统的强适应能力。

#### **其他真实数据 AUROC 结果**
| Dataset | DyCausal | CUTS+ | NGM |
|--------|---------|--------|-----|
| NetSim ($d=15$) | **0.9391** | 0.7058 | 0.7499 |
| DREAM-3 ($d=100$) | **0.6806** | 0.6229 | 0.5629 |
| Phoenix ($d=34$) | **0.5472** | 0.5104 | 0.4809 |
| CausalRivers ($d=5$) | **0.7214** | 0.6072 | 0.4905 |

> ✅ 在所有真实数据集上均取得最佳 AUROC。

---

### **消融实验结果**

#### **Ablation 1：移除线性插值（w/o inter）**
| 方法 | TPR ($t=1$) | F1 ($t=1$) | SHD ($t=1$) |
|------|------------|------------|-------------|
| DyCausal | **90.35±6.62** | **91.42±4.39** | **3.20±1.40** |
| w/o inter | 75.67±22.20 | 72.98±23.78 | 12.50±13.86 |

> 🔍 线性插值显著提升精度与稳定性，有助于跨窗口信息融合。

#### **Ablation 2：替换为原始 `hlog` 约束**
- 使用 `hlog` 时：
  - 训练过程频繁超出优化空间，被迫降低学习率并重启；
  - 最终运行超过 60,000 轮迭代（远高于 DyCausal 的 ~30,000）；
  - 仍无法准确恢复因果图（冗余/缺失边多）；
- 使用 `hnorm`：
  - 约束值平稳下降至接近 0；
  - 收敛更快，结构更准确。

> 🔍 `hnorm` 是实现高效稳定训练的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **DyCausal 成功实现了对完全动态因果结构的高效、稳定学习**，突破了传统方法局限于静态或部分动态的限制。
2. ✅ **Coarse-to-Fine 设计有效平衡了效率与精度**：通过粗粒度建模 + 线性插值，大幅减少参数量与计算负担。
3. ✅ **`hnorm` 约束具备理论与实践双重优势**：满足稳定性标准，避免重训练，提升优化效率。
4. ✅ 在**合成与真实数据**上全面超越 SOTA 方法，尤其在动态性强、噪声大、维度高的场景中优势明显。
5. ✅ 方法具有良好的**可扩展性与鲁棒性**：适用于不同节点数、时间长度、噪声强度、滞后阶数等设置。

---

### **局限性**
1. **假设因果变化平滑**：当前设计依赖于因果矩阵随时间连续变化的前提；若存在突变（如突发事件），可能需要更小的窗口或引入跳跃检测机制。
2. **未处理不规则采样时间序列**：论文假设时间序列为规则采样，虽提及可用补全策略预处理，但未集成端到端解决方案。
3. **变量固定参与**：假设所有变量在整个时间段内始终存在，未考虑变量动态加入或退出的情形。

---

### **未来工作方向**
1. **扩展至 irregular time series**：结合 imputation 或 continuous-time modeling 处理缺失与不规则观测。
2. **支持变量动态增减**：构建动态变量参与的 causal system。
3. **引入因果机制切换建模**：识别结构性断裂点（regime shifts）并进行分段建模。
4. **应用于更多领域**：如金融风险传导、神经科学、气候系统建模等。

---

> **总结**：DyCausal 提供了一种新颖且实用的框架，首次将“从粗到细”的思想系统应用于动态因果发现，并通过 `hnorm` 约束解决了长期困扰该领域的 acyclicity 优化难题，为现实世界中的复杂动态系统分析提供了强有力的工具。

</details>

---

### 12. [FactGuard: Agentic Video Misinformation Detection via Reinforcement Learning](https://arxiv.org/abs/2602.22963)

**Authors**: Zehao Li, Hongwei Yu, Hao Jiang, Qiang Sheng, Yilong Xu, Baolong Bi, Yang Li, Zhenlong Yuan, Yujun Cai, Zhaoqi Wang  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.22963v1  

#### Abstract
Multimodal large language models (MLLMs) have substantially advanced video misinformation detection through unified multimodal reasoning, but they often rely on fixed-depth inference and place excessive trust in internally generated assumptions, particularly in scenarios where critical evidence is s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《FactGuard: Agentic Video Misinformation Detection via Reinforcement Learning》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前基于 **Multimodal Large Language Models (MLLMs)** 的视频虚假信息检测方法存在以下关键缺陷：
- 依赖**单次推理（single-pass inference）**，缺乏对不确定性的感知能力；
- 在证据稀疏或模糊的情况下，过度依赖模型内部生成的假设，导致**跨模态幻觉（cross-modal hallucination）**；
- 无法主动获取外部关键证据（如事实核查、特定视频片段分析），从而做出“自信但错误”的判断。

这些问题在现实世界中尤为严重，因为误导性视频往往通过剪辑、断章取义或结合真实背景传播虚假主张。

---

### **提出了什么新方法或新思路**
作者提出 **FactGuard** —— 一种基于 MLLMs 的**智能体式（agentic）视频虚假信息检测框架**，其核心思想是将验证过程建模为一个**不确定性感知、迭代决策的过程**。

#### 主要创新点包括：
- ✅ **Agentic Reasoning Pipeline**：引入多轮推理机制，模型可自我评估任务难度，并在必要时调用外部工具补充证据。
- ✅ **Selective Tool Invocation**：设计两个专用工具：
  - `FactProbe`：用于检索外部知识（如维基百科、新闻网站）以验证事实主张；
  - `ClipScout`：聚焦视频中的关键时间片段进行视觉证据提取。
- ✅ **两阶段训练策略**：
  1. **Agentic Chain-of-Thought Supervised Fine-Tuning (SFT)**：构建带有工具使用意图和证据整合路径的标注推理轨迹，注入结构化推理行为；
  2. **Decision-aware Reinforcement Learning (RL)**：采用 **Group Relative Policy Optimization (GRPO)**，优化工具使用效率、风险敏感决策与证据驱动推理。
- ✅ **风险感知奖励函数**：在 RL 中显式建模**非对称误差成本（asymmetric error costs）**，支持调节 precision-recall 权衡（例如更重视避免误删 vs 避免漏检）。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | FactGuard |
|------|--------|----------|
| 推理方式 | 单次前向推理 | 多轮迭代、自省式推理 |
| 不确定性处理 | 忽略或隐式处理 | 显式识别并触发工具调用 |
| 证据来源 | 仅限输入模态 | 可动态接入外部知识与局部视觉证据 |
| 决策可控性 | 固定阈值 | 支持风险偏好调节（如高精度/高召回） |
| 可解释性 | 黑箱输出 | 完整记录推理链 + 工具调用依据 |

> ➤ **本质提升**：从“被动分类器”转变为“主动调查员”。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在三个主流视频虚假信息检测基准上进行评估：
- **FakeSV**：来自社交媒体的短新闻视频，含丰富社交上下文；
- **FakeTT**：TikTok 平台视频，强调多模态信号（视觉、音频转录、用户互动）；
- **FakeVV**：更具挑战性的复杂案例，强调推理需求。

所有数据集均按时间划分（最近15%作为测试集），确保评估符合现实部署场景。

---

### **实验设置与评估指标**
- **模型基础**：基于 `Qwen2.5-VL-7B` 构建；
- **训练流程**：
  1. **SFT阶段**：使用教师模型（Qwen2.5-VL-72B）生成高质量 agentic CoT 数据，经双层过滤后微调；
  2. **RL阶段**：使用 GRPO 进行强化学习，每条样本生成 8 条候选轨迹，KL 正则系数 β=0.04；
- **硬件配置**：8×NVIDIA H100 GPU；
- **最大上下文长度**：16,384 tokens。

#### **评估指标**
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1-score

此外还进行了：
- **可解释性分析**：使用 GPT-4o 自动评估推理质量（忠实性、逻辑一致性、洞察力）；
- **成本敏感分析**：调整 α:γ 比例控制假阳性与假阴性惩罚；
- **消融实验**：验证各组件贡献。

---

### **基线方法对比**
分为三类基线：
1. **判别式模型（Discriminative Models）**：
   - BERT、TikTec、FANVM、SV-FEND、FakingRec
2. **零样本 MLLMs（Zero-shot MLLMs）**：
   - Gemini2-thinking、GPT-4o、GPT-o1-mini、Qwen2.5-VL 系列、InternVL2.5、DeepSeek-R1
3. **任务对齐推理模型（Task-Aligned Reasoning Models）**：
   - Fact-R1（此前SOTA）、FactGuard（本文方法）

> 注：无原生视频支持的模型使用文本描述替代。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| Model | FakeSV (Acc/F1) | FakeTT (Acc/F1) | FakeVV (Acc/F1) |
|-------|------------------|------------------|------------------|
| FakingRec (best discriminative) | 69.5 / 70.0 | 71.0 / 72.0 | 72.1 / 72.0 |
| GPT-4o (best zero-shot MLLM) | 66.6 / 64.9 | 57.9 / 60.2 | 56.0 / 44.3 |
| Fact-R1 (prior SOTA) | 75.6 / 74.7 | 74.4 / 72.7 | 81.2 / 80.3 |
| **FactGuard (Ours)** | **79.3 / 81.4** | **75.3 / 75.2** | **83.0 / 83.9** |

> ✅ 在所有数据集上实现 **state-of-the-art 性能**，显著超越最强基线（Fact-R1）。

---

### **与基线方法的对比结果**
- 相比最强判别模型 **FakingRec**，FactGuard 在 FakeSV 上准确率提升 **+9.8%**，F1 提升 **+11.4%**；
- 相比最强零样本 MLLM **GPT-4o**，提升更为显著（如 FakeVV 上 Acc ↑27.0%，F1 ↑39.6%）；
- 相比同源推理模型 **Fact-R1**，仍取得稳定增益（平均 Acc ↑+3.7%）；
- 特别是在 **FakeVV** 这类需要深度推理的数据集上优势最明显，说明其**更强的复杂推理与证据整合能力**。

---

### **消融实验结果（见 Table 3）**

| 模型变体 | FakeSV (Acc/F1) | FakeTT (Acc/F1) |
|---------|------------------|------------------|
| Full FactGuard | 79.3 / 81.4 | 75.3 / 75.2 |
| w/o SFT | 73.1 / 74.7 | 71.6 / 68.2 |
| w/o RL | 62.0 / 63.2 | 67.9 / 65.7 |
| w/o Rtool（移除工具奖励） | 77.7 / 78.5 | 74.7 / 72.9 |
| w/o Rrisk（移除风险奖励） | 78.5 / 78.9 | 74.9 / 73.8 |
| Base (Qwen2.5-VL-7B + tools) | 57.6 / 60.6 | 56.8 / 55.4 |

#### 关键发现：
- **RL 最关键**：去掉 RL 导致性能暴跌（↓17.3% Acc on FakeSV），说明单纯加工具不足以形成有效推理策略；
- **SFT 是基础**：没有 SFT 的引导，RL 难以收敛到合理策略（policy collapse）；
- **工具与风险奖励均有贡献**：二者分别提升证据利用效率与决策稳健性；
- **端到端训练必要**：直接在 base model 上加工具效果有限（Base → FactGuard: +21.7% Acc）。

---

## **4. 关键结论和发现**

### **主要发现**
1. 🔍 **迭代式、工具增强的推理优于单次推理**：FactGuard 能在模糊情况下主动寻求外部证据，避免“盲目自信”；
2. 🧠 **结构化训练至关重要**：先通过 SFT 注入推理模式，再通过 RL 优化策略，形成协同效应；
3. ⚖️ **风险可调控性具有实际意义**：通过调整 α:γ 比例可在 precision 和 recall 间灵活权衡（见 Table 2），适用于不同应用场景（如内容审核需高 precision，预警系统需高 recall）；
4. 💬 **可解释性强**：完整记录 `<think>` 推理链与 `<tool>` 调用过程，便于人工审计与调试。

---

### **方法的局限性**
- **工具调用开销大**：每次调用 FactProbe 或 ClipScout 增加延迟与计算成本，不适合实时高频场景；
- **依赖工具质量**：若搜索引擎返回噪声或 ClipScout 抽帧不准确，会影响最终判断；
- **泛化边界待验证**：目前仅在英文短视频场景验证，跨语言、长视频、直播等场景尚未覆盖；
- **无法完全消除幻觉**：尽管大幅降低，但在极端误导下仍可能出现错误工具查询或误读结果。

---

### **未来工作方向**
- 扩展至更多工具类型（如语音情感分析、元数据分析、水印检测）；
- 引入 **Monte Carlo Tree Search (MCTS)** 或 **Bayesian reasoning** 实现更高效的探索策略；
- 探索轻量化代理架构，适应移动端或边缘设备部署；
- 构建更大规模的 agentic CoT 数据集，推动社区发展；
- 将框架推广至其他多模态任务（如医学影像报告生成、法律证据分析）。

---

## ✅ 总结一句话
> **FactGuard 首次将 agentic reasoning 与 decision-aware RL 结合，实现了从“静态分类”到“动态调查”的范式跃迁，在视频虚假信息检测任务中达到 SOTA 性能，同时具备高可解释性与风险可控性。**

</details>

---

### 13. [Discourse-Aware Dual-Track Streaming Response for Low-Latency Spoken Dialogue Systems](https://arxiv.org/abs/2602.23266)

**Authors**: Siyuan Liu, Jiahui Xu, Feng Jiang, Kuang Wang, Zefeng Zhao, Chu-Ren Huang, Jinghang Gu, Changqing Yin, Haizhou Li  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.23266v1  

#### Abstract
Achieving human-like responsiveness is a critical yet challenging goal for cascaded spoken dialogue systems. Conventional ASR-LLM-TTS pipelines follow a strictly sequential paradigm, requiring complete transcription and full reasoning before speech synthesis can begin, which results in high response...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Discourse-Aware Dual-Track Streaming Response for Low-Latency Spoken Dialogue Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在级联式（cascaded）**ASR-LLM-TTS** 流程的口语对话系统中，传统方法遵循严格的顺序执行模式：必须等待完整的语音识别（ASR）输出后，再进行大语言模型（LLM）推理，最后才启动语音合成（TTS）。这种设计导致**响应延迟过高**，难以满足人类自然对话所需的亚秒级（sub-second）响应要求。

该论文旨在解决这一“**响应起始困境**”（response-onset dilemma），即如何在不牺牲模块化优势的前提下，显著降低感知延迟。

---

### 🚀 提出的新方法：DDTSR 框架
作者提出 **Discourse-Aware Dual-Track Streaming Response (DDTSR)** 框架，其核心思想是模仿人类“边思考边说话”（speaking while thinking）的行为，通过三个关键机制实现低延迟响应：

#### （1）**Connective-Guided Small-Large Model Synergy**  
将响应生成解耦为两个角色：
- **轻量级小模型**（small model）：负责生成语义安全、无需深层推理的**话语连接词**（discourse connectives），如 “Hmm”, “Well”, “I see” 等。
- **大型LLM**（large model）：并行执行知识密集型的完整语义推理。
> 小模型可提前发声，避免空等，而大模型继续后台处理主内容。

#### （2）**Streaming-Based Cross-Modal Collaboration**  
打破传统 ASR→LLM→TTS 的串行流程，采用**流式跨模态协作**：
- 在 ASR 输出尚未完成时，小模型即可基于部分转录文本（partial transcript）预测连接词。
- 实现“听的同时思考”（listen while thinking）和“说的同时思考”（speak while thinking）。

#### （3）**Curriculum-Learning-Based Discourse Continuity Enhancement**  
确保早期发出的连接词与后续大模型生成的内容保持**话语连贯性**（discourse continuity）：
- 引入基于课程学习（curriculum learning）的联合训练策略，使小模型适应从完整到截断输入的渐进挑战。
- 设计风格一致性损失（style consistency loss）、连贯性损失（coherence loss）和先验正则化损失（prior regularization loss）。
- 推理阶段引入**置信度驱动的发射策略**（confidence-based emission policy），仅当连接词足够可靠时才提前输出。

---

### 🔍 相比现有方法的优势
| 维度 | DDTSR | 传统级联系统 | End-to-End 模型 |
|------|-------|--------------|----------------|
| **延迟** | 显著降低（19%-51%） | 高延迟（>1s） | 较低但不可控 |
| **模块化** | 支持插件式部署 | 高度模块化 | 黑箱、难替换组件 |
| **可控性** | 可独立优化各模块 | 高 | 低 |
| **实用性** | 兼容多种 LLM backbone | 是 | 否 |
| **训练成本** | 低（仅微调小模型） | 中等 | 极高 |

> ✅ DDTSR 是一个**即插即用**（plug-and-play）模块，无需修改现有 ASR/TTS 模块，适用于真实场景部署。

---

## 2. 核心实验方法和设置

### 📚 数据集
在两个主流口语对话基准上进行评估：
- **SD-Eval** [Ao et al., 2024]：日常口语对话数据集，涵盖口音、年龄、环境多样性。
- **SpokenNativQA** [Alam et al., 2025]：多语言日常问答数据集，本文使用其英文子集（en）。

| 数据集 | 子集 | 样本数 |
|--------|------|--------|
| SD-Eval | test-acc / test-age / test-env | 1,236 |
| SpokenNativQA | en | 2,322 |
| **总计** | — | **3,558** |

数据划分为 8:1:1 的训练/验证/测试集。

---

### ⚙️ 实验设置
- **小模型**：`Qwen3-0.6B` 微调用于连接词生成。
- **大模型**：`Qwen3-8B` 和 `Qwen3-32B`（通过 API 调用，解耦重推理负载）。
- **ASR**：基于 `sherpa-onnx` 的流式语音识别。
- **TTS**：`Cosy Voice2` 实现增量语音合成。
- **部署**：小模型、ASR、TTS 本地运行于单张 RTX 4090D GPU；大模型远程调用。

---

### 📊 评估指标

#### **延迟导向指标（Latency-Oriented Metrics）**
| 指标 | 定义 |
|------|------|
| **Perception Latency** | 用户说完最后一段音频到系统开始正常响应的时间（排除输入长度影响） |
| **Reaction Latency** | 系统开始响应到第一个合成语音块发出的时间（含 LLM 推理 + 初始 TTS） |
| **Waiting Latency** | 前两者之和，反映用户感知的整体延迟 |

> 注：报告“最优”（Opt.，含连接词预填充）和“剩余”（Rem.，不含）两种情况。

#### **质量导向指标（Quality-Oriented Metrics）**
| 指标 | 定义 |
|------|------|
| **Textual Coherence** | 使用 G-Eval 协议由 LLM-as-a-judge 评估逻辑一致性和话语连贯性 |
| **Speech Naturalness** | 使用 UTMOSv2 自动评估合成语音自然度（与人类评分高度相关） |

---

### 🔁 基线方法对比
#### **End-to-End 模型**
- Doubao-Realtime
- GLM-Realtime
- Qwen3-Omni-Flash-Realtime

#### **级联式 Baseline**
- **SSC**（Standard Single-Stream Cascade）：标准流式级联系统（基于 FunAudioLLM 修改）
- **SDC**（Standard Dual-Stream Cascade）：双轨并行小大模型，但无跨模态交错或早发机制

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2）

| 方法 | 数据集 | 大模型 | Waiting Latency (Avg.) | Reduction vs SSC |
|------|--------|--------|-------------------------|------------------|
| SSC | SD-Eval | — | 1003 ms | — |
| SDC | SD-Eval | 8B | 861 ms | 14.2% ↓ |
| **DDTSR** | **SD-Eval** | **—** | **548 ms** | **45.4% ↓** |
| SSC | SpokenNativQA | — | 931 ms | — |
| SDC | SpokenNativQA | 8B | 854 ms | 8.3% ↓ |
| **DDTSR** | **SpokenNativQA** | **8B** | **741 ms** | **20.4% ↓** |

> ✅ 总体延迟降低 **19%-51%**，首次将级联系统带入**亚秒响应区间**（平均 < 600ms）。

---

### 🔬 分项延迟分析
- **Perception Latency**：
  - SSC/SDC：约 388 ms
  - **DDTSR**：降至 **~100 ms**（最低达 89 ms）
  > 归因于流式 ASR 与小模型推理重叠，实现“听中思考”。

- **Reaction Latency**：
  - SSC（SD-Eval, 8B）：615 ms
  - **DDTSR**：435 ms（↓29.3%）
  > 小大模型并行执行有效缓解“思考完再说”的瓶颈。

---

### ✅ 输出质量保持
尽管延迟大幅下降，**话语质量和语音自然度未受损**（Table 3）：

| 方法 | Text Consistency / Coherence | UTMOSv2 |
|------|-------------------------------|--------|
| Gold Standard | 4.77 / 4.43 | — |
| SSC (8B) | 4.76 / 4.54 | 3.16 |
| **DDTSR (8B)** | **4.77 / 4.49** | **3.12** |

> ✔️ 与基线相当，甚至优于部分方法，说明连接词拼接不会破坏自然性。

---

### 📊 输入长度的影响（Figure 3）
随着输入音频变长（0-3s → 6-9+s）：
- DDTSR 的 **perception latency 几乎不变**（稳定 ~100ms），而基线明显上升。
- **waiting latency 改善比例从 38.3% 提升至 58.3%**，表明 DDTSR 在长交互中增益更大。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **人类启发的有效性**：模拟“边说边想”的行为可通过**话语连接词**作为低风险缓冲，显著提升系统响应速度。
2. **时间重组优于计算加速**：相比单纯优化模型推理速度，**重构级联系统的时间流程**（temporal reorganization）更能根本性降低感知延迟。
3. **即插即用性强**：DDTSR 不依赖特定模型架构，兼容不同规模 LLM（8B/32B）和主流 ASR/TTS 模块。
4. **鲁棒性好**：对不同输入长度、语境变化均表现稳定，尤其适合多轮、长对话场景。

---

### ⚠️ 局限性
1. **依赖连接词出现频率**：在连接词稀少的数据集（如 SpokenNativQA 中仅 38% 含 turn-initial connective）中，早发机会有限。
2. **文化/语言差异未充分探索**：当前连接词定义基于英语语料，跨语言泛化能力需进一步验证。
3. **极端短句收益较小**：对于极短输入（<1s），延迟压缩空间有限。

---

### 🔮 未来工作方向
1. 扩展至更多语言和文化背景下的连接词建模。
2. 结合语音韵律线索（如 pause detection）动态调整连接词选择。
3. 探索更复杂的双轨融合策略（如语音风格平滑过渡）。
4. 将 DDTSR 应用于全双工（full-duplex）对话系统，支持打断与即时反馈。

---

## 总结
> **DDTSR 成功将心理学中的“边说边想”机制工程化，提出了一种高效、实用、可扩展的低延迟口语对话框架。它不仅显著降低了响应延迟（19%-51%），还保持了高质量的话语连贯性与语音自然度，为构建真正类人交互的语音代理提供了新范式。**

🔗 代码已开源：[https://github.com/hlt-cuhksz/DDTSR](https://github.com/hlt-cuhksz/DDTSR)

</details>

---

### 14. [Energy Efficient Federated Learning with Hyperdimensional Computing (HDC)](https://arxiv.org/abs/2602.22290)

**Authors**: Yahao Ding, Yinchao Yang, Jiaxiang Wang, Zhonghao Liu, Zhaohui Yang, Mingzhe Chen, Mohammad Shikh-Bahaei  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.22290v1  

#### Abstract
This paper investigates the problem of minimizing total energy consumption for secure federated learning (FL) in wireless edge networks, a key paradigm for decentralized big data analytics. To tackle the high computational cost and privacy challenges of processing large-scale distributed data with c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Energy Efficient Federated Learning with Hyperdimensional Computing (HDC)》总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对**无线边缘网络中安全联邦学习（FL）的高能耗问题**，提出了一种综合解决方案。传统 FL 面临两大挑战：
- **高计算成本**：基于 Neural Networks (NN) 的模型在资源受限的边缘设备上训练开销大；
- **隐私泄露风险**：标准 FL 更新可能被用于梯度反演攻击，暴露本地敏感数据。

此外，现有研究大多孤立地优化能量、隐私或模型效率，缺乏对 **HDC 模型维度、通信与计算资源进行联合建模与协同优化**的统一框架。

### 提出了什么新方法或新思路
作者提出了一个名为 **FL-HDC-DP** 的新型框架，其核心是将三种关键技术融合：
- **Hyperdimensional Computing (HDC)**：作为轻量级本地训练模型，利用高维向量操作（如 bundling、binding）实现低功耗计算；
- **Differential Privacy (DP)**：通过添加高斯噪声（采用 zCDP 机制）保护上传的模型更新，提供严格的数学隐私保障；
- **Joint Optimization**：首次将 **HDC 维度 $d$**、**传输功率 $p_i$** 和 **CPU 频率 $f_i$** 进行联合优化，以最小化总能耗。

### 相比现有方法的优势
- **能效显著提升**：相比固定资源分配或非联合优化方案，实现了高达 **83.3% 的总能量减少**；
- **兼顾准确性与收敛速度**：尽管引入了 DP 噪声，仍能在目标精度（如 MNIST 上 88%）下快速收敛；
- **适用于资源受限场景**：特别适合电池供电的 IoT 设备等边缘节点；
- **系统级协同设计**：突破了以往仅优化单一维度（如带宽或频率）的局限，实现了跨层联合优化。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MNIST**：手写数字图像数据集，用于分类任务验证。
- 每个用户持有部分独立同分布（IID）数据，共 $U=50$ 用户参与训练。

### 实验设置和评估指标

#### 系统参数
| 参数 | 值 |
|------|-----|
| 用户数 $U$ | 50 |
| 总带宽 $B$ | 1 MHz |
| 最大传输功率 $P_{\text{max}}$ | 0.1–1 W |
| 噪声谱密度 $N_0$ | -174 dBm/Hz |
| CPU 最大频率 $f_{\text{max}}$ | 2.3 GHz |
| HV 维度范围 $d$ | 3000–10000（步长1000） |

#### 评估指标
- **总能量消耗 $E_{\text{total}}$**：所有用户在整个 FL 过程中的计算与通信能耗之和；
- **完成时间 $T$**：每轮最大允许执行时间（约束条件）；
- **收敛轮数 $J_a$**：达到目标准确率所需的全局通信轮次；
- **准确率**：测试集上的分类准确率（设定为 88%）；
- **隐私预算 $(\epsilon, \delta)$**：设为 $(25, 10^{-5})$，使用 zCDP 保证隐私。

#### 基线方法对比
1. **Fixed $f = f_{\text{max}}$**：CPU 频率固定为最大值，仅优化其他变量；
2. **Fixed $p = P_{\text{max}}$**：传输功率固定为最大值；
3. **Fixed $d = 3000$ / $d = 5000$**：HDC 维度固定，不进行维度选择。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在总时间限制 $T = 30s$ 下，最优 HDC 维度为 **$d^* = 4000$**；
- 此时所需收敛轮数 $J_a = 20$，远低于 $d=3000$ 时的 39 轮；
- 总能量消耗最低，约为 **~1000 J**（具体数值依图估算），而基线高达 ~6000 J。

### 与基线方法的对比结果
| 对比项 | 能量节省幅度 |
|--------|--------------|
| vs. Fixed $f = f_{\text{max}}$ | **83.3%** |
| vs. Fixed $p = P_{\text{max}}$ | **31.5%** |
| vs. Fixed $d = 3000$ | **54.9%** |
| vs. Fixed $d = 5000$ | **50.0%** |

> 图 2 显示：能量随维度呈“U”型曲线，在 $d=4000$ 处取得最小值——说明存在**最优维度平衡点**：过低则需更多通信轮次；过高则单轮计算/通信开销过大。

> 图 3 表明：随着总时间 $T$ 放宽，所有方案能耗下降，但所提方法始终领先，且优势在严格延迟约束下更明显。

### 消融实验结果（隐含分析）
虽然未明确标注“ablation study”，但从不同维度和资源配置的对比中可得出以下消融式结论：
- **HDC 维度选择最关键**：相比单纯调整 $p$ 或 $f$，选择合适的 $d$ 对节能影响最大；
- **联合优化优于单点优化**：任何单一资源固定都会导致次优解；
- **通信瓶颈效应明显**：当 $T$ 较小时，必须提高 $p_i$ 和 $f_i$ 来满足时限，导致能耗飙升。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **HDC 是边缘 FL 的理想候选模型**：因其低复杂度向量运算，天然适配资源受限设备；
2. **存在最优 HDC 维度 $d^*$**：并非越高越好，需在“降低收敛轮数”与“增加单轮开销”之间权衡；
3. **联合优化至关重要**：只有同时优化 $d$, $p_i$, $f_i$ 才能达到全局最优能效；
4. **所提混合算法高效可行**：外层枚举 $d$ + 内层一维搜索 $(f_i, p_i)$，复杂度仅为 $O(DUL)$，适合大规模部署。

### 方法的局限性
- **维度枚举依赖仿真**：$J_a(d)$ 关系无法解析表达，需通过 Monte Carlo 仿真预先获取；
- **假设信道状态已知**：未考虑动态信道变化下的自适应调度；
- **数据分布假设较理想**：实验基于 IID 数据，实际中 Non-IID 场景可能影响收敛行为；
- **仅考虑上行传输**：未涉及下行广播能耗。

### 未来工作方向
- 将框架扩展至 **Non-IID 数据场景** 和 **异构设备环境**；
- 探索 **在线维度自适应机制**，根据实时反馈动态调整 $d$；
- 结合 **Federated Split Learning** 或 **Model Pruning** 进一步压缩通信负载；
- 在真实硬件平台（如 HDC 加速器 [10]）上实现并验证能效增益；
- 探索与 **6G 中 Large AI Models + HDC 融合路径** 的结合潜力 [12]。

--- 

> ✅ **一句话总结**：本文提出的 **FL-HDC-DP** 框架通过联合优化 HDC 维度与系统资源，在保证隐私和精度的前提下，实现了高达 **83.3% 的能量节约**，为绿色、安全的边缘智能提供了新范式。

</details>

---

### 15. [Toward Expert Investment Teams:A Multi-Agent LLM System with Fine-Grained Trading Tasks](https://arxiv.org/abs/2602.23330)

**Authors**: Kunihiro Miyazaki, Takanobu Kawahara, Stephen Roberts, Stefan Zohren  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.23330v1  

#### Abstract
The advancement of large language models (LLMs) has accelerated the development of autonomous financial trading systems. While mainstream approaches deploy multi-agent systems mimicking analyst and manager roles, they often rely on abstract instructions that overlook the intricacies of real-world wo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Toward Expert Investment Teams: A Multi-Agent LLM System with Fine-Grained Trading Tasks

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **LLM 的多智能体交易系统**虽然模仿了投资分析师和经理的角色分工，但通常采用**粗粒度（coarse-grained）的任务指令**，例如简单地让“基本面分析代理”去“分析财务报表”。这种抽象指令忽略了真实投资流程中的复杂性和细节，导致两个核心问题：
- **性能下降**：LLM 在面对模糊、复杂的任务时容易推理中断或输出质量降低。
- **可解释性差**：仅能看到最终决策，无法追溯中间推理过程，在资产管理等高风险场景中难以实际部署。

### 提出了什么新方法或新思路
本文提出了一种**细粒度任务分解的多智能体 LLM 交易框架**（fine-grained task decomposition），其核心思想是：
- 将投资分析任务显式拆解为多个**标准操作流程（SOP-like）的子任务**，每个 Agent 接收具体、明确的操作指南。
- 以现实世界投资团队的工作流为蓝本，构建一个**三层层级化决策架构**（Hierarchical Decision-Making Process）：
  - **Level 1**：四个专家 Agent 分别执行量化（Quantitative）、定性（Qualitative）、新闻（News）、技术（Technical）分析；
  - **Level 2**：Sector Agent 和 Macro Agent 进行行业与宏观层面调整；
  - **Level 3**：PM Agent 综合所有信息做出最终投资组合构建。

该方法强调 **prompt design 的精细化**，而非仅仅增加 agent 数量或改进模型本身。

### 相比现有方法的优势
- ✅ **提升性能**：相比粗粒度设计，显著提高了风险调整后收益（Sharpe Ratio）。
- ✅ **增强可解释性**：通过细粒度 prompt 引导出更专业的术语和清晰的推理路径，便于审计与验证。
- ✅ **改善信息流动**：实验证明，细粒度设计能有效促进底层信号（尤其是 Technical 分析）向高层决策者的传递。
- ✅ **实用性强**：结合真实市场数据、防数据泄露机制、portfolio optimization 验证，贴近工业级应用需求。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验基于日本股市（Japanese equity market），涵盖以下四类数据源：
| 数据类别 | 具体来源 |
|--------|--------|
| **股票价格数据** | Yahoo Finance 提供的 TOPIX 100 成分股日频收盘价 |
| **财务报表数据** | 日本金融厅（FSA）的 EDINET API 获取季度/半年报/年报 |
| **新闻数据** | Ceek.jp News 聚合 Nikkei、Reuters、Bloomberg（日文版）的标题与摘要 |
| **宏观经济数据** | FRED 和 Yahoo Finance 提供的美日利率、通胀、就业、汇率、波动率指数等 |

> 所有数据均经过严格的时间对齐处理，确保无前瞻偏差（look-ahead bias）。

### 实验设置和评估指标

#### 回测设置（Backtesting）
- **投资标的**：TOPIX 100 大盘股
- **策略类型**：月度调仓的多空策略（long-short），市场中性（equal long/short positions）
- **回测周期**：2023年9月 – 2025年11月（共27个月），**晚于 GPT-4o 的知识截止日期（2023年8月）**，避免记忆效应干扰
- **模型选择**：GPT-4o（temperature=1），使用 median aggregation 减少随机性影响

#### 评估指标
| 类型 | 指标说明 |
|------|---------|
| **定量指标** | Sharpe Ratio（月度收益均值 / 标准差）为主，衡量风险调整后回报 |
| **定性指标** | 分析各 Agent 输出文本的内容特征（如关键词分布、语义相似度）以评估可解释性与信息传播效率 |

### 基线方法对比
- **主对比实验**：  
  - **Fine-grained setting**：提供预计算的技术指标（如 RoC、RSI、MACD）和财务比率（如 ROE、P/E）作为输入  
  - **Coarse-grained setting**：直接输入原始价格序列或原始财务数值（如收入、净利润），由 LLM 自主提取特征
- **消融实验（Ablation Study）**：逐一移除某一类 Agent（leave-one-out），观察性能变化
- **组合优化实验**：将多个 agent 策略合成一个 composite portfolio，并与 TOPIX 100 指数进行资产配置优化测试

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）细粒度 vs 粗粒度任务的整体表现（图2）
- 在不同投资组合规模（N=10~50）下，**细粒度设计在绝大多数情况下显著优于粗粒度设计**（p < 0.05 至 p < 0.0001）。
- 例如，在 N=50 时，细粒度设置的中位 Sharpe Ratio 达到约 **1.9**，而粗粒度仅为 **1.6** 左右，差距显著。

#### （2）消融实验结果（表1 & 表2）

##### 表1：细粒度减去粗粒度的 Sharpe Ratio 差异（ΔSR）
| Portfolio Size | 10 | 20 | 30 | 40 | 50 |
|----------------|----|----|----|----|----|
| All agents     | -0.12 | +0.19\*\*\* | +0.08 | +0.17\*\*\* | +0.26\*\*\*\* |
| w/o Technical  | +0.54\*\* | -0.07 | -0.34\*\*\* | -0.66\*\*\* | -0.79\*\*\*\* |

> 🔍 发现：当移除 Technical Agent 后，细粒度的优势消失甚至反转，说明 **Technical Agent 是驱动性能提升的关键组件**。

##### 表2：各 Agent 的消融效果（相对于全模型的 SR 变化）
| Agent Removed | Fine-grained setting 结果 | Coarse-grained setting 结果 |
|---------------|----------------------------|------------------------------|
| w/o Technical | ❌ 性能大幅下降（尤其大组合） | ➖ 影响较小或略有提升 |
| w/o News      | ➖ 影响较小                   | ❌ 移除后性能严重恶化         |

> 🔍 发现：
> - 在细粒度框架下，**Technical Agent 提供强预测信号**，不可或缺；
> - 在粗粒度框架下，**News Agent 更重要**，可能因其补偿了技术信号提取能力的不足。

#### （3）文本分析支持发现（6.3节）
- **关键词差异分析**显示：
  - 细粒度输出包含更多专业术语（如 *momentum*, *volatility*, *margins*, *growth-rate*）；
  - 粗粒度输出偏向表面描述（如 *price*, *trend*, *EPS*, *increase*）。
- **语义相似度分析**（表3）表明：
  - 细粒度设置下，**Technical Agent 与 Sector Agent 的输出相似度更高**（+0.022），说明技术洞察被更好整合进高层决策。

#### （4）组合优化结果（图3 & 表4）
| Portfolio | 年化收益 | 年化波动 | Sharpe Ratio（净，含交易成本） |
|----------|----------|-----------|-------------------------------|
| TOPIX 100 | 19.3%    | 11.5%     | 1.68                          |
| Agent Strategies | 10.6% | 11.2% | 0.95 |
| **50-50 Combined** | **15.2%** | **8.0%** | **1.91** ✅ |

> 💡 关键发现：尽管单一 agent 策略 Sharpe 不及大盘，但由于其与 TOPIX 100 收益相关性低（~0.4），**组合后能显著提升整体 Sharpe Ratio**，证明其具备独立 alpha 和分散化价值。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **细粒度任务分解显著提升 LLM 交易系统的性能**：相比于笼统指令，明确的任务拆解能引导 LLM 生成更高质量、更具逻辑性的分析。
2. ✅ **Technical Agent 是性能核心驱动力**：尤其是在细粒度设置下，其提供的 momentum 与 volatility 信号对最终决策至关重要。
3. ✅ **信息传播效率得到改善**：细粒度 prompt 促使专业术语向上层传递，增强了整个 multi-agent 架构的信息流动一致性。
4. ✅ **系统具有实际应用潜力**：通过 portfolio optimization 实验验证，该策略可作为主动管理模块与被动指数搭配使用，实现更优的风险调整收益。

### 方法的局限性
1. ⚠️ **依赖特定 LLM 的知识边界**：实验受限于 GPT-4o 的训练截止时间，长期稳健性仍需跨周期验证。
2. ⚠️ **尚未完全排除语言偏见的影响**：是否某些词汇模式更容易被 LLM “偏好”，从而影响下游判断？仍需进一步研究。
3. ⚠️ **仅在日本市场验证**：未在其他市场（如美国）进行泛化测试，文化与制度差异可能导致结果不可迁移。
4. ⚠️ **自然语言通信开销大**：虽然提升了可解释性，但也牺牲了效率；机器语言通信可能是未来方向之一。

### 未来工作方向
1. 🔄 探索 **time-aware LLM variants**（如 Time Machine GPT）以支持更长时间跨度的历史模拟；
2. 🔍 深入研究 **LLM 中的语言偏见与提示工程之间的交互机制**；
3. 🌐 扩展至其他金融市场（如 US market 或 emerging markets）进行跨市场验证；
4. 🤝 探索 **natural language 与 machine-oriented language 混合通信架构**，兼顾效率与可解释性；
5. 📈 将该框架应用于 real-world live trading pipeline，并引入 human-in-the-loop 审核机制。

--- 

> ✅ **总结一句话**：  
> 本文证明了在 LLM 多智能体交易系统中，**精细的任务设计（fine-grained task decomposition）比简单的角色分配更重要**——它不仅能提升 Sharpe Ratio，还能增强系统的透明度与可控性，为构建可信的 AI 投资团队提供了可行路径。

</details>

---

### 16. [Effective QA-driven Annotation of Predicate-Argument Relations Across Languages](https://arxiv.org/abs/2602.22865)

**Authors**: Jonathan Davidov, Aviv Slobodkin, Shmuel Tomi Klein, Reut Tsarfaty, Ido Dagan, Ayal Klein  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.22865v1  

#### Abstract
Explicit representations of predicate-argument relations form the basis of interpretable semantic analysis, supporting reasoning, generation, and evaluation. However, attaining such semantic structures requires costly annotation efforts and has remained largely confined to English. We leverage the Q...

---

### 17. [An Artificial Intelligence Framework for Joint Structural-Temporal Load Forecasting in Cloud Native Platforms](https://arxiv.org/abs/2602.22780)

**Authors**: Qingyuan Zhang  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.22780v1  

#### Abstract
This study targets cloud native environments where microservice invocation relations are complex, load fluctuations are multi-scale and superimposed, and cross-service impacts are significant. We propose a structured temporal joint load prediction framework oriented to microservice topology. The met...

---

### 18. [WaveSSM: Multiscale State-Space Models for Non-stationary Signal Attention](https://arxiv.org/abs/2602.22266)

**Authors**: Ruben Solozabal, Velibor Bojkovic, Hilal Alquabeh, Klea Ziu, Kentaro Inui, Martin Takac  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.22266v1  

#### Abstract
State-space models (SSMs) have emerged as a powerful foundation for long-range sequence modeling, with the HiPPO framework showing that continuous-time projection operators can be used to derive stable, memory-efficient dynamical systems that encode the past history of the input signal. However, exi...

---

### 19. [Hypernetwork-based approach for grid-independent functional data clustering](https://arxiv.org/abs/2602.22823)

**Authors**: Anirudh Thatipelli, Ali Siahkoohi  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.22823v1  

#### Abstract
Functional data clustering is concerned with grouping functions that share similar structure, yet most existing methods implicitly operate on sampled grids, causing cluster assignments to depend on resolution, sampling density, or preprocessing choices rather than on the underlying functions themsel...

---

### 20. [Agentic AI for Intent-driven Optimization in Cell-free O-RAN](https://arxiv.org/abs/2602.22539)

**Authors**: Mohammad Hossein Shokouhi, Vincent W. S. Wong  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.22539v1  

#### Abstract
Agentic artificial intelligence (AI) is emerging as a key enabler for autonomous radio access networks (RANs), where multiple large language model (LLM)-based agents reason and collaborate to achieve operator-defined intents. The open RAN (O-RAN) architecture enables the deployment and coordination ...

---

### 21. [Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search](https://arxiv.org/abs/2602.22983)

**Authors**: Xun Huang, Simeng Qin, Xiaoshuang Jia, Ranjie Duan, Huanqian Yan, Zhitao Zeng, Fei Yang, Yang Liu, Xiaojun Jia  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.22983v1  

#### Abstract
As Large Language Models (LLMs) are increasingly used, their security risks have drawn increasing attention. Existing research reveals that LLMs are highly susceptible to jailbreak attacks, with effectiveness varying across language contexts. This paper investigates the role of classical Chinese in ...

---

### 22. [Search More, Think Less: Rethinking Long-Horizon Agentic Search for Efficiency and Generalization](https://arxiv.org/abs/2602.22675)

**Authors**: Qianben Chen, Tianrui Qin, King Zhu, Qiexiang Wang, Chengjun Yu, Shu Xu, Jiaqi Wu, Jiayu Zhang, Xinpeng Liu, Xin Gui, Jingyi Cao, Piaohong Wang, Dingfeng Shi, He Zhu, Tiannan Wang, Yuqing Wang, Maojia Song, Tianyu Zheng, Ge Zhang, Jian Yang, Jiaheng Liu, Minghao Liu, Yuchen Eleanor Jiang, Wangchunshu Zhou  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.22675v1  

#### Abstract
Recent deep research agents primarily improve performance by scaling reasoning depth, but this leads to high inference cost and latency in search-intensive scenarios. Moreover, generalization across heterogeneous research settings remains challenging. In this work, we propose \emph{Search More, Thin...

---

### 23. [FLYING SERVING: On-the-Fly Parallelism Switching for Large Language Model Serving](https://arxiv.org/abs/2602.22593)

**Authors**: Shouwei Gao, Junqi Yin, Feiyi Wang, Wenqian Dong  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.22593v1  

#### Abstract
Production LLM serving must simultaneously deliver high throughput, low latency, and sufficient context capacity under non-stationary traffic and mixed request requirements. Data parallelism (DP) maximizes throughput by running independent replicas, while tensor parallelism (TP) reduces per-request ...

---

### 24. [Exploiting network topology in brain-scale simulations of spiking neural networks](https://arxiv.org/abs/2602.23274)

**Authors**: Melissa Lober, Markus Diesmann, Susanne Kunkel  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.23274v1  

#### Abstract
Simulation code for conventional supercomputers serves as a reference for neuromorphic computing systems. The present bottleneck of distributed large-scale spiking neuronal network simulations is the communication between compute nodes. Communication speed seems limited by the interconnect between t...

---

### 25. [Prediction of Diffusion Coefficients in Mixtures with Tensor Completion](https://arxiv.org/abs/2602.23142)

**Authors**: Zeno Romero, Kerstin M\"unnemann, Hans Hasse, Fabian Jirasek  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.23142v1  

#### Abstract
Predicting diffusion coefficients in mixtures is crucial for many applications, as experimental data remain scarce, and machine learning (ML) offers promising alternatives to established semi-empirical models. Among ML models, matrix completion methods (MCMs) have proven effective in predicting ther...

---

### 26. [Knob: A Physics-Inspired Gating Interface for Interpretable and Controllable Neural Dynamics](https://arxiv.org/abs/2602.22702)

**Authors**: Siyu Jiang, Sanshuai Cui, Hui Zeng  
**Category**: cs.AI  
**Published**: 2026-02-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.22702v1  

#### Abstract
Existing neural network calibration methods often treat calibration as a static, post-hoc optimization task. However, this neglects the dynamic and temporal nature of real-world inference. Moreover, existing methods do not provide an intuitive interface enabling human operators to dynamically adjust...

---

### 27. [Towards Faithful Industrial RAG: A Reinforced Co-adaptation Framework for Advertising QA](https://arxiv.org/abs/2602.22584)

**Authors**: Wenwei Li, Ming Xu, Tianle Xia, Lingxiang Hu, Yiding Sun, Linfang Shang, Liqun Liu, Peng Shu, Huan Yu, Jie Jiang  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.22584v1  

#### Abstract
Industrial advertising question answering (QA) is a high-stakes task in which hallucinated content, particularly fabricated URLs, can lead to financial loss, compliance violations, and legal risk. Although Retrieval-Augmented Generation (RAG) is widely adopted, deploying it in production remains cha...

---

### 28. [Towards Better RL Training Data Utilization via Second-Order Rollout](https://arxiv.org/abs/2602.22765)

**Authors**: Zhe Yang, Yudong Wang, Rang Li, Zhifang Sui  
**Category**: cs.CL  
**Published**: 2026-02-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.22765v1  

#### Abstract
Reinforcement Learning (RL) has empowered Large Language Models (LLMs) with strong reasoning capabilities, but vanilla RL mainly focuses on generation capability improvement by training with only first-order rollout (generating multiple responses for a question), and we argue that this approach fail...

---

### 29. [CARAT: Client-Side Adaptive RPC and Cache Co-Tuning for Parallel File Systems](https://arxiv.org/abs/2602.22423)

**Authors**: Md Hasanur Rashid, Nathan R. Tallent, Forrest Sheng Bao, Dong Dai  
**Category**: cs.DC  
**Published**: 2026-02-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.22423v1  

#### Abstract
Tuning parallel file system in High-Performance Computing (HPC) systems remains challenging due to the complex I/O paths, diverse I/O patterns, and dynamic system conditions. While existing autotuning frameworks have shown promising results in tuning PFS parameters based on applications' I/O pattern...

---

### 30. [TEFL: Prediction-Residual-Guided Rolling Forecasting for Multi-Horizon Time Series](https://arxiv.org/abs/2602.22520)

**Authors**: Xiannan Huang, Shen Fang, Shuhan Qiu, Chengcheng Yu, Jiayuan Du, Chao Yang  
**Category**: cs.LG  
**Published**: 2026-02-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.22520v1  

#### Abstract
Time series forecasting plays a critical role in domains such as transportation, energy, and meteorology. Despite their success, modern deep forecasting models are typically trained to minimize point-wise prediction loss without leveraging the rich information contained in past prediction residuals ...

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
