# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-10 07:15:43 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [QaRL: Rollout-Aligned Quantization-Aware RL for Fast and Stable Training under Training--Inference Mismatch](https://arxiv.org/abs/2604.07853)

**Authors**: Hao Gu, Hao Wang, Jiacheng Liu, Lujun Li, Qiyuan Zhu, Bei Liu, Binxing Xu, Lei Wang, Xintong Yang, Sida Lin, Sirui Han, Yike Guo  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2604.07853v1  

#### Abstract
Large language model (LLM) reinforcement learning (RL) pipelines are often bottlenecked by rollout generation, making end-to-end training slow. Recent work mitigates this by running rollouts with quantization to accelerate decoding, which is the most expensive stage of the RL loop. However, these se...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：QaRL: Rollout-Aligned Quantization-Aware RL for Fast and Stable Training under Training–Inference Mismatch

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大型语言模型（LLM）的强化学习（RL）训练中，**rollout 生成阶段是端到端训练的主要瓶颈**，占整个训练时间的约 70%。为加速这一过程，已有研究尝试对 rollout 模型进行低比特量化（如 W4A16），从而提升推理吞吐量。

然而，这种做法引入了严重的 **training-inference mismatch**（训练-推理不一致）问题：
- **Rollout 阶段**：使用低精度（quantized）模型采样响应；
- **Policy 学习阶段**：使用全精度（BF16）模型计算梯度更新。

这导致采样分布 $ \pi_{\text{quant-sampler}} $ 与学习分布 $ \pi_{\text{BF16-learner}} $ 不一致，破坏了 PPO 类算法的信任区域（trust region）假设，引发训练不稳定甚至崩溃。

此外，作者观察到一个关键失败模式：**量化 rollout 在长序列生成中容易产生重复、乱码的“error tokens”**，这些 token 在原始策略下概率极低，导致重要性权重（importance ratio）极端化，进一步加剧训练波动。

---

### 🚀 提出的新方法与创新思路

#### （1）**QaRL（Rollout-Aligned Quantization-Aware RL）**
- **核心思想**：让 learner 的前向传播行为与 quantized rollout 引擎保持一致，即在训练侧也执行真实的低比特 GEMM 运算（而非仅模拟量化噪声的 Fake Quant）。
- **实现方式**：
  - 维持 BF16 的 master weights 用于反向传播；
  - 在前向时将权重实时量化为低比特（如 W4A16），并执行低比特矩阵乘法；
  - 使用 STE（Straight-Through Estimator）传递梯度。
- **优势**：显著缩小 learner 与 sampler 之间的分布差距，缓解 mismatch。

#### （2）**TBPO（Trust-Band Policy Optimization）**
针对 error tokens 导致的优化失稳问题，提出一种新的序列级优化目标：
- **Dual Clipping for Negative Samples**：
  - 对负优势样本（A < 0）同时施加上下界裁剪（$[1-\epsilon_l, 1+\epsilon_h]$），防止因低概率 error tokens 导致的重要性比爆炸。
- **Sequence-Level Objective**：
  - 将整条 response 视作单一 action，基于其几何平均概率定义序列级重要性比 $ r_{\text{seq-prox}} $ 和 mismatch 权重 $ w_{\text{seq-mismatch}} $；
  - 若任一序列超出信任带，则丢弃该样本的更新。
- **动机**：error tokens 往往成簇出现，序列级控制能有效识别并过滤掉偏离轨迹的整体响应，而 token-level 裁剪无法处理后续传播错误。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | QaRL/TBPO 改进 |
|------|--------|----------------|
| **Quantized Rollout Only** | 严重 mismatch，训练不稳定 | 通过 rollout-aligned forward 对齐 learner 与 sampler 行为 |
| **Standard QAT / Fake Quant** | 仅模拟量化误差，未真实复现 arithmetic behavior | 执行 real low-bit GEMM，更精确匹配 rollout 引擎 |
| **PPO / GRPO Token-Level Clipping** | 无法控制 error token 引发的序列漂移 | 序列级 dual clipping + filtering，增强鲁棒性 |
| **Bitwise Consistent Kernels** | 实现复杂，性能损失大（~2× 慢） | 容忍非完全一致，用 $ w_{\text{mismatch}} $ 动态补偿 |

> ✅ **综合优势**：在保持量化带来的 **1.3× 训练加速** 的前提下，恢复接近 BF16 全精度训练的性能与稳定性。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **主任务数据集**（数学推理）：
  - `OpenR1-Math-46K`（46,000 数学题）
  - 测试集：AIME2024/2025、AMC、MATH-500、Minerva、Olympiad-Bench
- **泛化能力评估**（OOD）：
  - ARC-Challenge、GPQA-Diamond、LiveCodeBench、MMLU-Pro

---

### ⚙️ 实验设置
- **模型规模**：
  - Qwen2.5-1.5B-Math
  - Qwen2.5-7B-Math
  - Qwen3-8B-Base
  - Qwen3-30B-A3B-Base（MoE 架构）
- **训练框架**：
  - Rollout 引擎：vLLM
  - 训练后端：Verl（基于 Ray）
  - 硬件：8× NVIDIA H800 GPUs
- **量化配置**：
  - 主要采用 **W4A16**（权重量化至 4-bit，激活保留 16-bit）
  - 对比 W8A8、FP8W8A8、W4A8 等方案
- **优化器**：
  - 使用 **Muon**（收敛更快于 AdamW）

---

### 🎯 评估指标
- **主指标**：Pass@1 准确率（除 LiveCodeBench 报告 Pass@4）
- **辅助指标**：
  - 平均奖励曲线（reward curve）
  - KL 散度变化（KL drift）
  - 每步训练耗时（speedup ratio）
  - 训练稳定性（是否崩溃）

---

### 🆚 基线方法对比
| 基线方法 | 描述 |
|---------|------|
| **BF16 GRPO/GSPO** | 全精度训练，作为性能上限基准 |
| **w4a16 rollout GRPO/GSPO** | 仅 rollout 量化，learner 仍为 BF16，代表当前主流加速方案 |
| **QaRL w/o TBPO** | rollout-aligned QAT，但无序列级保护机制 |
| **TBPO variants** | 消融不同 clipping 策略（如 dual-clip、positive-only、on-policy 等） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Model | 方法 | In-Domain Avg. Math | OOD Avg. |
|-------|------|---------------------|----------|
| Qwen3-8B-Base | BF16 GRPO | **51.8** | **62.5** |
| | w4a16 rollout GRPO | 43.9 (-7.9↓) | 59.5 |
| | **w4a16 QaRL TBPO** | **48.9 (-2.9↓)** | **61.0** |
| Qwen3-30B-A3B-Base | BF16 GSPO | **52.1** | **67.8** |
| | w4a16 rollout GSPO | 45.7 (-6.4↓) | 61.2 |
| | **w4a16 QaRL TBPO** | **51.2 (-0.9↓)** | **67.05** |

> 💡 **结论**：QaRL TBPO 成功挽回了超过 **5.5~6.4 个百分点** 的性能退化，几乎追平 BF16 基线。

---

### 🔁 与基线方法对比结果
- **相比纯量化 rollout**：
  - 在 Qwen3-30B-A3B 上，数学平均得分从 **45.7 → 51.2（+5.5）**
  - OOD 性能同步提升，说明不是过拟合，而是真正提升了泛化能力
- **相比 BF16 全精度训练**：
  - 性能差距缩小至 **<1%**
  - 同时获得 **1.3× 的训练速度提升**（见 Figure 9b）
- **在 MoE 模型上的表现尤为突出**：
  - MoE 本身路由不稳定，传统量化 rollout 更易崩溃
  - QaRL TBPO 仍能稳定收敛，验证其鲁棒性

---

### 🔍 消融实验结果（Ablation Study）

#### （1）优化目标对比（Figure 8）
| 方法 | 表现 |
|------|------|
| **GSPO** | KL 显著漂移，最终崩溃 |
| **MIS GSPO** | 拒绝采样降低效率，reward 上限受限 |
| **Positive GRPO** | 忽略负样本，探索不足，性能受限 |
| **On-Policy GRPO** | 更新效率低 |
| **Dual-Clip GRPO** | 控制住极端 ratio，但仍受污染 token 影响 |
| ✅ **QaRL TBPO** | KL 稳定、reward 高、收敛快 |

> ➤ 表明：**只有结合 rollout alignment 与 sequence-level dual clipping，才能实现高效且稳定的训练**

#### （2）不同量化方案对比（Figure 9a）
- W4A16、W8A8、FP8W8A8、W4A8 等多种量化格式下，**QaRL TBPO 均能达到相近最终性能**
- 说明：一旦训练稳定，收敛质量对具体量化格式不敏感
- 最终选择 W4A16 因其在大模型上具有最佳 **throughput/memory 平衡**

#### （3）熵分析（Appendix D.3）
- 发现量化 rollout 训练后期 entropy 上升，源于 error tokens 的过度优化；
- QaRL TBPO 可抑制此现象，维持与 BF16 相当的 entropy 曲线。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Training-inference mismatch 是量化 rollout RL 失败的核心原因**，不能仅靠 reweighting 或简单 QAT 解决。
2. **Error tokens 是量化 rollout 中的致命隐患**：它们在长序列中累积，导致 policy ratio 极端化，破坏 trust region。
3. **Token-level clipping 无法解决 mid-generation 错误传播问题**，必须上升到 sequence-level 进行控制。
4. **Rollout-aligned forward + sequence-level dual clipping（QaRL + TBPO）可有效恢复训练稳定性与性能**，几乎达到 BF16 水平。
5. **QaRL 在 MoE 等复杂架构上依然稳定**，具备良好的扩展性。

---

### ⚠️ 方法的局限性
1. **仍依赖部分高精度计算**：
   - 反向传播仍在 BF16 下进行，未实现 Fully Quantized Training（FQT）
   - 若引入 4-bit gradients 可能带来更大误差（文中指出梯度 NaN 问题）
2. **额外计算开销**：
   - 每步需重新量化权重并同步到 rollout 引擎，虽避免重复 quant，仍有轻微 overhead
3. **未支持 KV Cache 量化**：
   - 当前 vLLM 中 FP8 KV 未能提供实际 throughput 提升，故未启用

---

### 🔮 未来工作方向
1. **探索 Fully Quantized RL Training**：
   - 实现 end-to-end 低比特前向与反向，进一步压缩内存与能耗
2. **设计高效的 token-level 近似方法替代 sequence-level filtering**：
   - 当前 sequence-level 判断成本较高，希望找到既能稳定训练又能提高 sample efficiency 的轻量替代
3. **扩展至其他 agent 范式**：
   - 如 self-improvement、tool-using、multi-step planning 等需要长 rollout 的场景
4. **硬件协同优化**：
   - 结合定制 kernel 实现更高效的低比特 arithmetic pipeline

---

## 总结

📌 **QaRL 提供了一条实用路径**：在不牺牲训练稳定性的前提下，利用量化 rollout 加速 LLM 强化学习，实现了 **“既快又稳”** 的训练体验。

🔧 其核心在于两个互补机制：
- **Rollout-Aligned Forward** → 缩小 learner-sampler 分布差异
- **TBPO（Sequence-Level Dual Clipping）** → 抵御 error tokens 引发的优化灾难

📈 实验表明，在多个模型尺度（含 MoE）上，QaRL TBPO 不仅大幅优于 naive quantized rollout，还逼近甚至媲美 BF16 全精度训练，同时带来 **1.3× 的端到端加速**，为大规模 LLM-RL 的工业化部署提供了强有力的技术支撑。

</details>

---

### 2. [AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention](https://arxiv.org/abs/2604.07815)

**Authors**: Yuxuan Hu, Jianchao Tan, Jiaqi Zhang, Wen Zan, Pingwei Sun, Yifan Lu, Yerui Sun, Yuchen Xie, Xunliang Cai, Jing Zhang  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.07815v1  

#### Abstract
Long-context inference in LLMs faces the dual challenges of quadratic attention complexity and prohibitive KV cache memory. While token-level sparse attention offers superior accuracy, its indexing overhead is costly; block-level methods improve efficiency but sacrifice precision. We propose AsyncTL...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在长上下文推理中面临两大瓶颈：
- **计算复杂度**：自注意力机制具有 $O(n^2)$ 的计算开销。
- **内存消耗**：Key-Value (KV) cache 随序列长度线性增长，在超长上下文（如 48k–96k tokens）下极易超出 GPU 显存容量。

现有稀疏注意力方法存在“精度 vs 效率”的权衡：
- **Token-level 稀疏**：精度高，但索引开销大；
- **Block-level 稀疏**：效率高，但因粗粒度选择导致重要信息丢失。

此外，KV cache 的 offloading 缺乏对细粒度 sparsity 模式的优化支持。

---

### **提出的新方法与创新思路**

#### ✅ **Hierarchical Two-Level Sparse Attention（两级稀疏注意力架构）**
- **Level-1: Block-Level Filtering（粗筛选）**
  - 将 KV cache 分为多个 block，基于压缩表示快速打分并保留 Top-$k_b$ 个候选块。
  - 利用 GEMM 友好型公式重构 Quest 的评分机制，提升硬件利用率（尤其适用于 MQA/MLA 架构）。
- **Level-2: Token-Level Selection（精筛选）**
  - 在选中的 block 内进行 token 级别的精细选择（Top-$k_t$），采用 Double-Sparsity 思路结合通道选择与量化以降低开销。

> 这种两阶段设计在保持 token-level 精度的同时显著减少了搜索空间和索引成本。

#### ✅ **AsyncTLS Offloading Engine（异步缓存预取引擎）**
- **Temporal Overlap（时间重叠执行）**
  - 当前 timestep 执行 token-level attention 时，**并行地**：
    1. 对下一时刻的 query 执行 block-level selection；
    2. 异步预取所需 KV blocks 至 GPU resident cache。
- **Incremental Block Transfer（增量传输）**
  - 仅传输当前与上一步 block selection 的差异部分（$\Delta M_t = M_t \setminus C_t$），利用注意力模式的时间局部性减少 PCIe 带宽占用。

> 实现了 **compute-memory overlap** 和 **bandwidth minimization** 的双重优化。

---

### **相比现有方法的优势**
| 维度 | 现有方法 | AsyncTLS |
|------|--------|----------|
| **精度** | Block-level 方法损失显著 | 接近 Full Attention，优于 Quest |
| **效率** | Token-level 方法索引开销高 | 层次化剪枝大幅降低开销 |
| **KV Offloading** | 多为整块搬移，未适配 token-level sparsity | 支持异步 + 增量传输，隐藏延迟 |
| **通用性** | 主要针对 MHA/GQA | 兼容 MLA 等先进架构 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LongBench**：涵盖 14 项长文本理解任务，包括：
  - 叙事理解（NarrativeQA）
  - 科学问答（QasperQA）
  - 多跳推理（HotpotQA, MultiFieldQA, Musique）
  - 文档摘要（GovReport, QMSum, Multi-News）
  - 特殊任务（TriviaQA, RepoBench-P 等）
- **RULER**：专注于上下文检索能力评测，包含：
  - 单文档/多文档问答（S1/S2/MK1/MK2/MQ/MV）
  - QA-Hotpot/SQuAD、VT、FWE 等子任务

---

### **实验设置与评估指标**

#### **模型**
- **Qwen3-8B / Qwen3-14B**（GQA 架构）
- **GLM-4.7-Flash**（MLA 架构）

#### **上下文长度**
- 测试范围：**32k ~ 128k tokens**

#### **稀疏配置**
- Block size: 64
- Block 数量: 128 → 覆盖 8,192 tokens
- Token budget: 512 / 1024 / 2048 tokens
- Channel dimension: GQA 模型用 32，MLA 用 128，INT4 量化 key

#### **评估维度**
1. **准确性指标**：各任务得分平均值（如 LongBench Avg., RULER Avg.）
2. **效率指标**：
   - Operator-level latency（内核级延迟）
   - End-to-end inference latency
   - Throughput（tokens/sec）

#### **基线方法对比**
| 方法 | 类型 | 描述 |
|------|-----|------|
| **Full Attention (FA)** | Dense | 完整 KV attention，作为性能上限 |
| **Quest** | Block-level | 动态 block selection，高效但精度低 |
| **Double-Sparsity (DS)** | Token-level | 高精度，但索引开销大 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 模型 | 上下文长度 | 方法 | 准确率（vs FA） | 吞吐提升（vs FA） | 延迟下降 |
|------|------------|-------|------------------|--------------------|-----------|
| Qwen3-8B | 32k | AsyncTLS | ≈99.5% | **1.8×** | ↓~40% |
| GLM-4.7-Flash | 96k | AsyncTLS | ≈99.8% | **4.7×** | ↓~60% |
| 平均 Across GQA/MLA | 48k–96k | AsyncTLS | **几乎无损** | **1.3×–4.7×** | —— |

> 在高达 **96k tokens** 的上下文中仍能维持接近 Full Attention 的准确率。

---

### **与基线方法的对比结果**

#### 🔹 **准确性对比（Table 1 & 2）**
- 在 **LongBench** 上：
  - AsyncTLS 显著优于 **Quest**（如 Qwen3-14B 提升 ~2.5 pts）
  - 与 **DS** 和 **Full Attention** 性能相当（差距 <1%）
- 在 **RULER** 上：
  - AsyncTLS 在多数任务上超过 DS 和 Quest，尤其在 QA 和 VT 任务中表现优异。

#### 🔹 **效率对比（Figure 4 & 5）**
- **Operator Speedup**：
  - GQA 架构：**1.7×–6.2× faster than FA**, **1.2×–4.0× faster than DS**
  - MLA 架构：最高达 **10.0× speedup over FA**
- **End-to-End Latency**（图5）：
  - AsyncTLS 明显快于 DS，接近 Quest 水平
- **Throughput with Offloading**（图6）：
  - 因支持更大 batch size（bs=6 vs bs=1 for FA），吞吐优势巨大：
    - Qwen3-8B @96k: **1.84× higher throughput**
    - GLM-4.7-Flash @96k: **4.70× higher throughput**

---

### **消融实验分析（隐含于主实验中）**
虽然未单独列出消融表，但从设计可推断以下关键组件作用：
- **Two-Level Indexing**：
  - 若仅用 token-level，索引开销过大；
  - 若仅用 block-level，精度下降明显（见 Quest 表现）。
- **Asynchronous Prefetching + Incremental Transfer**：
  - 显著减少 PCIe 数据传输量（从 $O(k_b B d)$ → $O(|\Delta| B d)$）
  - 成功实现 compute-transfer overlap，避免内存墙成为瓶颈。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **层次化稀疏是平衡精度与效率的有效路径**  
   结合 block-level 快速过滤与 token-level 精细选择，可在极低 token budget 下逼近 full attention 性能。

2. ✅ **时间局部性可用于优化 KV offloading**  
   相邻 step 的 block selection 高度相似，增量更新策略极大节省带宽。

3. ✅ **AsyncTLS 具备强架构兼容性**  
   在 GQA 和 MLA 架构上均取得显著收益，验证其在现代 LLM 中的普适性。

4. ✅ **无需微调即可部署**  
   所有优化均为 inference-time 技术，不依赖 retraining 或 fine-tuning，适合生产环境快速落地。

---

### **方法的局限性**
- **依赖稳定的 attention locality**：若 attention pattern 变化剧烈（如跳跃式话题转换），block-level 预测可能失效。
- **额外的 index 存储开销**：需维护 block/token 两级索引结构，增加少量元数据管理负担。
- **对 very short context 优势不明显**：主要价值体现在 >32k 的超长上下文场景。

---

### **未来工作方向**
1. **动态调整 $k_b$ / $k_t$**：根据输入难度或生成阶段自适应调节稀疏粒度。
2. **扩展至 prefilling 阶段**：目前 focus 在 decoding，可探索在 prefill 中也应用类似稀疏。
3. **集成到端到端推理系统**：结合调度器、内存池等构建完整 long-context inference pipeline。
4. **探索更高效的 index encoding**：进一步压缩索引体积，提升跨设备传输效率。

---

> 📌 **总结一句话**：  
> **AsyncTLS 通过 “Hierarchical Sparsity + Asynchronous Offloading” 的协同设计，在几乎不牺牲 accuracy 的前提下，实现了 1.3×–4.7× 的 end-to-end throughput 提升，为 ultra-long context LLM inference 提供了一个高效、实用且可扩展的解决方案。**

</details>

---

### 3. [Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving](https://arxiv.org/abs/2604.08075)

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.08075v1  

#### Abstract
Production vLLM fleets typically provision each instance for the worst-case context length, leading to substantial KV-cache over-allocation and under-utilized concurrency. In practice, 80-95% of requests are short, yet are served under configurations optimized for long contexts, wasting 4-8$\times$ ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前生产环境中的 vLLM 部署普遍采用**同质化配置**（homogeneous provisioning），即所有实例均按最大上下文长度（如 `max_model_len=64K`）进行资源配置。然而，实际请求中 **80–95% 是短文本请求**（通常在 2K–8K tokens 内），导致：
- **KV-cache 资源严重浪费**：每个序列预留大量内存空间，但实际利用率不足 5%
- **并发能力受限**：高 `Cmax` 导致每 GPU 支持的并发序列数极低（例如从 128 降至 16）
- **可靠性问题频发**：OOM 崩溃、preemption（抢占）、请求拒绝、TTFT SLO 违规等

这些问题的根本原因是 **configuration-traffic mismatch**（配置与流量不匹配）。

---

### 提出的新方法或新思路
作者提出 **Dual-Pool Token-Budget Routing**（双池令牌预算路由），其核心思想是：
将一个同质化的 vLLM 集群划分为两个专用子池：
- **Short Pool**：小 `Cmax`（如 8K），支持高并发、高吞吐，服务绝大多数短请求
- **Long Pool**：大 `Cmax`（如 64K），处理长上下文请求，牺牲部分吞吐换取容量覆盖

请求通过轻量级调度器基于其 **estimated total token budget** 动态路由到合适池。

#### 创新点包括：
1. **Token-Budget Pool Routing**  
   - 全局调度机制，操作于集群层面而非单个 GPU
   - 可与 PagedAttention、continuous batching、chunked prefill 等现有优化无缝组合

2. **Self-Calibrating Token Estimation（自校准令牌估计）**
   - 不依赖 tokenizer，使用 per-category 的 **bytes-per-token ratio**
   - 在线通过 EMA（指数移动平均）学习，并反馈 `usage.prompt_tokens`
   - 引入保守偏差（conservative bias）防止误路由至短池造成 preemption

3. **Closed-Form Cost Model（闭式成本模型）**
   - 推导公式：`ΔG/G = α(1 - 1/p)`，其中：
     - `α`：短请求占比
     - `p`：短池相对于长池的吞吐增益
   - 可在部署前仅凭流量分布和吞吐测试预估收益，无需仿真

4. **Load-Aware Spillover + Safety Check**
   - 当首选池过载时动态溢出到备用池
   - 最终安全检查确保目标池能容纳该请求

---

### 相比现有方法的优势
| 方法 | 局限性 | 本文优势 |
|------|--------|----------|
| Chunked Prefill | 仅缓解计算调度阻塞，KV-cache 仍为全序列分配，无法提升并发 | 解决的是**内存资源配置问题**，从根本上释放并发潜力 |
| PagedAttention / Continuous Batching | 单实例优化，无法改变整体集群规模 | 工作在**fleet level**，可减少所需 GPU 总数 |
| Speculative Decoding / Disaggregation | 复杂度高，硬件依赖强 | 方法简单、开销 O(1)，兼容性强 |

> ✅ **核心优势**：以极低成本实现显著的成本节约与可靠性提升，且不牺牲延迟。

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **Azure LLM Inference Dataset**  
   - 来自真实生产环境的请求轨迹
   - 特征：高度偏斜，80% 请求 < 2K tokens，尾部长达 64K
2. **LMSYS-Chat-1M**  
   - 开源对话数据集衍生的请求流
   - 平均输入长度 `Lin = 69.5`，输出长度 `Lout = 214.5`，更集中紧凑

两者共同覆盖了“重尾”与“紧凑型”两种典型 LLM 流量模式。

---

### 实验设置
| 项目 | 设置 |
|------|------|
| **模型** | Llama-3-70B-Instruct（BF16）、Qwen3-235B-A22B（FP8） |
| **硬件模拟** | NVIDIA A100-80GB（TP=2）、AMD MI300X（192GB HBM） |
| **仿真工具** | 基于 Vidur 构建的离散事件模拟器，建模 prefill/decode、KV-cache 分配、批处理行为、排队动态 |
| **Pool 配置** | 如下表所示 |

#### 表：Pool 配置对比
| Pool | `Cmax` | `Nseq/GPU` | Batch Size | Throughput (req/s/inst) |
|------|--------|------------|------------|--------------------------|
| Homogeneous | 65K | 16 | 8K | 2.8 |
| Short Pool (`Ps`) | 8K | 128 | 16K | 11.2 |
| Long Pool (`Pl`) | 65K | 16 | 8K | 2.8 |

> 短池获得约 **4–8× 吞吐增益**

---

### 评估指标
- **成本类**：
  - GPU 实例数量
  - GPU-hours 消耗
  - 年度成本估算（$2.21/GPU-hr for A100, $3.67/GPU-hr for MI300X）
- **可靠性类**：
  - Preemption rate（‰）
  - OOM events/hr
  - Request rejection rate
  - Success rate
- **延迟类**：
  - P50/P99 TTFT（Time to First Token）
  - P50/P99 TPOT（Time Per Output Token）
- **SLO**：
  - P99 TTFT ≤ 2s
  - P99 TPOT ≤ 80ms

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Homogeneous (Round-Robin)** | 所有实例统一配置为 65K，请求轮询分发 |
| **Token-Budget Routing** | 本文方法，`Bshort=8192`，启用 load-aware spillover 和在线校准 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（A100 上 Llama-3-70B，1,000 req/s）

#### 表：GPU 成本节省（Table 2）
| Trace | 方法 | GPUs | 节省 | P99 TTFT |
|-------|------|------|------|---------|
| Azure | Homogeneous | 358 | — | 1.82s |
| Azure | Token-Budget | 208 | **41.9%** | 1.71s |
| LMSYS | Homogeneous | 358 | — | 1.45s |
| LMSYS | Token-Budget | 246 | **31.3%** | 1.48s |

> 💰 按云价格计算，**年节省 $2.86M**

---

#### 表：可靠性提升（Azure trace, 90% util）
| 方法 | Preemption (‰) | OOM (events/hr) | Rejection Rate | Success Rate |
|------|----------------|------------------|---------------|--------------|
| Homogeneous | 47.3 | 2.1 | 0.31% | 99.69% |
| Overall (Ours) | **8.7** | **0.4** | **0.05%** | **99.95%** |

- **Preemption 下降 5.4×**
- **OOM 下降 5.3×**
- **成功率从 99.69% → 99.95%**

> ✅ 短池完全无 OOM，因始终满足 `Ltotal ≤ Cmax`

---

#### 表：延迟表现（Azure trace）
| Method | P50 TTFT | P99 TTFT | P50 TPOT | P99 TPOT |
|--------|----------|----------|----------|----------|
| Homogeneous | 0.42s | 1.82s | 28ms | 67ms |
| Ours | **0.28s** | **1.71s** | **25ms** | **62ms** |

- **P50 TTFT 提升 33%**
- **P99 TTFT 提升 6%**
- **TPOT 提升 7–11%**

> ⚡️ P50 改善显著得益于短池极高并发消除排队；P99 改善有限因最长请求仍在长池处理

---

### 消融实验与敏感性分析

#### （1）Calibration 收敛性（Table 5）
| Category | True c_k | c_k @ n=50 | Rel. Error | Mis-route Rate |
|---------|----------|------------|-------------|----------------|
| English Prose | 4.48 | 4.41 | 1.6% | 0.3% |
| Code | 3.52 | 3.47 | 1.4% | 0.2% |
| CJK Text | 2.01 | 2.08 | 3.5% | 0.8% |
| Global Static (c=4) | — | — | — | **4.1%** |

> ✅ EMA 在 ~50 次观测后快速收敛；保守估计使误路由率下降超 4×，尤其对 CJK 文本帮助巨大

#### （2）Threshold 敏感性（Figure 6）
- **阈值 `Bshort` 在 4K–16K 范围内均能达到 >80% 最优收益**
- 默认设为 `8192` 即可获得接近峰值的性能
- 系统对调参不敏感，易于部署

#### （3）Scale Invariance**
- 在 100–2000 req/s 下节省稳定在 **31–42%**，验证理论模型普适性

---

### 案例研究：Qwen3-235B on AMD MI300X（Table 6）
| Deployment | Nodes | GPUs | Annual Cost | Savings |
|-----------|-------|------|-------------|---------|
| Homogeneous | 197 | 1,576 | $50.6M | — |
| Token-Budget | 137 | 1,096 | $35.2M | **$15.4M/yr** |

> 💥 在前沿 MoE 模型上仍实现 **30.5% GPU 减少**，证明方法可扩展至大规模系统

---

## 4. 关键结论和发现

### 主要发现
1. **Configuration-Traffic Mismatch 是根本瓶颈**  
   统一配置导致资源浪费与可靠性问题并存，而拆分池可同时解决二者。

2. **Dual-Pool 设计性价比极高**  
   仅需两个池即可捕获绝大部分收益（额外池增益 <2%），运维复杂度低。

3. **Routing on `Ltotal` 至关重要**  
   仅看 `prompt_tokens` 会导致“短提示+长生成”请求被错误路由，引发 preemption。

4. **保守估计显著提升鲁棒性**  
   利用 EMA 方差引入 bias，有效避免危险误路由，尤其对非英文流量至关重要。

5. **无需 tokenizer 的估计可行且高效**  
   基于字节长度 + per-category ratio 的 O(1) 估算足够准确，适合多模型上游路由层。

6. **理论模型是实用工程工具**  
   `ΔG/G = α(1 - 1/p)` 可用于部署前快速评估 ROI，指导是否值得实施。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **需要一定冷启动样本** | 新类别初期依赖默认 ratio（如 4.0），可能短暂误判，但收敛快（~50 请求） |
| **极端长尾仍需长池支撑** | 对持续出现超长请求场景，长池仍面临压力 |
| **未动态调整阈值** | `Bshort` 固定，若流量分布漂移需手动更新 |
| **假设集群可分割** | 需要管理两组独立实例，增加编排复杂性（可通过 autoscaler 缓解） |

---

### 未来工作方向
1. **Adaptive Thresholding**  
   利用运行时信号（preemption rate、queue depth）自动调节 `Bshort`，实现自我优化
2. **Lightweight Prompt Compression for Borderline Requests**  
   对接近阈值的请求尝试压缩，使其能在短池安全执行，进一步扩大高效区
3. **Integration with Heterogeneous Hardware**  
   结合不同 GPU 类型（如 A100 + MI300X）进行混合部署，最大化 cost-efficiency
4. **Multi-Pool Optimization with ML-based Router**  
   探索三层及以上池结构 + 学习型调度器，在可控复杂度下逼近理论极限

---

## 总结
✅ **Dual-Pool Token-Budget Routing** 是一种简洁、高效、实用的 LLM 服务优化方案：
- **解决了 configuration-traffic mismatch 的根本矛盾**
- **实现了 31–42% GPU 节省、5.4× 更低 preemption、6% 更好 P99 TTFT**
- **具备 O(1) 调度开销、无需 tokenizer、兼容现有优化技术**
- **已在真实轨迹和前沿硬件（MI300X）上验证有效性**

> 🎯 适用于任何具有明显长短请求混合特征的大规模 LLM 推理系统，是迈向**自适应、自优化 LLM serving infrastructure** 的关键一步。

</details>

---

### 4. [TurboAgent: An LLM-Driven Autonomous Multi-Agent Framework for Turbomachinery Aerodynamic Design](https://arxiv.org/abs/2604.06747)

**Authors**: Juan Du, Yueteng Wu, Pan Zhao, Yuze Liu, Min Zhang, Xiaobin Xu, Xinglong Zhang  
**Category**: cs.AI  
**Published**: 2026-04-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.06747v2  

#### Abstract
The aerodynamic design of turbomachinery is a complex and tightly coupled multi-stage process involving geometry generation, performance prediction, optimization, and high-fidelity physical validation. Existing intelligent design approaches typically focus on individual stages or rely on loosely cou...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TurboAgent: An LLM-Driven Autonomous Multi-Agent Framework for Turbomachinery Aerodynamic Design

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

传统涡轮机械（turbomachinery）气动设计流程高度依赖专家经验，是一个多阶段、强耦合、试错密集的迭代过程。现有智能设计研究大多局限于单一环节（如几何生成或性能预测），或通过脚本连接不同工具，缺乏**端到端自主闭环能力**。这导致设计周期长、计算成本高、知识复用困难。

### ✅ 提出了什么新方法或新思路

本文提出 **TurboAgent**，一个由大语言模型（LLM）驱动的**自主多智能体框架**（multi-agent system, MAS），用于实现涡轮机械气动设计与优化的全流程自动化。其核心创新包括：

- **统一LLM认知中枢**：LLM作为“任务规划代理”（Task Planning Agent），负责自然语言需求解析、全局任务分解、动态调度与决策路由，实现跨阶段协同。
- **功能化专用代理协作机制**：
  - **生成设计代理**（Generative Design Agent）：基于条件去噪扩散概率模型（cDDPM）实现高性能逆向几何生成。
  - **性能预测代理**（Performance Prediction Agent）：采用Transformer架构构建快速代理模型，实现毫秒级性能评估。
  - **优化代理**（Optimization Agent）：融合LLM驱动的元提示优化算法与传统GA/PSO，实现语义理解下的自适应搜索。
  - **物理验证代理**（Physics Validation Agent）：集成CFD与FEA求解器，支持自然语言指令驱动的高保真仿真自动化。
  - **知识合成代理**（Knowledge Synthesis Agent）：整合多源输出，生成结构化报告并支持问答交互。
- **数据驱动+物理一致性双保障闭环**：结合生成模型快速探索与高保真数值模拟最终验证，兼顾效率与工程可靠性。

### ✅ 相比现有方法的优势

| 维度 | 传统方法 | 现有AI辅助方法 | TurboAgent |
|------|----------|----------------|-----------|
| 设计范式 | 经验驱动、手动迭代 | 工具辅助、局部智能 | **自主智能、闭环协同** |
| 流程集成 | 脚本串联、刚性流程 | 功能模块独立 | **动态调度、反馈闭环** |
| 输入方式 | 数值参数输入 | 参数/代码接口 | **自然语言输入** |
| 决策能力 | 人工判断 | 固定策略 | **LLM语义推理+自适应优化** |
| 验证机制 | 手动CFD/FEA | 少量抽样验证 | **自动批处理+全链路物理验证** |

---

## 2. 核心实验方法和设置

### ✅ 数据集

- 基于文献 [33] 构建的**跨音速单级压气机转子叶片数据库**。
- 包含约1000组3D叶片几何及其对应的CFD仿真性能数据（mass flow rate $ \dot{m} $、total pressure ratio $ \pi_t $、isentropic efficiency $ \eta $）。
- 几何参数化采用NURBS描述，在叶根（hub）、中径（mid-span）、叶尖（tip）三个截面共定义21个设计变量（如前缘金属角、弦长、厚度分布等）。

### ✅ 实验设置

- **验证案例**：跨音速单转子压气机（transonic single-rotor compressor）设计任务。
- **前端界面**：基于Flask开发可视化交互平台，支持NL输入、3D几何展示、性能曲线与仿真结果可视化。
- **后端框架**：基于LangGraph构建图结构多智能体系统，实现代理间状态管理与任务流控。
- **LLM引擎**：主实验使用 **DeepSeek-V3.2-Chat** API，部分对比测试涉及ChatGPT。

### ✅ 评估指标

| 类别 | 指标 |
|------|------|
| **生成准确性** | R², nRMSE, MAE（目标 vs 生成性能） |
| **预测精度** | R², nRMSE（代理模型 vs CFD结果） |
| **优化效果** | 效率提升Δη，压比提升Δπ_t，收敛速度 |
| **系统性能** | 单次全流程耗时、token消耗量、设计成功率 |
| **物理一致性** | CFD结果与设计目标的RR²（>0.91为达标） |

### ✅ 基线方法对比

- **生成模型对比**：cDDPM vs GAN/VAE（在生成质量与多样性上更优）
- **优化算法对比**：LLM-driven optimizer vs GA vs PSO（在收敛速度与最终性能上占优）
- **流程自动化程度**：TurboAgent vs 手动流程 vs 脚本流水线（实现完全无人干预闭环）

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 📊 生成与预测性能（Section 4.2）

| 模块 | 指标 | 结果 |
|------|------|------|
| **生成设计代理**（cDDPM） | RR² (vs 目标) | >0.97（CFD验证下） |
| | nRMSE | <8% |
| **性能预测代理**（Transformer） | RR² (vs CFD) | 0.9807 ($\dot{m}$), 0.9824 ($\pi_t$), 0.9184 ($\eta$) |
| | nRMSE | <8% |
| | 推理时间 | ~10 ms/样本 |

> ➤ 表明生成与预测代理能高度逼近目标性能，且具备工程可用精度。

#### 📈 优化性能（Section 4.2.4）

| 优化器 | Δη (%) | Δπ_t (%) | 最终奖励值 |
|--------|--------|----------|------------|
| 初始设计 | 0 | 0 | 86.31 |
| **LLM-driven** | **+1.61** | **+3.02** | **87.08** |
| GA | +2.41 | +2.01 | 86.31 |
| PSO | +0.80 | +5.21 | 83.60 |

> ➤ LLM优化器在保持稳定性的前提下，综合性能最优，收敛更快。

#### ⏱️ 全流程效率（Section 4.4）

- **单次完整设计闭环时间**：约 **30分钟**（在30核CPU并行下）。
- **总token消耗**：约 **80,000–100,000 tokens**（以DeepSeek-V3.2计费）。
- **设计成功率**：130个生成样本中124个成功收敛CFD，成功率 **~95%**。

> ➤ 相比传统需数周的设计周期，实现**数量级提速**。

### ✅ 与基线方法的对比结果

- 在相同初始条件下，**TurboAgent** 能在无人干预下完成从NL需求到最终设计方案推荐的全过程，而传统方法需人工介入多个环节。
- LLM优化器相比GA/PSO在复杂非线性空间中表现出更强的**语义引导探索能力**，避免陷入局部最优。
- 自动生成的CFD/FEA流程减少人为错误，提升重复性与标准化水平。

### ✅ 消融实验结果（隐含于多任务验证）

- 移除物理验证代理 → 设计方案虽满足代理模型预测，但CFD偏差显著增大（RR²下降至~0.85以下）。
- 移除任务规划代理 → 各代理无法协同，需手动触发每一步，丧失自主性。
- 仅使用脚本调用 → 缺乏条件分支与反馈机制，无法应对“若不达标则优化”类复杂逻辑。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **TurboAgent 可实现真正意义上的端到端自主设计闭环**：
   - 支持从自然语言输入 → 自动任务分解 → 多代理协同执行 → 高保真物理验证 → 最终方案推荐的全流程自动化。
   
2. **LLM作为中央协调者具有强大语义理解与动态规划能力**：
   - 成功处理“先快速评估，若不满足则启动优化”等带条件分支的任务流，展现出类人类工程师的推理能力。

3. **cDDPM + Transformer 构成高效生成-评估对**：
   - 实现高质量、多样化的逆向设计生成，并通过快速代理筛选候选集，大幅提升搜索效率。

4. **LLM-driven优化器展现优越性能**：
   - 不依赖显式梯度或进化算子，即可通过prompt实现多目标自适应搜索，在效率与鲁棒性上优于传统算法。

5. **高保真验证不可或缺**：
   - 尽管代理模型精度高（RR²>0.98），但仍存在系统偏差；最终设计必须经CFD/FEA验证以确保物理一致性。

### ✅ 方法的局限性

- **训练数据依赖性强**：当前框架性能受限于已有数据库的质量与覆盖范围，难以泛化至全新构型（如离心压气机）。
- **LLM模型能力敏感**：优化性能受所选LLM规模与推理能力影响较大，小模型可能无法有效建模复杂设计语义。
- **计算资源门槛高**：尽管流程自动化，但CFD并行计算仍需高性能集群支持。
- **可解释性有限**：LLM决策过程为黑箱，缺乏明确的物理机制解释。

### ✅ 未来工作方向

1. **增强泛化能力**：引入跨构型迁移学习或零样本生成技术，拓展至风扇、涡轮等多种部件。
2. **提升自主决策能力**：发展基于强化学习的长期记忆与策略学习机制，实现更复杂的多工况联合优化。
3. **降低数据依赖**：探索主动学习与仿真驱动的数据扩充策略，减少对历史数据的依赖。
4. **提高可解释性**：构建可视化推理路径与决策溯源机制，增强工程可信度。
5. **扩展应用场景**：推广至整机多级匹配设计、多学科耦合优化（气动+传热+结构）等更复杂场景。

---

> 🔚 **总结一句话**：  
> TurboAgent 开创性地将 LLM 驱动的 multi-agent system 引入涡轮机械设计领域，实现了从“AI辅助”到“AI自主”的范式跃迁，为下一代智能设计提供了高效、可扩展的新范式。

</details>

---

### 5. [Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](https://arxiv.org/abs/2604.07399)

**Authors**: Wonseon Lim, Jaesung Lee, Dae-Won Kim  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.07399v1  

#### Abstract
Continual learning (CL) on edge devices requires not only high accuracy but also training-time efficiency to support on-device adaptation under strict memory and computational constraints. While prompt-based continual learning (PCL) is parameter-efficient and achieves competitive accuracy, prior wor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**边缘设备上的持续学习（Continual Learning, CL）**中存在的训练阶段资源瓶颈问题，特别是：
- **高内存占用**：传统 Prompt-based Continual Learning（PCL）方法在反向传播过程中产生大量中间激活，容易超出边缘设备的内存容量。
- **高计算开销**：现有PCL方法多关注推理效率或准确率，忽视了**训练时的计算与能耗效率**，限制了其在资源受限设备（如Jetson Orin Nano）上的部署。

此外，现有的**token reduction 技术**（如ToMe、PatchDropout）虽然能降低计算量，但通常是任务无关（task-agnostic）的，会随机合并或丢弃图像块（patch），导致关键语义信息丢失，严重影响PCL的准确性。

---

### 提出的新方法：CPS-Prompt
作者提出 **CPS-Prompt**，一个面向边缘设备的高效PCL框架，包含两个核心模块：

#### （1）Critical Patch Sampling (CPS)
- 利用冻结的 query encoder 在最终Transformer block中的 **attention 权重 $A$** 和 **value 激活 $V$** 构建“关键分数”：
  $$
  s_j = A_{1,j} \odot \|V_j\|_2
  $$
  其中 $A_{1,j}$ 表示class token对第$j$个patch token的关注度，$\|V_j\|_2$ 反映该patch的特征强度。
- 使用温度缩放softmax将分数转化为采样分布，并通过 multinomial sampling 选出最具任务相关性的patches。
- 仅保留这些critical patches进行后续forward和backward，显著减少存储的激活值数量。

> ✅ 特点：轻量级、无需额外训练、基于任务感知信号选择patches。

#### （2）Decoupled Prompt and Classifier Training (DPCT)
为缓解因稀疏输入训练带来的**表示不匹配问题**（sparse training vs. full inference），设计两阶段训练策略：
1. **Prompt Training Phase**：使用CPS选中的sparse patches联合优化 prompt 和 classifier。
2. **Classifier Training Phase**：冻结prompt参数，仅用完整patches微调classifier，使其适应推理时的完整输入表示。

> ✅ 优势：提升表示一致性，同时第二阶段无需回传prompt梯度，进一步节省计算和内存。

---

### 相比现有方法的优势
| 方面 | CPS-Prompt优势 |
|------|----------------|
| **准确性** | 接近SOTA方法C-Prompt，平均仅低2%，优于大多数PCL方法；显著优于直接应用token reduction的方法（如ToMe、PD）。 |
| **训练效率** | 相比CODA-Prompt，峰值内存 ↓1.6×，训练时间 ↓1.5×，能耗 ↓1.6×。 |
| **鲁棒性** | 即使在60%以上patch reduction下仍保持90%以上的baseline accuracy。 |
| **适用性** | 显式优化训练阶段效率，更适合真实边缘场景（已在Jetson Orin Nano验证）。 |

---

## 2. 核心实验方法和设置

### 数据集
在三个主流类增量学习 benchmark 上进行评估：
- **CIFAR-100**：10个任务，每任务10类
- **ImageNet-R**：10个任务，每任务20类
- **CUB-200**：10个任务，每任务20类  
所有数据均划分为10个非重叠任务，符合标准 class-incremental setting。

---

### 实验设置
- **Backbone**：ViT-Tiny/16（适合边缘部署）
- **预训练权重**：ImageNet-21K初始化 → ImageNet-1K微调
- **优化器**：Adam，batch size=16
- **训练轮数**：
  - ImageNet-R：50 epochs
  - 其他数据集：20 epochs
- **学习率调度**：cosine decay，初始 lr=0.001
- **超参数设置**：
  - 温度 $T$ 和相位比 $\lambda$ 分别设为 (0.1, 0.4), (0.1, 0.2), (0.1, 0.6) 对应不同数据集
  - Patch reduction ratio 固定为 0.4 进行主比较

---

### 评估指标
#### 准确性指标：
- **Average Accuracy (ACC↑)**：所有任务结束后的平均分类精度
- **Forgetting (FGT↓)**：模型遗忘先前任务的程度

#### 效率指标（在 Jetson Orin Nano 上实测）：
- **Peak GPU Memory Usage (MB)**：训练过程最大显存占用
- **Per-task Training Time (s/task)**：每个任务所需训练时间
- **Energy Consumption (Joules/task)**：单任务能耗

---

### 基线方法对比
#### 主要对比类别：
| 类型 | 方法 |
|------|------|
| **常规CL方法** | SGD, LwF, ER |
| **Prompt-based CL** | L2P, DualPrompt, CODA-Prompt, C-Prompt, OS-Prompt/++ |
| **Token Reduction 方法** | ToMe, PatchDropout (PD) |

特别以 **CODA-Prompt** 作为主要平衡基线，**C-Prompt** 作为准确率SOTA参考。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table 1 和 Figure 3）

| 方法 | CIFAR-100 ACC | ImageNet-R ACC | CUB-200 ACC | Peak Mem (avg) | Train Time (avg) | Energy (avg) |
|------|---------------|----------------|-------------|----------------|------------------|--------------|
| C-Prompt | **68.34** | **53.32** | 52.64 | ~1100 MB | ~2200 s | ~3300 J |
| CODA-Prompt | 67.06 | 50.24 | **53.96** | ~700 MB | ~1800 s | ~2400 J |
| **CPS-Prompt (Ours)** | 66.89 | 49.96 | 52.85 | **~440 MB** | **~1200 s** | **~1500 J** |

> 💡 结论：CPS-Prompt 在准确率上仅比C-Prompt低约2%，但内存、时间和能耗分别降低约 **1.6×**。

---

### 与其他方法的关键对比
- **vs. C-Prompt**：
  - 内存 ↓4.3×，训练时间 ↓3.1×，能耗 ↓3.3×
  - 准确率仅下降 ~2%，但在边缘设备更可行。
- **vs. CODA-Prompt**：
  - 无统计显著差异（p>0.05）在多数任务上
  - 资源消耗全面优于基线（内存↓1.6×，时间↓1.5×）
- **vs. OS-Prompt**：
  - 尽管采用两阶段架构（通常更耗资源），仍实现更低内存和更快训练
  - 显示出CPS+DPCT机制的有效性

---

### 消融实验结果（Table 2，ImageNet-R, reduction ratio=0.5）

| 配置 | ACC | Memory | Train Time |
|------|-----|--------|------------|
| CODA-Prompt (Full) | 50.24 | 440 MB | 1788 s |
| w/ PD (random drop) | 45.32 | 253 MB | 1388 s |
| w/ CPS | **47.16** | 253 MB | 1389 s |
| w/ PD + DPCT | 47.96 | 253 MB | **1126 s** |
| **w/ CPS + DPCT** | **49.28** | 253 MB | **1126 s** |

> ✅ 发现：
> - CPS 比随机PD提升 +1.84% accuracy（相同资源）
> - DPCT 单独带来约 +2% accuracy 回升，并大幅缩短训练时间（↓662s）
> - CPS + DPCT 组合效果最佳，接近原始性能且效率极高

---

## 4. 关键结论和发现

### 主要发现
1. **任务感知的 token sparsity 是关键**：
   - 盲目减少tokens（如ToMe、PD）严重损害PCL性能
   - 利用query encoder输出的任务相关信号指导patch选择（CPS），可在大幅压缩输入的同时保持高准确率。

2. **解耦训练有效缓解表示失配**：
   - DPCT通过分离prompt学习与classifier对齐，解决了稀疏训练与全输入推理之间的gap。
   - 同时冻结prompt后减少反向传播范围，显著加速训练。

3. **CPS-Prompt 实现了最优权衡**：
   - 在多个数据集和真实边缘硬件上验证，实现了当前最好的 accuracy-efficiency trade-off。
   - 特别适用于内存敏感、需长期在线更新的边缘AI系统（如机器人、无人机、智能摄像头）。

---

### 方法的局限性
- 当前依赖于两阶段PCL架构（如CODA-Prompt），可能不完全兼容所有prompt设计。
- CPS依赖于attention和value激活，若query encoder未能充分捕捉任务语义，则patch选择可能失效。
- 超参数（如温度$T$、phase ratio $\lambda$）需要根据数据集调整，在动态环境中可能需自适应机制。

---

### 未来工作方向
- 扩展至 **dynamic resource-aware settings**，根据实时内存/电量自动调节reduction ratio。
- 探索 **更广泛的CL场景**，如domain-incremental、task-free continual learning。
- 引入 **adaptive temperature 或可学习采样策略**，替代固定温度的softmax采样。
- 结合 **hardware-aware NAS** 进一步优化端到端部署性能。

---

> 🔚 总结：**CPS-Prompt 是首个明确面向“训练时效率”的PCL框架**，通过 **task-aware patch selection (CPS)** 与 **decoupled optimization (DPCT)**，在几乎不牺牲准确率的前提下，显著提升了边缘设备上的训练速度、内存利用率和能效，为可持续的 on-device continual learning 提供了实用解决方案。

</details>

---

### 6. [A Novel Edge-Assisted Quantum-Classical Hybrid Framework for Crime Pattern Learning and Classification](https://arxiv.org/abs/2604.07389)

**Authors**: Niloy Das, Apurba Adhikary, Sheikh Salman Hassan, Yu Qiao, Zhu Han, Tharmalingam Ratnarajah, Choong Seon Hong  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.07389v1  

#### Abstract
Crime pattern analysis is critical for law enforcement and predictive policing, yet the surge in criminal activities from rapid urbanization creates high-dimensional, imbalanced datasets that challenge traditional classification methods. This study presents a quantum-classical comparison framework f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Novel Edge-Assisted Quantum-Classical Hybrid Framework for Crime Pattern Learning and Classification

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该研究针对**犯罪模式分析中的三大挑战**：
- 高维特征空间与复杂依赖关系
- 犯罪类别严重不平衡（如凶杀案等罕见但关键的“Critical”类）
- 在资源受限边缘设备上部署模型时面临的计算复杂度限制

传统机器学习方法在处理上述问题时存在性能瓶颈，尤其是在内存和通信开销敏感的应用场景中（如无线传感器网络、智慧城市监控系统）。

---

### 🚀 提出的新方法与创新思路

1. **首次系统性地比较量子、经典与混合范式的犯罪分类性能**  
   构建了一个四范式比较框架，涵盖：
   - 纯量子模型（VQC, QAOA, QKernel SVM）
   - 纯经典模型（Random Forest, SVM, Logistic Regression）
   - 双向混合架构：
     - **Q→C**：量子特征提取 + 经典分类器
     - **C→Q**：经典降维（PCA） + 量子建模

2. **提出“相关性感知”的量子电路设计（Correlation-Aware Quantum Circuit）**  
   利用Spearman相关系数识别高相关特征对，并在VQC中通过定向CNOT门实现**基于真实数据结构的纠缠策略**，提升模型表达能力。

3. **面向边缘部署优化的轻量级量子架构探索**  
   强调参数数量、电路深度、qubit效率等指标，推动适用于NISQ设备和边缘节点的实际应用。

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **模型紧凑性** | QAOA仅需16个可训练参数（vs. Random Forest ~15万），降低9000倍内存占用 |
| **边缘适用性** | 极低推理开销（O(p·nq)），适合部署于1–4MB内存的无线传感器节点 |
| **通信成本** | 推理输出仅为标签+置信度（9字节），显著减少分布式系统通信负担 |
| **对少数类表现潜力** | QAOA在“Critical”类上相对表现优于多数类，显示其对复杂交互建模的能力 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **Bangladesh Police官方犯罪统计数据（2010–2025年）**
- 覆盖18个报告单位（metropolitan areas 和 police ranges）
- 总样本数：**272个观测值**（18单位 × 16年）
- 包含16种犯罪类型的计数及社会时间变量
- 工程化后选取10个关键特征（如暴力犯罪总数、女性儿童压迫案件数等）

> ⚠️ 注：数据经标准化、PCA降维至4或6维以适配NISQ设备限制（保留≥95%方差）

---

### 🎯 目标变量构建（4分类任务）
定义犯罪严重程度为四类：
```math
S(x) = 
\begin{cases}
\text{Critical}, & r_v > 0.3 \land C > 30,000 \\
\text{High},     & r_v > 0.15 \land C > 15,000 \\
\text{Medium},   & r_v > 0.05 \land C > 5,000 \\
\text{Low},      & \text{otherwise}
\end{cases}
```
其中 $ r_v $ 为暴力犯罪比例，$ C $ 为总案件数。

---

### 🧪 实验设置与评估指标

#### ✅ 评估协议
- **Stratified 5-Fold Cross-Validation**（每折重复5次随机种子，共25次评估）
- 替代初步实验中的单次80/20划分，避免乐观偏差

#### 📈 主要评估指标

| 类别 | 指标 |
|------|------|
| **分类性能** | Accuracy, Precision, Recall, F1-Score（加权平均） |
| **量子效率** | Parameter Count, Circuit Depth, Qubit Efficiency (Acc/nq) |
| **训练效率** | Mean Training Time per Fold |
| **理论复杂度** | Inference Complexity (O(...)) |
| **统计显著性** | Paired t-test (α=0.05), Cohen’s d effect size |

#### 🆚 基线方法对比
| 类型 | 模型列表 |
|------|--------|
| **Pure Quantum** | VQC, QAOA, QKernel SVM |
| **Pure Classical** | Random Forest, SVM(RBF), Logistic Regression, Decision Tree |
| **Hybrid Q→C** | Q→RF, Q→SVM, Q→LogReg, Q→DecisionTree |
| **Hybrid C→Q** | PCA→VQC, PCA→QAOA, PCA→QKernel |

所有量子模型均通过**经典模拟器**进行仿真（遵循NISQ时代QML基准测试标准 [5]）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自稳健交叉验证）

| Model | Acc. ± CI | F1 ± CI | Params | Train Time (s/fold) |
|-------|-----------|---------|--------|---------------------|
| **Random Forest (Best Classical)** | **0.945 ± 0.016** | **0.944 ± 0.016** | ~150k | 0.273 |
| **QAOA (6q, 3L)** | 0.846 ± 0.019 | 0.830 ± 0.021 | **36** | **0.004** |
| **QAOA (4q, 2L)** | 0.803 ± 0.024 | 0.779 ± 0.026 | **16** | 0.006 |
| **PCA→QAOA** | 0.803 ± 0.024 | 0.779 ± 0.026 | 16 | 0.006 |
| **Logistic Regression** | 0.895 ± 0.019 | 0.887 ± 0.018 | d=10 | 0.012 |
| **VQC (6q, 3L)** | 0.686 ± 0.021 | 0.672 ± 0.022 | 18 | 0.027 |
| **QKernel SVM** | 0.359 ± 0.028 | 0.351 ± 0.029 | — | 0.075 |

> ✅ **最佳量子模型：QAOA (6q, 3L)** 达到 **84.6% 准确率**

---

### 🔁 与基线方法的对比结果

| 对比维度 | 结果 |
|--------|------|
| **准确性** | Classical 显著优于 Quantum（p < 0.001, t = -23.06）<br>Random Forest 比 QAOA 高约10个百分点 |
| **参数效率** | QAOA 参数量仅为 RF 的 **1/9000**（16 vs ~150,000）<br>存储需求从 ~1.2MB 降至 **128 bytes**（float64） |
| **训练速度** | QAOA 训练最快（0.004s/fold），比 RF 快 **68倍** |
| **推理复杂度** | QAOA 推理为 O(p·nq)，远低于 RF 的 O(T·depth) |
| **通信开销** | 每次推理仅传输 **9 bytes**（label + confidence），极适合边缘协同 |

> ⚠️ 注意：初步单次划分实验曾显示 QAOA 与 classical 相当（85% vs 75%，p=0.1835），但交叉验证揭示其存在**乐观偏倚**

---

### 🔍 消融实验与关键发现

| 实验 | 发现 |
|------|------|
| **不同层数 vs 表达力（Expressibility）** | 图4显示：随着layer增加，circuit expressibility从0.35升至0.79，表明更深电路能更好捕捉非线性模式 |
| **VQC underperformance** | 所有VQC配置仅达~55%-68%准确率，归因于cosine-squared特征映射压缩输入信息 |
| **QKernel失败原因** | 四qubit下指数特征空间不足以有效分离数据，导致仅40%性能 |
| **Hybrid效率分析** | Q→C 混合结构训练快（0.007–0.02s），但未提升精度；C→Q 中 PCA→QAOA 表现与原生QAOA一致 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **QAOA是当前最有效的量子启发模型**  
   在所有量子与混合架构中表现最优（最高84.6%准确率），且具备极高的参数效率和训练速度。

2. **纯经典模型仍占主导地位**  
   Random Forest 以 **94.5%** 的准确率显著领先，凸显当前量子方法在绝对性能上的差距。

3. **量子方法的核心优势在于“资源效率”而非“精度超越”**  
   尽管精度落后，但QAOA的**极小参数足迹**使其成为边缘计算场景下的理想候选者。

4. **相关性驱动的纠缠设计具有实际意义**  
   基于Spearman相关性的CNOT连接策略提升了VQC的表达能力，验证了领域知识融入量子电路的有效性。

5. **单次划分评估易产生误导**  
   初步结果显示QAOA媲美经典模型，但交叉验证暴露其偏差，强调**严格统计验证的重要性**。

---

### ⚠️ 局限性

| 问题 | 描述 |
|------|------|
| **仅基于经典模拟** | 所有结果来自模拟器，未在真实量子硬件上运行，忽略噪声、退相干等现实影响 |
| **数据规模较小** | N=272，难以充分训练复杂模型，尤其限制深度学习与高维QML发挥 |
| **类别不平衡未完全解决** | “Critical”类占比低，虽加权F1缓解，但仍可能影响泛化 |
| **缺乏实时性测试** | 未在真实边缘设备或WSN平台上部署验证延迟与能耗 |

---

### 🔮 未来工作方向

1. **真实NISQ设备部署与benchmarking**  
   在IBM Quantum、IonQ等平台实测QAOA性能，纳入噪声模型与纠错机制。

2. **扩展更大规模数据集**  
   应用城市级实时犯罪流数据，探索在线学习与增量更新能力。

3. **优化混合架构设计**  
   探索更高效的Q→C特征融合方式，例如使用Transformer解码量子嵌入。

4. **边缘-云协同推理框架开发**  
   设计分层决策系统：边缘端运行轻量QAOA初筛，云端执行精细分析。

5. **引入更多domain knowledge into quantum ansatz**  
   如将地理邻接、时间周期性编码进U(x)或H_c中，进一步提升先验建模能力。

---

## ✅ 总结一句话

> 本论文首次系统评估了**量子-经典混合框架在犯罪模式分类中的可行性**，发现虽然当前**经典模型在精度上仍遥遥领先**，但**QAOA凭借极低参数量和高效训练**展现出巨大潜力，特别适用于**资源受限的边缘智能监控系统**，为未来量子机器学习在公共安全领域的落地提供了重要实证基础与设计范式。

</details>

---

### 7. [Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference](https://arxiv.org/abs/2604.07472)

**Authors**: Jiaming Cheng, Duong Tung Nguyen  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.07472v1  

#### Abstract
Deploying large language model (LLM) inference at scale requires jointly selecting base models, provisioning heterogeneous GPUs, configuring parallelism, and distributing workloads under tight latency, accuracy, and budget constraints. Exact mixed-integer linear programming (MILP) approaches guarant...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对大规模 **LLM inference serving** 中的联合优化难题，即在严格的 **SLO（Service-Level Objectives）约束**（延迟、精度、预算）下，如何高效地进行以下决策：
- 基础模型选择（Foundation model selection）
- 异构 GPU 资源配置（Heterogeneous GPU provisioning）
- 并行策略配置（Tensor Parallelism 和 Pipeline Parallelism）
- 工作负载分配与路由（Workload allocation and routing）

传统基于 **Mixed-Integer Linear Programming (MILP)** 的精确求解器虽然能保证最优性，但计算复杂度高，难以扩展到大规模场景，无法满足实时重优化需求。

---

### 提出的新方法与新思路
作者提出了两种**约束感知的启发式算法**（constraint-aware heuristics）：

1. **Greedy Heuristic (GH)**  
   单次贪心分配算法，通过三个关键机制确保每一步都满足多维约束。

2. **Adaptive Greedy Heuristic (AGH)**  
   在 GH 基础上增强，引入：
   - 多起点构造（multi-start construction）
   - 基于重定位的局部搜索（relocate-based local search）
   - GPU 资源整合（consolidation）

#### 三大核心机制（M1–M3）——“可行性前提”而非简单优化
| 机制 | 功能 |
|------|------|
| **M1: TP-aware feasibility selection** | 对每个 `(query type, model, GPU tier)` 组合，预筛选满足内存和延迟 SLO 的最小成本 TP/PP 配置；若无可行配置则直接丢弃候选 |
| **M2: cost-per-effective-coverage ranking** | 不按原始成本排序，而是按单位有效服务量的成本排序（考虑误差和延迟累积限制） |
| **M3: TP upgrade for active GPUs** | 若已有部署的配置延迟超标，则尝试升级其 TP 度数（复用已加载模型），避免重复激活开销 |

这些机制是**确保最终解可行的关键**，而非仅用于提升质量。

---

### 相比现有方法的优势
| 维度 | 本工作（GH/AGH） | 现有方法（如 Helix [5], Jiang et al. [6]） |
|------|------------------|----------------------------------------|
| **可扩展性** | ✅ 子秒级运行（<1s），支持大规模实例 | ❌ MILP 求解时间随规模指数增长（可达分钟甚至小时） |
| **联合优化能力** | ✅ 同时优化模型选择、GPU 配置、并行度、路由 | ❌ 多数系统固定部分参数（如先定 TP 再路由） |
| **鲁棒性** | ✅ 在压力测试下保持稳定成本与可控 SLO 违规率 | ❌ 精确解对参数扰动敏感，性能急剧下降 |
| **实用性** | ✅ 支持滚动重优化（rolling re-optimization），适应动态负载 | ❌ 静态部署为主，难以频繁更新 |

> ⚡️ **关键优势总结**：  
> **>260× speedup** vs. exact MILP，同时接近最优成本，并具备更强的操作鲁棒性和实时响应能力。

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **Azure LLM Inference Trace (2025)** [13] 进行工作负载校准。
- 包含 6 类查询类型（Query Types）：
  - Summarization, Code Generation, Translation, Math Solving, Image Generation, Video Generation
- 查询特征包括到达率（arrival rate）、输入/输出长度等。

### 实验设置
- **模型集合 J**：6 个 Llama-3.x 模型（1B ~ 70B 参数）
- **GPU Tier K**：10 种异构组合（如 H100-FP16, A6000-INT8, RTX4090-INT4）
- **TP Degree**：{1,2,4,8}；**PP Depth**：{1,2,4}
- **规划周期 △T**：24 小时
- **预算上限**：$100（基础场景）

### 评估指标
| 指标 | 描述 |
|------|------|
| **Total Cost ($)** | 包括 GPU 租赁、模型存储、token 存储、延迟惩罚、未满足请求惩罚 |
| **SLO Violation Rate (%)** | 请求因延迟或错误超出阈值而未能被服务的比例 |
| **Runtime (s)** | 算法执行时间 |
| **Robustness** | 在 out-of-sample 压力测试下的稳定性表现 |

### 基线方法对比
| 方法 | 类型 |
|------|------|
| **Exact MILP Solver (DM)** | 使用 Gurobi 求解完整 MILP 问题，作为“最优”基准 |
| **Greedy Heuristic (GH)** | 单次贪心构造 |
| **Adaptive Greedy Heuristic (AGH)** | 多起点 + 局部搜索 + 整合优化 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 结果 |
|------|------|
| **运行时间** | GH: <0.01–0.9s；AGH: 0.01–2.3s；MILP: 最高达 >600s（超时） |
| **加速比** | AGH 相比 MILP 实现 **>260× speedup**（在大实例上） |
| **成本接近度** | AGH 成本仅比最优解高出约 5–10%，远优于 GH |
| **SLO 违规率** | 在 1.5× 延迟/误差膨胀的压力测试下，AGH 控制在 <4%，而 MILP 达到 14%+ |

---

### 与基线方法对比结果

#### 表格：Stage-2 实验（压力测试下实际成本与 SLO 违规）

| 场景 | 方法 | 成本 ($) | SLO Viol. (%) |
|------|------|----------|---------------|
| S3: Critical Budget ($72) | GH | 1162.0 | 14.2 |
| | AGH | **343.0** | **3.7** |
| | Gap | ↓70% | ↓74% |
| S5: High Penalty + Critical Budget | GH | 1811.0 | 14.2 |
| | AGH | **344.0** | **3.7** |
| | Gap | ↓81% | ↓74% |

> ✅ AGH 在紧预算和高压环境下显著优于 GH 和 MILP。

#### 图表观察（Fig. 2）
- 在 **1.5× delay/error inflation** 下：
  - MILP 的实际成本飙升（due to high unmet penalty）
  - AGH 成本稳定，SLO 违规率低
- AGH 的保守设计（provisioning headroom）天然具有抗扰动能力。

---

### 消融实验结果（Ablation Study）

| 配置 | 是否可行？ | 成本 ($) | 变化 |
|------|------------|---------|------|
| 完整 AGH（所有机制） | ✅ Yes | 89.88 | — |
| 移除 M1（TP selection） | ❌ No（内存/延迟违规） | — | 不可行 |
| 移除 M2（cost ranking） | ✅ Yes | 134.52 | ↑>50% |
| 移除 M3（TP upgrade） | ❌ No（延迟违规） | — | 不可行 |

> 🔍 **关键发现**：  
> **M1 和 M3 是可行性必要条件**，不是可选优化项。这表明本文提出的“约束感知”机制本质不同于普通 GRASP 类启发式。

---

## 4. 关键结论和发现

### 主要发现
1. **约束耦合性强导致贪婪策略易失败**  
   内存、延迟、误差、预算之间高度耦合，标准贪心极易产生不可行解。

2. **AGH 实现近似最优 + 极速求解 + 高鲁棒性三者平衡**  
   - 接近 MILP 最优成本
   - 运行时间控制在 **1 秒以内**
   - 在参数扰动下仍保持稳定性能

3. **“保守性”带来鲁棒性**  
   AGH 自然倾向于预留资源（headroom），从而吸收需求波动，而追求“极致节省”的 MILP 解反而更脆弱。

4. **滚动重优化创造复利优势**  
   利用 AGH 的高速特性，实现 **每 5 分钟重新优化一次**（rolling-horizon re-optimization），在高波动场景下相比静态部署最多节省 **48% 成本**。

5. **GH 对重优化不敏感，AGH 可从中获益**  
   因为 GH 是确定性顺序，需求变化后仍生成相同解；而 AGH 的随机多起点能探索新空间。

---

### 方法的局限性
- 当前假设为 **静态平均负载建模**，未显式处理队列动态或突发流量。
- 并行配置空间受限于预定义集合（如 TP∈{1,2,4,8}），未探索更细粒度或混合精度策略。
- 实验基于模拟器，尚未在真实集群中验证通信开销与调度延迟的影响。

---

### 未来工作方向
1. 引入 **stochastic optimization** 框架以显式建模不确定性。
2. 结合 **queuing theory** 与 **continuous batching dynamics** 更准确刻画推理延迟。
3. 扩展至 **multi-region, cross-cloud** 场景下的分布式部署。
4. 在真实生产环境中集成并验证（real cluster validation）。

---

## 总结

✅ **本文亮点一句话概括**：  
提出了一套**快速、可行、鲁棒且可滚动优化**的异构 LLM 推理资源配置框架（GH/AGH），通过三个**约束感知机制**解决了传统 MILP 方法不可扩展、系统级启发式不可行的痛点，在真实轨迹驱动的实验中实现了 **>260× 加速 + 接近最优成本 + 出色抗压能力**，为大规模 LLM serving 提供了实用高效的解决方案。

</details>

---

### 8. [Reduced-Mass Orbital AI Inference via Integrated Solar, Compute, and Radiator Panels](https://arxiv.org/abs/2604.07760)

**Authors**: Stephen Gaalema, Samuel Indyk, Clinton Staley  
**Category**: cs.DC  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.07760v1  

#### Abstract
We describe and analyze a distributed compute architecture for SSO computational satellites that can potentially provide >100 kW compute power per launched metric ton (including deployment and station keeping mass). The architecture co-locates and integrates the solar cells, radiator, and compute fu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Reduced-Mass Orbital AI Inference via Integrated Solar, Compute, and Radiator Panels*

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**轨道数据中心**（Orbital Data Center, ODC）在大规模部署AI计算任务时面临的关键挑战，提出了一种全新的系统架构。传统ODC设计通常采用分立子系统（独立的太阳能阵列、计算模块和热辐射器），存在以下问题：
- 质量效率低（specific power < 100 W/kg）
- 热管理复杂，高温导致IC能效下降
- 结构冗余，部署体积大
- 难以实现单次发射的大规模部署

### 提出的新方法与创新思路
作者提出了 **“集成式太阳能-计算-散热板”**（Integrated Solar Compute Radiator, **ISCR**）架构，其核心创新包括：

- **功能一体化设计**：将太阳能电池、计算芯片（ICs）和**vapor chamber**散热背板集成于同一块轻质面板中，形成多功能复合结构。
- **分布式计算拓扑**：数千个ISCR面板组成大型线性阵列，实现分布式计算与热管理，无需集中式冷却系统。
- **结构功能复用**：利用**vapor chamber**作为太阳能电池的唯一机械支撑基底，消除传统独立结构需求。
- **低温高效运行**：通过高效的被动散热，使IC结温（junction temperature）维持在约40°C，显著提升能效与可靠性。

### 相比现有方法的优势
| 维度 | ISCR 架构优势 |
|------|----------------|
| **Specific Power** | 达到 ~506 W/kg（太阳能阵列层面），系统级达 **112.5 kW/ton**，远超传统卫星的 <100 W/kg |
| **热管理效率** | 单面散热器可在370 W/m²功率密度下维持背板温度<30°C，支持低结温运行 |
| **部署可行性** | 可卷绕收纳，适配Starship货舱（8m×22m），支持单星16 MW级计算能力 |
| **可扩展性** | 支持从百kW到数十MW的平滑扩展，降低单位功率的总线开销 |
| **容错性** | 面板级故障不影响整体运行，支持动态重构 |

---

## 2. 核心实验方法和设置

> 注：本文为**概念性系统设计与建模分析**，非传统意义上的“实验”，所有结果基于物理建模、参数估算与仿真推导。

### 分析对象与场景设定
- **轨道环境**：晨昏线太阳同步轨道（Dawn-Dusk Sun-Synchronous Orbit, SSO），高度600–1000 km
- **电源假设**：采用**Perovskite/Si tandem solar cells**，效率27%，输出功率密度~363 W/m²
- **计算单元**：每块面板提供 **1 kW** 计算功耗，搭载定制ASIC或GPU级IC
- **散热机制**：基于水工质的**flat vapor chamber**，工作温度~25–30°C
- **通信架构**：面板间采用铜线**NVLink/UA Link**（100 GB/s双工），中心节点使用光纤

### 评估指标
| 指标 | 定义 |
|------|------|
| **Specific Power (kW/ton)** | 每吨发射质量提供的有效计算功率 |
| **Energy per Token (Joules/token)** | 推理一个token所需的能量，反映能效 |
| **Tokens/sec/session** | 单会话推理吞吐率 |
| **Concurrent Inferences** | 同时支持的在线推理会话数 |
| **Junction Temperature (Tj)** | IC工作结温，影响性能与寿命 |
| **Stowed Density** | 收纳状态下的面积密度（kW/m³） |

### 基线对比方法
论文未直接对比具体“基线算法”，而是对比如下典型ODC架构：
- **High-T Radiator Design**：高温双面辐射器（>80°C），类似地面服务器液冷
- **Centralized Compute Architecture**：分离式太阳能+集中计算+主动冷却
- **Tether-Based ODC**（如Bargatin等）：垂直系绳结构
- **Constellation Approach**（如Google Suncatcher）：多星编队

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 参数 | 数值 | 来源 |
|------|------|------|
| **面板尺寸** | 1.7m × 1.7m (~2.9 m²)，厚度<7 mm | Fig. 3, Table 5 |
| **单面板功率** | 1 kW（用于计算与通信） | Section 4.1 |
| **系统总功率** | **16.4 MW**（150吨卫星，16,000面板） | Section 6.2 |
| **Specific Power** | **112.5 kW/ton**（含部署与驻留推进） | Table 7 |
| **太阳能阵列比功率** | **506 W/kg** | Section 7.1 |
| **IC结温** | ~40–42°C（对应散热器35°C） | Table 4 |
| **能量每token** | **0.204 J/token**（优于高温设计30%以上） | Table 4 |
| **最大并发推理** | **>7,900 inferences**（全卫星） | Abstract |

### 与基线方法的对比结果
#### （1）能效对比（Energy per Token）
| 冷却方案 | 散热器温度 | IC结温 | Clock Speed | Energy/Token |
|---------|------------|--------|-------------|---------------|
| ISCR (vapor chamber) | 35°C | ~40–42°C | 2.6 GHz | **0.204 J** |
| Liquid Cooled (Baseline) | 45°C | ~85–90°C | 2.38 GHz | 0.213 J |
| High-T Radiator | 60°C | ~103–105°C | 2.05 GHz | 0.274 J |

> ✅ ISCR相比高温设计节能 **>30%**

#### （2）结构效率对比
| 架构类型 | 比功率 (W/kg) | 是否支持单星MW级 | 部署复杂度 |
|--------|----------------|--------------------|------------|
| 传统卫星 | <100 | ❌ | 中高 |
| ISCR Panel | **506**（仅太阳能层） | ✅ | 低（卷展式） |
| Tether-Based | ~300–400（估计） | ✅ | 高（姿态控制难） |

> ✅ ISCR在结构效率上具有数量级优势

### 大模型推理性能（LLM Inference）
| 配置 | 上下文长度 | 并发会话 | 吞吐量 (tokens/sec/session) | 面板数 |
|------|------------|----------|-------------------------------|--------|
| 512-panel heavy LLM | 500,000 | 256 | **553** | 512 |
| 全卫星（31个子阵列） | —— | **>7,900** | ~550 | 16,000 |

> 💡 支持长上下文（50万token）、高并发的LLM推理服务

---

## 4. 关键结论和发现

### 主要发现
1. **低温显著提升能效**：  
   将IC结温从105°C降至~40°C，可使**Energy per Token降低超过30%**，主要得益于更低的漏电流（leakage current）和更高的时钟频率。

2. **结构集成带来质量革命**：  
   利用**vapor chamber**作为太阳能电池的结构基底，可消除专用支架，实现**~5倍以上的比功率提升**（506 vs <100 W/kg）。

3. **分布式架构更具可扩展性**：  
   千级面板组成的线性阵列天然支持**pipeline parallelism**，适合LLM推理；且具备良好的容错性和部署灵活性。

4. **单星可承载超大规模计算**：  
   一颗150吨级卫星即可提供**16 MW计算能力**，相当于数万台高端服务器，且完全由太阳能驱动。

5. **经济潜力巨大**：  
   若硅片成本成为瓶颈，ISCR的高能效与高良率小芯片设计更具竞争力；若发射成本下降，其大尺寸优势将进一步放大。

---

### 方法的局限性
| 局限性 | 说明 |
|-------|------|
| **尚未实测验证** | 所有热力学、电学、结构性能均基于建模与外推，缺乏飞行或地面原型验证 |
| **辐射防护不确定性** | 当前HDPE屏蔽层（3 mm）对长期LEO辐射环境的有效性需进一步测试 |
| **动态稳定性风险** | 2.2 km长柔性结构的姿态控制、扭转振动等问题尚未进行完整FEA分析 |
| **制造工艺挑战** | 连续化生产万级同质面板的质量一致性要求极高 |
| **依赖定制组件** | 性能优势建立在定制太阳能电池、ASIC、vapor chamber基础上，初期NRE成本高 |

---

### 未来工作方向
1. **Panel-Level Investigations**
   - **3D thermal simulation** of vapor chamber + compute module
   - **Radiation qualification** of Si/Perovskite cells and ICs in LEO
   - **Custom ASIC architecture definition**（memory hierarchy, interconnect）
   - 分析**sub-2 nm节点**在低温下的VF特性优化空间

2. **Satellite-Level Investigations**
   - **Structural statics & dynamics FEA**：评估柔性阵列的平面保持能力
   - **Fault tolerance algorithms**：研究面板失效后的动态重配置策略
   - **Orbital transfer analysis**：计算LEO→SSO的推进剂预算与时间
   - **Single- vs Dual-Bus design trade-off**：比较不同部署构型的刚度与控制性能

3. **系统级成本建模**
   - 开展**launch cost vs component cost sensitivity analysis**
   - 对比ISCR与其他ODC架构在不同市场条件下的**$ per FLOP-year**经济性

---

> 🔚 **总结**：本文提出的ISCR架构是一种面向未来的**高能效、高密度、可扩展**的轨道AI基础设施范式，虽仍处于概念阶段，但其在**thermal management efficiency**、**structural integration** 和 **system scalability** 上展现出颠覆性潜力，为下一代太空计算提供了重要技术路径。

</details>

---

### 9. [Reinforcement Learning with Reward Machines for Sleep Control in Mobile Networks](https://arxiv.org/abs/2604.07411)

**Authors**: Kristina Levina, Nikolaos Pappas, Athanasios Karapantelakis, Aneta Vulgarakis Feljan, Jendrik Seipp  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.07411v1  

#### Abstract
Energy efficiency in mobile networks is crucial for sustainable telecommunications infrastructure, particularly as network densification continues to increase power consumption. Sleep mechanisms for the components in mobile networks can reduce energy use, but deciding which components to put to slee...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Reinforcement Learning with Reward Machines for Sleep Control in Mobile Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对移动网络中 **Radio Units (RUs)** 的 **Sleep Mode (SM) 控制问题**，旨在在保证服务质量（QoS）的前提下最大化 **Energy Efficiency (EE)**。该问题的关键挑战在于：
- QoS 约束是 **time-averaged** 的（如平均丢包率、最小吞吐量），而非瞬时性能；
- 这些约束具有 **非马尔可夫性（non-Markovian）**，即当前决策的影响依赖于历史状态；
- 传统方法难以有效建模长期约束与能量节省之间的权衡。

### 🚀 提出的新方法/新思路
提出将 **Reinforcement Learning (RL)** 与 **Reward Machines (RMs)** 结合，构建一个 **MDP with Reward Machines (MDPRM)** 框架来处理非马尔可夫奖励结构：
- 使用两个独立的 **Reward Machine** 分别跟踪：
  - Deadline-constrained 用户的 **packet drop rate violation**
  - Constant-rate 用户的 **throughput violation**
- RM 通过维护抽象状态（abstract state）显式记录历史违反情况，从而将原本非马尔可夫的问题转化为马尔可夫形式；
- 最终奖励函数为：  
  $$
  R_{\text{RM}} = \text{EE} + r_d + r_m
  $$  
  其中 $r_d$ 和 $r_m$ 来自两个 RM 的输出，体现对长期约束的动态响应。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | 本论文优势 |
|------|--------|------------|
| **Lyapunov Optimisation** | 需每时隙求解复杂优化问题，扩展性差；无法适应高维动作空间 | 不依赖系统模型，支持在线学习，适用于大规模 RU 联合控制 |
| **CMDP / Lagrangian 方法** | 政策无记忆性，难以捕捉时间相关性 | RM 显式引入有限状态记忆，能更好处理多时隙承诺和累积约束 |
| **标准 RL（Markovian Reward）** | 奖励仅基于当前状态，忽略历史趋势 | RM 提供结构化记忆，显著提升长期约束满足能力 |

> ✅ **核心创新**：首次将 Reward Machines 引入移动网络节能控制领域，为解决 **time-averaged QoS 约束下的非马尔可夫决策问题** 提供了一种可扩展、原理清晰的框架。

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真环境
- **未使用真实数据集**，而是基于 **自研系统级仿真工具** 构建场景；
- 采用简化的 **map-based ray-tracing 传播模型** 计算路径增益；
- 用户位置、信道条件、流量负载等随机生成并重置每轮 episode。

### ⚙️ 实验设置
| 参数 | 设置 |
|------|------|
| RUs 数量 | 4 |
| Sleep Modes (SMs) | 4 种：<br>- SM1: 71 μs<br>- SM2: 1 ms<br>- SM3: 10 ms<br>- SM4: 1 s |
| 动作空间大小 | $(H+1)^G = 5^4 = 625$ → 视为连续空间处理 |
| RL 算法 | **TD3**（Twin Delayed Deep Deterministic Policy Gradient） |
| 网络架构 | Actor/Critic 均为两层全连接网络（400, 300 neurons），ReLU 激活 |
| 输出处理 | Tanh 激活后离散化到最近 SM 等级 |
| 学习参数 | Adam 优化器（lr=3e-4），折扣因子 γ=0.2，回放缓冲区大小=1e6，mini-batch=256 |
| 训练配置 | 5000 episodes，每 episode 30 步，MacBook Pro M4 芯片运行 |

### 🎯 评估指标
1. **Energy Efficiency (EE)**：相对能耗节约比例
2. **Power Consumption**：总功耗
3. **Constraint Satisfaction**：
   - 平均丢包率是否低于阈值 $D$
   - 平均吞吐量是否高于要求 $\mu$
4. **Policy Behavior Analysis**：
   - Power cycling 频率
   - 各 SM 使用分布

### 🆚 基线方法对比
共比较四种奖励机制（其余设置相同）：
1. **Deep RM** ($L=100$) —— 本文提出，深度记忆
2. **Shallow RM** ($L=10$) —— 浅层记忆对照
3. **Markovian Reward** —— 即 $R = \text{EE} - \rho_d - \rho_m$
4. **Lagrangian Optimisation (LO)** —— 使用 Lagrange multiplier 更新：$\lambda_d, \lambda_m$

---

## 3. 主要实验结果和性能指标

### 📈 关键性能表现（见 Fig. 2）
| 方法 | Energy Efficiency | 功耗水平 | 约束满足度 |
|------|-------------------|----------|-----------|
| **Deep RM (L=100)** | ✅ **最高 EE (~0.78)** | ❌ 略高波动 | 接近边界但仍满足 |
| **Shallow RM (L=10)** | 中等 (~0.72) | 较低 | 更保守，远离边界 |
| **LO Reward** | 中等 (~0.70) | 中等 | 约束满足良好 |
| **Markovian Reward** | ❌ 最低 (~0.65) | 最高 | 容易违反约束 |

> 💡 **关键观察**：Deep RM 在保持 QoS 合格的同时实现了最佳 EE，说明其能更精细地利用“违规预算”（violation budget）换取长期节能。

### 🔬 消融实验分析（RM 深度影响）
- **RM 深度 $L$ 是关键设计参数**：
  - $L=100$：拥有更强的历史记忆能力，策略更具适应性；
  - $L=10$：记忆有限，行为更保守，错失节能机会；
- 图 3 显示：Deep RM 的 **power cycling 更频繁**，表明其策略更灵活、响应更快；
- 图 4 显示：Deep RM 更多地使用 **SM4（最长睡眠）**，但也合理调用短 SM（SM1–SM3）应对突发需求。

### 📊 性能对比总结
| 维度 | Deep RM 表现 |
|------|--------------|
| **EE 提升** | 比 Markovian 提升约 **20%** |
| **约束贴近性** | 可安全运行在约束边界附近，资源利用率更高 |
| **策略灵活性** | 支持细粒度、场景自适应的 sleep 控制 |
| **收敛稳定性** | 所有方法均稳定收敛，Deep RM 方差略大但可控 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Reward Machines 能有效建模非马尔可夫 QoS 约束**，使 RL 智能体能够学习兼顾短期节能与长期服务质量的策略；
2. **RM 的深度 $L$ 决定了历史记忆容量**，更深的 RM 可实现更优的 EE-QoS 权衡；
3. 与传统 Markovian RL 或 Lagrangian 方法相比，RM-based 方法在复杂无线环境中表现出更强的 **适应性和鲁棒性**；
4. 所提框架具有良好的 **scalability**，适合未来 6G 网络中高密度 RU 的联合控制。

### ⚠️ 方法局限性
1. **RM 设计依赖人工先验知识**：需预先定义命题符号（propositional symbols）和状态转移逻辑；
2. **计算开销随 $L$ 增加而上升**：过深的 RM 可能导致状态爆炸，影响训练效率；
3. **仿真环境简化**：未考虑完全真实的用户移动性、干扰模型或跨小区协作；
4. **泛化能力待验证**：不同拓扑或流量模式下的迁移性能尚未测试。

### 🔮 未来工作方向
1. **自动化 RM 构造**：探索从任务描述或专家轨迹中自动归纳 RM 结构的方法（如使用 LTL 或神经符号学习）；
2. **轻量化 RM 设计**：研究稀疏化、分层或模块化 RM 以降低复杂度；
3. **多基站协同控制**：扩展至 multi-cell 场景，结合 federated RL 或 multi-agent RM；
4. **实际部署验证**：在 testbed 或商用网络中进行原型验证；
5. **与其他约束方法融合**：例如结合 Lyapunov drift 或 safe RL 技术进一步增强安全性。

---

> 🏁 **总结一句话**：  
> 本文提出一种基于 **Reward Machines** 的新型 RL 框架，成功解决了移动网络中因 **time-averaged QoS 约束导致的非马尔可夫决策难题**，在保证服务质量的前提下显著提升了能源效率，为下一代绿色通信系统的智能管控提供了新范式。

</details>

---

### 10. [Bit-by-Bit: Progressive QAT Strategy with Outlier Channel Splitting for Stable Low-Bit LLMs](https://arxiv.org/abs/2604.07888)

**Authors**: Binxing Xu, Hao Gu, Lujun Li, Hao Wang, Bei Liu, Jiacheng Liu, Qiyuan Zhu, Xintong Yang, Chao Li, Sirui Han, Yike Guo  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.07888v1  

#### Abstract
Training LLMs at ultra-low precision remains a formidable challenge. Direct low-bit QAT often suffers from convergence instability and substantial training costs, exacerbated by quantization noise from heavy-tailed outlier channels and error accumulation across layers. To address these issues, we pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Bit-by-Bit: Progressive QAT Strategy with Outlier Channel Splitting for Stable Low-Bit LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大语言模型（LLMs）在超低比特（如2-bit）量化感知训练（QAT）中面临严重挑战，主要包括：
- **收敛不稳定**：直接进行低比特QAT容易陷入尖锐的损失尖峰（loss spike）和非光滑损失景观（loss landscape）。
- **误差累积**：深层Transformer块中的量化误差逐层累积，导致性能急剧下降。
- **计算开销大**：现有QAT方法依赖大量训练token和复杂的蒸馏机制，训练成本高昂。
- **缺乏灵活性**：传统方法需为不同比特宽度单独训练多个模型，存储和部署成本高。

---

### 🚀 提出的新方法与核心创新
作者提出 **BIT-BY-BIT**，一个稳定、高效的渐进式QAT框架，包含三大关键技术：

#### （1）**Block-wise Progressive QAT**
- **策略**：从高精度（如8-bit）逐步退火到低精度（如4-bit → 2-bit），先量化权重再量化激活。
- **优势**：提供良好的初始化，避免直接进入粗糙的低比特优化空间，显著提升稳定性。

#### （2）**Once-for-any-precision Framework**
- **思想**：利用低比特网格的**嵌套性**（nested grids），通过位移操作（bit shift）实现“一次训练，任意精度部署”。
- **实现**：共享一组主参数，在推理时按需截断至目标比特（如支持W8/W4/W2）。
- **优势**：无需为每个比特重新训练，节省训练和存储成本。

#### （3）**Rounding-aware Outlier Channel Splitting (OCS)**
- **机制**：对敏感通道（outlier channels）进行分裂，将原权重 $ w_m $ 拆分为两个分支 $ (w_m - s/2)/2 $ 和 $ (w_m + s/2)/2 $，保持输出不变。
- **优点**：
  - 减少动态范围，缓解量化误差；
  - 保留原始语义信息（不丢弃outliers）；
  - **四倍降低MSE误差**（理论证明见附录C）。

#### （4）**Microscaling with E4M3 FP8 Scales**
- 使用每组32个元素的微缩放（microscaling），并采用 **E4M3 FP8** 格式存储scale，兼顾精度与效率。
- 相比传统E8M0更适用于2-bit场景，仅增加0.25 bit/weight的开销。

---

### 🔍 相比现有方法的优势
| 维度 | BIT-BY-BIT | 传统QAT（如BitDistiller, EfficientQAT） |
|------|------------|----------------------------------------|
| 稳定性 | 高（无loss spike） | 易发散，需精细调参 |
| 训练成本 | 极低（3600× token减少） | 需数亿至数十亿token |
| 多精度支持 | 单模型支持多bit（train once, deploy any） | 每bit需独立训练 |
| 内核效率 | 自研W2A2/W2A16 CUDA kernel，达11×加速 | 依赖通用kernel，效率低 |
| 误差控制 | OCS + 渐进训练双重抑制误差 | 依赖剪枝或蒸馏 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **校准/训练数据**：
  - RedPajama 子集（4096样本，seq len=2048）
- **评估任务**：
  - **语言建模**：WikiText2、C4（测试集）
  - **零样本推理**：PIQA、ARC-Easy、ARC-Challenge、HellaSwag、Winogrande
  - **高级推理**：GSM8k、MathQA、MMLU、IFEval

### ⚙️ 实验设置
- **模型家族**：
  - LLaMA-2 / LLaMA-3（2B ~ 13B）
  - Mistral-7B
  - Qwen2.5（7B / 14B）
- **量化配置**：
  - **Weight-only**: W2A16
  - **Weight-Activation**: W2A2
  - Group size = 32
- **训练预算对齐**：
  - 所有QAT方法统一使用约2 epoch、4096样本，确保公平比较。

### 📊 评估指标
| 类型 | 指标 |
|------|------|
| 语言建模 | Perplexity (PPL) on WikiText2 / C4 |
| 推理能力 | Zero-shot Accuracy (%) |
| 效率 | GEMV延迟（μs）、端到端吞吐量（tokens/s） |
| 消融分析 | 不同组件组合下的PPL变化 |

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **PTQ** | GPTQ, AWQ, SmoothQuant, SpinQuant, OmniQuant |
| **QAT** | EfficientQAT, BitDistiller, ParetoQ |
| **Multi-bit** | MatQuant, OmniQuant（多精度版本） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（LLaMA-2 7B, W2A2）

| 方法 | WikiText2 PPL | 相比FP16增量 |
|------|---------------|--------------|
| FP16（全精度） | 5.47 | — |
| **BIT-BY-BIT (Ours)** | **7.72** | **+2.25** |
| EfficientQAT | 9.71 | +4.24 |
| BitDistiller | 29.66 | +24.19 |
| ParetoQ | 259.74 | +254.27 |

> ✅ **结论**：BIT-BY-BIT在W2A2下仅增加2.25 PPL，远优于所有基线。

---

### 📊 零样本推理准确率（LLaMA-3.2-3B）

| 方法 | Avg Accuracy (w2a16) | Avg Accuracy (w2a2) |
|------|------------------------|------------------------|
| FP16 | 67.67 | — |
| EfficientQAT | 55.89 | 40.10 |
| BitDistiller | 56.18 | 46.28 |
| **BIT-BY-BIT (Ours)** | **56.91** | **51.52** |

> ✅ 在w2a2下仍保持超过51%平均准确率，领先第二名（BitDistiller）**5.24个百分点**。

---

### ⏱️ 推理速度与能效
| 配置 | GEMV加速比（vs BF16） | 端到端吞吐量（Llama3-8B） |
|------|--------------------------|----------------------------|
| W2A2 | **最高11×**（大矩阵） | **76 tokens/s**（vs 49） |
| W2A16 | 最高1.2× | — |

> 💡 自定义CUDA kernel充分利用dp4a指令和bit-packing，显著提升内存带宽利用率。

---

### 🔬 消融实验（LLaMA-3.2-1B, W2A2）

| 组件 | WikiText2 PPL |
|------|----------------|
| Baseline（无任何改进） | 2.0e5（崩溃） |
| + Block-wise | 1441.9 |
| + Progressive | **42.2** |
| + OCS + Metric (`|x|₂·max|w|`) | **32.34** |

> ✅ 渐进训练是稳定性的关键；OCS进一步带来巨大收益。

#### 不同outlier metric效果对比：
| Metric | PPL |
|--------|-----|
| `wmax` | 36.75 |
| `xmax` | 32.34 |
| `|x|₂·max|w|` | 32.48 |

> ✅ 激活相关的metric在W2A2中更为有效。

#### Group Size影响：
| Group Size | PPL |
|------------|-----|
| 32 | 32.34 |
| 64 | 121.87 |
| 128 | 261.28 |

> ❗ 超低比特下必须使用细粒度分组（如32）以维持精度。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **渐进式训练是超低比特QAT稳定的基石**：从高到低的精度退火可避免陷入局部最优。
2. **嵌套量化网格支持“一次训练，任意部署”**：通过bit shift实现灵活多精度服务，极大降低运维成本。
3. **OCS显著降低量化误差且保持输出恒定**：相比clipping更安全，理论可降4× MSE。
4. **自定义W2A2 kernel可行且高效**：即使在无原生2-bit支持的GPU上也能实现10×以上加速。
5. **新型架构（如Qwen）对量化更鲁棒**：Qwen2.5-14B在W2A2下几乎无损，而Llama2-13B性能腰斩。

---

### ⚠️ 局限性
1. **对部分模型族泛化较差**：在Qwen系列上表现良好，但在其他模型（如文中未详述者）可能存在更大性能下降。
2. **分布式训练适配困难**：Block-wise训练破坏负载均衡，不利于大规模并行。
3. **尚未探索MoE或KV Cache量化**：当前方法聚焦Dense Transformer，未扩展至稀疏架构或长上下文场景。
4. **自动调度缺失**：split ratio、learning rate schedule仍需手动设定。

---

### 🔮 未来工作方向
1. **自动化layer-wise调度**：学习每层的最佳量化策略与split比例。
2. **混合精度搜索（Hardware-aware Mixed Precision Search）**：结合NAS技术寻找最优bit分配。
3. **扩展至MoE模型与KV Cache量化**：支持更复杂架构与长序列推理。
4. **轻量级蒸馏集成**：结合知识蒸馏进一步压缩表示差距。
5. **端侧联合优化**：将编译器、硬件特性纳入训练闭环，实现软硬协同设计。

---

> 🧩 **总体评价**：  
> BIT-BY-BIT 是首个在 **W2A2** 下实现接近全精度性能的QAT框架，兼具**稳定性、效率与灵活性**，为超低比特LLM部署提供了实用化路径。其提出的 **渐进训练 + OCS + 嵌套网格** 三重设计，有望成为未来低比特训练的标准范式。

</details>

---

### 11. [DMax: Aggressive Parallel Decoding for dLLMs](https://arxiv.org/abs/2604.08302)

**Authors**: Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, Xinchao Wang  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.08302v1  

#### Abstract
We present DMax, a new paradigm for efficient diffusion language models (dLLMs). It mitigates error accumulation in parallel decoding, enabling aggressive decoding parallelism while preserving generation quality. Unlike conventional masked dLLMs that decode through a binary mask-to-token transition,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DMax: Aggressive Parallel Decoding for dLLMs》总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现有的 **diffusion language models (dLLMs)** 虽然具备并行解码潜力，但在**激进并行解码**（aggressive parallel decoding）下性能急剧下降，其根本原因是 **error accumulation**（错误累积）。

传统基于 **masked dLLMs (MDLMs)** 的模型采用“二元、单向”的 `mask → token` 解码机制：一旦某个位置被解码为 token，该预测即被固定，并作为上下文用于后续步骤。早期错误无法修正，会污染后续生成，导致语义崩溃。

### **提出了什么新方法或新思路**

本文提出 **DMax**，一种全新的 dLLM 并行解码范式，核心思想是将传统的“一次性解码”转变为“渐进式自我精炼”（progressive self-refinement），从而缓解错误累积。

#### 主要创新点包括：

- **On-Policy Uniform Training (OPUT)**  
  一种新颖的训练策略，将预训练的 MDLM 扩展为具备自修正能力的 uniform dLLM。不同于传统 uniform 训练中从词表均匀采样噪声，OPUT 从**模型自身的预测分布**中采样噪声输入，从而**弥合了训练与推理之间的差距**（train-inference gap）。这使得模型能有效学习如何纠正自己的错误。

- **Soft Parallel Decoding (SPD)**  
  在推理阶段，不再将中间状态视为离散的 token，而是表示为 **hybrid embedding** —— 即预测 token embedding 与 mask embedding 的插值。这种软表示显式地传递了预测的不确定性，使模型在后续步骤中能更稳健地进行自我修正。

### **相比现有方法的优势**

- **更高的并行度**：实现了远超现有方法的 **Tokens Per Forward (TPF)**，显著提升推理吞吐量。
- **保持高准确性**：在大幅提升并行性的同时，几乎不牺牲生成质量。
- **更强的鲁棒性**：通过自我修正机制，有效抑制了错误传播。
- **无需额外监督**：训练数据通过 self-distillation 构建，仅依赖模型自身生成的数据。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **数学推理任务**：
  - **GSM8K**：小学数学应用题基准
  - **MATH500**：高中及以上难度数学题
  - **Minerva-Algebra**：代数问题
  - **ASDIV**：多样化算术题
- **代码生成任务**：
  - **HumanEval-Instruct**：Python 函数生成
  - **MBPP-Instruct**：面向初学者的编程任务

所有任务均采用 **zero-shot** 设置。

### **实验设置**

- **基础模型**：基于 **LLaDA-2.0-mini** 进行微调。
- **训练方式**：
  - 使用 OPUT 进行 full-parameter fine-tuning，共 2 个 epoch。
  - Batch size = 8，学习率 = 2e-5，cosine 调度。
  - 构造两个专用模型：**DMax-Math** 和 **DMax-Coder**。
- **推理设置**：
  - 采用 block-wise semi-autoregressive 解码，block size = 32。
  - 使用提出的 **Soft Parallel Decoding (SPD)** 策略。
  - 收敛阈值 $ T_{acc} = 0.9 $。
- **硬件平台**：训练使用 8×H200 GPU；推理在 2×H200 GPU 上进行。

### **评估指标**

- **TPF (Tokens Per Forward)**：每次前向传播生成的有效 token 数，衡量并行效率。
- **TPS (Tokens Per Second)**：每秒生成的 token 数，衡量实际吞吐量。
- **Accuracy**：任务最终准确率。
- **AUP Score (Area Under the Performance curve)**：综合衡量并行性与准确性的指标，越高越好。

### **基线方法对比**

| 基线方法 | 简要说明 |
|--------|--------|
| **LLaDA-2.0-mini** | 原始模型，使用 confidence-threshold 解码 |
| **Hierarchical Decoding** | 分层解码策略，提升并行性 |
| **dParallel SFT** | 使用 certainty-forcing 损失进行微调以加速收敛 |
| **Uniform Diffusion Training** | 传统 uniform 训练方式，用随机 token 替换 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 模型 | 数据集 | TPF | TPS | Accuracy | AUP Score |
|------|-------|-----|-----|----------|-----------|
| LLaDA-2.0-mini | GSM8K | 2.04 | 512 | 92.6% | 340 |
| **DMax-Math (Ours)** | GSM8K | **5.48** | **1258** | **92.1%** | **557** |
| LLaDA-2.0-mini | MBPP | 2.71 | 662 | 80.6% | 276 |
| **DMax-Coder (Ours)** | MBPP | **5.86** | **1264** | **79.2%** | **482** |
| LLaDA-2.0-mini | MATH500 | 2.58 | 626 | 75.8% | 257 |
| **DMax-Math (Ours)** | MATH500 | **5.94** | **1286** | **75.4%** | **507** |

> ✅ **结论**：DMax 将 TPF 提升约 **2.5–3 倍**，同时保持甚至略微提升准确率。

### **与基线方法的对比结果**

- 在 **GSM8K** 上，DMax 的 TPF 达到 **5.48**，远高于 dParallel SFT 的 2.79 和 Hierarchical Decoding 的 2.44。
- 在 **MBPP** 上，DMax 的 TPF 达到 **5.86**，而其他方法均未超过 3.7。
- **Uniform Diffusion Training** 表现极差，准确率大幅下降（如 GSM8K 降至 68.7%），验证了其存在严重 train-inference mismatch。
- 在 **2×H200 GPU** 上，DMax 实现平均 **1,338 TPS**（batch size=1），展示出强大的实际部署潜力。

### **消融实验结果**

#### 表格：不同训练与推理策略组合在 GSM8K 上的表现（$T_{dec}=0.5$）

| 训练策略 | 推理策略 | TPF | Accuracy |
|--------|--------|-----|---------|
| 原始训练 | 原始解码 | 2.04 | 92.6% |
| OPUT | 原始解码 | 4.47 | 78.0% |
| OPUT + SPD | 完整 DMax | **5.48** | **92.1%** |

> 🔍 **关键发现**：
> - **OPUT 是基础**：即使不使用 SPD，OPUT 已赋予模型一定的纠错能力。
> - **SPD 显著提升鲁棒性**：当 $T_{dec}=0$（最激进并行）时，仅 OPUT 的准确率仅为 68%，而加入 SPD 后恢复至 **90.4%**。
> - **OPUT 是 SPD 的前提**：若直接对原始模型应用 SPD，会导致生成崩溃（accuracy ≈ 0%），证明 OPUT 对 embedding 插值意义的建立至关重要。

---

## 4. 关键结论和发现

### **主要发现**

1. **错误累积是限制 dLLM 并行性的根本瓶颈**，而非单纯的解码策略问题。
2. **通过 OPUT 实现训练-推理一致性**，可让模型学会从自身错误中恢复。
3. **Soft Parallel Decoding 提供了一种有效的不确定性传播机制**，使模型能在 embedding 空间内进行渐进式自我修正。
4. DMax 成功实现了 **“激进并行”与“高质量生成”的统一**，在多个 benchmark 上实现 TPF 翻倍以上提升而不损失精度。

### **方法的局限性**

- 当前方法仍依赖于 block-wise 解码，在极端长序列上可能存在边界效应。
- OPUT 需要额外的 fine-tuning 成本，虽然不依赖人工标注，但仍需计算资源。
- 对于某些高度依赖顺序逻辑的任务（如复杂程序生成），完全并行可能仍有挑战。

### **未来工作方向**

- 将 DMax 思想扩展到 **fully autoregressive-free** 的全并行生成。
- 探索 **动态调整 confidence threshold** 以适应不同任务难度。
- 结合 **reinforcement learning** 进一步优化解码轨迹。
- 应用于 **multimodal** 和 **agent** 场景中的高效推理。

---

> 📌 **总结一句话**：  
> **DMax 通过 On-Policy Uniform Training 和 Soft Parallel Decoding，首次实现了在不牺牲准确性的前提下对 dLLMs 进行激进并行解码，为高效语言生成提供了新的强基线。**

</details>

---

### 12. [Quantization Impact on the Accuracy and Communication Efficiency Trade-off in Federated Learning for Aerospace Predictive Maintenance](https://arxiv.org/abs/2604.08474)

**Authors**: Abdelkarim Loukili  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.08474v1  

#### Abstract
Federated learning (FL) enables privacy-preserving predictive maintenance across distributed aerospace fleets, but gradient communication overhead constrains deployment on bandwidth-limited IoT nodes. This paper investigates the impact of symmetric uniform quantization ($b \in \{32,8,4,2\}$ bits) on...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Quantization Impact on the Accuracy and Communication Efficiency Trade-off in Federated Learning for Aerospace Predictive Maintenance

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文聚焦于**航空航天领域预测性维护**（predictive maintenance）中的联邦学习（Federated Learning, FL）部署挑战，具体解决了以下两个关键问题：

- **通信开销大**：在带宽受限的航空IoT节点（如通过LoRaWAN连接）上，全精度梯度传输成本高昂，限制了FL的实际应用。
- **硬件资源受限**：边缘设备多为FPGA等低功耗平台，对模型大小、计算复杂度和数值精度有严格限制。

此外，现有研究大多基于**IID数据假设**进行量化评估，而真实场景中各客户端的数据分布高度异构（Non-IID），导致评估结果存在偏差。

---

### 🚀 提出的新方法与创新思路

1. **提出轻量级1D-CNN模型 AeroConv1D**
   - 参数仅 **9,697**，专为FPGA推理优化。
   - 采用纯前馈结构（无RNN/LSTM），避免时间步间状态累积带来的量化噪声放大，提升量化鲁棒性。
   - 支持高效硬件流水线设计，降低延迟。

2. **系统性评估对称均匀量化（symmetric uniform quantization）的影响**
   - 在 **b ∈ {32, 8, 4, 2} bits** 下全面分析精度-效率权衡。
   - 强调将**量化应用于weight delta（Δw）而非原始权重**，保留客户端本地高精度梯度积累。

3. **揭示IID划分的评估偏见（Methodological Contribution）**
   - 首次通过实证表明：**IID客户端数据划分会人为压制方差，夸大低比特量化的“准确性优势”**。
   - 正确的Non-IID划分才能反映真实运行稳定性。

4. **引入FPGA资源投影与加密共设计潜力分析**
   - 基于 hls4ml scaling model 对 Xilinx ZCU102 进行**分析型FPGA资源估计**。
   - 发现INT4可在单个SoC上实现完整FL流程，并预留DSP资源用于NTT-based Homomorphic Encryption协处理器，支持端到端安全通信。

---

### 🔍 相比现有方法的优势

| 方面 | 本文优势 |
|------|--------|
| **任务针对性** | 聚焦航空航天RUL回归任务，非通用分类基准，更具现实意义 |
| **评估严谨性** | 多种子（N=10）、Non-IID划分、配对t检验，统计显著性强 |
| **量化策略实用性** | 选择可直接部署于固定点FPGA的对称均匀量化，不依赖浮点反量化 |
| **方法论警示** | 揭示IID评估的误导性，提醒从业者避免部署失配 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **NASA C-MAPSS** 公开数据集中的两个子集：
  - **FD001**：100台训练引擎，1种运行条件（较简单）
  - **FD002**：260台训练引擎，6种运行条件（更复杂、Non-IID程度更高）

> 输入特征：14个传感器通道，时间窗口长度50，标准化处理  
> 输出目标：Remaining Useful Life (RUL)，上限125 cycles

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|----------|
| **联邦学习框架** | 同步FedAvg，N=10客户端，每轮E=2 local epochs |
| **本地批量大小** | 32 |
| **优化器** | Adam，学习率 1e-3 |
| **训练轮数** | 20 rounds |
| **量化方式** | 对每层weight delta Δw(l)独立执行 per-tensor symmetric uniform quantization |
| **量化位宽** | b ∈ {32 (FP32), 8 (INT8), 4 (INT4), 2 (INT2)} |
| **随机性控制** | N=10 seeds 控制划分、初始化、mini-batch顺序 |
| **统计检验** | Two-tailed paired t-test (α=0.05, df=9) |

---

### 🎯 评估指标

| 指标 | 定义与用途 |
|------|-----------|
| **MAE** | Mean Absolute Error in RUL cycles，衡量平均误差 |
| **NASA Score (S)** | 不对称评分函数：<br> - 欠估惩罚轻 `exp(-d/13)`<br> - 过估惩罚重 `exp(d/10)-1`<br> 反映安全关键系统的代价敏感特性 |
| **Coefficient of Variation (CV)** | NASA Score跨seed的标准差归一化均值，衡量模型稳定性 |
| **Gradient-distortion Privacy Proxy (Lpriv)** | 平均每参数量化失真：<br>`Lpriv = ||Δw − Qb(Δw)||² / Θ`<br>作为梯度泄露攻击难度的启发式代理（非正式DP保证） |
| **Communication Cost** | 每轮传输的梯度体积（KiB） |
| **FPGA Resource Projection** | 基于 hls4ml 模型估算 LUT、DSP、BRAM 和延迟 |

---

### 🔁 基线方法对比

- 主要对比不同量化级别之间的性能差异：
  - **FP32**：全精度，通信成本最高，作为黄金标准
  - **INT8 / INT4 / INT2**：分别代表主流、轻量、极端压缩方案
- 特别设置了 **IID vs. Non-IID 划分对比组**，以揭示评估偏差

> 未与其他压缩技术（如QSGD、SignSGD、混合量化）直接比较，因本文旨在建立干净baseline并强调方法论问题。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（来自 Table 3）

| 子集 | 配置 | MAE (cycles) | NASA Score S (×10³) | CV(S) | p(MAE vs FP32) | p(Score vs FP32) | Comm. Cost |
|------|------|---------------|----------------------|-------|------------------|--------------------|-------------|
| FD001 | FP32 | 17.52±0.47 | 449±123 | 27.3% | — | — | 37.88 KiB |
| FD001 | INT8 | 17.51±0.48 | 447±127 | 28.6% | 0.520 | 0.746 | 9.47 KiB |
| FD001 | INT4 | 17.48±0.51 | 452±115 | 25.3% | 0.341 | 0.802 | **4.73 KiB** |
| FD001 | INT2 | 19.03±1.62 | 802±573 | 72.0% | 0.018 | 0.064 | 2.37 KiB |
| FD002 | FP32 | 26.99±1.69 | 923±206 | 22.3% | — | — | 37.88 KiB |
| FD002 | INT8 | 27.24±1.40 | 951±167 | 16.9% | 0.265 | 0.364 | 9.47 KiB |
| FD002 | INT4 | 27.20±1.84 | 938±233 | 24.8% | 0.264 | 0.534 | **4.73 KiB** |
| FD002 | INT2 | 21.53±2.31 | 749±347 | 45.8% | **0.001** | 0.207 | 2.37 KiB |

> ✅ 所有p > 0.05 表示与FP32无统计显著差异

---

### 🔍 与基线方法的对比结果

#### ✅ INT4 是最优操作点（Operating Point）
- **精度无损**：在FD001和FD002上，INT4的MAE和NASA Score均与FP32**无统计显著差异**（p > 0.05）
- **通信节省8倍**：从37.88 KiB降至**4.73 KiB/轮**
- **适用于LoRaWAN**：在5kbps下每轮约7.5秒传输时间，符合分钟级调度需求

#### ✅ INT8 性能稳定但收益有限
- 实现4×通信压缩，精度保持良好，适合对稳定性要求极高但带宽稍宽松的场景

#### ❌ INT2 虽然通信最小，但不可靠
- 在FD001上MAE显著恶化（+8.6%，p=0.018）
- 在FD002上出现“虚假改进”：MAE下降20.2%（p=0.001），但这是由于**极端过正则化（over-regularization）导致保守预测**
- NASA Score波动剧烈：
  - CV高达 **45.8%（FD002）和72.0%（FD001）**，远高于FP32的22.3%
  - 单次运行可能产生极端过高或过低评分，**缺乏可复现性**

> 💡 图3显示INT2的NASA Score在训练过程中剧烈震荡，验证其不稳定

---

### 🔍 消融实验与关键发现

#### ✅ 消融：IID vs. Non-IID 划分的影响（Table 4 & Figure 1）

| 分区方式 | 模型 | NASA Score Std |
|--------|------|----------------|
| IID | FP32 | 41k |
| IID | INT4 | 31k |
| Non-IID | FP32 | 123k |
| Non-IID | INT4 | 115k |

- **发现**：IID划分严重低估模型方差，使INT4看似优于FP32（Score更低）
- **真相**：Non-IID下二者统计等价，证明**IID评估会误导决策**

> 👉 这是本文最重要的方法论贡献之一：**必须在Non-IID下评估量化效果**

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **INT4 是航空航天FL的理想选择**
   - 在NASA C-MAPSS上实现**精度无损**（statistically indistinguishable from FP32）
   - 达成 **8×通信压缩**（37.88 → 4.73 KiB/round）
   - 满足带宽受限航空链路（如LoRaWAN）的部署需求

2. **INT2 不适合安全关键RUL任务**
   - 尽管在FD002上表现出更低MAE，但这是一种**由3-level量化网格引发的过正则化伪象**
   - NASA Score的极高变异系数（CV=45.8%）表明其**预测不可靠、不可复现**
   - 在异构运行条件下无法稳定部署

3. **IID客户端划分会导致评估偏差**
   - 人为平滑数据分布，掩盖真实方差
   - 可能错误地得出“量化提升精度”的结论
   - **应强制使用按引擎划分的Non-IID协议进行评估**

4. **INT4具备FPGA部署可行性**
   - 在Xilinx ZCU102上：
     - DSP利用率仅 **85.5%**
     - 剩余 **366个DSP** 可用于集成NTT-based HE协处理器
     - 推测可实现**单SoC完成训练-量化-推理-加密全流程**

---

### ⚠️ 局限性

| 限制项 | 说明 |
|-------|------|
| **FPGA投影未硅验证** | 所有资源估计基于 hls4ml scaling model，尚未在物理ZCU102芯片上综合验证 |
| **隐私代理非正式DP** | Gradient-distortion (Lpriv) 仅为启发式指标，**不能替代 (ε,δ)-DP 形式化隐私保障** |
| **仅评估对称均匀量化** | 未涵盖QSGD、SignSGD、动态混合量化等先进压缩方法 |
| **未考虑客户端异构硬件** | 假设所有客户端具有相同算力，忽略实际设备多样性 |

---

### 🔮 未来工作方向（原文建议）

1. **建立形式化DP分析框架**  
   针对RUL回归任务构建 (ε,δ)-DP 模型，抵御gradient inversion攻击。

2. **FPGA硅验证**  
   在真实ZCU102平台上实现AeroConv1D + INT4量化 pipeline，验证资源与延迟预测。

3. **扩展Non-IID严重性量化**  
   使用Earth Mover’s Distance（EMD）系统扫描不同划分策略下的异构程度影响。

4. **拓展至其他C-MAPSS子集**  
   将实验推广至FD003、FD004，覆盖更多故障模式。

5. **探索异构客户端FL设置**  
   研究不同客户端拥有不同硬件能力（如部分用INT4，部分用FP32）时的协同训练机制。

---

📌 **代码与数据公开**  
全部仿真代码、日志、FPGA估算脚本已开源：  
👉 [https://github.com/therealdeadbeef/aerospace-fl-quantization](https://github.com/therealdeadbeef/aerospace-fl-quantization)  
NASA C-MAPSS 数据集来源：NASA Prognostics Data Repository [16]

</details>

---

### 13. [DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification](https://arxiv.org/abs/2604.07622)

**Authors**: Ziyi Wang, Siva Rajesh Kasa, Ankith M S, Santhosh Kumar Kasa, Jiaru Zou, Sumit Negi, Ruqi Zhang, Nan Jiang, Qifan Song  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.07622v1  

#### Abstract
Speculative decoding is an effective technique for accelerating large language model inference by drafting multiple tokens in parallel. In practice, its speedup is often bottlenecked by a rigid verification step that strictly enforces the accepted token distribution to exactly match the target model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统的 **speculative decoding (SD)** 虽然能通过小模型（draft model）并行生成多个候选 token 来加速大语言模型（LLM）推理，但其性能受限于**严格的验证机制**。该机制要求接受的 token 分布必须严格匹配目标模型（target model）的分布，导致许多语义合理但不完全一致的 draft token 被拒绝，从而降低 **acceptance rate** 和整体加速效果。

此外，现有的 **lossy speculative decoding** 方法（如静态加权 ensemble）虽然放宽了验证标准，但采用固定的、全局的权重策略，无法适应不同上下文和任务对质量敏感度的差异，限制了效率-质量权衡的灵活性。

### **提出了什么新方法或新思路**
本文提出 **DIVERSED**（**Dynamic VErification RElaxed SpEculative Decoding**），一种基于动态集成验证的松弛 speculative decoding 框架，核心思想是：

- 引入一个可学习的 **ensemble-based verifier**，在每一步解码时，根据当前上下文 $x_{\leq t-1}$ 动态调整 draft model 与 target model 输出分布的混合权重 $w_t$。
- 权重 $w_t = f_\theta(h_t^{\text{draft}}, h_t^{\text{target}})$ 由一个小的神经网络（ensemble head）预测，利用 draft 和 target 模型的隐藏状态作为输入，实现**上下文感知**和**任务自适应**的验证策略。
- 在训练阶段使用 **reinforcement learning**（REINFORCE++）优化目标函数，平衡任务准确率与 acceptance rate。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **灵活性** | 相比 static ensemble 使用固定权重，DIVERSED 实现了 token-level 和 context-dependent 的动态调节，更精细地控制质量-效率权衡。 |
| **性能提升** | 显著提高 acceptance rate，在保持甚至提升 generation quality 的同时，获得更低的端到端延迟（wall-clock time）。 |
| **理论支持** | 推导出无需 i.i.d. 假设的期望接受长度精确表达式，提供了比前人更精确的理论分析。 |
| **通用性** | 不修改 draft 或 target 模型结构，仅在验证逻辑上增强，可与其他 SD 变体（如 DISCO, Medusa）兼容。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖多种任务类型，确保泛化能力：
- **GSM8K**：数学推理（mathematical reasoning）
- **CNNDM**：新闻摘要（news summarization）
- **XSum**：极端摘要（extreme summarization）
- **MBPP**：Python 编程任务（program synthesis）

### **实验设置和评估指标**
#### **模型组合（Target/Draft Pairs）**
共测试三组主流模型对：
1. `Llama-3.1-8B-Instruct` / `Llama-3.2-1B-Instruct`
2. `Qwen3-8B` / `Qwen3-0.6B`
3. `Gemma-3-12B-It` / `Gemma-3-4B-It`

#### **关键参数**
- Draft length $N = 5$（主实验），部分补充 $N=3,7$
- Temperature: 0 或 1.0
- Evaluation metrics:
  - **Acceptance Rate (%)**：被成功接受的 draft token 比例
  - **Wall-clock Time / Latency**：实际运行时间
  - **Task Accuracy**：
    - GSM8K: final answer accuracy
    - CNNDM/XSum: ROUGE-2 score
    - MBPP: pass@1

#### **训练方式**
- 使用序列级奖励 $R(x_{1:T})$ 进行强化学习训练（如数学题答对得1，否则0）
- 优化目标包含正则项 $(1 - \text{TV}(q, v))$ 鼓励高 acceptance，防止退化为纯 draft 或纯 target 模型

---

## **3. 主要实验结果和性能指标**

### **关键性能数据与对比结果**

#### ✅ **总体性能表现（Table 1 & Appendix Table 6）**

| Method | Acceptance Rate ↑ | Task Quality ↔/↑ | Wall-clock Time ↓ |
|--------|-------------------|------------------|--------------------|
| **Autoregressive** | NA | Baseline | High |
| **SD (Standard)** | ~60–70% | Match target | Moderate |
| **SD (Lossy)** | ~65–75% | Slight drop | Better |
| **SpecCascade** | ~50–60% | Match target | Moderate |
| **Static Ensemble** | ~70–85% | Slight drop | Good |
| **DIVERSED (Ours)** | **~85–95%** | **Match or exceed** | **Best (lowest)** |

> 示例：在 `Llama-3.1-8B/Llama-3.2-1B` 对上，DIVERSED 在 GSM8K 上达到 **84.82%** acceptance rate，显著高于 SD 的 61.53%，且 accuracy 保持 80%。

#### ✅ **速度提升（Figure 5）**
- DIVERSED 平均每次验证轮次接受的 token 数量最多，说明其有效扩展了可接受区域。
- 在所有 model pairs 上均优于 baselines，表明其动态加权机制确实提升了 token 利用率。

#### ✅ **效率-质量帕累托前沿（Figure 1, 4）**
- DIVERSED 成功突破了 **static ensemble** 所能达到的 Pareto front，实现了“更高 acceptance + 更好或相当 quality”的组合。
- 表明动态上下文感知的验证策略优于静态规则。

#### ✅ **跨任务迁移实验（Zero-shot Transfer）**
- 在 GSM8K 上训练的 DIVERSED 应用于 CNNDM（反之亦然），acceptance 提升但 task performance 下降。
- **结论**：验证策略应是 task-specific 的，不能简单通用化。

#### ✅ **与 fine-tuned draft model 对比（Table 3）**
- 单独微调 draft model 可提升 accuracy，但 acceptance 改善不稳定（有时上升，有时下降）。
- DIVERSED 在所有设置下都取得最高的 acceptance rate，说明其优势来自验证机制本身而非 draft model 能力。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **最优 acceptance 策略是 context- and task-dependent 的**，静态规则（如 fixed ensemble weight）无法充分挖掘潜力。
2. ✅ **更高的 acceptance rate 可靠地转化为更低的 wall-clock latency**，验证了 acceptance 是影响推理速度的关键瓶颈。
3. ✅ **DIVERSED 能自适应地在关键位置偏向 target model，在非关键位置更多采纳 draft token**，实现智能松弛验证。
4. ✅ **理论推导无需 i.i.d. 假设**，更真实刻画 speculative decoding 中的位置依赖效应。

### **方法的局限性**
- **训练成本较高**：需要额外训练 ensemble head，并使用 RL 优化，增加了部署复杂性。
- **依赖高质量 reward signal**：对于缺乏明确 success/failure 判断的任务（如开放生成），reward 设计可能成为挑战。
- **未探索 block-level 决策**：目前仍基于 token-level 接受/拒绝决策，未来可拓展至 chunk 或语义块级别。

### **未来工作方向**
- 将 relaxed verification 扩展到 **block-level decisions**，进一步提升吞吐量。
- 探索 **cross-task transferability** of the learned dynamic verifier，减少 per-task 训练开销。
- 结合其他 SD 技术（如 DISCO, Medusa），构建层级式高效推理系统。

---

> 🔗 **代码开源地址**：[https://github.com/comeusr/diversed](https://github.com/comeusr/diversed)

</details>

---

### 14. [Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](https://arxiv.org/abs/2604.07659)

**Authors**: Mingchen Li, Jiatan Huang, Zonghai Yao, Hong yu  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.07659v1  

#### Abstract
Large language models (LLMs) hold significant promise for healthcare, yet their reliability in high-stakes clinical settings is often compromised by hallucinations and a lack of granular medical context. While Retrieval Augmented Generation (RAG) can mitigate these issues, standard supervised pipeli...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction**

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

大型语言模型（LLMs）在医疗预测任务中展现出潜力，但在高风险临床环境中面临两大挑战：

- **幻觉（Hallucinations）** 和缺乏细粒度医学上下文支持；
- 传统 **Retrieval-Augmented Generation (RAG)** 方法依赖于大规模外部知识库（如知识图谱、文档集合）的检索，导致：
  - 高延迟（high latency），不适用于时间敏感的临床决策；
  - 推理时上下文过长，计算开销大；
  - 构建高质量检索器需要大量标注数据或复杂图搜索。

这些问题限制了 RAG 在实时医疗场景中的实用性。

---

### 🚀 **提出了什么新方法或新思路**

作者提出 **Keys-to-Knowledge (K2K)** —— 一种**基于内部记忆检索**的新型框架，其核心思想是：

> 将外部医学知识直接编码进 LLM 的参数空间，并通过访问模型内部的 **FFN 层 Key-Value 结构** 来实现快速、低延迟的知识检索，**无需外部检索系统**。

#### K2K 的三大核心模块：

1. **Internal Memory Construction（内部记忆构建）**
   - 利用 FFN 层中隐含存储的事实知识（keys as $W_1$）作为“内存”；
   - 对于预训练语料未覆盖的专业医学知识，采用 **LoRA** 注入到 FFN 参数中，形成增强型内部记忆（document-level 和 graph-level knowledge）。

2. **Activation-Guided Probe Construction（激活引导探针构造）**
   - 不再使用简单的均值池化生成查询向量；
   - 引入基于 **对角近似 Mahalanobis 距离** 的激活偏置机制，动态加权 token 表示，提升查询判别力，强调稀有且关键特征。

3. **Cross-Attentive Reranking（交叉注意力重排序）**
   - 对从内部 memory 中检索出的 top-k 知识进行动态融合；
   - 使用 cross-attention 机制根据当前输入上下文对 retrieved knowledge 进行自适应重加权，增强任务相关性。

---

### ⚡ **相比现有方法的优势**

| 维度 | K2K | 传统 RAG / 外部检索方法 |
|------|-----|------------------------|
| **检索方式** | 内部参数空间访问 | 外部数据库/图谱检索 |
| **延迟** | 极低（仅需前向传播） | 高（需索引、匹配、排序） |
| **可扩展性** | 支持长序列、无上下文膨胀 | 上下文长度受限，成本随规模上升 |
| **知识来源整合** | 文档 + 图结构知识统一处理 | 通常单一来源或复杂集成 |
| **训练效率** | LoRA 微调，轻量高效 | 全模型微调或端到端联合训练 |

✅ **核心优势总结**：
- 实现了 **zero-latency retrieval**，适合实时医疗应用；
- 显著降低推理时间和资源消耗；
- 提升了检索的相关性和上下文感知能力。

---

## 2. 核心实验方法和设置

### 📚 **使用的数据集**

在两个公开的电子健康记录（EHR）基准上评估：

- **MIMIC-III**（重症监护数据库）
- **MIMIC-IV**（更新版，更大规模）

#### 数据划分与任务定义：

| 任务 | 定义 |
|------|------|
| **Mortality Prediction** | 预测患者是否会在下一次就诊期间死亡 |
| **Readmission Prediction** | 预测患者是否在出院后 15 天内再入院 |

> 输入为纵向诊断代码序列（ICD codes），每个 code 关联文本描述（如 “Acute myocardial infarction”）。

| 数据集 | Train | Dev | Test |
|-------|-------|-----|------|
| MIMIC-III Mortality | 7,777 | 978 | 953 |
| MIMIC-III Readmission | 7,777 | 978 | 953 |
| MIMIC-IV Mortality | 100,125 | 12,547 | 12,667 |
| MIMIC-IV Readmission | 100,125 | 12,547 | 12,667 |

> 按 patient ID 分组划分，防止数据泄露。

---

### 🎯 **评估指标**

遵循临床预测标准，报告以下四个指标的平均值（Avg）：

- **F1-score**
- **Jaccard Similarity**
- **AUPRC**（Area Under Precision-Recall Curve）
- **AUROC**（Area Under ROC Curve）

---

### 🔁 **基线方法对比**

共比较三类主流方法：

#### （1）**Sequential Models（传统序列模型）**
- GRU, RETAIN, Deepr, AdaCare, StageNet, TCN

#### （2）**Retrieval-Based Models**
- **KARE**：当前 SOTA，结合知识图谱最短路径检索；
- **Standard RAG**：使用 Contriver 检索相似病例；
  
#### （3）**Generative Knowledge Models**
- **Prompt-Based Retrieval**：利用 in-context learning 让 LLM 自主生成医学知识。

> 主要 LLM backbone：**BioMistral-7B**, **Meditron3-Qwen2.5-7B**

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据（来自 Table 2）**

#### 在 **BioMistral-7B** 上的表现（MIMIC-III Mortality）：

| Model | F1 | Jaccard | AUPRC | AUROC | **Avg** |
|-------|----|---------|-------|-------|--------|
| Fine-tuned w/o retriever | 16.00 | 8.69 | 11.61 | 59.40 | 23.92 |
| KARE | 18.01 | 9.90 | 9.72 | 56.65 | 23.57 |
| Prompt-based | 15.05 | 8.13 | 10.78 | 58.72 | 23.17 |
| **K2K (Ours)** | **18.55** | **10.22** | **15.22** | **61.05** | **26.26** |

> ✅ K2K 在所有指标上全面领先，尤其 AUPRC 提升显著（+3.6 pts vs KARE），说明其对正样本识别更优。

#### 在 **MIMIC-IV Readmission** 上表现最强：

| Model | F1 | Jaccard | AUPRC | AUROC | **Avg** |
|-------|----|---------|-------|-------|--------|
| KARE | 64.62 | 47.74 | 68.50 | 66.76 | 61.91 |
| Standard RAG | 65.09 | 48.25 | 67.68 | 65.78 | 61.70 |
| **K2K (Ours)** | **67.11** | **50.50** | **67.60** | **66.28** | **62.87** |

> ✅ 达到 SOTA 性能，尤其在 F1 和 Jaccard 上优势明显。

---

### 🔍 **消融实验结果（Ablation Studies）**

#### （1）不同知识源的影响（Table 3）

| 变体 | MIMIC-III Mortality (Avg) | MIMIC-IV Readmission (Avg) |
|------|----------------------------|------------------------------|
| K2K (Full) | **26.26** | **62.87** |
| w/o document | 23.62 ↓ | 61.42 ↓ |
| w/o graph | 24.63 ↓ | 60.44 ↓ |

> ❗ 移除任一知识源都会导致性能下降，证明 **document + graph 双源协同有效**。

> ⚠️ 特别地，移除 graph 后 AUPRC/AUROC 下降明显，表明图结构知识有助于区分难例。

---

#### （2）查询表示策略对比（Table 5）

| 查询方式 | MIMIC-III Mortality (F1) | MIMIC-IV Readmission (F1) |
|----------|---------------------------|----------------------------|
| Mean Only（均值池化） | 12.06 | 56.26 |
| Euclidean 加权 | 16.97 | 63.56 |
| **Mahalanobis（本文）** | **18.55** | **67.11** |

> ✅ Mahalanobis 显著优于其他方法，因其考虑了维度方差，提升了低方差方向的敏感性。

---

#### （3）不同 Transformer 层的知识效果（Figure 3）

- 并非越深的层越好；
- 浅层（如 Layer 5, 8, 10）也包含重要实体/结构信息；
- 最佳性能分布在多个层级，支持多层融合策略。

---

### ⏱️ **效率对比（Appendix A.7 & Table 6）**

| 方法 | Avg Score (MIMIC-III) | **检索耗时** |
|------|------------------------|-------------|
| KARE | 21.11 | 33分52秒 |
| Prompt-based | 22.52 | **3小时26分钟** |
| **K2K (Ours)** | **22.89** | **仅 5 秒** |

> ✅ K2K 检索速度比 KARE 快 **~400 倍**，比 Prompt-based 快 **~2400 倍**！

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **LLM 的 FFN 参数空间可被显式用作可检索的“内部知识库”**，无需外部检索系统；
2. **LoRA 是有效的 domain adaptation 工具**，可用于将专业医学知识注入 FFN keys；
3. **激活信号指导下的 probe query 构造显著提升检索精度**，优于简单池化或欧氏距离；
4. **cross-attentive reranking 实现了上下文感知的知识融合**，增强了推理一致性；
5. **K2K 在多个医疗预测任务上达到 SOTA 性能，同时具备极高的推理效率**。

---

### ⚠️ **局限性（Limitations）**

1. **Layer Selection 固定**：
   - 当前固定选择某一层提取 key，尚未实现动态 layer selection 或 multi-layer weighted fusion；
   - 不同任务可能最优 layer 不同。

2. **领域泛化待验证**：
   - 目前仅在生物医学领域测试；
   - 是否适用于法律、金融等其他 high-stakes 领域尚不清楚。

3. **数据不平衡问题未充分解决**：
   - 医疗数据中阳性样本稀少，虽有改进但仍存在偏差风险。

---

### 🔮 **未来工作方向**

1. 设计 **dynamic layer-aware retrieval mechanism**，自动选择最相关的 FFN 层；
2. 扩展至更多模态（如影像、实验室数值）；
3. 探索 **multi-hop internal reasoning**，模拟更复杂的临床推断流程；
4. 应用于真实世界临床系统，进行前瞻性验证。

---

## ✅ 总结一句话

> **K2K 开创性地将 LLM 的内部参数视为可检索的记忆单元，通过 LoRA 注入医学知识、激活引导查询和 cross-attention 重排序，在保持超低延迟的同时实现了医疗预测任务上的 SOTA 效果，为高效可靠的 AI 医疗决策提供了新范式。**

</details>

---

### 15. [Validated Synthetic Patient Generation for Small Longitudinal Cohorts: Coagulation Dynamics Across Pregnancy](https://arxiv.org/abs/2604.07557)

**Authors**: Jeffrey D. Varner, Maria Cristina Bravo, Carole McBride, Thomas Orfeo, Ira Bernstein  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.07557v1  

#### Abstract
Small longitudinal clinical cohorts, common in maternal health, rare diseases, and early-phase trials, limit computational modeling: too few patients to train reliable models, yet too costly and slow to expand through additional enrollment. We present multiplicity-weighted Stochastic Attention (SA),...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Validated Synthetic Patient Generation for Small Longitudinal Cohorts: Coagulation Dynamics Across Pregnancy*

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**小规模纵向临床队列**（small longitudinal cohorts）在计算建模中的瓶颈问题：
- 在母体健康、罕见病和早期临床试验中，患者样本量（n）通常远小于特征维度（p），即 `n << p`。
- 这导致传统统计方法失效（如协方差矩阵秩亏）、模型过拟合严重，且难以对罕见亚群（如PCOS、preeclampsia）进行独立分析。

### 提出的新方法
提出了一种名为 **Multiplicity-weighted Stochastic Attention (SA)** 的生成框架，其核心思想基于 **modern Hopfield network theory** 和 **Langevin dynamics**：
- 将真实患者的多时间点表型作为“记忆模式”嵌入到一个连续的能量景观（energy landscape）中。
- 通过 Langevin 动态在这些模式之间插值生成新的合成患者，保留原始队列的几何结构。
- 引入 **per-pattern multiplicity weights**，允许在推理时对特定临床亚群（如PCOS）进行定向放大，而无需重新训练模型。

### 相比现有方法的优势
| 方法 | 局限性 | SA 的优势 |
|------|--------|----------|
| **Multivariate Normal (MVN)** | 在 `n < p` 下协方差矩阵奇异，需正则化（如Ledoit-Wolf），会扭曲跨时间点依赖关系，并引入虚假方差。 | 不估计完整协方差矩阵，避免秩亏；通过 PCA 降维后在低维流形上操作，保留真实数据几何结构。 |
| **GANs / VAEs (如 CTGAN, TVAE)** | 在极小样本下易发生模式崩溃（mode collapse），且无法有效捕捉跨访视（cross-visit）协方差。 | 在 `n=23` 极小样本下仍能稳定生成多样化样本，无模式崩溃现象。 |
| **条件生成能力** | MVN 无法对稀有亚群进行条件生成（子群体太小无法拟合分布）；TVAE 只能在单次访问上操作，无法支持纵向条件生成。 | 支持 inference-time 的条件生成，可从仅3例PCOS患者中生成100例合成患者并保留其特征签名。 |

---

## 2. 核心实验方法和设置

### 数据集
- **来源**：一项前瞻性妊娠研究，来自 University of Vermont Medical Center。
- **队列规模**：`K = 23` 名具有完整纵向数据的孕妇。
- **时间点**：3次访视（V1: 孕前基线，V2: 第一孕期，V3: 第三孕期）。
- **特征维度**：每访视 `72` 项测量指标，共 `216` 维向量（合并为单个纵向轨迹）。
  - 包括：凝血因子（Factor II, VIII, fibrinogen等）、抗凝物（antithrombin, protein C）、纤溶标志物、TGA参数、ROTEM粘弹性参数、激素水平。
- **罕见亚群**：
  - PCOS（n=3）
  - Developed PE（n=5）
  - Uncomplicated（n=18）

### 实验设置与评估框架
采用四层验证体系：

| 验证层级 | 内容 | 指标 |
|---------|------|------|
| **Level 1: Marginal Plausibility** | 单变量分布保真度 | Mean Relative Error (MRE)，特征均值、标准差、生理相关性散点图 |
| **Level 2: Cross-Visit Covariance Structure** | 跨时间点协方差结构保留 | 全局 `216×216` 相关矩阵比较、PCA 投影、特征值谱分析 |
| **Level 3: Conditional Generation** | 罕见亚群条件生成能力 | 条件组均值比较、bootstrap Mann-Whitney U 检验（p>0.05比例） |
| **Level 4: Mechanistic Consistency** | 生物学合理性验证 | 使用独立的 **BZ2012 ODE model** 推断 thrombin generation，比较预测/实测比值分布（cloud overlap, KS test） |

### 基线方法对比
- **MVN**：使用 Ledoit-Wolf 正则化的多元正态分布采样（same concatenated input）。
- **Deep Generative Models**：
  - **CTGAN**：在不同epoch（300–3000）下测试，均出现模式崩溃。
  - **TVAE**：虽边际保真度尚可，但只能在单次访问上训练（n=90），无法建模跨访视依赖，也不支持条件生成。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Level 1: Marginal Plausibility
- **总体 MRE**：所有 `216` 特征-访视组合的中位 MRE 仅为 **1.2%**（95% CI: 1.0–1.6%），其中 `89%` 的条目 MRE < 5%。
- **代表性特征 MRE**：
  - Factor II: < 2%
  - Antithrombin: < 3%
  - Fibrinogen: < 1%
- **新颖性检测**：
  - 平均 novelty score: **0.50**（角距离）
  - 合成-真实最近邻距离 / 真实-真实距离 ≈ **0.89**，表明合成样本与真实样本间距相当。

#### ✅ Level 2: Cross-Visit Covariance Preservation
- SA 完整保留了相关矩阵的块状结构（block structure），尤其是跨访视依赖（off-diagonal blocks）。
- **PCA 投影对比**：
  - SA 合成患者紧密围绕真实数据云。
  - MVN 显著扩散，尤其在 V2/V3，因其正则化在194个零空间维度引入虚假方差（见 Fig. S2）。

#### ✅ Level 3: Conditional Generation of Rare Subgroups
- 成功从 `n=3` PCOS 患者生成 `N=100` 合成患者，保留其高 Factor VIII 和 vWF 的特征。
- **Bootstrap Mann-Whitney 等价性检验**：
  - 在 `24` 个 feature-condition 对中，`20` 个（83%）在 ≥90% 的重抽样中无法区分（p > 0.05）。
  - 中位 “non-significance fraction” 达 **98.6%**。
- **PCA 可视化**（Fig. S3–S4）显示合成亚群聚集在其真实对应组周围。

#### ✅ Level 4: Mechanistic Consistency
使用 **BZ2012 ODE model**（58 ODEs, 64 parameters）进行生物学合理性验证：
- **Cloud Overlap**（合成 vs. 真实 ratio 分布重叠）：
  - TF-only 条件下：**86–93%**
  - TF+TM 条件下：**89–93%**
- **KS Test**：所有5个 TGA 特征的 p > 0.30（D = 0.08–0.12），表明 ODE 模型无法区分合成与真实患者。
- **校准泛化性**：仅用 V1 数据校准的参数，在 V2/V3 上预测仍接近1.0（Fig. S6），说明参数具稳定性。

#### ✅ Downstream Utility: Mechanistic Model Calibration
- 使用 `N=100` 合成 V1 患者训练的 BZ2012 模型，在预测 **held-out real V2/V3 患者**时表现：
  - **中位相对误差比（Synth/Real）为 0.94×**，略优于使用 `K=23` 真实患者训练的模型。
  - 表明合成数据足以支持下游机制建模任务。

---

## 4. 关键结论和发现

### 主要发现
1. **SA 能在极小样本（n=23）下生成统计与生物学双重可信的合成纵向患者**，通过四层验证均无法与真实患者区分。
2. **SA 保留了数据的低秩几何结构**，避免了 MVN 因正则化引入的虚假方差，也克服了 GAN/VAE 在小样本下的模式崩溃。
3. **Multiplicity weighting 提供了强大的条件生成能力**，使罕见亚群（如PCOS）的研究成为可能。
4. **合成数据可用于下游机制建模**：完全基于合成数据校准的 ODE 模型，能准确预测真实患者结局。

### 方法的局限性
- **PCA 假设线性结构**：可能无法捕捉复杂的非线性临床关系。
- **尾部分布压缩**：由于 PCA 和 magnitude sampling 有限，极端值（tails）保真度下降。
- **未验证临床结局关联**：当前验证止步于生物合理性，尚未与真实临床事件（如出血、血栓）挂钩。
- **ODE 模型本身存在偏差**：部分参数需大幅调整以匹配数据，反映体外模型与体内环境差异。

### 未来工作方向
- 扩展至其他领域：只要有可用的机制模型（如 PK/PD、肿瘤生长、代谢网络），即可应用类似验证范式。
- 结合非线性降维方法（如 autoencoder）替代 PCA，以更好捕捉复杂流形结构。
- 开发面向临床决策支持的端到端验证，例如预测治疗响应或不良事件。
- 探索联邦学习场景下，使用 SA 在多个小中心间安全共享合成数据。

---

> **总结一句话**：  
> 该论文提出了 **multiplicity-weighted Stochastic Attention (SA)**，首次实现了在 `n << p` 的极小纵向队列中生成**经机制模型验证的合成患者**，不仅统计保真，更具备**下游建模实用性**，为罕见病与母体健康研究提供了新的数据增强路径。

</details>

---

### 16. [SepSeq: A Training-Free Framework for Long Numerical Sequence Processing in LLMs](https://arxiv.org/abs/2604.07737)

**Authors**: Jie Sun, Yu Liu, Lu Han, Qiwen Deng, Xiang Shu, Yang Xiao, Xingyu Lu, Jun Zhou, Pengfei Liu, Lintao Ma, Jiancan Wu, Xiang Wang  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.07737v1  

#### Abstract
While transformer-based Large Language Models (LLMs) theoretically support massive context windows, they suffer from severe performance degradation when processing long numerical sequences. We attribute this failure to the attention dispersion in the Softmax mechanism, which prevents the model from ...

---

### 17. [GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-efficient Large Language Model Fine-tuning](https://arxiv.org/abs/2604.07808)

**Authors**: Kaiyuan Tian, Yu Tang, Gongqingjian Jiang, Baihui Liu, Yifu Gao, Xialin Su, Linbo Qiao, Dongsheng Li  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.07808v1  

#### Abstract
Full-parameter fine-tuning of large language models is constrained by substantial GPU memory requirements. Low-rank adaptation methods mitigate this challenge by updating only a subset of parameters. However, these approaches often limit model expressiveness and yield lower performance than full-par...

---

### 18. [Auto-Configured Networks for Multi-Scale Multi-Output Time-Series Forecasting](https://arxiv.org/abs/2604.07610)

**Authors**: Yumeng Zha, Shengxiang Yang, Xianpeng Wang  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.07610v1  

#### Abstract
Industrial forecasting often involves multi-source asynchronous signals and multi-output targets, while deployment requires explicit trade-offs between prediction error and model complexity. Current practices typically fix alignment strategies or network designs, making it difficult to systematicall...

---

### 19. [SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](https://arxiv.org/abs/2604.07663)

**Authors**: Wooin Lee, Hyun-Tae Kim  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.07663v1  

#### Abstract
The AdamW optimizer, while standard for LLM pretraining, is a critical memory bottleneck, consuming optimizer states equivalent to twice the model's size. Although light-state optimizers like SinkGD attempt to address this issue, we identify the embedding layer dilemma: these methods fail to handle ...

---

### 20. [Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](https://arxiv.org/abs/2604.07692)

**Authors**: Micky C. Nnamdi, Benoit L. Marteau, Yishan Zhong, J. Ben Tamo, May D. Wang  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.07692v1  

#### Abstract
Large Multimodal Models (LMMs) achieve state-of-the-art performance in high-stakes domains like healthcare, yet their reasoning remains opaque. Current interpretability methods, such as attention mechanisms or post-hoc saliency, often fail to faithfully represent the model's decision-making process,...

---

### 21. [Enabling Intrinsic Reasoning over Dense Geospatial Embeddings with DFR-Gemma](https://arxiv.org/abs/2604.07490)

**Authors**: Xuechen Zhang, Aviv Slobodkin, Joydeep Paul, Mandar Sharma, Samet Oymak, Shravya Shetty, Gautam Prasad  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.07490v1  

#### Abstract
Representation learning for geospatial and spatio-temporal data plays a critical role in enabling general-purpose geospatial intelligence. Recent geospatial foundation models, such as the Population Dynamics Foundation Model (PDFM), encode complex population and mobility dynamics into compact embedd...

---

### 22. [A Graph Foundation Model for Wireless Resource Allocation](https://arxiv.org/abs/2604.07390)

**Authors**: Yucheng Sheng, Jiacheng Wang, Le Liang, Hao Ye, Shi Jin  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.07390v1  

#### Abstract
The aggressive densification of modern wireless networks necessitates judicious resource allocation to mitigate severe mutual interference. However, classical iterative algorithms remain computationally prohibitive for real-time applications requiring rapid responsiveness. While recent deep learning...

---

### 23. [Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](https://arxiv.org/abs/2604.07397)

**Authors**: Jinhong Lin, Pan Wang, Zitong Zhan, Lin Zhang, Pedro Morgado  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.07397v1  

#### Abstract
A key inefficiency in diffusion training occurs when a randomly initialized network, lacking visual priors, encounters gradients from the full complexity spectrum--most of which it lacks the capacity to resolve. We propose Data Warmup, a curriculum strategy that schedules training images from simple...

---

### 24. [Reinforcement Learning with LLM-Guided Action Spaces for Synthesizable Lead Optimization](https://arxiv.org/abs/2604.07669)

**Authors**: Tao Li, Kaiyuan Hou, Tuan Vinh, Monika Raj, Zhichun Guo, Carl Yang  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.07669v1  

#### Abstract
Lead optimization in drug discovery requires improving therapeutic properties while ensuring that proposed molecular modifications correspond to feasible synthetic routes. Existing approaches either prioritize property scores without enforcing synthesizability, or rely on expensive enumeration over ...

---

### 25. [Value-Guidance MeanFlow for Offline Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.08174)

**Authors**: Teng Pang, Zhiqiang Dong, Yan Zhang, Rongjian Xu, Guoqiang Wu, Yilong Yin  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.08174v1  

#### Abstract
Offline multi-agent reinforcement learning (MARL) aims to learn the optimal joint policy from pre-collected datasets, requiring a trade-off between maximizing global returns and mitigating distribution shift from offline data. Recent studies use diffusion or flow generative models to capture complex...

---

### 26. [Long-Term Embeddings for Balanced Personalization](https://arxiv.org/abs/2604.08181)

**Authors**: Andrii Dzhoha, Egor Malykh  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.08181v1  

#### Abstract
Modern transformer-based sequential recommenders excel at capturing short-term intent but often suffer from recency bias, overlooking stable long-term preferences. While extending sequence lengths is an intuitive fix, it is computationally inefficient, and recent interactions tend to dominate the mo...

---

### 27. [Bias-Constrained Diffusion Schedules for PDE Emulations: Reconstruction Error Minimization and Efficient Unrolled Training](https://arxiv.org/abs/2604.08357)

**Authors**: Constantin Le Cle\"i, Nils Th\"urey, Xiaoxiang Zhu  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.08357v1  

#### Abstract
Conditional Diffusion Models are powerful surrogates for emulating complex spatiotemporal dynamics, yet they often fail to match the accuracy of deterministic neural emulators for high-precision tasks. In this work, we address two critical limitations of autoregressive PDE diffusion models: their su...

---

### 28. [Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](https://arxiv.org/abs/2604.08537)

**Authors**: Mu Nan, Muquan Yu, Weijian Mai, Jacob S. Prince, Hossein Adeli, Rui Zhang, Jiahang Cao, Benjamin Becker, John A. Pyles, Margaret M. Henderson, Chunfeng Song, Nikolaus Kriegeskorte, Michael J. Tarr, Xiaoqing Hu, Andrew F. Luo  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.08537v1  

#### Abstract
Visual decoding from brain signals is a key challenge at the intersection of computer vision and neuroscience, requiring methods that bridge neural representations and computational models of vision. A field-wide goal is to achieve generalizable, cross-subject models. A major obstacle towards this g...

---

### 29. [FVD: Inference-Time Alignment of Diffusion Models via Fleming-Viot Resampling](https://arxiv.org/abs/2604.06779)

**Authors**: Shivanshu Shekhar, Sagnik Mukherjee, Jia Yi Zhang, Tong Zhang  
**Category**: cs.AI  
**Published**: 2026-04-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.06779v1  

#### Abstract
We introduce Fleming-Viot Diffusion (FVD), an inference-time alignment method that resolves the diversity collapse commonly observed in Sequential Monte Carlo (SMC) based diffusion samplers. Existing SMC-based diffusion samplers often rely on multinomial resampling or closely related resampling sche...

---

### 30. [Large Language Model Post-Training: A Unified View of Off-Policy and On-Policy Learning](https://arxiv.org/abs/2604.07941)

**Authors**: Shiwan Zhao, Zhihu Wang, Xuyang Zhao, Jiaming Zhou, Caiyue Xu, Chenfei Liu, Liting Zhang, Yuhang Jia, Yanzhe Zhang, Hualong Yu, Zichen Xu, Qicheng Li, Yong Qin  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.07941v1  

#### Abstract
Post-training has become central to turning pretrained large language models (LLMs) into aligned and deployable systems. Recent progress spans supervised fine-tuning (SFT), preference optimization, reinforcement learning (RL), process supervision, verifier-guided methods, distillation, and multi-sta...

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
