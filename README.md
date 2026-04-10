# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-10 07:14:53 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [QaRL: Rollout-Aligned Quantization-Aware RL for Fast and Stable Training under Training--Inference Mismatch](https://arxiv.org/abs/2604.07853)

**Authors**: Hao Gu, Hao Wang, Jiacheng Liu, Lujun Li, Qiyuan Zhu, Bei Liu, Binxing Xu, Lei Wang, Xintong Yang, Sida Lin, Sirui Han, Yike Guo  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 14.5  
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
在大型语言模型（LLM）的强化学习（RL）训练中，**rollout 阶段的自回归解码是计算瓶颈**，占整个训练时间的约 70%。为加速这一过程，已有研究尝试对 rollout 模型进行低比特量化（如 W4A16），从而提升推理速度。

然而，这种做法引入了严重的 **training-inference mismatch**（训练-推理不一致）问题：
- **Rollout** 在低精度（如 INT4/FP4）下生成响应；
- **Policy learning 更新** 却基于全精度（如 BF16）模型计算梯度；
- 导致采样分布 $ \pi_{\text{sampler}} $ 与学习策略 $ \pi_{\text{learner}} $ 不一致，破坏信任区域（trust region），引发训练不稳定甚至崩溃。

此外，作者还发现一个关键失败模式：**量化 rollout 容易产生“error tokens”** ——即在长文本生成过程中出现重复、乱码等偏离轨迹的 token，这些 token 在原始策略下概率极低，导致重要性权重（importance ratio）剧烈波动，进一步加剧训练不稳定性。

---

### 🚀 提出的新方法与创新思路

#### （1）**QaRL（Rollout-Aligned Quantization-Aware RL）**
- **核心思想**：让 learner 的前向传播行为与 quantized rollout 引擎保持一致。
- 具体实现：
  - 在训练端维护高精度 master weights（如 BF16）；
  - 前向时动态执行 **low-bit GEMM**（如 W4A16），真实模拟量化引擎的行为；
  - 反向传播仍用 STE（Straight-Through Estimator）更新高精度参数；
  - 每步将训练后的低比特权重同步回 rollout 引擎，避免重复量化。
- 效果：显著缩小 $ \pi_{\text{learner}} $ 和 $ \pi_{\text{quant-sampler}} $ 的差距，缓解 mismatch。

#### （2）**TBPO（Trust-Band Policy Optimization）**
- 针对 error tokens 设计的新型优化目标：
  - **双侧剪裁（Dual Clipping）**：特别针对负优势样本（negative advantages），不仅限制下界（1−ε），也设置上界（1+δ），防止因低概率 error tokens 导致 ratio 爆炸；
  - **序列级目标（Sequence-Level Objective）**：以整条 response 作为 action，计算几何平均的重要性比率 $ r_{\text{seq-prox}} $ 和 mismatch 权重 $ w_{\text{seq-mismatch}} $；
  - 若整条序列超出信任带，则直接丢弃该样本，避免污染更新。

> 💡 创新本质：从 token-level 控制升级到 **sequence-level trust band control**，更鲁棒地应对量化噪声下的错误累积。

---

### 🔍 相比现有方法的优势

| 方法 | 缺陷 | QaRL/TBPO 改进 |
|------|------|----------------|
| **Quantized Rollout Only** | 严重 mismatch，训练不稳定 | QaRL 对齐 learner 前向行为，减少分布偏移 |
| **标准 PPO/GRPO** | 无法控制 error tokens 的极端 ratio | TBPO 引入 dual clipping + sequence-level clipping，有效抑制异常梯度 |
| **Bitwise Consistent Kernels** | 实现复杂，性能损失大（~2×慢） | QaRL 不要求完全一致 kernel，仅需 arithmetic alignment，性价比更高 |
| **纯 QAT 或 FQT** | 多用于 SFT，未适配 RL 场景 | QaRL 是首个专为 hybrid RL 架构设计的量化感知训练框架 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **主任务**：数学推理
  - `OpenR1-Math-46K`（46K 数学题）
  - 测试集：AIME2024/2025, AMC, MATH-500, Minerva, Olympiad-Bench
- **泛化能力评估（OOD）**
  - ARC-Challenge, GPQA-Diamond, LiveCodeBench, MMLU-Pro

---

### ⚙️ 实验设置
- **模型规模**：
  - Qwen2.5-1.5B-Math
  - Qwen2.5-7B-Math
  - Qwen3-8B-Base
  - Qwen3-30B-A3B-Base（MoE 模型）
- **训练框架**：
  - Rollout 引擎：vLLM
  - 训练后端：Verl（基于 Ray）
  - 硬件：8× NVIDIA H800 GPUs
- **量化配置**：
  - 主要采用 W4A16（权重量化为 4-bit，激活保持 16-bit）
  - 对比 W8A8、FP8 等方案
- **优化器**：Muon（收敛更快于 AdamW）
- **超参关键值**：
  - `seq_clip_ratio_high`: 0.0004
  - `neg_seq_clip_ratio_low/high`: 0.0003 / 0.0007
  - `seq_tis_imp_ratio_cap`: 2.0

---

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **BF16 GRPO/GSPO** | 全精度训练，无量化，作为性能上限基准 |
| **w4a16 rollout GRPO/GSPO** | 仅 rollout 量化，learner 全精度，典型 mismatch 设置 |
| **QaRL w/o TBPO** | 含对齐前向但无双剪裁机制 |
| **TBPO variants** | 消融不同 clipping 策略的影响 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Pass@1 平均准确率）

| Model | BF16 Baseline | Quantized Rollout | **QaRL + TBPO** | 提升幅度 |
|-------|---------------|--------------------|------------------|----------|
| Qwen2.5-1.5B-Math | 34.5 | 29.3 | **33.9** | **+4.6 pts** |
| Qwen2.5-7B-Math | 43.3 | 40.9 | **43.5** | **+2.6 pts** |
| Qwen3-8B-Base | 51.8 | 43.9 | **48.9** | **+5.0 pts** |
| Qwen3-30B-A3B-Base (MoE) | 52.1 | 45.7 | **51.2** | **+5.5 pts** |

> ✅ **QaRL + TBPO 几乎追平 BF16 性能，在 MoE 上实现 +5.5 的巨大提升**

---

### ⏱️ 推理与训练效率
- **训练速度**：
  - 相比 BF16，QaRL 实现 **1.3× 的 per-step 加速**
  - Quantized rollout 达到 1.4×，略快但牺牲稳定性
- **原因分析**：
  - QaRL 因需在训练中执行 low-bit GEMM，带来轻微开销；
  - 但在 MoE 模型上，W4 显著降低内存占用，减少通信开销，整体仍高效。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）**优化目标对比（Fig. 8）**
| 方法 | 表现 |
|------|------|
| **GSPO** | KL 漂移严重，reward 下降（error tokens 污染） |
| **MIS + GSPO** | 拒绝采样降低数据利用率，天花板受限 |
| **Positive-only GRPO** | 忽略负样本，探索不足，性能受限 |
| **On-policy GRPO** | 更新频率低，效率差 |
| **Dual-clip GRPO** | 抑制极端 ratio，但仍受残余 error tokens 影响 |
| ✅ **TBPO** | KL 控制稳定，reward 最高，收敛可靠 |

> ➤ 结论：**sequence-level + dual clipping 是稳定训练的关键组合**

#### （2）**量化方案对比（Fig. 9a）**
- W4A16、W4A8、W8A8、FP8W8A8 在 QaRL+TBPO 下表现相近
- 说明：一旦通过 TBPO 稳定训练，最终性能对具体 bit-width **不敏感**
- 选择 W4A16 主要因其在大模型上的硬件兼容性和吞吐优势

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Quantized Rollout 虽然提速，但会放大 training-inference mismatch，导致训练不稳定**；
2. **Error tokens 是量化 rollout 中训练崩溃的核心诱因**，尤其在长输出中积累并破坏 trust region；
3. **QaRL 通过对齐 learner 前向行为，显著减小分布差异**；
4. **TBPO 通过 sequence-level dual clipping 成功隔离 error tokens，保障更新稳健性**；
5. **QaRL + TBPO 在多个模型尺度上接近甚至匹配 BF16 性能，同时保留 ~1.3× 的训练加速**；
6. 方法在 MoE 架构（如 Qwen3-30B-A3B）上依然稳定，验证了通用性。

---

### ⚠️ 局限性
1. **尚未实现 Fully Quantized Training（FQT）**：
   - 当前 backward 仍使用 BF16，未尝试 4-bit 梯度（担心数值不稳定）；
2. **Sequence-level clipping 可能浪费样本**：
   - 一旦某条 response 出现 error token 就整条丢弃，可能影响样本效率；
3. **依赖低比特 GEMM 内核支持**：
   - 不同硬件平台对 W4A16 支持程度不同，部署门槛存在；
4. **未解决 KV Cache 量化问题**：
   - 当前未启用 FP8 KV quantization，因其在 vLLM 中未能提效。

---

### 🔮 未来工作方向
1. **探索 Fully Quantized RL Training**：
   - 实现 end-to-end 低比特 forward + backward，进一步压缩资源消耗；
2. **设计更高效的 token-level 近似方法**：
   - 替代昂贵的 sequence-level 判断，在保持稳定性的同时提高样本利用率；
3. **结合 off-policy correction 技术**：
   - 如与 TIS、MIS 更深度整合，增强对历史数据的利用；
4. **扩展至多模态与 Agent 场景**：
   - 将 QaRL 应用于需要长时间 rollouts 的 agent-based decision making。

---

> 📌 **一句话总结**：  
> **QaRL 通过 rollout-aligned quantization-aware training 与 TBPO 的 sequence-level trust band 控制，在不牺牲训练稳定性的前提下，实现了量化 rollout 的高速度与高性能兼得，为大规模 LLM-RL 的实用化提供了可行路径。**

</details>

---

### 2. [AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention](https://arxiv.org/abs/2604.07815)

**Authors**: Yuxuan Hu, Jianchao Tan, Jiaqi Zhang, Wen Zan, Pingwei Sun, Yifan Lu, Yerui Sun, Yuchen Xie, Xunliang Cai, Jing Zhang  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2604.07815v1  

#### Abstract
Long-context inference in LLMs faces the dual challenges of quadratic attention complexity and prohibitive KV cache memory. While token-level sparse attention offers superior accuracy, its indexing overhead is costly; block-level methods improve efficiency but sacrifice precision. We propose AsyncTL...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在长上下文推理中面临两大瓶颈：
- **计算复杂度**：自注意力机制具有 $O(n^2)$ 的计算复杂度，随序列长度增长而急剧上升。
- **内存开销**：Key-Value (KV) cache 占用大量高带宽 GPU 内存，在处理数十万 token 的长序列时极易超出显存容量。

现有稀疏注意力方法存在精度与效率之间的权衡：
- **Token-level 稀疏**：如 Double-Sparsity (DS)，能精准保留重要 token，准确率高，但索引开销大。
- **Block-level 稀疏**：如 Quest，硬件友好、效率高，但因粗粒度选择引入无关 token 或遗漏关键信息，影响精度。

### 提出的新方法与创新思路
论文提出 **AsyncTLS**，一种结合精度与效率的分层稀疏注意力系统，包含两个核心组件：

#### （1）Two-Level Sparse Attention 架构
- **Level-1: Block-Level Filtering**  
  使用块级索引对 KV cache 进行粗筛选，快速剔除不相关区域。
- **Level-2: Token-Level Selection**  
  在保留的块内进行细粒度 token 级别选择，确保语义关键信息被精确捕获。
- 该设计通过“先粗后精”的方式，在大幅减少搜索空间的同时保持 token-level 的高精度。

#### （2）AsyncTLS Offloading Engine
针对 KV cache 显存不足问题，提出异步卸载引擎：
- **Temporal Overlap（时间重叠）**  
  利用相邻解码步间注意力模式的时间局部性（temporal locality），将当前步的 block selection 结果用于预取下一步所需 KV 块。
- **Incremental Block Transfer（增量传输）**  
  只传输前后两步 block selection 的差异部分，显著降低 PCIe 带宽消耗。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **准确性** | 接近 Full Attention 和 token-level 方法（如 DS），优于 block-level 方法（如 Quest） |
| **效率** | 比 token-level 方法快 1.2×–4.0×，比 Full Attention 快 1.7×–10.0× |
| **内存管理** | 支持高效 KV cache offloading，实现更大 batch size 处理 |
| **架构兼容性** | 验证于 MHA、GQA 和 MLA 架构，尤其在 MLA 上表现优异 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LongBench**：涵盖14项长文本理解任务，包括：
  - 叙事理解（Narrative QA）
  - 科学问答（QasperQA）
  - 多跳推理（HotpotQA, MultiFieldQA, Musique）
  - 文档摘要（GovReport, QMSum, Multi-News）
  - 特殊任务（TriviaQA, RepoBench-P 等）

- **RULER**：专注于长上下文检索能力评估，包含：
  - 单/多关键词检索（S1/S2/MK1/MK2）
  - 多查询/值匹配（MQ/MV）
  - 开放式问答（QA-1/QA-2）
  - 视觉文本定位（VT）、事实验证（FWE）

### 实验设置与评估指标
| 设置项 | 描述 |
|-------|------|
| **模型** | Qwen3-8B、Qwen3-14B、GLM-4.7-Flash |
| **上下文长度** | 32k–128k tokens |
| **稀疏预算** | 512 / 1024 / 2048 tokens |
| **Block 配置** | Block size = 64，每步选取 128 blocks（共 8192 tokens） |
| **Token-level 维度** | GQA: 32 dims；MLA: 128 dims，INT4 量化压缩 |
| **评估指标** | 准确率（Accuracy）、Latency（延迟）、Throughput（吞吐量） |

### 基线方法对比
- **Full Attention (FA)**：完整注意力作为性能上限基准
- **Quest**：代表 block-level 动态稀疏方法
- **Double-Sparsity (DS)**：代表 token-level 稀疏方法，精度较高但开销大

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ 准确性结果（LongBench 平均得分 @1024 token budget）
| 方法 | Qwen3-8B | Qwen3-14B | GLM-4.7-Flash |
|------|----------|-----------|----------------|
| Full Attention | 50.14 | 52.16 | 50.83 |
| Quest | 47.49 | 49.33 | 46.75 |
| DS | 50.10 | 51.10 | 50.73 |
| **AsyncTLS** | **49.90** | **51.80** | **50.58** |

> ➤ AsyncTLS 准确率接近 Full Attention 和 DS，显著优于 Quest。

#### ✅ RULER 检索任务平均得分（@512 token budget）
| 方法 | Qwen3-14B | GLM-4.7-Flash |
|------|------------|------------------|
| Full Attention | 86.24 | 85.73 |
| Quest | 37.89 | 53.32 |
| DS | 77.20 | 82.38 |
| **AsyncTLS** | **87.21** | **81.50** |

> ➤ AsyncTLS 在 Qwen3 上超越 Full Attention，在 GLM 上仅次于 DS。

#### ✅ 效率提升（Operator Speedup vs. Full Attention）
| 架构 | Batch Size | Speedup 范围 |
|------|------------|-------------|
| GQA | 1–8 | **1.7× – 6.2×** |
| MLA | 1–8 | **3.3× – 10.0×** |

> ➤ AsyncTLS 在 MLA 架构上加速尤为明显，得益于其低秩 KV 表示特性。

#### ✅ 端到端吞吐量提升（With KV Offloading, @96k seq len）
| 模型 | 方法 | Batch Size | Throughput 提升 |
|------|------|------------|------------------|
| Qwen3-8B | FA → AsyncTLS | 1 → 6 | **1.84×** |
| GLM-4.7-Flash | FA → AsyncTLS | 1 → 6 | **4.70×** |

> ➤ 因 AsyncTLS 显著减小 KV cache 占用，支持更大 batch 推理，极大提升实际部署吞吐。

---

### 消融实验分析（隐含于主实验趋势中）
虽然未明确列出消融表，但从以下观察可推断各模块作用：
- **Hierarchical Selection**：相比纯 token-level 方法（DS），AsyncTLS 在几乎无损精度下实现了高达 **4× 的算子加速**，说明 block-level filtering 有效降低了 token-level indexing 开销。
- **Asynchronous Prefetching + Incremental Transfer**：通过隐藏内存传输延迟，使系统可在 batch size 扩展至 6 的情况下仍维持低延迟，证明了 offloading 引擎的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. **Hierarchical Sparsity 是平衡精度与效率的有效路径**  
   将 block-level 与 token-level 结合，既能享受 coarse-grained 的高效索引，又能保留 fine-grained 的高精度表达。

2. **Temporal Locality 可被有效利用以优化 KV Offloading**  
   相邻解码步的 block selection 具有高度一致性，可用于预测并预取下一阶段数据，实现计算与通信的重叠。

3. **Training-Free Token-Level Sparsity 可实用化**  
   以往认为 token-level 稀疏因索引开销难以部署，本工作表明通过分层设计和异步调度，可在无需微调的前提下实现高性能部署。

4. **MLA 架构特别适合 AsyncTLS**  
   由于 MLA 本身采用压缩 latent 表示存储 KV，与 token-level selection 更加契合，因此 AsyncTLS 在 MLA 上获得最大加速收益（达 10×）。

---

### 方法的局限性
- **依赖 temporal locality 假设**：若 attention pattern 变化剧烈（如跳跃式话题转换），block-level 预测可能失效，影响 token selection 效果。
- **实现复杂度较高**：需维护两级索引结构，并协调异步 prefetch 流程，工程实现难度高于传统方法。
- **对极短序列增益有限**：主要优势体现在 32k 以上长上下文场景，在常规长度下性价比不高。

---

### 未来工作方向
1. **动态调整层级参数**：根据输入内容自动调节 block 数量 $k_b$ 与 token 数量 $k_t$，进一步优化资源分配。
2. **扩展至训练阶段**：探索是否可将 hierarchical sparse attention 应用于训练过程以降低成本。
3. **跨设备协同优化**：结合更复杂的 CPU-GPU-NUMA 内存拓扑，最大化异步传输效率。
4. **支持流式输入更新**：适配 streaming LLM 场景下的动态 chunk 添加与缓存更新。

---

## 总结
**AsyncTLS** 成功解决了长上下文 LLM 推理中“精度 vs. 效率”的根本矛盾。它通过 **two-level sparse attention** 实现了 token-level 的精度与 block-level 的效率统一，并借助 **asynchronous offloading engine** 最大化利用时间局部性，实现了计算与内存访问的高效重叠。实验表明，该方法在多个主流模型和架构上均能达到与 Full Attention 相当的性能，同时带来 **1.2×–10.0× 的算子加速** 和 **1.3×–4.7× 的端到端吞吐提升**，为超长序列生成提供了可扩展、实用化的解决方案。

</details>

---

### 3. [A Novel Edge-Assisted Quantum-Classical Hybrid Framework for Crime Pattern Learning and Classification](https://arxiv.org/abs/2604.07389)

**Authors**: Niloy Das, Apurba Adhikary, Sheikh Salman Hassan, Yu Qiao, Zhu Han, Tharmalingam Ratnarajah, Choong Seon Hong  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 12.0  
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
- 犯罪类型严重类别不平衡（如凶杀案等罕见但关键事件）
- 资源受限环境下（如边缘设备）的传统机器学习模型计算开销大、部署困难

这些问题限制了传统 ML 在智能城市监控系统中分布式无线传感器网络（WSN）上的实时应用。

---

### 🚀 提出的新方法与创新思路

1. **首次系统性地比较量子、经典与混合范式的犯罪分类性能**
   - 构建了一个四范式比较框架：纯量子（Quantum）、纯经典（Classical）、Q→C（量子特征提取 + 经典分类）、C→Q（经典降维 + 量子建模）

2. **提出“相关性感知”的量子电路设计（Correlation-Aware Quantum Circuit）**
   - 利用 Spearman 相关性分析识别高相关特征对
   - 在 VQC 中引入基于相关性的选择性纠缠（targeted entanglement），提升模型表达能力

3. **双向 Hybrid 架构实现**
   - **Q→C**: 使用 VQC 提取量子特征，后接 Random Forest/SVM/Logistic Regression 等经典分类器
   - **C→Q**: 使用 PCA 进行维度压缩至 4 维，再输入 QAOA/VQC/QKernel 模型，适配 NISQ 设备的 qubit 数量限制

4. **面向边缘计算优化的设计理念**
   - 强调低参数量、小内存占用、低通信成本，适用于 WSN 节点部署

---

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **资源效率** | QAOA 仅需 16–36 可训练参数，相比 Random Forest (~150k) 减少超 9000×，适合 memory-constrained 边缘节点 |
| **训练效率** | QAOA 单 fold 训练时间仅 0.004–0.006 秒，显著快于多数经典模型 |
| **通信开销** | 推理阶段每条输出仅为 class label + confidence score（约 9 字节），极大降低无线传输负担 |
| **对少数类敏感** | QAOA 在 “Critical” 类别上表现优于部分经典模型，显示其在捕捉非线性交互方面的潜力 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **Bangladesh Police 官方发布的 16 年犯罪统计数据（2010–2025）**
- 包含 18 个报告单位（metropolitan areas 和 police ranges）
- 总样本数：272（18 × 16）
- 特征工程后保留 10 个关键特征（通过互信息筛选）：
  - Total Cases, Crime Std Dev, Woman & Child Repression, Violent Crime Total, Murder, Theft, etc.
- 目标变量构建为 4 分类任务（基于暴力犯罪比例 $ r_v $ 和案件总数 $ C $）：
  ```
  S(x) = {Critical, High, Medium, Low}
  ```

---

### ⚙️ 实验设置

#### 数据预处理流程
1. **Feature Engineering**：构造聚合特征（如暴力犯罪总量、多样性指数）
2. **Quantum-Compatible Dimensionality Reduction**：
   - 使用 PCA 将特征降至 4 或 6 维，保留 ≥95% 方差
   - 公式：$ X_{\text{reduced}} = W_{\text{PCA}} X_{\text{original}} $
3. **标准化**：StandardScaler 缩放所有输入

#### 模型配置（见 Table I）
| 类别 | 模型 | 主要参数 |
|------|------|---------|
| **Quantum** | VQC | 4–6 qubits, 2–3 layers |
| | QAOA | 4–6 qubits, 2–3 p-layers |
| | QKernel SVM | 4 qubits, RBF kernel |
| **Q→C Hybrid** | Q→RF, Q→SVM, etc. | 6q → classical classifier |
| **C→Q Hybrid** | PCA→QAOA, etc. | 4 PCs → 4-qubit quantum model |
| **Pure Classical** | Random Forest, SVM(RBF), Logistic Regression, Decision Tree | 标准参数设置 |

#### 评估协议
- **Stratified 5-Fold Cross-Validation** + **5 不同随机种子** ⇒ 共 25 次评估/模型
- 替代初步实验中的单次 80/20 划分，提高统计可靠性
- 所有量子模型通过 **classical simulation** 实现（模拟 variational circuits）

#### 评估指标（见 Table II）
| 指标类型 | 指标名称 | 公式/说明 |
|--------|--------|----------|
| **Classification** | Accuracy, Precision, Recall, F1-score | 加权平均，尤其关注 minority class（Critical）的 F1 |
| **Quantum Efficiency** | Parameter Count, Circuit Depth, Qubit Efficiency ($ \text{Acc}/n_q $) | 衡量资源利用率 |
| **Comparative Metrics** | Speedup Factor ($ T_C / T_Q $), Performance Gap ($ \Delta = \text{Acc}_C - \text{Acc}_Q $), Paired t-test p-value | 统计显著性检验 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table IV 和 V）

| 模型 | Accuracy (±CI) | Weighted F1 (±CI) | 参数数量 | 训练时间 (s/fold) |
|------|----------------|------------------|-----------|------------------|
| **Random Forest (Best Classical)** | **0.945 ± 0.016** | **0.944 ± 0.016** | ~150,000 | 0.273 |
| **QAOA (6q, 3L)** | 0.846 ± 0.019 | 0.830 ± 0.021 | **36** | **0.004** |
| **QAOA (4q, 2L)** | 0.803 ± 0.024 | 0.779 ± 0.026 | **16** | 0.006 |
| **PCA→QAOA** | 0.803 ± 0.024 | 0.779 ± 0.026 | 16 | 0.006 |
| **VQC (6q, 3L)** | 0.686 ± 0.021 | — | 18 | 0.027 |
| **QKernel SVM** | 0.359 ± 0.028 | — | — | 0.075 |

> 注：CI = 95% Confidence Interval；t-test 显示经典方法显著更优（p < 0.001）

---

### 🔁 与基线方法的对比结果

| 对比维度 | 结果 |
|--------|------|
| **整体准确率** | Random Forest 最佳（94.5%），明显优于最佳量子模型（QAOA: 84.6%） |
| **参数效率** | QAOA 参数量仅为 RF 的 **1/9000**，极具边缘部署优势 |
| **训练速度** | QAOA 是最快的模型之一（0.004s/fold），远快于 RF（0.273s） |
| **初步 vs 鲁棒评估差异** | 单次划分下 QAOA 曾达 85%，看似媲美经典模型；但交叉验证揭示其稳定性不足，凸显 single-split 的乐观偏差风险 |

---

### 🔍 消融实验与关键发现（隐含分析）

1. **电路深度影响表达能力（Figure 4）**
   - 随着层数增加，circuit expressibility 从 0.35 提升到 0.79
   - 支持深层电路能更好捕获非线性犯罪模式

2. **不同 Hybrid 架构效果对比**
   - **Q→C**：未能超越经典 baseline（如 Q→RF = 75%），表明单纯量子特征提取未带来增益
   - **C→Q**：PCA→QAOA 表现良好（80.3%），说明先降维有助于适配量子硬件约束

3. **VQC 与 QKernel 表现不佳原因分析**
   - **VQC underfitting**：可能因 $ \cos^2 $ 特征映射丢失幅度信息
   - **QKernel limited by qubit count**：仅 4 qubits 下 Hilbert 空间太小，无法体现核方法优势

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **QAOA 是当前最有效的量子启发模型**
   - 在多种设置下均表现最优（最高达 84.6% accuracy）
   - 天然适合建模特征间成对交互（通过 Cost Hamiltonian 编码相关性）

2. **量子方法尚未超越经典模型，但在资源效率方面具有巨大潜力**
   - 尽管 accuracy 存在差距（~10%），但其极低的参数量和训练时间支持其在边缘场景的应用前景

3. **correlation-aware circuit design 有效提升了量子模型性能**
   - 基于 Spearman 相关性的 entanglement pattern 设计增强了模型对领域知识的利用

4. **Hybrid 架构未展现出明显优势**
   - 当前 Q→C 和 C→Q 架构未能融合两者长处，仍需进一步架构探索

5. **Single-split 评估存在严重偏差**
   - 初步结果显示 QAOA 可媲美经典模型（p=0.1835），但交叉验证证实经典方法显著更优（p<0.001）

---

### ⚠️ 局限性

| 局限 | 描述 |
|------|------|
| **Simulation-only Evaluation** | 所有量子模型基于经典模拟，并未运行在真实量子硬件上，忽略噪声和误差影响 |
| **Small Dataset Size** | 仅 272 样本，难以充分训练复杂模型，尤其是 deep QML 模型 |
| **Class Imbalance** | 尽管目标变量设计考虑平衡性，“Critical” 类仍属少数，影响泛化能力 |
| **Limited Qubit Count** | 最多使用 6 qubits，限制了量子态空间的表达能力 |
| **No Hardware Noise Modeling** | 未考虑 NISQ 设备中的 decoherence、gate error 等实际因素 |

---

### 🔮 未来工作方向

1. **Real Quantum Hardware Deployment**
   - 在真实 NISQ 设备（如 IBM Quantum, Rigetti）上验证 QAOA 性能
   - 引入 error mitigation 技术应对噪声干扰

2. **Noise-Robust Circuit Design**
   - 开发抗噪的 variational ansatz 和训练策略

3. **Edge Benchmarking**
   - 在真实 WSN 节点（如 Raspberry Pi + IoT sensors）上部署轻量级 QAOA 推理模块
   - 测量能耗、延迟、通信开销等实际指标

4. **更大规模数据集扩展**
   - 应用于其他国家或城市的长期犯罪数据，增强模型普适性

5. **动态更新机制**
   - 探索 online learning 或 incremental training 框架，适应犯罪趋势变化

---

## ✅ 总结一句话
> 尽管当前 **QAOA 尚未在精度上超越 Random Forest**，但其**极低的参数量和训练开销**使其成为面向 **edge-assisted 智慧城市监控系统** 中极具潜力的候选方案，特别是在带宽与存储受限的无线传感器网络中。本研究为 QML 在公共安全领域的落地提供了首个实证基准。

</details>

---

### 4. [TurboAgent: An LLM-Driven Autonomous Multi-Agent Framework for Turbomachinery Aerodynamic Design](https://arxiv.org/abs/2604.06747)

**Authors**: Juan Du, Yueteng Wu, Pan Zhao, Yuze Liu, Min Zhang, Xiaobin Xu, Xinglong Zhang  
**Category**: cs.AI  
**Published**: 2026-04-10  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2604.06747v2  

#### Abstract
The aerodynamic design of turbomachinery is a complex and tightly coupled multi-stage process involving geometry generation, performance prediction, optimization, and high-fidelity physical validation. Existing intelligent design approaches typically focus on individual stages or rely on loosely cou...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统涡轮机械（turbomachinery）气动设计依赖专家经验与反复试错，流程复杂且高度耦合，涉及需求分析、几何生成、性能预测、优化决策和高保真物理验证等多个阶段。现有智能设计研究多局限于单一环节或通过脚本连接工具，缺乏**端到端自主闭环能力**，难以实现真正意义上的自动化设计。

### **提出的新方法与新思路**
本文提出了 **TurboAgent**，一个由大语言模型（LLM）驱动的**自主多智能体框架**（multi-agent framework），用于涡轮机械气动设计与优化。其核心创新包括：

- **统一的LLM认知中枢**：LLM作为中央协调者，负责高层次的任务规划、意图理解与动态调度，将自然语言设计需求转化为可执行的工作流。
- **功能化专用Agent协作机制**：
  - **任务规划Agent**（Task Planning Agent）：解析需求并动态构建执行路径。
  - **生成设计Agent**（Generative Design Agent）：基于条件去噪扩散概率模型（cDDPM）进行逆向几何生成。
  - **性能预测Agent**（Performance Prediction Agent）：采用Transformer架构的代理模型实现毫秒级性能评估。
  - **优化Agent**（Optimization Agent）：集成LLM驱动的元提示优化算法、遗传算法（GA）、粒子群优化（PSO）等。
  - **物理验证Agent**（Physics Validation Agent）：自动调用CFD（NUMECA/ANSYS CFX）和FEA进行高保真仿真验证。
  - **知识合成Agent**（Knowledge Synthesis Agent）：整合结果、生成报告、支持问答交互。
- **数据驱动与高保真验证融合**：结合快速生成/预测与最终CFD/FEA验证，确保设计既高效又符合物理一致性。

### **相比现有方法的优势**
| 维度 | 传统方法 | 现有智能方法 | TurboAgent |
|------|--------|------------|----------|
| **流程控制** | 人工主导 | 脚本串联 | LLM动态规划 |
| **闭环能力** | 弱 | 局部闭环 | 全流程自主闭环 |
| **人机交互** | 图形界面操作 | 参数输入 | 自然语言指令 |
| **知识复用** | 依赖专家 | 模型黑箱 | Prompt工程显式编码 |
| **灵活性** | 固定流程 | 静态管道 | 动态适应反馈 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 数据来源于文献 [33]，基于一个跨音速1.5级压气机转子叶片原型。
- 包含参数化后的三维叶片几何与对应气动性能（质量流量 $ \dot{m} $、总压比 $ \pi $、等熵效率 $ \eta $）。
- 参数化方式：在叶根（hub）、中径（mid-span）、叶尖（tip）三个截面使用NURBS描述弯度线与厚度分布，共定义21个设计变量（如前缘金属角、弦长、最大厚度位置、弯掠等）。

### **实验设置与评估指标**
#### **验证案例**
- **对象**：跨音速单级转子压气机（transonic single-rotor compressor）
- **目标性能**：$ \dot{m} = 15.2\,\text{kg/s},\ \pi = 1.62,\ \eta = 0.88 $

#### **评估维度**
1. **任务规划能力**：测试LLM对不同复杂度自然语言请求的理解与工作流构建能力。
2. **单Agent功能验证**：
   - 生成设计Agent：生成多样性与目标一致性。
   - 性能预测Agent：预测精度（R², nRMSE, MAE）。
   - 物理验证Agent：CFD/FEA自动化执行与结果准确性。
   - 优化Agent：收敛速度与优化效果。
3. **端到端全流程验证**：从自然语言输入到最终设计方案输出的完整闭环能力。
4. **成本分析**：Token消耗与计算时间。

#### **基线方法对比**
- **优化算法对比**：LLM-driven optimizer vs. GA vs. PSO
- **性能预测模型**：Transformer代理模型 vs. CFD真值
- **设计范式对比**：TurboAgent vs. 传统人工迭代设计（周期数周）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 指标 | 数值 | 说明 |
|------|-----|------|
| **生成设计一致性（vs. CFD）** | R² > 0.91, nRMSE < 8% | 所有三项性能指标均达到高一致性 |
| **性能预测准确率（vs. CFD）** | R²: 0.9807 ($ \dot{m} $), 0.9824 ($ \pi $), 0.9184 ($ \eta $); nRMSE < 8% | 代理模型具备高精度快速评估能力 |
| **优化提升效果** | $ \Delta\eta = +1.61\% $, $ \Delta\pi = +3.02\% $ | 相较初始设计显著提升 |
| **全流程耗时** | ~30分钟（并行计算下） | 含CFD仿真，相较传统方法提速1–2个数量级 |
| **Token消耗** | ~80,000–100,000 tokens | 完整闭环流程（不含报告生成） |

### **与基线方法的对比结果**

#### **优化算法对比（图13）**
- **LLM-driven optimizer** 在早期收敛更快，最终奖励值最高。
- 相比GA和PSO，无需显式定义进化算子，能根据语义自适应调整搜索策略。
- 最终设计：
  - LLM优化器：$ \dot{m}=14.995\,\text{kg/s},\ \eta=0.8830,\ \pi=1.637 $
  - GA：$ \eta=0.8930 $（更高），但 $ \pi=1.609 $ 偏低
  - PSO：$ \pi=1.641 $ 最高，但 $ \eta=0.8770 $ 较低
- 表明LLM优化器在多目标权衡上更具灵活性。

#### **性能预测 vs. CFD仿真（图11）**
- 设计目标与CFD结果高度一致：
  - $ R^2_{\dot{m}} = 0.9775,\ R^2_{\pi} = 0.9795,\ R^2_{\eta} = 0.9158 $
- 代理模型预测与CFD结果也高度吻合（$ R^2 > 0.91 $），证明其可靠性。

### **消融实验结果**
虽然未明确列出“消融实验”章节，但以下实验证实了各模块有效性：
- **任务规划Agent验证**（图5）：成功处理三种场景——简单生成、带条件分支的高保真验证、带反馈回路的优化流程，体现强推理与调度能力。
- **生成多样性验证**（图7）：同一目标下生成多个不同几何构型，增强设计探索空间。
- **多任务工作流验证**（图14–15）：完成从10个候选方案生成、筛选、优化、CFD/FEA验证、离工况分析到最终推荐的全过程，全程无手动干预。

---

## **4. 关键结论和发现**

### **主要发现**
1. **TurboAgent实现了真正的端到端自主设计闭环**：
   - 输入为自然语言需求，输出为经CFD/FEA验证的最优设计方案。
   - 整个流程由LLM统一规划，多Agent协同执行，形成“感知-决策-行动-验证”的智能循环。

2. **LLM不仅是接口，更是智能调度核心**：
   - 能够理解复杂工程语义，动态构建条件分支与迭代逻辑。
   - 支持自然语言交互下的实时修改与重规划（如材料更换、参数敏感性分析）。

3. **数据驱动与物理验证有效融合**：
   - 利用cDDPM + Transformer实现快速生成与评估，大幅提升迭代效率。
   - 保留CFD/FEA作为最终物理一致性检验，保障工程可用性。

4. **LLM-driven优化展现优越潜力**：
   - 在无需梯度信息的情况下，通过语义理解实现高效多目标优化。
   - 相比传统优化算法，在收敛速度与策略灵活性方面表现更优。

### **方法的局限性**
1. **训练数据依赖性强**：当前cDDPM与Transformer模型依赖特定压缩机数据库，泛化至其他机型或工况需重新训练。
2. **LLM性能波动风险**：优化结果受底层LLM能力影响较大，不同模型可能导致不一致行为。
3. **高保真仿真仍是瓶颈**：尽管自动化程度高，CFD求解仍占主要时间（约30分钟/工况）。
4. **缺乏不确定性量化**：未提供生成或预测结果的置信区间，可能影响工程决策安全性。

### **未来工作方向**
1. **增强泛化能力**：开发跨构型、跨工况的通用生成与预测模型。
2. **引入强化学习**：结合RL进一步提升LLM优化器的自主决策能力。
3. **提升可解释性与可信度**：增加设计过程的透明度与不确定性建模。
4. **扩展至多学科设计优化**（MDAO）：集成热、振动、噪声等更多物理场。
5. **部署轻量化版本**：降低对高性能计算资源的依赖，推动工业落地。

---

> ✅ **总结一句话**：  
> **TurboAgent首次实现了由LLM统一规划、多Agent协同执行、高保真仿真闭环验证的涡轮机械全自主气动设计框架，标志着从“经验驱动”向“智能自主”设计范式的重大跃迁。**

</details>

---

### 5. [Reinforcement Learning with Reward Machines for Sleep Control in Mobile Networks](https://arxiv.org/abs/2604.07411)

**Authors**: Kristina Levina, Nikolaos Pappas, Athanasios Karapantelakis, Aneta Vulgarakis Feljan, Jendrik Seipp  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.07411v1  

#### Abstract
Energy efficiency in mobile networks is crucial for sustainable telecommunications infrastructure, particularly as network densification continues to increase power consumption. Sleep mechanisms for the components in mobile networks can reduce energy use, but deciding which components to put to slee...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Reinforcement Learning with Reward Machines for Sleep Control in Mobile Networks*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文研究了移动网络中 **Radio Units (RUs)** 的 **Sleep Mode (SM) 控制优化问题**，旨在在保证服务质量（QoS）的前提下最大化 **Energy Efficiency (EE)**。具体挑战包括：
- 需要满足两类用户的长期时间平均约束：
  - **Deadline-constrained traffic** 用户的 **packet drop rate** 上限；
  - **Constant-rate users** 的 **minimum throughput** 下限。
- 这些约束具有 **非马尔可夫性（non-Markovian）**，即当前决策的影响依赖于历史状态（如累计丢包率、吞吐量），传统强化学习难以建模。

### 🚀 提出的新方法/新思路
提出将 **Reinforcement Learning (RL)** 与 **Reward Machines (RMs)** 结合，构建一个 **MDP with Reward Machines (MDPRM)** 框架来处理非马尔可夫奖励结构：
- 使用两个独立的 **Reward Machine** 分别跟踪：
  - 丢包率违反程度 `[D]`
  - 吞吐量违反程度 `[u]`
- RM 通过有限状态自动机显式维护历史信息，使原本非马尔可夫的问题转化为等效的马尔可夫问题。
- 最终奖励函数为：  
  $$
  R_{\text{RM}} = \text{EE} + r_d + r_m
  $$
  其中 $r_d$ 和 $r_m$ 是来自两个 RM 的非马尔可夫奖励项。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | 本论文优势 |
|------|--------|-----------|
| **Lyapunov Optimisation** | 需每时隙求解复杂优化问题，扩展性差；对大规模动作空间不适用 | 不依赖系统模型，支持在线学习，更适合高维控制 |
| **CMDP / Lagrangian 方法** | 政策无记忆性，无法捕捉长期依赖 | RM 显式建模历史，能更好平衡短期节能与长期QoS |
| **标准 RL（Markovian Reward）** | 奖励仅基于即时状态，忽略累积效应 | 引入抽象状态记忆，显著提升长期约束满足能力 |

> ✅ **核心创新**：首次将 **Reward Machines** 应用于移动网络节能控制，有效解决了 **time-averaged QoS constraints** 的非马尔可夫建模难题。

---

## 2. 核心实验方法和设置

### 🧪 数据集与仿真环境
- **未使用真实数据集**，而是基于自研的 **系统级仿真工具** 构建虚拟网络环境。
- 包含以下关键组件：
  - 简化的 **ray-tracing 传播模型** 计算路径增益；
  - 多个 RUs 支持多种 SM（共4种，从 71μs 到 1s）；
  - 用户分布、信道条件、流量负载动态变化。

### ⚙️ 实验设置
| 参数 | 设置 |
|------|------|
| RUs 数量 | 4 |
| Sleep Modes (SMs) | 4 种（SM1~SM4），不同持续时间和唤醒延迟 |
| 用户数量 | - Deadline-constrained users $N_d$: 4–5<br>- Constant-rate users $N_m$: 10–60 |
| 流量负载 | 0.1–0.2 Mbps 均匀分布 |
| 信道模型 | Gilbert-Elliot 模型（Good/Bad 两种状态） |
| 学习算法 | **TD3**（Twin Delayed Deep Deterministic Policy Gradient） |
| 网络结构 | Actor/Critic 均为两层全连接（400, 300 neurons），ReLU激活 |
| 动作空间 | $(H+1)^G = 5^4 = 625$，视为连续空间进行映射 |
| 训练配置 | Stable-Baselines3，默认 MlpPolicy；训练5000轮，每轮30步 |

### 📊 评估指标
1. **Energy Efficiency (EE)**：相对能耗节省比例
2. **Power Consumption**：总功耗趋势
3. **Constraint Satisfaction**：
   - 平均 packet drop rate vs. 允许阈值（0.1%）
   - 平均 throughput vs. 要求下限（35 Mbps）
4. **Policy Behavior Analysis**：
   - Power cycling 频率
   - 各 SM 使用分布

### 🆚 基线方法对比
共比较四种奖励设计下的 TD3 表现：
1. **Deep RM ($L=100$)**：深层 Reward Machine，精细粒度历史追踪
2. **Shallow RM ($L=10$)**：浅层 RM，粗略记忆
3. **Markovian Reward**：直接使用即时丢包/吞吐偏差作为惩罚（公式7）
4. **Lagrangian-Optimised (LO) Reward**：引入 Lagrange multiplier 的经典约束优化方法（公式10）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能表现（见 Fig. 2）

| 方法 | EE 表现 | QoS 满足情况 | 综合表现 |
|------|--------|-------------|---------|
| **Deep RM ($L=100$)** | ✅ **最高 EE**（接近 0.9） | 在约束边界附近运行，轻微波动但总体达标 | **最优权衡** |
| **Shallow RM ($L=10$)** | 中等 EE（约 0.75） | 更保守，远离约束边界 | 过于谨慎，牺牲节能潜力 |
| **Markovian Reward** | 较低 EE（约 0.6） | 明显低于目标吞吐，丢包较多 | 缺乏长期视角 |
| **Lagrangian LO** | EE 约 0.7，优于 Markovian | 可满足约束，但响应较慢 | 性能次优 |

> 💡 图2显示：**Deep RM 在几乎不违反QoS的前提下实现了最高的能量效率**，证明其能更智能地利用“违规预算”（violation budget）换取长期节能。

### 🔬 消融实验分析（Fig. 3–4）

#### Fig. 3: Power Cycling 与 SM 分布
- **Deep RM Agent**：
  - 最频繁切换 SM → 高适应性策略
  - 主要使用 **SM4（最长睡眠模式）**，但也灵活调用短时SM应对突发需求
- **Markovian Agent**：
  - 切换最少 → 策略僵化
  - 几乎只在 “active” 和 “SM4” 之间二选一 → 缺乏中间调节能力

#### Fig. 4: SM 使用分布热力图
- Deep RM 展现出丰富的多模式行为，表明其学会了根据场景动态选择最佳睡眠深度。
- Markovian 与 LO 方法表现出明显的双峰行为，缺乏细粒度调控。

> ✅ **消融结论**：RM 的深度 $L$ 是关键超参数。更深的 RM 提供更强的历史记忆能力，从而支持更精细、更高效的控制策略。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Reward Machines 成功建模了非马尔可夫 QoS 约束**，使得 RL 能够学习兼顾 **长期服务质量** 与 **即时节能** 的策略。
2. **RM 深度 $L$ 决定了策略的灵活性与性能上限**：更深的 RM（如 $L=100$）允许代理积累更丰富的历史信息，实现更优的能量-QoS 权衡。
3. 所提方法在多样化流量和动态无线环境中表现出良好的鲁棒性和适应性，适合下一代移动网络的智能化节能管理。

### ⚠️ 方法的局限性
1. **RM 设计依赖人工先验知识**：需要预先定义命题符号（如 `[D]`, `[u]`）和状态转移逻辑，自动化构造 RM 仍具挑战。
2. **计算开销随 RM 深度增加而上升**：过深的 RM 会扩大状态空间，影响学习效率。
3. **仿真实验尚未部署于真实网络**：实际部署需考虑信令开销、控制延迟等问题。

### 🔮 未来工作方向
1. 探索 **自动学习 Reward Machine 结构** 的方法（例如结合 LTL 或神经符号学习）。
2. 将框架扩展至 **多基站协同睡眠控制** 场景，解决更大规模网络中的分布式决策问题。
3. 引入 **因果推理机制** 提升策略可解释性，增强运营商信任（呼应 [16] 中关于 explainable RL 的讨论）。
4. 结合 **semantic communication** 或 **AI-native networking** 技术，进一步优化端到端资源利用。

---

## ✅ 总结一句话
> 本文提出了一种基于 **Reward Machines** 的新型 RL 框架，成功解决了移动网络中由 **time-averaged QoS constraints** 导致的非马尔可夫决策难题，在保障服务质量的同时显著提升了能源效率，为 6G 网络的智能节能提供了可行路径。

</details>

---

### 6. [Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving](https://arxiv.org/abs/2604.08075)

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

# **论文总结：Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前生产环境中的 vLLM 部署通常采用**同构配置**（homogeneous provisioning），即所有实例均按最大上下文长度（如 `max_model_len=64K`）进行资源配置。然而，实际请求中 **80–95% 是短文本请求**（多数在 2K–8K tokens 内），导致：
- **KV-cache 资源严重浪费**：每个序列槽位预留 64K tokens，但短请求仅使用 <5%，造成高达 4–8× 的吞吐容量损失。
- **可靠性问题频发**：高内存占用引发 OOM 崩溃、preemption（抢占）、请求拒绝和 **head-of-line blocking**（长请求阻塞短请求）。

这些问题的根本原因是 **configuration-traffic mismatch**：静态资源配置与动态流量分布不匹配。

---

### **提出了什么新方法或新思路**
作者提出 **Dual-Pool Token-Budget Routing**（双池令牌预算路由），一种轻量级、跨实例的调度机制，核心思想如下：

1. **双池架构**：
   - 将同构 vLLM 集群划分为两个专用池：
     - **Short Pool**：低 `Cmax`（如 8K），高并发（128 seq/GPU），高吞吐。
     - **Long Pool**：高 `Cmax`（如 64K），低并发（16 seq/GPU），处理长上下文请求。
   - 请求根据其 **总 token 预算**（`Ltotal = Lin + Lout`）被路由到合适池。

2. **Self-Calibrating Token Estimation**（自校准令牌估算）：
   - 不依赖 tokenizer，通过 **per-category 字节-令牌比率**（bytes-per-token）在线学习估算 `Ltotal`。
   - 使用 **指数移动平均**（EMA）从 `usage.prompt_tokens` 反馈中持续更新比率。
   - 引入 **保守估计策略**：`c_route = c_k - γσ_k`，偏向将边界请求分发至 Long Pool，避免因误判导致 preemption。

3. **负载感知溢出机制**（Load-aware Spillover）：
   - 当首选池过载时，临时将请求重定向至备用池，保障 SLO，同时不影响稳态效率。

4. **闭式成本模型**（Closed-form Cost Model）：
   - 推导公式：  
     $$
     \text{Savings} = \alpha \left(1 - \frac{1}{p}\right)
     $$
     其中 $\alpha$ 为短请求占比，$p$ 为短池相对吞吐增益。该模型可在部署前预测收益。

---

### **相比现有方法的优势**
| 维度 | 现有方法（如 Chunked Prefill） | 本文方法 |
|------|-------------------------------|--------|
| **优化层级** | 单 GPU 内部执行优化（如 PagedAttention） | **集群级资源分配优化**，跨实例协同 |
| **KV-cache 利用** | 仍为完整序列分配内存，无法提升并发 | **按需配置池大小**，显著提高短请求并发能力 |
| **是否需要 tokenizer** | 多数需 tokenizer 进行 token 数预估 | **无需 tokenizer**，通过字节长度 + EMA 估算，适用于多模型异构后端 |
| **兼容性** | 可独立存在 | **完全兼容** PagedAttention、continuous batching、prefill-decode disaggregation 等已有优化 |
| **开销** | 中等调度开销 | **O(1) 调度开销**，延迟可忽略 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Azure LLM Inference Dataset**：真实生产轨迹，100K 请求，泊松到达。
  - 特征：高度偏斜，80% 请求 < 2K tokens，尾部长达 64K。
- **LMSYS-Chat-1M**：真实对话数据子集。
  - 平均输入长度 `Lin = 69.5`，平均输出长度 `Lout = 214.5`，更集中于短请求。

---

### **实验设置**
- **模型**：Llama-3-70B-Instruct（BF16），部分实验扩展至 Qwen3-235B-A22B。
- **硬件**：NVIDIA A100-80GB（TP=2），部分案例模拟 AMD MI300X（192GB HBM）。
- **仿真器**：基于 Vidur 构建离散事件模拟器，建模 prefill/decode 阶段、KV-cache 分配、批处理行为和排队动态。
- **池配置**：

| Pool | Cmax | Nseq/GPU | Batch Size | Throughput (req/s/inst) |
|------|------|----------|------------|--------------------------|
| Homogeneous | 65K | 16 | 8K | 2.8 |
| Short Pool (Ps) | 8K | 128 | 16K | 11.2 |
| Long Pool (Pt) | 65K | 16 | 8K | 2.8 |

- **阈值设置**：`Bshort = 8192` tokens。
- **SLO 目标**：
  - P99 TTFT ≤ 2s
  - P99 TPOT ≤ 80ms

---

### **基线方法对比**
- **Homogeneous**：单池 + round-robin 路由，代表标准部署方式。
- **Token-budget Routing**：本文方法，启用 load-aware spillover。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **成本降低（GPU 使用量）**
| 数据集 | 方法 | GPU 数量 | 节省比例 |
|-------|------|---------|--------|
| Azure | Homogeneous | 358 | — |
| Azure | Token-budget | 208 | **41.9%** |
| LMSYS | Homogeneous | 358 | — |
| LMSYS | Token-budget | 246 | **31.3%** |

- 在 A100 上，**年节省 $2.86M**（按 $2.21/GPU-hr 计算）。
- 在 **Qwen3-235B on MI300X** 场景下（10,000 req/s）：
  - Homogeneous：1,576 GPUs → $50.6M/年
  - Token-budget：1,096 GPUs → **$35.2M/年**，**节省 $15.4M/年**

#### ✅ **可靠性提升**
| 指标 | Homogeneous | Token-budget | 改善倍数 |
|------|-------------|--------------|---------|
| Preemption Rate (Azure, 90% util) | 47.3‰ | **8.7‰** | ↓ **5.4×** |
| OOM Events/hr | 2.1 | 0.4 | ↓ **5.3×** |
| Success Rate | 99.69% | **99.95%** | ↑ |

> **原因分析**：
> - 短池几乎零失败：因请求严格满足 `Ltotal ≤ Cmax`，且利用率远低于上限（~115/128）。
> - 长池压力减轻：80% 流量被分流，降低并发竞争。

#### ✅ **延迟改善**
| 指标 | Homogeneous | Token-budget | 改善幅度 |
|------|-------------|--------------|---------|
| P50 TTFT | 0.42s | **0.28s** | ↓ **33%** |
| P99 TTFT | 1.82s | **1.71s** | ↓ **6%** |
| P50 TPOT | 28ms | **25ms** | ↓ **11%** |
| P99 TPOT | 67ms | **62ms** | ↓ **7%** |

> **原因**：
> - P50 改善主因是短池高并发消除排队延迟。
> - P99 改善有限，因最长请求仍在长池处理。
> - TPOT 改善源于短池更小 KV footprint，支持更大 decode batch。

#### ✅ **消融实验与敏感性分析**
- **Calibration 收敛性**（Table 5）：
  - 各类别（English, Code, CJK）在约 50 次反馈后收敛，误差 < 3.5%。
  - 保守估计使误路由率从全局静态值 4.1% 降至 **<1%**，尤其对 CJK 文本效果显著。
- **阈值鲁棒性**（Figure 6）：
  - `Bshort ∈ [4K, 16K]` 区间内均可实现 >80% 最优收益。
  - 默认值 `8192` 表现稳健，无需精细调参。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Configuration-traffic mismatch 是 LLM 服务中成本与可靠性问题的共同根源**，而 dual-pool routing 可一次性解决两者。
2. **Token-budget routing 显著降低 GPU 成本（31–42%）**，且收益随规模放大而稳定（scale-invariant）。
3. **无需 tokenizer 的在线 token 估算机制实用性强**，尤其适合多模型、异构部署场景。
4. **闭式成本模型可用于部署前 ROI 评估**，帮助团队快速判断是否值得引入该方案。
5. **方法与现有优化正交**，可在每个池内部继续应用 PagedAttention、speculative decoding 等技术，实现叠加增益。

---

### **方法的局限性**
1. **仅划分两个池**：虽然增加更多池（如 4K/16K/64K）理论上可进一步优化，但边际收益极小（~2%），却显著增加运维复杂度。
2. **依赖流量分布稳定性**：若短/长请求比例剧烈波动，可能影响收益；但可通过 adaptive threshold 扩展缓解。
3. **未处理极端长上下文**（>64K）：仍受限于模型最大上下文窗口。
4. **假设请求长度可预知**：依赖 `max_output_tokens` 参数，若用户未指定，则需合理默认值或启发式估计。

---

### **未来工作方向**
1. **自适应阈值调节**：利用运行时信号（如 preemption rate、queue depth）自动调整 `Bshort`，实现全自动优化。
2. **轻量级 prompt 压缩**：对边界请求进行压缩后再送入短池，扩大高效处理范围。
3. **与 disaggregation 技术深度集成**：如将 prefill-decode 拆分与 dual-pool 结合，进一步解耦资源瓶颈。
4. **支持动态弹性扩缩容**：结合 SageServe 类系统，在地理分布式环境中实现全局最优调度。

---

> **总结一句话**：  
> **Dual-pool token-budget routing 以极低开销（O(1)）打破了“一刀切”的资源配置模式，实现了成本、可靠性和延迟的三重提升，是迈向自优化 LLM serving 系统的关键一步。**

</details>

---

### 7. [Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](https://arxiv.org/abs/2604.07399)

**Authors**: Wonseon Lim, Jaesung Lee, Dae-Won Kim  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.07399v1  

#### Abstract
Continual learning (CL) on edge devices requires not only high accuracy but also training-time efficiency to support on-device adaptation under strict memory and computational constraints. While prompt-based continual learning (PCL) is parameter-efficient and achieves competitive accuracy, prior wor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在边缘设备（edge devices）上进行 **Continual Learning (CL)** 面临严峻挑战，尤其是在**训练阶段的内存占用和计算开销**方面。尽管现有的 **Prompt-based Continual Learning (PCL)** 方法在准确率和参数效率上表现良好，但它们通常忽视了在资源受限设备上的**训练时效率**（如峰值内存、训练时间和能耗），导致难以部署。

此外，直接应用通用的 **token reduction** 技术（如 ToMe、PatchDropout）会无差别地丢弃图像 patch，可能移除任务相关的关键信息，从而显著降低准确率。

### 提出的新方法与思路
本文提出 **CPS-Prompt**，一种面向边缘设备的高效 PCL 框架，其核心由两个模块组成：

- **Critical Patch Sampling (CPS)**  
  利用冻结的 query encoder 在最终 Transformer block 中提取的 **attention map** 和 **value activation**，计算每个 patch 的“关键分数”（critical score），并据此采样保留最相关的 patch。该过程是 **task-aware** 的，确保稀疏化不会破坏语义完整性。

- **Decoupled Prompt and Classifier Training (DPCT)**  
  将训练分为两个阶段：
  1. **Prompt Training Phase**：使用 CPS 选出的 **sparse patch 输入** 更新 prompt 参数；
  2. **Classifier Training Phase**：冻结 prompt，仅用 **full patch 输入** 微调分类器。  
  这种解耦策略缓解了训练与推理之间的表示不匹配问题，并减少了反向传播开销。

### 相比现有方法的优势
- **更高的训练效率**：相比 CODA-Prompt，峰值内存、训练时间、能耗均降低约 **1.6×**。
- **更好的准确率-效率权衡**：准确率仅比最先进的 C-Prompt 平均低 **2%**，但资源消耗远低于后者（内存少 4.3×，训练时间短 3.1×）。
- **优于传统 token reduction 方法**：在相同压缩比下，CPS-Prompt 显著优于 ToMe 和 PatchDropout (PD)，尤其在高稀疏度下仍保持鲁棒性。
- **真实边缘硬件验证**：在 Jetson Orin Nano 上实测验证，证明其适用于实际边缘场景。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CIFAR-100**：小尺度图像分类数据集，划分为 10 个增量任务。
- **ImageNet-R**：ImageNet 的子集，用于鲁棒性评估，同样划分为 10 个任务。
- **CUB-200**：鸟类细粒度分类数据集，也按类增量方式划分成 10 个任务。

### 实验设置与评估指标
- **骨干网络**：ViT-Tiny/16（适合边缘部署）
- **预训练权重**：ImageNet-21K 初始化，ImageNet-1K 微调
- **优化器**：Adam，batch size = 16，学习率 0.001，cosine 衰减
- **训练轮数**：ImageNet-R 为 50 epochs，其余为 20 epochs
- **patch reduction ratio**：统一设为 0.4（即保留 60% 的 patch）

#### 评估指标
| 类别 | 指标 |
|------|------|
| **准确性** | Average Accuracy (ACC↑), Forgetting (FGT↓) |
| **训练效率** | Peak GPU Memory Usage (MB), Training Time (s/task), Energy Consumption (J/task) |
| **平台** | 主要结果在 RTX 4090 上训练，在 **Jetson Orin Nano** 上测量效率 |

### 基线方法对比
- **传统 CL 方法**：SGD, LwF, ER
- **PCL 方法**：L2P, DualPrompt, CODA-Prompt, C-Prompt, OS-Prompt / OS-Prompt++
- **Token Reduction 方法**：ToMe, PatchDropout (PD)

---

## 3. 主要实验结果和性能指标

### 关键性能数据（综合三数据集平均）

| 方法 | 准确率 (ACC) | 峰值内存 | 训练时间 | 能耗 | 相对 CODA-Prompt 提升 |
|------|-------------|----------|----------|------|------------------------|
| **C-Prompt** | **最高** (~68%) | ~1100 MB | ~2600 s | ~3000 J | ❌ 不适用（资源过高） |
| **CODA-Prompt** | 高 (~67%) | ~700 MB | ~1800 s | ~1700 J | 基准 |
| **CPS-Prompt (Ours)** | **仅低 2%** | **~440 MB (-1.6×)** | **~1200 s (-1.5×)** | **~1100 J (-1.6×)** | ✅ 全面提升效率 |
| **OS-Prompt** | 略低 | ~700 MB | ~1300 s | ~1200 J | ✅ 内存更低，速度更快 |

> 注：以上数值为基于图示与表格的近似汇总。

### 与其他 Token Reduction 方法对比（Fig. 4 & 5）
- 在不同 memory reduction ratio 下，**CPS-Prompt 始终取得最佳 ACC-memory 权衡**。
- 当内存减少超过 60%，**ToMe 准确率急剧下降**，而 CPS-Prompt 仍能维持 >90% 的 baseline 准确率。
- **PD 表现稳定但次优**，且训练时间高于 CPS-Prompt。
- **CPS-Prompt 在所有稀疏度下训练时间最短**，得益于 DPCT 减少反向传播负担。

### 消融实验结果（Table 2, ImageNet-R, reduction ratio=0.5）

| 方法配置 | ACC (%) | Memory | Train Time |
|---------|--------|--------|------------|
| CODA-Prompt (baseline) | 50.24 | 440 MB | 1,788 s |
| w/ PD (random patch) | 45.32 | 253 MB | 1,388 s |
| w/ CPS (our sampling) | **47.16** | 253 MB | 1,389 s |
| w/ PD + DPCT | 47.96 | 253 MB | **1,126 s** |
| w/ CPS + DPCT | **49.28** | 253 MB | **1,126 s** |

> 结论：
- **CPS 比随机 PD 提升 +1.84% ACC**，说明 task-aware 采样有效。
- **DPCT 单独带来约 +2% ACC 回升**，缓解表示失配。
- **DPCT 显著缩短训练时间（-662s）**，因第二阶段无需更新 prompt。

---

## 4. 关键结论和发现

### 主要发现
1. **Task-aware patch selection 是关键**：通过 query encoder 的 attention 和 value 信号指导 patch 保留，可在大幅压缩输入的同时最小化精度损失。
2. **Decoupled training 提升效率与一致性**：将 prompt 与 classifier 分阶段训练，既减少梯度计算量，又对齐推理时的完整输入表示。
3. **CPS 与 DPCT 具有互补性**：联合使用可同时提升准确率和训练效率，单一模块无法达到最优。
4. **控制随机性优于确定性选择**：实验表明，**multinomial sampling（温度=0.1）** 比 top-k 更有利于泛化，尤其在复杂图像上能探索更多语义区域（见 Fig. 8）。
5. **相位比例（phase ratio λ）需平衡**：λ ∈ [0.4, 0.6] 效果最好，过早冻结 prompt 会影响特征适配。

### 方法的局限性
- 当前方法依赖于两阶段 PCL 架构（query + prompt-injected forward），可能不适用于单阶段 prompt 设计（如 OS-Prompt）。
- 对 query encoder 的 attention 机制有较强依赖，若该模块失效或噪声大，CPS 效果可能下降。
- 实验集中在类增量学习（class-incremental），未测试领域增量或其他 CL 场景。

### 未来工作方向
- 探索动态资源调度下的 CPS-Prompt 自适应机制（如根据设备负载调整 reduction ratio）。
- 扩展至更广泛的 CL 场景，如 domain-incremental 或 task-free continual learning。
- 将 CPS 思路迁移到其他视觉 backbone（如 CNN 或 Swin Transformer）中。

---

> ✅ **总体评价**：  
> CPS-Prompt 成功将 **task-aware token sparsity** 与 **decoupled optimization** 结合，为边缘端持续学习提供了新的高效范式，在准确率几乎无损的前提下实现了显著的训练资源节省，具有较强的实用价值和推广潜力。

</details>

---

### 8. [Validated Synthetic Patient Generation for Small Longitudinal Cohorts: Coagulation Dynamics Across Pregnancy](https://arxiv.org/abs/2604.07557)

**Authors**: Jeffrey D. Varner, Maria Cristina Bravo, Carole McBride, Thomas Orfeo, Ira Bernstein  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.07557v1  

#### Abstract
Small longitudinal clinical cohorts, common in maternal health, rare diseases, and early-phase trials, limit computational modeling: too few patients to train reliable models, yet too costly and slow to expand through additional enrollment. We present multiplicity-weighted Stochastic Attention (SA),...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Validated Synthetic Patient Generation for Small Longitudinal Cohorts: Coagulation Dynamics Across Pregnancy

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究针对**小样本纵向临床队列**（如妊娠、罕见病、早期临床试验）中数据稀缺导致的计算建模瓶颈问题。这类队列通常患者数量少（n < p，即样本数小于特征数），难以支持可靠的统计分析或机器学习模型训练，且扩大样本成本高昂、耗时长。

### 提出的新方法
作者提出了一种名为 **Multiplicity-weighted Stochastic Attention (SA)** 的生成框架，其核心思想基于**现代Hopfield网络理论**，将真实患者的多维表型作为“记忆模式”嵌入到一个连续的能量景观中，并通过**Langevin动力学**生成新的合成患者。

#### 创新点：
- **几何保持生成**：SA不拟合参数化分布，而是直接在原始数据的PCA降维子空间中操作，保留了小队列的低秩几何结构，避免了传统方法因 `n < p` 导致的协方差矩阵奇异问题。
- **条件生成能力**：引入**每模式多重性权重（multiplicity weighting）**，可在推理阶段动态放大特定临床亚群（如PCOS、preeclampsia等罕见组），而无需重新训练模型。
- **无监督、免训练调节**：通过调整softmax中的log-r权重实现从无条件生成到目标子群放大的平滑过渡。

### 相比现有方法的优势
| 方法 | 局限性 | SA的优势 |
|------|--------|---------|
| **Multivariate Normal (MVN)** | 需要正则化处理奇异协方差；假设高斯分布和线性相关；无法进行条件生成 | 不依赖完整协方差估计；保留跨时间点协变结构；可定向放大稀有亚群 |
| **GAN / VAE (如CTGAN, TVAE)** | 在极小样本下易发生mode collapse；需要大量训练数据；TV AE只能按单次访问行建模，丢失纵向依赖 | 在n=23时仍能稳定生成多样本；利用Hopfield注意力机制避免崩溃；显式建模完整纵向轨迹 |

---

## 2. 核心实验方法和设置

### 数据集
- 来源：一项前瞻性妊娠研究，经伦理批准并获得知情同意。
- 规模：**K = 23 名具有完整三访数据的孕妇**。
- 时间点：
  - Visit 1 (V1): 孕前基线（卵泡期）
  - Visit 2 (V2): 第一孕期末
  - Visit 3 (V3): 第三孕期中期
- 特征维度：每访次 **72个生物标志物**，共 **216维纵向向量**（拼接三个访次）。
- 类别覆盖：包含罕见亚群——**PCOS (n=3)** 和 **发展为子痫前期者 (Developed PE, n=5)**。

### 实验设置与评估指标
采用四级验证框架：

| 层级 | 评估内容 | 指标 |
|------|----------|------|
| **Level 1: Marginal Plausibility** | 单变量分布保真度 | Mean Relative Error (MRE)，标准差、生理关系可视化 |
| **Level 2: Cross-Visit Covariance Structure** | 跨时间点联合结构保持 | 全局Pearson相关矩阵比较、PCA投影分布、特征值谱分析 |
| **Level 3: Conditional Generation** | 罕见亚群放大能力 | Bootstrap Mann-Whitney检验（p > 0.05比例）、条件均值一致性 |
| **Level 4: Mechanistic Consistency** | 生物学合理性验证 | 使用独立的**BZ2012 ODE模型**模拟凝血酶生成，比较预测/实测比值分布（cloud overlap, KS test） |

### 基线方法对比
- **MVN**：对拼接后的216维数据拟合带Ledoit-Wolf正则化的多元正态分布。
- **Deep Models**：
  - **CTGAN**：在每访次90条记录上训练（23×3），测试不同epoch表现。
  - **TVAE**：同上设置，用于对比边缘保真度。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Level 1: 边缘保真度
- **总体MRE中位数仅为1.2%**（95% CI: 1.0–1.6%）
- 216个“特征-访次”组合中，**89%的MRE < 5%**
- 示例关键因子MRE：
  - Factor II: < 2%
  - Fibrinogen: < 1%
  - Antithrombin: < 3%

#### ✅ Level 2: 跨访次协方差结构保持
- SA生成的相关矩阵**完美保留了块状结构**（block structure），尤其是跨访次依赖（off-diagonal blocks）。
- **PCA投影显示**：SA合成患者紧密围绕真实患者云团，而MVN因正则化引入虚假方差，在V2/V3显著扩散。
- 特征值谱分析表明：MVN在超过rank=22的维度人为抬升特征值，SA则通过PCA截断（保留至第18主成分）自然维持低秩结构。

#### ✅ Level 3: 条件生成（罕见亚群放大）
- 成功以 `p ≈ 26.7` 放大PCOS组（仅3例）至100例合成患者。
- Bootstrap Mann-Whitney检验结果显示：
  - **24个“特征-条件”组合中，20个（83%）在≥90%重复中无法区分**（p > 0.05）
  - 中位“不可区分率”达 **98.6%**
- 生理特征正确保留：
  - PCOS: ↑Factor VIII, ↑vWF
  - PE: ↑α2-AP, ↑Factor IX

#### ✅ Level 4: 机械一致性验证（Mechanistic Validation）
使用**BZ2012 ODE模型**输入患者因子水平，预测TGA参数（lagtime, peak, ETP等）：

| 指标 | 结果 |
|------|------|
| **Cloud Overlap**（合成 vs 真实比率分布重叠） | **86–93%**（TF-only）；**89–93%**（TF+TM） |
| **Kolmogorov-Smirnov 检验** | 所有5项TGA特征 p > 0.30（最高p=0.84），**无法区分两组** |

> 注：系统偏差（如lagtime被低估）在真实与合成患者间一致，说明是ODE模型本身偏差，非合成数据问题。

#### ✅ 下游实用性测试（Downstream Utility）
- 分别用**真实V1患者**和**合成V1患者**校准BZ2012模型。
- 在未见过的**真实V2/V3患者**上测试预测误差：
  - 合成校准模型表现**持平甚至略优**于真实校准模型
  - 总体中位相对误差降低 **6%**（ratio = 0.94×）

---

## 4. 关键结论和发现

### 主要发现
1. **SA可在极小纵向队列（n=23）上生成统计与生物学双重可信的合成患者**，其合成数据在边际分布、协方差结构、罕见亚群特征及机制模型响应方面均与真实数据无异。
2. **Multiplicity-weighted SA实现了无需重训练的条件生成**，为罕见疾病研究提供了“虚拟扩增”工具。
3. **下游任务验证表明，完全基于合成数据训练的机制模型具备泛化能力**，预测真实患者结局效果与真实数据训练相当。
4. **小队列研究的瓶颈可能正在从“样本量不足”转向“表型深度不足”** —— 只要少数高质量纵向数据，结合SA即可支撑复杂建模。

### 方法局限性
- **尾部压缩效应**：由于PCA降维和有限范数采样，极端值（distribution tails）略有压缩，影响极高/低风险个体建模。
- **线性假设**：当前使用PCA处理线性流形，未考虑非线性临床关系。
- **未验证临床结局关联**：目前验证集中于生物标志物和机制模型输出，尚未链接到真实分娩结局或并发症事件。
- **ODE模型自身偏差**：BZ2012在TF+TM条件下对protein C通路存在系统性过校正，但该偏差在真实与合成间一致，不影响比较有效性。

### 未来工作方向
- 将SA扩展至其他领域（如肿瘤生长、儿科发育、药物代谢PK/PD模型）。
- 开发非线性版本（如使用autoencoder替代PCA）以更好捕捉复杂表型流形。
- 构建端到端的“合成数据+机制模型”pipeline用于假说生成与试验设计优化。
- 探索联邦学习场景下的多中心小队列合成数据共享框架。

---

> **一句话总结**：  
> 本文提出的 **multiplicity-weighted Stochastic Attention (SA)** 方法，首次实现了从小规模纵向妊娠队列（n=23）中生成经过多层次验证（统计+机制）的合成患者，不仅解决了小样本建模难题，还赋予了定向放大罕见亚群的能力，为罕见病与母胎医学研究开辟了新路径。

</details>

---

### 9. [DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification](https://arxiv.org/abs/2604.07622)

**Authors**: Ziyi Wang, Siva Rajesh Kasa, Ankith M S, Santhosh Kumar Kasa, Jiaru Zou, Sumit Negi, Ruqi Zhang, Nan Jiang, Qifan Song  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.07622v1  

#### Abstract
Speculative decoding is an effective technique for accelerating large language model inference by drafting multiple tokens in parallel. In practice, its speedup is often bottlenecked by a rigid verification step that strictly enforces the accepted token distribution to exactly match the target model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
传统的 **speculative decoding (SD)** 通过一个小的 draft model 并行生成多个候选 token，并由更大的 target model 进行验证，从而加速大语言模型（LLM）的推理过程。然而，其性能受限于一个**刚性的验证机制（rigid verification）**，该机制严格要求接受的 token 分布必须与 target model 完全一致。

这种严格的匹配导致许多语义合理、可能导向正确答案的 draft token 被拒绝，从而：
- 降低 **acceptance rate**
- 限制了实际的 **wall-clock speedup**
- 形成效率瓶颈

### 🚀 提出的新方法：**DIVERSED**
作者提出 **DIVERSED**（Dynamic VErification RElaxed SpEculative Decoding），一种**动态集成验证框架**，核心思想是：
- 引入一个可学习的 **ensemble-based verifier**
- 在每个时间步 $t$，根据上下文 $x_{\leq t-1}$ 动态地融合 draft model 和 target model 的输出分布：
  $$
  v_t(x) = w_t \cdot p_t(x) + (1 - w_t) \cdot q_t(x)
  $$
  其中权重 $w_t = f_\theta(h_t^{\text{draft}}, h_t^{\text{target}})$ 是一个轻量级神经网络（ensemble head），基于两个模型的隐藏状态计算得出。

这使得验证规则从“硬性对齐”变为“软性、自适应”的决策，允许在不影响最终任务质量的前提下接受更多高质量 draft token。

### 🔍 相比现有方法的优势
| 方法 | 是否动态 | 是否可训练 | 是否保持质量 |
|------|--------|-----------|-------------|
| Standard SD (Lossless) | ❌ | ❌ | ✅ |
| Static Ensemble | ❌（固定权重） | ❌ | ⚠️ 随权重退化 |
| Judge Decoding / Lossy SD | ❌ | ❌ | ❌ 易降质 |
| **DIVERSED (Ours)** | ✅（context-dependent） | ✅（RL训练） | ✅ |

- **灵活性更强**：支持任务和上下文相关的松弛策略，而非全局静态参数。
- **效率更高**：显著提升 acceptance rate，带来端到端更低延迟。
- **质量可控**：通过强化学习优化目标函数，在提高 acceptance 的同时维持甚至提升任务准确率。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
覆盖多种典型 NLP 任务，确保泛化性：
- **GSM8K**：数学推理（Mathematical Reasoning）
- **CNNDM**：新闻摘要（News Summarization）
- **XSum**：极端摘要（Extreme Summarization）
- **MBPP**：Python 编程任务（Code Generation）

### ⚙️ 实验设置
- **Target/Draft Model Pairs**：
  1. `Llama-3.1-8B-Instruct` / `Llama-3.2-1B-Instruct`
  2. `Qwen3-8B` / `Qwen3-0.6B`
  3. `Gemma-3-12B-It` / `Gemma-3-4B-It`

- **Draft Length $N$**：默认为 5，部分实验测试 $N=3,7$

- **温度设置**：$T=0$ 和 $T=1.0$ 多组对比

- **训练方式**：
  - 使用 **Reinforcement Learning (REINFORCE++)** 训练 ensemble head
  - 奖励函数 $R(x_{1:T})$ 为序列级奖励（如 GSM8K 中答案是否正确）
  - 正则项鼓励高 acceptance（通过 $1 - \text{TV}(q, v)$ 控制）

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Acceptance Rate (%)** | 成功被接受的 draft token 占比 |
| **Wall-clock Time / Latency** | 实际推理耗时（秒/样本） |
| **Task Accuracy** | 最终任务表现（如 Pass@1, ROUGE-2） |
| **Speedup** | 相对于 autoregressive 推理的速度提升倍数 |
| **Pareto Frontier** | 效率 vs. 质量的权衡曲线 |

### 🆚 基线方法对比
- **Lossless Methods**：
  - Standard SD（Leviathan et al., 2023）
- **Lossy / Adaptive Methods**：
  - SD (Lossy)
  - SpecCascade (Narasimhan et al., 2025)
  - Static Ensemble（本文引入作为基准）

> 注：未比较 Medusa 或 EAGLE，因其修改模型架构，会引入额外变量；DIVERSED 可与其正交结合。

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据汇总（见 Table 1 & Appendix Table 6）

#### 示例：Llama-3.1-8B / Llama-3.2-1B 对比（$T=0$, $N=5$）

| 方法 | Acceptance Rate | GSM8K Acc | CNNDM ROUGE-2 | MBPP Pass@1 |
|------|------------------|------------|----------------|--------------|
| Autoregressive | NA | 80% | 11.29 | 62% |
| SD | 61.53% | 80% | 11.26 | 72.18% |
| Static Ensemble | 82.58% | 79% | 11.45 | 84.67% |
| **DIVERSED (Ours)** | **84.82%** | **80%** | **12.37** | **85.03%** |

> ✅ 在所有任务上，DIVERSED 实现最高 acceptance rate，且任务质量持平或略优。

#### 更广泛结果趋势：
- **平均 acceptance 提升**：相比标准 SD 提升 **28%+**
- **速度增益明显**：acceptance rate 与 wall-clock time 呈强负相关（Figure 4）
- **每轮验证接受 token 数最多**：DIVERSED 平均接受 ~3.9 tokens/round，优于其他方法（Figure 5）

### 🔁 消融实验与关键分析

#### （1）**静态 vs 动态 ensemble 权重**
- Figure 6 显示：随着静态 ensemble 权重 $w$ 增加，acceptance 线性下降。
- 当 $w \to 1$，退化为 lossless SD。
- 表明单一固定权重无法兼顾不同上下文的需求。

#### （2）**跨数据集迁移能力**
- 在 GSM8K 上训练的 DIVERSED 应用于 CNNDM：
  - Acceptance ↑，但 Accuracy ↓
- 结论：最优的松弛策略是 **task-dependent**，不能通用化。

#### （3）**微调 draft model vs DIVERSED**
- 单独 fine-tune draft model 并不能稳定提升 acceptance：
  - 在 Llama 对上有轻微提升
  - 在 Qwen3 上反而下降
- 表明 acceptance 主要取决于 **distributional alignment**，而非 draft model 自身性能。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Acceptance-Quality Trade-off 是 context- and task-dependent**
   - 不同位置、不同类型的任务对 token 错误容忍度不同（如数学题关键步骤不可错）。
   - 固定规则（如 static ensemble）无法达到最优。

2. **DIVERSED 超越静态集成的 Pareto Frontier**
   - 如 Figure 1 所示，DIVERSED 在相同质量下实现更高效率，突破原有理论边界。

3. **Higher Acceptance ⇒ Lower Latency**
   - 实验验证 acceptance rate 与 end-to-end 推理时间高度负相关，提升 acceptance 可直接转化为速度优势。

4. **Verification 设计比 draft model 性能更重要**
   - 微调 draft model 不一定能提升 acceptance；
   - 改进 verification 策略更具性价比。

### ⚠️ 局限性
- **依赖 RL 训练**：需要任务级 reward signal，难以应用于无明确评价标准的任务。
- **增加少量计算开销**：ensemble head 需额外前向传播，虽轻量但仍有一定成本。
- **目前仅适用于 token-level 决策**：尚未扩展至 block-level 或 tree-level speculative 结构。

### 🔮 未来工作方向
1. 将 relaxed verification 扩展到 **block-level 或 tree-level speculative decoding**（如结合 SpecInfer）。
2. 探索 **cross-task transferability**：能否在一个任务上学出通用的动态 verifier？
3. 与 **DISCO、Medusa、EAGLE 等方法结合**，进一步叠加加速效果。
4. 引入更高效的 reward modeling 替代人工标注 reward。

---

## ✅ 总结一句话
> **DIVERSED 通过可学习的动态集成验证器，实现了 context-aware 的松弛验证机制，在几乎不牺牲生成质量的前提下显著提升了 speculative decoding 的 acceptance rate 和推理效率，突破了传统静态方法的 Pareto 边界。**

</details>

---

### 10. [Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](https://arxiv.org/abs/2604.07659)

**Authors**: Mingchen Li, Jiatan Huang, Zonghai Yao, Hong yu  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.07659v1  

#### Abstract
Large language models (LLMs) hold significant promise for healthcare, yet their reliability in high-stakes clinical settings is often compromised by hallucinations and a lack of granular medical context. While Retrieval Augmented Generation (RAG) can mitigate these issues, standard supervised pipeli...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在医疗预测任务中展现出巨大潜力，但在高风险临床场景中面临两大挑战：
- **幻觉（Hallucinations）** 和缺乏细粒度医学上下文；
- 传统 **Retrieval-Augmented Generation (RAG)** 虽能缓解上述问题，但依赖大规模外部知识库的检索，导致计算开销大、延迟高，难以满足实时医疗决策需求。

现有 RAG 方法存在两个瓶颈：
1. 外部知识通过输入 prompt 注入，扩展了上下文长度，增加推理成本；
2. 高质量检索器构建困难，监督式检索需要大量标注数据，而图结构检索则依赖昂贵的路径搜索。

### 提出了什么新方法或新思路
本文提出 **Keys-to-Knowledge (K2K)** 框架，其核心思想是：  
> **将 LLM 内部参数空间作为可检索的知识库，实现无需外部检索的“内部记忆访问”**。

K2K 包含三大模块：
1. **Internal Memory Construction**  
   利用 FFN 层中的 `W1` 权重矩阵作为 key，存储事实知识；对于预训练语料中缺失的领域知识（如医学图谱），采用 **LoRA (Low-Rank Adaptation)** 将其注入模型参数空间。
   
2. **Activation-Guided Probe Construction**  
   构造带有激活引导的 probe query，利用对角化 **Mahalanobis distance** 计算 token 的判别性权重，增强查询向量的区分能力，避免均值池化带来的语义稀释。

3. **Cross-Attentive Reranking**  
   引入跨窗口 **cross-attention** 机制，动态整合从 document-level 和 graph-level 内存中检索到的知识，并进行重排序，提升下游任务的相关性和适应性。

### 相比现有方法的优势
| 维度 | K2K | 传统 RAG |
|------|-----|----------|
| **检索方式** | 内部 key-value 访问（无额外 I/O） | 外部数据库/图谱检索 |
| **延迟** | 极低（仅 O(m) 或 O(mk)） | 高（需遍历海量文档或图结构） |
| **可扩展性** | 不受上下文长度限制 | 受限于最大 context window |
| **知识融合** | 参数级融合，支持端到端训练 | 通常为 pipeline 式处理 |

✅ **优势总结**：
- 显著降低检索延迟，适用于时间敏感型临床应用；
- 免除外部分布式索引系统，部署更轻量；
- 支持多源知识（文本 + 图谱）统一建模与集成。

---

## 2. 核心实验方法和设置

### 使用的数据集
基于两个公开电子健康记录（EHR）数据集进行评估：
- **MIMIC-III**：包含约 4 万住院患者记录；
- **MIMIC-IV**：更新版本，规模更大、结构更完整。

每个患者的病史表示为一系列就诊记录 $ V = \{v_1, ..., v_n\} $，每条就诊包含多个 ICD 编码及其对应的临床描述文本。

#### 任务定义
- **Mortality Prediction**：预测下一次就诊是否死亡；
- **Readmission Prediction**：预测出院后 15 天内是否再入院。

### 实验设置和评估指标

| 设置项 | 描述 |
|--------|------|
| 数据划分 | 按 patient ID 分组，8:1:1 划分 train/dev/test |
| LLM 主干 | BioMistral-7B、Meditron3-Qwen2.5-7B |
| 批大小 | 16 |
| 优化器 | AdamW（lr=2e-5） |
| Chunk size | 固定为 64 |
| Top-k | 根据任务调整（5~20） |

#### 评估指标（四个标准指标）
- **F1-score**
- **Jaccard Similarity**
- **AUPRC**（Area Under Precision-Recall Curve）
- **AUROC**（Area Under ROC Curve）

最终性能以四项指标的算术平均值（Avg）为主要比较依据。

### 基线方法对比
分为三类基线模型：

#### （1）序列模型（Sequential Models）
- GRU, RETAIN, Deepr, AdaCare, StageNet, TCN

#### （2）检索增强模型（Retrieval-Based Models）
- **KARE**：当前 SOTA，结合知识图谱社区检索；
- **Standard RAG**：使用 Contriver 进行密集检索；
- **Prompt-based Retrieval**：指令引导 LLM 自生成医学知识。

#### （3）生成式知识模型
- Prompt-Based Retrieval（同上）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 模型 | Mortality-MIMIC-III (Avg) | Readmission-MIMIC-III (Avg) | Mortality-MIMIC-IV (Avg) | Readmission-MIMIC-IV (Avg) |
|------|----------------------------|------------------------------|----------------------------|-------------------------------|
| **K2K (BioMistral-7B)** | **60.37** ✅ | **57.90** ✅ | **61.42** ✅ | **62.87** ✅ |
| KARE | 53.46 | 53.77 | 59.74 | 61.91 |
| Standard RAG | 58.33 | 55.38 | 59.98 | 61.70 |
| Prompt-based | 55.06 | 54.70 | 60.21 | 57.50 |
| Fine-tuned LLM (w/o retriever) | 58.93 | 57.66 | 59.80 | 57.58 |

📌 **关键观察**：
- 在所有任务和主干模型上，**K2K 均达到 SOTA 性能**；
- 特别是在 **Mortality-MIMIC-IV** 上，F1 提升显著（从 1.33→6.61 @ KARE → K2K）；
- 即使在小样本 MIMIC-III 上，也优于 prompt-based 方法约 **5+ pts** 平均得分。

### 与基线方法的对比结果
- **vs. KARE**：尽管 KARE 使用图结构最短路径增强推理，但仍可能遗漏关键关系；K2K 通过直接访问内部图谱知识（via LoRA），实现更全面的关系建模。
- **vs. Standard RAG**：受限于外部检索效率和噪声干扰，表现不稳定。
- **vs. Prompt-based**：虽能生成有用信息，但推理成本极高（见效率分析）。

### 消融实验结果（Ablation Studies）

#### （1）不同知识来源的影响（Table 3）
| 变体 | Mortality-III (Avg) | Readmission-IV (Avg) |
|------|----------------------|------------------------|
| K2K (full) | **60.37** | **62.87** |
| w/o graph | 58.12 | 60.23 |
| w/o document | 56.84 | 60.56 |

➡️ 移除任一知识源均导致性能下降，证明 **document + graph 双源协同有效**。

#### （2）查询构造策略比较（Table 5）
| 方法 | Mortality-III (Avg) |
|------|----------------------|
| K2K (Mahalanobis-guided) | **60.37** |
| K2K (Euclidean) | 56.82 |
| K2K (Mean Only) | 53.36 |

➡️ **Mahalanobis distance 明显优于 Euclidean 和均值池化**，说明考虑维度方差可提升检索精度。

#### （3）不同 Transformer 层的知识效果（Figure 3）
- 并非越深的层越好；
- 浅层（如 Layer 5, 8, 10）也能提供重要实体/结构信息；
- 表明知识分布在网络各层，应综合利用。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 参数不仅是推理载体，更是可显式访问的知识库**：FFN 中的 key 向量隐式编码了医学知识，可通过设计机制主动检索。
2. ✅ **内部记忆检索可行且高效**：K2K 成功绕过外部检索瓶颈，在保持高性能的同时大幅降低延迟。
3. ✅ **激活信号指导的 probe 查询显著提升检索质量**：引入 Mahalanobis 距离加权，增强了低方差方向的敏感性。
4. ✅ **跨注意力重排序机制有助于多源知识融合**：实现了 context-aware 的动态知识整合。

### 方法的局限性
1. **Layer Selection 固定**：未实现按 query 动态选择最优检索层，可能错过最佳知识位置；
2. **领域泛化待验证**：目前仅在生物医学领域测试，法律、金融等其他知识密集型领域尚需验证；
3. **数据不平衡问题**：真实世界中正负样本极度不均衡（如罕见病），当前框架对此鲁棒性有待加强。

### 未来工作方向
- 设计 **layer-wise adaptive selection mechanism**，根据 query 自动选择最相关层进行检索；
- 探索 **multi-modal extension**，将影像、实验室检测等非文本信息纳入内部记忆；
- 研究 **continual knowledge injection**，支持在线更新 LoRA 适配器以适应新医学发现；
- 在更多领域（如 Legal LLMs）验证 K2K 的通用性。

---

> 📌 **一句话总结**：  
> K2K 开辟了一条全新的“**参数即知识库**”路径，通过 **internal memory retrieval + activation-guided probing + cross-attentive reranking**，实现了高效、精准、低延迟的医疗预测，为 LLM 在高风险场景下的落地提供了有力支撑。

</details>

---

### 11. [LLM-Based Data Generation and Clinical Skills Evaluation for Low-Resource French OSCEs](https://arxiv.org/abs/2604.08126)

**Authors**: Tian Huang, Tom Bourgeade, Irina Illina  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.08126v1  

#### Abstract
Objective Structured Clinical Examinations (OSCEs) are the standard method for assessing medical students' clinical and communication skills through structured patient interviews. In France, however, the organization of training sessions is limited by human and logistical constraints, restricting st...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LLM-Based Data Generation and Clinical Skills Evaluation for Low-Resource French OSCEs 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本研究针对**法语 OSCE（Objective Structured Clinical Examinations）领域中数据稀缺和评估标准异构的问题**。具体挑战包括：
- 法国医学教育中缺乏公开、标注的法语 OSCE 对话转录数据集；
- 法语 OSCE 使用高度场景化（station-specific）、非标准化的评估标准，难以直接套用英语 OSCE 的自动化评估方法；
- 人工组织训练受限于人力与资源，学生实践机会有限。

### 提出了什么新方法或新思路
作者提出了一套**基于 LLM 的合成数据生成与自动评估框架**，其核心创新如下：

#### （1）受控的法语 OSCE 对话生成管道（Controlled Dialogue Generation Pipeline）
- 利用真实 OSCE 场景中的三个结构化文档（Doctor Sheet, Patient Sheet, Evaluation Sheet）作为输入；
- 引入 **LLM 驱动的标准排序机制**（OlAP 或 context-driven），将杂乱的评估标准重排为符合临床访谈逻辑的顺序（Opening → Information Collection → Assessment → Plan）；
- 通过分段生成（segment-wise generation）结合 N 个标准（N=4）逐步构建完整对话；
- 设计 **criteria perturbation 机制**：对部分“叶节点”标准进行扰动，模拟不同水平的学生表现（理想 vs. 次优），增强数据多样性。

#### （2）LLM 辅助的银标签标注框架（LLM-Assisted Silver-Labeling）
- 提出两种可调节严格度的评估模式：
  - **Soft Mode**：只要信息出现在对话中即视为达标（即使由患者主动提及）；
  - **Strict Mode**：要求医生明确询问或确认该信息才视为达标。
- 使用 GPT-4o 自动生成初步标签（含 justification 和 evidence），再经人工审核形成“银标签”（silver labels），提升标注效率与一致性。

#### （3）本地可部署的小型 LLM 评估系统探索
- 聚焦于参数量 ≤32B 的开源模型（如 qwen3-32b, deepseek-r1-32b, qwen3-8b），验证其在隐私敏感医疗教育场景下的可行性；
- 探索辅助工具以增强小型 LLM 表现，包括：
  - **Criterion Decomposition (CD)**：将复合标准（如 A AND B 或 A OR B）拆分为子项分别判断后聚合；
  - **Medical Definitions (MD)**：从 UMLS 注入医学术语定义以帮助理解专业词汇。

### 相比现有方法的优势
| 方面 | 本文优势 |
|------|--------|
| **语言与文化适配性** | 首次专注于**低资源法语 OSCE**，解决英语主导研究无法迁移的问题 |
| **数据生成控制性** | 生成过程受评估标准驱动，保证生成对话与评分体系强关联，利于后续评估任务 |
| **评估灵活性** | 支持 adjustable strictness 的评估模式，更贴近实际教学需求 |
| **实用性与隐私保护** | 强调使用本地部署的中小型 LLM，避免敏感数据外泄，适合真实教育环境应用 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **无真实公开法语 OSCE 文本数据集可用**，因此完全依赖**合成数据**。
- 基于 10 个真实的法语 OSCE 培训场景构建：
  - 共提取 **179 条二元评估标准**（binary criteria）；
  - 每个场景生成两个版本的对话：
    - **Unperturbed Corpus**：理想医生行为（最优执行）；
    - **Perturbed Corpus**：50% 叶节点标准被扰动，模拟较差表现；
  - 最终得到 20 个合成对话（每种各 10 个）。

> ✅ 所有数据均为 LLM 合成，不涉及真实患者或学生数据，符合伦理规范。

### 实验设置和评估指标
#### 评估任务
给定一个完整的对话转录、一条评估标准和任务描述，LLM 输出三部分内容：
1. **Justification**：简要解释判断依据；
2. **Evidence**：引用相关对话片段；
3. **Binary Decision (Done/Not Done)**：是否满足该标准。

#### 评估指标
- 主要指标：**Accuracy**（与银标签的一致率）
- 分别在 **perturbed** 和 **unperturbed** 两个 corpus 上测试；
- 报告多个模型在不同配置下的平均准确率。

#### 基线方法对比
- **Proprietary LLMs（高端参考）**：
  - GPT-4o
  - Claude Sonnet 4
  - Llama-4-Scout
- **Open-source LLMs（主攻目标）**：
  - 中等规模（20–32B）：`qwen3-32b`, `deepseek-r1-32b`, `gpt-oss-20b`, `gemma-3-27b`
  - 轻量级（~8B）：`qwen3-8b`, `llama3.1-8b`, `ministral-8b`

> 所有开源模型均采用量化版本（Q4_K_M），支持本地运行。

#### 提示策略（Prompting Strategies）
| 类型 | 描述 |
|------|------|
| **Zero-shot** | 仅提供任务说明 + 当前标准 + 转录 |
| **Few-shot** | 加入少量带标签示例 |
| **Multi-step** | 第一步提取相关片段 → 第二步基于片段做判断 |

#### 辅助工具
| 工具 | 功能 |
|------|------|
| **CD (Criterion Decomposition)** | 将复合标准拆解为原子项，防止逻辑误判 |
| **MD (Medical Definitions)** | 注入 UMLS 中的法语/英语医学定义，辅助术语理解 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 模型 | Perturbed Accuracy (%) | Unperturbed Accuracy (%) | 是否接近 GPT-4o |
|------|------------------------|----------------------------|------------------|
| **GPT-4o** | ~88.8 | ~90.5 | —— |
| **qwen3-32b** | **90.5** (CD) | 89.9 | ✅ 是 |
| **deepseek-r1-32b** | 86.0 | 88.8 | ✅ 接近 |
| **gpt-oss-20b** | 87.7 | 87.7 | ✅ 接近 |
| **qwen3-8b** | 85.5 | 87.7 | ✅ 轻量级中最佳 |
| **llama3.1-8b** | 50.8 | 51.4 | ❌ 明显落后 |
| **ministral-8b** | 76.5 | 84.4 | ⚠️ 不稳定 |

> 💡 **总体趋势**：多个 ≤32B 参数的开源模型（尤其是 `qwen3-32b`）在 accuracy 上达到甚至超过 GPT-4o 水平（约 90%）。

### 与基线方法的对比结果
- **GPT-4o 并未取得完美成绩**（因部分银标签经人工修正），说明评估本身存在挑战；
- 多个开源模型表现优于或媲美 GPT-4o，表明**大型闭源模型并非必需**；
- 特别是 `qwen3-32b` 在多种设置下表现最稳定，在 perturbed 数据上达到 **90.5% 准确率**（使用 CD 工具）；
- 轻量级模型中 `qwen3-8b` 表现突出，远超同类。

### 消融实验结果

#### （1）提示策略影响
| 策略 | 效果 |
|------|------|
| **Zero-shot** | 性能最好且最稳定，推荐使用 |
| **Few-shot** | 未见明显增益，反而使 `ministral-8b` 下降（可能因上下文过长） |
| **Multi-step** | **全面劣于 zero-shot**，原因：
  - 上下文丢失（只看到片段而非全局）；
  - 错误传播（第一步检索失败则全错） |

#### （2）辅助工具效果
| 工具 | 效果 |
|------|------|
| **CD (Criterion Decomposition)** | ✅ 显著提升 perturbed 数据上的表现（尤其对 `qwen3-32b` 和 `llama3.1-8b`），缓解 OR/AND 逻辑误解问题；但在 unperturbed 数据上作用较小 |
| **MD (Medical Definitions)** | ❌ 无显著提升，甚至轻微下降；推测因多数术语已可理解，且 NER 匹配引入噪声 |

#### （3）真实案例初步验证（Table 3）
使用两个真实教学会话转录（共 38 条标准）进行零样本测试：

| 模型 | Real-case Accuracy |
|------|--------------------|
| **qwen3-32b** | **86.97%** |
| **GPT-4o** | 83.47% |
| **qwen3-8b** | 75.77% |
| **llama3.1-8b** | 37.25% |

➡️ 结果趋势与合成数据一致，说明**合成数据具有一定的生态有效性（ecological validity）**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **中小型开源 LLM 完全可用于法语 OSCE 自动评估**，性能可达 GPT-4o 水平（~90% accuracy），无需依赖昂贵闭源 API；
2. ✅ **可控的合成数据生成 + LLM 辅助银标签** 是解决低资源语言医学教育数据瓶颈的有效路径；
3. ✅ **Criterion Decomposition 工具显著改善复合标准判断错误**，是提升鲁棒性的关键组件；
4. ✅ **Zero-shot 提示优于 multi-step 或 few-shot**，简单有效；
5. ✅ **真实案例趋势与合成数据一致**，初步证明方法具备外推能力。

### 方法的局限性
1. 🔒 **数据理想化严重**：
   - 生成对话结构规整、逻辑连贯，缺少真实学生常见的犹豫、重复、偏离话题等现象；
   - Perturbation 仍属“结构化错误”，不如真实表现复杂。
2. 🧪 **银标签非金标准**：
   - 使用 GPT-4o 生成并由作者审核，未经医学专家正式仲裁，属于“reviewed silver standard”；
   - 存在潜在模型偏见风险（如风格偏好、过度合作性）。
3. 📝 **未考虑非文本因素**：
   - 评估仅基于文本，忽略语气、停顿、肢体语言等重要沟通维度；
   - 也未处理 ASR 转录带来的 disfluency、中断等问题。
4. 🎯 **生成与评估同源偏差**：
   - 所有数据由 GPT-4o 生成，可能导致其他模型对其风格适应更好，造成性能高估。

### 未来工作方向
1. 🔄 **多样化生成器与风格控制**：使用多个 LLM 生成数据，减少单一模型风格偏倚；
2. 👩‍⚕️ **引入医学教育专家参与校准**：共同定义评估严格度、验证自动生成标签；
3. 🧩 **扩展至多模态评估**：整合语音特征、情感分析、非语言行为建模；
4. 📚 **扩大场景覆盖范围**：纳入更多 OSCE 类型（如儿科、精神科）、加入外部操作类任务；
5. 🧠 **深入分析 justification 与 evidence 质量**：不仅看 accuracy，还要评估反馈的 pedagogical value（教学价值）；
6. 🛡️ **开展 bias auditing 与伦理审查**：确保系统公平、透明、可信赖，方可用于真实培训。

---

> 📌 **一句话总结**：  
> 本文展示了在**低资源法语 OSCE 场景下，利用 LLM 生成高质量合成对话，并用中小规模开源模型实现接近 GPT-4 的自动评估性能**，为隐私保护、低成本、可复制的医学教育 AI 工具提供了可行范式。

</details>

---

### 12. [Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference](https://arxiv.org/abs/2604.07472)

**Authors**: Jiaming Cheng, Duong Tung Nguyen  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.07472v1  

#### Abstract
Deploying large language model (LLM) inference at scale requires jointly selecting base models, provisioning heterogeneous GPUs, configuring parallelism, and distributing workloads under tight latency, accuracy, and budget constraints. Exact mixed-integer linear programming (MILP) approaches guarant...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference  
——核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**大规模 LLM 推理服务中的联合优化难题**，即在严格的 **SLO（Service-Level Objective）约束**下，如何同时决策：
- 基础模型选择（foundation model）
- 异构 GPU 资源配置（heterogeneous GPU provisioning）
- 并行策略配置（tensor/pipeline parallelism, TP/PP）
- 工作负载分配（workload routing）

该问题是高度耦合的混合整数规划（MILP）问题，传统精确求解器（如 Gurobi）虽能保证最优性，但计算复杂度随规模指数增长，难以满足实时性需求。

---

### 🚀 提出的新方法与新思路

作者提出两种**约束感知的启发式算法**（constraint-aware heuristics）：

#### （1）Greedy Heuristic (GH)
- 单次遍历式贪心算法，分为两个阶段：
  - **Phase 1**: 覆盖预分配（Coverage pre-allocation），确保每种查询类型至少有一个可行部署路径。
  - **Phase 2**: 按请求率降序逐个处理 query type，进行顺序分配。
- 在每一步都集成三个关键机制以保障可行性。

#### （2）Adaptive Greedy Heuristic (AGH)
- 在 GH 基础上增强，引入三大改进：
  - **Multi-start construction**：尝试多种输入排序（按 arrival rate、error bound、storage footprint 等升/降序 + 随机排列），保留最佳结果。
  - **Relocate-based local search**：对已分配任务尝试迁移到更优配置，提升成本效益。
  - **Consolidation**：将低负载 GPU 上的任务合并到其他活跃配置中，并释放空闲实例，降低租金开销。

---

### 🔍 三大约束感知机制（Constraint-Aware Mechanisms）

这些机制不仅是优化手段，更是**可行性前提**（feasibility prerequisites）：

| 机制 | 功能 |
|------|------|
| **M1: TP-aware feasibility selection** | 对每个 `(query, model, tier)` 组合，仅保留满足内存和延迟 SLO 的最小成本 TP/PP 配置；否则直接丢弃候选。防止不可行部署。 |
| **M2: Cost-per-effective-coverage ranking** | 不按原始成本排序，而是依据单位有效服务能力的成本（cost per effective coverage）来优先级排序，兼顾延迟与误差预算限制下的实际可服务比例。 |
| **M3: TP upgrade for active GPUs** | 若已有激活的 `(j,k)` 配置无法满足新请求的延迟要求，则尝试升级其 TP degree（增加并行度），复用已加载权重，避免重复启动开销。 |

> ⚠️ 消融实验证明：移除 M1 或 M3 会导致严重违反内存或延迟约束，说明它们是**强制性的可行性保障机制**。

---

### ✅ 相比现有方法的优势

| 维度 | 本工作（GH/AGH） | 现有方法（如 Helix [5], Jiang et al. [6]） |
|------|------------------|----------------------------------------|
| **可扩展性** | 子秒级运行时间（<1 秒），支持大规模实例 | MILP 求解器运行时间可达分钟甚至小时级 |
| **联合优化能力** | 同时优化 model selection、GPU provisioning、TP/PP 配置、routing | 多数系统固定部分参数（如先定 parallelism 再 routing） |
| **鲁棒性** | 在 out-of-sample 压力测试下保持稳定成本与可控 SLO violation | 精确解在参数扰动后性能急剧下降 |
| **实用性** | 支持滚动重优化（rolling re-optimization），适应动态负载变化 | 静态部署为主，难以应对波动 |

> 💡 **核心优势总结**：  
> AGH 实现了 **>260× speedup** 于精确求解器，同时接近最优成本，并具备天然鲁棒性，适用于生产环境中的实时调度层。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用基于 **Azure LLM Inference Trace (2025)** 构建的工作负载。
- 包含 **6 类 query types**：
  - Summarization, Code Generation, Translation, Math Solving, Image Generation, Video Generation
- 模型池：**6 个 Llama-3.x 模型**（1B–70B 参数）
- GPU 层级：**10 种异构 tier**（A6000, RTX4090, A100, H100）搭配不同精度（FP16/INT8/INT4）

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| 规划周期 △T | 24 小时 |
| 查询到达率 λ | 1,000 ~ 25,000 queries/hour |
| 延迟 SLO Δ | 1.5 ~ 25 秒 |
| 误差阈值 ε | 2% ~ 8% |
| GPU 租金 | $0.35 ~ $2.50/hour |
| 总预算上限 | $100（基准场景） |
| 存储容量 | 1,000 GB |
| 并行度选项 | TP ∈ {1,2,4,8}, PP ∈ {1,2,4} |

---

### 📈 评估指标

| 指标 | 定义 |
|------|------|
| **Expected Total Cost** | 包括 GPU 租赁、模型存储、token 存储、延迟惩罚、未满足请求惩罚 |
| **SLO Violation Rate (%)** | 出现延迟或误差超限的请求占比 |
| **Runtime (seconds)** | 算法执行时间 |
| **Rolling Re-optimization Performance** | 每 5 分钟重新调度一次，在需求漂移下的累计成本表现 |

---

### 🆚 基线方法对比

| 方法 | 描述 |
|------|------|
| **Exact MILP Solver (DM)** | 使用 Gurobi 求解完整 MILP 问题，作为“最优”参考 |
| **GH** | 本文提出的单通路贪心算法 |
| **AGH** | 本文提出的自适应增强版本 |
| **Static vs Rolling** | 是否允许周期性重优化（如 AGH-5min vs AGH-24h） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 指标 | 结果 |
|------|------|
| **运行时间** | 
| - GH | < 0.01 ~ 0.9 秒（随规模线性增长） |
| - AGH | 0.01 ~ 2.3 秒 |
| - DM（精确求解器） | 从 0.39 秒增至 >600 秒（在 (20,20,20) 规模超时） |
| **加速比** | AGH 相比 DM 实现 **>260× speedup**（大实例下） |
| **成本接近度** | AGH 成本仅比最优解高约 **5–10%**，远优于 GH 在压力下的退化表现 |

---

### 🔁 与基线方法对比结果

#### 在标准压力测试下（1.2× 和 1.5× 延迟/误差膨胀）：

| 方法 | 实际总成本 | SLO 违规率 |
|------|------------|-----------|
| DM（静态） | 显著上升（$793 → $909） | 急剧恶化 |
| GH | $475 ~ $498 | 14.2% → 更高 |
| **AGH（滚动）** | **$434 ~ $474** | **3.7% → 仍可控** |

> ✅ AGH 在高压下仍维持低成本与低违规率，而 DM 因过度追求静态最优导致泛化差。

#### 滚动重优化效果（每 5 分钟 re-optimize）：

| 场景（volatility σ） | AGH-5min vs AGH-24h 节省 | vs DM-24h 节省 |
|---------------------|----------------------------|----------------|
| σ = 0.04 | -17.3% | -45.2% |
| σ = 0.05 | -16.0% | -47.9% |

> 💡 最高节省 **48% 成本**，体现 AGH 动态适应能力的巨大优势。

---

### 🔍 消融实验结果（Ablation Study）

| 配置 | 可行性？ | 成本（美元） | 说明 |
|------|---------|-------------|------|
| **完整 AGH** | ✅ Yes | 89.88 | 基准 |
| w/o M1（无 TP 可行性筛选） | ❌ No | — | 导致内存/延迟违规 |
| w/o M2（无 cost-per-coverage 排序） | ✅ Yes | **134.52 (+50%)** | 成本显著上升 |
| w/o M3（无 TP 升级机制） | ❌ No | — | 延迟违规频发 |

> 🔎 **结论**：M1 和 M3 是**可行性必要条件**，M2 是**成本有效性关键**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **精确最优 ≠ 实际最优**  
   - MILP 得到的“最小成本”部署在真实环境中面对不确定性时极易崩溃，产生高昂的 unmet penalty。
   - 启发式方法因内置保守性（headroom provisioning），反而更具鲁棒性。

2. **约束感知机制是核心创新**  
   - M1/M2/M3 不是普通优化技巧，而是确保每步操作都满足多维耦合约束的**可行性守门员**。

3. **速度带来结构性优势**  
   - AGH 的 sub-second runtime 支持 **rolling-horizon re-optimization**，可在需求波动中持续微调，积累显著收益。
   - GH 因确定性排序无法从中受益，凸显 AGH 的随机探索价值。

4. **AGH 在紧预算下优势更明显**  
   - 如 Table 2 所示，在 critical budget（$72）下，AGH 相比 GH 可减少 **70% 成本** 和 **74% SLO violation**。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **未显式建模不确定性** | 当前方法依赖隐式鲁棒性，而非概率建模（如 chance constraints） |
| **假设离散 query types** | 实际中 query 特征可能是连续分布，需聚类预处理 |
| **忽略 batching 与 queuing 效应** | 当前模型未考虑动态批处理（dynamic batching）带来的吞吐增益 |
| **依赖 trace calibration** | 参数校准依赖外部 trace，可能影响迁移性 |

---

### 🔮 未来工作方向

1. **引入 Stochastic Optimization**  
   - 将 arrival rate、latency、error 建模为随机变量，构建 chance-constrained 或 robust counterpart 优化框架。

2. **整合 Serving Engine Dynamics**  
   - 联合建模 vLLM 等系统的 PagedAttention、continuous batching 行为，实现端到端优化。

3. **在线学习与反馈控制**  
   - 利用实际观测延迟/错误反馈，动态调整 SLO margin 与配置策略。

4. **跨区域调度扩展**  
   - 结合 SkyLB 等工作，拓展至 multi-region、data-locality-aware 的全局负载均衡。

---

## ✅ 总结一句话

> 本文提出了 **GH 与 AGH** 两种高速、约束感知的启发式算法，解决了大规模 LLM 推理服务中模型、资源、并行、路由的联合优化难题，在实现 **>260× 加速**的同时逼近最优成本，并展现出卓越的鲁棒性和滚动优化潜力，为生产级 LLM serving 提供了一套实用高效的调度框架。

</details>

---

### 13. [GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-efficient Large Language Model Fine-tuning](https://arxiv.org/abs/2604.07808)

**Authors**: Kaiyuan Tian, Yu Tang, Gongqingjian Jiang, Baihui Liu, Yifu Gao, Xialin Su, Linbo Qiao, Dongsheng Li  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.07808v1  

#### Abstract
Full-parameter fine-tuning of large language models is constrained by substantial GPU memory requirements. Low-rank adaptation methods mitigate this challenge by updating only a subset of parameters. However, these approaches often limit model expressiveness and yield lower performance than full-par...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-efficient Large Language Model Fine-tuning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLM）的全参数微调（Full-parameter Fine-tuning, FFT）面临巨大的 **GPU 内存开销**，尤其是在大规模模型上难以部署。现有的参数高效微调（PEFT）方法如 LoRA 虽然节省内存，但由于其低秩参数化限制了模型表达能力，导致性能低于 FFT。

另一类方法是层级微调（layer-wise fine-tuning），例如 LISA，通过仅更新部分 Transformer 层来降低内存消耗。然而，这些方法采用**静态层重要性采样策略**，假设各层的重要性在任务和训练阶段中保持不变，忽略了实际中层重要性的动态变化，从而导致次优性能。

### 🚀 提出的新方法：GRASS
本文提出 **GRASS**（Gradient-based Adaptive Layer-wise Importance Sampling），一种基于梯度的自适应层级重要性采样框架，用于内存高效的 LLM 微调。

#### 核心创新点：
1. **动态层重要性度量**  
   使用 **Mean Gradient Norm (MGN)** 作为每层对损失下降贡献的指标。MGN 反映了当前任务和训练阶段下各层的敏感程度，具有任务感知（task-aware）和训练阶段感知（training-stage-aware）特性。

2. **自适应层采样机制**  
   在训练过程中周期性地更新各层的采样概率：
   - 初始阶段进行短暂的 **probing phase** 收集梯度统计信息；
   - 后续阶段根据最新的 MGN 动态调整 softmax 分布下的采样概率，优先更新更重要的层。

3. **层级优化器状态卸载（Optimizer State Offloading）**  
   引入一种**重叠式通信与计算**的层级别优化器状态管理机制：
   - 仅将当前激活层的优化器状态保留在 GPU；
   - 其余层的状态存储于 CPU，并通过异步预取和回写实现通信与计算的重叠，显著减少 GPU 显存占用而不牺牲吞吐量。

### 🔍 相比现有方法的优势
| 方法 | 缺陷 | GRASS 如何改进 |
|------|------|----------------|
| LoRA / DoRA | 低秩表示能力受限，性能低于 FFT | 不引入低秩约束，保留完整参数表达力 |
| LISA / LIFT | 静态层选择，无法适应不同任务或训练阶段 | 动态基于 MGN 更新采样策略，更贴合优化动态 |
| 层级微调通用方案 | 优化器状态需全部驻留 GPU → 显存瓶颈 | 层级卸载 + 通信/计算重叠 → 显著降显存 |

---

## 2. 核心实验方法和设置

### 📚 数据集
#### （1）算术推理任务（Arithmetic Reasoning）
- **训练数据**：Hu et al. (2023) 构建的数学数据集
- **测试基准**（6个）：
  - MultiArith
  - AddSub
  - GSM8K
  - AQuA
  - SingleEq
  - SVAMP

#### （2）常识推理任务（Commonsense Reasoning）
- **训练数据**：8个任务混合数据集（来自 Hu et al., 2023）
- **测试基准**（8个）：
  - BoolQ
  - PIQA
  - SIQA
  - HellaSwag
  - Winograd (Winogrande)
  - ARC-e/c
  - OBQA

此外还使用 Alpaca-GPT4 进行收敛性分析。

### ⚙️ 实验设置
- **模型规模**：
  - TinyLlama (~1.1B)
  - Gemma-2B
  - LLaMA2-7B
- **硬件环境**：2 × NVIDIA H100 80G GPU
- **批大小 & 序列长度**：多数设为 batch size=4 或 1，seq_len=1024
- **评估指标**：
  - 准确率（Accuracy）平均值（Avg. ↑）
  - 峰值 GPU 显存消耗（Peak GPU Memory Usage ↓）
  - 训练吞吐量（Throughput ↑）
  - 消融实验中的稳定性（多随机种子测试）

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| FFT | 全参微调 | 性能强但显存高 |
| LoRA (r=128/256) | Reparameterization | 低秩适配，广泛应用 |
| DoRA | LoRA 改进版 | 解耦权重方向与模长 |
| LISA | Layer-wise | 固定比例均匀采样层 |
| IST / OWS | Static Layer Selection | 基于启发式或离群值加权采样 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 2 & Table 3）

#### ✅ 算术推理任务（Arithmetic Reasoning）——平均准确率（↑）

| Model | FFT | LoRA (r=128) | DoRA | LISA | **GRASS** |
|-------|-----|--------------|------|------|-----------|
| TinyLlama | 33.48 | 29.84 | 33.36 | 33.63 | **34.22** ✅ |
| Gemma-2B | 60.16 | 58.75 | 59.23 | 56.46 | **60.65** ✅ |
| LLaMA2-7B | 60.46 | 58.83 | 59.23 | 56.57 | **59.59** ✅ |

> 💡 GRASS 在所有模型上均优于主流 PEFT 方法，在 Gemma-2B 上甚至超过 FFT。

#### ✅ 常识推理任务（Commonsense Reasoning）——平均准确率（↑）

| Model | FFT | LoRA (r=128) | DoRA | LISA | **GRASS** |
|-------|-----|--------------|------|------|-----------|
| TinyLlama | 39.34 | 37.20 | 37.51 | 37.34 | **38.59** ✅ |
| Gemma-2B | 69.51 | 66.04 | 66.32 | 68.63 | **69.30** ✅ |
| LLaMA2-7B | 77.80 | 75.29 | 76.15 | 75.05 | **76.30** ✅ |

> 💡 GRASS 接近甚至逼近 FFT 表现，在小模型上提升尤为明显。

### 📉 显存效率（Table 4）——峰值 GPU 显存（GB ↓）

| Model | FFT | LoRA | DoRA | LISA | **GRASS** |
|-------|-----|------|------|------|-----------|
| TinyLlama | 8.76 | 4.71 | 5.71 | 4.93 | **4.49** ✅ |
| Gemma-2B | 21.19 | 11.26 | 11.64 | 13.33 | **12.45** ✅ |
| LLaMA2-7B | 51.32 | 19.97 | 23.23 | 22.31 | **19.08** ✅ |

> ✅ GRASS 实现最低或接近最低显存占用，相比 LISA 最多降低 **19.97%** 显存。

#### 🔁 长序列扩展性（Figure 4 Right）
- 当序列长度达到 1792 时：
  - LoRA / DoRA 显存 > 24GB（超出单卡容量）
  - GRASS 仍维持在 **23.25GB**，可正常训练
- 表明 GRASS 更适合长上下文场景。

### ⏱️ 吞吐量表现（Figure 5）
- GRASS 通过 **通信/计算重叠机制**，吞吐量达 LoRA 的 **1.08×**
- 显著高于 FFT，且与主流 PEFT 方法相当
- 证明其在节省显存的同时未牺牲训练效率

### 🔬 消融实验结果

#### （1）是否需要自适应更新采样概率？→ 对比 GRASS vs GRASS*

| Model | GRASS*（静态） | GRASS（动态） | 差距 |
|-------|----------------|---------------|------|
| TinyLlama | 30.69 | 34.22 | +3.53 |
| Gemma-2B | 59.00 | 60.65 | +1.65 |
| LLaMA2-7B | 56.59 | 59.59 | +3.00 |

> ❗ 结果表明：固定采样分布严重损害性能，验证了“层重要性随训练演进”的假设。

#### （2）超参数敏感性分析（Table 6）
- **活跃层数 γ**：越大越好（更多更新预算 → 更接近 FFT）
- **采样周期 Ts**：中间值（25~50）最优
  - 太短（Ts=5）→ 梯度噪声大，采样不稳定
  - 太长 → 不能及时响应重要性变化

#### （3）随机种子鲁棒性（Table 7）
- 在三个不同 seed 下性能波动极小：
  - TinyLlama: ±0.12
  - LLaMA2-7B: ±0.11
- 表明 GRASS 训练稳定，随机采样不会引发不一致性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **层重要性是动态且任务相关的**  
   不同任务（如算术 vs 常识推理）中关键层的位置显著不同，且重要性随训练进程演变。静态采样策略无法捕捉此动态。

2. **MGN 是有效的层重要性代理指标**  
   平均梯度范数（MGN）能够有效反映各层对损失下降的短期影响，适合作为资源分配依据。

3. **GRASS 实现性能与效率双赢**  
   - 性能上：平均准确率最高提升 **+4.38 pts**（vs LoRA）
   - 效率上：显存最多减少 **19.97%**
   - 吞吐量与 LoRA 相当，优于 FFT

4. **层级优化器状态卸载 + 通信重叠 是关键技术支撑**  
   使得动态激活不同层成为可能，同时避免 CPU-GPU 频繁传输带来的性能损耗。

### ⚠️ 方法的局限性
1. **额外计算开销**  
   尽管占比小（约 2–6%），但仍需执行 probing 和周期性 MGN 统计，对极低延迟场景可能有影响。

2. **依赖梯度信号质量**  
   若 batch size 过小或梯度噪声大，MGN 估计可能不准，影响采样效果。

3. **目前仅验证于 Decoder-only 模型**  
   是否适用于 Encoder-Decoder、Vision-Language 模型等尚待研究。

### 🔮 未来工作方向
- 扩展至多模态模型（Multimodal LLMs）
- 探索非梯度信号（如 Hessian 近似）进行重要性估计
- 自动化调节 γ 和 Ts 的元控制机制
- 结合 LoRA 等方法形成 hybrid PEFT 方案

---

## ✅ 总结一句话
> **GRASS 通过基于 MGN 的自适应层采样 + 层级优化器状态卸载，在几乎不损失训练吞吐的前提下，实现了比现有 PEFT 方法更高的精度和更低的显存占用，为高效 LLM 微调提供了新的范式。**

</details>

---

### 14. [Auto-Configured Networks for Multi-Scale Multi-Output Time-Series Forecasting](https://arxiv.org/abs/2604.07610)

**Authors**: Yumeng Zha, Shengxiang Yang, Xianpeng Wang  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.07610v1  

#### Abstract
Industrial forecasting often involves multi-source asynchronous signals and multi-output targets, while deployment requires explicit trade-offs between prediction error and model complexity. Current practices typically fix alignment strategies or network designs, making it difficult to systematicall...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Auto-Configured Networks for Multi-Scale Multi-Output Time-Series Forecasting**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
工业场景中的时间序列预测通常面临以下挑战：
- **多源异构信号**：传感器采样频率不同，导致数据为**异步多尺度**（multi-source asynchronous）；
- **多输出目标**：需同时预测多个质量变量（如烧结过程中的 TFe, FeO, SiO₂ 等）；
- **部署约束**：实际应用中需要在**预测误差**（accuracy）与**模型复杂度**（complexity）之间进行权衡；
- **配置组合爆炸**：预处理（如对齐策略）、网络架构、训练超参数等共同构成庞大的混合搜索空间，传统人工调参或固定流程难以系统优化。

现有方法通常**固定对齐策略或网络结构**，无法联合优化整个 pipeline，且缺乏在有限计算预算下自动输出多种部署选项的能力。

---

### **提出的新方法与新思路**
本文提出一个**端到端的自动配置框架**（auto-configuration framework），其核心包括：

#### ✅ **1. 多尺度双分支卷积网络（MS-BCNN）**
- 设计了一个**短核与长核并行的双分支 CNN 架构**：
  - **短核分支**：捕捉局部波动（local fluctuations）；
  - **长核分支**：建模长期趋势（long-term trends）；
- 支持灵活的**特征融合方式**（如 concat, add, gating, attention 等）；
- 引入**周期性时间嵌入**（sine-cosine periodic embeddings）以增强时序建模能力。

#### ✅ **2. 层级条件混合配置空间（Hierarchical Conditional Mixed Configuration Space）**
- 将以下组件统一纳入可搜索空间：
  - 预处理操作（resampling, pooling）
  - 网络结构（kernel size, normalization, activation）
  - 融合策略（fusion operator）
  - 训练超参数（learning rate, loss function, scheduler）
- 支持**条件依赖**（conditional activation），例如某些参数仅在特定算子被选中时才激活。

#### ✅ **3. 基于玩家机制的混合多目标进化算法（PHMOEA）**
- 提出一种新的 **multi-objective evolutionary algorithm**，专为**有限预算下的黑盒优化**设计；
- 创新性地引入“**玩家追踪**”（player tracking）机制：
  - 每个维度候选值视为“玩家”，维护热度（heat）和出现次数（count）档案；
  - 动态调整采样策略（hot/cold pool），提升搜索效率；
- 支持**自适应离散化细化**（adaptive refinement）连续变量，实现粗到细搜索；
- 包含去重（deduplication）、修复（repair）、早停（early stopping）等工程优化。

#### ✅ **4. 输出帕累托模型集（Pareto Model Set）**
- 不返回单一最优模型，而是输出一组在 **error-complexity trade-off 上非支配的模型集合**；
- 用户可根据部署需求（轻量 or 高精度）灵活选择。

---

### **相比现有方法的优势**
| 维度 | 本文方法 | 传统方法 |
|------|---------|--------|
| **搜索范围** | 联合优化预处理 + 架构 + 超参 | 固定预处理或仅调参 |
| **输出形式** | 可部署的帕累托模型集 | 单一模型 |
| **搜索效率** | PHMOEA 显著减少无效评估，收敛更快 | 评估成本高，易陷入局部最优 |
| **适用性** | 适用于真实工业异步多尺度任务 | 多用于标准同频数据 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### ✅ **真实世界数据集：钢铁烧结过程（Sintering Dataset）**
- 来源于某大型钢厂，共 **2283 个生产周期样本**；
- 输入变量：
  - 原料成分、工艺参数、设备状态等 **多源异步信号**（采样率不同）；
- 输出目标（multi-output）：
  - 5 个终端质量指标：`TFe`, `FeO`, `SiO₂`, `CaO`, `Basicity`；
- 数据划分：
  - **主设置**：按时间顺序划分（chronological split, 0.7/0.15/0.15），模拟真实部署；
  - **对照设置**：随机打乱划分（shuffled split），近似 i.i.d. 场景。

#### ✅ **合成基准：H-DTLZ2 和 H-DTLZ7**
- 构造具有**层级条件决策结构**的多目标优化测试函数；
- 用于分析搜索算法在可控环境下的行为；
- 几何特性覆盖：
  - H-DTLZ2：光滑凸帕累托前沿；
  - H-DTLZ7：不连通、多模态前沿。

---

### **实验设置与评估指标**

#### 🔹 **搜索阶段评估指标（用于比较进化算法性能）**
| 指标 | 含义 |
|------|------|
| **IGD ↓** | Inverted Generational Distance，越小越好，衡量解集逼近参考前沿的程度 |
| **HV ↑** | Hypervolume，越大越好，反映非支配解集覆盖的目标空间体积 |
| **FEs ↓** | Function Evaluations，即完整训练验证次数，衡量搜索效率 |

> 注：每个 evaluation 对应一次完整的 train-validate 运行，成本高昂。

#### 🔹 **预测性能评估指标**
| 指标 | 定义 |
|------|------|
| **NMSE**, **NMAE** | Normalized MSE/MAE，跨目标归一化后平均，消除量纲影响 |
| **MAPE (%)** | Mean Absolute Percentage Error，相对误差 |
| **MSE**, **MAE** | 每目标原始误差 |

---

### **基线方法对比**

#### 📌 **进化算法基线（搜索层面）**
- **MOEA/D**
- **NSGA-II**
- **NSGA-III**
- **SMS-EMOA**

> 所有方法共享相同配置空间与预算（最多 1500 FEs）。

#### 📌 **预测模型基线（性能层面）**
- **OB-ISSID**（state-of-the-art 工业软传感器）
- **Ventingformer**
- **Transformer**
- **LSTM**
- **GRU-PLS**

> 使用 PHMOEA 搜索得到的最佳 MS-BCNN 配置与这些模型对比预测性能。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **搜索质量与效率（表1 & 图2）**

| 方法 | IGD ↓ | HV ↑ | FEs ↓ |
|------|-------|-----|-------|
| **PHMOEA** | **0.0163±0.0026** | **0.986±0.005** | **1113.6±1.9** |
| MOEA/D | 0.1400±0.0150⁺ | 0.781±0.020⁺ | 1500 |
| NSGA-II | 0.0308±0.0040⁺ | 0.984±0.006⁼ | 1500 |
| NSGA-III | 0.0282±0.0035⁺ | 0.986±0.005⁼ | 1500 |
| SMS-EMOA | 0.0175±0.0030⁼ | 0.980±0.006⁼ | 1500 |

> 符号说明：⁺ 表示显著差于 PHMOEA；⁼ 表示无显著差异。

🔹 **结论**：
- PHMOEA 在**更少的函数评估次数下**达到最佳或相当的搜索质量；
- 收敛速度明显快于其他方法（见图2 IGD 曲线）；
- 特别是 MOEA/D 表现较差，表明其在复杂混合空间中适应性不足。

---

#### ✅ **预测性能对比（表2 & 表3）**

##### **随机打乱设置（i.i.d. 近似）**

| Model | NMSE ↓ | NMAE ↓ | MAPE (%) ↓ |
|-------|--------|--------|------------|
| **MS-BCNN** | **0.268±0.217** | **0.348±0.138** | **1.19±0.04** |
| OB-ISSID | 0.304±0.201 | 0.381±0.135 | 1.31±0.07 |
| Ventingformer | 0.279±0.199 | 0.366±0.136 | 1.25±0.07 |
| Transformer | 0.456±0.177 | 0.483±0.103 | 1.65±0.17 |

🔹 **结论**：MS-BCNN 在所有指标上均领先，显示其强大的建模能力。

##### **时间顺序设置（真实部署模拟）**

| Model | NMSE ↓ | NMAE ↓ | MAPE (%) ↓ |
|-------|--------|--------|------------|
| **MS-BCNN** | **4.446±2.193** | **1.682±0.490** | **3.02±0.66** |
| OB-ISSID | 119.7±101.7⁺ | 8.003±3.659⁺ | 13.57±7.50⁺ |
| Ventingformer | 4.789±4.111⁼ | 1.698±0.746⁼ | **2.89±0.87** |
| LSTM | 114.8±206.4⁺ | 6.334±7.545⁺ | 4.23±1.53⁺ |

🔹 **结论**：
- 所有模型误差上升，体现**非平稳性带来的挑战**；
- MS-BCNN 仍保持最低 NMSE 和 NMAE，稳定性最强；
- Ventingformer 在 MAPE 上略优，说明其在相对误差上有互补优势。

---

#### ✅ **消融实验结果（图4 & 表10）**

##### **PHMOEA 消融（图4a）**
- 移除 **Elitism（精英保留）** 导致性能大幅下降 → 精英机制对稳定收敛至关重要；
- 移除 **De-duplication（去重）** 影响较小 → 主要作用是提升效率而非性能。

##### **MS-BCNN 消融（图4b & 表10）**
| 变体 | NMSE ↑ | 结论 |
|------|--------|------|
| Full (MS-BCNN) | 4.446 | 基准 |
| w/o Time Embedding | 5.660 | 时间嵌入显著提升时序预测能力 |
| Single-branch (Short-only) | 4.910 | 缺乏长期建模能力 |
| Single-branch (Long-only) | 4.957 | 忽视短期波动影响精度 |

🔹 **结论**：
- **双分支结构**有效兼顾长短程依赖；
- **时间嵌入**对真实工业时序预测至关重要。

---

## **4. 关键结论和发现**

### **主要发现**
1. **联合优化优于分步设计**：
   - 将预处理、架构、超参数统一建模为层级条件空间，能发现更优的 error-complexity trade-off；
2. **PHMOEA 高效且鲁棒**：
   - 在有限预算下能快速逼近帕累托前沿，显著优于主流 MOEA 方法；
3. **MS-BCNN 更适合工业多尺度预测**：
   - 双分支结构 + 时间嵌入使其在局部与全局模式建模上表现优异；
4. **输出帕累托集更具实用性**：
   - 提供多样化的部署选项，满足不同资源约束下的需求。

---

### **方法的局限性**
1. **静态搜索**：
   - 当前框架为离线搜索，未考虑在线漂移（distribution drift）；
2. **搜索空间仍受限**：
   - 固定了层数和堆叠顺序，未完全开放 NAS；
3. **依赖完整训练评估**：
   - 虽通过 deduplication 和 early stopping 优化，但仍需大量 full training runs。

---

### **未来工作方向**
1. **动态学习与持续更新**：
   - 结合在线监测机制，在分布漂移时触发模型再配置或微调；
2. **将帕累托集作为先验初始化**：
   - 利用搜索得到的优质结构作为动态学习的起点，降低adaptation成本；
3. **构建标准化工业 AutoML 基准**：
   - 推动更多研究关注真实工业场景下的多尺度、多输出、预算受限 AutoML 问题；
4. **结合 freeze-thaw 或代理模型进一步提速**：
   - 如引入部分训练评估（partial evaluation）或 surrogate modeling 提升搜索效率。

---

> **总结一句话**：  
> 本文提出了首个面向**工业多尺度多输出时间序列预测**的**自动配置框架**，通过 **MS-BCNN + PHMOEA** 的协同设计，在有限预算下高效生成高质量的帕累托模型集，兼具理论创新与工程实用价值。

</details>

---

### 15. [Bit-by-Bit: Progressive QAT Strategy with Outlier Channel Splitting for Stable Low-Bit LLMs](https://arxiv.org/abs/2604.07888)

**Authors**: Binxing Xu, Hao Gu, Lujun Li, Hao Wang, Bei Liu, Jiacheng Liu, Qiyuan Zhu, Xintong Yang, Chao Li, Sirui Han, Yike Guo  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.07888v1  

#### Abstract
Training LLMs at ultra-low precision remains a formidable challenge. Direct low-bit QAT often suffers from convergence instability and substantial training costs, exacerbated by quantization noise from heavy-tailed outlier channels and error accumulation across layers. To address these issues, we pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Bit-by-Bit: Progressive QAT Strategy with Outlier Channel Splitting for Stable Low-Bit LLMs**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代大语言模型（LLMs）在超低比特（如 2-bit）下的量化感知训练（QAT）面临严重挑战：
- **收敛不稳定**：直接进行低比特 QAT 容易陷入非光滑的损失景观（loss landscape），导致训练发散或出现 loss spike。
- **误差累积**：深层 Transformer 块中量化误差显著累积，尤其在权重和激活都低精度时（W2A2）。
- **异常值（outlier）影响**：重尾分布的权重和激活通道扩大动态范围，加剧量化误差。
- **训练成本高**：现有 QAT 方法需要大量 token 和复杂的蒸馏机制，计算开销大。

### **提出了什么新方法或新思路**
作者提出 **BIT-BY-BIT**，一个渐进式 QAT 框架，包含三大核心组件：

#### **(1) 渐进式量化策略（Progressive QAT）**
- 从高精度（如 8-bit）开始训练，逐步降低至目标低比特（如 2-bit）。
- 先量化权重（W），再量化激活（A），形成 `w8a16 → w4a16 → w2a16 → w2a2` 的平滑过渡。
- 利用高比特阶段提供良好初始化，避免低比特直接优化的不稳定性。

#### **(2) 一次训练，任意精度部署（Once-for-any-precision）**
- 利用低比特网格是高比特子集的“嵌套”特性（nested grids），通过位移操作（bit shift）实现多精度支持。
- 单一模型可动态部署为 W8/W4/W2，无需重复训练，实现 “train once, deploy any precision”。

#### **(3) 四舍五入感知的异常通道分裂（Rounding-aware Outlier Channel Splitting, OCS）**
- 对敏感通道（metric = `||x||₂ · max|w|`）进行分裂：将一个通道拆分为两个幅度减半的副本。
- 分裂后仍保持原始输出不变，同时减少量化步长（scale），从而降低四舍五入误差。
- 采用块级调度，越深的层分裂比例越高，匹配误差累积趋势。

#### **其他技术细节**
- 使用 **E4M3 FP8** 存储每组（group=32）的缩放因子（scale），提升微缩放（microscaling）精度。
- 开发了高效的 **W2A2 和 W2A16 CUDA 内核**，解决 2-bit 缺乏原生硬件支持的问题。

### **相比现有方法的优势**
| 维度 | BIT-BY-BIT 优势 |
|------|------------------|
| **稳定性** | 渐进策略避免 loss spike，训练更稳定 |
| **效率** | 训练 token 需求减少 **3600×**（vs. ParetoQ） |
| **灵活性** | 支持多精度部署，无需重新训练 |
| **性能** | 在 W2A2 下接近全精度表现，显著优于 BitDistiller、EfficientQAT 等基线 |
| **硬件适配** | 自定义 kernel 实现高达 **11×** 推理加速（vs. BF16） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：
  - RedPajama 子集（4096 samples, seq len=2048）
- **校准/验证数据**：
  - WikiText2
  - C4
  - RedPajama（用于 OCS 通道选择）

### **实验设置**
- **模型家族**：
  - LLaMA-2 / LLaMA-3（2B ~ 13B）
  - Mistral-7B
  - Qwen2.5（7B / 14B）
- **量化配置**：
  - **Weight-only**: W2A16
  - **Weight-activation**: W2A2
  - Group size = 32
- **训练预算控制**：
  - 所有 QAT 方法统一使用约 2 epoch 的 RedPajama + Alpaca 数据，确保公平比较。

### **评估指标**
| 类型 | 指标 |
|------|------|
| **语言建模** | Perplexity (PPL) on WikiText2 / C4 |
| **零样本推理** | Accuracy on PIQA, ARC-Easy, ARC-Challenge, HellaSwag, Winogrande |
| **复杂推理** | GSM8k, MathQA, MMLU, IFEval |
| **推理速度** | GEMV 延迟（μs）、端到端解码吞吐（tokens/s） |

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **PTQ** | GPTQ, AWQ, SmoothQuant, SpinQuant, OmniQuant |
| **QAT** | EfficientQAT, BitDistiller, ParetoQ |
| **多精度** | MatQuant, OmniQuant（需分别训练各比特） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **W2A2 设置下 WikiText2 PPL 表现（LLaMA-2 7B）**
| 方法 | PPL |
|------|-----|
| FP16（全精度） | 5.47 |
| **BIT-BY-BIT (Ours)** | **7.72** |
| EfficientQAT | 9.71 |
| BitDistiller | 29.66 |
| ParetoQ | 259.74 |

> ✅ **仅增加 +2.25 PPL**，远优于所有基线。

#### **零样本平均准确率（LLaMA-3.2-3B）**
| 配置 | 方法 | 平均准确率 |
|------|------|------------|
| W2A16 | Bit-by-Bit | **56.91** |
| W2A2 | Bit-by-Bit | **51.52** |

> ✅ 在 W2A2 下仍保持 **51.52%** 准确率，领先第二名（BitDistiller）**5+ pts**。

#### **推理加速**
- **GEMV 内核延迟**（W2A2 vs BF16）：
  - 在 `(4096,14336)` 形状下，**加速 11×**
- **端到端解码吞吐**（Llama3-8B）：
  - BF16: **49 tokens/s**
  - W2A2: **76 tokens/s** → **1.5× 加速**

---

### **与基线方法的对比结果**
| 指标 | 结果 |
|------|------|
| **PPL 降低** | 在 LLaMA-3.2-3B 上，W2A2 PPL 从 ParetoQ 的 1018.61 降至 **13.87** |
| **零样本任务** | 在 W2A2 下，平均准确率比 EfficientQAT 高 **11.4 pts** |
| **训练效率** | Token 需求仅为 ParetoQ 的 **1/3600**（图 2c） |
| **多精度部署** | 单次训练即可支持 W8/W4/W2，性能与单独训练相当 |

---

### **消融实验结果**

在 LLaMA-3.2-1B 上进行消融（W2A16）：

| 方法 | WikiText2 PPL | 任务平均准确率 |
|------|---------------|----------------|
| End-to-end + NLL | 1700+ | 35.09 |
| Block-wise | 31.88 | 40.87 |
| + Progressive | 24.60 | 43.26 |
| + OCS (`||x||₂·max|w|`) | **17.07** | **45.18** |

> 🔍 **关键发现**：
> - Block-wise 损失至关重要
> - 渐进策略带来巨大提升
> - OCS 中联合 metric `||x||₂·max|w|` 效果最佳
> - Group size 从 32 增至 128 导致性能骤降（38.60 → 45.18），说明细粒度分组必要

---

## **4. 关键结论和发现**

### **主要发现**
1. **渐进式训练是超低比特 QAT 成功的关键**：直接 W2A2 训练几乎无法收敛，而逐步退火能提供稳定初始化。
2. **嵌套量化网格支持“一次训练，任意精度”**：无需额外训练即可灵活部署不同比特宽度。
3. **OCS 显著缓解异常值问题**：通过分裂而非剪裁保留语义信息，且四舍五入感知设计保证输出不变。
4. **自定义 kernel 可实现高效推理**：即使无原生 2-bit 支持，也能通过优化实现显著加速。
5. **新型架构更抗量化**：Qwen2.5 在 W2A2 下几乎无损，而 LLaMA-2 性能下降明显。

### **方法的局限性**
1. **对某些模型族效果较差**：在 Qwen 家族上性能下降较大，需进一步分析。
2. **分布式训练不友好**：Block-wise 策略增加通信和负载均衡复杂度。
3. **未探索自动调度**：分裂比例、学习率等仍依赖人工设定。
4. **未扩展至 MoE 或 KV Cache 量化**：当前仅适用于标准 Dense Transformer。

### **未来工作方向**
- 学习层自适应的分裂比例和量化策略
- 扩展至 MoE 架构和长上下文（KV-cache quantization）
- 硬件感知的混合精度搜索（hardware-aware mixed-precision）
- 轻量级蒸馏与 BIT-BY-BIT 结合
- 探索 LoRA-based distribution-preserving progression（见 Appendix B.3）

---

> 📌 **总体评价**：  
> BIT-BY-BIT 是目前最稳定、高效的超低比特 QAT 框架之一，首次在 W2A2 下实现接近全精度的语言建模性能，并支持灵活部署与高效推理，为边缘设备上的 LLM 部署提供了实用路径。

</details>

---

### 16. [DMax: Aggressive Parallel Decoding for dLLMs](https://arxiv.org/abs/2604.08302)

**Authors**: Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, Xinchao Wang  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.08302v1  

#### Abstract
We present DMax, a new paradigm for efficient diffusion language models (dLLMs). It mitigates error accumulation in parallel decoding, enabling aggressive decoding parallelism while preserving generation quality. Unlike conventional masked dLLMs that decode through a binary mask-to-token transition,...

---

### 17. [SepSeq: A Training-Free Framework for Long Numerical Sequence Processing in LLMs](https://arxiv.org/abs/2604.07737)

**Authors**: Jie Sun, Yu Liu, Lu Han, Qiwen Deng, Xiang Shu, Yang Xiao, Xingyu Lu, Jun Zhou, Pengfei Liu, Lintao Ma, Jiancan Wu, Xiang Wang  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.07737v1  

#### Abstract
While transformer-based Large Language Models (LLMs) theoretically support massive context windows, they suffer from severe performance degradation when processing long numerical sequences. We attribute this failure to the attention dispersion in the Softmax mechanism, which prevents the model from ...

---

### 18. [Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](https://arxiv.org/abs/2604.07692)

**Authors**: Micky C. Nnamdi, Benoit L. Marteau, Yishan Zhong, J. Ben Tamo, May D. Wang  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.07692v1  

#### Abstract
Large Multimodal Models (LMMs) achieve state-of-the-art performance in high-stakes domains like healthcare, yet their reasoning remains opaque. Current interpretability methods, such as attention mechanisms or post-hoc saliency, often fail to faithfully represent the model's decision-making process,...

---

### 19. [Value-Guidance MeanFlow for Offline Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.08174)

**Authors**: Teng Pang, Zhiqiang Dong, Yan Zhang, Rongjian Xu, Guoqiang Wu, Yilong Yin  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.08174v1  

#### Abstract
Offline multi-agent reinforcement learning (MARL) aims to learn the optimal joint policy from pre-collected datasets, requiring a trade-off between maximizing global returns and mitigating distribution shift from offline data. Recent studies use diffusion or flow generative models to capture complex...

---

### 20. [TR-EduVSum: A Turkish-Focused Dataset and Consensus Framework for Educational Video Summarization](https://arxiv.org/abs/2604.07553)

**Authors**: Figen E\u{g}in, Aytu\u{g} Onan  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.07553v1  

#### Abstract
This study presents a framework for generating the gold-standard summary fully automatically and reproducibly based on multiple human summaries of Turkish educational videos. Within the scope of the study, a new dataset called TR-EduVSum was created, encompassing 82 Turkish course videos in the fiel...

---

### 21. [Reduced-Mass Orbital AI Inference via Integrated Solar, Compute, and Radiator Panels](https://arxiv.org/abs/2604.07760)

**Authors**: Stephen Gaalema, Samuel Indyk, Clinton Staley  
**Category**: cs.DC  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.07760v1  

#### Abstract
We describe and analyze a distributed compute architecture for SSO computational satellites that can potentially provide >100 kW compute power per launched metric ton (including deployment and station keeping mass). The architecture co-locates and integrates the solar cells, radiator, and compute fu...

---

### 22. [Bayesian Optimization for Mixed-Variable Problems in the Natural Sciences](https://arxiv.org/abs/2604.07416)

**Authors**: Yuhao Zhang, Ti John, Matthias Stosiek, Patrick Rinke  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.07416v1  

#### Abstract
Optimizing expensive black-box objectives over mixed search spaces is a common challenge across the natural sciences. Bayesian optimization (BO) offers sample-efficient strategies through probabilistic surrogate models and acquisition functions. However, its effectiveness diminishes in mixed or high...

---

### 23. [A Systematic Framework for Tabular Data Disentanglement](https://arxiv.org/abs/2604.07940)

**Authors**: Ivan Tjuawinata, Andre Gunawan, Anh Quan Tran, Nitish Kumar, Payal Pote, Harsh Bansal, Chu-Hung Chi, Kwok-Yan Lam, Parventanis Murthy  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.07940v1  

#### Abstract
Tabular data, widely used in various applications such as industrial control systems, finance, and supply chain, often contains complex interrelationships among its attributes. Data disentanglement seeks to transform such data into latent variables with reduced interdependencies, facilitating more e...

---

### 24. [Provably Adaptive Linear Approximation for the Shapley Value and Beyond](https://arxiv.org/abs/2604.08438)

**Authors**: Weida Li, Yaoliang Yu, Bryan Kian Hsiang Low  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.08438v1  

#### Abstract
The Shapley value, and its broader family of semi-values, has received much attention in various attribution problems. A fundamental and long-standing challenge is their efficient approximation, since exact computation generally requires an exponential number of utility queries in the number of play...

---

### 25. [Quantization Impact on the Accuracy and Communication Efficiency Trade-off in Federated Learning for Aerospace Predictive Maintenance](https://arxiv.org/abs/2604.08474)

**Authors**: Abdelkarim Loukili  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.08474v1  

#### Abstract
Federated learning (FL) enables privacy-preserving predictive maintenance across distributed aerospace fleets, but gradient communication overhead constrains deployment on bandwidth-limited IoT nodes. This paper investigates the impact of symmetric uniform quantization ($b \in \{32,8,4,2\}$ bits) on...

---

### 26. [SymptomWise: A Deterministic Reasoning Layer for Reliable and Efficient AI Systems](https://arxiv.org/abs/2604.06375)

**Authors**: Isaac Henry, Avery Byrne, Christopher Giza, Ron Henry, Shahram Yazdani  
**Category**: cs.AI  
**Published**: 2026-04-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.06375v1  

#### Abstract
AI-driven symptom analysis systems face persistent challenges in reliability, interpretability, and hallucination. End-to-end generative approaches often lack traceability and may produce unsupported or inconsistent diagnostic outputs in safety-critical settings. We present SymptomWise, a framework ...

---

### 27. [Enabling Intrinsic Reasoning over Dense Geospatial Embeddings with DFR-Gemma](https://arxiv.org/abs/2604.07490)

**Authors**: Xuechen Zhang, Aviv Slobodkin, Joydeep Paul, Mandar Sharma, Samet Oymak, Shravya Shetty, Gautam Prasad  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.07490v1  

#### Abstract
Representation learning for geospatial and spatio-temporal data plays a critical role in enabling general-purpose geospatial intelligence. Recent geospatial foundation models, such as the Population Dynamics Foundation Model (PDFM), encode complex population and mobility dynamics into compact embedd...

---

### 28. [Guaranteeing Knowledge Integration with Joint Decoding for Retrieval-Augmented Generation](https://arxiv.org/abs/2604.08046)

**Authors**: Zhengyi Zhao, Shubo Zhang, Zezhong Wang, Yuxi Zhang, Huimin Wang, Yutian Zhao, Yefeng Zheng, Binyang Li, Kam-Fai Wong, Xian Wu  
**Category**: cs.CL  
**Published**: 2026-04-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.08046v1  

#### Abstract
Retrieval-Augmented Generation (RAG) significantly enhances Large Language Models (LLMs) by providing access to external knowledge. However, current research primarily focuses on retrieval quality, often overlooking the critical ''integration bottleneck'': even when relevant documents are retrieved,...

---

### 29. [LLM-Generated Fault Scenarios for Evaluating Perception-Driven Lane Following in Autonomous Edge Systems](https://arxiv.org/abs/2604.07362)

**Authors**: Faezeh Pasandideh, Achim Rettberg  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.07362v1  

#### Abstract
Deploying autonomous vision systems on edge devices faces a critical challenge: resource constraints prevent real-time and predictable execution of comprehensive safety tests. Existing validation methods depend on static datasets or manual fault injection, failing to capture the diverse environmenta...

---

### 30. [A Graph Foundation Model for Wireless Resource Allocation](https://arxiv.org/abs/2604.07390)

**Authors**: Yucheng Sheng, Jiacheng Wang, Le Liang, Hao Ye, Shi Jin  
**Category**: cs.LG  
**Published**: 2026-04-10  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.07390v1  

#### Abstract
The aggressive densification of modern wireless networks necessitates judicious resource allocation to mitigate severe mutual interference. However, classical iterative algorithms remain computationally prohibitive for real-time applications requiring rapid responsiveness. While recent deep learning...

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
