# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-03 06:16:19 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification](https://arxiv.org/abs/2603.01399)

**Authors**: Guang Huang, Zeyi Wen  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2603.01399v1  

#### Abstract
Speculative Decoding (SD) has emerged as a premier technique for accelerating Large Language Model (LLM) inference by decoupling token generation into rapid drafting and parallel verification. While recent advancements in self-speculation and lookahead decoding have successfully minimized drafting o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

- **验证阶段成为新的性能瓶颈**：尽管现有的 Self-Speculative Decoding（如 Ngram、Lookahead、Medusa/EAGLE）有效加速了 drafting 阶段，但其 **verification 阶段仍需对目标模型进行全精度（如 BF16）前向传播**，导致该过程严重受限于 **memory bandwidth**。
- 在高带宽压力场景下，即使 draft 长度增加，验证开销也抵消了潜在加速收益，形成“**memory wall**”。

### 🚀 提出的新方法与思路

- **提出 Quasar**：一种无需训练的 **Quantized Verification** 框架，首次将低比特量化应用于 speculative decoding 中的 **verifier 而非 drafter**。
- 核心思想：使用 **W8A8 量化的模型作为 verifier**，在保持生成质量的同时显著减少内存访问量（降低 50%），从而加速验证阶段。

### 🔍 相比现有方法的优势

| 对比维度 | 传统方法 | Quasar |
|--------|--------|-------|
| **Verifier 精度** | 必须使用 full-precision（BF16/FP16）以保证质量 | 使用 W8A8 量化，仍能保持高接受率（acceptance length） |
| **内存带宽消耗** | 高（加载 full-precision 权重） | 降低约 50%（INT8 权重） |
| **与 drafting 策略关系** | 多数依赖特定 drafting 架构 | **正交于 drafting 方法**，可无缝集成到各类 self-speculation 框架中 |
| **实现成本** | 通常需要额外训练或 distillation | **training-free**，仅需离线量化校准 |

> 💡 创新点总结：**首次挑战“verifier 必须高精度”的假设，证明 W8A8 量化足以胜任 verification 任务，直接打破 memory-bound 瓶颈。**

---

## 2. 核心实验方法和设置

### 📚 数据集与模型

- **模型**：
  - `Qwen3-8B`
  - `OpenPangu-7B`
- **任务与对应数据集**：
  - 多轮对话 → **MT-bench**
  - 代码生成 → **HumanEval**
  - 数学推理 → **GSM8k**
  - 指令遵循 → **Alpaca**
  - 文本摘要 → **CNN/Daily Mail**

### ⚙️ 实验设置

- **硬件平台**：单张 Ascend 910B2 NPU（64GB）
- **推理引擎**：基于 vLLM-Ascend 实现，集成 NCLL 支持 INT8 GEMM
- **量化方案**：
  - 采用增强版 **SmoothQuant** 进行 W8A8 量化，缓解激活值中的 outlier 问题
  - 离线完成权重平滑与量化，运行时动态对激活进行 on-the-fly 平滑与量化
- **drafting 方法**：`Prompt Lookup Decoding (Ngram)`，最大 draft 长度为 4

### 📊 评估指标

| 指标 | 含义 |
|-----|------|
| **Speedup / Throughput** | 相对于 vanilla auto-regressive 推理的端到端吞吐提升倍数 |
| **Mean Acceptance Length (L)** | 平均每次成功接受的 token 数量，反映生成一致性与质量 |
| **Acceptance Rate (α)** | 成功通过 rejection sampling 的 token 比例 |
| **Accuracy** | 在下游任务（如 MMLU、CEval 等）上的得分，衡量模型智能保留能力 |

### 🆚 基线方法对比

| 方法 | 描述 |
|------|------|
| **Vanilla (Auto-regressive)** | 标准逐 token 生成，无 speculative decoding |
| **Ngram (Self-Speculative, BF16 Verifier)** | 使用 n-gram 匹配生成 draft，但 verifier 仍为 full-precision |
| **Quasar (W8A8 Verifier)** | 本文方法，使用量化 verifier，drafting 策略与 Ngram 一致 |

---

## 3. 主要实验结果和性能指标

### 📈 端到端吞吐提升（Speedup）

| 模型 | 设置 | Ngram (BF16) | **Quasar (W8A8)** | 提升幅度 |
|------|------|-------------|------------------|---------|
| Qwen3-8B | T=0（greedy） | 1.18× | **1.28×** | **+8.5%** |
| OpenPangu-7B | T=0 | 1.08× | **1.13×** | **+4.6%** |
| Qwen3-8B | T=1（stochastic） | 1.15× | **1.23×** | **+7.0%** |

> ✅ 在所有任务上均优于基线，尤其在 **GSM8k（数学推理）** 上达到最高 **1.64× speedup**（Qwen3），表明 memory-bound 任务受益最明显。

### 🎯 接受长度（Quality 保持）

| 模型 | 方法 | T=0 (L) | T=1 (L) |
|------|------|--------|--------|
| Qwen3-8B | Ngram | 1.33 | 1.31 |
| Qwen3-8B | **Quasar** | **1.40** | **1.36** |
| OpenPangu-7B | Ngram | 1.27 | 1.24 |
| OpenPangu-7B | **Quasar** | **1.29** | **1.27** |

> ✅ **Quasar 不仅未降低接受长度，反而更高**，说明 W8A8 量化并未破坏 logit 分布的相对排序，甚至可能因噪声起到轻微正则化作用。

### 🔍 消融实验与敏感性分析

#### （1）不同 draft 长度的影响（Table 3）

- 当 speculative token 数量 $ y = 5 $ 且 prompt lookup 范围 $ K=(1,3) $ 时，Quasar 达到峰值 **1.47× speedup**。
- 更长的 draft（如 $ y=9 $）反而导致性能下降，因 verification 开销超过增益 → 表明存在 **最优 draft 长度**。

#### （2）温度鲁棒性测试（Table 2）

- 即使在 $ T=1 $ 高熵采样下，Quasar 仍保持 **1.23× speedup** 和 **L=1.36**，稳定性优于 Ngram。
- 表明方法在多样化生成场景中依然可靠。

#### （3）结构剪枝 vs. 量化（Table 5）

| 方法 | 层保留 | L | Speedup |
|------|--------|----|----------|
| Vanilla | 100% / BF16 | 1.00 | 1.00× |
| Pruned-90% | 90% / BF16 | 1.62 | 0.80× |
| Pruned-75% | 75% / BF16 | 1.27 | 0.68× |
| Pruned-50% | 50% / BF16 | 1.03 | 0.62× |
| **Quasar** | **100% / W8A8** | **1.40** | **1.28×** |

> ❌ 结构剪枝虽减少层数，但破坏网络拓扑结构，导致分布偏移严重，acceptance length 下降或计算节省不足以补偿。
>
> ✅ **Quasar 保持完整深度，仅压缩数值精度，实现了“速度↑ + 质量↑”的双赢**。

### ✅ 准确率保留（Table 4）

| 模型 | 平均准确率差异（Δ） |
|------|--------------------|
| OpenPangu-7B | +3.1% |
| Qwen3-8B | +2.9% |

> 实际 Δ 为正，表示 **Quasar 在多数 benchmark 上得分略高于原模型**，作者认为可能是量化噪声起到了正则化效果。总体可视为 **近似 lossless**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Verification 是当前 speculative decoding 的主要瓶颈**，尤其是在 memory-bound 场景下。
2. **W8A8 量化 verifier 是可行且高效的**：现代 PTQ 技术（如 SmoothQuant）已足够成熟，能在几乎不损失 logit 分布保真度的前提下实现高效验证。
3. **量化优于结构剪枝**：保持网络拓扑完整性比维持高数值精度更重要；**depth > precision** for verification。
4. **Quasar 是通用插件式加速方案**：正交于 drafting 策略，适用于 Ngram、Lookahead、Medusa/EAGLE 等各类 self-speculative 框架。
5. **实现了“免费加速”（free lunch）**：在几乎不牺牲生成质量的前提下，获得高达 **1.28× 的端到端吞吐提升**。

### ⚠️ 方法的局限性

1. **依赖硬件支持 INT8 tensor core**：若设备缺乏高效 INT8 计算单元（如某些 GPU/NPU），加速效果会受限。
2. **极端复杂推理任务中可能存在微小 acceptance drop**：虽然整体稳定，但在超高精度要求的任务中，量化误差可能略微影响 top-1 一致性。
3. **目前仅验证 W8A8，更低比特（如 W4A4）尚未探索**：进一步压缩需权衡误差累积风险。

### 🔮 未来工作方向

1. **Ultra-low Bit Verification**：探索 W4A4 或混合精度量化，进一步突破带宽限制。
2. **Dynamic Precision Scaling**：根据 draft confidence 动态调整 verifier 精度，实现细粒度 speed-accuracy trade-off。
3. **Hardware-Aware Optimization**：针对特定芯片（如 Ascend 910C、H100）优化 kernel 调度与内存布局。
4. **Integration with Tree-based Speculation**：结合 EAGLE/Medusa 的树形 speculative decoding，验证其在非线性路径下的扩展性。

---

> 🔗 **开源地址**：https://github.com/Tom-HG/Quasar  
> 📄 **一句话总结**：*Quasar 通过将 verifier 从 BF16 替换为 W8A8，打破了 speculative decoding 的 memory wall，在保持生成质量的同时实现高达 1.28× 的端到端加速，是首个真正面向 “verification acceleration” 的通用解决方案。*

</details>

---

### 2. [Trident: Adaptive Scheduling for Heterogeneous Multimodal Data Pipelines](https://arxiv.org/abs/2603.02075)

**Authors**: Ding Pan, Zhuangzhuang Zhou, Long Qian, Binhang Yuan  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2603.02075v1  

#### Abstract
The rapid adoption of large language models and multimodal foundation models has made multimodal data preparation pipelines critical AI infrastructure. These pipelines interleave CPU-heavy preprocessing with accelerator-backed (GPU/NPU/TPU) inference and produce massive intermediate artifacts. Achie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Trident: Adaptive Scheduling for Heterogeneous Multimodal Data Pipelines**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代多模态数据处理流水线（如用于训练大语言模型和多模态基础模型的数据管道）面临以下挑战：
- **高度非平稳的工作负载**：输入数据特性（如文档长度、视频复杂度）导致执行时间和资源消耗剧烈波动。
- **异构算力调度困难**：流水线混合了 CPU 密集型操作（如解析、过滤）和 AI 加速器（GPU/NPU/TPU）上的推理任务，资源需求差异大。
- **动态批处理与内存瓶颈**：LLM 和视觉模型采用连续批处理（continuous batching），其吞吐量与延迟行为复杂，且配置不当易引发 **Out-of-Memory (OOM)** 故障。
- **现有调度器效率低下**：传统方法依赖阈值触发的自动扩缩容（threshold-based autoscaling），假设算子同步运行、忽略通信开销，无法适应真实场景。

### 🚀 提出的新方法：TRIDENT
TRIDENT 是一个面向固定资源集群的自适应调度框架，集成三个紧密耦合的闭环控制层：

#### （1）**Observation Layer（观测层）**
- 使用 **Gaussian Process (GP) 回归** 建模算子吞吐能力，以输入特征（如 token 长度、图像分辨率）为条件预测可持续吞吐量。
- 引入 **两阶段异常过滤机制**：
  - **信号级过滤**：基于利用率（GPU/CPU）、队列长度排除饥饿或积压状态下的观测。
  - **模型级过滤**：利用 GP 预测置信区间识别并剔除残差过大的离群样本。
- 输出噪声鲁棒的容量估计，供上层决策使用。

#### （2）**Adaptation Layer（适配层）**
- 实现在线聚类（incremental clustering）检测工作负载模式漂移（workload regime shifts）。
- 采用 **Memory-Constrained Bayesian Optimization (BO)** 在安全前提下搜索最优配置（如 batch size, max sequence length）。
  - 将峰值显存作为黑盒约束建模，避免 OOM 探索。
  - 通过可行性概率（Probability of Feasibility, PoF）引导采样，兼顾性能与安全性。

#### （3）**Scheduling Layer（调度层）**
- 构造 **Mixed-Integer Linear Program (MILP)** 联合优化：
  - 算子并行度（parallelism）
  - 实例部署位置（placement）
  - 配置迁移策略（rolling updates）
- 显式建模异构资源池（CPU/GPU/NPU）、网络带宽限制及跨节点传输成本。
- 引入 **滚动更新机制**，权衡预期吞吐增益与冷启动开销，仅在收益大于代价时执行配置切换。

#### 🔁 闭环反馈设计
三层形成闭环控制：
- 观测层提供容量估计 → 支持调度与调参；
- 调度层执行配置变更 → 反馈至观测层清空旧样本，触发模型重建；
- 保证系统状态一致性，防止“陈旧模型”误导后续决策。

### ⭐ 相比现有方法的优势
| 维度 | 现有方法缺陷 | TRIDENT 改进 |
|------|--------------|---------------|
| 容量建模 | 依赖 `useful-time` 指标，在异步批处理下失效 | 使用 GP + 异常过滤，准确捕捉可持续吞吐 |
| 配置调优 | 离线静态调参，不适应运行时变化；无内存保护 | 在线 BO + 内存约束，动态响应负载漂移 |
| 资源调度 | 忽略 placement 与通信开销；独立扩缩各算子 | 联合优化 parallelism + placement + config transition |
| 控制逻辑 | 各模块解耦，缺乏协同 | 三层次闭环反馈，保持全局一致性 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **PDF 处理流水线**：
  - ~200k 文档，包含学术论文、年报、财务报告三类。
  - 流水线含 17 个算子，涉及文件 I/O、布局检测、OCR（Text/Table/Formula）、聚合等。
- **视频清洗流水线**：
  - ~410k 视频片段，分为短格式（10–30s, ≤720p）和长格式（5–10min, 1080p–4K）。
  - 流水线含 9 个算子，包括场景分割、美学评分、文本过滤、LLM 字幕生成等。

### 💻 实验环境
- **集群配置**：8 台服务器，每台配备：
  - 8 × Huawei Ascend 910B NPU
  - 256 CPU 核心
  - 1TB 内存
  - 100 Gbps 网络互联
- **实现平台**：基于 **Ray Data v2.46.0** 扩展实现 TRIDENT。

### 🎯 评估指标
- **End-to-End Throughput**（端到端吞吐量）：原始输入记录/秒，为主要性能指标。
- **Mean Absolute Percentage Error (MAPE)**：衡量观测层容量估计准确性。
- **OOM Events & Downtime**：验证内存安全机制有效性。
- **Migration Cost / Cold Start Overhead**：评估配置切换稳定性。
- **Ablation Studies**：分析各组件对整体性能的贡献。

### 🔁 基线方法对比
| 方法 | 特点 |
|------|------|
| **Static** | 手动调优固定资源配置，无动态适应 |
| **Ray Data** | 默认基于利用率阈值的 autoscaler，逐算子独立扩缩 |
| **DS2** | 基于 useful-time 的并行度推导，假设同步执行 |
| **ContTune** | 在 DS2 基础上引入保守贝叶斯优化学习非线性关系 |
| **SCOOT** | 对每个 LLM 算子进行离线 BO 调参，部署后不再变化 |

> 注：所有方法均在同一 Ray Data 框架下公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（相对 Static 的加速比）

| 方法 | PDF Pipeline | Video Pipeline |
|------|---------------|----------------|
| **TRIDENT (Full)** | **2.01×** | **1.88×** |
| SCOOT | 1.21× | 1.17× |
| ContTune | 1.04× | 0.96× |
| Ray Data | 1.12× | 1.18× |
| DS2 | 0.87× | 0.79× |
| Static | 1.00× | 1.00× |

> ✅ TRIDENT 在两种生产级流水线上分别实现 **2.01× 和 1.88× 的端到端吞吐提升**，显著优于所有基线。

### 🔍 控制变量实验（RQ2）：相同观测+适配输入下的调度层对比
| 方法 | PDF | Video |
|------|-----|-------|
| TRIDENT | **2.01×** | **1.88×** |
| TRIDENT (All-at-once) | 1.92× | 1.79× |
| ContTune | 1.42× | 1.36× |

> 即使共享相同的容量估计与配置建议，TRIDENT 仍大幅领先，说明其 **MILP 联合优化机制是性能优势主因**。滚动更新额外带来约 5% 提升。

### 📈 观测层精度（MAPE%）

| 方法 | PDF | Video |
|------|-----|-------|
| True Processing Rate | 62.7% | 54.3% |
| EMA | 28.3% | 25.7% |
| GP w/o filtering | 24.3% | 21.8% |
| GP + signal filtering | 8.4% | 7.1% |
| **TRIDENT (two-stage filtering)** | **5.6%** | **4.8%** |

> 两阶段异常过滤显著提升 GP 模型准确性，误差降低 **80% 以上**，证明其对异步算子建模至关重要。

### 🔬 适配层有效性（RQ4）

#### 工作负载聚类准确率（vs 离线算法）
| 方法 | Pipeline | Clusters | Purity | ARI |
|------|----------|---------|--------|-----|
| TRIDENT (online) | PDF | 3 | 0.95 | 0.89 |
| K-means (offline) | PDF | 3 | 0.97 | 0.94 |
| TRIDENT (online) | Video | 2 | 0.97 | 0.93 |

> 在线聚类能自动发现正确类别数，纯度与 ARI 接近离线方法，具备强实用性。

#### 配置优化效果（Normalized to Default = 1.0×）
| 方法 | TextOCR (PDF) | Captioning (Video) |
|------|----------------|--------------------|
| Default Config | 1.00× | 1.00× |
| Random Search | 1.18× | 1.14× |
| Grid Search | 1.22× | 1.19× |
| Unconstrained BO | 1.38× | 1.35× (**t**) |
| **Constrained BO (TRIDENT)** | **1.36×** | **1.33×** |

> 内存约束 BO 性能接近无约束版本，但 **避免了 OOM 故障（标记为 t 的配置在持续运行中崩溃）**。

#### OOM 事件减少效果
| 指标 | PDF Pipeline (Uncon.) | PDF (Constr.) | Video (Uncon.) | Video (Constr.) |
|------|------------------------|---------------|----------------|------------------|
| OOM Events | 14 | **3** | 11 | **2** |
| Downtime (s) | 462 | **102** | 352 | **68** |
| Throughput Loss vs Oracle | 8.7% | **3.2%** | 7.2% | **2.7%** |

> 内存约束 BO 减少 **79–82% 的 OOM 事件**，累计停机时间下降超 75%，有效提升可用性。

### 🔪 消融实验（Ablation Study）

| 变体 | PDF Pipeline | Video Pipeline |
|------|---------------|----------------|
| **TRIDENT (Full)** | 100.0% | 100.0% |
| w/o Observation Layer | 66.5% | 60.9% |
| w/o Adaptation Layer | 79.6% | 78.1% |
| w/o Placement-Aware Scheduling | 90.5% | 84.0% |
| w/o Rolling Update | 95.5% | 95.2% |

> - **观测层影响最大**：错误容量估计导致资源错配，性能腰斩。
> - **适配层贡献显著**：动态调参带来近 20% 增益。
> - **placement-aware 重要性凸显**：尤其在视频流水线中，通信密集阶段需共置优化。
> - **滚动更新缓解冷启动冲击**：虽影响较小，但在频繁切换场景中累积效应明显。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **异步多模态算子的容量建模必须结合 workload-aware modeling 与 robust anomaly filtering**，否则会导致严重资源浪费。
2. **内存安全是在线配置调优的前提**，盲目追求高吞吐可能引发 OOM 级联失败；**memory-constrained BO 是高效且安全的选择**。
3. **全局联合优化（parallelism + placement + config transition）远胜局部决策**，尤其在网络受限环境下，placement 成为关键因素。
4. **闭环反馈机制保障系统一致性**：配置变更后及时清空历史观测，避免“模型滞后”问题。
5. **TRIDENT 实现高达 2.01× 的端到端吞吐提升**，且开销极低（MILP 求解平均 < 200ms），适合在线重优化。

### ⚠️ 局限性
- 当前假设线性流水线 DAG 结构，未充分支持复杂分支或循环结构。
- MILP 规模随算子数量增长而扩大，大规模图可能面临求解延迟问题（尽管已在异步 Actor 中处理）。
- 聚类与调优过程仍有一定探索成本，极端快速的负载漂移可能导致短暂性能下降。

### 🔮 未来工作方向
- 扩展至更复杂的 DAG 拓扑结构（如 fan-out/fan-in）。
- 引入增量 MILP 求解或强化学习代理以应对更大规模调度问题。
- 探索轻量化 surrogate model 替代 GP，进一步降低观测层开销。
- 结合 trace 预测实现前瞻性调度（proactive scheduling）而非反应式调整。

---

> **总结一句话**：  
> TRIDENT 通过 **Observation-Adaptation-Scheduling 三层次闭环控制**，首次实现了在固定资源下对异构多模态数据流水线的高性能、高安全、自适应调度，解决了传统调度器在容量建模、配置调优与资源协调方面的根本缺陷，为下一代 AI 数据基础设施提供了关键支撑。

</details>

---

### 3. [A Cascaded Graph Neural Network for Joint Root Cause Localization and Analysis in Edge Computing Environments](https://arxiv.org/abs/2603.01447)

**Authors**: Duneesha Fernando, Maria A. Rodriguez, Rajkumar Buyya  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.01447v1  

#### Abstract
Edge computing environments host increasingly complex microservice-based IoT applications that are prone to performance anomalies propagating across dependent services. Identifying the faulty component (root cause localization) and the underlying fault type (root cause analysis) is essential for tim...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Cascaded Graph Neural Network for Joint Root Cause Localization and Analysis in Edge Computing Environments

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在边缘计算环境中，基于微服务的IoT应用日益复杂，性能异常会通过服务依赖关系传播，导致多个服务表现出异常行为。传统的 **Root Cause Localization (RCL)** 和 **Root Cause Analysis (RCA)** 方法通常采用集中式处理全系统图的方式进行诊断，这在大规模分布式边缘环境中面临以下挑战：
- **高推理延迟**：随着图规模增大，GNN的消息传递复杂度呈二次增长（$O(N^2)$），难以满足实时性要求。
- **可扩展性差**：无法有效应对成百上千个微服务构成的大规模系统。

此外，大多数现有方法仅关注诊断准确性，而忽视了效率与延迟之间的平衡。

### 提出的新方法与创新思路
本文提出了一种**级联式图神经网络框架（Cascaded GNN）**，用于联合执行 **RCL** 与 **fault type identification**（即RCA的一种高级形式）。其核心思想是：
- **通信驱动的聚类（Communication-driven Clustering）**：利用 **Louvain算法** 对微服务按通信强度进行社区划分，形成高度交互的子集群。
- **两级级联架构（Two-stage Cascade Architecture）**：
  - **Proposal Network (P-Net)**：在每个集群内运行，完成局部RCL并生成集群级嵌入；
  - **Output Network (O-Net)**：将每个集群视为一个节点，构建跨集群图，执行全局RCA任务。

该设计实现了**分层推理（Hierarchical Reasoning）**，显著缩小了每阶段的搜索空间。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **可扩展性** | 推理时间几乎不随图规模增长而增加，实现“规模不变”（scale-invariant）推理 |
| **效率** | 显著降低计算复杂度，尤其适用于大型边缘部署场景 |
| **精度保持** | 在中等规模数据集上达到与集中式GNN相当的诊断准确率 |
| **实用性** | 同时输出“故障位置”（RCL）和“故障类型”（RCA），构成完整的“failure unit”，支持AIOps自动化响应 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **MicroCERCL** | 公开的真实世界基准数据集，包含81个微服务（来自SockShop、Hipster等4个应用），部署于云边协同环境。注入多种异常如CPU耗尽、内存泄漏、网络延迟等。共682个故障场景，图规模约~50节点。 |
| **iAnomaly** | 基于仿真框架生成的大规模合成数据集，支持从50到10,000节点的服务图生成。保留真实执行轨迹的时间与因果特性，用于评估可扩展性。 |

### 实验设置与评估指标

#### 评估任务
- **RCL（Root Cause Localization）**
  - 指标：`Acc@K`, `MAR`（Mean Average Rank）, `MRR`（Mean Reciprocal Rank）
- **RCA（Root Cause Analysis）**
  - 指标：`Accuracy`, `Precision`, `Recall`, `F1-score`

#### 模型配置
- **Baseline模型**：Centralized GNN（共享CNN/GCN编码器 + 双头输出）
- **Proposed模型**：Cascaded GNN（P-Net + O-Net + EdgeTemporalNet共享模块）
- 超参数通过 **Tree-structured Parzen Estimator (TPE)** 进行优化
- 所有实验在 **Spartan HPC集群** 上完成，使用PyTorch Geometric实现

#### 对比基线
- Centralized Joint RCL/RCA GNN
- 不同变体（消融实验中的随机聚类、非联合学习、GAT替换等）

---

## 3. 主要实验结果和性能指标

### MicroCERCL 数据集上的性能表现

#### ✅ Baseline验证（证明实现可靠）
| 模型 | Acc@1 (RCL) | F1-score (RCA) |
|------|------------|---------------|
| MicroCERCL原论文报告 | ~0.63 | — |
| 本文实现的Centralized GNN（单独训练） | 0.9203 | 0.8512 |
| **Joint Centralized GNN** | **0.9275** | **0.8715** |

> 表明本文baseline优于原始论文结果，具备公平比较基础。

#### 🔍 Cascaded vs Centralized（联合任务性能）
| 模型 | Acc@1 (RCL) | Acc@5 (RCL) | MAR | F1-score (RCA) |
|------|-------------|------------|-----|----------------|
| **Centralized Joint** | **0.9275** | **1.0000** | 1.1377 | **0.8715** |
| **Cascaded GNN** | 0.9130 | 0.9493 | 1.9058 | 0.8640 |

> 结论：**精度基本持平**，仅轻微下降，但在小图（~50节点）下尚未体现效率优势。

#### ⏱️ 推理延迟对比（MicroCERCL）
| 模型 | 平均诊断时间 |
|------|-------------|
| Centralized GNN | 17.779 ms |
| Cascaded GNN | 20.237 ms |

> 小规模图中因引入聚类开销，反而略慢，符合预期。

---

### iAnomaly 数据集上的可扩展性分析

#### 📈 随图规模扩大的推理时间趋势（见 Fig. 3）
| 模型 | 图大小从50 → 10,000节点的表现 |
|------|-------------------------------|
| **Centralized GNN** | 推理时间急剧上升，呈现明显非线性增长 |
| **Cascaded GNN（固定10个cluster）** | 增长放缓但仍上升（单cluster变大） |
| **Cascaded GNN（自适应clusters，~每20节点1 cluster）** | **推理时间基本恒定！** |

> ✅ **关键发现**：自适应聚类策略使推理延迟近乎与图规模无关，真正实现**高效可扩展诊断**。

---

### 消融实验（Ablation Study）

| 变体 | Acc@1 (RCL) ↓ | F1-score (RCA) ↓ | 分析 |
|------|----------------|------------------|------|
| **完整模型（Proposed）** | 0.9130 | 0.8640 | 基准 |
| 移除**联合学习**（Separate Models） | 0.8696 | 0.6692 | 性能大幅下降 → **共享表示至关重要** |
| 改为**随机聚类**（Random Clustering） | 0.7971 | 0.8512 | RCL严重退化 → **通信驱动聚类有效保留拓扑信息** |
| 替换GCN为**GAT层** | 0.8478 | 0.8346 | 精度下降且计算更重 → **GCN更适合本架构** |

> 结论：所有设计选择均有明确增益，尤其是**联合学习**对RCA影响最大。

---

## 4. 关键结论和发现

### 主要发现
1. **级联架构可在不牺牲太多精度的前提下极大提升可扩展性**：
   - 在中等规模图上，诊断精度与集中式GNN相当；
   - 在超大规模图上，推理时间保持稳定，远胜传统方法。

2. **通信驱动的聚类是成功的关键**：
   - 异常传播路径与通信依赖高度一致，因此聚类能有效保留关键上下文。

3. **联合学习显著增强诊断能力**：
   - RCL与RCA共享早期特征提取器，有助于捕捉跨任务的共性模式。

4. **真正的“可行动诊断”需要同时提供“where”和“what”**：
   - 输出“failure unit”（故障组件 + 故障类型）才能驱动AIOps自动修复策略。

---

### 方法的局限性
- **依赖高质量的标注数据**：作为监督学习方法，需大量带标签的故障样本训练；
- **聚类质量影响性能**：若通信图结构稀疏或噪声多，Louvain可能产生不合理分区；
- **当前为集中推理**：虽高效，但仍未完全去中心化，仍需将数据汇聚至推理节点。

---

### 未来工作方向
1. **分布式/联邦部署**：
   - 将P-Net部署在边缘设备本地，减少数据传输开销；
   - 使用**Federated Learning**协调各节点模型更新。

2. **改进聚类机制**：
   - 融合**colocation dependency**（共置依赖，如同主机部署）与通信依赖，提升物理感知能力。

3. **动态图支持**：
   - 当前假设静态图结构，未来可扩展至动态演化图（dynamic service graphs）。

4. **零样本/少样本RCA**：
   - 结合LLM或元学习技术，识别未见过的新型故障类型。

--- 

> ✅ **总体评价**：本文首次提出了面向边缘计算环境的**高效、可扩展、精准**的联合RCL/RCA框架，在保证诊断质量的同时解决了GNN在大规模系统中的瓶颈问题，为构建实用化的AIOps系统提供了重要技术路径。

</details>

---

### 4. [SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs](https://arxiv.org/abs/2603.00357)

**Authors**: Jin Lee, Zhonghao Chen, Xuhang He, Robert Underwood, Bogdan Nicolae, Franck Cappello, Xiaoyi Lu, Sheng Di, Zheng Zhang  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.00357v1  

#### Abstract
In large-scale LLM pre-training systems with 100k+ GPUs, failures become the norm rather than the exception, and restart costs can dominate wall-clock training time. However, existing fault-tolerance mechanisms are largely unprepared for this restart-dominant regime. To address this challenge, we pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SPARe: Stacked Parallelism with Adaptive Reordering for Fault-Tolerant LLM Pretraining Systems with 100k+ GPUs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在超大规模（10万+ GPU）的 LLM 预训练系统中，**故障已成为常态而非例外**。随着 GPU 数量增加，系统的平均无故障时间（MTBF）急剧下降，而每次全局重启（global restart）的成本极高，尤其是在同步 Data Parallelism 中，一次失败可能导致整个训练中断。传统容错机制如 **checkpointing** 主要缓解进度丢失后的重算开销，但在“重启主导”（restart-dominant）场景下，**系统宕机时间远超有效计算时间**，导致训练效率极低。

此外，传统复制（replication）虽能提高可用性，但其计算开销随冗余度 $ r $ 线性增长（$ r \times $），在高冗余时不可行。

### 提出了什么新方法或新思路
本文提出 **SPARe（Stacked Parallelism with Adaptive Reordering）**，一种新型容错框架，核心思想是：

- **不复制整个组的计算任务**，而是将数据分片（shards）以循环轮转方式跨并行组堆叠（stacked），形成冗余副本。
- 在每个训练步骤中，**动态决定何时触发 all-reduce** —— 即当所有类型的梯度都能被收集到时立即进行同步。
- 引入 **自适应重排序机制（Adaptive Reordering）**：通过一个控制器 **RECTLR**，在发生节点失效后，重新安排各组内堆栈的执行顺序，以最小化所需计算的堆栈数（all-reduce stack）。

该方法实现了“故障屏蔽”（failure masking），即在部分节点失效的情况下仍可继续训练，避免全局重启。

### 相比现有方法的优势
| 特性 | Checkpointing Only | Replication + CKPT | SPARe + CKPT |
|------|--------------------|---------------------|---------------|
| 容错能力 | 依赖恢复重算 | 冗余掩码失败 | 冗余掩码失败 |
| 计算开销 | 接近 1× | $ r \times $（线性增长） | **仅 2~3×**（近常量） |
| 可用性提升潜力 | 有限 | 高但代价大 | **高且代价小** |
| 与 checkpointing 兼容性 | 是 | 是 | **正交兼容** |

> ✅ **核心优势**：SPARe 能达到与传统 replication 相当的容错能力和系统可用性，但计算开销仅为 **2~3×**，即使在高冗余（如 $ r=20 $）下也几乎不变，显著优于 replication 的线性膨胀。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
本研究未使用真实训练数据集进行端到端训练，而是基于 **SimGrid** 构建了一个**离散事件模拟器（discrete-event simulator）** 来评估系统级性能。因此，实验中使用的“数据”为合成参数，符合现代 LLM 训练特征。

- 模型规模：**10T 参数（约 20TB FP16）**
- 数据分片大小：每 stack 包含 256M tokens（4×64M）
- 并行组数量 $ N $：{200, 600, 1000}

### 实验设置和评估指标
#### 系统配置（针对 600k H100 GPU 集群）
| 参数 | 设置 |
|------|------|
| MTBF（节点故障间隔） | 5 分钟（Weibull 分布，形状参数 $ k=0.78 $） |
| Global Restart Latency ($ T_r $) | 60 秒 |
| Checkpoint Save Time ($ T_s $) | 60 秒 |
| 每 stack 计算时间 ($ T_{\text{comp}} $) | 64 秒 |
| All-Reduce 时间 ($ T_a $) | 2–10 秒（随 $ N $ 增加） |
| 通信收缩（shrink）延迟 | 0.1 秒 |
| 事件抖动（jitter） | $ \times \mathcal{N}(1, 0.05^2) $ |

#### 评估指标
- **Time-to-train**：完成 10,000 步训练所需的总墙钟时间
- **Normalized Time-to-train / $ T_0 $**：相对于无故障情况的时间倍数
- **System Availability**：系统处于运行状态的比例
- **Average Computation Overhead**：平均每步需计算的 stack 数量
- **Endurable Failure Count**：首次 wipe-out 前可承受的平均失败次数

### 基线方法对比
- **CKPT-only**：仅使用 checkpointing，无冗余
- **Rep+CKPT**：传统 degree-$ r $ replication + checkpointing
- **SPARe+CKPT**：本文方法 + checkpointing（联合优化）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| $ N $ | 方法 | Min Time-to-train / $ T_0 $ | Availability | Optimal $ r^* $ | Gain vs Rep+CKPT |
|-------|--------|-------------------------------|--------------|------------------|------------------|
| 200 | Rep+CKPT | 6.07 | 61.74% | 3 | — |
|     | SPARe+CKPT | **2.92** | **87.00%** | 9 | **51.9%** |
| 600 | Rep+CKPT | 4.27 | 79.89% | 3 | — |
|     | SPARe+CKPT | **2.49** | **93.90%** | 8 | **41.7%** |
| 1000 | Rep+CKPT | 3.88 | 84.41% | 3 | — |
|      | SPARe+CKPT | **2.34** | **96.54%** | 9 | **39.6%** |

> 🔺 **平均节省 40~50% 的训练时间**

### 与基线方法的对比结果
- **Rep+CKPT** 在低冗余（$ r=3 $）时表现最佳，但更高冗余会导致计算开销线性上升，反而恶化 time-to-train。
- **SPARe+CKPT** 在高冗余下依然保持高效，因其计算开销稳定在 **2~2.8×**，远低于 replication 的 $ r \times $ 开销。
- 在 $ N=600 $、$ r=20 $ 时，SPARe 可平均容忍 **426 次故障** 才发生 wipe-out，而计算开销仅 **2.34×**。

### 消融实验结果（理论验证）
- **公式准确性验证**：
  - 平均可容忍故障数 $ u(N,r) $ 的闭式表达误差 < **1.13%**
  - 平均计算开销 $ S(N,r) $ 的预测误差 < **0.60%**
  - 仿真与理论高度一致（相关系数 > 0.996）
- **低冗余性能下降分析**：
  - 在 $ r=2 $ 附近，SPARe 表现略差于预期，原因是 Weibull 分布下早期故障累积更快，导致更频繁的全局重启。
  - 作者指出可通过 **dynamic checkpointing** 改善此问题。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **在“重启主导”时代，避免全局重启是提升可用性的最直接路径**。
2. **SPARe 成功解耦了“容错能力”与“计算开销”的强绑定关系**，实现了高可用性下的近恒定计算成本。
3. 自适应重排序机制（via RECTLR）可在多项式时间内求解最优执行计划，实际开销可忽略（< 0.1s）。
4. SPARe 与任何 checkpointing 方案正交兼容，适合集成进现有训练框架（如 PyTorch、NCCL）。

### 方法的局限性
- 当前模型假设节点故障独立且均匀分布，现实中可能存在相关性（如电源故障影响整机架）。
- 对 Weibull 分布下早期高失效率敏感，在极低冗余时性能不如理论预期。
- 依赖 communicator shrinking 和 runtime 重建支持，需底层通信库（如 NCCL）配合。

### 未来工作方向
- 结合 **dynamic checkpointing** 进一步优化在非指数故障分布下的表现。
- 将 SPARe 扩展至 **异构集群** 和 **混合精度训练** 场景。
- 探索与 **pipeline parallelism** 和 **tensor parallelism** 的协同优化。
- 实现原型系统并在真实超算平台上部署验证。

---

> 💡 **总体评价**：SPARe 是面向下一代十万级 GPU LLM 预训练系统的**关键基础设施创新**。它通过精巧的堆叠与重排序策略，在不牺牲容错能力的前提下，将传统复制带来的高昂计算代价降至最低，为实现 **>90% 系统可用性** 和 **可持续的大模型训练** 提供了切实可行的技术路径。

</details>

---

### 5. [Optimizing In-Context Demonstrations for LLM-based Automated Grading](https://arxiv.org/abs/2603.00465)

**Authors**: Yucheng Chu, Hang Li, Kaiqi Yang, Yasemin Copur-Gencturk, Kevin Haudek, Joseph Krajcik, Jiliang Tang  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.00465v1  

#### Abstract
Automated assessment of open-ended student responses is a critical capability for scaling personalized feedback in education. While large language models (LLMs) have shown promise in grading tasks via in-context learning (ICL), their reliability is heavily dependent on the selection of few-shot exem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimizing In-Context Demonstrations for LLM-based Automated Grading

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Models (LLMs)** 的自动评分系统依赖于 **In-Context Learning (ICL)**，其性能高度敏感于所选的 few-shot 示例（exemplars）及其对应的 rationale（推理过程）。然而，传统方法存在以下问题：

- **标准检索策略**（如基于语义相似度的 KNN）倾向于选择表面相似但未必能反映评分边界（decision boundary）的示例，无法有效区分相邻分数等级之间的细微差异。
- 手动编写高质量的 **expert rationales** 成本高昂，难以规模化。
- 现有优化方法（如 BRIDGE）关注全局准确率，忽视了对“边界案例”（borderline cases）的建模，导致在实际教学场景中鲁棒性不足。

### 🚀 提出的新方法：GUIDE
作者提出 **GUIDE**（Grading Using Iteratively Designed Exemplars），一个将 exemplar 选择与优化重构为“边界定义问题”的框架。其核心思想是：

- 将 ICL 中的演示集构建视为一种 **boundary-focused optimization** 任务，而非简单的代表性样本选取。
- 引入两个关键机制：
  1. **Contrastive Operators**：主动识别并引入“边界对”（boundary pairs）——即语义高度相似但得分不同的学生回答对，以强化模型对评分标准边界的理解。
  2. **Discriminative Rationale Generation**：通过提示 LLM 生成对比性推理文本，明确解释为何某回答得分为 *y* 而非 *y−1* 或 *y+1*，从而增强决策依据的可解释性和判别力。

### 🔍 相比现有方法的优势
| 方面 | GUIDE 的优势 |
|------|-------------|
| **目标导向** | 不再追求泛化准确率最大化，而是聚焦于最难判断的“边缘情况”，提升 rubric adherence（评分标准一致性） |
| **自动化程度** | 自动合成高质量 discriminative rationales，显著减少人工标注负担 |
| **上下文效率** | 在仅 4–16 个 exemplars 的小 context window 内实现高性能，适合部署 |
| **泛化能力** | 在多个学科领域（物理、化学、教师教育）均表现优异，具备跨域适应性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
论文在三个真实教育数据集上进行验证，涵盖科学教育与教师培训场景：

| 数据集 | 领域 | 样本数 | 分数等级 | 是否含专家 rationale |
|-------|------|--------|----------|------------------|
| **Dr** | 物理（电学交互） | 314 | {0,1}（二元） | 否 |
| **Dc** | 化学（3DLP 框架） | ~163–184 | {0,1,2}（有序） | 是（每类 3 条） |
| **Dt** | 教师教育（数学教学知识） | ~229–236 | {0,1,2}（有序） | 是（每类 3–5 条） |

所有数据集划分为 train:valid:test = 3:1:1。

### ⚙️ 实验设置
- **主干模型**：GPT-4o-mini（用于 grading、rationale generation 和 embedding）
- **嵌入模型**：text-embedding-3-small（计算语义相似度）
- **优化轮次**：T = 5 轮迭代
- **Bayesian Optimization 参数**：
  - 每轮评估候选子集数量：neval = 32
  - 候选池大小：256
  - 最大全局 exemplar pool 大小：Nmax = 512
  - 边界对相似度阈值：sim ≥ 0.7
- **Demonstration Set Size**：限制在 [4, 16] 范围内
- **温度参数**：generation 温度设为 0.2，确保稳定性

### 📊 评估指标
采用多维度指标全面评估性能：

| 指标 | 描述 |
|------|------|
| **Accuracy** | 完全匹配的真实得分比例 |
| **Quadratic Weighted Kappa (QWK)** | 衡量序数变量的一致性，对偏离越远的错误惩罚越重 |
| **Adjacent Error Rate (AdjErr)** | 预测偏移 ±1 分的比例（如真值为1预测为2），反映边界模糊问题 |
| **Non-Adjacent Error Rate (NonAdjErr)** | 预测跳级错误（如真值为2预测为0），反映严重逻辑失误 |

### 🔁 基线方法对比
共比较五种典型 exemplar selection 方法：

| 基线方法 | 类型 | 简介 |
|--------|------|------|
| **NAIVE** | 固定 | 使用原始提供的 few-shot 示例，无优化 |
| **RANDOM** | 固定 | 随机从训练集中抽取固定数量 exemplars |
| **KNN SBERT** | 动态 | 对每个查询动态检索最相似的 k 个示例（基于 SBERT + cosine） |
| **Vote-K** | 固定 | 选择语义多样性最大的 k 个示例，覆盖更广空间 |
| **BRIDGE** | 优化 | 当前最先进的 optimize-generate 方法，以验证集准确率为优化目标，但不关注边界 |

---

## 3. 主要实验结果和性能指标

### 📈 总体性能对比（见 Table 2）

| 方法 | Dr (Acc/QWK) | Dc (Acc/QWK) | Dt (Acc/QWK) |
|------|---------------|--------------|--------------|
| NAIVE | 0.74 / 0.42 | 0.69 / 0.39 | 0.59 / 0.54 |
| RANDOM | 0.75 / 0.43 | 0.58 / 0.26 | 0.52 / 0.52 |
| KNN SBERT | 0.78 / 0.44 | 0.58 / 0.26 | 0.52 / 0.52 |
| BRIDGE | 0.90 / 0.57 | 0.76 / 0.53 | 0.66 / 0.65 |
| **GUIDE (Ours)** | **0.92 / 0.62** | **0.80 / 0.59** | **0.71 / 0.67** |

✅ **关键发现**：
- GUIDE 在所有数据集上均取得 **最高 Accuracy 和 QWK**，尤其在复杂任务 Dt 上相对 NAIVE 提升约 **20% 准确率**。
- 即使优于先进方法 BRIDGE，GUIDE 仍能进一步提升 4–5 个百分点。

### 🎯 边界错误分析（AdjErr 显著降低）

| 方法 | Dr (AdjErr) | Dc (AdjErr) | Dt (AdjErr) |
|------|-------------|-------------|-------------|
| NAIVE | 0.26 | 0.31 | 0.37 |
| BRIDGE | 0.19 | 0.24 | 0.32 |
| **GUIDE** | **0.08** | **0.20** | **0.28** |

📌 **结论**：
- GUIDE 将 **Adjacent Error Rate 降低最多达 69%**（Dr 上从 0.26 → 0.08），说明其在处理“模棱两可”的边缘案例方面具有压倒性优势。
- NonAdjErr 始终保持极低水平（≤0.02），表明 GUIDE 并未牺牲整体稳定性来换取边界精度。

### 🔍 消融实验与关键组件分析（文中隐含分析）
虽然未单独列出消融表，但从设计逻辑和讨论部分可推断：

| 组件 | 作用 | 支持证据 |
|------|------|---------|
| **Contrastive Operators** | 引导搜索朝向 boundary pairs | BO 过程中 Contrastive Density 作为优化目标之一，直接影响最终选择 |
| **Discriminative Rationale Generation** | 提供更强的学习信号 | Phase 2 利用最优 E* 重新生成 rationale，形成正反馈循环；图2展示 rationale 明确提及“为什么不是其他分数” |
| **Tchebycheff Scalarization** | 多目标平衡（accuracy, sparsity, contrastive） | 避免手动调参，探索 Pareto 前沿，提升鲁棒性 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **边界案例是自动评分的关键瓶颈**  
   多数误判发生在相邻分数之间，而非极端错误。因此，提升模型对 **rubric decision boundaries** 的敏感度比提高整体准确率更重要。

2. **“对比性学习”优于“代表性学习”**  
   GUIDE 成功证明：有效的 ICL 示例不应只是“典型代表”，而应是“反例对照”。通过提供 **semantically similar but differently scored pairs**，模型能更好掌握评分标准的本质区别。

3. **自动生成 discriminative rationale 可替代专家标注**  
   通过 contrastive infilling prompt（如：“解释为什么这是1分而不是0或2分”），LLM 能生成媲美甚至超越人工撰写的 rationale，极大缓解冷启动问题。

4. **小规模高质量 context > 大规模普通 context**  
   GUIDE 仅需 4–16 个精心挑选的 exemplars 即可达到最优性能，说明 **context quality 远重于 quantity**。

### ⚠️ 局限性
- **依赖初始训练数据质量**：若训练集中缺乏真正的边界案例，GUIDE 可能难以生成有效 contrastive pairs。
- **计算成本较高（训练阶段）**：尽管推理轻量，但 Bayesian Optimization 涉及多次 LLM 查询，在大规模场景下可能耗时较长。
- **目前仅适用于文本输入**：尚未扩展到图像、公式等多模态作答形式（如手写解题步骤、电路图绘制等）。

### 🔮 未来工作方向
- 扩展至 **multimodal automated grading**，例如结合 diagram parsing 与 textual reasoning。
- 探索 **real-time adaptive exemplar selection**，根据不同学生群体动态调整 context。
- 将 GUIDE 框架应用于 **formative feedback generation**，不仅给出分数，还能生成个性化改进建议。
- 研究如何将 human-in-the-loop 机制融入迭代流程，实现人机协同优化。

---

## 总结
> **GUIDE 通过将 exemplar selection 重构为 boundary-focused optimization 问题，首次系统性地解决了 LLM-based automated grading 中“边缘案例难判”的核心挑战。其实验充分证明：聚焦于“语义相近但评分不同”的 contrastive pairs，并辅以 discriminative rationale generation，能够显著提升 rubric adherence 与评分鲁棒性，为可信、可扩展的 AI 教育评估系统铺平道路。**

</details>

---

### 6. [HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents](https://arxiv.org/abs/2603.00977)

**Authors**: Hongbo Jin, Rongpeng Zhu, Jiayu Ding, Wenhao Zhang, Ge Li  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.00977v1  

#### Abstract
Large language model (LLM) agents have recently demonstrated strong capabilities in interactive decision-making, yet they remain fundamentally limited in long-horizon tasks that require structured planning and reliable execution. Existing approaches predominantly rely on flat autoregressive policies...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **LLM Agent** 在长视野任务（long-horizon tasks）中面临三大挑战：
- **指数级探索复杂度**（exponential exploration complexity）
- **延迟奖励分配**（delayed credit assignment）
- **语义漂移**（semantic drift），即推理过程中逐渐偏离原始目标

现有方法多采用“扁平化”（flat）的自回归策略，将高层推理与低层动作生成混在同一 token 序列中，导致错误传播严重、探索效率低下。

---

### 提出了什么新方法或新思路
本文提出 **HiMAC**（Hierarchical Macro-Micro Agentic Control），一种分层强化学习框架，通过显式解耦决策过程为两个层级：

- **Macro-Policy（规划器）**：负责生成结构化的自然语言子目标序列（structured blueprint），作为全局战略计划。
- **Micro-Policy（执行器）**：在蓝图指导下，逐段生成可执行的原子动作。

该框架将原本联合的“推理-行动”空间分解为两个独立优化路径，显著降低搜索空间维度，并限制错误传播范围。

---

### 相比现有方法的优势
- ✅ **更高效的探索**：通过分层抽象减少组合爆炸。
- ✅ **更强的鲁棒性**：执行阶段的错误被限制在单个子目标内，不会影响整个任务流程。
- ✅ **无需 Critic 的稳定训练**：引入 **Critic-Free Hierarchical Policy Optimization** 和 **Iterative Co-Evolution Training**，避免高维稀疏语言空间中 value network 学习不稳定的问题。
- ✅ **自发涌现高级行为**：如自我验证（self-verification）、适应性终止等。

---

## 2. 核心实验方法和设置

### 使用的数据集
在三个具有代表性的长视野基准上进行测试：

| 数据集 | 任务类型 | 特点 |
|-------|--------|------|
| **ALFWorld** | 多步具身推理（embodied reasoning） | 模拟家庭环境中完成物品操作任务（如加热某物） |
| **WebShop** | 长视野网页导航与购物 | 在噪声大、观测复杂的电商网站中定位并购买指定商品 |
| **Sokoban** | 视觉接地的空间规划 | 推箱子游戏，需精确顺序推理，支持视觉输入 |

---

### 实验设置和评估指标
- **主干模型**：
  - 文本任务：`Qwen2.5-Instruct`（1.5B / 7B）
  - 视觉任务：`Qwen2.5-VL`, `Qwen3-VL`
- **训练机制**：
  - 使用 **Group-based RL** 范式（类似 GRPO），无 Critic。
  - 采用 **Hierarchical Relative Advantage Estimation** 进行两级优势估计。
  - 实施 **Iterative Co-Evolution Training**：交替更新 Macro 与 Micro 策略。
- **评估指标**：
  - **Success Rate (%)**：任务成功完成的比例
  - **Score**：综合得分（如 WebShop 中的价格折扣率）
  - **Sample Efficiency**：达到特定成功率所需的训练迭代次数

---

### 基线方法对比
涵盖以下几类主流方法：

#### Prompting 方法
- ReAct
- Reflexion

#### 强化学习方法
- PPO
- RLOO
- GRPO
- GiGPO（最新 critic-free 多轮 RL 方法）

#### 闭源模型参考
- GPT-4o
- Gemini-2.5-Pro
- Claude Sonnet 4.5

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

| 方法 | ALFWorld (7B) | WebShop Success (7B) | WebShop Score (7B) | Sokoban Success (7B) |
|------|----------------|-------------------------|------------------------|------------------------|
| **HiMAC (Ours)** | **92.1%** | **84.1%** | **93.8** | **87.5%** |
| GiGPO | 90.8% | 75.2% | 86.2 | 82.8% |
| GRPO | 77.6% | 65.7% | 79.3 | 83.9% |
| PPO | 80.4% | 68.7% | 81.4 | — |
| ReAct | 31.2% | 19.5% | 46.2 | — |

> 💡 **最大提升出现在 WebShop 上：相比最强 RL 基线 GiGPO 提升 16.0% 成功率（67.4% → 83.4%，1.5B 模型下）**

---

### 与基线方法的对比结果
- 在所有任务上均取得 **state-of-the-art 性能**。
- 即使使用较小模型（1.5B），HiMAC 的表现也远超闭源强模型（如 Gemini-2.5-Pro 在 ALFWorld 上仅 60.3%）。
- 在 **WebShop** 这种易发生 context drift 的任务中优势尤为明显，说明 structured blueprint 有效防止了目标丢失。

---

### 消融实验结果（Table 3）

| 变体 | ALFWorld ↓ | WebShop Score ↓ | WebShop Succ. ↓ |
|------|------------|------------------|------------------|
| **HiMAC (Full)** | 92.1 | 93.8 | 84.1 |
| w/o Hierarchy (Flat GRPO) | 77.6 (-14.5) | 79.3 (-14.5) | 66.1 (-18.0) |
| w/o Iterative Co-Evolution | 85.3 (-6.8) | 86.7 (-7.1) | 74.8 (-9.3) |
| w/o `<sub_done>` token | 88.2 (-3.9) | 90.1 (-3.7) | 79.8 (-4.3) |
| Random Blueprint | 89.7 (-2.4) | 91.6 (-2.2) | 81.5 (-2.6) |

> 🔍 结论：
> - 分层结构本身带来最大收益（-14.5%），证明其必要性。
> - **Iterative Co-Evolution** 对稳定性至关重要，尤其在长任务中（WebShop 下降 9.3%）。
> - `<sub_done>` 机制允许动态控制节奏，优于固定步数预算。
> - 高质量蓝图对 Micro-Policy 训练有显著正向引导作用。

---

## 4. 关键结论和发现

### 主要发现
1. **结构化先于规模**：相较于单纯扩大模型参数，引入 **hierarchical inductive bias** 是实现稳健长视野智能的关键。
2. **分层设计提升样本效率**（见 Table 4）：
   - HiMAC 在更少训练迭代中达到目标成功率（如 WebShop 达到 65% 仅需 ~220 轮 vs GiGPO 的 ~230，GRPO 的 ~380）。
3. **涌现能力**：
   - Planner 在后期训练中自发学会添加 “Inventory or look to confirm” 类似自我验证步骤，形成闭环控制。
4. **可扩展性强**：
   - 从 1.5B 到 7B 模型，性能持续提升，且增益正交于模型大小。

---

### 方法的局限性
- 当前蓝图仍依赖自然语言表示，可能存在歧义或冗余。
- 所有实验基于模拟环境，真实世界部署尚待验证。
- 蓝图生成与执行共享同一 LLM 参数，可能限制两者的专业化程度。

---

### 未来工作方向
- 将 HiMAC 应用于更开放、动态的真实环境（如安卓设备控制、机器人操作）。
- 探索跨任务、跨领域的 **blueprint 迁移与复用**。
- 引入更丰富的中间状态监督信号以进一步提升规划质量。

---

> ✅ **最终结论**：  
> **HiMAC 表明，通过构建 Macro-Micro 分层架构并辅以稳定的 critic-free 优化机制，可以显著提升 LLM Agent 在长视野任务中的性能、鲁棒性和样本效率。这标志着从“更大模型”向“更好结构”的范式转变，是迈向真正自主代理的重要一步。**

</details>

---

### 7. [Scalable Gaussian process modeling of parametrized spatio-temporal fields](https://arxiv.org/abs/2603.00290)

**Authors**: Srinath Dama, Prasanth B. Nair  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.00290v1  

#### Abstract
We introduce a scalable Gaussian process (GP) framework with deep product kernels for data-driven learning of parametrized spatio-temporal fields over fixed or parameter-dependent domains. The proposed framework learns a continuous representation, enabling predictions at arbitrary spatio-temporal co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Scalable Gaussian process modeling of parametrized spatio-temporal fields 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**高维参数化时空场（parametrized spatio-temporal fields）建模中的可扩展性挑战**。传统 Gaussian Process (GP) 回归在处理来自高保真仿真（如 PDE 求解器）的大规模时空数据时面临严重瓶颈，其计算复杂度为 $O(n^3)$，其中 $n$ 是训练样本数，这使得直接应用不切实际。

此外，现有的算子学习方法（如 FNO、DeepONet）虽然灵活，但通常是确定性的，缺乏对预测不确定性的量化能力，而这对下游任务（如贝叶斯反演、优化）至关重要。

### 提出的新方法与创新点
作者提出了一种**基于深度乘积核（deep product kernel, DPK）的可扩展 GP 框架**，其核心创新包括：

- **深度乘积核 (Deep Product Kernel, DPK)**：将输入分解为参数 $\mu$、空间坐标 $x$ 和时间 $t$，并为每个分量设计一个由神经网络驱动的“深核”（deep kernel）。具体形式为：
  $$
  k_{\text{DPK}}(z, z') = k_\mu(G_\mu(\mu), G_\mu(\mu')) \times k_x(G_x(x), G_x(x')) \times k_t(G_t(t), G_t(t'))
  $$
  这种结构允许模型捕捉跨参数、空间和时间的非线性相关性，同时通过乘积结构引入**Kronecker 结构**以实现高效计算。

- **Kronecker 矩阵代数加速**：当训练数据位于笛卡尔网格（Cartesian grid）上时，协方差矩阵具有 Kronecker 分解形式 $K_z = K_u \otimes K_x \otimes K_t$。利用此性质，作者实现了训练和推理的计算复杂度从 $O((NMN_t)^3)$ 降低到**近似线性于空间网格点数 $M$**，即 $O(N^3 + dM^{3/d} + N_t^3)$，极大地提升了可扩展性。

- **Gappy-Grid 扩展以处理非结构化网格**：针对真实世界中常见的非结构化空间网格，提出了一种“gappy-grid”方法。将原始数据嵌入到一个更大的规则背景网格中，并将物理域外的位置视为“缺失值”（gaps）。通过引入伪观测值（pseudovalues），证明了可以在保持完整 Kronecker 结构的前提下，精确求解原问题的系数。

- **高效的后验方差计算**：
  - 对于笛卡尔网格，可以**精确且高效地计算后验方差**，成本与后验均值相当。
  - 对于非结构化网格，提出了**严格的理论上下界**来估计后验方差，这些界限同样可以通过 Kronecker 代数高效计算，从而实现了可扩展的不确定性量化。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **计算效率** | 复杂度近似线性于空间点数，远优于标准 GP 的立方复杂度。 |
| **不确定性量化** | 提供自然的概率输出和校准良好的置信区间，这是大多数算子学习方法所缺乏的。 |
| **灵活性与表达力** | DPK 能够捕捉复杂的非平稳性和非线性依赖关系，超越了传统的平稳核或简单乘积核。 |
| **适用范围广** | 可处理固定域、参数化几何域以及结构化/非结构化网格。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖了多个流体力学和固体力学领域的基准问题，包括：
- **1D 非定常 Burgers' 方程**：用于与投影型降阶模型（如 POD-Galerkin）进行比较。
- **超弹性材料问题 (Hyper-elastic problem)**：涉及中心有孔洞的单位域上的应力场预测，数据位于 O 型非结构化网格上。
- **2D 跨音速流 (Transonic flow)**：围绕参数化翼型（NACA0012 变形）的速度场预测。
- **Navier-Stokes 方程（管道流）**：参数化管道几何下的水平速度场预测。

所有数据均来自公开文献 [56] 或通过有限元/有限体积法生成。

### 实验设置和评估指标
- **评估指标**：采用 **$\ell^2$ 相对测试误差**：
  $$
  \text{Relative Test Error} = \frac{\|\mathbf{u}_h(\mu^{(i)}) - \hat{\mathbf{u}}(\mu^{(i)})\|_2}{\|\mathbf{u}_h(\mu^{(i)})\|_2}
  $$
  其中 $\mathbf{u}_h$ 为高保真解，$\hat{\mathbf{u}}$ 为预测解。

- **参数预处理**：对于高维参数（如翼型形状），先使用 PCA 将其降至低维潜空间（如前 6–20 个主成分）再作为输入。

- **训练/测试划分**：
  - Burgers' 问题：80 个快照训练，2 个特定参数测试。
  - 其他问题：1000 个快照训练，200 个快照测试。

- **实现细节**：使用 Adam 优化器最小化负对数边缘似然（NLML）以学习超参数；神经网络架构为三层全连接网络（1000-500-50）。

### 基线方法对比
- **算子学习方法**：
  - **Fourier Neural Operator (FNO)**
  - **DeepONet**
  - **Geo-FNO**（专为一般几何设计的 FNO 变体）
- **物理驱动降阶模型**：
  - **POD-Galerkin**
  - **POD-LSPG**
  - **Deep-Galerkin / Deep-LSPG**

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 1D Burgers' 问题（见 Table 1）
| 方法 | 测试误差 ($\mu_1$) | 测试误差 ($\mu_2$) |
|------|------------------|------------------|
| **Proposed GP (DPK-Matern-5/2)** | **0.0033** | **0.0029** |
| Deep-LSPG (p=20) | 0.0009 | 0.001 |
| Deep-Galerkin (p=20) | 0.015 | 0.013 |
| POD-Galerkin (p=20) | 0.032 | 0.03 |

- **结论**：提出的 GP-DPK 方法显著优于所有投影型 ROM（POD-Galerkin, POD-LSPG, Deep-Galerkin），精度接近但略低于最先进的 Deep-LSPG。相比标准 Matérn 核 GP，性能大幅提升。

#### 超弹性问题（见 Table 2）
| 方法 | 训练误差 | **测试误差** |
|------|--------|------------|
| **Proposed GP (DPK)** | 0.0181 | **0.0326** |
| Geo-FNO (O-mesh) | 0.0344 | 0.0363 |
| FNO interpolation | 0.0314 | 0.0508 |
| DeepONet | 0.0528 | 0.0965 |

- **结论**：在非结构化 O 型网格上，GP-DPK 在测试误差上**优于 FNO 和 DeepONet**，并**略优于 Geo-FNO**，展示了其在复杂几何上的强大建模能力。

#### 2D 跨音速流与管道流（见 Table 3）
| 方法 | 翼型测试误差 | 管道测试误差 |
|------|-------------|-------------|
| **Proposed GP (DPK)** | **0.0142** | 0.0131 |
| Geo-FNO | 0.0138 | **0.0067** |
| FNO interpolation | 0.0421 | 0.0151 |

- **结论**：在翼型流动中，GP-DPK 性能与 Geo-FNO 相当且优于 FNO；在管道流中，Geo-FNO 表现最佳，但 GP-DPK 仍优于 FNO 和 UNet。

### 消融实验结果
- **不同核函数对比**：在多个任务中，使用 **DPK 的 GP 显著优于使用标准 Matérn-5/2 乘积核的 GP**，验证了深度特征映射带来的表达力提升。
- **数据量影响研究（附录 B.3）**：随着训练快照数量增加（从 50 到 5000），GP-DPK 的测试误差持续下降，在足够多数据下**最终超过了物理驱动的 POD-Galerkin 方法的精度**，表明其强大的数据驱动潜力。

---

## 4. 关键结论和发现

### 主要发现
1. **可扩展 GP 是可行且有效的**：通过深度乘积核与 Kronecker 代数结合，成功构建了一个能够处理百万级时空网格点的 GP 模型，解决了传统 GP 的可扩展性难题。
2. **高精度与不确定性量化兼得**：该方法在多个基准问题上达到了与先进算子学习方法（FNO, DeepONet, Geo-FNO）**相当甚至更优的精度**，同时**天然提供高质量的不确定性估计**。
3. **通用性强**：框架统一处理了结构化/非结构化网格、固定/参数化域等复杂场景，具有广泛的适用性。
4. **数据驱动替代方案**：在 1D Burgers' 问题上，其性能超越了多种投影型降阶模型，为纯数据驱动的代理建模提供了强有力工具。

### 方法的局限性
- **分离性假设**：DPK 假设协方差在参数、空间、时间之间是可分离的，这可能限制其对强耦合交互作用的建模能力。
- **嵌入误差**：将非结构化网格数据映射到规则背景网格会引入插值误差。
- **gappy-grid 求解效率**：尽管使用共轭梯度法（CG），求解伪值 $y_g$ 仍需迭代，可能成为性能瓶颈。

### 未来工作方向
1. **放松分离性假设**：开发部分不可分离的核（partially non-separable kernels），例如通过低秩修正或耦合选定维度。
2. **减少嵌入误差**：改进非结构化网格到规则网格的映射策略，降低插值带来的信息损失。
3. **提升 gappy-grid 求解效率**：探索更好的预条件子（preconditioners）以加速 CG 收敛。
4. **扩展至更高维问题**：进一步优化算法以应对三维时空场等更大规模问题。

</details>

---

### 8. [Energy-Efficient Information Representation in MNIST Classification Using Biologically Inspired Learning](https://arxiv.org/abs/2603.00588)

**Authors**: Patrick Stricker, Florian R\"ohrbein, Andreas Knoblauch  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.00588v1  

#### Abstract
Efficient representation learning is essential for optimal information storage and classification. However, it is frequently overlooked in artificial neural networks (ANNs). This neglect results in networks that can become overparameterized by factors of up to 13, increasing redundancy and energy co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Energy-Efficient Information Representation in MNIST Classification Using Biologically Inspired Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前深度神经网络（DNNs）普遍存在**过参数化**（overparameterization）问题，尤其是在使用 Backpropagation（BP）进行训练时。这种现象导致模型冗余度高、存储效率低、能耗大，尤其在大规模语言模型（LLMs）快速发展的背景下，引发了严重的环境与伦理问题。

此外，传统方法如 dropout 和 skip connection 只是掩盖了模型膨胀的后果，并未从根本上优化信息表示效率。

### 🚀 提出的新方法与新思路
作者提出了一种**受生物启发的学习框架**（biologically inspired learning rule），其核心机制包括：
- **竞争性兴奋性 Hebbian 塑性**（Competitive Excitatory Hebbian Plasticity）
- **非负权重约束**（nonnegativity constraints）
- **权重扰动学习**（Weight Perturbation, WP）
- **偏置神经元上的稳态可塑性**（homeostatic plasticity）

该方法模拟大脑中的**结构可塑性**（structural plasticity），即通过动态生成或修剪突触来优化连接结构，仅保留对分类任务至关重要的突触。

特别地，该框架将 MNIST 分类任务重新定义为一个**异联想记忆问题**（heteroassociative memory task），并构建了一个两层的关联网络（associative network）进行建模。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **信息效率** | 显著提升 mutual information 压缩能力，减少冗余信息存储 |
| **存储效率** | 使用更少的非静默突触（nonsilent synapses）实现高效记忆编码 |
| **能源效率** | 减少活跃连接数，降低计算与信号传输能耗 |
| **无需架构预优化** | 自动调节网络复杂度，避免人工调参 |
| **适应性强** | 动态保留“空间”供未来学习新记忆，模仿大脑持续学习能力 |

相比标准 BP 和 Chorowski 的约束型 BP 方法，本方法在**synaptic capacity**（突触容量）上表现最优。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **MNIST 手写数字数据集**
  - 图像尺寸：28×28 灰度图
  - 总样本量：70,000 张（60,000 训练 + 10,000 测试）
  - 实验聚焦于子集：数字 `1`, `2`, `6`

> 注：文中也提及 FashionMNIST 的相关研究支持该范式的有效性。

### ⚙️ 实验设置
- **网络结构**：单隐藏层前馈网络（one-hidden-layer feedforward network）
- **激活函数**：修改后的 Sigmoid 函数，输出范围映射至 `[0.0, 1.0]` 以匹配非负处理需求
- **训练方式**：
  - 隐藏层采用公式 (1) 更新权重（基于局部 Hebbian 规则）
  - 输出层结合 Hebbian 与 Weight Perturbation（公式 2）
  - 偏置更新采用 homeostatic 规则（公式 4）
- **初始化**：权重从均匀分布 U(0.01, 0.1) 初始化
- **损失函数**：交叉熵损失（cross-entropy loss）
- **框架实现**：Keras + TensorFlow 后端
- **硬件平台**：本地 RTX 4080 进行原型开发；最终实验运行于 bwUniCluster 2.0 上的 A100/H100/P100 GPU

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Test Accuracy** | 测试准确率，衡量分类性能 |
| **Mutual Information I(X;Z)** | 输入 X 与潜在变量 Z 之间的互信息，反映信息压缩程度 |
| **Synaptic Capacity $ C_S $** | 定义为 $ C_S = \frac{I(Z;X)}{\text{Number of nonsilent synapses}} $（单位：bits/synapse），用于量化每单位突触的信息存储效率，是核心评价指标 |

### 🆚 基线方法对比
- **Backpropagation (BP)**：标准反向传播算法
- **Constrained BP (Chorowski et al. [17])**：施加非负性和稀疏性约束的 BP 方法
- **本文方法（Authors）**：所提出的生物启发式学习规则

所有方法均在同一条件下复现以确保可比性。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 2）

| 隐藏神经元数 | 方法 | Test Accuracy | I(X;Z) (bits) | $ C_S $ (bits/synapse) |
|-------------|-------|----------------|----------------|----------------------------|
| 10          | BP | 99.01% | 22.50 | 2.87×10⁻³ |
|             | Chorowski [17] | 98.34% | 16.10 | 1.08×10⁻² |
|             | **Authors** | **64.29%** | **12.80** | **1.63×10⁻²** ✅ |
| 30          | BP | 99.17% | 127.62 | 5.43×10⁻³ |
|             | Chorowski [17] | 99.01% | 72.99 | 2.22×10⁻² |
|             | **Authors** | **86.21%** | **52.18** | **6.66×10⁻²** ✅ |
| 100         | BP | 99.23% | 435.93 | 5.56×10⁻³ |
|             | Chorowski [17] | 99.10% | 217.43 | 5.93×10⁻² |
|             | **Authors** | **89.79%** | **198.33** | **2.53×10⁻¹** ✅ |
| 200         | BP | 99.17% | 518.79 | 3.31×10⁻³ |
|             | Chorowski [17] | 99.10% | 402.91 | 1.31×10⁻¹ |
|             | **Authors** | **95.55%** | **372.66** | **4.75×10⁻¹** ✅ |

> ✅ 表示在对应列中表现最佳

### 🔬 对比分析
- **准确性方面**：BP 和 Constrained BP 在精度上略优于本文方法，尤其在小规模网络中差距明显。
- **信息压缩与效率方面**：
  - 本文方法始终具有最低的 **I(X;Z)**，说明其有效抑制了冗余信息存储。
  - 尽管准确率稍低，但在所有配置下实现了最高的 **$ C_S $**（突触容量），表明其**单位突触的信息利用率最高**。
- **随网络增大趋势**：
  - 当隐藏层扩大时，本文方法的性能差距显著缩小，且 $ C_S $ 持续增长，而 BP 的 $ C_S $ 反而下降，显示出严重过参数化倾向。

### ❌ 消融实验（Ablation Study）
虽然文中未明确列出消融实验表格，但从机制设计与讨论部分可推断以下关键要素的作用：
- **Weight Perturbation (WP)**：使系统能基于性能反馈调整权重方向，增强鲁棒性
- **Nonnegativity Constraint**：促进稀疏性，符合生物学合理性
- **Homeostatic Bias Update**：稳定学习过程，提高类别区分能力
- **Modified Sigmoid Activation**：解决了早期版本中难以训练的问题，显著提升了性能

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **高效的突触利用机制**  
   所提方法通过模拟大脑的结构可塑性，自然防止过参数化，在仅保留必要突触的前提下完成分类任务，极大提高了 **synaptic capacity**。

2. **信息压缩与泛化潜力正相关**  
   结果验证了 Alemi et al. [8] 的观点：较低的 $ I(X;Z) $ 通常意味着更强的信息压缩能力和更好的泛化潜力。相比之下，BP 虽然能达到高精度，但容易陷入“完美记忆”陷阱，存储大量噪声。

3. **无需预先设计网络结构**  
   本方法能够根据输入数据自适应地决定所需网络复杂度，减少了对人工调参的依赖。

4. **具备持续学习潜力**  
   动态调控参数保留机制，使得网络能够在稳定状态下仍保留“空间”用于后续学习，更贴近真实大脑的学习模式。

### ⚠️ 局限性
- **分类准确率略低于 BP**：在当前实现下，特别是在小型网络中，性能仍有差距。
- **尚未扩展到深层网络或多任务场景**：目前仅在浅层网络和单一任务（MNIST 子集分类）上验证。
- **训练速度较慢**：由于依赖批处理时间序列动态，可能影响收敛速度。

### 🔮 未来工作方向
1. **提升分类性能**：改进学习规则以缩小与 BP 的准确率差距。
2. **多任务学习能力探索**：研究同一网络内连续学习多个任务的能力。
3. **应用于更深网络结构**：测试在 Deep DNNs 或 ConvNet 中的可扩展性。
4. **模拟更广泛的脑功能**：进一步融合神经科学成果，提升模型的认知模拟能力。
5. **实际部署节能评估**：量化在边缘设备上的功耗节省效果。

---

## 总结

该论文提出了一种真正意义上**受脑启发、面向可持续 AI 的学习范式**。它不仅在理论上揭示了高效信息表示的本质——即通过结构可塑性实现“最少必要连接”，还在实践中证明了其在 **energy-efficient AI** 和 **scalable memory modeling** 中的巨大潜力。尽管当前性能尚不及传统 BP，但其在资源效率方面的突破为下一代绿色人工智能提供了重要思路。

</details>

---

### 9. [Multi-Head Low-Rank Attention](https://arxiv.org/abs/2603.02188)

**Authors**: Songtao Liu, Hongwu Peng, Zhiwei Zhang, Zhengyu Chen, Yue Guo  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.02188v1  

#### Abstract
Long-context inference in large language models is bottlenecked by Key--Value (KV) cache loading during the decoding stage, where the sequential nature of generation requires repeatedly transferring the KV cache from off-chip High-Bandwidth Memory (HBM) to on-chip Static Random-Access Memory (SRAM) ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Multi-Head Low-Rank Attention**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在大语言模型（Large Language Models, LLMs）的长上下文推理过程中，解码阶段的 **Key-Value (KV) cache** 加载成为性能瓶颈。传统的 Multi-Head Attention (MHA) 和其变体（如 Multi-Head Latent Attention, MLA）虽然能压缩 KV cache 大小，但在使用 **Tensor Parallelism (TP)** 进行分布式解码时存在严重缺陷。

具体而言，MLA 虽然将 KV cache 压缩为一个“潜在头”（latent head），显著减少了缓存总量，但由于该潜在头无法被分片（sharded），在 TP 场景下每个设备都必须重复加载完整的 KV cache，导致内存带宽浪费，削弱了并行化带来的收益。

### **提出的新方法**
本文提出了 **Multi-Head Low-Rank Attention (MLRA)**，一种支持高效张量并行解码的新型注意力机制。其核心思想是：

- 将 MLA 中单一的 latent head **显式分解为多个独立的 latent heads**（如 MLRA-4 使用 4 个）。
- 每个 latent head 独立进行上投影（up-projection）生成 NoPE 键值对。
- 最终的注意力输出是各个分支输出的加权和。

这一设计使得每个 latent head 可以被分配到不同的 TP 设备上，从而实现真正的 KV cache 分片，大幅降低单设备的内存负载。

### **相比现有方法的优势**
- ✅ **支持高效的 4-way Tensor Parallelism**：解决了 MLA 在 TP 下冗余加载 KV cache 的问题。
- ✅ **更低的解码延迟**：在长序列场景下，MLRA-4 相比 MLA 实现了 **2.8× 的解码速度提升**。
- ✅ **更高的吞吐量**：在多设备部署中，MLRA-4 实现了最高的解码吞吐量。
- ✅ **更优的模型性能**：在困惑度（perplexity）和零样本推理任务上达到 SOTA 表现。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **预训练数据**：
  - `FineWeb-Edu-100B`（98.3B 训练 token，0.1B 验证 token）
- **评估数据集**（用于计算困惑度）：
  - Wikipedia, C4, The Pile, RefinedWeb, Cosmopedia, FineWeb, FineWeb-Edu
- **下游零样本推理基准**：
  - ARC-Easy, ARC-Challenge, OpenBookQA, BoolQ, HellaSwag, Winogrande, PIQA

### **实验设置**
- **模型规模**：基于 Llama-3 架构，在 **2.9B 参数量级** 上进行实验。
- **上下文长度**：预训练使用 2048 tokens；解码效率测试扩展至 **最长 2M tokens**。
- **硬件平台**：
  - 单卡测试：NVIDIA H100 80GB GPU
  - 多卡吞吐测试：8× H100 80GB GPUs
- **优化器**：AdamW，学习率 1.6e-4，cosine 衰减
- **实现框架**：基于 nanoGPT，并集成 FlashAttention-3 和自定义 MLRA 内核。

### **评估指标**
| 指标类别 | 具体指标 |
|--------|--------|
| **模型质量** | 验证集平均困惑度（PPL）、零样本任务平均准确率 |
| **解码效率** | 解码延迟（latency）、解码吞吐量（throughput） |
| **系统开销** | 每设备 KV cache 加载量（bytes/token） |

### **基线方法对比**
- **传统注意力**：MHA, MQA, GQA
- **低秩/压缩注意力**：MLA, MFA, TPA, GLA-2, GLA-4, GTA
- 特别关注与 **MLA** 的对比，因其同属 latent attention 范式。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 模型性能（Main Results）**
| 方法 | 平均困惑度 (↓) | 零样本平均准确率 (↑) |
|------|----------------|-----------------------|
| GQA | 14.139 | 57.89% |
| MLA | 13.727 | 58.75% |
| **MLRA-4** | **13.672** | **58.84%** |

👉 **结论**：MLRA-4 在所有模型中取得了最低的困惑度和最高的零样本推理准确率，优于 MLA。

#### **(2) 解码效率**
- **解码速度提升**：
  - MLRA-4 相比 GQA 实现 **1.05–1.26× 速度提升**。
  - 相比 MLA 实现 **高达 2.8× 的解码加速**。
- **KV Cache 加载量（每设备）**：
  - MLA (TP=1): 4.5dn
  - GLA-2 (TP=2): 2.5dn
  - **MLRA-4 (TP=4)**: **1.5dn** ✅
  - GQA 需要 TP=8 才能达到 2dn，而 MLRA-4 仅需 TP=4 即可达到更低的 1.5dn。

#### **(3) 吞吐量表现**
- 在批量为 128、序列长度从 1K 到 16K 的测试中：
  - MLRA-4 在所有长度上均实现了 **最高吞吐量**。
  - 显著优于 MLA（使用 DP=8）和 GQA（TP=8），证明其在实际部署中的优势。

#### **(4) 消融实验结果**
| 实验类型 | 发现 |
|--------|------|
| **初始化策略** | 使用 **zero initialization** 比 N(0,0.02) 更优，尤其对 TPA 类方法影响显著。 |
| **Scaling 因子** | 应用方差校准（variance calibration）后，MLA、GLA-2、MLRA-2 收敛更快，困惑度更低。 |
| **双倍注意力头数** | 将 GQA/MLA/GLA-2 的头数翻倍（保持参数不变）反而导致性能下降，说明“更多头”不等于更好性能。 |
| **门控机制（Gating）** | 引入门控注意力后，所有模型困惑度进一步降低，MLRA-4 达到 **13.621** 的最佳平均 PPL。 |

---

## **4. 关键结论和发现**

### **主要发现**
1. **MLRA 成功解决了 MLA 在张量并行下的瓶颈问题**，通过将 latent state 分支化，实现了真正可分片的 KV cache 结构。
2. **MLRA-4 在模型质量和解码效率上均达到 SOTA**：
   - 是当前 **2.9B 规模下困惑度最低** 的模型。
   - 在长上下文解码中实现了 **最短延迟和最高吞吐**。
3. **增加分支数（如 MLRA-4 vs MLRA-2）有助于提升性能**，表明多分支低秩建模具有更强表达能力。
4. **方差校准和门控机制对性能有正向作用**，验证了数值稳定性和非线性建模的重要性。

### **方法的局限性**
- 当前 MLRA 设计主要针对 **4-way TP**，对其他并行度（如 2/8-way）的支持未深入探讨。
- 分支间求和操作可能引入额外的同步开销，在极端低延迟场景下需进一步优化。
- 实验集中在 2.9B 模型，更大规模（如 7B+）的表现有待验证。

### **未来工作方向**
- 探索 **动态分支选择** 或 **稀疏激活** 机制，进一步提升效率。
- 将 MLRA 与 **量化**、**分页注意力（PagedAttention）** 等技术结合，构建端到端高效的推理系统。
- 扩展至 **MoE 架构** 或 **长文本微调** 场景，验证其通用性。

---

> 🔗 **代码与权重公开**：  
> - GitHub: [https://github.com/SongtaoLiu0823/MLRA](https://github.com/SongtaoLiu0823/MLRA)  
> - Hugging Face: [https://huggingface.co/Soughing/MLRA](https://huggingface.co/Soughing/MLRA)

</details>

---

### 10. [CARD: Towards Conditional Design of Multi-agent Topological Structures](https://arxiv.org/abs/2603.01089)

**Authors**: Tongtong Wu, Yanming Li, Ziye Tang, Chen Jiang, Linhao Luo, Guilin Qi, Shirui Pan, Gholamreza Haffari  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.01089v1  

#### Abstract
Large language model (LLM)-based multi-agent systems have shown strong capabilities in tasks such as code generation and collaborative reasoning. However, the effectiveness and robustness of these systems critically depend on their communication topology, which is often fixed or statically learned, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CARD: Towards Conditional Design of Multi-agent Topological Structures

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的基于 **Large Language Model (LLM)** 的多智能体系统（Multi-Agent Systems, MAS）在代码生成、协同推理等任务上表现出色，但其通信拓扑（communication topology）通常是**固定的**或**静态学习的**。这导致系统在面对动态环境变化时表现脆弱，例如：
- LLM模型升级（如 GPT-4o → GPT-5）
- 外部工具可用性变化（如搜索API失效）
- 数据源质量波动

这些变化会破坏预设的信息流，造成冗余交互或信息中断。

### 提出的新方法与新思路
作者提出 **CARD (Conditional Agentic gRaph Designer)**，一个**条件图生成框架**，用于实现 **AMACP (Adaptive Multi-Agent Communication Protocol)** 协议。其核心思想是：
- 将多智能体系统的通信结构建模为一个**可动态调整的有向图** $ G = (V, E) $
- 显式地将**环境状态信号**（如模型能力、工具性能、资源成本）作为输入，**条件化地生成通信拓扑**
- 在训练和运行时均可适应环境变化，无需重新训练

#### 主要创新点：
1. **形式化了 AMACP 协议**  
   定义了一个理想的通信拓扑应满足三个属性：
   - **Effectiveness**：能有效解决问题
   - **Cost-efficiency**：最小化资源消耗（token cost, API calls）
   - **Adaptiveness**：能随环境变化动态调整

2. **提出了 CARD 框架**
   - 包含四个阶段：Agent Representation → Conditional Graph Generation → Environment-Aware Training → Runtime Adaptation
   - 使用双通道编码器（profile + condition）生成节点嵌入
   - 引入 query 节点作为辅助注意力机制
   - 支持运行时一键更新拓扑（one-pass recomputation），无需微调

3. **实现了真正的“条件设计”**
   - 不同于仅靠 prompt 注入环境信息的方法，CARD 将环境信号直接嵌入图生成过程，实现更稳定、鲁棒的适应

### 相比现有方法的优势
| 方法类型 | 代表 | 缺陷 | CARD 的优势 |
|--------|------|------|------------|
| 手动设计 | CoT, LLM-Debate | 固定结构，无法适应变化 | 动态调整拓扑 |
| 自动学习静态拓扑 | GPT-Swarm, G-Designer | 学得的是固定结构，out-of-domain 泛化差 | 条件化生成，跨模型/工具泛化强 |
| Prompt 注入条件 | w/ Cond.p | 易引发负向影响（如 -12.5% 性能下降） | 结构级适应，更可靠 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在三个标准基准上进行评估：
- **HumanEval**：编程代码生成任务（功能正确性）
- **MATH**：数学推理任务（复杂推导）
- **MMLU**：大规模多任务语言理解（涵盖57个学科）

### 实验设置与评估指标
- **Agent 设置**：每个任务配置多个角色智能体（如 Project Manager, Programming Expert, Searcher 等）
- **环境变量模拟**：通过切换不同 LLM（gpt4o-mini, deepseek-v3, llama3-70B, gpt4o, qwen-72B）、搜索工具（Google, Wiki）、知识源（Wikipedia, Quora）来模拟真实世界的变化
- **评估方式**：
  - 报告各模型-任务组合下的准确率（Accuracy）
  - 对比平均性能（Avg.）
  - 分析 in-domain vs. out-of-domain 表现差距
- **训练策略**：使用环境感知训练（Environment-Aware Training），在多种配置下联合优化

### 基线方法对比
共三类基线：
1. **单智能体方法**：
   - Vanilla LLM
   - Chain-of-Thought (CoT)

2. **固定多智能体拓扑**：
   - Random Graph
   - LLM-Debate

3. **自动拓扑学习方法**：
   - GPT-Swarm
   - G-Designer
   - Aflow

所有方法均在同一 agent 配置下比较，确保公平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | HumanEval (Avg.) | MATH (Avg.) | MMLU (Avg.) |
|------|------------------|-------------|-------------|
| Vanilla | 85.50 | 63.33 | 81.04 |
| CoT | 87.66 | 64.33 | 84.18 |
| LLM-Debate | 86.00 | 66.83 | 84.05 |
| G-Designer | 86.50 | 72.66 | 82.87 |
| Aflow | 89.83 | 73.83 | 82.87 |
| **CARD** | **90.50** | **74.50** | **84.44** |

> ✅ CARD 在三项任务上均取得**最高平均准确率**

### 与基线方法的对比结果
- **全面领先**：在 15 个 model-benchmark 组合中，CARD 在 **13 项达到第一或并列第一**
- **显著提升**：
  - HumanEval 上比最强基线 Aflow 提升 **+0.67 pp**
  - MATH 上提升 **+0.67 pp**
  - MMLU 上提升 **+1.57 pp**
- **out-of-domain 更稳健**：
  - G-Designer 在从 deepseek-v3 切换到 qwen-72B 时，MATH 准确率从 91.66% ↓ 至 79.16%（↓12.5 pp）
  - CARD 同样切换下仅从 91.66% ↓ 至 82.50%（↓9.16 pp），**更具泛化性**

### 消融实验结果（Ablation Study）

#### （1）是否使用条件信号？（Figure 3）
- **w/o Cond.**（无条件）：作为基线
- **w/ Cond.p**（prompt 注入条件）：效果不稳定，在某些设置下甚至**性能下降最多达 -12.5%**
- **CARD**（图结构条件化）：在所有设置下都带来**正向增益**（+0.83 ~ +3.34% on MATH, +2.5 ~ +23.33% on HumanEval）

> 🔍 结论：**结构级条件适应远优于 prompt 级注入**

#### （2）局部条件更新 vs 全局重生成（Appendix B.2）
- 当仅有一个 agent 的条件发生变化（如工具更换），只需更新该节点 condition embedding 并重新 decode 图
- 实验显示：即使只替换 head node condition，仍能保持 **88.33% accuracy**，显著高于其他方法
- 表明 CARD 支持**轻量级在线适应**，适合生产部署

---

## 4. 关键结论和发现

### 主要发现
1. **通信拓扑必须是可适应的**  
   固定或静态学习的拓扑在现实动态环境中不可靠；**环境感知的拓扑设计至关重要**

2. **CARD 实现了高效且鲁棒的条件适应**
   - 显式建模环境信号（model capability, tool quality, cost）
   - 在训练和推理阶段都能动态调整图结构
   - 不依赖 prompt 工程，避免语义干扰

3. **弱模型更依赖密集协作**
   - 使用较小模型（如 gpt-4o-mini）时，CARD 自动生成更高密度的通信图（更强协作）
   - 使用大模型（如 GPT-4o）则趋向稀疏连接，节省成本

4. **检索质量影响信息流向**
   - 当使用 Wiki 替代 Google 搜索时，Knowledge Expert → Searcher 边权重下降，说明系统识别出检索质量差异并调整信任分配

5. **条件适应提升抗攻击能力**
   - 在中间节点被攻击时，CARD 可动态绕过故障节点，恢复能力强于 G-Designer 和 LLM-Debate（Figure 8）

6. **具备良好的成本-效益平衡**
   - 在相同预算下，CARD 实现更高的准确率
   - 或在相同性能下，使用更少 token 和 API 调用

### 方法的局限性
1. **未联合优化 agent 内部行为**  
   CARD 仅优化通信拓扑，不调整 agent 的 prompt 或内部参数。未来可探索 joint optimization。

2. **图表示可能不足以表达复杂流程约束**  
   如软件工程中的执行顺序、工具依赖等，纯图结构可能不够，需引入符号先验或混合建模。

3. **当前实验基于 API 调用模拟，尚未在真实复杂 workflow 中验证**

### 未来工作方向
- 扩展至更大规模 agent ensemble
- 引入 online reinforcement learning 实现持续自适应
- 探索 human-in-the-loop 的 hybrid design
- 应用于真实场景（如 DevOps pipeline, 科研助手系统）
- 结合 symbolic rules 增强领域特定约束建模

---

> 📚 **开源信息**：源码已公开于 GitHub：[https://github.com/Warma10032/CARD](https://github.com/Warma10032/CARD)  
> 🔗 论文链接：https://arxiv.org/abs/2603.01089

</details>

---

### 11. [3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs](https://arxiv.org/abs/2603.01376)

**Authors**: Mehdi Makni, Xiang Meng, Rahul Mazumder  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.01376v1  

#### Abstract
Sparse plus Low-Rank $(\mathbf{S} + \mathbf{LR})$ decomposition of Large Language Models (LLMs) has emerged as a promising direction in model compression, aiming to decompose pre-trained model weights into a sum of sparse and low-rank matrices $(\mathbf{W} \approx \mathbf{S} + \mathbf{LR})$. Despite...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：3BASiL: An Algorithmic Framework for Sparse plus Low-Rank Compression of LLMs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）虽然在多项任务上表现出色，但其庞大的参数量导致部署成本高昂，尤其是在资源受限的设备上。现有的 **Sparse plus Low-Rank (S + LR)** 分解方法虽能压缩模型，但在**性能恢复**和**优化质量**方面存在显著退化，且缺乏理论收敛保证。

本文旨在解决以下核心问题：
- 如何更有效地将预训练权重矩阵 $W$ 分解为稀疏（Sparse）和低秩（Low-Rank）两部分（即 $W \approx S + LR$），以实现高效压缩。
- 如何克服现有交替最小化（Alternating Minimization）方法在联合优化稀疏与低秩分量时的收敛性差、优化不充分等问题。

---

### 提出的新方法与创新思路

作者提出了 **3BASiL-TM** 框架，由两个核心组件构成：

#### （1）3BASiL：基于 3-Block ADMM 的层级重建算法
- **方法**：提出一种新颖的 **3-Block Alternating Direction Method of Multipliers (ADMM)** 算法，用于联合优化稀疏分量 $S$ 和低秩分量 $L$。
- **创新点**：
  - 将稀疏分量引入辅助变量 $D$，形成 $S, L, D$ 三块更新结构，确保每次迭代中都能精确满足稀疏性和低秩约束。
  - 提供**理论收敛保证**：在惩罚参数 $p_t$ 适当增长的条件下，证明分解序列收敛到一个稳定解。
  - 显著提升优化效率，相比现有方法（如 HASSLE-free）**加速超过 7 倍**。

#### （2）Transformer Matching (TM)：跨层联合微调机制
- **方法**：在完成各层的 $S+LR$ 分解后，引入一个轻量级的 **Transformer-level Matching** 步骤，通过梯度下降联合优化所有层的稀疏和低秩分量，使其输出尽可能匹配原始 Transformer 的输出。
- **创新点**：
  - **通用性强**：可作为“即插即用”模块，增强任何已有的 $S+LR$ 或纯稀疏压缩方法（如 SparseGPT, ALPS）。
  - **更优代理损失**：相比逐层重建误差，TM 使用 Transformer 级别的输出对齐作为中间目标，更接近真实端到端损失，减少累积误差。
  - 微调后的低秩分量可直接作为 **Smart LoRA Initialization**，用于后续下游任务适配。

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **优化质量** | 3BASiL 在层重建误差上显著优于 OATS、HASSLE-free 等交替最小化方法（见图5）。 |
| **收敛性** | 提供理论收敛保证，而现有方法多无此保障。 |
| **速度** | 3BASiL 压缩运行时比 HASSLE-free 快 **7 倍以上**；3BASiL-TM 整体仍快 **2.5–3 倍**。 |
| **性能恢复** | 3BASiL-TM 将 LLaMA-8B 在 WikiText2 上的 **PPL 与稠密模型的差距缩小超 30%**。 |
| **通用性** | TM 可提升任意 $S+LR$ 方法，甚至纯稀疏方法（如 Wanda, SparseGPT）。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据集（Calibration Set）**：从 C4 数据集中随机抽取 **128 个文本段**（每段 2048 tokens），用于一次性压缩（one-shot compression）。
- **评估数据集**：
  - **困惑度（Perplexity）**：WikiText2 (WT2), Penn Treebank (PTB), C4 validation。
  - **零样本任务（Zero-Shot Tasks）**：PIQA, ARC-Easy/Challenge, HellaSwag, Winogrande, RTE, OpenBookQA, BoolQ。

---

### 实验设置与评估指标

| 类别 | 设置说明 |
|------|----------|
| **模型** | LLaMA-3, LLaMA-3.2 (1B, 3B, 8B), OPT-30B |
| **压缩配置** | 主要测试 $(N:M + r)$ 配置，如 `(2:4 + 64LR)` 表示每 4 个权重保留 2 个，低秩为 64。 |
| **评估指标** | 
| - **Perplexity (↓)** | 越低越好，衡量语言建模能力。 |
| - **Zero-Shot Accuracy (↑)** | 越高越好，衡量下游任务表现。 |
| - **Compression Runtime** | 压缩耗时，越短越好。 |
| **LoRA 微调** | 在 C4 的 10% 数据上进行有限微调，验证恢复能力。 |

---

### 基线方法对比

| 基线方法 | 简介 |
|---------|------|
| **OATS** | 基于交替最小化的 $S+LR$ 方法，利用异常值感知稀疏性。 |
| **HASSLE-free-SparseGPT / HASSLE-free-ALPS** | 结合 SparseGPT 或 ALPS 的稀疏化步骤与交替最小化进行 $S+LR$ 分解。 |
| **EoRA** | 仅执行一次低秩拟合的快速方法，作为下限参考。 |
| **Wanda, SparseGPT, ALPS** | 纯稀疏压缩方法，用于验证 TM 的通用性。 |

> 注：作者对 HASSLE-free 系列进行了改进，采用闭式解（closed-form solution）替代梯度下降进行低秩拟合，提升了性能和速度。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 LLaMA-8B 为例）

#### 表：One-shot 压缩在 `2:4 + 64LR` 配置下的 C4 Perplexity（越低越好）

| 方法 | C4 PPL | WT2 PPL | PTB PPL |
|------|--------|---------|---------|
| **Dense** | 9.44 | 6.14 | 11.18 |
| OATS | 21.59 | 14.76 | 23.41 |
| Hf-SparseGPT | 17.77 | 12.38 | 18.71 |
| Hf-ALPS | 16.15 | 11.38 | 16.71 |
| **3BASiL** | 15.76 | 11.23 | 16.25 |
| **3BASiL-TM** | **14.34** | **9.78** | **14.88** |

> ✅ **3BASiL-TM 将 WT2 PPL 与稠密模型的差距缩小了超过 30%**。

---

#### 零样本任务平均准确率（Avg ↑）

| 方法 | Avg Zero-Shot |
|------|----------------|
| Dense | 69.99 |
| Hf-ALPS | 61.61 |
| 3BASiL | 62.80 |
| **3BASiL-TM** | **63.34** |

---

#### 压缩运行时对比（Llama3-8B on A100 80GB）

| 方法 | 运行时间（小时） | 相对速度 |
|------|------------------|----------|
| Hf-ALPS | 15.71 | 1× |
| **3BASiL-TM** | **4.08** | **>3.8× 更快** |

> ✅ **3BASiL-TM 在性能更强的同时，压缩速度提升超过 2.5 倍**。

---

### 消融实验结果

#### （1）3BASiL vs. Alternating Minimization
- 图5 显示，在 Attention 层，3BASiL 的重建误差远低于其他方法，表明其联合优化更有效。
- 3BASiL 在首次迭代后即快速收敛，而交替方法波动较大。

#### （2）TM 对不同方法的增益
- 表3 显示，TM 不仅提升 $S+LR$ 方法（如 OATS, Hf-ALPS），也显著提升纯稀疏方法（如 Wanda, SparseGPT）：
  - **Wanda → Wanda-TM**：C4 PPL 从 38.21 → 15.91（↓58%）
  - **SparseGPT → SparseGPT-TM**：C4 PPL 从 22.65 → 15.30（↓32%）

#### （3）LoRA 微调后性能恢复
- 图4 显示，即使经过 LoRA 微调，3BASiL-TM 仍保持领先优势。
- 例如在 `2:8+64LR` 配置下，LFT-3BASiL-TM 的 C4 PPL 比 LFT-Hf-ALPS 低约 8%。

---

## 4. 关键结论和发现

### 主要发现

1. **联合优化优于交替优化**：3BASiL 的 3-Block ADMM 框架能更有效地联合优化稀疏与低秩分量，避免交替方法的局部最优陷阱。
2. **Transformer Matching 是关键增强器**：TM 通过跨层输出对齐，显著提升稀疏分量质量，是连接压缩与下游适配的桥梁。
3. **Smart LoRA Initialization 更高效**：由压缩得到的低秩分量作为 LoRA 初始化，比随机初始化收敛更快、效果更好。
4. **通用性极强**：TM 可作为“即插即用”模块，提升几乎所有 $S+LR$ 或稀疏压缩方法。

---

### 方法的局限性

1. **未探索动态稀疏/秩分配**：当前方法对所有层使用固定 $N:M$ 和 $r$，未像 OWL 那样进行层间自适应分配。
2. **依赖校准数据**：虽然只需少量数据，但仍需输入激活 $X$ 来计算 $H = XX^T + \lambda I$。
3. **未报告统计显著性**：由于算力限制，实验未提供多次运行的标准差或置信区间。
4. **扩展至量化等场景待验证**：目前聚焦于稀疏+低秩，是否适用于量化+低秩尚未验证。

---

### 未来工作方向

1. **结合稀疏分配机制**：将 3BASiL 与 OWL 等方法结合，实现**层自适应的 $S+LR$ 压缩**。
2. **扩展至 Quantized + LR**：将框架推广至量化+低秩分解，进一步提升压缩率。
3. **端到端训练集成**：探索在预训练阶段就引入 $S+LR$ 结构（类似 SLTrain），而非仅用于后处理压缩。
4. **硬件协同优化**：针对特定推理引擎（如 DeepSparse, TensorRT）定制 $S+LR$ 模式，最大化推理加速。

---

> **代码开源**：https://github.com/mazumder-lab/3BASiL

</details>

---

### 12. [Accelerating PDE Surrogates via RL-Guided Mesh Optimization](https://arxiv.org/abs/2603.02066)

**Authors**: Yang Meng, Ruoxi Jiang, Zhuokai Zhao, Chong Liu, Rebecca Willett, Yuxin Chen  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.02066v1  

#### Abstract
Deep surrogate models for parametric partial differential equations (PDEs) can deliver high-fidelity approximations but remain prohibitively data-hungry: training often requires thousands of fine-grid simulations, each incurring substantial computational cost. To address this challenge, we introduce...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accelerating PDE Surrogates via RL-Guided Mesh Optimization*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

深度学习驱动的 **PDE surrogate models**（如 FNO、DeepONet）在科学计算中展现出强大的近似能力，但其训练极度依赖大量高保真度的数值模拟，这些模拟通常在精细网格上运行，计算成本极高。这种“数据饥渴”特性成为实际部署的关键瓶颈。

此外，传统方法通常在**均匀、密集的网格**上进行求解，而许多物理系统的解具有**空间异质性**（如激波、边界层、高导通道），在平滑区域浪费了大量计算资源。

### **提出了什么新方法或新思路**

本文提出 **RLMESH** —— 一个端到端的、基于强化学习（Reinforcement Learning, RL）的自适应网格优化框架，用于高效训练 PDE surrogate。

#### 核心思想：
- 将每个 PDE 实例的**网格点选择**建模为一个**序列决策过程**（Sequential Decision Process）。
- 在每个实例的求解过程中，一个 **RL policy** 动态地、非均匀地选择少量最关键的网格点进行求解器查询。
- 利用一个轻量级的 **proxy model**（如核岭回归）来快速估计新采集数据对下游 surrogate 模型的改进程度，并将其作为 RL 的奖励信号，避免频繁重训昂贵的 surrogate 模型。

### **相比现有方法的优势**

| 对比维度 | 现有方法 | RLMESH |
|--------|--------|--------|
| **适应性粒度** | 实例级选择（选哪个 PDE 实例）或全局分辨率选择 | **实例内空间点级选择**，实现细粒度的空间自适应 |
| **反馈机制** | 依赖完整场或启发式不确定性 | 使用 **proxy model** 提供与 surrogate 改进强相关的终端奖励 |
| **效率** | 需要全网格模拟 | 只在选定的稀疏点上查询求解器，大幅减少计算开销 |
| **通用性** | 多假设固定网格结构 | 支持不规则、非均匀采样，更灵活 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

论文在三个经典的 PDE 和动力系统上进行了验证：

1. **1D Burgers’ Equation**  
   - 描述粘性流体中的激波形成。
   - 输入：初始条件 $ u_0(x) $；输出：终态解 $ u(x,1) $。
   - 特征：存在陡峭前沿（shock fronts），适合测试空间自适应能力。

2. **2D Darcy Flow**  
   - 描述多孔介质中的稳态流动。
   - 输入：扩散系数场 $ c(x) $；输出：压力场 $ u(x) $。
   - 特征：解在高导通道和边界层附近变化剧烈。

3. **Lorenz-96 System**  
   - 一维周期性格点上的混沌动力系统。
   - 输入：初始状态 $ x(0) $；输出：终态 $ x(1) $。
   - 特征：无连续空间几何，用于测试方法在非 PDE 场景下的泛化性。

> 所有数据均来自 **PDEBench** 数据集。

### **实验设置和评估指标**

- **训练/测试划分**：1000 个训练实例，200 个测试实例。
- **预训练阶段**：使用 100 个全网格观测实例对 surrogate（FNO）和 proxy 进行预训练。
- **主动学习阶段**：在剩余 900 个实例上进行 18 轮主动采样，每轮 50 个实例。
- **每实例预算**（Per-instance Budget B）：限制每次模拟最多查询 B 个网格点（如 60、80 等）。
- **评估指标**：
  - **RMSE**：在高分辨率密集网格上的均方根误差。
  - **时间-误差权衡**：累计求解时间 vs. 测试误差。
- **重复性**：所有结果取 5 次独立随机种子的平均值 ± 标准差。

### **基线方法对比**

与以下启发式策略进行公平比较（相同预算、相同 retrain 频率）：

- **Uniform**：均匀分布采样。
- **Random**：随机采样。
- **Gradient**：优先选择梯度大的区域。
- **Variance**：基于模型预测方差选择不确定区域。
- **Intensity**：基于输入场强度选择。

此外，还与实例级主动学习方法（如 **Self-MI**, **LCMD**）进行对比，以凸显“空间点级”自适应的独特优势。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ 在所有三个任务上，RLMESH 显著优于所有基线：

| 方法 | Burgers (Iter 6) | Darcy (Iter 6) | Lorenz-96 (Iter 6) |
|------|------------------|----------------|--------------------|
| **RLMESH (Ours)** | **0.0109 ± 0.0021** | **0.030 ± 0.002** | **0.018 ± 0.001** |
| Variance | 0.0126 ± 0.0007 | 0.038 ± 0.003 | 0.022 ± 0.002 |
| Gradient | 0.0146 ± 0.0014 | 0.040 ± 0.004 | 0.023 ± 0.001 |
| Uniform | ~0.018 | ~0.045 | ~0.025 |

> *注：具体数值见原文图3及附录Table 1*

- **加速效果显著**：在 Burgers 上，RLMESH 达到 RMSE=0.02 仅需约 **6 轮迭代**，而 variance/heuristic 方法需要 9–10 轮，uniform/random 需要 12 轮以上，相当于节省 **33%~50% 的标注成本**。

#### ✅ 时间-误差权衡优势明显（使用自定义非均匀求解器）：

- 为达到相同的 RMSE=0.02，RLMESH 仅需约 **40 秒**累计求解时间。
- Variance/Gradient 等基线需 **80–120 秒**。
- Uniform/Random 超过 **150 秒**。
- 表明 RLMESH 更高效地将计算资源转化为精度提升。

#### ✅ 最优预算分析：

- 当每实例预算 $ B=60 $ 时，性能与 $ B=80/100 $ 接近，但成本更低。
- $ B=20/40 $ 时性能饱和早，受限于 FNO 容量而非策略质量。
- 因此 **$ B=60 $ 是性价比最高的操作点**。

### **消融实验结果**

1. **Proxy Model 的有效性**：
   - 使用 **RBF Kernel Ridge Regression** 作为 proxy。
   - 与真实 FNO surrogate 的 RMSE 改进之间 **Spearman 相关系数高达 0.9908**，说明 proxy 能准确反映相对排序。
   - 替换为 MLP、GP、SVR 等其他 proxy，相关性均低于 Kernel Ridge。

2. **Per-instance Adaptivity 的必要性**：
   - 可视化显示，RL policy 学会根据不同输入特征动态调整采样策略：
     - Burgers：集中在 shock front 和 turning points。
     - Darcy：沿高对比度通道和边界层聚集。
   - 证明策略不是静态模板，而是真正实现了**按需定制**。

3. **与实例级主动学习对比**：
   - 将 MRA-FNO 和 AL4PDE 改造为单保真度设置并与 RLMESH 对比。
   - 结果表明，即使结合最优实例选择，若缺乏**内部空间点选择**，仍无法匹敌 RLMESH 的性能（见 Table 1）。

---

## 4. 关键结论和发现

### **主要发现**

1. **Solver-level spatial adaptivity 可极大提升 surrogate 训练效率**：通过在最关键的空间位置集中计算资源，可以在远少于全网格的数据下达到相近甚至更好的精度。

2. **RL + Proxy 是可行且高效的闭环框架**：轻量 proxy model 成功替代了昂贵的 surrogate 重训作为 RL 奖励来源，实现了稳定、高效的在线策略优化。

3. **Per-instance 自适应优于全局或实例级策略**：针对每个输入动态生成采样策略，能更好地捕捉局部物理特征，是实现高效学习的关键。

4. **该方法不仅适用于经典 PDE，也适用于混沌动力系统**（如 Lorenz-96），显示出良好的泛化潜力。

### **方法的局限性**

1. **依赖高质量的非均匀求解器**：当前实验中，为保证稀疏点上的解准确，需对极端不规则网格进行几何增强（如插入虚拟中点）。构建工业级鲁棒的自适应求解器仍是挑战。

2. **尚未扩展到时空联合采样或多保真度设置**：目前仅考虑空间点选择，未涉及时间步长或不同精度层级的协同优化。

3. **RL 训练本身有一定开销**：虽然 proxy 加速了奖励计算，但 RL policy 的训练仍需一定样本积累，在极小预算下可能不划算。

4. **2D+ 扩展复杂度高**：文中未提供完整的 2D 非均匀 Darcy 求解器的时间-误差分析，因其实现更为复杂。

### **未来工作方向**

1. **扩展至时空感知的 sensing 策略**：联合优化时间和空间上的采样点。
2. **支持多保真度成本模型**：结合粗/细网格、快/慢求解器进行混合查询。
3. **应用于更高维或不规则几何问题**：结合 geometry-aware neural operators（如 GNN、Transformer）处理复杂域。
4. **理论分析样本效率**：为 joint instance-and-grid-point selection 提供理论保障。
5. **集成到工业级 AMR 框架中**：与 Firedrake、 deal.II 等工具链对接，推动实用化落地。

---

> **总结一句话**：  
> RLMESH 通过引入 **RL-guided per-instance mesh optimization** 和 **proxy-based reward estimation**，首次实现了在严格局部查询预算下的高效 PDE surrogate 训练，在多个基准上显著优于传统启发式与实例级主动学习方法，为低成本、高精度的科学机器学习提供了新范式。

</details>

---

### 13. [CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging](https://arxiv.org/abs/2603.00573)

**Authors**: Jie Cao, Zhenxuan Fan, Zhuonan Wang, Tianwei Lin, Ziyuan Zhao, Rolan Yan, Wenqiao Zhang, Feifei Shao, Hongwei Wang, Jun Xiao, Siliang Tang  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.00573v1  

#### Abstract
Large language models (LLMs) achieve remarkable performance on diverse downstream and domain-specific tasks via parameter-efficient fine-tuning (PEFT). However, existing PEFT methods, particularly MoE-LoRA architectures, suffer from limited parameter efficiency and coarse-grained adaptation due to t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CoMoL: Efficient Mixture of LoRA Experts via Dynamic Core Space Merging**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **MoE-LoRA** 架构在参数高效微调（PEFT）中面临两大挑战：
- **Limited Parameter Efficiency（参数效率低下）**：引入多个 LoRA 专家和路由网络显著增加了可训练参数量，违背了 PEFT 的初衷。
- **Coarse-grained Adaptation（粗粒度适配）**：如 SMEAR 等方法采用实例级（instance-level）路由，无法实现对每个 token 的细粒度动态适应。

### **提出了什么新方法或新思路**
本文提出 **CoMoL（Core Space Mixture of LoRA）**，一种新型 MoE-LoRA 框架，包含两个核心组件：
- **Core Space Experts（核心空间专家）**  
  将每个 LoRA 专家的知识压缩到一个低秩的 **core matrix $ M_i \in \mathbb{R}^{r \times r} $** 中，共享主干的奇异子空间（$ U_B, V_A $），从而极大减少参数开销。
- **Core Space Routing（核心空间路由）**  
  路由器输入被投影至 LoRA 的低秩空间（即 $ v_c = A x $），并在该空间进行 token-level 动态路由决策，进一步降低路由模块的参数量。

此外，CoMoL 在核心空间内通过 **soft-merging** 策略将激活专家融合为单一核心专家，再与共享 LoRA 结合生成专用适配模块。

### **相比现有方法的优势**
| 维度 | CoMoL | 其他 MoE-LoRA 方法 |
|------|-------|------------------|
| **参数量** | ≈ Standard LoRA（~1.0×） | 多数为 N× 参数增长 |
| **计算开销（FLOPs）** | ≈1.0× LoRA | Soft-weighted: N×；Sparse: 高延迟 |
| **路由粒度** | Token-level | Instance-level 或高成本稀疏路由 |
| **路由延迟** | 低 | Sparse MoE 路由机制带来显著延迟 |

> ✅ **核心优势**：在保持 **token-level 细粒度适应能力** 的同时，实现了与标准 LoRA 相当的参数和计算效率。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
#### 数学推理任务（Mathematical Reasoning）
- **微调数据**：Math14k（GSM8K + AQuA，含 Chain-of-Thought 标注）
- **测试数据**：GSM8K, SVAMP, MultiArith, AddSub, AQuA, SingleEq

#### 代码生成任务（Code Generation）
- **微调数据**：CodeAlpaca-20k
- **测试数据**：HumanEval
- **评估指标**：pass@1, pass@5, pass@10

### **实验设置**
- **主干模型**：
  - Qwen3-8B 和 Qwen3-14B（数学 & 编程）
  - Llama3.1-8B（编程，跨架构验证）
- **LoRA 设置**：
  - Rank 固定为 8（除标准 LoRA、DenseLoRA、FlyLoRA 外）
  - 适配层：Query, Key, Value, Output, Down-projection
- **训练配置**：
  - 数学任务：训练 1 轮
  - 编程任务：Qwen3-8B 训练 5 轮，Llama3.1-8B 训练 1 轮
  - 统一框架基于 Transformers 库，控制变量公平比较

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Standard PEFT** | LoRA |
| **Soft-weighted MoE-LoRA** | MoLoRA, HydraLoRA |
| **Sparse MoE-LoRA** | MoLA, AdaMoLE, SparseMoA |
| **Advanced LoRA Variants** | DenseLoRA, FlyLoRA |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **数学推理任务（Qwen3-8B）**
| 方法 | Average Accuracy | Trainable Params |
|------|------------------|------------------|
| LoRA | 82.78% | 24.77M |
| MoLoRA | 83.85% | 107.35M |
| SparseMoA | 83.69% | 24.49M |
| **CoMoL** | **84.48%** | **25.16M** |

> 🔺 CoMoL 以仅比 LoRA 多 **0.39M 参数**，平均准确率提升 **1.7 个百分点**，且在 AddSub 和 AQuA 上达到 SOTA。

#### **数学推理任务（Qwen3-14B）**
| 方法 | Average Accuracy | Trainable Params |
|------|------------------|------------------|
| LoRA | 85.18% | 35.39M |
| MoLoRA | 86.10% | 76.84M |
| **CoMoL** | **86.34%** | **35.82M** |

> 🔺 在更大模型上仍保持领先，参数仅为 MoLoRA 的 ~46%，性能更高。

#### **代码生成任务（HumanEval）**
| 方法 | Llama3.1-8B (pass@1) | Qwen3-8B (pass@1) | Params |
|------|------------------------|--------------------|--------|
| LoRA | 26.22% | 39.69% | ~24M |
| MoLoRA | 34.15% | 43.78% | >100M |
| FlyLoRA | 32.32% | 20.18% | ~23M |
| **CoMoL** | **35.00%** | **48.11%** | **23.24M / 24.97M** |

> ✅ CoMoL 在两个模型上均取得最佳性能，尤其在 Qwen3-8B 上远超其他方法，显示其强大学习容量。

### **消融实验结果**
- **CoMoL w/o CR**（无 Core Space Routing）：
  - 性能接近 CoMoL，但参数随专家数增加而上升（如 64 专家时达 93.78M vs CoMoL 的 27.91M）
  - 表明 **Core Space Routing 对参数恒定至关重要**
- **不同专家数量（#Experts）影响**：
  - 最佳性能出现在 8 专家
  - 即使扩展到 64 专家，CoMoL 仍稳定运行，而 HydraLoRA 在 16 专家即出现 OOM
- **不同 LoRA rank 影响**：
  - CoMoL 在所有 rank 下均优于 LoRA，最佳表现于 rank=16
  - 参数增长平缓，体现良好扩展性

---

## **4. 关键结论和发现**

### **主要发现**
1. **专家冗余严重**：传统 MoE-LoRA 中大量专家存在参数冗余，CoMoL 证明少量核心矩阵即可捕获多专家集体知识。
2. **细粒度路由可行且高效**：通过在低秩核心空间完成 token-level soft-merging，可在不牺牲效率的前提下实现精细化适配。
3. **参数效率与表达力可兼得**：CoMoL 成功打破“更多专家 → 更多参数”的范式，在参数量≈LoRA 的前提下超越多数 MoE-LoRA 方法。
4. **跨模型/任务鲁棒性强**：在 Qwen 和 Llama 系列、数学与编程任务中均表现优异，具备广泛适用性。

### **方法的局限性**
- 当前研究聚焦于 **参数效率优化**，未系统探讨不同 PEFT 方法的 **学习容量边界**。
- 如 FlyLoRA 所示，某些方法在特定模型（Llama）上表现好但在另一些（Qwen）上差，说明 **适配能力依赖于模型先验分布**。
- 缺乏统一基准来评估 PEFT 方法的学习潜力与泛化能力。

### **未来工作方向**
- 建立 **系统性的 PEFT 学习容量评测框架**，涵盖不同任务、模型族、数据分布。
- 探索更智能的 **core space 初始化与正则化策略**，进一步提升小参数下的表达能力。
- 将 CoMoL 思想推广至其他 PEFT 范式（如 IA³、Adapter）或 MoE 架构中。

---

> 📌 **总结一句话**：  
> **CoMoL 通过将 MoE-LoRA 的专家存储与路由操作全部迁移至低秩核心空间，首次实现了“token-level 动态路由 + ≈LoRA 参数量”的理想组合，在多项任务上达成高效性与性能的双重突破。**

</details>

---

### 14. [GroupGPT: A Token-efficient and Privacy-preserving Agentic Framework for Multi-User Chat Assistant](https://arxiv.org/abs/2603.01059)

**Authors**: Zhuokang Shen, Yifan Wang, Hanyu Chen, Wenxuan Huang, Shaohui Lin  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.01059v1  

#### Abstract
Recent advances in large language models (LLMs) have enabled increasingly capable chatbots. However, most existing systems focus on single-user settings and do not generalize well to multi-user group chats, where agents require more proactive and accurate intervention under complex, evolving context...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《GroupGPT: A Token-efficient and Privacy-preserving Agentic Framework for Multi-User Chat Assistant》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前大多数基于 **Large Language Models (LLMs)** 的聊天机器人研究集中于**单用户场景**，在多用户群聊环境中的应用存在显著不足。具体挑战包括：
- **干预时机不准确**：难以判断何时介入对话（过早打断、过晚响应）。
- **Token 消耗过高**：传统方法频繁调用 LLM 进行推理，导致高昂的 API 成本。
- **隐私泄露风险**：原始消息直接上传至云端 LLM，可能暴露用户敏感信息（如姓名、关系等）。
- **缺乏标准化评估**：缺少高质量、标注丰富的多用户群聊干预行为评测数据集。

### **提出了什么新方法或新思路**
本文提出 **GroupGPT** —— 一种面向多用户群聊助手的 **token-efficient** 且 **privacy-preserving** 的智能体框架，其核心创新如下：

#### ✅ **小模型 + 大模型协同架构（Small-Large Model Collaboration）**
- 将“**是否干预**”（intervention timing）与“**如何回应**”（response generation）解耦：
  - **Intervention Judge**：轻量级语言模型（SLM），实时监控上下文并决定是否干预，大幅减少 LLM 调用频率。
  - **Final Respondent**：大语言模型（LLM），仅在需要时生成最终回复。
- 实现高效决策与高质量输出的平衡。

#### ✅ **隐私保护机制（Privacy Transcriber）**
- 引入 **Privacy Transcriber** 模块，在本地对每条消息进行 PII（Personally Identifiable Information）检测与泛化重写（如将 “with Cindy” 替换为 “with my partner”），确保敏感信息不会被上传到云 LLM。

#### ✅ **支持多模态输入处理（Multimodal Message Processor）**
- 支持图像、表情包（meme）、视频、语音等多种输入形式，并通过专用模型将其转化为结构化文本描述（如 `<meme>A dog with tears</meme>`），统一输入格式以提升效率。

#### ✅ **构建首个干预推理基准数据集 MUIR**
- 提出 **MUIR**（Multi-User Intervention Reasoning dataset）：
  - 包含 **2,500 条真实群聊片段**，涵盖日常交流、学术讨论、情感支持等多个主题。
  - 每条样本标注了 **干预类型**（如 Emotional Support, Fact Correction）及 **干预理由**（rationale）。
  - 分为训练集（2,000）和测试集（500），支持量化评估干预时机与合理性。

### **相比现有方法的优势**
| 维度 | GroupGPT | 传统方法（如 MUCA、HUMA） |
|------|--------|--------------------------|
| **Token 效率** | 减少高达 **3倍以上** 的 token 使用 | 固定间隔调用 LLM，浪费严重 |
| **响应延迟控制** | 平均端到端延迟 **~4.3秒**，接近人类反应速度 | 高频调用导致高延迟 |
| **隐私保护** | 本地完成 PII 检测与脱敏，保障数据安全 | 原始消息直传云端，风险高 |
| **评估能力** | 提供标准 benchmark MUIR，支持客观比较 | 依赖主观用户研究或人工设计场景 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MUIR 数据集**：本文构建的核心 benchmark。
  - 来源：来自 30 名志愿者的真实英文群聊记录（经知情同意与匿名化处理）。
  - 内容覆盖：日常生活、技术讨论、粉丝社群、心理健康、烹饪、宠物、体育、编程等。
  - 多模态支持：包含图像、语音、视频等内容的文本转录版本。
  - 标注方式：结合 GPT-4o 自动生成 + 人工校验，保证质量。

### **实验设置**
- **GroupGPT 实现细节**：
  - **Intervention Judge**: Qwen-3-4B（fine-tuned on MUIR）
  - **Privacy Transcriber**: Llama-3.2-Instruct-3B（trained on [11] 的隐私数据集）
  - **Multimodal Processor**:
    - 图像/表情包：Qwen-2.5-VL-32B
    - 视频：采样 1FPS 后 caption
    - 语音：Qwen3-ASR-Flash 转录
  - **Final Respondent**: GPT-4o
  - 窗口大小：短期窗口 $N_{sw}=20$，长期窗口 $N_{lw}=50$
  - 训练配置：LoRA 微调，batch size=16，lr=2e-4，warmup ratio=0.1

### **评估指标**
| 类别 | 指标说明 |
|------|--------|
| **干预准确性** | Accuracy, Macro-F1, F1（用于 Chime-in Reason 和 Timing 任务） |
| **综合性能** | Weighted Score = (Reason + Timing) / 2 |
| **响应质量** | 使用 **LLM-as-a-judge** 方法（GPT-4 对话评估），从 **Relevance, Coherence, Fluency, Helpfulness** 四个维度打分（1–5 Likert scale） |
| **系统效率** | Token 消耗总量、推理延迟（latency）、GPU 显存占用 |
| **用户体验** | 用户问卷调查（post-study questionnaire），评估实用性、隐私感知、舒适度等 |

### **基线方法对比**
| 基线类型 | 具体模型/策略 |
|---------|--------------|
| **随机猜测** | Random Guess（baseline） |
| **人类评估者** | Three human annotators（upper bound） |
| **纯 LLM 方法** | Qwen3-Max, Gemini-2.5-Pro, DeepSeek-V3.2（prompt engineering） |
| **Embedding + KNN** | GPT-4o, Gte-large-en-v1.5, Bge-m3 等嵌入模型检索最近邻 |
| **小型模型微调** | Gemma-2-it, Qwen-2.5-Instruct, Llama-3.1-Instruct 等 SLM fine-tuned on MUIR |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **MUIR 基准表现（Table 1）**
| 方法 | Chime-in Reason Acc | Chime-in Timing Acc | Weighted Score |
|------|--------------------|---------------------|----------------|
| **Human Evaluator** | **0.8859** | **0.8642** | **0.8775** |
| **Qwen-2.5-Instruct (3B)** | **0.8628** | 0.7536 | **0.8125** |
| **Qwen-3 (4B)** | 0.7859 | **0.8340** | 0.8062 |
| GPT-4o (w/ prompt) | 0.8333 | 0.6358 | 0.7223 |
| Random Guess | 0.1721 | 0.5926 | 0.4120 |

> 📌 **结论**：微调后的 SLM 在 Timing 任务上显著优于大模型，实现了更主动的干预判断；Qwen-2.5-Instruct 3B 取得最佳综合得分。

#### 🔹 **响应质量评估（LLM-as-a-judge）**
对 300 个 GroupGPT 生成的回复进行 GPT-4 打分（Table 2）：

| 维度 | 平均分 | 评分分布（5分占比） |
|------|-------|------------------|
| **Relevance** | 4.74 | 82.3% |
| **Coherence** | 4.79 | 85.6% |
| **Fluency** | 4.90 | 93.3% |
| **Helpfulness** | 4.46 | 67.7% |

> 📌 **结论**：生成内容高度相关、连贯流畅，获得接近满分的语言质量评价。

#### 🔹 **Token 消耗对比（Figure 2）**
- 在模拟 **500 条消息** 的群聊中：
  - **Baseline（LLM-only）**：消耗约 **1.8M tokens**
  - **GroupGPT**：仅消耗 **0.6M tokens**
  - ➔ **节省达 3.1× token 使用量**

> 📌 **推论**：若按每日 1,500 条消息计算，年 token 消耗可从 **2B** 降至 **0.66B**，极大降低部署成本。

#### 🔹 **推理延迟与资源占用（Table 3）**
| 组件 | 平均延迟（s） | 最大延迟（s） | GPU 显存（GB） |
|------|-------------|------------|---------------|
| Intervention Judge | 2.40 | 6.17 | 10.12 |
| Privacy Transcriber | 0.77 | 1.61 | 8.29 |
| **GroupGPT（完整流程）** | **4.36** | **10.96** | **18.41** |

> 📌 **结论**：平均响应时间 **<5 秒**，符合人类对话节奏；可在消费级 GPU（如 3080Ti）运行，具备实际部署可行性。

#### 🔹 **用户研究反馈（Figure 3）**
- **实用性**：
  - 64% 用户认为“介入时机恰当”
  - 71% 认为“回应有趣、有帮助且简洁”
- **隐私与效用**：
  - 84% 认为“改写后移除了大部分私人信息”
  - 88% 认为“原意得以保留”
- **舒适度**：
  - 仅 9% 表示感到“烦人或尴尬”
  - 56% 愿意继续与该 agent 互动
- **总体印象**：
  - 66% 认为“新颖且具潜在影响力”
  - 61% 愿意向他人推荐

> 📌 **结论**：用户普遍接受度高，兼具功能性与隐私友好性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **小模型可用于精准干预决策**：轻量级模型在 fine-tuned 后可在干预时机判断上媲美甚至超越大模型，是实现高效群聊交互的关键。
2. ✅ **解耦设计显著提升效率**：将“判断是否说话”与“说什么”分离，可减少 **3倍以上 token 消耗**，降低成本。
3. ✅ **本地隐私处理可行且有效**：Privacy Transcriber 能在设备端完成 PII 检测与脱敏，兼顾安全性与语义完整性。
4. ✅ **MUIR 是可靠 benchmark**：数据集质量高，能有效区分不同模型的能力，填补领域空白。
5. ✅ **用户体验良好**：多数用户认可其介入时机、表达质量和隐私保护机制，愿意持续使用。

### **方法的局限性**
- **语言限制**：目前 MUIR 和实验均基于 **英语** 数据，未验证跨语言适用性。
- **角色个性化不足**：尚未建模每个用户的个性特征或群体级集体记忆（group-level personalization）。
- **极端高频聊天压力测试缺失**：未在超高频（如直播弹幕级）场景下验证稳定性。
- **完全自动化标注依赖 LLM**：虽有人工校验，但仍可能存在标注偏差。

### **未来工作方向（Future Work）**
- **Group-level Personalization**：让 agent 学会记住群组共享概念（如某成员的宠物名字、过往梗图）。
- **合成群聊数据生成**：利用 LLM + 多模态生成模型创建逼真的虚拟群聊数据，缓解数据稀缺问题。
- **引入 RLHF**：采用强化学习从人类偏好中优化干预策略。
- **统一多模态理解模型**：使用单一 MLLM（如 Qwen3-Omni）替代多个专用处理器，简化架构。
- **多智能体协作系统**：部署多个专业化 agent（如哲学家、教师、调解员）共同参与群聊。
- **数字身份集成**：结合 user’s digital twin 或 persona，实现更自然的角色扮演式交互。

---

> ✅ **总结一句话**：  
> **GroupGPT 通过“小模型判时机 + 大模型产回复 + 本地隐私脱敏”的架构创新，在保证高质量、低延迟响应的同时，实现了高达 3 倍的 token 节省和强隐私保护，为多用户群聊助手的实际落地提供了高效、安全、可评估的新范式。**

</details>

---

### 15. [AeroDaaS: A Programmable Drones-as-a-Service Platform for Intelligent Aerial Systems](https://arxiv.org/abs/2603.00506)

**Authors**: Kautuk Astu, Suman Raj, Priyanshu Pansari, Yogesh Simmhan  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.00506v1  

#### Abstract
The increasing adoption of UAVs equipped with advanced sensors and GPU-accelerated edge computing has enabled real-time AI-driven applications in domains such as precision agriculture, wildfire monitoring, and environmental conservation. However, the integrated design and orchestration of navigation...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AeroDaaS: A Programmable Drones-as-a-Service Platform for Intelligent Aerial Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前无人机（UAV）应用面临以下挑战：
- **平台碎片化**：不同厂商（如 DJI、Parrot、Skydio）提供专有 SDK，导致代码不可移植，开发成本高。
- **低级抽象**：现有框架（如 PX4、ArduPilot）仅提供底层飞行控制接口，缺乏对 **sensing、navigation、analytics** 的高层集成支持。
- **分析与导航脱节**：DNN 驱动的实时分析（如目标检测）难以与轨迹规划协同调度，无法实现动态任务优先级调整。
- **部署复杂性**：缺乏统一的跨边缘-云（edge-cloud continuum）资源调度机制，且仿真与真实环境切换困难。

### **提出了什么新方法或新思路**
本文提出 **AeroDaaS** —— 一个可编程的 **Drones-as-a-Service (DaaS)** 平台，其核心创新包括：

- **服务化架构（Service-Oriented Framework）**  
  将无人机的 sensing、navigation 和 analytics 抽象为可组合的微服务（composable microservices），支持跨平台部署。

- **统一编程模型（DaaS Programming Framework）**  
  提供高层 Python API（如 `IEnvironment`, `IRobot`, `IAnalyse`, `INavigate`），开发者无需关心底层硬件差异即可构建 DaaS 应用。

- **模块化调度器插件机制**  
  支持即插即用的 **Waypoint Scheduler** 和 **Analytics Scheduler**，实现轨迹优化与推理任务的联合调度（co-scheduling）。

- **跨边缘-云分析执行**  
  分析任务可在 **edge（如 Jetson Orin Nano）、fog、cloud（如 AWS Lambda）** 上灵活分布，由调度器自动管理负载均衡与延迟约束。

- **仿真-现实一致性设计**  
  支持在 Gazebo 中进行仿真测试，并保证同一份代码可无缝部署到物理无人机上，降低开发-测试摩擦。

### **相比现有方法的优势**

| 特性 | AeroDaaS | 其他框架（如 DJI SDK, PX4, Buzz, SkyQuery） |
|------|---------|------------------------------------------|
| 跨平台兼容性 | ✅ 支持多硬件（Tello, PX4等）及仿真 | ❌ 多数绑定特定厂商或平台 |
| 高层抽象 | ✅ 提供 `IAnalyse`, `INavigate` 等声明式API | ❌ 多为底层控制命令 |
| 分析-导航协同 | ✅ 支持触发式 mission update/abort | ❌ 缺乏闭环反馈机制 |
| 边缘-云协同 | ✅ 内建 analytics 调度器支持 offloading | ❌ 通常需手动配置 |
| 可扩展性 | ✅ 插件式调度器，易于替换策略 | ❌ 架构固化 |

> 如表1所示，AeroDaaS 是唯一同时支持 **cross-hardware、navigation、sensing、analytics、generalizability、edge+cloud** 的框架。

---

## **2. 核心实验方法和设置**

### **使用的数据集 / 场景**
本研究未使用传统“数据集”，而是基于 **六类真实世界 DaaS 应用场景** 进行验证，涵盖仿真与实机测试：

| 应用 | 类型 | 环境 |
|------|------|------|
| Farm Survey | 静态航点任务 | Gazebo 仿真 |
| Disaster Survey (Human-in-the-loop) | 动态航点注入 | Gazebo 仿真 |
| Situation Awareness | 传感器驱动跟踪 | 实地户外（IISc 校园） |
| Vehicle Tracking | 分析驱动任务更新 | Gazebo 仿真 |
| Search and Rescue | 分析驱动任务中断 | 实地户外 |
| Radio Tower Inspection | 分析驱动中断与检查 | Gazebo 仿真 |

视频流来自无人机摄像头（720p@30FPS 或 15FPS），部分任务使用预定义移动目标（如 Prius Hybrid 车模）。

### **实验设置**

#### **硬件配置**
- **边缘设备**：NVIDIA Jetson Orin Nano（6核 ARM CPU + 1024 CUDA 核，8GB 统一内存）
- **云端**：AWS Lambda（用于 cloud-based inference）
- **工作站（仿真）**：AMD Ryzen 9 3900X + 24GB RAM + NVIDIA RTX 3090（128GB VRAM）
- **无人机**：
  - 实体：DJI Ryze Tello（搭载摄像头）
  - 仿真：Quadrotor Base（支持 PX4 控制器）

#### **软件栈**
- 仿真环境：Gazebo (GZ Garden) + ROS2 Humble
- 容器化：Docker（隔离服务运行环境）
- 深度学习模型：YOLOv8-nano（目标检测）、ResNet18+SVM（姿态识别）

### **评估指标**
- **Lines of Code (LoC)**：衡量开发效率
- **End-to-End Latency**：从图像采集到控制指令下发的时间
- **AeroDaaS Overhead**：排除 DNN 推理后的系统开销
- **资源占用**：CPU、GPU、RAM、VRAM 使用率
- **任务完成质量**：是否成功完成 mission update/abort、轨迹精度等

### **基线方法对比**
与以下平台进行 LoC 对比：
- **PX4**
- **Aerostack2**
- **Native DJI TelloPy**

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 指标 | 结果 |
|------|------|
| **平均端到端延迟（仿真）** | 35 ms |
| **平均端到端延迟（实机）** | 73 ms |
| **AeroDaaS 开销（中位数）** | ≤ 20 ms（仿真 13ms，实机 17ms） |
| **内存占用增加** | ≤ 1 GB（Orin Nano 上约 3.2GB 总用量） |
| **VRAM 占用（RTX 3090）** | 增加 ~4.8 GB（+20%） |
| **所需代码量（LoC）** | ≤ 40 行（所有应用） |

> 图10–11 显示，AeroDaaS 自身引入的处理延迟极低，仅为 **10–17ms**，远小于 DNN 推理时间。

### **与基线方法的对比结果**

#### **开发效率（LoC 对比）**
- **PX4**：>200 LoC（analytics-driven），>130 LoC（waypoint-driven）
- **Aerostack2**：>140 LoC / >100 LoC
- **DJI TelloPy**：>100 LoC / ~70 LoC
- **AeroDaaS**：<40 LoC（两种任务均适用）

👉 **AeroDaaS 减少代码量达 5× 以上**，显著提升开发效率。

#### **调度器灵活性验证**
- 使用 **NearestNeighborScheduler** 替换默认调度器，在 Disaster Survey 中减少飞行距离 **22 米**，体现节能潜力。
- 使用 **OcularOne 的 DEMS analytics scheduler**，实现 VIP detection 与 body pose analysis 的并行调度，端到端延迟 **158ms**，其中 AeroDaaS 开销仅 **10ms**。

### **消融实验结果**
虽然文中未明确标注“ablation study”，但通过多个应用场景展示了各组件的作用：

- **MonitoringAnalytics 模块**：实时输出电池电量、高度、GPS 位置（图16），可用于后续 energy-aware handoff 或 PID 控制优化。
- **stat-stream 传感器**：实现 13 种状态的状态机（图3），增强任务鲁棒性。
- **PriorityQueue 支持**：允许高优先级任务（如 car tracking）抢占原 mission（图8d）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **AeroDaaS 能有效统一 DaaS 开发流程**  
   通过高层 API 实现 **sensing、navigation、analytics** 的解耦与组合，极大简化了复杂无人机应用的开发。

2. **系统开销极低，适合实时应用**  
   在嵌入式设备（Orin Nano）和高性能工作站上均表现出色，**<20ms 的处理延迟** 和 **<1GB 内存增长** 证明其轻量化设计成功。

3. **支持动态任务重调度**  
   成功实现了 **mission update**（如车辆追踪后返回原路径）和 **mission abort**（如搜救中立即转向目标）两类高级行为，体现了闭环智能决策能力。

4. **跨平台与仿真-现实一致性良好**  
   同一份代码可在 Gazebo 和实体无人机间无缝迁移，加速了算法迭代与安全验证。

### **方法的局限性**
- 当前仅支持单机无人机，尚未扩展至 **multi-drone swarm coordination**。
- 所有实验基于 **WiFi 或局域网通信**，未考虑广域 cellular 网络下的不稳定连接影响。
- 调度器插件虽灵活，但目前只验证了几种简单策略（如 nearest neighbor），缺乏复杂优化算法（如 RL-based）集成。
- 安全机制（如避障）假设已内置，未深入讨论故障恢复与容错机制。

### **未来工作方向**
1. **引入 Agentic AI**：利用大模型生成高层任务逻辑，自动生成 AeroDaaS 应用代码。
2. **支持多用户并发应用**：在同一架无人机上协调多个用户的请求。
3. **扩展 fleet-level primitives**：支持无人机集群的任务编排与资源协同。
4. **增强弹性与容错能力**：应对网络中断、节点失效等异常情况。
5. **集成更多硬件平台**：如 Skydio、Autel 等商用无人机。

---

> ✅ **总体评价**：AeroDaaS 是一个极具前景的 **高效、灵活、可扩展** 的 DaaS 编程框架，填补了当前无人机系统在 **高层抽象、分析-导航协同、边缘-云融合** 方面的关键空白。其实验验证充分，性能优越，为未来智能空中系统的发展提供了坚实基础。

</details>

---

### 16. [SWE-Hub: A Unified Production System for Scalable, Executable Software Engineering Tasks](https://arxiv.org/abs/2603.00575)

**Authors**: Yucheng Zeng, Shupeng Li, Daxiang Dong, Ruijie Xu, Zimo Chen, Liwei Zheng, Yuxuan Li, Zhe Zhou, Haotian Zhao, Lun Tian, Heng Xiao, Tianshu Zhu, Longkun Hao, Jianmin Wu  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.00575v1  

#### Abstract
Progress in software-engineering agents is increasingly constrained by the scarcity of executable, scalable, and realistic data for training and evaluation. This scarcity stems from three fundamental challenges in existing pipelines: environments are brittle and difficult to reproduce across languag...

---

### 17. [Fair in Mind, Fair in Action? A Synchronous Benchmark for Understanding and Generation in UMLLMs](https://arxiv.org/abs/2603.00590)

**Authors**: Yiran Zhao, Lu Zhou, Xiaogang Xu, Zhe Liu, Jiafei Wu, Liming Fang  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.00590v1  

#### Abstract
As artificial intelligence (AI) is increasingly deployed across domains, ensuring fairness has become a core challenge. However, the field faces a "Tower of Babel'' dilemma: fairness metrics abound, yet their underlying philosophical assumptions often conflict, hindering unified paradigms-particular...

---

### 18. [TraceSIR: A Multi-Agent Framework for Structured Analysis and Reporting of Agentic Execution Traces](https://arxiv.org/abs/2603.00623)

**Authors**: Shu-Xun Yang, Cunxiang Wang, Haoke Zhang, Wenbo Yu, Lindong Wu, Jiayi Gui, Dayong Yang, Yukuo Cen, Zhuoer Feng, Bosi Wen, Yidong Wang, Lucen Zhong, Jiamin Ren, Linfeng Zhang, Jie Tang  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.00623v1  

#### Abstract
Agentic systems augment large language models with external tools and iterative decision making, enabling complex tasks such as deep research, function calling, and coding. However, their long and intricate execution traces make failure diagnosis and root cause analysis extremely challenging. Manual...

---

### 19. [Harmonizing Dense and Sparse Signals in Multi-turn RL: Dual-Horizon Credit Assignment for Industrial Sales Agents](https://arxiv.org/abs/2603.01481)

**Authors**: Haojin Yang, Ai Jian, Xinyue Huang, Yiwei Wang, Weipeng Zhang, Ke Zeng, Xunliang Cai, Jingqing Ruan  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.01481v1  

#### Abstract
Optimizing large language models for industrial sales requires balancing long-term commercial objectives (e.g., conversion rate) with immediate linguistic constraints such as fluency and compliance. Conventional reinforcement learning often merges these heterogeneous goals into a single reward, caus...

---

### 20. [Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents](https://arxiv.org/abs/2603.01548)

**Authors**: Neeraj Bholani  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.01548v1  

#### Abstract
Tool-using LLM agents face a reliability-cost tradeoff: routing every decision through the LLM improves correctness but incurs high latency and inference cost, while pre-coded workflow graphs reduce cost but become brittle under unanticipated compound tool failures. We present Self-Healing Router, a...

---

### 21. [S5-HES Agent: Society 5.0-driven Agentic Framework to Democratize Smart Home Environment Simulation](https://arxiv.org/abs/2603.01554)

**Authors**: Akila Siriweera, Janani Rangila, Keitaro Naruse, Incheon Paik, Isuru Jayanada  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.01554v1  

#### Abstract
The smart home is a key domain within the Society 5.0 vision for a human-centered society. Smart home technologies rapidly evolve, and research should diversify while remaining aligned with Society 5.0 objectives. Democratizing smart home research would engage a broader community of innovators beyon...

---

### 22. [RLAR: An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models](https://arxiv.org/abs/2603.00724)

**Authors**: Andrew Zhuoer Feng, Cunxiang Wang, Bosi Wen, Yidong Wang, Yu Luo, Hongning Wang, Minlie Huang  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.00724v1  

#### Abstract
Large language model alignment via reinforcement learning depends critically on reward function quality. However, static, domain-specific reward models are often costly to train and exhibit poor generalization in out-of-distribution scenarios encountered during RL iterations. We present RLAR (Reinfo...

---

### 23. [LaSER: Internalizing Explicit Reasoning into Latent Space for Dense Retrieval](https://arxiv.org/abs/2603.01425)

**Authors**: Jiajie Jin, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengjun Xie, Yutao Zhu, Zhicheng Dou  
**Category**: cs.CL  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.01425v1  

#### Abstract
LLMs have fundamentally transformed dense retrieval, upgrading backbones from discriminative encoders to generative architectures. However, a critical disconnect remains: while LLMs possess strong reasoning capabilities, current retrievers predominantly utilize them as static encoders, leaving their...

---

### 24. [FWeb3: A Practical Incentive-Aware Federated Learning Framework](https://arxiv.org/abs/2603.00666)

**Authors**: Peishen Yan, Shuang Liang, Yang Hua, Linshan Jiang, Kuai Yu, Yulin Sun, Yaozhi Zhang, Tao Song, Ningxin Hu, Xinran Liang, Bingsheng He, Haibing Guan  
**Category**: cs.DC  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.00666v1  

#### Abstract
Federated learning (FL) enables collaborative model training over distributed private data. However, sustaining open participation requires incentive mechanisms that compensate contributors for their resources and risks. Enabled by Web3 primitives, especially blockchains, recent FL proposals incorpo...

---

### 25. [Interpretable Cross-Network Attention for Resting-State fMRI Representation Learning](https://arxiv.org/abs/2603.00786)

**Authors**: Karanpartap Singh, Adam Turnbull, Mohammad Abbasi, Kilian Pohl, Feng Vankee Lin, Ehsan Adeli  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.00786v1  

#### Abstract
Understanding how large-scale functional brain networks reorganize during cognitive decline remains a central challenge in neuroimaging. While recent self-supervised models have shown promise for learning representations from resting-state fMRI, their internal mechanisms are difficult to interpret, ...

---

### 26. [Subliminal Signals in Preference Labels](https://arxiv.org/abs/2603.01204)

**Authors**: Isotta Magistrali, Fr\'ed\'eric Berdoz, Sam Dauncey, Roger Wattenhofer  
**Category**: cs.LG  
**Published**: 2026-03-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.01204v1  

#### Abstract
As AI systems approach superhuman capabilities, scalable oversight increasingly relies on LLM-as-a-judge frameworks where models evaluate and guide each other's training. A core assumption is that binary preference labels provide only semantic supervision about response quality. We challenge this as...

---

### 27. [EmCoop: A Framework and Benchmark for Embodied Cooperation Among LLM Agents](https://arxiv.org/abs/2603.00349)

**Authors**: Hanqing Yang, Shiyu Chen, Narjes Nourzad, Marie Siew, Jingdi Chen, Carlee Joe-Wong  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00349v1  

#### Abstract
Real-world scenarios increasingly require multiple embodied agents to collaborate in dynamic environments under embodied constraints, as many tasks exceed the capabilities of any single agent. Recent advances in large language models (LLMs) enable high-level cognitive coordination through reasoning,...

---

### 28. [NeuroHex: Highly-Efficient Hex Coordinate System for Creating World Models to Enable Adaptive AI](https://arxiv.org/abs/2603.00376)

**Authors**: Quinn Jacobson, Joe Luo, Jingfei Xu, Shanmuga Venkatachalam, Kevin Wang, Dingchao Rong, John Paul Shen  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00376v1  

#### Abstract
\textit{NeuroHex} is a hexagonal coordinate system designed to support highly efficient world models and reference frames for online adaptive AI systems. Inspired by the hexadirectional firing structure of grid cells in the human brain, NeuroHex adopts a cubic isometric hexagonal coordinate formulat...

---

### 29. [LOGIGEN: Logic-Driven Generation of Verifiable Agentic Tasks](https://arxiv.org/abs/2603.00540)

**Authors**: Yucheng Zeng, Weipeng Lu, Linyun Liu, Shupeng Li, Zitian Qu, Chenghao Zhu, Shaofei Li, Zhengdong Tan, Mengyue Liu, Haotian Zhao, Zhe Zhou, Jianmin Wu  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00540v1  

#### Abstract
The evolution of Large Language Models (LLMs) from static instruction-followers to autonomous agents necessitates operating within complex, stateful environments to achieve precise state-transition objectives. However, this paradigm is bottlenecked by data scarcity, as existing tool-centric reverse-...

---

### 30. [Advancing Multimodal Judge Models through a Capability-Oriented Benchmark and MCTS-Driven Data Generation](https://arxiv.org/abs/2603.00546)

**Authors**: Zeyu Chen, Huanjin Yao, Ziwang Zhao, Min Yang  
**Category**: cs.AI  
**Published**: 2026-03-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00546v1  

#### Abstract
Using Multimodal Large Language Models (MLLMs) as judges to achieve precise and consistent evaluations has gradually become an emerging paradigm across various domains. Evaluating the capability and reliability of MLLM-as-a-judge systems is therefore essential for ensuring trustworthy assessment. Ex...

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
