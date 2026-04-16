# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-16 07:16:58 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](https://arxiv.org/abs/2604.13634)

**Authors**: Xuwen Zhou, Fangxin Liu, Chao Wang, Xiao Zheng, Hao Zheng, Min He, Li Jiang, Haibing Guan  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.13634v1  

#### Abstract
Speculative decoding accelerates autoregressive generation by letting draft tokens bypass full verification, but conventional frameworks suffer from frequent false rejections, particularly when draft models produce semantically correct but lexically divergent outputs. In this paper, we present Calib...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Speculative Decoding (SD)** 虽然通过 draft model 加速了 LLM 推理，但其验证机制依赖于严格的 **token-level exact matching**，导致大量“语义正确但词法不同”的 draft tokens 被错误拒绝（False Rejections）。例如，`x` 和 `*` 在数学表达中含义相同，却因符号差异被丢弃，造成计算资源浪费。

这种“**Semantics-Alignment Mismatch**”限制了现代小型语言模型（SLMs）作为 draft model 的潜力——尽管它们具备较强的推理能力，但输出风格或词汇选择上的微小差异仍会导致高拒绝率。

---

### 提出了什么新方法或新思路
本文提出 **Calibrated Speculative Decoding (CSD)**，一种无需训练、轻量级的推理加速框架，核心思想是：

> **“Frequency-Guided Candidate Selection, Probability-Guarded Acceptance”**

CSD 包含两个关键模块：

1. **Online Correction Memory (OCM)**  
   - 在线记录历史中频繁出现的 draft-target token 对（如 `"x"` vs `"*"`），形成一个“纠错记忆库”。
   - 当发生拒绝时，若当前 token 对在 OCM 中频率超过阈值，则将其作为候选进行救援。

2. **Semantic Consistency Gating (SCG)**  
   - 不再要求完全匹配，而是基于目标模型对候选 token 的置信度判断是否可接受。
   - 使用 logit 空间中的概率比（`log(p(x)/p(t*)) ≥ log τ`）进行快速、零开销的语义一致性校验，避免 Softmax 开销。

该双阶段机制允许系统动态恢复本应有效的 draft tokens，同时保证生成质量。

---

### 相比现有方法的优势
| 方法 | 缺陷 | CSD 如何改进 |
|------|------|---------------|
| **Standard SD** | 严格匹配导致 false rejection 多 | 引入频率引导 + 概率门控，智能恢复有效 token |
| **Lossy SD / Static Gating** | 固定阈值过于粗糙，易引入错误 | 动态结合上下文置信度与历史模式，更安全 |
| **Tree-based / Learned Verifiers** | 架构复杂、需训练、额外开销大 | 完全无需训练，仅增加 <0.02% 开销 |
| **Swift / Lookahead** | 层跳或迭代解码带来额外计算瓶颈 | 直接提升 acceptance rate，无结构性负担 |

✅ **优势总结**：
- **Training-free**：无需任何参数更新或微调。
- **Lightweight**：算法开销极低（<0.02% 总延迟）。
- **通用性强**：适用于所有基于 rejection sampling 的 SD 框架。
- **性能提升显著**：最高达 **2.33× 吞吐加速**，且保持甚至提升准确率。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖多个任务领域，确保泛化性：
- **GSM8K**：数学推理（8-shot）
- **MATH500**：高等数学问题（4-shot）
- **HumanEval**：代码生成（0-shot, Pass@1）
- **CNN/DailyMail (CNN/DM)**：摘要任务（0-shot, ROUGE-L）
- （附录）**IFEval**：指令遵循能力
- （附录）**RULER**：长上下文理解

---

### 实验设置和评估指标

#### 模型配置
- **Llama-3 系列**：`Llama-3-70B-Instruct`（target） + `Llama-3.2-1B-Instruct`（draft）
- **Qwen-2.5 系列**：`Qwen-2.5-72B-Instruct`（target） + `Qwen-2.5-7B-Instruct`（draft）

#### 主要评估指标
| 指标 | 含义 |
|------|------|
| **Acc** | 下游任务准确率（Pass@1 或 ROUGE-L） |
| **Tp (Throughput)** | 每秒生成 token 数（tokens/s） |
| **Spd (Speedup)** | 相对于 Vanilla Decoding 的加速比 |
| **AR (Acceptance Rate)** | draft token 被接受的比例 |
| **Avg. Speedup** | 所有任务平均加速比 |

#### 基线方法对比
| 类别 | 方法 |
|------|------|
| 基础基准 | Vanilla Decoding, SpecDecode |
| 松弛验证 | Lossy SD (T=0.6) |
| 高级加速 | Swift, Lookahead Decoding |
| 并发语义方法 | Fly, Reflective Verification |

#### 实现细节
- 使用 PyTorch + HuggingFace Transformers 实现
- 测试环境：8×NVIDIA H20 GPU，单 batch 推理
- CSD 参数设置：lookahead steps `γ=6`, frequency threshold `λ=6`, gating threshold `τ=0.01`
- 校准阶段使用 2k–8k 无标签样本进行离线初始化（zero online overhead）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | Avg. Speedup (Llama) | 最高 Speedup | Acc 提升 |
|------|------------------------|--------------|-----------|
| **Vanilla Decoding** | 1.00× | — | — |
| **SpecDecode** | 1.75× | 1.90× | ≈持平 |
| **Lossy SD (T=0.6)** | 1.85× | 2.04× | 小幅下降或持平 |
| **CSD (Ours)** | **2.02×** | **2.33×** (MATH500/HumanEval) | ✅ 显著提升 |

#### 在 Llama-3 上的具体表现：
- **MATH500**: 从 1.89× → **2.33×**, Acc 从 45.4 → **48.0** (+2.0)
- **HumanEval**: 从 1.90× → **2.33×**, Acc 从 76.8 → **79.3** (+2.5)
- **CNN/DM**: 从 1.49× → **1.59×**, Acc 持平

#### 在 Qwen-2.5 上的表现：
- 平均加速从 **1.66× → 1.86×**
- MATH500 达到 **2.12× speedup**，Acc 提升 +1.0

---

### 与并发语义方法对比（Table 2）

| 方法 | Avg. Speedup | Acceptance Rate | Acc |
|------|----------------|------------------|-----|
| **SpecDecode** | 1.56× | 38.0% | 58.7 |
| **Fly** | 1.75× | 44.8% | 58.9 |
| **Reflective Verification** | 1.76× | 52.6% | 59.9 |
| **CSD (Ours)** | **1.89×** | **50.2%** | **59.9** |

- CSD 在 **wall-clock speedup** 上优于 Reflective（1.89× vs 1.76×），因其不引入冗余 prompt 模板。
- 相比 Fly，CSD 更鲁棒于边界和风格变化，acceptance rate 更稳定。

---

### 消融实验结果（Table 3）

| 变体 | HumanEval Acc | AR | Speedup |
|------|----------------|-------|----------|
| **SpecDecode (baseline)** | 76.8 | 59.7% | 1.00× |
| **SD w/ OCM only** | 70.7 | 71.6% | 1.24× |
| **SD w/ SCG only** | 70.7 | 88.6% | 1.48× |
| **CSD (full)** | **79.3** | **67.9%** | **1.23×** |

🔍 **发现**：
- 单独使用 OCM 或 SCG 会显著降低 accuracy（跌至 ~70.7），说明粗粒度过滤会引入有害 hallucination。
- **只有两者协同工作**，才能实现“高接受率 + 高准确性”的最优平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **False Rejections 是普遍且可预测的**  
   - 分析显示，前 20% 的 rejection patterns 占据了 69% 的总拒绝数（Pareto 分布），表明存在大量重复性的词法差异。
   
2. **历史频率 + 上下文置信度 = 安全恢复的关键**  
   - 仅靠频率不可靠（context-sensitive errors），仅靠置信度也不够（忽略系统性偏差），必须联合建模。

3. **CSD 实现了真正的“语义级”验证**  
   - 成功将验证逻辑从 “exact match” 升级为 “semantic equivalence”，突破传统 SD 的天花板。

4. **不仅加速，还能提准**  
   - 在 HumanEval 和 MATH500 上分别提升 **+2.5 和 +2.0 点准确率**，表明 draft model 提供了有价值的替代路径，帮助跳出 greedy decoding 的局部最优。

5. **开销几乎为零**  
   - CSD 带来的额外延迟仅为 **0.01%~0.02%**，真正做到了“无痛加速”。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **偏离分布精确性** | 不再满足 rejection sampling 的理论分布一致性，属于 heuristic 改进，可能在未知任务上不稳定。 |
| **依赖 draft model 质量** | 若 draft model 本身 hallucinates 严重，OCM 无法学习有效 pattern，SCG 也会拒绝多数候选。 |
| **高并发扩展未验证** | OCM 的在线更新机制在多请求并行环境下可能存在同步竞争问题，尚未在生产级系统中测试。 |
| **缺乏工业引擎集成** | 当前未接入 vLLM、TensorRT-LLM 等主流推理框架，实际部署潜力有待验证。 |

---

### 未来工作方向
1. **构建跨模型通用的 Correction Memory**  
   - 探索是否可以预训练一个通用 OCM，适配多种 target-draft 组合。
   
2. **动态调整 gating threshold τ**  
   - 根据输入难度、领域自动调节语义宽容度，进一步优化 trade-off。

3. **支持 streaming 和 interactive 场景**  
   - 将 CSD 应用于对话系统，在持续交互中不断进化 OCM。

4. **与 Medusa / Eagle 等多头解码结合**  
   - 探索 CSD 是否可用于提升 tree-based speculative inference 的 acceptance 效率。

5. **集成至 vLLM 等工业系统**  
   - 实现端到端高性能服务，验证其在真实负载下的稳定性与收益。

---

> ✅ **总体评价**：  
> CSD 是一项简洁而深刻的改进，抓住了当前 Speculative Decoding 中“过度严格验证”这一核心痛点，以极低成本实现了显著性能突破。它不仅是工程上的胜利，也为 LLM 推理中的“语义等价性”研究开辟了新方向。

</details>

---

### 2. [A KL Lens on Quantization: Fast, Forward-Only Sensitivity for Mixed-Precision SSM-Transformer Models](https://arxiv.org/abs/2604.13440)

**Authors**: Jason Kong, Nilesh Prasad Pandey, Flavio Ponzina, Tajana Rosing  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.13440v1  

#### Abstract
Deploying Large Language Models (LLMs) on edge devices faces severe computational and memory constraints, limiting real-time processing and on-device intelligence. Hybrid architectures combining Structured State Space Models (SSMs) with transformer-based LLMs offer a balance of efficiency and perfor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**A KL Lens on Quantization: Fast, Forward-Only Sensitivity for Mixed-Precision SSM-Transformer Models**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **混合架构模型（SSM-Transformer）在边缘设备部署时面临量化敏感性不均的问题**：
  - 当前主流的均匀量化（Uniform Quantization）策略在 SSM-Transformer 混合模型上表现不佳，因为不同组件对量化的鲁棒性差异显著。
  - 特别是 SSM 中的 `state transition` 和 `output projections` 对量化噪声极为敏感，而 Transformer 部分（如 FFN、Attention）则更鲁棒。
  - 现有基于梯度或 SQNR 的敏感性分析方法在语言建模任务中无法准确反映实际性能下降。

### 🚀 提出的新方法与创新
- **提出一种轻量级、无需反向传播的敏感性分析框架**：
  - 名为 **KL-based forward-only sensitivity analysis**，仅依赖前向推理信号进行层敏感性排序。
  - 使用 **KL divergence（Kullback-Leibler Divergence）** 作为量化敏感性的代理指标，衡量教师模型（Full Precision）与学生模型（Quantized）输出分布之间的差异。
  - 引入 **KLstudent→teacher** 方向（而非传统 KLteacher→student），更有效地捕捉由量化引起的“错误概率分配”行为。

- **理论证明 KL 比 SQNR 更适合作为语言模型的敏感性指标**：
  - 形式化推导表明：**Perplexity 可分解为 H(p) + DKL(p||q)**，因此 KL divergence 是 PPL 变化的直接上界。
  - 而 SQNR 在 logits 层面测量保真度，但存在非单调性——即使 SQNR 很低，PPL 也可能不变（例如常数偏移不影响 softmax 输出）。

### 🔍 相比现有方法的优势
| 维度 | 本文方法（KL-MP） | 传统方法（如 HAWQ、SQNR-based） |
|------|------------------|-------------------------------|
| 是否需要梯度 | ❌ 否（Forward-only） | ✅ 是（Backpropagation required） |
| 是否需微调/重训练 | ❌ 否（Post-training only） | ✅ 通常需要校准或微调 |
| 敏感性指标相关性 | ⬆️ KLstudent→teacher 与 PPL 高度一致（Kendall’s τ ≈ 0.79） | ⬇️ SQNR 与 PPL 关联弱（τ ≈ 0.71） |
| 适用场景 | 边缘设备、隐私受限环境（无训练权限） | 数据可访问且允许计算梯度 |

> ✅ **优势总结**：**更快、更轻、更适合边缘部署的语言模型量化决策工具**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **WikiText-2**：用于评估语言模型的 Perplexity（PPL），标准文本建模基准。
- **Calibration Set**：少量代表性样本（未指定具体大小），用于执行量化并收集 forward-pass 输出以计算 KL/SQNR。

### ⚙️ 实验设置
- **模型架构**：
  - **Hybrid Models**：Hymba、Zamba
  - **Pure SSM Models**：Mamba-130M、Mamba-380M、Mamba-1.4B、Mamba2-130M
- **量化配置**：
  - **Uniform Baselines**：FP16、INT8、INT4（全模型统一精度）
  - **Mixed-Precision (MP)**：高敏感层保留 FP16，其余压缩至 INT4（CPU）或 INT8（GPU）
- **硬件平台**：
  - **Intel Lunar Lake** 平台（集成 CPU + GPU + NPU），代表下一代 AI 边缘设备。
  - 使用 **OpenVINO IR** 进行模型转换与部署。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | 主要质量指标，越低越好 |
| **Model Size (MB)** | 压缩率的关键指标 |
| **Latency (ms)** | 单次推理延迟 |
| **Throughput (FPS)** | 每秒帧数（推理速度） |
| **Kendall’s τ** | 排序一致性指标，衡量敏感性评分与真实 PPL 下降的相关性 |
| **Compression Ratio** | FP16 / Quantized Size |

### 🆚 基线方法对比
- **Uniform INT8 / INT4**：工业级常用方案
- **SQNR-based MP**：传统信号保真度驱动的混合精度
- **Hessian-based (HAWQ)**：依赖梯度的敏感性分析（虽未直接实现，但在理论上比较）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（来自 Table 4）

| Model & Config | PPL | Size | FPS | Latency |
|----------------|-----|------|-----|---------|
| **Mamba-130M (CPU)** | | | | |
| FP16 | 21.59 | 493MB | 22.6 | 44ms |
| INT8 | 25.32 | 126MB | 35.5 | 28ms |
| INT4 | 30.35 | 84MB | 39.7 | 25ms |
| **KL-MP p05 (Ours)** | **21.61** | **136MB** | **35.9** | **28ms** |
| **Mamba-1.4B (CPU)** | | | | |
| FP16 | 11.22 | 5.2GB | 2.6 | 384ms |
| INT8 | 11.25 | 1.3GB | 5.1 | 196ms |
| INT4 | 60.55 | 723MB | 5.3 | 190ms |
| **KL-MP p05 (Ours)** | **11.22** | **1.4GB** | **5.6** | **178ms** |
| **Mamba2-130M (GPU)** | | | | |
| FP16 | 46.45 | 81MB | 0.02 | 60,006ms |
| **KL-MP p02 (Ours)** | **46.46** | **45MB** | **0.29** | **3,417ms** |

> ✅ **核心亮点**：
> - KL-MP 在 **几乎保持 FP16 级别的 PPL** 的同时，实现了高达 **7.2× 模型压缩**（Mamba-1.4B 从 5.2GB → 723MB）。
> - 在 CPU 上，**吞吐量媲美甚至超过 Uniform INT4**，同时 **延迟更低**（Mamba-1.4B: 178ms vs. 190ms）。
> - 在 GPU 上，**延迟降低达 17.6×**（60s → 3.4s），得益于 INT8 加速。

### 🔁 排名相关性实验结果（Table 2）
| Metric | Avg. Kendall’s τ |
|--------|------------------|
| **KLstudent→teacher** | **0.7911** ✅ |
| SQNR(dB) | 0.7111 |
| KLteacher→student | -0.1275 |
| ΔCross Entropy | -0.0645 |

> ✅ **KLstudent→teacher 显著优于 SQNR**（p < 10⁻⁶），说明其更能准确预测哪一层量化后会导致最大 PPL 上升。

### 🔍 消融实验结果（Ablation Studies on Hymba）
- **子模块敏感性不平衡**：
  - `mamba.x_proj` 是最敏感组件（ΔPPL = 0.27），远高于其他模块（≤0.014）。
  - `mamba.dt_proj` 几乎无影响（ΔPPL = 3.6e-4），适合极低位宽表示。
- **局部热点层**：
  - Block 31 占总敏感预算的 **70%以上**，少量其他块（2, 11, 16, 22）也有小高峰。
- **MoE 与 x_proj 耦合效应**：
  - x_proj 的量化噪声会通过专家路由机制被放大，导致整体性能下降加剧。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KL divergence（尤其是 KLstudent→teacher）是比 SQNR 更优的语言模型量化敏感性指标**：
   - 因其直接关联到 Perplexity 的变化，具有理论保障。
2. **混合 SSM-Transformer 架构具有高度异质的量化敏感性**：
   - SSM 中的投影层（特别是 `x_proj`）必须保留高精度。
   - Transformer 部分（FFN、MoE）可以安全地进行激进量化。
3. **无需梯度即可实现高性能混合精度量化**：
   - 所提 forward-only 方法适用于资源受限、数据不可见的真实边缘场景。
4. **KL-guided mixed-precision 实现帕累托最优**：
   - 在精度、大小、延迟、吞吐之间取得最佳平衡，**Uniform Quantization 无法达到该区域**。

### ⚠️ 方法的局限性
- **依赖代表性 calibration set**：虽然不需要大量数据，但仍需一定数量的输入样本来估计输出分布。
- **静态分配策略**：当前为离线确定各层精度，缺乏运行时动态调整能力。
- **特定于生成式任务**：KL-based 方法在分类等任务中的泛化性有待验证。

### 🔮 未来工作方向
1. **扩展至更多混合架构**（如不同比例的 SSM/Transformer 块）。
2. **探索动态量化策略**：根据输入序列长度或内容自适应调整精度。
3. **结合硬件感知优化**：将内存带宽、缓存命中率纳入联合优化目标。
4. **应用于更大规模模型**（如 7B+ 级别的 Zamba）。

---

> 💡 **一句话总结**：  
> 本文提出了一个**快速、免梯度、基于 KL 散度的敏感性分析框架**，成功指导了 SSM-Transformer 混合模型的混合精度量化，在 **Intel Lunar Lake** 上实现了 **接近 FP16 精度 + INT4 级效率** 的卓越性能，为边缘端大模型部署提供了实用解决方案。  
> 开源代码地址：[https://github.com/jasonkongie/kl-ssm-quant](https://github.com/jasonkongie/kl-ssm-quant)

</details>

---

### 3. [OmniTrace: A Unified Framework for Generation-Time Attribution in Omni-Modal LLMs](https://arxiv.org/abs/2604.13073)

**Authors**: Qianqi Yan, Yichen Guo, Ching-Chen Kuo, Shan Jiang, Hang Yin, Yang Zhao, Xin Eric Wang  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.13073v1  

#### Abstract
Modern multimodal large language models (MLLMs) generate fluent responses from interleaved text, image, audio, and video inputs. However, identifying which input sources support each generated statement remains an open challenge. Existing attribution methods are primarily designed for classification...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*OmniTrace: A Unified Framework for Generation-Time Attribution in Omni-Modal LLMs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代 **Omni-Modal LLMs**（如 Qwen2.5-Omni、MiniCPM-o）能够处理文本、图像、音频、视频等多模态交织输入，并生成流畅的开放式输出。然而，当前缺乏有效机制来解释**每个生成语句背后的支撑证据来源**（即 attribution），尤其是在自回归解码过程中。

现有 attribution 方法（如 attention rollout、Grad-CAM、LRP）主要针对分类任务或编码器架构设计，存在以下局限：
- 依赖固定预测目标（如类别 logit）
- 无法适应动态增长的因果图（decoding step-by-step）
- 难以跨模态统一处理（text/image/audio/video 共享 token timeline）

### 🆕 提出的新方法：OmniTrace
作者提出 **OmniTrace**，一个轻量级、模型无关（model-agnostic）、信号无关（signal-agnostic）的框架，将 attribution 形式化为 **generation-time tracing problem**，在解码过程中实时追踪每一步生成 token 的来源。

#### 核心思想：
- **Generation-Aware Tracing**：在每个 decoding step $ t $，利用 token-level attribution signal（如 attention weights 或 gradients）追踪当前生成 token 最相关的输入源。
- **Span-Level Aggregation**：将 token-level 追踪结果聚合为语义连贯的输出片段（如句子或短语），提升可解释性。
- **Confidence-Weighted & Temporal Coherence Filtering**：通过置信度加权投票和时序一致性约束，筛选出简洁且可靠的支撑源，无需训练或监督。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | OmniTrace |
|------|--------|-----------|
| **适用场景** | 分类 / 固定目标 | 开放式生成（autoregressive） |
| **模态支持** | 单一模态为主 | 支持 text/image/audio/video 统一处理 |
| **粒度** | Token-level（碎片化） | Span-level（语义完整） |
| **计算方式** | Post-hoc（事后分析） | Generation-time（在线追踪） |
| **灵活性** | 依赖特定信号机制 | Signal-agnostic（兼容多种 scoring functions） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验覆盖视觉、音频、视频三大模态，共 **759 个样本**，涵盖多种任务类型：

| Modality | Task Type | Dataset | #Examples | 特点 |
|---------|----------|--------|----------|-----|
| Visual | QA | Mantis-eval [18] | 200 | 多图推理 |
| Visual | Summarization | MMDialog [19], CliConSummation [20] | 257 | 图文对话摘要 |
| Audio | QA / Summarization | MMAU [21], MISP [22] | 134 / 66 | 会议录音理解与摘要 |
| Video | QA | Video-MME [23] | 102 | 视频问答，含时间戳标注 |

### ⚙️ 实验设置
- **模型**：Qwen2.5-Omni-7B 和 MiniCPM-o-4.5-9B（均为开源 decoder-only omni-modal LLMs）
- **硬件**：H200 GPU，确定性解码（deterministic decoding）确保可复现
- **解码策略**：默认 greedy decoding
- **attribution signals**：
  - `OTAttMean`：跨层平均 attention 权重
  - `OTRawAtt`：最后一层原始 attention
  - `OTAttGrads`：基于梯度的 attention 影响力（memory-intensive）

### 📊 评估指标
| 模态 | 评估方式 | 指标 |
|------|--------|------|
| Text/Image | 离散 span 匹配 | **Span-level F1**（multi-label 预测） |
| Audio/Video | 时间区间匹配 | **Time-F1**（按秒分桶，二值标签 F1） |

### 🆚 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Self-Attribution** | 用同一模型对自身输出进行 post-hoc 自我归因（prompting） |
| **Embedprocessor** | 使用模型自身的 processor embedding 计算生成 span 与源之间的余弦相似度 |
| **EmbedCLIP** | 使用 CLIP-ViT-Large-Patch14 提取图文嵌入并计算相似度 |
| **Random** | 随机分配源作为负例对照 |

> ❗注意：embedding-based 方法不适用于 audio/video 的连续时间戳任务（标记为 ×），因为无法定义任意时间段的语义嵌入。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2）

#### 在 Qwen2.5-Omni 上的表现（F1 值）：

| Method | Visual Summ. (Text) | Visual Summ. (Image) | Visual QA (Image) | Audio Summ. (Time) | Audio QA (Time) | Video QA (Time) |
|-------|---------------------|------------------------|--------------------|---------------------|------------------|------------------|
| **OTAttMean** | **75.66** | **76.59** | 56.60 | **83.12** | **49.90** | **40.16** |
| OTRawAtt | 72.51 | 51.82 | **65.44** | 76.69 | 47.64 | 36.53 |
| OTAttGrads | 67.70 | 42.24 | 65.02 | + | 47.56 | + |
| Self-Attribution | 9.25 | 40.60 | 61.03 | 4.43 | 29.01 | 13.67 |
| Embedprocessor | 17.30 | 14.55 | 36.88 | × | × | × |
| Random | 10.98 | 8.38 | 24.70 | × | × | × |

#### 在 MiniCPM-o-4.5 上的表现：

| Method | Visual Summ. (Text) | Visual Summ. (Image) | Visual QA (Image) | Audio Summ. (Time) | Audio QA (Time) | Video QA (Time) |
|-------|---------------------|------------------------|--------------------|---------------------|------------------|------------------|
| **OTRawAtt** | 37.32 | **76.46** | **45.41** | **49.21** | 41.06 | 21.59 |
| OTAttMean | 30.57 | 75.43 | 37.00 | 33.52 | **46.94** | **22.85** |
| Self-Attribution | 9.06 | 66.53 | 39.39 | 0.08 | 34.66 | 18.26 |

> ✅ 所有 OmniTrace 变体均显著优于所有 post-hoc 基线方法，尤其在 audio/video 时间定位任务上优势明显。

### 🔍 消融实验结果（Table 3）

在 Qwen2.5-Omni + OTAttMean 设置下进行 ablation study，验证各组件作用：

| 方法变体 | Text F1 | Image F1 | Audio F1 | Video F1 |
|--------|--------|---------|---------|---------|
| **Full Model** | 75.66 | **76.59** | 49.90 | 40.16 |
| w/o POS Weighting | 76.69 | 20.79 | 50.07 | 37.46 |
| w/o Confidence Weight | 74.59 | 19.88 | 50.83 | 38.82 |
| w/o Run Coherence | 75.93 | 19.88 | 48.69 | 35.79 |
| w/o pmin Filtering | 76.00 | 19.88 | 48.85 | 36.22 |

> ⚠️ **关键发现**：移除任一过滤机制都会导致 **image attribution 性能崩溃至 ~20 F1**，说明：
- 原始 token-level tracing 极不稳定
- **POS-aware weighting、confidence shaping、run-level coherence、pmin filtering** 是实现稳定跨模态归因的关键

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Generation-time tracing 更有效**  
   相比 post-hoc 方法（如 self-attribution），在生成过程中动态追踪能获得更准确、稳定的 attribution 结果。

2. **Span-level aggregation 提升可解释性**  
   将 token-level 信号聚合成语义完整的 span-level 解释，显著提高 human-interpretable 能力。

3. **OmniTrace 具有强鲁棒性和泛化能力**  
   - 在不同 scoring signals（attention/gradient）下表现一致
   - 跨模态通用（text/image/audio/video）
   - 不依赖特定模型结构（适用于 Qwen 和 MiniCPM）

4. **高质量 ASR 分割至关重要（Fig 2a）**  
   使用 Paraformer/Scribe v2 等高质量 ASR 显著优于 raw token 输入，说明**语义连贯的时间段划分是 audio attribution 成功的前提**。

5. **多模态互补提升 attribution 准确率（Fig 2b）**  
   同时使用 visual + audio 输入比单独任一模态效果更好，证明 OmniTrace 能有效融合异构证据。

6. **attribution quality ≠ generation quality（Fig 4）**  
   - 有些错误答案仍有高 attribution F1 → 模型“理由正确但结论错”
   - ROUGE-L 与 attribution F1 弱相关 → attribution 是独立维度，需专门评估

7. **High faithfulness（可信度高）**  
   在 multiple-choice QA 中，OmniTrace 对最终选项的 attribution 与其实际选择高度一致（Top-1 consistency 达 **93.84%**），表明其反映的是模型真实决策路径。

### ⚠️ 方法的局限性
- **早期位置偏差（early-token bias）**：归因倾向于集中在输入序列前半部分（CDF 曲线高于对角线）
- **图像归因更脆弱**：相比文本，视觉 grounding 更容易受噪声干扰，需强过滤机制
- **梯度计算内存开销大**：OTAttGrads 在长音频/视频输入中不可行（+ 表示未报告）
- **依赖 ASR 质量**：低质量语音识别会严重损害 audio attribution 效果

### 🔮 未来工作方向
- 探索更高效的 generation-time attribution scoring 方法（如 probe-based 或 cached attention）
- 设计抗偏置机制缓解 positional bias
- 扩展到更多模态（如 3D point clouds、sensor streams）
- 构建交互式 attribution interface 用于模型调试与用户信任建立
- 推动标准化 benchmark 与 human evaluation protocol

---

## 📎 附加信息
- **项目主页**：[https://github.com/eric-ai-lab/OmniTrace](https://github.com/eric-ai-lab/OmniTrace)
- **代码与数据发布计划**：将开源完整实现、测试集及评估脚本，遵循 MIT License
- **关键词**：Omni-Modal LLM, Generation-Time Attribution, Span-Level Explanation, Cross-Modal Tracing, Interpretability, Faithfulness, Ablation Study

</details>

---

### 4. [FAST: A Synergistic Framework of Attention and State-space Models for Spatiotemporal Traffic Prediction](https://arxiv.org/abs/2604.13453)

**Authors**: Xinjin Li, Jinghan Cao, Mengyue Wang, Yue Wu, Longxiang Yan, Yeyang Zhou, Ziqi Sha, Yu Ma  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.13453v1  

#### Abstract
Traffic forecasting requires modeling complex temporal dynamics and long-range spatial dependencies over large sensor networks. Existing methods typically face a trade-off between expressiveness and efficiency: Transformer-based models capture global dependencies well but suffer from quadratic compl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FAST: A Synergistic Framework of Attention and State-space Models for Spatiotemporal Traffic Prediction

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
交通流量预测需要同时建模**复杂的时序动态**、**长距离空间依赖**以及在大规模传感器网络上的**计算可扩展性**。现有方法面临以下权衡：
- **Transformer-based 模型**：能捕捉全局依赖，但具有 **quadratic complexity**，难以扩展到长历史窗口和大规模路网。
- **GNN-based 模型**：利用图结构建模空间相关性，但受限于局部消息传递机制，难以捕获远距离节点间的功能关联。
- **Selective State-Space Models (如 Mamba)**：具备线性复杂度，适合长序列建模，但在处理图结构数据中的显式空间推理方面能力较弱。

因此，如何在**表达力、效率与泛化性之间取得平衡**是当前的核心挑战。

---

### 🚀 提出的新方法与创新思路

作者提出 **FAST**（**F**ramework of **A**ttention and **S**tate-space for **T**raffic prediction），其核心思想是：  
> **将时间建模与空间建模解耦，并为两者分配最适合的模型组件**。

#### 主要创新点如下：

1. **Temporal-Spatial-Temporal (TST) 架构设计**
   - 采用 `Temporal → Spatial → Temporal` 的交替堆叠结构：
     - 第一个 **Temporal Attention Module** 提取每个节点的时间特征；
     - 中间的 **Mamba-based Spatial Module** 在传感器维度上传播信息，建模跨节点的长程空间依赖；
     - 第二个 **Temporal Attention Module** 在更新后的空间上下文中进一步优化时间动态。
   - 这种设计实现了**时空交互的迭代增强**。

2. **角色专精（Role-Specialized）建模**
   - **Attention 用于时间维度**：灵活捕捉短期波动与长期趋势；
   - **Mamba 用于空间维度**：以线性复杂度实现高效的跨传感器传播，避免显式邻接矩阵依赖。

3. **Learnable Multi-Source Spatiotemporal Embedding**
   - 融合四类输入信息：
     - 历史交通流 (`Xdata`)
     - 时间上下文（day-of-week, time-of-day）
     - 节点身份与位置联合编码（`Epn`）
   - 统一嵌入空间有助于建模周期性和异质性行为。

4. **Hierarchical Skip Prediction Mechanism**
   - 引入多层级跳跃连接，聚合不同深度块的输出表示，提升多步预测稳定性并促进特征复用。

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **准确性** | 在多个指标上全面超越 GNN、Transformer、Attention 和 Mamba 类基线模型 |
| **效率** | 空间模块使用 Mamba 实现 $O(N)$ 复杂度（$N$: 传感器数量），显著优于 Attention 的 $O(N^2)$ |
| **可扩展性** | 适用于大规模传感器网络部署 |
| **通用性** | 不依赖预定义图结构，通过数据驱动学习动态空间依赖 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个广泛使用的现实世界交通数据集上进行验证：
- **PeMS04**
- **PeMS07**
- **PeMS08**

这些数据来自加州高速公路系统的感应线圈检测器，采样频率为每 **5分钟一次**。

| 数据集 | 传感器数 (N) | 时间长度 | 特点 |
|--------|----------------|-----------|-------|
| PeMS04 | ~307 | ~15k | 中等规模，交通动态较复杂 |
| PeMS07 | ~883 | ~15k | 规模较大，覆盖区域广 |
| PeMS08 | ~170 | ~15k | 小规模但高密度观测 |

---

### ⚙️ 实验设置

- **任务类型**：标准多步预测（multi-step forecasting）
- **输入历史**：最近 12 个时间步（即过去 1 小时）
- **预测目标**：未来 12 个时间步（即未来 1 小时）
- **评估指标**：
  - **MAE**（Mean Absolute Error）
  - **RMSE**（Root Mean Squared Error）
  - **MAPE**（Mean Absolute Percentage Error）

- **训练配置**：
  - 硬件：NVIDIA RTX 3080 Ti GPU
  - 框架：PyTorch 2.0.0
  - 优化器：AdamW（lr=0.001, dropout=0.1）
  - 批大小：32
  - 最大训练轮次：100，早停策略（patience=5）

- **划分比例**：训练 : 验证 : 测试 = 8 : 1 : 1

---

### 🆚 基线方法对比

涵盖四大类代表性模型：

| 类别 | 基线模型 |
|------|----------|
| **Time Series Forecasting** | iTransformer, PatchTST |
| **Graph Neural Network-Based** | STGCN, DCRNN, GWNET, STGNCDE |
| **Attention-Based** | GMAN, ST-WA |
| **Mamba-Based** | MCST-Mamba, TSMamba |

---

## 3. 主要实验结果和性能指标

### 📊 性能汇总（来自 Table I）

| Dataset | Metric | Best Baseline | **FAST (Ours)** | Improvement |
|--------|--------|----------------|------------------|-------------|
| **PeMS04** | MAE | 19.08 (TSMamba) | **19.00** | ↓0.08 |
|          | RMSE | 30.94 (ours) | **30.94** | SOTA |
|          | MAPE | 12.53 (ST-WA) | 13.44 | — |
| **PeMS07** | MAE | 20.50 (ours) | **20.50** | SOTA |
|           | RMSE | 33.58 (ours) | **33.58** | SOTA |
|           | MAPE | 8.63 (ours) | **8.63** | SOTA |
| **PeMS08** | MAE | 15.08 (GWNET) | **14.66** | ↓**2.8%** |
|           | RMSE | 24.54 (MCST-Mamba) | **23.59** | ↓**4.3%** |
|           | MAPE | 9.68 (ours) | **9.68** | SOTA |

> ✅ **FAST 在 9 项 metric-dataset 组合中，获得 6 项第一、2 项第二，整体表现最优。**

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）Multi-Source Spatiotemporal Embedding 的作用（Table II）

移除该嵌入导致性能严重下降：

| 模型 | PeMS08 MAE | RMSE |
|------|------------|--------|
| w/o Embedding | 19.98 | 30.76 |
| FAST (full) | **14.66** | **23.59** |
| ➜ 下降幅度 | ↑**+5.32 (+36.3%)** | ↑**+7.17 (+30.4%)** |

👉 表明融合多源信息对建模异构交通上下文至关重要。

---

#### （2）TST 结构设计的有效性（Fig. 3）

比较四种变体：
- **w/ Attention**：用 Attention 替代空间 Mamba 模块
- **w/ Mamba**：用 Mamba 替代时间 Attention 模块
- **Swapped Mamba-Attention**：交换模块顺序（先 Mamba 后 Attention）
- **FAST (proposed)**：原始 TST 设计

| 变体 | MAE | 训练时间 | GPU 内存 |
|------|-----|----------|----------|
| w/ Attention | 更高 | 显著增加（↑~2x） | ↑↑↑ |
| w/ Mamba | 升高 | 略低 | 略低 |
| Swapped | 升高 | 接近 | 接近 |
| **FAST** | **最低** | **适中** | **最小** |

👉 结果表明：
- 使用 Attention 做空间建模会带来巨大计算开销；
- 用 Mamba 建模时间模式会损害表达能力；
- 模块顺序不可随意调换；
- **只有“Attention for time + Mamba for space”的 TST 结构才能实现最佳权衡**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Attention 和 State-Space Models 可以协同互补**：
   - Attention 更适合精细的时间建模；
   - Mamba 更适合高效的空间传播；
   - 二者结合可在精度与效率之间取得理想平衡。

2. **TST 架构有效促进时空交互迭代增强**：
   - 时空模块交替执行，使得时间演化受空间影响，空间传播也基于最新时间表征。

3. **无需显式图结构也能学得有效空间依赖**：
   - FAST 不依赖预设邻接矩阵，而是通过 Mamba 动态选择性地传播信息，适应功能相关的远程节点。

4. **多源嵌入与层次化预测头显著提升鲁棒性**：
   - 融合时间、空间、节点特性信息，增强了对复杂城市交通模式的建模能力；
   - Skip-connected prediction 改善了多步预测的稳定性。

---

### ⚠️ 局限性

1. **未明确建模道路拓扑约束**：
   - 虽然不依赖固定图结构是优点，但也可能忽略真实的物理连通性限制。

2. **Mamba 参数量仍较高**：
   - 尽管复杂度线性增长，但 Mamba 模块本身参数较多，在边缘设备部署仍有挑战。

3. **仅考虑单变量预测（单车道流量）**：
   - 未扩展至多变量场景（如速度、占有率等联合预测）。

---

### 🔮 未来工作方向

1. **拓展至 Multivariate Traffic Forecasting**
   - 联合建模速度、密度、事件等多种信号。

2. **自适应空间结构学习**
   - 引入动态图生成机制，结合 Mamba 的选择性传播。

3. **多任务 Urban Prediction**
   - 将 FAST 扩展为统一的城市感知框架，支持拥堵预警、事故检测、出行需求预测等任务。

4. **轻量化与边缘部署优化**
   - 探索模型压缩、量化、蒸馏等技术，推动实际交通系统落地应用。

---

## ✅ 总结

FAST 是一种新颖且高效的 **spatiotemporal traffic forecasting** 框架，它通过 **Temporal-Spatial-Temporal (TST)** 架构巧妙融合了 **Attention 的表达力** 与 **Mamba 的效率优势**，解决了传统方法在**时间建模、空间建模与计算效率之间的根本性权衡问题**。实验证明其在多个真实数据集上达到 SOTA 性能，尤其在 RMSE 和 MAE 上相比最强基线分别降低最多达 **4.3%** 和 **2.8%**，展现出卓越的准确性、可扩展性与泛化能力，为智能交通系统的实时预测提供了强有力的技术支撑。

</details>

---

### 5. [SparseBalance: Load-Balanced Long Context Training with Dynamic Sparse Attention](https://arxiv.org/abs/2604.13847)

**Authors**: Hongtao Xu, Jianchao Tan, Yuxuan Hu, Pengju Lu, Hongyu Wang, Pingwei Sun, Yerui Sun, Yuchen Xie, Xunliang Cai, Mingzhen Li, Weile Jia  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.13847v1  

#### Abstract
While sparse attention mitigates the computational bottleneck of long-context LLM training, its distributed training process exhibits extreme heterogeneity in both \textit{1)} sequence length and \textit{2)} sparsity sensitivity, leading to a severe imbalance problem and sub-optimal model accuracy. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*SparseBalance: Load-Balanced Long Context Training with Dynamic Sparse Attention*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在长上下文大语言模型（LLM）训练中，**稀疏注意力（sparse attention）** 虽然缓解了标准注意力机制带来的二次计算复杂度瓶颈，但在分布式训练过程中仍面临严重的**负载不平衡（load imbalance）** 问题。该问题源于两个维度的异构性：
- **序列长度异构性（sequence length heterogeneity）**：真实数据集中长短序列混合，导致不同 micro-batch 的计算负载差异巨大。
- **稀疏敏感性异构性（sparsity sensitivity heterogeneity）**：不同序列、不同 Transformer 层对稀疏程度的敏感度不同，固定稀疏度无法兼顾效率与精度。

现有方法通常只单独优化其中一个方面，未能实现算法与系统的协同优化，导致训练效率低下或模型精度下降。

### 提出的新方法与新思路
本文提出 **SparseBalance**，一种**算法-系统协同设计框架（algorithm-system co-design）**，通过动态调整稀疏度来联合优化训练效率与模型精度。其核心创新包括：

- **Workload-aware Dynamic Sparsity Tuning (DST)**  
  在运行时进行**双向稀疏度调整（bidirectional sparsity adjustment）**：
  - 对“瓶颈” micro-batch（执行时间长）**减少 attention budget（提高稀疏度）**，以加速执行、消除 straggler。
  - 对非瓶颈 micro-batch **增加 attention budget（降低稀疏度）**，利用 pipeline bubbles “免费”提升模型精度。
  - 引入 **anchor-guided thresholding** 机制，基于预测延迟设定调优目标，并通过路由 logits 控制调优幅度，保障模型质量。

- **Sparsity-Aware Batching (SAB)**  
  设计面向稀疏训练的批处理策略：
  - 利用轻量级**稀疏度估计器**和**基于延迟的打包策略**，在数据加载阶段实现粗粒度负载均衡。
  - 与 DST 形成“粗调 + 细调”的两级优化流水线。

- **Profiling-based Latency Predictor**  
  构建一个离线性能分析驱动的延迟预测模块，将序列长度与稀疏度映射为实际执行延迟，为 DST 和 SAB 提供精准性能指导。

### 相比现有方法的优势
| 方面 | 现有方法 | SparseBalance |
|------|--------|-------------|
| **优化维度** | 单一（仅序列长度或仅稀疏度） | 双重异构性联合优化 |
| **负载均衡粒度** | 静态批处理（coarse-grained） | 动态层级别调优（fine-grained） |
| **系统感知** | 缺乏运行时反馈 | 基于延迟预测的闭环控制 |
| **精度影响** | 固定稀疏度可能损失关键信息 | 动态补偿非瓶颈任务，维持甚至提升精度 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ChatQA2-Long-SFT**：开源长上下文 SFT 数据集，呈现双峰分布（大量 <4K 和 >16K 序列）。
- **LongAlign-10k**：另一个真实长文本对齐数据集，具有明显的长尾分布（8K–72K tokens）。

### 实验设置
- **硬件环境**：
  - 两组集群：H200 和 H20 GPU 集群，每节点 8×GPU，NVLink + InfiniBand 互联。
- **模型**：
  - Qwen2.5-0.5B 和 Qwen2.5-3B。
  - 使用 **MoBA** 作为基础稀疏注意力算法。
- **并行策略**：
  - 采用 4D 并行：DP=4, PP=4, TP=2, SP=2。
  - 微批次大小（micro-batch size）= 1，全局批次大小 = 16。
- **训练配置**：
  - 3 轮 SFT 训练，学习率 1e-6，warm-up 20%。
  - 使用 LoRA 进行参数高效微调（rank=32, alpha=64）。

### 评估指标
- **系统效率**：
  - 端到端训练速度（speedup ×）
  - 迭代时间（iteration time）
  - 负载不平衡度（Imbalance = max / mean micro-batch latency）
- **模型精度**：
  - 训练 loss 收敛情况
  - 下游任务表现：
    - **LongBench**：综合长上下文理解能力
    - **Needle-in-a-Haystack (NIAH)**：精确检索能力
    - **ARC, BoolQ, HellaSwag 等**：通用零样本推理能力

### 基线方法对比
- **Baseline**：原始 MoBA + 固定稀疏度 + 长度感知批处理（Length-Based Batching, LBB）。
- **消融变体**：
  - `+DST`：仅启用动态稀疏调优
  - `+SAB`：仅启用稀疏感知批处理
  - `+SAB+DST`：完整 SparseBalance
  - 不同 anchor 策略（MEAN, MIN, MAX）和阈值 $ p $（0.1, 0.2）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 & 数据集 | 端到端加速比（Speedup） | LongBench 准确率提升 |
|--------------|--------------------------|------------------------|
| Qwen2.5-3B + LongAlign-10k | **1.33×** | **+0.46%**（从 39.28 → 39.46） |
| Qwen2.5-3B + ChatQA2 | **1.30×** | 基本持平或轻微提升 |

> ✅ **核心结论：在显著提升训练效率的同时，未牺牲模型精度，反而略有增益。**

### 与基线方法的对比结果
- **效率方面**：
  - 相比 Baseline，SparseBalance 实现 **平均 1.30× 端到端加速**。
  - 在 H20 和 H200 上均表现出一致的加速效果，说明其硬件鲁棒性强。
- **精度方面**：
  - **LongBench 总体得分最高的是 MEAN0.1 配置（39.46）**，优于 Baseline（39.28）。
  - NIAH 任务在 64K 上保持 97.53% vs Baseline 97.60%，几乎无损。
  - 通用推理任务（如 BoolQ, Piqa）上也保持竞争力，部分指标反超。

### 消融实验结果
| 配置 | 加速比 | LongBench 分数 | 备注 |
|------|-------|------------------|------|
| Baseline | 1.00× | 39.28 | — |
| +DST | 1.08× | ≈39.2 | 单独 DST 提升有限 |
| +SAB | 1.21× | ≈39.2 | 初步平衡有效 |
| **+SAB+DST** | **1.35×** | **39.46** | 完整方案最优 |
| +LBB+DST | 1.28× | — | LBB 效果弱于 SAB |

> 🔍 发现：**SAB 与 DST 具有强协同效应**。SAB 提供良好的初始负载分布，使 DST 更高效地进行细粒度调优。

#### 超参数敏感性分析
- **阈值 $ p $**：
  - $ p $ 越大（允许更大稀疏度变化），加速越明显（$ p=0.3 $ → 1.45×），但精度下降。
  - $ p=0.1 $ 是效率与精度的最佳折衷点。
- **Anchor 策略**：
  - **MEAN anchor** 表现最佳，在效率和精度间取得最好平衡。
  - MIN anchor 易过度压缩瓶颈任务，损害精度；MAX anchor 几乎无加速。

---

## 4. 关键结论和发现

### 主要发现
1. **稀疏度可作为运行时负载调节自由度**：首次将稀疏注意力中的 `attention budget` 视为可动态调整的系统资源，用于解决分布式训练中的负载不均衡问题。
2. **双向调优实现“免费精度”**：通过在非关键路径上增加计算资源，可“吸收” pipeline bubbles，实现精度提升而不增加端到端延迟。
3. **算法-系统协同至关重要**：单纯改进稀疏算法或批处理策略无法根本解决问题，必须结合运行时反馈与性能建模。
4. **MEAN0.1 是最优实践配置**：在 trade-off 曲线上位于帕累托前沿最优点，同时提升效率与精度。

### 方法的局限性
- **依赖稀疏注意力 indexer**：需复用现有算法（如 MoBA、DSA）的 routing logits，对非 block-sparse 方法适配性待验证。
- **额外开销虽小但仍存在**：SAB 和 DST 引入约 1.5% 的额外开销（主要来自表查找与搜索）。
- **静态 profile 表**：延迟预测依赖离线 profiling，若硬件或并行策略变更需重新校准。

### 未来工作方向
- 扩展至更多类型的稀疏模式（如 sliding window, local-global）。
- 探索在线自适应 profiling，实现跨环境迁移。
- 将动态稀疏思想应用于推理阶段，进一步提升部署效率。
- 结合更复杂的调度策略（如弹性 batch size）与通信优化。

---

> 📌 **总结一句话**：  
> **SparseBalance 通过算法-系统协同设计，将稀疏注意力的“副作用”转化为负载均衡的“调控手段”，实现了“更快 + 更准”的长上下文训练新范式。**

</details>

---

### 6. [Hardware-Efficient Neuro-Symbolic Networks with the Exp-Minus-Log Operator](https://arxiv.org/abs/2604.13871)

**Authors**: Eymen Ipek  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.13871v1  

#### Abstract
Deep neural networks (DNNs) deliver state-of-the-art accuracy on regression and classification tasks, yet two structural deficits persistently obstruct their deployment in safety-critical, resource-constrained settings: (i) opacity of the learned function, which precludes formal verification, and (i...

---

### 7. [Automated co-design of high-performance thermodynamic cycles via graph-based hierarchical reinforcement learning](https://arxiv.org/abs/2604.13133)

**Authors**: Wenqing Li, Xu Feng, Peixue Jiang, Yinhai Zhu  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.13133v1  

#### Abstract
Thermodynamic cycles are pivotal in determining the efficacy of energy conversion systems. Traditional design methodologies, which rely on expert knowledge or exhaustive enumeration, are inefficient and lack scalability, thereby constraining the discovery of high-performance cycles. In this study, w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Automated co-design of high-performance thermodynamic cycles via graph-based hierarchical reinforcement learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统热力学循环设计依赖专家经验或穷举法，存在以下局限：
- **效率低下**：人工设计周期长，难以系统探索复杂拓扑空间；
- **可扩展性差**：固定结构优化范式（“fixed structure + parameter optimization”）限制了新型高性能循环的发现；
- **缺乏自主学习能力**：现有算法方法（如HEN、superstructure、graph theory）虽部分实现自动化，但计算效率低、智能程度不足。

该研究旨在突破这些瓶颈，实现**全自动化的热力学循环结构与参数协同优化（co-design）**。

---

### ✨ 提出的新方法与创新思路
提出了一种**基于图表示的分层强化学习框架（graph-based hierarchical reinforcement learning, HRL）**，用于热力学循环的结构-参数联合优化。其核心创新包括：

#### （1）**图编码（Graph-based Encoding）**
- 将热力学循环抽象为有向图（Directed Graph）：
  - 节点（Nodes）代表组件（如 Compressor、Turbine、IHX、Ejector 等）；
  - 边（Edges）表示工质流动路径；
- 引入语法约束规则（grammatical constraints），确保生成的图在物理上可行（如压力平衡、能量守恒等）。

#### （2）**物理感知解码（Physics-informed Graph Decoding）**
- 构建一个基于深度学习的 **thermophysical surrogate model（MLP）**，替代传统的NIST REFPROP/CoolProp求解器；
- 实现稳定、快速、可微分的状态点求解，支持端到端训练；
- 支持同时求解所有节点状态（simultaneous solution），避免手动设定求解顺序。

#### （3）**Manager-Worker 分层强化学习架构**
- **高层 Manager Agent**：负责探索结构演化，通过激活边（edge activation）逐步构建图结构；
- **底层 Worker Agent**：对给定结构进行连续参数优化（如压比、分流比等），返回性能奖励（如 COP 或 η）；
- 形成“结构探索 → 参数优化 → 性能反馈”的闭环搜索机制。

#### （4）**联合训练策略提升稳定性**
- 引入三项关键技术缓解稀疏奖励与收敛不稳定问题：
  1. **Performance Feedback Backpropagation**：将最终性能反向传播至每一步决策；
  2. **Elite Cycle Memory**：保留高绩效循环轨迹，引导后续搜索；
  3. **Staged Training**：前期鼓励探索（高熵权重），后期聚焦优化（加强精英记忆）。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 设计范式 | 固定结构 + 参数调优 | 结构-参数协同自动设计 |
| 探索能力 | 受限于专家假设 | 自主发现新拓扑 |
| 智能水平 | 规则驱动 / 数值优化 | 学习驱动 / 自适应进化 |
| 可扩展性 | 难以处理复杂组合 | 图表示天然支持扩展 |
| 计算效率 | 单次仿真耗时高 | Surrogate加速，支持大规模采样 |

> ✅ **优势总结**：首次实现了从“人为设计”到“机器自主创造”的跃迁，兼具高效性、通用性和高性能潜力。

---

## 2. 核心实验方法和设置

### 📚 数据集与仿真环境
- **无外部数据集**：采用**第一性原理建模 + 内部生成数据**；
- 工质模型基于 **CO₂**（跨临界热泵）和 **sCO₂**（超临界布雷顿循环）；
- 使用开源库 **CoolProp** 生成训练数据，训练 **MLP surrogate** 模型预测 T-S-Q、H-P-S 等物性；
- 所有循环结构由Agent在符合物理规则的前提下自动生成。

---

### ⚙️ 实验设置
| 模块 | 设置说明 |
|------|----------|
| **Graph Rules** | 定义合法连接规则（如每个环路必须含增压/减压元件）、组件数量上限（如仅允许1个压缩机） |
| **State Representation** | 当前邻接矩阵 $ A \in \{0,1\}^{N\times N} $ |
| **Action Space** | Manager 动作：选择一对 (i,j)，激活边 $ a_{ij}=1 $ |
| **Reward Signal** | Worker 返回的系统性能指标：<br>- 热泵：COP（Coefficient of Performance）<br>- 热机：Thermal Efficiency $ \eta $ |
| **Optimization Objective** | 最大化 COP 或 $ \eta $ |
| **Training Episodes** | 5,000 轮迭代 |
| **Baseline** | Random Search（随机搜索）作为对照 |

---

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **有效性** | Valid Cycle Generation Rate（有效循环占比） |
| **多样性** | 发现的新结构数量（novel configurations） |
| **性能优越性** | 最佳新结构 vs. 基准循环的性能提升百分比 |
| **收敛性** | 平均性能随训练轮次的变化趋势 |

---

### 🆚 基线方法对比
- **Random Search**：完全随机生成邻接矩阵并验证合法性；
- **Expert-designed Cycles**：经典循环配置（如基本跨临界CO₂热泵、再生式布雷顿循环）作为性能基准；
- **枚举法（implicit comparison）**：文中指出传统方法无法覆盖如此庞大的搜索空间。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**有效循环生成率显著高于随机搜索**
| 方法 | 热泵循环有效率 | 热机循环有效率 |
|------|----------------|----------------|
| HRL Agent | **93.50%** | **82.82%** |
| Random Search | ~0.02% | <0.06% |

> ➤ 表明 HRL 能高效聚焦于可行设计区域，避免无效探索。

---

#### （2）**发现大量新颖高性能循环结构**

| 系统类型 | 总有效循环数 | 已知循环数 | 新发现循环数 |
|----------|---------------|-------------|----------------|
| Heat Pump | 22 | 4（cycles 1,2,3,5） | **18** |
| Heat Engine | 26 | 5（cycles 1,2,3,8,13） | **21** |

> ➤ 显示模型具备强大创新能力，远超人类经验范围。

---

#### （3）**性能超越经典循环**

| 场景 | 最佳新结构 | 性能提升 |
|------|------------|-----------|
| **Heat Pump**（Case 1） | Cycle No. 3,5,6,7 | **COP 提升 4.6%**（vs. 基础 cycle 1） |
| **Heat Engine**（Case 1） | Cycle No. 13 | **效率提升 133.3%**（η 从 0.177 → 0.413） |

> ➤ 特别是在热机系统中实现翻倍以上效率增长，极具工程价值。

---

#### （4）**消融实验与训练分析（隐含验证）**
虽然未明确列出消融表格，但从方法描述中可推断关键模块作用：
- **Without Elite Memory**：易遗忘高绩效结构，收敛慢；
- **Without Performance Backpropagation**：信用分配不均，难以学习长期策略；
- **Without Staged Training**：早期陷入局部最优或过度探索导致不收敛。

> ➤ 所提联合训练策略对稳定性和性能至关重要。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **HRL 可成功应用于热力学系统设计**：
   - 实现了从“结构生成 → 参数优化 → 性能反馈”的全自动化流程；
   - 成功复现经典循环，并发现了数十种前所未见的高性能新结构。

2. **内部能量回收是提升性能的关键机制**：
   - 多数高性能循环都包含 **IHX（Internal Heat Exchanger）** 或类似回热结构；
   - 表明 Agent 能自主“学习”到经典热力学原则（如预热降低压缩功）。

3. **结构偏好随运行条件动态调整**：
   - 在不同加热温度下，最优结构排序发生变化（尤其在热泵中）；
   - 显示框架具有良好的**环境适应性**。

4. **热机结构更稳健，性能排序稳定**：
   - 不同热源温度下最优结构保持一致；
   - 反映其能量转换机制更具普适性。

---

### ⚠️ 局限性
1. **图规则仍部分依赖专家定义**：
   - 当前 grammar rules（如连接有效性、压力约束）需人工设定；
   - 限制了向更广泛系统的直接迁移能力。

2. **单目标优化**：
   - 仅以 COP 或 效率 为目标，未考虑设备成本、复杂度、可靠性等实际工程因素。

3. **应用范围有限**：
   - 目前仅验证于热泵与热机系统；
   - 尚未拓展至联合循环（combined cycle）、多能互补系统（integrated energy systems）等更复杂场景。

4. **Surrogate 模型泛化边界待明确**：
   - MLP 替代物性求解器虽快且稳，但在极端工况下的误差需进一步控制。

---

### 🔮 未来工作方向
1. **自动化图规则提取**：
   - 利用无监督学习或元学习让 Agent 自主归纳有效连接模式，减少专家干预。

2. **多目标优化扩展**：
   - 引入 Pareto front 优化，兼顾效率、成本、紧凑性等多重指标。

3. **向复杂系统延伸**：
   - 应用于 **combined cycle power plant**、**integrated energy systems**（IES）、**power-to-X** 等综合能源系统设计。

4. **实验验证与硬件部署**：
   - 将发现的新结构投入原型测试，形成“AI设计 → 实验验证 → 反馈优化”闭环。

5. **结合物理约束的更强正则化**：
   - 进一步融合 thermodynamic laws into RL reward shaping or policy constraints。

---

## ✅ 总结
本论文提出了一种革命性的 **graph-based hierarchical RL 框架**，实现了热力学循环的**全自动协同设计**。实验表明，该方法不仅能重现经典结构，还能发现大量性能显著提升的新颖循环，在热泵和热机系统中分别实现 **4.6% 和 133.3% 的性能增益**。它标志着热力学系统设计正从“经验主导”迈向“AI原生创新”的新时代，为未来智能能源系统的设计提供了强有力的工具和范式。

</details>

---

### 8. [RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair](https://arxiv.org/abs/2604.12820)

**Authors**: Jagadeesh Rachapudi, Pranav Singh, Ritali Vatsi, Praful Hambarde, Amit Shukla  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.12820v1  

#### Abstract
Large language models (LLMs) inherently absorb harmful knowledge, misinformation, and personal data during pretraining on large-scale web corpora, with no native mechanism for selective removal. While machine unlearning offers a principled solution, existing approaches are provider-centric, requirin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Machine Unlearning (MU)** 方法存在以下核心缺陷：
- **Provider-centric**：依赖模型服务提供商（MSP）进行干预，普通用户无法自主控制自己的数据是否被遗忘。
- **训练依赖**：大多数方法需要重新训练或微调，计算成本高，难以在推理时（test time）执行。
- **批量处理**：通常要求批量的遗忘样本（forget samples）和保留样本（retain samples），不支持单样本即时遗忘。

这导致用户在发现模型泄露其个人隐私或错误知识后，缺乏直接、透明、高效的“数字遗忘权”行使机制。

### 🚀 提出的新方法与新思路
本文提出 **Interactive Machine Unlearning (IMU)** ——一种全新的机器遗忘范式，允许**终端用户通过自然语言指令**，在推理阶段直接让 LLM “忘记”特定知识。

为实现 IMU，作者设计了 **RePAIR** 框架，包含三个核心组件：
1. **Mwatchdog**：从对话历史中检测用户的遗忘意图，并提取需遗忘的 `(pf, rf)` 对。
2. **Msurgeon**：生成可执行的修复代码（repair code），指导如何修改模型参数。
3. **Mpatient**：原始模型，在运行时被自动更新为 `Mhealed`，实现知识移除。

核心技术是 **STAMP**（Steering Through Activation Manipulation with PseudoInverse）：
- 一种**无需训练**（training-free）、**单样本**（single-sample）的遗忘方法。
- 利用伪逆（pseudoinverse）对 MLP 层的激活进行闭式更新，将“遗忘样本”的激活导向“拒绝子空间”（refusal subspace）。
- 推出低秩变体 **STAMP-LR**，显著降低计算复杂度（从 $O(d^3)$ 降至 $O(r^3 + r^2 \cdot d)$），支持设备端高效执行。

### ⭐ 相比现有方法的优势
| 维度 | 现有方法（如 GA, NPO, FLAT 等） | RePAIR / STAMP |
|------|-------------------------------|----------------|
| 用户参与 | ❌ 完全由 MSP 控制 | ✅ 支持用户交互式请求 |
| 是否训练 | ✅ 需要梯度回传与优化 | ❌ 完全无训练（training-free） |
| 样本粒度 | ❌ 批量遗忘为主 | ✅ 单样本即可有效遗忘 |
| 推理效率 | ❌ 耗时长（~10–12 分钟） | ✅ 快速响应（最快 ~2.57 分钟） |
| 实现场景 | ❌ 云端集中处理 | ✅ 可部署于边缘设备 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验覆盖三大典型遗忘任务：
1. **有害知识抑制**（Harmful Knowledge Suppression）
   - 数据集：**WMDP-Bio**（1K 样本）
2. **虚假信息纠正**（Misinformation Removal）
   - 数据集：**MMLU**（1K 错误标注样本）
3. **个人数据擦除**（Personal Data Erasure）
   - 数据集：基于 **Mistral-7B API** 生成的 2K 合成人物档案

所有数据集均划分为等量的 `Df`（forget set）和 `Dr`（retain buffer ≤10% of total），并引入一个包含 200 条“我不知道”类提示的 `Dref` 作为拒绝参考集。

### 🧪 实验设置与评估指标
#### 模型配置
- **Mpatient**: Llama-3-8B
- **Mwatchdog**: Mistral-7B（用于意图识别与配对抽取）
- **Msurgeon**: Qwen2.5-Coder-7B-Instruct（生成修复代码）

#### 评估指标
| 类别 | 指标 | 说明 |
|------|------|------|
| 遗忘效果 | `Accf ↓`, `F-RL ↓` | 越低越好，表示目标知识已被成功遗忘 |
| 保留能力 | `Accr ↑`, `R-RL ↑` | 越高越好，表示无关知识未受损 |
| 模型效用 | Perplexity on **TinyStories** | 衡量整体语言建模能力保持情况 |
| 运行效率 | **RTE (Runtime Efficiency)** | 单位：分钟，越小越好 |

#### 基线方法对比
共比较六种 SoTA 方法：
- Gradient Ascent (GA)
- Negative Preference Optimization (NPO)
- Representation Misdirection for Unlearning (RMU)
- Forget-data-only Loss Adjustment (FLAT)
- Weighted Gradient Ascent (WGA)
- Attention Smoothing Unlearning (ASU)

同时引入 **Oracle 模型**（仅在完整保留集上训练）作为理论上限。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 2）

| 方法 | Harmful Accf | Harmful Accr | Misinfo Accf | Misinfo Accr | Personal F-RL | Personal R-RL | RTE (min) |
|------|--------------|--------------|---------------|---------------|----------------|----------------|------------|
| **Base** | 75.30 | 78.50 | 83.70 | 86.30 | 0.87 | 0.89 | N/A |
| **Oracle** | – | 77.37 | – | 85.30 | – | 0.90 | – |
| **GA** | 0.00 | 73.27 | 0.00 | 83.21 | 0.13 | 0.81 | 10.41 |
| **STAMP** | **0.00** | 70.13 | **0.00** | 80.13 | **0.00** | 0.79 | 6.48 |
| **STAMP-LR** | **0.00** | 73.27 | **0.00** | **84.47** | **0.00** | **0.88** | **4.01** |

> ✅ **关键亮点**：
> - 在所有任务中实现 **Accf ≈ 0.00**, **F-RL = 0.00**，达到近乎完美的遗忘效果。
> - **STAMP-LR** 在误导信息修正任务中取得 **84.47 Accr**，接近 Oracle（85.30），远超其他方法。
> - **RTE 最快达 4.01 分钟**，相比训练型基线（平均 ~10 min）提速约 **3×**。

### 🔍 消融实验结果

#### （1）单层 vs 全层干预（Table 5）
| 设置 | F-RL | R-RL | Utility | RTE(s) |
|------|-----|------|--------|--------|
| Layer 7 only | 0.00 | 0.85 | 6.07 | **4.36** |
| All layers | 0.00 | 0.88 | 6.02 | 15.40 |

> 💡 发现：仅干预 **第7层 MLP** 即可获得接近全层更新的效果，且速度提升 **~3.8×**。原因在于该层在 WMDP 和拒绝激活之间具有最大余弦分离度（0.867，见 Figure 5）。

#### （2）低秩分解中的秩 `r` 影响（Table 6）
| Rank(r) | R-RL | RTE(mins) |
|--------|------|-----------|
| 64     | 0.85 | 4.01      |
| 128    | 0.88 | 5.24      |

> 💡 结论：当 `r ≥ 64` 时性能稳定；低于此值则保留能力下降，表明足够秩对结构捕捉至关重要。

#### （3）保留缓冲区大小影响（Table 7）
即使将 `Dr` 减少到总保留集的 **10%**，STAMP-LR 仍能维持良好性能（R-RL=0.88），适合资源受限场景。

#### （4）单样本遗忘能力测试（Table 8）
| Method | Accf (|Df|=1) |
|--------|----------------|
| GA, NPO, RMU 等 | **100**（完全失败） |
| **STAMP-LR** | **0.00** |

> ✅ 唯一能在单样本设定下成功遗忘的方法，验证其真正满足 IMU 场景需求。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **IMU 是可行且必要的新范式**：用户应能通过自然语言指令直接控制模型遗忘行为，无需依赖 MSP。
2. **STAMP 实现了真正的 test-time unlearning**：
   - 首个支持 **training-free + single-sample** 的 LLM 遗忘方法。
   - 通过激活操纵而非参数微调实现高效知识编辑。
3. **RePAIR 框架端到端可用性强**：
   - Mwatchdog 成功检测 >96% 的遗忘请求。
   - Msurgeon 生成的有效代码率高达 97%，支持自动化修复。
4. **性能全面超越 SoTA**：
   - 遗忘质量接近 Oracle。
   - 效用损失极小，运行速度快 **~3×**。

### ⚠️ 方法的局限性
1. **仍需少量 retain buffer (`Dr`)**：
   - 尽管只需 10%，但在严格合规环境下存储任何旧数据仍可能违反 GDPR / CCPA。
2. **测试时资源消耗仍较高**：
   - 虽优于训练方法，但 STAMP-LR 仍需一定内存与算力，尚未完全适配移动端。
3. **目前仅适用于文本模态**：
   - 多模态扩展尚未探索。

### 🔮 未来工作方向
1. 开发 **fully retain-free unlearning** 方法（如结合 FLAT 思路）。
2. 将 RePAIR 扩展至 **multimodal foundation models**（如视觉-语言模型）。
3. 探索更轻量化的 **on-device implementation**，推动隐私保护落地。
4. 引入 **formal verification** 机制，确保遗忘操作的可审计性与可信度。

---

> 📌 **总结一句话**：  
> RePAIR 首次实现了**用户驱动、无需训练、实时高效的交互式机器遗忘**，为 LLM 的隐私安全与可控性提供了实用化路径，是迈向“负责任 AI”的重要一步。

</details>

---

### 9. [BioTrain: Sub-MB, Sub-50mW On-Device Fine-Tuning for Edge-AI on Biosignals](https://arxiv.org/abs/2604.13359)

**Authors**: Run Wang, Victor J. B. Jung, Philip Wiese, Sebastian Frey, Giusy Spacone, Francesco Conti, Alessio Burrello, Luca Benin  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.13359v1  

#### Abstract
Biosignals exhibit substantial cross-subject and cross-session variability, inducing severe domain shifts that degrade post-deployment performance for small, edge-oriented AI models. On-device adaptation is therefore essential to both preserve user privacy and ensure system reliability. However, exi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《BioTrain: Sub-MB, Sub-50mW On-Device Fine-Tuning for Edge-AI on Biosignals》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
生物信号（如 EEG 和 EOG）在跨被试（cross-subject）和跨会话（cross-session）场景下表现出显著的变异性，导致模型部署后因**域偏移**（domain shift）而性能严重下降。传统的边缘设备受限于内存、计算和功耗资源，难以支持完整的 **Backpropagation (BP)** 进行全网络微调，通常只能采用浅层更新（如 Last-Layer Training / Linear Probing, LP）或稀疏更新策略，这些方法适应能力有限。

此外，现有 on-device training 框架缺乏对完整训练流程的编译器级优化支持，尤其在内存管理方面依赖手动调整，限制了可扩展性和实用性。

### 提出的新方法与创新思路
本文提出 **BioTrain**，一个面向资源受限 MCU 平台的端到端 on-device fine-tuning 编译框架，其核心创新包括：

- **支持全网络 BP 的自动化代码生成**：基于 Deeploy 编译器架构，首次实现从 PyTorch 模型到可在 MCU 上运行的 bare-metal C 代码的全流程训练代码生成，涵盖前向传播、反向传播及参数更新。
- **结合 Gradient Accumulation 与 Group Normalization (GN)**：为解决小批量训练下的内存瓶颈和 BN 层的跨样本依赖问题，用 GN 替代 BN，并引入梯度累积机制，在保持大有效 batch size 优化优势的同时大幅降低峰值内存占用。
- **编译器驱动的内存优化机制**：
  - 扩展 Deeploy 的前端、中端和后端以支持训练图（包括 autograd 子图）；
  - 引入针对 CNN 梯度算子的 **tiling 策略** 和静态内存分配；
  - 利用 OR-Tools 解决带约束的内存调度问题，最大化片上内存复用，最小化片外传输。

### 相比现有方法的优势
| 方面 | BioTrain | 现有方法（如 TinyOL, MiniLearn, TTE, AIfES） |
|------|--------|------------------------------------------|
| **更新深度** | 支持 Full-network Fine-Tuning | 多为 Last-Layer 或 Sparse BP |
| **Batch Size > 1** | ✅ 支持（通过梯度累积 + GN） | ❌ 多数仅支持 batch size = 1 |
| **内存效率** | 峰值内存降低 8.1×（5.4MB → 0.67MB） | 高内存占用，无法片上执行 |
| **自动化程度** | 完整编译器支持，自动内存调度 | 依赖手工优化，扩展性差 |
| **适用性** | 可推广至多种 biosignal 模型架构 | 多限于简单 MNIST/Iris 类任务 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **EEG Dataset** [4]：
  - 来源：5 名被试，每人 4 个会话
  - 通道数：8 个 EEG 通道
  - 采样率：500 Hz
  - 任务：二分类（舌头运动 vs. 休息），每段 trial 3.8 秒（T=1900）
  - 模型：MI-BMINet（轻量级 CNN，7.9k 参数）

- **EOG Dataset** [7]（来自 GAPses 智能眼镜平台）：
  - 来源：5 名被试，每人 2 个会话
  - 电极配置：3 个干电极（左/右鼻托 + 鼻梁），推导出水平（VH）和垂直（VV）EOG 信号
  - 采样率：500 Hz
  - 任务：11 类眼动识别，每段 trial 2.0 秒（T=1000）
  - 模型：EpiDeNet（紧凑 CNN，4.1k 参数）

### 实验设置与评估指标
#### 硬件平台
- **GAP9 MCU**：
  - 9 核 RISC-V 可编程集群
  - 片上存储：128kB L1 SRAM（用户管理），1.5MB L2 SRAM
  - DMA 支持计算与内存传输重叠
  - 功耗测量工具：Nordic Power Profiler Kit II

#### 评估场景
1. **Scenario A: Day-1 Calibration（冷启动校准）**
   - 使用 LOSO（Leave-One-Subject-Out）预训练全局模型
   - 对新用户的第一会话数据进行微调（80% 训练，20% 测试）
   - 模拟初始个性化过程

2. **Scenario B: Longitudinal Adaptation（纵向自适应）**
   - 以 S1 微调后的模型为起点
   - 在后续会话（S2~S4）中持续增量学习
   - 每次使用当前会话前 80% 数据微调，后 20% 测试
   - 模拟长期生理漂移跟踪

#### 基线方法对比
| 方法 | 描述 |
|------|------|
| **No-FT** | 不微调，直接推理（zero-shot） |
| **LP (Linear Probing)** | 仅微调最后一层分类头 |
| **Full-FT** | 全网络微调（原始结构含 BN） |
| **Edge-FT (BioTrain)** | 本文方法：GN 替代 BN + 梯度累积 + 编译优化 |

#### 超参数设置
- **云预训练**：AdamW，batch size=64，lr=1e-3，40 epochs
- **设备端微调**：SGD + momentum (0.9)，cosine annealing，batch size=8（有效），lr=5e-3，weight decay=1e-3，30 epochs
- 所有实验重复 5 次取均值 ± 标准差

---

## 3. 主要实验结果和性能指标

### 准确率提升（Table III & Figure 4）
| 方法 | EEG (Subj.) | EEG (Sess.) | EOG (Subj.) | EOG (Sess.) |
|------|-------------|-------------|-------------|-------------|
| No-FT | 50.8±8.8% | 49.8±3.8% | 78.1±18.1% | 84.1±11.2% |
| LP | 74.8±11.4% | 76.2±7.6% | 83.3±13.0% | 86.5±10.7% |
| Full-FT | 80.0±11.4% | 78.7±7.3% | 88.7±10.3% | 89.1±8.8% |
| **Edge-FT (BioTrain)** | **86.4±7.9%** | **83.9±13.3%** | **87.7±10.4%** | **87.2±7.6%** |

- **相比 No-FT 最高提升达 35%**
- **相比 LP 提升约 7%（EEG 场景）**
- **Edge-FT 在 EEG 上优于 Full-FT（+6.4%）且方差更小**

> ⚠️ 注意：Full-FT 因 BN 导致跨样本同步，无法与梯度累积兼容，实际无法在设备上运行（需 >5.4MB 内存）

### 性能、内存与能耗分析（Table IV）
| 指标 | 方法 | EEG 结果 | EOG 结果 |
|------|------|---------|---------|
| **Peak L2 Memory** | Edge-FT | **0.67 MB** | **0.28 MB** |
| | Full-FT (理论) | ~5.4 MB (> GAP9 L2 容量) | — |
| **Training Throughput** | Edge-FT | 17 samples/s | 85 samples/s |
| **Power Consumption** | Edge-FT | < **50 mW** | < **50 mW** |
| **Energy per Session** (40 epochs, 200 samples) | Edge-FT | 20.16 mJ | 4.48 mJ |
| **Battery Life Support** (320mAh, 3.7V) | Edge-FT | **~211 次 EEG 训练** | **~951 次 EOG 训练** |
| **Compute Efficiency** | Edge-FT | 0.89 GFLOPs/s | 0.22 GFLOPs/s |

- **内存减少 8.1×**：从 5.4MB → 0.67MB（EEG）
- **完全片上执行**：得益于 tiling 和 GN 设计，避免频繁片外访问
- **低功耗可持续训练**：单节电池支持数百次个性化训练会话

### 消融实验与关键发现（隐含于设计选择）
- **GN vs. BN**：实验证明 GN 在小 batch 下稳定性更好，适合边缘训练；且无跨样本统计依赖，兼容梯度累积。
- **梯度累积的作用**：允许使用大有效 batch size（如 8），提升收敛速度与最终精度，同时将峰值内存控制在 sub-MB 级别。
- **编译器优化有效性**：通过 tiling 和静态内存分配，实现了高效的数据重用与 DMA-Compute Pipeline，显著提升吞吐量。

---

## 4. 关键结论和发现

### 主要发现
1. **全网络 fine-tuning 显著优于 last-layer 更新**：特别是在跨被试和长期漂移场景下，LP 方法存在明显性能天花板。
2. **BioTrain 实现了真正的 on-device full BP**：在 sub-MB 内存和 sub-50mW 功耗下完成端到端训练，是目前唯一支持此能力的自动化框架。
3. **GN + 梯度累积是边缘训练的关键组合**：解决了小批量与内存限制之间的矛盾，使高性能训练成为可能。
4. **编译器驱动的系统优化至关重要**：仅靠算法改进不足以突破硬件瓶颈，必须结合底层 tiling、内存调度和 kernel 优化。

### 方法的局限性
- 当前仅支持 FP32 精度训练，未涉及量化训练（Quantized Training），仍有进一步压缩空间。
- 实验集中在 EEG/EOG 两类 biosignal，尚未验证在 EMG、ECG 等其他模态上的泛化能力。
- 仍需要短期监督标签用于校准（如临床协议或参考设备），不支持完全无监督 adaptation。

### 未来工作方向
- 扩展至 **quantized training**（如 INT8/INT4），进一步降低计算与内存开销。
- 支持更多 biosignal 模态（如 EMG、PPG、ECG）和模型架构（如 Transformers）。
- 探索 **test-time adaptation (TTA)** 与 BioTrain 的融合，减少对标注数据的依赖。
- 开源发布：项目已开源至 GitHub（https://github.com/pulp-platform/Deeploy），鼓励社区共建。

--- 

> ✅ **一句话总结**：  
> BioTrain 是首个在 sub-MB 内存和 sub-50mW 功耗下实现全网络 on-device fine-tuning 的编译框架，通过 GN + 梯度累积 + 编译器级内存优化，在 EEG/EOG 任务上实现了高达 35% 的准确率提升，推动了 TinyML 向“持续学习 + 个性化”方向迈进。

</details>

---

### 10. [Outperforming Self-Attention Mechanisms in Solar Irradiance Forecasting via Physics-Guided Neural Networks](https://arxiv.org/abs/2604.13455)

**Authors**: Mohammed Ezzaldin Babiker Abdullah, Rufaidah Abdallah Ibrahim Mohammed  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.13455v1  

#### Abstract
Accurate Global Horizontal Irradiance (GHI) forecasting is critical for grid stability, particularly in arid regions characterized by rapid aerosol fluctuations. While recent trends favor computationally expensive Transformer-based architectures, this paper challenges the prevailing "complexity-firs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Outperforming Self-Attention Mechanisms in Solar Irradiance Forecasting via Physics-Guided Neural Networks*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **高噪声气象环境下的GHI预测难题**：在撒哈拉以南等干旱半干旱地区，气溶胶浓度高、沙尘暴频繁，导致全球水平辐照度（GHI）剧烈波动，传统纯数据驱动模型难以准确捕捉其动态变化。
- **Transformer类模型的“复杂性陷阱”**：当前研究普遍追求更复杂的架构（如Self-Attention、Transformers），但这些模型在连续时间序列任务中易出现相位滞后（Phase Lag）、过平滑等问题，且计算成本高昂。

### 提出的新方法与思路
- **提出一种轻量级物理引导混合深度学习框架（Physics-Informed Hybrid CNN-BiLSTM）**：
  - 结合 **CNN** 提取局部空间特征（如云层瞬变）；
  - 利用 **BiLSTM** 捕捉长期双向时序依赖；
  - 引入 **15个工程化物理特征** 显式注入先验知识，而非依赖模型从原始数据中隐式学习。

- **核心思想：显式物理约束 > 隐式注意力机制**
  - 将 Clear-Sky GHI、Solar Zenith Angle、Clearness Index（KT）、Volatility Index 等作为输入特征，使模型具备对太阳几何和大气透射率的基本理解。
  - 这些物理特征充当“软约束”，防止模型输出违反自然规律（如夜间正辐射、超过Clear-Sky上限）。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **准确性** | 在高噪声环境下显著优于Transformer及标准LSTM/CNN模型 |
| **鲁棒性** | 对未见天气条件（如突发沙尘暴）泛化能力强 |
| **效率** | 参数量仅约492,200，远低于典型Transformer模型（常超百万） |
| **可解释性** | 物理特征提供明确因果路径，减少黑箱行为 |

---

## 2. 核心实验方法和设置

### 数据集
- **来源**：NASA POWER 数据库（基于卫星观测与再分析数据）
- **地点**：苏丹乌姆杜尔曼（Omdurman, 14.7°N, 33.2°E），典型的干旱气候区
- **时间范围**：
  - **训练/验证/内部测试**：2010–2015年（按70%/15%/15%划分）
  - **外部压力测试**：2020–2024年（独立五年数据，用于检验跨时段泛化能力）
- **变量**：共15维输入特征，包括：
  - 辐射项：GHI, DNI, DHI
  - 气象项：Tamb, RH, Tdew, Twet
  - 物理导出项：Clear-Sky GHI, KTcalc, KTsat, Volatility Index
  - 时间编码：Sin/Cos(Hour), Sin/Cos(Day)

### 实验设置
- **任务类型**：Sequence-to-One 回归预测（使用过去24小时数据预测下一小时GHI）
- **预处理流程**：
  - 夜间掩码（Night Masking）：当理论Clear-Sky GHI < 0时强制设为0
  - 混合归一化策略：Z-Score标准化 + Min-Max缩放目标变量
  - 循环时间编码（Cyclical Time Encoding）避免端点不连续
- **优化方法**：**Bayesian Optimization** 自动调参，搜索空间包括学习率、Dropout率、BiLSTM单元数、L2正则系数等

### 评估指标
| 指标 | 公式 | 作用 |
|------|------|------|
| **RMSE** | $\sqrt{\frac{1}{N}\sum(y_{\text{act}} - y_{\text{pred}})^2}$ | 主要评价指标，敏感于大误差 |
| **MAE** | $\frac{1}{N}\sum|y_{\text{act}} - y_{\text{pred}}|$ | 平均绝对误差，反映整体偏差 |
| **R²** | $1 - \frac{\sum(y_{\text{act}} - y_{\text{pred}})^2}{\sum(y_{\text{act}} - \bar{y}_{\text{act}})^2}$ | 决定系数，衡量方差解释比例 |

### 基线方法对比
| 模型 | 架构 | 输入特征 | 是否使用BO |
|------|------|----------|------------|
| **Standard Baseline** | CNN-BiLSTM | 12维（无物理导出特征） | 否（手动调参） |
| **Attention-Hybrid** | CNN-BiLSTM-Attention | 15维全特征 | 否 |
| **Proposed PI-Hybrid** | CNN-BiLSTM | 15维全特征 | 是（Bayesian Optimization） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（2023年测试集）
| 模型 | RMSE (W/m²) | MAE (W/m²) | R² |
|------|-------------|-----------|-----|
| **Proposed PI-Hybrid** | **19.53** | **10.68** | **0.997** |
| Attention-Hybrid | 30.64 | 22.31 | 0.992 |
| Standard Baseline | 55.32 | 40.00 | 0.970 |

> ✅ **结论**：物理引导模型 RMSE 比Attention模型低 **36%**，比标准基线低 **65%**

### 与基线方法的对比结果
- **vs. Attention-Hybrid**：
  - 尽管后者结构更复杂并包含Self-Attention机制，但性能反而下降；
  - 错误分布更宽，尤其在正午高峰时段误差显著上升（见Fig. 20）；
  - 表明在强物理规律支配的任务中，“增加复杂性”未必带来收益。

- **vs. Standard Baseline**：
  - 缺乏物理特征导致严重“几何盲区”（Geometric Blindness）；
  - 出现明显**相位滞后**（Phase Lag），无法提前感知日出/日落；
  - 在过渡期（黎明/黄昏）预测不稳定，甚至产生负值或非物理跳跃。

### 消融实验结果
#### （1）物理特征的作用（Ablation: Physics Impact）
- 移除 Clear-Sky GHI、KTcalc、Volatility Index 后，RMSE 从19.53升至55.32；
- R² 下降近3个百分点（0.997 → 0.970）；
- 证明**物理导出特征是性能提升的关键驱动力**。

#### （2）复杂性的代价（Ablation: Complexity Impact）
- 在最优物理模型基础上添加Self-Attention层：
  - RMSE 从 **19.53 → 30.64**
  - R² 从 **0.997 → 0.992**
- **结论**：在已有强物理信号的前提下，Self-Attention不仅无效，反而引入噪声，降低稳定性。

> 🔍 **发现**：“**Complexity Paradox**” —— 在高噪声、有明确物理规律的任务中，**显式物理引导比增加模型复杂度更有效**。

---

## 4. 关键结论和发现

### 主要发现
1. **物理先验知识是最高效的“注意力机制”**  
   - Clear-Sky指数、太阳天顶角等物理特征天然地聚焦于关键时刻（如日出、云过境），其效果优于通过训练学习的Softmax权重分配。

2. **“复杂性优先”范式存在局限性**  
   - Transformer和Self-Attention在图像、NLP领域成功，但在**短周期、低维度、强物理约束的时间序列任务中可能失效**；
   - 它们容易丢失精确时间顺序，在连续预测中表现退化（Zeng et al., 2023）。

3. **轻量化+物理引导 = 更优性价比**  
   - 所提模型参数量不到50万，却实现SOTA精度；
   - 特别适合部署于边缘设备（如光伏电站控制器）进行实时MPPT控制。

4. **模型具有良好的物理一致性**
   - 输出永不超出Clear-Sky极限；
   - 能正确建模温度与辐照度之间的热力学关系；
   - 夜间自动归零，无需后处理干预。

### 方法的局限性
- **依赖高质量物理模型计算Clear-Sky GHI**：若地理位置或大气参数不准，会影响特征质量；
- **仍需一定历史数据训练**：虽强调物理引导，但仍非完全免训练的物理模型；
- **未提供不确定性估计**：当前输出为点预测，缺乏置信区间支持风险决策。

### 未来工作方向
1. **模型压缩与边缘部署**：
   - 应用 **Knowledge Distillation** 训练小型“学生网络”；
   - 使用 **Pruning & Quantization** 技术降低内存占用，适配MCU运行。

2. **构建前瞻性灌溉控制系统**：
   - 利用5小时预测窗口实施“虚拟储能”——在光照充足时提前抽水储存在土壤/水库中；
   - 实现能源-水资源协同调度。

3. **发展概率预测能力**：
   - 输出预测区间而非单一数值；
   - 支持风险感知控制（如高波动期延迟施肥灌溉以防堵塞）。

4. **扩展至其他气候区域验证泛化性**：
   - 当前验证集中于干旱区，未来可在热带多云、温带季风等区域测试迁移能力。

---

> 📌 **一句话总结**：  
> 本文挑战了“越复杂越好”的主流AI范式，实证表明：**在太阳能预测这类受物理定律严格约束的任务中，一个精心设计的、由物理知识引导的轻量级Hybrid模型，可以显著超越昂贵的Self-Attention机制**。这一发现为可再生能源系统的高效智能管理提供了新范式。

</details>

---

### 11. [Jump-Start Reinforcement Learning with Vision-Language-Action Regularization](https://arxiv.org/abs/2604.13733)

**Authors**: Angelo Moroncelli, Roberto Zanetti, Marco Maccarini, Loris Roveda  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.13733v1  

#### Abstract
Reinforcement learning (RL) enables high-frequency, closed-loop control for robotic manipulation, but scaling to long-horizon tasks with sparse or imperfect rewards remains difficult due to inefficient exploration and poor credit assignment. Vision-Language-Action (VLA) models leverage large-scale m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Jump-Start Reinforcement Learning with Vision-Language-Action Regularization*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**强化学习**（RL）在**长时程任务**（long-horizon tasks）和**奖励设计不完善**（imperfect reward design）场景下存在的两大挑战：
- **低效探索**（inefficient exploration）
- **信用分配困难**（suboptimal credit assignment）

尤其是在机器人操作任务中，由于奖励稀疏或延迟，RL 难以在合理交互次数内收敛。

同时，尽管 Vision-Language-Action **（VLA）模型**具备强大的语义理解和泛化能力，但其**推理频率低、控制精度不足**，难以直接用于高频率闭环控制。

### 提出的新方法：VLAJS（Vision-Language-Action Jump-Starting）
提出了一种名为 **VLAJS** 的新框架，将 VLA 模型作为**稀疏、瞬态的辅助指导源**，而非主导控制器。其核心思想是：
- 利用 VLA 模型提供**高层动作先验**（high-level action priors），引导 RL 早期探索；
- 在训练后期逐渐退场，让 RL 策略自主优化并超越教师策略。

#### 关键创新点：
1. **稀疏且瞬态的指导机制**（Sparse and Transient Guidance）  
   - VLA 教师仅在训练初期被**稀疏查询**（如每回合数次），避免高昂的持续推理成本。
   - 指导信号通过**时间离散化**（temporal discretization）扩展为多个控制步的动作目标。

2. **基于奖励趋势的跳过启动机制**（Reward-Trend-Based Jump-Start）  
   - 动态监测 PPO 学习进展（rolling reward improvement）。
   - 当学习进入稳定上升阶段后，**永久关闭 VLA 指导**，实现“跳过启动”后的自由探索。

3. **方向一致性损失**（Directional Action-Consistency Loss, `L_dir`）  
   - 不强制模仿 VLA 输出的具体动作值（如 MSE 损失），而是采用**余弦对齐损失**，仅保留动作方向信息：
     $$
     \mathcal{L}_{\text{dir}}(x, y) = 1 - \frac{x \cdot y}{\|x\|\|y\| + \epsilon}
     $$
   - 允许 RL 自主决定动作幅度和微调，提升灵活性与最终性能。

### 相比现有方法的优势
| 方法 | 局限性 | VLAJS 的改进 |
|------|--------|-------------|
| **Vanilla PPO** | 探索效率低，难处理稀疏奖励 | 显著加速早期学习 |
| **DAgger / 行为克隆** | 持续依赖专家行为，无法超越教师 | 瞬态指导，支持策略超越 |
| **Policy Distillation / RPD** | 持续使用 MSE 损失，限制策略优化空间 | 方向性正则化，更灵活 |
| **端到端 VLA 控制** | 控制频率低，抗干扰差 | 保留高频率状态反馈控制 |

> ✅ **核心优势**：结合了 VLA 的语义先验与 RL 的精确控制，在**样本效率、鲁棒性和部署可行性**之间取得平衡。

---

## 2. 核心实验方法和设置

### 数据集与环境
所有仿真实验均在 **ManiSkill** 操作环境中进行，涵盖六项复杂任务：
- `PickCube`, `PickPlaceCube`
- `LiftPegUpright`, `PegInsertionSide`
- `PokeCube`, `PushCube`

并在真实世界中于 **Franka Panda 机械臂** 上验证零样本迁移能力。

### 观测与动作空间
- **RL 策略**：基于状态的控制器（state-based），输入包括机器人本体感知和模拟器特权状态（如物体位姿）。
- **VLA 教师**：接收 RGB 图像 + 语言指令，输出末端执行器的增量动作（delta action）。
- 动作频率：RL 运行在高控制频率，VLA 查询极稀疏（最多占总步数 20%）。

### 评估指标
1. **Success Rate at t*** (`SR_t*`)  
   在特定交互预算 $t^*$ 内完成任务的成功率。
2. **Area Under the Success Curve** (`AUC`)  
   在整个训练周期内的成功率积分，反映学习速度与最终性能。

报告 **bootstrap 95% 置信区间** 和跨任务**宏平均**（macro-average）。

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **PPO** | 标准 PPO，无任何外部指导 |
| **Sparse RPD** | 稀疏版本的 Refine Policy Distillation，持续使用 MSE 损失进行动作匹配 |
| **VLAJS (RPD)** | 消融实验：使用相同稀疏查询与退火机制，但替换为 MSE 损失 |
| **VLAJS (ours)** | 完整方法：稀疏查询 + 方向一致性损失 + 奖励驱动退火 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table I）
| 方法 | SR_t* (%) | AUC (%) |
|------|----------|--------|
| **PPO** | 34.2 | 61.8 |
| **VLAJS (RPD)** | 35.3 | 59.3 |
| **VLAJS (ours)** | **78.1** | **78.4** |

> 🔺 **VLAJS 相比 PPO 提升超过 40 个百分点的 SR_t***，AUC 提升约 16.6%，显示显著的样本效率增益。

### 与基线方法的对比结果
- 在所有任务上，**VLAJS 均优于 PPO 和 VLAJS (RPD)**。
- 特别是在长时程任务（如 `PickPlaceCube-v2`）中：
  - PPO 成功率为 0%
  - VLAJS 达到 **65.9%**
- 在奖励稀疏的任务中（如 `LiftPegUpright-v3`）：
  - PPO: 13.2%
  - VLAJS (RPD): 19.2%
  - **VLAJS**: **63.3%**

> 💡 表明：**方向性损失 + 瞬态指导** 是成功的关键。

### 消融实验结果
- **VLAJS (RPD)**（MSE 损失）表现不佳，说明：
  - 强制动作匹配会引入噪声梯度，尤其当 VLA 动作未对齐学生控制频率时。
- **稀疏查询 + 时间传播** 有效降低计算开销，同时维持探索引导效果。
- **奖励趋势退火机制** 能可靠检测学习拐点，平均在 3~5 百万步内关闭指导。

> 📈 图 5(c)(d) 显示：所有任务中均出现明显的“jump-start”现象——早期回报迅速上升，随后指导关闭，策略继续独立优化。

### 零样本真实世界部署（Table III）
| 方法 | Lift Cube | Pick & Place | Peg Reorientation |
|------|----------|------------|------------------|
| **OpenVLA-best** | 47% | 40% | — |
| **VLAJS (zero-shot)** | **70%** | **80%** | **20%** |

> ✅ **无需真实世界微调**，VLAJS 训练的策略即可在真实机器人上实现高性能、强鲁棒性操作。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **VLA 可作为有效的“认知导师”**，即使其本身不适合直接部署，也能显著加速 RL 学习。
2. ✅ **稀疏、瞬态、方向性的指导优于持续模仿**，既能引导探索，又不限制策略上限。
3. ✅ **方向一致性损失** 是关键设计，允许策略从粗略建议中提取有用方向信息，而不受尺度误差影响。
4. ✅ **奖励趋势可作为可靠的指导退火信号**，实现自动化的“启动即走”机制。
5. ✅ 所得策略具备出色的**零样本 sim-to-real 能力**，在遮挡、扰动、对象变化等现实条件下仍能稳健运行。

### 方法的局限性
- 依赖一个至少能提供**基本正确方向提示**的 VLA 教师；若教师完全错误，则可能误导学习。
- 当前训练引入额外系统复杂性（GPU 推理服务、内存管理），增加工程负担。
- 依赖特权状态（privileged state）进行 RL 控制，尚未完全扩展至纯视觉输入的 RL。
- 指导退火机制基于启发式规则，在高度随机环境中可能不稳定。

### 未来工作方向
1. **动态查询机制**：仅在策略不确定性高时才激活 VLA 查询，进一步节省计算资源。
2. **全视觉 RL 扩展**：将 VLAJS 应用于端到端视觉策略训练。
3. **多阶段任务与记忆机制**：结合分层 RL 或记忆模块，应对更复杂的长期任务。
4. **真实世界在线微调**：探索如何在真实环境中联合优化 RL 策略与 VLA 模型。
5. **不确定性感知门控**：用策略不确定性或优势估计替代启发式退火条件，提高鲁棒性。

---

> 🏁 **总结一句话**：  
> **VLAJS 成功地将 VLA 的“大脑”与 RL 的“小脑”结合起来——前者提供战略方向，后者负责战术执行，实现了高效、精准、可部署的机器人学习范式。**

</details>

---

### 12. [Physics-Informed Neural Networks for Methane Sorption: Cross-Gas Transfer Learning, Ensemble Collapse Under Physics Constraints, and Monte Carlo Dropout Uncertainty Quantification](https://arxiv.org/abs/2604.13992)

**Authors**: Mohammad Nooraiepour, Zezhang Song, Wei Li, Sarah Perez  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.13992v1  

#### Abstract
Accurate methane sorption prediction across heterogeneous coal ranks requires models that combine thermodynamic consistency, efficient knowledge transfer across data-scarce geological systems, and calibrated uncertainty estimates, capabilities that are rarely addressed together in existing framework...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
本研究旨在解决**地质材料中甲烷吸附容量预测**中的三个关键挑战：
- **数据稀缺性**：在不同煤阶（coal ranks）下获取大规模、高质量的实验数据成本高昂且耗时。
- **物理一致性缺失**：传统机器学习模型虽能拟合数据，但可能违反热力学基本原理（如非单调压力-吸附关系），导致外推不可靠。
- **不确定性量化（UQ）不准确**：现有UQ方法在强物理约束下的表现尚不清楚，尤其当模型需满足多个热力学约束时。

### 提出的新方法与思路
作者提出了一种**融合物理信息神经网络（PINNs）、跨气体迁移学习（cross-gas transfer learning）与蒙特卡洛Dropout（Monte Carlo Dropout）不确定性量化的综合框架**，其核心创新包括：

- **跨气体迁移学习（H₂ → CH₄）**：
  - 利用在氢气（H₂）吸附任务上预训练的PINN作为源模型，通过**Elastic Weight Consolidation (EWC)** 正则化策略，将知识迁移到甲烷（CH₄）吸附预测任务。
  - 物理依据：H₂和CH₄均以伦敦色散力（London dispersion forces）为主导吸附机制，且均符合Sips等温线行为，为跨气体迁移提供了理论基础。

- **三阶段课程学习（three-phase curriculum learning）**：
  - **Phase 1（Warmup）**：冻结编码器，仅训练特征投影层和输出头，建立稳定的特征映射。
  - **Phase 2（Fine-tuning）**：解冻编码器，引入强EWC正则化，防止灾难性遗忘。
  - **Phase 3（Full Optimization）**：逐步放松EWC约束，加强物理损失权重，实现最终微调。

- **系统性贝叶斯不确定性量化比较**：
  - 首次在强物理约束的PINN架构下，系统比较了五种贝叶斯UQ方法（MC Dropout、深集成、拉普拉斯近似等），揭示了“集成坍缩”（ensemble collapse）现象。

### 相比现有方法的优势
- **更高的预测精度与泛化能力**：相比经典等温线模型提升227%，相比随机初始化PINN提升18.9% RMSE。
- **更快的收敛速度**：迁移学习使收敛时间缩短19.4%。
- **更可靠的不确定性估计**：MC Dropout在物理约束下仍能提供良好校准的不确定性，而深集成则失效。
- **物理可解释性**：通过SHAP和ALE分析验证了模型学到的特征重要性与已知煤吸附机理一致。

---

## 2. 核心实验方法和设置

### 数据集
- **来源**：114个独立的煤样吸附实验，共993个平衡测量点。
- **覆盖范围**：涵盖从褐煤（lignite）到无烟煤（anthracite）的全煤阶谱系。
- **输入特征**：5个原始测量值（温度、压力、水分、灰分、挥发分） + 7个物理信息衍生特征（如约化变量 $T_r, P_r$、固定碳、有机质分数、$P \times T$、$\beta=1/(RT)$、水分×挥发分交互项），共12维。
- **目标变量**：甲烷吸附容量（m³/t），经 $\log(y+1)$ 变换以稳定方差。

### 实验设置
- **数据划分**：采用**实验级分组划分（group-aware splitting）**，确保同一煤样的所有测量点只出现在训练集或测试集中（80/20），避免样本泄露。
- **训练协议**：三阶段课程学习，使用AdamW优化器，结合梯度裁剪与早停。
- **物理约束**：四类热力学一致性损失：
  - **Sips一致性**：预测必须符合Sips等温线形式。
  - **物理边界**：吸附量非负且不超过单层容量。
  - **单调性**：在恒温下，吸附量随压力非减。
  - **范特霍夫一致性**：平衡常数 $K(T)$ 满足 $\partial \ln K / \partial (1/T) = -\Delta H/R$。

### 评估指标
- **预测性能**：$R^2$、RMSE、MAE、MaxAE。
- **不确定性校准**：
  - **Expected Calibration Error (ECE)**：越低越好（<0.1为佳）。
  - **误差-不确定性Spearman相关性 ($p_s$)**：越高越好（反映不确定性能否识别高误差）。
  - **Sharpness**：预测区间宽度，越窄越好。
  - **覆盖率（Coverage）**：实际值落入预测区间的比例。
- **统计检验**：Bootstrap配对t检验 + Bonferroni校正，效应量Cohen’s $d$。

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Classical Isotherms** | Langmuir, Freundlich, Sips 等温线（仅用压力作为输入） |
| **Classical Ensemble** | 上述三种等温线的平均预测 |
| **Composition-aware Classical** | 引入挥发分和水分修正的等温线 |
| **Random-random PINN** | 编码器随机初始化，物理头随机初始化 |
| **Random-classical PINN** | 编码器随机初始化，物理头用Sips拟合参数初始化 |
| **Deep Ensemble** | 10个独立训练的random-random PINNs的集成 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 测试集 $R^2$ | RMSE (m³/t) | 提升幅度 |
|------|--------------|-------------|----------|
| Classical Ensemble | 0.285 | 7.41 | - |
| Composition-aware Classical | 0.505 | 6.41 | +77% |
| **Transfer-learned PINN (本文)** | **0.932** | **2.29** | **+227%** |
| Random-random PINN | 0.942 (log) / 0.932 (orig) | 2.29 | - |
| Deep Ensemble (10模型) | 0.941 (log) / 0.931 (orig) | 2.30 | 无显著提升 |

> 注：Transfer-learned PINN在log空间 $R^2=0.962$，优于random-random的0.942。

### 与基线方法对比
- **相比经典等温线**：$R^2$ 从0.285提升至0.932，证明**组成异质性是主导因素**，多变量建模至关重要。
- **相比composition-aware模型**：$R^2$ 从0.505提升至0.932，表明**非线性交互作用**（如水分×挥发分）是进一步提升的关键。
- **迁移学习优势**：
  - **RMSE降低18.9%**（0.139 vs 0.171，log空间）。
  - **收敛速度快19.4%**（达到$R^2 \geq 0.90$所需epoch减少130）。
- **深集成无增益**：10模型集成的性能与单个random-random PINN无统计差异（$p=0.815$），计算成本却增加10倍。

### 消融实验结果
| 对比 | $t$-stat | $p$-value | Cohen’s $d$ | 结论 |
|------|---------|----------|------------|------|
| Transfer vs Random-random | -18.0 | 4.9e-33 | -1.80 | 迁移学习显著更优 |
| Transfer vs Ensemble | -18.9 | 1.2e-34 | -1.89 | 单一迁移模型优于10模型集成 |
| Random-classical vs Transfer | 24.1 | 3.2e-43 | 2.41 | 经典物理头初始化需配合迁移编码器才有效 |
| Random-random vs Ensemble | -0.23 | 0.815 | -0.02 | 深集成无性能增益 |

> 关键发现：**EWC正则化**是迁移成功的关键，它允许编码器在保留H₂知识的同时进行可控适应。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **跨气体迁移学习有效**：H₂ → CH₄的知识迁移显著提升了CH₄吸附预测的准确性与效率，验证了基于共享物理机制的迁移可行性。
2. ⚠️ **深集成在物理约束下失效（Ensemble Collapse）**：
   - 所有集成成员在训练后几乎收敛到相同解。
   - 原因：共享的物理约束（如Sips一致性、单调性等）极大地**压缩了可行解流形（solution manifold）**，导致模型间缺乏功能性分歧（functional disagreement），无法用于epistemic uncertainty估计。
   - 即使采用多样化架构或初始化，也无法避免此现象。
3. ✅ **MC Dropout是更优的UQ方法**：
   - 在物理约束下仍能提供良好校准的不确定性（ECE=0.101, $p_s=0.708$）。
   - 计算开销极低（仅1.5×推理成本），远低于深集成（5–10×）。
   - 观察到**aleatoric uncertainty主导**（占总不确定性的98.3%），epistemic uncertainty仅占1.7%，这反映了物理约束确实减少了模型不确定性。
4. 🔍 **模型具有物理可解释性**：
   - SHAP和ALE分析显示，**水分-挥发分交互项**最重要（17.2%），符合“低阶煤中水分加剧孔隙堵塞”的机理。
   - 温度呈现U型非单调效应（20–30°C增强吸附，30–50°C抑制），解释了为何其线性相关性接近零。
   - 11/12个特征表现出非单调效应，支持使用非线性模型。

### 方法的局限性
- **气体适用范围有限**：该迁移框架适用于以色散力为主的气体（如H₂, CH₄），但对于强极性或偶极相互作用的气体（如CO₂, H₂S）可能不适用。
- **地质材料限制**：若目标材料具有多峰孔径分布或不同的吸附机制（如微孔填充），可能需要不同的源任务。
- **数据规模依赖**：若目标数据集远小于源数据集，EWC强度需重新校准以防过度保留源表示。

### 未来工作方向
- 将框架扩展至**多组分气体混合物**（如CH₄/CO₂竞争吸附）。
- 纳入**非平衡动力学**，处理时间依赖过程。
- 验证迁移学习在**页岩、粘土、富有机沉积物**等其他地质材料中的通用性。
- 探索**线性化拉普拉斯近似**（linearized Laplace approximation）于EWC正则化MAP估计，以获得更严谨的转移学习不确定性分解。

</details>

---

### 13. [Enhancing Clustering: An Explainable Approach via Filtered Patterns](https://arxiv.org/abs/2604.12460)

**Authors**: Motaz Ben Hassine (CRIL), Sa\"id Jabbour (CRIL)  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12460v1  

#### Abstract
Machine learning has become a central research area, with increasing attention devoted to explainable clustering, also known as conceptual clustering, which is a knowledge-driven unsupervised learning paradigm that partitions data into $\theta$ disjoint clusters, where each cluster is described by a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Enhancing Clustering: An Explainable Approach via Filtered Patterns*

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **explainable clustering**（即可解释聚类，又称概念聚类）中的一个关键瓶颈：在基于 **k-Relaxed Frequent Patterns (k-RFPs)** 的聚类框架中，多个不同的 k-RFPs 可能诱导出相同的 **k-cover**，导致候选模式集中存在大量冗余。这种冗余不仅扩大了搜索空间，还显著增加了后续 **ILP**（Integer Linear Programming）求解器的计算开销。

### 提出的新方法与新思路
作者提出了 **Optimized Conceptual Clustering Method (OCCM)**，其核心思想是通过**过滤冗余模式**来优化聚类过程。具体贡献如下：

- **理论分析**：形式化地刻画了何时两个不同的 k-RFPs 会诱导出相同的 k-cover（Proposition 2），为冗余检测提供了理论基础。
- **模式过滤算法**：提出了一种高效的 **Pattern Filtering Algorithm**（Algorithm 1），对生成的 k-RFPs 进行后处理，为每个唯一的 k-cover 仅保留一个代表性模式。
- **代表性选择策略**：在多个共享相同 k-cover 的模式中，优先保留**最大项集**（largest itemset），以增强聚类结果的可解释性和描述能力。
- **可解释性度量**：引入了两个新的理论度量来评估所选模式的代表性和鲁棒性：
  - **Shapley Value Variance (SVV)**：衡量模式内各 item 贡献的分布差异。
  - **Average Cluster Stability (ACS)**：衡量从模式中移除单个 item 后，其诱导的 cluster 的稳定性。

### 相比现有方法的优势
- **提升效率**：显著减少输入到 ILP 模型的候选模式数量，从而大幅降低 ILP 求解时间。
- **保持甚至提升聚类质量**：在多数数据集上保持了与基线相当的 F1-score，并在部分数据集（如 Mushroom 和 Primary-Tumor）上实现了质量提升。
- **增强可解释性**：通过保留更大的模式，提供更丰富、更具代表性的聚类描述。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在六个真实世界数据集上进行，均为事务型数据（transactional data）：

| 数据集 | |D|（事务数） | |U|（项数） | 密度 (%) |
|--------|------------|----------|---------|
| Lymph | 148 | 68 | 40 |
| Mushroom | 8124 | 119 | 18 |
| Primary-Tumor | 336 | 31 | 48 |
| Soybean | 630 | 50 | 32 |
| Tic-tac-toe | 958 | 27 | 33 |
| Vote | 435 | 48 | 33 |

### 实验设置
- **k 值**：固定为 `k=1`。
- **聚类数**：固定为 `θ=2`，因所有数据集的真实划分均为二分类。
- **最小支持度阈值 σ**：在不同实验阶段取值不同（10%~40%）。
- **超时限制**：ILP 求解最大运行时间为 1 小时。

### 评估指标
1. **模式数量对比**：比较过滤前后的 k-RFP 数量，计算减少百分比。
2. **计算效率**：ILP 求解的 CPU 时间。
3. **聚类质量**：使用 **F1-score** 与真实标签（ground-truth clusters）对比。
4. **可解释性分析**：
   - **SVV**：Shapley Value Variance
   - **ACS**：Average Cluster Stability
   - 分析 SVV、ACS 与模式大小之间的相关性。

### 基线方法对比
- **CCA-k-RFP-M1**：由 Hassine et al. (2024) 提出的基于 k-RFPs 的概念聚类方法，作为主要基线。
- **OCCM**：本文提出的优化方法，即在 CCA-k-RFP-M1 的 k-RFP 生成之后加入过滤步骤。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 模式过滤效果（Phase I）
- 在所有数据集和 σ 设置下均观察到冗余模式的存在。
- 最高减少比例达 **26.67%**（Tic-tac-toe, σ=40%）。
- 即使在低支持度（σ=10%）下，如 Mushroom 数据集也减少了 **8.32%** 的模式。

#### ILP 求解时间与聚类质量（Phase II）

| 数据集 | 方法 | |A| / |Af| | F1-score | CPU Time (s) |
|--------|------|--------|----------|--------------|
| Lymph | CCA-k-RFP-M1 | 91,888 | 0.71 | 49.66 |
| Lymph | OCCM | 85,470 | 0.71 | **29.74** |
| Mushroom | CCA-k-RFP-M1 | 19,712 | 0.34 | 3133.34 |
| Mushroom | OCCM | 18,176 | **0.73** | **1144.46** |
| Primary-Tumor | CCA-k-RFP-M1 | 45,465 | 0.25 | 55.88 |
| Primary-Tumor | OCCM | 45,250 | **0.33** | **55.71** |
| Soybean | CCA-k-RFP-M1 | 11,900 | 0.29 | 18.29 |
| Soybean | OCCM | 11,664 | 0.29 | **16.99** |
| Vote | CCA-k-RFP-M1 | 280,386 | 0.51 | 1206.17 |
| Vote | OCCM | 280,179 | 0.51 | **234.30** |
| Tic-tac-toe | 两种方法 | — | — | 超时未解出 |

> ✅ **结论**：OCCM 在所有可解出的数据集上均**显著缩短了 ILP 求解时间**，且在 Mushroom 和 Primary-Tumor 上**提升了 F1-score**。

#### 可解释性分析（Phase III）
- **SVV 与 ACS 的关系**：大多数数据集呈现负相关——贡献越均衡（SVV 越小），集群越稳定（ACS 越高）。
- **模式大小与 ACS 的关系**：在所有数据集上呈现**强正相关**（见 Figure 4）。更大的模式产生更稳定的集群。
- **支持过滤策略**：由于更大的模式更稳定且更具代表性，因此在过滤时保留最大项集是合理且有效的设计。

---

## 4. 关键结论和发现

### 主要发现
1. **冗余普遍存在**：在 k-RFPs 中，多个不同模式共享相同 k-cover 是常见现象，验证了理论分析（Proposition 2）。
2. **过滤有效提升效率**：通过去除冗余模式，OCCM 显著减少了 ILP 输入规模，平均求解时间大幅下降。
3. **不牺牲质量，反可能提升**：过滤操作不仅保持了聚类质量，在某些情况下还因选择了更具代表性的大模式而提升了 F1-score。
4. **模式大小影响稳定性**：实验证明，**更大的模式倾向于产生更稳定的 cluster**（高 ACS），支持了“保留最大项集”的策略。
5. **贡献均衡性与稳定性相关**：item 贡献越均衡（低 SVV），cluster 对 item 移除越鲁棒（高 ACS）。

### 方法的局限性
- **依赖后处理**：当前的过滤是在 SAT 生成 k-RFPs 之后进行的，属于后处理步骤，未能从根本上避免冗余生成。
- **SAT 编码复杂性**：尝试将去重逻辑直接编码进 SAT 求解器会导致约束过于复杂，影响可扩展性。
- **目标函数局限**：当前 ILP 仅最大化模式大小，尚未整合更复杂的可解释性指标（如 SVV 或 ACS）作为优化目标。

### 未来工作方向
1. **将去重机制嵌入 SAT 生成过程**：设计更紧凑、高效的 SAT 编码，直接生成无冗余的 k-RFPs。
2. **扩展 ILP 目标函数**：在优化过程中显式考虑可解释性指标（如最小化 SVV 或最大化 ACS），而不仅仅是模式大小。
3. **探索其他模式模型**：研究是否可以推广该过滤思想到其他类型的 relaxed pattern 模型中。
4. **应用于更大规模和多类别场景**：当前实验限于 θ=2，未来可拓展至更复杂的聚类任务。

</details>

---

### 14. [RPRA: Predicting an LLM-Judge for Efficient but Performant Inference](https://arxiv.org/abs/2604.12634)

**Authors**: Dylan R. Ashley, Ga\"el Le Lan, Changsheng Zhao, Naina Dhingra, Zhipeng Cai, Ernie Chang, Mingchen Zhuge, Yangyang Shi, Vikas Chandra, J\"urgen Schmidhuber  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12634v1  

#### Abstract
Large language models (LLMs) face a fundamental trade-off between computational efficiency (e.g., number of parameters) and output quality, especially when deployed on computationally limited devices such as phones or laptops. One way to address this challenge is by following the example of humans a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RPRA: Predicting an LLM-Judge for Efficient but Performant Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在计算效率与输出质量之间存在根本性权衡，尤其在资源受限设备（如手机、笔记本）上部署时面临高昂的推理成本。小模型虽然高效，但在复杂任务上表现不稳定，容易产生“诡异”或错误的输出（如对“strawberry中有几个r？”这类问题回答错误）。因此，如何让小模型在自身能力范围内自信作答，而在超出能力时主动求助大模型，是提升系统整体效率与可靠性的关键。

本文提出了一种**自我认知机制**，使模型能在生成响应前预测一个外部LLM judge对其输出的评分，从而实现智能路由（small model处理简单查询，large model处理困难查询），在保证质量的同时显著降低计算开销。

---

### 提出的新方法与新思路
论文提出了两个核心范式：

- **Predict-Answer/Act (PA)**  
  模型在生成答案前，先预测一个LLM judge会如何评价其即将生成的回答（如“great”、“ok”、“bad”），并基于此决定是否直接作答或转交给更大模型。

- **Reason-Predict-Reason-Answer/Act (RPRA)**  
  在PA基础上扩展，要求模型先进行初步推理（reason），再预测judge评分，最后再次推理并作答。该范式更适用于具备多步推理能力的模型。

为实现上述范式，作者探索了三种具体实现方式：

1. **Zero-shot Prediction**  
   不依赖任何历史信息，仅通过prompt让模型直接预测judge评分。

2. **In-context Learning with Report Card**  
   为每个模型提供一份“成绩单”（report card），汇总其在多个数据集上的历史表现模式（mode score），作为上下文辅助预测。

3. **Supervised Fine-tuning with Hindsight Trick**  
   利用“事后诸葛亮”技巧（hindsight trick）构建训练数据：将实际的judge评分作为标签，对小模型进行监督微调，使其学会预测自身性能。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **灵活性** | 使用LLM/agentic judge而非固定metric，支持灵活、可定制的评估标准（通过修改prompt即可调整评判维度）。 |
| **通用性** | 可应用于任意规模模型（包括closed-weight模型如ChatGPT），无需修改架构。 |
| **效率提升** | 小模型可通过报告卡或微调获得接近大模型的预测能力，避免频繁调用昂贵的大模型。 |
| **无需实时生成** | 预测发生在响应生成之前，节省了无效生成的成本。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用5个多样化数据集，覆盖不同任务类型：
- **MedQA**：医学诊断问答（来自专业考试）
- **LongFact**：事实性冷知识问答
- **AIME 2024**：竞赛级数学题（需多步推理）
- **MMLU-Pro**：本科水平多学科选择题（涵盖科学、人文等）
- **SciCode**：科研编程任务（需代码实现）

这些数据集确保了评估的广度与挑战性。

---

### 实验设置与评估指标

#### 模型集合
共测试11个模型，参数量从0.9B到120B不等，包括：
| 模型简称 | 参数量 | 是否具备Reasoning能力 |
|---------|--------|------------------|
| M09B (MobileLLM 0.9B) | 0.9B | 否 |
| L318B (Llama 3.1 8B) | 8.03B | 否 |
| DSQ32B (DeepSeek R1 Distilled Qwen 32B) | 32.8B | 是 |
| GPT120B (GPT OSS 120B) | 120B | 是 |
| ... | ... | ... |

完整列表见原文Table 1。

#### Judge 设置
- 使用 **Llama 3.3 70B** 作为主judge模型。
- 所有模型对同一query的响应被**同时提交给judge**进行相对评估（避免独立打分导致的偏差）。
- Judge依据预定义的细粒度rubric进行评分（见Prompt 11），分为三类：
  - `great`：完全符合所有标准
  - `ok`：多数达标，无严重缺陷
  - `bad`：有一项及以上为“坏”

#### 评估任务
目标是预测judge对自己响应的评分（三分类任务），评估指标为：
- **Accuracy**：正确预测judge评分的比例
- 对比三种方法下的准确率提升：
  - Zero-shot
  - In-context（带report card）
  - Fine-tuned（监督微调后）

---

### 基线方法对比
- **Zero-shot baseline**：直接预测，无额外信息输入
- **Random Guess**：随机猜测三个等级，期望准确率为~33%
- **Report Card vs. No Report Card**：验证上下文信息的有效性
- **Fine-tuned vs. Zero-shot**：验证微调带来的增益

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 报告卡方法（In-context）效果（Table 3）
| 方法 | 平均准确率提升（vs. zero-shot） |
|------|-------------------------------|
| **Report Card** | 最高 **+55%**（L321B在MedQA上） |
| 多数小模型提升显著，尤其是非推理模型 |

典型结果示例（Figure 8, MedQA）：
- M09B（0.9B）zero-shot准确率 ≈ 30%，加入report card后提升至约 **85%**
- L318B（8B）从 ~40% 提升至 ~90%

> ✅ **结论**：即使是极小模型，在获得历史表现摘要后也能大幅提高自我评估准确性。

---

#### 微调方法（Fine-tuning）效果（Table 2）
| 方法 | 平均准确率提升（vs. zero-shot） |
|------|-------------------------------|
| **Supervised Fine-tuning** | 最高 **+52%**（LP321B） |
| 微调后的模型预测性能达到甚至超过report card方法 |

例如：
- LP321B（Llama 3.2 1B微调版）在AIME 2024上比zero-shot提升 **+66%**
- 在MMLU-Pro上平均提升 **+64%**

> ✅ **结论**：微调能内化预测能力，避免推理时加载长report card的token开销。

---

### 与基线方法的对比结果
| 方法 | 准确率 | 优点 | 缺点 |
|------|-------|------|------|
| **Zero-shot** | 低（部分接近随机） | 无需训练或上下文 | 小模型严重过自信（overconfident） |
| **Report Card** | 显著提升（最高+55%） | 无需训练，适用于闭源模型 | 增加输入长度，带来延迟 |
| **Fine-tuning** | 最高（媲美或超越report card） | 推理高效，无需额外context | 需要训练资源，可能牺牲原始性能 |

> 📌 总体趋势：**大模型（尤其reasoning models）zero-shot表现良好；小模型严重依赖report card或fine-tuning才能有效预测。**

---

### 消融实验结果

#### （1）Judge的一致性分析（Figure 12）
- 将MMLU-Pro按学科拆分，观察judge打分分布。
- 发现**模型本身的影响远大于题目类别** → 表明judge打分稳定，非随机波动。

#### （2）不同rubric的鲁棒性测试（Appendix I）
- 使用“恶意反转”的评分标准（mischievous rubric），即把原本“好”的标准定义为“差”
- 结果显示模型仍能以高于随机的准确率进行预测 → 表明模型不仅依赖模板匹配，还理解任务难度和自身能力。

#### （3）短版report card测试（Appendix H）
- 使用简化版report card（Prompt 6）
- 结果显示性能明显下降 → 说明**详细的历史反馈信息至关重要**

#### （4）独立vs联合打分（Appendix E）
- 若judge单独评估每个模型输出，评分多样性降低
- 联合评估（all-at-once）能更好地区分模型间差异 → 支持本文采用的方法

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **大模型（尤其是reasoning models）具备较强的zero-shot自我评估能力**，而小模型普遍过自信或欠自信。
2. ✅ **Report Card机制能极大提升小模型的预测准确性**（平均提升可达55%），使其可用于PA范式中的智能路由。
3. ✅ **通过supervised fine-tuning + hindsight trick，可将预测能力“固化”进小模型中**，实现免上下文的高效PA推理。
4. ✅ **模型在难题上的自我意识更强**：越难的任务，模型越倾向于正确预测自己会得低分。
5. ✅ **LLM judge的评分具有较高一致性**，适合作为统一评估代理。

---

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **Report Card带来额外token开销** | 虽然输入token便宜，但仍增加延迟，影响实时性 |
| **Fine-tuning需额外训练成本** | 需维护两套权重或牺牲原模型性能 |
| **当前仅限单轮交互** | 未考虑对话场景下的动态能力变化 |
| **依赖高质量judge模型** | 若judge本身不可靠，则预测失去意义 |
| **未在真实路由系统中验证端到端收益** | 当前为可行性研究，尚未量化实际成本节约 |

---

### 未来工作方向
1. **多轮对话中的动态能力预测**  
   扩展至对话系统，根据上下文动态判断何时需要升级模型。

2. **集成预测与生成于同一forward pass**  
   设计统一prompt，让模型在同一轮中完成“预测+作答”，减少冗余计算。

3. **结合人类反馈进行强化学习**  
   使用RLHF或RLAIF优化预测策略，使其更贴近真实用户满意度。

4. **轻量化微调方案**  
   探索LoRA、Adapter等参数高效微调技术，降低fine-tuning门槛。

5. **支持动态alignment需求**  
   利用LLM judge的灵活性，预测模型在特定伦理、安全、风格等维度的表现。

---

## 总结
本论文开创性地提出了 **PA/RPRA范式**，通过让模型预测LLM judge的评分来实现**自我认知驱动的高效推理**。实验证明，无论是通过**in-context report card**还是**supervised fine-tuning**，都能显著提升小模型的自我评估能力，为其在资源受限环境下安全、高效运行提供了可行路径。这标志着向**更自知、更节能、更智能的AI系统**迈出了重要一步。

</details>

---

### 15. [BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design](https://arxiv.org/abs/2604.12898)

**Authors**: Chuyang Xiang, Yichen Wei, Jiale Ma, Handing Wang, Junchi Yan  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12898v1  

#### Abstract
Large Language Model-based Hyper Heuristic (LHH) has recently emerged as an efficient way for automatic heuristic design. However, most existing LHHs just perform well in optimizing a single function within a pre-defined solver. Their single-layer evolution makes them not effective enough to write a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design —— 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现有的 **Language Hyper-Heuristics (LHH)** 方法在自动启发式设计（Automatic Heuristic Design, AHD）中存在两大根本性缺陷：

1. **结构与提示策略缺陷**：大多数 LHH 是单层进化框架，将整个算法视为一个“个体”进行演化。这导致其难以生成复杂的完整求解器（complete solver），通常只能优化预定义框架内的单一函数（如优先级函数、惩罚项等），缺乏对高层算法结构的建模能力。
2. **知识增强缺失或不足**：现有方法要么让 LLM 从零开始设计算法（缺乏外部知识支持），要么依赖人工设计的代码模板（限制多样性且需大量手动干预），无法有效引导 LLM 进行高质量、可复用的复杂代码生成。

### **提出了什么新方法或新思路**

为解决上述问题，作者提出 **BEAM (Bi-level Memory-adaptive Algorithmic Evolution)**，其核心思想是将启发式设计重构为一个 **Bi-level Optimization** 问题，并引入自适应记忆机制：

- **双层优化架构 (Bi-level Optimization Framework)**：
  - **外层（Exterior Layer）**：使用 **Genetic Algorithm (GA)** 演化高层算法结构（如流程控制、模块组合），即“算法骨架”。
  - **内层（Interior Layer）**：使用 **Monte Carlo Tree Search (MCTS)** 实现结构中的函数占位符（function placeholders），即“功能实现”。
  - 该分层设计模仿人类专家“先规划框架，再填充细节”的设计范式。

- **自适应记忆（Adaptive Memory, AM）**：
  - 在进化过程中动态维护一个高价值函数池，允许 LLM 直接调用历史生成的优质函数（通过 `import` 形式），避免重复生成冗长代码。
  - 支持函数的重用与重组，提升代码生成效率与多样性。

- **知识增强管道（Knowledge Augmentation, KA）**：
  - 构建两个结构化数据库供 LLM 调用：
    - **HeuBase**：可调用的启发式组件库（包括 pip 安装库与手写高性能组件）。
    - **KnoBase**：基于任务标签检索的文本型知识库，用于注入领域先验知识。
  - 该管道旨在支持“从组件构建完整求解器”的评估范式，而非仅优化单一函数。

### **相比现有方法的优势**

| 维度 | 传统 LHH（如 EoH, ReEvo） | BEAM |
|------|--------------------------|------|
| **设计粒度** | 单一函数（within fixed framework） | 完整算法（entire algorithm / hybrid solver） |
| **搜索结构** | 单层（Single-layer） | 双层（Bi-level） |
| **函数复用** | 不支持 | 支持（Adaptive Memory） |
| **知识利用** | 零知识 或 固定模板 | 动态构建 HeuBase & KnoBase |
| **探索效率** | 易陷入局部最优，退化为随机搜索 | 结构与功能分离，探索更高效 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

实验覆盖多种优化问题类型，包括：

| 问题类别 | 具体任务 | 数据集 |
|--------|--------|-------|
| **组合优化 (CO)** | TSP, BPP, CVRP, MIS | TSP-50/100/500, Weibull5k/10k/100k, CVRP-100/200/500, RB 200-300 / RB 800-1200 |
| **贝叶斯优化 (BO)** | CAF (Cost-aware Acquisition Function) | Ackley, Rastrigin, Griewank, Levy 等合成函数 |
| **连续优化 (BBO)** | Black Box Optimization | BBOB 基准套件（Rastrigin, Rosenbrock, Sphere, Ackley, Griewank） |
| **调度问题** | PMSP (Parallel Machine Scheduling) | ZeroBubble 文献中的真实数据 |

### **实验设置和评估指标**

- **LLM 模型**：主要使用 `DeepSeek-V3` 和 `DeepSeek-R1`，温度设置为 0.7–1.0。
- **预算控制**：严格控制各方法的运行时间或 Token 消耗，确保公平比较（见 Table II）。
- **评估指标**：
  - **Gap**：与已知最优解或当前最优解之间的差距（越小越好）。
  - **Objective Value (OBJ)**：最终目标函数值（越小越好）。
  - **Best & Average Performance**：多次运行的最佳与平均表现。
- **硬件环境**：Apple M3 CPU；部分依赖 GPU/CUDA 的算法在 NVIDIA RTX 4070 Ti 上运行。

### **基线方法对比**

对比了多种主流 LHH 与 SOTA 求解器：

- **LHH 方法**：
  - `EoH`, `ReEvo`, `MCTS-AHD`, `AlphaEvolve`, `PoH`, `FunSearch`
- **SOTA 求解器**：
  - `KaMIS`（MIS 问题）
  - `HGS`（CVRP 问题）
  - `CMA-ES`, `PSO` 等经典优化器

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **CVRP 混合算法设计（Table V）**
- BEAM 在 CVRP 上相较现有 LHH 平均 **降低优化差距（optimality gap）达 37.84%**。
- 在 CVRP-500 上，BEAM 达到 **0.86% Gap**，显著优于 ReEvo (1.52%) 和 EoH (1.09%)。

#### ✅ **最大独立集（MIS）问题（Table V）**
- BEAM 设计的启发式算法在 RB-800-1200 实例上达到 **43.03 OBJ**，**超越 SOTA 求解器 KaMIS (43.00)**。
- 在 SATLIB 上也实现了 **425.95 OBJ**，接近 KaMIS 表现。

#### ✅ **连续优化（BBOB）（Table VI）**
- BEAM 在多个 BBOB 函数上接近或达到 SOTA 水平：
  - **Rastrigin**: 0.026 Gap
  - **Rosenbrock**: 0.000 Gap
  - **Sphere**: 0.000 Gap
  - **平均 Gap 仅为 0.007**，远优于 ReEvo (0.957) 和 EoH (6.032)。

#### ✅ **传统单函数任务（Table III & IV）**
- 尽管 BEAM 并非专为此类任务设计，但在 TSP、BPP、CAF 上仍表现出色：
  - 在 CAF 任务上，BEAM 在多数数据集上优于 `AlphaEvolve`。
  - 在 TSP 上，BEAM 实现 **0.88% Gap**，优于 ReEvo (1.00%) 和 EoH (0.85%)。

### **与基线方法的对比结果**

| 方法 | CVRP Gap ↓ | MIS OBJ ↑ | BBOB Avg Gap ↓ |
|------|------------|-----------|----------------|
| ReEvo | 1.52% | 41.21 | 0.957 |
| EoH | 1.09% | 41.65 | 6.032 |
| MCTS-AHD | 1.29% | 41.13 | 0.386 |
| **BEAM (Ours)** | **0.86%** | **43.03** | **0.007** |

> ✅ BEAM 在所有复杂任务上均显著领先。

### **消融实验结果（Ablation Study）**

#### 🔹 **自适应记忆（Adaptive Memory）的影响（Table VII & VIII）**
- 移除 AM 后（记为 BE），在 TSP 上性能下降至 **-8.12%**（相对 BEAM 的 -9.55%），稳定性变差（方差从 0.01 升至 0.19）。
- 说明 AM 显著提升了 **收敛速度、峰值性能与稳定性**。

#### 🔹 **教育方法（Education Method）对比**
- 内层采用 **MCTS** 比 **One-Shot** 更优：
  - 在 MIS 上：MCTS → 3.05% Gap，One-Shot → 3.63% Gap
  - 在 CVRP 上：MCTS → 0.86% Gap，One-Shot → 1.07% Gap
- 证明 MCTS 能更有效地探索函数实现空间。

#### 🔹 **模型泛化性测试**
- 使用较小模型（如 GPT-3.5 Turbo）时，BEAM 仍能生成高质量代码，表明其性能不完全依赖于大模型规模。

---

## 4. 关键结论和发现

### **主要发现**

1. **双层架构是关键**：将算法结构演化与函数实现分离，使 LLM 能专注于各自层级的优化，显著提升复杂求解器的设计能力。
2. **自适应记忆促进创新**：通过函数复用与重组，BEAM 鼓励 LLM 探索新的组合方式，而非重复发明轮子。
3. **知识增强不可或缺**：HeuBase 与 KnoBase 的引入使 LLM 能基于已有知识构建更可靠、高效的算法，避免“从零开始”的盲目探索。
4. **BEAM 能超越 SOTA 求解器**：在 MIS 问题上，BEAM 自动生成的算法已超过手工设计的 KaMIS，验证了其强大潜力。

### **方法的局限性**

1. **计算开销较大**：由于双层结构与 MCTS 搜索，BEAM 生成首个个体所需 Token 数量较多（见 Fig. 8）。
2. **可能过度复杂化简单任务**：对于简单目标（如 BPP），BEAM 生成的代码往往比必要更复杂（code length 更长），存在“杀鸡用牛刀”现象。
3. **依赖高质量组件库**：HeuBase 的质量直接影响最终性能，若缺少关键组件，可能限制设计上限。

### **未来工作方向**

1. 扩展 BEAM 至更复杂的领域（如多目标优化、约束满足问题）。
2. 探索更高效的 **Knowledge Augmentation** 方式，例如结合 RAG 或向量检索。
3. 优化双层协同机制，减少通信成本与 Token 消耗。
4. 研究如何自动构建与更新 HeuBase，形成“自我进化的启发式生态系统”。

---

> **总结**：BEAM 通过 **Bi-level Optimization + Adaptive Memory + Knowledge Augmentation** 三重创新，成功突破了传统 LHH 的能力边界，首次实现了由 LLM 自主设计出超越 SOTA 求解器的完整启发式算法，为自动化算法设计开辟了新路径。

</details>

---

### 16. [YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference](https://arxiv.org/abs/2604.13556)

**Authors**: You Wu, Ziheng Chen, Yizhen Zhang, Haoyi Wu, Chengting Yu, Yuchi Xu, Wenbo Su, Bo Zheng, Kewei Tu  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.13556v1  

#### Abstract
Cross-layer key-value (KV) compression has been found to be effective in efficient inference of large language models (LLMs). Although they reduce the memory consumption of the KV cache, such methods usually introduce non-negligible performance degradation. In this work, we aim to enhance the perfor...

---

### 17. [SAKURAONE: An Open Ethernet-Based AI HPC System and Its Observed Workload Dynamics in a Single-Tenant LLM Development Environment](https://arxiv.org/abs/2604.13600)

**Authors**: Fumikazu Konishi, Yuuki Tsubouchi, Hirofumi Tsuruta  
**Category**: cs.DC  
**Published**: 2026-04-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.13600v1  

#### Abstract
SAKURAONE is a managed high performance computing (HPC) cluster developed and operated by the SAKURA Internet Research Center. It builds on the KOKARYOKU PHY bare metal GPU platform and is optimized for advanced workloads, including large language model (LLM) training. In ISC 2025 TOP500, SAKURAONE ...

---

### 18. [Multi-Task LLM with LoRA Fine-Tuning for Automated Cancer Staging and Biomarker Extraction](https://arxiv.org/abs/2604.13328)

**Authors**: Jiahao Shao, Anam Nawaz Khan, Christopher Brett, Tom Berg, Xueping Li, Bing Yao  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.13328v1  

#### Abstract
Pathology reports serve as the definitive record for breast cancer staging, yet their unstructured format impedes large-scale data curation. While Large Language Models (LLMs) offer semantic reasoning, their deployment is often limited by high computational costs and hallucination risks. This study ...

---

### 19. [Optimization with SpotOptim](https://arxiv.org/abs/2604.13672)

**Authors**: Thomas Bartz-Beielstein  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.13672v1  

#### Abstract
The `spotoptim` package implements surrogate-model-based optimization of expensive black-box functions in Python. Building on two decades of Sequential Parameter Optimization (SPO) methodology, it provides a Kriging-based optimization loop with Expected Improvement, support for continuous, integer, ...

---

### 20. [LLM-HYPER: Generative CTR Modeling for Cold-Start Ad Personalization via LLM-Based Hypernetworks](https://arxiv.org/abs/2604.12096)

**Authors**: Luyi Ma, Wanjia Sherry Zhang, Zezhong Fan, Shubham Thakur, Kai Zhao, Kehui Yao, Ayush Agarwal, Rahul Iyer, Jason Cho, Jianpeng Xu, Evren Korpeoglu, Sushant Kumar, Kannan Achan  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12096v1  

#### Abstract
On online advertising platforms, newly introduced promotional ads face the cold-start problem, as they lack sufficient user feedback for model training. In this work, we propose LLM-HYPER, a novel framework that treats large language models (LLMs) as hypernetworks to directly generate the parameters...

---

### 21. [Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization](https://arxiv.org/abs/2604.12290)

**Authors**: Yizhe Chi, Deyao Hong, Dapeng Jiang, Tianwei Luo, Kaisen Yang, Boshi Zhang, Zhe Cao, Xiaoyan Fan, Bingxiang He, Han Hao, Weiyang Jin, Dianqiao Lei, Qingle Liu, Houde Qian, Bowen Wang, Situ Wang, Youjie Zheng, Yifan Zhou, Calvin Xiao, Eren Cai, Qinhuai Na  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12290v1  

#### Abstract
Current LLM agent benchmarks, which predominantly focus on binary pass/fail tasks such as code generation or search-based question answering, often neglect the value of real-world engineering that is often captured through the iterative optimization of feasible designs. To this end, we introduce Fro...

---

### 22. [KnowRL: Boosting LLM Reasoning via Reinforcement Learning with Minimal-Sufficient Knowledge Guidance](https://arxiv.org/abs/2604.12627)

**Authors**: Linhao Yu, Tianmeng Yang, Siyu Ding, Renren Jin, Naibin Gu, Xiangzhao Hao, Shuaiyi Nie, Deyi Xiong, Weichong Yin, Yu Sun, Hua Wu  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12627v1  

#### Abstract
RLVR improves reasoning in large language models, but its effectiveness is often limited by severe reward sparsity on hard problems. Recent hint-based RL methods mitigate sparsity by injecting partial solutions or abstract templates, yet they typically scale guidance by adding more tokens, which int...

---

### 23. [DeEscalWild: A Real-World Benchmark for Automated De-Escalation Training with SLMs](https://arxiv.org/abs/2604.13075)

**Authors**: Md Hasebul Hasan, Krity Haque Charu, Eshwara Prasad Sridhar, Shuchisnigdha Deb, Mohammad A. Islam  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.13075v1  

#### Abstract
Effective de-escalation is critical for law enforcement safety and community trust, yet traditional training methods lack scalability and realism. While Large Language Models (LLMs) enable dynamic, open-ended simulations, their substantial computational footprint renders them impractical for deploym...

---

### 24. [ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding](https://arxiv.org/abs/2604.13519)

**Authors**: Heming Xia, Yongqi Li, Cunxiao Du, Mingbo Song, Wenjie Li  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.13519v1  

#### Abstract
Tool calling has greatly expanded the practical utility of large language models (LLMs) by enabling them to interact with external applications. As LLM capabilities advance, effective tool use increasingly involves multi-step, multi-turn interactions to solve complex tasks. However, the resulting gr...

---

### 25. [Debate to Align: Reliable Entity Alignment through Two-Stage Multi-Agent Debate](https://arxiv.org/abs/2604.13551)

**Authors**: Cunda Wang, Ziying Ma, Po Hu, Weihua Wang, Feilong Bao  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.13551v1  

#### Abstract
Entity alignment (EA) aims to identify entities referring to the same real-world object across different knowledge graphs (KGs). Recent approaches based on large language models (LLMs) typically obtain entity embeddings through knowledge representation learning and use embedding similarity to identi...

---

### 26. [Doc-V*:Coarse-to-Fine Interactive Visual Reasoning for Multi-Page Document VQA](https://arxiv.org/abs/2604.13731)

**Authors**: Yuanlei Zheng, Pei Fu, Hang Li, Ziyang Wang, Yuyi Zhang, Wenyu Ruan, Xiaojin Zhang, Zhongyu Wei, Zhenbo Luo, Jian Luan, Wei Chen, Xiang Bai  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.13731v1  

#### Abstract
Multi-page Document Visual Question Answering requires reasoning over semantics, layouts, and visual elements in long, visually dense documents. Existing OCR-free methods face a trade-off between capacity and precision: end-to-end models scale poorly with document length, while visual retrieval-base...

---

### 27. [MAny: Merge Anything for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2604.14016)

**Authors**: Zijian Gao, Wangwang Jia, Xingxing Zhang, Pengfei Qian, Tao Sun, Bo Ding, Yong Dou, Huaimin Wang, Kele Xu  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.14016v1  

#### Abstract
Multimodal Continual Instruction Tuning (MCIT) is essential for sequential task adaptation of Multimodal Large Language Models (MLLMs) but is severely restricted by catastrophic forgetting. While existing literature focuses on the reasoning language backbone, in this work, we expose a critical yet n...

---

### 28. [A hierarchical spatial-aware algorithm with efficient reinforcement learning for human-robot task planning and allocation in production](https://arxiv.org/abs/2604.12669)

**Authors**: Jintao Xue, Xiao Li, Nianmin Zhang  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.12669v1  

#### Abstract
In advanced manufacturing systems, humans and robots collaborate to conduct the production process. Effective task planning and allocation (TPA) is crucial for achieving high production efficiency, yet it remains challenging in complex and dynamic manufacturing environments. The dynamic nature of hu...

---

### 29. [Cycle-Consistent Search: Question Reconstructability as a Proxy Reward for Search Agent Training](https://arxiv.org/abs/2604.12967)

**Authors**: Sohyun An (Meta Superintelligence Labs, UCLA), Shuibenyang Yuan (Meta Superintelligence Labs), Hayeon Lee (Meta Superintelligence Labs), Cho-Jui Hsieh (UCLA), Alexander Min (Meta Superintelligence Labs)  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.12967v1  

#### Abstract
Reinforcement Learning (RL) has shown strong potential for optimizing search agents in complex information retrieval tasks. However, existing approaches predominantly rely on gold supervision, such as ground-truth answers, which is difficult to scale. To address this limitation, we propose Cycle-Con...

---

### 30. [Bi-Predictability: A Real-Time Signal for Monitoring LLM Interaction Integrity](https://arxiv.org/abs/2604.13061)

**Authors**: Wael Hafez, Amir Nazeri  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.13061v1  

#### Abstract
Large language models (LLMs) are increasingly deployed in high-stakes autonomous and interactive workflows, where reliability demands continuous, multi-turn coherence. However, current evaluation methods either rely on post-hoc semantic judges, measure unidirectional token confidence (e.g., perplexi...

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
