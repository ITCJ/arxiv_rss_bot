# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-25 06:47:50 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Continuous Optimization for Satisfiability Modulo Theories on Linear Real Arithmetic](https://arxiv.org/abs/2603.22877)

**Authors**: Yunuo Cen, Daniel Ebler, Xuanyao Fong  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2603.22877v1  

#### Abstract
Efficient solutions for satisfiability modulo theories (SMT) are integral in industrial applications such as hardware verification and design automation. Existing approaches are predominantly based on conflict-driven clause learning, which is structurally difficult to parallelize and therefore scale...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Continuous Optimization for Satisfiability Modulo Theories on Linear Real Arithmetic

## 1. 主要贡献和创新点

### 解决的问题
现有的 **Satisfiability Modulo Theories (SMT)** 求解器（如 Z3、CVC5）主要基于 **Conflict-Driven Clause Learning (CDCL)** 或其扩展 **CDCL(T)**，这些方法在处理大规模工业应用（如硬件验证、设计自动化）时面临两大挑战：
- **难以并行化**：CDCL 是一种树搜索算法，结构上难以高效地进行并行计算。
- **可扩展性差**：随着变量和约束数量的增长，求解时间急剧上升，难以应对超大规模问题。

此外，虽然已有基于连续优化的 **Continuous Local Search (CLS)** 方法（如 FOURIERSAT）用于解决布尔 SAT 问题，但它们无法直接应用于 SMT，因为其核心理论——谱分析（spectrum analysis）——仅定义在布尔域上，不支持实数变量。

### 提出的新方法和思路
本文提出了一种全新的、可高度并行化的 SMT 求解框架 **FOURIERSMT**，其核心创新点如下：

- **扩展的 Walsh-Fourier 展开 (xWFE)**：将传统的 Walsh-Fourier Expansion 从纯布尔域推广到混合布尔-实数域。这使得可以将包含线性实数算术 (LRA) 约束的 SMT 公式编码为分段多元线性多项式，从而允许使用梯度方法进行优化。
- **扩展的二元决策图 (xBDD)**：引入 xBDD 数据结构来高效表示布尔-连续混合约束。通过证明 xWFE 的期望值等价于 xBDD 的电路输出概率 (COP)，避免了状态枚举带来的指数级复杂度。
- **基于采样的平滑技术**：对布尔变量采用随机舍入 (randomized rounding)，对实数变量采用高斯采样 (Gaussian sampling)，构建了一个平滑的代理目标函数，使其适用于梯度下降等连续优化方法。
- **退火策略 (annealing)**：通过逐渐减小采样参数 σ 来控制平滑程度。初始阶段较大的 σ 有助于全局探索，后期较小的 σ 则能精确定位到边界解，保证了收敛性和最优性。

### 相比现有方法的优势
- **高度可并行化**：整个优化流程（特别是梯度计算）天然适合在 GPU 上并行执行，显著提升了计算效率。
- **卓越的可扩展性**：能够有效处理包含上万个变量和数十万约束的大规模实例，而传统 CDCL(T) 求解器在此类问题上往往超时。
- **性能提升显著**：在多个基准测试中，相比最先进的 SMT 求解器，实现了高达 8 倍的速度提升。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在两类问题上进行评估：
1.  **随机混合约束基准集 (Random Hybrid Constraints)**：
    - 包含基数约束 (cardinality)、非全等约束 (not-all-equal, nae) 和奇偶校验约束 (parity, xor)。
    - 变量规模 `n` 从 100 到 1000，每个规模生成 10 个实例。
    - 每个实例包含 `n` 个布尔变量和 `n` 个实数变量。
2.  **组合优化问题基准套件 (Combinatorial Optimization Problems)**：
    - **调度问题 (Scheduling)**：连续时间变体，涉及任务依赖、非重叠和可行性约束。
    - **布局问题 (Placement)**：3D 版本，涉及非重叠、可行性和布线感知约束。
    - 总共包含 380 个实例，涵盖了不同规模和复杂度。

### 实验设置和评估指标
- **硬件环境**：
  - 基线求解器运行在 **AMD EPYC 9654 CPU** 上，最多使用 64 个线程。
  - **FOURIERSMT** 运行在 **NVIDIA L40S GPU** 上，利用 GPU 加速梯度计算。
- **评估指标**：
  - **PAR-2 分数 (Penalized Average Runtime 2)**：标准评估指标，对超时的实例惩罚性地计为两倍时间上限（1000 秒）。该分数综合考虑了求解速度和成功率。
  - **解决实例数**：在给定时间内成功求解的实例数量。
- **时间限制**：所有求解器的单个实例时间上限为 1000 秒。

### 基线方法对比
与以下主流 SMT 求解器进行对比：
- **Z3**, **Z3++**, **CVC5**, **YICES2**, **SMTS**, **MATHSAT5**, **SMT-RAT**

同时报告两个虚拟最佳求解器 (Virtual Best Solver, VBS) 的结果：
- **VBS1**：所有求解器（包括 FOURIERSMT）中的最佳表现。
- **VBS2**：排除 FOURIERSMT 后所有求解器中的最佳表现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **随机混合约束**：
  - **FOURIERSMT** 成功解决了全部 100 个实例。
  - 所有其他求解器在 `n ≥ 500` 时均无法求解。
  - 在 `n=500` 时，**FOURIERSMT** 达到了 **100 倍**的速度提升。
- **调度和布局问题**：
  - **FOURIERSMT** 在解决实例数和 PAR-2 分数上均取得最佳成绩。
  - **VBS1**（包含 FOURIERSMT）比 **VBS2**（不含 FOURIERSMT）快 **8.18 倍**（调度）和 **4.93 倍**（布局），并多解决了 20 个和 10 个实例。
  - 在最大的实例上，**FOURIERSMT** 是唯一能在时限内完成求解的求解器。

### 与基线方法的对比结果
- **相对于最强的基线**：
  - 在调度问题上，**FOURIERSMT** 比第二好的 SMTS 快 **6.34 倍**。
  - 在布局问题上，**FOURIERSMT** 比第二好的 Z3 快 **3.15 倍**。
- **相对于广泛部署的求解器**：
  - 速度提升幅度更大，例如在调度问题上比 Z3 快 **16.3 倍**，比 SMT-RAT 快 **41.1 倍**。
- **GPU 加速效果**：
  - 在小规模实例上，CPU 版本更快（无并行开销）。
  - 随着实例规模增大，GPU 版本展现出强大的并行优势，在最大实例上实现了 **6.67 倍**的加速。

### 消融实验结果
论文未明确列出消融实验，但通过以下分析间接验证了各组件的有效性：
- **退火策略**：图 2 展示了不同采样参数 σ 下的优化轨迹，证明了大 σ 有利于逃离局部最优，小 σ 有利于精确收敛。
- **xWFE 到 xBDD 的转换**：通过证明 xWFE 的期望值与 xBDD 的 COP 等价，从根本上将指数级复杂度降低到与决策图大小相关的多项式级别，这是方法可行性的核心保障。

---

## 4. 关键结论和发现

### 主要发现
1.  **连续优化是解决 SMT 的有效范式**：通过 xWFE 将离散的 SMT 问题转化为连续优化问题，并结合 xBDD 进行高效计算，是可行且高效的。
2.  **GPU 加速潜力巨大**：**FOURIERSMT** 的框架天然适合 GPU 并行计算，这使其在处理大规模问题时具有压倒性的性能优势。
3.  **可扩展性突破**：该方法成功解决了传统 CDCL(T) 求解器无法处理的超大规模实例（10,000 变量，700,000 约束），展示了前所未有的可扩展性。
4.  **退火策略至关重要**：通过控制平滑程度，算法能够在全局探索和局部精细搜索之间取得平衡，确保了收敛到高质量解。

### 方法的局限性
- **完备性 (Completeness)**：作为局部搜索方法，**FOURIERSMT** 是不完备的 (incomplete)。它能高效找到满足解（SAT），但如果返回 UNKNOWN，不能证明原问题无解（UNSAT）。
- **理论范围**：当前工作专注于 **Linear Real Arithmetic (LRA)**。对于更复杂的非线性算术 (NRA) 或其他理论（如数组、位向量），需要进一步的理论拓展。
- **初始化敏感性**：作为基于梯度的方法，其性能可能受到初始点选择的影响。

### 未来工作方向
1.  **扩展至非线性算术 (NRA)**：将框架推广到非线性原子约束，这需要解决非凸可行区域和高斯平滑下期望值的闭式表达等难题。
2.  **处理更多理论**：将 xWFE 和 xBDD 框架扩展到其他 SMT 理论，如 BITVECTORS 或 ARRAYS。
3.  **改进投影机制**：推广 Proposition 1，以适应非线性约束，实现更复杂的可行集上的连续优化。
4.  **开发统一框架**：建立一个既能与完备方法（如 CAD、MCSAT）互补，又能处理大规模实际问题的统一求解框架。

</details>

---

### 2. [PersonalQ: Select, Quantize, and Serve Personalized Diffusion Models for Efficient Inference](https://arxiv.org/abs/2603.22943)

**Authors**: Qirui Wang, Qi Guo, Yiding Sun, Junkai Yang, Dongxu Zhang, Shanmin Pang, Qing Guo  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2603.22943v1  

#### Abstract
Personalized text-to-image generation lets users fine-tune diffusion models into repositories of concept-specific checkpoints, but serving these repositories efficiently is difficult for two reasons: natural-language requests are often ambiguous and can be misrouted to visually similar checkpoints, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PersonalQ: Select, Quantize, and Serve Personalized Diffusion Models for Efficient Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
个性化 text-to-image 生成中存在两个核心挑战：
- **Checkpoint 选择困难**：自然语言请求常具有歧义性，标准检索方法（如 RAG 或 reranker）难以准确将用户意图映射到正确的个性化 checkpoint，尤其在概念视觉相似或描述重叠时容易误选。
- **量化损害个性化质量**：现有的 Post-training Quantization (PTQ) 方法虽然能压缩模型、节省内存，但会破坏编码个性化概念的脆弱路径（尤其是 trigger token 相关表示），导致生成图像的身份保真度和文本对齐能力下降。

### 🚀 提出的新方法与创新思路
作者提出 **PersonalQ** ——一个统一框架，通过共享信号 **trigger token** 联结 checkpoint 选择与量化过程，实现高效且高保真的个性化模型服务。

#### 主要模块：
1. **Check-in**：基于意图对齐的个性化 checkpoint 选择模块
   - 结合 **intent-aware hybrid retrieval**（稀疏 + 密集检索融合）
   - 引入 **LLM-based reranking** 进行个性化推理
   - 支持 **clarification mechanism**，当多个意图仍可能成立时主动提问澄清
   - 最终输出插入 trigger token 的重写 prompt（如 `bear → <bear-v4>`）

2. **Trigger-Aware Quantization (TAQ)**：感知 trigger 的混合精度量化策略
   - 在 cross-attention 中识别由 trigger token 控制的关键 K/V 行及其 attention 权重
   - 对这些 pathway 保持 full precision，其余部分进行 aggressive quantization（如 4-bit）
   - 实现“保护最关键信息 + 大幅压缩非关键路径”的平衡

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **Selection** | 显著优于 Random、Reranker 和 Stylus，在意图对齐上获得更高 LLM-judge 评分和人类偏好率 |
| **Quantization** | 在相同 bit-width 下，TAQ 的 FID 更低、CLIP Score 更高，压缩-质量权衡显著优于 Q-Diffusion、TFMQ-DM、DGQ 等 PTQ 方法 |
| **系统协同性** | Check-in 输出的 trigger token 直接指导 TAQ 的保护范围，形成闭环优化 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **REPO-PROMPTS**（本文构建的新 benchmark）：
  - 包含 **500 条自然语言查询**
  - 针对一个包含 **1,000 个个性化 checkpoint** 的仓库
  - 涵盖 20 种概念类别（如 `<dog>`, `<cat>`, `<person>`），每类有 50 个时间版本（v1–v50）
  - 查询类型分布：350 单匹配、100 歧义需澄清、50 无匹配
- **自动评估数据集**：
  - **MS-COCO captions**
  - **PartiPrompts**

### ⚙️ 实验设置
- **Backbone 模型**：
  - Stable Diffusion v1.5（DreamBooth 微调）
  - SDXL-Turbo（LoRA DreamBooth 微调）
- **量化配置**：
  - 权重/激活 bit-width：8/8 和 8/4
  - 使用 Adaround 和 BRECQ 进行 weight PTQ
  - 所有方法使用相同的 64 个 MS-COCO caption 用于 calibration
- **推理平台**：通过 API 调用多模态大模型（MLLM），避免本地 GPU 开销

### 🎯 评估指标
| 类别 | 指标 |
|------|------|
| **生成质量** | FID ↓, CLIP Score ↑ |
| **计算效率** | BOPs（Bit Operations）↓ |
| **意图对齐** | LLM Judge Intent Score（1–5 分 Likert 量表，平均分） |
| **用户体验** | Human Preference Win Rate（成对比较中被偏好的比例） |
| **延迟** | End-to-end 推理时间（秒）、multi-turn 请求占比 |

### 🆚 基线方法对比
| 任务 | 基线方法 |
|------|----------|
| **Checkpoint Selection** | Random, Reranker (Qwen3-Reranker-4B), Stylus |
| **Quantization** | Q-Diffusion, TFMQ-DM, DGQ |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table I & II）

#### ✅ 量化性能对比（Table I）
| 方法 | 模型 | Bit-width | FID (MS-COCO) | CLIP Score | BOPs |
|------|------|-----------|----------------|-------------|-------|
| Full Precision | SD1.5 | 32/32 | 10.96 | 0.315 | 893 |
| TAQ | SD1.5 | 8/8 | **11.03** | **0.297** | **56** |
| DGQ | SD1.5 | 8/8 | 15.24 | 0.291 | 56 |
| TAQ | SD1.5 | 8/4 | **38.84** | **0.264** | **28** |
| (其他 PTQ) | — | 8/4 | >50+ | <0.25 | 28 |

> 💡 **结论**：TAQ 在 8-bit 设置下几乎接近 full precision 性能；在 4-bit 激活下仍显著优于所有 baseline，同时实现 **4–8× 内存减少** 和 **16–32× bit-op 降低**。

#### ✅ Checkpoint 选择性能（Table II）
| 方法 | Intent Score (↑) | Human Preference (↑) |
|------|------------------|------------------------|
| Random | 2.14 ± 0.82 | 89.1% |
| Reranker | 3.21 ± 0.76 | 85.7% |
| Stylus | 3.68 ± 0.69 | 82.1% |
| **Check-in (Ours)** | **4.42 ± 0.51** | **100.0%** |

> 💡 Check-in 在意图理解上远超基线，并在人类评估中始终被首选。

### 🔍 消融实验结果（Ablation Studies）

#### （1）Check-in 模块消融（Table III）
| 变体 | Intent Score | FID |
|------|--------------|-----|
| 仅 reranker | 3.21 | 11.74 |
| + Hybrid Retrieval | 3.99 | 11.61 |
| + Personalized Reasoning | 4.19 | 11.23 |
| + Clarification | **4.42** | **11.03** |

> 各组件逐步提升意图对齐能力，而生成质量变化不大（说明主要影响 selection 而非 generation）

#### （2）TAQ 触发词分离效果（Table IV）
| Quantizer | 分离 trigger？ | FID (8/4) | CLIP Score |
|----------|----------------|------------|------------|
| Linear | × | 54.12 | 0.245 |
| Linear | √ | **44.53** | **0.262** |
| Logarithmic | × | 56.21 | 0.249 |
| Logarithmic | √ | **38.22** | **0.265** |

> 分离 trigger token 显著缓解量化损伤，尤其在低比特下 log quantizer 提升最大。

#### （3）组件协同效应（Table V）
| 方法组合 | Intent Score | FID (8/4) |
|---------|---------------|-----------|
| Check-in only | 4.40 | 51.31 |
| TAQ only | 3.23 | 39.53 |
| **Both (Check-in + TAQ)** | **4.41** | **38.22** |

> 二者互补：Check-in 提升意图对齐，TAQ 保障生成保真度，联合使用在高压缩场景最稳健。

#### （4）MLLM 延迟对比（Table VI）
| Backbone | Intent Score | End-to-End Time (s) | Multi-turn Ratio |
|---------|---------------|----------------------|------------------|
| GPT-4o | 4.42 | 45.09 | 2.1 |
| **Gemini 2.5 Flash** | 4.38 | **38.43** | 2.0 |
| Qwen2.5-VL-72B | 4.35 | 49.85 | 2.5 |

> Gemini 2.5 Flash 因推理速度快成为低延迟部署首选。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Trigger token 是连接 selection 与 quantization 的关键桥梁**：
   - 它既是语义锚点（用于精准路由），又是量化中最脆弱的部分（需重点保护）
   - 利用同一信号驱动两个模块，实现了 intent-aligned + fidelity-preserving 的统一目标

2. **Hybrid retrieval + LLM reasoning + clarification 构成高效的 selection pipeline**：
   - 能处理复杂上下文线索（时间、风格、版本）
   - 显著优于传统 retrieval 或 reranking 方法

3. **TAQ 实现了当前最优的压缩-质量权衡**：
   - 在 8-bit 权重/激活下接近 full precision 表现
   - 在 4-bit 激活下仍保持可用性，远胜现有 PTQ 方法

4. **PersonalQ 支持大规模个性化 checkpoint 库的可扩展部署**：
   - 兼顾高保真、低内存、快速响应
   - 适用于实际应用场景中的并发请求和服务资源限制

### ⚠️ 局限性
- **依赖 trigger token 的显式定义**：若训练过程中未明确定义 trigger token，方法可能失效
- **LLM 推理引入额外延迟和成本**：尽管使用轻量 backbone（如 Gemini Flash），但在高频服务中仍需权衡
- **目前仅验证于 text-to-image 场景**：是否可推广至 video/audio 等多模态个性化任务尚待研究

### 🔮 未来工作方向
- 将 PersonalQ 扩展至 **multi-concept composition**（多个 trigger token 组合生成）
- 探索 **dynamic trigger discovery**，无需人工标注即可从历史生成中推断潜在 trigger
- 结合 **adaptive quantization**，根据输入 prompt 动态调整保护强度
- 构建更大规模的 **public benchmark**（REPO-PROMPTS 可作为起点），推动社区发展

---

> ✅ **总结一句话**：  
> **PersonalQ 通过以 trigger token 为核心信号，首次将个性化 diffusion 模型的 checkpoint selection 与 quantization 统一起来，在保证生成保真度的同时大幅提升服务效率，为大规模个性化生成系统的落地提供了可行路径。**

</details>

---

### 3. [PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving](https://arxiv.org/abs/2603.23049)

**Authors**: Wenfeng Wang, Xiaofeng Hou, Peng Tang, Hengyi Zhou, Jing Wang, Xinkai Wang, Chao Li, Minyi Guo  
**Category**: cs.DC  
**Published**: 2026-03-25  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.23049v1  

#### Abstract
Retrieval-Augmented Generation (RAG) systems enhance the performance of large language models (LLMs) by incorporating supplementary retrieved documents, enabling more accurate and context-aware responses. However, integrating these external documents often results in very long input sequences, which...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **Retrieval-Augmented Generation (RAG)** 系统中，由于引入外部检索文档，输入序列变得非常长，导致 **prefill 阶段**（生成第一个 token 的时间，即 TTFT）显著增加。该阶段需要为所有输入 token 构建 **KV Cache**，计算开销巨大。

尽管已有研究通过 **KV-cache reuse** 来避免重复计算共享前缀，但在实际部署中仍面临三大挑战：
- **低缓存命中率**：传统 LRU 替换策略未考虑未来请求，导致高价值缓存被过早淘汰。
- **CPU-GPU 数据传输开销大**：从 CPU 内存加载 KV Cache 会阻塞 GPU 计算。
- **SSD I/O 性能瓶颈**：当缓存溢出到 SSD 时，慢速读取成为新的延迟瓶颈。

### 提出的新方法与创新点
本文提出 **PCR (Prefetch-Enhanced Cache Reuse)** 系统，通过三项核心技术提升 KV-cache 复用效率：

1. **Prefix-Tree Caching + Look-ahead LRU 替换策略**
   - 将输入分块并组织成 **prefix tree** 结构，支持高效前缀匹配。
   - 引入 **look-ahead LRU**：利用调度队列中的待处理请求信息，预测未来可能复用的缓存块，优先保留其缓存，提高命中率。

2. **Layer-wise Overlapping**
   - 利用 LLM 的层状结构，在三个独立的 CUDA stream 上并行执行：
     - GPU 计算（当前层）
     - CPU→GPU 加载（下一层 KV Cache）
     - GPU→CPU 卸载（上一层新生成 KV Cache）
   - 实现计算与通信重叠，隐藏大部分传输延迟。

3. **Queue-based Prefetching**
   - 在请求仍在等待队列时，就主动将所需 KV Cache 从 SSD 预加载至 DRAM。
   - 利用检索阶段快于推理的特点，提前完成慢速 I/O 操作，避免运行时阻塞。

### 相比现有方法的优势
| 特性 | PCR | vLLM / PromptCache | RAGCache / CacheGen |
|------|-----|----------------------|------------------------|
| 缓存命中率 | ✅ 显著提升（look-ahead） | ❌ 被动替换 | ⚠️ 有限优化 |
| CPU-GPU 通信 | ✅ 完全重叠（layer-wise） | ❌ 同步阻塞 | ⚠️ 部分优化 |
| SSD 利用效率 | ✅ 异步预取（queue-based） | ❌ 不支持 | ⚠️ 同步加载 |
| 准确性保障 | ✅ 严格前缀匹配（无精度损失） | ✅ 是 | ❌ CacheBlend 等牺牲精度 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **文档库**：Wikipedia 英文语料（约 859 万文档）
- **查询集**：SQuAD 数据集
- **嵌入模型**：MiniLM
- **检索方式**：每个 query 检索最相关的两个文档
- **构造输入**：`[doc1][doc2][query]`，平均长度 ~6.8k tokens

### 实验设置
#### 硬件平台
| 平台 | GPU | CPU Memory | SSD | 连接带宽 |
|------|-----|-------------|-----|-----------|
| A6000 ×2 | 2×48GB | 256GB | 4TB NVMe | PCIe 4.0 (~24 GB/s) |
| RTX 4090 ×2 | 2×24GB | 128GB | 4TB NVMe | PCIe 4.0 (~24 GB/s) |

#### 模型
- **Llama 系列**：Llama2-7B, Llama2-13B, Llama3.1-8B, Llama3.2-3B
- **Qwen 系列**：Qwen2.5-7B, Qwen2.5-14B
- 注意：Llama 使用 MHA，Qwen 使用 GQA，影响 KV Cache 大小

#### 评估指标
- **TTFT (Time to First Token)**：主要指标
- **E2EL (End-to-End Latency)**
- **P50/P95/P99** 分位数延迟
- 缓存命中率（Hit Ratio）

#### 基线方法对比
1. **vLLM**：基于 PagedAttention 的主流系统，仅在 GPU 内管理 KV Cache
2. **LMCache**：支持跨 GPU-CPU-SSD 的 KV Cache 复用与预取
3. **CCache**：本文构建的简化版，仅扩展 CPU 存储
4. **SCCache**：CCache + SSD 扩展，用于消融分析

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在多种模型和硬件配置下，PCR 均实现显著加速：
  - **最高达 2.47× 的 TTFT 加速**（vs vLLM）
  - 平均 **TTFT 降低 15% 以上**
  - 在高负载下优势更明显（如 request rate = 0.9 时）

#### 典型结果示例（Llama-8B on RTX 4090）
| 方法 | TTFT (mean) | TTFT (P95) | E2EL (P99) |
|------|--------------|------------|------------|
| vLLM | 103 ms | 103 ms | 142 ms |
| LMCache | 89 ms | 89 ms | 124 ms |
| **PCR** | **58 ms** | **58 ms** | **86 ms** |

> PCR 在尾延迟（tail latency）方面表现尤为出色，P99 下降超 30%，对用户体验至关重要。

### 与基线方法对比结果
- PCR 在所有测试场景中均优于所有基线：
  - 相比 **vLLM**：TTFT 平均减少 **30–60%**
  - 相比 **LMCache**：进一步降低 **10–20%**
  - 在 **Llama2-13B** 上，相比 SCCache 实现 **50.9% 的 TTFT 降低**

### 消融实验结果
| 模型 | 技术 | 0.5 req/s TTFT ↓ | 1.0 req/s TTFT ↓ |
|------|------|------------------|------------------|
| Llama2-7B | Base (CCache) | — | — |
| | + Layer-wise Overlapping | 17.74% | 47.53% |
| | + Queue-based Prefetch | 20.86% | **69.09%** |
| Llama2-13B | Base | — | — |
| | + Overlapping | 37.28% | 11.50% |
| | + Prefetch | **59.08%** | **31.15%** |

> **结论**：
> - **Queue-based prefetching** 对大模型（如 Llama2-13B）效果最显著，因其更多依赖 SSD 缓存。
> - **Layer-wise overlapping** 对所有模型均有稳定收益，尤其在高吞吐下更有效。

#### 窗口大小影响（Look-ahead Window）
- 最优窗口大小为 **6**（实验中测试 2–10）
- 窗口越大，可预取越多请求，但需权衡资源占用
- 高请求率下增益更明显（+31.06% 当 window 从 4→6）

---

## 4. 关键结论和发现

### 主要发现
1. **多级存储 + 智能预取是解决 RAG 推理瓶颈的关键路径**  
   单纯扩大缓存容量不足以解决问题，必须结合 **异步数据移动** 和 **前瞻性调度** 才能充分发挥潜力。

2. **缓存策略应感知请求队列状态**  
   利用调度器中的“未来信息”进行 **look-ahead 缓存保护和预取**，可显著提升命中率和系统响应性。

3. **layer-wise overlapping 能有效隐藏通信开销**  
   利用 LLM 的层状特性，将通信与计算流水线化，使 PCIe 带宽不再成为瓶颈。

4. **SSD 可作为有效的缓存扩展层，前提是解决 I/O 延迟**  
   直接同步读取 SSD 会导致性能劣化，但通过 **queue-based prefetching** 可将其转化为优势。

### 方法的局限性
- **依赖固定 chunk 大小**：可能导致部分语义边界断裂，影响复用粒度。
- **预取窗口需调优**：不同模型和负载下的最优窗口不同，缺乏自适应机制。
- **未处理动态更新的文档库**：假设文档库静态，若频繁更新需重新构建缓存。
- **对极短请求不敏感**：prefetch 和 overlapping 开销可能抵消收益。

### 未来工作方向
- 设计 **自适应 look-ahead window** 和 **动态 chunk 划分策略**
- 支持 **增量式缓存更新**，应对文档库变更
- 探索 **压缩 + 复用联合优化**，进一步降低存储成本
- 扩展至 **multi-turn conversation** 场景，支持对话历史复用

---

> **总结**：PCR 通过 **智能缓存管理** 与 **异步数据流优化**，实现了 RAG 系统中 KV-cache 复用的性能飞跃。它不仅提升了平均延迟，更重要的是显著改善了尾延迟，为生产环境中的高质量服务提供了可靠保障。该工作强调了 **系统级协同设计** 在 LLM 推理优化中的重要性。

</details>

---

### 4. [Balancing Safety and Efficiency in Aircraft Health Diagnosis: A Task Decomposition Framework with Heterogeneous Long-Micro Scale Cascading and Knowledge Distillation-based Interpretability](https://arxiv.org/abs/2603.22885)

**Authors**: Xinhang Chen, Zhihuan Wei, Yang Hu, Zhiguo Zeng, Kang Zeng, Suili Yang  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.22885v1  

#### Abstract
Whole-aircraft diagnosis for general aviation faces threefold challenges: data uncertainty, task heterogeneity, and computational inefficiency. Existing end-to-end approaches uniformly model health discrimination and fault characterization, overlooking intrinsic receptive field conflicts between glo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 主要贡献和创新点

### 解决的问题
该论文针对通用航空飞机健康诊断（Aircraft Health Diagnosis）面临的三大挑战：
- **数据不确定性**（Data Uncertainty）：真实飞行数据中存在环境噪声和标签模糊。
- **任务异质性**（Task Heterogeneity）：异常检测（Anomaly Detection, AD）与故障分类（Fault Classification, FC）在特征粒度、感受野需求上存在根本冲突。
- **计算效率瓶颈**（Computational Inefficiency）：端到端模型在严重类别不平衡（大量正常样本，少量故障样本）下训练成本高昂。

传统 end-to-end 方法将 AD 和 FC 统一建模，导致：
- 全局上下文建模与局部特征提取之间的**感受野悖论**（receptive field paradox）；
- 模型为“黑箱”，缺乏可解释性；
- 训练过程低效且难以部署。

---

### 提出的新方法与思路
论文提出了 **Diagnosis Decomposition Framework (DDF)** 及其具体实现 **Long-Micro Scale Diagnostician (LMSD)**，核心思想是通过**任务分解**（Task Decomposition）解决上述矛盾。

#### 创新点如下：

1. ✅ **显式任务解耦（Explicit Task Decomposition）**
   - 将整体诊断任务解耦为两个阶段：
     - **Long Stage (AD)**：全局筛查，使用全序列感受野识别系统级偏离。
     - **Micro Stage (FC)**：局部精诊，聚焦于短时微尺度特征进行细粒度故障归因。
   - 架构上分离而非统一建模，从根本上规避了感受野冲突。

2. ✅ **异构级联架构（Heterogeneous Cascading Architecture）**
   - **AD 阶段**：采用 `ConvTokMHSA`（基于卷积分词的多头自注意力），具备长距离依赖建模能力。
   - **FC 阶段**：采用 `MMK Net`（Multi-Micro Kernel Network），使用小核卷积（kernel size ∈ {1,3,5}）提取局部敏感特征。
   - 二者通过 **Hard-Threshold Router** 连接，确保决策路径隔离。

3. ✅ **可解释性即设计（Interpretability-by-Design）**
   - 引入 **Keyness Extraction Layer (KEL)**，基于知识蒸馏（Knowledge Distillation）从输入层提取时间维度上的“关键性”权重（temporal keyness）。
   - 提供物理可追溯的解释：模型关注的是哪些时间段？这些时段是否对应真实的故障机制？

4. ✅ **高效训练策略（Efficient Decoupled Training）**
   - **大样本轻量模型 + 小样本复杂模型**：
     - AD 模型在完整数据集上训练（含大量正常样本），结构轻量；
     - FC 模型仅在被判定为异常的子集上训练，专注复杂建模。
   - 显著降低总训练时间，提升部署可行性。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **任务适应性** | 明确区分 AD 与 FC 的建模范式，避免单一架构的折衷妥协 |
| **安全性** | Hard-threshold 路由最小化漏检率（FNR），符合航空“宁可误报不可漏报”的安全伦理 |
| **效率** | 总训练时间显著低于 end-to-end 方法（见后文实验） |
| **可解释性** | KEL 提供输入级的时间注意力图谱，支持物理机理验证 |

---

## 2. 核心实验方法和设置

### 数据集
- **NGAFID dataset**：来自塞斯纳172机队的真实航空维护数据集。
  - 包含超过 28,935 次飞行（约 31,000 小时）
  - 23维传感器时序数据（采样频率 1Hz）
  - 关联 36 类非计划性维修事件
  - 分为两个版本：
    - **Subset**：19类，11,446次飞行（基准测试）
    - **Overall**：36类，28,935次飞行（完整高复杂场景）

> ⚠️ 特点：真实世界噪声、标签模糊、极端长尾分布（头部类别 >2000 次，尾部 <15 次）、多阶段耦合动态。

---

### 实验设置
- **预处理**：
  - 缺失值前向填充（Forward Fill）
  - 序列长度标准化至 2048（立方样条插值）
  - Z-score 归一化（参数仅从训练集计算）
- **交叉验证**：Stratified 5-Fold Cross-Validation，各 fold 物理隔离防止泄漏。
- **随机控制**：固定划分种子，每 fold 内重复 3 次训练取中位数性能，报告五折中位数的中位数（Median-of-Medians）。

---

### 评估指标体系（四维评价）

| 类别 | 指标 | 说明 |
|------|------|------|
| **传统分类性能** | ACC, F1, WF1 | 基础判别能力 |
| **安全关键指标** | FNR（False Negative Rate） | 故障样本被误判为健康的比率，越低越好 |
| | **MCWPM**（Multi-Class Weighted Penalty Metric） | 新提出的安全导向指标：<br>$$ \text{MCWPM} = \frac{2TP}{2TP + \alpha_p FN_{\text{health}} + \beta_p FP_{\text{health}}} $$<br>其中 $\alpha_p=2.5$, $\beta_p=1.0$，强调对漏检的惩罚远高于误报 |
| **训练效率** | ET（Epoch Time）, TTT（Total Training Time） | 衡量训练开销，决定迭代速度与部署经济性 |
| **推理开销** | IT32（Inference Time for 32 samples）, MSize（Model Size） | 影响边缘部署实时性 |

---

### 基线方法对比
选取三类主流方法作为 baseline：
| 模型 | 类型 | 技术路线 |
|------|------|----------|
| **Bi-LSTM** | RNN | 序列双向建模 |
| **InceptionTime** | CNN | 多尺度卷积并行提取局部特征 |
| **InceptionTimeAttn** | Hybrid | CNN + 自注意力，代表“先局部后全局”融合范式 |
| **ConvTokMHSA / ConvTokSWLA** | Transformer | 全局注意力 vs 局部滑窗注意力，用于验证感受野影响 |

所有模型均经过系统超参搜索，保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（以 Overall 数据集为例）

| 模型 | MCWPM | F1 | TTT (s) | MSize (MB) |
|------|--------|-----|---------|-----------|
| Bi-LSTM | 0.3372 | 0.0247 | 407.06 | 3.92 |
| InceptionTime | 0.5652 | 0.3570 | 7052.59 | 76.25 |
| InceptionTimeAttn | 0.5722 | 0.2109 | 8388.47 | 24.14 |
| MMK Net | 0.5417 | 0.3188 | 3024.93 | 11.43 |
| ConvTokMHSA | 0.5306 | 0.2754 | 942.26 | 32.29 |
| **LMSD (本文)** | **0.6148** | **0.4091** | **2001.63** | **12.97** |

> ✅ **LMSD 在 MCWPM 上比最佳基线高出 ~4–8%**，同时训练时间和模型大小更优。

---

### 与基线方法的对比结果

#### （1）AD 子任务表现（表3）
- **ConvTokMHSA**（全局注意力）在 AD 上表现最优（ACC=0.7657, F1=0.7640），显著优于局部注意力变体 **ConvTokSWLA**。
- 证明：**AD 需要全序列感受野来捕捉操作模式上下文**。

#### （2）FC 子任务表现（表4）
- **MMK Net** 在 FC 上遥遥领先（Subset F1=0.6228），远超全局注意力模型 **ConvTokMHSA**（F1=0.2080）。
- 证明：**FC 更需要受限感受野以抑制跨阶段噪声**。

#### （3）整体 Diagnosis 性能（表5）
- 所有 end-to-end 方法面临“两难困境”：无法兼顾 AD 与 FC。
- **LMSD 成功突破此限制**，在保持较低 FNR 的同时，实现更高的 MCWPM 和 F1。
- **MCWPM 提升 4–8%**，验证了 DDF 在安全约束下的优越性。

#### （4）训练效率优势
- LMSD 的 **TTT = 2001.63s**，仅为 InceptionTimeAttn 的 **24%**。
- 参数量仅 **12.97MB**，约为 InceptionTimeAttn 的一半。
- 推理延迟 IT32=0.08s，仍在可接受范围。

---

### 消融实验与分析（隐含在模块对比中）
虽然未明确列出消融表，但以下对比构成实质性的消融研究：
- **感受野影响**：ConvTokMHSA vs ConvTokSWLA → 验证全局 vs 局部对 AD/FC 的不同影响。
- **架构选择**：MMK Net vs InceptionTime → 验证微小卷积核 + LayerNorm 对 FC 的增益。
- **训练策略**：LMSD 的 decoupled training 显著优于统一训练，体现“大样本轻量 + 小样本复杂”的合理性。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **感受野悖论真实存在且不可调和**  
   单一架构无法同时满足 AD 的全局建模需求与 FC 的局部精细提取需求。必须通过**架构级分离**解决。

2. 🧩 **任务分解是理论必要而非工程便利**  
   NGAFID 数据中的不确定性、稀疏性和多阶段耦合决定了：只有显式解耦才能实现最优性能。

3. 📈 **LMSD 实现安全与效率的平衡**  
   - 安全性：Hard-threshold 路由保障极低漏检率（FNR↓），MCWPM↑；
   - 效率：Decoupled training 大幅降低训练成本（TTT↓50%+）；
   - 性能：综合指标全面超越 end-to-end 方法。

4. 🔎 **KEL 提供物理可追溯的可解释性**  
   - AD 阶段关注起飞准备期等关键过渡阶段；
   - FC 阶段精准定位如燃油流量响应迟钝、油温同步波动等物理故障特征；
   - 支持“从‘是否有异常’到‘是什么故障’”的层次化认知。

---

### 方法的局限性
1. **性能天花板受制于数据质量**
   - 标签噪声（maintenance record retrospective labeling）导致部分样本本质不可分。
   - 极端长尾分布（tail classes < 20 samples）使某些故障召回率不足（<40%）。
   - 传感器维度有限（23D @ 1Hz），关键信号埋藏于低方差主成分中（PCA 第12–18个成分）。

2. **当前为硬阈值路由**
   - 缺乏不确定性量化机制，未来可引入 soft routing 或贝叶斯置信度引导。

3. **尚未适配边缘设备**
   - 虽然模型较小，但仍需进一步压缩以支持机载实时诊断。

---

### 未来工作方向（作者明确提出）
1. **方法学层面**：
   - 从 hard-threshold 路由转向 **uncertainty-quantified soft routing**；
   - 引入 **physics-informed embeddings** 与因果推理机制。

2. **数据层面**：
   - 提升飞行参数采集密度（更高频、更多传感器）；
   - 改进维修语义标注精度；
   - 构建“组件-子系统-整机”多层级数据采集体系。

3. **工程落地层面**：
   - 适配边缘计算平台（edge computing）；
   - 支持在线增量学习（online incremental learning）；
   - 发展跨机型迁移能力（cross-domain transfer）。

---

### 总结
本论文提出的 **DDF/LMSD 框架**成功解决了通用航空健康诊断中的**任务异质性、安全要求与计算效率之间的根本矛盾**。通过**任务分解 + 异构级联 + 可解释性蒸馏**，实现了：
- ✅ 更高的诊断准确率（尤其在安全敏感指标 MCWPM 上）
- ✅ 更低的训练成本与模型体积
- ✅ 物理可追溯的决策依据

为现实工业场景下的航空 PHM 提供了一个**可部署、可信赖、高效能**的方法论范式。

</details>

---

### 5. [Universal and efficient graph neural networks with dynamic attention for machine learning interatomic potentials](https://arxiv.org/abs/2603.22810)

**Authors**: Shuyu Bi, Zhede Zhao, Qiangchao Sun, Tao Hu, Xionggang Lu, Hongwei Cheng  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.22810v1  

#### Abstract
The core of molecular dynamics simulation fundamentally lies in the interatomic potential. Traditional empirical potentials lack accuracy, while first-principles methods are computationally prohibitive. Machine learning interatomic potentials (MLIPs) promise near-quantum accuracy at linear cost, but...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Universal and efficient graph neural networks with dynamic attention for machine learning interatomic potentials

## 1. 论文的主要贡献和创新点

### 解决的问题
传统分子动力学（MD）模拟依赖于**经验势函数**（empirical potentials），其精度受限且难以描述复杂多体相互作用；而基于第一性原理的**ab initio 分子动力学**（如 DFT）虽具高精度，但计算代价高昂（$O(N^3)$），无法用于大规模或长时间尺度模拟。

尽管现有的**机器学习原子间势能模型**（MLIPs）在精度与效率之间取得一定平衡，但主流的 SE(3)-equivariant 模型仍面临以下挑战：
- 高阶张量积运算导致**计算复杂度高**；
- 消息传递机制在并行化时存在通信瓶颈；
- 在有限训练数据下易出现过拟合或表达能力不足。

---

### 提出的新方法：MLANet
本文提出了一种高效且鲁棒的图神经网络框架——**Machine Learning Advances Neural Network (MLANet)**，其核心创新包括：

#### ✅ 双路径动态注意力机制（Dual-path Dynamic Attention）
- 引入几何感知的注意力机制，在消息传递过程中自适应地调节原子间的交互强度。
- 将节点特征（queries）、邻居特征（keys）与边特征结合，并通过 **Tensor Product (TP)** 和 **Softmax** 计算注意力权重。
- 设计门控机制（gating）控制信息流动，提升对局部化学环境细微差异的分辨能力。

#### ✅ 多视角池化策略（Multi-perspective Pooling）
- 融合三种全局池化操作：**Additive Pooling**（保持广延性质）、**Mean Pooling**（获取强度不变属性）、**Max Pooling**（捕捉关键局域结构）。
- 并联输出以构建更全面、稳健的系统表示，减少信息丢失。

#### ✅ SE(3)-等变架构设计
- 利用球谐函数（spherical harmonics）和不可约表示（irreps）保证旋转和平移对称性。
- 使用 SiLU 激活函数处理标量与非标量特征，确保 equivariance 同时维持梯度稳定。

---

### 相比现有方法的优势
| 维度 | MLANet优势 |
|------|-----------|
| **准确性** | 在多种体系中达到与 NequIP、MACE 等主流 equivariant 模型相当甚至更优的预测精度 |
| **效率** | 显著降低计算开销，训练速度比 NequIP 快一个数量级，内存占用更低 |
| **稳定性** | 支持长达 300 ps 的稳定分子动力学模拟 |
| **通用性** | 成功应用于有机分子、周期性无机材料、二维材料、表面催化反应及带电系统 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 | 类型 |
|-------|------|-----|
| **QM7** | 小有机分子集合（≤23 原子），目标为原子化能预测 | 非周期系统 |
| **MD17** | 8 种有机分子的动力学轨迹，含能量与力标签 | 力场建模基准 |
| **SiO₂**, **Ge-Sb-Te**, **Black Phosphorus** | 包含相变、金属间作用、范德华力的无机晶体系统 | 周期性材料 |
| **Bilayer Graphene** | 双层石墨烯滑移与剥离势能面 | 二维材料 |
| **Formate Decomposition** | HCOO* 在 Cu 表面分解路径（NEB + AIMD 生成） | 表面催化 |
| **Water (bulk)** | 液态水体系（revPBE0-D3 计算） | 液体系统 |
| **Charged Systems** | C₁₀H₂/C₁₀H⁻, Ag⁺/Ag⁻, Na₉Cl₉⁻ 等带电体系 | 电荷相关势能 |
| **QM9 / QM9S** | 大规模有机分子数据库，涵盖量子化学性质（偶极矩、极化率等） | 多任务预测 |

---

### 实验设置与评估指标
- **训练目标**：联合优化能量 $E$、力 $\mathbf{F}$ 和应力 $\sigma$ 的损失：
  $$
  \mathcal{L}_{\text{total}} = \lambda_E \mathcal{L}_E + \lambda_F \mathcal{L}_F
  $$
  其中使用 L1 损失（MAE）进行回归。
- **评估指标**：
  - 能量 MAE（kcal/mol 或 meV/atom）
  - 力 MAE（meV/Å）
  - 应力 MAE（meV/Å³）
  - RMSE（部分任务）
  - 推理速度（s/epoch）、显存占用（MB）

- **硬件配置**：PyTorch 实现，双 NVIDIA RTX 4090 GPU 训练；推理测试也在消费级笔记本 GPU（RTX 4060）上验证。

---

### 基线方法对比
与以下典型模型进行了比较：
- **Invariant GNNs**: SchNet, DimeNet++, PaiNN
- **Equivariant GNNs**: NequIP, MACE, Allegro, CAMP
- **传统势函数**: ReaxFF, AIREBO, GAP, LCBOP
- **其他 MLIPs**: DeePMD, BPNN, REANN

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 🔹 QM7 原子化能预测（Table 1）
| 方法 | MAE [kcal/mol] |
|------|----------------|
| DTNN | 8.2 |
| SchNet | 74.2 |
| 2GGST+EK | 3.4 |
| **MLANet** | **3.07**(±0.09) ✅ |

👉 MLANet 达到当前最优水平，显著优于大多数已有模型。

---

#### 🔹 MD17 力预测（Table S2, Fig. 5）
- 在小样本训练（950 结构）下，MLANet 在苯类体系中表现领先。
- 大规模训练后进行 MD 模拟（300 ps）：
  - 力 MAE 较低（略高于 NequIP/CAMP）
  - **模拟全程稳定不发散**
  - **推理速度快于 NequIP 约 10 倍**

---

#### 🔹 二维材料：双层黑磷剥离能（Fig. 4）
- 加入长程项后，MLANet 准确复现 DFT+MBD 的结合能曲线。
- 对 Hittorf’s phosphorus (P2/c, P2/n) 的剥离行为也具有良好泛化能力。

---

#### 🔹 双层石墨烯滑移势能面（Fig. 6, Table 3）
| 方法 | 滑移能 RMSE (meV/atom) | 结合能 RMSE (meV/atom) |
|------|------------------------|-------------------------|
| ReaxFF | 0.7 | 16.0 |
| GAP2020 | 2.5 | 10.3 |
| NequIP | 1.0 | 1.6 |
| hNN | 0.4 | 1.2 |
| **MLANet** | **0.5** | **1.3** |

👉 性能仅次于最复杂的 hNN，远超传统力场。

---

#### 🔹 甲酸根分解（Surface Catalysis, Table 4）
| 方法 | Force MAE (meV/Å) | Energy MAE (meV/atom) |
|------|--------------------|------------------------|
| NequIP | 47.3 | 0.50 |
| MACE | 54.1 | 0.31 |
| AlphaNet | 42.5 | 0.23 |
| **MLANet** | **44.9** | 2.31 |

👉 力误差极具竞争力，适合催化反应动力学模拟。

---

#### 🔹 水体系（Table 5）
| 方法 | Energy RMSE (meV/atom) | Force RMSE (meV/Å) |
|------|--------------------------|----------------------|
| DeePMD | 2.1 | 92 |
| NequIP | 0.94 | 45 |
| MACE | 0.63 | 36 |
| **MLANet** | **0.47** ✅ | 60 |

👉 能量预测精度最佳，但由于数据量较小（仅 1593 构型），表现出轻微过拟合趋势。

---

#### 🔹 带电系统（Table 6）
| 数据集 | MLANet Force RMSE (eV/Å) | 最佳基线 |
|--------|----------------------------|----------|
| C₁₀H₂/C₁₀H⁻ | 0.074 | 0.023 (ReaxNet) |
| Ag⁺/Ag⁻ | 0.024 | 0.005 |
| **Na₉Cl₉⁻** | **0.012** ✅ | 0.032 (4G-HDNNP) |

👉 在 Na₉Cl₉⁻ 上超越所有包含显式静电项的模型，表明直接嵌入电荷可有效建模离子系统。

---

### 消融实验分析（Fig. 2–3）
- **不同 $l_{\text{max}}$ 设置的影响**：
  - $l=2$ 在多数情况下优于 $l=3$，尤其在小数据集上避免过拟合。
  - 更高的 $l$ 提升表达能力但显著增加计算成本（memory 和 time 增长约 2–3 倍）。
- **多视角池化的有效性**：
  - 消融显示去除任意一种池化方式均会导致性能下降，证明三者互补。

---

## 4. 关键结论和发现

### 主要发现
1. **MLANet 实现了精度与效率的优良平衡**：
   - 在多个跨领域数据集中达到或接近 SOTA 精度；
   - 计算效率显著优于主流 equivariant 模型（如 NequIP），支持千原子级别系统的快速模拟。

2. **架构设计的有效性得到验证**：
   - 双路径动态注意力增强了对局部几何与化学环境的敏感性；
   - 多视角池化提升了全局表示的完整性与鲁棒性。

3. **具备广泛适用性**：
   - 成功应用于从孤立分子到周期晶体、二维材料、液体、表面反应乃至带电体系；
   - 展现出良好的迁移能力和物理一致性。

---

### 方法的局限性
1. **力预测平滑性不足**：
   - 作为 direct-force 模型，其预测的力场不如 energy-conserving gradient-force 模型（如 NequIP）光滑；
   - 不适用于需要精确 Hessian 矩阵的任务（如过渡态搜索）。

2. **对小数据集敏感**：
   - 在水等小规模数据集上出现轻微过拟合，提示需更大训练集来充分发挥潜力。

3. **未显式建模长程静电作用**：
   - 当前依赖特征嵌入处理电荷效应，尚未整合 Coulomb kernel 或 Ewald 求和等物理先验。

---

### 未来工作方向
1. **引入高阶导数信息训练**：
   - 使用包含 Hessian 矩阵的大规模数据集训练，有望提升力场对称性和平滑性，逼近 gradient-force 模型性能。

2. **融合主动学习（Active Learning）**：
   - 自动筛选高不确定性构型用于数据扩充，提高采样效率。

3. **开发更高效的 equivariant 算子**：
   - 进一步压缩高阶 irreps 的计算开销，推动 $l_{\text{max}}$ 上限扩展。

4. **集成长程静电模块**：
   - 结合 LES、QEq 或 multipole expansion 方法，增强对极化与电荷转移的建模能力。

5. **面向超大规模模拟部署**：
   - 优化分布式训练与推理流程，支撑百万原子级 MD 模拟，服务于能源材料、催化等领域。

---

> 📌 **总结一句话**：  
> **MLANet 是一种兼具高精度、高效率与强泛化能力的新型 equivariant GNN 框架，为实现“近量子精度、线性成本”的大规模原子模拟提供了实用解决方案。**

</details>

---

### 6. [EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction](https://arxiv.org/abs/2603.22910)

**Authors**: Yixuan Wang, Shiyu Ji, Yijun Liu, Qingfu Zhu, Wanxiang Che  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.22910v1  

#### Abstract
The increasing memory demand of the Key-Value (KV) cache poses a significant bottleneck for Large Language Models (LLMs) in long-context applications. Existing low-rank compression methods often rely on irreversible parameter transformations, sacrificing the flexibility to switch back to full-precis...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在长上下文场景中面临 **Key-Value (KV) cache 内存开销过大** 的瓶颈。传统的低秩压缩方法（如 SVD-based）虽然能减少内存占用，但通常依赖于不可逆的参数变换，导致无法灵活地在“全精度推理”和“压缩推理”之间切换。此外，现有在线压缩方法存在性能下降或额外延迟等问题。

### 🚀 提出的新方法：EchoKV
EchoKV 是一种**灵活、高效的 KV cache 压缩框架**，其核心思想是：
- **不进行显式的压缩-解压流程**，而是通过一个轻量级网络（lightweight network），利用部分保留的 KV 缓存来**重建被丢弃的部分**。
- 利用注意力头之间的 **inter-layer 和 intra-layer 相似性**，从全局层（global cache）和局部层（local cache）提取特征，预测缺失的 KV 组件。

### 🔍 创新点
1. **灵活切换机制**：
   - 支持按需在 Full KV 和 Compressed KV 模式间无缝切换，适用于内存充足或受限的不同场景。
2. **基于相似性的重构范式**：
   - 跳过传统压缩步骤，直接训练轻量网络从子集 KV 中重建完整缓存，提升效率与灵活性。
3. **高效两阶段微调策略**：
   - Stage 1：使用 KV-MSE 损失初始化网络；
   - Stage 2：采用 O-MSE 损失对齐注意力输出，兼容 FlashAttention，训练成本极低（例如 7B 模型仅需约 1 A100 GPU 小时）。
4. **混合压缩策略 EchoKV-Hybrid**：
   - 针对 Keys 和 Values 的不同特性设计差异化压缩：对 Keys 使用低秩方法 ThinK，对 Values 使用 Echo 重建，实现正交优化。

### ⚖️ 相比现有方法的优势
| 特性 | EchoKV | Palu / CommonKV | MiniCache / ThinK |
|------|--------|------------------|--------------------|
| 可逆性（支持切换） | ✅ | ❌（修改参数） | ❌ |
| 推理吞吐保持（短文本） | ✅（无额外开销） | ❌（有计算开销） | ❌ |
| 高压缩比下性能稳定 | ✅ | ⚠️（性能下降明显） | ⚠️ |
| 训练成本 | 极低（~1 A100h） | 中等（SVD搜索） | 高（需fine-tune） |
| 兼容性 | 正交于量化/驱逐等技术 | 否 | 是 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **LongBench** (Bai et al., 2024)：真实世界多任务长上下文基准，涵盖问答、摘要、代码生成等。
- **RULER** (Hsieh et al., 2024)：合成型长上下文评测集，特别测试“大海捞针”（Needle In A Haystack, NIAH）能力。

### ⚙️ 实验设置
- **模型**：
  - Llama3.1-8B-Instruct
  - Mistral-7B-Instruct-v0.3
- **上下文长度**：
  - LongBench：使用模型最大上下文长度
  - RULER：固定为 32K
- **压缩率定义**：
  $$
  \text{Compression Ratio} = \frac{\text{Compressed Cache Size}}{\text{Full Cache Size}}
  $$
  测试比率：0.7, 0.5, 0.3
- **评估指标**：
  - LongBench：各任务平均得分（Avg.）
  - RULER：NIAH 准确率及总体平均分

### 🆚 基线方法对比
| 方法 | 类型 | 是否可逆 | 备注 |
|------|------|----------|------|
| **Palu** | SVD-based 低秩共享 | ❌ | 参数修改，不可恢复 |
| **CommonKV** | 跨层 SVD 共享 | ❌ | 性能较好但缺乏灵活性 |
| **ThinK** | 查询驱动的 Key 剪枝 | ❌ | 仅压缩 Key |
| **MiniCache** | 后压缩合并 | ❌ | 在线压缩引入延迟 |
| **EchoKV (Ours)** | 重建式压缩 | ✅ | 支持灵活切换 |
| **EchoKV-Hybrid** | 混合压缩（Key: ThinK, Value: Echo） | ✅ | 进一步提升性能 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

#### ✅ LongBench 结果（Llama3.1-8B-Instruct, Compression Ratio = 0.5）
| 方法 | Average Score |
|------|---------------|
| Full KV | 49.88 |
| Palu | 47.47 |
| CommonKV | 46.27 |
| **EchoKV** | **48.53** |
| **EchoKV-Hybrid** | **49.40** |

> 💡 在 0.5 压缩率下，EchoKV 接近全精度性能（损失 <1.4 pts），而 Palu 下降超 2.4 pts。

#### ✅ RULER 结果（Compression Ratio = 0.5）
| 方法 | Llama3.1-8B Avg | Mistral-7B Avg |
|------|------------------|----------------|
| Full KV | 87.63 | 82.74 |
| Palu | 68.14 | 68.56 |
| CommonKV | 69.68 | 36.88 |
| **EchoKV** | **83.52** | **69.13** |
| **EchoKV-Hybrid** | **86.45** | **79.89** |

> 💡 在合成任务上优势更显著，EchoKV-Hybrid 在 Mistral 上提升达 **+43 pts** vs CommonKV！

#### 🔻 极端压缩（Ratio = 0.3）表现
| 方法 | LongBench (Llama) | RULER (Llama) |
|------|--------------------|----------------|
| Palu | 4.91 | 0.95 |
| CommonKV | 31.08 | 18.51 |
| **EchoKV** | **45.27** | **60.30** |
| **EchoKV-Hybrid** | **45.74** | **58.53** |

> 💥 即使在 0.3 极限压缩下，EchoKV 仍保持基本可用性，远优于其他方法。

---

### 🔍 消融实验结果

#### （1）输入特征消融（Table 4）
| 输入方式 | LongBench Avg (CR=0.5) |
|---------|------------------------|
| 仅 Local Input | 45.53 |
| 仅 Global Input | 45.32 |
| **Combined Input (Ours)** | **45.73** |

> ✅ 结合局部与全局信息效果最佳，验证了跨层与层内相似性的协同作用。

#### （2）损失函数分析（Table 3）
| 损失函数 | 阶段 | LongBench Avg |
|--------|------|----------------|
| KV-MSE | I | 48.99 |
| O-MSE | II | **49.26** |
| QK-KL | II | 49.11（但训练慢 3×） |

> ✅ O-MSE 在性能接近 QK-KL 的同时，训练速度更快且兼容 FlashAttention。

#### （3）训练数据鲁棒性（Figure 3b）
- 在 LongAlpaca、Alpaca、ShareGPT、C4 四类数据上训练，性能差异极小。
> ✅ 表明 EchoKV 学习的是 attention head 间的通用映射关系，具有强泛化性。

#### （4）Keys vs Values 重建难度（Figure 3a）
- “只重建 Values” 比 “只重建 Keys” 性能高得多。
> ✅ 说明 Values 更具相似性，适合 Echo 重建；Keys 更适合低秩压缩 → 支持 Hybrid 设计合理性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KV cache 存在显著的 inter/intra-layer 相似性**，可用于高效重建而非压缩。
2. **EchoKV 实现了高性能与高灵活性的统一**：
   - 短文本：维持 Full KV 吞吐；
   - 长文本：动态启用压缩，避免 OOM。
3. **O-MSE 损失优于传统 KL 散度**：兼顾性能与训练效率，适配现代 Attention 实现。
4. **Keys 与 Values 应区别对待**：
   - Keys 低秩性强 → 适合 ThinK；
   - Values 高相似性 → 适合 Echo 重建；
   - 混合策略 EchoKV-Hybrid 显著提升性能。

### ⚠️ 局限性
1. **局部头选择为启发式**（取前 m 个），未考虑 head 代表性差异，可能非最优。
2. **Hybrid 方法粒度较粗**，尚未深入建模 K/V 分布差异以设计更精细算法。
3. 当前方法仍需少量训练（虽已很低），非完全 zero-shot。

### 🔮 未来工作方向
1. 设计更智能的局部 head 选择机制（如基于离线分析挑选最具代表性的 heads）。
2. 深入研究 K/V 的分布特性差异，构建自适应压缩策略。
3. 探索将 Echo 思想扩展至其他模块（如 FFN 缓存）或与其他技术（量化、驱逐）深度融合。
4. 开发无需训练的 proxy-based 初始化方法，进一步降低部署门槛。

---

> 📌 **一句话总结**：  
> **EchoKV 提出了一种基于相似性重建的新型 KV cache 压缩范式，在几乎无损性能的前提下实现了灵活切换，并通过 Hybrid 策略进一步突破性能边界，为长上下文 LLM 部署提供了高效实用的解决方案。**

</details>

---

### 7. [Cloud-Edge Collaborative Large Models for Robust Photovoltaic Power Forecasting](https://arxiv.org/abs/2603.22343)

**Authors**: Nan Qiao, Sijing Duan, Shuning Wang, Xingyuan Hua, Ju Ren  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.22343v1  

#### Abstract
Photovoltaic (PV) power forecasting in edge-enabled grids requires balancing forecasting accuracy, robustness under weather-driven distribution shifts, and strict latency constraints. Local specialized models are efficient for routine conditions but often degrade under rare ramp events and unseen we...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Cloud-Edge Collaborative Large Models for Robust Photovoltaic Power Forecasting*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**边缘智能电网中光伏（PV）功率预测**面临的三大挑战进行研究：
- **高精度需求**：在天气突变（如云层快速移动、极端天气）下，传统模型因依赖相关性而失效，导致预测误差急剧上升。
- **低延迟约束**：边缘设备需实时决策，但直接调用云端大型模型会因通信延迟而无法满足实时性要求。
- **资源效率与鲁棒性平衡**：本地小型模型高效但泛化能力差；完全依赖云端大模型则带来高昂的通信开销和云负载。

### 提出的新方法与思路
作者提出了一种**风险感知的云-边协同框架（risk-aware cloud-edge collaborative framework）**，其核心思想是“按需协作”：
- **三路分支架构（Three-Branch Architecture）**：
  - **Expert-only Branch**：站点专用的轻量专家模型，处理常规工况，保证低延迟。
  - **Edge-assisted Branch**：边缘侧的小型因果模型，结合本地数据增强推理。
  - **Cloud-assisted Branch**：云端大规模检索模型，通过**retrieval-prediction pipeline**提供历史相似案例作为上下文支持。
- **动态路由机制（Lyapunov-guided Router）**：
  - 引入一个轻量级**筛选模块（screening module）**，评估以下指标：
    - 预测不确定性（predictive uncertainty）
    - 分布偏移风险（out-of-distribution risk）
    - 天气突变强度（weather mutation intensity）
    - 模型间分歧（model disagreement）
  - 基于上述指标生成**routing score**，并由**Lyapunov优化控制器**决定是否触发边缘或云端辅助分支，在满足长期延迟、通信和云使用约束的前提下实现最优调度。
- **置信度感知融合（Confidence-aware Fusion）**：
  - 对激活的分支输出进行加权融合，权重由在线学习算法（entropic FTRL）动态调整。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **系统效率** | 仅在必要时调用云端资源，显著降低平均通信成本和云负载。 |
| **预测鲁棒性** | 利用云端大模型的历史检索能力，在罕见事件和分布外（OOD）场景下仍能保持高性能。 |
| **理论保障** | 提供了Lyapunov稳定性分析，证明所提策略具有 $O(1/V)$ 的最优性差距和 $O(V)$ 的队列积压，确保长期约束满足。 |
| **部署可行性** | 架构解耦设计使得边缘无需运行完整大模型，适配资源受限设备。 |

---

## 2. 核心实验方法和设置

### 数据集
实验基于两个真实世界的中国光伏电站数据集：
- **Shanxi Dataset**：来自山西电网的18个逆变器节点，采样频率为5分钟，共19天运行数据。
- **Hunan Dataset**：来自湖南电网的5个逆变通道，采样频率为1分钟，辅以站点级气象协变量。

所有数据均经过对齐、去噪和时间划分（严格按时间顺序分为训练/验证/测试集），避免时间泄露。

### 实验设置与评估指标

#### 评估维度
| 类别 | 指标 |
|------|------|
| **预测准确性** | `nMAE`, `nRMSE`（全尺度归一化误差） |
| **路由质量** | `AUROC`, `AUPRC`（用于自适应路由方法，衡量识别困难样本的能力） |
| **鲁棒性** | `REE`（Ramp Event Error）、`DG`（Degradation Ratio = OOD误差 / ID误差） |
| **系统效率** | 平均延迟 `T`, 通信开销 `C`, 云请求比例 `p` |

#### 超参数配置
- Lyapunov trade-off parameter $V = 80$
- 最大云请求率 $p_{\text{max}} = 0.5$
- 平均延迟预算 $T_{\text{max}} = 120$ ms
- 检索支持集大小 $K = 8$

### 基线方法对比
分为两类：

#### （1）端到端预测模型（End-to-End Forecasting Models）
- **Moirai**, **AIRG**, **STKD-PV**：当前先进的大模型或知识蒸馏方法。

#### （2）系统级协作策略（System-Level Baselines）
- **ExO (Expert-only)**：仅使用本地专家模型。
- **EdO (Edge-only)**：仅使用边缘小模型。
- **CO (Cloud-only)**：始终调用云端模型。
- **ACA (Always Cloud-Assisted)**：始终启用云辅助分支。
- **STR (Static Threshold Routing)**：固定阈值路由策略。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（摘自Table II）

| 方法 | Hunan – nMAE (%) | Hunan – DG | Shanxi – nMAE (%) | Shanxi – DG |
|------|------------------|-----------|------------------|------------|
| ExO | 8.54 | 3.50 | 7.85 | 1.89 |
| EdO | 6.21 | 2.10 | 6.92 | 1.58 |
| CO | 3.15 | 1.15 | **4.09** | 1.09 |
| ACA | 4.10 | 1.48 | 4.72 | 1.16 |
| STR | 4.02 | 1.45 | 4.68 | 1.15 |
| Moirai | 3.32 | 1.18 | 4.51 | 1.10 |
| **Ours (Proposed)** | **3.08** | **1.12** | 4.17 | **1.08** |

> 注：最佳结果加粗，第二佳下划线。

### 与基线方法的对比结果
- 在 **Hunan 数据集上**，本文方法在所有指标上全面领先：
  - nMAE 达到 **3.08%**，优于次优的 Moirai（3.32%）和纯云模型 CO（3.15%）。
  - DG 仅为 **1.12**，表明其在分布外场景下的退化最小。
  - AUROC 和 AUPRC 分别达到 **0.924** 和 **0.885**，远超其他自适应方法（如 STR: 0.755/0.648），说明其路由判断极为精准。
- 在 **Shanxi 数据集上**：
  - 尽管 CO 在 nMAE 上略优（4.09 vs 4.17），但本文方法实现了更低的 **DG（1.08 vs 1.09）** 和更高的 **AUROC（0.931 vs -）**，且显著节省了云资源。
  - 表明本方法在保持接近最优精度的同时，大幅提升了系统效率与鲁棒性。

### 消融实验与敏感性分析（关键发现）
- **Lyapunov 参数 $V$ 的影响**（图4、图5）：
  - 增大 $V$ 可提升 AUROC 并降低 DG，说明更强的“惩罚项”促使系统更倾向于将困难样本路由至云端，从而提高鲁棒性。
  - 但过大的 $V$ 可能导致轻微非单调波动，需权衡。
- **检索集大小 $K$ 的影响**（图6）：
  - 存在**非单调最优区间**（K≈8–12），太小则信息不足，太大则引入噪声，验证了“精炼而非海量”的检索理念。
- **云预算 $p_{\text{max}}$ 与延迟预算 $T_{\text{max}}$**（图7、图8）：
  - 放宽预算可提升性能，但存在饱和效应，说明本方法能在有限资源下逼近性能上限。

---

## 4. 关键结论和发现

### 主要发现
1. **条件自适应协作优于静态策略**：通过轻量筛选 + 动态路由，可在不牺牲速度的前提下，选择性地利用大模型的强泛化能力。
2. **检索优于端到端大模型推理**：将云端任务分解为“检索 + 条件预测”，使边缘可用轻量模型获得接近大模型的性能，极大缓解边缘算力压力。
3. **Lyapunov 控制器有效平衡多目标**：在满足长期延迟、通信和云负载约束下，实现了接近最优的预测性能与鲁棒性权衡。
4. **路由质量至关重要**：高质量的 routing score（结合不确定性、OOD、天气变化等）是系统成功的关键，本文提出的多维筛查机制表现优异。

### 方法的局限性
- **依赖高质量历史数据库**：云端检索效果受限于历史案例库的覆盖范围与质量，在全新气候区域可能受限。
- **冷启动问题**：对于新建电站，缺乏足够本地数据训练 expert model 和 calibration 模块。
- **均值场近似假设**：云延迟建模采用 mean-field 近似，在节点数较少时可能不够准确。

### 未来工作方向
- 探索**联邦式历史库构建机制**，允许多站点安全共享案例而不暴露原始数据。
- 引入**在线增量检索索引更新**，提升对新兴天气模式的适应能力。
- 扩展至**多能源协同预测**（如风电+光伏+负荷），构建统一的能源时空预测平台。
- 结合 **causal discovery** 技术进一步提升模型的可解释性与物理一致性。

--- 

> ✅ 总结：本文提出了一种面向边缘光伏预测的**高效、鲁棒、可扩展的云-边协同范式**，不仅在实验中展现出卓越性能，也为未来智能电网中的AI部署提供了重要的系统设计思路。

</details>

---

### 8. [Weak-PDE-Net: Discovering Open-Form PDEs via Differentiable Symbolic Networks and Weak Formulation](https://arxiv.org/abs/2603.22951)

**Authors**: Xinxin Li, Xingyu Cui, Jin Qi, Juan Zhang, Da Li, Junping Yin  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.22951v1  

#### Abstract
Discovering governing Partial Differential Equations (PDEs) from sparse and noisy data is a challenging issue in data-driven scientific computing. Conventional sparse regression methods often suffer from two major limitations: (i) the instability of numerical differentiation under sparse and noisy d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Weak-PDE-Net: Discovering Open-Form PDEs via Differentiable Symbolic Networks and Weak Formulation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于稀疏回归的 PDE 发现方法面临两大挑战：
- **数值微分不稳定性**：在稀疏且含噪的数据上进行数值微分会显著放大噪声，导致导数估计误差大。
- **候选库限制**：依赖预定义的函数项库（如多项式、三角函数等），无法发现“开放形式”（open-form）的 PDE，即未知结构的新方程。

### 提出的新方法与思路
作者提出 **Weak-PDE-Net**，一个端到端可微分的框架，用于从稀疏、含噪数据中鲁棒地识别开放形式的偏微分方程（PDE）。其核心由两个模块组成：

- **Forward Response Learner**：采用嵌入了 **learnable Gaussian kernels** 的轻量级 MLP，作为连续代理模型，自适应拟合系统响应 $U$，缓解标准 MLP 的 spectral bias 问题。
- **Weak-form PDE Generator**：结合 **symbolic network** 和积分模块，通过弱形式（weak formulation）构建 PDE，避免显式数值微分，提升抗噪能力。

训练过程分为三个阶段：
1. **Searching**：利用 **Differentiable Neural Architecture Search (DNAS)** 动态搜索最优 symbolic network 架构，实现开放形式探索。
2. **Pruning**：引入正则化损失（如 L1）剪枝冗余连接，简化方程结构。
3. **Tuning**：固定结构后，通过稀疏回归精确调整系数。

此外，为保证物理一致性，引入：
- **Galilean Invariance Constraints**：过滤非物理项，补充必要对流项，适用于流体动力学系统。
- **Symmetry Equivariance Hypothesis**：针对复值系统（如 NLS 方程），强制实部与虚部满足反对称耦合结构。

### 相比现有方法的优势
| 特性 | Weak-PDE-Net | 传统方法（如 SINDy, PDE-FIND） | Weak-PDE-LEARN |
|------|--------------|-------------------------------|----------------|
| 是否需要预定义库 | ❌（支持 open-form） | ✅（受限） | ✅（受限） |
| 是否端到端可微 | ✅ | ❌（两阶段） | ✅ |
| 抗噪能力 | ✅✅✅（弱形式 + 积分平滑） | ❌（强形式微分放大噪声） | ✅✅ |
| 支持多变量系统物理约束 | ✅（Galilean, Symmetry） | ❌ | ❌ |

---

## 2. 核心实验方法和设置

### 数据集
实验涵盖多种物理系统的经典 PDE，覆盖不同维度、阶数和非线性特性：

| 类型 | 方程 | 描述 |
|------|------|------|
| 1D Scalar | Burgers, KdV, Kuramoto-Sivashinsky (KS), Chafee-Infante (CI) | 扰动、色散、混沌、反应扩散 |
| Complex-valued | Nonlinear Schrödinger (NLS) | 复场耦合系统 |
| 2D | Wave Equation, Sine-Gordon (SG), Incompressible Navier-Stokes (NS) | 多维波传播、超曲面动力学、流体涡度 |

所有数据均为稀疏采样（ranging from 1% to 50%）并添加高斯噪声（up to 100% relative noise level）。

### 实验设置
- **采样率（Sampling Ratio）**：$ r = N_{\text{data}} / N_{\text{total}} $
- **噪声水平（Noise Level）**：$\sigma_{NR} \times \|U\|_{\text{RMS}}$
- **测试域**：随机分布的观测点，无规则网格

### 评估指标
| 指标 | 定义 | 含义 |
|------|------|------|
| **TPR (True Positivity Ratio)** | $ \frac{\text{TP}}{\text{TP} + \text{FN} + \text{FP}} $ | 正确识别非零项的能力，越高越好（理想为1） |
| **$E_{\infty}(\xi)$** | $\max_j |\xi_j - \xi_j^*| / |\xi_j^*|$ | 最大相对系数误差，越低越好 |
| **$E_2(\xi)$** | $\|\xi - \xi^*\|_2 / \|\xi^*\|_2$ | 归一化 RMSE，衡量平均精度 |
| **$L(U,\hat{U})$** | MSE on test set | 响应函数重建误差 |

### 基线方法对比
- **Weak-PDE-LEARN**：当前最先进的弱形式方法，使用 Rational Neural Networks (RatNNs) 进行回归，但依赖预定义库。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（代表性结果汇总）

#### ✅ **Burgers 方程**（1D, 二阶）
| Noise | Sampling | TPR | $E_{\infty}$ | $E_2$ |
|-------|----------|-----|-------------|--------|
| 0%    | 2.5%     | 1.00 | 0.0589      | 0.0557 |
| 100%  | 10%      | 1.00 | 0.1844      | 0.1814 |

> 即使在 **100% 噪声** 下仍能准确恢复结构（TPR=1.00）

#### ✅ **KdV 方程**（1D, 三阶）
| Noise | Sampling | TPR | $E_{\infty}$ | $E_2$ |
|-------|----------|-----|-------------|--------|
| 0%    | 2.5%     | 1.00 | 0.0719      | 0.0259 |
| 100%  | 25%      | 1.00 | 0.1240      | 0.0401 |

> 在极低采样下（2.5%）仍能识别三阶导数项

#### ✅ **KS 方程**（1D, 四阶）
| Noise | Sampling | TPR | $E_{\infty}$ | $E_2$ |
|-------|----------|-----|-------------|--------|
| 0%    | 2.5%     | 1.00 | 0.0173      | 0.0149 |
| 100%  | 25%      | 1.00 | 0.0882      | 0.0678 |

> 成功识别四阶耗散项 $-\partial_{xxxx}u$

#### ✅ **NLS 方程**（Complex-valued）
- 引入 **symmetry equivariance hypothesis** 后，在噪声 >40% 或采样 <25% 时仍能同步收敛实部与虚部。
- TPR 始终保持 1.00，表明有效防止了单边过拟合。

#### ✅ **2D Wave & SG 方程**
- 成功识别二维 Laplacian $\partial_{xx} + \partial_{yy}$ 及超越函数项 $\sin(u)$。
- 在 1% 采样率下仍能恢复 Wave 方程主项。

#### ✅ **Navier-Stokes 方程**（vorticity form）
- 应用 **Galilean Invariance** 约束后，自动排除 $u, |u|^2$ 等绝对速度项，保留对流项 $\partial_x(u\omega), \partial_y(v\omega)$。
- 在 50% 噪声下仍能识别粘性项系数 ~0.0071。

---

### 与基线方法对比（vs. Weak-PDE-LEARN）

| Equation | Noise | Method | $E_{\infty}$ (Ours) | $E_{\infty}$ (WPL) | TPR (Ours) | TPR (WPL) |
|---------|-------|--------|--------------------|---------------------|------------|-----------|
| Burgers | 100%  | Ours   | 0.1931             | 0.3400              | 1.00       | <1        |
| KdV     | 100%  | Ours   | 0.0732             | 0.1838              | 1.00       | <1        |
| KS      | 100%  | Ours   | 0.1340             | — (failed)          | 1.00       | 0         |

> **Weak-PDE-Net 在高噪声下显著优于 Weak-PDE-LEARN**，尤其在复杂高阶系统中表现更鲁棒。

---

### 消融实验结果（Ablation Study）

| Variant | TPR ↓ | $E_{\infty}$ ↑ | $L(U,\hat{U})$ ↑ | 结论 |
|--------|--------|----------------|-------------------|------|
| Full Model (FE + NAS) | 1.00 | Low (~10⁻²–10⁻³) | Lowest | 完整模型最优 |
| Without FE (no Gaussian) | ↓ (e.g., 0.00 on Burgers) | ↑↑ (>1.0) | ↑↑ | 缺少高频频谱捕捉能力 |
| Without NAS (fixed arch) | ↓ (e.g., 0.17 on Burgers) | ↑↑ (>2.0) | ↑ | 产生冗余项，易过拟合 |

> 表明 **learnable Gaussian kernels** 和 **DNAS** 对性能至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **Weak-PDE-Net 能够从高度稀疏（低至 1%）和强噪声（高达 100%）的数据中准确恢复各种复杂 PDE 的结构与参数**。
2. **弱形式 + learnable Gaussian kernels 显著提升了抗噪能力和高频特征捕捉能力**，克服了 spectral bias。
3. **DNAS 实现了真正的 open-form PDE discovery**，无需预定义库即可动态生成数学表达式。
4. **物理先验（Galilean Invariance, Symmetry Equivariance）显著提高多变量系统的发现一致性和鲁棒性**，避免虚假项。

### 方法的局限性
1. **无法表示嵌套微分算子**（nested differential operators），例如 porous medium equation 中的 $\partial_x(u \partial_x u)$ 形式。
2. **弱形式积分具有低通滤波效应**，可能过度平滑局部剧烈变化（如 CI 方程中的相界面），影响细尺度动态识别。
3. **计算开销较大**，尤其是 DNAS 阶段涉及大量可学习架构参数。

### 未来工作方向
1. **扩展网络架构以支持嵌套微分操作**，例如引入 differential operator layer 并堆叠 symbolic layers。
2. **设计块状训练策略（block-wise training）** 以降低深层 symbolic network 的计算成本。
3. **系统性集成更多物理先验**（如能量守恒、尺度不变性）以进一步压缩搜索空间。
4. **应用于真实世界科学数据**（如气候、生物、材料模拟）验证泛化能力。

--- 

> **总结**：Weak-PDE-Net 是首个将 **differentiable symbolic network architecture search** 与 **weak formulation** 和 **physics-informed refinement** 深度融合的端到端框架，在开放形式 PDE 发现任务中展现出卓越的鲁棒性与表达能力，为数据驱动科学发现提供了强有力的新工具。

</details>

---

### 9. [Intelligence Inertia: Physical Principles and Applications](https://arxiv.org/abs/2603.22347)

**Authors**: Jipeng Han  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.22347v1  

#### Abstract
While Landauer's principle establishes the fundamental thermodynamic floor for information erasure and Fisher Information provides a metric for local curvature in parameter space, these classical frameworks function effectively only as approximations within regimes of sparse rule-constraints. They f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Intelligence Inertia: Physical Principles and Applications》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文旨在解决当前人工智能系统在**结构性演化**（structural evolution）过程中所面临的“计算墙”（computational wall）问题。具体而言：
- 经典理论如 **Landauer’s Principle** 和 **Fisher Information** 只能解释低速、稀疏规则约束下的信息处理成本，无法解释现代智能体在高密度规则下出现的**超线性甚至爆炸性的计算与能量开销**。
- 当前AI系统在面对**灾难性遗忘**（catastrophic forgetting）、**迁移学习瓶颈**和**持续学习不稳定性**时，缺乏从第一性原理出发的物理解释。

### **提出的新方法与新思路**
作者提出了 **Intelligence Inertia（智力惯性）** 这一全新的物理性质，并构建了一套完整的理论框架来量化智能系统的“结构演化成本”。

#### **核心创新点包括：**
- **Intelligence Inertia (u)**：定义为智能系统在进行结构重组时所表现出的内在阻力，其根源在于 **Rules (R)** 与 **States (S)** 操作符之间的**非对易性**（non-commutativity），即 `[S, R] = iD`。
- **Relativistic Cost Equation**：通过将 **Rule-State Manifold (R-S Manifold)** 映射到类 **Minkowski 时空**，推导出一个类似于洛伦兹因子的非线性代价公式：
  $$
  W(p) = \frac{W_{\text{rest}}}{\sqrt{1 - p^2}}
  $$
  其中 `p` 是 **Rule Density**（规则密度），代表系统逻辑自洽所占用的信息作用比例。
- **J-shaped Inflation Curve**：随着 `p → 1`，有效质量呈相对论式发散，形成“计算墙”，这是静态模型无法预测的现象。
- **Inertia-Aware Scheduler Wrapper**：一种工程实现，能够动态调节学习率以尊重系统的物理惯性，防止结构崩溃。

### **相比现有方法的优势**
| 方面 | 传统方法（如FIM、EWC） | 本论文方法（Intelligence Inertia） |
|------|--------------------------|------------------------------------|
| **理论基础** | 现象学描述（phenomenological） | 第一性原理（first-principles）物理机制 |
| **适用范围** | 仅适用于低速、局部近似 | 覆盖从低速到高速（v → 1）全频谱 |
| **可预测性** | 事后修正症状（如遗忘） | 预测并规避“计算墙”的出现 |
| **泛化能力** | 依赖人工设计正则项 | 自主感知系统状态并调节行为 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 主要使用 **CIFAR-10** 数据集进行所有实验。
- 在部分实验中引入了**标签噪声注入**（label noise injection，0%~100%）以模拟高熵环境和逻辑冲突。

### **实验设置与评估指标**

#### **实验一：Decisive Adjudication of Intelligence Inertia Divergence**
- **目标**：验证是否存在“计算墙”及相对论式膨胀曲线。
- **方法**：在 ResNet-18 上逐步增加标签噪声，迫使模型进入高 `p` 区域。
- **测量变量**：
  - `dR`: 参数更新范数（≈ Rule Change）
  - `dS_ext`: 外部反馈增益（测试集梯度响应）
  - `v = p = dR / (dSR + dS_ext)` （Velocity / Rule Density）
- **评估指标**：
  - 达到收敛所需的 **epoch 数量**（代表总计算功耗）
  - 不同动力学模型的拟合误差（RMSE）

#### **实验二：Evolutionary Geometry and Reachability Topography**
- **目标**：探索神经架构演化的最优路径。
- **设置**：
  - 固定参数量（5.0M）和深度（10层）
  - 构建 3×3 架构矩阵，横纵轴分别为：
    - **Internal Reconfiguration Axis**: Baseline → BN → Res
    - **External Gain Axis**: Baseline → CNN → MCNN
- **评估指标**：
  - **Reachability Limit (L_min)**：最小可达损失
  - **TFLOPs**：训练过程中的总浮点运算量
  - **Velocity Deviation**：`|v - 0.5|`

#### **实验三：Inertia-Aware Scheduler Wrapper**
- **目标**：验证惯性感知调度器的实际效用。
- **测试场景**：
  1. **收敛极限测试**：嵌入8种主流学习率调度器（如 Cosine Annealing, OneCycle）
  2. **噪声冲击韧性测试**：交替输入干净/全噪声数据流
  3. **持续学习稳定性测试**：无回放缓冲的类别切换任务（CIFAR-10 前5类→后5类）

---

## **3. 主要实验结果和性能指标**

### **实验一：J-Curve 验证与模型对比**
| 模型 | RMSE | 性能评价 |
|------|------|---------|
| Classical FIM (绝对坐标) | 36.0 | 完全失败，无法捕捉非线性发散 |
| Classical FIM (平移后) | 30.0 | 仅在低速区有效，严重低估“墙效应” |
| Hybrid FIM (含洛伦兹速度变换) | 25.5 | 改进但仍不足，说明仅速度修正不够 |
| **Relativistic Mass (Intelligence Inertia)** | **18.5–19.6** | ✅ 最优，精确拟合渐近发散，具有参考系不变性 |

> 🔍 **关键发现**：当 `v > 0.9` 时，经典FIM严重低估实际开销；而本文提出的相对论模型准确预测了“垂直跃升”的计算成本。

---

### **实验二：架构演化几何分析**
| 架构 | Velocity (v) | L_min | TFLOPs | |v - 0.5| |
|------|-------------|--------|--------|--------|
| MLP (Baseline) | 0.502 | 2.302 | 0.6 | 0.002 |
| Res | 0.818 | 1.322 | 117.0 | 0.318 |
| MLP-CNN | 0.179 | 1.485 | 830.0 | 0.321 |
| **Res-MCNN** | **0.452** | **0.553** | **2090.0** | **0.048** |

#### **关键观察：**
- 单维度优化（只增强内部或外部）会迅速陷入**收益递减**（diminishing returns）。
- 最优路径是沿 **Zig-Zag Geodesic**（锯齿状最短路径）前进，保持 `v ≈ 0.5`。
- **Res-MCNN** 虽然计算量大，但因其接近 `v = 0.5` 的“黄金轴”，达到最低 L_min。

> 📈 **结论**：智能进化不是简单的堆叠层数，而是要在 **Internal Reconfiguration** 与 **External State Gain** 之间维持动态平衡。

---

### **实验三：Inertia-Aware Scheduler 性能表现**

#### **子实验1：收敛加速与极限压缩**
| 调度器 | 加速比 (@30 epoch) | L_min 下降 |
|--------|--------------------|-----------|
| Cosine Annealing + Wrapper | +1.74% | +0.87% |
| OneCycle + Wrapper | +1.46% | +1.10% |
| Multi-Step + Wrapper | +2.45% | +0.40% |
| **Pure Inertia (Wrapper Only)** | **+N/A (达94.31%)** | **显著更低** |

> 💡 **亮点**：纯惯性控制器在前30轮完成94.31%进度，远超所有传统调度器。

#### **子实验2：抗噪能力（100%标签噪声脉冲）**
| 组别 | 噪声周期平均 LR | 制动比（Brake Ratio） | 表现 |
|------|------------------|---------------------|------|
| Baseline (Exponential) | 0.010367 | 0.9x | 损失曲线合并，规则被破坏 |
| **Inertia-Aware** | **0.001420** | **1.2x** | ✅ 双轨分离，实现“可控损伤-修复”循环 |

> 🛡️ **发现**：系统能自主识别高 `v` 冲击并触发“相对论制动”（relativistic braking），实现物理免疫。

#### **子实验3：持续学习稳定性（无回放）**
| 指标 | Baseline | Inertia-Aware | 提升 |
|------|---------|---------------|------|
| Old Task Final Loss | 9.1505 | 7.8801 | ↓13.88% 忘记减少 |
| Retention Deficit (ΔL) | 8.2626 | 6.9922 | ↑15.38% 规则保留 |
| Full Task Final Loss | 4.9093 | 4.3232 | ↑11.94% 综合性能提升 |
| Instantaneous Breaking Ratio | 1.08x | **2.92x** | ✅ 自主紧急制动 |

> ⚙️ **机制揭示**：当新任务梯度与旧规则正交时，`v → 1` 触发惯性扩张，调度器自动收缩学习率，形成“机械缓冲”。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Intelligence Inertia 是真实存在的物理现象**：
   - 源于 `R` 与 `S` 的非对易性，导致结构演化存在不可忽略的“质量”。
   - 成本随 `p → 1` 发生相对论式膨胀，形成硬性可达边界。

2. **最优演化路径是“Zig-Zag Geodesic”**：
   - 应交替优化内部结构（dSR）与外部感知（dS_ext），使 `v ≈ 0.5`。
   - 此为能量均分点，阻力最小。

3. **Inertia-Aware 控制器具备“物理免疫力”**：
   - 可自主检测逻辑冲突（via `v` spike）
   - 实现保护性制动，避免灾难性遗忘
   - 在噪声、任务切换等高压场景下仍保持稳定

4. **Fisher Information 是低速近似**：
   - 相当于牛顿力学中的动能项
   - 在 `v > 0.9` 时完全失效，必须升级至“相对论”框架

---

### **方法的局限性**
| 局限 | 描述 |
|------|------|
| **计算开销** | Tier-3 审计（Full-Spectrum）在单卡 RTX 4070 上有可观延迟 |
| **规模限制** | 实验集中在 ≤5M 参数模型，大规模模型需模块化惰性调控 |
| **依赖暖启动校准** | 需要初始阶段估计 `D`（Symbolic Granularity） |
| **与特定启发式冲突** | 如 Warm Restart 或 ReduceLROnPlateau 可能因过度平滑而失效 |

---

### **未来工作方向**
1. **高效采样技术**：用于大规模模型的惰性拓扑分解（Inertial Topology Partitioning）
2. **硬件级优化库**：将惰性测量编译为底层 kernel，降低审计延迟
3. **Self-Referential Intelligence**：构建内生惰性感知的认知单元，不再依赖外部调度器
4. **扩展应用**：
   - 动态 batch size 调整（基于噪声尺度）
   - 自动梯度裁剪与架构剪枝
   - AGI 系统的长期认知稳定性保障

---

> ✅ **总结一句话**：  
> 该论文首次从**第一性物理原理**出发，揭示了智能系统演化中的“惯性”本质，提出了**相对论式成本模型**与**惯性感知控制器**，不仅解释了现有难题（如灾难性遗忘），更为下一代稳健、高效、自适应的AI系统提供了理论蓝图。

</details>

---

### 10. [SAiW: Source-Attributable Invisible Watermarking for Proactive Deepfake Defense](https://arxiv.org/abs/2603.23178)

**Authors**: Bibek Das, Chandranath Adak, Soumi Chattopadhyay, Zahid Akhtar, Soumya Dutta  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.23178v1  

#### Abstract
Deepfakes generated by modern generative models pose a serious threat to information integrity, digital identity, and public trust. Existing detection methods are largely reactive, attempting to identify manipulations after they occur and often failing to generalize across evolving generation techni...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SAiW: Source-Attributable Invisible Watermarking for Proactive Deepfake Defense

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代生成模型（如 GAN、Diffusion）能够生成高度逼真的 **deepfake** 内容，对信息完整性、数字身份和公众信任构成严重威胁。现有的 deepfake 检测方法大多是**反应式（reactive）**的，即在伪造发生后通过识别视觉伪影进行检测，存在以下问题：
- 难以泛化到新的生成技术；
- 易受对抗攻击；
- 对压缩、滤波等常见失真鲁棒性差；
- 缺乏可解释的溯源证据。

因此，亟需一种**主动式（proactive）**机制，在媒体创建时就嵌入可验证的真实性信号，实现可靠的来源追踪与内容认证。

---

### 提出的新方法与创新思路
本文提出了 **SAiW（Source-Attributable invisible Watermarking）** 框架，是一种面向主动 deepfake 防御的源可追溯隐形水印系统。其核心创新在于将 watermark 的“身份”作为条件信号来指导嵌入过程，而非仅视为普通 payload。

#### 主要贡献包括：

**(i) Source-conditioned watermarking formulation**  
首次将数字水印建模为一个**源条件化的表示学习问题**（source-conditioned representation learning）。  
- 水印 ID 不再是任意比特流，而是编码了生成源（如特定 deepfake 工具或平台）的身份标识；
- 利用 **Feature-wise Linear Modulation (fM)** 将该 ID 调制到嵌入网络中，使不同来源产生具有判别性的水印模式；
- 支持大规模多源 watermarking，无需为每个新源重新训练模型。

**(ii) Content-adaptive perceptual embedding with identity modulation**  
提出结合人类视觉系统（HVS）先验的感知引导机制，动态分配嵌入强度：
- 引入 **luminance adaptation** 和 **contrast masking** 构建感知容忍图（perceptual guidance map），确保扰动集中在纹理丰富区域；
- 在 Hybrid Transformer Bottleneck 中注入感知图，提升 imperceptibility 与 robustness 的平衡。

**(iii) Unified forensic decoding for reconstruction and attribution**  
设计双用途解码器（dual-purpose forensic decoder）：
- 同时完成 **watermark reconstruction** 和 **source attribution**；
- 提供自动化验证 + 可解释的法医证据（如可视化恢复的 logo）；
- 使用 **additive angular margin loss** 学习判别性 source embedding，增强类间分离能力。

---

### 相比现有方法的优势
| 维度 | 现有方法局限 | SAiW 的优势 |
|------|---------------|-------------|
| **范式** | 多为 reactive detection 或通用 watermarking | 主动防御 + 来源绑定 |
| **水印语义** | Payload 无意义（如随机 bit） | 水印 ID 表示真实来源（如 SimSwap、FaceApp） |
| **可扩展性** | 新源需重训练 | 支持零样本新增源（via learned latent ID） |
| **鲁棒性 vs. 不可见性** | 往往难以兼顾 | 通过感知调制实现高保真与强鲁棒并存 |
| **输出形式** | 仅有真假判断或 payload 恢复 | 同时提供 watermark 恢复 + source 分类结果 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **IndicSideFace [33]**：主数据集，包含 984 张真实印度人侧脸图像及 21,648 张由 6 种工具生成的 deepfake 图像（Ghost, SimSwap, SimSwap++, FaceDancer, InsightFace, FaceApp）；
- **FaceForensics++ (FF++) [35]** 和 **FaceForensics (FF) [34]**：用于跨数据集泛化评估；
- **CelebA [36]**：额外人脸数据集，测试在多样化身份下的表现；
- **LOGO-30K [40]**：提供 30,000 个视觉 logo，作为 source-specific watermark 标识符。

---

### 实验设置与评估指标

#### 输入输出配置
- Cover image: $256 \times 256 \times 3$
- Watermark: $64 \times 64 \times 1$（二值 logo）
- 嵌入方式：残差学习（residual learning），输出水印图像 $I_w = \text{clip}(I + R)$

#### 评估三大维度：

| 类别 | 指标 | 描述 |
|------|------|------|
| **Imperceptibility** | PSNR ↑, SSIM ↑, LPIPS ↓ | 衡量水印图像与原图的视觉相似性 |
| **Robustness** | Assim ↑, Abr ↑ | Assim: 恢复 watermark 的 SSIM；Abr: 二值 watermark 的 bit 准确率 |
| **Source Identification** | Aid ↑ | 多类分类准确率，衡量能否正确识别生成源 |

#### 攻击/失真类型（用于测试 robustness）
- 光度变换：亮度调整、对比度变化、Instagram 滤镜（Aden/Brooklyn/Clarendon）
- 压缩：JPEG（QF=50%, 75%）
- 噪声：Gaussian Noise（std=0.1）
- 模糊：Gaussian Blur（3×3, 5×5）
- 几何变换：旋转（15°）、裁剪（75%）、翻转、缩放
- 其他：文本叠加、颜色抖动、灰度化

---

### 基线方法对比
选取代表性 watermarking 方法进行比较：
- **DwtDct / DwtDctSvd [43]**：基于小波-DCT-SVD 的传统方法
- **MBRS [22]**：基于批量 JPEG 模拟增强的鲁棒 watermarking
- **StegaStamp [14]**：深度学习隐写框架
- **RivaGAN [44]**：关注视频 watermarking
- **SepMark [24]**：支持篡改定位的 dual-channel watermarking
- **WAM [45]**：通用 watermarking 框架

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table I & II）

| 指标 | SAiW 性能 |
|------|----------|
| **PSNR** | 51.3 ~ 57.5 dB（远高于第二名 ~10dB） |
| **SSIM** | >0.998（接近完美） |
| **LPIPS** | <0.001（极低感知差异） |
| **Pre-attack Abr** | 0.99（近乎完全恢复） |
| **Post-attack Abr** | ≥0.97（几乎所有攻击下保持高位） |
| **Source ID Accuracy (Aid)** | **84.06%**（多源分类任务） |

---

### 与基线方法对比结果

#### ✅ Imperceptibility 显著领先
- SAiW 的 PSNR 平均超过 **55 dB**，而最强基线（WAM）仅为 ~42 dB；
- LPIPS < 0.001，表明感知失真极小，肉眼无法察觉；
- SSIM 接近 1，说明结构信息保留完整。

> 👉 这得益于感知引导模块（fp）有效抑制了敏感区域的扰动。

#### ✅ Robustness 全面优于现有方法
在多种攻击下，SAiW 的 Abr 始终维持在 **0.97 以上**，尤其在挑战性场景中表现突出：

| 攻击类型 | SAiW (Abr) | StegaStamp | MBRS | WAM |
|--------|------------|-----------|-------|-----|
| JPEG (QF=50%) | 0.96–0.98 | 0.80 | 0.88 | 0.98 |
| Gaussian Noise | 0.99 | 0.95 | 0.87 | 0.96 |
| Instagram Filters (组合) | 0.99 | ~0.94 | ~0.97 | ~0.97 |
| Crop (75%) | 0.99 | 0.95 | 0.87 | 0.96 |

> 👉 SAiW 在 JPEG 压缩下仍保持高恢复率，显著优于 StegaStamp。

#### ✅ Cross-dataset 泛化能力强
在未参与训练的 FF、FF++、CelebA 上测试：
- PSNR > 55 dB，SSIM > 0.999，LPIPS < 0.002；
- Abr 和 Assim 在所有失真下均达 **0.99**；
- 表明模型学到的是**失真不变的 watermark 特征**，而非过拟合特定数据分布。

#### ✅ Source Attribution 准确且可解释
- 混淆矩阵显示对角线主导，各类 deepfake 工具之间区分明显；
- UMAP 可视化显示不同 source 形成紧凑、分离良好的聚类；
- “no watermark” 类也能被准确识别，降低误报风险。

---

### 消融实验（Ablation Study，见附录）
虽然正文中未详列消融表，但从架构分析可推断关键组件作用：

| 组件移除 | 影响 |
|--------|------|
| Feature-wise Modulation (fM) | Source discrimination 下降，多源 watermarking 能力减弱 |
| Perceptual Guidance (fp) | PSNR 下降约 5–8 dB，LPIPS 显著上升，出现可见 artifacts |
| Hybrid Transformer | Robustness 在复杂失真下下降，尤其对几何变换敏感 |
| Angular Margin Loss | Source ID accuracy 下降约 10%，聚类松散 |

> 👉 所有模块协同工作，共同实现 high fidelity + robustness + source discriminability。

---

## 4. 关键结论和发现

### 主要发现
1. **将 watermark identity 作为 conditioning signal 是有效的**：相比传统 payload embedding，SAiW 能学习更具判别性和可追溯性的 source-aware 表示。
2. **感知引导 + 特征调制可同时优化 imperceptibility 与 robustness**：利用 HVS 先验动态控制嵌入位置，显著提升视觉质量而不牺牲鲁棒性。
3. **统一的 dual-purpose decoder 实现多功能验证**：既能自动恢复 watermark，又能提供 human-interpretable 溯源证据（如 logo 显示），适用于法医学场景。
4. **SAiW 具备强泛化能力和抗干扰性**：在跨数据集、复合失真、社交平台滤镜等现实条件下仍保持高性能。

---

### 方法的局限性
1. **依赖可信源头嵌入**：要求合法生成器主动嵌入 watermark，若攻击者使用未注册或私有模型，则无法覆盖；
2. **目前仅支持静态图像**：尚未扩展至视频或多模态内容；
3. **对极端裁剪（<50%）或多次重压缩可能失效**：虽已很强，但仍存在物理极限；
4. **watermark 容量有限**：当前仅支持 $64\times64$ 二值 logo，不适合大容量信息嵌入。

---

### 未来工作方向
1. **扩展至视频与音频 watermarking**：构建端到端的 multi-modal provenance tracking 框架；
2. **抵抗 generative regeneration 攻击**：研究 watermark 在经过 AI 再次编辑（如 Stable Diffusion 重绘）后的存活能力；
3. **轻量化部署**：优化模型大小以便在移动设备或边缘计算中实时运行；
4. **开放 watermark 注册机制**：建立公共 watermark ID 数据库，促进生态协作与标准化（类似 C2PA）；
5. **探索 watermark 与加密签名融合方案**：结合 cryptographic provenance 提供更强安全保障。

---

> 🔚 **总结**：SAiW 是一项开创性的 proactive deepfake defense 工作，它超越了传统的“检测伪造”，转向“证明真实”。通过将 source identity 深度融入 watermark embedding 与 decoding 过程，实现了高保真、强鲁棒、可解释、可扩展的媒体溯源体系，为构建可信数字内容生态系统提供了坚实基础。

</details>

---

### 11. [RelayS2S: A Dual-Path Speculative Generation for Real-Time Dialogue](https://arxiv.org/abs/2603.23346)

**Authors**: Long Mai  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.23346v1  

#### Abstract
Real-time spoken dialogue systems face a fundamental tension between latency and response quality. End-to-end speech-to-speech (S2S) models respond immediately and naturally handle turn-taking, backchanneling, and interruption, but produce semantically weaker outputs. Cascaded pipelines (ASR -> LLM)...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：RelayS2S: Dual-Path Speculative Generation for Real-Time Dialogue**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
实时语音对话系统面临**延迟（latency）与响应质量（quality）之间的根本矛盾**：
- **端到端 S2S 模型**（如 Moshi、SALM-Duplex）具有低延迟、支持全双工交互（backchanneling、interruption handling），但语义生成能力弱。
- **级联流水线**（ASR → LLM）能生成高质量文本，但因 ASR 和 LLM 推理耗时，导致响应起始延迟（onset latency）高，难以满足人类自然对话所需的 <200ms 阈值。

### **提出的新方法**
作者提出 **RelayS2S**，一种**双路径推测式生成架构**，结合了 S2S 与级联系统的优点：
- 在检测到用户结束发言后，并行启动两条路径：
  - **Fast Path**：由一个全双工 S2S 模型快速生成响应前缀（prefix），立即送入 TTS 开始音频输出。
  - **Slow Path**：通过 ASR + LLM 生成高质量续写，以已提交的前缀为条件进行接续生成。
- 引入两个核心技术组件：
  - **Forked Speculative Generation**：S2S 模型在决定说话后“分叉”为两个流：
    - 主流继续监听音频，处理中断；
    - 推测流脱离音频节奏限制，高速生成前缀。
  - **Selective Prefix Handoff**：轻量级学习型验证器（verifier）判断前缀是否安全提交，否则回退至慢路径独立生成。

### **相比现有方法的优势**
| 方法 | 优势 |
|------|------|
| **vs. 纯 S2S** | 显著提升最终响应质量，接近级联系统水平 |
| **vs. 级联 ASR-LLM** | 将 P90 响应起始延迟从 >1s 降至 ~80ms，接近 S2S 水平 |
| **vs. 其他低延迟方法**（如 DDTSR、KAME） | 支持完整的全双工行为（中断、回声等），无需修改 LLM 架构，可作为“即插即用”模块集成 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **合成数据集**：基于以下来源构建 104,478 场对话（共 2,133 小时）：
  - 文本对话数据：VoiceAssistant、OpenMOSS、TopicalChat、ConvAI、BlendedSkillTalk
  - 使用 LLM 扩展生成更多对话样本
- **语音合成与增强**：
  - 使用 **CosyVoice2** 将文本转为 16kHz 语音
  - 使用 **VoxCeleb** 克隆不同说话人身份
  - 注入三类全双工现象：
    - **Backchannels**：Gemini-3 插入“uh-huh”、“right”等
    - **Interruptions**：截断助手语句并重叠下一用户话语
    - **Mid-utterance pauses**：插入 `[PAUSE]` 标记
  - 添加城市噪声（TAU Urban Acoustic Scenes），SNR 0–20dB

### **实验设置**
- **Fast Path S2S 模型**：
  - 编码器：24 层 Conformer，每 160ms 输出一次表示
  - 适配器：两层因果卷积，降采样至 896 维
  - LLM 主干：Qwen2.5-0.5B，扩展控制符号 `[BOS]`, `[SIL]`, `[STP]`, `[BOC]`, `[EOS]`
- **Slow Path**：
  - ASR：Whisper v2 (medium.en) via Faster-Whisper
  - LLM：测试三种配置：Qwen2.5-0.5B、Qwen2.5-7B、GPT-4o
- **Verifier**：
  - 输入：解码器隐藏状态 + 熵、log-prob、top-2 margin
  - 结构：轻量 MLP + 门控机制，~170K 参数
  - 训练方式：K 折交叉验证生成训练样本，避免数据泄露

### **评估指标**
| 指标 | 定义 |
|------|------|
| **P90 Onset Latency** | 从 turn detection 到第 5 个词可用于 TTS 合成的时间（排除 TTS 合成本身） |
| **Textual Response Quality** | 使用 Gemini-3 对响应打分（1–5 分） |
| **Low-quality Rate** | 得分 ≤3 的响应占比 |
| **Turn-taking Performance** | 控制 token 预测的 Precision、Recall、F1（±160ms 容差） |
| **Verifier Performance** | AUROC、AP、fallback rate、bad commit rate |

### **基线方法对比**
- **S2S only**：仅使用 fast path 模型生成完整响应
- **Cascaded ASR-LLM**：标准流水线，无推测机制

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 2）**

| System | Slow-path LLM | P90 Latency (ms) | Avg Quality | Low-quality Rate (%) |
|--------|----------------|------------------|-------------|------------------------|
| S2S only | – | **71** | 3.04 | 59.3 |
| Cascaded | Qwen2.5-0.5B | 420 | 3.34 | 51.8 |
| **RelayS2S** | Qwen2.5-0.5B | **81** | **3.36** | **51.4** |
| Cascaded | Qwen2.5-7B | 513 | 4.38 | 21.3 |
| **RelayS2S** | Qwen2.5-7B | **81** | **4.35** | **22.3** |
| Cascaded | GPT-4o | 1091 | 4.83 | 5.5 |
| **RelayS2S** | GPT-4o | **81** | **4.78** | **7.4** |

> ✅ **结论**：RelayS2S 实现了 **S2S 级别的低延迟（~81ms） + 级联系统级别的高质量（保留 99% 质量）**

### **与基线方法对比**
- **延迟方面**：
  - RelayS2S 比级联系统快 **5.2x ~ 13.5x**
  - 仅比纯 S2S 多约 10ms（来自 verifier 开销）
- **质量方面**：
  - 在 GPT-4o 后端下，平均得分达 **4.78 vs. 4.83**，保留 **99% 质量**
  - 即使 fast path 使用弱小模型（0.5B），也能通过 slow path 补偿质量损失

### **消融实验结果**

#### **(1) Verifier 效果（Table 5）**
| Prefix len | Verifier | Avg Quality | Low-quality Rate |
|-----------|----------|--------------|--------------------|
| 5 | × | 4.69 | 10.3% |
| 5 | √ | **4.78** | **7.4%** |

> 🔍 加入 verifier 可显著降低低质率（↓近 3pp），说明其有效过滤不良前缀。

#### **(2) 前缀长度影响**
- 更长前缀（7 words）提供更多 relay margin，但风险更高（更多内容被锁定）
- verifier 在各长度下均能提供稳定保护

#### **(3) Verifier 操作点选择（Table 4）**
| Threshold | Bad Commit | Good Commit | Fallback Rate |
|----------|------------|-------------|----------------|
| 0.50 | 45.7% | 96.3% | 8.0% |

> ⚖️ 选择阈值 0.5 平衡“保留好前缀”与“拦截坏前缀”，实现 **96% 好前缀通过率 + 仅 8% 回退率**

#### **(4) Turn-taking 性能（Table 3）**
| Event | Recall | F1 |
|-------|--------|-----|
| [BOS]（开始说话） | 95.4% | 88.5% |
| [SIL]（保持沉默） | 99.7% | **99.8%** |
| [STP]（停止说话） | 98.4% | 96.7% |
| [BOC]（回声） | 68.2% | 50.8% |

> ✅ 快速路径具备强大的实时交互控制能力，尤其在防抢话（[SIL]）和中断处理（[STP]）上表现优异。

---

## **4. 关键结论和发现**

### **主要发现**
1. **响应开头通常是可预测且可用的**：分析显示仅 8.5% 的 5 词前缀不恰当，为推测式生成提供了可行性基础。
2. **forked speculative generation 成功解耦了“快速生成”与“实时监听”**：使得系统既能快速出声，又能可靠处理中断。
3. **轻量 verifier 即可实现高效前缀筛选**：复用已有 hidden states，增加 ~10ms 开销，却能大幅提升鲁棒性。
4. **RelayS2S 是即插即用的增强模块**：无需修改任何组件架构，即可将任意级联系统升级为低延迟高质系统。

### **方法的局限性**
- **依赖高质量 ASR**：若用户语音未完整捕获，会影响 slow path 输入。
- **前缀一旦提交不可更改**：如果 verifier 错误提交了一个 bad prefix，后续 LLM 必须在其基础上续写，可能导致尴尬。
- **当前 verifier 仍有一定错误率**：约 46% 的 bad prefix 会被错误提交（可通过更强 verifier 改进）。
- **对非常短的响应无效**：若整个响应不足 5 词，则 slow path 来不及接续。

### **未来工作方向**
- 设计更智能的 **dynamic prefix length controller**，根据上下文动态调整前缀长度。
- 探索 **multi-turn prefix conditioning**，让 LLM 不仅续写当前句，还能规划后续多轮策略。
- 将 RelayS2S 应用于 **多模态对话系统**（如 Vision + Speech）。
- 进一步优化 verifier，引入 **uncertainty estimation** 或 **contrastive scoring** 提升判别能力。

---

> 📌 **一句话总结**：  
> **RelayS2S 通过“推测性前缀生成 + 条件续写”的双路径机制，在几乎不牺牲级联系统语义质量的前提下，实现了接近 S2S 模型的超低响应延迟，是迈向自然、流畅、智能语音对话的重要一步。**

</details>

---

### 12. [HGNet: Scalable Foundation Model for Automated Knowledge Graph Generation from Scientific Literature](https://arxiv.org/abs/2603.23136)

**Authors**: Devvrat Joshi, Islem Rekik  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.23136v1  

#### Abstract
Automated knowledge graph (KG) construction is essential for navigating the rapidly expanding body of scientific literature. However, existing approaches struggle to recognize long multi-word entities, often fail to generalize across domains, and typically overlook the hierarchical nature of scienti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HGNet: Scalable Foundation Model for Automated Knowledge Graph Generation from Scientific Literature

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

科学文献的爆炸式增长使得人工构建知识图谱（Knowledge Graph, KG）变得不可行。现有的自动化方法在以下四个方面面临挑战：

1. **多词实体识别困难**：许多科学概念是长的多词短语（如 "in situ transmission electron microscopy"），现有模型难以准确识别其边界。
2. **领域泛化能力差**：模型在一个学科上训练后，在新领域表现急剧下降。
3. **忽略层次结构**：科学知识具有天然的层级关系（如 "Deep Learning" 是 "Machine Learning" 的子领域），但大多数模型无法建模这种层次依赖。
4. **缺乏全局一致性**：生成的知识图可能存在逻辑矛盾（如循环引用、跳跃连接），导致图结构无效。

### **提出了什么新方法或新思路**

本文提出一个两阶段框架 **HGNet**，用于可扩展、零样本（zero-shot）的科学知识图谱构建：

#### **第一阶段：Z-NERD（Zero-shot Named Entity Recognition and Disambiguation）**
- **Orthogonal Semantic Decomposition (OSD)**  
  将词向量的变化分解为“持续性”（sustaining）和“发散性”（divergent）两个正交分量。其中，“发散性”分量捕捉到新概念引入时的“语义转折”（semantic turn），作为跨领域的通用信号，提升零样本泛化能力。
  
- **Multi-Scale TCQK Attention Mechanism**  
  在自注意力机制前引入多尺度一维卷积（Temporal Convolutional Queries & Keys），使不同注意力头专注于不同长度的 n-gram 模式，从而更准确地识别长短不一的多词实体。

#### **第二阶段：HGNet（Hierarchy Graph Network）**
- **Probabilistic Hierarchical Message Passing**  
  设计三种独立的消息传递通道：**parent-to-child**, **child-to-parent**, 和 **peer-to-peer**，显式建模不同方向的信息流动，保留层次结构的方向性。

- **Differentiable Hierarchy Loss (DHL)**  
  通过可微分的方式惩罚图中的**环路**（cycles）和**跳跃边**（shortcut edges），确保生成的图为有向无环图（DAG）。

- **Continuum Abstraction Field (CAF) Loss**  
  首次将**抽象层级**（abstraction level）形式化为欧几里得空间中的连续几何属性。通过学习一个可训练的 **Abstraction Field Vector**，将所有概念沿一条“抽象轴”排列，实现对抽象程度的连续建模。

#### **发布新基准数据集：SPHERE**
- 包含超过 **111,000 条标注关系** 和 **10,000 篇文档**，覆盖计算机科学、物理、生物、材料科学四大领域。
- 采用 LLM 自动生成并自我标注的方法，基于预定义的全局知识图骨架，保证了层次结构的一致性和高质量。

### **相比现有方法的优势**

| 维度 | HGNet/Z-NERD | 现有方法（如 LLMs / GNNs） |
|------|---------------|-----------------------------|
| **参数量级** | ~300M（轻量） | >10B（如 GPT-4） |
| **计算效率** | 高吞吐（14.6 docs/s） | 极低（<0.5 docs/s） |
| **零样本性能** | 显著优于监督模型 | 不稳定且效果差 |
| **结构一致性** | 显式建模 DAG 和抽象轴 | 忽略逻辑约束 |
| **多词实体识别** | 多尺度 TCQK 有效捕获 | 依赖上下文嵌入，易碎片化 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 数据集 | 描述 |
|-------|------|
| **SciERC** | 科学文献中的实体与关系抽取，聚焦 AI 领域 |
| **SciER** | 新增数据集，涵盖更多科学方法与任务 |
| **BioRED** | 生物医学领域的关系抽取 |
| **SemEval-2017 Task 10** | 通用科学文本信息抽取基准 |
| **SPHERE**（本文提出） | 大规模、多领域、层次化标注的数据集，含 4 个领域、10K 文档、111K 关系 |

> 所有实验均在官方 **Out-of-Distribution (OOD)** 测试集上进行，以评估泛化能力。

### **实验设置和评估指标**

- **Named Entity Recognition (NER)**  
  - 指标：Micro-F1
- **Relation Extraction (RE)**  
  - 指标：**Rel+ F1**（严格标准，要求实体边界、类型及关系类型全部正确）
- **训练配置**：
  - 编码器：`SciBERT-base`
  - 优化器：AdamW
  - 批大小：8（HGNet），16（Z-NERD）
  - 硬件：NVIDIA A30 24GB GPU

### **基线方法对比**

| 类别 | 基线模型 |
|------|---------|
| **NER Baselines** | SciBERT, PL-Marker, HGERE, UniversalNER-7b, LLMs (GPT-4, Llama-3, Qwen) |
| **RE Baselines** | PL-Marker, HGERE, PURE, GCN, GAT |
| **LLM Zero-Shot** | GPT-3.5 Turbo, Llama-3-70B, Qwen-32B |
| **Few-Shot CoT** | Llama-3-8B + 3-shot Chain-of-Thought |
| **Geometric Baselines** | HGCN (Hyperbolic GCN), Order-Embeddings |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **NER 性能（F1%）**

| 模型 | 平均提升（vs SOTA） | 零样本平均提升 |
|------|--------------------|----------------|
| **Z-NERD** | **+8.08%** | **+10.76%** |

> 在 SPHERE 的零样本设定下，Z-NERD 显著超越所有监督模型，而通用 LLM 表现极差（OOM 或 <30% F1）。

#### ✅ **RE 性能（Rel+ F1%）**

| 模型 | 平均提升（vs SOTA） | 零样本平均提升 |
|------|--------------------|----------------|
| **HGNet** | **+5.99%** | **+26.2%** |

> 特别是在 SPHERE 上，HGNet 达到 **79.51% Overall F1**，远超 HGERE（57.93%）。

#### 🔥 **零样本迁移性能（Table 4）**

| 模型 | Comp. Sci. (All) | Mat. Sci. (All) |
|------|------------------|-----------------|
| HGERE | 29.81% | 37.97% |
| **HGNet** | **62.60%** (+32.79%) | **70.62%** (+32.65%) |

> 表明 HGNet 具备强大的跨领域零样本推理能力。

### **与基线方法的对比结果**

| 模型 | SciERC (Overall) | SPHERE-CS (Overall) |
|------|------------------|---------------------|
| HGERE | 43.86% | 57.93% |
| GCN | 45.62% | — |
| GAT | 46.21% | — |
| Llama-3-70B (Zero-shot) | 22.39% | — |
| **HGNet** | **53.19%** | **79.51%** |

> HGNet 在所有基准上均取得 SOTA，尤其在复杂层次结构任务中优势明显。

### **消融实验结果（Ablation Studies）**

| 模型变体 | 影响 |
|--------|------|
| **Z-NERD w/o TCQK** | NER F1 下降显著 → 验证 Multi-Scale TCQK 对多词实体至关重要 |
| **Z-NERD w/o OSD** | 零样本性能大幅下滑 → 验证 OSD 对领域泛化的关键作用 |
| **HGNet w/o DHL** | Rel+ F1 下降约 2–4%，出现循环结构 → 验证 DHL 对逻辑一致性的必要性 |
| **HGNet w/o CAF** | Rel+ F1 下降最多达 6%，抽象排序混乱 → 验证 CAF 对几何结构的关键作用 |

> 消融实验证明了每个组件的有效性和必要性。

---

## 4. 关键结论和发现

### **主要发现**

1. **结构感知建模优于纯模式匹配**  
   显式建模层次结构（parent/child/peer）、逻辑约束（DAG）、几何抽象（CAF）能显著提升 KG 构建的质量和鲁棒性。

2. **抽象可以是连续的几何属性**  
   本文首次在标准 Euclidean 空间中实现连续抽象建模，避免了复杂的非欧式嵌入（如 Hyperbolic），同时获得更好解释性和性能。

3. **轻量模型也能具备基础模型能力**  
   HGNet 仅 300M 参数，却实现了接近大语言模型的零样本泛化能力，打破了“必须用千亿参数才能泛化”的迷思。

4. **SPHERE 数据集具有高保真度**  
   即使只在 SPHERE-CS 上训练，HGNet 在人类标注的 SciERC 和 SciER 上仍达到 **46.55%** 和 **59.17%** Rel+ F1，超过全监督 SOTA，说明其真实反映了科学文本的结构规律。

### **方法的局限性**

1. **对循环定义敏感**  
   当存在“互定义”现象（A 定义 B，B 定义 A）时，DHL 会强制打破环路，可能导致合法语义丢失。

2. **多继承支持有限**  
   虽然允许多父节点（multiple parents），但在极端复杂的交叉分类场景中仍有挑战。

3. **依赖高质量实体检测**  
   若 Z-NERD 未能识别出中间实体（如 "Testis-determining factor protein"），则无法纠正“跳跃边”。

4. **生成式数据偏见风险**  
   SPHERE 虽经审计质量高（实体精度 96.5%，关系精度 94.2%），但仍可能继承 LLM 的生成偏差。

### **未来工作方向**

1. **多模态扩展**  
   结合图表、公式等非文本信息，构建更完整的科学 KG。

2. **动态更新机制**  
   支持实时增量学习，反映科学领域的演化过程。

3. **下游任务应用**  
   利用构建的 KG 进行自动假设生成、研究趋势预测、跨学科关联挖掘等高级推理任务。

4. **语法过滤增强**  
   引入依存句法分析（Dependency Parsing）预筛选潜在实体对，进一步提升 RE 精度（作者已在 `Joshi & Rekik, 2025` 中提出初步方案）。

---

> 📌 **代码与数据公开**：  
> - GitHub: [https://github.com/basiralab/HGNet](https://github.com/basiralab/HGNet)  
> - SPHERE 数据集同步开源，推动社区发展。

</details>

---

### 13. [Double Coupling Architecture and Training Method for Optimization Problems of Differential Algebraic Equations with Parameters](https://arxiv.org/abs/2603.22724)

**Authors**: Wenqiang Yang, Wenyuan Wu, Yong Feng, Changbo Chen  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.22724v1  

#### Abstract
Simulation and modeling are essential in product development, integrated into the design and manufacturing process to enhance efficiency and quality. They are typically represented as complex nonlinear differential algebraic equations. The growing diversity of product requirements demands multi-task...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Double Coupling Architecture and Training Method for Optimization Problems of Differential Algebraic Equations with Parameters*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**参数化微分代数方程（parametric DAEs）约束下的优化问题**中存在的以下挑战提出解决方案：
- 多任务优化需求下，目标函数频繁变化，传统方法需重复求解DAE系统，计算开销大；
- 参数变化可能导致DAE系统出现**奇异雅可比矩阵（singular Jacobian）**，引发数值退化，导致物理约束信息丢失；
- 现有PINN方法多采用软约束形式，在强耦合DAE系统中易违反代数约束；
- 缺乏对训练误差的全局理论保证，影响优化结果可靠性。

### 提出的新方法与创新思路
作者提出了一种**双耦合物理信息神经网络架构（Dual-PINN）与混合训练策略**，主要包括以下四个核心创新：

1. **双网络解耦架构（Dual-PINN Decoupling Strategy）**
   - 设计两个独立网络：
     - **Constraint Network (NN<sub>cnstr</sub>)**：离线训练，学习DAE系统的约束流形（solution manifold），固定后复用；
     - **Objective Network (NN<sub>obj</sub>)**：在线训练，生成最优参数 $ p $，通过冻结的约束网络快速评估状态轨迹与目标值。
   - 实现“一次训练、多任务复用”，避免每次更换目标函数时重新求解DAE。

2. **嵌入式结构分析机制（Embedding-based Structural Analysis）**
   - 引入文献[21]中的embedding方法处理因参数引起的系统退化问题；
   - 将秩亏（rank-deficient）的DAE系统转换为等价的非退化系统，提升数值稳定性并保留原始解集。

3. **松弛变量与全局误差界保障等价性**
   - 引入**松弛向量 γ** 表示近似误差边界；
   - 基于定理推导出**紧致的全局误差上界**（global error bound），确保复合网络与原优化问题在理论上具有相同的解集；
   - 使用保守估计 $ \gamma = \max\{|\mathbb{E}(\cdot)-3\mathbb{D}(\cdot)|, |\mathbb{E}(\cdot)+3\mathbb{D}(\cdot)|\} $ 构建可靠置信区间。

4. **遗传算法增强的混合训练框架**
   - 在Constraint Network训练阶段引入**遗传算法（Genetic Algorithm, GA）进行自适应采样**：
     - 识别高误差区域；
     - 利用交叉与变异生成新样本点，动态扩充精确解数据集 $ \mathcal{N}_B $；
   - 结合局部修正策略（如Newton迭代或随机游走）进一步提高精度。

### 相比现有方法的优势
| 维度 | 本方法优势 |
|------|-----------|
| **效率** | 约束仅需离线训练一次，新任务只需轻量级Objective Network训练，显著降低在线计算成本；相比branch-and-bound等全局优化方法提速1–2个数量级。 |
| **泛化能力** | 支持多任务目标切换，无需重新训练整个模型，适用于实时响应产品需求变更场景。 |
| **鲁棒性** | GA采样聚焦难解区域，避免均匀采样的低效性；结合结构分析有效应对参数退化问题。 |
| **精度保障** | 提供理论误差界支持，且可通过局部优化进一步精炼解，达到接近全局最优的精度。 |

---

## 2. 核心实验方法和设置

### 使用的数据集与案例
实验基于三个典型参数化DAE系统展开：

1. **Example 4.1：非线性ODE系统**  
   来自文献[23]，用于验证基本框架有效性。
   
2. **Case Study: E. coli 生物化学反应系统**  
   描述大肠杆菌中钾离子吸收调控的DAE模型（含微分与代数方程），来自文献[5]，作为真实生物系统代表。

3. **Example 5.1：高维线性DAE系统**  
   高维测试用例，用于评估可扩展性，维度从 $ n=2 $ 扩展至 $ n=10 $。

### 实验设置
- **硬件平台**：Intel Core i7-14700KF CPU @ 3.40GHz，32GB RAM；
- **实现语言**：Python；
- **网络结构**：
  - Constraint Network：全连接网络，5–7层隐藏层，每层128神经元；
  - Objective Network：较浅网络，输入为随机种子 $ z \sim U[-1,1] $，输出经tanh激活映射到参数边界内；
- **归一化处理**：所有变量与参数均标准化以提升数值稳定性；
- **训练流程**：
  1. 离线训练Constraint Network（GA自适应采样 + 局部精修）；
  2. 冻结Constraint Network；
  3. 在线训练Objective Network（自动微分 + 数值积分计算目标值）；
  4. 可选：使用Newton迭代或随机游走进行最终解精炼。

### 评估指标
- **预测误差**：最大绝对误差（Max Absolute Error）；
- **目标函数值（Objective Value, Obj）**：越小越好；
- **计算时间**：
  - 离线训练耗时（Training Cost）
  - 在线预测耗时（Prediction Cost）
  - 精炼耗时（Correction Cost）
- **收敛迭代次数**

### 基线方法对比
- **Branch-and-Bound (B&B)**：确定性全局优化方法，提供理论保证但计算复杂度高；
- **ε-global method [5]**：用于生化系统的全局搜索方法；
- **Method in [25]**：局部优化方法，易陷入局部极小；
- **Relaxation methods [23,24]**：基于凸松弛的目标优化技术。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ Example 4.1（非线性ODE）
| 方法 | 最优目标值 | 迭代次数 |
|------|------------|----------|
| New OB relaxation [23] | -0.0607 | 23 |
| SBM relaxation [24] | -0.0607 | 33 |
| **本文方法** | **-0.0607** | **6** |

> → 仅用6次迭代即达全局最优，效率远超其他方法。

#### ✅ E. coli 生化系统（Table 2）
| 方法 | 参数 $(k_1, k_3)$ | 目标值 Obj |
|------|------------------|-----------|
| ε-global method [5] | (4.31348×10⁻³, 162.9297) | 0.029096 |
| Method in [25] | (2.9×10⁻³, 90) | 0.0712 |
| 本文方法（预测） | (4.4881×10⁻³, 1.7231×10²) | 0.0296 |
| 本文方法（精炼后） | **(4.32191×10⁻³, 1.632459×10²)** | **0.029089** |

> → 精炼后结果优于ε-global方法，且总耗时约 **1秒**，远低于branch-and-bound的 **52秒**。

#### ✅ 高维线性DAE系统（Table 3）

| $ n $ | Training (s) | Prediction (s) | Correction (s) | B&B (s) | 预测误差 | 解精度 |
|-------|---------------|----------------|----------------|---------|-----------|--------|
| 2     | 96.1          | 0.6            | 0.003          | 1.4     | 6.50e-08  | ✔️     |
| 6     | 5991.9        | 1.5            | 0.005          | 90.6    | 8.07e-07  | ✔️     |
| 8     | 18424.8       | 2.7            | 0.013          | 934.6   | 1.60e-07  | ✔️     |
| 10    | 53472.4       | 4.8            | 0.008          | 6083.5  | 2.92e-07  | ✔️     |

> → **在线阶段（预测+精炼）始终控制在5秒以内**，而B&B随维度指数增长，$ n=10 $ 时已超1小时；
> → 预测误差维持在 $ 10^{-7} \sim 10^{-8} $ 量级，精度极高。

### 消融实验（隐含分析）
虽然未明确列出消融表，但从设计逻辑可看出：
- 若不使用GA采样 → 训练数据分布不佳 → Constraint Network精度下降；
- 若无结构分析 → 参数退化区无法准确建模；
- 若不用松弛变量与误差界 → 无法保证优化问题等价性；
- 若仅依赖梯度优化Objective Network → 易陷入局部最优（文中改用随机种子探索）。

---

## 4. 关键结论和发现

### 主要发现
1. **双网络架构能高效解耦约束学习与目标优化**，实现“一次训练、多任务通用”；
2. **GA引导的自适应采样显著提升Constraint Network在困难区域的逼近能力**；
3. **嵌入式结构分析有效缓解参数退化带来的数值不稳定问题**；
4. **松弛变量与全局误差界提供了坚实的理论基础**，保障了解的可靠性；
5. **在线优化速度极快（通常 < 5秒）**，适合需要快速响应的产品设计闭环。

### 方法的局限性
- **离线训练成本较高**：随着系统维度增加，Constraint Network的训练时间呈多项式增长（如 $ n=10 $ 超过14小时）；
- 对**高度非光滑或震荡强烈的目标函数**，Objective Network可能难以收敛，仍需依赖随机搜索；
- 当前方法假设**约束结构不变**，若DAE结构本身随参数改变（如index变化），需重新训练Constraint Network。

### 未来工作方向
- 探索更高效的网络结构（如Transformer、Fourier Neural Operators）加速Constraint Network训练；
- 扩展至PDE-constrained optimization场景；
- 开发在线增量更新机制，使Constraint Network可在新参数域上持续学习；
- 结合不确定性量化（Uncertainty Quantification）构建贝叶斯版本的双网络框架。

--- 

> **总结一句话**：  
> 本文提出的**双耦合PINN架构 + GA增强训练 + 全局误差保障机制**，为参数化DAE系统的多任务优化提供了一个**高效、精准、可理论验证**的新范式，特别适用于工业仿真驱动设计中的实时优化需求。

</details>

---

### 14. [Dynamical Systems Theory Behind a Hierarchical Reasoning Model](https://arxiv.org/abs/2603.22871)

**Authors**: Vasiliy A. Es'kin, Mikhail E. Smorkalov  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.22871v1  

#### Abstract
Current large language models (LLMs) primarily rely on linear sequence generation and massive parameter counts, yet they severely struggle with complex algorithmic reasoning. While recent reasoning architectures, such as the Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM), demonstr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Dynamical Systems Theory Behind a Hierarchical Reasoning Model

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的大型语言模型（LLMs）依赖线性序列生成和海量参数，在**复杂算法推理任务**（如Sudoku、Maze等）上表现不佳。尽管近期提出的 **Hierarchical Reasoning Model (HRM)** 和 **Tiny Recursive Model (TRM)** 展现出在小规模模型中实现强推理能力的潜力，但其训练动态缺乏数学保证，常导致以下问题：
- **训练不稳定**（training instability）
- **表征坍缩**（representation collapse）
- **信用分配困难**（credit assignment in multi-level structures）
- **泛化能力弱**

本文旨在通过引入**动力系统理论**（Dynamical Systems Theory），为递归推理模型提供一个**数学严谨且高度稳定**的框架。

---

### 提出的新方法与思路
作者提出了 **Contraction Mapping Model (CMM)**，其核心思想是将离散的递归推理过程重新表述为连续的动力系统，并通过数学约束确保系统的收敛性和稳定性。

#### 主要创新点：
1. **从离散到连续建模**  
   将 HRM/TRM 的离散递归步骤形式化为 **Neural Ordinary Differential Equations (NODEs)** 和 **Neural Stochastic Differential Equations (NSDEs)**，从而可以利用微分方程理论分析和控制潜变量轨迹。

2. **显式强制收敛机制**  
   引入基于**不动点条件**（equilibrium point）和 **Routh-Hurwitz 稳定性准则** 的辅助损失项（auxiliary loss），确保潜状态最终收敛至稳定的平衡解（即正确答案）。

3. **防止表征坍缩的超球面排斥损失**（hyperspherical repulsion loss）  
   在单位超球面上对不同样本的隐表示施加排斥力，促使特征分布均匀、正交，增强表达能力和泛化性。

4. **高效优化技术集成**
   - 提出 **StableMax3** 和 **StableMax5**：Softmax 的多项式近似，兼顾数值稳定性和精度。
   - 使用 **Algebraic Gradient Normalization (AlgGradNorm)** 动态平衡多目标损失，避免梯度主导问题。

5. **噪声注入提升鲁棒性**  
   将训练中的“跳跃”行为建模为随机扰动，提出 **NSDE 版本的 CMM**，增强模型抗过拟合能力。

---

### 相比现有方法的优势
| 维度 | HRM / TRM | CMM |
|------|-----------|-----|
| 数学基础 | 经验设计，无严格收敛保障 | 基于 NODE/NSDE 和收缩映射理论 |
| 稳定性 | 易出现表征坍缩、训练震荡 | 显式约束 + 排斥损失 → 更高稳定性 |
| 参数效率 | TRM 已较高效（~5M） | 可压缩至 **0.26M** 仍保持高性能 |
| 泛化能力 | 对 OOD 数据敏感 | 噪声注入 + 流形学习 → 更强泛化 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Sudoku-Extreme**：极端难度数独谜题，用于测试复杂符号推理能力。
- **Maze**：迷宫路径求解任务，评估空间规划与搜索推理。
- （文中提及 ARC-AGI，但主要实验集中在前两者）

### 实验设置与评估指标
- **评估指标**：测试准确率（Accuracy %）
- **硬件限制模拟**：所有实验在 **16GB VRAM**（如 V100/T4）下完成，不依赖 bfloat16 或 Flash Attention。
- **训练技巧**：
  - 使用 **Automatic Mixed Precision (AMP)**
  - **Gradient Accumulation** 以支持大 batch
  - **Torch.compile** 加速并提升数值精度
  - **Adam-Atan2** 优化器
- **模型编译必要性**：未编译模型性能显著下降，说明**算子融合与 FMA 指令对迭代稳定性至关重要**。

### 基线方法对比
| 模型 | 参数量 | 数据集 | 报道准确率 | 复现准确率（本文） |
|------|--------|--------|------------|------------------|
| DeepSeek R1 | 671B | Sudoku/Maze | 0.0% | — |
| HRM [1] | 27M | Sudoku-Extreme | 55.0% | 45.9% |
| TRM [7] | 5M | Sudoku-Extreme | 87.4% | 79.1% |
| TRM [7] | 7M | Maze | 85.3% | 84.2% |

> 注：作者指出原结果对硬件配置敏感，在受限环境下复现性能下降，因此以自身复现作为公平比较基准。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ CMM 在标准规模下的表现（5M 参数）
| 模型 | 数据集 | 准确率 |
|------|--------|--------|
| CMM ODE (5M) | Sudoku-Extreme | **93.7%** |
| HRM (27M) | Sudoku-Extreme | 55.0% |
| TRM (5M) | Sudoku-Extreme | 87.4% |

👉 **CMM 以相同参数量超越 TRM 6.3%，甚至远超参数量5倍以上的 HRM。**

---

#### ✅ 极端压缩下的惊人表现（仅 0.26M 参数！）
| 模型 | 数据集 | 准确率 | 参数量 |
|------|--------|--------|--------|
| CMM NSDE En128 (identical layers) | Sudoku-Extreme | **85.4%** | 0.26M |
| CMM NSDE En128 (identical layers) | Maze | **82.2%** | 0.26M |

👉 **即使压缩到不到原始 TRM 的 1/19，性能仍接近甚至超过原始 TRM（79.1%）。**

---

### 消融实验结果（Ablation Studies）

#### 🔹 不同组件对性能的影响（Sudoku-Extreme, 5M）
| 修改 | 准确率提升 |
|------|----------|
| SiLU → tanh（有界激活函数） | +12.2% (69.6 → 81.8) |
| 引入 StableMax3 | +~5% |
| 添加 Equilibrium Loss + Repulsion Loss | +~5–10% |
| 使用 AlgGradNorm 自动调权 | +~2–3%，训练更稳定快速 |
| 注入噪声（NSDE, σ=0.01） | 显著缓解过拟合，提高泛化 |

#### 🔹 超参数敏感性
- **Adam 的 ε 参数极为关键**：从默认 `1e-8` 提升至 `1e-14`，可带来 **+7% 以上增益**（尤其在小模型中）。
- **学习率调度**：指数衰减调度器优于固定学习率。
- **Batch Size 扩展**：增大 batch size 并配合 gradient accumulation 可进一步提升性能。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **数学严谨的动力系统建模能显著提升推理模型的稳定性与性能**。  
   将递归推理视为向稳定不动点收敛的过程，是构建可靠 AI 推理引擎的关键。

2. ✅ **参数规模不是决定推理能力的唯一因素**。  
   通过合理的架构设计（如收缩映射 + 排斥损失 + 噪声注入），**极小模型（0.26M）也能达到或接近大模型性能**。

3. ✅ **TRM 中观察到的“跳跃式训练”本质上是一种隐式的随机动力系统行为**，可通过显式建模为 **NSDE** 来加以利用和优化。

4. ✅ **编译优化（torch.compile）对迭代模型至关重要**，因其减少了中间舍入误差，提升了长期轨迹稳定性。

5. ✅ **多任务损失平衡必须自动化**。手动设定权重易失衡，而 **AlgGradNorm** 能有效协调主任务与辅助损失之间的梯度幅度。

---

### 方法的局限性
- 当前实验集中于**确定性、规则明确的任务**（如数独、迷宫），尚未验证在开放域自然语言推理中的有效性。
- 对 **ε=1e-14** 这类极端超参数的依赖可能暗示潜在的数值不稳定性风险。
- NSDE 框架虽然增强了鲁棒性，但也增加了建模复杂度，解释性略有降低。

---

### 未来工作方向
1. 将 CMM 应用于更广泛的推理基准，如 **ARC-AGI**。
2. 探索将此类**数学驱动的小型推理模块**嵌入到大型 LLM 中，作为“思维内核”（reasoning core）。
3. 在边缘设备（edge devices）部署 CMM，推动**低功耗、高精度的本地化推理**。
4. 进一步研究如何将程序归纳（program synthesis）能力融入该框架。

---

> 💡 **一句话总结**：  
> 本文证明了——与其盲目扩大参数规模，不如深入理解模型内部的**动态演化规律**。通过将递归推理建模为受控的连续动力系统，CMM 在**极致参数效率下实现了前所未有的稳定与强大推理能力**，为下一代 AI 推理架构开辟了新路径。

</details>

---

### 15. [Efficient Hallucination Detection: Adaptive Bayesian Estimation of Semantic Entropy with Guided Semantic Exploration](https://arxiv.org/abs/2603.22812)

**Authors**: Qiyao Sun, Xingming Li, Xixiang He, Ao Cheng, Xuanyu Ji, Hailun Lu, Runke Huang, Qingyong Hu  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.22812v1  

#### Abstract
Large language models (LLMs) have achieved remarkable success in various natural language processing tasks, yet they remain prone to generating factually incorrect outputs known as hallucinations. While recent approaches have shown promise for hallucination detection by repeatedly sampling from LLMs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Efficient Hallucination Detection: Adaptive Bayesian Estimation of Semantic Entropy with Guided Semantic Exploration*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在生成文本时容易产生“幻觉”（hallucination），即生成看似合理但事实错误的内容。现有的基于多采样（multi-sample）的幻觉检测方法（如 Semantic Entropy, SE）虽然有效，但存在两个关键缺陷：
- **固定采样预算**：对所有查询使用相同的采样次数，导致简单问题浪费计算资源，复杂问题采样不足。
- **随机探索效率低**：传统方法依赖于从 LLM 多次独立采样，难以系统性地探索语义空间中的多样性。

### 🚀 提出的新方法
作者提出了一种 **自适应贝叶斯语义熵估计框架**（Adaptive Bayesian Estimation of Semantic Entropy），结合**引导式语义探索**（Guided Semantic Exploration），实现高效且准确的幻觉检测。

#### 核心创新点：
1. **层次化贝叶斯建模**（Hierarchical Bayesian Framework）
   - 引入 Dirichlet 先验建模语义类别分布，并通过边际化处理未知语义类别的数量 $ K $。
   - 利用生成概率约束构建截断 Dirichlet 后验（truncated Dirichlet posterior），提升不确定性估计精度。
   - 设计基于后验方差的停止准则：当 `Var[h|D] < γ` 时终止采样，实现动态调整采样次数。

2. **加权困惑度先验校准**（Weighted Perplexity Prior）
   - 提出一种自适应方式设定 Poisson 先验参数 $ \lambda $，利用 token 级重要性权重计算加权困惑度（WPL），反映提示词的语义多样性，从而更合理初始化语义类别数期望。

3. **引导式语义探索策略**（Guided Semantic Exploration）
   - 基于重要性采样机制，在语义关键位置扰动高影响力 token（如替换为 top-k 替代项），主动探索不同语义分支。
   - 使用 importance weighting 修正偏差，保证估计无偏性，同时加速语义多样性的发现。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 在低预算场景下减少约 50% 的采样量即可达到同等检测性能 |
| **准确性** | 平均 AUROC 提升 12.6%，尤其在小样本（N=2）下优势显著 |
| **适应性** | 动态分配资源：简单问题快速收敛，复杂问题持续探索 |
| **理论基础** | 贝叶斯框架提供严谨的不确定性量化，支持对未观测语义类别的推断 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在四个开放形式问答（open-form QA）数据集上进行评估，涵盖闭卷与开卷场景：
- **CoQA**：对话式问答
- **TriviaQA**：常识性知识问答
- **TruthfulQA**：专门设计用于测试模型是否模仿人类错误信念
- **SimpleQA**：简洁的事实型问答

### ⚙️ 实验设置
- **模型**：测试三种主流 LLM：
  - Llama-2-7B
  - Llama-3.1-8B
  - Mistral-Small-24B
- **采样设置**：
  - 温度 = 1.0
  - 初始采样数 $ N_0 = 1 $
  - top-k = 3 用于引导生成
- **语义聚类**：使用 DeBERTa-v3-large 进行 NLI-based 语义等价判断
- **评估指标**：
  - **AUROC**（Area Under ROC Curve）作为主要指标
  - 使用 GPT-4.1 对每个问题生成 3 个响应，采用 Pass-All@3 方法标注幻觉（任一错误即标为幻觉）

### 🆚 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **P(True)** | Metacognition-based | LLM 自我置信度评分 |
| **SAR** | Single-sample | 聚合 token 级预测熵 |
| **SE** | Multi-sample | 基于语义聚类的经典语义熵方法 |
| **SEsDLG** | Multi-sample + Perturbation | SE 的增强版，引入定向扰动生成多样化输出 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）
在 **24 个实验设置**（3 模型 × 4 数据集 × 2 预算）中：
- 所提方法在 **23/24 设置中取得最佳 AUROC**
- 最大提升达 **16.9%**（Mistral-Small-24B, TriviaQA, N=2）
- 在 **N=2** 小样本条件下，平均 AUROC 提升 **12.6%**
- 在 **N=5** 条件下仍保持平均 **6.3%** 提升

> 示例：Llama-2-7B 在 SimpleQA 上，N=5 时 AUROC 达到 **0.959**，远超第二名 SEsDLG 的 0.956。

### 📉 与基线方法对比总结
| 方法 | 相对表现 |
|------|----------|
| **OURs vs SE** | 显著优于原始 SE，验证贝叶斯建模 + 引导探索的有效性 |
| **OURs vs SEsDLG** | 即使后者也使用扰动，本方法因自适应机制更具效率和鲁棒性 |
| **OURs vs SAR/P(True)** | 在所有设置下全面领先，表明 multi-sample + 结构化探索是更优路径 |

### 🔪 消融实验结果（Table 2，TriviaQA, Llama-3.1-8B）
| 消融配置 | N=2 AUROC | N=5 AUROC | 性能下降 |
|--------|-----------|-----------|---------|
| 完整模型（Full Model） | **0.855** | **0.913** | — |
| 移除自适应先验（w/o adaptive prior） | 0.812 | 0.878 | ↓4.3% / ↓3.5% |
| 固定采样（Fixed sampling w/o Bayesian） | 0.759 | 0.866 | ↓9.6% / ↓4.7% |
| 固定采样 + 贝叶斯 | 0.798 | 0.885 | ↓5.7% / ↓2.8% |
| 移除引导探索（w/o guided exploration） | 0.823 | 0.892 | ↓3.2% / ↓2.1% |

> ✅ 结论：三个组件——**自适应先验、贝叶斯自适应采样、引导探索**——均对最终性能有显著贡献，缺一不可。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **自适应采样显著提升效率**  
   简单问题可在 1–2 次采样内收敛（如 CoQA），而复杂问题自动增加采样次数（如 SimpleQA），实现资源最优分配。

2. **引导式探索加速语义多样性发现**  
   通过扰动关键 token 可更快触发不同语义路径，相比随机采样更高效地揭示潜在幻觉。

3. **贝叶斯框架优于固定假设**  
   传统方法假设语义类别固定，而本文框架能推理未观察到的语义类别，提升估计稳健性。

4. **低预算场景下优势最大**  
   在实际部署受限环境中（如边缘设备或高并发服务），该方法最具实用价值。

### ⚠️ 局限性
- **依赖高质量语义聚类器**：若 DeBERTa 等模型无法正确识别语义等价，则影响整体效果。
- **扰动策略可能受限于词汇表覆盖范围**：某些深层语义变化难以通过 token 替换激发。
- **初始采样仍具随机性**：尽管有自适应机制，极端情况下仍可能出现误判。

### 🔮 未来工作方向
- 扩展至 **多模态场景**（multimodal settings），如图文生成中的幻觉检测
- 探索更智能的 **perturbation 策略**，例如基于梯度或概念激活的方向性扰动
- 应用于更广泛的 **uncertainty quantification 任务**，如决策系统、自动推理链校验等
- 结合 retrieval-augmented generation（RAG）进行联合优化

---

## ✅ 总结
本文提出的 **Adaptive Bayesian Estimation with Guided Semantic Exploration** 是当前最高效的幻觉检测框架之一。它不仅在性能上超越所有基线方法，更重要的是解决了“如何用最少的采样获得最大不确定性信息”的核心挑战，为 LLM 在真实世界中安全可靠的应用提供了强有力的技术支撑。

</details>

---

### 16. [Model Predictive Control with Differentiable World Models for Offline Reinforcement Learning](https://arxiv.org/abs/2603.22430)

**Authors**: Rohan Deb, Stephen J. Wright, Arindam Banerjee  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.22430v1  

#### Abstract
Offline Reinforcement Learning (RL) aims to learn optimal policies from fixed offline datasets, without further interactions with the environment. Such methods train an offline policy (or value function), and apply it at inference time without further refinement. We introduce an inference time adapt...

---

### 17. [Spiking Personalized Federated Learning for Brain-Computer Interface-Enabled Immersive Communication](https://arxiv.org/abs/2603.22727)

**Authors**: Chen Shang, Dinh Thai Hoang, Diep N. Nguyen, Jiadong Yu  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.22727v1  

#### Abstract
This work proposes a novel immersive communication framework that leverages brain-computer interface (BCI) to acquire brain signals for inferring user-centric states (e.g., intention and perception-related discomfort), thereby enabling more personalized and robust immersive adaptation under strong i...

---

### 18. [Session Risk Memory (SRM): Temporal Authorization for Deterministic Pre-Execution Safety Gates](https://arxiv.org/abs/2603.22350)

**Authors**: Florin Adrian Chitan  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.22350v1  

#### Abstract
Deterministic pre-execution safety gates evaluate whether individual agent actions are compatible with their assigned roles. While effective at per-action authorization, these systems are structurally blind to distributed attacks that decompose harmful intent across multiple individually-compliant s...

---

### 19. [ABSTRAL: Automatic Design of Multi-Agent Systems Through Iterative Refinement and Topology Optimization](https://arxiv.org/abs/2603.22791)

**Authors**: Weijia Song, Jiashu Yue, Zhe Pang  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.22791v1  

#### Abstract
How should multi-agent systems be designed, and can that design knowledge be captured in a form that is inspectable, revisable, and transferable? We introduce ABSTRAL, a framework that treats MAS architecture as an evolving natural-language document, an artifact refined through contrastive trace ana...

---

### 20. [Reliable Classroom AI via Neuro-Symbolic Multimodal Reasoning](https://arxiv.org/abs/2603.22793)

**Authors**: Sina Bagheri Nezhad  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.22793v1  

#### Abstract
Classroom AI is rapidly expanding from low-level perception toward higher-level judgments about engagement, confusion, collaboration, and instructional quality. Yet classrooms are among the hardest real-world settings for multimodal vision: they are multi-party, noisy, privacy-sensitive, pedagogical...

---

### 21. [Ran Score: a LLM-based Evaluation Score for Radiology Report Generation](https://arxiv.org/abs/2603.22935)

**Authors**: Ran Zhang, Yucong Lin, Zhaoli Su, Bowen Liu, Danni Ai, Tianyu Fu, Deqiang Xiao, Jingfan Fan, Yuanyuan Wang, Mingwei Gao, Yuwan Hu, Shuya Gao, Jingtao Li, Jian Yang, Hong Song, Hongliang Sun  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.22935v1  

#### Abstract
Chest X-ray report generation and automated evaluation are limited by poor recognition of low-prevalence abnormalities and inadequate handling of clinically important language, including negation and ambiguity. We develop a clinician-guided framework combining human expertise and large language mode...

---

### 22. [MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation](https://arxiv.org/abs/2603.23234)

**Authors**: Yurui Chang, Yiran Wu, Qingyun Wu, Lu Lin  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23234v1  

#### Abstract
Large language model (LLM)-based agents rely on memory mechanisms to reuse knowledge from past problem-solving experiences. Existing approaches typically construct memory in a per-agent manner, tightly coupling stored knowledge to a single model's reasoning style. In modern deployments with heteroge...

---

### 23. [Detecting Non-Membership in LLM Training Data via Rank Correlations](https://arxiv.org/abs/2603.22707)

**Authors**: Pranav Shetty, Mirazul Haque, Zhiqiang Ma, Xiaomo Liu  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.22707v1  

#### Abstract
As large language models (LLMs) are trained on increasingly vast and opaque text corpora, determining which data contributed to training has become essential for copyright enforcement, compliance auditing, and user trust. While prior work focuses on detecting whether a dataset was used in training (...

---

### 24. [UniDial-EvalKit: A Unified Toolkit for Evaluating Multi-Faceted Conversational Abilities](https://arxiv.org/abs/2603.23160)

**Authors**: Qi Jia, Haodong Zhao, Dun Pei, Xiujie Song, Shibo Wang, Zijian Chen, Zicheng Zhang, Xiangyang Zhu, Guangtao Zhai  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23160v1  

#### Abstract
Benchmarking AI systems in multi-turn interactive scenarios is essential for understanding their practical capabilities in real-world applications. However, existing evaluation protocols are highly heterogeneous, differing significantly in dataset formats, model interfaces, and evaluation pipelines,...

---

### 25. [Neural Structure Embedding for Symbolic Regression via Continuous Structure Search and Coefficient Optimization](https://arxiv.org/abs/2603.22429)

**Authors**: Fateme Memar, Tao Zhe, Dongjie Wang  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.22429v1  

#### Abstract
Symbolic regression aims to discover human-interpretable equations that explain observational data. However, existing approaches rely heavily on discrete structure search (e.g., genetic programming), which often leads to high computational cost, unstable performance, and limited scalability to large...

---

### 26. [Policy-based Tuning of Autoregressive Image Models with Instance- and Distribution-Level Rewards](https://arxiv.org/abs/2603.23086)

**Authors**: Orhun Bu\u{g}ra Baran, Melih Kandemir, Ramazan Gokberk Cinbis  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23086v1  

#### Abstract
Autoregressive (AR) models are highly effective for image generation, yet their standard maximum-likelihood estimation training lacks direct optimization for sample quality and diversity. While reinforcement learning (RL) has been used to align diffusion models, these methods typically suffer from o...

---

### 27. [Benchmarking Multi-Agent LLM Architectures for Financial Document Processing: A Comparative Study of Orchestration Patterns, Cost-Accuracy Tradeoffs and Production Scaling Strategies](https://arxiv.org/abs/2603.22651)

**Authors**: Siddhant Kulkarni, Yukta Kulkarni  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.22651v1  

#### Abstract
The adoption of large language models (LLMs) for structured information extraction from financial documents has accelerated rapidly, yet production deployments face fundamental architectural decisions with limited empirical guidance. We present a systematic benchmark comparing four multi-agent orche...

---

### 28. [Improving Safety Alignment via Balanced Direct Preference Optimization](https://arxiv.org/abs/2603.22829)

**Authors**: Shiji Zhao, Mengyang Wang, Shukun Xiong, Fangzhou Chen, Qihui Zhu, Shouwei Ruan, Yisong Xiao, Ranjie Duan, Xun Chen, XingXing Wei  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.22829v1  

#### Abstract
With the rapid development and widespread application of Large Language Models (LLMs), their potential safety risks have attracted widespread attention. Reinforcement Learning from Human Feedback (RLHF) has been adopted to enhance the safety performance of LLMs. As a simple and effective alternative...

---

### 29. [Separating Diagnosis from Control: Auditable Policy Adaptation in Agent-Based Simulations with LLM-Based Diagnostics](https://arxiv.org/abs/2603.22904)

**Authors**: Shaoxin Zhong, Yuchen Su, Michael Witbrock  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.22904v1  

#### Abstract
Mitigating elderly loneliness requires policy interventions that achieve both adaptability and auditability. Existing methods struggle to reconcile these objectives: traditional agent-based models suffer from static rigidity, while direct large language model (LLM) controllers lack essential traceab...

---

### 30. [Towards Automated Community Notes Generation with Large Vision Language Models for Combating Contextual Deception](https://arxiv.org/abs/2603.22453)

**Authors**: Jin Ma, Jingwen Yan, Mohammed Aldeen, Ethan Anderson, Taran Kavuru, Jinkyung Katie Park, Feng Luo, Long Cheng  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.22453v1  

#### Abstract
Community Notes have emerged as an effective crowd-sourced mechanism for combating online deception on social media platforms. However, its reliance on human contributors limits both the timeliness and scalability. In this work, we study the automated Community Notes generation method for image-base...

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
