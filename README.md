# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-23 06:56:07 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [NASimJax: GPU-Accelerated Policy Learning Framework for Penetration Testing](https://arxiv.org/abs/2603.19864)

**Authors**: Raphael Simon, Jos\'e Carrasquel, Wim Mees, Pieter Libin  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2603.19864v1  

#### Abstract
Penetration testing, the practice of simulating cyberattacks to identify vulnerabilities, is a complex sequential decision-making task that is inherently partially observable and features large action spaces. Training reinforcement learning (RL) policies for this domain faces a fundamental bottlenec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：NASimJax: GPU-Accelerated Policy Learning Framework for Penetration Testing**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前基于强化学习（RL）的自动化渗透测试面临两大瓶颈：
- **训练效率低下**：现有模拟器（如 NASim、CyberBattleSim）基于 CPU 和 Python 实现，环境交互速度慢，无法充分利用现代 GPU 加速硬件，导致大规模训练不可行。
- **泛化能力差**：大多数模拟器仅支持固定或参数受限的网络场景，缺乏结构性多样性，导致训练出的策略难以在未见过的网络拓扑中实现零样本迁移（Zero-Shot Policy Transfer, ZSPT）。

### **提出了什么新方法或新思路**
本文提出 **NASimJax**，一个完全基于 JAX 的高性能渗透测试 RL 框架，其核心创新包括：

- **全栈硬件加速**：将整个训练流程（包括环境模拟和策略更新）迁移到 JAX 上，利用 JIT 编译、`vmap` 向量化等特性，在单个入门级 GPU 上实现高达 **100× 的吞吐量提升**。
- **Contextual POMDP 建模**：将渗透测试形式化为 **Contextual POMDP**，每个 episode 对应一个由随机生成的网络实例定义的“上下文”（context），从而系统性研究跨网络结构的零样本泛化。
- **可配置且保证可解的网络生成器**：引入一套可控的网络生成流程，通过调节 `topology density (ta)`、`service density` 等参数，生成结构多样、现实合理且**保证可解**的网络场景。
- **两阶段动作选择机制（2SAS）**：针对随主机数量线性增长的动作空间，提出 Two-Stage Action Selection（2SAS），先选择目标主机，再在其上选择具体攻击动作，显著降低探索复杂度。

### **相比现有方法的优势**
| 维度 | NASimJax | 传统模拟器（如 NASim、CyberBattleSim） |
|------|--------|----------------------------|
| 性能 | 最高 **1.6M steps/sec**（GPU） | 数千步/秒（CPU） |
| 可扩展性 | 支持最大 **40 主机**网络 | 多数 ≤16 主机 |
| 泛化支持 | 内建分布式训练与 ZSPT 能力 | 固定或窄分布场景 |
| 动作空间处理 | 引入 **2SAS** 分解机制 | 平坦动作掩码（Flat Action Masking） |
| 开放性 | 完全开源，兼容 Gymnax API | 部分闭源或接口不统一 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **无真实世界数据集依赖**：所有实验均基于 NASimJax 自带的**程序化生成网络环境**。
- 网络规模覆盖三种配置：
  - **16 hosts**（7 subnets）
  - **26 hosts**（10 subnets）
  - **40 hosts**（16 subnets）
- 每种规模下通过调整 `topology density (ta)` 构建不同难度的训练/测试分布，用于评估零样本迁移。

### **实验设置和评估指标**
#### **训练设置**
- 使用 **PPO** 作为基础 RL 算法（PureJAX 实现）。
- 所有实验运行在 **Intel Xeon Gold 6230R + NVIDIA RTX A4000** 上。
- 环境并行数：最多 **4096 个向量化环境**。
- 训练预算：
  - 16-host: 100M steps
  - 26-host: 500M steps
  - 40-host: 1B steps

#### **评估指标**
- **Solve Rate（解决率）**：在一组独立生成的评估网络中，成功攻陷所有敏感主机的比例。
- 报告 **5 次随机种子平均值 + 95% 置信区间 / IQR**。
- 零样本迁移任务中，训练与测试使用不同的 `ta` 分布。

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **DR-Masked** | Domain Randomization + Flat Action Masking |
| **DR-2SAS** | Domain Randomization + Two-Stage Action Selection |
| **PLR-Masked / PLR--Masked** | Prioritized Level Replay（含梯度更新控制）+ 掩码 |
| **PLR-2SAS / PLR--2SAS** | PLR + 2SAS 组合 |

此外还进行了消融实验，验证 2SAS、奖励归一化、PLR 参数的影响。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | 结果 |
|------|------|
| **最大吞吐量** | **1.6M steps/sec**（NASimJax vs ~16k steps/sec for NASim）→ **100× 加速** |
| **40-host Solve Rate (最优)** | **42%**（PLR--2SAS），而基线仅 14% |
| **动作空间大小** | 40-host 网络可达 **640 维离散动作空间** |

### **与基线方法的对比结果**

#### **(1) 动作空间扩展性能（Fig. 4）**
| 方法 | 16-host | 26-host | 40-host |
|------|-------|--------|--------|
| **Flat Masking** | ~90% | 66% | **14%** |
| **2SAS** | ~90% | **82%** | **42%** |

✅ **结论**：随着网络规模扩大，2SAS 显著优于平坦掩码，尤其在 40-host 场景下性能差距达 **3 倍以上**。

#### **(2) 零样本迁移性能（Fig. 5）**
- 在 26-host 和 40-host 上：
  - **训练于低密度拓扑（low `ta`）效果最好**，即使测试于更高密度网络也能保持良好表现 → 存在**隐式课程学习（implicit curriculum）**。
  - **PLR 类方法整体优于 DR**，尤其是在高密度训练时更稳定。
  - **DR 在高密度训练时性能急剧下降**，表明其对训练分布敏感。

#### **(3) 特殊失败模式识别**
- **PLR-2SAS 在 40-host, `ta=0.15` 下完全崩溃（solve rate ≈ 0）**
- 原因分析（见第 7 节）：
  1. **PLR 的 episode reset 行为** 导致长周期任务无法跨 rollout 积累进度；
  2. **2SAS 的信用分配结构** 使得错误决策被双向惩罚（host head & action head）；
  3. 成功轨迹稀少 → **replay buffer 无法填充** → 优先回放失效。

> ✅ 但 **PLR--2SAS**（关闭探索分支梯度更新）避免了该问题，说明可通过架构设计规避此风险。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **硬件加速至关重要**：  
   将 RL 环境全面迁移到 JAX 可带来 **近两个数量级的速度提升**，使大规模、高并发训练成为可能。

2. ✅ **2SAS 是应对大规模动作空间的有效方案**：  
   在超过 26 主机的环境中，2SAS 明显优于传统 action masking，是实现可扩展性的关键技术。

3. ✅ **低密度训练促进更好泛化**：  
   在稀疏拓扑上训练能形成**隐式课程**，显著提升对密集网络的零样本迁移能力 —— 这是一个**实用且经济高效的训练策略**（sparse 更便宜）。

4. ✅ **PLR 比 DR 更适合复杂分布学习**：  
   PLR 能自动构建有意义的学习课程，在多种拓扑上表现更鲁棒，尤其适用于高难度任务。

5. ⚠️ **存在方法间交互的失败模式**：  
   PLR 与 2SAS 的组合在大尺度下会因 credit assignment 和 episode reset 的冲突而崩溃，揭示了当前 UED 方法与分层策略结合时的设计挑战。

### **方法的局限性**
- **依赖程序化生成**：尚未接入真实企业网络拓扑，泛化到真实世界的有效性仍需验证。
- **部分机制假设理想观测**：虽然建模为 POMDP，但仍假设扫描结果完全准确，未考虑噪声或欺骗机制（如蜜罐）。
- **2SAS 增加模型复杂度**：需要双头策略网络和联合损失函数设计，调参难度上升。

### **未来工作方向**
1. **集成真实网络数据源**：结合 CVE 数据库、Nmap 扫描报告等增强现实性。
2. **引入防御方建模**：构建攻防对抗环境（Adversarial RL），例如动态防火墙、入侵检测系统（IDS）响应。
3. **改进 credit assignment 机制**：设计更精细的反向传播策略，避免 2SAS 中的误惩罚问题。
4. **探索其他 UED 方法**：如 PAIRED、ALP-GMM，进一步优化课程生成。
5. **多智能体扩展**：支持多个攻击 agent 协同作战。

---

> **总结一句话**：  
> NASimJax 不仅是一个更快的模拟器，更是首个支持**大规模、可泛化、可复现**的渗透测试 RL 研究平台，推动该领域从“玩具实验”迈向真正的科学基准。

</details>

---

### 2. [BEAVER: A Training-Free Hierarchical Prompt Compression Method via Structure-Aware Page Selection](https://arxiv.org/abs/2603.19635)

**Authors**: Zhengpei Hu, Kai Li, Dapeng Fu, Chang Zeng, Yue Li, Yuanhao Tang, Jianqiang Huang  
**Category**: cs.CL  
**Published**: 2026-03-23  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2603.19635v1  

#### Abstract
The exponential expansion of context windows in LLMs has unlocked capabilities for long-document understanding but introduced severe bottlenecks in inference latency and information utilization. Existing compression methods often suffer from high training costs or semantic fragmentation due to aggre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BEAVER: A Training-Free Hierarchical Prompt Compression Method via Structure-Aware Page Selection

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着大语言模型（LLMs）的上下文窗口（context window）从数万扩展到百万级 token，虽然增强了长文本理解能力，但也带来了两大瓶颈：
- **计算墙（Computation Wall）**：自注意力机制的 `O(L)` 复杂度导致推理延迟显著增加，尤其是首 token 时间（time to first token）和尾延迟（tail latency）。
- **信息利用率下降（Diminishing Returns）**：上下文越长，模型越容易“迷失在中间”（Lost in the Middle），忽略关键信息。

现有 prompt compression 方法存在两个主要缺陷：
1. **依赖训练的方法**（如 LLMLingua-2）部署成本高、泛化性差；
2. **基于 token 的粗暴剪枝**破坏语义连贯性和句法完整性，影响下游任务表现。

---

### ✅ 提出的新方法：BEAVER
BEAVER 是一种**无需训练**（training-free）、**结构感知**（structure-aware）的分层 prompt 压缩框架，其核心思想是将传统的线性 token 剪枝转变为**基于页面的层级选择**。

#### 创新架构三大组件：
| 组件 | 功能 |
|------|------|
| **Segmenter** | 将原始文本按自然分隔符（如换行、标题）切分为逻辑段落，并打包为固定大小的“页”（page），形成二维张量结构，提升 GPU 并行效率并保留局部语义边界。 |
| **PageEncoder** | 使用双路径池化（dual-path pooling）编码每一页：<br>- **加权平均池化**：捕获全局语义<br>- **最大池化**：捕捉稀有关键词等局部高亮特征<br>引入 **In-Context ITF 权重** 抑制高频无意义词的影响。 |
| **QueryPlanner** | 结合语义与词法匹配，通过混合得分筛选重要页面：<br>- **语义分支**：基于嵌入的余弦相似度（加权 ITF）<br>- **词法分支**：精确 token 匹配<br>融合后结合三种**结构先验**构建最终压缩上下文：<br>1. **Anchor Pages**：保留前 k 页元信息（如标题）<br>2. **Flow Pages**：保留查询前连续窗口以维持局部记忆<br>3. **Flash Pages**：选择全局最高分页作为远距离证据 |

最后通过 **sentence-level smoothing** 扩展选中片段至完整句子边界，确保语法完整性。

---

### ✅ 相比现有方法的优势
| 特性 | BEAVER | 传统方法（如 LongLLMLingua, LLMLingua-2） |
|------|--------|------------------------------------------|
| 是否需要训练 | ❌ 否（Training-Free） | ✅ 是（需微调或蒸馏） |
| 泛化能力 | 强，跨模型尺度稳定 | 弱，依赖特定模型对齐 |
| 语义完整性 | 高，保留句子结构和逻辑流 | 中低，易产生碎片化 |
| 推理效率 | 极高，支持张量并行 | 较低，常为序列处理 |
| 超参数依赖 | 中等（如页面大小 M=64） | 高（需调优压缩策略） |

---

## 2. 核心实验方法和设置

### ✅ 数据集
在四个具有挑战性的长上下文基准上进行全面评估：

| 数据集 | 类型 | 任务 | 上下文长度范围 | 特点 |
|-------|------|------|----------------|------|
| **LongBench** (Bai et al., 2024) | 多语言多任务 | 单/多文档 QA、摘要、少样本学习、代码补全 | 1k–22k tokens | 综合性强，覆盖中英文及代码 |
| **ZeroSCROLLS** (Shaham et al., 2023) | 零样本长文本理解 | 摘要、问答、聚合、推理 | ~10k words | 真实长文档，强调零样本泛化 |
| **RULER** (Hsieh et al., 2024) | 合成可控测试 | 多针检索（Multi-Needle）、变量追踪、关键词提取、QA | 4k–128k tokens | 可控注入关键信息，精准衡量保留能力 |
| **L-Eval** (An et al., 2024) | 高质量人工标注 | 法律合同、小说、学术 QA 等 | 最长达 62k tokens | 强调真实世界复杂推理场景 |

---

### ✅ 实验设置与评估指标
- **目标模型**：统一使用 `gpt-3.5-turbo-instruct` 进行下游任务推理。
- **压缩预算**：严格控制压缩后 token 数为 2,000 或 3,000。
- **评估指标**：
  - **准确性（Accuracy / F1 / Rouge-L）**：根据任务类型采用标准指标。
  - **延迟（Latency）与加速比（Speedup）**：测量压缩阶段耗时，在 NVIDIA A100 (80GB) 上运行。
  - **Token 保留率（1/T）**：衡量压缩强度。

---

### ✅ 基线方法对比
分为三类进行公平比较：

| 类别 | 代表方法 |
|------|---------|
| **(1) 无监督统计方法** | Selective-Context, LongLLMLingua |
| **(2) 有监督/专用学习方法** | LLMLingua, LLMLingua-2, SBERT, OpenAI Embeddings |
| **(3) 零开销方法（本文提出）** | **BEAVER (ours)** |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总

#### 🔹 在 **LongBench** 上的表现（2k token 预算）
| 方法 | 单文档 QA | 多文档 QA | 总体平均 | Latency (s) | Speedup |
|------|-----------|------------|----------|-------------|---------|
| LongLLMLingua | 39.0 | 42.2 | 48.0 | 6.1 | 2.6x |
| LLMLingua-2 | 29.8 | 33.1 | 39.1 | 3.7 | 4.2x |
| **BEAVER (ours)** | **40.7** | **37.6** | **42.2** | **3.0** | **5.2x** |

> ✅ BEAVER 在单文档 QA 上达到新的 SOTA（40.7），同时实现最低延迟和最高加速比。

---

#### 🔹 在 **RULER** 上的多针检索能力（16k context, 3k budget）
| 方法 | 单针准确率 | 多针准确率 | 平均得分 |
|------|------------|------------|----------|
| LongLLMLingua | 100.0% | 9.0% | 28.8 |
| LLMLingua-2 | 27.0% | 72.0% | 47.9 |
| **BEAVER (ours)** | **100.0%** | **99.0%** | **83.7** |

> ✅ BEAVER 几乎完美保持了多针信息的召回能力，远超所有基线，验证其对“Lost in the Middle”的缓解效果。

---

#### 🔹 在 **L-Eval** 上的跨域泛化能力（2k token 预算）
| 方法 | 平均得分 |
|------|----------|
| LongLLMLingua | 51.5 |
| LLMLingua-2 | 54.6 |
| **BEAVER (ours)** | **57.6** |

> ✅ 在法律、金融、小说等复杂领域表现最优，尤其在 SFictionQA 和 Legal Contract QA 上优势明显。

---

#### 🔹 效率分析：128k 上下文下的压缩延迟
| 方法 | 压缩时间（秒） | 加速比 |
|------|----------------|--------|
| LongLLMLingua | ~31.7 s | 1.0x |
| LLMLingua-2-small | ~5.4 s | 5.9x |
| **BEAVER (ours)** | **~1.2 s** | **26.4x** |

> ✅ 在超长上下文（128k）下，BEAVER 实现 **26.4倍速度提升**，且延迟增长更平缓，具备极佳可扩展性。

---

### ✅ 消融实验结果（Ablation Study）
在 LongBench QA 子集上进行消融，关键发现如下：

| 配置修改 | 性能下降 Δ | 说明 |
|--------|------------|------|
| 移除 Max-Pooling | -2.7 | 双路径互补，缺失局部激活检测能力 |
| 移除 Mean-Pooling | -2.6 | 损失全局语义整合 |
| 移除 Multi-Token Query | -2.9 | 细粒度查询交互至关重要 |
| 移除 ITF 权重 | -2.7 | 上下文频率加权有效抑制噪声 |
| 仅用 Semantic 匹配 | -6.0 | 词法匹配对精确信息定位不可或缺 |
| 仅用 Lexical 匹配 | -3.1 | 语义匹配提供必要泛化能力 |
| 移除 Sentence Smoothing | -1.6 | 修复截断句边界显著提升连贯性 |
| 仅用 Anchor Pages | -21.7 | 单一策略无法兼顾全局与局部 |

> ✅ 验证了各模块设计的有效性，特别是**混合匹配机制**和**三级结构先验**的协同作用。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **层级页面选择优于线性 token 剪枝**：将文本组织为 page tensor 不仅提升硬件并行效率，还能天然保留语义块结构。
2. **无需训练即可实现高性能压缩**：利用预训练模型的 embedding 和统计信号（如 ITF），BEAVER 在多个任务上媲美甚至超越有监督方法。
3. **结构先验显著增强鲁棒性**：Anchor + Flow + Flash 的组合模仿人类阅读认知过程，有效防止关键信息遗漏。
4. **极高的推理效率与可扩展性**：在 128k 上下文下实现 **26.4× 加速**，适合高吞吐应用场景。
5. **强大的跨模型泛化能力**：在 Qwen3 系列（0.6B–32B）上均表现出色，性能保留率达 84%–98%，远超依赖外部训练的方法。

---

### ⚠️ 局限性
1. **粒度限制**：页面级操作不如 token 级精细，可能保留少量冗余内容。
2. **深层多跳推理挑战**：当支持证据与查询无直接表面重叠时，依赖语义/词法相似性的机制可能失效。
3. **超参数敏感性**：虽无需训练，但如页面大小 `M`、融合权重 `λ` 等仍需手动调整以适应不同领域。

---

### 🔮 未来工作方向
- 探索动态页面划分策略，替代固定大小分页。
- 引入轻量级适配机制，自动优化超参数配置。
- 研究公平性问题：避免因 embedding 偏见导致少数群体文本被误判为“冗余”而过滤。
- 扩展至多模态 prompt 压缩场景（如图文混合输入）。

---

## 总结
**BEAVER** 提出了一种全新的、无需训练的 prompt 压缩范式，通过**结构感知的层级页面选择机制**，在保证语义完整性和推理准确性的前提下，实现了前所未有的**高效性与可扩展性**。它不仅在多项长上下文任务上达到 SOTA 表现，还在极端压缩条件下展现出卓越的信息保留能力，为大规模长文档理解提供了实用且稳健的解决方案。

</details>

---

### 3. [Federated Hyperdimensional Computing for Resource-Constrained Industrial IoT](https://arxiv.org/abs/2603.20037)

**Authors**: Nikita Zeulin, Olga Galinina, Nageen Himayat, Sergey Andreev  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.20037v1  

#### Abstract
In the Industrial Internet of Things (IIoT) systems, edge devices often operate under strict constraints in memory, compute capability, and wireless bandwidth. These limitations challenge the deployment of advanced data analytics tasks, such as predictive and prescriptive maintenance. In this work, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Federated Hyperdimensional Computing for Resource-Constrained Industrial IoT*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **Industrial Internet of Things (IIoT)** 系统中边缘设备普遍存在的**资源受限**问题（如计算能力弱、内存小、电池容量有限、无线带宽紧张），探索如何在这些设备上实现高效、可扩展且隐私保护的分布式机器学习。

传统联邦学习（Federated Learning, FL）虽然能保护数据隐私，但其主流方法（如基于梯度的 FedAvg）依赖复杂的神经网络模型，通信开销大、训练能耗高，难以部署在低功耗 IIoT 设备上。此外，轻量级模型（如 XGBoost、SVM）在分布式场景下仍面临高通信成本或计算复杂度问题。

### 🚀 提出的新方法与创新思路
作者提出了一种结合 **Hyperdimensional Computing (HDC)** 与 **Federated Learning (FL)** 的新型框架——**Federated HDC**，其核心创新包括：

- **采用 HDC 替代传统 NN 模型作为基础学习范式**  
  利用高维向量（hypervectors）进行信息编码和分类，通过简单的向量操作（如 bundling 和 binding）完成训练与推理，避免了矩阵乘法等高耗能运算。

- **原型聚合（prototype aggregation）代替梯度交换**  
  在 FL 框架中，各设备仅上传代表各类别的 prototype hypervectors 给中心服务器进行聚合，显著降低通信负载，并天然支持隐私保护（不传输原始数据或梯度）。

- **引入随机子模型重训练机制（Randomized HDC Sub-model Retraining）**  
  每轮本地训练只更新一个随机选择的 HDC 子模型（即部分维度），其余“冻结”，类似 Dropout，既减少计算负担又缓解过拟合。

- **通信效率优化设计**  
  由于通信内容是固定长度的 prototype 向量，通信开销主要取决于类别数而非模型参数总量，适合大规模部署。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **计算效率** | 无需矩阵乘法，仅需位运算（XOR、加法），可在 MCU/FPGA 上低成本运行 |
| **内存占用** | 模型存储需求低，适合内存受限设备 |
| **通信效率** | 通信量比传统 FL 减少最多达 75%，尤其适用于低带宽无线网络 |
| **鲁棒性** | 高维表示具有容错性和抗噪能力，对数据缺失或硬件错误更稳健 |
| **能源消耗** | 极低功耗，适合电池供电的长期监测系统 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
实验在三个公开数据集上验证方法有效性：
- **MNIST**：手写数字图像分类（10 类）
- **Fashion-MNIST**：服装图像分类（10 类）
- **UCI HAR**（Human Activity Recognition）：智能手机传感器时间序列数据分类（6 种人体活动）

所有数据均模拟由 **N = 20 台资源受限 IIoT 设备** 分布持有。

### ⚙️ 实验设置
- **联邦架构**：标准 FL 设置，中央服务器协调，设备间无直接通信。
- **HDC 参数**：
  - 总向量维度 $ D = 5000 $
  - 子模型维度 $ D' \in \{2500, 1000, 500\} $，对应压缩率 $ M = D/D' = 2, 5, 10 $
  - 使用 OnlineHD 映射函数进行特征编码：$ \phi(x) = [\cos(xW + \phi), \sin(xW)] $
- **训练配置**：
  - 全局轮次（global epochs）: $ G = 100 $
  - 本地轮次（local epochs）: $ L = 5 $（i.i.d. 场景），$ L = 3 $（non-i.i.d. 场景）
  - 学习率：$ \alpha = 0.01 $

### 🎯 评估指标
- **分类准确率（Classification Accuracy）**：测试集上的最高准确率
- **上行链路流量（Uplink Traffic Consumption）**：达到某一性能水平所需的总上传数据量（MB）
- **收敛速度**：达到稳定性能所需的通信轮次

### 🔁 基线方法对比
- **Baseline Federated HDC**：传统的全维度 HDC 联邦训练，无子模型抽样
- 对比不同模型大小 $ D \in \{5K, 2.5K, 1K, 0.5K\} $ 下的表现

同时考虑两种数据分布场景：
- **i.i.d. 场景**：每个设备拥有均衡的类别分布
- **non-i.i.d. 场景**：极端非独立同分布，每台设备仅有 2 个类别（MNIST/Fashion-MNIST）或 2–3 个（UCI HAR）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据与对比结果

#### （1）准确率表现（见 Fig. 3）
- 在 **i.i.d. 场景**下：
  - 所有配置中，**提出的子模型方法（proposed method）在相同通信/计算成本下达到了更高或相当的准确率**
  - 即使使用 $ D'=1K $（仅更新 20% 维度），也能接近甚至超过完整 $ D=5K $ 模型的性能
- 在 **non-i.i.d. 场景**下（见 Fig. 5）：
  - 对于 **UCI HAR 数据集**，$ D'=2.5K $ 配置显著优于 baseline（准确率提升明显），同时节省高达 **50% 上行流量**
  - 对于 MNIST 和 Fashion-MNIST，性能略逊于 baseline，但通信代价更低

> ✅ 结论：该方法在某些任务（尤其是时间序列）中具备更强的泛化能力和通信优势。

#### （2）通信效率（见 Fig. 4）
- 为达到 baseline 方法的最终准确率，所提方法所需上传数据量大幅减少：
  - **最多减少约 75% 的上行链路流量**
  - 流量节省程度随 $ D' $ 减小而增加（即压缩率越高，节省越多）
- 这意味着设备可以以更少的通信轮次或更低的带宽参与协作学习

#### （3）消融分析（隐含于实验设计）
尽管未明确列出“ablation study”章节，但从以下设计体现了消融思想：
- 固定总计算/通信预算，比较是否采用“子模型更新”策略 → 显示后者性能更优
- 比较不同 $ D' $ 设置的影响 → 揭示存在最优子空间规模（如 UCI HAR 中 $ D'=2.5K $ 最佳）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Federated HDC 是一种极具潜力的轻量级 FL 范式**，特别适用于资源极度受限的 IIoT 环境。
2. **原型聚合机制显著降低了通信开销**，且不受模型参数爆炸影响，通信量仅与类别数相关。
3. **随机子模型重训练不仅降低成本，还能提升模型鲁棒性和准确性**，机制类似于 Dropout，在 non-i.i.d. 场景下表现出一定适应性。
4. **在时间序列分类任务（如 UCI HAR）中，Federated HDC 表现尤为出色**，兼具高性能与高效率。
5. **HDC 天然适合硬件加速**（如 FPGA、MCU），无需 GPU 支持即可实现实时推理与训练。

### ⚠️ 方法的局限性
- **原始精度上限低于先进深度模型**：尽管效率极高，但在绝对准确率上无法媲美大型 DNN。
- **对数据预处理敏感**：依赖 Random Fourier Features (RFFM) 等映射方式，可能影响端到端自动化。
- **non-i.i.d. 场景下的泛化能力不稳定**：在图像数据上表现不如 baseline，说明当前聚合策略对异构数据分布仍具挑战。
- **尚未集成差分隐私或安全聚合机制**：虽具隐私友好特性，但未显式提供形式化隐私保障。

### 🔮 未来工作方向
1. **适配更先进的 FL 算法**：将 FedProx、SCAFFOLD 等用于处理 non-i.i.d. 数据的方法融入 HDC 框架。
2. **探索自适应子模型选择机制**：动态决定每次更新哪些维度，而非完全随机。
3. **结合预训练特征提取器**：利用轻量 CNN 或 Transformer 提取特征后再输入 HDC，提升表达能力。
4. **二值化 HDC 实现极致节能**：使用 binary hypervectors + bitwise operations，进一步降低功耗与通信成本。
5. **真实工业场景部署验证**：在预测性维护、设备故障检测等实际 IIoT 应用中测试系统稳定性与实用性。

---

## 总结
> **Federated HDC 为资源受限 IIoT 提供了一个全新的“超轻量 + 高效 + 鲁棒”的分布式智能路径**。它不是为了追求最高精度，而是为了解决“能否在微型设备上持续运行智能算法”这一根本问题。在未来 B5G/6G 支持的大规模工业传感网络中，这类方法有望成为支撑边缘智能的基础组件之一。

</details>

---

### 4. [A Subgoal-driven Framework for Improving Long-Horizon LLM Agents](https://arxiv.org/abs/2603.19685)

**Authors**: Taiyi Wang, Sian Gooding, Florian Hartmann, Oriana Riva, Edward Grefenstette  
**Category**: cs.AI  
**Published**: 2026-03-23  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.19685v1  

#### Abstract
Large language model (LLM)-based agents have emerged as powerful autonomous controllers for digital environments, including mobile interfaces, operating systems, and web browsers. Web navigation, for example, requires handling dynamic content and long sequences of actions, making it particularly cha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Subgoal-driven Framework for Improving Long-Horizon LLM Agents

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Model (LLM)** 的智能体在处理长视野任务（long-horizon tasks）时存在显著缺陷，尤其是在 **Web导航** 这类复杂、动态环境中。主要问题包括：
- **在线执行中的规划失败**：随着新信息不断出现，代理容易偏离目标路径，缺乏对最终目标的自适应追踪能力。
- **强化学习（RL）微调中的稀疏奖励问题**：由于奖励信号延迟且稀疏，代理难以将成功归因于早期的关键动作，导致信用分配（credit assignment）困难。

### 提出的新方法与思路
本文提出了一种结合**推理时规划**与**训练时奖励塑造**的统一框架，旨在提升LLM代理在长程任务中的表现。其两大核心贡献为：

#### （1）**推理时子目标驱动规划（Subgoal-driven Inference-Time Planning）**
- 利用强大的专有模型（如Gemini）将高层任务分解为一系列可验证的**Subgoals**。
- 在推理过程中引入**动态里程碑机制（Dynamic Milestoning）**，让代理通过自我反思（self-reflection）实时检查已完成的里程碑，并据此调整下一步行动。
- 该机制增强了代理的“情境感知”能力，使其能够检测停滞并进行纠错。

#### （2）**MiRA：基于里程碑的强化学习微调框架（Milestone-based Reinforcement Learning Fine-Tuning）**
- 提出 **MiRA (Milestoning your Reinforcement Learning Enhanced Agent)**，一种离线RL训练框架。
- 引入**密集的、基于里程碑的奖励信号**作为辅助奖励，解决原始任务中奖励稀疏的问题。
- 设计双批评家架构（Dual-Critic Architecture）：
  - **Value Critic (Vb)**：预测最终任务成功的概率（基于稀疏的ORM信号）。
  - **Potential Critic (Pb)**：学习一个连续的进展函数，输出当前状态相对于各子目标的完成度（基于密集的Subgoal Checker信号）。
- 使用**潜在函数奖励塑造（Potential-Based Reward Shaping, PBRS）** 将两者结合，形成更有效的优势估计。

### 相比现有方法的优势
- **显式而非隐式分解**：不同于依赖潜变量或树搜索的方法，本文的Subgoals是语义明确、可验证的中间目标，提高了可靠性和可解释性。
- **训练与推理协同优化**：MiRA在训练阶段内化了子目标依赖关系，而推理时的动态里程碑则提供了运行时纠错能力，二者互补。
- **避免过优化风险**：与Process Reward Models (PRMs) 不同，本文的里程碑是硬性检查点，减少了因学习软信号而导致的噪声和过优化问题。

---

## 2. 核心实验方法和设置

### 数据集
- 主要评估基准：**WebArena-Lite**，这是从WebArena中精选的165个高质量任务子集，覆盖五个真实应用领域：
  - Shopping Admin (35 tasks)
  - Map (26 tasks)
  - Shopping (45 tasks)
  - Reddit (19 tasks)
  - Gitlab (30 tasks)
- 选择此数据集的原因是其任务定义清晰、可行性高，避免了原版WebArena中存在的环境故障和评估不稳定性。

### 实验设置与评估指标
- **评估指标**：
  - **Success Rate (SR)**：任务完成率，即Pass@1，强调一次性正确完成的能力。
  - **Failure Mode Analysis**：通过自动化分析器对失败轨迹进行分类，识别主要错误模式。
  - **Pass@k**：在k次尝试中至少有一次成功的概率，用于衡量采样效率。
- **训练协议**：
  - 所有基于学习的方法均从相同的SFT检查点初始化。
  - MiRA采用迭代式课程学习（iterative curriculum），每轮收集新轨迹，更新模型，并生成更具挑战性的任务分布。
  - 使用**Experience Replay Buffer**和**Perplexity Filtering**筛选高质量训练数据。

### 基线方法对比
| 类别 | 基线方法 |
|------|--------|
| **专有LLM** | GPT-4-Turbo, GPT-4o, Gemini 2.5 Pro/Flash |
| **开源LLM + SFT** | Llama3.1-SFT, Gemma3-SFT |
| **开源LLM + RL** | AWR, DigiRL, WebRL |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 平均SR (%) |
|------|----------|
| GPT-4-Turbo | 17.6 |
| GPT-4o | 13.9 |
| Gemini 2.5 Pro | 23.0 |
| **Gemini-SGO (ours)** | **32.1** |
| WebRL (Llama3-8B) | 38.8 |
| WebRL (Gemma3-12B) | 35.1 |
| **Gemma3 + MiRA (ours)** | **43.0** |

> ✅ **结论**：提出的 **Gemma3 + MiRA** 框架以 **43.0%** 的成功率大幅超越所有基线，包括专有系统（GPT-4-Turbo: 17.6%, GPT-4o: 13.9%）和此前开源SOTA（WebRL: 38.8%）。

### 与基线方法的对比结果
- **推理时增强（SGO）效果**：
  - 在Gemini 2.5 Pro上应用SGO，成功率从 **23.0%** 提升至 **32.1%**（绝对提升约10%）。
  - 显著降低了“中途卡住（Get Stuck Midway）”的错误率（从48.41%降至39.87%）。
- **训练时增强（MiRA）效果**：
  - Gemma3-12B模型的成功率从SFT的30.9%提升至MiRA的43.0%，增幅超过12个百分点。
  - 在Gitlab和Shopping Admin等复杂流程任务上提升尤为明显（分别达56.7%和54.3%）。

### 消融实验结果
消融研究验证了MiRA各组件的必要性：

| 配置 | 最终SR (%) | 分析 |
|------|-----------|------|
| **MiRA (Full)** | **43.0** | 完整框架 |
| MiRA w/o PC (无Potential Critic) | ~35.0 | 学习迅速饱和，证明密集奖励对长程信用分配至关重要 |
| MiRA w/o Doubly Robust | ~37.0 | 早期性能崩溃，表明双重鲁棒估计对稳定训练至关重要 |
| MiRA w. KL (使用KL散度) | ~33.0 | 收敛慢且性能差，证明MSE回归更适合离线策略学习 |
| AWR (标准RL基线) | ~30.3 | 性能最低，凸显奖励塑造的有效性 |

> 🔍 **关键发现**：移除任何一个核心组件都会导致性能显著下降，证明了完整MiRA框架的必要性。

---

## 4. 关键结论和发现

### 主要发现
1. **“中途卡住”是主导失败模式**：无论模型大小或是否经过SFT，**Get Stuck Midway** 是最常见的失败原因（占比高达42–49%），表明现有代理普遍缺乏持续规划和自我纠正能力。
2. **显式子目标分解有效**：将任务分解为可验证的Subgoals，不仅能指导推理，还能提供密集监督信号，显著改善长程信用分配。
3. **训练与推理协同增效**：
   - MiRA在训练中“编译”了子目标逻辑到模型权重中。
   - SGO在推理时提供独立的“护栏”，防止错误传播。
   - 二者结合实现了**鲁棒的长视野自主性**。
4. **开源模型可超越专有模型**：通过MiRA框架，**Gemma3-12B** 的性能不仅远超同类开源模型，还显著优于GPT-4-Turbo等大型专有模型。

### 方法的局限性
- **子目标生成依赖强教师模型**：Subgoals由Gemini-2.5-Pro生成，若教师模型本身无法理解任务，则整个框架失效。
- **冷启动探索问题**：当初始子目标极难达成时，辅助奖励无法激活，系统退化为稀疏奖励学习。
- **固定粒度限制**：目前Subgoals数量固定（如4步），可能不适合所有任务复杂度。
- **对终端判断的依赖**：AutoRater可能仅验证页面状态而忽略语义细节，导致“错误终止（Wrong Termination）”增加。

### 未来工作方向
- **可学习的子目标生成器**：从启发式提示转向可训练的分层子目标生成模型，实现动态粒度控制。
- **非线性进展估计**：改进进展函数，考虑不同子目标的难度差异，而非均匀加权。
- **信号退火策略（Signal Annealing）**：在训练后期逐渐减少对辅助奖励的依赖，确保最终策略聚焦于真实任务目标。
- **闭环自进化系统**：构建完全自包含的循环，其中同一模型负责路径规划、进度判断、课程生成和训练信号合成，实现真正的递归自我改进。

> 🚀 **总体结论**：本文提出的**子目标驱动框架**通过结合**显式的推理时规划**与**基于里程碑的奖励塑造**，显著提升了LLM代理在长视野任务中的表现。其实验结果证明，**结构化的规划机制**对于构建真正自主的数字助手至关重要，为下一代通用智能体的发展指明了方向。

</details>

---

### 5. [SWARM+: Scalable and Resilient Multi-Agent Consensus for Fully-Decentralized Data-Aware Workload Management](https://arxiv.org/abs/2603.19431)

**Authors**: Komal Thareja, Krishnan Raghavan, Anirban Mandal, Ewa Deelman  
**Category**: cs.DC  
**Published**: 2026-03-23  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.19431v1  

#### Abstract
Distributed scientific workflows increasingly span heterogeneous compute clusters, edge resources, and geo-distributed data repositories. In these environments, a centralized orchestrator is an architectural bottleneck -- introducing a single point of failure, limiting scalability, and constraining ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SWARM+: Scalable and Resilient Multi-Agent Consensus for Fully-Decentralized Data-Aware Workload Management

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代科学工作流在异构、地理分布的计算资源（如超级计算机、边缘设备）和分布式数据存储之间运行，传统的**集中式调度器**（如 SLURM、HTCondor）存在以下瓶颈：
- 单点故障（Single Point of Failure）
- 可扩展性差（Scalability Bottleneck）
- 难以适应动态资源变化和故障
- 忽视数据局部性（Data Locality），导致高传输开销

SWARM+ 旨在构建一个**完全去中心化**（Fully-Decentralized）、**可扩展**（Scalable）、**容错性强**（Resilient）且**数据感知**（Data-Aware）的工作负载管理系统。

---

### 提出的新方法与创新点

SWARM+ 在其前作 SWARM 的基础上，提出三大核心创新：

#### （1）**Hierarchical Consensus（分层共识机制）**
- 将 agents 组织为树状层级结构（如 Level 1 CoordinatorAgent 负责多个 Level 0 ResourceAgent）
- 共识过程分解为两级：
  - **Level 1**：CoordinatorAgent 从全局任务池中选择任务并分配给子组
  - **Level 0**：ResourceAgent 在本地组内通过共识完成具体资源分配
- **优势**：将通信复杂度从 $O(n^2)$ 降低至 $O(\log n)$，显著提升可扩展性

#### （2）**Comprehensive Resilience Mechanisms（综合容错机制）**
- **多信号故障检测**（Multi-signal Failure Detection）：结合 gRPC 连接状态回调与 Redis 心跳超时
- **自动任务重选**（Automatic Job Reselection）：失败 agent 的任务被重新置为待处理并参与新一轮选择
- **自适应法定人数**（Adaptive Quorum）：动态调整达成共识所需的活跃 agent 数量，无需显式成员重组
- 支持弹性扩缩容（Dynamic Membership）

#### （3）**Data-Aware Cost Modeling（数据感知的成本模型）**
- 成本函数综合考虑：
  - 资源利用率（CPU、RAM、GPU、Disk）
  - DTN（Data Transfer Node）连接质量
  - 数据局部性惩罚项（Connectivity Penalty）
  - 长作业惩罚（Long-Job Penalty）
- 引入 **LRU 缓存优化** 减少重复计算开销

---

### 相比现有方法的优势

| 方面 | SWARM+ 优势 |
|------|-------------|
| 架构 | 完全去中心化，无单点故障 |
| 扩展性 | 分层结构支持千级 agent 规模 |
| 容错性 | 自动恢复、自适应 quorum，系统降级优雅 |
| 性能 | 选择延迟低，尾部延迟显著降低 |
| 数据感知 | 显式建模 DTN 连接与数据局部性，优化调度效率 |

---

## 2. 核心实验方法和设置

### 实验平台
- 使用 **FABRIC testbed**：一个国家级可编程网络研究基础设施，覆盖超过 30 个地理分布站点
- 支持跨站点部署，模拟真实 WAN 环境（RTT: 2.36–68.33 ms）

---

### 实验设置

#### Agent 配置
- **数量范围**：10 到 990 个 agents
- **拓扑类型**：
  - Mesh（全连接）
  - Ring（稀疏环形）
  - Hierarchical（分层结构，最多三层）
- **资源类型**：Small / Medium / Large（异构配置，含 GPU 差异）

#### DTN 设置
- 10 个 DTN 节点，连接评分 [0.6, 0.95]
- 每个 agent 关联 0–4 个 DTN，体现数据局部性偏好

#### 工作负载（Synthetic Workload）
- 任务数量：100 至 9000
- 资源需求：按比例采样自 agent 容量
- 任务类别：
  - Lightweight（轻量级）
  - Standard（标准）
  - Resource-Intensive（资源密集型）
- 分布偏重小任务（指数分布，指数为 3）

---

### 评估指标

| 指标 | 定义 |
|------|------|
| **Selection Time** | 从开始选择到任务分配最终确定的时间（均值 & P95） |
| **Scheduling Latency** | 从任务提交到分配完成的时间 |
| **Wait Time** | 从提交到开始选择的时间 |
| **Job Completion Rate** | 成功完成的任务占比 |
| **Failure Detection Latency** | 故障被检测到的时间 |
| **Recovery Time** | 故障后恢复至稳定状态的时间 |
| **Agent Participation Rate** | 新增 agent 参与任务选择的比例 |

---

### 基线方法对比
- **Baseline SWARM**：原始版本，采用扁平化 PBFT 共识，无分层、无数据感知
- 对比重点：相同配置下（Mesh-10, 100 jobs, 多站点）的延迟表现

---

## 3. 主要实验结果和性能指标

### ✅ 与 SWARM 的性能对比（Table I）

| 指标 | SWARM | SWARM+ | 提升幅度 |
|------|--------|---------|----------|
| 平均 Selection Time | 40.03 s | 1.20 s | **97.0%** |
| 中位 Selection Time | 36.60 s | 1.15 s | **96.9%** |
| P95 Selection Time | 85.47 s | 1.54 s | **98.2%** |
| P99 Selection Time | 130.61 s | 2.67 s | **98.0%** |
| 平均 Scheduling Latency | 325.22 s | 5.41 s | **98.3%**（约 60× 加速） |

> 所有改进均统计显著（p < 10⁻²³, Cohen’s d > 6.4）

#### 性能提升来源分析：
- **gRPC 替代 Kafka**：实现亚毫秒级通信（vs. Kafka 的 10–50ms 开销）
- **协议缓冲区序列化** + **索引查找优化**：减少消息处理开销 60–70%
- **LRU 缓存**：避免重复成本计算，提高缓存命中率

---

### ✅ 可扩展性测试（Table II & IV）

| 拓扑 | Agents | Jobs | 平均 Selection Time (s) | P95 (s) |
|------|-------|------|------------------------|--------|
| Mesh-90 | 90 | 450 | 5.95 | 8.67 |
| Ring-30-500 | 30 | 500 | 51.98 | 95.60 |
| **Hier-110** | **110** | **1000** | **1.01** | **1.34** |
| **Hier-990** | **990** | **9000** | **46.12** | **208.20** |

- **Hierarchical 拓扑在大规模下仍保持亚秒级延迟**
- 层级间负载均衡良好（Level 0 与 Level 1 处理任务数接近 50%/50%）
- **验证了 $O(\log n)$ 复杂度的有效性**

---

### ✅ 容错性实验结果（Figure 5）

| 故障场景 | Job Completion Rate | 影响程度 |
|--------|--------------------|---------|
| 单 agent 失败（3.3%） | **>99.8%** | 可忽略 |
| 8 agents 失败（26.7%） | ~94.9% | 轻微下降 |
| 15 agents 失败（50%） | **≥92.5%** | 最大影响仅 **7.5%** |

> 不完整任务主因是剩余资源无法满足任务需求（非系统崩溃）

#### 故障检测延迟对比：
- **gRPC-based detection**: **13.8 ms ± 1.6 ms**
- **Redis-based detection**: **54.2 s ± 0.5 s**
- SWARM+ 同时启用两者：gRPC 快速响应 + Redis 回退防误报

---

### ✅ 动态扩缩容能力（Figure 6）

- 新增 agent 可在任意时间加入系统
- **早期加入（t=10–20s）**：参与率达 **86.7–93.3%**
- **晚期加入（t=60s）**：参与率降至 **66.7%**（因多数任务已被分配）
- 表明系统具备良好的 **elastic scaling** 能力

---

### ✅ 地理分布影响（Table IV）

| 拓扑 | 单站点延迟 (s) | 多站点延迟 (s) | 慢化倍数 |
|------|---------------|---------------|---------|
| Mesh-30 | 2.79 | 5.77 | 2.07× |
| Hier-30 | 0.93 | 1.19 | **1.28×** |
| Hier-110 | 1.01 | 3.77 | 3.73× |

- **分层结构对 WAN 更鲁棒**：大部分共识发生在本地 site 内
- Hier-30 表现最优，说明“site-aligned”设计有效减少跨域通信

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **分层共识显著提升可扩展性**  
   - 支持近 **1000 个 agents** 和 **9000 个任务** 的调度
   - 通信复杂度控制在 $O(\log n)$，实现近线性扩展

2. **系统高度容错且降级优雅**  
   - 单点故障几乎不影响任务完成率（>99%）
   - 即使 50% agent 失效，系统仍能完成 **>92.5%** 的任务
   - 自适应 quorum 保证系统持续前进（liveness）

3. **工程优化带来巨大性能收益**  
   - gRPC + 缓存 + 序列化优化共同促成 **97–98% 的延迟降低**

4. **数据感知调度提升实际效率**  
   - DTN 连接质量作为成本因子，引导任务向数据近端执行
   - 长作业惩罚防止热点形成

---

### ⚠️ 方法的局限性

1. **WAN 延迟仍构成挑战**  
   - 多站点部署下延迟上升明显（尤其 Hier-110 达 3.73×）
   - 当前未使用优化 overlay 网络（如 DGRO [20]）

2. **静态层级结构**  
   - 层级深度需预先配置，缺乏动态重构能力
   - 未来可引入自适应 hierarchy construction

3. **合成 workload**  
   - 实验基于 synthetic data，尚未在真实科学 workflow（如 Pegasus、Nextflow）上验证

4. **缓存 TTL 与一致性权衡**  
   - LRU 缓存依赖版本号，极端情况下可能短暂不一致

---

### 🔮 未来工作方向

1. **网络优化部署**  
   - 结合 **overlay communication networks**（如 DGRO）降低 WAN 开销

2. **自适应层级构建**  
   - 根据负载动态调整 hierarchy depth 与 group size

3. **大规模弹性与容错实验**  
   - 在更大规模（>1000 agents）下测试长期稳定性

4. **集成主流 Workflow Systems**  
   - 开发适配器对接 **Pegasus**、**Nextflow** 等系统，推动实用化

5. **支持更复杂的资源类型与约束**  
   - 如能耗、SLA、安全隔离等维度扩展 cost model

---

> **总结一句话**：SWARM+ 通过 **Hierarchical Consensus + Adaptive Resilience + Data-Aware Cost Model**，实现了面向大规模科学计算的**可扩展、容错、去中心化**任务调度系统，在真实测试床 FABRIC 上验证了其优越性能与实用性。

</details>

---

### 6. [A Dynamic Bayesian and Machine Learning Framework for Quantitative Evaluation and Prediction of Operator Situation Awareness in Nuclear Power Plants](https://arxiv.org/abs/2603.19298)

**Authors**: Shuai Chen, Huiqiao Jia, Tao Qing, Li Zhang, Xingyu Xiao  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.19298v1  

#### Abstract
Operator situation awareness is a pivotal yet elusive determinant of human reliability in complex nuclear control environments. Existing assessment methods, such as SAGAT and SART, remain static, retrospective, and detached from the evolving cognitive dynamics that drive operational risk. To overcom...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
传统对核电厂操作员**Situation Awareness (SA)** 的评估方法（如 SAGAT 和 SART）存在以下局限性：
- **静态性**：依赖事后问卷或冻结探针，无法捕捉动态认知过程；
- **主观性**：基于自我报告，缺乏客观生理证据支持；
- **孤立性**：忽略多个**Performance Shaping Factors (PSFs)** 之间的因果关系与时序演化；
- **非预测性**：难以实现早期预警和实时监控。

该研究旨在构建一个**可量化、可解释、可预测**的操作员 SA 动态模型，以支持下一代数字化主控室（Digital Main Control Room, DMCR）中的人因可靠性管理。

---

### 提出的新方法：DBML-SA 框架
提出了一种融合概率推理与数据驱动智能的混合框架 —— **Dynamic Bayesian-Machine Learning Framework for Situation Awareness (DBML-SA)**，其核心思想是将两种范式优势互补：

| 组件 | 功能 |
|------|------|
| **Dynamic Bayesian Network (DBN)** | 建模 PSFs 间的因果-时序结构，进行时间演化的不确定性推理 |
| **Neural Network (FCN)** | 学习从 PSFs 到 SART 分数的非线性映射，实现自动预测 |

#### 框架四层架构：
1. **Layer I: 数据源与预处理**  
   整合历史事件报告与模拟器实验数据，提取 PSFs、生理信号与主观 SA 测量值。
   
2. **Layer II: 统计与因子分析**  
   通过相关性分析与探索性因子分析（EFA），识别影响 SA 的四大潜在维度。

3. **Layer III: DBN 构建**  
   建立包含静态 PSFs 与动态认知状态（stress, attention）的时间扩展贝叶斯网络。

4. **Layer IV: 机器学习预测模块**  
   使用全连接神经网络（Fully Connected Network, FCN）训练 SA 预测模型，并与 DBN 耦合形成闭环系统。

---

### 相比现有方法的优势
| 特性 | 传统方法（SAGAT/SART） | DBML-SA 框架 |
|------|------------------------|---------------|
| 时间分辨率 | 静态/离散 | 连续、动态推断 |
| 可解释性 | 弱（仅评分） | 强（因果路径可视化） |
| 客观性 | 主观依赖强 | 融合客观生理指标（EDA, 瞳孔直径） |
| 预测能力 | 无 | 支持前向预测与后向诊断 |
| 实时性 | 不适用 | 可嵌入监控系统实现实时反馈与早期预警 |

> ✅ **核心创新**：首次将 DBN 的**因果可解释性**与 ML 的**高精度拟合能力**结合，实现了 SA 的“既知其然，又知其所以然”的建模目标。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据来源 | 描述 | 规模 |
|---------|------|-----|
| **Operation Event Reports** | 来自中国商用核电站 2007–2021 年间共 212 起运行事件报告 | 包含 SA 失败相关的定性描述 |
| **Simulator Experiments** | 在高保真数字核电厂模拟器上开展的 SGTR（Steam Generator Tube Rupture）小破口事故场景实验 | 41 名参与者（33 名持证操作员 + 8 名研究人员）组成团队参与 |

---

### 实验设置
- **任务设计**：SGTR 场景持续约 25 分钟，分为故障触发、诊断、响应与恢复阶段；
- **暂停点设置**：在关键节点插入两个 SAGAT 探针（pause point），用于测量实时 SA；
- **数据采集**：
  - **生理数据**：穿戴式 EDA 传感器（Shimmer3）、眼动仪（Tobii Pro Fusion）
  - **行为数据**：控制日志、语音记录、程序执行步数
  - **主观评估**：SAGAT（感知/理解/预测三级 SA）、SART（综合注意力需求与信息供给评价）

---

### 评估指标
| 指标 | 公式/说明 | 应用场景 |
|------|----------|--------|
| **MAPE** | $\frac{1}{n}\sum \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$ | 衡量 FCN 对 SART 得分的预测误差 |
| **R²** | 决定系数，反映预测值与真实值的相关程度 | 评估模型拟合优度 |
| **One-way ANOVA** | 单因素方差分析（p > 0.05 表示无显著差异） | 比较 DBN 输出与 SART 自评得分的一致性 |
| **Sensitivity Analysis** | 输入扰动下输出变化率 | 识别关键影响因素 |
| **Diagnostic Reasoning** | 后向推理各 PSF 在 SA 失败条件下的后验概率提升 |

---

### 基线方法对比
虽然文中未直接列出多个外部基线模型作为比较对象，但在第 5.3 节进行了内部模型横向对比：

| 模型 | 类型 | 输入特征 | 是否考虑时序 |
|------|------|-----------|-------------|
| FCN (Instantaneous) | 静态 | 11 PSFs | ❌ |
| Multi-Factor FCN | 静态扩展 | 12 PSFs + embedding | ❌ |
| LSTM | 动态 | PSFs 序列（to-t9） | ✅ |

> ⚠️ 注意：所有模型均在同一数据集上训练并使用相同优化器（Adam, lr=0.001）进行独立测试。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 | 说明 |
|------|------|------|
| **FCN 预测 MAPE** | **13.8%**（整体 SART 得分） | 最终模型平均绝对百分比误差低，具备实用价值 |
| **FCN 预测 R²** | **0.83**（p < 0.01） | 与主观 SART 评分高度一致 |
| **DBN vs SAGAT 差异** | 平均绝对偏差 < **5.0%** | 时间趋势高度吻合 |
| **DBN vs SART ANOVA** | **p = 0.385 > 0.05** | 统计上无显著差异，验证有效性 |

---

### 与基线方法的对比结果（见 Table 8）

| Model | R² | MAPE (%) | Key Advantage |
|-------|-----|----------|--------------|
| FCN (Instantaneous) | 0.83 | **14.3** | 非线性映射能力强 |
| **Multi-Factor FCN** | 0.62 | **9.47** | **最低 MAPE，最佳整体表现** |
| LSTM | 0.34 | 14.83 | 可捕获时序动态，但精度较低 |

> 🔍 发现：尽管 LSTM 设计为处理序列，但由于样本有限且噪声较大，其泛化能力不如静态 FCN；而 Multi-Factor FCN 因引入嵌入特征，在少量数据下仍表现出最优性能。

---

### 消融实验与敏感性分析结果
#### （1）诊断推理（Diagnostic Reasoning）
当设定 SA 失败为目标状态时，DBN 反向推断出最可能导致失败的因素：

| PSF | 后验概率增幅 (%) |
|-----|------------------|
| **Training Quality** | **+18.7%** |
| **Stress Variation** | **+16.4%** |
| Attention Level | +9.3% |
| Task Complexity | +7.8% |
| Teamwork | +6.5% |

> 💡 **结论**：培训质量与压力波动是导致 SA 下降的最主要驱动因素。

#### （2）全局敏感性分析
| 敏感度等级 | 影响因素 |
|------------|----------|
| >10% 变化 | Training, Stress |
| 5–10% 变化 | Attention, Task Complexity |
| <5% 变化 | Interface Quality, Automation Level |

> 📉 表明 SA 更易受**人为认知与情绪因素**影响，而非环境或自动化水平等结构性因素。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **DBML-SA 成功实现了 SA 的动态建模与预测**：
   - DBN 能准确再现 SAGAT 所观测到的认知演变轨迹；
   - FCN 可以仅凭 PSFs 输入稳定预测 SART 得分（MAPE≈13.8%）；
   - 两者结合提供“解释+预测”双重能力。

2. ✅ **培训质量与压力是影响 SA 的最关键因素**：
   - 缺乏充分训练会加剧高压下的判断失误；
   - 生理应激反应（由 EDA 反映）直接影响注意力稳定性。

3. ✅ **多因素交互作用显著**：
   - 因子分析揭示 SA 受四大潜变量调控：**认知负荷、程序支持、人机界面、组织协同**；
   - 忽视这些交互将低估风险传播路径。

4. ✅ **框架具有实际部署潜力**：
   - 可集成至 DMCR 监控系统，生成 SA 风险仪表盘；
   - 支持早期预警、自适应辅助决策与个性化培训推荐。

---

### 方法的局限性
1. **数据规模限制**：仅基于 212 份事件报告与 41 名被试，可能存在过拟合风险；
2. **任务语义缺失**：当前模型未编码具体操作语义（如“是否开启安全注入”），限制跨场景迁移能力；
3. **生理信号信噪比问题**：EDA 和瞳孔数据易受个体差异与外部干扰影响；
4. **静态 PSF 假设**：部分 PSFs（如可用时间）在任务中实际是动态变化的，但模型中视为常量。

---

### 未来工作方向
1. **多模态数据融合**：整合语音通信、行为日志、注视热图等更多信号以丰富认知表征；
2. **知识图谱增强**：引入 Knowledge Graph 或 Large Language Models 实现任务语义编码与情境理解；
3. **实时可视化模块开发**：构建 SA 动态仪表板，支持监督员实时监控与干预；
4. **跨任务迁移学习**：利用语义表示提升模型在新型事故场景中的泛化能力；
5. **闭环人机协同系统**：将 SA 预测结果反馈给自动化系统，实现动态任务分配与辅助策略调整。

---

> 🏁 **总结一句话**：  
> 本研究提出的 **DBML-SA 框架**突破了传统 SA 评估的静态与主观局限，首次实现了基于因果推理与机器学习融合的**可解释、可预测、可实时监控**的操作员认知状态建模，为智能核电厂人因可靠性管理提供了坚实的技术基础。

</details>

---

### 7. [TTQ: Activation-Aware Test-Time Quantization to Accelerate LLM Inference On The Fly](https://arxiv.org/abs/2603.19296)

**Authors**: Toshiaki Koike-Akino, Jing Liu, Ye Wang  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.19296v1  

#### Abstract
To tackle the huge computational demand of large foundation models, activation-aware compression techniques without retraining have been introduced. However, since these methods highly rely on calibration data, domain shift issues may arise for unseen downstream tasks. We propose a test-time quantiz...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TTQ: Activation-Aware Test-Time Quantization to Accelerate LLM Inference On The Fly

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）虽然在多种任务上表现出色，但其巨大的计算和内存开销限制了实际部署。现有的后训练量化（Post-Training Quantization, PTQ）方法如 **AWQ** 和 **GPTQ** 虽然能压缩模型，但存在以下关键问题：
- **依赖离线校准数据（offline calibration）**：需要预先准备与目标任务分布一致的数据。
- **领域偏移（domain shift）风险**：当测试任务与校准数据不匹配时，性能显著下降。
- **不可逆性**：一旦模型被量化并部署，无法重新校准以适应新任务。

### 🚀 提出的新方法：TTQ（Test-Time Quantization）
本文提出 **TTQ** ——一种**在推理时动态进行激活感知量化的框架**，其核心思想是：
- 在每个输入 prompt 到来时，**在线计算激活统计量**（activation statistics），实时调整量化参数（scale 和 zero-point）。
- 实现“即用即压”（compress on the fly），无需任何离线校准数据。

### 🔍 相比现有方法的优势
| 特性 | AWQ / GPTQ（静态量化） | TTQ（动态量化） |
|------|------------------------|----------------|
| 是否需要校准数据 | 是（必须） | 否（zero calibration） |
| 是否受 domain shift 影响 | 是 | 否（自适应） |
| 是否支持部署后重校准 | 否 | 是（on-device self-calibration） |
| 推理速度提升 | 有（依赖硬件优化） | 有，且额外开销极小 |
| 支持低秩补偿 | 静态（QLoRA） | 动态适配 |

此外，TTQ 还引入了：
- **低秩分解集成**（Low-Rank Decomposition）：通过 $ \mathbf{W} = \mathbf{W}_q + \mathbf{BA} $ 补偿量化损失，其中 $\mathbf{W}_q$ 是动态量化的残差权重。
- **动态适应性**：$\mathbf{W}_q$ 随输入变化而变化，优于 QLoRA 中固定的 $\mathbf{W}_q$。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **语言建模基准（Perplexity Evaluation）**：
  - `WikiText-2` (WT2)
  - `Penn Treebank` (PTB)
  - `C4`（cleaned web text）
- **视觉语言模型（VLM）任务**：
  - `TextVQA`：基于图像中文本的问答
  - `COCO-Caption`, `OK-VQA`, `ChartQA`：用于 AWQ 的不同校准数据
- **视觉语言动作模型（VLA）任务**：
  - `LIBERO`：机器人操作任务，包含空间、物体、目标等子任务

### ⚙️ 实验设置
- **模型族**：
  - **OPT**（125M ~ 6.7B）
  - **Qwen3**（0.6B ~ 32B）
  - **Gemma3**（270M ~ 12B）
  - **Qwen3-VL**（多模态）
  - **T0.5**（VLA 模型）
- **量化配置**：
  - 位宽：2~5 bits
  - groupsize：默认 32，部分实验测试 8~1024
  - TTQ 中低秩秩 $ r = 0 $ 或 $ 16 $
- **评估指标**：
  - **Perplexity ↓**（越低越好）
  - **Accuracy ↑ / Success Rate ↑**
  - **Runtime Speed (k tokens/sec)**：解码阶段吞吐量
- **硬件平台**：
  - NVIDIA A40, A100, L40, RTX3090, RTX4090
  - 使用 `Marlin` 和 `awq_gemm` 等高效 int matmul kernel

### 🆚 基线方法对比
| 方法 | 类型 | 是否需校准 | 备注 |
|------|------|------------|------|
| **RTN** | Round-To-Nearest | 否 | 最基础量化 |
| **AWQ** | Activation-Aware Weight Quantization | 是 | 使用 C4 / WT2 / PTB 等校准 |
| **TTQ(r=0)** | 仅量化 | 否 | 无低秩补偿 |
| **TTQ(r=16)** | 量化 + 低秩 | 否 | 引入 BA 分解 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 3）

#### OPT 系列平均 Perplexity（↓）
| 方法 \ 位宽 | 2-bit | 3-bit | 4-bit | 5-bit |
|-----------|-------|-------|-------|-------|
| RTN       | 5058.5 | 56.3 | 33.5 | 31.8 |
| AWQ       | 381.7 | 37.4 | 32.3 | **31.4** |
| **TTQ(r=16)** | **141.7** | **35.8** | **31.8** | ***31.1*** ✅ |

> ✅ **TTQ 在所有位宽下均优于 AWQ，尤其在极端低位宽（2/3-bit）优势明显**

#### Qwen3-1.7B 在 WT2 上的 Perplexity（3-bit）
| 方法 | Perplexity |
|------|----------|
| RTN | 162.8 |
| AWQ (best) | 28.2 |
| **TTQ(r=16)** | **26.4** ✅ |

#### Gemma3-1B 平均 Perplexity（5-bit）
| 方法 | Perplexity |
|------|----------|
| AWQ (best) | 93.6 |
| **TTQ(r=16)** | **90.3** ✅ |

> 💡 **多个模型在 5-bit 时达到甚至略超原始未压缩模型性能（标 *）**

---

### 📈 与基线方法的对比结果
- **TTQ 显著优于 AWQ**，尤其是在校准数据不足或分布不匹配时：
  - 如 Table 1 所示，在仅用少量 calibration token 时，AWQ 性能急剧下降，而 TTQ 不受影响。
- **对 groupsize 更鲁棒**：
  - Table 2 显示 TTQ 可容忍更大的 groupsize（如 512），意味着更少的 scale/zero-point 存储需求。
- **在 VLM/VLA 任务中表现稳定**：
  - **TextVQA**（Table 12）：TTQ 在 2-bit 下仍保持 7.47%~47.22% 准确率，远高于 AWQ 的 <1%。
  - **LIBERO**（Table 13）：TTQ 在长程任务（Libero-10）上成功率 **87.5%**，显著高于 AWQ（最高 84.5%）。

---

### 🔬 消融实验结果
- **低秩补偿有效**：
  - 加入 $ r=16 $ 的低秩因子后，几乎所有模型在低位宽下进一步提升性能（如 OPT-125M 从 257.4 → 141.7 @2bit）。
- **无需精细调参**：
  - TTQ 使用固定超参数（α=0.5, λ=0.4, p=2）即可取得最优效果（见 Appendix F 图 2）。
- **运行时开销极小**：
  - 公式 (3) 显示额外复杂度为 $ O[dT + 3dd'] / O[d'dT] = O(1/T + 1/d') $，在大 $ T $ 和 $ d' $ 下可忽略。
  - 实测运行时速度接近 AWQ（Table 8），甚至在大模型上更快（因内存瓶颈缓解）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **TTQ 实现了真正意义上的“零校准”动态量化**，解决了 domain shift 问题。
2. **在线 activation-aware 量化可行且高效**，额外计算开销可忽略。
3. **TTQ 在 2~5 bit 下全面超越 AWQ/GPTQ 等 SOTA 方法**，尤其在低位宽和跨域场景下优势显著。
4. **集成低秩分解进一步提升性能**，且不影响推理效率。
5. **TTQ 更适合大模型**：随着模型增大，weight traffic 成为主导，量化带来的缓存收益更明显。

### ⚠️ 局限性
- 当前实现未完全融合到 kernel 中（如 `prologue fusion`），仍有优化空间。
- 低秩因子目前为静态初始化（top-r PCA），未探索动态更新。
- 超参数（α, λ, p）虽可固定，但尚未实现自动在线调整。

### 🔮 未来工作方向
- 设计专用 CUDA kernel 支持 **prologue fusion**，进一步加速。
- 探索 **test-time pruning + TTQ** 的混合 MoE 架构（文中提及 μ-MoE）。
- 实现 **动态超参数调节机制**，根据输入自适应选择 α, p 等。
- 将 TTQ 扩展至 **vector quantization** 和 **非均匀量化格式**（如 NF4）。

---

> **总结一句话**：  
> **TTQ 开创了一种“推理时即时压缩”的新范式，摆脱了对校准数据的依赖，在保持高速的同时实现了更优的量化性能，是迈向高效、自适应 LLM 部署的重要一步。**

</details>

---

### 8. [PowerLens: Taming LLM Agents for Safe and Personalized Mobile Power Management](https://arxiv.org/abs/2603.19584)

**Authors**: Xingyu Feng, Chang Sun, Yuzhu Wang, Zhangbing Zhou, Chengwen Luo, Zhuangzhuang Chen, Xiaomin Ouyang, Huanqi Yang  
**Category**: cs.AI  
**Published**: 2026-03-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.19584v1  

#### Abstract
Battery life remains a critical challenge for mobile devices, yet existing power management mechanisms rely on static rules or coarse-grained heuristics that ignore user activities and personal preferences. We present PowerLens, a system that tames the reasoning power of Large Language Models (LLMs)...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《PowerLens: Taming LLM Agents for Safe and Personalized Mobile Power Management》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
移动设备的**电池续航**仍是用户的核心痛点。现有的电源管理机制（如 Android 的 Battery Saver、App Standby Buckets）依赖静态规则或粗粒度启发式策略，存在以下缺陷：
- **缺乏上下文感知**：无法理解用户当前活动（如导航 vs 阅读），导致资源分配不合理。
- **忽略个性化偏好**：不同用户对亮度、刷新率等参数的容忍度差异大，但系统采用统一策略。
- **安全性不足**：盲目降功耗可能破坏关键功能（如导航时关闭 GPS 或调暗屏幕）。

### 提出了什么新方法或新思路
作者提出 **PowerLens**，一个基于 **Large Language Model (LLM)** 的系统级电源管理系统，其核心创新包括：

#### ✅ 多智能体架构（Multi-Agent Architecture）
将复杂的电源管理任务分解为四个协同工作的 LLM Agent：
- **Activity Agent**：从 UI 树和设备状态识别用户活动语义（如“正在视频会议”）。
- **Policy Agent**：结合活动上下文、设备能力与用户偏好生成优化策略。
- **Execution Agent**：验证并执行策略，分两步调用 LLM（合法性检查 + 命令生成）以确保安全。
- **Feedback Agent**：通过**状态差分**（state differencing）检测用户的手动调整，作为隐式反馈。

#### ✅ 双层记忆系统（Two-Tier Memory System）
实现无需显式配置的个性化学习：
- **STM (Short-Term Memory)**：会话级缓存，记录用户锁定参数与原始事件日志。
- **LPM (Long-Term Personal Memory)**：持久化存储用户偏好规则，通过**置信度蒸馏机制**逐步提升观察到的行为模式为稳定规则。
- 学习方式：基于**隐式反馈**（implicit feedback），即当用户手动修改系统设置时视为不满意，系统自动学习。

#### ✅ PDL 约束验证框架（Propositional Dynamic Logic-based Safety Verification）
防止 LLM 幻觉导致的功能破坏：
- 定义领域特定的安全约束（如“导航期间 location_mode ≥ 3”）。
- 所有 LLM 输出动作在执行前必须通过 PDL 规则验证。
- 实现“信任但验证”（trust but verify）机制，在保留 LLM 创造性的同时保障安全性。

### 相比现有方法的优势
| 维度 | 传统方法 | PowerLens |
|------|--------|---------|
| 上下文感知 | ❌ 缺乏 | ✅ 利用 LLM 理解 UI 语义 |
| 个性化 | ❌ 固定规则 | ✅ 从隐式反馈中学习偏好 |
| 安全性 | ⚠️ 有限保护 | ✅ PDL 强制执行硬性约束 |
| 泛化能力 | ❌ 需大量训练数据 | ✅ 零样本推理（zero-shot） |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **PowerLensBench**：本文构建的新基准，涵盖 **7 类主流应用**：
  - Navigation（导航）
  - Video Watching（视频观看）
  - Meeting（会议）
  - Social（社交）
  - Music（音乐）
  - Content Feed（内容流）
  - Reading（阅读）
- 包含 **25 款主流 App**（如 Google Maps、YouTube、Zoom、Spotify 等），定义 **48 个典型任务场景**。
- 每个任务在 **3 种电量水平**（高 >60%，中 30–60%，低 <30%）下测试，共 **144 个实例**。

### 实验设置和评估指标

#### 实验平台
- 设备：OnePlus ACE 5（Snapdragon 8 Gen 3, 12GB RAM）
- 系统：Android 15 + KernelSU（需 root 权限）
- LLM 后端：Gemini-2.5-Flash（Google API）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Action Accuracy (%)** | 动作与用户真实偏好的匹配程度（加权平均） |
| **Energy Saving (ES, %)** | 相比 Stock Android 节省的能耗 |
| **Safety Violation Rate (%)** | 违反 PDL 安全约束的动作比例 |
| **User Experience Score (UES)** | 用户体验评分（1–5 分），考虑参数敏感性权重 |
| **Preference Convergence** | 用户偏好规则收敛所需时间（天数） |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Stock Android** | 无干预，默认行为（ES = 0%） |
| **Battery Saver** | Android 内建省电模式，全局限制 |
| **Rule-Based** | 基于类别预设规则，无 LLM 推理 |
| **Single-Agent LLM** | 单一 LLM 完成全部任务，无多智能体拆分、无记忆、无 PDL 验证 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 和全文）

| 方法 | Action Accuracy (%) | Energy Saving (%) | Violation Rate (%) | UES (1–5) |
|------|---------------------|-------------------|--------------------|-----------|
| Stock Android | — | 0.0 | — | — |
| Battery Saver | 48.3 | 4.6 | 0.8 | 3.6 |
| Rule-Based | 63.5 | 19.9 | 1.2 | 3.4 |
| Single-Agent LLM | 52.1 | 48.4 | 12.5 | 2.5 |
| **PowerLens (Ours)** | **81.7** | **38.8** | **0.6** | **4.3** |

### 与基线方法的对比结果
- **节能效果**：
  - PowerLens 实现 **38.8% 平均节能**，远超 Battery Saver（4.6%）和 Rule-Based（19.9%）。
  - 尽管 Single-Agent LLM 节能更高（48.4%），但牺牲了用户体验和安全性。
- **准确性与体验**：
  - PowerLens 准确率达 **81.7%**，显著优于其他所有方法。
  - 用户体验得分 **4.3/5.0**，表明其决策高度符合用户预期。
- **安全性**：
  - 安全违规率仅 **0.6%**，而 Single-Agent LLM 高达 12.5%。
  - PDL 框架消除了 **96.5% 的原始 LLM 安全违规**。

### 消融实验结果（Ablation Study）
#### 影响准确性和节能（Fig. 14a）
| 配置 | Accuracy ↓ | ES ↑ |
|------|----------|------|
| Full System | 81.7% | 38.8% |
| w/o Multi-Agent | 52.1% | 48.4% |
| w/o Memory | 71.4% | 35.1% |
| w/o Feedback | 75.2% | 37.0% |
| w/o PDL | 80.0% | 40.1% |

> **结论**：多智能体设计是准确性的关键；移除后虽节能更高，但严重损害用户体验。

#### 影响安全性和 UES（Fig. 14b）
| 配置 | Violation Rate ↑ | UES ↓ |
|------|------------------|--------|
| Full System | 0.6% | 4.3 |
| w/o PDL | 17.0% | 3.72 |
| Single-Agent LLM | 12.5% | 2.5 |

> **结论**：PDL 是安全保障的核心组件。

#### 记忆系统的影响（Fig. 11）
- 加入 LPM 后，**整体准确率提升 +10.3%**（71.4% → 81.7%）。
- 在 **Reading** 和 **Navigation** 场景中增益最大（+16.8%, +15.5%），因这些场景偏好差异大。
- 用户体验平均提升 **+0.8 分**，学生和通勤者受益最多。

#### 偏好学习过程（Fig. 10）
- 用户偏好规则在 **3–5 天内收敛**。
- 强信号（strong feedback）规则可在第 3 天达到推广阈值（confidence ≥ 0.8）。
- 支持偏好迁移：当用户习惯改变时，旧规则被新规则自然替换。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 可作为有效的系统级推理引擎**：能够理解 UI 语义并进行跨参数协调优化，实现零样本（zero-shot）电源策略生成。
2. **多智能体架构优于单体模型**：任务分解提升了系统的可解释性、鲁棒性和准确性。
3. **隐式反馈足以支持个性化学习**：无需用户主动配置，仅通过检测手动调整即可建立长期偏好模型。
4. **PDL 安全验证至关重要**：有效拦截了超过 96.5% 的潜在危险操作，保障系统可用性。
5. **实际开销极低**：系统自身仅消耗 **0.5% 日常电量**，延迟约 **12.2 秒/周期**，具备实用价值。

### 方法的局限性
1. **反馈盲区问题**：
   - 当参数变化效果延迟或不可见时（如后台进程限制），用户难以及时纠正，导致反馈缺失。
2. **隐私依赖云端推理**：
   - 当前使用云上 LLM，尽管有 PII 过滤机制，但仍存在数据外泄风险。
3. **需要 root 权限**：
   - 限制了在普通消费设备上的部署可行性。
4. **UI 语义解析依赖 Accessibility API**：
   - 对某些动态或加密 UI 元素识别能力有限。

### 未来工作方向
1. **部署轻量级模型（SLM）于设备端**：
   - 消除对云服务的依赖，增强隐私保护。
2. **扩展至更多设备类型**：
   - 如平板、可穿戴设备、电动汽车等。
3. **引入更主动的反馈机制**：
   - 例如浮动提示框让用户快速确认或否决调整，弥补状态差分的不足。
4. **探索更细粒度的状态建模**：
   - 结合传感器数据（如环境光、运动状态）进一步提升上下文感知精度。

---

> **总结一句话**：  
> PowerLens 成功将 LLM 的常识推理能力引入移动系统级资源管理，通过 **multi-agent + memory + PDL verification** 架构，在保证安全的前提下实现了高效、个性化的电源优化，为“AI 原生操作系统”提供了重要范例。

</details>

---

### 9. [ShobdoSetu: A Data-Centric Framework for Bengali Long-Form Speech Recognition and Speaker Diarization](https://arxiv.org/abs/2603.19256)

**Authors**: Md. Nazmus Sakib, Shafiul Tanvir, Mesbah Uddin Ahamed, H. M. Aktaruzzaman Mukdho  
**Category**: cs.CL  
**Published**: 2026-03-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.19256v1  

#### Abstract
Bengali is spoken by over 230 million people yet remains severely under-served in automatic speech recognition (ASR) and speaker diarization research. In this paper, we present our system for the DL Sprint 4.0 Bengali Long-Form Speech Recognition (Task~1) and Bengali Speaker Diarization Challenge (T...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*ShobdoSetu: A Data-Centric Framework for Bengali Long-Form Speech Recognition and Speaker Diarization*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **低资源语言的长语音识别与说话人分离挑战**：尽管孟加拉语（Bengali）拥有超过2.3亿使用者，但在自动语音识别（ASR）和说话人分离（speaker diarization）领域仍严重缺乏高质量标注数据。
- **真实场景下的复杂音频条件**：竞赛数据包含模糊区域（muffled zones）、背景噪音、代码混合（code-mixing）等现实干扰，现有模型难以应对。

### 提出的新方法与新思路
1. **数据为中心的ASR训练流程（Data-Centric Pipeline）**
   - 利用YouTube视频中的有声书和戏剧音频构建高质量训练语料。
   - 引入 **LLM-in-the-loop 数据清洗机制**：使用 Gemini 3 Flash 进行语言归一化（如纠正Chirp自动生成转录中的Hindi token），但仅用于窄任务（endpoint word识别），避免LLM幻觉。
   - 采用 **fuzzy-matching-based chunk boundary validation** 验证字幕块边界，提升时间对齐精度。
   - 设计 **muffled-zone augmentation** 模拟测试集中的“被遮挡麦克风”效应，增强模型鲁棒性。

2. **低资源条件下的说话人分离方案**
   - 在仅有10个标注文件的极端低资源设定下，基于 `pyannote.audio` 的 community-1 分割模型进行微调。
   - 结合系统性的 **超参数优化（Hyperparameter Optimization, HPO）**，显著降低DER。

### 相比现有方法的优势
- **无需大规模标注数据即可实现竞争性性能**：通过精心的数据工程弥补标注稀缺。
- **更贴近实际部署需求**：处理长语音、噪声、模糊音频的能力优于通用Whisper模型。
- **可靠性高**：相比端到端LLM修正，混合式（LLM + fuzzy matching）边界验证在14,000+样本上表现更稳定。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 来源于 **DL Sprint 4.0 竞赛** 提供的 **Bengali-Loop** 数据集：
  - **ASR任务**：191段YouTube来源的Bengali有声书与戏剧录音，共158.6小时；ground truth为YouTube Chirp生成的无时间戳文本。
  - **Diarization任务**：24段录音（22小时），含人工标注的说话人切换时间戳。
- 最终用于训练的ASR数据来自其中可匹配URL的72个音频文件，提取约14,000个有效chunk，并通过数据增强扩展至约20,500个训练样本。

### 实验设置与评估指标

#### ASR任务
- **模型选择**：以 `tugstugi/whisper-medium` 为基础模型（769M参数），进行全量微调（full fine-tuning）。
- **训练配置**：
  - 使用 NVIDIA A100 GPU，训练12轮（epochs）
  - Batch size: 16，学习率：2×10⁻⁵，cosine调度器，700步warmup
  - 训练/验证划分：90%/10%
- **推理策略**：使用 **beam search**，比较不同beam size的影响。
- **评估指标**：
  - **Word Error Rate (WER)**：衡量识别准确率

#### Diarization任务
- **基础模型**：`pyannote.audio` 社区版 segmentation 模型（community-1）
- **训练策略**：
  - 仅微调分割子模块（segmentation head）
  - 使用Azure Speech Services生成伪标签扩展数据
  - 应用muffled-zone数据增强
- **超参数搜索**：对以下四个参数进行网格搜索：
  - `min_duration_off`, `clustering_threshold`, `fa` (false alarm weight), `fb` (missed detection weight)
- **评估指标**：
  - **Diarization Error Rate (DER)**：综合衡量漏检、误报和混淆错误

### 基线方法对比
| 方法 | 类型 | WER/DER |
|------|------|--------|
| `tugstugi/whisper-medium` (原始) | 零样本迁移 | WER 34.8% |
| Whisper large-v3 | 多语言通用 | WER 75.0% |
| IndicWav2Vec Bengali | 自监督预训练 | WER >40% |
| WhisperX (chunk=5s) | 流水线方案 | DER ~0.274 |
| Raw pyannote 3.1 | 开源默认 | DER 0.410 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ASR任务结果
| 模型/阶段 | Public WER | Private WER |
|----------|------------|-------------|
| tugstugi base (未微调) | 34.8% | — |
| 中间模型（50%数据） | 21.893% | — |
| + HTDemucs vocal extraction | 21.00% | — |
| **最终模型（全量微调 + beam=5）** | **16.751%** | **15.551%** |

- 实现 **53% 的相对WER下降**（34.8 → 16.751）
- 排名 **Public 和 Private 榜单双第一**

#### Diarization任务结果
| 系统配置 | DER（Public） | DER（Private） |
|--------|----------------|----------------|
| Raw pyannote 3.1 | 0.410 | — |
| + segment merging | 0.290 | — |
| Community-1 + FT (10 epochs) | 0.199 | — |
| **Community-1 + FT + HPO** | **0.194** | **0.26723** |
| （边界舍入后） | **0.18999** | — |

- 在仅10个训练文件条件下，DER从0.41降至0.194（**相对下降53%**）
- 最终排名第七

### 消融实验结果
| 改进措施 | 对WER影响 | 发现说明 |
|--------|-----------|---------|
| HTDemucs语音分离 | ↓ ~0.9 pts | 表明**环境噪声是主要误差源**而非语言难度 |
| Muffled-zone augmentation | 显著改善muffled段落表现 | 模型学会在退化音频中恢复内容 |
| Beam size=1 → 5 | WER从18.42%↓至16.751% | **beam search对噪声鲁棒性至关重要** |
| Full fine-tuning vs LoRA | 全量微调收敛更快且效果相当 | 在中等规模数据下推荐使用full fine-tuning |
| 固定阈值合并（diarization） | Public提升但Private退化 | **固定规则泛化差，需模型级优化**

---

## 4. 关键结论和发现

### 主要发现
1. **数据质量远胜于模型架构选择**  
   - 数据清洗（语言过滤、边界校正）、增强（muffled模拟）带来的增益超过任何模型替换。
   
2. **LLM辅助需谨慎设计作用范围**  
   - 将LLM限制在分类任务（如预测末尾词）可避免过度修正；开放式重写会导致语法正确但声学不符的结果。

3. **训练与推理一致性重要**  
   - 微调时若使用beam=5生成目标序列，则推理也应保持相同beam size，否则性能下降。

4. **公共集表现优于私有集的现象解释**  
   - 最终模型在private test上的WER更低（15.551 < 16.751），原因在于训练数据与Chirp生成文本分布一致，而private集可能更接近训练域。

5. **领域适应 vs 泛化能力的权衡**  
   - 模型越深入拟合竞赛特定领域（YouTube drama/audiobook），其在其他真实场景（如嘈杂食堂、vlog）中的表现反而略降（OOD WER从40.92%升至41.13%）。

### 方法的局限性
- **依赖Chirp生成文本作为GT**：虽提高与测试集的一致性，但也继承了其拼写错误、遗漏等问题。
- **训练数据多样性不足**：局限于YouTube有声书与戏剧，缺乏日常对话、多人交互等场景。
- **diarization存在过拟合风险**：所有10个训练样本均用于HPO，导致public-private DER差距较大（0.19 vs 0.267）。

### 未来工作方向
- **集成去噪模块进训练流程**：将HTDemucs类denoising作为可学习组件，统一训练与测试条件。
- **引入多样化训练数据与转录校正**：加入真实世界多场景音频，并进行word-level纠错，以提升OOD泛化能力。
- **chunk质量过滤机制**：基于音频长度与文本长度比例设计置信度过滤器，剔除低质量chunk。
- **交叉验证策略改进diarization HPO**：采用leave-one-out方式防止超参数过拟合。
- **拓展至其他南亚低资源语言**：本框架具备良好可复制性，适用于类似语言环境。

---

> ✅ **总结一句话**：  
> *ShobdoSetu* 展示了在低资源语言（如Bengali）中，**精细的数据工程 + 领域适配微调** 能够超越更大模型和复杂架构，在长语音ASR与说话人分离任务上取得领先性能。

</details>

---

### 10. [SAGE: Sustainable Agent-Guided Expert-tuning for Culturally Attuned Translation in Low-Resource Southeast Asia](https://arxiv.org/abs/2603.19931)

**Authors**: Zhixiang Lu, Chong Zhang, Yulong Li, Angelos Stefanidis, Anh Nguyen, Imran Razzak, Jionglong Su, Zhengyong Jiang  
**Category**: cs.CL  
**Published**: 2026-03-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.19931v1  

#### Abstract
The vision of an inclusive World Wide Web is impeded by a severe linguistic divide, particularly for communities in low-resource regions of Southeast Asia. While large language models (LLMs) offer a potential solution for translation, their deployment in data-poor contexts faces a dual challenge: th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SAGE: Sustainable Agent-Guided Expert-tuning for Culturally Attuned Translation in Low-Resource Southeast Asia

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本论文针对**低资源语言（Low-Resource Languages, LRLs）在东南亚地区的机器翻译（Machine Translation, MT）困境**提出解决方案。当前主流的神经机器翻译（NMT）模型严重依赖大规模高质量平行语料，而这些资源在低收入和中等收入国家（LMICs）极度匮乏。此外，传统“大数据”范式不仅加剧了数字鸿沟，还因训练海量噪声数据导致高昂的**能源消耗和碳排放**。

更深层次的问题是：即使有少量数据，标准模型也难以捕捉社区对话中的**文化细微差别**（如敬语、语境依赖代词、方言混合等），导致翻译虽语法正确却文化失当，影响公共健康、教育等关键服务的可及性。

---

### 提出了什么新方法或新思路
作者提出了 **SAGE（Sustainable Agent-Guided Expert-tuning）框架**，其核心思想是：  
> **从“更多数据”（more data）转向“正确的数据”（right data）**。

该框架由两个阶段构成：

1. **Expert-Informed Data Curation（专家引导的数据筛选）**  
   - 使用一个基于**强化学习（Reinforcement Learning, RL）的智能体**，通过 **Group Relative Policy Optimization (GRPO)** 算法优化策略。
   - 智能体的任务是从大规模、高噪声的平行语料 $D_{\text{noisy}}$ 中自动筛选出与**专家构建的小型高质量参考集 $D_{\text{expert}}$** 在语义上最接近的样本子集 $D_{\text{cur}}$。
   - 奖励信号来自**语义相似度**（使用 LaBSE 编码器计算候选翻译与专家参考之间的余弦相似度），从而将专家的文化领域知识编码进数据选择过程。

2. **Parameter-Efficient Fine-Tuning（高效微调）**  
   - 在筛选出的高质量小数据集 $D_{\text{cur}}$ 上，使用 **Low-Rank Adaptation (LoRA)** 对开源 LLM（如 Qwen-3-8B）进行微调。
   - 冻结主干模型参数，仅训练低秩适配矩阵，显著降低计算开销。

---

### 相比现有方法的优势
| 维度 | SAGE 优势 |
|------|-----------|
| **性能** | 在多个低资源语言上达到新的 SOTA，超越闭源大模型（如 GPT-4o、Claude-3.5）。 |
| **效率** | 数据使用量减少 **97.1%**，训练能耗降低 **95.2%**，实现“绿色AI”。 |
| **文化适应性** | 显式建模文化对齐，能正确处理敬语、社会等级等本地化表达。 |
| **可扩展性** | 框架通用，适用于不同基础模型（Qwen、Llama、Gemma）。 |
| **人效比** | 仅需约 20 小时专家标注即可获得近峰值性能，适合资源受限场景。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **$D_{\text{noisy}}$**: 超过 5000 万句对的大规模网络爬取语料，整合自 CCMatrix、CCAligned 和 ParaCrawl。
- **$D_{\text{expert}}$**: 人工精心标注的 2,000 句对/语言，涵盖医疗、公民参与等高价值社区场景，由专业译员完成。
- **$D_{\text{eval}}$**: Asian Language Treebank (ALT)，用于跨语言评估。
- **$D_{\text{test}}$**: 每语言保留 500 句测试集，确保无数据泄露。

目标语言（共7种）：
> Bengali (bn), Filipino (fil), Hindi (hi), Khmer (km), Lao (lo), Burmese (my), Vietnamese (vi)

---

### 实验设置和评估指标

#### 模型配置
- **基础模型**：Qwen-3-8B、Llama-3.1-8B、Gemma-3-9B
- **RL Agent**：轻量级 BERT-based reward model
- **微调方式**：LoRA（rank=64, alpha=16）
- **硬件环境**：8×NVIDIA A100-80GB GPU

#### 评估指标
| 指标 | 描述 |
|------|------|
| **BLEU-4** | 衡量 n-gram 精确率，反映词汇层面匹配度（SacreBLEU 实现） |
| **COMET-22** | 基于 XLM-R 的语义评分，衡量深层语义保真度，更贴近人类判断 |
| **Avg. Tok. ↓** | 推理阶段平均 token 消耗，衡量推理效率 |
| **CO₂eq (kg)** | 使用 Algorithm 2 估算训练碳足迹，综合考虑功耗、PUE 和电网碳强度 |

---

### 基线方法对比
| 类别 | 基线模型 |
|------|--------|
| **闭源模型** | GPT-4o, Claude-3.5 Sonnet, Grok-3, Gemini-2.5 pro |
| **开源模型** | DeepSeek-v3, Gemma-3-9B, Qwen-3-8B, Llama-3.1-8B, NLLB-200-3.3B, M2M-100-1.2B |
| **消融基线** | 随机采样、启发式过滤、无专家奖励的 RL 代理 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 模型 | 平均 BLEU-4 ↑ | 平均 COMET-22 ↑ |
|------|----------------|------------------|
| GPT-4o | ~39.0 | ~83.0 |
| Grok-3 | ~40.5 | ~84.0 |
| **SAGE (Qwen-3-8B)** | **43.6** | **86.3** |

- 在 **Hindi** 上，SAGE 实现 **BLEU-4 = 48.80**，比最强闭源模型 Grok-3（43.85）高出 **+5.0 分**。
- 所有 SAGE 变体均在 **6/7 种语言**上取得 BLEU-4 和 COMET-22 的 SOTA。
- 推理速度达 **52–54 tokens/sec**，优于闭源 API 模型（平均 34 tokens/sec）。

---

### 与基线方法的对比结果
- **相比全数据微调基线**（使用 100% 噪声数据）：
  - BLEU-4 提升 **+11.46 分**（32.14 → 43.60）
  - 数据用量仅为 **3%**（即减少 97.1%）
  - 能耗从 **85.6 kg CO₂eq** 降至 **4.2 kg CO₂eq**（降幅 95.1%）

- **相比其他数据筛选方法**（Table 3）：
  - 相比无过滤基线，SAGE 带来 **+79.1% BLEU 提升**，远超 QE 过滤（+57.7%）和 BLEU-Reward RL（+52.6%）。

---

### 消融实验结果（Table 2 & Table 4）

| 配置 | 平均 BLEU-4 | 相对下降 |
|------|-------------|----------|
| Full SAGE Framework | 43.60 | — |
| - w/o Expert Reward（替换为 QE 启发式） | 38.84 | ↓4.76 |
| - w/o RL Curation（随机采样） | 33.73 | ↓9.87 |
| Baseline（全数据微调） | 32.14 | ↓11.46 |

- **统计显著性验证**（paired t-test）：
  - 所有语言上的提升均显著（p < 0.05），平均提升 **+7.20 BLEU**，p < 0.001。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **“正确的数据” > “更多的数据”**：在低资源社区翻译任务中，**高质量、文化对齐的小数据集**远胜于大规模噪声数据。
2. ✅ **专家知识可通过奖励机制注入 RL 代理**：语义相似度作为 reward signal，有效指导智能体识别文化相关样本。
3. ✅ **SAGE 是模型无关的通用框架**：在 Qwen、Llama、Gemma 上均带来巨大增益，证明其泛化能力。
4. ✅ **实现高性能与可持续性的统一**：以极低资源代价达成 SOTA 性能，推动“绿色AI”落地。
5. ✅ **文化对齐可量化且可学习**：案例研究表明 SAGE 能正确使用越南语中的敬语 “bác”，而基线模型误用 “ban”。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖专家参考集 $D_{\text{expert}}$** | 构建成本较高，可能引入标注者偏见或覆盖偏差。 |
| **领域特异性较强** | 模型专精于社区对话，在法律、科技等 out-of-domain 文本上表现可能下降。 |
| **静态一次性筛选** | 当前 RL 代理执行一次选择后不再更新，缺乏与模型训练的动态交互。 |
| **潜在的灾难性遗忘风险** | 专注特定分布可能导致对通用语言能力的遗忘（alignment tax）。 |

---

### 未来工作方向
1. **迭代式协同训练**：构建 RL Agent 与翻译模型的闭环，实现 curriculum learning 或持续学习。
2. **主动学习增强**：让模型识别不确定性高的样本，交由专家标注，进一步提升人效比。
3. **多模态扩展**：结合语音、图像上下文，提升跨模态社区内容理解。
4. **去中心化协作标注平台**：建立本地社区驱动的标注生态，促进公平数据治理。
5. **Continual Learning 技术应用**：缓解领域专业化带来的通用能力退化问题。

---

> 🔚 **最终结论**：  
> SAGE 成功验证了一条**可持续、可扩展、文化敏感**的低资源机器翻译路径。它不仅打破了“大模型=高性能”的迷思，更为全球南方（Global South）的语言平等提供了切实可行的技术方案，真正践行了 **AI for Social Good** 的愿景。

</details>

---

### 11. [A Mathematical Theory of Understanding](https://arxiv.org/abs/2603.19349)

**Authors**: Bahar Ta\c{s}kesen  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.19349v1  

#### Abstract
Generative AI has transformed the economics of information production, making explanations, proofs, examples, and analyses available at very low cost. Yet the value of information still depends on whether downstream users can absorb and act on it. A signal conveys meaning only to a learner with the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Mathematical Theory of Understanding

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**信息吸收瓶颈**（absorption bottleneck）问题。尽管生成式AI极大地降低了信息生产成本，使得解释、证明、示例和分析变得唾手可得，但这些信息的价值最终取决于下游用户是否能够理解并利用它们。一个信号对具备先决条件的学习者可能是清晰的解释，而对缺乏背景知识的用户则可能只是噪声。

传统信息理论关注信道容量和编码效率，但忽略了接收端的解码能力依赖于已有知识结构这一事实。本文提出，**理解是一个结构性过程**，其速度受限于学习者的“心智”（mind）结构。

---

### 提出的新方法与新思路
作者提出了一个**基于闭包系统**（closure system）的数学框架来建模“理解”的过程。其核心要素包括：

- **Mind（心智）**：定义为一个三元组 $(C, A_m, \mathcal{E}_m)$，其中：
  - $C$ 是概念空间（concept space）
  - $A_m$ 是公理集（axioms），即初始已知概念
  - $\mathcal{E}_m$ 是扩展规则集（expansion rules），表示掌握某些前提概念后可解锁新概念

- **理解闭包**（understanding closure）$\text{cl}_m(K)$：从当前知识集 $K$ 出发，在扩展规则下能推导出的所有可达概念集合。

- **可达获得概念集族**（reachable acquired concept sets）$\mathcal{K}_m$：通过一系列符合前提条件的教学步骤所能达到的知识状态集合。在有限假设下，该集合构成一个 **antimatroid** 或等价地，一个 **learning space**。

- **教学动态模型**：将教学建模为教师向具有潜在目标概念 $O$ 的学习者发送信号的过程。信号需经由**前提门控解析器**（prerequisite-gated parser）过滤，只有当目标概念当前“有序”（ordered）时才能被解析，否则坍缩为 null 观测。

- **相对随机性**（relativity of randomness）：同一信号对一个学习者是信息，对另一个可能是噪声——这取决于其当前知识状态与心智结构。

---

### 相比现有方法的优势
| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| **信息观** | 信息是信号本身的属性（Shannon） | 信息是信号与学习者心智之间的关系 |
| **学习模型** | 黑箱更新（如神经网络梯度下降） | 显式建模前提依赖结构（combinatorial structure） |
| **教学复杂性** | 教学维度（teaching dimension）仅考虑识别难度 | 同时考虑**结构性障碍**（structural barrier）和**认知障碍**（epistemic barrier） |
| **资源分配** | 假设平滑收益 | 揭示非凹回报与阈值效应 |

> ✅ **优势总结**：本文提供了一个形式化工具，揭示了教学效率的根本限制来源于学习者的内部结构，而非仅仅是信息量或策略设计问题。

---

## 2. 核心实验方法和设置

> ⚠️ 注意：本论文是一篇**理论研究**，并未进行传统意义上的机器学习实验（无数据集、无代码实现）。所谓“实验”实为**数学建模与定理推导**，并通过构造反例验证边界情况。

### 使用的方法与设定
- **理论框架**：结合了组合优化中的 **antimatroid / learning space** 理论、信息论（Shannon, Blackwell）、以及教学理论（machine teaching）。
- **建模对象**：
  - 学习者：抽象为具有固定前提结构的心智 $m$
  - 教师：知道目标 $O$，选择信号序列以最小化完成时间
  - 信号系统：$(Z, \text{tgt})$，其中 $\text{tgt}: Z \to C$ 将原始信号映射到其所教授的概念
- **动态过程**：
  - 时间离散化：每轮教师发送一个信号 $Z_t$
  - 学习者观测 $Y_t = p_m(Z_t, K_{t-1})$，若可解析则更新知识集 $K_t = K_{t-1} \cup \{\text{tgt}(Y_t)\}$
  - 学习者维护关于目标 $O$ 的贝叶斯信念 $\tau_t$

### 评估指标（理论意义）
- **完成时间**（completion time）$T$：满足以下两个条件的最早时刻：
  1. **目标获取**：$O \in K_T$
  2. **目标识别**：$\tau_T(O) = 1$
- **最优成功概率** $V(t)$：在 $t$ 轮内完成教学的最大概率
- **期望完成时间下界**：$\mathbb{E}[T] \geq \max\left\{ \mathbb{E}[L_m(O)], \frac{H(O)}{C^*_{\max}} \right\}$

### 基线对比方式
并非数值比较，而是通过构造性证明展示：
- 个性化教学 vs 公共广播教学的时间差距
- 不同心智结构下的教学路径差异
- 阈值前后性能跃变

---

## 3. 主要实验结果和性能指标

### 关键理论结果（性能边界）

#### 定理 4.11（全局结构性-信息性下界）
$$
\mathbb{E}[T] \geq \max\left\{ \mathbb{E}[L_m(O)],\ \frac{H(O)}{C^*_{\max}} \right\}
$$
其中：
- $L_m(c)$：从公理集到达含 $c$ 的状态所需的最短路径长度（结构性距离）
- $H(O)$：目标的香农熵（初始不确定性）
- $C^*_{\max} = \max_{K \in \mathcal{K}_m} C_m(K)$：最大解析熵界（通道容量）

> 🔹 这表明教学时间受双重约束：必须跨越结构深度，也必须传输足够信息。

---

#### 命题 5.3（确定性目标的阶跃函数）
对于确定性目标 $g$，存在严格阈值：
$$
V_g(t) =
\begin{cases}
0, & t < L_m(g) \\
1, & t \geq L_m(g)
\end{cases}
$$
> 🔹 在达到结构深度前，任何教学都无效；一旦达到，即可保证完成。

---

#### 定理 5.6（通用广播的线性惩罚）
存在构造使得：
- 对每个学习者 $i$，个性化教学可在 $L$ 步内完成
- 任意公共广播序列要使所有 $k$ 个学习者均完成，则至少需要 $k(L-1)+1$ 步

> 🔹 广播教学代价随学习者类型数呈**线性增长**，因无法共享私有前提链。

---

### 消融实验（理论层面）
虽然没有传统消融实验，但文中通过多个例子展示了不同组件的影响：

| 组件变化 | 结果影响 |
|--------|--------|
| 改变心智结构（例 2.4） | 不同学习路径，相同起点无法共用课程 |
| 移除前提门控 | “相对随机性”消失，信息不再依赖状态 |
| 引入记忆缓冲机制（Remark 2.17） | 可降低教学时间下界（早发信号可延迟解析） |

---

## 4. 关键结论和发现

### 主要发现
1. **理解是结构性的**：能否理解一个概念不仅取决于信息本身，更取决于学习者是否已掌握其前提。
2. **教学存在结构性阈值**：在未达到目标的结构深度前，额外教学资源无效；一旦越过，即可快速完成。
3. **非凹回报与资源错配风险**：均匀分配训练资源可能导致零产出，集中投入少数人反而更高效。
4. **广播教学存在内在低效性**：面对异构学习者，公共课程必须重复支付各心智的私有前提成本，导致线性开销。
5. **信息的相对性**：“随机性”不是信号固有属性，而是观察者心智结构的产物 —— 即 **relativity of randomness**。

---

### 方法的局限性
- **静态心智假设**：心智结构 $A_m, \mathcal{E}_m$ 被视为固定不变。现实中人类可通过元学习改变自身架构。
- **无遗忘机制**：模型中知识永不丢失，不适用于存在遗忘或干扰的情境。
- **无主动探索**：学习者被动接受信号，未建模主动提问或自我驱动学习。
- **理想化解析器**：sharp parsing 假设要么完全解析，要么完全失效，忽略部分理解的可能性。
- **有限表达粒度**：要求每个扩展规则的前提集为有限，限制了对无限归纳推理的建模。

---

### 未来工作方向
1. **动态心智演化模型**：允许 $\mathcal{E}_m$ 随时间演进，模拟认知发展或元学习过程。
2. **带记忆的延迟解析模型**：引入缓冲区，允许存储未解析信号并在后续激活时重新处理。
3. **主动教学博弈**：建模师生之间的策略互动，如学生提问、教师试探。
4. **应用于AI教育系统设计**：指导自适应课程推荐、个性化辅导路径规划。
5. **连接神经网络可解释性**：将DNN的中间表示映射为概念空间，分析其隐式前提结构。
6. **扩展至社会学习场景**：研究群体中知识传播如何受个体心智结构差异影响。

---

## 总结

| 维度 | 内容 |
|------|------|
| **核心思想** | 理解的速度受限于学习者的前提结构，信息的有效性是相对的 |
| **关键创新** | 提出 mind + closure + teaching dynamics 的形式化框架，揭示结构性瓶颈 |
| **主要结论** | 存在教学阈值、非凹回报、广播线性惩罚、相对随机性 |
| **适用领域** | 教育科技、AI教学系统、技能形成模型、组织学习、认知科学 |
| **一句话概括** | **你听不懂，不是因为我说得不够多，而是因为你还没学会听懂它所需要的前置知识。**

</details>

---

### 12. [Quantifying Gate Contribution in Quantum Feature Maps for Scalable Circuit Optimization](https://arxiv.org/abs/2603.19805)

**Authors**: F. Rodr\'iguez-D\'iaz, D. Guti\'errez-Avil\'es, A. Troncoso, F. Mart\'inez-\'Alvarez  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.19805v1  

#### Abstract
Quantum machine learning offers promising advantages for classification tasks, but noise, decoherence, and connectivity constraints in current devices continue to limit the efficient execution of feature map-based circuits. Gate Assessment and Threshold Evaluation (GATE) is presented as a circuit op...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前量子机器学习（QML）在 **Noisy Intermediate-Scale Quantum (NISQ)** 设备上面临诸多挑战，主要包括：
- **噪声与退相干**：导致计算错误累积。
- **连接性限制**：硬件拓扑约束增加了电路深度和SWAP门数量。
- **资源效率低下**：传统优化方法多关注电路深度最小化，但未系统评估单个量子门对最终计算结果的实际贡献。

这些问题使得基于 **feature map** 的量子电路难以高效执行，且容易因冗余或低效门的存在而降低模型性能。

### 提出的新方法与新思路
本文提出了 **Gate Assessment and Threshold Evaluation (GATE)** 方法论，其核心是引入了一个全新的 **Gate Significance Index (GSI)**。

#### GSI 的构成
GSI 是一个综合量化指标，通过结合以下三个关键物理量来评估每个量子门的重要性：
- **Fidelity (保真度)**：衡量门操作前后量子态的变化程度。
- **Entanglement (纠缠)**：衡量该门对生成量子纠缠的贡献。
- **Sensitivity (敏感性)**：衡量参数化门对参数扰动的响应强度，反映其鲁棒性。

GSI 的计算公式为：
$$
\text{GSI} = \frac{F + E + (1 - P)}{3}
$$
其中 $F$、$E$、$P$ 分别归一化到 [0,1] 区间。高 GSI 值表示该门对计算至关重要。

#### GATE 方法流程
1. 计算原始电路中所有门的 GSI。
2. 设置一个 GSI 阈值范围 $[\text{GSI}_l, \text{GSI}_u)$。
3. 迭代移除 GSI 低于阈值的门，生成多个简化电路。
4. 在验证集上评估这些简化模型的性能（准确率、运行时间等）。
5. 根据预定义标准（最佳准确率、最快时间、最佳平衡）选出最优模型。
6. 在测试集上进行最终评估。

### 相比现有方法的优势
| 特性 | 传统方法 | GATE 方法 |
|------|---------|-----------|
| **优化目标** | 仅减少门数/深度 | 综合考虑准确性、效率与稳定性 |
| **门级分析** | 缺乏 | 显式量化每个门的贡献 |
| **硬件适应性** | 多为模拟器设计 | 支持真实硬件（通过测量估计 GSI） |
| **误差缓解** | 可能增加复杂度 | 移除低贡献门可直接减少噪声源 |
| **通用性** | 常依赖特定架构 | 方法独立于硬件，可跨平台应用 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用 **9 个二分类数据集**，涵盖医疗、材料科学等领域：
- **PMLB 套件中的数据集**：`BreastW`, `Corral`, `Glass2`, `Monk`, `Flare`, `Vote`, `Saheart`
- **其他来源**：`Heart`（心脏病预测）、`Fitness`（健身会员留存）

所有数据集均为二分类任务，符合所用 QML 模型的要求。

### 实验设置
#### 执行环境
- **Free Noise Simulator (FNS)**：理想无噪声模拟器。
- **Noise Simulator (NS)**：基于 IBM `ibm_brisbane` 后端的真实噪声模型。
- **Real Device (RD)**：在真实的 IBM `ibm_strasbourg` 127-qubit 超导处理器上运行。

#### QML 模型
- **PegasosQSVM**：基于量子核的支持向量机。
- **Quantum Neural Network (QNN)**：变分量子线路。

#### 评估指标
- **Accuracy (A)**：分类准确率。
- **Execution Time (T)**：模型训练/推理耗时。
- **Balanced Metric (B)**：结合准确率提升与时间节省的综合评分：
  $$
  B = (A_n - A_b) + \left(1 - \frac{T_n}{T_b}\right)
  $$
  其中下标 $n$ 表示新模型，$b$ 表示基线模型。

#### 基线方法对比
- **Baseline Model**：保留全部原始门的完整电路。
- GATE 方法生成的多个简化版本作为对比对象。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 电路压缩效果
- 最大实现了 **高达 40% 的门数减少**（如 Heart 数据集从 52 降至 24 门）。
- 平均压缩率约为 20–30%，具体取决于数据集和模型。

#### 准确率表现
- 在多数情况下，**简化后的模型保持甚至提升了准确率**。
- 例如，在 `BreastW` 上，PegasosQSVM 的准确率从基线 0.807 提升至 **0.900**。
- 在 `Corral` 上，准确率从 0.562 提升至 **1.000**（FNS），表明原电路存在冗余。

#### 运行时间
- 所有场景下，**运行时间显著下降**。
- 以 `Heart` 为例，FNS 中时间从 127s 降至 92s；NS 中从 1541s 降至 1394s。

#### 性能排名（RATB）
- 最优配置通常出现在 **中间 GSI 阈值**，而非极端值。
- 如 `Heart` 在 PegasosQSVM 下的最佳平衡模型（RB=1）对应 GSI=0.633，此时门数为 43（原为 52）。

### 与基线方法的对比结果
| 指标 | 对比结果 |
|------|----------|
| **Accuracy** | 7/9 数据集优于或持平基线 |
| **Runtime** | 所有数据集均更快（平均提速 ~30%） |
| **Balanced Score** | 多数达到 RB=1，说明综合性能更优 |

### 消融实验结果
虽然文中未明确标注“消融实验”，但通过不同 GSI 阈值下的性能变化曲线，实际上完成了对 **门重要性** 的系统性分析：
- **过低阈值**（保留太多门）：接近基线，未能有效减小电路。
- **过高阈值**（移除过多门）：准确率急剧下降，说明关键门被误删。
- **中间阈值**：实现最佳权衡，证明 GSI 能有效识别冗余门。

此外，作者还测试了不同 **simulator backend**（DM, MPS, TN, RD）下的 GSI 计算开销，发现：
- **MPS** 和 **TN** 在扩展性上远优于 **DM**。
- **RD** 方法虽慢，但成本随 qubit 数增长缓慢，适合大规模部署。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **适度压缩优于全保留**：  
   并非所有原始门都必要。**中间 GSI 阈值** 得到的模型往往在准确率和效率之间取得最佳平衡。

2. ✅ **移除低贡献门可提升性能**：  
   冗余或不稳定的门会引入额外噪声。移除它们不仅能加速，还能 **提高预测准确率**。

3. ✅ **GSI 具有跨环境一致性**：  
   在 FNS、NS 和 RD 上，GSI 排序趋势一致，验证了其物理意义的有效性。

4. ✅ **兼容性强**：  
   GATE 可与动态解耦（Dynamic Decoupling）等 error mitigation 技术结合，进一步增强鲁棒性。

5. ✅ **支持真实硬件部署**：  
   即使无法访问量子态，也可通过辅助电路和测量重构 GSI，实现在真实设备上的优化。

### 方法的局限性
1. **GSI 估算精度受噪声影响**：  
   在高噪声环境下，测量统计波动可能影响 GSI 的准确性。

2. **独立性假设限制**：  
   GSI 假设门的影响可独立评估，但在高度纠缠的电路中，移除一个门可能引发非局域效应。

3. **计算开销**：  
   尤其是在真实设备上，需运行大量辅助电路来估计 GSI，总采样次数为 $O(N)$，仍有一定延迟。

4. **适用范围**：  
   当前主要适用于存在冗余结构的 QML feature maps，对已高度优化的算法可能增益有限。

### 未来工作方向
1. **自适应阈值选择**：开发自动确定最优 GSI cut-off 的策略。
2. **改进噪声下的估计技术**：利用 error mitigation 或机器学习提升 GSI 估计鲁棒性。
3. **考虑门间交互**：将 GSI 扩展为群体贡献评估，捕捉门之间的协同作用。
4. **动态权重机制**：根据不同任务动态调整 Fidelity、Entanglement、Sensitivity 的权重。
5. **扩展至更大规模系统**：应用于 multi-class 分类、hybrid quantum-classical 架构及更大规模量子芯片。

---

> 🔗 **补充材料**：代码、数据集和详细结果已公开于 GitHub：  
> [https://github.com/Data-Science-Big-Data-Research-Lab/GATE_GSI](https://github.com/Data-Science-Big-Data-Research-Lab/GATE_GSI)

</details>

---

### 13. [AgenticRS-EnsNAS: Ensemble-Decoupled Self-Evolving Architecture Search](https://arxiv.org/abs/2603.20014)

**Authors**: Yun Chen, Moyu Zhang, Jinxin Hu, Yu Zhang, Xiaoyi Zeng  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.20014v1  

#### Abstract
Neural Architecture Search (NAS) deployment in industrial production systems faces a fundamental validation bottleneck: verifying a single candidate architecture pi requires evaluating the deployed ensemble of M models, incurring prohibitive O(M) computational cost per candidate. This cost barrier s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AgenticRS-EnsNAS: Ensemble-Decoupled Self-Evolving Architecture Search 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
工业级推荐系统中广泛采用 **ensemble-based deployment**（即部署 $ M $ 个独立模型构成预测集成）以提升鲁棒性和准确性。然而，在此类系统中进行 **Neural Architecture Search (NAS)** 面临一个根本瓶颈：验证每一个候选架构都需要训练并评估整个 $ M $ 模型的集成，导致单次验证成本为 $ O(M \times C_{\text{learner}}) $，在 $ M=50\sim200 $ 的实际场景下计算开销极高，严重限制了架构迭代频率。

现有方法存在以下缺陷：
- **代理指标（proxy metrics）**：如 Zico Abdelfattah 等提出的零成本代理，在复杂 CTR 任务中与最终集成性能相关性差。
- **参数共享（parameter sharing）**：如 ENAS、DARTS 引入权重耦合偏差，且搜索空间受限于超网络子图。
- **早停策略（early stopping）**：可能误判收敛较慢但潜力高的架构。

### 提出的新方法与核心思想
本文提出 **Ensemble-Decoupled Architecture Search**，其核心是通过 **Ensemble-Decoupled Theory** 实现从“全集成训练”到“常数成本评估”的解耦。

#### 核心理论：单调改进条件（Monotonic Improvement Condition）
在同构假设（homogeneity assumption）下——即集成成员是同一架构的不同独立训练实例——定义三个可估计的架构级属性：
- $ \Delta E(\pi) = E(\pi) - E(\pi_{\text{old}}) $：新架构相对于当前基线的期望误差变化
- $ p(\pi) $：独立实例间的预测相关性（衡量多样性）
- $ \sigma^2(\pi) $：预测方差

**定理 3.1（单调集成改进）**：若满足  
$$
p(\pi) < p(\pi_{\text{old}}) + \frac{M}{M-1} \cdot \frac{\Delta E(\pi)}{\sigma^2(\pi)}
$$  
则将当前集成替换为 $ M $ 个 $ \pi $ 架构的实例，能保证降低整体集成误差。

该条件允许仅通过训练少量代理模型（通常 2–3 个）来估计上述量，从而避免对每个候选都训练完整集成。

### 相比现有方法的优势
| 维度 | 传统 NAS | 本方法 |
|------|--------|-------|
| 单候选搜索成本 | $ O(M) $ | $ O(1) $ |
| 验证方式 | 完整训练 $ M $ 模型 | 只需双模型轻量训练 + 历史统计 |
| 理论保障 | 无或启发式 | 有充分条件保证单调改进 |
| 搜索灵活性 | 受限于代理/权重共享 | 支持闭式优化、可微优化、LLM 驱动搜索 |

此外，框架揭示了两种正交增益机制：
- **Base Diversity Gain**：源于基础模型本身低相关性（$ p < 1 $），随 $ M $ 增大而增强
- **Accuracy Gain / Dropout Gain**：来自架构优化带来的精度提升或主动引入多样性（如特征丢弃）

---

## 2. 核心实验方法和设置

> ⚠️ 注意：本文目前为 arXiv 预印本，**尚未包含完整的实验部分**。所有实验结果均为“计划中”或基于内部试点研究的初步观察。

### 使用的数据集（计划）
- **Criteo Display Advertising Challenge Dataset**（CTR 预测标准基准）
- **Avazu Click-Through Rate Prediction Dataset**
- **NAS-Bench-201**（用于离散架构搜索测试）

### 实验设置与评估指标
| 类别 | 设置 |
|------|-----|
| 集成规模 $ M $ | 测试 $ M \in \{10, 50, 100\} $ |
| 评估指标 | AUC, LogLoss, Ensemble MSE |
| 成本对比 | 总 GPU 小时数（Training Hours） |
| 搜索预算 | 固定资源下比较探索候选数量 $ N_{\text{trials}} $ |

### 基线方法对比（计划）
- **Traditional NAS**：完整训练每候选 $ M $ 模型
- **Random Search**
- **Evolutionary Baselines**
- **LLM-NAS without theoretical filter**

### 消融实验设计（隐含）
- 是否使用 Theorem 3.1 作为接受准则的影响
- 不同估计方式（zero-cost proxy vs dual-training）的效果
- 同构性假设的有效性分析

---

## 3. 主要实验结果和性能指标

> 所有结果均来自 **第 6.2 节“Preliminary Observations”**，非完整实验。

### 关键性能数据（初步观察，$ M=50 $, Criteo 子集）
| 指标 | 结果 |
|------|------|
| **搜索成本缩放** | 与 $ M $ 无关，呈 $ O(1) $ 特性；传统方法随 $ M $ 线性增长 |
| **成本节省倍数** | 在 $ M=100, N_{\text{trials}}=1000 $ 下：<br>传统：~100,000 GPU 小时<br>本方法：~1,100 GPU 小时 → **约 90× 节省** |
| **候选探索能力** | 相同预算下可探索 **90× 更多候选架构** |
| **单调条件有效性** | 正确接受 ~85% 的改进候选，拒绝 ~70% 的退化候选 |
| **闭式解准确性** | 特征保留率 $ \alpha^* $ 的理论预测值与实证最优值相差 < 5% |

### 与基线方法的对比结果（预期）
- **效率上显著超越**所有依赖完整集成验证的方法（速度提升 $ \sim M \times $）
- **效果上优于随机搜索与无理论引导的 LLM-NAS**，因后者缺乏有效剪枝机制
- **相比 proxy-based 方法更具鲁棒性**，因其结合轻量训练而非纯梯度代理

### 消融实验结果（暂无）

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **首次建立单学习器属性与系统级集成性能之间的理论桥梁**，实现了 NAS 中搜索成本与集成规模 $ M $ 的解耦。
2. ✅ **提出统一框架支持多种搜索范式**：
   - 连续可解析管道：**闭式求解**（closed-form optimization）
   - 连续不可解析管道：**约束可微优化**
   - 离散拓扑结构：**LLM 驱动 + 迭代单调接受算法**（Algorithm 1）
3. ✅ **揭示双路径增益机制**：
   - Base Diversity Gain（固有）
   - Accuracy/Regularization-induced Gain（可控）
4. ✅ **工程指导意义强**：
   - 最优 dropout rate $ \beta^* \propto \frac{(M-1)}{M} $，表明大集成可容忍更高丢弃率
   - 收益递减点约在 $ M \approx 50 $，超过后边际收益下降

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **同构性假设（Homogeneity Assumption）** | 要求所有集成成员为同一架构的独立训练实例，不适用于异构集成（heterogeneous ensembles）。作者承认此为近似成立，尤其在一般网络结构搜索中。 |
| **代理估计可靠性依赖建模假设** | 如 $ \Delta E(\pi) $ 使用 zero-cost proxy 或双模型验证损失，需校准；若代理失真可能导致错误决策。 |
| **当前仅支持均匀加权集成** | 未考虑非均匀权重分配下的优化空间。 |
| **尚无公开完整实验验证** | 当前仅为理论预印本，完整实验将在期刊投稿中补充（目标 Q2 2026）。 |

### 未来工作方向
1. **扩展至异构集成**：放宽同构假设，引入有界偏差分析。
2. **代理可靠性分析**：推导 $ \Delta E(\pi) $ 估计的置信区间与失败模式。
3. **加权集成优化**：支持动态权重调整以进一步平衡 diversity-accuracy trade-off。
4. **在线演化机制**：支持流式数据与动态集成规模调整。
5. **与 DARTS 结合**：将单调条件作为可微搜索中的约束项。
6. **Meta-Learning 加速估计**：跨任务学习预测 $ \Delta E, p, \sigma^2 $，消除搜索阶段的训练需求。

---

## 总结

**AgenticRS-EnsNAS** 是首个为工业级集成部署场景量身打造的 **理论驱动型 NAS 框架**。它通过 **Ensemble-Decoupled Theory** 和 **单调改进条件**，成功将 NAS 的搜索成本从 $ O(M) $ 降至 $ O(1) $，同时保留 $ O(M) $ 的部署成本仅用于胜出者，极大提升了架构探索效率。

尽管当前仍处于理论构建阶段，其实验验证正在进行中，但其提出的 **gain decomposition**、**solution unification** 以及 **LLM + 理论过滤** 的新范式，已展现出推动 NAS 从“启发式探索”迈向“自演化系统”的巨大潜力。

> 🔗 **代码与复现承诺**：作者承诺在期刊提交时开源实现（Algorithm 1、条件检查脚本、特征袋装案例复现代码）及预计算统计，仓库链接待公布。

</details>

---

### 14. [Learning to Disprove: Formal Counterexample Generation with Large Language Models](https://arxiv.org/abs/2603.19514)

**Authors**: Zenan Li, Zhaoyu Li, Kaiyu Yang, Xiaoxing Ma, Zhendong Su  
**Category**: cs.AI  
**Published**: 2026-03-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19514v1  

#### Abstract
Mathematical reasoning demands two critical, complementary skills: constructing rigorous proofs for true statements and discovering counterexamples that disprove false ones. However, current AI efforts in mathematics focus almost exclusively on proof construction, often neglecting the equally import...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Learning to Disprove: Formal Counterexample Generation with Large Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于大语言模型（LLM）的数学推理研究主要集中于**定理证明**（theorem proving），而对**反例生成**（counterexample generation）这一同等重要的任务关注甚少。然而，在数学中，反例对于推翻错误猜想、精炼理论和提升模型自省能力至关重要。

本文首次系统地研究了**形式化反例生成**（formal counterexample generation）——即让LLM不仅提出反例，还要用形式化语言（如Lean 4）生成可被自动验证的证明。

该任务面临两大挑战：
- **训练数据稀缺**：现有反例数据集（如CounterMath）仅有约1.2K自然语言问题，不足以支撑有效训练。
- **奖励信号稀疏**（sparse reward）：当模型无法生成正确反例时，强化学习框架中的奖励为零，导致训练停滞。

---

### **提出的新方法与创新思路**

#### ✅ **Symbolic Mutation Strategy（符号化变异策略）**
- 从大量已证明的定理（seed theorems）出发，**移除一个必要假设**（drop a necessary hypothesis），使其变为假命题，从而**构造出新的反例问题**。
- 利用Lean 4定理证明器分析原定理证明过程，确保所删假设非冗余，保证变异后问题的有效性和难度。
- 此方法可大规模合成高质量、多样化的反例训练数据。

#### ✅ **Multi-Reward Guided Training（多奖励引导训练）**
- 设计双奖励机制以缓解稀疏奖励问题：
  1. **主奖励**（`r_M`）：验证生成的反例是否能证明变异后的存在性命题（mutated version）。
  2. **辅助奖励**（`r_H`）：验证同一反例是否满足被删除的假设的否定（dropped hypothesis）。
- 即使主任务失败，只要辅助任务成功，仍可获得部分奖励，显著提升训练稳定性与效率。
- 最终样本权重为 `r = α·r_M + (1−α)·r_H`，其中 α ∈ [0,1]。

#### ✅ **Integrated Framework：Expert Iteration + Weighted Fine-Tuning**
- 结合专家迭代（expert iteration）框架，持续生成候选解并筛选成功案例进行监督微调。
- 使用加权损失函数，赋予高奖励样本更高权重，进一步优化训练效果。

---

### **相比现有方法的优势**
| 维度 | 现有方法 | 本工作 |
|------|--------|-------|
| 数据来源 | 手动标注 / 小规模NL数据（如CounterMath） | 自动合成575K形式化反例数据 |
| 形式化程度 | 多为自然语言输出 | 要求完整Lean 4形式证明，支持自动验证 |
| 训练信号 | 单一奖励，易陷入稀疏困境 | 双重奖励机制，增强反馈密度 |
| 推理范式 | 链式思维（chain-of-thought）为主 | “猜测-验证”循环，更贴近反例发现本质 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
四个来源共提供约 **321,929 条种子定理**，经变异生成 **575,039 条反例问题**：

| 数据源 | 描述 |
|-------|------|
| **Mathlib** | Lean 4官方数学库，涵盖广泛基础数学概念 |
| **Leanworkbook** | 来自10余个领域的形式化习题集 |
| **MiniF2F** | 中学竞赛级数学问题的形式化版本 |
| **PutnamBench** | 大学生数学竞赛难题，更具挑战性 |

> 所有数据均通过自定义Lean tactic `mutate` 进行处理，确保语法正确且逻辑有效。

---

### **实验设置**

#### **模型架构**
- **非正式推理模型**（Informal Reasoning）：Qwen3 8B —— 提出反例候选。
- **形式化推理模型**（Formal Reasoning）：DeepSeek-Prover-v2 7B —— 生成Lean 4证明。

#### **训练流程**
- 采用 **Algorithm 1** 所述的两阶段框架：
  1. **数据变异阶段**：批量生成反例问题。
  2. **专家迭代阶段**：每轮执行大规模推理 → 定理验证 → 收集成功样本 → 加权微调。

#### **评估指标**
- **pass@k**（k=1,4,9）：在k次采样中至少有一次成功的比例。
- 在以下三个新构建的基准上测试：
  - **FOR-COUNTER**：1,058个来自教科书的反例问题（经Kimina-Autoformalizer形式化）。
  - **VERI-FORMALIZE**：3K个由错误形式化产生的不可证命题。
  - **VERI-REASON**：3K个在定理证明过程中出现错误推理步骤的问题。

#### **基线方法对比**
- **Proprietary Reasoning Models**：
  - Gemini-2.5-Flash, Grok-3-mini, GPT-4.1-mini, Deepseek-R1
- **Open-Sourced Neural Provers**：
  - Leanabell-prover, STP-prover, Kimina-prover-distill, Goedel-prover-v2, Deepseek-prover-v2

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **RQ1：数据变异效率**
- 平均**变异率**（mutation ratio）达 **1.65–2.48**（即每个种子定理平均生成近2个有效反例）。
- 平均执行时间仅 **0.3–0.71秒/定理**，计算高效（见Figure 3）。

#### ✅ **RQ2：多奖励训练优势**
- 在验证集上的 **pass@1** 达到 **49%**（multi-reward） vs. **43%**（single-reward）。
- 收敛速度更快，最终性能更高（见Figure 4），表明多奖励机制显著提升了训练效率与稳定性。

#### ✅ **RQ3：整体性能超越SOTA**

| 模型 | FOR-COUNTER √@1 | VERI-REASON √@1 | VERI-FORMALIZE √@1 |
|------|------------------|------------------|---------------------|
| 最强开源基线（Deepseek-prover-v2） | 127 | 144 | 69 |
| **本文方法（Ours）** | **222** (+95) | **213** (+69) | **174** (+63) |

> 相对提升幅度达 **47%–74%**（见Table 2最后一行百分比）。

此外，在所有k值（1,4,9）下均取得最优表现，说明泛化能力强。

---

### **消融实验结果（Ablation Study）**

#### ✅ **模块有效性验证**（Table 4）
| 模型组合 | Validation √@1 | FOR-COUNTER √@1 |
|---------|----------------|------------------|
| Qwen3 × Ds-prv-v2（原始） | 26.2% | 14.1% |
| Ours × Ds-prv-v2（仅微调Qwen3） | 30.9% | 15.2% |
| Qwen3 × Ours（仅微调Prover） | 47.2% | 19.1% |
| **Ours × Ours（全微调）** | **49.8%** | **20.9%** |

> 表明两个模块都得到提升，但**形式化证明模型的改进贡献更大**。

#### ✅ **多奖励机制作用明确**
- 即使主任务未完成，辅助奖励也能维持训练信号流动，避免梯度消失。
- 减少对简单样本的过拟合，鼓励探索复杂反例空间。

---

## **4. 关键结论和发现**

### **主要发现**
1. **反例生成是可训练的**：通过合理设计数据合成与训练机制，LLM可以学会系统性地生成形式化反例。
2. **symbolic mutation 是高效的数据增强手段**：不仅能扩展数据规模，还能控制问题难度与有效性。
3. **multi-reward 显著缓解稀疏奖励问题**：为未来在困难逻辑任务上的RL训练提供了新范式。
4. **“guess-and-check”范式优于纯deductive推理**：更适合反例这类需要创造性试探的任务。

---

### **局限性**
1. **合成数据质量参差**：尽管数量庞大，但部分生成问题重复或过于简单，影响训练效率。
2. **模型容量限制**：
   - 当前使用的是7B–8B级别模型，在复杂计算或长链推理中表现不佳。
   - 形式化模型有时忽略提供的反例，自行构造错误证明。
3. **依赖Lean生态系统**：方法目前局限于Lean 4环境，向其他形式化系统迁移需重新适配。

---

### **未来工作方向**
- 引入**数据去重与难度分级机制**，提升合成数据质量。
- 探索**更大规模模型**（如70B）或结合**tool use**（计算器、SAT solver等）来增强数值与逻辑能力。
- 将框架推广至**交互式数学助手**场景，实现实时猜想检验与理论修正。
- 研究如何将反例生成能力**反哺定理证明**，实现“自省式”数学推理闭环。

---

> 🔚 **总结一句话**：  
> 本文开创性地将LLM应用于**形式化反例生成**，提出**symbolic mutation + multi-reward training**框架，在数据稀缺与稀疏奖励双重挑战下实现了对现有SOTA的显著超越，为构建具备自我纠错能力的数学AI迈出了关键一步。

</details>

---

### 15. [HyEvo: Self-Evolving Hybrid Agentic Workflows for Efficient Reasoning](https://arxiv.org/abs/2603.19639)

**Authors**: Beibei Xu, Yutong Ye, Chuyun Shen, Yingbo Zhou, Cheng Chen, Mingsong Chen  
**Category**: cs.AI  
**Published**: 2026-03-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19639v1  

#### Abstract
Although agentic workflows have demonstrated strong potential for solving complex tasks, existing automated generation methods remain inefficient and underperform, as they rely on predefined operator libraries and homogeneous LLM-only workflows in which all task-level computation is performed throug...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*HyEvo: Self-Evolving Hybrid Agentic Workflows for Efficient Reasoning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **agentic workflows** 在解决复杂任务时面临两大瓶颈：
- **效率低下**：依赖预定义的 operator 库进行流程编排，缺乏自动化，且所有计算均通过 LLM 的概率推理完成，导致高延迟和高成本。
- **同质化设计（Homogeneous Workflows）**：仅使用 LLM 节点执行全部语义推理，即使是规则明确、可确定性执行的子任务也未被优化。

这限制了系统的可扩展性和经济可行性。

---

### 🚀 提出的新方法与创新思路

作者提出 **HyEvo** —— 一种**自演化的混合型智能体工作流生成框架**，其核心创新在于：

#### （1）异构原子合成（Heterogeneous Atomic Synthesis）
- 不再依赖预定义 operator 库，而是**自主合成功能性的原子单元**：
  - **LLM Nodes**：负责语义理解与复杂推理。
  - **Code Nodes**：执行确定性逻辑（如格式校验、数学运算等），避免不必要的 LLM 推理开销。
- 实现了从“operator orchestration”到“atomic node synthesis”的范式跃迁。

#### （2）LLM 驱动的多岛进化策略（Multi-Island Evolutionary Strategy）
- 引入基于 **MAP-Elites** 的多岛种群机制，维持多样性，防止早熟收敛。
- 设计 **reflect-then-generate** 机制：
  - 反思阶段：分析失败案例与历史最优方案，诊断结构缺陷。
  - 生成阶段：基于反思指导新 workflow 的拓扑与节点逻辑设计。
- 结合 **cascaded sandbox evaluation** 快速筛选无效候选，提升搜索效率。

#### （3）端到端自演化闭环
- 构建了一个完整的“生成 → 执行 → 反馈 → 进化”循环，实现 workflow 的持续自我改进。

---

### 🔍 相比现有方法的优势
| 维度 | HyEvo | 现有方法 |
|------|-------|--------|
| 自动化程度 | 完全自动（无需人工设计 operators） | 依赖手工构建 operator 库 |
| 架构类型 | **Heterogeneous**（LLM + Code Nodes） | **Homogeneous**（全 LLM 节点） |
| 成本与延迟 | 显著降低（最高达 19× 和 16×） | 高昂的 token 消耗与响应时间 |
| 搜索空间探索能力 | 多样性强，支持细粒度优化 | 受限于粗粒度模块组合 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在五个主流 benchmark 上进行评估，涵盖两类任务：

#### 数学推理（Mathematical Reasoning）
- **GSM8K**：小学数学应用题
- **MATH**：高中难度数学问题
- **MultiArith**：多步算术推理

#### 编程生成（Code Generation）
- **HumanEval**：函数级代码生成
- **MBPP**：面向实际场景的 Python 编程任务

---

### ⚙️ 实验设置与评估指标

#### 模型后端（Backbone LLMs）
- `gpt-4o-mini`（主比较）
- 开源模型：`DeepSeek-V3`, `Qwen3-Max`

#### 评估指标
| 类别 | 指标 |
|------|------|
| 性能 | Accuracy（数学）、Pass@1（编程） |
| 效率 | **Cost**（千美元级 token 花费）、**Latency**（端到端执行时间，秒） |
| 综合目标 | 加权奖励函数 $ R(G) = \lambda_1 S_q + \lambda_2 U(C_q) + \lambda_3 U(T_q) $ |

#### 参数配置
- 进化代数：40 轮
- 岛屿数量（K）：2
- 迁移间隔（Δmig）：每 15 轮一次环形迁移
- 温度参数：1.0
- 数据划分：验证集 : 测试集 = 1:4

---

### 🆚 基线方法对比

分为三类：

#### （1）手动设计的 Workflow
- Prompting 方法：Vanilla, CoT, ComplexCoT, SC
- 多智能体系统：MultiPersona, LLM-Debate, DyLAN, AgentVerse, MacNet

#### （2）自动化的 Agentic Workflow 框架
- 学习型：GPTSwarm, MaAS
- 搜索型：AutoAgents, ADAS, AgentSquare, **AFlow**（本文主要对比基线）

> 注：选择 **AFlow** 作为主要 baseline 是因其为当前最先进的开源自动化 workflow 生成器。

---

## 3. 主要实验结果和性能指标

### 📊 性能对比（Table 1）

| 方法 | GSM8K | MATH | MultiArith | HumanEval | MBPP | **Avg.** |
|------|-------|------|------------|-----------|--------|----------|
| MaAS | 92.30 | 51.82 | 98.80 | 92.85 | 82.17 | 83.59 |
| AFlow | 91.16 | 51.28 | 96.22 | 90.93 | 81.67 | 82.25 |
| **HyEvo (Ours)** | **93.36** | **53.91** | **99.67** | **93.89** | **83.28** | **84.82** |

✅ **HyEvo 在所有五个 benchmark 上均取得 SOTA 表现**，平均得分领先第二名 MaAS 达 **1.23%**，较 AFlow 提升 **2.57%**。

---

### 💰 效率分析（Table 2）——以 MATH 和 MBPP 为例

| Dataset | Metric | AFlow | **HyEvo** | 提升倍数 |
|--------|--------|--------|-----------|---------|
| **MATH** | Perf. ↑ | 51.65 | **53.91** | +2.26 pts |
|          | Cost ↓ | 3.78 | **1.85** | **≈2.0× 更便宜** |
|          | Time ↓ | 138.86s | **30.76s** | **≈4.5× 更快** |
| **MBPP** | Perf. ↑ | 81.52 | **83.28** | +1.76 pts |
|          | Cost ↓ | 1.05 | **0.08** | **≈13.1× 更便宜** |
|          | Time ↓ | 23.93s | **2.42s** | **≈9.9× 更快** |

📌 在不同 backbone 下趋势一致，**最大效率增益可达 19×（成本）和 16×（延迟）**。

> 特别是在 MBPP 上，HyEvo 将成本从 \$1.05×10⁻³ 降至 \$0.08×10⁻³，说明其对规则性强的任务极具优势。

---

### 🔪 消融实验（Ablation Study）

在 MATH 数据集上对关键组件进行消融：

#### （1）移除 Reflect-Then-Generate 机制（w/o Reflect）
- 验证准确率快速上升至 76.47% 后停滞。
- 最终仅达 76.47%，而完整 HyEvo 达 **80.67%**（+4.2% 绝对提升）。
- ❗ 缺乏反思机制易陷入局部最优，无法持续进化。

#### （2）移除 MAP-Elites 策略（w/o MAP-Elites）
- 收敛速度慢，最终性能下降 1.68%（78.99% vs 80.67%）。
- 种群多样性不足，导致探索受限。

✅ 结论：**reflect-then-generate 与 MAP-Elites 对高效搜索至关重要**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **混合架构优于纯 LLM 架构**  
   将确定性任务卸载给 code node 显著提升了效率与可靠性，验证了 **hybrid agentic workflow** 的必要性。

2. **原子级合成比 operator 编排更具表达力**  
   自主生成 atomic nodes 支持更精细的控制流与逻辑抽象，突破了传统方法的模块粒度限制。

3. **进化 + 反思 = 高效定向搜索**  
   LLM 的推理能力可用于指导进化过程，形成“智能突变”，显著加速高质量 workflow 的涌现。

4. **Population Diversity 是成功的关键**  
   多岛机制与精英归档（elite archive）有效防止模式坍塌，促进跨路径的知识融合（如 Island 1 与 Island 2 的协同进化）。

---

### ⚠️ 局限性
1. **依赖高质量 LLM 作为 Meta-Agent**  
   若 meta-agent 推理能力弱，则反思与生成质量下降，影响整体进化效率。
   
2. **初始收敛较慢**  
   由于搜索空间极大，前几轮性能增长平缓，需足够迭代次数才能显现优势。

3. **Code Node 的安全性与可解释性未深入讨论**  
   自动生成的代码可能存在潜在漏洞或难以调试。

---

### 🔮 未来工作方向
- 扩展至多模态任务（vision + language + code）
- 引入形式化验证机制保障 code node 正确性
- 动态调整 hybrid ratio（LLM vs Code 节点比例）以适应不同任务类型
- 探索分布式并行进化以进一步加速搜索

---

## ✅ 总结

**HyEvo** 是首个实现**完全自动化、异构化、自演化**的 agentic workflow 生成框架。它通过引入 **LLM + Code 节点混合架构** 与 **LLM 驱动的多岛进化算法**，不仅在性能上超越现有 SOTA 方法，在推理成本和执行延迟方面更是实现了**数量级的优化**（up to 19× cost saving, 16× speedup）。该工作标志着从“人工设计流程”向“机器自我创造智能架构”的重要迈进，为下一代高效 AI Agent 系统提供了新范式。

</details>

---

### 16. [Stepwise: Neuro-Symbolic Proof Search for Automated Systems Verification](https://arxiv.org/abs/2603.19715)

**Authors**: Baoding He, Zenan Li, Wei Sun, Yuan Yao, Taolue Chen, Xiaoxing Ma, Zhendong Su  
**Category**: cs.AI  
**Published**: 2026-03-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19715v1  

#### Abstract
Formal verification via interactive theorem proving is increasingly used to ensure the correctness of critical systems, yet constructing large proof scripts remains highly manual and limits scalability. Advances in large language models (LLMs), especially in mathematical reasoning, make their integr...

---

### 17. [Cooperation and Exploitation in LLM Policy Synthesis for Sequential Social Dilemmas](https://arxiv.org/abs/2603.19453)

**Authors**: V\'ictor Gallego  
**Category**: cs.CL  
**Published**: 2026-03-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19453v1  

#### Abstract
We study LLM policy synthesis: using a large language model to iteratively generate programmatic agent policies for multi-agent environments. Rather than training neural policies via reinforcement learning, our framework prompts an LLM to produce Python policy functions, evaluates them in self-play,...

---

### 18. [Translation from the Information Bottleneck Perspective: an Efficiency Analysis of Spatial Prepositions in Bitexts](https://arxiv.org/abs/2603.19924)

**Authors**: Antoine Taroni, Ludovic Moncla, Frederique Laforest  
**Category**: cs.CL  
**Published**: 2026-03-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19924v1  

#### Abstract
Efficient communication requires balancing informativity and simplicity when encoding meanings. The Information Bottleneck (IB) framework captures this trade-off formally, predicting that natural language systems cluster near an optimal accuracy-complexity frontier. While supported in visual domains...

---

### 19. [Optimizing Resource-Constrained Non-Pharmaceutical Interventions for Multi-Cluster Outbreak Control Using Hierarchical Reinforcement Learning](https://arxiv.org/abs/2603.19397)

**Authors**: Xueqiao Peng, Andrew Perrault  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19397v1  

#### Abstract
Non-pharmaceutical interventions (NPIs), such as diagnostic testing and quarantine, are crucial for controlling infectious disease outbreaks but are often constrained by limited resources, particularly in early outbreak stages. In real-world public health settings, resources must be allocated across...

---

### 20. [Neural Uncertainty Principle: A Unified View of Adversarial Fragility and LLM Hallucination](https://arxiv.org/abs/2603.19562)

**Authors**: Dong-Xiao Zhang, Hu Lou, Jun-Jie Zhang, Jun Zhu, Deyu Meng  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19562v1  

#### Abstract
Adversarial vulnerability in vision and hallucination in large language models are conventionally viewed as separate problems, each addressed with modality-specific patches. This study first reveals that they share a common geometric origin: the input and its loss gradient are conjugate observables ...

---

### 21. [Scalable Learning of Multivariate Distributions via Coresets](https://arxiv.org/abs/2603.19792)

**Authors**: Zeyu Ding, Katja Ickstadt, Nadja Klein, Alexander Munteanu, Simon Omlor  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19792v1  

#### Abstract
Efficient and scalable non-parametric or semi-parametric regression analysis and density estimation are of crucial importance to the fields of statistics and machine learning. However, available methods are limited in their ability to handle large-scale data. We address this issue by developing a no...

---

### 22. [Constraint-aware Path Planning from Natural Language Instructions Using Large Language Models](https://arxiv.org/abs/2603.19257)

**Authors**: Dylan Shim, Minghan Wei  
**Category**: cs.CL  
**Published**: 2026-03-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.19257v1  

#### Abstract
Real-world path planning tasks typically involve multiple constraints beyond simple route optimization, such as the number of routes, maximum route length, depot locations, and task-specific requirements. Traditional approaches rely on dedicated formulations and algorithms for each problem variant, ...

---

### 23. [DAPA: Distribution Aware Piecewise Activation Functions for On-Device Transformer Inference and Training](https://arxiv.org/abs/2603.19338)

**Authors**: Maoyang Xiang, Bo Wang  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.19338v1  

#### Abstract
Non-linear activation functions play a pivotal role in on-device inference and training, as they not only consume substantial hardware resources but also impose a significant impact on system performance and energy efficiency. In this work, we propose Distribution-Aware Piecewise Activation (DAPA), ...

---

### 24. [Two-Time-Scale Learning Dynamics: A Population View of Neural Network Training](https://arxiv.org/abs/2603.19808)

**Authors**: Giacomo Borghi, Hyesung Im, Lorenzo Pareschi  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.19808v1  

#### Abstract
Population-based learning paradigms, including evolutionary strategies, Population-Based Training (PBT), and recent model-merging methods, combine fast within-model optimisation with slower population-level adaptation. Despite their empirical success, a general mathematical description of the result...

---

### 25. [MeanFlow Meets Control: Scaling Sampled-Data Control for Swarms](https://arxiv.org/abs/2603.20189)

**Authors**: Anqi Dong, Yongxin Chen, Karl H. Johansson, Johan Karlsson  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.20189v1  

#### Abstract
Steering large-scale swarms in only a few control updates is challenging because real systems operate in sampled-data form: control inputs are updated intermittently and applied over finite intervals. In this regime, the natural object is not an instantaneous velocity field, but a finite-window cont...

---

### 26. [EvidenceRL: Reinforcing Evidence Consistency for Trustworthy Language Models](https://arxiv.org/abs/2603.19532)

**Authors**: J. Ben Tamo, Yuxing Lu, Benoit L. Marteau, Micky C. Nnamdi, May D. Wang  
**Category**: cs.CL  
**Published**: 2026-03-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.19532v1  

#### Abstract
Large Language Models (LLMs) are fluent but prone to hallucinations, producing answers that appear plausible yet are unsupported by available evidence. This failure is especially problematic in high-stakes domains where decisions must be justified by verifiable information. We introduce \textbf{Evid...

---

### 27. [RiboSphere: Learning Unified and Efficient Representations of RNA Structures](https://arxiv.org/abs/2603.19636)

**Authors**: Zhou Zhang, Hanqun Cao, Cheng Tan, Fang Wu, Pheng Ann Heng, Tianfan Fu  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.19636v1  

#### Abstract
Accurate RNA structure modeling remains difficult because RNA backbones are highly flexible, non-canonical interactions are prevalent, and experimentally determined 3D structures are comparatively scarce. We introduce \emph{RiboSphere}, a framework that learns \emph{discrete} geometric representatio...

---

### 28. [Ontology-Based Knowledge Modeling and Uncertainty-Aware Outdoor Air Quality Assessment Using Weighted Interval Type-2 Fuzzy Logic](https://arxiv.org/abs/2603.19683)

**Authors**: Md Inzmam, Ritesh Chandra, Sadhana Tiwari, Sonali Agarwal, Triloki Pant  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.19683v1  

#### Abstract
Outdoor air pollution is a major concern for the environment and public health, especially in areas where urbanization is taking place rapidly. The Indian Air Quality Index (IND-AQI), developed by the Central Pollution Control Board (CPCB), is a standardized reporting system for air quality based on...

---

### 29. [FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization](https://arxiv.org/abs/2603.19835)

**Authors**: Chiyu Ma, Shuo Yang, Kexin Huang, Jinda Lu, Haoming Meng, Shangshang Wang, Bolin Ding, Soroush Vosoughi, Guoyin Wang, Jingren Zhou  
**Category**: cs.LG  
**Published**: 2026-03-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.19835v1  

#### Abstract
We present Future-KL Influenced Policy Optimization (FIPO), a reinforcement learning algorithm designed to overcome reasoning bottlenecks in large language models. While GRPO style training scales effectively, it typically relies on outcome-based rewards (ORM) that distribute a global advantage unif...

---

### 30. [From Comprehension to Reasoning: A Hierarchical Benchmark for Automated Financial Research Reporting](https://arxiv.org/abs/2603.19254)

**Authors**: Yiyun Zhu, Yidong Jiang, Ziwen Xu, Yinsheng Yao, Dawei Cheng, Jinru Ding, Yejie Zheng, Jie Xu  
**Category**: cs.CL  
**Published**: 2026-03-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.19254v1  

#### Abstract
Large language models (LLMs) are increasingly used to generate financial research reports, shifting from auxiliary analytic tools to primary content producers. Yet recent real-world deployments reveal persistent failures--factual errors, numerical inconsistencies, fabricated references, and shallow ...

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
