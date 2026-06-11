# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-11 09:57:32 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Beyond Fully Random Masking: Attention-Guided Denoising and Optimization for Diffusion Language Models](https://arxiv.org/abs/2606.12273)

**Authors**: Jia Deng, Junyi Li, Wayne Xin Zhao, Jinpeng Wang, Hongyu Lu, Ji-Rong Wen  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2606.12273v1  

#### Abstract
Diffusion large language models (dLLMs) offer an efficient alternative to autoregressive models through parallel decoding, yet existing post-training methods largely rely on random masking strategies that overlook intrinsic token dependencies. In this work, we present an empirical analysis of attent...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Beyond Fully Random Masking: Attention-Guided Denoising and Optimization for Diffusion Language Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **diffusion large language models (dLLMs)** 在 post-training 阶段普遍采用**完全随机掩码（fully random masking）策略**，忽略了语言生成过程中 token 之间的内在依赖关系。这种训练方式与推理时的动态过程不一致，导致模型在复杂推理任务（如数学和代码生成）中表现受限。

此外，尽管 dLLMs 具备双向注意力机制，允许每个 token 动态关注上下文，但传统方法仍强行施加固定的解码顺序（如从左到右或块状 unmasking），未能充分利用其结构优势。

### 🚀 提出的新方法：AGDO 框架
作者提出 **AGDO (Attention-Guided Denoising and Optimization)**，一个统一的后训练框架，通过引入基于注意力机制的信号来指导去噪顺序和优化目标。

#### 主要组成部分：
1. **Attention-Guided Denoising Order (AGDO-SFT)**  
   - 利用最终层的注意力得分计算“有效注意力”（valid attention score $S$），即 token 对已去噪上下文的关注程度。
   - 构建一种新的去噪顺序：优先恢复那些对已有上下文有更强注意力依赖的 token。
   - 该顺序更符合模型内部语义依赖结构，提升生成稳定性。

2. **Attention-Based Loss Reweighting**
   - 定义“影响分数”（influence score $I_k$）衡量某个 token 被其他 token 注意的程度，识别出关键的“注意力枢纽”（attention hubs）。
   - 在 SFT 中对这些关键 token 的损失进行加权，增强其学习强度。

3. **Attention-Guided Reinforcement Learning (AGDO-RL)**
   - 将上述注意力信息融入 GRPO（Group Relative Policy Optimization）算法。
   - 修改优势函数：$A'_k = A_k + \text{sign}(A_k)\cdot\delta\cdot I_k$，使策略更新更倾向于高影响力的 token。
   - 实现偏好优化与模型内在推理结构的一致性。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | AGDO |
|------|--------|-------|
| 掩码策略 | 随机或固定顺序（blockwise / left-to-right） | 基于注意力依赖动态决定去噪顺序 |
| 依赖建模 | 忽视 token 间动态依赖 | 显式利用注意力稀疏性和时间一致性 |
| 优化重点 | 平等对待所有 token | 强调“注意力中心”token |
| 训练-推理一致性 | 存在 mismatch | 更好对齐推理轨迹 |

> ✅ **核心思想**：不应人为规定去噪顺序，而应让模型“自然地”按照其注意力所揭示的语义依赖路径进行恢复。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### 数学推理任务：
- **MATH-500**：500道高中及以上难度数学题，用于主评估。
- **GSM8K**：小学级别数学应用题，测试基础推理能力。
- **Minerva**：多领域科学数学问题集合。

#### 编程任务：
- **LiveBench**：真实编程挑战，强调执行正确性。
- **LiveCodeBench-V2**：污染可控的代码生成评测基准。

#### 泛化任务（验证非推理性能）：
- **HellaSwag**：常识填空任务。
- **CommonsenseQA (CSQA)**：常识问答。

---

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **主模型** | `Dream-v0-Instruct-7B` 和 `LLaDA-8B-Instruct` |
| **训练配置** | 所有实验重复 8 次取平均；静态采样（temperature=0.1），每次去噪一步，最大长度 1024 |
| **SFT 设置** | 学习率 $1\times10^{-6}$，batch size=128，单轮训练 |
| **RL 设置** | 使用 GRPO 框架，rollout 时每步采样 32 prompts，每个 prompt 生成 8 responses；batch size=512 |
| **评估指标** | 准确率（Accuracy），以程序执行通过率或答案匹配为准 |

---

### 🔁 基线方法对比

#### SFT 阶段基线：
- **Standard SFT**：全随机掩码
- **Blockwise SFT**：分块自回归式掩码（Sun et al., 2025）

#### RL 阶段基线：
- **diff-GRPO**（Zhao et al., 2025）
- **Coupled RL**（Gong et al., 2025）
- **TraceRL**（Wang et al., 2025b）

> 所有基线均进行了数据增强（多次随机掩码）以保证前向传播次数可比。

---

## 3. 主要实验结果和性能指标

### 📊 主要性能对比（Table 2 & Table 4）

#### 在 Dream-v0-Instruct-7B 上的结果（平均准确率）：

| 方法 | GSM8K | MATH500 | Minerva | LiveBench | LiveCodeBench | **Average** |
|------|-------|---------|--------|-----------|----------------|-------------|
| Dream-v0-Instruct-7B (原模型) | 69.4 | 38.9 | 11.6 | 10.7 | 10.7 | 28.3 |
| SFT (标准) | 83.5 | 48.3 | 14.8 | 11.3 | 11.5 | 33.9 |
| Blockwise SFT | 86.0 | 51.7 | 12.3 | 10.2 | 11.8 | 34.4 |
| **AGDO-SFT (Ours)** | 85.3 | **53.7** | **15.3** | **12.5** | **13.1** | **36.0** |
| diff-GRPO | 85.0 | 45.5 | 15.3 | 15.2 | 13.9 | 35.0 |
| TraceRL | 86.3 | 52.8 | 16.4 | 14.0 | 13.0 | 36.5 |
| **AGDO-RL (Ours)** | **87.7** | **53.7** | **16.1** | **18.3** | **14.7** | **38.1** |
| **AGDO (完整流程)** | **86.9** | **56.2** | **17.0** | **18.4** | **15.6** | **38.8** |

> ✅ **AGDO 整体优于所有基线，在多个任务上刷新 SOTA**，尤其在编程任务（LiveBench 提升超 3%）和数学综合表现上显著领先。

#### 在 LLaDA-8B-Instruct 上的验证（Table 4）：

| 方法 | GSM8K | MATH500 |
|------|-------|--------|
| LLaDA-1.5 (SOTA) | 83.3 | 42.6 |
| **AGDO (Ours)** | **85.3** | **42.8** |

> ✅ 即使在先进基线上仍有提升，证明方法具有良好的迁移性和通用性。

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）关于 $\gamma$ 和 $\delta$ 的调参分析（Figure 4）
- 在 AGDO-SFT 中，$\gamma=100$ 时达到最佳效果。
- 即使 $\gamma=0$（无 loss weighting），仅靠 attention-guided order 已超越 blockwise SFT，说明**去噪顺序本身至关重要**。
- 在 RL 中，$\delta=10$ 表现最优；过大 ($\delta=20$) 导致梯度剧烈变化，违反 PPO 信任域约束。

#### （2）不同 block size 下的表现（Table 3，L=512）
- 当上下文受限时，blockwise SFT 性能大幅下降（avg 45.8%），而 **AGDO-SFT 反而保持稳定（49.6%）**。
- AGDO-RL 在各种 block size 下均优于所有基线，最高达 **53.7%**（block=64）。

#### （3）注意力来源分析（Tables 5 & 6）
| 设置 | MATH500 Acc (%) |
|------|------------------|
| 第一层注意力 | 48.8 |
| 中间层 | 52.5 |
| **最后一层（本文选择）** | **53.7** |
| 局部注意力头 | 53.0 |
| 全局注意力头 | 52.2 |
| **所有注意力头聚合（本文选择）** | **53.7** |

> ✅ 最终层捕捉高级语义依赖，且局部与全局注意力共同作用效果最好。

#### （4）分离“顺序”与“加权”的贡献（Table 7）
| 方法 | MATH500 Acc (%) |
|------|------------------|
| Blockwise SFT (baseline) | 51.7 |
| 仅使用 attention-guided order ($\gamma=0$) | 52.7 (+1.0%) |
| 仅使用 influence weighting（随机顺序） | 52.4 |
| 随机权重（非 attention-derived） | 51.9 |
| **Order + Weight (完整 AGDO-SFT)** | **53.7** |

> ✅ 两者均有独立增益，结合后效果最强，且增益来自 attention 信号而非随机正则化。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **dLLMs 的注意力具有强稀疏性与时序一致性**  
   - 大多数 token 主要关注自身及邻近 token（对角模式）；
   - 少数关键 token 成为“注意力中心”，被广泛引用（垂直亮柱）；
   - 同一 token 的注意力分布跨步骤高度稳定。

2. **有效注意力（Valid Attention）与生成稳定性正相关**  
   - token 若更多关注已去噪上下文，则其预测概率更稳定（$\Delta P$ 更小）；
   - 引入 $S$ 指导去噪顺序可显著提高生成质量（见 Table 1）。

3. **AGDO 显著提升推理能力而不损害泛化**
   - 在数学与编程任务上持续超越 SOTA；
   - 在 HellaSwag 和 CommonsenseQA 上也全面领先（Table 9），表明未引入领域偏见。

4. **训练效率影响极小**
   - 注意力分析仅需一次额外前传，占总 rollout 时间约 **3%**（Table 8），几乎无开销。

---

### ⚠️ 方法的局限性
- **目前仅适用于 full-attention dLLMs**，未扩展至 block attention 或稀疏注意力架构。
- 依赖最终层注意力，可能忽略中间层的有用信息（尽管实验显示最后一层最优）。
- 当前框架假设注意力能准确反映语义依赖，但在某些噪声场景下可能存在偏差。

---

### 🔮 未来工作方向
1. 将 AGDO 扩展至 **block-based dLLMs**，设计适配局部注意力的引导策略。
2. 探索 **动态选择注意力层或头** 的机制，实现更精细控制。
3. 结合 **remasking** 或 **adaptive decoding** 技术，在推理阶段进一步优化 AGDO 路径。
4. 探索将 AGDO 应用于 **多模态扩散模型** 中的语言分支训练。

---

## ✅ 总结一句话
> **AGDO 首次系统性地将 dLLMs 的注意力结构用于指导去噪顺序与优化目标，实现了训练与推理路径的高度对齐，在数学与编程等复杂推理任务上显著超越现有 post-training 方法，同时具备高效性与通用性。**

</details>

---

### 2. [Energy-Efficient On-Device RAG on a Mobile NPU: System Design and Benchmark on Snapdragon X Elite](https://arxiv.org/abs/2606.11257)

**Authors**: Zhiyuan Cheng, Longying Lai  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.11257v1  

#### Abstract
Retrieval-Augmented Generation (RAG) pipelines are compute-intensive, combining embedding, retrieval, reranking, and large language model (LLM) generation. Running them entirely on-device benefits privacy, latency, and offline use, but the energy cost of CPU inference is a major barrier. We present ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Energy-Efficient On-Device RAG on a Mobile NPU: System Design and Benchmark on Snapdragon X Elite*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Retrieval-Augmented Generation (RAG)** 管道计算密集，通常依赖云端执行，存在**隐私泄露、延迟高、无法离线使用**等问题。虽然可在设备端运行以缓解这些问题，但传统在 **CPU 上进行推理**能耗过高，难以满足移动设备的能效需求。

此外，尽管现代 SoC 集成了专用 **Neural Processing Unit (NPU)**，但此前尚无研究实现并评测一个完整的 RAG 流程在移动 NPU 上的端到端部署。

### 🚀 提出的新方法与创新
本文提出了据作者所知**首个在移动 NPU 上实现全神经阶段端到端 RAG 的系统设计**，其核心创新包括：

- **完整 NPU 加速的 RAG 管道**：将 RAG 中所有神经网络组件——**Embedding 生成（EmbeddingGemma）**、**Cross-encoder 重排序（Jina Reranker）** 和 **LLM 回答生成（Qwen3-4B-Instruct）** ——全部部署在 **Qualcomm Hexagon NPU** 上，通过 **QAIRT/QNN SDK** 运行静态计算图。
- **统一接口支持多后端对比**：构建了一个可切换的推理后端框架，允许在同一代码逻辑下分别在 **NPU、CPU（llama.cpp）、GPU（OpenCL）** 上运行模型，确保公平比较。
- **面向 NPU 的工程优化实践**：揭示并解决了在 NPU 上部署多模型 RAG 所面临的实际挑战，如：
  - 模型加载顺序必须按参数量从大到小（LLM → Embedder → Reranker），否则会因内存分配失败而崩溃；
  - NPU 静态图限制导致上下文长度受限，需调整 chunk 大小（1,000 字符 vs CPU 的 2,500）和保留 chunk 数（Top-7 vs Top-10）；
  - Windows ARM64 生态不成熟带来的依赖编译难题。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **能效** | NPU 在索引和查询阶段分别降低系统能耗 **12.3×** 和 **4.0×**，显著优于 CPU 和 GPU |
| **速度** | 查询延迟降低至 **4.0× 快于 CPU**，预填充（prefilling）速度快达 **18.1×** |
| **实用性** | 实现真正意义上的“绿色边缘智能”（green edge intelligence），为隐私敏感、低功耗场景提供可持续方案 |
| **质量无损** | 尽管配置受限，答案质量经 GPT-4.1 judge 评估与 CPU/GPU 基本持平，在评估噪声范围内 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **索引数据集**：10 家公司的 SEC 10-K 财报文件（AAPL, ABBV 等），共解析出 **9,324 个文本块**（约 204 万 tokens）
- **查询测试集**：
  - `wiki_minirag`：来自 *rag-mini-wikipedia* 的 **120 个通用知识问答对**，用于主性能与能效评测
  - `FinDER`：金融领域 QA 数据集中的 **120 个问题**，限定于上述 10 家公司，用于测试在小众领域的拒绝回答能力（refusal behavior）

### ⚙️ 实验设置
- **硬件平台**：Dell XPS 13 9345 笔记本，搭载：
  - **SoC**: Snapdragon X Elite (Oryon 12核 CPU)
  - **NPU**: Hexagon NPU (45 TOPS INT8)
  - **GPU**: Adreno X1-85
  - **内存**: 64GB LPDDR5x
  - **操作系统**: Windows 11 ARM64
- **软件栈**：
  - NPU 后端：**QAIRT/QNN SDK** 编译的静态图模型
  - CPU/GPU 后端：**llama.cpp + GGUF 量化模型**，GPU 使用 OpenCL 卸载
- **固定项**：重排序器（reranker）始终运行在 NPU 上（因 OpenCL 存在批处理缺陷）

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **性能** | 各阶段耗时（wall-clock time）、总查询延迟（end-to-end latency）、P95 尾延迟、token/s（prefill & decode） |
| **能效** | 系统平均功率（avg power, W）、总能耗（total energy, J/kJ），通过 HWiNFO64 共享内存采集（500ms 间隔） |
| **质量** | 使用 **GPT-4.1 作为 LLM-as-a-judge**，基于 1–10 分制评分：
  - 平均得分
  - 正确率（≥7）
  - 完美匹配率（=10）
  - 失败率（=1）
  - 配对 Wilcoxon 检验分析差异显著性 |

### 🔁 基线方法对比
| 后端 | LLM & Embedder 执行位置 | Reranker 位置 | 特点 |
|------|--------------------------|---------------|------|
| **NPU** | Hexagon NPU (QAIRT/QNN) | Hexagon NPU | 主要目标方案，全加速 |
| **CPU** | CPU (llama.cpp, Q4_K_M) | Hexagon NPU | 主要对比基线 |
| **GPU** | GPU (llama.cpp + OpenCL) | Hexagon NPU | 功能性验证，非有效加速路径 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### （1）索引阶段性能（Indexing Pipeline）
| 指标 | NPU | CPU | 提升倍数 |
|------|-----|-----|---------|
| Embedding 吞吐量 | 3,325 tokens/s | 367 tokens/s | **9.1×** |
| 总索引时间 | 710.7 s | 5,648.2 s | **7.9×** |
| 系统总能耗 | 19,592 J | 241,020 J | **12.3× 更节能** |
| 平均系统功耗 | 27.6 W | 42.7 W | ↓35% |

> 💡 **说明**：Embedding 阶段是瓶颈，NPU 极大缓解该瓶颈；同时由于 CPU 几乎空闲，系统整体功耗下降明显。

#### （2）查询阶段性能（Query Pipeline on `wiki_minirag`）
| 指标 | NPU | CPU | GPU | NPU vs CPU | GPU vs CPU |
|------|-----|-----|-----|------------|-----------|
| LLM Prefill Speed | 786.7 t/s | 43.4 t/s | 25.2 t/s | **18.1×** | 0.58× |
| LLM Decode Speed | 14.2 t/s | 8.2 t/s | 4.7 t/s | **1.74×** | 0.57× |
| Time-to-First-Token (TTFT) | 1.30 s | 24.77 s | 42.24 s | **19.1× 更快** | 0.59× |
| 总查询延迟 | 9.48 s | 37.98 s | 63.61 s | **4.0× 更快** | 0.60× |
| P95 尾延迟 | 17.90 s | 69.56 s | 107.40 s | **3.9× 更快** | 0.65× |
| 系统总能耗（120 queries） | 37.83 kJ | 150.12 kJ | 246.14 kJ | **4.0× 更节能** | 仅 0.61×（即更耗电） |

> ⚠️ **重要发现**：集成 GPU（Adreno X1-85）在此任务中表现差于 CPU，且能耗高达 NPU 的 **6.5×**

#### （3）能耗分解（Per-Component Energy）
| 组件 | NPU 模式（kJ） | CPU 模式（kJ） | 差异原因 |
|------|----------------|----------------|----------|
| CPU Clusters | ~5.4 kJ | ~79 kJ | NPU 卸载后 CPU 接近休眠 |
| GPU | 0.10 kJ | 0.33 kJ → GPU 模式达 16.4 kJ | GPU 实际活跃但效率极低 |
| System Total | 37.83 kJ | 150.12 kJ | 时间主导能耗差异 |

> ✅ **结论**：能耗节省主要来自**运行时间缩短 + 系统其余部分进入低功耗状态**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **NPU 是移动端 RAG 的理想加速器**：
   - 在 Snapdragon X Elite 平台上，NPU 可实现 **端到端全神经阶段 RAG 的高效运行**，相比 CPU 实现 **4.0× 速度提升** 和 **4.0× 能耗降低**。
   - 对于 embedding 密集型索引任务，节能高达 **12.3×**。

2. **性能增益具有超线性节能效应**：
   - 不仅因为运行更快，还因为 **CPU 资源释放、系统平均功耗下降**，使得能耗减少幅度大于速度提升比例。

3. **答案质量未受损**：
   - 尽管 NPU 方案使用更短上下文（chunk size 1,000 vs 2,500），仅保留 Top-7 文档（vs Top-10），但 **GPT-4.1 judge 评分为 9.32（NPU） vs 8.95（CPU） vs 9.03（GPU）**，差异在评估噪声内。
   - **86.7% 的查询三者评分一致**，Wilcoxon 检验显示多数差异不显著。

4. **集成 GPU 并非优选方案**：
   - Adreno X1-85 GPU 在此负载下比 CPU **慢 1.7×**，能耗却是 NPU 的 **6.5×**，表明其不适合此类 LLM 推理任务。
   - 这是一个**硬件天花板问题**，而非软件栈不成熟所致。

5. **适用于绿色边缘 AI 发展方向**：
   - 若每日处理 1,000 次查询，NPU 比 CPU 节省约 **260 Wh/天（94.9 kWh/年）**，对大规模设备部署有显著碳减排意义。

---

### ⚠️ 局限性
| 限制 | 描述 |
|------|------|
| **静态计算图约束** | 当前 QAIRT 不支持动态形状，导致 context length 固定，限制灵活性 |
| **模型加载顺序要求** | 必须先加载最大模型（LLM），否则内存分配失败，反映当前 NPU 内存管理机制僵化 |
| **模型生态有限** | 并非所有主流 embedding/reranker/LLM 模型都有 NPU 编译版本可用 |
| **单平台验证** | 实验仅在 Snapdragon X Elite 上完成，尚未扩展至 Apple Neural Engine、Intel NPU 或 MediaTek APU |
| **单一 judge 评估** | 使用 GPT-4.1 单一裁判可能存在偏见，缺乏人工标注或多 judge 交叉验证 |

---

### 🔮 未来工作方向
1. **支持动态 shape 的 NPU 运行时**：突破固定 context 长度限制，提升 LLM 使用灵活性。
2. **更灵活的内存管理系统**：消除“必须先加载大模型”的工程约束。
3. **扩大 NPU 兼容模型库**：推动更多轻量级 embedding、reranker 和 LLM 支持 NPU 部署。
4. **跨平台迁移研究**：将在 Snapdragon 上的经验迁移到 Apple、Intel、MediaTek 等其他移动 NPU 架构。
5. **多模态 RAG on NPU**：探索图像+文本混合检索增强生成在 NPU 上的可行性。
6. **引入 human-in-the-loop 评估**：加强答案质量评估的可靠性。

---

## ✅ 总结一句话
> 本论文首次实现了在移动 NPU 上端到端运行完整的 RAG 流程，证明了 **Hexagon NPU 能以 4–12× 的能效优势、4× 以上的速度提升，且不牺牲回答质量地执行 on-device RAG**，为绿色、隐私友好的边缘 AI 提供了一条切实可行的技术路径。

</details>

---

### 3. [Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/abs/2606.12232)

**Authors**: Stipe Frkovic, Metod Jazbec, Dan Zhang, Christian A. Naesseth, Ilija Bogunovic, Eric Nalisnick  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.12232v1  

#### Abstract
Masked diffusion language models (dLLMs) have recently emerged as a competitive alternative to autoregressive language models, with the promise of faster inference via parallel token generation. A notable limitation of the masked formulation, however, is that once a token has been unmasked it can no...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Re-evaluating Confidence Remasking in Masked Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文重新审视了 **post-hoc confidence-based remasking** 在 **masked diffusion language models (dLLMs)** 中的有效性。尽管近期如 WINO 等方法声称通过训练后、无需微调的方式引入 remasking 能提升生成质量，但其实际增益在标准解码设置下是否显著仍不明确。

本文系统性地质疑并验证了这类方法的实际价值，指出其宣称的性能提升可能源于不公平的比较基准或特定非标准设置。

### 提出了什么新方法或新思路
本文**并未提出新的 remasking 方法**，而是提出了一个更全面、更严谨的 **evaluation framework** 来重新评估现有 post-hoc remasking 方法（以 WINO 为代表）的真实有效性。

其核心思路是：
- 将 remasking 的收益与强大的 **confidence-based unmasking**（如 Fast-dLLM）进行隔离比较；
- 在多种解码设置下（block length、sampling temperature、unmasking strategy）进行测试；
- 引入 **flip-flop frequency** 等诊断性指标分析 remasking 失败的原因。

### 相比现有方法的优势
- **批判性视角**：揭示了当前文献中对 remasking 效果的高估，强调需与更强基线（Fast-dLLM）对比。
- **系统性评估框架**：首次在不同 block length、greedy/non-greedy decoding、确定性/随机性 unmasking 下统一评估 remasking。
- **深入归因分析**：通过 flip-flop 分析指出，remasking 失效的根本原因在于 dLLM 自身无法在 remask 后提出更好的替代 token，而非 shadow token 近似不准。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GSM8k**：数学应用题数据集
- **MATH-500**：高等数学问题
- **HumanEval**：代码生成任务
- **MBPP**：面向编程的自然语言到代码任务

### 实验设置和评估指标
| 设置项 | 描述 |
|------|------|
| **模型** | LLaDA-8B-Instruct, Dream-v0-Instruct-7B |
| **Block Length (BL)** | 32（标准设置）、128（WINO 原文设置） |
| **Decoding Temperature (T)** | 0（greedy）、0.8、1.5（non-greedy） |
| **Unmasking Strategy** | Fast-dLLM（confidence thresholding）、dUltra（learned Bernoulli policy） |
| **Remasking Method** | WINO（shadow tokens + confidence thresholding） |
| **评估指标** | Accuracy、pass@k（k=1,64）、NFE（Network Function Evaluations）、throughput（tokens/sec）、flip-flop frequency |

### 基线方法对比
- **Fast-dLLM**：confidence-based adaptive unmasking，作为主要强基线
- **High-confidence sampling**：早期方法，用于与 WINO 原文对比
- **Saber**：另一 post-hoc remasking 方法，在附录中补充对比
- **dUltra**：基于 RL 的 stochastic unmasking 策略，用于测试 remasking 在随机环境下的效果

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 在标准设置下（BL=32, T=0）
- **WINO vs Fast-dLLM**：
  - 准确率提升极小：平均仅 **+0.4%~0.5%**
  - 在 **throughput**（考虑 shadow token 开销）上，WINO 反而更低
  - **结论**：remasking 带来的收益被其额外计算开销抵消，实际无优势

#### ✅ 在大 block length 下（BL=128）
- WINO 提升较明显（平均 **+1.3%~1.5%**），但：
  - Fast-dLLM 在 BL=128 下本身性能下降严重
  - WINO 的“提升”实为**修复 Fast-dLLM 在大 block 下的退化**，而非真正提升 dLLM 上限
  - 且 latency 更高

#### ✅ 在非贪婪解码下（T > 0）
- **pass@1 提升**：T=0.8 时平均 **+2.6%**
- **但多样性下降**：
  - pass@64 仅提升 **+0.7%**
  - 表明 remasking 加剧了 confidence-based unmasking 已有的 **diversity collapse** 问题
- T=1.5 时增益进一步缩小（pass@1 +1.6%，pass@64 +0.3%）

#### ✅ 与随机 unmasking 结合（dUltra + WINO）
- 在 **dUltra（Bernoulli unmasking）** 上添加 WINO：
  - 平均准确率提升 **+3.2%**
  - 仅增加约 **2 NFE**
- **结论**：remasking 在**引入更多随机性的采样策略中更有效**，因其提供了更多可修正的错误机会

### 消融实验结果

#### 🔹 Shadow Token 质量分析（Figure 2）
- WINO 的 shadow token 预测与 oracle leave-one-out 预测一致性高达 **>97%**
- 说明 shadow token 是高质量近似，**不是失败原因**

#### 🔹 Flip-flop Frequency（Figure 3）
- **75–90%（LLaDA） 和 85–95%（Dream）** 的 remasked 位置最终恢复原 token
- 说明模型**识别出错误位置的能力尚可，但无法生成更好替代**

#### 🔹 Cascading Dependency 测试（Appendix B.2）
- 扩展 remasking 到邻居或历史 token（WINO-T1/T2/S1/S2）
- 结果：**accuracy 下降，flip-flop 未减少**
- 说明问题不在上下文依赖，而在模型自身预测分布僵化

#### 🔹 其他 WINO 设计选择消融（Appendix B.1）
- 替换为 consistency-based remasking、修改 loop-guard、attention mask 等
- **所有变体均未带来显著提升**
- 说明 WINO 架构本身已接近最优，但受限于 dLLM 能力

#### 🔹 Saber 方法复现（Figure 8）
- 在相同设置下测试另一 post-hoc 方法 **Saber**
- 结果：同样在 BL=32 下**无显著增益**
- 支持结论：**post-hoc remasking 的低效可能是普遍现象**

---

## 4. 关键结论和发现

### 主要发现
1. **Post-hoc remasking（如 WINO）在标准设置下（BL=32, greedy）几乎无收益**，尤其考虑延迟成本后。
2. 其在大 block length 下的“成功”主要是**补偿 Fast-dLLM 的缺陷**，而非提升模型能力上限。
3. 在非贪婪或随机 unmasking 设置下，remasking 可缓解部分错误，但会**加剧 diversity collapse**。
4. **Flip-flop 率极高**，表明 dLLM 在 remask 后仍倾向于重复原 token，暴露了模型缺乏真正的“自我纠正”能力。
5. remasking 的有效性**高度依赖于 unmasking 策略**，在随机策略（如 dUltra）下表现更好。

### 方法的局限性
- 当前 masked dLLMs 的 reverse process 缺乏灵活性，难以在已有 context 下生成不同于历史的 token。
- Post-hoc remasking 依赖模型自身判断，若模型无法提供替代方案，则机制失效。
- Shadow token 虽高效，但在深层模型中仍是近似，存在理论偏差。

### 未来工作方向
- 探索需要**微调或 RL 训练**的 remasking 方法，可能比 post-hoc 更有效。
- 考虑转向 **uniform discrete diffusion** 范式，天然支持 token 修改。
- 设计能**鼓励多样性**的 remasking 机制，避免加剧 confidence collapse。
- 开发更鲁棒的 **evaluation protocol**，涵盖 block size、temperature、unmasking strategy 等维度。

---

> **总结一句话**：  
> 本文揭示了当前 post-hoc confidence remasking 方法（如 WINO）在标准 dLLM 解码设置下的收益被高估，其有效性高度依赖于非标准或次优的基线设置，未来需更全面的评估框架与更根本的建模改进。

</details>

---

### 4. [Accurate and Resource-Efficient Federated Continual Learning](https://arxiv.org/abs/2606.11480)

**Authors**: Jebacyril Arockiaraj, Dhruv Parikh, Jayashree Adivarahan, Rajgopal Kannan, Viktor Prasanna  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.11480v1  

#### Abstract
Federated continual learning (FCL) must learn from distributed task streams under limited resources, such as communication, computation, memory, and label availability. Existing FCL methods often rely on repeated local optimization, replay, and full supervision. Analytic alternatives avoid iterative...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Accurate and Resource-Efficient Federated Continual Learning**

---

## 1. **论文的主要贡献和创新点**

### **解决的问题**
本文针对**联邦持续学习**（Federated Continual Learning, FCL）中的多重资源约束问题，提出了一种高效且准确的解决方案。FCL 面临以下挑战：
- **通信成本高**：传统方法依赖多轮模型更新，上传大量参数。
- **计算开销大**：客户端需进行多次本地迭代训练（backpropagation）。
- **内存与标签稀缺**：难以存储历史数据或获取充足标注。
- **灾难性遗忘与特征漂移**：非独立同分布（non-IID）数据导致模型在不同客户端间表示不一致。

现有方法通常采用梯度优化、回放机制或蒸馏策略，但这些方法在资源受限场景下表现不佳。

---

### **提出的新方法：FedRAN**
作者提出了 **FedRAN**（Federated Random-feature Analytic Network），一个**资源感知的解析式联邦持续学习框架**，其核心思想是：
- **用前向统计量替代反向传播**：冻结预训练骨干网络（backbone），仅通过一次前向推理提取特征。
- **引入随机投影增强表达能力**：使用固定随机投影（random projection）将低维特征映射到高维空间，提升线性可分性。
- **压缩二阶统计信息**：客户端上传其局部随机特征矩阵的**截断SVD摘要**（truncated-SVD summary），而非完整的 Gram 矩阵。
- **两层 QR-SVD 子空间合并**：服务器端对客户端间的（空间）和任务间的（时间）子空间进行高效合并，形成全局低秩近似。
- **闭式分类器求解**：基于保留的子空间直接求解岭回归分类器（ridge classifier）。
- **支持半监督学习**：提出 **FedRAN-SSL** 变体，利用原型相似性为高置信度无标签样本分配伪标签（pseudo-labeling），以应对标签稀缺。

---

### **相比现有方法的优势**
| 维度 | 传统优化型 FCL | 现有解析方法 | **FedRAN（本文）** |
|------|----------------|-------------|--------------------|
| 更新方式 | 迭代训练（iterative training） | 前向统计（forward-only） | ✅ 前向统计 |
| 通信模式 | 多轮模型交换 | 传输完整 Gram 矩阵（$O(M^2)$） | ✅ 传输低秩摘要（$O(Mr)$） |
| 资源效率 | 高计算/通信开销 | 通信仍昂贵 | ✅ 极低通信与计算 |
| 表示稳定性 | 易发生特征漂移（feature drift） | 固定特征 | ✅ 特征完全稳定 |
| 标签利用率 | 主要依赖全监督 | 忽略无标签数据 | ✅ 支持伪标签利用 |

> ✅ FedRAN 在准确性、通信效率、计算速度和标签效率之间实现了更优平衡。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
实验在三个主流视觉基准上进行：
| 数据集 | 类别数 | 任务数 | 特点 |
|--------|-------|--------|------|
| **CIFAR-100** | 100 | 5 / 10 | 标准图像分类，用于类增量学习 |
| **ImageNet-R** | 200 | 10 | Out-of-distribution 渲染鲁棒性测试 |
| **VTAB** | 50 | 5 | 包含自然、医学、遥感等多样化领域 |

---

### **实验设置**
- **客户端数量**：$K=5$
- **非IID划分**：使用 Dirichlet 分布（$\beta \in \{0.1, 0.5, 1.0\}$）模拟类别偏斜程度，$\beta$ 越小越不均衡。
- **骨干网络**：ResNet-18 和 ViT-B/16（均 ImageNet 预训练）
- **随机投影维度 $M$**：
  - ResNet 设置：$M=8192$
  - ViT 设置：$M=2048$
- **保留秩 $r$**：2048（ResNet）、512（ViT）
- **伪标签阈值 $\tau$**：0.5（cosine 相似度）

---

### **评估指标**
#### **准确性指标**
- **最终准确率 $A_T$**：所有任务完成后，在全部已见类上的测试准确率。
- **平均准确率 $A_{avg}$**：各任务结束后的准确率平均值，衡量持续学习能力。

#### **资源效率指标**
- **通信成本**：单个客户端每任务最大上传字节数（MB）
- **运行时间**：完成一个任务所需的总墙钟时间（wall-clock time），含客户端与服务器计算。

---

### **基线方法对比**
#### **基于优化的方法（Optimization-based）**
- **Finetune**：标准微调 + FedAvg
- **FedLwF**, **FedEWC**, **FediCaRL**：适配联邦场景的经典 CL 方法
- **TARGET**：基于蒸馏的免回放示范方法

#### **解析式方法（Analytic Methods）**
- **STSA**：当前最先进的解析式 FCL，估计 Gram 矩阵（first-order 统计）

#### **提示/适配器方法（Prompt/Adapter）**
- **DualPrompt**, **CodaPrompt**, **PiLoRA**：基于预训练模型的轻量级持续学习

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### **总体准确率提升（vs 最强基线 STSA）**
| 数据集 | $A_{avg}$ 提升 | $A_T$ 提升 |
|--------|----------------|-----------|
| CIFAR-100 | **+4.80 pp** | +3.12 pp |
| ImageNet-R | **+4.24 pp** | +1.50 pp |
| VTAB | **+2.68 pp** | +1.67 pp |

> ✅ FedRAN 在所有数据集上显著优于最强基线 STSA。

---

#### **通信效率对比**
| 方法 | CIFAR-100 | ImageNet-R | VTAB |
|------|----------|------------|------|
| **TARGET**（代表优化方法） | 31.90× FedRAN | 30.65× | 121.75× |
| **STSA** | 1.45× | 2.77× | 2.77× |
| **FedRAN**（本文） | **1.00×** | **1.00×** | **1.00×** |

> ✅ FedRAN 比优化方法节省 **30.6–121.8×** 通信量，比 STSA 再降 **1.45–2.77×**。

---

#### **计算效率对比**
| 方法 | 平均加速倍数（vs 梯度方法） |
|------|----------------------------|
| **FedRAN vs 所有梯度方法** | **190.3× 更快** |
| - CIFAR-100 | 96.9× |
| - ImageNet-R | 246.9× |
| - VTAB | 227.2× |

> ✅ 单任务处理仅需几秒，远超需要数百秒的迭代训练方法。

---

#### **伪标签效果（FedRAN-SSL）**
在仅有 **20% 标签可用**时：
| 数据集 | $A_{avg}$ 提升 |
|--------|----------------|
| CIFAR-100 | +3.26 pp |
| ImageNet-R | **+6.61 pp** |
| VTAB | +5.76 pp |

> ✅ 伪标签有效利用无标签数据，在极端标签稀缺下仍能大幅提升性能。

---

### **消融实验结果**

#### **组件消融（CIFAR-100 + ViT）**
| 组件 | $A_{avg}$ |
|------|----------|
| 原始 ViT 特征（baseline） | 87.71% |
| + 随机投影（Random Projection） | 90.21% |
| + ReLU 非线性 | 93.95% |
| + 低秩 SVD 摘要（完整 FedRAN） | **93.96%** |

> ✅ 各模块逐步提升性能，且引入低秩压缩**几乎无精度损失**。

---

#### **投影维度 $M$ 与秩 $r$ 影响**
- 增大 $r$ 或 $M$ 可提升准确率，但收益递减。
- 例如：当 $M=8192$, $r=2048$ 时，通信达 70.3MB；而 $r=512$ 已捕获大部分有用信息，通信降至 ~17.6MB。
> ✅ 中等秩即可获得接近最优性能，实现良好精度-通信权衡。

---

#### **ViT 骨干上的对比**
| 方法 | CIFAR-100 $A_{avg}$ | ImageNet-R $A_{avg}$ |
|------|---------------------|-----------------------|
| PiLoRA | 83.59 | 55.47 |
| STSA | 93.34 | 68.60 |
| **FedRAN** | **93.96** | **69.80** |

> ✅ 即使在强大 ViT 上，FedRAN 依然领先。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **解析式方法可以同时实现高性能与高效率**：FedRAN 证明无需迭代训练也能达到甚至超越传统 FCL 方法的准确率。
2. ✅ **低秩 Gram 摘要比一阶统计更有效**：直接传输主导特征方向优于从类均值估计 Gram 矩阵。
3. ✅ **冻结骨干 + 子空间合并可避免特征漂移**：解决了 non-IID 下客户端表示不一致的根本问题。
4. ✅ **伪标签能有效缓解标签稀缺**：在冻结特征空间中进行原型匹配安全可靠，无需额外训练。
5. ✅ **通信瓶颈可通过结构性压缩突破**：将 $O(M^2)$ 通信降至 $O(Mr)$ 是可行且高效的。

---

### **方法的局限性**
- **依赖预训练骨干**：未探索从零开始训练的场景。
- **理论分析假设理想伪标签**：未建模伪标签噪声的影响。
- **安全性未考虑**：上传的统计量可能泄露隐私，需结合 Secure Aggregation 等机制。
- **仅限类增量设定**：未扩展至域增量或任务增量等其他持续学习范式。

---

### **未来工作方向**
1. **隐私保护版本**：集成差分隐私或安全聚合（Secure Aggregation）于统计上传过程。
2. **扩展至其他任务类型**：如 domain-incremental、task-incremental 场景。
3. **动态秩调整机制**：根据任务复杂度自适应选择 $r$，进一步优化资源使用。
4. **伪标签噪声建模**：理论分析伪标签错误如何影响最终性能边界。
5. **跨模态应用**：推广至文本、语音等非图像领域的联邦持续学习。

---

> 🔗 **代码开源地址**：[https://github.com/JebacyrilArockiaraj/Fed-RAN-SSL](https://github.com/JebacyrilArockiaraj/Fed-RAN-SSL)

</details>

---

### 5. [Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization](https://arxiv.org/abs/2606.12373)

**Authors**: Hao Xiang, Qiaoyu Tang, Le Yu, Yaojie Lu, Xianpei Han, Ben He, Le Sun, Bowen Yu, Peng Wang, Hongyu Lin, Dayiheng Liu  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.12373v1  

#### Abstract
Reinforcement Learning (RL) with verifiable environments has emerged as a powerful approach for enhancing the reasoning capabilities of Large Language Models (LLMs). While prior research demonstrates that scaling environment quantity improves RL performance, existing manual or individual constructio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前基于 **Reinforcement Learning with Verifiable Rewards (RLVR)** 的大模型推理能力提升方法面临以下瓶颈：
- 大多数研究通过手动或独立生成方式构建 **verifiable environments**（如代码、数学题、逻辑谜题），导致环境池的扩展是**线性增长**。
- 线性扩展限制了训练数据的多样性，在固定成本下难以实现有效的**推理泛化**（reasoning generalization）。

### 🚀 提出的新方法：RACES
本文提出 **RACES**（**Recursive Automated Composition for Environment Scaling**），其核心思想是：
> 将可验证环境视为 **LEGO积木块**，当一个环境的输出类型（codomain）匹配另一个环境的输入类型（domain）时，它们可以自动组合成一个新的、仍可验证的复合环境，并支持递归嵌套。

#### 创新机制：
- **递归组合性**（Recursive Composition）：利用函数复合的闭包性质（closure property），将有限的基础环境组合为指数级增长的复合任务空间。
- **程序化组合操作符**（Composition Operators）：定义了四种组合方式，诱导不同的推理模式：
  - `SEQUENTIAL`：链式执行，要求模型预测中间结果
  - `PARALLEL`：并行处理多个独立子任务
  - `SORT`：打乱顺序，让模型恢复正确执行路径
  - `SELECT`：从候选集中选择正确的子集与顺序

### 🔍 相比现有方法的优势
| 方面 | 传统方法（如 SCALER, RESYN） | RACES |
|------|-------------------------------|-------|
| 扩展方式 | 单个环境独立生成 → 线性扩展 | 组合式构造 → **组合爆炸式扩展** |
| 推理深度 | 浅层、单一任务 | 支持多步、状态保持、顺序推断等深层推理 |
| 数据效率 | 需要大量基础环境 | 用少量基础环境即可达到甚至超越大规模独立环境的效果 |
| 泛化能力 | 易过拟合训练分布 | 引导模型学习通用推理策略（如中间表示、自洽校验） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
#### （1）训练环境池（Environment Pool）
- 构建了包含 **300个 verifiable environments** 的初始池，来源包括：
  - 标准算法题数据集（如 LeetCode-style）
  - LLM 自动生成（Claude-Sonnet-4.5 + 双阶段过滤）
  - 人工编写
- 每个环境满足四元组形式：`(Ge, fe, De, Ve)`，确保可采样、可执行、可描述、可验证。

#### （2）评估基准（Unseen Benchmarks）
所有评估均在**未参与训练环境构建的外部测试集**上进行，体现真正的推理泛化能力：
| Benchmark | 类型 |
|---------|------|
| **LiveCodeBench (LCB)** | 编程生成 |
| **AIME 2024/2025** | 数学竞赛题（AMC系列） |
| **Enigmata** | 合成逻辑推理谜题 |
| **IFEval** | 指令遵循精确度 |
| **LongBench-v2** | 长上下文理解与推理 |

> ⚠️ 注意：这些 benchmark **不用于训练环境的设计或筛选**，完全“out-of-distribution”。

### ⚙️ 实验设置
| 项目 | 设置说明 |
|------|----------|
| **模型** | 主实验：`DeepSeek-R1-Distill-Qwen-14B`, `Qwen3-14B`<br>分析实验：`Qwen3-4B-Instruct-2507` |
| **训练框架** | VERL + GRPO 算法，无KL正则，clip ratio=0.28 |
| **硬件** | 32×NVIDIA A100 80GB GPU 集群，vLLM 推理加速 |
| **训练步数** | 300 steps（主实验），每步128 batch size，共12,800训练样本 |
| **序列长度** | 最长支持 32K tokens 上下文窗口 |
| **对比配置** | 
| - `Base`: 无RL微调<br>- `RL_individual`: 在原始300个环境中采样训练<br>- `RL_RACES`: 使用组合环境训练 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Model | LCB | Enigmata | LBench-V2 | IFEval | AIME | **Avg** |
|-------|-----|----------|-----------|--------|------|--------|
| **DeepSeek-R1-Distill-Qwen-14B** (Base) | 47.2 | 32.3 | 32.5 | 70.6 | 58.5 | **48.2** |
| → + `RL_individual` | 46.9 | 34.2 | 33.7 | 69.3 | 59.8 | **48.8** |
| → + `RL_RACES` | **48.8** | **35.4** | **36.0** | **74.6** | **61.7** | **51.3** |
| **↑ Gain** | +1.6 | +1.2 | +3.5 | +4.0 | +3.2 | **+3.1 pts** |

| **Qwen3-14B** (Base) | 55.0 | 47.4 | 32.5 | 74.8 | — | **58.8** |
| → + `RL_individual` | 56.3 | 48.2 | 34.1 | 84.5 | 76.0 | **60.1** |
| → + `RL_RACES` | **57.0** | **49.2** | **35.5** | **85.7** | **77.0** | **61.1** |
| **↑ Gain** | +2.0 | +1.8 | +3.0 | +10.9 | +1.0 | **+1.0 pt vs RL_individual** |

> ✅ 结论：**RACES 在两个主流 backbone 上均显著提升平均得分**，尤其在 IFEval 和 LongBench-v2 上表现突出。

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）环境利用率效率（Table 2）
使用更少的基础环境即可超越大规模独立训练：

| 方法 | Avg Score (Qwen3-4B) |
|------|------------------|
| `RL_individual` (300 envs) | 50.4 |
| `RL_RACES` (**仅50 envs**) | **50.8** ✅ |
| `RL_RACES` (300 envs) | **51.9** |

> 💡 **仅用1/6的基础环境数量，RACES就超过了全量独立训练效果**，证明其极高的环境利用效率。

#### （2）组合规模的影响（Composition Size）
研究 `SEQUENTIAL` 操作符下的不同组合长度（2~6）对性能影响：

| Composition Size | 2 | 3 | 4 | 5 | 6 |
|------------------|----|----|----|----|----|
| **Average Score** | 50.8 | 50.7 | 51.0 | **51.2** | 50.7 |

> 🔁 发现非单调关系：适度增加组合深度（≤5）有助于提升推理能力；但过深（=6）反而因优化困难导致性能下降。

#### （3）训练动态分析（Figure 2）
- `RL_individual`：初期奖励上升快，但很快饱和（overfitting to simple tasks）
- `RL_RACES`：训练奖励增长缓慢，但**下游泛化性能持续提升**，最终反超

> 📌 表明：复合环境虽然难学，但提供了更强的迁移信号，促进通用推理能力形成。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **组合优于枚举**：  
   > 将 verifiable environments 视为可组合模块，能以**亚线性成本实现超线性推理能力增长**。

2. **推理模式被显式诱导**：  
   不同 composition operator 引导出特定推理行为：
   - `SEQUENTIAL` → 中间状态维护
   - `PARALLEL` → 子任务分离
   - `SORT/SELECT` → 顺序/结构推理 + 抗干扰能力

3. **质变发生在推理策略层面**（见 Appendix E）：
   - RACES 模型倾向于使用：
     - 函数抽象（functional abstraction）
     - 自洽验证（self-consistency check）
     - 显式索引追踪（index-value tracking）
     - 轴向重构搜索（axis-switching search）
   - 而 baseline 模型常陷入局部错误传播或无效假设循环。

4. **组合大小是可控的教学变量**：  
   Composition size 可作为 curriculum learning 的调节器，平衡难度与可学习性。

---

### ⚠️ 局限性（Limitations）
1. **操作符设计有限**：目前仅支持 `SEQUENCE`, `PARALLEL`, `SORT`, `SELECT`，尚未涵盖条件分支（if-else）、循环（loop）等复杂控制流。
2. **依赖基础模型能力**：低能力模型在复合任务中稀疏奖励严重，难以有效学习。
3. **高上下文需求**：复合任务产生长推理轨迹，需至少 **32K token 上下文窗口**才能完整训练。
4. **领域覆盖受限**：当前环境集中在算法、数学、逻辑类任务，是否适用于开放域有待验证。

---

### 🔮 未来工作方向
1. 设计更复杂的组合操作符（如 IF, WHILE, TRY-CATCH）
2. 引入 curriculum learning 动态调整 composition size
3. 探索跨模态环境组合（code + vision + text）
4. 开发轻量化版本以适配小模型和短上下文场景
5. 将 RACES 思想应用于 agent planning、tool calling 等高级任务编排

---

## ✅ 总结一句话
> **RACES 通过将 verifiable environments 视为可递归拼接的 LEGO 积木，实现了环境空间的指数级扩展，在更少基础资源下显著提升了 LLM 的推理泛化能力，为高效、可控的 RLVR 训练提供了新范式。**

</details>

---

### 6. [Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching](https://arxiv.org/abs/2606.11583)

**Authors**: Zhuoyi Peng, Hanlin Gu, Lixin Fan, Yi Yang  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.11583v1  

#### Abstract
Text-attributed graphs (TAGs) underlie real-world applications such as citation networks, social media, and e-commerce. Few-shot graph learning on TAGs is hard: with only a handful of labels per class and the rest of the graph unannotated, neither GNNs nor LLMs can learn well on their own. GNNs read...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**Few-shot 图学习**中的文本属性图（Text-attributed Graphs, TAGs）节点分类任务。在仅有少量标注样本（如每类仅3个标签）的情况下，现有方法面临两大挑战：
- **GNN** 依赖拓扑结构，在低度节点（cold nodes）上表现差（邻居信息不足）；
- **LLM** 依赖文本语义，在文本模糊或简短时难以准确判断。

传统方法普遍采用“**Golden Teacher**”范式：固定一个模型作为教师（如 LLM 或 GNN），其输出被视为“真值”，用于指导另一个模型训练。但在稀疏监督下，任一模型都不可靠，这种单向教学会将教师的盲区直接传递给学生，限制性能上限。

### 提出的新方法与新思路
作者提出 **LLM-GNN Co-Teaching**，一种**双向协同教学框架**，彻底摒弃“Golden Teacher”假设：

- **双向伪标签交换**：GNN 和 LLM 在每一轮中分别选出自己最自信的预测（基于 small-loss 准则），并将这些高置信伪标签交换给对方作为训练信号。
- **动态更新机制**：两个模型在每轮都同时更新，而非一方冻结、一方学习。
- **RPL-PO（Round-based Pseudo-Label Preference Optimization）**：利用跨轮次的一致性变化生成偏好对（preference pair）。当某节点从第 $t$ 轮的“跨模型矛盾”变为第 $t+1$ 轮的“跨模型一致”时，LLM 在这两轮对该节点的回答构成一个偏好对（旧错 vs 新对），用于 DPO 训练。

> ✅ **核心思想**：不指定任何模型为权威教师，而是让两者通过多轮互教互学，逐步纠正彼此错误，并从学习轨迹中自挖掘监督信号。

### 相比现有方法的优势
| 维度 | 传统方法（如 GNN-as-Judge） | LLM-GNN Co-Teaching |
|------|----------------------------|---------------------|
| 教学模式 | 单向、固定教师 | 双向、动态互教 |
| 监督来源 | 外部标签或固定模型输出 | 自身演化轨迹中的共识信号 |
| 错误修正能力 | 无法修正教师错误 | 可通过共识实现自我纠正 |
| 适用场景 | 标签较充足时有效 | 尤其适合 Few-shot 极端稀疏场景 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共六个文本属性图基准，涵盖多种领域：
| 数据集 | 类型 | 节点数 | 边数 | 类别数 | 领域 |
|--------|------|--------|------|--------|------|
| **Cora** | 引用网络 | 2.7K | 10.9K | 7 | 学术论文 |
| **Citeseer** | 引用网络 | 3.2K | 4.3K | 6 | 学术论文 |
| **PubMed** | 引用网络 | 19.7K | 88.7K | 3 | 医学文献 |
| **WikiCS** | Wikipedia 超链接 | 11.7K | 431.7K | 10 | 计算机科学文章 |
| **ogbn-arxiv** | arXiv 引用网络 | 169.3K | 1.17M | 40 | 计算机科学子领域 |
| **ogbn-products** | 亚马逊商品共购图 | 54.0K | 72.3K | 47 | 商品分类 |

> 所有数据集均采用标准 Few-shot 设置：每类仅取 k 个标签作为训练集（k=3,5,10），其余未标注。

### 实验设置与评估指标
- **任务**：Few-shot Semi-supervised Node Classification
- **评估指标**：测试集上的分类准确率（Accuracy %）
- **训练方式**：
  - 每轮从无标签节点中采样 batch 进行交叉推理；
  - 使用 LoRA 对 Llama-3-8B-Instruct 微调；
  - GNN 使用 GCN 架构（可替换为 GAT/SAGE）；
  - 总共运行 T=20 轮，每隔两轮执行一次 RPL-PO。

### 基线方法对比
分为三类进行比较：
1. **经典 GNN 模型**：
   - GCN, GAT, GraphSAGE
2. **LLM-as-Predictor 方法**：
   - Zero-shot, Graph-CoT, Neighbor-Augmented Prompting
3. **LLM-GNN 融合方法**：
   - GLEM, TAPE, LLM-GNN, LLaGA, GraphGPT
   - **GNN-as-Judge**（当前最优基线）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（3-shot setting 下的 Accuracy %）

| Method | Cora | Citeseer | PubMed | WikiCS | ogbn-arxiv | ogbn-products |
|--------|------|----------|--------|--------|------------|--------------|
| GNN-as-Judge | 77.89 | 73.59 | 87.12 | 67.50 | 62.21 | 81.02 |
| **LLM-GNN Co-Teaching** | **85.75** | **77.12** | **91.32** | **74.80** | **69.94** | **82.82** |
| **绝对提升 Δ** | **+7.86** | **+3.53** | **+4.20** | **+7.30** | **+7.73** | **+1.80** |

> 🔥 在所有六个数据集上均取得 SOTA 表现，尤其在 Cora 和 ogbn-arxiv 上分别领先 **7.86%** 和 **7.73%**。

### 与基线方法的对比结果
- **显著超越所有基线**：无论是纯 GNN、纯 LLM 提示法，还是现有的 LLM-GNN 融合方法（如 TAPE、GraphGPT、GNN-as-Judge），Co-Teaching 均全面胜出。
- **优势随任务难度增加而扩大**：
  - 在类别多（40类）、结构复杂（ogbn-arxiv）或异质性强（WikiCS）的任务上，传统方法性能急剧下降，而 Co-Teaching 仍保持稳健。
  - 例如，LLaGA 和 GraphGPT 在 ogbn-arxiv 上仅达约 30%，而 Co-Teaching 达到近 70%。

### 消融实验结果（Ablation Study）

| 变体 | Cora | ogbn-arxiv | 说明 |
|------|------|------------|------|
| 完整模型（Full Model） | 85.75 | 69.94 | 基准 |
| 移除双向教学（w/o bidirectional） | 78.66 | 65.50 | 性能大幅回落至接近 GNN-as-Judge |
| 移除 RPL-PO | 83.03 | 69.51 | 显示 RPL-PO 提供额外增益 |
| 固定选择比例（R=0.5） | 83.20 | 67.10 | 动态 annealing 更优 |
| 使用一致性筛选（agreement selection） | 82.52 | 68.47 | small-loss ranking 更有效 |
| 移除邻居信息输入 | 85.08 | 68.76 | 结构上下文仍有帮助 |

> ✅ 实验证明：**双向协同机制**是性能提升的关键，**RPL-PO** 提供了额外优化路径，且**动态置信度选择策略优于静态规则**。

---

## 4. 关键结论和发现

### 主要发现
1. **“Golden Teacher”假设在 Few-shot 场景下失效**：任一模型都无法单独胜任教师角色，强行指定会导致错误传播。
2. **双向 Co-Teaching 实现互补纠错**：
   - GNN 弥补 LLM 的文本歧义问题；
   - LLM 弥补 GNN 的冷启动节点问题；
   - 二者通过 small-loss 机制实现可靠的知识迁移。
3. **RPL-PO 成功挖掘时间维度监督信号**：
   - 利用“从分歧到共识”的转变构建 preference pair；
   - 无需人工标注、奖励模型或外部裁判，即可完成对齐训练。
4. **具备良好的零样本迁移能力**：
   - 在 **Zero-shot Cross-dataset Transfer** 实验中（如 arxiv → Cora），Co-Teaching 明显优于其他方法，表明其提升了 LLM 的通用图推理能力。

### 方法的局限性
1. **计算开销较高**：
   - 多轮迭代导致训练时间增长（约为单次流程的 T 倍）；
   - 在 ogbn-arxiv 上需约 4.7 小时（T=20），虽可通过早停缓解，但仍高于多数基线。
2. **依赖高质量预训练 LLM**：
   - 若使用能力较弱的 LLM（如 Vicuna-7B），性能显著下降（↓4–7%）；
   - 表明框架效果受限于 LLM 的基础语义理解能力。
3. **应用场景受限于文本质量**：
   - 当节点文本本身噪声大、稀疏或缺失时（如分子图、金融交易图），LLM 的作用受限，需重新设计融合机制。

### 未来工作方向
- 探索更高效的调度策略（如 adaptive round termination）以降低计算成本；
- 将 Co-Teaching 范式扩展至其他模态（如视觉-图、音频-图）或多任务场景；
- 引入去偏机制防止 LLM 偏见在迭代中被放大；
- 研究如何在无文本或弱文本图上构建有效的互补表示学习框架。

---

> 📌 **代码开源地址**：[https://github.com/llmgnncoteaching/LLM-GNN-Coteaching](https://github.com/llmgnncoteaching/LLM-GNN-Coteaching)

</details>

---

### 7. [Efficient Time Series Clustering from Multiscale Reservoir Dynamics with Granular-Ball Anchoring Graph Optimization](https://arxiv.org/abs/2606.12077)

**Authors**: Yifan Wang, Lifeng Shen, Shuyin Xia, Yi Wang  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.12077v1  

#### Abstract
Time-series clustering remains challenging due to the inherent trade-off between clustering effectiveness and computational efficiency. Similarity-based methods often suffer from quadratic complexity caused by pairwise distance computations, while deep learning-based approaches typically rely on cos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient Time Series Clustering from Multiscale Reservoir Dynamics with Granular-Ball Anchoring Graph Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
时间序列聚类面临两大核心挑战：
- **计算效率低**：基于相似性的方法（如 DTW）需要成对距离计算，导致 $O(N^2)$ 复杂度，难以扩展到大规模数据。
- **表示能力与训练成本的权衡**：深度学习方法虽能学习复杂动态特征，但依赖迭代训练、大量参数调优，计算开销大。

此外，如何有效融合多尺度 temporal dynamics 并避免 point-wise affinity modeling 的高复杂性仍是一个未解难题。

### 提出的新方法与创新思路
作者提出 **MSRGC-Net**（Multi-Scale Reservoir Granular-ball Consensus Network），一种高效的时间序列聚类框架，结合以下三大核心技术：

#### i) **Training-free Multiscale Reservoir Encoding**
- 利用多个具有不同谱半径（spectral radii）的固定 reservoir（如 ESN），无需反向传播即可从原始时间序列中提取多尺度 temporal 表示。
- 不同 spectral radius 控制短期瞬态动态 vs 长期依赖建模，实现互补表示生成，且无需重复训练。

#### ii) **Granular-Ball-Based Anchoring Graph Construction**
- 引入 granular-ball computing 对 reservoir states 进行区域级抽象，自适应地划分密度一致的局部区域（granular balls）。
- 选取高质量的 granular-ball 中心作为“anchors”，构建样本到 anchor 的 anchoring graph，显著降低内存和计算复杂度。

#### iii) **Consensus-Based Anchoring Graph Optimization**
- 将多视图（multi-view）的 anchoring graphs 融合为统一的共识图（consensus graph），通过优化目标联合建模：
  - 视图内重建误差
  - 跨视图一致性
  - 全局结构正则化
- 使用加权聚合策略自动调整各视图权重，提升鲁棒性和集成效果。

### 相比现有方法的优势
| 维度 | MSRGC-Net | 传统方法 |
|------|-----------|--------|
| **训练方式** | Training-free（无反向传播） | 深度模型需迭代训练 |
| **复杂度** | 近线性 $O(N)$ | 相似性方法为 $O(N^2)$ |
| **可扩展性** | 支持百万级样本 | 多数方法在千级即受限 |
| **表示多样性** | 显式建模多尺度动态 | 单一尺度或隐式集成 |
| **图构建** | Anchor-level graph（$m \ll N$） | Point-wise affinity graph |

---

## 2. 核心实验方法和设置

### 数据集
在 **UCR 和 UEA 归档**中的 10 个基准数据集上进行评估，涵盖多种应用场景：
- **Multivariate（5个）**：
  - `CharacterTrajectories` (CT)
  - `JapaneseVowels` (JV)
  - `BasicMotions` (BM)
  - `Cricket` (Cric)
  - `SelfRegulationSCP1` (SCP1)
- **Univariate（扩展测试）**：
  - `BCCrop`, `EPG-R/S`, `Wafer` 等
- **大规模测试**：Pedestrian 数据集（约 **180万样本**）

### 评估指标
采用三个标准聚类评价指标：
- **NMI**（Normalized Mutual Information）
- **ARI**（Adjusted Rand Index）
- **RI**（Rand Index）

所有实验重复 10 次取均值 ± 标准差。

### 基线方法对比
分为三类共 11 种代表性方法：
1. **Raw data-based**：
   - `k-Shape`, `Fuzzy-kShape`, `TCK`
2. **Representation learning-based**：
   - `Modular-RC`, `GRAIL`, `Time2Feat`, `DEC`, `TimeSURL`, `TFMCC`
3. **Multi-view clustering**：
   - `GB-SMKKM`, `MV-CAGAF`

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Multivariate 数据集平均表现）
| 方法 | NMI ↑ | ARI ↑ | RI ↑ |
|------|-------|-------|-----|
| **MSRGC-Net (Ours)** | **0.678** | **0.615** | **0.857** |
| GRAIL | 0.703 | 0.581 | 0.832 |
| Time2Feat | 0.610 | 0.417 | 0.869 |
| GB-SMKKM | 0.692 | 0.558 | 0.840 |
| TFMCC | 0.625 | 0.468 | 0.766 |

> ✅ 在 **15项指标中取得12项最优**，其余3项排名第二。

#### 典型案例亮点：
- 在 `JapaneseVowels` 上 ARI 达 **0.750**，比次优方法 GRAIL 提升 **16.9%**。
- 在超长序列 `Cricket`（T=1197）上保持稳定高性能（RI=0.983）。
- 在 20 类分类任务 `CharacterTrajectories` 上 NMI 达 **0.789**，远超单视图 rm-ESN（0.476）。

### 与基线方法的对比结果
- 相比 **raw-based 方法**（如 k-Shape）：利用 reservoir 特征提取捕捉跨变量动态，显著优于直接距离匹配。
- 相比 **deep learning 方法**（如 DEC, TFMCC）：在无需任何网络训练的情况下达到甚至超越其性能。
- 相比 **multi-view 方法**（如 GB-SMKKM）：运行时间降低一个数量级以上，同时精度更高。

### 消融实验结果（Ablation Study）
使用 **RI 指标**验证各组件贡献（见 Table 2）：

| 变体 | Multiscale | Granular-ball | Optimization | Avg RI (Multivariate) |
|------|------------|---------------|--------------|------------------------|
| w/o Multiscale | × | √ | √ | 0.844 |
| w/o Granular-ball | √ | × | √ | 0.832 |
| w/o Optimization | √ | √ | × | 0.811 |
| **Full MSRGC-Net** | √ | √ | √ | **0.857** |

> 所有模块均有正向贡献，其中 **multi-scale reservoir 设计最为关键**。

---

## 4. 关键结论和发现

### 主要发现
1. **Training-free + Multiscale Reservoir 是高效的表示学习范式**  
   固定 reservoir 结构可低成本生成多样化 temporal 表示，避免深度模型昂贵训练过程。

2. **Granular-ball anchoring 实现结构感知的轻量图建模**  
   区域级 anchor 替代 point-wise affinity，既保留局部结构又抑制噪声，大幅提升可扩展性。

3. **Consensus graph optimization 有效融合多视图信息**  
   自适应加权机制使模型更关注高质量视图，增强鲁棒性与泛化能力。

4. **近线性复杂度支持百万级规模聚类**  
   在 1.8M 样本的 Pedestrian 数据集中，MSRGC-Net 仍保持 **RI=0.947**，运行时间随样本数近线性增长。

### 方法的局限性
- 当前 reservoir 参数（如 spectral radius 集合）仍需预设，尚未完全自动化。
- 对极短时间序列（T < 10）的效果可能受限于 reservoir 动态演化不足。
- 仅适用于数值型时间序列，未考虑符号或事件序列场景。

### 未来工作方向
- 探索 **adaptive spectral radius selection** 机制，根据数据自动生成最优 reservoir 配置。
- 将框架拓展至 **time series anomaly detection** 或 **classification** 任务。
- 结合 **causal discovery** 或 **interpretable modeling**，挖掘聚类背后的语义解释。
- 应用于 **real-time streaming clustering** 场景，进一步验证在线部署潜力。

---

> 🔚 **总结**：MSRGC-Net 成功实现了 **expressiveness 与 efficiency 的平衡**，为大规模、高维时间序列聚类提供了一种新颖且实用的解决方案，在准确率、速度和可扩展性方面全面领先现有方法。

</details>

---

### 8. [Automated Mediator for Human Negotiation: Pre-Mediation via a Structured LLM Pipeline](https://arxiv.org/abs/2606.11379)

**Authors**: Jamie Bergen, Sarit Kraus  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.11379v1  

#### Abstract
Pre-mediation, the preparatory phase preceding direct human negotiation, plays a critical role in achieving mutually beneficial agreements, yet is often omitted due to cost, time, and limited access to trained mediators. We introduce an automated mediator for human negotiation, implemented as a stru...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Automated Mediator for Human Negotiation: Pre-Mediation via a Structured LLM Pipeline*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **Pre-Mediation（调解前准备）资源不足**：尽管pre-mediation在冲突解决中至关重要（帮助明确利益、建立信任、提升谈判信心），但由于时间、成本和专业调解员（human mediators）稀缺，该阶段常被省略或简化。
- **现有AI系统局限性**：当前基于LLM的调解支持多集中于联合会议（joint session）中的消息转述或建议生成，而对**pre-mediation阶段的自动化支持几乎空白**。
- **单提示（single-prompt）LLM系统的缺陷**：传统“端到端”对话模型难以同时胜任心理推理、情感共情、策略引导和伦理监督等复杂任务，易出现逻辑退化、过度迎合（sycophancy）等问题。

### 🚀 提出的新方法与创新
- **提出首个面向pre-mediation的Structured LLM Pipeline架构**：
  - 将整个准备流程分解为四个专用LLM模块（称为“agents”，但非自主运行）：
    1. **User Prediction Agent**：从对话中预测用户的偏好、情绪、合作倾向等11个维度（基于SVI框架）。
    2. **Pre-Mediation Dialogue Agent**：主对话接口，遵循八阶段协议（rapport building → preference exploration → trade-offs → perspective-taking 等）进行引导。
    3. **Critic Agent**：独立于生成器的审查模块，在每条回复发出前进行approval/rejection判断，确保内容质量。
    4. **Summary Generation Agent**：会话结束后生成结构化报告，供人类调解员后续使用，实现human-in-the-loop oversight。
  - 可选集成 **Voice-to-Text Agent**（Whisper-1）支持语音输入。

- **单方设计（Single-party design）支持并行部署**：
  - 每个模块仅与一方当事人交互，不接触对方偏好，模拟真实调解员的一对一会谈。
  - 支持所有争议方**并行运行**，极大提升可扩展性（scalability）。

### 🔍 相比现有方法的优势
| 维度 | 本工作 | 现有方法 |
|------|--------|----------|
| **任务聚焦** | 明确针对pre-mediation阶段 | 多数关注joint session或训练型negotiation agents |
| **系统架构** | 分解式pipeline，职责分离 | 单一prompt或多轮自洽对话 |
| **监督机制** | 引入dedicated critic agent避免自我强化错误 | 依赖self-critique或无显式监督 |
| **可用性设计** | 输出summary report供human mediator使用 | 多为直接替代人类角色 |
| **可扩展性** | 支持多方并行处理 | 通常需协调双方同步参与 |

---

## 2. 核心实验方法和设置

### 📊 数据集与场景
- **无公开数据集**，采用**受控的人类被试实验（controlled human-subject experiments）**。
- **模拟室友冲突场景**（ecologically valid for university populations）：
  - 三方合住，需协商三项议题：
    1. **Chore schedules**（清洁分工）
    2. **Quiet hours**（安静时段）
    3. **Guest policies**（访客频率）
  - 虚构两位室友具有预设偏好，创造**integrative trade-off机会**（如一人重安静、轻客人；另一人反之）。
  - 所有参与者扮演其中一位真实室友，与AI或人类调解员进行pre-mediation对话。

### ⚙️ 实验设置
- **Study 1**（N=38）：
  - 20人使用AI mediator（原始版本）
  - 18人使用专业human mediator
  - 对话时长：8–10分钟
  - 测量前后变化（pre-/post-conversation surveys）

- **Study 2**（N=22）：
  - 使用改进版AI mediator（优化prompt以减少affirmation）
  - 验证优化效果

- **平台**：聊天界面（类似常见即时通讯应用），支持文本或语音输入（via Whisper-1）

### 📈 评估指标
#### 主要outcome measures（基于Subjective Value Inventory, SVI）：
- Trust in mediator
- Confidence in positive outcome
- Negotiation confidence
- Preparedness to stay true to principles
- Perspective-taking readiness
- Emotional regulation (frustration management)

#### 技术性评估：
- **Preference inference accuracy**：使用RMSE衡量User Prediction Agent对用户偏好的预测误差，并与human mediators比较。
- **Affirmation rate**：通过GPT-4o标注+人工校验，统计对话中“纯粹肯定”内容的比例。
- **Issue importance change**：测量用户对各议题重要性评分的变化（下降表示灵活性增强，上升表示立场固化/entrenchment）。

### 🆚 基线方法对比
- **Human mediator baseline**：由训练有素的专业调解员执行相同protocol。
- **AI baseline**：本文提出的structured pipeline（Study 1原始版 vs Study 2优化版）。

---

## 3. 主要实验结果和性能指标

### ✅ Study 1 结果
| 指标 | AI Mediator | Human Mediator | 是否显著改善（p值） |
|------|------------|----------------|--------------------|
| Trust in mediator | 2.80 → 3.47 | 3.09 → 3.73 | 均显著（p < .05） |
| Confidence in outcome | 3.27 → 4.07 | 3.18 → 3.91 | 均显著（p < .01 / p < .05） |
| Negotiation confidence | 3.53 → 3.87 | 2.91 → 3.82 | 仅human显著（p < .01） |
| Stay true to principles | 3.60 → 4.07 | 4.09 → 4.27 | 仅AI显著（p < .05） |

- **H1 支持**：AI与human均显著提升trust和confidence。
- **H2 支持**：User Prediction Agent在偏好推断上表现更优：
  - **RMSE = 0.61（AI） vs 0.95（human）→ 降低36%**
  - 准确率随对话轮次增加而提升（见Figure 4）

- **负面发现**：
  - **Affirmation rate过高**：AI中affirming content占比达 **36.6%**，是human（18.9%）的近两倍。
  - 导致**issue entrenchment**：AI组用户对议题的重要性评分平均上升（+0.20），而human组下降（-0.36），表明后者更具灵活性。

---

### ✅ Study 2 结果（Prompt优化后）
- **目标**：降低affirmation，增强“productive friction”和现实检验（reality testing）。
- **Prompt修改**：
  1. 明确指令减少validation，挑战模糊回答。
  2. 加强perspective-taking提问（如“What if your roommate sees this completely differently?”）。
  3. 新增reality testing phase，要求考虑成功率与备选方案。
  4. Critic Agent新增**WARNING**层级，加强过滤。

| 指标 | Pre | Post | p-value |
|------|-----|------|--------|
| Negotiation confidence | 3.68 | 4.41 | **p < .01** |
| Confidence in outcome | 3.68 | 4.09 | p = .07（边际显著） |
| Trust in mediator | 3.00 | 3.41 | p = .21（不显著） |

- **H3（维持有效性）**：部分支持 —— negotiation confidence显著提升，其他指标呈积极趋势。
- **H4（降低affirmation）**：**强烈支持**：
  - Affirmation rate从 **36.6%（Study 1）降至16.8%（Study 2）**
  - **低于human baseline（18.9%）**
  - Critic Agent拒绝率从2.6%升至22.2%，说明新规则有效拦截过度肯定内容。

---

## 4. 关键结论和发现

### 🧠 主要发现
1. **Structured LLM Pipeline可在short-term preparation outcomes上达到与human mediator相当的效果**：
   - 在trust、confidence等主观感知指标上无明显差距。
   - **User Prediction Agent甚至在偏好推断准确率上超越人类**（RMSE ↓36%）。

2. **过度affirmation是一个关键风险点**：
   - 初始AI系统因频繁肯定导致用户立场固化（entrenchment），不利于创造性解决方案。
   - 这反映了LLM常见的**sycophancy问题**：倾向于取悦用户而非推动成长。

3. **Prompt engineering + dedicated critic可有效缓解sycophancy**：
   - 单靠message-level critic不足以控制累积效应。
   - **必须结合source-level干预**（即在dialogue agent prompt中明确要求“challenge”和“reality test”）才能根本改变行为模式。

4. **模块化解耦带来灵活性与可维护性**：
   - Study 2仅调整dialogue和critic prompts，不影响prediction和summary模块，验证了architecture的robustness。

---

### ⚠️ 局限性（Limitations）
- **样本量较小**：Study 1 (N=38)，Study 2 (N=22)，统计功效有限。
- **场景限制**：室友冲突虽具生态效度，但无法推广至高 stakes 场景（如家庭、职场纠纷）。
- **缺乏行为层面验证**：仅测量即时态度变化，未追踪实际谈判过程中的**behavioral outcomes**或agreement quality。
- **无跨研究因果推断**：Study 1与Study 2之间无直接对照，无法严格归因于prompt修改。
- **未评估summary report的实际效用**：尚未让human mediators使用AI生成报告并评估其价值。

---

### 🔮 未来工作方向（Future Work）
1. **并行部署实验**：将系统同时用于所有争议方，研究聚合后的summary如何辅助human mediator主持joint session。
2. **引入human evaluator**：邀请专业mediators评估AI生成的summary report的质量与实用性。
3. **拓展至更高风险领域**：应用于workplace mediation、family disputes等更复杂的场景。
4. **长期效果追踪**：跟踪参与者完成真实谈判后的结果，验证preparation是否转化为更好的agreements。
5. **动态trade-off识别**：探索如何在保护隐私的前提下，利用多方预测结果识别潜在integrative opportunities。

---

## 总结

> 本文提出了一个**模块化、可扩展、以人为本的AI pre-mediation pipeline**，首次系统性地将LLM应用于调解前准备阶段。实验证明其在短期准备效果上**媲美人类调解员**，且在偏好推断精度上更优。更重要的是，研究揭示了**AI在人际互动中易陷入过度肯定的陷阱**，并通过**prompt重构 + 分离式critic机制**成功纠正。这一工作不仅填补了自动化调解的技术空白，也为构建负责任、可监督的interpersonal AI系统提供了重要设计范式。

</details>

---

### 9. [SVoT: State-aware Visualization-of-Thought for Spatial Reasoning via Reinforcement Learning](https://arxiv.org/abs/2606.11770)

**Authors**: Chao Lei, Yanbei Jiang, Markus Hiller, Zhijian Zhou, Xunye Tian, Krista A. Ehinger, Nir Lipovetzky  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.11770v1  

#### Abstract
Spatial reasoning remains a challenge for Multimodal Large Language Models (MLLMs), as it requires reliable multi-hop inference over both intermediate states and state transitions. Current studies often leave intermediate states unverified and treat state transitions as implicit processes, which lim...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SVoT: State-aware Visualization-of-Thought for Spatial Reasoning via Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
多模态大语言模型（MLLMs）在**空间推理**任务中面临挑战，尤其是在需要多步状态追踪（multi-hop spatial reasoning）的动态环境中。现有方法如 VoT 和 MVoT 存在两个关键缺陷：
- **中间状态未被验证**：模型生成的状态无法被系统性地检查，导致错误累积。
- **状态转移过程隐式化**：动作前条件与后效（preconditions and effects）未被显式建模，削弱了对复杂交互的推理能力。

此外，现有基准测试通常将状态转移简化为单变量更新，忽略了真实世界中多对象、数值和因果关系的复杂性。

### 提出了什么新方法或新思路
作者提出 **SVoT (State-aware Visualization-of-Thought)**，一个基于强化学习（RL）的框架，其核心创新包括：

- **Verifiable Intermediate States**：将中间状态形式化为 `(action, state_description)` 元组，并通过确定性转移函数 `f(s_{t-1}, a_t)` 生成真值，支持定量验证。
- **Transition Reasoning Chains (CoT)**：引入显式的推理链，在每一步中验证动作的前提条件与效果，实现文本与视觉的交错推理（interleaved textual and visual reasoning）。
- **Reinforcement Learning with Fine-grained Rewards**：采用 **Group Relative Policy Optimization (GRPO)** 进行训练，设计细粒度奖励机制：
  - `r_z`：中间状态正确性
  - `r_v`：可视化保真度
  - `r_c`：推理链忠实度（Process Reward Model, PRM）
- **五个新的空间推理域**：扩展经典环境并引入两个新域：
  - **PACMAN**：收集硬币 + 数值计数
  - **GATHER**：多步移动（如“Go right 2”）+ 多色球收集，要求长程状态追踪与数值推理

### 相比现有方法的优势
- **可靠性提升**：通过显式状态表示与推理链，使模型决策过程可解释、可验证。
- **更强泛化能力**：在 **Out-of-Distribution (OOD)** 测试集上表现显著优于基线。
- **多模态一致性增强**：联合优化文本描述与图像生成，确保语义一致。
- **支持细粒度监督**：PRM 式奖励引导模型学习正确的推理路径，而非仅匹配最终输出。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
构建了五个基于网格的世界（grid-based domains），均通过 PDDLGym 自动化生成：

| Domain        | 特点 |
|---------------|------|
| **MAZE**      | 包含边界与墙碰撞检测，位置不变即为无效动作 |
| **FROZENLAKE** | 冰面行走，需区分冰、洞、目标格子 |
| **SOKOBAN**   | 推箱子谜题，玩家推动前方方块，涉及多对象交互 |
| **PACMAN**    | 收集金币，需跟踪累计数量与坐标 |
| **GATHER**    | 新增域，支持多步动作（如“Go down 4”）与按颜色分类的球体收集 |

每个 domain 在不同网格大小（4×4 到 7×7）下构造，训练/验证/测试集无重叠。

### 实验设置和评估指标
- **输入格式**：初始状态图 `v₀`、领域描述 `d`、初始状态描述 `s₀`、动作序列 `A`
- **输出格式**：
  - **Free Response (F)**：自由文本描述最终状态
  - **Classification (C)**：从四个选项中选择正确答案
- **评估方式**：预测必须完全匹配真值才算正确（exact match）
- **训练流程**：
  1. **Supervised Fine-Tuning (SFT)**：5 轮，学习生成带推理链的状态与图像
  2. **GRPO 训练**：5 轮，使用分组相对策略优化，采样 G=4 条候选路径
- **硬件**：4×NVIDIA A100 GPU，总耗时约 10 天

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **GPT-4o (T-CoT)** | 强大的非微调 MLLM，使用文本 Chain-of-Thought 推理 |
| **Anole (T-CoT)** | 同款 backbone 模型，指令微调用于逐步推理 |
| **MVoT** | 当前 SOTA 方法，生成图文交错的推理轨迹，但缺乏显式状态验证 |
| **SVoT_o (SVoT+ORM)** | SVoT 变体，仅用结果奖励（Outcome Reward Model） |
| **SVoTp (SVoT+PRM)** | 完整版 SVoT，使用过程奖励（Process Reward Model） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **Table 1** 中报告了 ID 和 OOD 设置下的目标覆盖率准确率（Goal Coverage Accuracy %）：

| 方法 | 最高绝对增益（vs MVoT） | OOD Free-Response 表现 |
|------|------------------------|-------------------------|
| **SVoTp** | **+65%**（SOKOBAN, size 4） | 显著领先所有基线 |
| **SVoTo** | ~+50% | 次优，但仍大幅超越 MVoT |
| **MVoT** | —— | 在 OOD 下严重退化 |
| **GPT-4o** | —— | 在 PACMAN/GATHER 上因数值推理失败而表现差 |

例如，在 **SOKOBAN (size 4, OOD, free-response)** 上：
- MVoT: 3.3%
- SVoTo: 16.7%
- **SVoTp: 46.7%**

### 与基线方法的对比结果
- **SVoTp > SVoTo > MVoT > T-CoT**：完整 SVoT 在几乎所有 domain-size 组合中达到 SOTA。
- **SVoTp ≈ 或 > GPT-4o**：尽管 GPT-4o 是更大更强的模型，SVoTp 仍能在多数任务上匹敌甚至超越它，证明了方法的有效性。
- **Free-response 难于 Classification**：分类任务可通过猜测获得一定分数，而自由响应更能反映真实推理能力。

### 消融实验结果（Ablation Study）
见 **Table 4**：

| 消融变体 | 性能下降情况 | 结论 |
|----------|--------------|------|
| **w/o-V (无可视化)** | 明显下降 | 图像提供关键视觉线索，促进状态追踪 |
| **w/o-RL (仅 SFT)** | 大幅下降 | GRPO 对复杂约束下的推理至关重要 |
| **w/o-RL-C (无推理链 + 无 RL)** | 进一步轻微下降 | 即便有状态和图像，缺少 RL 也无法有效利用推理链 |

> 💡 **关键发现**：在 MAZE 和 SOKOBAN 中，移除 RL 导致性能崩溃，说明这些 domain 对精确动作判断高度敏感，必须依赖 RL 的探索与反馈机制。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **显式状态建模 + 显式推理链** 是可靠多跳空间推理的关键。
2. ✅ **细粒度过程奖励（PRM）优于结果奖励（ORM）**：指导模型“如何思考”比只看“是否答对”更有效。
3. ✅ **图文交错生成 + 联合优化** 提升了多模态一致性与视觉保真度（见 Table 3 & Figure 5）。
4. ✅ **SVoT 在 OOD 场景下鲁棒性强**：尤其在动作更长、对象更多的情况下仍保持较高准确率。
5. ✅ **GATHER 是极具挑战的新 benchmark**：暴露了当前方法在长程路径依赖与数值推理上的不足。

### 方法的局限性
- **推理成本高**：相比仅输出结果的方法，SVoT 需自回归生成多个中间状态、图像和推理链，带来额外计算开销。
- **依赖高质量奖励设计**：视觉奖励 `r_v` 和推理奖励 `r_c` 的有效性依赖人工定义的相似性度量与关键词提取。
- **GATHER 域仍有较大改进空间**：即使 SVoTp 在该域也仅取得较低准确率（~10–30%），表明长期状态追踪仍是开放难题。

### 未来工作方向
- **效率优化**：引入选择性可视化、自适应推理长度、中间状态缓存等机制降低延迟。
- **更通用的奖励机制**：探索无需人工标注的自监督或对比学习方式来构建奖励信号。
- **扩展至连续空间**：将 SVoT 应用于机器人导航、自动驾驶等真实场景中的连续状态空间。
- **结合 symbolic planning**：融合符号规划器（symbolic planner）作为外部工具，辅助验证推理链的逻辑正确性。

---

> 🔚 **总结一句话**：  
> **SVoT 通过“状态感知 + 可视化思维 + 强化学习”三位一体的设计，首次实现了可验证、可解释、高性能的多跳空间推理框架，在多个新构建的挑战性 benchmark 上取得了突破性进展。**

</details>

---

### 10. [A Lightweight Multi-Agent Framework for Automated Concrete Barrier Design](https://arxiv.org/abs/2606.12040)

**Authors**: Wanting Wang, Xiye Ma, Yuyang He, Minghui Cheng, Ran Cao  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.12040v1  

#### Abstract
The design of reinforced concrete highway barriers is a safety-critical process that requires strict compliance with regulatory provisions such as the AASHTO-LRFD bridge design guidelines. Current engineering practice relies heavily on manual, iterative, and heuristic calculations to satisfy complex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Lightweight Multi-Agent Framework for Automated Concrete Barrier Design

## 1. 论文的主要贡献和创新点

### 解决的问题
- **传统混凝土护栏设计依赖人工迭代**：当前工程实践中，设计人员需手动进行反复试错计算以满足 AASHTO-LRFD 规范中关于抗横向冲击力（transverse resistance）的要求，过程繁琐且高度依赖经验。
- **大模型直接应用存在风险**：尽管 Large Language Models (LLMs) 在生成任务上表现出色，但其在结构工程中的直接应用受限于**幻觉（hallucination）** 和缺乏物理一致性（physical grounding），难以保证安全关键型设计的可靠性。

### 提出的新方法与创新思路
- **提出“生成-评估-优化”闭环多智能体框架（Multi-Agent Framework, MAF）**：
  - 利用 **AutoGen** 构建一个由多个专业化 LLM Agent 协同工作的系统，实现任务分工与闭环反馈。
  - 框架包含五大模块：
    1. **Designer Agent**：解析自然语言输入并生成初始参数（JSON格式）。
    2. **Message Parser & Validator**：通过正则提取、文本清洗、缺省填充和硬约束拦截确保参数合法性。
    3. **Calculator & Evaluator**：基于 **Yield-Line Method** 计算实际抗力 $R_w$ 并与目标 $F_t$ 对比，判断状态为 `UNSAFE` / `WASTEFUL` / `OPTIMAL`。
    4. **Optimizer Agent**：接收错误上下文（error context），执行基于力学规则的参数调整（如减小截面或钢筋用量）。
    5. **Drawing Tool**：输出可执行的 AutoLISP 脚本，在 AutoCAD 中自动生成标准图纸。

- **轻量化模型高效替代巨型模型的设计范式**：
  - 发现 **8B 参数的小型模型在 MAF 框架下表现优于 671B 的旗舰模型**，挑战了“越大越好”的主流认知。
  - 强调**架构设计 > 模型规模**，为降低 AI 工程工具部署成本提供新路径。

### 相比现有方法的优势
| 维度 | 传统方法 | 单独 LLM | 本文 MAF |
|------|--------|---------|----------|
| 设计精度 | 高（依赖专家） | 极低（随机性强） | **>98%** |
| 自动化程度 | 低（手工迭代） | 中（一次生成） | 高（闭环优化） |
| 安全性保障 | 人为控制 | 不可控 | 内置物理校验 + 显式约束 |
| 成本效益 | 时间成本高 | 推理资源消耗大 | 可用小型模型，节省算力 |

---

## 2. 核心实验方法和设置

### 数据集与设计场景
- **设计标准依据**：AASHTO-LRFD (2024) 与 MASH (2016)，选取三种防护等级：
  - **TL-3**, **TL-4**, **TL-5**
- **几何类型**：单坡形（single-slope）护栏
- **每种测试等级生成 20 个案例**，共 **60 个设计案例**
- **变量范围**（按等级设定高度区间）：
  - TL-3: 685.8–780.8 mm
  - TL-4: 812.8–907.8 mm
  - TL-5: 1066.8–1161.8 mm

### 实验设置
- **基础 LLM 模型选择**（均来自 DeepSeek 系列）：
  - **DeepSeek-8B**
  - **DeepSeek-32B**
  - **DeepSeek-671B**
- 所有模型用于构建两种模式：
  - **Standalone LLM**：直接生成设计参数，无验证/优化循环
  - **MAF + LLM**：将各 LLM 作为 Agent 的底层引擎，嵌入多智能体协作流程

### 评估指标
- **Precision（精度）**：
  $$
  \text{Precision} = \frac{n_{\text{success}}}{n}, \quad \text{其中 } R_w \in [1.4F_t, 1.6F_t]
  $$
  表示设计落在经济与安全平衡区间的成功率。
  
- **Mean Squared Error (MSE)**：
  $$
  e_i = \max(0, 1.4F_t - R_w, R_w - 1.6F_t), \quad \text{MSE} = \frac{1}{n}\sum e_i^2
  $$
  衡量偏离理想区间的平均误差平方。

- **目标区间设定依据**：参考 NCHRP RR 1109 中 9 个真实设计案例，统计得平均 $R_w/F_t = 1.49$，故取 **[1.4, 1.6]** 为“OPTIMAL”区间。

### 基线方法对比
- **Baseline**：
  - DeepSeek-8B / 32B / 671B 单独推理（zero-shot）
- **Proposed Method**：
  - MAF-DS-8B
  - MAF-DS-32B
- 所有实验重复运行多次以消除随机性影响。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 多智能体框架显著提升精度
| 方法 | TL-3 | TL-4 | TL-5 | **Avg Precision** |
|------|------|------|------|------------------|
| DeepSeek-8B | 0% | 5% | 15% | **6.7%** |
| DeepSeek-32B | 10% | 0% | 25% | **11.7%** |
| DeepSeek-671B | 5% | 5% | 15% | **8.3%** |
| **MAF-DS-8B** | **100%** | **100%** | **98.3%** | **98.3%** |
| **MAF-DS-32B** | 80% | 90% | 95% | **88.3%** |

> 🔍 **结论**：引入 MAF 后，即使是最小的 8B 模型也能达到接近完美的设计成功率。

#### ✅ MSE 显著下降，接近零误差
| 方法 | TL-3 ($10^4$) | TL-4 ($10^4$) | TL-5 ($10^4$) | **Avg MSE ($10^4$)** |
|------|---------------|---------------|---------------|--------------------|
| DeepSeek-8B | 355.44 | 216.63 | 423.83 | **331.97** |
| DeepSeek-32B | 25.31 | 30.89 | 9.35 | **21.85** |
| DeepSeek-671B | 2.91 | 3.01 | 8.65 | **4.86** |
| MAF-DS-32B | 0.19 | 0.14 | 0.12 | **0.15** |
| **MAF-DS-8B** | **0** | **0** | **0** | **0** |

> 📉 **说明**：MAF 极大抑制了“维度幻觉”（dimensional hallucinations），使输出几乎完全收敛于目标区间。

### 消融实验分析（隐含在结果中）
虽然未明确列出消融实验表格，但从以下现象可推断关键组件作用：
- **无闭环机制 → 输出极不稳定**（如 DS-8B 出现从 200kN 到 4000kN 的剧烈波动）
- **加入 Validator → 消除非法参数格式问题**
- **引入 Optimizer Agent → 实现定向修正**（如减少 $n$, $d_z$, $B_{top}$ 等）
- **多轮迭代 → 收敛至 OPTIMAL 区间**

> 💡 示例见 Figure 2：初始设计被判定为 WASTEFUL（$R_w = 485.56 > 1.6F_t$），经优化后降至 $360.05$，成功进入目标区间。

---

## 4. 关键结论和发现

### 主要发现
1. **多智能体协同远胜单一 LLM**：
   - 单独使用任何规模的 LLM 进行端到端设计都不可靠，**最高精度仅 11.7%**。
   - 引入 **generation-evaluation-optimization 闭环** 后，精度跃升至 **98.3%**，证明结构化协作是解决工程幻觉的关键。

2. **模型规模不再是决定性因素**：
   - 在 MAF 框架内，**8B 模型略优于 32B 模型**（98.3% vs 88.3%），颠覆“更大即更强”的假设。
   - 表明：**通过合理的系统架构设计，可以解耦性能对模型参数量的依赖**。

3. **轻量化 + 领域专用 = 更高性价比**：
   - 小模型在专用框架中能实现更优性能，大幅降低推理成本，有利于推动 AI 工具在中小型工程公司普及。

4. **成功实现从自然语言到 CAD 图纸的自动化链路**：
   - 用户输入一句话需求（如“设计一个高 857.8mm 的 TL-4 护栏”），系统可自动输出符合规范的参数与 AutoLISP 脚本，支持一键绘图。

### 方法的局限性
- **适用范围有限**：目前仅针对 **单坡形混凝土护栏**，尚未扩展至其他复杂形式（如双坡、F-shape、复合材料等）。
- **依赖外部计算器**：力学计算仍由确定性程序完成，未实现 LLM 内部物理推理。
- **未处理动态碰撞模拟**：仅基于静态 Yield-Line Theory，未集成 LS-DYNA 或 ANSYS 等 FEA 动态仿真验证。
- **Agent 编排依赖人工设计规则**：Optimizer 的调整策略为预设启发式规则，非自主学习所得。

### 未来工作方向
- **构建端到端工程流水线**：
  - 将当前输出接入下游 **FEA 自动建模 Agent**，实现从设计 → 仿真 → 验证的全自动闭环。
- **拓展至更多结构构件**：
  - 应用于桥墩、梁板、挡土墙等常见结构元素的设计自动化。
- **引入强化学习优化策略**：
  - 让 Optimizer Agent 学习最优调参路径，而非依赖固定规则。
- **开源生态建设**：
  - 当前代码已发布于 GitHub：[https://github.com/MXY820/barrier-design](https://github.com/MXY820/barrier-design)，后续计划支持插件化扩展与行业协作。

---

> ✅ **总结一句话**：  
> 本研究提出了一种基于 AutoGen 的轻量级多智能体框架，首次实现了高精度、可解释、低成本的混凝土护栏自动化设计，揭示了“**架构优于规模**”的新范式，为 AI 赋能安全关键型土木工程提供了可复制的技术蓝图。

</details>

---

### 11. [Teaching Diffusion to Speculate Left-to-Right](https://arxiv.org/abs/2606.11552)

**Authors**: Lexington Whalen, Yuki Ito, Ryo Sakamoto  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.11552v1  

#### Abstract
Large language models (LLMs) achieve remarkable performance across a wide range of tasks, but their autoregressive decoding process incurs substantial inference costs due to inherently sequential token generation. Speculative decoding addresses this bottleneck by employing a lightweight draft model ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Teaching Diffusion to Speculate Left-to-Right*

---

## 1. 主要贡献和创新点

### 解决的问题
该论文针对**基于 diffusion 的 speculative decoding** 中存在的**训练-验证目标不一致**（training-verification mismatch）问题。

- **背景**：在 speculative decoding 中，一个轻量级的 *drafter* 模型生成多个候选 token，由大型 *target* 模型并行验证。Block-diffusion drafters（如 DFlash）能并行生成整个 token block，显著提升吞吐。
- **问题**：diffusion 模型在训练时使用 **bidirectional attention**，每个位置对称地依赖上下文；但在推理时，target 模型以 **strictly left-to-right** 方式验证，一旦某个位置被拒绝，其后所有 token 都被丢弃。
- **后果**：这种不对称导致大量本应正确的 draft token 因前序错误而被浪费（wasted），标准 cross-entropy 损失无法有效优化这一目标。

### 提出的新方法
作者提出了三种**正交的、可叠加的训练时干预策略**，使 drafter 的训练目标更贴近实际的 left-to-right 验证机制：

1. **Position-wise Loss Decay（损失衰减）**  
   - 对不同位置的 token 施加指数衰减的权重：$ w_k(\gamma) = \exp(-\gamma \cdot k) $
   - 早期位置获得更高梯度权重，因其决定整个 block 的接受长度。

2. **First-Error Focal Loss（首错焦点损失）**  
   - 引入辅助损失项，仅作用于 block 中**第一个预测错误的位置**（即 chain breaker）。
   - 该位置决定了 block 的截断点，因此是优化接受长度的关键。
   - 具有自适应性：随着模型改进，错误位置后移，焦点自动转移。

3. **Chain Reward（链式奖励）**  
   - 引入一个可微的代理目标，近似 **expected accepted length**：
     $$
     R_{\text{chain}} = \frac{1}{K-1} \sum_{k=1}^{K-1} \exp\left(\sum_{j=1}^k \log p_j\right)
     $$
   - 梯度按前缀联合概率加权，直接优化长序列的一致性。

### 相比现有方法的优势
- **无需额外推理开销**：不增加 forward pass，不改变 inference pipeline，保持 rejection-sampling 的 **exactness**。
- **正交且可组合**：三个方法作用于不同维度（位置、块条件首错、联合前缀），可叠加增益。
- **兼容性强**：可与 test-time 方法（如 ddTree、SpecDiff-2）结合，实现进一步加速。
- **普适性好**：在多种 target model（Llama-3、Qwen）和任务上均有效。

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据**：
  - 主实验使用 **ShareGPT**：多轮用户-助手对话，转换为 Llama-3 格式。
  - 补充分析使用 **Nemotron-V2 + CodeAlpaca** 的 target-aligned 版本（由 target model 重新生成 response）。
- **评估基准**（共6个）：
  - 推理：**GSM8K**, **AIME**
  - 编码：**HumanEval**, **MBPP**, **LiveCodeBench**
  - 对话：**MT-Bench**

### 实验设置
- **Target Model**：Meta-Llama-3-8B-Instruct（主实验）
- **Drafter Model**：4-layer DFlash denoiser，block size $ K=16 $
- **特征输入**：从 target 的第 {0,10,20,30} 层提取 hidden states 作为条件
- **优化器**：AdamW，cosine schedule，effective batch size 32，peak LR 调优至 $10^{-3}$

### 评估指标
- **Average Accepted Length ($ \bar{T} $)**：每轮 speculative decoding 平均接受的 draft token 数（核心指标）
- **Throughput (TPS)**：每秒生成 token 数
- **Waste Ratio**：正确但因上游拒绝而被丢弃的 token 占比

### 基线方法
- **Position-uniform Baseline**：标准 DFlash，使用均匀权重的 cross-entropy 损失
- **EAGLE-3**：当前主流 feature-level autoregressive drafter
- **ddTree**：test-time 的树形 draft 结构，用于验证兼容性
- **SpecDiff-2**：同期工作，使用 streak-distillation 进行对齐

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | Avg. $ \bar{T} $ | 相对提升 |
|------|------------------|---------|
| Position-uniform baseline | 2.376 | — |
| + Loss Decay ($\gamma=10$) | 2.583 | +8.7% |
| + Focal Loss ($\alpha_f=0.3$) | 2.892 | +21.7% |
| + Chain Loss ($\alpha_c=40$) | **3.420** | **+43.9%** |

- 在 **HumanEval** 上，接受长度从 2.803 提升至 **4.336**（+54.7%）
- 在 **AIME** 上，从 2.705 提升至 **4.757**（+75.9%）

### 与基线对比
- 相比 EAGLE-3，block-diffusion + 对齐训练在 $K=16$ 下理论速度上限高 **3倍以上**。
- 与 **SpecDiff-2**（streak-distillation）结合后，$ \bar{T} $ 达到 **4.071**，相对 streak-distilled baseline 提升 **74.0%**。

### 消融实验结果
#### （1）各方法单独效果（Table 4）
| 方法 | $ \bar{T} $ |
|------|------------|
| Loss Decay only | 2.583 |
| Focal Loss only | 2.736 |
| Chain Reward only | **2.919** |

👉 结论：**Chain Reward 是最有效的单一方法**，说明直接优化联合前缀概率最为关键。

#### （2）跨模型泛化（Table 3）
在 Llama-3.2-3B、Qwen-3-4B/8B 上均观察到一致增益，表明方法具有**目标模型无关性**。

#### （3）训练数据影响（Table 5）
- 使用 target-aligned 数据后，baseline $ \bar{T} $ 从 2.376 → **3.914**（+64.7%）
- 叠加三项技术后达 **4.985**，相对原始 baseline 提升 **+109.8%**
👉 结论：**训练数据对齐与损失函数设计是正交且可叠加的增益来源**

#### （4）block size 影响（Table 6）
- 最佳表现出现在训练与推理 block size 一致时（$K=16$）
- 当 $K>16$，性能下降，说明模型未泛化到更长 block
- 对齐训练的增益随 $K$ 增大而扩大（在 $K=16$ 时达 +72%）

#### （5）与 ddTree 结合（Table 7）
| 方法 | TPS | $ \bar{T} $ |
|------|-----|-------------|
| Baseline (no ddTree) | 126.75 | 2.376 |
| Fully aligned (no ddTree) | 225.12 | 3.420 |
| Fully aligned + ddTree | **294.68** | **4.609** |

👉 两项技术**完全正交**，可叠加实现 **132.5% 吞吐提升**

---

## 4. 关键结论和发现

### 主要发现
1. **训练-验证不一致是 diffusion drafter 的核心瓶颈**：标准 bidirectional 训练导致大量正确 token 被浪费（平均浪费率 **46.9%**）。
2. **三种正交干预策略显著提升性能**：
   - Loss decay 提供简单有效的先验偏置；
   - Focal loss 精准打击“链断裂点”；
   - Chain reward 直接优化期望接受长度，效果最强。
3. **方法高效且通用**：无额外推理成本，适用于多种模型和任务，且可与 test-time 技术（ddTree、SpecDiff-2）无缝结合。

### 方法的局限性
- 所有方法仍基于 **fixed block size**（$K=16$），超出训练长度时性能下降。
- 依赖高质量的 **multi-layer feature extraction**，对 target 模型结构有一定要求。
- Chain reward 和 Focal loss 依赖 argmax 解码选择 breaker，可能受采样噪声影响。

### 未来工作方向
- 将 block size **动态化** 或引入层次化 block 结构。
- 探索 **test-time 与 training-time 的联合优化**，例如 adaptive loss weighting based on runtime difficulty。
- 将该框架扩展至 **非 diffusion 类 non-autoregressive drafters**。
- 研究如何将 **rejection feedback** 更有效地反向传播到 drafter 训练中。

--- 

> ✅ **总结一句话**：本文通过三项正交的训练目标对齐技术，显著提升了 block-diffusion drafters 在 speculative decoding 中的实际效用，在不增加推理成本的前提下，将平均接受长度提升 **21–76%**，为高效 LLM inference 提供了一套实用且可扩展的解决方案。

</details>

---

### 12. [DeMix: Debugging Training Data with Mixed Data Error Types by Investigating Influence Vectors](https://arxiv.org/abs/2606.11616)

**Authors**: Jiale Deng, Yanyan Shen, Xiaogang Shi, Chai Junjun  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11616v1  

#### Abstract
High-quality training data is essential for the success of machine learning models. However, real-world datasets often contain mixed types of errors arising from systematic flaws in data preparation pipelines, including label errors, feature errors, and spurious correlations. Effective debugging of ...

---

### 13. [MODF-SIR: A Multi-agent Omni-modal Distilled Framework for Social Intelligence Reasoning](https://arxiv.org/abs/2606.12018)

**Authors**: Shang Ma, Jisheng Dang, Wencan Zhang, Yifan Zhang, Bimei Wang, Hong Peng, Bin Hu, Qi Tian, Tat-Seng Chua  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.12018v1  

#### Abstract
We propose a multi-agent collaborative framework built upon a lightweight Multimodal Large Language Model (MLLM), specifically designed for social intelligence reasoning. A key feature of our approach is that both the training and inference phases are augmented via knowledge distillation. Within thi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MODF-SIR: A Multi-agent Omni-modal Distilled Framework for Social Intelligence Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**Social Intelligence Reasoning（社会智能推理）**这一复杂任务，旨在让多模态大语言模型（MLLM）能够理解人类意图、情绪、人际动态及隐含的社会规范。现实场景中，人类交流往往通过非显式的多模态信号（如语言、表情、动作等）传递意图，传统端到端的黑箱推理范式容易产生“认知过载”和幻觉（hallucination），难以精准捕捉细微且稀疏的**long-tail events**（长尾事件），例如微表情或语气变化。

### 提出的新方法与创新思路
作者提出了一种名为 **MODF-SIR**（Multi-agent Omni-modal Distilled Framework for Social Intelligence Reasoning）的轻量级、基于知识蒸馏的多智能体协作框架，其核心创新如下：

- ✅ **多智能体协同架构**  
  首次将多智能体系统引入社会智能推理领域，各模块分工明确，模拟人类“双过程理论”（Dual-Process Theory）：
  - **System 1（快速直觉）**：由 ELT Retriever Agent 执行粗粒度扫描，提取潜在 long-tail 事件线索。
  - **System 2（慢速分析）**：由 OMLT Reasoner Agent 进行细粒度因果推理，结合上下文进行深度分析。

- ✅ **动态路由机制（AKD Router Agent）**  
  引入一个基于**非对称知识蒸馏**（Asymmetric Knowledge Distillation, AKD）训练的路由器智能体，根据查询是否隐含以及是否存在 long-tail 事件，动态决定是否需要先执行时空定位（temporal grounding）。这避免了在简单任务上不必要的计算开销。

- ✅ **精确时空定位（GRPO Grounder Agent）**  
  针对 long-tail 事件难以全局搜索的问题，设计了一个基于 **GRPO**（Group Relative Policy Optimization）算法优化的定位器，能高效地从长视频流中精确定位与问题相关的片段，显著缩小推理范围。

- ✅ **测试时自适应（Test-Time Adaptation, TTA）闭环反馈机制**  
  构建了一个包含 **TTA Reviser Agent** 的自我修正循环：
  - 利用 LLM 更擅长“评估”而非“生成”的特性（generation-evaluation gap），由外部教师模型评估中间输出（Context、Chain-of-Thought、Answer）。
  - 若评分不足，则通过 **LoRA** 对基础模型进行实例级参数更新并重试，直至满足阈值。
  - 推理结束后丢弃 LoRA 权重，防止灾难性遗忘（catastrophic forgetting）。

- ✅ **文本化增强 long-tail 信号表达**  
  将多模态信号转化为结构化的自然语言描述（`Ctext`），使关键但微弱的 long-tail 信息在 tokenization 阶段不被主导事件或噪声淹没。

### 相比现有方法的优势
| 维度 | MODF-SIR 的优势 |
|------|------------------|
| **推理可靠性** | 多阶段验证 + TTA 自我修正，显著减少幻觉和错误传播 |
| **资源效率** | 动态路由机制按需启用复杂流程，节省计算成本 |
| **可解释性** | 显式展示检索、定位、推理全过程，提升透明度 |
| **泛化能力** | 蒸馏+TTA 设计使其在少量训练数据下仍表现优异（仅用 IntentTrain 的 ~30% 数据） |

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在三个主流多模态基准上进行了全面评估：

| 数据集 | 描述 |
|--------|------|
| **Daily-Omni [10]** | 包含日常生活中真实音视频片段，强调跨模态时间对齐与情境理解，涵盖不同长度子集（30s/60s）。 |
| **IntentBench [11]** | 专注于人类意图识别，包含 Why / How / When / Who 等类型问题，测试模型对心理状态和社会动机的理解能力。 |
| **WorldSense [9]** | 评估世界知识和特定领域理解（科技、文化、体育、音乐等），检验模型的知识广度与深度。 |

### 实验设置与评估指标
- **模型规模**：主干使用 **7B 参数级别的轻量 MLLM**（如 Qwen2.5-Omni-7B），确保公平比较。
- **输入形式**：原始视频 + 音频 + 用户查询。
- **输出形式**：答案 + Chain-of-Thought 推理链。
- **评估方式**：
  - 定量指标：准确率（Accuracy）、平均得分（Avg）。
  - 定性分析：可视化推理路径、消融研究。
- **TTA 设置**：最大迭代次数 `T_max` 控制重试次数，LoRA 更新为临时增量，推理后清除。

### 基线方法对比
分为两类进行对比：

#### Proprietary MLLMs（闭源商业模型）
- GPT-4o [12]
- Gemini-2.5-Pro (think) [13]
- Claude 3.5 Sonnet [51]

#### Open-Source Video-Audio MLLMs（开源多模态模型）
- Unified-IO-2 [45]
- VideoLLaMA2 [46]
- Qwen2.5-Omni [31]
- HumanOmniV2 [11]（当前最强开源基线之一）
- MiniCPM-o [49], Ola [47], VITA-1.5 [50]

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 📊 在 **Daily-Omni [10]** 上的表现（Table I）
| 方法 | 模型大小 | 平均得分（Avg） |
|------|----------|----------------|
| Gemini 2.0 Flash | - | 67.8 |
| HumanOmniV2 | 7B | 58.5 |
| **MODF-SIR (Ours)** | **7B** | **64.9** ✅ |

> 🔺 超越所有开源模型，接近闭源 Gemini Flash，领先 HumanOmniV2 达 **6.4 个百分点**。

#### 📊 在 **IntentBench [11]** 上的表现（Table II）
| 方法 | 模型大小 | 平均得分（Avg） |
|------|----------|----------------|
| GPT-4o | - | 60.0 |
| Gemini-2.5-Pro (think) | - | 67.2 |
| HumanOmniV2 | 7B | 69.3 |
| **MODF-SIR (Ours)** | **7B** | **70.3** ✅ |

> 🔺 **超越所有开源与多数闭源模型**，达到当前最优水平（SOTA）。

#### 📊 在 **WorldSense [9]** 上的表现（Table III）
| 方法 | 模型大小 | 平均得分（Avg） |
|------|----------|----------------|
| GPT-4o | - | 42.6 |
| Gemini 1.5 Pro | - | 48.0 |
| HumanOmniV2 | 7B | 47.1 |
| **MODF-SIR (Ours)** | **7B** | **51.5** ✅ |

> 🔺 显著优于最强开源基线（+4.4 pts），逼近甚至超过部分闭源模型。

---

### 消融实验结果（Table IV）

| 消融配置 | Avg 得分 | 分析说明 |
|---------|--------|--------|
| HumanOmniV2 (Baseline) | 58.5 | 原始基线 |
| + TTA Reviser | 64.0 | 单独加入 TTA 已带来巨大提升（+5.5 pts），证明自我评估机制有效 |
| + TTA + GRPO Grounder（无 Router） | 58.6 | 盲目启用定位反而引入噪声，性能下降 |
| + TTA + GRPO Grounder + AKD Router | 59.4 | 路由机制恢复性能，验证其必要性 |
| 完整 MODF-SIR | **64.9** | 所有组件协同作用达到峰值 |

> 🔍 发现：**AKD Router 是关键枢纽**，它决定了何时调用复杂流程；**TTA Reviser 是性能飞跃的核心驱动力**。

此外，还验证了 GRPO Grounder 相比传统 VideoMind Grounder 提升明显（58.6 vs 57.3），说明 GRPO 更好地优化了 IoU 指标。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **多智能体协作显著提升社会智能推理能力**：通过角色分工与流程编排，实现了更可靠、可解释的推理。
2. ✅ **TTA + LoRA 是实现高精度的关键**：利用生成-评估差距构建闭环反馈，使模型具备“边想边改”的能力。
3. ✅ **动态路由机制平衡效率与性能**：仅在必要时启动复杂流程，避免资源浪费。
4. ✅ **知识蒸馏可在小数据下获得高性能**：仅使用约 30% 的 IntentTrain 数据即达成 SOTA，显示方法高效性。
5. ✅ **文本化 long-tail 信号可防止信息丢失**：将视觉细节转为文本描述，有助于 LLM 注意力聚焦。

### 方法的局限性
- ⚠️ **依赖高质量教师模型进行蒸馏**：若教师模型本身存在偏见或错误，会污染伪标签。
- ⚠️ **推理延迟较高**：由于 TTA 多轮迭代机制，在线服务场景可能面临响应速度挑战。
- ⚠️ **目前主要针对单轮问答**：尚未扩展至连续对话或多轮交互中的社会推理。

### 未来工作方向
- 🔮 构建 **Video Shot Segmentation Agent**：利用 CV 中的镜头边界检测技术，将视频划分为语义稳定的片段，支持跨时段因果推理。
- 🔮 探索 **跨镜头语义关联挖掘**：使用 LLM 分析多个片段间的逻辑联系，建立长期记忆与推理链条。
- 🔮 扩展至 **多轮社会交互建模**：支持人机之间的情感演化与意图追踪。
- 🔮 降低 TTA 的计算开销：探索更高效的参数更新策略或提前终止机制。

---

> 💡 **总体评价**：  
> MODF-SIR 是一个多模态 AI 社会智能推理领域的突破性工作。它不仅在性能上刷新了多项开源记录，更重要的是提出了一个**结构清晰、可解释性强、具备自我纠错能力**的新范式，为构建真正理解人类行为的智能系统提供了重要路径。

</details>

---

### 14. [Benchmarking Large Language Models for Safety Data Extraction](https://arxiv.org/abs/2606.11204)

**Authors**: Jonas Grill, Thomas Bayer, S\"oren Berlinger  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.11204v1  

#### Abstract
Accurate extraction of structured information from Safety Data Sheets (SDS) remains challenging in industrial safety due to heterogeneous document formats and the limitations of traditional rule-based methods. This study benchmarks state-of-the-art Large Language Models (LLMs) for automated SDS data...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Benchmarking Large Language Models for Safety Data Extraction

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本研究针对工业安全领域中**Safety Data Sheets (SDS)** 数据提取的挑战展开。尽管SDS是符合GHS、REACH等法规的关键文件，但由于制造商之间在格式、术语和完整性上的高度异质性，传统基于规则或OCR的方法难以实现高精度、可扩展的自动化提取。手动处理成本高昂且易出错，限制了企业对化学品安全管理的效率。

### 提出了什么新方法或新思路
本文提出并系统评估了一套**基于Large Language Models (LLMs) 的SDS结构化信息提取框架**，其核心创新在于：
- 首次对当前最先进的四种主流LLMs（Gemini 1.5 Pro、GPT-4o、Claude 3.7 Sonnet、Llama 3.1-70B）进行了**端到端的基准测试（benchmarking）**。
- 对比了两种处理范式：**text-based pipeline**（将PDF转为Markdown后输入纯文本LLM） vs. **multimodal pipeline**（直接以图像形式输入支持视觉的LLM）。
- 系统性地评估了三种**Prompt Engineering策略**：Zero-Shot、Few-Shot 和 Chain-of-Thought，并分析其对性能的影响。

### 相比现有方法的优势
相比传统的OCR+规则匹配或早期机器学习模型，该方法具有以下优势：
- 利用LLMs强大的语义理解能力，能够应对非标准结构、自由文本描述和嵌套列表等复杂情况。
- 支持**in-context learning**，无需重新训练即可适配不同类型的SDS文档。
- 提供统一的评估框架（accuracy、latency、cost），便于横向比较不同模型与配置的实际部署可行性。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 数据来源：来自公开数据库 ChemicalSafety.com 的 **10份真实SDS文档**。
- 内容覆盖：涵盖多个制造商（如BASF、Sigma-Aldrich、Merck），均为近期版本，确保符合最新的GHS和REACH/CLP标准。
- 字段规模：共标注约 **50,000个字段**，涉及所有主要SDS章节（如产品标识、危害识别、急救措施、成分信息等）。
- 复杂性设计：包含key-value对、嵌套列表、表格混合数值与单位、图像型GHS象形图等多种结构。

### 实验设置和评估指标

#### 模型选择
| Model | 类型 | 是否多模态 |
|-------|------|------------|
| Gemini 1.5 Pro | Closed-source | ✅ |
| GPT-4o | Closed-source | ✅ |
| Claude 3.7 Sonnet | Closed-source | ✅ |
| Llama 3.1-70B | Open-source | ❌（仅文本） |

#### 提示策略（Prompting Strategies）
| 策略 | 描述 |
|------|------|
| Zero-shot | 无示例，仅通过任务说明引导 |
| Few-shot | 提供若干输入-输出样例 |
| Chain-of-Thought (CoT) | 要求模型先推理再输出JSON |

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy** | 正确提取字段占比：<br>`(TP + TN) / (TP + FP + FN + TN)` |
| **Not-Found Rate (NF Rate)** | 应提取但未提取的比例：<br>`FN / (TP + FN)` |
| **False-Positive Rate (FP Rate)** | 错误生成（幻觉）字段比例：<br>`FP / (FP + TN)` |
| **BERTScore** | 基于上下文嵌入的语义相似度评分 |
| **Processing Time (Latency)** | 单文档端到端处理时间（秒） |
| **Cost** | 基于token计费模型计算的成本（$） |
| **Normalized Score** | 综合得分：<br>`0.7×Accuracy_norm + 0.2×Time_norm + 0.1×Cost_norm` |

#### 基线方法对比
本文没有传统算法作为显式基线，而是将以下组合视为“baseline”进行内部比较：
- 不同模型之间的横向对比
- 文本 vs. 多模态处理路径
- 不同prompt策略的效果差异

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 模型 | 最佳准确率 | 处理时间（最佳） | 总成本（最低） |
|------|-----------|------------------|----------------|
| **Gemini 1.5 Pro** | **84%**（text + CoT） | 77.58s | $0.15 |
| GPT-4o | 81%（text + Zero-shot） | **73.51s** | $0.57 |
| Claude 3.7 Sonnet | 79% | 180.29s | $0.42 |
| Llama 3.1-70B | 71%（text + Few-shot） | 271.82s | **$0.07** |

> 注：所有准确率均未达到工业级所需的 **90% 可靠性阈值**

### 与基线方法的对比结果

#### ✅ 文本 vs. 多模态
- **文本管道全面优于多模态管道**：
  - 平均提升 **4–9个百分点** 准确率
  - 多模态引入额外OCR误差和延迟（如GPT-4o从73s增至近300s）
  - 示例：Gemini 1.5 Pro 文本准确率为82%，多模态仅为73%

#### ✅ Prompt策略影响
- **Zero-shot 表现最优或持平 Few-shot**：
  - 在多数情况下，添加few-shot示例并未提高准确率，反而增加token消耗和延迟
  - 支持“lost in the middle”现象假设——模型对提示中间部分关注度下降
- **Chain-of-Thought 仅对Gemini有效**：
  - 显著提升Gemini表现（从~82% → 84%）
  - 对其他模型无明显增益

#### ✅ 模型间差距显著
- **闭源商业模型远超开源模型**：
  - Gemini/GPT-4o/Claude 准确率集中在76–84%
  - Llama 3.1-70B 最高仅达71%，且延迟最长、错误率更高
- **模型选择 > 提示工程 > 输入模式**

### 消融实验结果
- **消融维度**：分别控制模型、输入方式（text/multimodal）、prompt策略
- **关键发现**：
  - 模型本身是最大变量（解释约70%性能方差）
  - 输入方式次之（文本优于多模态）
  - Prompt策略影响最小，甚至可能因增加长度而降低稳定性

---

## 4. 关键结论和发现

### 论文的主要发现
1. **文本优先原则成立**：对于数字原生PDF文档，**text-based extraction consistently outperforms multimodal processing**，因其避免了OCR噪声和视觉解析不确定性。
2. **Gemini 1.5 Pro 是综合最优解**：结合Chain-of-Thought提示，在准确率（84%）、延迟（77.58s）和成本（$0.15）上取得最佳平衡，归一化得分为 **0.88**。
3. **尚不满足无人监督部署要求**：尽管表现优异，但**没有任何模型突破90%准确率门槛**，尤其在防止hallucination（假阳性）方面仍有风险。
4. **Prompt Engineering 效果有限**：Zero-shot常优于Few-shot；CoT仅特定模型受益，表明通用提示难以替代领域优化。

### 方法的局限性
- **数据集规模小**：仅使用10份SDS，泛化能力有待验证。
- **缺乏细粒度错误分析**：未区分不同类型字段（如CAS号 vs. 浓度范围）的提取难度。
- **未考虑降质扫描件**：实际场景中的模糊、倾斜、水印等情况未被测试。
- **多模态OCR质量依赖API实现**：各厂商OCR能力不透明，影响公平比较。

### 未来工作方向
1. **Domain-adapted fine-tuning**：在大规模SDS语料上进行指令微调（instruction tuning），提升领域适应性。
2. **Human-in-the-Loop verification**：引入人工审核机制，特别是在高风险字段（如急救措施、PPE建议）上进行确认。
3. **Confidence calibration**：开发可靠的置信度估计机制，辅助决策是否需要人工干预。
4. **扩展benchmark规模**：纳入更多语言、更多行业、更复杂的SDS变体，并发布标准化评测集（类似ChemTEB）。
5. **Hybrid架构探索**：结合LLM语义理解与结构化规则引擎，兼顾灵活性与安全性。

---

> 📌 **一句话总结**：  
> 当前最先进的LLMs在SDS数据提取任务中展现出强大潜力，尤其是**Gemini 1.5 Pro + text-based + Chain-of-Thought**组合表现领先，但仍**未达到工业级90%可靠性标准**，需结合**人类复核机制**才能安全落地应用。

</details>

---

### 15. [uva-irlab-conv at SemEval-2026 Task 8: Multi-Turn RAG with Learned Sparse Retrieval and Listwise Reranking](https://arxiv.org/abs/2606.11945)

**Authors**: Simon Lupart, Kidist Amde Mekonnen, Zahra Abbasiantaeb, Mohammad Aliannejadi  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.11945v1  

#### Abstract
This report describes our participation in SemEval-2026 Task 8 on multi-turn retrieval and question answering. The task evaluates conversational systems across four domains (finance, cloud documentation, government, Wikipedia), and includes unanswerable queries where the available collection does no...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：uva-irlab-convy at SemEval-2026 Task 8: Multi-Turn RAG with Learned Sparse Retrieval and Listwise Reranking

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对 **multi-turn retrieval and question answering (QA)** 中的关键挑战，特别是在多轮对话中处理复杂信息需求、话题漂移、指代消解和模糊性等问题。任务特别强调在**领域特定集合**（如金融、云文档、政府文件）中的表现，并要求系统能够识别和妥善处理 **unanswerable 或 underspecified queries**（即缺乏足够支持证据的问题）。

### 提出的新方法与创新点
作者提出了一种**多阶段级联的 RAG（Retrieval-Augmented Generation）pipeline**，其核心创新包括：

- **Learned Sparse Retrieval (LSR) 作为第一阶段检索**  
  采用基于 LLM backbone 的稀疏检索模型 **LION-SP-8B**，结合神经语义建模与词法稀疏性，在跨域场景下具有强泛化能力，同时兼容高效的倒排索引搜索。

- **LLM-based Conversational Query Rewriting**  
  利用 GPT-4.1 对话历史重写最新用户提问为独立查询（standalone query），解决指代（anaphora）和省略（ellipsis）问题，提升后续检索兼容性。

- **两阶段 LLM Reranking：Pointwise + Listwise**  
  - 先通过 **Qwen3-Reranker-8B** 进行 pointwise 重排序，筛选 top 20 候选段落；
  - 再引入 **GPT-4.1 的 listwise reranking**，利用完整对话上下文对候选进行联合比较与精细排序，实现更优的证据选择。

- **Zero-shot RAG Generation**  
  最终响应生成完全在 zero-shot 设置下完成，使用 GPT-4.1 结合 top 5 排名段落与完整对话历史生成答案，无需任务微调。

### 相比现有方法的优势
- **轻量且无需监督训练**：整个流程基于 zero-shot LLM 操作，避免了昂贵的任务特定标注与训练。
- **强跨域适应性**：LSR 在未见过的领域（如 FiQA、Cloud 文档）仍保持良好检索效果。
- **上下文整合更充分**：listwise reranking 显式利用全对话历史，缓解 query rewriting 压缩上下文导致的信息丢失。
- **高效与精度平衡**：先用高效稀疏检索缩小候选集，再用高成本 LLM 进行精细化排序，兼顾性能与效率。

---

## 2. 核心实验方法和设置

### 数据集
基于 **MT-RAG (MTRAGEval)** benchmark，包含四个英文领域的对话 QA 数据集：
- **ClapNQ**：自然问题长文本问答
- **FiQA**：金融领域问答
- **GovT**：政府文档
- **IBM Cloud**：云计算技术文档

共 332 轮对话交互，按领域划分 turn 数分别为：84 (ClapNQ), 59 (FiQA), 106 (GovT), 87 (IBM Cloud)

### 实验设置
- **任务分解**：
  - **Task A**: Conversational Search（检索任务）
  - **Task B**: Oracle RAG Generation（使用黄金段落生成）
  - **Task C**: Predicted RAG Generation（使用自己检索的结果生成）

- **模型组件**：
  - Query Rewriting & Listwise Reranking & Response Generation: **GPT-4.1**
  - Pointwise Reranker: **Qwen3-Reranker-8B**
  - First-stage Retriever: **LION-SP-8B**（基于 llama3 架构的 LSR 模型）

- **实现工具**：
  - 倒排索引使用 **Seismic library**
  - 所有 LLM 组件均以 **zero-shot 方式运行**，无任务微调
  - 检索模型仅在 MSMARCO 上预训练，未在目标领域 fine-tune

### 评估指标
| 任务 | 主要指标 |
|------|---------|
| **Task A (Retrieval)** | **nDCG@5**（官方排名依据），nDCG@1, nDCG@10 |
| **Task B/C (Generation)** | 
| - 内容质量 | **RB_agg**（BERT-Recall/K-Precision/ROUGE-L 调和平均）、**RB_llm**（LLM-as-judge） |
| - 忠实度（Faithfulness） | **RL_F**（基于 RAGAS 框架） |
| - 综合得分 | **H.Avg**：上述三项的调和平均（IDK-conditioned） |

> 注：IDK-conditioned 表示排除“无法回答”类别的 turn 后计算，反映系统在可答情况下的表现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Task A - Retrieval 性能（Table 1）
| 方法 | nDCG@1 | **nDCG@5*** | nDCG@10 |
|------|--------|--------------|----------|
| Baseline (GPT-OSS-20B + ELSER) | — | 0.4795 | — |
| LSR (LION-SP-8B) | 0.4910 | 0.4841 | 0.5343 |
| + Pointwise Reranking | 0.5120 | **0.5477** | 0.5921 |
| + GPT-4.1 Listwise Reranking (**最终系统**) | **0.5331** | **0.5475** | **0.5943** |

> 🏆 **最终系统在 38 支队伍中排名第 2**，nDCG@5 达到 **0.5475**

#### 🔍 Domain-Specific Retrieval Performance（Table 2）
| 领域 | nDCG@5 |
|------|--------|
| **ClapNQ** | 0.645 |
| **FiQA** | 0.330 ⚠️（最难） |
| **GovT** | 0.587 |
| **IBM Cloud** | 0.552 |

> FiQA 表现最弱，但首相关段落平均出现在 rank 3，说明仍具备基本检索能力。

#### ✅ Task B/C - Generation 性能（Table 3）

| 系统 | **H.Avg\*** (IDK-cond.) | RB_agg | RB_llm | RL_F |
|-------|--------------------------|--------|--------|------|
| **uva-oracle (Task B)** | 0.5123 (**23/26**) | 0.3683 | 0.6807 | 0.5981 |
| **uva-rag (Task C)** | 0.4865 (**20/29**) | 0.3197 | 0.6538 | **0.6626** |

> 尽管生成排名不高，但 **faithfulness (RL_F) 高于 oracle 设置**，表明系统更忠实于检索证据。

#### 💡 非 IDK-conditioned 分析（更真实场景）
当包含所有 turn（含 unanswerable）时：
- uva-rag 的 H.Avg 提升至 **0.5619**
- RL_F 高达 **0.8035**
- 此时性能甚至超过 oracle 配置 → 体现检索模块的强大稳定性

### 消融实验结果（Ablation Study）
从 Table 1 可见各阶段增益：
- **Query Rewriting + LSR** 已超越官方 baseline
- **Pointwise Reranking** 贡献最大提升（+6.36 pts @nDCG@5）
- **Listwise Reranking** 主要优化 top-1 排名（+2.1 pts @nDCG@1），对 nDCG@5 影响较小但仍有益

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Strong Retrieval → Better Generation**  
   图 3 显示 retrieval (nDCG@10) 与 generation 质量（RB_agg, RB_llm）呈正相关（Pearson r = 0.46），验证高质量检索是优质生成的基础。

2. ✅ **Faithfulness 不依赖于检索排名**  
   RL_F 与 nDCG 几乎无相关性，说明忠实度更多取决于生成策略而非检索顺序，提示需专门设计机制保障事实一致性。

3. ✅ **系统在 unanswerable 场景下仍保持高 Faithfulness**  
   即使面对无答案问题，系统也极少幻觉（hallucination），归因于保守的 prompt 设计（见附录）。

4. ✅ **对话深度不影响检索，但影响生成**  
   - 图 4：随着 conversation depth 增加，检索性能稳定；
   - 图 5：生成质量随轮次加深而下降，可能因上下文过长或信息累积噪声所致。

5. ✅ **FiQA 是最具挑战性的领域**  
   因其专业术语密集、文档结构复杂，导致检索延迟（first relevant at rank 3 vs ~2 in others）

### 方法的局限性
- ❌ **未显式建模 unanswerable queries**  
  导致在 IDK-conditioned 指标上排名偏低，未能主动声明“我不知道”或请求澄清。
- ❌ **Listwise reranking 成本高**  
  受限于 context window，只能作用于 top 20 段落，难以扩展到更大范围。
- ❌ **依赖强大 LLM（如 GPT-4.1）**  
  整个 pipeline 高度依赖闭源、高性能 LLM，限制可复现性和部署灵活性。
- ❌ **未探索 advanced RAG 技术**  
  如 query decomposition、self-reflection、agent-style reasoning 等未纳入当前框架。

### 未来工作方向
- 引入 **explicit IDK detection module**，增强对 unanswerable query 的识别与响应能力。
- 探索 **unified modeling of retrieval and generation**，减少 pipeline 错误传播。
- 尝试 **open-source LLM 替代方案**，提升系统的开放性与实用性。
- 研究 **long-context management techniques**，应对深层对话中的信息衰减问题。
- 加入 **query decomposition 与 self-correction 机制**，提升复杂多跳问题处理能力。

--- 

> **总结一句话**：该工作展示了以 **Learned Sparse Retrieval + LLM-based Listwise Reranking** 为核心的 zero-shot 多轮 RAG pipeline 在跨域对话 QA 中的强大检索能力和高忠实度生成潜力，虽生成排名受限，但在稳健性和可信性方面表现出色。

</details>

---

### 16. [Can News Predict the Market? Limits of Zero-Shot Financial NLP and the Role of Explainable AI](https://arxiv.org/abs/2606.12210)

**Authors**: Ali M Karaoglu, Shreyank N Gowda  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.12210v1  

#### Abstract
Can financial news reliably predict short-term stock movements? Despite advances in large language models, this question remains unresolved. We revisit this problem using a zero-shot natural language processing framework, investigating whether models can extract actionable signals from financial new...

---

### 17. [On The Effectiveness-Fluency Trade-Off In LLM Conditioning: A Systematic Study](https://arxiv.org/abs/2606.12234)

**Authors**: Iuri Macocco, Pau Rodr\'iguez, Arno Blaas, Luca Zappella, Marco Baroni, Xavier Suau  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.12234v1  

#### Abstract
Controlling the output of Large Language Models (LLMs) is a central challenge for their reliable deployment, yet a clear understanding of the involved trade-offs remains elusive. Current approaches to conditioning are often evaluated with a narrow focus on their effectiveness at injecting or removin...

---

### 18. [SwiftCTS: Fast Cross-Design Prediction and Pareto Optimization of Clock Tree Metrics via Few-Shot Calibration](https://arxiv.org/abs/2606.11348)

**Authors**: Barsat Khadka, Kawsher Roxy, Md Rubel Ahmed  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.11348v1  

#### Abstract
Clock Tree Synthesis (CTS) is a computationally expensive stage in the physical design flow, requiring iterative EDA tool invocations to navigate a vast configuration space for optimal power, wirelength, and timing skew. Existing machine learning approaches require computationally expensive retraini...

---

### 19. [A Riemannian Approach to Low-Rank Optimal Transport](https://arxiv.org/abs/2606.12120)

**Authors**: Pratik Jawanpuria, Bamdev Mishra  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.12120v1  

#### Abstract
Low-rank optimal transport (OT) mitigates the quadratic scaling of classical solvers, yet existing approaches rely heavily on first-order mirror-descent updates that require careful hyperparameter tuning and ignore the optimization landscape's curvature. To address these limitations, we propose a un...

---

### 20. [PCA-Enhanced Adaptive NVAR Framework for High-Resolution Sea Surface Temperature Forecasting in the East Sea](https://arxiv.org/abs/2606.12141)

**Authors**: Sherkhon Azimov, Susana L\'opez-Moreno, Eric Dolores-Cuenca, JinYong Choi, Sangil Kim  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.12141v1  

#### Abstract
Accurate forecasting of sea surface temperature (SST) in regional seas such as the East Sea is crucial for monitoring marine ecosystems, assessing climate risks, managing fisheries, and conducting naval operations. Traditional numerical ocean models provide reliable predictions but are computational...

---

### 21. [TouchThinker: Scaling Tactile Commonsense Reasoning to the Open World with Large-scale Data and Action-aware Representation](https://arxiv.org/abs/2606.11637)

**Authors**: Kailin Lyu, Di Wu, Pengwei Zhang, Yuhang Zheng, Yingxin Lai, Long Xiao, Kangyi Wu, Pengna Li, Chen Gao, Lianyu Hu, Xiaobin Hu, Jie Hao, Ce Hao, Weihao Yuan, Shuicheng Yan  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11637v1  

#### Abstract
Touch is a key modality for embodied agents to understand the physical world. Although recent work has incorporated tactile signals into language systems for tactile commonsense reasoning, scaling such systems to realistic open-world settings remains challenging due to two key bottlenecks: (1) curre...

---

### 22. [Organize then Retrieve: Hierarchical Memory Navigation for Efficient Agents](https://arxiv.org/abs/2606.11680)

**Authors**: Hao-Lun Hsu, Nikki Lijing Kuang, Boyi Liu, Zhewei Yao, Yuxiong He  
**Category**: cs.AI  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11680v1  

#### Abstract
Large language model (LLM) agents struggle with long-horizon tasks due to their inherent statelessness, requiring all task-relevant information to be encoded in growing input contexts. The resulting degraded reasoning quality, increased inference cost, and higher latency necessitate efficient workin...

---

### 23. [EverydayGPT: Confidence-Gated Routing for Efficient and Safe Hybrid GPT-RAG Conversational QA](https://arxiv.org/abs/2606.11212)

**Authors**: Jaspreet Singh Nahal  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11212v1  

#### Abstract
Standard Retrieval-Augmented Generation (RAG) pipelines route every query through retrieval and generation unconditionally, incurring unnecessary computation and propagating low-quality context to the generator. We introduce EverydayGPT, a lightweight conversational QA system built around a Confiden...

---

### 24. [GraspLLM: Towards Zero-Shot Generalization on Text-Attributed Graphs with LLMs](https://arxiv.org/abs/2606.11898)

**Authors**: Hengyi Feng, Zeang Sheng, Meiyi Qiang, Meiyi Qiang, Wentao Zhang  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11898v1  

#### Abstract
Research on Text-Attributed Graphs (TAGs) has gained significant attention recently due to its broad applications across various real-world data scenarios, such as citation networks, e-commerce platforms, social media, and web pages. Inspired by the remarkable semantic understanding ability of Large...

---

### 25. [Context-Driven Incremental Compression for Multi-Turn Dialogue Generation](https://arxiv.org/abs/2606.12411)

**Authors**: Yeongseo Jung, Jaehyeok Kim, Eunseo Jung, Jiachuan Wang, Yongqi Zhang, Ka Chun Cheung, Simon See, Lei Chen  
**Category**: cs.CL  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.12411v1  

#### Abstract
Modern conversational agents condition on an ever-growing dialogue history at each turn, incurring redundant attention and encoding costs that grow with conversation length. Naive truncation or summarization degrades fidelity, while existing context compressors lack cross-turn memory sharing or revi...

---

### 26. [Harnessing Routing Foresight for Micro-step-level MoE load balancing in RL Post-training](https://arxiv.org/abs/2606.11867)

**Authors**: Yuming Zhou, Haoyang Li, Sheng Lin, Yanfeng Zhao, Tong Zhao, Xupeng Miao, Jie Jiang, Fangcheng Fu, Bin Cui  
**Category**: cs.DC  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11867v1  

#### Abstract
Mixture-of-Experts (MoE) and reinforcement learning (RL) post-training now dominate large language model (LLM) development, yet expert load imbalance remains a critical challenge. Existing load-balancing systems target pre-training by relying on historical step-level statistics. However, these metho...

---

### 27. [APEX: A Network-Native Time-Series Foundation Model for Forecasting and Anomaly Detection for Wireless Edge Operations](https://arxiv.org/abs/2606.11553)

**Authors**: Swadhin Pradhan, Niloo Bahadori, Peiman Amini  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11553v1  

#### Abstract
Generic time-series foundation models transfer poorly to wireless network telemetry whose signals are bursty, zero-inflated, and coupled across protocol layers. We present APEX, a network-native, decoder-only transformer for forecasting enterprise AP telemetry, and evaluate it on DHCP degradation as...

---

### 28. [Range-Aware Bayesian Optimization for Discovering Diverse Designs within Target Property Windows](https://arxiv.org/abs/2606.11574)

**Authors**: Shengli Jiang, Jason Wu, Charles M. Schroeder, Michael A. Webb  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11574v1  

#### Abstract
In many materials and product design problems, desirable candidates exhibit properties that fall within an acceptable range rather than achieve a single optimum. Recovering multiple, distinct solutions that satisfy such specifications is also practically valuable, as some candidates may be preferred...

---

### 29. [TimeRouter: Efficient and Adaptive Routing of Time-Series Foundation Models](https://arxiv.org/abs/2606.11625)

**Authors**: Kanghui Ning, Yushan Jiang, Kashif Rasul, Anderson Schneider, Yuriy Nevmyvaka, Dongjin Song  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11625v1  

#### Abstract
Time-series foundation models (TSFMs) are increasingly explored as predictive experts within emerging agentic time-series systems. However, TSFMs exhibit heterogeneous inductive biases, and no single model consistently dominates across forecasting regimes, making expert selection a critical challeng...

---

### 30. [Spectrally Regularized Latent Flow Matching for Turbulence Generation](https://arxiv.org/abs/2606.11691)

**Authors**: Khalid Rafiq, Aditya G. Nair  
**Category**: cs.LG  
**Published**: 2026-06-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2606.11691v1  

#### Abstract
Latent diffusion and flow matching have emerged as leading approaches for synthetic turbulence generation, yet they systematically under-represent dissipation-range amplitudes. We introduce a latent flow matching framework with a spectrally regularized compression stage that directly targets this fa...

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
