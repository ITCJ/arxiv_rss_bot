# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-21 07:19:04 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing](https://arxiv.org/abs/2604.18170)

**Authors**: Ziyang Liu  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2604.18170v1  

#### Abstract
LLMs edit text and code by autoregressively regenerating the full output, even when most tokens appear verbatim in the input. We study Copy-as-Decode, a decoding-layer mechanism that recasts edit generation as structured decoding over a two-primitive grammar:  references an input line range, ... emi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）在执行文本或代码编辑任务时，通常采用**自回归生成**（autoregressive decoding）方式从头生成整个输出。然而，大多数编辑操作中，大部分内容是直接复制输入的（如保留未修改的代码段），这导致大量计算资源被浪费在重复生成已存在的内容上。

这种模式带来了两个核心问题：
- **高延迟**：推理时间与输出长度成正比，而非实际“编辑量”。
- **语义漂移风险**：模型可能对本应保持不变的内容进行错误改写。

### 提出的新方法：Copy-as-Decode
本文提出了一种新的解码层机制 **Copy-as-Decode**，其核心思想是将编辑过程建模为一个**结构化解码程序**，通过语法约束引导模型只生成必要的新内容，并高效地“复制”已有部分。

#### 创新点
1. **两原语语法（Two-Primitive Grammar）**  
   定义了一个简单的程序化语法：
   ```xml
   <program>
     <copy lines="i-j"/>  <!-- 复制输入中的第 i 到 j 行 -->
     <gen>new content</gen> <!-- 生成新内容 -->
   </program>
   ```
   输出不再是自由文本，而是由 `<copy>` 和 `<gen>` 构成的程序。

2. **Token-Level FSM 强制语法合规**  
   使用一个**有限状态机**（Finite-State Machine, FSM）在 token 层面强制模型遵守上述语法，确保所有输出都可被解析，**解析率（parse rate）达到 100%**。

3. **Parallel Prefill 实现 Copy 加速**  
   当模型决定执行 `<copy lines="i-j"/>` 时，系统不再逐 token 自回归生成这些内容，而是直接将对应输入 token 批量插入 KV Cache，仅需一次 **parallel-prefill forward**，从而跳过 N 次自回归步骤。

4. **确定性解析器（Deterministic Resolver）**  
   将程序转换回最终编辑文档的过程是完全确定性的，不依赖模型，保证了可复现性和正确性。

### 相比现有方法的优势
| 方法 | 缺陷 | Copy-as-Decode 的改进 |
|------|------|------------------------|
| **Full Regeneration** | 重新生成所有内容，效率低 | 只生成必要部分，其余复制 |
| **Search/Replace Blocks**（如 Aider） | 存在锚点歧义（anchor ambiguity），多个匹配项导致错误替换 | 使用行号索引，无歧义 |
| **Unified Diff** | 需要冗余头部信息，增加输出 token 数 | 无需头部，更紧凑 |
| **Speculative Decoding** | 加速自由生成，但无法保证复制区域的 token identity | 输入即“草稿”，接受率为 1，零拒绝风险 |

> ✅ **本质区别**：Copy-as-Decode 不是分布近似，而是一种**程序化、语法驱动的确定性加速机制**。

---

## 2. 核心实验方法和设置

### 数据集
- **ProbeEdit**：154 个短文本编辑样本，涵盖会议记录、技术文档等四类文本。
- **HumanEvalPack-Fix (Python & JavaScript)**：各 164 个代码 bug 修复样本，用于评估代码编辑能力。
- 总计 **482 个 (input, gold) 对**。

### 实验设置
- **模型**：Qwen2.5-{1.5B, 7B}-Instruct（主实验）、Qwen2.5-Coder-1.5B（微调实验）
- **硬件**：单张 A100 80GB GPU，bf16 精度
- **运行环境**：HuggingFace Transformers（eager 模式），无 vLLM/SGLang 集成
- **上下文长度**：1024 tokens 前缀

### 评估指标
1. **Kernel Speedup**：单次 copy 操作中，parallel prefill 相比 N 次自回归 decode 的速度提升倍数。
2. **Copy Ceiling**：黄金输出中有多少比例的 token 可以通过 `<copy>` 原语实现（理想上限）。
3. **bndexact**：结合 span 分布和 kernel 加速曲线得出的**端到端理论最大加速比**。
4. **Pipeline Losslessness**：使用 oracle program 能否完美重构黄金输出（字节级相等）。
5. **Exact Match (EM)**：最终输出是否与黄金完全一致。
6. **Span Selection Accuracy**：模型选择的 copy 范围是否准确。

### 基线方法对比
- **Full Regeneration**：标准自回归生成。
- **Search/Replace**：Aider 风格的搜索替换块。
- **Unified Diff**：标准 diff 格式。
- **Prompt-level CRP**：早期工作，仅作为提示格式使用，无 KV 拼接。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）Kernel Speedup（单次 Copy 加速）
在 Qwen2.5-1.5B 上，复制 N 个 token 的速度提升高达 **303×**（N=512）；在 7B 上达 **90.5×**。即使对于小 span（N=8），也有 **6.8–7.0×** 加速。

> 🔹 图 3 显示：随着 span 增长，加速比单调上升，在 1.5B 上持续增长，在 7B 上趋于饱和（进入计算瓶颈区）。

#### （2）Copy Ceiling（复制覆盖率）
- **ProbeEdit**：97.8% 的黄金 token 可通过 `<copy>` 实现。
- **HEvalFix-Py**：74.1%
- **HEvalFix-JS**：78.8%
- **总体平均**：93.8%

这意味着超过 90% 的编辑内容本质上是“复制粘贴”，仅有少量需要生成。

#### （3）理论最大加速比 `bndexact`
这是本文的核心量化结论，表示任何基于该机制的系统所能达到的**理论性能天花板**：

| 数据集 | `bndexact`（理论最大加速比） |
|--------|-------------------------------|
| ProbeEdit | **29.0×** |
| HEvalFix-Py | **3.4×** |
| HEvalFix-JS | **4.2×** |
| **总体池化（pooled）** | **13.0×** |

> 💡 注：该值由 `T / (Tgen + Σ Nk/s(Nk))` 计算而来，考虑了每个 copy span 的实际长度及其对应的 kernel 加速比。

#### （4）Pipeline Losslessness（管道无损性）
- 在 **482 个案例**中，使用 oracle program 经 resolver 解析后，**全部实现了字节级精确重建**（byte-exact EM = 100%）。
- 这表明：只要 span 选得准，就能完美还原输出——失败只能归因于 span selection 错误。

#### （5）Span Selection 精度要求（扰动研究）
引入行号噪声（±e lines）测试鲁棒性：

| 噪声半径 e | Pool EM 下降至 |
|-----------|---------------|
| 0         | 100.0%        |
| 1         | **15.48%**    |
| 2         | 9.42%         |

👉 结论：**span 边界必须近乎完美**，哪怕偏移一行也会导致 EM 断崖式下跌。

#### （6）监督微调试点（SFT Pilot）
在 Qwen2.5-Coder-1.5B 上进行小规模 SFT（131–385 样本，3 种随机种子）：
- **未训练模型**：EM = 0/33（0%）
- **SFT 后**：EM 提升至 **12–17%**（Wilson 95% CI [7.0, 26.0]）
- 表明 span selection 是**可学习的**，但远未达到部署水平。

### 与基线方法对比（Table 3）

| 方法 | 输出 token 数 | Round-Trip EM | 是否有锚点歧义 |
|------|----------------|----------------|----------------|
| Full Regeneration | 最多 | 1.00 | 否 |
| Search/Replace | 较少 | **0.81–0.94** | ✅ 是（常见失败原因） |
| Unified Diff | 中等 | 1.00 | 否（但头部开销大） |
| **Copy-as-Decode** | **最少或相当** | **1.00** | ❌ 否 |

✅ **唯一同时实现最低 token 开销和 100% 回放成功率的方法**。

### 消融实验结果
- **移除 FSM**：模型可能偏离语法，导致解析失败（EM 崩溃）。
- **移除 Splice**：失去 kernel 加速优势，退化为普通格式压缩。
- **闭源模型使用 Prompt-Level 版本**（Appendix Q）：
  - 如 GPT-5-mini 上 EM 达 80%，说明格式本身有价值。
  - 但无 KV 拼接，**无法获得 kernel speedup**。

---

## 4. 关键结论和发现

### 主要发现
1. **编辑的本质是“复制+局部生成”**：在真实编辑任务中，**74–98% 的输出 token 可直接从输入复制**，传统自回归生成严重低效。
2. **Copy-as-Decode 可带来巨大加速潜力**：理论上可达 **13–29× 的端到端加速**（取决于任务类型）。
3. **加速的关键在于“确定性拼接”**：利用输入作为“草稿”，通过语法承诺实现 **acceptance=1**，避免 speculative decoding 的验证成本和不确定性。
4. **当前瓶颈是 span selection**：模型能否精准识别哪些行应该 copy、哪些应该 gen，决定了最终性能。
5. **该机制独立于模型质量**：提出的三个核心属性（kernel speedup、copy ceiling、pipeline losslessness）均不依赖模型训练，是对机制本身的分析。

### 方法的局限性
1. **尚未集成到生产级服务框架**：未接入 vLLM 或 SGLang 等支持批处理和分页注意力的系统。
2. **缺乏高性能的 span selector**：当前 SFT 仅实现 12–17% EM，离实用差距较大。
3. **仅支持单文件编辑**：不适用于 SWE-bench 类的多文件代理任务。
4. **粒度限制**：当前 `<copy lines="i-j"/>` 是按行复制，无法处理行内细粒度修改（如变量重命名）。虽提出了 `<copy tokens="a-b"/>` 的 token 级扩展，但尚未实现训练。
5. **依赖 KV Cache 访问权限**：闭源模型无法启用 kernel 加速，只能享受格式压缩好处。

### 未来工作方向
1. **扩大 span selector 训练规模**：使用更大模型（如 7B）和更多数据进行 SFT 或 QLoRA 微调。
2. **实现 token-level copy primitive**：支持 `<copy tokens="a-b"/>`，进一步提高覆盖率达到 91–99%。
3. **集成到 vLLM/SGLang**：实现真正的 batched serving 支持，测量真实吞吐和延迟。
4. **组合其他加速技术**：如与 **Speculative Decoding** 结合，在 `<gen>` 区域继续加速自由生成。
5. **扩展至多文件和 AST-level 编辑**：支持跨文件引用和语法树节点级别的复制操作。

---

> 📌 **总结一句话**：  
> **Copy-as-Decode 提供了一种语法驱动、确定性加速的 LLM 编辑新范式，通过将“复制”操作转化为一次 parallel prefill，理论上可实现数十倍加速，其性能上限已被严格界定，下一步关键是训练出高精度的 span selector 并完成工程集成。**

</details>

---

### 2. [Cloud-native and Distributed Systems for Efficient and Scalable Large Language Models -- A Research Agenda](https://arxiv.org/abs/2604.17227)

**Authors**: Minxian Xu, Jingfeng Wu, Shengye Song, Satish Narayana Srirama, Bahman Javad, Rajiv Ranjan, Devki Nandan Jha, Sa Wang, Wenhong Tian, Huanle Xu, Li Li, Zizhao Mo, Shuo Ren, Thomas Kunz, Petar Kochovski, Vlado Stankovski, Kejiang Ye, Chengzhong Xu, Rajkumar Buyya  
**Category**: cs.DC  
**Published**: 2026-04-21  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2604.17227v1  

#### Abstract
The rapid rise of Large Language Models (LLMs) has revolutionized various artificial intelligence (AI) applications, from natural language processing to code generation. However, the computational demands of these models, particularly in training and inference, present significant challenges. Tradit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Cloud-native and Distributed Systems for Efficient and Scalable Large Language Models – A Research Agenda*

本文并非一篇以实验为核心的传统研究论文，而是一篇**研究议程（Research Agenda）性质的综述与展望文章**。其核心目标是系统性地梳理当前在支持大规模语言模型（LLMs）的云原生（cloud-native）与分布式系统领域所面临的挑战，并提出一个全面的研究蓝图，而非报告单一的新方法或其实验结果。

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在解决以下核心问题：
- **LLM工作负载的独特性与传统系统的不匹配**：传统的云原生和分布式系统是为微服务、批处理等任务设计的，而LLM在训练和推理时表现出极高的计算密度、内存压力、动态请求模式（如bursty流量）、长上下文依赖和强通信耦合，导致现有系统效率低下。
- **资源管理复杂化**：如何在异构硬件（GPU/NPU/TPU）、多租户环境、云边协同架构下高效调度LLM任务，同时优化延迟、吞吐量、成本、能耗和公平性。
- **缺乏统一标准与可复现性**：当前LLM系统生态碎片化严重，缺乏跨平台的标准接口、部署规范和基准测试，阻碍了技术比较与协作发展。

### 提出了什么新方法或新思路
本论文并未提出具体算法或系统原型，而是提出了一个**系统性的研究框架和未来方向**，其创新点体现在以下几个方面：

| 创新维度 | 具体内容 |
|--------|--------|
| **系统级整合视角** | 首次将 cloud-native 架构（如容器化、Kubernetes、服务网格）与 LLM 特定需求深度结合，倡导“LLM-aware”系统设计。 |
| **多层次挑战分类** | 明确划分出六大类系统挑战：计算、软件系统、运维与资源管理、隐私安全、数据管理和标准化，构建完整分析框架。 |
| **前瞻性趋势引导** | 引入并强调多个新兴方向作为未来突破口，例如：<br>• **Serverless Inference**：实现按需弹性伸缩，降低运营负担；<br>• **Federated & Decentralized Training**：保护数据隐私的同时利用分散算力；<br>• **Quantum & Neuromorphic Accelerators**：探索超越经典计算范式的新型硬件支持；<br>• **AI-Driven Orchestration**：使用 RL 和 LLM 自身来优化资源调度与决策。 |

### 相比现有方法的优势
相比已有文献仅关注某一方面（如仅优化推理延迟或仅讨论联邦学习），本文的优势在于：
- **综合性强**：覆盖从底层硬件到上层API的全栈视角，提供端到端的理解。
- **前瞻性强**：不仅总结现状，更系统性地提出未来5–10年可能的发展路径。
- **指导意义明确**：为学术界和工业界提供了清晰的合作方向与研究优先级排序。

---

## 2. 核心实验方法和设置

由于这是一篇**愿景型论文（Vision Paper）**，它**没有进行原始实验**，也未报告具体的实验配置或数据集。相反，作者通过以下方式支撑论点：

- **文献综述与案例引用**：广泛引用近年来顶级会议（OSDI, SOSP, ASPLOS, MLSys 等）上的代表性工作作为证据，例如：
  - **vLLM**：用于说明 PagedAttention 对 KV Cache 的优化；
  - **DistServe / Splitwise**：展示 Prefill/Decode 分离调度的有效性；
  - **Alpa / Megatron-LM**：体现分布式训练中的混合并行策略；
  - **Hugging Face Transformers / Ollama**：反映开源社区对事实标准的影响。

- **概念性架构图示**：使用多张图表（如 Figure 1–5）直观呈现系统层级、挑战分类与未来方向。

- **定性分析与归纳**：基于对大量已有工作的理解，提炼共性问题、开放挑战与潜在解决方案。

因此，不存在传统意义上的“实验设置”、“数据集”或“基线对比”。

---

## 3. 主要实验结果和性能指标

同样，由于无原始实验，**文中未报告任何新的性能数字或量化结果**。

然而，论文通过对已有研究成果的总结，间接指出了某些技术路线带来的**潜在性能增益**：

| 技术方向 | 引用成果 | 报告的性能提升（来自被引文献） |
|---------|--------|-----------------------------|
| **Prefill/Decode Disaggregation** | DistServe, Splitwise | 可显著改善 TTFT（Time to First Token）和 TPOT（Time Per Output Token） |
| **PagedAttention** | vLLM | 提升 GPU 内存利用率，实现高吞吐共享服务 |
| **Speculative Decoding** | Medusa | 加速自回归生成，在预测准确时减少端到端延迟 |
| **Token-Level Preemption** | FastServe | 减少头阻塞（head-of-line blocking），提高响应性 |
| **Energy-Aware Scheduling** | DynamoLLM, TAPAS | 动态调节 GPU 频率与任务放置，降低能耗而不牺牲 QoS |

这些数据均非本文实测，而是作为论证当前进展的基础。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **LLM 已成为系统研究的新驱动力**：其独特的计算特征正在重塑 cloud-native 与分布式系统的设计原则。
2. **必须转向“LLM-aware”系统设计**：不能再简单套用通用微服务架构，需针对 LLM 的相位异质性（prefill vs decode）、KV Cache 压力、长序列处理等特性进行定制化优化。
3. **弹性、能效与可持续性至关重要**：未来的 LLM 系统必须联合优化性能、成本与碳足迹，尤其在大规模部署场景下。
4. **去中心化与隐私保护是必然趋势**：Federated Learning、Decentralized Inference 等范式将在医疗、金融等领域发挥重要作用。
5. **智能化自治是终极目标**：通过 RL、AI-driven 控制器甚至 LLM 自身来实现自动化的资源编排、故障诊断与策略调优。

### 方法的局限性
- **非实证性**：所有主张均为理论推导与趋势预测，缺乏统一实验验证。
- **广度优先于深度**：涵盖主题广泛，但每个子领域的探讨相对宏观，缺少细节机制剖析。
- **依赖生态系统演进**：许多设想（如量子加速、神经形态计算）仍处于早期阶段，短期内难以落地。

### 未来工作方向
论文在第6节明确提出多项未来研究方向：

| 方向 | 描述 |
|-----|------|
| **LLMs for System Software** | 使用 LLM 来辅助或自动化云资源调度、任务分配与异常检测（即 “LLM for Automatic Cloud Management”）。 |
| **Quantum Computing for LLMs** | 探索量子算法在参数优化、采样等子任务中的应用潜力。 |
| **Prompt Engineering as System Optimization** | 将 prompt 压缩、结构化视为系统级优化手段，减轻 prefill 阶段负担。 |
| **Information Management & Context Engineering** | 构建智能上下文管理系统，动态选择是否使用 RAG 或 long-context 模型。 |
| **Decentralized Intelligence** | 发展去中心化的 LLM 协作网络，支持跨设备、跨组织的知识共享与推理。 |
| **Incentive Models for LLM Training** | 设计激励机制鼓励高质量数据贡献与算力共享（如基于 Shapley Value 的数据估值）。 |
| **Personalized LLM & LLM@Home** | 支持本地化、个性化的模型服务，兼顾性能与隐私。 |

---

## 总结

| 维度 | 内容 |
|-----|------|
| **论文类型** | Vision Paper / Research Agenda |
| **核心贡献** | 提出首个聚焦于 *cloud-native + distributed systems* 支持 LLM 的系统性研究议程，识别六大挑战并规划未来方向。 |
| **关键价值** | 为研究人员、工程师和政策制定者提供了一个共同的语言体系和发展蓝图，推动跨学科合作。 |
| **适用读者** | 系统研究员、AI基础设施工程师、云计算平台开发者、科研项目规划者。 |

> ✅ **一句话总结**：  
> 本文不是一篇报告实验结果的论文，而是一份呼吁行动的“宣言书”，主张我们必须重新思考整个系统栈的设计哲学，以应对LLM带来的根本性变革，并为此绘制了一幅通往高效、可扩展、可持续且可信的LLM基础设施的路线图。

</details>

---

### 3. [SinkRouter: Sink-Aware Routing for Efficient Long-Context Decoding in Large Language and Multimodal Models](https://arxiv.org/abs/2604.16883)

**Authors**: Junnan Liu, Xinyan Liu, Peifeng Gao, Zhaobo Qi, Beichen Zhang, Weigang Zhang, Antoni Bert Chen  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2604.16883v1  

#### Abstract
In long-context decoding for LLMs and LMMs, attention becomes increasingly memory-bound because each decoding step must load a large amount of KV-cache data from GPU memory. Existing acceleration strategies often trade efficiency for accuracy by relying on heuristic pruning that may discard useful i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SinkRouter**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在大语言模型（LLMs）和大视觉多模态模型（LMMs）的**长上下文解码**过程中，每一解码步都需要从 GPU 内存中加载不断增长的 **KV-Cache** 数据进行注意力计算，导致解码过程严重受限于内存带宽（memory-bound），造成显著延迟。

现有加速策略（如 token 剪枝、KV-Cache 压缩）通常以牺牲准确性为代价，且缺乏对“注意力沉降”（attention sink）现象的深入机制理解，往往：
- 不加区分地保留高注意力分数的 token；
- 将早期 token 视为不可替代的锚点；
- 或依赖启发式头路由（head routing）。

这些方法未能有效识别真正冗余的计算。

---

### **提出的新方法与新思路**
本文提出 **SinkRouter**，一种**无需训练**（training-free）、即插即用（plug-and-play）的选择性路由框架，其核心思想是：

> **将 attention sink 现象视为一种可学习的“近零操作”（near-no-op）更新机制，并利用其作为运行时信号来跳过冗余计算。**

#### **关键洞察（Mechanistic Insight）**
- **BOS token 的 value 向量范数极小**（≈0），形成“数值真空”；
- **BOS token 的 key 向量在几何上与其他语义 token 分离**，使其成为一个稳定且易到达的目标；
- 当 query 与 BOS key 高度对齐时，注意力输出接近零，残差流几乎不变 → 可视为一个 **ε-固定点**（ε-fixed point）。

因此，sink-dominant 注意力头本质上是在执行低影响更新，可被安全跳过。

---

### **相比现有方法的优势**
| 维度 | SinkRouter | 传统方法（如 H2O、Scissorhands、StreamingLLM） |
|------|------------|---------------------------------------------|
| **粒度** | **Head-level**（基于 KV Group） | Token-level 或 Cache-level |
| **机制理解** | 基于对 sink 的**机制建模**（低影响更新） | 启发式剪枝或压缩 |
| **是否需训练** | ❌ 无需任何训练 | 多数无需训练 |
| **硬件友好性** | ✅ 自研 **Triton kernel** 支持 block-level branching 和 Split-K 并行 |
| **兼容性** | ✅ 可与 KV-Cache 压缩方法（如 H2O）**正交结合** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 类型 | 数据集 | 说明 |
|------|--------|------|
| **文本长上下文** | `LoNGBENCH`, `INFINITEBENCH` | 覆盖问答、推理、检索等任务 |
| **多模态长上下文** | `MMVP`, `CVBENCH`, `MILEBENCH` | 包含图像理解、视频推理等复杂场景 |

---

### **实验设置与评估指标**
- **模型**：
  - 文本模型：`Llama-3.1-8B`, `Llama-3.1-70B`, `Yi-9B-200K`
  - 多模态模型：`LLaVA-1.5-7B`, `LLaVA-1.5-13B`
- **精度指标**：各基准官方指标（如准确率、F1、AUC 等）
- **效率指标**：
  - 每 token 解码延迟（ms/token）
  - 端到端速度提升倍数（speedup）
- **硬件**：NVIDIA RTX PRO 6000 GPU，bf16 精度
- **预算对齐**：所有方法控制在约 **40% KV-Cache 保留率** 下比较，确保公平

---

### **基线方法对比**
| 类型 | 方法 | 说明 |
|------|------|------|
| **Token/KV 剪枝** | `H2O`, `Scissorhands` | 基于重要性保留 token 或 heavy hitters |
| **Sink 利用** | `StreamingLLM` | 利用 sink token 保持状态连续性 |
| **自适应缓存** | `FastGen` | 动态分配缓存策略 |
| **多模态优化** | `LOOK-M`, `FASTV` | 针对视觉 token 进行压缩或过滤 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在 **512K 上下文长度** 下，SinkRouter 达到 **最高 2.03× 的端到端解码加速**。
- 在 128K 上下文下，通过调节路由阈值，可在保持精度几乎无损的情况下实现显著加速。

---

### **与基线方法的对比结果**

#### **表：LoNGBENCH 上的平均准确率对比（Llama-3.1-8B）**
| 方法 | Avg. Score | Δ vs Full Attention |
|------|-----------|---------------------|
| Full Attention | 50.58 | — |
| H2O | 50.61 | +0.03 |
| FastGen | 45.67 | -4.91 |
| StreamingLLM | 32.82 | -17.76 |
| **SinkRouter (Ours)** | **49.89** | **-0.69** |

> ✅ **SinkRouter 几乎无损精度**，远优于 FastGen 和 StreamingLLM。

#### **表：InfiniteBench 上的表现（Llama-3.1-8B）**
| 方法 | Avg. Score | Δ vs Full Attention |
|------|-----------|---------------------|
| Full Attention | 53.43 | — |
| FastGen | 49.48 | -3.95 |
| **SinkRouter (Ours)** | **53.47** | **+0.04** |

> ✅ 在极端长上下文检索任务中表现稳健，甚至略超全注意力。

#### **多模态结果（LLaVA-1.5-7B & 13B）**
| 方法 | MMVP/CVBench/MileBench Avg. | Δ vs Full Attention |
|------|-------------------------------|---------------------|
| Full Attention | ~57.37 | — |
| LOOK-M | ~52.18 | -5.19 |
| FastV | ~55.35 | -2.02 |
| **SinkRouter (Ours)** | **~57.13** | **~-0.24** |

> ✅ 多模态任务中同样保持高保真，显著优于 LOOK-M。

---

### **消融实验与分析（来自附录）**
- **Layer-wise 分析**：BOS 占主导的 attention head 输出的残差写入量更小，且与整体 attention 更新方向对齐度更低 → 支持“弱贡献”假设。
- **Proxy 可靠性验证**：
  - 使用 `cos(q, k_BOS)` 作为路由代理，在 KV Group 级别上的 AUPRC 达到 **0.773**，表明其具有强预测能力。
- **动态阈值校准**：
  - 固定阈值在不同上下文长度下不稳定；
  - 提出的 **长度自适应阈值函数**（cubic fit）能稳定维持目标跳过比例（如 60%）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Attention Sink 是一种可解释的机制**：
   - 不是架构副产物，而是训练中学得的一种“可控静默更新”策略。
   - BOS token 具备“数值真空”（value norm ≈ 0）和“几何隔离”（key 方向分离）特性。

2. **SinkRouter 实现高效选择性计算**：
   - 在解码前通过轻量级对齐检测（`cos(q, k_0)`）判断是否进入 sink 模式；
   - 若满足条件，则跳过历史 KV-Cache 加载，直接输出零代理（zero surrogate）；
   - 所有决策在 **KV Group 级别**进行，天然契合 GQA 架构。

3. **系统级优化至关重要**：
   - 自研 **Triton kernel** 实现：
     - Inline routing（在 kernel 内部完成路由判断）
     - Block-level branching（避免冗余内存传输）
     - Split-K parallelism（提升活跃组的利用率）
   - 使得算法稀疏性转化为实际的 wall-clock 加速。

4. **性能随上下文增长而增强**：
   - 当 context < 64K 时增益有限（内存压力未凸显）；
   - 当 context > 128K 后，加速效果迅速上升，**在 512K 达到 2.03× speedup**。

---

### **方法的局限性**
- **仅适用于 GQA 架构**：依赖 KV Group 结构进行分组路由，对 MHA 或 MLA 不直接适用。
- **依赖 BOS token 的 sink 特性**：若模型不表现出明显的 sink 行为（如某些修改版 attention），效果可能下降。
- **前两层被排除路由**：实验发现前两层 sink 行为异常，需特殊处理，限制了最大收益。

---

### **未来工作方向**
- 将 sink-aware 思路扩展至 **prefill 阶段**，进一步减少初始计算开销。
- 探索其他类型的 sink（如句首、段落标记）用于更细粒度的上下文管理。
- 结合 **KV-Cache 压缩方法**（如 H2O + SinkRouter），实现双重加速。
- 推广至非 BOS 类 sink 场景，构建通用的“idle head detection”机制。

---

> ✅ **总结一句话**：  
> SinkRouter 从机制层面重新理解 attention sink，将其转化为高效的运行时路由信号，结合硬件感知内核，在几乎无损精度的前提下实现了高达 **2.03× 的长上下文解码加速**，为 LLM/LMM 的高效部署提供了新范式。

</details>

---

### 4. [CoLLM: A Unified Framework for Co-execution of LLMs Federated Fine-tuning and Inference](https://arxiv.org/abs/2604.16400)

**Authors**: Shaoyuan Huang, Xiaokai Wang, Na Yan, Xiaofei Wang, Wenyu Wang, Yansha Deng  
**Category**: cs.DC  
**Published**: 2026-04-21  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2604.16400v1  

#### Abstract
As Large Language Models (LLMs) are increasingly adopted in edge intelligence to power domain-specific applications and personalized services, the quality and efficiency of the LLM post-training phase-including fine-tuning and inference, have become critical due to constrained resources. Although re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CoLLM: A Unified Framework for Co-execution of LLMs Federated Fine-tuning and Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在边缘智能（edge intelligence）场景中，**Large Language Models (LLMs)** 被广泛用于个性化服务和领域特定应用。然而，当前系统通常将 **Federated Parameter-Efficient Fine-tuning (FL PEFT)** 和 **推理（inference）** 视为两个独立任务，导致以下问题：

- **资源争用严重**：fine-tuning 占用大量 GPU 资源，影响实时推理的吞吐量和 SLO（Service-Level Objective）满足；
- **模型更新延迟**：fine-tuning 的改进需等待训练完成并重新部署后才能用于推理，无法及时提升用户体验；
- **冗余部署开销大**：采用时间复用（temporal multiplexing）或空间复用（spatial multiplexing）方式运行两个任务时，重复加载 LLM backbone 导致内存和计算浪费。

### 🚀 提出的新方法与思路
作者提出 **CoLLM** —— 一种统一的 **FL PEFT 与推理共执行框架**，其核心思想是：

> 在共享的模型副本（replica）上同时执行 fine-tuning 和 inference，并通过协调机制实现参数实时共享与负载动态平衡。

#### 主要创新技术：
1. **Intra-replica Model Sharing（副本内模型共享机制）**
   - 采用 **unmerged inference** 范式，保持 adapter 参数独立，避免合并权重破坏可训练性；
   - 引入 **shadow adapter 策略**：维护两组 adapter（`A_act` 和 `A_shd`），推理读取激活版本，fine-tuning 更新影子版本，通过原子交换实现无锁同步，解决并发访问冲突。

2. **Inter-replica Joint Coordination（跨副本联合调度算法）**
   - 设计 **Two-Timescale Coordination Algorithm (TTCA)**：
     - **粗粒度 FL round-level scheduler**：基于 EMA 预测请求趋势，优化每轮 fine-tuning 批大小和推理强度；
     - **细粒度 slot-level dispatcher**：实时调整请求分发，应对短期负载波动，支持过载/欠载自动校正。

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 FedLS、dLoRA） | CoLLM |
|------|-----------------------------|--------|
| 架构设计 | 分离式执行（fine-tune then serve） | 统一共执行（co-execution） |
| 模型更新延迟 | 高（需 redeployment） | 极低（单步延迟内生效） |
| 内存开销 | 高（双份 backbone 或频繁 reload） | 低（单一共享 backbone） |
| 推理质量提升 | 滞后、不连续 | 实时、持续增强 |
| 资源利用率 | 受限于静态分配 | 动态自适应协调 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
使用多个开源指令微调数据集模拟不同 domain-specific 任务，每个副本分配不同子集以体现数据异构性（data heterogeneity）：

| 数据集 | 任务类型 | 样本数 |
|-------|----------|--------|
| ManimCode [28] | 代码生成 | 4.4k |
| CodeAlpaca [29] | 代码生成 | 20k |
| CodeInstruct [30] | 代码生成 | 122k |
| alpaca [31] | 对话 | 50k |
| GPTeacher [32] | 对话 | 89.3k |
| OpenInstruct [33] | 对话 | 499k |

> ⚠️ 每个数据集中 30% 样本作为推理请求重放。

### 🧪 实验设置
- **硬件环境**：
  - 2 台服务器，各配 4×NVIDIA A30 GPUs（24GB），通过 RDMA 连接；
  - 构建含 8 个逻辑副本的联邦边缘集群（federated edge cluster），每 GPU 一个 replica；
  - 1 个作为 server 负责聚合，其余为 client 执行本地 PEFT + 推理。

- **模型选择**：
  - LLaMA3.1-8B 和 Qwen3-4B；
  - 均启用 LoRA 进行参数高效微调。

- **真实流量轨迹**：
  - 使用来自 Azure 的真实 LLM 服务 trace：
    - `Azure-Code`（代码生成）
    - `Azure-Conv`（对话）
  - 四小时请求回放，模拟动态负载。

### 🎯 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput (req/s)** | 成功响应且满足 SLO（<0.8s）的请求数每秒 |
| **Goodput (Q-req/s)** | 吞吐量 × 推理质量：<br> $ \text{goodput} = F \cdot Q(R) $，其中 $ Q(R) = 1 / \text{CE Loss} $ |
| **SLO Violation Rate** | 响应超时请求占比 |
| **Memory Footprint** | GPU 显存占用情况 |

### 🔁 基线方法对比
| 基线 | 类型 | 特点 |
|------|------|------|
| **dLoRA [35]** | 推理专用系统 | 支持动态批处理和迁移 |
| **Shepherd [17]** | 推理专用系统 | SLO 感知调度，优化模型分配 |
| **Vanilla PEFT [36]** | 推理专用系统 | HuggingFace Transformers 默认配置 |
| **FedLS [18]** | 联合训练-推理系统 | 原为 CNN 设计，适配 LoRA 后用于对比 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Fig. 5）

#### ✅ 推理吞吐量（Throughput）
- 在 **LLaMA-8B** 上：
  - CoLLM 达到最高 throughput，比 FedLS 高 **1.5×**；
  - 尽管并发执行 fine-tuning，仍优于或持平纯推理系统（dLoRA, Shepherd）；
- 在 **Qwen3-4B** 上：
  - 各系统差距缩小（因小模型推理快），但 CoLLM 仍保持竞争力。

> 💡 表明 CoLLM 的模型共享有效抑制了 fine-tuning 干扰。

#### ✅ Goodput 性能（质量感知吞吐）
- **LLaMA-8B + Code Generation**：
  - CoLLM 比 dLoRA 高 **2.2×**，比 Shepherd 高 **2.0×**，比 PEFT 高 **3.0×**，比 FedLS 高 **1.4×**；
- **Qwen3-4B**：
  - 所有任务下均高出基线 **1.6× ~ 1.9×**；

> ✅ 表明 CoLLM 不仅维持高吞吐，还能持续提供高质量响应。

### 🔍 消融实验（Ablation Study）

#### （1）模型共享对推理质量的影响（Fig. 7a）
- **Model Sharing** 方案：
  - >96% 请求质量得分 ≥1（逆 CE Loss）；
- **Separated Deployment**：
  - 仅约 40% 达到相同水平；
- **Inference-only**：
  - <10% 达标；

> ✔️ 证明模型共享显著提升了响应质量和一致性。

#### （2）TTCA 算法的有效性（Fig. 7b）
- 对比变体 **CoLLM@fixed**（固定 batch size + 轮询调度）：
  - 最佳固定设置下 goodput 仅为 CoLLM 的 **~50%**；
  - CoLLM 利用 TTCA 自动调节，在动态负载下表现更稳定、高效；

> ✔️ 验证了两阶段协调算法的必要性和优越性。

### 📈 可扩展性测试（Fig. 6）
- 工作负载从 0.5× 增至 3×：
  - CoLLM 展现出近线性的 throughput 和 goodput 增长；
  - FedLS 在高负载（3×）时性能骤降，出现请求堆积和质量退化；
- CoLLM 通过 **reactive correction** 抑制过载，保障系统稳定性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **分离式架构已不再适用于边缘 LLM 服务**：
   - 时间/空间复用带来高昂的部署与切换成本；
   - 无法利用 fine-tuning 的早期高质量增益。

2. **模型共享 + 共执行 是提升边缘 LLM 效率的关键路径**：
   - unmerged inference + shadow adapter 实现安全高效的参数共享；
   - 推理质量可在训练过程中实时提升，消除“冷启动”问题。

3. **两阶段协调机制（TTCA）能有效平衡长期质量与短期效率**：
   - 粗粒度规划 + 细粒度反馈控制，适应动态负载变化；
   - 实现高达 **3× 的 goodput 提升**，远超现有系统。

4. **CoLLM 在多种 LLM 和任务上具有良好的泛化能力**：
   - 在 LLaMA-8B 和 Qwen3-4B 上均取得一致优势；
   - 支持代码生成与对话等多样化应用场景。

### ⚠️ 方法的局限性
- 当前主要针对 **LoRA-based PEFT**，虽声称可推广至 QLoRA、LoHA 等，但未在实验中验证；
- 实验集中在 GPU 集群模拟边缘节点，尚未在真实移动设备（如手机、IoT）上部署；
- 通信开销被简化忽略（假设 adapter 很小），但在大规模分布式场景可能不可忽视。

### 🔮 未来工作方向
- 扩展支持更多类型的 PEFT 方法（如 IA³、AdapterHub）；
- 结合 RLHF 或 user-in-the-loop labeling，形成闭环学习系统；
- 探索在异构硬件（CPU/NPU/TPU）上的轻量化部署方案；
- 引入安全性与隐私保护机制（如差分隐私、加密聚合）以增强 FL 安全性。

---

## ✅ 总结
CoLLM 提出了一种全新的 **LLM 联邦微调与推理共执行范式**，通过 **intra-replica 模型共享** 和 **inter-replica 两阶段协调**，解决了传统系统中存在的资源浪费、更新延迟和效率低下等问题。实验证明其在 **throughput** 和 **goodput** 上全面超越 state-of-the-art 系统，最高可达 **3× 的 goodput 提升**，为构建高效、智能、实时进化的边缘 LLM 服务提供了坚实基础。

</details>

---

### 5. [HieraSparse: Hierarchical Semi-Structured Sparse KV Attention](https://arxiv.org/abs/2604.16864)

**Authors**: Haoxuan Wang, Chen Wang  
**Category**: cs.DC  
**Published**: 2026-04-21  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2604.16864v1  

#### Abstract
The deployment of long-context Large Language Models (LLMs) poses significant challenges due to the intense computational cost of self-attention and the substantial memory overhead of the Key-Value Cache (KV Cache). In this paper, we introduce HieraSparse, a hierarchical KV Cache compression framewo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HieraSparse: Hierarchical Semi-Structured Sparse KV Attention 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在处理长上下文时面临两大瓶颈：
- **计算成本高**：self-attention 的时间复杂度为 $O(n^2)$，随着序列长度增长，attention 占据预填充（prefill）阶段超过 80% 的延迟。
- **内存开销大**：Key-Value Cache（KV Cache）的内存占用随序列长度线性增长。例如，Llama-3.1-8B 在百万级 token 上下文中需超 125 GiB KV Cache 内存，远超现代 GPU 容量。

现有稀疏化方法存在效率瓶颈：
- **非结构化稀疏（Unstructured Sparsity）**：虽能精细控制剪枝位置，但执行框架多采用“load-as-sparse, compute-as-dense”模式，无法降低计算量，且格式转换带来额外开销。
- **粗粒度剪枝（Coarse-grained Pruning）**：如通道或头剪枝，灵活性差，难以实现细粒度的质量-效率权衡。

---

### 提出的新方法与创新思路
作者提出 **HieraSparse**，一个**分层半结构化稀疏 KV Attention 框架**，结合 GPU 稀疏张量核（Sparse Tensor Core）实现高效加速。

#### 核心创新点：
1. ✅ **Hierarchical Block-Based Memory Management**
   - 将 KV Cache 划分为块（block），支持混合存储：部分块保持稠密（dense），部分块压缩为 2:4 半结构化稀疏格式。
   - 引入**块索引映射表**（block index map），正数表示稠密块偏移，负数表示稀疏块偏移，实现灵活调度。

2. ✅ **GPU 加速内核设计（Acceleration Kernels）**
   - 首次将 **N:M semi-structured sparsity** 应用于 KV Cache，并利用 NVIDIA Ampere/Hopper 架构中的 **Sparse Tensor Core** 进行硬件加速。
   - 设计 **Trans-Both 架构**：对两个 GEMM 操作均转置，使 $K$ 和 $V^T$ 成为稀疏操作数，从而兼容标准 KV 剪枝算法并最大化加速潜力。
   - 实现端到端 2× 计算吞吐提升（理论值）。

3. ✅ **零开销在线压缩机制**
   - 开发高效的压缩 kernel，在 prefill 阶段同步完成 KV Cache 的稀疏化，压缩开销仅占 prefill 延迟的 **0.5%**，几乎无感知。

4. ✅ **统一支持 Prefill 与 Decode 阶段**
   - 是首个将半结构化稀疏 KV Cache 压缩扩展至 **prefill 阶段**的工作，而此前类似方法（如 MUSTAFAR）仅限 decode。

---

### 相比现有方法的优势
| 维度 | HieraSparse | MUSTAFAR（SOTA Unstructured） |
|------|-------------|-------------------------------|
| 稀疏类型 | N:M Semi-structured | Unstructured |
| 加速方式 | Sparse Tensor Core（真正降算力） | Load-as-sparse, Compute-as-dense（不降算力） |
| 支持阶段 | Prefill + Decode | 仅 Decode |
| 压缩率 | 更高（metadata 更紧凑） | 较低（bitmap 开销大） |
| 实际速度提升 | 最高达 4.57× | 几乎无加速甚至变慢 |

> 🔥 **核心优势**：成功将“稀疏性”转化为“实际效率”，解决了“有稀疏但无加速”的根本矛盾。

---

## 2. 核心实验方法和设置

### 数据集
- **生成质量评估**：使用 **LongBench**，一个多任务、双语、长上下文理解基准，涵盖：
  - 单/多文档问答
  - 摘要
  - 少样本学习
  - 合成任务
  - 代码补全
- 测试模型：
  - Llama-3.1-8B-Instruct
  - Mistral-7B-Instruct-v0.2
  - Qwen3-8B

### 实验设置
- **硬件平台**：NVIDIA L40S GPU（48 GiB 显存）
- **软件环境**：PyTorch 2.10.0, CUDA 12.8
- **稀疏配置**：
  - 默认使用 magnitude-based 剪枝策略。
  - 固定前 64 个 “sink” tokens 和最后 256 个 “local window” tokens 为稠密。
  - 其余区域按 block-level 和 element-level 分层剪枝。
- **块大小**：64 tokens
- **稀疏粒度**：2:4 N:M 结构化稀疏

### 评估指标
| 类别 | 指标 |
|------|------|
| **质量保留** | LongBench 平均得分（越高越好） |
| **计算效率** | Attention kernel 延迟、Speedup（vs Dense） |
| **内存效率** | KV Cache 压缩率（Compression Ratio） |
| **端到端性能** | TTFT（Time to First Token）、TPOT（Time Per Output Token） |

### 基线方法对比
- **Dense Baseline**：原始 full attention（FlashAttention）
- **MUSTAFAR**：当前最先进的 fine-grained unstructured KV Cache pruning 方法
  - 同样控制相同稀疏比例进行公平比较

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 场景 | 性能表现 |
|------|--------|
| **KV Cache 压缩率** | 达到 **1.2×** 高于 MUSTAFAR（同稀疏度下） |
| **Attention 速度提升（Decode）** | 相比 MUSTAFAR 提升 **4.57×** |
| **Prefill 阶段最大加速** | 最高达 **1.85×**（当 key 和 value 均剪枝） |
| **端到端加速（无显著质量下降）** | Prefill 加速 **1.37×**，Decode 加速 **1.77×** |
| **压缩开销** | 仅占 prefill 延迟 **0.5%**，可忽略不计 |

---

### 与基线方法对比（Table III & IV）

#### 在 50% Value Sparsity 下（Decode Only）：
| 方法 | 模型 | LongBench Score | Attention Speedup | Compression Ratio |
|------|------|------------------|--------------------|-------------------|
| Dense | Llama-3.1-8B | 49.96 | 1.0× | 1.0× |
| MUSTAFAR | Llama-3.1-8B | 49.94 | 0.32× | 1.5× |
| **HieraSparse** | Llama-3.1-8B | **49.90** | **1.28×** | **1.8×** |

> 💡 尽管 MUSTAFAR 能压缩更多内存，但由于其 compute-as-dense 特性，实际 attention 反而更慢！

#### 不同 sparsity 设置下的综合表现（Table IV）：
| Prefill Sparsity | Decode Sparsity | Prefill Speedup | Decode Speedup | Avg Score Drop |
|------------------|----------------|------------------|----------------|---------------|
| $S_k=0.0, S_v=1.0$ | $S_k=0.0, S_v=1.0$ | 1.34× | 1.28× | <0.5 pt |
| $S_k=0.0, S_v=1.0$ | $S_k=1.0, S_v=1.0$ | 1.34× | **1.71×** | ~1.7 pt |

> ✅ 表明可通过差异化设置 prefill/decode 稀疏度来灵活调节速度-质量平衡。

---

### 消融实验结果（Ablation Study）

#### （1）优化技术对 Prefill Kernel 的影响（Figure 4）
引入以下三项关键技术后，prefill kernel 性能得到显著提升：
- **Async Pipelining**（异步流水线）：隐藏内存加载延迟
- **In-Fragment Re-layout**（寄存器内重排布）：避免 shared memory/warp shuffle 开销
- **On-chip Memory Specialization**（专用 kernel）：减少共享内存占用，提高 occupancy

> ➕ 组合使用后达到最高 1.85× 加速。

#### （2）不同稀疏粒度的影响（Figure 8a）
- Decode kernel 的实测加速曲线紧贴理论上限。
- Prefill kernel 在低稀疏度下略超理论值（得益于缓存命中率提升），在高稀疏度下趋于平缓（受限于 shared memory 占用）。

#### （3）Key vs Value 缓存剪枝敏感性分析（Figure 6b）
- **Key Cache 更敏感**：因其数值幅度更大，剪枝导致更大的 magnitude loss。
- **Softmax 放大误差**：query-key 点积进入 softmax，微小扰动会被指数放大。
- 推荐策略：**prefill 阶段保持 key cache 稠密，优先剪枝 value cache**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **稀疏 ≠ 快**：传统 unstructured 方法因“compute-as-dense”机制，无法将稀疏性转化为实际加速，甚至拖慢推理。
2. ✅ **结构化稀疏 + 硬件协同是出路**：利用 GPU 的 **Sparse Tensor Core** 是实现高效稀疏 attention 的关键路径。
3. ✅ **分层设计提供灵活性**：通过 block-level + element-level 双层控制，可在质量与效率间自由调节。
4. ✅ **prefill 阶段也可安全剪枝**：首次证明在 prefill 中应用 semi-structured pruning 是可行且高效的。
5. ✅ **zero-overhead online compression 可实现**：压缩过程可在 prefill 中无缝集成，不影响首 token 延迟。

---

### 方法的局限性
1. **依赖特定硬件**：必须使用支持 N:M sparse tensor core 的 GPU（如 NVIDIA Ampere/Hopper 或 AMD MI300X）。
2. **magnitude-based 剪枝仍有改进空间**：当前剪枝策略较简单，未充分挖掘离线优化潜力。
3. **对某些模型更敏感**：如 Mistral-7B 对结构化剪枝更敏感，需调参以维持质量。
4. **未结合量化或其他压缩技术**：潜力尚未完全释放。

---

### 未来工作方向
1. **探索更高级的离线剪枝策略**：结合训练或校准信息，进一步提升 prefill 阶段的加速潜力。
2. **支持 unstructured-to-structured mapping**：借鉴 TASDER、VENOM 等工作，动态将非结构化稀疏映射到结构化加速器上。
3. **融合 Quantization + Pruning**：联合优化 KV Cache 的数值精度与稀疏性，适用于资源受限设备。
4. **适配 Prefix Caching 场景**：在 RAG、Agent 系统中复用压缩后的 KV Cache，进一步降低延迟。

---

> 📌 **总结一句话**：  
> **HieraSparse 是首个成功将半结构化稀疏 KV Cache 与 GPU 稀疏张量核深度融合的系统，实现了从“稀疏”到“高效”的跨越，在 prefill 和 decode 阶段均取得显著加速，同时保持生成质量稳定。**

</details>

---

### 6. [AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation](https://arxiv.org/abs/2604.16625)

**Authors**: Weihua Du, Jingming Zhuo, Yixin Dong, Andre Wang He, Weiwei Sun, Zeyu Zheng, Manupa Karunaratne, Ivan Fox, Tim Dettmers, Tianqi Chen, Yiming Yang, Sean Welleck  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2604.16625v1  

#### Abstract
Recent large language model (LLM) agents have shown promise in using execution feedback for test-time adaptation. However, robust self-improvement remains far from solved: most approaches still treat each problem instance independently, without accumulating reusable knowledge. This limitation is par...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Models (LLMs)** 的代码生成在 **低资源、领域特定语言（如 Triton）** 上表现不佳，尤其是在 **GPU kernel 代码生成与优化** 场景中面临两大挑战：

- **可行性瓶颈（Feasibility Bottleneck）**：由于语法、内存访问或并行化错误，大量生成的 kernel 编译失败或运行出错，导致无效样本比例高。
- **局部最优陷阱（Locality Bottleneck）**：性能优化空间高度非线性且组合复杂，仅靠局部微调难以跳出局部最优。

此外，大多数现有方法缺乏跨任务的知识积累机制，无法实现持续的自我改进。

---

### 提出的新方法：AdaExplore

提出 **AdaExplore**，一个两阶段的 LLM 代理框架，用于高效生成高性能 kernel 代码：

#### （1）Adapt: 失败驱动的适应（Failure-Driven Adaptation）
- 在合成的任务上运行 agent，收集执行失败反馈。
- 将重复出现的失败模式提炼为可复用的 **cross-task skill memory**（跨任务技能记忆），即一组简洁的约束规则（如“不能在 Triton kernel 中调用 `tl.float32()`”）。
- 这些规则作为系统提示注入后续生成过程，显著提升生成正确性。

#### （2）Explore: 多样性保持搜索（Diversity-Preserving Search）
- 构建一棵候选 kernel 的搜索树（而非单一链式迭代），支持多路径探索。
- 定义两种动作：
  - **Small Step**：局部补丁式修改，用于精细调优。
  - **Large Step**：结构性重建，打破当前设计限制，探索新策略。
- 使用 **UCT-style 节点选择策略** 平衡探索与利用，并通过 **代表内核池（representative kernel pool）** 保留历史优秀解以指导搜索。

---

### 相比现有方法的优势
| 方法 | 局限性 | AdaExplore 的优势 |
|------|--------|------------------|
| 单次生成（Single-pass） | 正确率低，性能差 | 显著提高正确性和速度提升 |
| 并行采样（Parallel Sampling） | 依赖大量采样，效率低 | 更高效的结构化搜索 |
| 迭代精炼（Iterative Refinement） | 易陷入局部最优 | 支持大步跳跃，突破局部限制 |
| OpenEvolve 等进化方法 | 缺乏失败知识沉淀 | 引入 failure-driven memory 提升泛化能力 |

> ✅ **核心优势**：无需额外训练或外部知识，仅通过执行反馈即可实现 **自适应 + 自主探索**，兼具 **高正确率** 和 **强优化潜力**。

---

## 2. 核心实验方法和设置

### 数据集
- **KernelBench (Ouyang et al., 2025)**：主要测试平台，包含真实收集的 GPU kernel 优化任务，按难度分为三级：
  - Level-1：单算子（用于训练任务合成）
  - Level-2：简单融合 kernel（评估重点之一）
  - Level-3：模型级 workload（如 ResNet、LSTM 组件）
- **FlashInfer-Bench (Xing et al., 2026)**：来自生产级 LLM 推理流水线的真实 kernel，输入形状取自部署模型（如 Llama-3.1-8B），提供专家编写的 CUDA 基线。

---

### 实验设置
- **硬件环境**：NVIDIA A6000 GPU（固定频率 1500MHz），部分实验在 B200 上验证迁移性。
- **基础模型**：默认使用 **GPT-5-mini**，也测试了 GPT-5、Claude-4.6-Opus、Qwen3-Coder-Next。
- **预算控制**：多数实验设定 **50 步、100 步、200 步** 的 test-time 搜索预算。

---

### 评估指标
| 指标 | 含义 |
|------|------|
| **Acc.** | 在预算内至少生成一个功能正确的 kernel 的比例 |
| **Speedup** | 最佳生成 kernel 相对于 PyTorch eager 实现的运行时加速比（上限截断为 10×） |
| **Fast@1.2 / Fast@2** | 成功生成速度超过基准 1.2× 或 2× 的正确 kernel 的比例 |

> ⚠️ 特别说明：作者对 Speedup 进行了 **outlier trimming**（去除最快最慢各 5%），提升了结果稳定性。

---

### 基线方法对比
| 类型 | 方法 |
|------|------|
| 单次生成 | GPT-5-mini, GPT-5, Claude-4.6-Opus, AutoTriton |
| 多轮优化 | Parallel Sampling (PS), Iterative Refinement (IR), OpenEvolve, DR.Kernel |
| 变体增强 | PS w. SM, IR w. SM（加入 cross-task skill memory） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（KernelBench）

| 方法 | KernelBench L2 Speedup | KernelBench L3 Speedup | Acc. (L2/L3) |
|------|------------------------|------------------------|-------------|
| GPT-5-mini (single-pass) | 0.34× | 0.21× | 22%/22% |
| IR w. SM (best baseline) | 2.59× | 1.31× | 100%/100% |
| **AdaExplore (50 steps)** | **2.65×** | **1.55×** | **100%/100%** |
| **AdaExplore (100 steps)** | **3.12×** | **1.72×** | **100%/100%** |
| **AdaExplore (200 steps)** | **3.41×** | **1.78×** | **100%/100%** |

> 🔥 **结论**：AdaExplore 在所有层级均达到最高 Speedup，且随计算预算增加持续提升，无饱和迹象。

---

### 与其他方法对比结果
- 在 Level-2 上，AdaExplore 达到 **3.12× 加速**，优于第二名 IR w. SM（2.59×）。
- 在更难的 Level-3 上，仍取得 **1.72× 加速**，远超其他方法（如 OpenEvolve w. SM 为 1.47×）。
- **Fast@2 指标** 上，AdaExplore (100 steps) 在 L2 达到 **44%**，表明其能稳定产出显著优于基线的 kernel。

---

### 消融实验结果（Ablation Study）

| 变体 | Speedup (L2) | Fast@1.2 |
|------|--------------|----------|
| **AdaExplore (Full)** | **2.65×** | **71%** |
| w/o MCTS（改为链式搜索） | 2.48× | 64% |
| w/o Large Step | 2.62× | 71% |
| w/o Small Step | 2.35× | 60% |
| w/o Skill Memory | 2.32× | 56% |
| w/o Representative Kernel Pool | 2.30× | 63% |

> 📌 **发现**：
> - **Tree Search（MCTS）** 对多样性至关重要，移除后性能下降明显。
> - **Large Step** 虽然单看影响不大，但在长程搜索中是突破的关键。
> - **Cross-task Skill Memory** 是提升正确性的核心，移除后 Acc. 下降至 99%，Speedup 明显降低。
> - **Representative Kernel Pool** 有助于维持长期进展信号。

---

## 4. 关键结论和发现

### 主要发现
1. **失败中学习有效**：kernel 生成失败集中在少数几类可复用的约束上，通过 failure-driven memory 可系统性避免常见错误。
2. **多样性搜索优于局部精炼**：尤其在复杂任务（Level-3）中，单纯迭代 refinement 难以突破设计瓶颈，必须引入结构性再生。
3. **AdaExplore 具备良好扩展性**：随着 test-time compute 增加，性能持续上升，优于所有基线。
4. **跨模型与跨 GPU 泛化能力强**：
   - Skill Memory 可迁移到 GPT-5、Qwen3-Coder、Claude 等不同模型，Pass@1 正确率从 22% 提升至 54%。
   - 在 A100、L40S、GB200 等不同架构 GPU 上均能取得 2.13×–3.07× 加速。

---

### 方法的局限性
1. **对极端优化 kernel 难以超越专家**：
   - 如 GEMM kernel 仅达 cuBLAS 的 42%，因底层已高度工程化。
   - 黑胶指令（如 Blackwell QMMA）、TMA 加速等需专业知识，当前 agent 难以掌握。
2. **依赖高质量合成任务**：虽然使用 mutation-based prompting 生成训练任务，但仍受限于初始种子质量。
3. **上下文长度限制**：尽管采用双记忆机制（working memory + kernel pool），但在极长搜索轨迹中仍有信息丢失风险。

---

### 未来工作方向
1. **引入专家知识蒸馏**：将专家 kernel 或 vendor library 中的最佳实践编码为 agent 可理解的形式。
2. **支持多硬件平台联合优化**：构建统一的 cross-architecture skill memory。
3. **结合强化学习进行端到端训练**：将 AdaExplore 的搜索策略纳入 RL 框架，进一步提升效率。
4. **扩展至其他 DSL**：应用于 Halide、CUDA、SYCL 等其他高性能编程语言。

---

> ✅ **总结一句话**：  
> **AdaExplore 通过“从失败中学规则” + “多样树状搜索”，实现了无需微调的高效 kernel 自动生成，在 KernelBench 上达到 3.12× 加速，是迈向自主代码优化的重要一步。**

GitHub 开源地址：[https://github.com/StigLidu/AdaExplore](https://github.com/StigLidu/AdaExplore)

</details>

---

### 7. [Cross-Family Speculative Decoding for Polish Language Models on Apple~Silicon: An Empirical Evaluation of Bielik~11B with UAG-Extended MLX-LM](https://arxiv.org/abs/2604.16368)

**Authors**: Krzysztof Fonal  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.16368v1  

#### Abstract
Speculative decoding accelerates LLM inference by using a small draft model to propose k candidate tokens for a target model to verify. While effective for same-tokenizer pairs on high-bandwidth GPUs, its applicability to cross-family pairs with mismatched tokenizers and consumer-grade unified memor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Cross-Family Speculative Decoding for Polish Language Models on Apple Silicon

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究解决了在 **Apple Silicon** 平台上进行 **跨家族（cross-family）speculative decoding** 的两大挑战：
1. **跨 tokenizer 不兼容问题**：主流的 speculative decoding 要求 draft 和 target 模型共享 tokenizer，但波兰语模型家族 **Bielik** 内部（如 Bielik 11B 和 Bielik 1.5B）因采用不同架构（Mistral vs. Qwen2.5）和 tokenizer（Mistral vs. APT4），无法直接配对。
2. **Apple Silicon 统一内存架构（unified memory）下的性能瓶颈**：现有 speculative decoding 的理论加速假设（验证 k 个候选 token 成本 ≈ 生成 1 个 token）基于高带宽 GPU，但在 Apple Silicon 的低带宽统一内存系统中是否成立尚不明确。

### 提出的新方法和创新
- **实现了 UAG（Universal Assisted Generation）在 MLX-LM 中的完整支持**：
  - 首次将 UAG 的 **string-level token translation** 机制移植到 Apple 的 **MLX** 推理框架。
  - 实现了两种 token translation 策略：
    - **Naive translation**：简单字符串 round-trip。
    - **Context-aware translation**：引入一个可配置的前缀窗口（`p=5`），在 re-tokenization 时加入上下文以缓解边界对齐问题。
- **提出了硬件感知的参数化加速公式**：
  - 基于实测数据推导出适用于 Apple Silicon 的 speculative decoding 速度提升模型，解释了为何标准理论在该硬件上失效。

### 相比现有方法的优势
- **填补技术空白**：首次为 Apple Silicon 用户提供跨 tokenizer speculative decoding 能力，使他们能利用异构模型组合（如通用小模型辅助专用大模型）。
- **更贴近现实的性能建模**：提出的硬件感知公式比传统理论更能准确预测在统一内存架构上的实际表现，为部署决策提供了可靠依据。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个波兰语数据集上进行，覆盖不同类型的内容：
1. **Polish Wikipedia**：结构化、重复性强的文本（如列表、信息框）。
2. **pl_alpaca**：指令跟随任务，内容多样，变异性高。
3. **Synthetic short questions**：人工生成的短问答，模拟对话场景。

### 实验设置
- **目标模型（Target）**：`Bielik 11B-Instruct`（Mistral 架构，Mistral tokenizer，8-bit 量化）。
- **草案模型（Drafters）**：
  - `Bielik 1.5B`（Qwen2.5 架构，APT4 Polish tokenizer）
  - `Qwen2.5-1.5B`（通用模型）
  - `Llama 3.2-1B`（通用模型）
- **硬件平台**：Apple M2 Pro（32GB unified memory，~200 GB/s 带宽）。
- **draft 长度 k**：`{2, 4}`，并在 pl_alpaca 上进行了 `k=6` 的确认性实验。
- **每条件样本数**：每个配置下测试 50 个 prompt。

### 评估指标
- **Token acceptance rate (α)**：目标模型接受的 draft token 比例。
- **Tokens per second (TPS)**：端到端生成吞吐量。
- **Speedup**：speculative decoding 的 TPS 相对于自回归基线的加速比。

### 基线方法对比
- **Baseline**：仅使用目标模型的自回归解码（autoregressive decoding）。
- **对比条件**：对每个 drafter，比较以下三种情况：
  1. **No translation**：直接传递 draft token（无效，用于基准）。
  2. **Naive translation**：无上下文的字符串转换。
  3. **Context-aware translation**：带前缀上下文的转换。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### Token Acceptance Rate (k=2)
| Drafter | Condition | Wikipedia | pl_alpaca | Synthetic |
|--------|----------|-----------|-----------|----------|
| **Bielik-1.5B** | Context-aware | 31.1% | 23.9% | 22.2% |
| **Llama-3.2-1B** | Context-aware | 42.0% | 36.0% | 36.5% |
| **Qwen2.5-1.5B** | Context-aware | **44.6%** | **41.0%** | **42.7%** |

> ✅ **Context-aware translation 在所有 9 种组合中均取得最高 acceptance rate**。

#### Speedup (k=2, Wikipedia, Context-aware)
| Drafter | Speedup |
|--------|---------|
| Bielik-1.5B | 0.87× |
| **Llama-3.2-1B** | **1.06×** |
| **Qwen2.5-1.5B** | **1.06×** |

> ✅ **通用模型（Qwen/Llama）作为 drafter 反而优于专用的 Bielik-1.5B**。

#### Draft Length 影响 (k=4 vs k=2)
- 所有 drafter 在 `k=4` 时的 **speedup 均显著下降**，即使 acceptance rate 提升。
- 例如，Qwen2.5-1.5B 在 Wikipedia 上：
  - `k=2`：44.6% acceptance → 1.06× speedup
  - `k=4`：53.8% acceptance → **0.70× speedup**

### 与基线方法的对比结果
- **Context-aware translation 显著优于其他策略**：
  - Naive translation 因边界错位导致 acceptance 率极低（如 Qwen 从 44.6% 降至 10.5%），**吞吐量仅为基线的 58–70%**。
  - No translation 在 Wikipedia 上表现尚可（因 ASCII 共享 token 多），但在其他数据集上迅速恶化。
- **通用模型优于专用模型**：
  - 尽管 Bielik-1.5B 是波兰语专用模型，其 acceptance 率仍低于 Qwen/Llama，归因于 APT4 tokenizer 与 Mistral tokenizer 的严重不匹配。

### 消融实验结果
- **Context-aware translation 的有效性**：通过对比 `naive` 与 `context-aware`，证明了上下文对缓解 token 边界错位至关重要。
- **k 值的影响**：`k=6` 实验显示，即使 acceptance 达到 45%，speedup 也仅 **0.40×**，证实 `k>4` 在当前硬件上完全不可行。
- **break-even 分析**：
  - `k=2` 时，break-even acceptance 率约为 **38–53%**。
  - `k=4` 时，break-even 率飙升至 **77–92%+**，远超实际可达水平。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Context-aware translation 是必要前提**：在跨 tokenizer speculative decoding 中，必须使用上下文感知的 token 对齐策略，否则性能会劣于基线。
2. ❌ **专用小模型不一定更优**：尽管 Bielik-1.5B 专为波兰语设计，但因 tokenizer 不匹配，其表现不如通用的 Qwen/Llama 模型。
3. ⚠️ **Apple Silicon 上的 speculative decoding 效果高度依赖内容**：
   - 在结构化、重复性高的文本（如 Wikipedia）上，可实现 **1.7×** 加速。
   - 在多变的指令跟随任务中，难以超越自回归解码。
4. 🔧 **验证成本未按预期摊销**：由于 draft 和 target 模型推理均为 memory-bandwidth-bound，且需顺序执行，`k` 次 draft 推理的成本过高，导致 `k>2` 时加速比反而下降。

### 方法的局限性
- **硬件限制**：当前 Apple Silicon 的内存带宽（~200 GB/s）远低于数据中心 GPU（如 A100 的 1555 GB/s），是 speculative decoding 难以发挥优势的根本原因。
- **token translation 开销**：context-aware translation 引入了额外的 re-tokenization 和 cache 同步开销，且存在“context absorption”等边界问题。
- **TLI 无法应用**：Token-Level Intersection (TLI) 因 Bielik 与通用 tokenizer 的交集过小而不可行。

### 未来工作方向
- **改进 token translation 机制**：
  - 动态调整 prefix 窗口大小以避免“吸收”问题。
  - 实现局部 cache rewinding 以降低重计算开销。
- **实现 TLI 支持**：探索在 MLX-LM 中集成 TLI，尤其适用于 tokenizer 交集较大的语言对。
- **适配更高带宽硬件**：在 M3 Ultra / M4 Max 等设备上验证 speculative decoding 是否更有效。
- **扩展模型家族评估**：随着 Bielik 家族新增 Nemotron 架构模型，可进一步测试跨架构 speculative decoding 的表现。
- **探索 adaptive draft length**：根据内容动态选择 `k` 值，以最大化吞吐量。

</details>

---

### 8. [Towards Intelligent Legal Document Analysis: CNN-Driven Classification of Case Law Texts](https://arxiv.org/abs/2604.17674)

**Authors**: Moinul Hossain, Sourav Rabi Das, Zikrul Shariar Ayon, Sadia Afrin Promi, Ahnaf Atef Choudhury, Shakila Rahman, Jia Uddin  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.17674v1  

#### Abstract
Legal practitioners and judicial institutions face an ever-growing volume of case-law documents characterised by formalised language, lengthy sentence structures, and highly specialised terminology, making manual triage both time-consuming and error-prone. This work presents a lightweight yet high-a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Intelligent Legal Document Analysis: CNN-Driven Classification of Case Law Texts

## 1. 论文的主要贡献和创新点

### 解决了什么问题
法律从业者和司法机构面临日益增长的判例文书（case law）数量，这些文本具有正式语言、长句结构和高度专业化的术语，导致人工分类耗时且易出错。现有的主流方法（如 fine-tuned BERT）虽然性能较强，但依赖大量计算资源，推理延迟高，难以部署于实际场景。

本文旨在解决**高效、准确、轻量级的法律文本分类**问题，特别是在“引用处理”（citation treatment）任务上实现高性能的同时降低模型复杂度。

### 提出了什么新方法或新思路
提出了一种**轻量级但高精度的深度学习框架**，其核心由三个关键组件构成：
- **基于 Lemmatisation 的预处理流程**：通过词形还原（lemmatisation）减少形态变异带来的噪声。
- **子词感知的 FastText 嵌入**：利用 FastText 的 subword 能力有效处理罕见法律术语、拉丁语短语和未登录词（OOV）。
- **多核一维卷积神经网络（Multi-kernel 1D-CNN）**：并行使用不同卷积核大小（2, 3, 5）捕捉从双词组到长子句的多层次法律模式。

该架构被称为 **CNN + FastText + Lemmatisation**，是一种无需依赖大规模 Transformer 的替代方案。

### 相比现有方法的优势
- **更高效率**：仅含 5.1 million 参数，推理延迟低至 **0.31 ms / 文档**，比 BERT 快 **13 倍以上**。
- **更强准确性**：在多个指标上超越包括 fine-tuned BERT 在内的多种基线模型。
- **更低资源消耗**：训练更快（epoch 时间 26.8 秒 vs BERT 的 215.6 秒），GPU 内存占用更少（1.3 GB vs 5.8 GB）。
- **良好的泛化性和鲁棒性**：对输入扰动（如嵌入层加噪）表现出较强抵抗力。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 数据集名称：**Legal Citation Text Classification Dataset** [18]
- 来源：Kaggle 公开数据集（由 Bansal 编译）
- 规模：约 **25,000 篇标注判例文书**
- 字段：
  - `Case ID`：唯一标识符
  - `Case Outcome`：引用类别标签（如 cited, referred to, distinguished, overruled, positive, neutral, negative）
  - `Case Title` 和 `Case Text`：案件标题与全文意见
- 划分方式：**75% 训练集，25% 测试集**，采用分层抽样保持类别平衡

### 实验设置和评估指标
- **编程环境**：Python 3.10，PyTorch 2.1，spaCy 进行 lemmatisation，FastText 初始化嵌入
- **硬件平台**：NVIDIA RTX 3060 GPU（12GB VRAM），Intel i7-12700F CPU，Ubuntu 22.04
- **超参数配置**：
  - 嵌入维度：500
  - 卷积核尺寸：{2, 3, 5}
  - 每个核滤波器数：128
  - 激活函数：ReLU
  - 池化策略：Global Max Pooling
  - Dropout rate：0.4
  - 优化器：Adam（学习率 1e-3）
  - Batch size：32，最大训练 50 轮，早停机制（3 轮无提升即停止）

### 评估指标
- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- Macro F1-score（宏平均 F1 分数）
- AUC-ROC（受试者工作特征曲线下面积）
- 推理延迟（Inference Latency）
- 模型参数量（Params）、GPU 内存占用、训练 epoch 时间

### 基线方法对比
| 模型 | 类型说明 |
|------|--------|
| **KNN (TF-IDF)** | 传统可解释模型，缺乏上下文与子词感知能力 |
| **CNN (Random Embedding)** | 能检测局部模式但无法利用语义词几何 |
| **LSTM + FastText** | 序列建模能力强，支持长距离依赖，但训练慢 |
| **BERT (Base, fine-tuned)** | 强大的上下文表示模型，但计算开销大 |
| **Legal-BERT [19]** 和 **BERT-CNN [17]** | 已有文献中报道的先进模型，用于横向比较 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | AUC (%) |
|-------|--------------|----------------|------------|---------|----------|
| KNN (TF-IDF) | 89.42 | 88.95 | 89.42 | 88.60 | 90.10 |
| CNN (Random) | 93.51 | 93.20 | 93.51 | 92.88 | 94.21 |
| LSTM + FastText | 95.68 | 95.40 | 95.68 | 95.10 | 96.02 |
| BERT (fine-tuned) | 97.12 | 97.05 | 97.12 | 96.88 | 97.45 |
| **Proposed (Ours)** | **97.26** | **97.28** | **97.26** | **96.82** | **97.83** |

> ✅ 所有指标均优于所有基线模型，首次在轻量模型中达到甚至超过 BERT 级别的性能。

### 与基线方法的对比结果
- **准确率**：比 BERT 高 **0.14%**，显著优于其他轻量模型（+3.75% vs LSTM）
- **AUC-ROC**：达到 **97.83%**，为所有模型最高，表明分类判别力最强
- **推理速度**：**0.31 ms / document**，是 BERT（4.25 ms）的 **13.7 倍快**
- **参数量**：仅 **5.1M**，远低于 BERT 的 110M（约为其 **4.6%**）
- **训练效率**：每 epoch 仅需 **26.8 秒**，而 BERT 需 **215.6 秒**

### 消融实验结果（Ablation Study）
研究了不同卷积核组合的影响（固定其余参数）：

| Kernel Sizes | Accuracy (%) |
|-------------|---------------|
| [3]         | 95.8          |
| [3, 4]      | 96.4          |
| **[2, 3, 5]**（提出） | **97.26**     |

> 结果验证了多尺度卷积的重要性：
> - kernel=2：捕获短引用线索（如 “cited”、“affirmed”）
> - kernel=3：识别中等长度法律短语
> - kernel=5：编码更长的子句上下文
>
> 多核组合提供了非冗余特征，显著提升性能。

此外，混淆矩阵分析显示错误主要集中在语义相近类别之间（如 Positive ↔ Neutral，Distinguished ↔ Overruled），这反映了法律语言本身的模糊性，而非模型缺陷。

---

## 4. 关键结论和发现

### 主要发现
1. **精心设计的 CNN 架构可在法律文本分类任务中媲美甚至超越 Transformer 模型**，尤其在效率方面优势明显。
2. **Lemmatisation + FastText 的组合能有效应对法律文本中的形态多样性与 OOV 问题**，提升语义一致性。
3. **多核 1D-CNN 可并行提取多粒度法律模式**，形成丰富且判别性强的文档表示。
4. 模型具备良好正则化效果，训练/验证曲线贴合紧密，**无明显过拟合现象**。
5. 在嵌入层加入高斯噪声后仍保持 **95.9% 准确率**（仅下降 1.3%），证明其**鲁棒性强**。

### 方法的局限性
- 当前框架基于**英文法律语料**，尚未验证在多语言或多司法管辖区（multijurisdictional）场景下的适应性。
- 尽管性能优异，但在**极长文档或需要全局上下文推理的任务**中可能不如 Transformer。
- 模型本身**可解释性有限**，虽优于黑箱 LLM，但仍需结合 XAI 技术增强透明度。

### 未来工作方向
1. **融合轻量级 Transformer 层**以增强长距离依赖建模能力。
2. **扩展至多领域、多语言法律文本处理**，构建跨法域通用框架。
3. **引入可解释 AI（Explainable AI）技术**，帮助法律从业者理解预测依据。
4. 探索 **Retrieval-Augmented Generation（RAG）与数据增强技术**，提升低资源场景下的表现。
5. 将本框架应用于其他法律 NLP 任务，如 Legal Judgment Prediction（LJP）、holdings extraction 等。

---

> **总结**：本文提出的 **CNN + FastText + Lemmatisation** 框架为智能法律文档分析提供了一个**高效、精准、可部署**的新范式，在性能不妥协的前提下大幅降低了资源需求，是迈向下一代 AI 法律系统的务实一步。

</details>

---

### 9. [Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](https://arxiv.org/abs/2604.17769)

**Authors**: Yuan Fang, Yiming Luo, Aimin Zhou, Fei Tan  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.17769v1  

#### Abstract
Ensuring the safety of large language models (LLMs) requires robust red teaming, yet the systematic synthesis of high-quality toxic data remains under-explored. We propose Reverse Constitutional AI (R-CAI), a framework for automated and controllable adversarial data generation that moves beyond isol...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大语言模型（LLMs）的安全性评估依赖于“红队测试”（red teaming），即寻找能绕过安全机制的对抗性输入（如 jailbreak prompts）。然而，现有方法存在以下局限：
- **孤立性**：主要关注单个有害提示的发现，缺乏系统性和可扩展性。
- **低质量数据**：手动或自动化生成的有毒数据往往结构松散、重复性强，难以覆盖多样化的失败模式。
- **奖励欺骗**（reward hacking）：在强化学习优化过程中，模型可能通过生成语义不连贯但高毒性得分的内容来“作弊”，导致生成的数据实用性下降。

该研究旨在解决**高质量、可控且多维度的有毒数据合成**这一关键挑战，以支持更全面、系统的安全性评估。

### 提出的新方法与新思路
作者提出了 **Reverse Constitutional AI (R-CAI)** 框架，其核心思想是将用于对齐人类价值观的 *Constitutional AI* 范式进行“逆向”应用。

- **核心方法**：
  1. **反向宪法构建**（Constitution of Toxicity）：定义一个由四类有害行为构成的“毒性宪法”，作为优化目标。这四类包括：
     - **法律与伦理**（Legal & Ethical）
     - **社会偏见**（Social Bias）
     - **行为后果**（Behavioral Consequence）
     - **信任与欺骗**（Trust & Deception）
  2. **自举式批判-修订循环**（Self-Bootstrapped Critique-Revision Pipeline）：利用一个基础模型（base model）作为批判者（critic）和修订者（reviser），在毒性宪法的指导下，迭代地对初始响应进行批评和修改，从而逐步放大其有害性并提升结构完整性。
  3. **概率钳制的 RLAIF**（Probability-Clamped RLAIF）：为了解决奖励欺骗问题，在强化学习从 AI 反馈中学习（RLAIF）的过程中引入**概率钳制**（probability clamping）机制，限制奖励模型对偏好对的置信度，从而稳定训练过程。

### 相比现有方法的优势
| 方面 | R-CAI | 传统方法（如手动标注、Prompt Search） |
| :--- | :--- | :--- |
| **自动化程度** | 完全自动，无需人工干预 | 依赖大量人工或简单模板 |
| **数据质量** | 生成的有毒数据**多样性高、逻辑连贯性强** | 数据往往重复、浅层、关键词堆砌 |
| **控制性** | 可控地生成特定类别和强度的有害内容 | 难以精确控制输出的类型和深度 |
| **系统性** | 将红队测试重构为**系统性的对抗数据合成**问题 | 主要聚焦于孤立的攻击实例 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **基础数据集**：一个包含 **30,000 个有害诱导提示**（harm-inducing prompts）的精选红队测试数据集。
- **合成数据**：通过 R-CAI 框架的批判-修订循环，从上述提示生成大规模、结构化的有毒响应对（prompt-response pairs）。

### 实验设置和评估指标
所有实验均基于 **Llama3-8B** 作为基础策略模型（base policy）和奖励模型（reward model），并使用 **Llama3-70B** 作为 AI 评委（AI judge）进行自动评分。

#### 评估指标
使用 Llama3-70B 作为评委，对生成的响应进行自动化打分，共三个维度：
- **毒性分数 (STox)**：衡量响应在四个毒性维度上的平均表现（1-5 分）。
- **连贯性分数 (Scoh)**：衡量响应的逻辑流畅性和语义一致性（1-5 分）。
- **多样性分数 (SDiv)**：通过计算响应嵌入（Sentence-BERT）的平均成对余弦相似度来衡量。
- **综合效用分数 (STox-Utility)**：加权分数，`STox-Utility = α * STox + (1-α) * Scoh`，其中 α=0.7，强调毒性的同时兼顾连贯性。

### 基线方法对比
实验对比了四种变体：
1. **Base Model**：未经微调的原始 Llama3-8B 模型。
2. **SFT Model**：在 R-CAI 批判-修订循环生成的数据上进行监督微调（Supervised Fine-Tuning）的模型。
3. **R-CAI (w/o clamping)**：使用标准 RLAIF 进行强化学习，**未使用概率钳制**的模型。
4. **R-CAI (Ours)**：提出的完整框架，包含**概率钳制**的 RLAIF 模型。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **毒性对齐效果显著**：
  - R-CAI (Ours) 在几乎所有宪法维度上都实现了最强的恶意对齐。例如，在“法律与伦理”维度，其 `STox` 达到 **3.28**，相比 Base Model 提升了 **77.3%**。
- **连贯性大幅提升**：
  - R-CAI (w/o clamping) 虽然毒性高，但因奖励欺骗导致连贯性下降（`Scoh=2.82`）。
  - 引入概率钳制后，R-CAI (Ours) 的连贯性分数显著提升至 **3.24**，**提高了约 15%**。
- **综合性能最优**：
  - R-CAI (Ours) 的综合效用分数（`STox-Utility`）达到 **3.00**，远超 w/o clamping 版本的 **2.81**。
- **多样性增强**：
  - 在温度参数 `temp=0.8` 下，R-CAI (Ours) 的多样性分数（`SDiv`）为 **5.46**，相比无钳制版本提升了 **42.5%**，表明其有效防止了模式崩溃（mode collapse）。

### 消融实验结果
- **概率钳制边界的影响**：
  - 实验对比了不同的钳制边界 `[emin, emax]`，如 `[0.2, 0.8]`, `[0.3, 0.7]`, `[0.4, 0.6]`。
  - 结果显示，边界越紧（尤其是 `[0.4, 0.6]`），对连贯性和多样性的提升越明显。与无钳制基线相比，`[0.4, 0.6]` 设置使连贯性提升 **14.9%**，多样性提升 **42.6%**。
- **批判-修订轮次的影响**：
  - 实验发现，经过 **3 轮**修订时，连贯性达到峰值（sweet spot）。
  - 第 **4 轮**修订反而导致连贯性下降，因为模型过度专注于最大化毒性关键词而牺牲了因果推理，验证了全局过滤机制的必要性。

---

## 4. 关键结论和发现

### 主要发现
1. **R-CAI 是一个有效的框架**：成功地将红队测试从“寻找攻击”转变为“系统性合成对抗数据”的过程，能够全自动、可控地生成高质量的有毒数据。
2. **概率钳制至关重要**：该机制能有效缓解 RLAIF 中的奖励欺骗问题，通过“压平”奖励景观（flatten the reward landscape），在保持高毒性的同时，显著提升生成内容的**连贯性**和**多样性**。
3. **对齐的脆弱性**：实验结果表明，现有的安全对齐（如 RLHF）并未真正消除模型的有害能力，而是将其“抑制”在潜空间中。R-CAI 能够作为一种“潜在能力提取器”（latent capability extractor），揭示这些被掩盖的风险。

### 方法的局限性
- **依赖 AI 评委**：生成的奖励信号质量受限于 AI 评委的能力，可能存在偏见或盲点。
- **静态稳定设计**：概率钳制的边界是固定的，动态或自适应的调度策略可能带来更好的权衡。
- **模型覆盖有限**：实验仅在 Llama3 系列模型上进行，其泛化性有待在更大规模或不同架构的模型上验证。
- **下游安全评估缺失**：未系统评估 R-CAI 生成的数据对下游安全训练的实际影响。

### 未来工作方向
- 探索**自适应的概率钳制策略**，根据训练阶段动态调整边界。
- 将 R-CAI 框架应用于**良性领域**（如科学推理、法律分析），通过“正向宪法”来激发模型的潜在高性能。
- 开展**标准化的下游安全评估**，量化 R-CAI 数据对提升模型鲁棒性的实际贡献。
- 研究批判-修订循环的**收敛性**和最优停止准则。

</details>

---

### 10. [Sampling for Quality: Training-Free Reward-Guided LLM Decoding via Sequential Monte Carlo](https://arxiv.org/abs/2604.16453)

**Authors**: Jelena Markovic-Voronov, Wenhui Zhu, Bo Long, Zhipeng Wang, Suyash Gupta, Kayhan Behdin, Bee-Chung Chen, Deepak Agarwal  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.16453v1  

#### Abstract
We introduce a principled probabilistic framework for reward-guided decoding in large language models, addressing the limitations of standard decoding methods that optimize token-level likelihood rather than sequence-level quality. Our method defines a reward-augmented target distribution over compl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Sampling for Quality: Training-Free Reward-Guided LLM Decoding via Sequential Monte Carlo*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 LLM 解码方法（如 beam search、nucleus sampling）主要优化**token-level likelihood**，而非**sequence-level 质量**。这导致生成的文本在流畅性上表现良好，但在需要全局一致性的任务（如代码生成、数学推理）中可能失败。尽管已有工作引入 reward signals（如 verifier、process reward model），但这些信号通常以启发式方式使用（如 best-of-N、re-ranking），并未真正融入生成过程。

本文旨在解决以下问题：
- 如何将 reward signals **原则性地**整合到 LLM 解码过程中？
- 如何在**不进行额外训练**的前提下，提升序列级生成质量？

---

### 🚀 提出的新方法与创新思路

提出一种**基于 Sequential Monte Carlo (SMC)** 的 reward-guided 解码框架，核心思想如下：

#### （1）定义 Reward-Augmented Sequence Distribution
构建一个目标分布：
$$
\pi(x_{1:T}|q) \propto \prod_t m_t(x_t|q,x_{<t}) \cdot \prod_t p_t(x_{1:t}, q)
$$
其中：
- $ m_t $：来自 base LLM 的 transition factor
- $ p_t $：prefix-dependent reward potential（如 verifier 分数）
该分布将语言模型的生成能力与外部 reward 信号结合，形成一个新的采样目标。

> 这是一个 **training-free** 方法：**不修改模型权重**，仅通过 inference-time sampling 改变输出分布。

#### （2）统一多种解码策略
证明了以下常见方法是该框架的特例：
- Temperature sampling → 对应 *Tempered Base* 目标
- Power-tempered sampling → 对应 *Powered Base* 目标

从而实现对多种 decoding objective 的统一建模。

#### （3）设计两种 Intermediate Target 用于 SMC 采样
- **Prefix-only Target**：计算高效，但中间分布不匹配真实边际。
- **Lookahead Target**：引入 future-value correction term $ L_t $，使中间分布**精确等于完整路径分布的边际**，理论上最优。

> Lookahead term 可通过 Monte Carlo rollouts 在线估计，无需额外训练。

#### （4）Resample-Move SMC + MH Rejuvenation
采用 block-wise SMC 框架，并集成：
- **Selective MH rejuvenation**：仅对 resampling 后重复的低 reward 粒子执行 Metropolis-Hastings 更新
- 利用 lookahead term 在 acceptance ratio 中评估未来延续质量，避免“弱换弱”

这种组合实现了**高效探索 + 精准修正**的双重优势。

---

### 🔍 相比现有方法的优势

| 方面 | 本方法优势 |
|------|-----------|
| **是否需训练** | ❌ 不需要 —— 完全 training-free |
| **reward 整合方式** | ✅ 原生嵌入采样分布，非后处理 |
| **理论基础** | ✅ 基于 Feynman-Kac / SMC 理论，有 principled 推导 |
| **scaling behavior** | ✅ 性能随 compute 单调提升，无 plateau |
| **适用任务** | ✅ 支持 code generation、math reasoning、scientific QA 等 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个具有挑战性的推理基准上评估：
- **HumanEval**：Python 代码生成，评估 pass@1（功能正确性）
- **MATH500**：高中数学题，exact match 准确率
- **GPQA Diamond Split**：科学领域多选问答，multiple-choice accuracy

---

### ⚙️ 实验设置

| 组件 | 设置说明 |
|------|--------|
| **Base Models** | Qwen2.5-7B, Qwen2.5-Math-7B, DeepSeek-Math-7B |
| **最大长度** | T = 3072 tokens，遇到 EOS 提前终止 |
| **实现平台** | 基于 vLLM 构建统一 inference-time sampling 框架 |
| **Block Size (B)** | HumanEval: 64；MATH/GPQA: 512（适应不同推理粒度） |
| **Rollout Temp (Troll)** | HumanEval: 0.1；MATH/GPQA: 0.3 |
| **其他参数** | α=4.0, N=16 particles, J=2 lookahead samples, S=2 MH steps |

---

### 🆚 基线方法对比

| 类型 | 基线方法 |
|------|--------|
| 基础解码 | Base (temp=1), Low-temperature (temp=1/α) |
| 多样本策略 | Best-of-N |
| MCMC 方法 | MCMC Power Sampling (Karan & Du, 2025) |
| SMC 方法 | Scalable Power Sampling (Ji et al., 2026), Power-SMC (Azizi et al., 2026) |
| Reward-guided SMC | SMC (reward) [Lew et al., 2023]（无 lookahead/MH） |
| 强化学习方法 | GRPO（可用时作为上限参考） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Pass@1）

| Model / Benchmark | MATH500 | HumanEval | GPQA |
|------------------|--------|----------|------|
| **Qwen2.5-7B** | | | |
| Base | 0.498 | 0.329 | 0.278 |
| Best-of-N | 0.650 | 0.609 | 0.282 |
| SMC (reward) | 0.710 | 0.787 | 0.323 |
| **Ours (SMC reward-guided lookahead)** | **0.790** | **0.878** | **0.384** |
| GRPO | 0.740 | 0.561 | 0.354 |
| | | | |
| **Qwen2.5-Math-7B** | | | |
| Base | 0.496 | 0.329 | 0.278 |
| SMC (reward) | 0.756 | 0.750 | 0.384 |
| **Ours** | **0.782** | **0.854** | **0.424** |
| GRPO | 0.785 | 0.537 | 0.399 |
| | | | |
| **DeepSeek-Math-7B** | | | |
| Base | 0.362 | 0.415 | 0.333 |
| SMC (reward) | 0.516 | 0.628 | 0.388 |
| **Ours** | **0.604** | **0.781** | **0.424** |
| GRPO | 0.492 | 0.524 | 0.333 |

---

### 📈 与基线方法的对比结果

- 在 **HumanEval** 上：
  - 相比 base model 最高提升 **+54.9% absolute**
  - 超越最强 sampling baseline（Scalable Power Sampling）**+9.1% ~ +15.3%**
  - 达到 **87.8% pass@1**（Qwen2.5-7B），超过 GRPO（56.1%）

- 在 **MATH500** 上：
  - 最高达到 **79.0%**（Qwen2.5-7B）
  - 超过 GRPO 和所有其他 sampling 方法

- 在 **GPQA** 上：
  - 最高达 **42.4%**，显著优于 baseline

> ✅ **在所有三个 7B 模型上均取得 SOTA 表现**

---

### 🔬 消融实验结果（Ablation Study）

#### ▶️ Lookahead Mechanism 的影响
比较完整方法 vs. SMC (reward)（即 prefix-only + 无 MH）：

| Model | HumanEval Δ↑ | MATH500 Δ↑ |
|-------|---------------|------------|
| Qwen2.5-7B | +9.1% | +8.0% |
| Qwen2.5-Math-7B | +10.4% | +2.6% |
| DeepSeek-Math-7B | **+15.3%** | **+8.8%** |

> 结论：**Lookahead 显著防止 myopic sampling**，尤其在长程推理任务中效果更明显。

#### ▶️ Token Budget vs. Accuracy（图1）
- Best-of-N 和 SMC (reward) 在 N=16 后迅速饱和（plateau at ~0.73）
- 本文方法持续提升，在 ~144K tokens/problem 时达 **0.94 pass@1**
- 表明性能增益**并非来自更多 token 生成**，而是更高效的 compute allocation

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Reward signals 应被 principled 地纳入生成分布**，而不仅是 post-hoc ranking。
2. **Lookahead intermediate targets 匹配真实 marginal 分布**，在理论上和实践中都优于 prefix-only 方法。
3. **SMC + MH rejuvenation 是有效的 resample-move 架构**，特别适合 sharply-peaked 或 reward-modified 分布。
4. **Training-free 方法可超越 RL 微调方法（如 GRPO）**，说明当前 base model 潜力尚未被充分挖掘。
5. **性能随 compute 单调增长**，为 test-time compute scaling 提供新路径。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **计算开销较高** | 需要 multiple rollouts 和 MH 步骤，延迟高于 greedy decoding |
| **依赖高质量 reward model** | 若 reward signal 噪声大或稀疏，性能会下降 |
| **超参敏感性** | block size、α、J、Troll 等需针对任务调整 |
| **难以部署实时系统** | 更适合 offline high-stakes generation（如代码补全、考试答题） |

---

### 🔮 未来工作方向

1. **Adaptive Compute Allocation**：
   - 动态分配 lookahead rollout budget，基于粒子不确定性
   - 清晰决策 early stop，模糊情况增加 compute

2. **Efficient Lookahead Estimation**：
   - 使用轻量模型近似 $ L_t $
   - 引入 learned twist functions（但保持 training-free）

3. **Integration with Agent Systems**：
   - 将本方法作为 reasoning agent 的核心 generator
   - 支持 multi-step planning 与 self-correction

4. **Data Generation for RL**：
   - 利用高质量 samples 训练更强的 reward model 或 policy

---

> 💡 **一句话总结**：  
> 本文提出了首个将 reward-guided generation 建立在 **principled probabilistic framework** 上的 training-free 解码方法，通过 **Sequential Monte Carlo + Lookahead + MH rejuvenation** 实现了在 code、math、science 多项任务上的显著突破，展示了 **test-time compute scaling** 的巨大潜力。

</details>

---

### 11. [Multi-Label Phase Diagram Prediction in Complex Alloys via Physics-Informed Graph Attention Networks](https://arxiv.org/abs/2604.16468)

**Authors**: Eunjeong Park, Amrita Basak  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.16468v1  

#### Abstract
Accurate phase equilibria are foundational to alloy design because they encode the underlying thermodynamics governing stability, transformations, and processing windows. However, while the CALculation of Phase Diagrams (CALPHAD) provides a rigorous thermodynamic framework, exploring multicomponent ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Multi-Label Phase Diagram Prediction in Complex Alloys via Physics-Informed Graph Attention Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
- **多组分合金相图预测**是材料设计中的核心任务，传统CALPHAD方法虽精确但计算成本高，难以在高维成分-温度空间中进行快速扫描。
- 现有机器学习模型通常仅预测**相数**或**特定边界**（如液相线），而忽略了对**完整共存相集合（phase set）** 的显式建模，导致无法支持下游冶金分析（如tie-line推断）。
- 数据驱动模型容易产生**物理上不可行的相组合**（如违反Gibbs相律），尤其在相界附近概率分布模糊时。

### 🚀 提出的新方法与创新思路
- 首次将**multi-label phase-set prediction**框架应用于复杂合金系统，直接预测每个状态点下所有可能共存的相。
- 构建了一个**基于元素的图注意力网络（element-aware Graph Attention Network, GATv2）**：
  - 将每个合金成分-温度点表示为一个四节点完全连接图（Ag, Bi, Cu, Sn）。
  - 节点特征融合原子分数（atomic fraction）与Magpie描述符（electronegativity, atomic radius等）。
- 引入**双阶段物理一致性保障机制**：
  1. **Physics-Informed Loss**：训练阶段加入三项轻量级热力学约束作为损失项（避免梯度冲突，逐个施加）。
  2. **Physics-Informed Decoding**：推理阶段通过确定性投影强制输出满足物理可行性。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **任务设定** | 从“相数分类”升级到“相集预测”，更贴近实际应用需求 |
| **模型表达能力** | GATv2动态注意力机制能捕捉上下文相关的元素交互作用 |
| **物理一致性** | 显式嵌入热力学先验，显著减少非物理解（如三相出现在二元系中） |
| **泛化能力** | 在未见的ternary/quaternary系统上仍保持高准确率 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **来源**：使用`pycalphad`结合NIST Solder Thermodynamic Database生成约 **25,000个平衡态数据点**。
- **覆盖范围**：
  - 所有6个二元子系统（Ag-Bi, Ag-Cu, ..., Cu-Sn）
  - 3个三元子系统（Ag-Bi-Cu, Ag-Cu-Sn, Bi-Cu-Sn）在700°C下的等温截面
  - 测试集包含两个**未见过的系统**：
    - 三元 Ag-Bi-Sn（extrapolation）
    - 四元 Ag-Bi-Cu-Sn（quaternary surface at 700°C）
- **标签构建**：根据pycalphad输出的相分数 > $10^{-6}$ 判定某相存在，形成multi-label向量（共9类相）。

### ⚙️ 实验设置
- **输入表示**：
  - 图结构：4-node fully-connected directed graph
  - 节点特征：$[x_e; \text{Magpie}_e]$，共9维（1维原子分数 + 8维属性）
  - 全局输入：温度 $T$
- **模型架构**：
  - 3层 GATv2（每层4头注意力）
  - Global Mean Pooling 得到图嵌入
  - 拼接温度后送入MLP输出9个phase logits
- **优化策略**：
  - 使用 **AdamW + Cosine Annealing LR Schedule**
  - 采用 **Class-Balanced Focal Loss** 处理类别不平衡
  - 使用 **Optuna** 进行超参数自动搜索（learning rate, dropout, hidden dim等）

### 📏 评估指标
| 指标 | 定义与用途 |
|------|-----------|
| **Macro-F1 Score** | 各相F1的无偏平均，强调稀有相的表现 |
| **Exact-Set Match Accuracy** | 预测相集与真实相集完全一致的比例（关键指标） |
| **Per-system breakdown** | 分别报告各binary/ternary系统的性能 |
| **Seed Ensembling** | 对10次独立训练的结果取概率平均以提升鲁棒性 |

### 🔁 基线方法对比
本文主要比较以下三种设置：
1. **Baseline GNN**：标准GATv2模型，无任何物理约束
2. **GNN + Physics-Informed Loss**：分别添加Gibbs Phase Rule / Local Smoothness / Pure Phase Feasibility 损失项
3. **GNN + Physics-Informed Decoding**：推理时依次执行：
   - Corner phase cleanup
   - Local smoothing (Gaussian neighbor averaging)
   - Gibbs phase rule cap (最多允许C个相共存)

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自Table 2 & Table 4）

| 模型 | Overall Macro-F1 | Exact-Set Accuracy (in-domain avg.) |
|------|------------------|-------------------------------|
| Baseline GNN | 0.9513 ± 0.0141 | 93.98% |
| + Physics-Informed Loss | 0.9577 ± 0.0049 | ~94.60% |
| + Physics-Informed Decoding | **0.9623 ± 0.0035** | **~96%** |

> 物理感知解码不仅提升了均值，还大幅降低了方差，说明其增强了预测稳定性。

#### 外推性能（Out-of-Domain Generalization）
| 系统 | 类型 | Exact-Set Accuracy |
|------|------|--------------------|
| Ag-Bi-Sn | ternary (700°C) | **99.32%** |
| Ag-Bi-Cu-Sn | quaternary (700°C) | **91.78%** |

> 即使在从未训练过的四元体系中，模型仍能达到超过90%的完全匹配精度，显示极强泛化能力。

### 🔍 与基线方法的对比结果
- **Physics-Informed Decoding > Physics-Informed Loss**
  - 解码方式在所有系统中均优于损失函数方式
  - 尤其在边界密集区域（如Bi-Sn, Cu-Sn）改善明显
- **Gibbs Phase Rule Loss 最有效**
  - 在消融实验中，$\lambda_{\text{GPR}} = 0.15$ 时达到最优Macro-F1（0.9382）
  - 局部平滑和纯相损失效果较弱且不稳定

### 🔧 消融实验结果
- **物理约束顺序至关重要**：
  - 正确顺序：**Pure Phase → Local Smoothness → Gibbs Cap**
  - 若颠倒顺序会导致无效修正甚至破坏合理预测
- **Operator组合效应**：
  - 单独使用任一约束均有提升
  - 三者联合使用（在解码阶段）可协同消除几乎所有非物理解
- **可视化验证**（Figures 7–12）：
  - Baseline GNN 存在大量三相预测于二元系内部或角点处（违反Gibbs规则）
  - Physics-Informed Loss 减少但未根除此类错误
  - Physics-Informed Decoding **彻底消除**所有违反相律的情况

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Multi-label phase-set prediction 是可行且必要的**：
   - 可准确建模多个相共存关系，支持后续微结构分析。
2. **GATv2 + Element Graph 是有效的表示学习范式**：
   - 动态注意力机制能够学习复杂的跨元素相互作用。
3. **物理一致性必须显式建模**：
   - 单靠数据拟合不足以保证热力学合理性。
4. **推理阶段的物理投影优于训练期软约束**：
   - **Physics-Informed Decoding** 提供硬保障，确保输出始终物理可行。
   - 分离训练与推理约束可避免梯度冲突，提高稳定性。

### ⚠️ 方法的局限性
- 当前模型依赖预定义的**9个目标相**，不能发现新相。
- 图结构固定为4节点，扩展至更高组分需重新设计拓扑。
- 物理约束目前为启发式实现，尚未端到端可微分。
- 训练数据来源于CALPHAD模拟，若数据库本身有误会影响模型表现。

### 🔮 未来工作方向
1. **引入更丰富的热力学描述符**：
   - 如相特异性Gibbs自由能、活度系数、lever rule约束等。
2. **发展可微化解码器（Differentiable Projection Layer）**：
   - 结合OptNet思想，实现端到端物理一致性优化。
3. **扩展至更大元素空间与开放数据库集成**：
   - 利用Matminer等工具聚合多源CALPHAD数据。
4. **联合预测相存在与相分数（fraction regression）**：
   - 实现完整的相图代理模型（surrogate model）。

---

> 💡 **总结一句话**：  
> 该研究提出了一种**物理感知的图注意力网络**，通过**multi-label phase-set预测 + 推理时物理投影**，实现了**高精度、高可靠性、强泛化能力**的多组分合金相图快速映射，在保持机器学习效率的同时，严格遵守热力学基本定律。

</details>

---

### 12. [River-LLM: Large Language Model Seamless Exit Based on KV Share](https://arxiv.org/abs/2604.18396)

**Authors**: Yingtao Shen, An Zou  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.18396v1  

#### Abstract
Large Language Models (LLMs) have demonstrated exceptional performance across diverse domains but are increasingly constrained by high inference latency. Early Exit has emerged as a promising solution to accelerate inference by dynamically bypassing redundant layers. However, in decoder-only archite...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# River-LLM: Large Language Model Seamless Exit Based on KV Share 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题  
论文针对 **decoder-only LLM** 中 **Early Exit** 技术在实际推理中难以实现显著加速的根本瓶颈——**KV Cache Absence 问题**。

- 在传统的 token-level Early Exit 中，当某个 token 在较浅层提前退出时，其跳过的深层无法生成对应的 Key-Value (KV) 缓存。
- 后续 token 在自回归生成过程中依赖这些历史 KV 缓存进行 self-attention，缺失会导致计算不完整或精度严重下降。
- 现有解决方案如 **KV Recompute、State Propagation、KV Masking** 等，要么引入高延迟开销，要么导致显著精度损失，无法实现“理论层数减少”到“实际 wall-clock 加速”的转化。

### 🚀 提出的新方法：River-LLM  
提出一种 **无需训练（training-free）** 的框架 **River-LLM**，实现真正意义上的 **seamless token-level Early Exit**。

#### 核心创新点：
1. **KV-Shared Exit Layer（KV共享出口层）**
   - 设计轻量级的“Exit River”，每个 exit layer 与主干 decoder 层一一对应，继承其架构并采用 **4-bit weight-only quantization (W4A16)** 进行压缩。
   - 所有 exit layer 共享同一套 KV Cache 地址空间，确保跳过层的 KV 信息被自然填充，保持 **Intrinsic KV Integrity**。
   - 利用部分图编译优化推理内核，使 exit layer 吞吐提升 **2.4×** 于原始 full-precision 主干块。

2. **基于状态转移相似性的退出决策机制**
   - 发现早期 decoder 层的输入输出之间的 **state transition similarity**（余弦相似度）可预测后续累积量化误差。
   - 利用该指标动态判断是否满足退出条件，避免因过早退出导致语义漂移。

3. **Backbone Offloading 内存优化策略**
   - 大多数 token 在浅层退出 → 深层主干模块可从 VRAM 卸载。
   - 显著降低内存占用，接近全模型量化水平，同时保留对复杂 token 的高精度处理能力。

### 🔍 相比现有方法的优势

| 特性 | River-LLM | 传统方法（如 EE-LLM, D-LLM, SkipDecode） |
|------|-----------|----------------------------------------|
| **Granular Freedom** | ✅ 支持任意 token 在任意层独立退出 | ❌ 受限于单调递减约束或批处理限制 |
| **Intrinsic KV Integrity** | ✅ 自然生成完整 KV 缓存，无恢复开销 | ❌ 需要 recompute/masking/propagation，带来额外延迟或精度损失 |
| **无需训练** | ✅ 完全 post-training setup，零微调成本 | ❌ 多数需训练 exit predictor 或 exit layer |
| **精度保持** | ✅ 几乎无损（near-lossless），甚至某些任务反超 | ❌ 普遍存在明显精度下降 |
| **部署效率** | ✅ 内存更低，延迟开销极小（~0.0688%） | ❌ 内存膨胀，调度复杂 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

涵盖多种推理与生成任务，分为两类：

#### （1）常识推理类（Zero/Few-shot Accuracy）
- **BoolQ**, **HellaSwag**, **ARC-Challenge/Easy**, **MMLU**

#### （2）长序列生成类（用于测速）
- **GSM8K**（数学推理）
- **MATH**（复杂数学问题）
- **HumanEval**（代码生成）

> 所有任务均以 **autoregressive generation** 模式测试，重点评估 wall-clock 推理速度与生成质量平衡。

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|---------|
| **主干模型** | Llama3.2 1B (16层), Llama3.1 8B (32层), Phi4-mini, Ministral3 8B |
| **硬件平台** | NVIDIA A40 GPU |
| **批次大小** | Batch size = 1（生成阶段） |
| **评估协议** | <br>- GSM8K: 5-shot<br>- MATH: 4-shot<br>- 其他: 0-shot |
| **量化方案** | Exit River 使用 **HQQ/W4A16**；对比基线使用相同量化框架 |
| **退出阈值 T** | 默认 T=0.5（Llama系列），Phi/Ministral 使用更高 T=0.8~0.9 模拟近无损场景 |

---

### 🆚 基线方法对比

| 方法 | 类型 | 是否需训练 | KV 处理方式 |
|------|------|------------|-------------|
| **Baseline** | Full Precision | – | 无 Early Exit |
| **Full Quantization** | 静态量化（4-bit） | 否 | 无 Early Exit，仅压缩权重 |
| **Balcony** | Sequence-level Exit | 是（fine-tune exit layer） | 固定退出深度 |
| **EE-LLM** | Token-level Exit | 是 | Batching Recompute |
| **D-LLM** | Token-level Exit | 是 | KV Masking |
| **SkipDecode** | Token-level Exit | 是 | Mono-Decreasing Constraint |
| **River-LLM (Ours)** | Token-level Exit | **否** | **KV Share（无缝填充）** |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 3）

| 模型 | 任务 | 方法 | Accuracy | Tokens/s | Speedup |
|------|------|--------|----------|-----------|---------|
| Llama3.2 1B | GSM8K | Backbone | 33.2 | 84.5 | 1.00× |
| | | River-LLM | **29.3** | **182.9** | **2.16×** |
| | MATH | River-LLM | 14.6 | 189.2 | 1.88× |
| | HumanEval | River-LLM | 23.2 | 171.8 | **1.71×** |
| Llama3.1 8B | GSM8K | River-LLM | 74.4 | 45.0 | **1.78×** |
| | MATH | River-LLM | 26.6 | 43.1 | 1.72× |
| | HumanEval | River-LLM | 55.5 | 44.7 | **1.77×** |
| Phi4-mini | GSM8K | River-LLM | 81.0 | 115.0 | 1.61× |
| Ministral3 8B | GSM8K | River-LLM | 84.3 | 61.8 | 1.77× |

> ✅ **总体实现 1.71× ~ 2.16× 的实际 wall-clock 速度提升**，且在 T=0.7 下可达几乎无损精度。

---

### 🔁 与基线方法对比结果

#### （1）速度 vs 精度权衡（Fig. 8）
- River-LLM 在 **GSM8K** 上定义了更优的 **Pareto Frontier**。
- 相比 masking/recompute/propagation 方法，在同等精度下提速更高，不存在“悬崖式”精度崩塌。

#### （2）内存消耗对比（Fig. 9 & Table 6）
| 方法 | Seq Len=64K 时峰值显存（Llama3.1 8B） |
|------|-------------------------------|
| Backbone | 22.96 GB |
| Balcony | 26.90 GB |
| EE-LLM | 26.47 GB |
| Full Quant (4-bit) | 12.47 GB |
| **River-LLM** | **14.73 GB** ✅ |

> River-LLM 显存远低于其他 Early Exit 方法，接近全量化模型，得益于 **Backbone Offloading** 和 **单套 KV Cache 共享机制**。

#### （3）延迟开销分析
- **Exit Decision 开销**：约 **100 微秒 / token**（Llama3.1 8B），占总推理时间 **0.0688%**，可忽略不计。

---

### 🔍 消融实验与补充发现（Appendix）

#### A.1 不同量化后端的影响（Table 4）
| Exit River Quant | GSM8K Score | Throughput |
|------------------|-------------|------------|
| HQQ (default) | 74.4 | 45.0 |
| **AWQ** | **77.3** ↑ | 46.9 |

> 表明 River-LLM 可作为路径级优化，**增强先进量化方法的表现**，形成互补。

#### A.2 新架构更强早期理解能力（Table 5）
- Phi4-mini 和 Ministral3 8B 更容易在浅层收敛，平均退出位置更靠前（如 HumanEval 平均仅执行 2.5 层）。
- 支持更高的退出阈值仍能保持高精度，验证 **KV-Shared 机制的架构通用性（architecture-agnostic）**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **KV Cache Absence 是阻碍 Early Exit 实际加速的核心障碍**，现有方法无法兼顾效率与完整性。
2. **River-LLM 成功实现了“seamless exit”**：
   - 实现真正的 **token-level 自由退出**
   - 通过 **KV-Shared Exit River** 自然维持 KV 完整性
   - **无需任何训练**，部署简单高效
3. 在多个主流 LLM 上实现 **1.71× ~ 2.16× 的 wall-clock 加速**，同时保持接近原模型的生成质量，部分任务甚至略有提升（如 HumanEval）。
4. **定义了新的 accuracy-speed Pareto frontier**，优于现有所有 Early Exit 与静态量化方法。

---

### ⚠️ 局限性（Limitations）

1. **当前评估集中在 ≤8B 参数模型**，尚未在更大规模（如 24B/70B）上验证 scalability。
2. **加速效果主要体现在 generation phase**，对于 prefill 占主导的任务（如 MMLU）增益有限。
3. 当前 prefill 阶段仍采用 sequence-level exit，未来可探索更细粒度控制。

---

### 🔮 未来工作方向

1. 将 River-LLM 扩展至 **multi-modal models** 与 **encoder-decoder 架构**。
2. 探索 **prefill 阶段的动态 exit 策略**，进一步提升端到端推理效率。
3. 结合 **mixed-precision quantization** 与 **dynamic depth**，构建统一的高效推理栈。
4. 在真实服务场景中集成 River-LLM，研究其在 **batch serving、streaming output** 中的实际表现。

---

## 总结

> **River-LLM 提出了一种简洁而强大的训练免费 Early Exit 框架，通过 KV-Shared Exit River 解决了长期困扰 decoder-only LLM 的 KV Cache Absence 问题。它不仅实现了高达 2.16× 的实际推理加速，而且在精度、内存、部署成本等方面全面超越现有方法，为高效大模型推理提供了新的范式。**

</details>

---

### 13. [Global Attention with Linear Complexity for Exascale Generative Data Assimilation in Earth System Prediction](https://arxiv.org/abs/2604.16590)

**Authors**: Xiao Wang, Zezhong Zhang, Isaac Lyngaas, Hong-Jun Yoon, Jong-Youl Choi, Siming Liang, Janet Wang, Hristo G. Chipilski, Ashwin M. Aji, Feng Bao, Peter Jan van Leeuwen, Dan Lu, Guannan Zhang  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.16590v1  

#### Abstract
Accurate weather and climate prediction relies on data assimilation (DA), which estimates the Earth system state by integrating observations with models. While exascale computing has significantly advanced earth simulation, scalable and accurate inference of the Earth system state remains a fundamen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Global Attention with Linear Complexity for Exascale Generative Data Assimilation in Earth System Prediction*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **Data Assimilation (DA)** 在地球系统预测中面临三大瓶颈：
- **极端时空分辨率需求**：公里级（km-scale）全球建模导致计算成本爆炸。
- **大规模集合（Ensemble）需求**：准确量化极端事件不确定性需要 $ \mathcal{O}(10^3) $ 以上集合成员，而现有系统受限于计算资源仅能支持数十至数百。
- **实时操作延迟限制**：业务化 DA 需在数分钟内完成，但传统流程常需小时级。

根本挑战在于：**传统 DA 采用两阶段“预报-更新”循环**，该结构高度依赖全局同步与频繁数据移动，导致其为内存和通信密集型，在 GPU 架构上难以高效扩展。

此外，基于 Transformer 的 AI 模型虽有潜力，但受制于 **自注意力机制的二次复杂度 $ \mathcal{O}(K^2N + KN^2) $**，无法处理亿级 spatiotemporal tokens 和长时序上下文（如 >100 帧），严重制约其在高分辨率地球系统建模中的应用。

---

### 提出的新方法与新思路

作者提出一个 **统一的一阶段生成式 DA 框架（Unified One-Stage Generative DA）**，将 DA 重构为 **贝叶斯后验采样问题**，并引入以下核心技术：

#### ✅ **STORM：具有线性复杂度全局注意力的时空 Transformer**
- 基于 **Vision Transformer (ViT)** 构建，专为高维地球系统场设计。
- 引入一种新型 **线性复杂度缩放算法（Linear-Complexity Scaling Algorithm）**，打破传统注意力的二次复杂度壁垒。
- 实现 **$ \mathcal{O}(N) $ 全局注意力**，支持高达 **200 亿 spatiotemporal tokens** 和 **177k 时间帧** 的建模能力。

#### ✅ **生成式 DA 范式转变**
- 将 DA 视为条件生成任务：直接从历史状态 $ x_{k-K:k-1} $ 和观测 $ y_k $ 条件下采样当前状态 $ x_k $。
- 使用 **扩散模型（Diffusion Model）** 进行后验采样，通过迭代去噪过程实现物理一致的状态估计。
- 后验得分分解为先验得分 $ S_{\text{prior}} $ 与似然得分 $ S_{\text{likelihood}} $，分别由 STORM 和 DPS（Diffusion Posterior Sampling）建模。

#### ✅ **噪声门控注意力机制（Noise-Gated Attention）**
- 动态调节空间自注意力与时间交叉注意力的权重，依据扩散噪声水平 $ \sigma_t $：
  - 高噪声阶段更依赖历史信息（强时间注意力）
  - 低噪声阶段聚焦当前状态精细化（强空间注意力）

#### ✅ **分层并行策略（Hierarchical Parallelism）**
- 设计与超算硬件层级对齐的并行架构：
  - **DDP** → 子集群间（低频通信）
  - **FSDP** → 节点间（中频）
  - **Tensor Parallelism** → 单节点多 GPU（高频）
  - **Tile Parallelism** → 单 GPU 内多个 SM（中频但低延迟）

---

### 相比现有方法的优势

| 维度 | 传统 DA / Two-Stage AI | 本文方法（STORM + Generative DA） |
|------|------------------------|-------------------------------|
| **范式** | 分离的预报-更新循环 | 统一的一阶段生成式推理 |
| **计算特性** | 通信/内存密集，低算术强度 | Compute-dense，适合 GPU 并行 |
| **注意力复杂度** | $ \mathcal{O}(K^2N + KN^2) $ 或局部窗口 | $ \mathcal{O}(N) $ 全局注意力 |
| **最大 token 数量** | < 10M（如 TimeSformer ~3M） | 达 **20B** |
| **时间上下文长度** | ≤ 96 帧 | 高达 **177k 帧** |
| **集合规模** | $ \mathcal{O}(10^2) $ | 支持 $ \mathcal{O}(10^4) $ 实时生成 |
| **不确定性建模** | 受限于高斯假设、定位等近似 | 多模态、非高斯后验采样 |

> ⚡️ **核心优势总结**：首次实现 **可扩展、长上下文、大集合、高分辨率、不确定性感知的统一 DA 框架**，填补了前向模拟与反演推断之间的“exascale 推理鸿沟”。

---

## 2. 核心实验方法和设置

### 数据集
使用两个互补的地球系统数据集进行训练与评估：

| 数据集 | 描述 |
|-------|------|
| **ERA5** | 全球再分析数据，~28km 分辨率（720×1440 网格），每帧含 6 变量（地形、海陆掩膜、纬度、10m 风速、2m 温度等），时间跨度 1978–2019，用于预训练与全球验证 |
| **HRRR** | 区域高分辨率快速刷新数据，5km 分辨率（512×1024 网格），覆盖美国本土，含 4 变量，2021–2024 数据，用于微调与区域温度预测评估 |

---

### 实验设置

#### 模型配置
评估三种尺度模型：
- **10M 参数**（嵌入维度 256，6 层）
- **200M 参数**（1024，8 层）
- **10B 参数**（3072，40 层）

#### 系统平台
- **Frontier 超算**（ORNL）：每个节点含 1×AMD EPYC CPU + 8×MI250X GPU（共 64GB 显存/GPU）
- 总计使用最多 **32,768 GPUs**
- 软件栈：PyTorch v2.7, ROCm v7.1, libfabric v1.22

#### 评估指标

##### 🔹 计算性能指标
- **Memory footprint**：训练峰值显存占用（GB）
- **Training/inference time-to-solution**：单样本平均墙钟时间
- **Strong scaling efficiency**：相对于 512 GPU 基线的速度提升归一化值
- **Sustained throughput**：实测持续浮点性能（ExaFLOP/s），使用 DeepSpeed FlopsProfiler 测量

##### 🔹 科学性能指标
- **RMSE**：均方根误差（预测 vs 真实值）
- **STD / Spread**：集合标准差，衡量不确定性
- **CRPS**（Continuous Ranked Probability Score）：综合评估概率预测质量
- **Variance Reduction & Spread-Skill Alignment**：评估不确定性校准能力

#### 基线方法对比
- **Conventional ViT**（全局注意力，$ \mathcal{O}(N^2) $）
- **TimeSformer**（时空因子化注意力，$ \mathcal{O}(K^2N + KN^2) $）
- **ORBIT-2**（局部注意力，支持 4B spatial tokens，无时间建模）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **线性复杂度验证（Table II）**
| 方法 | Tokens | Memory (GB) | Time (s) |
|------|--------|-------------|----------|
| STORM (w/ tiling, O(N)) | 196.6M | 46.0 | 1.54 |
| TimeSformer (O(N²)) | 3.1M | 55.0 | 0.080 |

→ STORM 在 **token 数量增加两个数量级** 的情况下，仍保持合理内存增长与可接受运行时间，证明其 **线性扩展能力**。

---

#### ✅ **最大序列扩展能力（Table III）**

| 维度 | 最大能力 |
|------|---------|
| **Temporal Scaling**（固定空间） | 支持 **177.5k 时间帧**（635M tokens） |
| **Spatial Scaling**（单时间帧） | 支持 **30,400 × 60,800 网格**（3.7B tokens） |
| **Joint Spatio-temporal Scaling** | 在 128 GPUs 上可达 **~20B tokens** |

> 💥 此类规模远超任何现有 ViT 或 DA 方法，首次进入“sub-kilometer global + long-horizon”建模范畴。

---

#### ✅ **强扩展效率（Fig. 5）**
- 在 **32,768 GPUs** 上达到 **61%–64% 强扩展效率**
- 持续性能达 **1.6 ExaFLOP/s**
- 表明通信开销被有效控制，得益于分层并行与 tile-local gradient propagation

---

#### ✅ **大规模集合生成能力（Table IV）**
| Model | GPUs | Ensemble Size | Wall Time (s) |
|-------|------|---------------|----------------|
| 10M | 64 | 64 | 34.6 |
| 10M | 4096 | 32768 | 34.1 |
| 10B | 10000 | 10000 | 242.9 |

→ **集合大小增加三个数量级，墙钟时间几乎不变**，表明近乎理想的并行采样效率。

---

#### ✅ **科学性能提升**

##### 🌀 **飓风跟踪（Hurricane Tracking）**
- 使用 **CRPS** 评价路径预测不确定性
- 结果（Table V）显示：
  - 仅用 **20% 观测数据**，CRPS 下降 **~60%**
  - 即使 **10% 观测**，也有显著改进
- 图 6 显示 DA 显著减少集合发散，提高轨迹一致性

##### 🌀 **飓风强度预测**
- 图 7 显示：
  - Forecast-only（$ S_{\text{prior}} $）系统低估峰值风速（如 Laura 差 4 m/s）
  - DA（$ S_{\text{posterior}} $）使 KDE 分布向真实值偏移，改善均值与不确定性
  - 减少沿轨迹与结构误差

##### 🌡️ **区域高温预测（HRRR）**
- 72 小时预测显示：
  - Forecast-only 在昼夜加热区（如美墨边境）低估温度 >10K
  - DA 成功纠正偏差，RMSE 降低 **2–3K**
  - 集合误差分布显著收窄，不确定性更好校准（图 8d）

---

### 消融实验结果（隐含于分析中）

- **是否启用 tiling**：
  - Tiles=1（无 tiling）：仅支持百万级 tokens
  - Tiles>1：突破至十亿级，验证线性缩放有效性
- **是否解耦时空注意力**：
  - STORM 架构本身已实现时空解耦，允许独立扩展空间与时间维度
- **噪声门控机制**：
  - 提升采样稳定性与收敛速度，尤其在高噪声阶段避免过拟合局部模式

---

## 4. 关键结论和发现

### 主要发现

1. **DA 可以且应当成为 exascale 推理的核心能力**  
   本文首次证明：通过算法-硬件协同设计，DA 不再是瓶颈，而是可以与前向模拟并列的 **新一代气候建模支柱**。

2. **线性复杂度全局注意力是可行且高效的**  
   通过 **tile-based gradient propagation + halo averaging + Hanning weighting**，可在不牺牲全局相关性的前提下实现 $ \mathcal{O}(N) $ 注意力，彻底打破 ViT 扩展瓶颈。

3. **生成式 DA 实现真正意义上的“统一”框架**  
   将预报与同化融合为单一去噪过程，消除传统 DA 中的重复耦合与同步开销，极大提升效率与保真度。

4. **前所未有的建模能力成为现实**  
   - 支持 **km-scale 全球网格 + 177k 时间步**
   - 实现 **万级集合成员秒级生成**
   - 实测 **1.6 ExaFLOP/s 持续性能**

5. **科学价值显著提升**  
   - 更好捕捉 **快速增强飓风** 与 **区域极端温度事件**
   - 提供 **可靠不确定性估计**，支持风险感知决策

---

### 方法的局限性

1. **依赖高质量训练数据**  
   - 当前依赖 ERA5/HRRR 等再分析或高分辨率观测产品
   - 对稀疏、异构、有偏观测的鲁棒性有待进一步研究

2. **扩散模型采样延迟**  
   - 尽管并行效率高，但仍需 **~80 diffusion steps**
   - 实时性要求极高的场景（<1min）可能仍有压力

3. **泛化能力边界尚待探索**  
   - 是否适用于其他地球子系统（如海洋、冰盖）？
   - 对未见极端事件（out-of-distribution）的外推能力？

4. **物理守恒律嵌入不足**  
   - 当前主要依赖数据驱动学习物理结构
   - 可考虑引入更多物理约束（如质量/能量守恒）以增强长期稳定性

---

### 未来工作方向

1. **向全地球系统扩展**  
   - 耦合大气、海洋、陆面、冰雪模块，构建端到端生成式 ESM

2. **加速采样算法开发**  
   - 探索 DDIM、Consistency Models 等快速采样器，缩短 inference wall time

3. **主动观测规划与 DA 联合优化**  
   - 利用不确定性地图指导最优观测部署（如无人机路径规划）

4. **跨模态 DA**  
   - 融合卫星图像、雷达、地面站、社交媒体等多源异构数据

5. **边缘部署轻量化版本**  
   - 开发蒸馏或量化版 STORM，用于区域气象中心或移动平台

---

> 📌 **总结一句话**：  
> 本论文通过提出 **STORM —— 一种具备线性复杂度全局注意力的时空扩散模型**，实现了 **首个可扩展、高保真、不确定性感知的统一生成式 DA 框架**，在 Frontier 超算上达成 **1.6 ExaFLOP/s 与 20B tokens 建模能力**，标志着地球系统预测正式迈入 **exascale inference 时代**。

</details>

---

### 14. [FedLLM: A Privacy-Preserving Federated Large Language Model for Explainable Traffic Flow Prediction](https://arxiv.org/abs/2604.16612)

**Authors**: Seerat Kaur, Sukhjit Singh Sehra, Dariush Ebrahimi  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.16612v1  

#### Abstract
Traffic prediction plays a central role in intelligent transportation systems (ITS) by supporting real-time decision-making, congestion management, and long-term planning. However, many existing approaches face practical limitations. Most spatio-temporal models are trained on centralized data, rely ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FedLLM: A Privacy-Preserving Federated Large Language Model for Explainable Traffic Flow Prediction

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前交通流量预测面临三大挑战：
- **数据隐私与分布性**：交通数据通常由不同区域机构独立持有，受隐私和治理限制，难以集中化训练。
- **可解释性不足**：传统 spatio-temporal 模型仅输出数值预测，缺乏对决策支持所需的推理过程。
- **模型泛化能力弱**：多数模型在非 IID（non-IID）数据分布下表现不佳，且难以跨区域迁移。

### 提出的新方法与创新思路
本文提出 **FedLLM**，一个结合 **Federated Learning (FL)** 与 **Large Language Model (LLM)** 的新型框架，用于可解释的短时交通流预测（15–60分钟）。其四大核心贡献如下：

1. **Composite Selection Score (CSS)**  
   - 提出一种多标准评分机制，综合考虑交通量、时间变异性、传感器可靠性与空间覆盖度，用于选择具有代表性的高速公路走廊进行训练，提升客户端多样性与领域适应质量。

2. **Domain-Adapted LLM**  
   - 将 Qwen2.5-1.5B-Instruct 模型通过 **QLoRA** 在结构化交通提示（prompt）上微调，使其能理解并生成基于上下文的自然语言解释，同时输出多步流量预测。

3. **集成 FL 与 LLM 的 FedLLM 框架**  
   - 首次将 LLM 应用于联邦学习环境下的交通预测任务。
   - 客户端本地训练，仅交换轻量级 **LoRA adapter 参数**（约 70.5MB/轮），而非完整模型权重（2.9GB），显著降低通信开销并保护原始数据隐私。

4. **结构化 Prompt 表示法**  
   - 设计包含静态属性（坐标、车道数）、统计摘要（均值、标准差、拥堵率）、动态时间序列（最近12个观测值）及空间邻域信息的 prompt，增强模型的情境推理与跨区域泛化能力。

### 相比现有方法的优势
| 维度 | 传统方法 | FedLLM |
|------|--------|--------|
| 数据隐私 | 中心化训练，需共享原始数据 | 联邦架构，数据保留在本地 |
| 可解释性 | 数值输出，无推理过程 | 输出 JSON + 自然语言解释链 |
| 通信效率 | 全模型参数传输，高带宽消耗 | 仅 LoRA 参数交换，通信成本下降 >95% |
| 泛化能力 | 对非 IID 分布敏感 | 多样化客户端训练，提升鲁棒性 |
| 架构通用性 | 专用图神经网络为主 | 基于通用 LLM，灵活适配多种场景 |

---

## 2. 核心实验方法和设置

### 数据集
- **主数据集**：**PeMS LargeST**（California PeMS），涵盖2017–2021年加州8,600个环形检测器站点。
- **研究区域**：
  - **训练/测试区**：洛杉矶大都会区（Greater Los Angeles Area, GLA）第12区，含24条高速路段。
  - **零样本迁移区**：旧金山湾区（Greater Bay Area, GBA）第4区，完全未参与训练，用于评估跨区域泛化能力。

### 实验设置
- **任务**：多步短期交通流预测（15, 30, 45, 60分钟）。
- **输入历史长度**：过去3小时（12个15分钟间隔）。
- **客户端配置**：4个异构客户端（SR261-S, SR57-N, SR133-N, SR133-S），代表不同交通模式（低流量、高流量、稳定中等、稀疏中等）。
- **联邦训练协议**：
  - 使用 **Flower** 框架实现联邦协调。
  - 采用 **FedAvg** 进行聚合。
  - 仅交换 **LoRA adapter 参数**。
  - 2轮通信，每轮本地训练200步。

### 评估指标
- **预测准确性**：
  - RMSE（Root Mean Squared Error）
  - MAE（Mean Absolute Error）
  - MAPE（Mean Absolute Percentage Error）
  - R²（Coefficient of Determination）
- **解释性**：人工验证生成的自然语言解释是否合理、一致。
- **零样本迁移**：在 GBA 区域直接测试，不进行任何再训练。

### 基线方法对比
#### 中央化基线（Centralized Baselines）
- **时序模型**：GRU, FC-LSTM
- **时空图模型**：STGCN, DCRNN, AGCRN, ASTGNN
- **LLM 基线**：Centralized Qwen（未微调）、Domain-Adapted LLM（集中式微调版）

#### 联邦学习基线（Federated Baselines）
- Fed-GDAN（图扩散注意力网络）
- FedASTGCN（拓扑感知联邦框架）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（整体平均）

| 模型 | RMSE ↓ | MAE ↓ | R² ↑ | MAPE ↓ |
|------|-------|-------|------|--------|
| **FedLLM (Ours)** | **23.31** | **15.07** | **0.985** | **21.84%** |
| Domain-Adapted LLM | 35.66 | 24.38 | 0.960 | 16.02% |
| Centralized Qwen | 44.63 | 32.02 | 0.775 | 28.26% |
| STGCN | 43.43 | 32.07 | 0.910 | 21.06% |
| FC-LSTM | 44.92 | 29.47 | 0.930 | 25.45% |
| DCRNN | 64.80 | 48.70 | 0.867 | 45.36% |
| FedASTGCN | 42.91 | 26.57 | 0.947 | 37.83% |

> ✅ **FedLLM 在所有指标上全面超越所有基线模型**

### 各预测时域表现（FedLLM）

| Horizon | RMSE | MAE | R² |
|---------|------|-----|----|
| 15 min | 16.30 | 10.56 | 0.989 |
| 30 min | 21.33 | 13.80 | 0.985 |
| 45 min | 25.42 | 16.61 | 0.983 |
| 60 min | 30.18 | 19.70 | 0.983 |

> 🔍 **R² 下降极小（0.989 → 0.983）**，表明模型在长时域仍保持高度稳定性。

### 与 Domain-Adapted LLM 对比
- FedLLM 相较于集中式微调版本：
  - RMSE 降低 **34.6%**（35.66 → 23.31）
  - R² 提升至 **0.985**（vs. 0.960）
- 表明：**分布式训练反而提升了全局模型性能**，得益于客户端间的数据多样性融合。

### 消融实验（Ablation Study）

| 训练/测试规模 | RMSE | MAE | R² |
|--------------|------|-----|----|
| 1000/500 | 23.31 | 15.07 | 0.985 |
| 2000/1000 | 24.40 | 15.46 | 0.936 |
| 5000/3000 | 24.20 | 15.51 | 0.935 |

> 📌 即使使用最小数据量（1000训练/500测试），FedLLM 仍优于所有基线，证明其**强数据效率与鲁棒性**。

### 零样本跨区域泛化（Zero-Shot Cross-Region Generalization）

在 **GBA 第4区** 上测试（从未见过的地理区域）：

| Horizon | R² |
|--------|-----|
| 15 min | 0.966 – 0.969 |
| 60 min | 0.864 – 0.881 |
| **Overall** | **0.916 – 0.927** |

> 🎯 性能接近原测试集（R²=0.985），说明模型具备**强大的跨区域迁移能力**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM + FL 是可行且高效的组合**：首次成功将 LLM 引入联邦交通预测框架，兼顾隐私、性能与可解释性。
2. ✅ **联邦训练可提升性能**：尽管数据分散，FedLLM 表现优于集中式训练模型，说明异构客户端协同有助于构建更泛化的全局模型。
3. ✅ **结构化 Prompt 是关键**：将交通上下文编码为自然语言 prompt，使 LLM 能够进行情境推理，是实现高精度与可解释性的基础。
4. ✅ **轻量通信设计实用性强**：LoRA 参数交换大幅减少通信负担，适合边缘设备部署与带宽受限环境。
5. ✅ **模型具备零样本迁移能力**：无需目标区域数据即可取得良好预测效果，适用于新城市快速部署。

### 局限性
1. **聚合策略简单**：目前使用标准 **FedAvg**，未考虑交通特性的动态加权（如拥堵程度、数据新鲜度）。
2. **模型容量有限**：使用的是 1.5B 参数的 Qwen 模型，更大模型（如 LLaMA-3, DeepSeek-V3）可能进一步提升性能。
3. **应用场景局限**：仅在高速公路验证，尚未扩展到城市道路、信号灯交叉口等复杂场景。
4. **未引入多模态数据**：如天气、事故报告、社交媒体事件等外部因素未被整合。

### 未来工作方向
1. **Traffic-Aware Aggregation**：开发基于交通特征（如流量波动、拥堵等级）的动态客户端加权机制。
2. **更大模型与更多客户端**：探索更大规模 LLM 在联邦设置中的潜力，并增加客户端数量以模拟真实多机构协作。
3. **扩展至城市路网**：在 METR-LA、PEMS-BAY、CityPulse 等城市数据集上验证模型普适性。
4. **多模态输入融合**：结合文本（新闻、警报）、图像（摄像头）、气象数据，构建更全面的预测系统。
5. **分层联邦学习（Hierarchical FL）**：支持市级 → 区级 → 路段级的多层级协同训练。
6. **领域预训练（Domain-Specific Pretraining）**：在大规模交通语料上预训练 LLM，减少对下游微调数据的依赖。

---

> 💡 **总体评价**：FedLLM 开创性地将 **Federated Learning** 与 **Large Language Model** 结合，解决了交通预测中的 **隐私、可解释性、泛化性** 三重难题。其实验设计严谨，结果显著优于现有方法，为智能交通系统的实际部署提供了极具前景的技术路径。

</details>

---

### 15. [Chronax: A Jax Library for Univariate Statistical Forecasting and Conformal Inference](https://arxiv.org/abs/2604.16719)

**Authors**: Xan Carey, Yash Deshmukh, Aileen Huang, Sunit Jadhav, Omkar Tekawade, Lorraine Yang, Anvesha Tiwary, Gerardo Riano, Amy Greenwald, Denizalp Goktas  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.16719v1  

#### Abstract
Time-series forecasting is central to many scientific and industrial domains, such as energy systems, climate modeling, finance, and retail. While forecasting methods have evolved from classical statistical models to automated, and neural approaches, the surrounding software ecosystem remains anchor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Chronax: A Jax Library for Univariate Statistical Forecasting and Conformal Inference — 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的时间序列预测库（如 StatsForecast）虽然在传统 CPU 上通过 Numba 加速提升了性能，但仍存在以下三大瓶颈：
- **并行能力受限**：难以高效处理大规模异构时间序列集合（multi-series forecasting）。
- **执行效率低下**：依赖 Python 解释器驱动的控制流，限制了编译优化和硬件加速（GPU/TPU）的应用。
- **集成困难**：面向对象的设计阻碍了与现代可微分机器学习流程（differentiable ML pipelines）的无缝整合。

这些问题在需要频繁重训练、低延迟推理和不确定性量化的工业级应用中尤为突出。

### 提出的新方法与新思路
作者提出了 **Chronax**，一个基于 **JAX** 构建的原生时间序列预测库，其核心设计围绕以下原则重构了统计预测范式：

#### ✅ 主要贡献（Key Contributions）
1. **Functional Forecasting Framework**  
   所有组件（预处理、模型计算、多步预测）均以纯函数（pure functions）形式实现，兼容 JAX 的 `jit`, `vmap`, `grad` 等程序变换，支持端到端优化。

2. **Unified Abstraction for Multi-Series Forecasting**  
   利用 `jax.vmap` 实现跨多个时间序列的自动向量化，无需手动批处理或进程池管理，极大简化了大规模预测任务。

3. **End-to-End Differentiable Pipeline**  
   整个预测流程（从输入特征到输出预测）均可求导，为超参数优化（hypergradient）、联合学习等高级应用提供基础。

4. **Model-Agnostic Conformal Inference Integration**  
   将保形推断（conformal prediction）自然嵌入 JAX 函数式框架，利用 `vmap` 并行化“walk-forward”交叉验证过程，在 GPU 上显著提速。

5. **Modern Reinterpretation of Classical Models**  
   重新实现了 ARIMA、ETS、Theta、TBATS 等经典模型，采用统一的 JAX 架构替代传统的混合 Python-Numba-C++ 实现，提升可维护性和扩展性。

### 相比现有方法的优势
| 维度 | Chronax | StatsForecast / 其他传统库 |
|------|--------|-----------------------------|
| **执行模式** | 编译优先（XLA），函数式 | 解释器驱动，命令式 |
| **硬件支持** | 原生支持 CPU/GPU/TPU | 主要针对 CPU 优化 |
| **并行能力** | `vmap` 支持数千序列并行 | 多进程或串行为主 |
| **可微分性** | 完全支持 `jax.grad` | 不支持或部分支持 |
| **不确定性量化** | 内建 conformal inference，可加速 | 需外部实现，效率低 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个真实世界时间序列上进行，覆盖不同领域和长度：

| 数据集 | 领域 | 观测数 | 描述 |
|-------|------|--------|------|
| **Airline Passengers** | Travel | 144 | 1949–1960 年每月乘客数量 |
| **Daily Female Births** | Health | 365 | 1959 年加州每日新生儿数 |
| **Room Temperatures** | Physics | 7,056 | IoT 设备记录的每小时室温 |

> 注：数据集规模从小到大递增，用于测试可扩展性。

### 实验设置
- **对比框架**：Chronax vs. **StatsForecast**（当前最快的统计预测库）
- **运行环境**：多环境隔离架构，避免 JAX 与 Numba 冲突；使用统一 `fit_predict` 接口封装。
- **硬件**：启用 GPU 加速（具体型号未说明），测量时排除数据传输开销。
- **重复次数**：每个模型执行 5 次 warm run 取平均。

### 评估指标
共六项指标，分为两类：

#### ⏱️ 性能（Speed）
- `Cold Start Time (T_cold)`：首次运行耗时（含 JIT 编译）
- `Warm Start Time (T_warm)`：稳定状态下的平均推理延迟（同步等待 `.block_until_ready()`）

#### 📊 准确性（Accuracy）
- `MAPE`：Mean Absolute Percentage Error
- `MAE`：Mean Absolute Error
- `RMSE`：Root Mean Squared Error
- `MASE`：Mean Absolute Scaled Error（相对于 naive 预测）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（见 Table 3）

| 指标 | Airline Passengers (Median) | Daily Female Births (Median) | Room Temperature (Median) |
|------|-------------------------------|-------------------------------|----------------------------|
| **MAPE Ratio (Chronax/Statsforecast)** | 1.00× | 1.00× | 1.00× |
| **MAE Ratio** | 1.00× | 1.00× | 1.00× |
| **RMSE Ratio** | 1.00× | 1.00× | 1.00× |
| **MASE Ratio** | 1.00× | 1.00× | 1.00× |
| **Cold Start Ratio** | 10.19× | 9.23× | 1.77× |
| **Warm Start Ratio** | 0.053× | 0.040× | 0.074× |

> ✅ **所有中位数误差指标均为 1.00×** → 表示 Chronax 与 StatsForecast 在预测精度上完全相当。

### 与基线方法的对比结果

#### 🔹 Warm Start 执行速度
- Chronax 在 warm inference 上**大幅领先**：
  - 最快达到 **StatsForecast 的 19 倍以上**（median 仅需 4–7.4% 时间）。
  - 在最大数据集（Room Temperature）上仍保持 **约 13.5 倍加速**（1 / 0.074 ≈ 13.5）。
- 原因：JAX 的 `jit` 和 `vmap` 实现了高度优化的并行执行。

#### 🔹 Cold Start 初始化延迟
- Chronax 明显更慢（高 6–10 倍），这是由于 **JIT 编译开销**所致。
- 但随着数据量增大，相对差距缩小（从 ~40× 下降到 ~6×），表明编译成本趋于固定。

#### 🔹 准确性表现
- 所有误差指标（MAPE/MAE/RMSE/MASE）的 **median ratio 均为 1.00×**，说明 Chronax **没有牺牲任何预测质量**。
- 在较大数据集上，**mean error ratio < 1.0×**（如 Room Temperature 上 MAE=0.94×），显示 Chronax 在大数据下可能略有优势。

### 消融实验（隐含分析）
尽管文中未明确列出消融实验，但从设计可推断：
- **`vmap` 对 conformal validation 的加速效果显著**：将原本需 K 次串行拟合的任务并行化，是 warm performance 提升的关键。
- **JIT 编译带来长期收益**：虽然 cold start 成本高，但一旦完成，后续调用极快，适合频繁推理场景。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Chronax 实现了与 StatsForecast 相当甚至更优的预测准确性**，证明 JAX 重写不影响模型正确性。
2. ⚡ **Warm inference 性能远超传统库**，尤其适用于需要高频调用的大规模预测系统。
3. 🔄 **函数式 + JAX 转换机制天然支持 conformal inference 的并行化**，解决了传统方法中 walk-forward CV 效率低下的痛点。
4. 💡 **JAX 的“高冷启动成本 + 低运行时开销”架构权衡合理**：适用于生产环境中“一次编译、多次执行”的典型模式。
5. 🧩 **统一的 JAX 接口为未来扩展奠定基础**：易于接入深度学习模型、可微调超参、构建混合 pipeline。

### 方法的局限性
- ❌ **Cold Start 较慢**：不适合一次性、低频调用的场景（如单次报告生成）。
- ❌ **目前仅支持 univariate 模型**：尚未涵盖 multivariate 或 deep learning forecasters。
- ❌ **依赖 JAX 生态**：对不熟悉 JAX 的用户有一定学习门槛。
- ❌ **GPU 内存占用较高**：大规模并行可能导致显存溢出（文中未讨论）。

### 未来工作方向
1. **扩展至 multivariate 和 deep learning 模型**：构建统一接口下的混合预测系统。
2. **增强 conformal inference 功能**：
   - 支持 hierarchical forecasting 中的区间一致性（coherence）
   - 提高对分布漂移（distribution shift）的鲁棒性
3. **深化与 JAX 生态集成**：
   - 可微分超参数优化（differentiable hyperparameter tuning）
   - 与科学模拟器联合优化
   - 构建端到端的 forecasting-as-a-service 系统
4. **探索分布式训练与推理**：支持 TPU Pods 等更大规模硬件。

---

> 🔗 开源地址：[https://github.com/Smlcrm/Chronax](https://github.com/Smlcrm/Chronax)

</details>

---

### 16. [UniCon: Unified Framework for Efficient Contrastive Alignment via Kernels](https://arxiv.org/abs/2604.16678)

**Authors**: Hangke Sui, Yuqing Wang, Minh N Do  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.16678v1  

#### Abstract
Contrastive objectives power state-of-the-art multimodal models, but their training remains slow, relying on long stochastic optimization. We propose a Unified Framework for Efficient Contrastive Alignment via Kernels (UniCon), which spans linear and nonlinear encoders as well as one-to-one and many...

---

### 17. [HopRank: Self-Supervised LLM Preference-Tuning on Graphs for Few-Shot Node Classification](https://arxiv.org/abs/2604.17271)

**Authors**: Ziqing Wang, Kaize Ding  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.17271v1  

#### Abstract
Node classification on text-attributed graphs (TAGs) is a fundamental task with broad applications in citation analysis, social networks, and recommendation systems. Current GNN-based approaches suffer from shallow text encoding and heavy dependence on labeled data, limiting their effectiveness in l...

---

### 18. [Open-TQ-Metal: Fused Compressed-Domain Attention for Long-Context LLM Inference on Apple Silicon](https://arxiv.org/abs/2604.16957)

**Authors**: Sai Vegasena  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.16957v1  

#### Abstract
We present Open-TQ-Metal, the first implementation of fused compressed-domain attention on Apple Silicon, enabling 128K-context inference for Llama 3.1 70B on a single 64GB consumer Mac -- a configuration impossible with all existing inference frameworks. Open-TQ-Metal quantizes the KV cache to int4...

---

### 19. [Towards a Data-Parameter Correspondence for LLMs: A Preliminary Discussion](https://arxiv.org/abs/2604.17384)

**Authors**: Ou Wu  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.17384v1  

#### Abstract
Large language model optimization has historically bifurcated into isolated data-centric and model-centric paradigms: the former manipulates involved samples through selection, augmentation, or poisoning, while the latter tunes model weights via masking, quantization, or low-rank adaptation. This pa...

---

### 20. [FlashFPS: Efficient Farthest Point Sampling for Large-Scale Point Clouds via Pruning and Caching](https://arxiv.org/abs/2604.17720)

**Authors**: Yuzhe Fu (Helen), Hancheng Ye (Helen), Cong Guo (Helen), Junyao Zhang (Helen), Qinsi Wang (Helen), Yueqian Lin (Helen), Changchun Zhou (Helen),  Hai (Helen),  Li, Yiran Chen  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.17720v1  

#### Abstract
Point-based Neural Networks (PNNs) have become a key approach for point cloud processing. However, a core operation in these models, Farthest Point Sampling (FPS), often introduces significant inference latency, especially for large-scale processing. Despite existing CUDA- and hardware-level optimiz...

---

### 21. [Universally Empowering Zeroth-Order Optimization via Adaptive Layer-wise Sampling](https://arxiv.org/abs/2604.18264)

**Authors**: Fei Wang, Li Shen, Liang Ding, Chao Xue, Ye Liu, Changxing Ding  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.18264v1  

#### Abstract
Zeroth-Order optimization presents a promising memory-efficient paradigm for fine-tuning Large Language Models by relying solely on forward passes. However, its practical adoption is severely constrained by slow wall-clock convergence and high estimation variance. In this work, we dissect the runtim...

---

### 22. [OPSDL: On-Policy Self-Distillation for Long-Context Language Models](https://arxiv.org/abs/2604.17535)

**Authors**: Xinsen Zhang, Zhenkai Ding, Tianjun Pan, Run Yang, Chun Kang, Xue Xiong, Jingnan Gu  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.17535v1  

#### Abstract
Extending the effective context length of large language models (LLMs) remains a central challenge for real-world applications. While recent post-training methods have made progress in long-context scaling, they either rely on high-quality supervision data or sparse sequence-level rewards, leading t...

---

### 23. [GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling](https://arxiv.org/abs/2604.18556)

**Authors**: Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtic, Dan Alistarh  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.18556v1  

#### Abstract
Weight quantization has become a standard tool for efficient LLM deployment, especially for local inference, where models are now routinely served at 2-3 bits per parameter. The state of the art is currently split into two sets of methods: simple scalar quantization techniques, such as GPTQ or AWQ, ...

---

### 24. [AutoOR: Scalably Post-training LLMs to Autoformalize Operations Research Problems](https://arxiv.org/abs/2604.16804)

**Authors**: Sumeet Ramesh Motwani, Chuan Du, Aleksander Petrov, Christopher Davis, Philip Torr, Antonio Papania-Davis, Weishi Yan  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.16804v1  

#### Abstract
Optimization problems are central to decision-making in manufacturing, logistics, scheduling, and other industrial settings. Translating complicated descriptions of these problems into solver-ready formulations requires specialized operations research (OR) expertise, making it hard to scale. We pres...

---

### 25. [Reference-state System Reliability method for scalable uncertainty quantification of coherent systems](https://arxiv.org/abs/2604.17066)

**Authors**: Ji-Eun Byun, Hyeuk Ryu, Junho Song  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.17066v1  

#### Abstract
Coherent systems are representative of many practical applications, ranging from infrastructure networks to supply chains. Probabilistic evaluation of such systems remains challenging, however, because existing decomposition-based methods scale poorly as the number of components grows. To address th...

---

### 26. [Federated Rule Ensemble Method in Medical Data](https://arxiv.org/abs/2604.17956)

**Authors**: Ke Wan, Kensuke Tanioka, Toshio Shimokawa  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.17956v1  

#### Abstract
Machine learning has become integral to medical research and is increasingly applied in clinical settings to support diagnosis and decision-making; however, its effectiveness depends on access to large, diverse datasets, which are limited within single institutions. Although integrating data across ...

---

### 27. [Scalable Physics-Informed Neural Differential Equations and Data-Driven Algorithms for HVAC Systems](https://arxiv.org/abs/2604.18438)

**Authors**: Hanfeng Zhai, Hongtao Qiao, Hassan Mansour, Christopher Laughman  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.18438v1  

#### Abstract
We present a scalable, data-driven simulation framework for large-scale heating, ventilation, and air conditioning (HVAC) systems that couples physics-informed neural ordinary differential equations (PINODEs) with differential-algebraic equation (DAE) solvers. At the component level, we learn heat-e...

---

### 28. [Uncertainty Quantification in PINNs for Turbulent Flows: Bayesian Inference and Repulsive Ensembles](https://arxiv.org/abs/2604.17156)

**Authors**: Khemraj Shukla, Zongren Zou, Theo Kaeufer, Michael Triantafyllou, George Em Karniadakis  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.17156v1  

#### Abstract
Physics-informed neural networks (PINNs) have emerged as a promising framework for solving inverse problems governed by partial differential equations (PDEs), including the reconstruction of turbulent flow fields from sparse data. However, most existing PINN formulations are deterministic and do not...

---

### 29. [Fully Analog Resonant Recurrent Neural Network via Metacircuit](https://arxiv.org/abs/2604.17277)

**Authors**: Zixin Zhou, Tianxi Jiang, Menglong Yang, Zhihua Feng, Qingbo He, Shiwu Zhang  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.17277v1  

#### Abstract
Physical neural networks offer a transformative route to edge intelligence, providing superior inference speed and energy efficiency compared to conventional digital architectures. However, realizing scalable, end-to-end, fully analog recurrent neural networks for temporal information processing rem...

---

### 30. [LLM-AUG: Robust Wireless Data Augmentation with In-Context Learning in Large Language Models](https://arxiv.org/abs/2604.17770)

**Authors**: Pranshav Gajjar, Manan Tiwari, Sayanta Seth, Vijay K. Shah  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.17770v1  

#### Abstract
Data scarcity remains a fundamental bottleneck in applying deep learning to wireless communication problems, particularly in scenarios where collecting labeled Radio Frequency (RF) data is expensive, time-consuming, or operationally constrained. This paper proposes LLM-AUG, a data augmentation frame...

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
