# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-29 07:45:27 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [CacheFlow: Efficient LLM Serving with 3D-Parallel KV Cache Restoration](https://arxiv.org/abs/2604.25080)

**Authors**: Sean Nian, Jiahao Fang, Qilong Feng, Zhiyu Wu, Fan Lai  
**Category**: cs.DC  
**Published**: 2026-04-29  
**Score**: 15.0  
**Type**: new  
**ArXiv ID**: 2604.25080v1  

#### Abstract
KV cache restoration has emerged as a dominant bottleneck in serving long-context LLM workloads, including multi-turn conversations, retrieval-augmented generation, and agentic pipelines. Existing approaches treat restoration as a per-request tradeoff between recomputation and I/O transfer, recomput...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CacheFlow: Efficient LLM Serving with 3D-Parallel KV Cache Restoration**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在长上下文 LLM 推理场景中（如多轮对话、检索增强生成 RAG 和智能体流水线），**KV Cache 的恢复（restoration）已成为影响延迟的关键瓶颈**。现有方法通常将恢复视为单个请求层面的“重新计算 vs. I/O 加载”二元选择，存在以下问题：
- **忽略结构性并行性**：未充分利用 token、layer 和 GPU 之间的并行潜力；
- **成本非均匀性**：注意力机制的二次复杂度导致后序 token 重计算代价极高；
- **批处理资源争用**：多个请求共享计算与 I/O 资源时产生竞争和拖尾效应（straggler effects）。

因此，如何高效协调大规模、异构、批处理下的 KV Cache 恢复成为核心挑战。

---

### **提出了什么新方法或新思路**
提出 **CacheFlow** —— 一种将 KV Cache 恢复重构为**三维并行执行问题**的新框架，其核心创新包括：

#### ✅ **统一的 3D 并行抽象**
识别出三个可并行维度，并通过结构化依赖关系实现细粒度重叠：
1. **Token-level Parallelism**：利用因果依赖，从前向后重计算早期 token，同时从后向前加载后期 token；
2. **Layer-level Parallelism**：利用前馈结构，在低层自底向上重计算的同时，高层自顶向下加载缓存；
3. **Multi-GPU Parallelism**：借助轻量级边界隐藏状态（boundary hidden states），各 GPU 可独立并行恢复本地模型分片的 KV Cache。

#### ✅ **Batch-aware Two-Pointer Scheduler**
设计了一个全局感知的双指针调度器：
- 每个请求维护一个 **compute pointer** 和 **I/O pointer**；
- 全局调度器动态优先服务那些能带来最大边际重计算节省的请求（即长前缀请求）；
- 实现跨请求、跨设备的资源协同优化。

#### ✅ **理论最优性保障**
证明该两指针策略达到 **harmonic-mean bound**，即恢复时间 $ T^* \propto \frac{T_{\text{comp}} \cdot T_{\text{IO}}}{T_{\text{comp}} + T_{\text{IO}}} $，优于单纯的 `min(T_comp, T_IO)`，实现了计算与 I/O 的最优平衡。

---

### **相比现有方法的优势**
| 维度 | CacheFlow | 现有方法（如 vLLM, LMCache, Cake） |
|------|----------|-------------------------------|
| 并行粒度 | 支持 token-, layer-, GPU- 三级并行 | 仅支持单一维度或粗粒度混合 |
| 批处理优化 | 全局调度，缓解资源争用 | 请求级独立决策，易引发拥塞 |
| 成本建模 | 考虑二次重计算代价与带宽限制 | 忽视非均匀成本分布 |
| 性能表现 | 显著降低 TTFT，尤其对长序列和尾部延迟 | 在极端条件下性能退化严重 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LMSYS-Chat**：真实世界 ChatGPT 多轮对话轨迹，反映常见聊天机器人部署；
- **WildChat**：开放域大规模对话语料，涵盖多种任务与语言，前缀长度分布广泛；
- **SWE-Bench**：代码智能体基准测试，涉及工具调用与共享仓库上下文，体现系统性前缀复用。

这些工作负载覆盖了从短到超长上下文（最高达 32K–128K tokens）的不同场景。

---

### **实验设置和评估指标**

#### **模型**
- Qwen3-8B（dense）
- Llama-3.1-8B（dense）
- Qwen3-30B-A3B（MoE 架构，激活参数约 3B）

#### **硬件平台**
- 单卡或多卡配置：NVIDIA L40S (46GB), A100 (40GB), H100 (80GB)
- 网络带宽模拟：10 Gbps（典型云间节点）、40 Gbps（SSD读速）、80 Gbps（InfiniBand）

#### **评估指标**
- **Time-To-First-Token (TTFT)**：首要指标，直接反映用户感知延迟；
- GPU Compute Utilization & I/O Bandwidth Utilization：衡量资源效率；
- 尾部延迟（P90–P99）分析：检验系统鲁棒性。

---

### **基线方法对比**
| 基线 | 类型 | 描述 |
|------|------|------|
| **vLLM** | 重计算主导 | 完全通过 prefill 重新生成 KV Cache |
| **LMCache** | I/O 主导 | 完全从外部存储加载 KV Cache |
| **SGLang (HiCache)** | 存储层级缓存 | 扩展 RadixAttention 至外存 |
| **Cake** | 混合策略 | 按 token 分块进行局部重算/加载 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **TTFT 减少 10%–62%**，平均提升 **1.1×–1.7×**，在尾部延迟（P99）上增益更显著；
- 在 **SWE-Bench 和 LMSYS-Chat** 上效果最明显，因其具有大量长前缀复用；
- 最高提速出现在 **10 Gbps 低带宽环境 + 高负载批次** 下，表明 CacheFlow 对现实约束更具鲁棒性。

---

### **与基线方法的对比结果**
| 方法 | 相对于最佳基线的 TTFT 改进 |
|------|------------------------|
| vLLM | ——（基线之一） |
| LMCache | I/O 利用率高但 GPU 闲置（仅 10% 利用） |
| Cake | 缺乏跨层与跨 GPU 协同，收益有限 |
| **CacheFlow** | **全面左移 CDF 曲线，P90/P99 改善显著** |

> 如图 4 所示，在所有工作负载下，CacheFlow 的 TTFT CDF 均显著左偏，说明其在各类请求中均有效。

---

### **消融实验结果**

#### 🔹 **按请求长度拆解（Figure 6）**
- vLLM 的 TTFT 随长度呈**超线性增长**（因 attention 二次开销）；
- CacheFlow 通过优先加载后段 token 有效抑制了这一趋势；
- 序列越长（如从 6K → 30K），相对加速比从 **1.1× 提升至 1.7×**。

#### 🔹 **禁用 Multi-GPU 并行性（Figure 7）**
- 移除 multi-GPU 并行后，平均恢复延迟从 **0.21s 升至 0.29s（↑38%）**；
- 即便如此，仍优于 vLLM（↓24%），说明 token-/layer-wise 并行本身已具价值。

#### 🔹 **不同 I/O 带宽影响（Figure 8）**
- 在 **40 Gbps** 下 TTFT 加速 **1.7×**；
- 在 **80 Gbps** 下仍有 **1.5×** 提升；
- 表明 CacheFlow 能自适应带宽变化，动态调整 compute/I/O 交界点。

#### 🔹 **不同 GPU 硬件表现（Figure 9）**
- 在 **L40S 和 A100** 上分别实现 **1.6× 和 1.5×** 加速；
- 显示方法对硬件差异具备良好泛化能力。

#### 🔹 **批大小影响（Figure 10）**
- 随 batch size 增大（2→8），TTFT 改善更加显著（**1.6×→2.6×**）；
- 验证了 batch-aware 调度在高并发下的有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **KV Cache 恢复本质上是多维调度问题**，不能简化为 per-request 决策；
2. **3D 并行 + 两指针机制** 可最大化 compute 与 I/O 的重叠，逼近理论最优；
3. **批处理调度必须考虑边际效益**：优先传输长前缀请求的 KV Cache 可显著减少整体计算负担；
4. **边界隐藏状态的设计** 是实现 multi-GPU 并行恢复的关键，大幅降低跨设备依赖。

---

### **方法的局限性**
- 当前实现依赖于 **FlashAttention 的 block 对齐机制**，可能不适用于所有 Attention 实现；
- 对 **极短序列（<512 tokens）** 收益较小，此时固定开销占主导；
- **边界状态需额外存储**，虽远小于完整 KV Cache，但仍引入轻微元数据管理开销；
- 实验集中在文本生成场景，尚未验证在语音、视觉等多模态中的扩展性。

---

### **未来工作方向**
- 将 3D 并行思想推广至 **KV Cache Prefetching 与 Eviction 策略**；
- 结合 **reuse pattern prediction**（如 Continuum）实现更智能的缓存生命周期管理；
- 探索 **异构设备集群**（CPU+FPGA+GPU）下的分布式恢复调度；
- 支持 **动态 sequence length prediction** 以进一步优化 pointer 初始化位置。

---

> 💡 **一句话总结**：  
> CacheFlow 通过引入 **token-, layer-, GPU-level 的三维并行恢复机制** 与 **batch-aware 两指针调度器**，首次将 KV Cache 恢复建模为多维调度问题，在真实场景下实现了 **10%–62% 的 TTFT 降低**，为下一代高效 LLM Serving 系统提供了新范式。

</details>

---

### 2. [FED-FSTQ: Fisher-Guided Token Quantization for Communication-Efficient Federated Fine-Tuning of LLMs on Edge Devices](https://arxiv.org/abs/2604.25421)

**Authors**: Changyu Li, Shuanghong Huang, Jiashen Liu, Ming Lei, Jidu Xing, Kaishun Wu, Lu Wang, Fei Luo  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.25421v1  

#### Abstract
Federated fine-tuning provides a practical route to adapt large language models (LLMs) on edge devices without centralizing private data, yet in mobile deployments the training wall-clock is often bottlenecked by straggler-limited uplink communication under heterogeneous bandwidth and intermittent p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FED-FSTQ: Fisher-Guided Token Quantization for Communication-Efficient Federated Fine-Tuning of LLMs on Edge Devices

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **边缘设备上的联邦学习**（Federated Learning, FL）中，对 **大语言模型**（LLMs）进行微调时面临严重的通信瓶颈：
- 上行链路带宽异构且有限（heterogeneous uplink），导致训练过程受“**straggler-limited**”（最慢客户端主导）影响。
- 即使采用 **参数高效微调**（PEFT, 如 LoRA），每轮传输的数据量依然巨大。
- 在 **非独立同分布**（Non-IID）数据下，均匀压缩会丢失稀有但关键的语义信号（如医学文本中的否定词、代码中的分隔符），损害模型性能。

### 🚀 提出的新方法：FED-FSTQ
提出了一种名为 **FED-FSTQ**（Fisher-Spectrum-aware Token Quantization）的系统原语，其核心思想是：
- 利用 **Fisher Information** 作为动态通信控制信号，指导 **token-level** 的敏感度感知压缩。
- 结合 **重要性感知的 token 选择** 与 **混合精度量化**（mixed-precision quantization），实现：
  - 对语义关键 token 分配高保真度传输；
  - 对冗余 token 进行剪枝或低比特量化。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | FED-FSTQ |
|------|--------|---------|
| **压缩依据** | 参数级、幅度驱动（magnitude-driven）、盲压缩 | **token级、语义敏感度驱动**，基于 Fisher 代理 |
| **适用性** | 通用但易丢关键信号 | 显式保留任务关键 token（如 negations, delimiters） |
| **部署兼容性** | 需修改聚合逻辑或训练流程 | **即插即用模块**（drop-in module），无需修改服务器聚合规则（FedAvg-compatible） |
| **资源利用** | 均匀分配带宽 | 动态优化 **rate-distortion trade-off**，提升信噪比 |

---

## 2. 核心实验方法和设置

### 📚 数据集
实验在三个具有挑战性的联邦 LLM 微调任务上进行：
- **Fed-Aya**：多语言问答任务，模拟语言分布异构（Dirichlet α=0.1），涵盖 `{ar,en,es,fr,pt,ru,te,zh}`。
- **Fed-Med**：医学问答任务（基于 PubMedQA），测试对罕见医学实体和逻辑操作符（如否定）的鲁棒性。
- **Fed-Code**：代码生成任务（CodeAlpaca-20k），评估语法正确性（Pass@1）。

### ⚙️ 实验设置
- **客户端配置**：`K=100` 客户端，每轮采样 `10` 个参与（10% 参与率）。
- **模型架构**：LLaMA-2-7B 和 Llama-3-8B，使用 **LoRA**（rank=16）进行 PEFT。
- **网络模拟**：
  - **Controlled LTE-20Mbps**：固定速率用于分析 payload 影响。
  - **Heterogeneous LTE**：引入慢速 straggler（0.5–2 Mbps），模拟真实移动网络。
- **客户端掉线**：以 `P_drop=0.1` 模拟间歇性连接。

### 🎯 评估指标
| 类别 | 指标 |
|------|------|
| **通信效率** | 累计上行流量（cumulative uplink traffic） |
| **端到端延迟** | time-to-accuracy（达到目标准确率所需的 wall-clock 时间） |
| **系统开销** | 每轮耗时分解（计算 vs 通信）、能耗（energy-per-round） |
| **语义可靠性** | Token Recall（保留关键 token 的比例）、ROUGE-L、METEOR、LLM-as-a-judge |
| **内存占用** | 峰值内存（peak memory footprint） |

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **无压缩基准** | FedAvg-LoRA |
| **参数中心化压缩** | QSGD（量化）、Top-k Sparsification、FedPAQ（周期平均+量化）、FedBAT（可学习二值化） |
| **启发式数据压缩** | Fed-ToMe（attention-driven token merging） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ 通信效率
- **累计上行流量减少 46×**：相比标准 Fed-LoRA，在达到相同质量阈值时，FED-FSTQ 将总上传数据量降低 **46倍**。
- **多语言成本均衡**：在 Fed-Aya 上，中文（zh）通信成本从 4.35 降至 2.08（↓52%），优于所有基线。

#### ⏱️ 端到端延迟与能效
- **time-to-accuracy 加快 52%**：由于显著缓解了 straggler 问题，训练更快收敛。
- **单轮延迟下降 6.8×**：从 414.60s → 61.05s（Jetson 测试平台）。
- **能耗大幅降低**：
  - 每轮能量从 634.40J（FedAvg）降至 **98.50J**（FED-FSTQ）。
  - 即使在高发射功率（5W）下仍节省超 80% 能耗（见 Table III）。

#### 💡 推理加速
- 启用 Fisher 引导的推理阶段 token 减少后，在 NVIDIA Jetson 设备上实现 **1.55× 端到端推理速度提升**。

### 🔁 与基线方法对比结果

| 方法 | 相对 Fed-LoRA 的通信量 | time-to-accuracy 提升 | Token Recall |
|------|------------------------|------------------------|-------------|
| QSGD (4-bit) | ~4× 减少 | ~15% 更快 | 0.7845 |
| FedPAQ | ~1.25× 减少 | ~10% 更快 | 0.7620 |
| Fed-ToMe | ~2× 减少 | ~20% 更快 | 0.6540 |
| **FED-FSTQ (Ours)** | **46× 减少** | **52% 更快** | **0.8320** |

> ✅ FED-FSTQ 不仅通信最少，且 **Token Recall 最高**，说明其在压缩同时更好保留了关键语义。

### 🔍 消融实验结果（Ablation Study）

| 变体 | ROUGE-L | Payload (MB) | 说明 |
|------|--------|--------------|------|
| **Full FED-FSTQ** | 0.6610 | 153.6 | 完整方法 |
| w/o Fisher（随机策略） | 0.4215 | 153.6 | 性能暴跌 → **Fisher 指导至关重要** |
| w/o Token Pruning | 0.6650 | 512.0 | 质量略升但通信翻 3 倍 |
| w/o Quantization | 0.6720 | 614.4 | 通信进一步恶化 |

> ✅ 表明 **Fisher 指导 + token 剪枝 + 混合精度量化** 三者协同才能实现最优权衡。

#### 匹配预算消融（Matched-Budget）
在相同上行预算（150MB/轮）下比较：
- FED-FSTQ 在第 50 轮 ROUGE-L 达 **36.15**，显著高于 Uniform Fisher（31.50）和 Random Support（18.05）。
- 证明 **token-parameter coupling** 机制有效提升了通信内容的质量。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **通信不应盲目压缩**：在 Non-IID 场景下，**语义敏感度感知的压缩** 比统一量化更有效。
2. **Fisher 是有效的轻量级代理**：通过输入嵌入梯度构建的 token-level Fisher proxy，能够准确识别决定性 token。
3. **边际效益原则指导比特分配**：只有当增加某坐标精度带来的 Fisher 加权失真下降超过通信代价时才值得传输。
4. **训练与推理双重收益**：Fisher 得分可在推理时复用，实现动态 token 压缩，带来额外 **1.55× 推理加速**。
5. **边缘可行性高**：
   - 峰值内存仅 **1450MB**，低于 2GB Jetson 限制；
   - 额外计算开销（+0.85s）被通信节省完全抵消。

### ⚠️ 局限性
- 当前依赖同步协议，未考虑异步或部分同步场景下的更新陈旧性（staleness）问题。
- Fisher 估计假设局部二次近似成立，在极端非凸区域可能失效。
- 未集成 Secure Aggregation 或 Differential Privacy，这些机制可能限制元数据传输。

### 🔮 未来工作方向
1. 扩展至 **异步联邦学习** 协议，平衡 straggler 缓解与更新陈旧性。
2. 在 **安全聚合**（Secure Aggregation）和 **差分隐私**（DP）约束下增强鲁棒性。
3. 探索更高效的 Fisher 近似方法（如 Kronecker-factored）以支持更大模型。
4. 将 Fisher-guided 控制信号推广至其他模态（如视觉、语音）的联邦适应任务。

---

> **一句话总结**：  
> FED-FSTQ 通过引入 **Fisher-guided token sensitivity** 作为通信控制系统，在不牺牲语义可靠性的前提下，实现了高达 **46× 的通信压缩** 和 **52% 的端到端加速**，为边缘设备上的高效联邦 LLM 微调提供了实用且可部署的解决方案。

</details>

---

### 3. [MetaGAI: A Large-Scale and High-Quality Benchmark for Generative AI Model and Data Card Generation](https://arxiv.org/abs/2604.23539)

**Authors**: Haoxuan Zhang, Ruochi Li, Yang Zhang, Zhenni Liang, Junhua Ding, Ting Xiao, Haihua Chen  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.23539v1  

#### Abstract
The rapid proliferation of Generative AI necessitates rigorous documentation standards for transparency and governance. However, manual creation of Model and Data Cards is not scalable, while automated approaches lack large-scale, high-fidelity benchmarks for systematic evaluation. We introduce Meta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MetaGAI: A Large-Scale and High-Quality Benchmark for Generative AI Model and Data Card Generation

## 1. 论文的主要贡献和创新点

### 解决的问题
当前，随着生成式人工智能（GenAI）模型的爆炸式增长，社区平台（如 Hugging Face）上已有超过两百万个模型和数据集。然而，高质量的 **Model and Data Card** 文档严重不足，手动编写存在可扩展性差、不一致、主观性强等问题。现有的自动化生成方法受限于缺乏大规模、高保真度的基准数据集来系统评估其准确性。

该论文旨在解决以下核心瓶颈：
- 缺乏一个大规模、高质量、经过验证的基准（benchmark），用于客观评估自动化文档生成系统的性能。
- 现有方法多将文档生成视为简单的文本摘要任务，忽略了从异构来源（论文、代码库、模型页面）进行多源信息三角验证的复杂性。

### 提出的新方法与创新思路
作者提出了 **MetaGAI**，这是一个大规模、高质量的基准数据集和评估框架，专门用于自动化生成 **Model and Data Card**。

其核心创新点包括：

- **多源三角验证构建 Ground Truth**：
  - 不同于以往仅依赖单一来源（如论文）的方法，MetaGAI 通过语义三角化（semantic triangulation）整合来自 **arXiv 论文、GitHub 仓库 和 Hugging Face 模型页面** 的信息，构建出更完整、更可靠的 Ground Truth。
  - 这种方法能够捕捉在论文中常被省略的实现细节（如超参数、许可证、部署约束）。

- **多智能体生成框架（Multi-Agent Framework）**：
  - 设计了一个由三个专业化智能体组成的生成流程：
    1. **Retriever Agent**：负责从分块的文档中检索与预定义模式（schema）相关的证据。
    2. **Generator Agent**：采用 **Ensemble Generation**，使用三种架构不同的 LLM（OLMo-3-7B, Llama-3.1-8B, Qwen2.5-7B）独立生成草稿，以减少单个模型的架构偏见。
    3. **Editor Agent**：作为“主编”，对多个候选草稿进行交叉验证，合并非冗余信息，并过滤归因错误，最终生成高保真的卡片。这一步显著提升了事实准确性（faithfulness）。

- **四维人类在环验证协议（Four-Dimensional Human-in-the-Loop Validation）**：
  - 通过四个维度的人工评估确保整个流程的质量：
    1. **D1**: 检索策略的有效性。
    2. **D2**: 生成器之间的多样性分析。
    3. **D3**: 编辑器提升质量的效果。
    4. **D4**: 不同编辑器架构的选择。
  - 特别是 D3 验证表明，经过 Editor 处理的卡片质量相比原始生成草稿有 **15-20% 的绝对提升**。

### 相比现有方法的优势
- **规模与质量**：MetaGAI 是目前最大的同类基准，包含 **2,541 个经过验证的三元组**（论文、GitHub、Hugging Face）。
- **真实性**：评估任务设定为仅输入论文（paper-only），这反映了现实中最常见且最具挑战性的场景，而 Ground Truth 则利用了所有三个来源的信息，从而可以量化“仅凭论文能恢复多少信息”。
- **严谨性**：通过多智能体框架和四维人工验证，确保了生成的 Ground Truth 具有高保真度，避免了传统方法中常见的幻觉和不一致性问题。

---

## 2. 核心实验方法和设置

### 数据集
- **MetaGAI 自建数据集**：从 arXiv、GitHub 和 Hugging Face 中系统性地收集并筛选，初始候选 15,727 个，经过规范过滤和语义一致性验证后，得到 **2,541 个高质量的三元组**。
- **数据构成**：涵盖 **Model Cards** 和 **Data Cards** 两类，时间跨度从 2019 到 2025 年，领域以计算机视觉（cs.CV）和计算语言学（cs.CL）为主。

### 实验设置
- **任务形式**：Zero-shot 生成，即模型 **仅基于论文文本（P）** 来生成完整的 Model/Data Card。
- **评估范围**：
  - **自动化指标**：在全部 2,541 个样本上计算。
  - **LLM-as-a-Judge**：在分层随机抽样的 500 个样本上进行。

### 评估指标
采用双层评估体系，结合自动化指标与专家判断：

#### 自动化指标
- **Completeness**：衡量字段级别的召回率，即生成的卡片覆盖了多少 Ground Truth 字段。
- **Semantic Similarity**：
  - **ROUGE-L**：基于词法重叠。
  - **BERTScore (F1)**：基于语义对齐。
- **Cost Index**：引入成本效率指标，标准化每张卡片生成的推理成本（基于 1M 输入 + 0.2M 输出 tokens 的价格）。

#### LLM-as-a-Judge 评估
- 使用一个由三个大型 LLM 组成的评审团（**GPT-OSS-120B, Llama-3.3-70B-Instruct, Qwen3-235B-A22B-2507**）对生成内容进行评分。
- 评分维度（1-5 分制）：
  - **Faithfulness**（忠实性）
  - **Relevance**（相关性）
  - **Accuracy**（准确性）
  - **Consistency**（一致性）
  - **Usefulness**（实用性）
- 最终得分为多个评委的平均值，以减少单个模型的偏见。

### 基线方法对比
对比了多种类型的模型，按访问方式和架构分类：

| 类型 | 模型列表 |
| :--- | :--- |
| **Dense Models** | Mistral-Small-3.2-24B-Instruct, Gemma-3-27B-IT, Qwen3-32B |
| **MoE Models** | GPT-OSS-20B, NVIDIA-Nemotron-3-Nano-30B-A3B, Qwen3-30B-A3B-Instruct |
| **Closed-Source Models** | GPT-5.1-Mini/Nano, Gemini-2.5-Flash/Flash-Lite |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **最佳性能模型**：**Qwen3-30B-A3B-Instruct**（稀疏 MoE 架构）在多项指标上取得最优。
  - **Model Card 质量得分**：4.55
  - **Data Card 质量得分**：4.30
  - **BERTScore**：0.246
  - **Completeness**：0.786 (Model), 0.702 (Data)
  - **Cost Index**：0.15（最低之一）

### 与基线方法的对比结果
- **MoE 架构优势明显**：
  - 在同等参数量级下，MoE 模型（如 Qwen3-30B-A3B-Instruct）显著优于密集模型（如 Qwen3-32B）。
  - 但 MoE 的优势需要达到一定规模（约 30B 参数），较小的 MoE（如 GPT-OSS-20B）表现反而最差。
- **开放权重模型更具成本效益**：
  - Qwen3-30B-A3B-Instruct 在成本-质量权衡上占据帕累托前沿（Pareto-optimal），其成本效率是 GPT-5-Mini 的 **4.3 倍**。
  - 闭源模型如 Gemini-2.5-Flash 虽然性能尚可，但成本过高（Cost Index: 0.80），性价比极低。

### 消融实验结果
- **信息来源消融（Ablation Study）**：
  - 实验对比了 **仅论文输入** 与 **提供全部三个来源信息** 的情况。
  - 结果显示，即使提供了全部信息，模型性能提升也微乎其微（△<0.02），说明性能瓶颈并非源于信息缺失，而是模型自身的 **长上下文推理能力有限**，难以从分散的文档中有效提取和整合信息。
- **编辑器有效性（D3）**：
  - 经过 Editor 处理的卡片质量（Bench_GPT-OSS: 4.475）远高于未经处理的原始生成草稿（Raw Baseline: 3.709），证明了编辑器在提升质量和减少幻觉方面的关键作用。

---

## 4. 关键结论和发现

### 主要发现
1. **稀疏 MoE 架构在成本-质量效率上最优**：对于大规模文档生成任务，经过优化的开放权重 MoE 模型（如 Qwen3-30B-A3B-Instruct）提供了最佳的经济回报。
2. **传统词法指标失效**：ROUGE-L 和 BERTScore 等传统指标与实际语义质量呈**反向关联**。例如，简单复制原文的模型可能获得高 ROUGE-L，但实际质量很低；而进行抽象综合的优质模型反而得分较低。这凸显了依赖 LLM-as-a-Judge 进行语义评估的必要性。
3. **完整性与质量正交**：高 Completeness 并不保证高质量。一些模型虽然填充了很多字段，但内容空洞或不准确。
4. **存在根本性的“忠实性-完整性”权衡（Faithfulness-Completeness Trade-off）**：
   - 模型为了保持高忠实性（避免幻觉），往往会遗漏约 21% 的真实信息（Completeness ≈ 0.786）。
   - 这种“精度-召回”失衡的根本原因在于模型的**长上下文推理能力不足**，而非信息本身不可用。
5. **Data Card 生成更具挑战性**：所有模型在生成 Data Card 时的表现均低于 Model Card，因为论文通常将数据集视为附属品，描述稀疏，导致生成难度更大。

### 方法的局限性
- **纯文本限制**：当前框架仅处理文本信息，忽略了图表、表格等非文本模态中的重要信息。
- **孤立任务假设**：将每个文档生成视为独立任务，未建模论文、模型、数据集之间复杂的生态系统级依赖关系。
- **潜在风险**：自动生成的文档若存在错误或遗漏，可能会在大规模复用时传播误导性信息，削弱透明度努力。

### 未来工作方向
- **多模态理解**：将图像、表格等内容纳入文档生成流程。
- **生态系统级建模**：通过图结构等方式建模 GenAI 生态中实体间的复杂关系。
- **鲁棒验证机制**：开发更强大的验证和人工监督机制，以降低自动化文档带来的风险。

</details>

---

### 4. [STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator](https://arxiv.org/abs/2604.24544)

**Authors**: Alessio Sordo, Lingxiao Du, Meeka-Hanna Lenisa, Evgeny Bogdanov, Maxim Romanovsky  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.24544v1  

#### Abstract
The increasing reliance on Large Language Models (LLMs) across diverse sectors highlights the need for robust domain-specific and language-specific evaluation datasets; however, the collection of such datasets is challenging due to privacy concerns, regulatory restrictions, and the time cost for man...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前对 **Large Language Models (LLMs)** 的评估严重依赖人工标注的真实数据集，这在实际应用中面临以下挑战：
- **隐私与合规风险**：尤其在金融、医疗等受监管领域，真实数据难以获取；
- **成本高、扩展难**：手动构建高质量数据集耗时且昂贵；
- **多语言支持不足**：现有基准多集中于英语，非英语语种资源稀缺；
- **数据污染与饱和**：静态数据集易被模型“记忆”，导致评估失真。

此外，现有自动化生成方法通常：
- 依赖已有数据进行增强或匿名化；
- 缺乏可控性和多样性；
- 难以支持跨语言、跨领域的定制化需求。

---

### 🚀 提出的新方法与思路
本文提出 **STELLAR-E** —— 一个**完全自动化的端到端系统**，用于生成高质量、可定制的合成指令-答案对（Instruction-Answer, I&A），并直接用于评估 LLM 应用。

#### 核心架构分为两个阶段：
1. **合成数据引擎（Synthetic Data Engine）**
   - 改进 **TGRT Self-Instruct 框架**，实现从零开始生成 I&A 数据；
   - 支持按需控制：语言、领域、数量、难度、多样性等维度。

2. **评估流水线（Evaluation Pipeline）**
   - 结合统计指标与 **LLM-as-a-Judge** 方法（基于改进版 G-Eval）；
   - 自动完成对生成数据质量及其作为评测基准有效性的元评估。

#### 创新机制：
- **DVE（Diversity Enhancement）**：通过嵌入空间距离过滤重复项，提升语义与表达多样性；
- **DFE（Difficulty Enhancement）**：利用 LLM 对原始指令进行对抗性改写，提高任务挑战性；
- **反馈循环（Feedback Loop）**：低质量样本由独立 LLM 提供反馈后重新生成，而非简单丢弃，显著减少数据损失。

---

### 🔍 相比现有方法的优势
| 特性 | STELLAR-E | YourBench / OmniEval / BENCHAGENTS |
|------|-----------|-------------------------------|
| 是否依赖输入文档 | ❌ 否 | ✅ 是（限制领域迁移能力） |
| 是否支持全合成生成 | ✅ 是 | ⚠️ 多为增强或转换 |
| 可控性（语言/领域/规模） | ✅ 高度可配置 | ⚠️ 有限 |
| 多语言原生支持 | ✅ 内建多语言 embedding 和生成 | ❌ 多依赖翻译 |
| 质量优化机制 | ✅ DVE + DFE + Feedback Loop | ⚠️ 主要靠过滤 |
| 端到端自动化程度 | ✅ 完整闭环：生成 → 过滤 → 评估 | ⚠️ 多需人工干预 |

> ✅ **优势总结**：STELLAR-E 实现了无需真实数据输入、高度可定制、可扩展、多语言兼容的全自动评估框架，适用于 LLMOps 中的持续集成与监控。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **真实基准数据集**：`Mintaka`（复杂、自然、多语言问答数据集）
  - 包含英文原版（`mintaka_en_real`）和专业意大利语翻译版（`mintaka_it_real`）
- **机器翻译对照组**：将 `mintaka_en_real` 自动翻译成意大利语（`mintaka_it_translated`），模拟常见做法
- **合成数据集**：
  - `mintaka_en_synthetic`：本系统生成的英文合成数据
  - `mintaka_it_synthetic`：本系统生成的意大利语合成数据

所有数据集最终均采样至 **1,500 条 I&A 对**以保证公平比较。

---

### ⚙️ 实验设置
#### 生成参数（见 Table 2）
| 参数 | 值 |
|------|----|
| G-Eval 阈值 T | 8/10 |
| Question Types (QTs) 数量 | 8 |
| 每轮采样主题数 | 5 |
| 迭代次数 | 50 |
| 每次生成指令数 | 50 |
| DVE 相似度阈值 | 0.3 |

> 总计每轮可生成约 2,500 条 I&A 对。

#### 评估模型（强弱搭配，避免自增强偏差）
| 模型 | 类型 | MMLU-Pro 得分 |
|------|------|----------------|
| **Gemini 2.5 Flash (2025)** | 强模型 | 83.6% |
| **Llama 2 Chat 13B (2023)** | 弱模型 | 25.3% |

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **ROUGE-L** | 衡量生成答案与标准答案之间的最长公共子序列，反映词汇重叠 |
| **BERTScore F1** | 基于上下文嵌入的语义相似度，适合多语言场景 |
| **Answer Relevance** | 评估回答是否紧扣问题，避免冗余或偏离 |
| **Custom G-Eval** | 改进版 LLM-as-a-Judge 框架，使用 Gemini 2.5 Pro 作为裁判模型 |
| - 评分范围 | 1–10 分（更细粒度） |
| - 温度设置 | 0（确保输出稳定） |
| - 逐项打分 | 减少锚定偏见（anchoring bias） |
| - 元评估标准 | Accuracy, Relevance, Completeness |

---

### 🆚 基线方法对比
| 方法 | 类型 | 是否依赖真实数据 | 是否支持多语言 | 是否端到端 |
|------|------|------------------|----------------|------------|
| **YourBench** | 文档驱动 QA 生成 | ✅ 是 | ❌ 有限 | ✅ 是 |
| **OmniEval** | 金融领域 RAG 评估 | ✅ 是 | ❌ 否 | ✅ 是 |
| **BENCHAGENTS** | Agent 协作生成 | ❌ 否 | ⚠️ 间接支持 | ✅ 是 |
| **STELLAR-E (Ours)** | 全合成、无文档依赖 | ❌ 否 | ✅ 原生支持 | ✅ 是 |

> 本文不仅与上述方法理念上对比，在实验中也隐含地展示了其优越性：更高的可控性、更低的成本、更强的泛化能力。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 3 & 4）

#### 在强模型上的表现（Gemini 2.5 Flash）
| 数据集 | G-Eval 分数 | 相对于 Real EN 的 Δ |
|--------|-------------|--------------------|
| Real English (`REAL EN`) | 8.17 | — |
| Synthetic EN (no DVE/DFE) | 9.43 | +12.6% |
| Synthetic EN + DVE | 9.14 | +9.7% |
| **Synthetic EN + DVE & DFE** | **8.74** | **+5.7%** ✅ 最接近 |

> ➤ **结论**：加入 DVE 和 DFE 后，合成数据与真实数据的平均差距缩小至 **+5.7%**，表明其具备相当的评估效力。

#### 在弱模型上的表现（Llama 2 Chat 13B）
| 数据集 | G-Eval 分数 | 相对于 Real EN 的 Δ |
|--------|-------------|--------------------|
| Real English | 5.69 | — |
| **Synthetic EN + DVE & DFE** | **6.78** | **+10.9%** |

> ➤ 小模型在合成数据上得分更高，说明其可能未充分挑战小模型，存在“简单化”倾向。

#### 意大利语结果（验证多语言能力）
| 数据集 | G-Eval 分数 | 相对于 Real IT 的 Δ |
|--------|-------------|--------------------|
| Real Italian | 4.28 | — |
| **Synthetic IT + DVE & DFE** | **4.35** | **+0.7%** ✅ 几乎持平！ |

> ➤ 在弱模型上，合成意大利语数据几乎与真实数据效果一致，证明其在低资源语言中的潜力。

---

### 🔬 消融实验结果（Ablation Study）
| 组件 | 影响 |
|------|------|
| **无 DVE/DFE** | G-Eval 显著偏高（+12.6%），数据过于简单 |
| **仅 DVE** | 多样性提升，但仍偏容易（+9.7%） |
| **DVE + DFE** | 显著降低分数膨胀，最接近真实分布（+5.7%） |
| **DFE 单独作用** | 成功引入复杂句式与歧义，使模型更难正确作答 |
| **DVE 过滤后 Rouge-L 下降最多** | 表明模型被迫采用不同表达方式，减少模板化输出 |

> ➤ **关键发现**：DFE 是拉近合成与真实数据差距的关键组件。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **合成数据可以逼近真实数据的评估效果**：
   - 在强模型上，STELLAR-E 生成的合成数据与真实 Mintaka 数据的 G-Eval 差距仅为 **+5.7%**；
   - 加入 DVE 和 DFE 后，能有效缓解“合成数据太简单”的问题。

2. **多语言支持表现优异**：
   - 在意大利语任务中，合成数据甚至接近真实数据水平（仅差 +0.7%），优于机器翻译版本；
   - 机器翻译数据因“translationese”现象反而降低了挑战性。

3. **模块化设计带来灵活性**：
   - 各阶段解耦，便于替换模型或插入新功能；
   - 支持训练、微调、强化学习等多种下游用途。

4. **小模型受益更多于合成数据**：
   - 弱模型在合成数据上表现相对更好，提示真实数据更具挑战性；
   - 可能由于合成数据保留了结构性线索，利于检索匹配。

---

### ⚠️ 局限性
1. **仍略低于真实数据的挑战性**：
   - 尽管已优化，真实数据对小模型仍构成更大压力，说明合成过程尚未完全复现人类思维复杂性。

2. **潜在系统性偏见**：
   - 整个流程依赖 LLM（生成 + 评估），可能存在隐性偏好或模式复制；
   - 自我增强偏差（self-enhancement bias）虽有缓解，但无法根除。

3. **文化特异性未充分验证**：
   - 虽然支持多语言，但缺乏本地母语者的人工评估来确认文化适配性。

4. **仅在一个基准上验证（Mintaka）**：
   - 泛化能力有待在更多领域（如医学、法律）进一步测试。

---

### 🔮 未来工作方向
1. **扩大元评估范围**：
   - 在更多模型家族（如 Llama、Qwen、DeepSeek）上测试，减少模型偏差影响；
   - 引入 judge ensemble 提升评估稳定性。

2. **扩展至 RAG 场景**：
   - 生成合成源文档 + 对应 I&A，实现完整的 Retrieval-Augmented Generation 评估闭环。

3. **引入人工评估**：
   - 对少量样本进行母语专家评审，分析语言自然性与文化相关性；
   - 特别针对非英语语种开展 human-in-the-loop 研究。

4. **探索动态难度调节机制**：
   - 根据被测模型能力实时调整 DFE 强度，实现个性化 benchmarking。

5. **应用于 LLMOps CI/CD 流程**：
   - 将该系统集成进自动化测试管道，实现实时性能监控与回归检测。

---

## ✅ 总结
**STELLAR-E** 是首个真正意义上**无需真实数据输入、端到端自动化、支持多语言与领域定制**的 LLM 应用评估框架。其实验证明，其生成的合成数据在质量上可媲美中等难度的真实基准，尤其在多语言场景下展现出巨大潜力。虽然目前尚不能完全替代人工构建数据，但它为快速、安全、高效的 LLM 质量保障提供了全新范式，是迈向 **automated, scalable, and responsible AI evaluation** 的重要一步。

</details>

---

### 5. [Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis](https://arxiv.org/abs/2604.23072)

**Authors**: Junyan Cheng, Kyle Richardson, Peter Chin  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.23072v1  

#### Abstract
Large language model (LLM) agents are increasingly tasked with complex real-world analysis (e.g., in financial forecasting, scientific discovery), yet their reasoning suffers from stochastic instability and lacks a verifiable, compositional structure. To address this, we introduce Analytica, a novel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Model (LLM)** 的智能体在进行复杂现实世界分析（如金融预测、科学发现）时，存在两大核心缺陷：
- **推理过程不稳定**：LLM 的生成具有随机性（stochastic instability），导致相同任务多次运行结果不一致。
- **缺乏可验证性和结构性**：传统基于文本的推理（如 Chain-of-Thought）难以形式化建模，无法有效追踪误差来源，也难以支持交互式“what-if”分析。

### 提出的新方法与思路
本文提出 **Analytica**，一种基于 **Soft Propositional Reasoning (SPR)** 的新型 LLM 智能体架构。其核心思想是将复杂分析任务重构为对一系列命题（propositions）软真值（soft truth value）的估计过程。

#### 核心创新点：
- **Soft Propositional Reasoning (SPR)**  
  将复杂问题分解为一个命题树，每个节点代表一个可验证的子命题，并赋予一个 `[0,1]` 区间的软真值（degree of belief）。最终通过合成规则（synthesis rule）从叶子节点向上聚合，计算根命题的最终真值。

- **三阶段并行 Divide-and-Conquer 架构**  
  Analytica 采用三个核心组件协同工作：
  1. **Analyzer**：将根命题递归分解为子命题树。
  2. **Grounder**：并行验证叶子节点，利用工具（如 Web Search、Jupyter Notebook）获取证据并打分。
  3. **Synthesizer**：自底向上聚合子命题的软真值，计算父命题的最终得分。

- **误差最小化的理论框架**  
  借助 **Mean Squared Error (MSE)** 分解为 **Bias** 和 **Variance**，系统性地优化推理质量：
  - **降低 Bias**：通过深度分解使叶子命题更简单，易于准确判断。
  - **降低 Variance**：使用 **robust linear synthesis rule** 对多个子命题的估计进行加权平均，平滑随机噪声。

- **高效的交互式场景分析（Resynthesis）**  
  支持用户手动修改任意节点的真值或假设，系统仅重新计算受影响的分支，实现快速“what-if”分析。

### 相比现有方法的优势
| 方面 | 传统方法（如 CoT, ToT） | Analytica |
|------|------------------------|---------|
| 推理结构 | 线性或图状文本流 | 结构化命题树 |
| 可解释性 | 黑箱式文本推理 | 显式的软真值传播 |
| 稳定性 | 高方差，结果波动大 | 低方差，结果稳定 |
| 可扩展性 | 深度增加导致指数级延迟 | 近线性时间复杂度 |
| 成本效率 | 依赖昂贵模型 | 支持轻量模型 + 工具增强 |

---

## 2. 核心实验方法和设置

### 数据集
在 **736 个真实世界的经济、金融与政治预测任务** 上进行评估，涵盖两类任务：
- **Financial Market Tasks**：对股票、指数、商品等资产做出“长期持有 vs 做空”的一年期预测。
- **Predictive Market Tasks**：来自 Kalshi 和 Polymarket 的预测市场选项（如“谁将赢得2024年美国总统选举？”）。

所有事件的解决日期均在模型知识截止日期（2024年6月1日）之后，确保为前瞻性预测。

### 实验设置
- **基础模型**：主要使用 `o3-2025-04-16` 模型，温度设为 0.1 以减少随机性。
- **Grounder 类型**：
  - **Basic Search**：仅使用 Web Search。
  - **Deep Research**：OpenAI 提供的深度研究能力。
  - **Jupyter Notebook Grounder**：本文提出的高级 Grounder，在 Jupyter 环境中执行代码、调用 API、生成报告。
- **Synthesis Rules 对比**：
  - **Vanilla**：直接由 LLM 输出 `p_true`。
  - **Simple Logic**：使用模糊逻辑算子（AND/OR/NOT）组合子命题。
  - **Linear**：使用线性加权模型：`p_true = β₀ + Σβᵢ·p_true_i`。

### 评估指标
| 指标 | 含义 |
|------|------|
| **Accuracy** | 正确选择最优选项的比例（Top-1 准确率） |
| **Soft Score** | 所有选项 `p_true` 加权后的期望回报 |
| **Hard Score** | 最高 `p_true` 选项的实际回报 |
| **Brier Score (BS)** | 预测分布的 MSE，衡量校准性 |
| **Variance (Var.)** | 多次运行下 Hard Score 的方差，衡量稳定性 |
| **Confidence** | 平均最高 `p_true`，反映模型自我置信度 |
| **Cost & Time** | API 花费与响应时间，衡量效率 |

### 基线方法对比
- **独立基线**：Random、Basic Search、Deep Research、Jupyter Notebook。
- **结构化推理基线**：Tree-of-Thoughts (ToT)、Graph-of-Thoughts (GoT)、Forest-of-Thoughts (FoT)，均基于 Basic Search 实现。
- **Analytica 变体**：结合不同 Grounder 与 Synthesis Rule 的组合。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）
| 方法 | Accuracy (%) | Improvement (%) | Variance (%) | Cost ($) | Time (min) |
|------|------------|----------------|-------------|----------|-----------|
| **Basic Search** | 53.94 | — | 10.30 | 0.02 | 0.54 |
| **+ Tree of Thoughts** | 60.19 | +11.59 | 9.21 | 0.28 | 6.55 |
| **Deep Research** | 63.04 | — | 9.28 | 4.02 | 7.60 |
| **+ Analytica-L (Linear)** | **71.06** | **+12.72** | **6.02** | 14.10 | 30.01 |
| **Jupyter Notebook** | 61.96 | — | 12.28 | 0.07 | 2.61 |
| **+ Analytica-L** | **70.11** | **+13.15** | **7.28** | **1.36** | **14.15** |

> ✅ **Analytica-L** 在 Deep Research 基础上提升 **12.72%** 准确率，达到 **71.06%**，同时将方差降至 **6.02%**，显著优于所有基线。

### 与基线方法的对比结果
- **相比结构化推理方法**（ToT, GoT, FoT）：
  - Analytica 在准确率上全面超越，且稳定性更高（更低方差）。
  - 特别是在 **Predictive Markets** 等高不确定性领域，提升最为显著（见 Table 5）。
- **相比 Deep Research**：
  - 即使使用相同的 Grounder，Analytica 仍能带来 **+8.02%** 的绝对提升。
  - 表明 **SPR 架构本身** 是性能增益的关键。

### 消融实验结果
#### （1）Grounder 消融（Table 3）
- **Jupyter Notebook Grounder** 展现出极高的成本效益：
  - 准确率达到 **70.11%**，仅比 Deep Research + Analytica 低 **0.95%**。
  - 但成本仅为 **90.35%**，耗时减少 **52.85%**。
  > 💡 表明 **工具增强的轻量 Grounder** 可替代昂贵的黑盒研究服务。

#### （2）Synthesis Rule 消融（Table 2）
- **Linear Rule** 表现最佳（71.06%），稳定性最高（Var=6.02%）。
- **Simple Logic Rule** 效果最差（+4.22%），且对噪声极为敏感（见 Figure 5）。
- **Vanilla Rule** 居中，但不如 Linear 稳健。

#### （3）开放权重模型与科学领域泛化（Table 6 & 7）
- 在 **Open-weight 模型**（如 DeepSeek-v3, GLM-4.6）上，Analytica 仍能带来 **+3–15%** 的提升。
- 在 **科学声明验证**（Matter-of-Fact benchmark）上，GPT-5-mini + Analytica 达到 **73%** 准确率，优于基线。
  > 🌐 表明 Analytica 具有良好的 **跨模型与跨领域适应性**。

---

## 4. 关键结论和发现

### 主要发现
1. **SPR 架构显著提升 LLM 分析的准确性与稳定性**  
   通过将复杂推理转化为软命题的估计与合成，实现了 **15.84%** 的平均准确率提升，并大幅降低方差。

2. **Linear Synthesis Rule 是鲁棒性的关键**  
   理论分析表明，线性规则具有 **恒定敏感性、平滑噪声、优雅退化** 三大特性，使其在噪声环境下表现远超逻辑规则。

3. **Jupyter Notebook Grounder 实现高效性价比**  
   利用代码执行与 API 调用进行定量分析，可在 **极低成本下接近顶级模型性能**，为实际部署提供可行路径。

4. **高度可扩展与可交互**  
   - 支持递归并行执行，分析节点数增长 **54倍** 时，计算时间仅增加 **12倍**（近线性）。
   - 支持 **Resynthesis**，实现快速“what-if”分析，适用于决策支持系统。

### 方法的局限性
1. **子命题独立性假设**：框架在子命题高度相关时性能可能下降。
2. **Synthesizer 错误传播风险**：若 Synthesizer 学习到错误的权重（β），可能导致系统性偏差。
3. **静态 Grounder 选择**：目前对所有叶子节点使用同一 Grounder，未实现动态路由（dynamic routing）。
4. **依赖高质量工具接口**：Jupyter Grounder 的性能受限于 API 的可用性与稳定性。

### 未来工作方向
- 引入 **Probabilistic Graphical Models (PGMs)** 或 **Neuro-symbolic 方法** 来显式建模子命题间的依赖关系。
- 开发 **Adaptive Grounder Routing** 机制，根据不同命题类型自动选择最优 Grounder。
- 探索 **端到端训练** Synthesizer，使其学习更优的合成策略。
- 将 Analytica 应用于更多高风险领域，如医疗诊断、政策制定与机器人规划。

--- 

> 🔚 **总结**：Analytica 提出了一种**结构化、可验证、可扩展**的 LLM 分析范式，通过 **Soft Propositional Reasoning** 与 **divide-and-conquer** 架构，在保持高效率的同时显著提升了推理的准确性与稳健性，为构建可靠 LLM 智能体提供了重要实践路径。

</details>

---

### 6. [Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](https://arxiv.org/abs/2604.23623)

**Authors**: Zichuan Fu, Xian Wu, Guojing Li, Yejing Wang, Yijun Chen, Zihao Zhao, Yixuan Luo, Hanyu Yan, Yefeng Zheng, Xiangyu Zhao  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.23623v1  

#### Abstract
Recent advancements in large language models (LLMs) have catalyzed the rise of reasoning-intensive inference paradigms, where models perform explicit step-by-step reasoning before generating final answers. While such approaches improve answer quality and interpretability, they incur substantial comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Models (LLMs)** 的 **reasoning-intensive inference** 范式（如 DeepSeek-R1）虽然能显著提升推理质量和可解释性，但其生成的 **长推理链（thousands of tokens）** 导致极高的计算开销（inference latency 和 operational cost），严重阻碍了在实时或预算受限场景下的部署。

此外，已有优化方法（如 Reinforcement Fine-Tuning, RFT）存在两大缺陷：
- 需要对 LLM 进行持续训练，可能损害模型通用能力；
- 不适用于仅提供 API 接入的闭源模型。

### 🚀 提出的新方法：Tandem 框架
提出一种新型 **LLM-SLM 协作范式**，灵感来源于“导师-实习生”（mentor-intern）关系：

- **LLM 作为战略导师（strategic coordinator）**：生成轻量级、高价值的 **Thinking Insights**（思考洞察），而非完整推理链。
- **SLM 作为执行实习生（agile intern）**：接收这些洞察，并完成详细推理与最终答案生成。
- 引入 **cost-aware termination mechanism**：由 SLM 动态判断当前 LLM 提供的指导是否已足够，决定是否提前终止 LLM 的生成过程。

#### 四类 Structured Thinking Insights
为结构化地提取 LLM 的核心推理内容，设计了以下四种关键洞察：
1. **Goal**：明确问题目标与约束；
2. **Planning**：制定高层策略与子问题分解；
3. **Retrieval**：召回相关知识、公式或定义；
4. **Action**：执行具体逻辑步骤或计算。

这些洞察通过结构化 prompt 引导 LLM 输出，确保信息紧凑且可解析。

### 🔍 相比现有方法的优势
| 维度 | Tandem | 传统方法 |
|------|--------|---------|
| **无需训练 LLM** | ✅ 支持 API 模型协作 | ❌ 多数需微调或强化学习 |
| **计算效率** | ⬇️ 降低约 40% 成本 | ⬆️ 完整推理链成本高 |
| **性能表现** | ✅ 超越单独 LLM | ⚠️ 缩短推理链常导致性能下降 |
| **动态适应性** | ✅ 自适应控制指导深度 | ❌ 固定长度或静态路由 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **MATH** (Hendrycks et al., 2021)：5,000 测试样本，涵盖 7 个数学领域（代数、几何等），难度等级从 1 到 5。
- **GSM8K** (Cobbe et al., 2021)：1,000 小学数学题，强调多步推理。
- **HumanEval** (Chen et al., 2021)：164 个编程任务，用于跨域泛化测试。

### ⚙️ 实验设置
- **模型组合**：
  - LLMs: DeepSeek-32B, Qwen3-32B, GPT-4o-mini, gpt-oss-120b（API）
  - SLMs: DeepSeek-7B, Qwen3-8B
- **推理模式**：
  - LLM 在 thinking mode 下运行（temperature=0, top_p=1.0）
  - 分三个 effort level（低/中/高）逐步生成 insights
- **分类器训练**：
  - 使用 MLP 分类器预测当前指导是否“充分”
  - 输入特征：基于 SLM 的 **token-level perplexity** 与 **entropy**
  - 训练数据来自 MATH 训练集，标签为 SLM 是否答对

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy** | 正确解答的比例 |
| **Inference Length** | 总生成 token 数（LLM + SLM） |
| **Computational Cost** | 近似 TFLOPs：<br>$ \text{Cost} = |\theta_L| \cdot L_L + |\theta_S| \cdot (L_L + L_S) $ |

### 🔀 基线方法对比
- **Single Model**：仅用 LLM 或 SLM 推理
- **Fixed-Length Collaboration**：固定 LLM 生成 100/500/1000 tokens 后交由 SLM
- **Budget Forcing**：截断 LLM 推理长度以节省成本
- **LLM Cascade**：二分类路由，选择走 SLM 或 Full LLM
- **API-based Baselines**：验证 Tandem 对闭源模型的兼容性

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（MATH 数据集）

| 方法 | 平均 Accuracy (%) | 计算成本 (TFLOPs) | 相对 LLM 成本降幅 |
|------|-------------------|--------------------|------------------|
| SLM-only (7B) | 77.14 | 38.25 | — |
| LLM-only (32B) | 80.90 | 168.35 | — |
| Tandem (7B+32B) | **83.46** | **99.72** | **↓ 40.8%** |

> ✅ Tandem 不仅将准确率提升了 **+2.56%**，还降低了近 **40% 的计算成本**。

#### 更细粒度结果（Table 1）：
- 在所有 7 个数学子科目上，Tandem 均取得 **最高准确率**
- 成本平均降低 **41%**（范围：29%-48%）
- 即使在最难的 “Intermediate Algebra” 上仍实现 **+5% 提升**

### 🔁 跨模型家族泛化（RQ2）
| 协作组合 | MATH Acc (%) | 成本 (TFLOPs) |
|----------|---------------|----------------|
| DeepSeek-7B + Qwen3-32B | **79.96** | 58.06 |
| Qwen3-32B 单独 | 69.50 | 193.41 |

> ✅ 跨家族协作依然有效，证明 **structured insights 具有良好的跨模型可理解性**。

### 🧪 消融实验与关键发现

#### （1）指导长度 vs. 性能（Figure 5）
- 即使仅提供 **200 tokens 的轻量指导**，也能显著超越 SLM-only
- 性能随指导长度增加而上升，但存在波动 → 支持 **adaptive guidance** 必要性

#### （2）模型大小匹配（Table 3）
- 最佳协作发生在 **能力差距适中** 的模型之间（如 7B + 32B）
- 若 SLM 过小（如 1.5B），难以理解复杂指导，增益有限
- 若差距太小（如 14B + 32B），收益边际递减

#### （3）非思考模式下的有效性（Table 7）
即使 LLM 不启用 thinking mode，Tandem 仍能：
- 准确率提升至 **82.56%**（vs. 79.98% 单独 32B）
- 成本降低 **36.7%**

> 表明框架不依赖特定推理范式，具有广泛适用性。

#### （4）跨域迁移能力（Table 5）
- Sufficiency classifier **直接迁移到 HumanEval（代码生成）而不重训**
- Tandem 达到 **85.37%** 准确率，优于最佳固定预算基线（83.54%）

> ✅ 表明 **perplexity/entropy 特征捕捉的是 domain-agnostic 的置信信号**

#### （5）与其他高效方法对比（Table 6）
| 方法 | Accuracy (%) | Cost (TFLOPs) |
|------|--------------|----------------|
| LLM Cascade | 82.60 | 95.33 |
| Budget Forcing | 82.18 | 108.74 |
| **Tandem** | **83.46** | **99.72** |

> Tandem 在保持竞争力成本的同时实现了 **最高精度**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **高质量轻量指导 > 完整冗长推理**  
   LLM 提供的 **structured thinking insights** 可作为 full reasoning chain 的高效替代，既能保留认知深度，又能大幅降低成本。

2. **动态判断机制至关重要**  
   cost-aware continual judgment 能自适应分配计算资源：简单问题早停，复杂问题深入，实现 **精度与效率的最佳平衡**。

3. **协作需合理的能力配比**  
   SLM 必须具备一定理解能力才能有效利用 LLM 指导；过弱则无法受益，过强则冗余。

4. **跨模型、跨任务、跨部署方式均有效**  
   - 支持不同 LLM 家族（DeepSeek/Qwen/GPT）
   - 支持 API 接入模型
   - 分类器可在 MATH 上训练后直接用于 HumanEval

5. **实际部署优势明显**  
   - 延迟降低 **1.8×**（Table 13）
   - 成本更低，尤其适合 token 计费的 API 场景

---

### ⚠️ 局限性
1. **领域泛化待验证**  
   当前实验集中于数学与代码生成，尚未验证在常识推理、开放问答等领域的效果。

2. **仍需标注数据训练 sufficiency classifier**  
   尽管支持跨域迁移，但仍需至少一个领域的 labeled data 进行初始训练。

3. **协作模式较简单**  
   当前为固定“一对一”协作，未探索更复杂的多模型协同或角色动态切换机制。

---

### 🔮 未来工作方向
- 探索 **zero-shot 或 self-supervised sufficiency detection**，减少监督需求
- 构建 **multi-agent Tandem**，引入多个专家模型进行分工协作
- 扩展至 **vision-language、robotics planning** 等多模态任务
- 研究 **feedback-driven insight refinement**，形成闭环优化

---

> 🔗 开源地址：[https://github.com/Applied-Machine-Learning-Lab/ACL2026_Tandem](https://github.com/Applied-Machine-Learning-Lab/ACL2026_Tandem)

</details>

---

### 7. [CUDA Kernel Optimization and Counter-Free Performance Analysis for Depthwise Convolution in Cloud Environments](https://arxiv.org/abs/2604.25422)

**Authors**: Huriyeh Babak, Melanie Schaller  
**Category**: cs.DC  
**Published**: 2026-04-29  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.25422v1  

#### Abstract
Efficient GPU execution of convolution operators is governed by memory-access efficiency, on-chip data reuse, and execution mapping rather than arithmetic throughput alone. This paper presents a controlled operator-level study of CUDA kernel optimization for the depthwise convolution used in Structu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CUDA Kernel Optimization and Counter-Free Performance Analysis for Depthwise Convolution in Cloud Environments

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在云环境（如 Kaggle、Google Colab、AWS）中，由于缺乏对硬件性能计数器（hardware performance counters）和高级分析工具（如 Nsight Compute）的访问权限，难以进行深入的 GPU 内核级性能分析。本文旨在解决以下核心问题：

> **如何在受限的云环境中，不依赖硬件性能计数器的情况下，实现可复现、细粒度的 CUDA kernel 性能分析？**

此外，研究聚焦于 **depthwise convolution** 在 **S4ConvD 模型**中的执行效率，特别关注前向传播与反向传播路径之间的性能差异。

---

### 🚀 提出的新方法与创新思路

#### （1）提出了一种 **cloud-compatible, counter-free performance analysis methodology**
- **无需硬件性能计数器**，仅通过标准 CUDA API 和分析建模即可获得架构级洞察。
- 方法整合了多个技术组件：
  - `CUDA-event` 时间测量
  - 执行路径分解（execution-path decomposition）
  - 分析性内存流量建模（analytical memory-traffic modeling）
  - 有效带宽估计（effective bandwidth estimation）
  - Roofline 分析

> 这使得开发者可以在普通云平台（无特权访问）下进行类似 profiler 的性能诊断。

#### （2）构建了一个**受控的操作符级评估框架**
- 固定模型（S4ConvD）、数据集（GEPIII）、训练配置和算子定义
- 只改变 CUDA kernel 实现方式，从而**精确归因性能变化到执行映射和内存层次利用**

#### （3）首次系统性地进行了 **execution-path-aware 性能分析**
- 明确区分 forward、input-gradient 和 weight-gradient 路径
- 揭示不同路径的优化极限存在显著差异，尤其是 reduction-dominated weight-gradient 成为瓶颈

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本论文方法 |
|------|--------|-----------|
| 工具依赖 | 需要 Nsight Compute / HW counters | 仅需 CUDA events，适用于所有云平台 |
| 控制变量 | 多数研究修改模型或融合算子 | 完全固定外部因素，只变 kernel |
| 分析深度 | 常见端到端加速报告 | 细分至每个执行路径的性能瓶颈 |
| 可复现性 | 受限于本地硬件和驱动版本 | 云端标准化环境提升可复现性 |

> ✅ **优势总结**：该方法将“限制”转化为“机会”，使云环境成为标准化、可复现性能分析的理想平台。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用 **ASHRAE Great Energy Predictor III (GEPIII)** 数据集
  - 包含建筑能耗与气象特征
  - 特征维度低、序列长度固定（L=48），适合控制变量研究
  - depthwise convolution 成为主要计算瓶颈

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| GPU | NVIDIA Tesla P100-PCIE-16GB（Pascal 架构，compute capability 6.0） |
| CUDA 平台 | 标准 CUDA C++，使用 `cudaEvent_t` 测量时间 |
| 精度 | float32 |
| Batch Size | 16,384 |
| Latent Dimension H | 128 |
| Sequence Length L | 48 |
| Kernel Size K | 48 |

> 所有实验排除数据加载、optimizer 更新、host-device transfer 等开销，专注于 convolution kernel 自身性能。

---

### 📈 评估指标

| 指标 | 说明 |
|------|------|
| `Kernel Runtime`（ms） | 主要性能指标，分别测量 forward、BWD_in、BWD_k |
| `Epoch Time`（s） | 端到端训练速度参考 |
| `Effective Bandwidth`（GB/s） | 基于运行时间和理论内存流量估算 |
| `Roofline Plot` | 展示 arithmetic intensity vs. achieved performance |
| `Speedup` | 相对于 naive baseline 的加速比 |

---

### 🆚 基线方法对比

共实现并比较四种 CUDA kernel 变体：

| Kernel Variant | 关键特性 |
|----------------|---------|
| **Naive CUDA** | 最基础实现，每线程一个输出，无共享内存，重复访存严重 |
| **GMC (Global-Memory Coalesced)** | 优化 warp-level 内存合并访问，减少事务开销 |
| **Shared-Memory Cache Blocked** | 利用 shared memory 缓存输入和 kernel 权重，支持片上复用 |
| **Warp-Tiled Execution** | 每个 warp 处理一个 (b,h) 实例，最大化 on-chip data reuse |

> ❗注意：PyTorch 实现用于数值验证，**不作为架构级性能基线**，因其底层调用 cuDNN 等黑盒库。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table II）

| Method | FWD (ms) | BWD_in (ms) | BWD_k (ms) | Conv Total (ms) | Epoch (s) |
|--------|----------|-------------|------------|------------------|------------|
| Naive CUDA | 29.97 | 30.25 | 73.26 | **133.47** | 44.82 |
| GMC | 28.23 | 28.78 | 49.64 | 106.65 | 40.31 |
| Shared | 16.36 | 16.03 | 34.17 | 66.57 | 36.91 |
| **Warp-tiled** | **10.46** | **10.61** | **19.91** | **40.99** | **34.74** |

---

### 🔁 对比结果与加速比

| 指标 | 加速效果 |
|------|--------|
| **Kernel-level Conv Speedup** | Warp-tiled 较 naive 实现快 **3.26×**（133.47 → 40.99 ms） |
| **End-to-End Training Speedup** | Epoch 时间从 44.82s → 34.74s，仅 **1.29×** 加速 |
| **GMC 提升有限** | 仅 1.25× kernel 加速，表明 coalescing 单独作用小 |
| **Shared Memory 改进明显** | 减少冗余访存，带来显著收益 |
| **Warp-tiled 表现最佳** | 充分利用 warp-centric 设计 + full on-chip staging |

---

### 🔍 消融实验与路径分析（Execution-Path Breakdown）

| 执行路径 | 性能表现 | 观察结论 |
|--------|--------|--------|
| **Forward** | 从 ~30ms → ~10.5ms（~2.9×） | 显著受益于 locality 和 data reuse |
| **Input Gradient (BWD_in)** | 同样下降至 ~10.6ms（~2.8×） | 类似 forward，属 throughput-oriented |
| **Weight Gradient (BWD_k)** | 从 73.26ms → 19.91ms（**3.68× 相对加速**） | 虽相对提升最大，仍是绝对最慢部分 |

> ⚠️ 尽管 weight-gradient kernel 得到最多优化收益，但由于其本质是 reduction-dominated，仍构成整体瓶颈。

---

### 📉 Counter-Free Effective Bandwidth 估计（Table III）

| Variant | Eff. BW (GB/s) | Peak Util. |
|--------|----------------|-----------|
| Naive CUDA | N/A | — |
| GMC | ~42 | ~6% |
| Shared | ~75 | ~10% |
| **Warp-tiled** | **~115** | **~16%** |

> ➕ 趋势清晰：更好的 on-chip reuse → 更高有效带宽 → 更低 runtime  
> ❌ 仍远低于 P100 的峰值 732 GB/s，符合 memory-bound 预期

---

### 📈 Roofline 分析结论（Fig. 10）
- 所有变体均位于 **memory-bound 区域**，远离 compute roof
- Naive kernel arithmetic intensity 最低（冗余访存多）
- Shared 和 Warp-tiled 明显右移（arithmetic intensity ↑），说明数据复用改善
- Weight-gradient 路径即使优化后仍受限于同步与累加开销

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **内存访问效率 > 算术吞吐量**
   - 对 memory-bound operator 来说，减少数据移动比提高 ALU 利用更重要
   - 单纯 global-memory coalescing 效果有限（仅 1.25×）

2. **On-chip data reuse 是性能飞跃的关键**
   - shared memory staging 和 warp-tiled 设计通过消除冗余 global memory 访问，实现高达 3.26× kernel 加速

3. **优化效果高度依赖执行路径**
   - Forward 和 BWD_in 明显受益于 locality 优化
   - BWD_k 受限于 reduction 结构，始终是瓶颈

4. **Kernel-level speedup ≠ End-to-end speedup**
   - kernel 加速 3.26×，但 epoch 仅提速 1.29×
   - 原因：非 kernel 开销（framework overhead、synchronization、optimizer）占比上升

5. **Occupancy 不等于高性能**
   - 高占用率 kernel 若存在冗余访存，依然低效
   - 性能由 memory efficiency 主导，而非 thread 数量

6. **Counter-free analysis 可行且有效**
   - 仅靠 CUDA events + analytical modeling 就能识别主要瓶颈
   - 可在无特权访问的云环境中复现高质量分析

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **无法获取真实硬件事件** | 如 L1 miss rate、warp stall reason 等仍不可知 |
| **Bandwidth 为估算值** | Effective BW 是逻辑推导，非实测 DRAM throughput |
| **特定于 depthwise 结构** | 当前分析集中于 S4ConvD 中的一维 depthwise conv |
| **未引入算法重构** | 如 hierarchical reduction 或 kernel fusion 以突破 BWD_k 瓶颈 |

---

### 🔮 未来工作方向

1. **算法层面突破 weight-gradient 瓶颈**
   - 探索更高效的 reduction 策略（如 tree-based、fused accumulation）
   - 引入 mixed-precision accumulation 降低通信成本

2. **扩展 counter-free 方法至更多算子**
   - 应用于 multi-kernel pipeline、attention、state-space models 等复杂结构

3. **自动化分析流程**
   - 构建自动化的 performance diagnosis workflow，集成到 CI/CD 或模型部署中

4. **跨平台可移植性验证**
   - 在 Ampere、Hopper 架构上验证方法通用性

5. **推动云原生性能分析标准**
   - 倡导建立不依赖专有工具的开放、可复现 GPU 分析范式

---

## ✅ 总结一句话

> 本文证明，在云环境下即使没有硬件性能计数器，也能通过 **counter-free methodology** 实现精准的 CUDA kernel 性能分析；并通过系统性的 operator-level 优化揭示：**减少冗余数据移动** 和 **按执行路径差异化优化** 是提升 memory-bound 卷积性能的核心，而 **weight-gradient 的 reduction 开销** 是最终瓶颈。

</details>

---

### 8. [PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks](https://arxiv.org/abs/2604.23148)

**Authors**: Tianlong Yu, Yang Yang, Ziyi Zhou, Jiaying Xu, Siwei Li, Tong Guan, Kailong Wang, Ting Bi  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.23148v1  

#### Abstract
The emerging threat of AR-LLM-based Social Engineering (AR-LLM-SE) attacks (e.g. SEAR) poses a significant risk to real-world social interactions. In such an attack, a malicious actor uses Augmented Reality (AR) glasses to capture a target visual and vocal data. A Large Language Model (LLM) then ana...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **AR-LLM** 的社会工程攻击框架（如 SEAR）在实际应用中面临两大瓶颈：

- **Cold-start personalization（冷启动个性化）**：依赖 **Retrieval-Augmented Generation (RAG)** 进行目标画像构建，导致首次交互时存在显著延迟（约 43.3 秒），破坏对话流畅性。
- **Static attack strategies（静态攻击策略）**：采用固定阶段的手工设计策略（如“先破冰、再建立信任”），无法根据目标实时反应动态调整，降低说服力和真实性。

### 提出的新方法与创新
作者提出 **PhySE** —— 一个融合心理学理论的实时 AR-LLM 社会工程攻击框架，具备两个核心创新模块：

#### （1）VLM-Based Social-Context Training（基于视觉语言模型的社会上下文训练）
- 利用 **Parameter-Efficient Fine-Tuning (PEFT)** 和 **LoRA** 对 **LLaVA-v1.5-7B** 进行微调，将社交上下文知识内化到 **Vision-Language Model (VLM)** 中。
- 在推理阶段无需重复检索外部数据库，实现 **on-the-fly profile generation**，大幅减少冷启动延迟。
- 引入 **cross-modal contrastive alignment**（跨模态对比对齐）优化图像与文本描述的一致性，提升画像准确性。

#### （2）Adaptive Psychological Agent（自适应心理代理）
- 设计一个基于心理学理论的 **routing agent**，依据当前交互状态动态选择三类策略路径：
  - **Warmth/Rapport**（亲和力建立）
  - **Credibility/Commitment**（可信度强化）
  - **Motivation/Action**（行动引导）
- 基于 **Stereotype Content Model (SCM)** 和 **Trust & Influence Model** 构建 **latent trust state**，量化目标信任水平，并据此决定是否升级请求强度。
- 使用 **Leaky Integrator** 动态更新信任值 $ T_t $，结合响应特征（如回应积极性、犹豫程度）进行策略路由。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | Profile 生成延迟从 43.3s 降至 10.5s，消除冷启动瓶颈 |
| **真实性** | 动态策略切换更符合真实人际互动节奏，避免过早施压引发怀疑 |
| **可控性** | 显式的心理状态建模使攻击过程更具解释性和可干预性 |
| **鲁棒性** | 对非预期用户反应具有更强适应能力，减少脚本断裂风险 |

---

## 2. 核心实验方法和设置

### 数据集
- **PhySE Dataset**：由作者构建并公开发布，包含：
  - **360 条标注对话**，来自 **60 名参与者** 在真实社交场景中的交互（如咖啡馆、社交活动）。
  - 多模态数据流：AR glasses 捕获的视频、音频、环境元数据。
  - 公开社交痕迹：用于个性化的公开资料（文本、图片、短视频）。
  - 后续调查问卷与路由记录（turn-level strategy decisions）。
- 所有研究均通过 **IRB 审批**，数据匿名化处理，符合伦理规范。

### 实验设置
- **硬件平台**：
  - AR 设备：RayNeo X2 AR Glasses（支持实时音视频采集）
  - 推理服务器：NVIDIA RTX 4090 + Intel Platinum 8352 CPU
- **模型配置**：
  - Base Model: LLaVA-v1.5-7B
  - VLM 微调：LoRA (r=128, α=256)，CLIP ViT-L/14 视觉编码器
  - Agent 控制：ReAct-style reasoning loop

### 评估指标
| 类型 | 指标 |
|------|------|
| **用户体验质量** | Social Experience Score（5 分制 Likert 量表） |
| **攻击有效性** | 四项行为诱导成功率：<br>• Photo Link（点击共享链接）<br>• Social App（添加社交好友）<br>• SMS（打开短信）<br>• Phone Call（接听电话） |
| **系统性能** | Latency（最小、最大、P90、平均延迟） |
| **主观感知维度** | 包括 Relevance, Naturalness, Sincerity, Depth 等 11 个维度评分 |
| **消融分析** | 分别移除 VLM 优化或 Psychological Agent 模块后的性能变化 |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Basic Conversation** | 无技术辅助的自然对话 |
| **Naive AR + LLM** | 使用 AR 感知 + 多模态 LLM，但无策略控制 |
| **SEAR** | 当前主流 AR-LLM-SE 框架，依赖 RAG 和固定阶段策略 |
| **PhySE (Full)** | 完整框架（VLM + Adaptive Agent） |
| **PhySE w/o VLM** | 移除 VLM 社会上下文训练 |
| **PhySE w/o Agent** | 移除心理路由代理 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）社交体验得分（Social Experience Score）
| 方法 | 平均分 (E[Score]) | 标准差 (σ) | 5分占比 |
|------|------------------|-----------|--------|
| Basic Conversation | 3.03 | 1.30 | 25.0% |
| Naive AR + LLM | 4.13 | 0.72 | 33.3% |
| SEAR | 4.73 | 0.51 | 76.7% |
| **PhySE (Ours)** | **4.83** | **0.37** | **83.3%** |

> ✅ PhySE 达到最高平均分且方差最低，表明其不仅效果最好，且用户体验最稳定。

#### （2）攻击有效性（Social-Engineering Effectiveness）
在四项诱导任务上，PhySE 均优于 SEAR，尤其在高摩擦渠道（SMS、Phone Call）提升明显：

- **Photo Link**: ~+5%
- **Social App**: ~+7%
- **SMS**: ~+12%
- **Phone Call**: ~+15%

> 🔍 表明 PhySE 更能建立深层信任，促成高风险行为。

#### （3）延迟表现（Latency Comparison）
| 模块 | 方法 | 平均延迟 | P90 延迟 |
|------|------|---------|--------|
| Multimodal LLM | SEAR | 43.3 s | 52.7 s |
| Multimodal LLM | **PhySE** | **10.5 s** | **19.7 s** |
| Social Agent | SEAR | 2.8 s | 4.0 s |
| Social Agent | PhySE | 5.8 s | 6.1 s |

> ⏱️ PhySE 将 profile 生成延迟降低 **75.7%**，虽 agent 推理稍慢，但波动极小（4.8–6.3s），更适合自然对话节奏。

#### （4）消融实验（Ablation Study）
| 配置 | 社交体验得分 | 下降幅度 |
|------|--------------|----------|
| Full PhySE | 4.83 ± 0.37 | — |
| w/o Trained VLM | 2.93 ± 1.14 | ↓39.3% |
| w/o Psychological Agent | 3.00 ± 1.13 | ↓37.9% |

> 📉 任一组件缺失都会导致性能断崖式下降，证明两个模块缺一不可。

---

## 4. 关键结论和发现

### 主要发现
1. **理论驱动的心理适应机制显著提升 AR-LLM 社会工程攻击的有效性**：
   - 基于 SCM 和 Trust-Influence Model 的动态策略路由比固定脚本更具说服力。
2. **VLM 内化社会上下文知识可有效解决冷启动问题**：
   - 无需在线 RAG 检索即可快速生成连贯、一致的目标画像。
3. **PhySE 实现了高效、真实、稳定的面对面操纵能力**：
   - 在多个通信渠道上诱导成功率更高，用户体验更自然。
4. **多模态感知 + 心理建模是下一代社会工程攻击的关键范式**：
   - 不仅“知道你是谁”，还能“判断你现在信不信我”。

### 方法的局限性
- **依赖高质量公开社交数据**：若目标数字足迹稀少，VLM 画像精度可能下降。
- **心理模型仍为简化抽象**：latent trust state 是标量估计，未完全捕捉复杂情感动态。
- **现实部署成本较高**：需高性能边缘计算支持，目前难以完全本地化运行。
- **伦理争议大**：尽管用于防御研究，但技术本身极易被滥用。

### 未来工作方向
1. **轻量化模型部署**：探索小型化 VLM 与 agent，适配移动端 AR 设备。
2. **增强反检测能力**：研究如何规避目标对 AR 设备或异常对话模式的警觉。
3. **跨文化心理建模**：扩展 Trust-Influence Model 至不同文化背景下的说服策略。
4. **防御机制开发**：
   - 实时 **behavior-level risk detection**
   - 用户端 **intervention alerts**
   - 政策层面推动 **AR sensing transparency regulations**

---

> 🔗 **代码与数据集已开源**：[https://github.com/2192537130/PhySE](https://github.com/2192537130/PhySE)  
> 🛑 **警告**：该研究揭示了 AR + LLM 技术组合的巨大安全威胁，亟需同步发展检测、防护与监管体系。

</details>

---

### 9. [Backtranslation Augmented Direct Preference Optimization for Neural Machine Translation](https://arxiv.org/abs/2604.25702)

**Authors**: Mehrdad Ghassabi, Spehr Rajabi, Hamidreza Baradaran Kashani, Sadra Hakim, Mahshid Keivandarian  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.25702v1  

#### Abstract
Contemporary neural machine translation (NMT) systems are almost exclusively built by training on supervised parallel data. Despite the tremendous progress achieved, these systems still exhibit persistent translation errors. This paper proposes that a post-training paradigm based on reinforcement le...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Backtranslation Augmented Direct Preference Optimization for Neural Machine Translation》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Neural Machine Translation (NMT)** 模型依赖大规模平行语料进行监督训练，容易产生“**translationese**”现象——即译文虽语法正确但不自然、缺乏流畅性和语义保真度。这些系统在低资源场景下表现更差，且难以通过传统微调进一步提升质量。

此外，现有的基于 **Reinforcement Learning (RL)** 或 **Direct Preference Optimization (DPO)** 的后训练方法在机器翻译中应用有限，尤其缺乏对偏好数据质量的有效控制机制。

### 🚀 提出的新方法与创新思路
本文提出了一种新颖的 **DPO-based 后训练框架**，其核心创新在于：

- **Backtranslation-Augmented Preference Pair Construction**  
  利用 **backtranslation** 技术自动生成高质量的偏好对（preference pairs）：
  - 由专家翻译器（human 或 AI）生成目标语言翻译 $ t $
  - 学生模型 $ T_g $ 将 $ t $ 回译为源语言 $ s' $
  - 若回译结果 $ s' $ 与原始句子 $ s $ 差距较大，则构成一个“劣质输出”，形成偏好三元组 $ (x, y_w=s, y_l=s') $

- **双阶段过滤机制确保偏好信号质量**
  - 第一阶段：使用 **BLEU** 过滤明显合格的样本
  - 第二阶段：基于 **COMET score 分布 + elbow method** 自动确定阈值，仅保留语义退化严重的样本用于训练，增强偏好对比强度

- **无需额外奖励模型的高效优化**
  直接采用 **DPO** 进行优化，避免传统 RL 中需训练独立 reward model 的复杂流程，实现稳定、高效的偏好对齐。

### 🔍 相比现有方法的优势
| 对比维度 | 本文方法 | 现有方法（如 Luu et al. [5], Vajda et al. [8]） |
|--------|---------|---------------------------------------------|
| 数据来源 | 单语语料 + 专家翻译（无需平行句对） | 多依赖已有平行数据或简单合成策略 |
| 偏好对构建 | 引入 backtranslation + 双重质量过滤 | 缺乏严格筛选，噪声较多 |
| 优化方式 | DPO 直接优化，参数效率高（LoRA） | 多数仍用 SFT 或传统 RL |
| 可扩展性 | 支持 human/AI 作为 expert，适用于低资源场景 | 高度依赖强 LLM 或标注数据 |

> ✅ 总体优势：**降低对平行语料依赖，提升偏好学习信号质量，提供一种可扩展、低成本、高性能的 NMT 后训练范式**

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **WMT14 英德翻译数据集**（English→German）
  - 标准 MT benchmark，包含大量高质量人工翻译句对
  - 实验中主要用于评估，而训练偏好数据从其中提取单语句构造

### ⚙️ 实验设置
- **基础模型（Student Model）**: `gemma3-1b`（Google DeepMind 发布的小型多语言 LLM）
- **目标模型命名**: 经 DPO 微调后的模型命名为 `amestris-1b`
- **微调方式**: 使用 **LoRA (Low-Rank Adaptation)** 实现参数高效微调
  - 冻结主干权重，仅训练 adapter 层
- **DPO 超参数配置**（见 Table I）：
  - Epochs: 1
  - DPO temperature β: 0.1
  - LoRA rank: 32, alpha: 32
  - Gradient accumulation steps: 4
  - Warmup ratio: 0.03
- **偏好数据过滤标准**：
  - BLEU 阈值：不限制（let all pass）
  - COMET score 下限：低于 elbow point（0.7233）的样本被保留

### 📊 评估指标
所有指标自动计算，方向如下（↑越高越好，↓越低越好）：

| 指标 | 类型 | 描述 |
|------|-----|------|
| **BLEU ↑** | n-gram 精确率 | 衡量参考译文与生成译文的词重叠度 |
| **COMET-DA ↑** | 神经评估模型 | 基于 WMT22 数据集训练，高度相关人类判断 |
| **COMET-QE ↑** | QE-based 评估 | 基于 unlabel/cometkiwi-da，反映翻译质量估计能力 |
| **METEOR ↑** | 精确/召回平衡 | 考虑同义词和词干匹配 |
| **TER ↓** | 编辑距离 | 达到参考所需编辑操作数，越低越好 |
| **chrF++ ↑** | 字符级 F-score | 更关注形态和拼写准确性 |

### 🔁 基线方法对比
- **Baseline**: 原始 `gemma3-1b` 模型（未经 DPO 微调）
- **对比对象**: 本文提出的 `amestris-1b`（gemma3-1b + backtranslation-augmented DPO）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table II）

| Metric | gemma3-1b (baseline) | amestris-1b (ours) | 变化趋势 |
|--------|------------------------|--------------------|----------|
| **BLEU ↑** | 0.1572 | 0.1500 | ↓（轻微下降） |
| **COMET-DA ↑** | 0.7698 | **0.7810** | ↑ +0.0112 |
| **COMET-QE ↑** | 0.7031 | **0.7476** | ↑ **+0.0445** |
| **METEOR ↑** | 0.3861 | **0.3969** | ↑ +0.0108 |
| **TER ↓** | 77.65 | **76.21** | ↓ -1.44 |
| **chrF++ ↑** | 41.93 | **43.82** | ↑ +1.89 |

### 🔬 结果分析
- **核心突破**：**COMET-QE 提升达 +0.0445**，表明模型在语义一致性、流畅性和整体翻译质量上显著改善。
- 尽管 **BLEU 微降**，但在 **chrF++ 和 METEOR 上提升明显**，说明模型生成文本更具字符级准确性和语义覆盖能力。
- **TER 下降** 表明译文更接近参考译文，编辑成本更低，体现更强的实际可用性。

> 💡 特别指出：**COMET score 从 0.703 提升至 0.747** 是摘要中强调的关键成果，验证了 DPO 在翻译任务中的有效性。

### ❌ 消融实验（Ablation Study）
论文未明确报告消融实验（如移除 backtranslation 或取消 COMET 过滤的影响），但通过以下设计间接体现其必要性：
- 使用 **elbow method 筛选 COMET 分数低的样本** → 确保偏好对具有清晰优劣区分
- **backtranslation 作为诊断工具** → 客观识别学生模型缺陷
- 虽无显式 ablation，但从方法论设计可见各模块功能明确、环环相扣

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DPO 可有效用于 NMT 后训练**  
   即使不引入复杂的 RL 架构，也能通过 preference learning 显著提升翻译质量，尤其是在 **COMET 等神经评估指标上表现突出**。

2. **Backtranslation 是构建高质量偏好数据的有效手段**  
   通过让模型“自我纠错”的方式暴露其翻译弱点，能系统性生成有意义的“winner-loser”对，优于随机采样或多模型对比。

3. **偏好数据质量至关重要**  
   仅靠 BLEU 不足以捕捉语义退化，结合 **COMET + elbow method** 可自动识别最具训练价值的样本，强化学习信号。

4. **小型语言模型亦可通过 DPO 实现跃迁式提升**  
   `gemma3-1b`（1B 参数）经轻量级 DPO 微调后，在英德翻译任务中达到接近更大模型的质量水平，证明该方法对 **Small Language Models (SLMs)** 具备高性价比优势。

### ⚠️ 方法的局限性
- 当前实验集中于 **高资源语言对（English→German）**，尚未验证在真正低资源语言上的泛化能力。
- 依赖一个“expert translator”生成初始翻译，若使用 AI 作为 expert，则可能引入级联错误。
- 未进行人工评估（如 MQM），完全依赖自动指标，可能存在偏差。
- 仅执行一轮 DPO 微调，未探索迭代反馈机制的长期效果。

### 🔮 未来工作方向（Future Research）
1. **Domain Adaptation 扩展**
   - 应用于医疗、法律等专业领域，结合领域专家提供偏好反馈，提高术语准确性和减少幻觉。

2. **改进偏好对构建策略**
   - 探索不同采样策略（如主动学习）、替代 DPO 方法（如 **SimPO**, **ORPO**）以去除 reference model 依赖。

3. **参数高效技术深化**
   - 结合 **QLoRA** 等量化微调方法，进一步降低计算开销，推动边缘部署。

4. **闭环反馈系统设计**
   - 构建 human-in-the-loop 或 AI-agent 协作的持续优化 pipeline，实现动态进化式翻译系统。

---

> 🔗 **开源声明**：作者已将全部代码、模型、检查点及实验资源公开于 GitHub：[github.com/mehrdadghassabi/Amestris](https://github.com/mehrdadghassabi/Amestris)，支持复现与社区拓展研究。

</details>

---

### 10. [A Comparative Analysis on the Performance of Upper Confidence Bound Algorithms in Adaptive Deep Neural Networks](https://arxiv.org/abs/2604.24810)

**Authors**: Grigorios Papanikolaou, Ioannis Kontopoulos, Konstantinos Tserpes  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.24810v1  

#### Abstract
Edge computing environments impose strict constraints on energy consumption and latency, making the deployment of deep neural networks a significant challenge. Therefore, smart and adaptive inference strategies that dynamically balance computational cost or latency with predictive accuracy are criti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Comparative Analysis on the Performance of Upper Confidence Bound Algorithms in Adaptive Deep Neural Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在边缘计算（edge computing）场景中，深度神经网络（DNNs）面临严格的**能效**（energy consumption）和**延迟**（latency）约束。传统的 Early-Exit DNNs 虽然通过动态提前退出（early-exit）机制降低计算开销，但其阈值选择策略多依赖静态设定或简单的 **UCB1** 算法，未能充分探索不同 Upper Confidence Bound（UCB）算法在精度、能耗与延迟之间权衡上的潜力。

本文旨在解决以下问题：
- 如何更智能地在线选择最优 confidence threshold，以实现更好的 accuracy-latency 和 accuracy-energy 权衡？
- 不同 UCB 算法在 Adaptive Deep Neural Networks（ADNNs）中的表现差异如何？

### 提出的新方法与新思路
作者首次将四种先进的 **UCB 变体**引入到 Early-Exit DNNs 的阈值选择任务中，并进行系统性比较分析：

| 算法 | 特点 |
|------|------|
| **UCB-V** | 引入奖励方差感知（variance-aware），对高方差臂给予更大探索权重 |
| **UCB-Tuned** | 结合经验方差并设上限（cap at 0.25），防止早期过度自信 |
| **UCB-Bayes** | 基于贝叶斯推断（Bayesian uncertainty），使用 NIG 先验建模后验分布 |
| **UCB-BwK** | 显式考虑“拉臂成本”（computational cost），优化 reward-to-cost ratio |

这些算法被集成至 Multi-Armed Bandit（MAB）框架中，用于动态决策每个输入样本的最佳退出点。

### 相比现有方法的优势
- ✅ **超越单一 UCB1 的局限性**：现有研究大多仅使用 UCB1，忽略其他具备更强理论保障和适应性的 UCB 变体。
- ✅ **更精细的探索-利用平衡**：如 UCB-V 和 UCB-Tuned 能根据历史奖励方差调整探索强度，避免浪费资源在次优策略上。
- ✅ **支持多样化需求匹配**：不同 UCB 策略适用于不同目标（例如快速收敛 vs 最终性能），为实际部署提供灵活选择。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CIFAR-10**：标准图像分类数据集，50k训练 + 10k测试，用于主实验。
- **CIFAR-100**：更复杂的分类任务，同样 50k/10k 划分，验证方法泛化能力。
- **CIFAR-10.1v6**：CIFAR-10 测试集的扩展版本，含轻微分布偏移（distribution shift），共 2k 新样本，用于评估鲁棒性。

### 实验模型架构
- **ResNet 系列**：ResNet18、ResNet34、ResNet50 —— 经典 CNN 架构，参数量递增。
- **MobileViT**（xxs-MobileViT）：轻量级 CNN-Transformer 混合架构，更适合移动端部署。

所有模型均构建为具有多个 exit branches 的 Early-Exit 版本（ResNet: 4个出口；MobileViT: 3个出口）。

### 实验设置
- **训练方式**：
  - 默认模型无 early exit 分支，作为 baseline。
  - EE 模型采用 joint optimization 进行训练。
  - MobileViT 采用两阶段训练：先训主干，再附加 exit branches 并联合微调。
- **MAB 框架实现**：
  - 每个“arm”代表一个 confidence threshold。
  - Reward 函数定义为：  
    $$
    \text{reward} = C_{\text{EEDNN}} \times (1 - C_{\text{Gating}}) - \lambda \times \text{cost}
    $$
    其中 $C_{\text{Gating}}$ 是由 gating network 输出的可靠性评分，$\text{cost}$ 为退出层索引（反映计算代价），$\lambda$ 控制风险容忍度（设为 0.01 / exit_num）。
- **评估流程**：
  - 在线学习 threshold 策略，在推理过程中实时更新 UCB 值。
  - 所有 UCB 变体均从零开始初始化，确保公平比较。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | Top-1 分类准确率 |
| **Energy Consumption** | 使用 CodeCarbon 工具估算总能耗（单位：kWh × 1e⁻⁴） |
| **Latency / Inference Time** | 处理整个测试集所需时间（秒） |
| **Cumulative Regret** | 衡量算法收敛速度的关键指标，越低越好 |
| **Pareto Frontier** | 展示 accuracy-energy 与 accuracy-latency 的最优权衡边界 |

### 基线方法对比
- **Default Models**：无 early exit 的原始 ResNet / MobileViT
- **Static Threshold EE Models**：固定阈值的 early-exit 模型
- **Dynamic Threshold with UCB1**：当前主流 MAB 方法
- **Proposed Methods**：UCB-V, UCB-Tuned, UCB-Bayes, UCB-BwK

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Fig. 2–4）
#### 在 ResNet 上的表现（CIFAR-10.1v6 测试）
| 方法 | Accuracy (%) | Energy (×1e⁻⁴ kWh) | Latency (s) | Pareto Dominance |
|------|--------------|--------------------|-------------|------------------|
| UCB-V | ~81.5 | ~2.6 | ~10 | ✅ |
| UCB-Tuned | ~81.6 | ~2.7 | ~11 | ✅ |
| UCB-Bayes | ~82.0 | ~3.0 | ~14 | ❌（高延迟） |
| UCB1 | ~80.8 | ~2.9 | ~13 | ❌ |
| Static EE | ~80.5 | ~3.0 | ~15 | ❌ |

> ✅ 表示位于 Pareto 前沿，即无法在不牺牲另一项指标的情况下提升某一项。

#### 在 MobileViT 上的表现（CIFAR-10 & CIFAR-100）
- **CIFAR-10**：UCB-BwK 表现最佳，因其 reward-cost ratio 更适合 MobileViT 的高效结构。
- **CIFAR-100**：难度更高，但趋势一致。

### 与基线方法的对比结果
- 所有 UCB 变体相比 **static threshold** 方法显著改善了 accuracy-energy 和 accuracy-latency 权衡。
- 相比 **UCB1**：
  - **UCB-V** 和 **UCB-Tuned** 在 Pareto 前沿上全面占优，尤其在低能耗/低延迟区域表现更好。
  - **UCB-Bayes** 达到最高 accuracy，但因贝叶斯计算开销大，导致 latency 明显增加。
  - **UCB-BwK** 在 MobileViT 上表现优异，说明其 cost-aware 设计更适合轻量模型。

### 消融实验与关键观察（Fig. 5–6）
#### Cumulative Regret 收敛行为
| 方法 | 收敛速度 | 最终 Regret | 是否 sub-linear |
|------|----------|------------|----------------|
| **UCB-Bayes** | ⭐ 最快（~4000 steps 后趋于平稳） | 最低 | ✅ |
| **UCB-Tuned / UCB-V** | 中等 | 较低 | ✅ |
| **UCB1 / UCB-BwK** | 缓慢 | 较高 | ✅（仍 sub-linear） |

> 所有算法均表现出 **sub-linear cumulative regret**，满足风险控制理论要求（Bajpai et al., 2025）。

#### 架构影响
- 参数更多的 **ResNet50** 比 ResNet18 更受益于动态阈值策略。
- **MobileViT** 因本身 representational capacity 强，reward variance 小，因此 UCB-BwK 成为主导策略。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **UCB-V 和 UCB-Tuned 在综合性能上最优**：它们在 accuracy-energy 和 accuracy-latency 的 Pareto Frontiers 上占据主导地位，实现了最佳权衡。
2. ✅ **UCB-Bayes 收敛最快**：得益于贝叶斯不确定性建模，能在较少步数内识别最优 arm，适合需要快速稳定策略的场景。
3. ✅ **不同 UCB 算法适用于不同需求**：
   - 追求极致效率 → 选 **UCB-V / UCB-Tuned**
   - 需要快速收敛 → 选 **UCB-Bayes**
   - 注重成本敏感 → 选 **UCB-BwK**
4. ✅ **传统 UCB1 并非最优选择**：尽管简单易用，但在性能和收敛速度上均落后于先进变体。

### 方法的局限性
- **UCB-Bayes 计算开销大**：虽然收敛快，但每次迭代需维护后验参数（NIG），增加了推理延迟，不适合严格实时系统。
- **Reward 方差估计不稳定**：特别是在 MobileViT 上，reward 分布较集中，导致 UCB-V/Tuned 提升有限。
- **未考虑上下文信息（contextual bandits）**：当前 MAB 设置为非上下文化的，若结合输入特征可进一步提升性能。

### 未来工作方向
- 探索更多高级 UCB 变体，如 **LinUCB** 或 **Neural UCB**，引入 contextual information 提升个性化决策能力。
- 将该框架应用于 Transformer-based 模型（如 BERT、ViT）的 adaptive inference。
- 研究 multi-objective bandits，直接优化 accuracy、energy、latency 的联合目标。
- 在真实边缘设备（如 Raspberry Pi、Jetson Nano）上进行端到端部署测试，获取真实能耗与延迟数据。

--- 

> 📌 **一句话总结**：本文系统比较了多种 UCB 算法在 Adaptive DNNs 中的性能，揭示了 UCB-V 和 UCB-Tuned 在 accuracy-efficiency 权衡上的优越性，推动了 MAB 在智能边缘推理中的精细化应用。

</details>

---

### 11. [Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence](https://arxiv.org/abs/2604.24954)

**Authors**: NVIDIA,  :, Amala Sanjay Deshmukh, Kateryna Chumachenko, Tuomas Rintamaki, Matthieu Le, Tyler Poon, Danial Mohseni Taheri, Ilia Karmanov, Guilin Liu, Jarno Seppanen, Arushi Goel, Mike Ranzinger, Greg Heinrich, Guo Chen, Lukas Voegtle, Philipp Fischer, Timo Roman, Karan Sapra, Collin McCarthy, Shaokun Zhang, Fuxiao Liu, Hanrong Ye, Yi Dong, Mingjie Liu, Yifan Peng, Piotr Zelasko, Zhehuai Chen, Nithin Rao Koluguri, Nune Tadevosyan, Lilit Grigoryan, Ehsan Hosseini Asl, Pritam Biswas, Leili Tavabi, Yuanhang Su, Zhiding Yu, Peter Jin, Alexandre Milesi, Netanel Haber, Yao Xu, Sarah Amiraslani, Nabin Mulepati, Eric Tramel, Jaehun Jung, Ximing Lu, Brandon Cui, Jin Xu, Zhiqi Li, Shihao Wang, Yuanguo Kuang, Shaokun Zhang, Huck Yang, Boyi Li, Hongxu Yin, Song Han, Pavlo Molchanov, Adi Renduchintala, Charles Wang, David Mosallanezhad, Soumye Singhal, Luis Vega, Katherine Cheung, Sreyan Ghosh, Yian Zhang, Alexander Bukharin, Venkat Srinivasan, Johnny Greco, Andre Manoel, Maarten Van Segbroeck, Suseella Panguliri, Rohit Watve, Divyanshu Kakwani, Shubham Pachori, Jeffrey Glick, Radha Sri-Tharan, Aileen Zaman, Khanh Nguyen, Shi Chen, Jiaheng Fang, Qing Miao, Wenfei Zhou, Yu Wang, Zaid Pervaiz Bhat, Varun Praveen, Arihant Jain, Ramanathan Arunachalam, Tomasz Kornuta, Ashton Sharabiani, Amy Shen, Wei Huang, Yi-Fu Wu, Ali Roshan Ghias, Huiying Li, Brian Yu, Nima Tajbakhsh, Chen Cui, Wenwen Gao, Li Ding, Terry Kong, Manoj Kilaru, Anahita Bhiwandiwalla, Marek Wawrzos, Daniel Korzekwa, Pablo Ribalta, Grzegorz Chlebus, Besmira Nushi, Ewa Dobrowolska, Maciej Jakub Mikulski, Kunal Dhawan, Steve Huang, Jagadeesh Balam, Yongqiang Wang, Nikolay Karpov, Valentin Mendelev, George Zelenfroynd, Meline Mkrtchyan, Qing Miao, Omri Almog, Bhavesh Pawar, Rameshwar Shivbhakta, Sudeep Sabnis, Ashrton Sharabiani, Negar Habibi, Geethapriya Venkataramani, Pamela Peng, Prerit Rodney, Serge Panev, Richard Mazzarese, Nicky Liu, Michael Fukuyama, Andrii Skliar, Roger Waleffe, Duncan Riach, Yunheng Zou, Jian Hu, Hao Zhang, Binfeng Xu, Yuhao Yang, Zuhair Ahmed, Alexandre Milesi, Carlo del Mundo, Chad Voegele, Zhiyu Cheng, Nave Assaf, Andrii Skliar, Daniel Afrimi, Natan Bagrov, Ran Zilberstein, Ofri Masad, Eugene Khvedchenia, Natan Bagrov, Borys Tymchenko, Tomer Asida, Daniel Afrimi, Parth Mannan, Victor Cui, Michael Evans, Katherine Luna, Jie Lou, Pinky Xu, Guyue Huang, Negar Habibi, Michael Boone, Pradeep Thalasta, Adeola Adesoba, Dina Yared, Christopher Parisien, Leon Derczynski, Shaona Ghosh, Wes Feely, Micah Schaffer, Radha Sri-Tharan, Jeffrey Glick, Barnaby Simkin, George Zelenfroynd, Tomasz Grzegorzek, Rishabh Garg, Aastha Jhunjhunwala, Sergei Kolchenko, Farzan Memarian, Haran Kumar, Shiv Kumar, Isabel Hulseman, Anjali Shah, Kari Briski, Padmavathy Subramanian, Joey Conway, Udi Karpas, Jane Polak Scowcroft, Annie Surla, Shilpa Ammireddy, Ellie Evans, Jesse Oliver, Tom Balough, Chia-Chih Chen, Sandip Bhaskar, Alejandra Rico, Bardiya Sadeghi, Seph Mard, Katherine Cheung, Meredith Price, Laya Sleiman, Saori Kaji, Wesley Helmholz, Wendy Quan, Michael Lightstone, Jonathan Cohen, Jian Zhang, Oleksii Kuchaiev, Boris Ginsburg, Jan Kautz, Eileen Long, Mohammad Shoeybi, Mostofa Patwary, Oluwatobi Olabiyi, Andrew Tao, Bryan Catanzaro, Udi Karpas  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.24954v1  

#### Abstract
We introduce Nemotron 3 Nano Omni, the latest model in the Nemotron multimodal series and the first to natively support audio inputs alongside text, images, and video. Nemotron 3 Nano Omni delivers consistent accuracy improvements over its predecessor, Nemotron Nano V2 VL, across all modalities, ena...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence*

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决**高效、开放且具备原生多模态能力的大模型构建难题**，特别是：
- 如何在保持强大文本推理能力的同时，有效融合视觉（images）、视频（videos）和音频（audio）等多模态输入；
- 如何处理长上下文（long-context）的多模态序列，尤其是在文档理解、长时间音视频分析等任务中；
- 如何降低多模态大模型的推理延迟（latency）和计算成本，实现高效的部署。

### 提出的新方法与创新点
NVIDIA 提出了 **Nemotron 3 Nano Omni**，这是 Nemotron 多模态系列中的最新模型，也是首个**原生支持音频输入**的 omni-modal 模型。其核心创新包括：

- **强大的 MoE 混合骨干网络**：采用基于 **Nemotron 3 Nano 30B-A3B** 的 Mixture-of-Experts (MoE) 架构作为语言模型（LLM）主干，显著提升了对长多模态序列的处理效率和推理吞吐量。
  
- **原生音频支持**：首次集成 **Parakeet-TDT-0.6B-v2** 音频编码器，支持对语音、环境声、音乐等多种音频信号的理解，并能与图像、视频进行联合时空推理。

- **动态分辨率图像处理**：摒弃传统的图像分块（tiling）策略，采用**动态分辨率策略**，保留原始宽高比，提升图像理解质量。

- **时间维度压缩技术**：
  - 对视频引入 **Conv3D patch embedder**，将每两帧压缩为一个“tubelet”，实现 **2× 时间token减少**。
  - 提出 **Efficient Video Sampling (EVS)** 技术，在运行时进一步剪枝空间冗余token，大幅降低输入长度。

- **超长上下文支持**：将最大上下文长度从 128K 扩展至 **256K tokens**，显著增强长文档、长视频的理解能力。

- **多阶段训练策略（Multi-stage Training）**：
  - 分阶段逐步引入新模态（vision → audio → omni），并逐步扩展上下文长度，缓解灾难性遗忘（catastrophic forgetting），稳定跨模态对齐。

- **高效的 token 减少技术**：结合 Conv3D 和 EVS，使 512 帧视频的输入 token 数从 ~141K 降至 ~42K（降幅达 70%），极大提升推理效率。

### 相比现有方法的优势
- 在多个权威 benchmark 上达到或接近 SOTA 表现，尤其在 **document understanding、long audio-video comprehension、agentic computer use** 等实际场景中表现领先。
- 推理效率远超同类模型：在 NVIDIA B200 上，相比 Qwen3-Omni，单流输出 token 吞吐量提升 **3×**，单位交互目标下的吞吐量提升 **9×**。
- 是目前 **MediaPerf 上最具成本效益的开源视频理解模型**。
- 完全开源：发布 BF16、FP8、FP4 格式的模型权重、部分训练数据集、pipeline 和代码，推动社区发展。

---

## 2. 核心实验方法和设置

### 使用的数据集
训练数据覆盖广泛模态与任务，总计约 **466.9B tokens**，主要来自以下来源：

| 类别 | 数据集示例 | 描述 |
|------|-----------|------|
| **视觉（Vision）** | OCRBench-V2, MMLongBench-Doc, ChartQA, DocVQA, AI2D, RefCOCO, ScreenSpot 系列 | 图像描述、OCR、图表理解、GUI 操作、视觉定位等 |
| **音频（Audio）** | OpenASR, TED-LIUM Longform, MMAU, VoiceBench | 自动语音识别（ASR）、长语音转录、声音理解、语音助手评测 |
| **音视频（Omni）** | DailyOmni, WorldSense | 跨模态时序对齐、事件理解、因果推理、复杂音视频问答 |
| **文本（Text）** | MMLU-Pro, GPQA, AIME-2025, LiveCodeBench, SciCode | 学术知识、数学推理、编程能力、科学写作等 |

### 实验设置与评估指标
- **评估框架**：
  - 视觉/音视频：`VLMEvalKit` + `vLLM`
  - 文本：`NeMo-Skills`
- **量化格式测试**：BF16、FP8、NVFP4
- **推理硬件**：NVIDIA B200 GPU
- **关键指标**：
  - **准确率（Accuracy）**：用于分类、VQA、文档理解等任务
  - **Word Error Rate (WER)**：用于 ASR 任务（越低越好）
  - **Pass@1**：用于数学与编程任务
  - **Throughput (tok/s)**：输出 token 吞吐量
  - **Time-to-First-Token (TTFT)**：首 token 延迟
  - **Median Accuracy Drop**：量化后精度损失

### 基线方法对比
主要对比对象包括：
- **Nemotron Nano V2 VL**：前代版本，无音频支持，较小规模
- **Qwen3-Omni**：通义千问系列的 omni-modal 模型
- **Qwen3.5-Omni**：更先进版本（闭源）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 多模态综合性能（部分关键指标）
| Benchmark | Nemotron 3 Nano Omni | Qwen3-Omni | 提升情况 |
|----------|------------------------|------------|---------|
| **OCRBench-V2 (EN/ZH)** | 65.8 / 52.0 | 54.8 / 39.8 | ↑ 显著 |
| **MMLongBench-Doc** | 57.5 | 49.5 | ↑ 8.0 pts |
| **VoiceBench (avg)** | 89.4 | 88.8 | ↑ 微优 |
| **DailyOmni** | 74.5 | 73.6 | ↑ |
| **WorldSense** | 55.4 | — | 接近 SOTA |
| **Video-MME** | 72.2 | 70.5 | ↑ |

#### ⚙️ 推理效率（NVIDIA B200）
| 指标 | Nemotron 3 Nano Omni | Qwen3-Omni | 提升倍数 |
|------|------------------------|------------|---------|
| **Single-stream output throughput** | >500 tok/s | 175–210 tok/s | **2.4–2.9×** |
| **Max output throughput (multi-doc)** | 5000 tok/s | — | — |
| **Output throughput per GPU @ iso-interactivity** | — | — | **9× (video), 7.5× (doc)** |
| **TTFT (multi-doc workload)** | ~1.3s | >2.5s | ↓ 超过 50% |

#### 🔍 量化影响（NVFP4 vs BF16）
- **模型大小**：61.5 GB (BF16) → **20.9 GB (NVFP4)**
- **有效 bit 数**：16.0 → **4.98 bpw**
- **平均精度下降**：<1% （median drop）
- **吞吐量提升**：可达 **7.5×**（在单图推理场景下）

#### 🪓 消融实验结果（Ablation Studies）

##### (1) Conv3D + EVS 对推理效率的影响（见 Table 12）
- **Conv3D 单独启用**：TTFT 从 7969ms → 5984ms（↓25%）
- **EVS 单独启用**：TTFT → 6452ms（↓19%）
- **两者叠加**：TTFT → **5313ms（↓33%）**，仅损失约 0.5 点平均精度
- 输入 token 数从 ~141K → ~42K（↓70%）

##### (2) EVS 剪枝率（pruning rate q）影响（见 Table 13）
- 当 q ≤ 0.7 时，精度基本持平
- q = 0.9 时开始明显下降，尤其 **LongVideoBench 最敏感**

##### (3) 推理预算控制（Reasoning Budget Control）（见 Table 11）
- 启用 13K reasoning budget 后：
  - **MathVista-Mini**: 80.3 → **82.8**
  - **MMLongBench-Doc**: 54.5 → **56.8**
  - 其他任务无退化，说明该机制可智能终止无效推理链。

---

## 4. 关键结论和发现

### 主要发现
1. **Nemotron 3 Nano Omni 是当前最高效的开源 omni-modal 模型之一**，在文档理解、长音视频理解、GUI 操作代理等方面达到领先水平。
2. **原生音频支持 + 多模态对齐训练策略** 成功实现了高质量的跨模态联合推理。
3. **Conv3D + EVS 的组合设计** 是实现高效推理的关键，可在几乎不损精度的前提下大幅降低延迟与计算开销。
4. **NVFP4 量化方案极为成功**，在压缩至 5bpw 的同时，多数任务精度损失 <1%，适合生产部署。
5. 模型在保持强大文本能力（如 MMLU-Pro 达 77.3）的同时，成功扩展了多模态能力，验证了 MoE 主干的有效性。

### 方法的局限性
- 尽管音频能力较强，但在某些音乐理解或极端噪声环境下的表现仍有提升空间。
- EVS 在极高剪枝率（q > 0.8）时会导致性能显著下降，限制了极端压缩场景的应用。
- 部分训练数据仍依赖合成生成，可能存在偏差或分布外泛化问题。
- 虽然发布部分数据，但完整训练集未完全公开。

### 未来工作方向
- 进一步优化音频编码器与跨模态对齐机制，提升复杂声学场景下的鲁棒性。
- 探索更智能的动态 token 剪枝策略（如基于内容重要性评分）。
- 扩展至更多语言和文化背景下的多模态理解任务。
- 结合更强的 agent planning 与工具调用能力，打造真正的“通用多模态智能体”。
- 继续推进模型小型化与边缘设备部署方案。

---

> ✅ **总结一句话**：  
> *Nemotron 3 Nano Omni 通过 MoE 主干、原生音频支持、动态分辨率、Conv3D+EVS 压缩、256K 上下文和多阶段训练，实现了高性能、高效率、全开源的 omni-modal 智能，是迈向实用化多模态 AI 的重要一步。*

</details>

---

### 12. [Large Language Models Explore by Latent Distilling](https://arxiv.org/abs/2604.24927)

**Authors**: Yuanhao Zeng, Ao Lu, Lufei Li, Zheng Zhang, Yexin Li, Kan Ren  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.24927v1  

#### Abstract
Generating diverse responses is crucial for test-time scaling of large language models (LLMs), yet standard stochastic sampling mostly yields surface-level lexical variation, limiting semantic exploration. In this paper, we propose Exploratory Sampling (ESamp), a decoding approach that explicitly en...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Large Language Models Explore by Latent Distilling*

## 1. 论文的主要贡献和创新点

### 解决的问题
标准的随机采样（stochastic sampling）在大语言模型（LLM）推理时扩展（test-time scaling）中存在**语义探索不足**的问题。尽管能生成词汇层面多样的响应，但这些响应往往基于相同的底层推理路径，导致候选解高度冗余。这种“表面多样性”限制了后续选择机制（如重排序、多数投票）的有效性，尤其是在数学、科学和代码等需要深度推理的任务中。

### 提出的新方法：Exploratory Sampling (ESamp)
本文提出了一种新的解码算法——**Exploratory Sampling (ESamp)**，其核心思想是通过估计LLM内部表示空间中的**新颖性（novelty）** 来引导生成过程，从而鼓励模型探索未被充分覆盖的语义区域。

- **Latent Distiller (LD)**：一个轻量级的在线训练模块（一个2层MLP），用于学习从LLM浅层隐藏状态到深层隐藏状态的映射关系。
- **新颖性信号**：利用LD的预测误差作为新颖性度量。高预测误差意味着当前上下文的表示映射是“未见过的”，因此具有较高的探索价值。
- **KL正则化优化**：将生成过程建模为一个KL正则化的强化学习目标，其中内在奖励（intrinsic reward）由新颖性信号驱动。最终的采样分布通过对基础模型的logits进行重加权得到。

### 相比现有方法的优势
- **超越表面多样性**：与仅操作词表空间（如Min-p, OverRIDE）的方法不同，ESamp在连续的**隐空间（latent space）** 进行探索，能够识别并抑制语义上重复但词汇上不同的序列。
- **高效且低开销**：采用异步训练-推理流水线，将LD的计算与主LLM的前向传播重叠，实现了**低于5%的吞吐量开销**（优化版本低至1.2%），适合大规模部署。
- **鲁棒泛化能力**：在数学、科学、代码生成和创意写作等多个领域均表现出色，打破了多样性与连贯性之间的权衡。
- **协同探索**：在批量生成时，共享的LD充当了一个隐式的“先到先得”协调器，有效避免了多个并行序列重复探索相同的语义模式。

---

## 2. 核心实验方法和设置

### 数据集
实验涵盖了四个主要领域的基准：
- **数学推理**：`AIME 2024` 和 `AIME 2025`，竞赛级别的数学题，要求多步推理得出0-999的整数答案。
- **科学问答**：`GPQA-Diamond`，涵盖生物、物理、化学的高难度多项选择题，需专家验证。
- **代码生成**：`LiveCodeBench v5`，来自LeetCode等平台的编程题，强调无数据污染。
- **创意写作**：`BookCorpus`，用于评估故事生成的多样性和质量。

### 实验设置和评估指标
- **模型**：在多种架构和规模的模型上测试，包括 `Qwen2.5-7B/32B-Instruct`、`Qwen3-8B` 和 `GPT-OSS-20B`。
- **批处理**：在 `Test-Time Scaling` 场景下，使用 `B=32` 的提示批次和 `K=16` 的每个请求采样数。
- **评估指标**：
  - **Pass@k**：k个样本中至少有一个正确解的概率，衡量推理效率。
  - **Embedding Similarity**：生成响应嵌入的平均成对余弦相似度，衡量语义冗余。
  - **Vendi Score**：量化一批次中独特语义簇的有效数量，衡量多样性。
  - **Perplexity (PPL)**：衡量生成文本的语言流畅性和一致性。

### 基线方法对比
- **随机采样**：`Vanilla` (温度=1), `Min-p`, `FIRE`。
- **结构化搜索**：`Tree of Thoughts (ToT)`。
- **Logit级控制**：`Contrastive Decoding`, `OverRIDE`。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **Pass@k 效率显著提升**：
  - 在 `GPT-OSS-20B` 上，`ESamp` 用 `Pass@8` 就能达到基线方法 `Pass@64` 的性能水平，效率提升高达8倍。
  - 在 `AIME25` 上，`ESamp` 的 `Pass@64` 达到 **63.9%**，远超 `Vanilla` 的 58.9% 和 `OverRIDE` 的 60.6%。
- **语义多样性增强**：
  - 在 `Creative Writing` 任务中，`ESamp` 取得了最高的 **Vendi Score (1.67)** 和最低的 **Embedding Similarity (0.57)**，同时保持了最佳的生成质量（PPL=3.55）。
  - 在 `AIME` 数学推理中，`ESamp` 的 `Vendi Score` 也最高（0.46），表明其探索了更多样化的有效推理路径。

### 与基线方法的对比结果
- **优于所有基线**：`ESamp` 在大多数任务和模型上均表现最优或相当，尤其在推理密集型任务（如AIME）上优势明显。
- **泛化性强**：`FIRE` 在AIME上表现好但在LiveCodeBench上受挫，而 `ESamp` 在所有领域都保持稳健。
- **打破多样性-连贯性权衡**：传统方法常牺牲一方来换取另一方，而 `ESamp` 同时提升了多样性和质量。

### 消融实验结果
- **探索强度 β**：`β=0.25` 是最优设置。过小（β=0.1）退化为 `Vanilla`，过大（β=0.5）会过度惩罚高置信度token，损害性能。
- **Logit融合方式**：提出的 `(1+β)logit_ref - β*logit_dist` 形式优于简单的减法，能更好地保留基础模型的概率分布，防止生成语法错误。
- **噪声消融**：用相同幅度的高斯噪声替换LD的真实误差向量，性能退化到接近 `Vanilla` 水平，证明了**误差方向**包含有意义的结构信息。
- **隐空间 vs 词表空间**：在词表空间训练的Distiller不稳定且性能差，验证了在紧凑的连续隐空间进行在线学习更稳定有效。

---

## 4. 关键结论和发现

### 主要发现
1. **隐空间新颖性是有效的探索信号**：通过在线蒸馏（online distillation）预测LLM的深度表示过渡，并利用其误差作为奖励，可以有效引导模型探索新的语义区域。
2. **ESamp实现了高效且高质量的探索**：该方法在不牺牲生成质量的前提下，显著提升了候选解的语义多样性，从而大幅提高了 `Pass@k` 效率。
3. **协同探索机制有效**：共享的LD使得并行生成的序列能够隐式协调，避免重复劳动，实现了高效的批量级探索。
4. **方法具有强鲁棒性和泛化性**：在不同模型家族、大小和任务上均表现一致优异，不受特定领域启发式约束的影响。

### 方法的局限性
- **依赖于模型的深度结构**：方法假设LLM有明确的浅层和深层表示，对于结构过于简单或特殊的模型可能不适用。
- **在线训练稳定性**：虽然设计为轻量级，但在线训练仍可能受到梯度噪声影响，需要仔细调参（如学习率）。
- **共享Distiller的潜在干扰**：在某些任务（如AIME）上，为每个提示独立维护Distiller效果更好，表明共享策略并非总是最优。

### 未来工作方向
- **自适应共享策略**：研究何时应共享Distiller，何时应为每个提示独立维护，以平衡效率和性能。
- **更复杂的Novelty信号**：探索结合长期记忆或外部知识来定义新颖性，而不仅仅是基于最近的生成历史。
- **与其他Test-Time Scaling方法的组合**：进一步研究 `ESamp` 如何与 `Self-Consistency` 或 `Verifier` 等聚合方法更有效地结合。
- **应用于其他生成任务**：探索该框架在对话系统、规划、决策等需要多样化输出的场景中的应用。

> **开源代码**：https://github.com/LinesHogan/tLLM

</details>

---

### 13. [Heterogeneous Variational Inference for Markov Degradation Hazard Models: Discretized Mixture with Interpretable Clusters](https://arxiv.org/abs/2604.24818)

**Authors**: Takato Yasuno  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.24818v1  

#### Abstract
Bayesian finite mixture models can identify discrete risk clusters (low-risk vs. high-risk equipment), but face three critical bottlenecks: (1) insufficient degradation signals from coarse state discretization, (2) unstable cluster identification when data inherently supports fewer clusters than exp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对工业基础设施（如泵设备）退化建模中的三个关键挑战：
- **信号不足**：传统粗粒度状态离散化（如4个健康等级）导致退化事件稀少（仅1.3%），难以支撑稳定的有限混合模型（finite mixture model）聚类。
- **模型选择不稳定**：基于WAIC等信息准则的网格搜索易选出过拟合且不可解释的小簇（如占比2.9%的集群），缺乏面向实践者的可操作性约束。
- **计算不可行**：传统的MCMC方法（特别是NUTS采样器）推理耗时极长（单次运行超7小时），无法满足生产环境下的快速迭代需求。

### 提出的新方法与思路
作者提出了一套完整的、面向生产的异质性变分推断框架，包含以下四项核心创新：

#### （1）细粒度全局百分位离散化（8-State Global Percentile Discretization）
将连续健康指标按所有设备在全时段上的**全局12.5%分位数**划分为8个离散状态，而非传统的固定阈值4状态。此举使退化事件率从1.3%提升至**2.4%（+83%）**，显著增强了状态转移多样性，为混合模型提供更丰富的信号。

#### （2）综合特征工程策略（30-Dimensional Feature Engineering）
构建了一个融合多源信号的30维协变量向量：
- **统计趋势特征**（22维）：过去90天内的均值、标准差、偏度、峰度、趋势斜率、波动性等；
- **连续健康指标**（2维）：归一化测量值及其变化；
- **文本嵌入特征**（3维）：通过Sentence-BERT提取巡检文本描述的语义信息，并用PCA压缩至3维。

#### （3）可解释性驱动的模型选择规则（Interpretability-Constrained Model Selection）
引入三层筛选机制防止过拟合并确保实用性：
- **WAIC容忍度**：候选模型WAIC不超过最优模型±50；
- **最小簇占比 ≥5%**：避免微小簇无实际维护意义；
- **最小簇分离 Δμ ≥0.15**：保证相邻簇间风险差异显著（exp(0.15)≈1.16倍以上退化速率差异）。

#### （4）全秩ADVI替代NUTS进行高效推断
采用**Automatic Differentiation Variational Inference (ADVI)** 并使用**full-rank covariance**近似后验分布，相比NUTS实现**84倍加速**（5分钟 vs. 7小时40分钟），同时解决了MCMC常见的标签切换（label switching）和收敛失败问题。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 推理效率 | NUTS: >7h | ADVI: **5分钟（84×加速）** |
| 聚类稳定性 | 易出现空簇或微小簇 | 引入可解释性规则，保障簇有效性 |
| 信号质量 | 4-state → 1.3%事件率 | **8-state → 2.4%事件率（+83%）** |
| 可部署性 | 不适用于实时更新 | 支持月度重训练与动态调度 |

---

## 2. 核心实验方法和设置

### 数据集
- **数据来源**：来自多个工业设施的**280台泵设备**
- **时间跨度**：1991年至2025年（共34年）
- **记录数量**：共计 **104,703条巡检记录**
- **观测内容**：
  - 连续健康指标（振动、温度等）
  - 巡检时间间隔（平均约91天）
  - 文本注释（82,416条，用于生成text embeddings）

### 实验设置
#### 模型配置
| 模型 | 方法 | 推理算法 | 执行时间 |
|------|------|----------|---------|
| `mcl` | 随机效应基准模型 | NUTS (4 chains) | 45分钟 |
| `mc2` | 随机效应基准模型 | full-rank ADVI (20k iter) | 3分钟 |
| `mix1` | 有限混合模型（K=2~5） | full-rank ADVI | 每K约5分钟 |
| `mix2` | 有限混合模型（K=2） | NUTS (6 chains, init='advi') | 7h 40min |

#### 评估指标
- **统计拟合优度**：WAIC（Widely Applicable Information Criterion）
- **收敛诊断**：
  - $\hat{r}$（Gelman-Rubin statistic）应 < 1.01
  - ESS（Effective Sample Size）应 > 400
- **聚类可解释性**：
  - 最小簇占比（min share）
  - 相邻簇均值差（Δμ）
- **相关性验证**：Pearson相关系数 $r$ 对比不同方法估计的随机效应 $u_i$

#### 基线方法对比
- **MCMC vs. VI**：比较NUTS与ADVI在相同模型下的参数一致性与收敛表现
- **不同离散化方案**：4-state vs. 8-state 对比对聚类稳定性的影响
- **不同特征集**：7维基础特征 vs. 30维综合特征对模型鲁棒性的提升

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）ADVI vs. NUTS 在随机效应模型中的准确性验证
| 指标 | NUTS (`mcl`) | ADVI (`mc2`) | 对比结果 |
|------|-------------|--------------|----------|
| 执行时间 | 45 min | **3 min** | **15×加速** |
| $u_i$ 范围 | [-3.52, +2.85] | [-3.56, +2.86] | 几乎一致 |
| Pearson $r(u)$ | — | **0.997** | 极高相关性 |
| 显著异质性泵数 | 223/280 (79.6%) | 223/280 (79.6%) | 完全一致 |

> ✅ 结论：ADVI在随机效应模型上与NUTS几乎完全一致，验证其可靠性。

#### （2）有限混合模型中ADVl vs. NUTS的表现对比（K=2）
| 指标 | ADVI (`mix1`) | NUTS (`mix2`) |
|------|---------------|----------------|
| 执行时间 | **5分钟** | 7h 40min |
| 速度提升 | **84×** | 1× |
| $\mu_1$（低风险簇） | -0.98 | -3.52（异常） |
| $\mu_2$（高风险簇） | 0.00 | +0.88（异常） |
| 簇分离 Δμ | 0.98 | 4.40（不合理） |
| $\hat{r}(\mu)$ | N/A | **1.19–1.28（严重发散）** |
| Min ESS | N/A | **17（严重不足）** |
| Cluster 1 占比 | 72.9% | 39.6%（颠倒） |
| 是否发生label switching | 否 | 是 |
| 收敛状态 | **稳定** | **失败** |

> ❗ 结论：NUTS因标签切换导致严重不收敛；而ADVI提供稳定、可复现的结果。

#### （3）模型选择结果（ADVI网格搜索 K=2~5）
| K | WAIC | 最小簇占比 | 最小Δμ | 是否通过三重规则 | 决策 |
|----|-------|------------|---------|------------------|------|
| 2 | **19,788** | **27.1%** | **0.98** | ✅✅✅ | **最优模型** |
| 3 | 19,814 | 2.9% | 0.56 | ✅❌✅ | 失败（min share < 5%） |
| 4 | 19,842 | 1.8% | 0.42 | ❌ | 失败 |
| 5 | 19,875 | 0.7% | 0.29 | ❌ | 失败 |

> ✅ 最终选定 **K=2**：一个包含204台泵（72.9%）的“低风险”簇和76台泵（27.1%）的“高风险”簇。

#### （4）消融实验结果
| 实验设置 | C=2 是否稳健 | C=3 是否可行 |
|--------|-------------|--------------|
| 8-state + 30维特征 | ✅ 是 | ❌ 否（虽有簇但占比仅2.9%） |
| 4-state + 30维特征 | ⚠️ 边缘 | ❌ 是（但出现空簇） |
| 8-state + 7维特征 | ✅ 是 | ⚠️ 微弱可行（min share=5.2%，勉强达标） |

> 🔍 发现：
> - **8-state离散化是稳定聚类的前提**；
> - **特征工程能消除空簇，但不能改变数据内在结构（天然支持K=2）**。

---

## 4. 关键结论和发现

### 主要发现
1. **细粒度状态划分至关重要**  
   将健康状态从4级扩展到8级（基于全局百分位），使退化事件增加83%，极大提升了混合模型的识别能力与稳定性。

2. **ADVI优于NUTS用于有限混合模型**  
   尽管传统观点认为MCMC更准确，但在有限混合模型中，**NUTS因标签切换而导致严重收敛失败**，而**full-rank ADVI不仅速度快84倍，且结果更稳定可靠**，是生产系统的首选。

3. **可解释性规则有效防止过拟合**  
   若仅依赖WAIC，会错误选择K=3模型；加入**min share ≥5% 和 Δμ ≥0.15**后，成功排除无意义的小簇，确保输出具有运维价值。

4. **真实设备存在两个自然风险群体**  
   数据天然支持两群结构：约73%设备退化缓慢（exp(-0.98)≈0.38倍基准速率），27%接近或高于平均水平，适合实施差异化维护策略。

### 方法的局限性
1. **未建模最终故障**：当前模型预测的是状态转移概率，而非设备彻底失效（failure），需补充更换/维修事件数据。
2. **忽略时间自相关性**：未显式建模同一设备多次测量间的序列依赖关系，未来可用GP或HMM增强。
3. **文本信息利用不足**：仅用PCA压缩至3维，可能丢失重要语义细节，建议微调领域专用语言模型。
4. **单一设备类型**：结论是否泛化至涡轮机、压缩机等其他设备尚待验证。
5. **协变量未优化选择**：30个协变量中仅有6个显著，可引入spike-and-slab或horseshoe先验进行稀疏化。

### 未来工作方向
- **短期**：
  - 扩展至**time-to-failure modeling**，结合右删失数据；
  - 引入**因果推断**，评估维修动作对退化路径的影响；
  - 构建**跨设备类型的多层级模型**，实现知识迁移。
- **长期**：
  - 开发**自适应ADVI在线学习机制**，实现秒级增量更新；
  - 结合**Bayesian decision theory**，联合成本模型优化干预时机；
  - 训练**领域专用Transformer模型**（如Maintenance-BERT），深度挖掘巡检日志语义。

---

> 📌 **总结一句话**：  
> 本文首次系统证明，在工业设备退化建模中，**全秩ADVI + 8-state离散化 + 可解释性规则**构成了一套高效、稳定、可落地的贝叶斯混合建模范式，**颠覆了“MCMC一定优于VI”的传统认知**，为AI驱动的资产健康管理提供了新的工程范本。

</details>

---

### 14. [Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines](https://arxiv.org/abs/2604.23178)

**Authors**: Sadman Kabir Soumik  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23178v1  

#### Abstract
LLM-as-a-Judge has become the dominant paradigm for evaluating language model outputs, yet LLM judges exhibit systematic biases that compromise evaluation reliability. We present a comprehensive empirical study comparing nine debiasing strategies across five judge models from four provider families ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前，**LLM-as-a-Judge** 已成为评估语言模型输出的主流范式（如 MT-Bench、Chatbot Arena），但已有研究表明 LLM 评委存在系统性偏差（如位置偏见、冗长偏好、自偏好、风格偏见等），严重影响评估可靠性。然而，现有研究大多孤立地提出缓解策略，缺乏统一框架下的**系统性比较**。

本文旨在解决以下关键问题：
- 不同类型的偏见在当前主流 LLM 中的实际严重程度如何？
- 各种去偏策略（debiasing strategies）的效果是否具有模型依赖性？
- 是否存在跨偏见的交互效应？哪种组合策略最有效？

### 提出的新方法与新思路
1. **统一的基准评测框架（Unified Benchmark）**  
   首次在**五个主流 LLM**（Gemini Pro/Flash, Claude Sonnet 4, GPT-4o, Llama 3.3-70B）、**四种厂商家族**、**三个基准数据集**上，对**九种去偏策略**进行系统性对比。

2. **受控偏见测量数据集（Controlled Dataset, n=225）**  
   构建了一个包含已知“ground truth”的合成数据集，用于精确分离并量化四类偏见：
   - **LENGTH**（扩展 vs 截断）
   - **POSITION**（顺序相同）
   - **STYLE**（Markdown vs 纯文本）
   - **MODEL ORIGIN**（Gemini vs Claude 回答）

3. **跨偏见交互分析（Cross-Bias Interaction Analysis）**  
   揭示了某些策略在缓解一种偏见的同时可能加剧另一种（例如 Position Swap 减少 style bias 却增加 verbosity bias）。

4. **统计严谨的结果报告**  
   所有结果均提供 Bootstrap 95% CI 和 McNemar 检验，支持模型级别的推荐。

### 相比现有方法的优势
| 维度 | 以往工作 | 本论文 |
|------|--------|-------|
| 范围 | 单一模型或单一策略 | 多模型、多策略、多偏见联合分析 |
| 数据控制 | 使用真实 benchmark | 引入可控合成数据验证因果关系 |
| 分析深度 | 忽略策略间交互 | 明确揭示 cross-bias effects |
| 可复现性 | 缺乏完整实验记录 | 开源框架、数据、缓存结果 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 样本量 | 特点 |
|-------|------|--------|-----|
| **MT-Bench** | 自然成对比较 | 400 instances | 包含人类偏好标签，用于主评估 |
| **LLMBar** | 对抗性测试集 | 200 instances | 存在误导性但表面吸引人的回答 |
| **Custom Controlled Dataset** | 合成控制对 | 225 pairs | 地面真值明确，用于精准测偏 |

> 其中，**Custom Dataset** 是核心创新，包含：
> - 200 对期望为“平局”（tie）的样本（每类偏见50对）
> - 25 对截断样本（期望选择更完整的长版本）

### 实验设置
- **任务形式**：Pairwise Comparison（A/B 二选一或平局）
- **输出格式**：JSON 结构化判断 + 推理过程
- **温度设置**：所有模型设为 `temperature=0.1` 以减少随机性
- **模型来源**：来自 Google、Anthropic、OpenAI、Meta 四大厂商

### 评估指标
- **Human Agreement Rate**：与人类标注的一致率
- **Cohen’s Kappa (K)**：校正偶然一致后的信度
- **Bias Score**：`P(prefers A) - P(prefers B)`，衡量偏见强度
- **Statistical Testing**：McNemar 检验 + Holm-Bonferroni 多重检验校正

### 去偏策略（Debiasing Strategies）
共实现 **9 种策略**，分为三类：

#### （1）单次调用提示法（1× cost）
| 编号 | 名称 | 描述 |
|------|------|------|
| B0 | Baseline (Naive) | 最简提示 |
| S4 | Calibrated Rubric | 五维评分表（准确性、相关性等） |
| S5 | Chain-of-Thought (CoT) | 强制逐步推理再判断 |

#### （2）多次聚合法（2–3× cost）
| S1 | Position Swap | 两次调用交换 A/B 顺序，取共识 |
| S2 | Same-Family Ensemble | 温度扰动投票（temp={0.0,0.3,0.7}) |

#### （3）组合策略（2× cost）
| S8 | Combined Budget | Position Swap + CoT + Rubric 融合提示 |

> 注：S3（跨家族集成）、S6（参考引导）、S7（全量组合）也实现，结果见附录。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（MT-Bench 上的人类一致性）

| Model | B0 (Baseline) | Best Strategy | Improvement | Significance |
|-------|----------------|---------------|-------------|--------------|
| **Claude Sonnet 4** | 58.8% | **S8 (Combined)** → **70.0%** | **+11.2 pp** | ***p < 0.0001*** ✅ |
| | | S5 (CoT) → 66.0% | +7.2 pp | *p = 0.004* ✅ |
| **Gemini 2.5 Pro** | 65.2% | S1 (Pos. Swap) → 69.8% | +4.6 pp | *p = 0.012*（未校正）⚠️ |
| **GPT-4o** | 63.4% | S8 (Combined) → 66.2% | +2.8 pp | 不显著 |
| **Llama 3.3-70B** | 64.7% | S8 (Combined) → 68.5% | +3.8 pp | 不显著 |
| **Gemini 2.5 Flash** | 64.0% | S5/S1 → 66.5% | +2.5 pp | 不显著 |

> ✅ 表示经 Holm-Bonferroni 校正后仍显著；⚠️ 表示边际显著。

### 与基线方法的对比结果
- **仅 2/20 非基线配置出现负向效果**：
  - GPT-4o 使用 Position Swap 下降 2.4 pp
  - Gemini Pro 使用 CoT 微降 0.2 pp（仍在置信区间内）
- **总体趋势积极**：20 个非基线配置中 **18 个提升**，sign test 显示整体方向显著正向（p < 0.001）

### 消融实验与关键发现

#### （1）偏见严重程度排序（Baseline 下）
| 偏见类型 | 平均偏见得分 | 发现 |
|---------|---------------|------|
| **Style Bias** | **0.76 – 0.92** | ⬆️ **绝对主导**，远超其他 |
| Verbosity Bias | 0.20 – 0.76 | 全部模型偏好简洁回答（conciseness preference） |
| Position Bias | ≤ 0.04 | ❌ 当前模型中几乎消失 |
| Self-Preference | 不一致 | 如 GPT-4o 反而偏好 Claude 输出 |

> 🔍 **Style Bias**：所有模型强烈偏好 Markdown 格式而非纯文本，即使内容完全相同。

#### （2）CoT 是唯一普适有效的单策略
- 在 **MT-Bench 和 LLMBar 上均无负面作用**
- 对所有模型均有正向或中性影响
- 在 LLMBar 上带来 +1.5 至 +13.0 pp 提升

#### （3）交叉偏见交互（Cross-Bias Interactions）
| 策略 | Style Bias ↓ | Verbosity Bias ↑/↓ | 说明 |
|------|--------------|--------------------|------|
| CoT | -0.14 avg | -0.03 | 有效缓解 style bias |
| Rubric | -0.11 | -0.11 | 更关注质量维度 |
| Position Swap | -0.12 | **+0.07 ↑** | **副作用明显！** |
| **S8 (Combined)** | **-0.26 ↓** | -0.07 | 综合优势显著 |

> 💡 S8 成功的关键在于融合多种机制，抵消了单一策略的副作用。

#### （4）不同场景下的最优策略
| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 自然评估（MT-Bench 类） | 模型定制化选择（见 Algorithm 1） | 效果差异大 |
| 对抗性/高风险评估（LLMBar 类） | **S5 (CoT Forcing)** | 唯一对所有模型有益 |
| 避免格式干扰 | **标准化输出格式** 或 显式忽略样式指令 | Style bias 过强 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Style Bias 是当前最严重的偏见**（0.76–0.92），远高于曾被广泛关注的 Position Bias（≤0.04），但长期被忽视。
2. ✅ **“冗长偏见”实为“简洁偏好”**：模型惩罚填充内容（expansion pairs），但能正确识别真正完整的内容（truncation accuracy 达 0.92–1.00），表明其具备**质量敏感判断能力**。
3. ✅ **去偏策略有效但高度模型依赖**：
   - Claude 系列从 S8（Combined Budget）获益最大（+11.2 pp）
   - Gemini Pro 适合使用 Position Swap
   - GPT-4o 和 Llama 改进方向不显著，但趋势正向
4. ✅ **CoT 是最安全的默认策略**：在所有模型和任务中均表现稳健，是未知场景下的最佳选择。
5. ✅ **组合策略优于单一策略**：S8 通过整合 Position Swap + CoT + Rubric，在平均 style bias 上降低达 -0.26。

### 方法的局限性
1. **LENGTH 扩展对可能混淆长度与填充质量**，尽管截断控制部分缓解此问题。
2. **Markdown 偏好是否完全属于“偏见”有待商榷**：可能反映真实可读性优势。
3. **未覆盖全部开源家族**：缺少 Mistral、Qwen 等模型，泛化性待验证。
4. **GPT-4o token 数估算误差**：使用字符数近似，影响成本分析精度。
5. **样本量限制**：n=400 导致部分改进未能达到统计显著性，需更大规模验证。

### 未来工作方向
- 扩展至更多模型家族（尤其是开源模型）
- 构建动态监控工具，持续追踪 LLM 评委偏见演化
- 探索轻量化 fine-tuning + prompting 的混合去偏方案
- 设计针对文化偏见、性别偏见等社会性偏差的测试集
- 将本框架应用于 Reward Modeling 和 RLHF 流程优化

---

## 附录：实用建议（Algorithm 1 总结）

```python
def JUDGE_STRATEGY_SELECTOR(model_family, task_type, budget_multiplier):
    if task_type == "adversarial / high-stakes":
        return S5  # CoT Forcing
    elif model_family == "Claude":
        return S8 if budget_multiplier > 2 else S5
    elif model_family == "Llama":
        return S8 if budget_multiplier > 2 else S4
    elif model_family == "Gemini Pro":
        return S1
    elif model_family == "Gemini Flash":
        return S5
    elif model_family == "GPT-4o":
        return S8 if budget_multiplier > 2 else S5
    else:
        return S5  # Safe default
```

> 📦 **资源链接**：代码、数据、缓存结果已开源 → [GitHub: sksoumik/llm-as-judge](https://github.com/sksoumik/1lm-as-judge)

--- 

> **一句话总结**：  
> 本文首次系统揭示了 **Style Bias 是 LLM-as-a-Judge 中最严重却被忽视的偏见**，并通过大规模实证证明 **去偏策略的有效性高度依赖于模型与任务情境**，提出了首个基于证据的策略选择指南，并强调 **Chain-of-Thought 是最安全的通用去偏手段**。

</details>

---

### 15. [Agentic Adversarial Rewriting Exposes Architectural Vulnerabilities in Black-Box NLP Pipelines](https://arxiv.org/abs/2604.23483)

**Authors**: Mazal Bethany, Kim-Kwang Raymond Choo, Nishant Vishwamitra, Peyman Najafirad  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23483v1  

#### Abstract
Multi-component natural language processing (NLP) pipelines are increasingly deployed for high-stakes decisions, yet no existing adversarial method can test their robustness under realistic conditions: binary-only feedback, no gradient access, and strict query budgets. We formalize this strict black...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Agentic Adversarial Rewriting Exposes Architectural Vulnerabilities in Black-Box NLP Pipelines*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**多组件 NLP 管道系统**（如 RAG、multi-agent systems）在现实部署中的**对抗鲁棒性测试难题**。现有对抗攻击方法（如 token-level 替换）无法有效应用于现代黑盒 NLP 系统，原因如下：
- **仅提供二值反馈**（binary decision），无梯度或概率输出；
- 存在严格的**查询预算限制**（query budget），API 调用成本高；
- 多阶段架构使得攻击需同时破坏多个模块（如 retrieval 和 inference）。

因此，缺乏能在真实黑盒条件下（no gradient, binary-only feedback, low query budget）有效测试 NLP 管道鲁棒性的方法。

### 提出了什么新方法或新思路
提出了一种**基于代理（agentic）的双智能体对抗重写框架**（two-agent adversarial rewriting framework），其核心思想是：
- 在**语义扰动空间**中生成保持原意但可规避检测的改写文本；
- 使用两个协作的 LLM Agent：
  - **Attacker Agent**：负责生成语义等价、语言流畅的对抗性重写；
  - **Prompt Optimization Agent**：基于每次查询的**二值反馈**（成功/失败），迭代优化 Attacker 的提示指令，引导搜索更有效的攻击策略。

该方法完全运行于推理时，无需微调、无梯度访问，仅依赖最多 **10 次查询**即可完成攻击。

### 相比现有方法的优势
- ✅ **适用于严格黑盒场景**：唯一能在 binary-only 反馈 + 低查询预算下工作的攻击方法；
- ✅ **语义级扰动优于 token-level 攻击**：通过整体句子重构而非局部替换，能跨阶段干扰 pipeline；
- ✅ **无需 surrogate model**：直接在目标系统上进行攻击，避免因模型差异导致迁移失败；
- ✅ **自适应性强**：Prompt Optimization Agent 能从历史尝试中学习并调整策略，尤其对鲁棒系统效果显著。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LIAR-New dataset**：包含 1,957 条来自 PolitiFact 的政治声明，涵盖医疗、经济、移民等多个领域。
- 选取其中“Possible”和“Hard”两类（共 500 条样本用于实验），标签映射为二分类：
  - `False`：Pants-fire / False / Mostly-false
  - `True`：Half-true / Mostly-true / True

### 实验设置和评估指标
#### 目标系统（Target Pipelines）
评估四个证据型 misinformation detection 系统：
| 系统 | 架构特点 |
|------|---------|
| **Verifact** | Google Search API + GPT-4o-mini |
| **ICL** | In-context learning + Qwen2-VL-72B-Instruct |
| **ClaimBuster** | 基于关键词匹配的静态检索（legacy system） |
| **Perplexity** | Sonar model + 实时网络搜索 |

#### 攻击设置
- **最大查询预算**：10 次 / 输入
- **约束条件验证模块**（Constraint Validation Module）确保每次生成满足：
  1. **语义等价性**（Semantic Equivalence）：
     - MPNet 相似度 ≥ 0.61
     - BERTScore F1 ≥ 0.77
     - GPT-4o 判断核心断言一致
  2. **语言连贯性**（Linguistic Coherence）：由 GPT-4o-mini 判定语法正确、自然流畅
- 成功攻击定义为：`f(x') ≠ y*` 且所有约束均满足

#### 评估指标
- **Attack Success Rate (ASR)**：成功诱导误分类的比例
- 95% Wilson 置信区间
- 文本分析指标：Flesch Reading Ease, BERTScore, Levenshtein Distance, Parse Tree Distance 等

### 基线方法对比
采用四种主流 token-level 黑盒攻击方法，均通过 **TextAttack** 框架实现，并使用 **RoBERTa-base** 作为 surrogate model：
- **CLARE**
- **DeepWordBug**
- **TextBugger**
- **TextFooler**

> ⚠️ 注意：这些基线必须依赖 surrogate model，因为它们需要预测概率来指导搜索，而真实目标系统不提供此类信息 —— 因此属于“结构不对称比较”。

此外还设置了内部消融变体：
- **Ours (Attacker Only)**：无 Prompt Optimization，固定提示
- **Ours (Previous Step)**：仅用最近一次尝试更新提示
- **Ours (Full History)**：利用全部历史尝试进行策略优化

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Attack Success Rate %）

| Method | ICL | Verifact | ClaimBuster | Perplexity |
|--------|-----|----------|--------------|------------|
| CLARE | 3.90 | 1.46 | 26.56 | 0.46 |
| DeepWordBug | 1.97 | 1.46 | 10.82 | 0.23 |
| TextBugger | 2.80 | 1.94 | 22.95 | 0.00 |
| TextFooler | 3.09 | 2.18 | 23.93 | 1.16 |
| **Ours (Attacker Only)** | 20.11 | 35.92 | 95.02 | 14.55 |
| **Ours (Previous Step)** | 25.00 | 39.56 | 93.05 | 18.14 |
| **Ours (Full History)** | **30.35** | **40.34** | **97.02** | **19.95** |

> 所有 CI ∈ ±1.5–4.3 pp，且与基线无重叠，统计显著。

### 与基线方法的对比结果
- 我们的方法在现代 LLM-based 系统上实现了 **19.95% ~ 40.34%** 的攻击成功率；
- 最强 token-level 基线（CLARE）最高仅达 **3.90%**；
- 即使是最简单的 **Attacker Only** 版本，也远超所有基线（14.55% 起步），说明**语义级重写本身具有根本优势**。

### 消融实验结果
- **Prompt Optimization 显著提升性能**：
  - 对最鲁棒的目标（ICL），Full History 比 Attacker Only 提升 **+10.24 pp**；
  - 对脆弱系统（ClaimBuster），仅提升 +2.00 pp，表明优化价值随系统鲁棒性增加而增大；
- **Full History > Previous Step > Attacker Only**，证明历史记忆有助于发现复杂攻击模式。

---

## 4. 关键结论和发现

### 论文的主要发现
1. 🎯 **多阶段 NLP 管道存在严重黑盒漏洞**：即使没有梯度和概率反馈，在仅 10 次查询内仍可实现高达 **40.34%** 的逃逸率。
2. 🔍 **攻击有效性高度依赖架构设计**，形成“脆弱性谱系”（vulnerability spectrum），由三大因素决定：
   - **Evidence Retrieval Mechanism**：基于关键词匹配的系统（如 ClaimBuster）极易被绕过（97.02%）；
   - **Retrieval-Inference Coupling**：紧耦合设计更能抵抗跨阶段攻击；
   - **Baseline Accuracy**：准确率越高，决策边界越宽，抗攻击能力越强。
3. 💡 **成功攻击呈现四大 exploitation patterns**：
   - **Pattern 1**: Hedging and Ambiguity Injection（引入模糊词如 “might”, “reportedly”）
   - **Pattern 2**: Structural Elaboration（扩展文本长度，稀释核心主张）
   - **Pattern 3**: Complexity Escalation（提高语言复杂度，降低可读性）
   - **Pattern 4**: Syntactic Restructuring（大幅修改句法结构但仍保语义）
4. 🛡️ **提出 Pattern-Informed Defense**：基于 Pattern 3 设计文本简化预处理，可将攻击成功率降低 **最多 65.18%**（在 ICL 上从 30.35% → 10.57%）。

### 方法的局限性
1. **语义等价性验证可能存在偏差**：使用 GPT-4o 验证 GPT-4o 生成的内容，存在同族模型偏见（same-family bias），尽管已用 MPNet/BERTScore 缓解。
2. **人类评估不足**：阈值校准未覆盖实际攻击输出，需进一步人工验证 false positive rate。
3. **单次运行估计**：未进行 multi-seed 实验，LLM 生成随机性未充分建模。
4. **数据集和任务范围有限**：目前仅在 LIAR-New 和 misinformation detection 上验证，泛化性待检验。
5. **暂不支持多类别输出或自由文本响应系统**。
6. **依赖商业 API**：GPT-4o 可能变更，影响复现性。

### 未来工作方向
- 在其他多阶段 NLP 任务中验证框架通用性，如：
  - **RAG-based QA**
  - **Tool-augmented LLM applications**
- 开展**受控消融实验**（controlled ablation），明确各 exploitation pattern 的因果贡献。
- 构建**组合式防御机制**，结合：
  - 文本简化（text simplification）
  - 扰动检测（perturbation detection）
  - 推理一致性检查（consistency checking）
- 探索**跨 pipeline 的攻击迁移性**，识别共享漏洞。
- 将该框架用于**红队测试**（red-teaming）标准流程，推动安全评估规范化。

---

> ✅ **总体评价**：本文首次实现了在真实黑盒条件下对现代多组件 NLP 管道的有效对抗测试，揭示了当前系统存在的深层架构脆弱性，并提出了可解释的攻击模式与相应防御方案，为构建更鲁棒的复合 AI 系统提供了重要洞见。

</details>

---

### 16. [LLM-Guided Agentic Floor Plan Parsing for Accessible Indoor Navigation of Blind and Low-Vision People](https://arxiv.org/abs/2604.23970)

**Authors**: Aydin Ayanzadeh, Tim Oates  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23970v1  

#### Abstract
Indoor navigation remains a critical accessibility challenge for the blind and low-vision (BLV) individuals, as existing solutions rely on costly per-building infrastructure. We present an agentic framework that converts a single floor plan image into a structured, retrievable knowledge base to gene...

---

### 17. [Beyond the Attention Stability Boundary: Agentic Self-Synthesizing Reasoning Protocols](https://arxiv.org/abs/2604.24512)

**Authors**: Dahlia Shehata, Ming Li  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.24512v1  

#### Abstract
As LLM agents transition to autonomous digital coworkers, maintaining deterministic goal-directedness in non-linear multi-turn conversations emerged as an architectural bottleneck. We identify and formalize a systemic failure mode termed the Attention Latch in decoder-only autoregressive Transformer...

---

### 18. [XGRAG: A Graph-Native Framework for Explaining KG-based Retrieval-Augmented Generation](https://arxiv.org/abs/2604.24623)

**Authors**: Zhuoling Li, Ha Linh Hong Tran Nguyen, Valeria Bladinieres, Maxim Romanovsky  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.24623v1  

#### Abstract
Graph-based Retrieval-Augmented Generation (GraphRAG) extends traditional RAG by using knowledge graphs (KGs) to give large language models (LLMs) a structured, semantically coherent context, yielding more grounded answers. However, GraphRAG reasoning process remains a black-box, limiting our abilit...

---

### 19. [BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate](https://arxiv.org/abs/2604.25203)

**Authors**: Arnon Mazza, Elad Levi  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.25203v1  

#### Abstract
Deploying guardrails for custom policies remains challenging, as generic safety models fail to capture task-specific requirements, while prompting LLMs suffers from inconsistent boundary-case performance and high inference costs. Training custom classifiers achieves both accuracy and efficiency, yet...

---

### 20. [From World-Gen to Quest-Line: A Dependency-Driven Prompt Pipeline for Coherent RPG Generation](https://arxiv.org/abs/2604.25482)

**Authors**: Dominik Borawski, Marta Szulc, Robert Chudy, Ma{\l}gorzata Giedrowicz, Piotr Mironowicz  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.25482v1  

#### Abstract
Large Language Models (LLMs) have shown strong potential for narrative generation, but their use in complex, multi-layered role-playing game (RPG) worlds is still limited by issues of coherence, controllability, and structural consistency. This paper explores a dependency-aware, multi-stage prompt p...

---

### 21. [LLM-ReSum: A Framework for LLM Reflective Summarization through Self-Evaluation](https://arxiv.org/abs/2604.25665)

**Authors**: Huyen Nguyen, Haoxuan Zhang, Yang Zhang, Junhua Ding, Haihua Chen  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.25665v1  

#### Abstract
Reliable evaluation of large language model (LLM)-generated summaries remains an open challenge, particularly across heterogeneous domains and document lengths. We conduct a comprehensive meta-evaluation of 14 automatic summarization metrics and LLM-based evaluators across seven datasets spanning fi...

---

### 22. [Comparative Study of Bending Analysis using Physics-Informed Neural Networks and Numerical Dynamic Deflection in Perforated nanobeam](https://arxiv.org/abs/2604.24768)

**Authors**: Ramanath Garai, Iswari Sahu, S. Chakraverty  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.24768v1  

#### Abstract
In this chapter, we investigate the bending behavior of a perforated nanobeam subjected to sinusoidal loading using an efficient and computationally robust Physics-Informed Functional Link Constrained Framework with Domain Mapping (DFL-TFC) method. Our aim is to determine the relationship between st...

---

### 23. [Nautile-370M: Spectral Memory Meets Attention in a Small Reasoning Model](https://arxiv.org/abs/2604.24809)

**Authors**: Maixent Chenebaux  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.24809v1  

#### Abstract
We present Nautile-370M, a 371-million-parameter small language model designed for efficient reasoning under strict parameter and inference budgets. Nautile-370M uses a hybrid backbone in which two SeqCond Attention (SCA) layers, a linear-time spectral sequence operator inspired by SeqCondenser, alt...

---

### 24. [LegalMidm: Use-Case-Driven Legal Domain Specialization for Korean Large Language Model](https://arxiv.org/abs/2604.25297)

**Authors**: Youngjoon Jang, Chanhee Park, Hyeonseok Moon, Young-kyoung Ham, Jiwon Moon, Jinhyeon Kim, JuKyung Jung, Heuiseok Lim  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.25297v1  

#### Abstract
In recent years, the rapid proliferation of open-source large language models (LLMs) has spurred efforts to turn general-purpose models into domain specialists. However, many domain-specialized LLMs are developed using datasets and training protocols that are not aligned with the nuanced requirement...

---

### 25. [Marco-MoE: Open Multilingual Mixture-of-Expert Language Models with Efficient Upcycling](https://arxiv.org/abs/2604.25578)

**Authors**: Fan Jiang, Yu Zhao, Chenyang Lyu, Tianqi Shi, Yichao Du, Feihu Jiang, Longyue Wang, Weihua Luo  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.25578v1  

#### Abstract
We present Marco-MoE, a suite of fully open multilingual sparse Mixture-of-Experts (MoE) models. Marco-MoE features a highly sparse design in which only around 5\% of the total parameters are activated per input token. This extreme sparsity, combined with upcycling from dense models, enables efficie...

---

### 26. [From Syntax to Emotion: A Mechanistic Analysis of Emotion Inference in LLMs](https://arxiv.org/abs/2604.25866)

**Authors**: Bangzhao Shu, Arinjay Singh, Mai ElSherief  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.25866v1  

#### Abstract
Large language models (LLMs) are increasingly used in emotionally sensitive human-AI applications, yet little is known about how emotion recognition is internally represented. In this work, we investigate the internal mechanisms of emotion recognition in LLMs using sparse autoencoders (SAEs). By ana...

---

### 27. [Why Search When You Can Transfer? Amortized Agentic Workflow Design from Structural Priors](https://arxiv.org/abs/2604.25012)

**Authors**: Shiyi Du, Jiayuan Liu, Weihua Du, Yue Huang, Jiayi Li, Yingtao Luo, Xiangliang Zhang, Vincent Conitzer, Carl Kingsford  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.25012v1  

#### Abstract
Automated agentic workflow design currently relies on per-task iterative search, which is computationally prohibitive and fails to reuse structural knowledge across tasks. We observe that optimized workflows converge to a small family of domain-specific topologies, suggesting that this combinatorial...

---

### 28. [Diverse Image Priors for Black-box Data-free Knowledge Distillation](https://arxiv.org/abs/2604.25794)

**Authors**: Tri-Nhan Vo, Dang Nguyen, Trung Le, Kien Do, Sunil Gupta  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.25794v1  

#### Abstract
Knowledge distillation (KD) represents a vital mechanism to transfer expertise from complex teacher networks to efficient student models. However, in decentralized or secure AI ecosystems, privacy regulations and proprietary interests often restrict access to the teacher's interface and original dat...

---

### 29. [GamED.AI: A Hierarchical Multi-Agent Framework for Automated Educational Game Generation](https://arxiv.org/abs/2604.23947)

**Authors**: Shiven Agarwal, Yash Shah, Ashish Raj Shekhar, Priyanuj Bordoloi, Vivek Gupta  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.23947v2  

#### Abstract
We introduce GamED.AI, a hierarchical multi-agent framework that transforms instructor-provided questions into fully playable, pedagogically grounded educational games validated through formal mechanic contracts. Built on phase-based LangGraph sub-graphs, deterministic Quality Gates, and structured ...

---

### 30. [FastOMOP: A Foundational Architecture for Reliable Agentic Real-World Evidence Generation on OMOP CDM data](https://arxiv.org/abs/2604.24572)

**Authors**: Niko Moeller-Grell, Shihao Shenzhang, Zhangshu Joshua Jiang, Richard JB Dobson, Vishnu V Chandrabalan  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.24572v1  

#### Abstract
The Observational Medical Outcomes Partnership Common Data Model (OMOP CDM), maintained by the Observational Health Data Sciences and Informatics (OHDSI) collaboration, enabled the harmonisation of electronic health records data of nearly one billion patients in 83 countries. Yet generating real-wor...

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
