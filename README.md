# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-19 08:52:05 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [KVDrive: A Holistic Multi-Tier KV Cache Management System for Long-Context LLM Inference](https://arxiv.org/abs/2605.18071)

**Authors**: Jian Lin, Jiazhi Mi, Zicong Hong, Haodong Wang, Qianli Liu, Haodyue Zhang, Peng Li, Song Guo  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2605.18071v1  

#### Abstract
Supporting long-context LLMs is challenging due to the substantial memory demands of the key-value (KV) cache. Existing offloading systems store the full cache in host memory and selectively fetch critical entries during decoding, but this strategy quickly hits a ceiling: sparsity cannot be pushed f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：KVDRIVE: A Holistic Multi-Tier KV Cache Management System for Long-Context LLM Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在长上下文（long-context）大语言模型（LLM）推理中，**Key-Value (KV) cache 的内存开销巨大**，其大小随上下文长度和批处理大小线性增长，远超 GPU 显存容量。现有 offloading 系统虽将 KV cache 存储于主机内存（DRAM），但仍面临以下瓶颈：
- **KV selection 和 fetching 阶段导致严重的 GPU 等待（stall）**；
- 缓存管理策略（如 LRU、LFU）未考虑注意力机制的语义特性，导致重复数据移动；
- 仅依赖 DRAM 不足以应对超长上下文，而直接将 KV cache 迁移到 SSD 会因带宽低引发严重性能退化。

### **提出了什么新方法或新思路**
作者提出 **KVDRIVE**，一个端到端的多层级 KV cache 管理系统，从**系统协同设计**角度优化缓存、调度与存储，而非依赖算法稀疏性改进。其三大核心技术为：

#### **(1) Attention-Based Cache Management（基于注意力的缓存管理）**
- 利用注意力机制中的**时间局部性**（temporal locality）：相邻 token 的关键 KV entries 高度重叠。
- 引入 **sliding window + lookahead eviction** 策略，在 GPU 内维护最近多个 token 的关键 KV entries，仅更新差异部分。
- 设计 **2D layer-head window scaling**，根据不同层和头的重用模式动态分配窗口大小，最大化缓存效率。

#### **(2) Elastic Pipeline Scheduling（弹性流水线调度）**
- 将传统串行流程（selection → fetching → computation）解耦为独立可调度阶段（SFC disaggregation）。
- 采用 **micro-batching** 实现细粒度并行，使 selection、fetching 与 computation 在不同微批次间重叠执行，消除流水线气泡。
- 联合调优 index size、cache size 和 micro-batch size，实现吞吐量与准确率的平衡。

#### **(3) Coordinated Multi-Tier KV Storage（协调的多级 KV 存储）**
- 构建 **HBM-DRAM-SSD 三级存储架构**，突破 GPU 和 DRAM 容量限制。
- **Importance-guided warm-up**：利用预填充阶段的注意力分布，初始化高重要性 KV entries 的层级放置。
- **SSD-aware layout**：通过语义连续打包（semantic-contiguity packing）和层头分区（layer-head partitioning）提升 I/O 局部性。
- **Parallel sparse synchronization**：按需异步加载特定 KV 块，避免全层迁移带来的 I/O 放大。

### **相比现有方法的优势**
| 维度 | 现有方法（如 Quest, ShadowKV, RetroInfer） | KVDRIVE |
|------|------------------------------------------|--------|
| **Caching** | 通用策略（LRU/LFU），忽略注意力行为 | 注意力感知，支持滑动窗口重用 |
| **Scheduling** | 串行流水线，存在严重 stall | 弹性调度，细粒度重叠，消除 stall |
| **Tiering** | 仅 DRAM offloading 或粗粒度 SSD 加载 | 协调 HBM-DRAM-SSD，高效利用 SSD 容量 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LongBench**：双语、多任务长上下文理解基准。
- **RULER**：涵盖检索、多跳推理、聚合与问答的长上下文评测集。

### **实验设置和评估指标**
- **模型**：
  - `Llama-3-8B-1048K`（1M 上下文）
  - `Qwen-3-8B`, `Qwen-3-14B`（128K 上下文）
  - `Phi-4-Mini-128K`（128K 上下文）
- **硬件平台**：
  - **L20 Server**（48GB GPU, 100GB DRAM）
  - **H20 Server**（96GB GPU, 200GB DRAM）
  - **RTX 4090 Workstation**（24GB GPU, 120GB DRAM）
  - 所有系统均配备 NVMe U.2 SSD
- **评估指标**：
  - **Generation Throughput (tokens/s)**：主性能指标
  - **Accuracy**：在 RULER 和 LongBench 上的任务得分
  - **GPU Memory Usage**
  - **Prefill Latency**
  - **I/O Overhead**

### **基线方法对比**
| 基线方法 | 类型 |
|---------|------|
| `Original` | 全 KV cache 存于 GPU（无 offloading） |
| `FlexGen` | 全层 offloading，每步重新加载 |
| `Quest` | 基于 chunk 的稀疏 attention |
| `ShadowKV` | 基于 mean key 的 chunking + LRU 缓存 |
| `RetroInfer` | ANNS 向量检索 |
| `PQCache` | 乘积量化索引 |
| `MagicPIG` | LSH 分组采样 |

所有基线均在统一框架下复现以确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **最高达 1.74× 吞吐提升**：相比当前最优系统（如 ShadowKV），KVDRIVE 在多种配置下实现高达 **1.74× 的生成吞吐提升**。
- **维持 80%+ 缓存命中率**：得益于 lookahead eviction 和 sliding window，关键 KV entries 的命中率超过 80%，显著减少冗余传输。
- **支持百万级上下文**：在 Llama-3-8B-1048K 上成功运行 360k 上下文，batch size=1。

### **与基线方法的对比结果**
| 场景 | KVDRIVE vs 最佳基线 | 结果 |
|------|---------------------|------|
| L20 Server, Llama-3-8B, 120k ctx | **+70% 吞吐** | 显著优于 ShadowKV |
| H20 Server, Qwen-3-14B | **1.53× 更高吞吐** | 扩展性强 |
| RTX 4090, 120k ctx | **3× 吞吐于 H20 原生方案** | 成本效益极高 |
| 跨模型平均 | **1.23× ~ 1.74× 吞吐增益** | 一致性好 |

> **图 13 & 14** 显示，KVDRIVE 在所有上下文长度和 batch size 下均保持领先。

### **消融实验结果**
#### **(1) Lookahead Eviction 效果（表 3）**
| 方法 | Llama3-8B Hit Rate 提升 |
|------|------------------------|
| Quest | +0.9% ~ +1.3% |
| ShadowKV | +2.9% |
| RetroInfer | +1.3% ~ +3.9% |
| KVDRIVE | +1.5% ~ +3.0% |

👉 表明 lookahead eviction 具有广泛适用性，能有效提升各类系统的缓存命中率。

#### **(2) 2D Window Scaling（图 15）**
- 相比均匀窗口分配，**2D scaling 减少 15–30% 数据传输量**。
- 特定层（如深层）和特定头（如全局关注头）被分配更大窗口，提升资源利用率。

#### **(3) Window Size 影响（图 16）**
- 小 batch size（BS=1）时，较小窗口（size=2）更优（lookup 开销低）；
- 大 batch size（BS=4）时，较大窗口（size=4）更优（I/O 压力主导）；
- 说明需根据 workload 动态调优。

#### **(4) Chunk Size 与 Centroids 数量（图 17–18）**
- **最佳 chunk size = 4**：太小 → I/O 频繁；太大 → 冗余数据多。
- **Centroids 数量与上下文成正比**：60k 上下文用 4096 centroids，120k 用 8192。
- **减少 centroids 至 2048 可缩小索引 4×，精度无损**（图 19）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **注意力机制具有强时间局部性**，可通过 sliding window 实现高效缓存重用。
2. **串行调度是性能瓶颈**，SFC disaggregation 与 micro-batching 可有效隐藏 I/O 延迟。
3. **SSD 可作为有效扩展层**，只要配合 importance-aware warm-up 与 parallel sparse sync。
4. **系统级协同设计 > 算法级稀疏优化**：KVDRIVE 不依赖更强稀疏性，而是通过系统协同实现更高吞吐。

### **方法的局限性**
- **依赖离线 profiling**：2D window scaling 需预先分析各层头的重用模式。
- **对极端稀疏场景适应性未知**：极低 sparsity budget（<1%）下的表现未充分验证。
- **未集成量化或压缩技术**：仍以 FP16/BF16 存储 KV，未来可进一步压缩。

### **未来工作方向**
1. **扩展至多模态模型**：处理图像、音频等非文本 token 的 KV cache 模式。
2. **探索 Processing-in-Memory (PIM)**：在存储端执行部分 selection 或聚类计算，进一步减少数据移动。
3. **Tiered Mixed-Precision Storage**：
   - HBM 中 hot blocks 保留 FP16，
   - SSD 中 cold blocks 使用 INT4 量化，
   - 平衡精度与 I/O 带宽。

---

> ✅ **总结一句话**：  
> **KVDRIVE 通过“注意力感知缓存 + 弹性流水线 + 协同多级存储”的系统级协同设计，在不牺牲准确率的前提下，实现了高达 1.74× 的长上下文 LLM 推理吞吐提升，首次证明消费级 GPU 可高效服务百万级上下文任务。**

</details>

---

### 2. [HPC-LLM: Practical Domain Adaptation and Retrieval-Augmented Generation for HPC Support](https://arxiv.org/abs/2605.16347)

**Authors**: Nourin Shahin, Izzat Alsmadi  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2605.16347v1  

#### Abstract
Modern scientific research increasingly depends on High-Performance Computing (HPC) infrastructures, yet many researchers face significant operational barriers when interacting with cluster environments, job schedulers, GPU resources, and parallel computing frameworks. General-purpose large language...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《HPC-LLM: Practical Domain Adaptation and Retrieval-Augmented Generation for HPC Support》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代科学研究高度依赖 **High-Performance Computing (HPC)** 系统，但研究人员在使用集群环境、作业调度器（如 Slurm）、GPU 资源管理、并行计算框架（如 MPI）时面临显著的操作障碍。通用的 **Large Language Models (LLMs)** 在代码生成方面表现良好，但由于缺乏领域特定知识（如集群配置、调度语义、文件系统策略），难以提供可靠、准确的 HPC 支持。

此外，HPC 环境具有**机构特异性**（institution-specific），例如不同大学的集群命名规则、分区设置、模块加载方式各不相同，这使得通用模型无法直接适用。

---

### 提出了什么新方法或新思路
本文提出 **HPC-LLM** —— 一个面向 HPC 领域的、可本地部署的 **Retrieval-Augmented Generation (RAG)** 与 **轻量级领域适配（domain adaptation）** 结合的语言助手框架。其核心设计思想是：

- **自动化文档摄入**：通过爬虫从多个高校和研究机构的公开 HPC 文档中提取知识。
- **向量检索增强生成（RAG）**：在推理时动态检索相关文档片段作为上下文输入，确保响应基于权威来源。
- **轻量级微调（QLoRA）**：对 `Llama 3.1 8B` 模型进行参数高效微调，提升其对 HPC 指令的理解能力。
- **本地化部署架构**：整个系统可在资源受限的本地 GPU 上运行，无需依赖云 API。

该框架采用模块化设计，包含五个组件：API 层、多智能体协调器、爬虫代理、检索代理、生成代理和评估代理。

---

### 相比现有方法的优势
| 维度 | HPC-LLM 的优势 |
|------|----------------|
| **部署成本** | 支持在仅需 **5GB VRAM** 的设备上运行（如 RTX 3090），适合资源受限的科研机构。 |
| **准确性** | 结合 RAG 和领域微调，在 HPC 任务上的表现接近甚至优于更大规模的通用模型（如 Qwen 2.5 14B）。 |
| **可扩展性与定制性** | 可持续更新机构专属文档，支持个性化知识注入。 |
| **隐私与安全** | 完全本地运行，无数据外泄风险，适用于敏感科研环境。 |
| **反馈闭环** | 用户正面反馈自动加入训练集和检索库，实现持续学习。 |

> ✅ **核心创新点总结**：
> - 构建首个面向 HPC 的指令微调数据集（约 9,000–24,000 条问答对）
> - 提出“**轻量级微调 + RAG**”双轮驱动模式，平衡性能与资源消耗
> - 实现完全开源、可复现、可扩展的本地 HPC 助手基础设施

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
#### （1）训练数据构建流程（五阶段）
| 阶段 | 内容 | 规模 |
|------|------|------|
| Stage 1: Web Crawling | 抓取来自 TACC、SDSC、Harvard RC、NCAR、Texas A&M HPRC 等 35+ 个机构的 HPC 官方文档 | >2,880 个网页 |
| Stage 2: LLM-driven Q&A Generation | 使用 **Qwen 2.5 14B Instruct** 对每个文本块生成 3 组 Q&A 对 | ~9,000–21,000 对（占总量 60–70%） |
| Stage 3: Curated Expert Pairs | 手动编写 30 组高质量问答，覆盖 Slurm、MPI、PyTorch DDP、Singularity 等关键主题 | 30 对 |
| Stage 4: GPU Advisor Pairs | 编写约 50 组关于 H100/A100/V100 规格、VRAM 估算、NVLink、CUDA 优化等问题 | ~50 对 |
| Stage 5: Deduplication & Filtering | 去重（MD5）、过滤短文本（问题≥20字符，答案≥50字符） | 总计：**9,000–24,000 对** |

> 数据格式统一为 `Llama 3` 的 chat template，并附加 HPC 系统提示。

#### （2）测试数据
- `hpc_1000_prompts.txt`：包含 1,000 个 HPC 相关问题，涵盖 11 类任务：
  - Job scheduling, MPI, GPU computing, filesystems, modules, containers, data transfer, cluster access, debugging, policy, workflows
- 子集来自更大的 5,000 问题语料库

---

### 实验设置和评估指标

#### 实验平台
- **JetStream2** 云计算基础设施
- 多轮独立 benchmark 运行（Run 1 和 Run 2）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **BERTScore F1** | 主要质量指标，衡量生成文本与参考答案之间的语义相似性（经 baseline rescaling） |
| **ROUGE-L** | 最长公共子序列 F1 分数，用于评估表面词汇重叠（预期较低，因命令类输出差异大） |
| **Cosine Similarity** | 使用 BGE-large-en-v1.5 编码 prompt 与 response 后计算余弦相似度 |
| **HPC Domain Score** | 回答中出现的 HPC 专业术语比例（共 50+ 项） |
| **RAG Relevance** | 回答与 top-3 检索文档的嵌入相似度 |
| **Latency (s)** | 推理延迟（秒） |
| **VRAM Usage** | 显存占用 |

#### Composite Leaderboard Score
$$
\text{Score} = 0.35 \cdot \text{BERTScore F1} + 0.25 \cdot \text{Cosine} + 0.20 \cdot \text{HPC\_score} + 0.10 \cdot \text{ROUGE-L} - 0.10 \cdot \min(\frac{\text{Latency}}{20}, 1)
$$

---

### 基线方法对比
| 模型 | 参数量 | 是否微调 | 是否启用 RAG |
|------|--------|----------|-------------|
| **HPC-LLM (ours)** | 8B | 是（QLoRA） | 是 |
| Qwen 2.5 14B Instruct | 14B | 否（通用） | 是 |
| Phi-2 | 2.7B | 否 | 是 |
| Phi-3 | 14B | 否 | 是 |
| Mistral Nemo 12B | 12B | 否 | 是 |
| TinyLlama | 1.1B | 否 | 是 |

> 所有模型均使用 **Flash Attention 2**, **auto-quantization (4-bit NF4/BF16)**, **torch.compile** 等优化技术以保证公平比较。

---

## 3. 主要实验结果和性能指标

### Run 1: HPC-LLM vs. Phi-2, Phi-3, Mistral Nemo, TinyLlama  
**(n=1,000, RAG top-k=5)**

| Model | Size | Latency (s) | Cosine | ROUGE-L | **BERTScore F1** | Resp. Length |
|-------|------|-------------|--------|---------|------------------|---------------|
| Phi-2 | 2.7B | 5.71 | 0.696 | 0.070 | **0.846** | 138.1 |
| Phi-3 | 14B | 7.30 | **0.724** | 0.071 | 0.841 | 125.0 |
| Mistral Nemo | 12B | 6.12 | 0.715 | 0.065 | 0.841 | 123.5 |
| TinyLlama | 1.1B | 4.27 | 0.632 | 0.066 | 0.829 | 94.3 |
| **HPC-LLM (ours)** | **8B** | **5.22** | 0.608 | 0.070 | **0.808** | **90.7** |

> 📌 尽管 BERTScore 略低，但 HPC-LLM 是第二快且最简洁的回答者。

---

### Run 2: HPC-LLM vs. Qwen 2.5 14B（关键对比）

| Model | Size | Latency (s) | Cosine | ROUGE-L | **BERTScore F1** | Resp. Length |
|-------|------|-------------|--------|---------|------------------|---------------|
| Phi-2 | 2.7B | 9.12 | 0.677 | 0.072 | 0.845 | 135.4 |
| **Qwen 2.5 14B** | **14B** | **12.11** | 0.697 | 0.060 | **0.832** | 140.3 |
| TinyLlama | 1.1B | 6.97 | 0.616 | 0.072 | 0.833 | 95.7 |
| **HPC-LLM (ours)** | **8B** | **9.27** | **0.693** | 0.064 | **0.831** | 131.3 |

> 🔥 **核心发现**：
> - HPC-LLM 的 **BERTScore F1 (0.831)** 仅比 Qwen 2.5 14B（0.832）低 **0.001**
> - 推理速度 **快 23%**（9.27s vs 12.11s）
> - VRAM 占用仅需 **5GB**，约为 Qwen 14B 的 **1/3**

---

### 其他重要观察
- **ROUGE-L 普遍偏低（0.06–0.072）**：说明正确答案与提问之间词汇重叠少，符合命令类任务特性。
- **HPC-LLM 输出最简洁**：平均长度仅为 90–131 词，更适合 CLI 用户。
- **跨运行波动存在**：同一模型在不同硬件/提示子集下得分变化，强调 benchmark 控制条件的重要性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **轻量级领域适配 + RAG 可弥补模型规模劣势**  
   经过 QLoRA 微调的 **8B 模型**在 HPC 任务上能达到 **14B 通用模型**的性能水平，验证了“**domain calibration > pure scale**”的理念。

2. ✅ **资源效率极高**  
   HPC-LLM 仅需 **5GB VRAM** 即可部署，适用于 RTX 3090/4080 等消费级显卡，极大降低使用门槛。

3. ✅ **响应更简洁实用**  
   相比大模型倾向于冗长解释，HPC-LLM 更聚焦于提供精准命令和操作步骤，符合工程师需求。

4. ✅ **本地化 + 可持续进化可行**  
   支持自动抓取机构文档、用户反馈自动回流训练，形成闭环学习系统。

---

### 方法的局限性
| 限制 | 说明 |
|------|------|
| ❌ 缺乏专家验证基准 | 当前评估依赖自动指标（如 BERTScore），缺少人工标注的黄金标准答案集 |
| ❌ 合成数据可能引入幻觉 | 部分训练数据由 LLM 自动生成，可能存在错误或风格同质化问题 |
| ❌ 检索质量依赖文档质量 | 若原始文档陈旧、碎片化或结构混乱，会影响最终效果 |
| ❌ 未验证命令安全性与可执行性 | 未进行大规模命令执行测试，潜在存在危险指令风险（如 `rm -rf`） |
| ❌ 多集群泛化能力未知 | 当前主要针对美国高校集群，是否适用于工业级超算尚待验证 |

---

### 未来工作方向
1. **构建专家验证的 HPC Benchmark Dataset**  
   包含真实场景下的问题、标准答案、可执行命令验证机制。

2. **引入可执行命令验证机制**  
   在沙箱环境中模拟执行建议命令，检测语法正确性和副作用。

3. **加强幻觉与安全行为评估**  
   设计专门测试集评估模型是否会推荐不存在的命令或高危操作。

4. **改进检索鲁棒性**  
   引入 adaptive chunking、query rewriting、multi-hop retrieval 等策略提升召回率。

5. **探索 Retrieval-Aware Fine-Tuning (RAFT)**  
   在微调阶段就引入检索上下文，使模型学会区分“已知”与“需查证”的知识。

6. **集成 Tool Calling 与 Scheduler Integration**  
   实现真正的“AI Agent”，不仅能回答问题，还能提交作业、监控状态、自动调试。

---

## 总结

📌 **HPC-LLM 成功展示了：**

> “**一个经过轻量级领域适配的小模型 + RAG 检索增强 + 本地部署架构**”  
> 可以在 HPC 这类**高专业性、强上下文依赖**的任务中，达到媲美大模型的效果，同时大幅降低资源消耗和部署难度。

🎯 该工作为科研机构提供了**实用、开放、可持续演进**的 AI 辅助基础设施范本，推动了 **LLM 在科学计算领域的落地应用**。

</details>

---

### 3. [Roll Out and Roll Back: Diffusion LLMs are Their Own Efficiency Teachers](https://arxiv.org/abs/2605.16941)

**Authors**: Fanqin Zeng, Feng Hong, Geng Yu, Huangjie Zheng, Xiaofeng Cao, Ya Zhang, Bo Han, Yanfeng Wang, Jiangchao Yao  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.16941v1  

#### Abstract
Diffusion Large Language Models (DLLMs) promise fast parallel generation, yet open-source DLLMs still face a severe quality-speed trade-off: accelerating decoding by revealing multiple tokens often causes substantial quality degradation. We attribute this dilemma to a train-inference mismatch amplif...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Roll Out and Roll Back: Diffusion LLMs are Their Own Efficiency Teachers

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
**Diffusion Large Language Models (DLLMs)** 虽然理论上支持并行生成，具备高速推理潜力，但在实际开源模型中仍面临严重的 **quality-speed trade-off**（质量-速度权衡）：
- 若逐个解码（标准 decoding），虽能保证质量，但失去了并行优势；
- 若强行并行解码多个 token（naive parallel sampling），则因错误传播导致质量显著下降。

作者指出，这一困境源于 **train-inference mismatch**（训练-推理不匹配）和 **不可逆的解码过程**：
- **训练阶段**：从随机掩码状态重建 token，恢复顺序是隐式随机的；
- **推理阶段**：需要一个自适应的去噪顺序——简单 token 应早揭示，依赖上下文的 token 应延迟揭示；
- 一旦 token 被错误地提前揭示且无法撤销，错误将被固化并传播。

---

### ✅ 提出的新方法与思路

#### （1）**WINO (Wide-In, Narrow-Out)**：一种无需训练的可撤销解码算法
- **核心思想**：打破传统解码的“不可逆”特性，允许在后续步骤中重新修正早期生成的 token。
- **机制**：
  - **Draft（宽进）**：每步激进地解码多个 token（基于较低置信度阈值 `T1`）；
  - **Verify（验证）**：引入一个辅助的 **shadow block**，利用更丰富的全局上下文重新评估已生成 token 的可靠性；
  - **Fallback（窄出）**：对低置信度 token 进行 **re-mask**，留待后续步骤重生成。
- **优点**：
  - 无需额外训练，可直接应用于现有 DLLMs；
  - 实现“并行加速 + 动态纠错”，兼顾效率与质量。

#### （2）**WINO+**：通过轨迹注入（trajectory injection）将经验内化到模型参数中
- **核心思想**：让 DLLM 自己成为自己的“效率教师”。
- **机制**：
  - 利用 WINO 在离线推理中产生的 **verified denoising trajectories**（验证后的去噪路径）；
  - 提取每个 token 的 **finalization step**（最终稳定步数），构建有序的训练样本；
  - 修改训练目标为 **trajectory-ordered denoising**，而非随机重建：
    - 早稳定的 token 早监督；
    - 晚稳定的 token 保持掩码直到对应步数。
- **训练目标设计**：
  - `L_tok`：监督当前步应揭示的 token；
  - `L_defer`：抑制对“应延迟位置”的高置信错误预测；
  - `L_sharp`：增强正确但低置信预测的置信度。

---

### ✅ 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **与标准 DLLM 相比** | 显著提升推理速度（6–16× 步骤减少），同时提高或保持生成质量； |
| **与 naive parallel sampling 相比** | 避免因固定多 token 解码导致的质量崩塌； |
| **与 KV Cache 加速方法** | 不依赖缓存机制，解决的是根本性的解码顺序问题； |
| **与其它采样策略（如 EB Sampler）相比** | 引入动态回滚机制，主动纠正错误而非仅控制数量； |
| **与传统训练方式相比** | WINO+ 将推理中发现的有效顺序反馈给训练，实现训练-推理对齐。 |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集

| 类型 | 数据集 | 任务类型 |
|------|--------|----------|
| **语言任务** | GSM8K, MATH-500, HumanEval, MBPP, Countdown, Sudoku, ARC-E, ARC-C | 数学推理、代码生成、逻辑推理、常识推理 |
| **多模态任务** | Flickr30K, AI2D, MATH-Vision, MathVista, MMMU, ScienceQA | 图像描述、图表理解、视觉数学推理、跨学科推理 |

> 所有任务均采用 **zero-shot** 设置（Sudoku 为 4-shot）。

---

### ✅ 实验设置与评估指标

| 项目 | 设置 |
|------|------|
| **基础模型** | LLaDA-8B-Instruct（语言）、MMaDA-8B-MixCoT（多模态） |
| **解码策略** | Semi-autoregressive decoding（块长 128，生成长度 256） |
| **评估指标** | - 准确率（Accuracy）用于除 Flickr30K 外的所有任务<br>- CIDEr 用于 Flickr30K 图像描述任务 |
| **效率指标** | - 解码步数（Decoding Steps）<br>- 每秒生成 token 数（Tokens Per Second, TPS） |
| **WINO 参数** | Draft 阈值 `T1 ∈ {0.5, 0.6, 0.7}`，Verify 阈值 `T2 = 0.9` |
| **WINO+ 训练** | 基于 LoRA 的 post-training，使用 WINO 生成的轨迹进行监督 |

---

### ✅ 基线方法对比

| 基线方法 | 描述 |
|---------|------|
| **Standard Decoding** | 每步只解码 1 个 token，作为性能上限基准 |
| **Naive Parallel Sampling** | 每步固定解码 M 个 token（如 M=4 或 8），速度快但质量差 |
| **Random Trajectory Injection** | 控制变量：使用随机顺序而非 WINO 发现的顺序进行训练 |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table I 和 II）

#### 🔹 在 **GSM8K** 上的结果：
| 方法 | Accuracy | 解码步数 | 步骤缩减倍数 | TPS | TPS 加速比 |
|------|----------|-----------|----------------|-----|-------------|
| LLaDA (标准) | 73.24% | 256 | 1.00× | 17.76 | 1.00× |
| WINO | **75.82%** (+2.58) | 41.93 | **6.10×** | 100.53 | 5.66× |
| WINO+ | **76.58%** (+3.34) | **37.47** | **6.83×** | 121.86 | 6.86× |

> ✅ **不仅提速 6.8 倍，还提升了准确率！**

#### 🔹 在 **Flickr30K** 上的结果：
| 方法 | CIDEr | 解码步数 | 步骤缩减倍数 | TPS | TPS 加速比 |
|------|-------|-----------|----------------|-----|-------------|
| MMaDA (标准) | 53.67 | 256 | 1.00× | 6.41 | 1.00× |
| WINO | 53.83 | 25.47 | 10.05× | 55.11 | 8.60× |
| WINO+ | **63.38** (+9.71) | **15.78** | **16.22×** | 106.07 | **16.55×** |

> ✅ **CIDEr 提升近 10 分，TPS 加速超 16 倍！**

#### 🔹 在其他任务上的表现趋势一致：
- **Countdown**：WINO+ 将准确率从 24.21% → **48.05%**，TPS 加速 4.15×；
- **ARC-E**：从 59.13% → **84.97%**，TPS 加速 10.36×；
- **ScienceQA**：从 30.89% → **53.84%**，TPS 加速 11.02×。

---

### ✅ 消融实验结果

#### （1）**移除验证模块（Only Draft）**
- 若仅保留 Draft 模块（无 re-mask 能力），即使降低 `T1` 加快生成，也会因错误累积导致性能下降（见 Table IV）；
- 例如在 GSM8K 上，`T1=0.6` 时 Accuracy 仅为 70.28%，低于完整 WINO 的 75.82%；
> ❗ 验证与回滚机制对维持高质量至关重要。

#### （2）**轨迹来源消融（Random vs. WINO Trajectory）**
| 轨迹类型 | GSM8K Accuracy | 步骤 | MMMU-val Accuracy |
|--------|----------------|------|------------------|
| Random | 72.63% | 46.69 | 26.67% |
| WINO（本文） | **76.58%** | **37.47** | **28.11%** |
> ✅ 表明 **WINO 发现的真实去噪顺序** 是有效监督信号的关键。

#### （3）**损失函数组件分析**
| 组件组合 | GSM8K Acc / Steps | 结论 |
|--------|--------------------|------|
| `L_tok` only | 73.16 / 42.28 | 基础有效，但仍有错误揭示 |
| `+ L_defer` | 75.59 / 39.60 | 抑制错误揭示，显著提分 |
| `+ L_sharp` | **76.58 / 37.47** | 增强信心，进一步提速 |
> ✅ 三个损失项协同作用，缺一不可。

#### （4）**GPU 内存开销**
- **WINO**：因 shadow block 导致序列变长，内存略增（+2.4%）；
- **WINO+**：推理时不需 shadow block，**内存反而低于原始模型**；
> ✅ WINO+ 实现了 **更高效、更低内存占用** 的推理。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **DLLMs 可以作为自身的“效率教师”**：
   - 通过可撤销解码（WINO）发现可靠的去噪顺序；
   - 再通过轨迹注入（WINO+）将该顺序内化为模型能力；
   > 💡 “Roll Out and Roll Back”：先展开探索，再回滚学习。

2. **训练-推理对齐是提升 DLLM 效率的关键**：
   - 传统随机掩码训练无法反映高效推理所需的自适应顺序；
   - WINO+ 成功弥合了这一 gap。

3. **并行生成不必牺牲质量**：
   - WINO 实现了“大胆生成 + 安全修正”的机制；
   - WINO+ 进一步将这种行为“编译”进模型，实现轻量级高速推理。

---

### ⚠️ 局限性

1. **依赖 semi-autoregressive 结构**：目前方法基于块状解码设计，是否适用于完全自由的全序列扩散还需验证；
2. **offline 轨迹收集成本**：WINO+ 需要运行大量 WINO 推理来构建训练数据，增加预处理开销；
3. **对复杂任务依赖更强上下文建模**：极端复杂的推理任务可能仍受限于模型本身能力边界。

---

### 🔮 未来工作方向

1. **扩展到更多模态与架构**：如语音、视频等序列生成任务；
2. **在线自适应阈值调整**：动态调节 `T1`, `T2` 以适配不同输入难度；
3. **结合强化学习优化去噪策略**：将“何时揭示哪个 token”建模为决策问题；
4. **探索更高效的 trajectory compression 方法**：减少轨迹存储与训练负担。

---

## ✅ 总结一句话

> **本论文提出 WINO 与 WINO+，首次实现了 DLLM 在不牺牲质量的前提下大幅加速推理，并证明了模型可通过“自我教学”机制（roll out → roll back）自动发现并学习最优生成顺序，为下一代高效生成模型提供了新范式。**

🔗 代码地址：[https://github.com/Feng-Hong/WINO-DLLM/tree/WINO-plus](https://github.com/Feng-Hong/WINO-DLLM/tree/WINO-plus)

</details>

---

### 4. [CoX-MoE: Coalesced Expert Execution for High-Throughput MoE Inference with AMX-Enabled CPU-GPU Co-Execution](https://arxiv.org/abs/2605.17889)

**Authors**: Mu-Young Son, Yi Chen, Seungjae Yoo, Soongyu Choi Joo-Young Kim  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.17889v1  

#### Abstract
The Mixture-of-Experts (MoE) architecture improves computational efficiency via sparse expert activation, but throughput-oriented inference faces substantial GPU memory pressure due to a significant parameter size and intermediate data. Prior works attempt to mitigate this using expert offloading wi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CoX-MoE: Coalesced Expert Execution for High-Throughput MoE Inference with AMX-Enabled CPU-GPU Co-Execution

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Mixture-of-Experts (MoE) 模型虽然在计算效率上优于密集型 LLMs，但由于其巨大的参数量和中间激活数据，在**高吞吐量推理场景下面临严重的 GPU 显存（VRAM）压力**。现有方法如 micro-batching 和专家卸载（expert offloading）存在以下瓶颈：

- **Micro-batching 导致操作强度（operational intensity）下降**，使专家计算变为内存受限（memory-bound），降低整体吞吐；
- **CPU 卸载受限于 PCIe 带宽瓶颈**，且传统依赖 AVX 的 CPU 计算能力不足，难以有效分担 GEMM 密集型任务；
- **动态负载分配导致 GPU-CPU 工作负载失衡**，影响系统利用率。

### 🚀 提出的新方法与创新思路
本文提出 **CoX-MoE** —— 一种基于 **AMX-enabled CPU-GPU 协同执行** 的 MoE 推理优化框架，核心创新包括：

#### （1）**Coalescing-Aware Orchestration Policy（聚合感知调度策略）**
- 放弃对专家计算使用 micro-batch，改为在整个 batch 上进行 **coalesced expert execution（聚合专家执行）**，显著提升操作强度；
- 引入 **attention offloading** 策略：将 prefill 阶段中产生大量中间数据的 attention 运算卸载到 CPU 执行，从而释放 VRAM 给更多专家权重驻留；
- 联合优化非 MoE 操作（如 QKV 投影、attention）和专家计算的设备分配，实现资源高效利用。

#### （2）**Expert-Aware Stratification（专家感知分层预部署机制）**
- 利用输入数据语义聚类（clustering）选取代表性样本进行轻量级探测（probing），预测各 layer 中高频激活的专家；
- 将这些“热”专家静态预加载至 GPU 显存，减少运行时 PCIe 数据传输开销；
- 实现 **静态优先级部署**，缓解因稀疏路由带来的负载不均问题。

#### （3）充分利用 Intel AMX 加速能力
- 利用现代 CPU 支持的 **Advanced Matrix Extensions (AMX)** 提供高达 ~144 TFLOPs 的 BF16 矩阵乘法性能（相比 AVX-512 提升约 8 倍），使得 CPU 可以高效承担部分 GEMM 密集型计算；
- 为 CPU-GPU 协同提供了实际可行的计算基础。

### 🔍 相比现有方法的优势
| 方面 | CoX-MoE | Prior Works（如 FlexGen, MoE-Lightning） |
|------|--------|------------------------------------------|
| 批处理方式 | 对专家使用 full batch，避免 micro-batch 分割 | 普遍采用 micro-batch，加剧内存瓶颈 |
| 卸载对象 | 主动卸载 attention 以腾出 VRAM 存放专家权重 | 多数卸载专家权重本身，增加 PCIe 通信 |
| CPU 利用 | 充分利用 AMX 进行 GEMM 计算，支持 prefill 和 decode 阶段 | 仅用于 decode 阶段 GEMV 类操作 |
| 负载均衡 | 静态识别热点专家并合理分布，平衡 GPU/CPU 负载 | 动态决定，易造成负载倾斜 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集与模型
- **评估模型**：
  - **Mixtral-8x7B-Instruct (Mixtral)**：较少但较大的专家
  - **DeepSeek-V2-Lite (DeepSeek)**：较多小专家
  - **Qwen3-30B-A3B (Qwen3)**：典型 MoE 架构，广泛测试基准
- **任务类型**：文本生成（自回归解码）
- **输入/输出长度组合**：
  - 输入长度 `Lin`：97 和 800（分别代表短上下文与长上下文）
  - 输出长度 `Lout`：32 和 256（控制 decode 时间占比）

### ⚙️ 实验平台配置（见 Table 2）
| 系统编号 | CPU | GPU | 内存 | PCIe 版本 |
|---------|-----|-----|-------|-----------|
| I | Intel Xeon Platinum 8452Y (36核, AMX) | RTX 6000 Ada (48GB) | 512GB DDR5 | PCIe 4.0 |
| II | 同上 | A100 (80GB) | 同上 | PCIe 4.0 |
| III | 同上 | H100 (80GB) | 同上 | PCIe 5.0 |

> 所有系统均启用 AMX 支持，并通过扩展 Intel Extension for PyTorch (IPEX) 实现 NVIDIA GPU 兼容性。

### 📈 评估指标
- **主要指标**：**End-to-end Throughput（tokens/s）**
- 辅助分析指标：
  - 每层延迟（per-layer latency）
  - 显存占用分布（VRAM breakdown）
  - 专家命中率（expert hit ratio）
  - PCIe 传输开销

### 🆚 基线方法对比
- **FlexGen**：基于 micro-batch 和 zig-zag 调度的经典 offloading 框架
- **MoE-Lightning**：当前最先进的 batch inference 系统，采用 CGO 流水线 + 分页专家管理 + 层级 roofline 模型选择 micro-batch

> 两者均将 decode 阶段 attention 卸载至 CPU，但未考虑 prefill 阶段的优化。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（图 7）
在 `batch=1024` 下，CoX-MoE 在不同系统和模型上的平均吞吐表现如下：

| 对比项 | vs. MoE-Lightning | vs. FlexGen |
|--------|------------------|-------------|
| 平均吞吐提升 | **1.7× – 2.4×** | **3.4× – 7.1×** |
| 最大提升（Mixtral on Sys I） | **2.4×** | **7.1×** |

> **特别地，在 prefill-heavy 场景（如 Lin=800）中优势更明显**，因为 attention offloading 更有效地缓解了显存压力。

### 🔍 不同模型的表现差异
- **Mixtral 提升最大**：因其隐藏维度更大，decode 阶段 attention 更加 memory-bound，PCIe 传输代价更高；CoX-MoE 的 attention offloading 策略对此尤为有效。
- **Qwen3 和 DeepSeek**：同样获得稳定 1.7–2.0× 提升，验证通用性。

### 🧪 消融实验（Ablation Study，见 Table 3）
在 Qwen3、System I 上逐步添加组件的结果表明：

| 技术模块 | 吞吐量 (tokens/s) | 相对提升 |
|--------|------------------|----------|
| Baseline (MoE-Lightning) | 16.8 | — |
| + (a) Coalesced expert exec + AMX co-execution | 25.5 | **1.51×** |
| + (b) Attention offloading | 32.3 | **1.26×** |
| + (c) Expert-aware stratification (80% hit ratio) | 34.1 | **1.05×** |

> **结论**：三大技术均有正向贡献，其中 **coalesced execution + AMX 协同是最大驱动力**。

### 🎯 专家命中率分析（图 8）
- 在有限 VRAM 条件下（如只能缓存 30–50 个专家），EAS（Expert-Aware Stratification）相比随机选择可将 **hit ratio 提高约 40%**；
- 更高的 hit ratio 直接带来 **1.47–1.50× 的吞吐增益**，证明静态热点专家部署的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Micro-batching 并不适合 MoE 专家计算**：它严重削弱操作强度，导致专家计算陷入 memory-bound，反而成为性能瓶颈。
2. **卸载 attention 比卸载专家更优**：attention 是中间数据的主要来源，将其卸载能显著释放 VRAM，让更多专家驻留在 GPU，从而减少 PCIe 通信。
3. **AMX 使 CPU 成为有效的协同计算单元**：其高达 144 TFLOPs 的矩阵运算能力足以承担 GEMM 密集型任务，打破了以往“CPU 只能处理轻量任务”的限制。
4. **静态专家部署优于动态决策**：通过轻量级采样与探测即可准确预测热点专家，提前部署可大幅降低运行时开销。

### ⚠️ 方法的局限性
- **依赖 AMX 硬件支持**：目前仅限第4代及以后的 Intel Xeon CPU，不具备跨厂商兼容性（如 AMD 或 Apple Silicon）；
- **适用于批处理场景**：EAS 需要预先访问整个 batch 数据以进行聚类分析，对 online streaming 推理支持较弱；
- **模型结构敏感性**：对于专家数量极多或路由高度随机的 MoE 模型，stratification 效果可能下降。

### 🔮 未来工作方向
- 扩展至 **multi-GPU + multi-CPU** 架构下的分布式协同调度；
- 开发 **adaptive runtime profiler**，实现在无先验 knowledge 的情况下动态调整专家部署；
- 探索 **AMX + GPU Tensor Core 的细粒度流水线融合**，进一步重叠计算与通信；
- 支持更多硬件平台（如集成 AMX-like 单元的国产 CPU）。

---

> 💡 **总结一句话**：  
> **CoX-MoE 通过“聚合专家执行 + 注意力卸载 + 热点专家预部署”三位一体的设计，结合 AMX 强大的 CPU 算力，首次实现了在单 GPU 上高效运行大规模 MoE 模型的高吞吐推理，平均达到 SOTA 方法 2.4× 的性能提升。**

</details>

---

### 5. [Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery](https://arxiv.org/abs/2605.17976)

**Authors**: Xinzhe Yuan, Zhuo Chen, Jianshu Zhang, Huan Xiong, Nanyang Ye, Yuqiang Li, Qinying Gu  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.17976v1  

#### Abstract
Scientific discovery is increasingly constrained by costly experiments and limited resources, underscoring the need for efficient optimization in AI for science. Bayesian Optimization (BO), though widely adopted for balancing exploration and exploitation, often exhibits slow cold-start performance a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

**论文标题**: *Unleashing LLMs in Bayesian Optimization: Preference-Guided Framework for Scientific Discovery*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

传统 **Bayesian Optimization (BO)** 在科学发现任务中面临两大挑战：
- **Cold-start problem**: 初始阶段因缺乏有效先验而探索效率低、收敛慢。
- **高维可扩展性差**: 高维参数空间下性能下降明显。

尽管已有研究尝试引入 **Large Language Models (LLMs)** 来辅助 BO（如 warm-start 初始化或候选生成），但这些方法通常将 LLM 作为一次性或辅助组件，未能**持续、系统地整合其语义推理能力**到优化循环中。

### ✅ 提出了什么新方法或新思路

本文提出 **LLM-Guided Bayesian Optimization (LGBO)** ——首个将 LLM 的偏好引导**连续嵌入**到 BO 循环中的框架。

#### 核心机制：Region-Lifted Preference
- LLM 不再仅提供初始点或候选列表，而是每轮输出一个 **preferred region** 或 **point**，并附带一个置信度 `c ∈ [0,1]`。
- 该偏好被转化为对 **Gaussian Process (GP)** surrogate model 的**均值偏移 (mean shift)**，而不改变协方差结构。
- 数学上等价于在函数空间施加一个指数提升（exponential lift），通过 **Proposition 1** 可精确实现为 GP prior 的线性平移。

> 这种设计实现了“语义引导”与“统计严谨性”的统一：LLM 提供方向性建议，BO 保留决策控制权。

### ✅ 相比现有方法的优势

| 方法 | 局限性 | LGBO 如何改进 |
|------|--------|----------------|
| **标准 BO (GPBO)** | 冷启动慢，依赖随机初始化 | 引入 LLM 提供领域知识引导 |
| **LLAMBO / ADO-LLM** | LLM 仅用于 warm-start 或候选生成，后续仍由 acquisition function 主导 | **每轮更新 LLM 偏好**，持续影响 surrogate model |
| **ColaBO 类人类专家框架** | 接受一次偏好后固定，无法适应动态推理 | 支持 LLM 动态演化偏好，避免信息浪费 |

**优势总结**：
- **连续集成**：LLM 成为“语义专家”，全程参与优化。
- **稳定可控**：仅调整均值，不破坏 GP 结构，保证理论安全性。
- **无需额外超参**：指导强度 `λ` 由置信度 `c` 和不确定性自动校准。

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集

实验涵盖 **4 个 dry benchmark** 和 **1 个 wet-lab 实验**，覆盖多个科学领域：

| 数据集 | 领域 | 维度 | 目标 |
|--------|------|-------|------|
| **LNP3** | 药物递送（脂质纳米粒） | d=5 | 最大化载药量、包封率；最小化粒径 |
| **Cross-barrel** | 3D 打印结构设计 | d=4 | 最大化机械韧性 |
| **Concrete** | 水泥材料配方 | d=7 | 最大化抗压强度 |
| **HPLC** | 高效液相色谱 | d=6 | 最大化峰面积 |
| **Fe-Cr Redox Flow Battery**（湿实验） | 电池电解液优化 | d=3 | 最大化综合性能（粘度、电导率加权） |

> 所有 dry 实验基于公开数据集构建黑箱 oracle；wet 实验在真实实验室进行，无已知最优解。

### ✅ 实验设置和评估指标

- **评估方式**：
  - **收敛速度**：达到目标性能所需的迭代次数。
  - **最终性能**：优化结束时的最佳观测值。
  - **稳定性**：多次运行的标准差与轨迹热图（trajectory heatmap）。
- **运行配置**：
  - 每个任务运行 5 次（不同随机种子）。
  - 每轮采集 1 个新样本。
  - 初始 2 个点由 LLM 建议（LGBO 与 LLAMBO 共享以公平比较）。
- **Surrogate Model**：Matérn-5/2 kernel 的 GP。
- **Acquisition Function**：log-qEI。
- **LLM 模型**：Intern-S1-241B（科学领域预训练大模型）。

### ✅ 基线方法对比

| 基线方法 | 简介 |
|---------|------|
| **GPBO** | 标准 BO，Sobol 初始化 |
| **LLAMBO** | 使用 LLM 进行 warm-start 和候选生成，state-of-the-art LLM-augmented BO 方法 |
| **ColaLLM**, **BOPRO**, **CAKE**（扩展对比） | 新增三种近期 LLM-BO 方法，验证泛化性 |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 🔹 Dry Benchmark 结果（Figure 3, 9, 13, 14, 18）
- **LNP3**：LGBO 在约 10 轮内达到接近最优值，显著快于 GPBO 和 LLAMBO。
- **HPLC**：在高噪声环境下仍保持最高平均性能，且波动更小。
- **Concrete**：虽初期略慢，但后期迅速超越，最终达成最高抗压强度。
- **Cross-barrel**：快速逼近全局最优，并在后期集中探索高价值区域。

#### 🔹 Wet-Lab 实验结果（Figure 4）
- **Fe-Cr 电池电解液优化**：
  - **LGBO 在第 6 轮即达到最佳观测值的 90%**。
  - GPBO 和 LLAMBO 需超过 10 轮才能达到相同水平。
  - 最终性能高出约 15%，且跨运行方差更低。

> 表明 LGBO 在真实、昂贵、低数据量场景下具有显著实用价值。

### ✅ 与基线方法的对比结果

| 方法 | 收敛速度 | 最终性能 | 稳定性 | 备注 |
|------|----------|-----------|--------|------|
| **LGBO** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 全面领先 |
| **LLAMBO** | ⭐⭐⭐☆ | ⭐⭐⭐☆ | ⭐⭐☆ | 初期受益于 warm-start，但很快 plateau |
| **GPBO** | ⭐⭐ | ⭐⭐ | ⭐⭐ | 冷启动严重，探索分散 |
| **BOPRO / CAKE / ColaLLM** | ⭐⭐☆~⭐⭐⭐ | ⭐⭐☆~⭐⭐⭐☆ | ⭐⭐☆~⭐⭐⭐ | 均弱于 LGBO，尤其在早期 |

> 在所有 5 个数据集上（含高维 COF, d=14），LGBO 均表现最优或接近最优。

### ✅ 消融实验结果（Ablation Study）

#### （1）不同 LLM backbone 的影响（Figure 5）
- 更大的模型（如 Qwen3-235B）带来更强稳定性。
- 科学领域微调模型（Intern-S1）优于通用模型。
- **Thinking 模式模型收敛慢**：因其指令跟随能力较弱，在严格格式约束下表现不佳。

✅ 结论：**LGBO 框架对 backbone 具有鲁棒性**，但高质量 LLM 可进一步提升性能。

#### （2）随机 region lifting 对比
- 将 LLM 输出替换为**同大小、同置信度的随机区域**。
- 结果显示：收敛速度大幅下降，探索更加发散。
  
✅ 结论：**性能增益来自 LLM 的语义知识，而非 lifting 机制本身**。

#### （3）初始化消融
- LGBO 与 LLAMBO 使用相同的 LLM 初始化点。
- 但只有 LGBO 实现持续加速 → 说明优势来自**持续偏好集成**，而非 warm-start。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM 的语义推理可以被安全、有效地嵌入 BO 框架**，成为“持续专家顾问”。
2. **Region-lifted preference 机制兼具表达力与数学可处理性**，是连接语言模型与概率建模的理想接口。
3. **LGBO 在 worst case 下不会显著劣于标准 BO**，而在偏好对齐时能显著加速收敛（见 Theorem 1）。
4. **在真实湿实验中，LGBO 仅用 6 次迭代即达 90% 最优性能**，展示了其在资源受限科学任务中的巨大潜力。

### ✅ 方法的局限性

- **依赖 LLM 输出质量**：若 LLM 缺乏相关领域知识，可能提供误导性建议（尽管理论证明其损害有限）。
- **提示工程敏感**：需精心设计 system prompt 以确保输出格式一致性和科学合理性。
- **当前仅支持单目标优化**，多目标扩展尚未验证。

### ✅ 未来工作方向

- 扩展至 **multi-objective BO** 和 **constrained optimization**。
- 探索 **feedback loop**：将实验结果反馈给 LLM 以增强其自我修正能力。
- 结合 **vision-language models** 处理图像型实验数据（如显微镜图像、光谱图）。
- 在更多 **self-driving lab** 平台中部署，推动全自动科学发现。

---

> **总结一句话**：  
> LGBO 开创性地将 LLM 从“一次性助手”转变为 BO 中的“持续语义引导者”，通过 **region-lifted preference** 实现了高效、稳健、可解释的科学优化新范式，在多个真实任务中展现出显著优于现有方法的性能。

</details>

---

### 6. [Latent Action Reparameterization for Efficient Agent Inference](https://arxiv.org/abs/2605.18597)

**Authors**: Wenhao Huang, Qingwen Zeng, Qiyue Chen, Zijie Guo, Yu Sun, Cheng Yang, Siru Ouyang, Jiri Gesi, Fang Wu, Jiayi Zhang, Huaming Chen, Bang Liu, Xiangru Tang, Chenglin Wu  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.18597v1  

#### Abstract
Large language model (LLM) agents often rely on long sequences of low-level textual actions, resulting in large effective decision horizons and high inference cost. While prior work has focused on improving inference efficiency through system-level optimizations or prompt engineering, we argue that ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Latent Action Reparameterization for Efficient Agent Inference

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Model (LLM)** 的智能体（agent）在执行复杂任务时，通常依赖于长序列的低层级文本动作（low-level textual actions），导致决策步数（effective decision horizon）过长，推理成本高昂。尽管已有研究通过系统优化、prompt engineering 或硬件加速来提升效率，但这些方法并未触及根本——**动作空间本身的表示方式**。

论文指出，过度细粒度的动作表示是推理效率的瓶颈，即使每 token 生成更快，总步数不变仍会导致高延迟和计算开销。

### 提出的新方法：Latent Action Reparameterization (LAR)
作者提出 **Latent Action Reparameterization (LAR)**，一种学习紧凑潜在动作空间（latent action space）的框架。其核心思想是：
- 将多个语义上连贯的低级动作序列压缩为一个 **latent action**（潜在动作）。
- 每个 latent action 对应一个多步语义行为（multi-step semantic behavior），从而减少有效决策步数。
- 动作抽象需满足 **transition equivalence** 和 **executability**：即替换后不改变环境交互结果，且下游系统仍可解码执行。

与手工宏指令（macros）或分层控制器不同，LAR 中的 latent actions 是从 agent 轨迹中自动学习得到，并直接集成到模型中，实现端到端的高层决策。

### 相比现有方法的优势
| 方法类型 | 代表 | 局限性 | LAR 的优势 |
|--------|------|--------|-----------|
| Prompt Engineering | CoT, ConciseHint | 可能牺牲准确性，仅缩短文本长度 | 不改变语义，保持甚至提升准确率 |
| Token-Level 控制 | TokenSkip | 干预生成过程，可能破坏逻辑流 | 在更高级别重参数化，保留语义完整性 |
| Context 压缩 | ACON | 压缩历史记忆，不影响动作粒度 | 直接减少决策步数，降低 prefill 和 KV-cache 开销 |

> ✅ **核心优势**：LAR 通过改变“决策单元”本身，在不牺牲性能的前提下显著提升推理效率，是一种与模型架构、硬件优化互补的新路径。

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖三类典型 LLM agent 任务，体现多样性：
- **TriviaQA**：多步推理问答任务，强调 chain-of-thought 推理。
- **KodCode**：代码生成任务，具有高度结构化的动作模式（如函数模板、工具调用格式）。
- **Mind2Web**：网页交互任务，涉及丰富的工具使用和 HTML 协议 scaffold。

此外还测试了零样本迁移能力：
- **Musique**, **HumanEval**, **MBPP**：作为 held-out benchmark 验证泛化性。

### 实验设置
- **Backbone Models**：
  - `Qwen3-8B`
  - `Llama-3.1-8B-Instruct`
- **训练方式**：
  - 使用 **trajectory-level distillation** 进行训练。
  - 教师模型处理原始轨迹 $T$，学生模型处理 reparameterized 轨迹 $\tilde{T}$。
  - 仅更新 LoRA 参数（rank=8）和新增 latent action embeddings，冻结主干权重（parameter-efficient）。
- **Latent Action 学习流程**：
  1. 从 rollouts 中提取高频 n-gram。
  2. 使用 **next-token entropy** 作为 transition equivalence 的代理指标（低熵 ≈ 行为稳定）。
  3. 过滤条件：`freq ≥ f_min`, `H(s) ≤ H_max`。
  4. 构建 latent vocabulary 并进行 longest-first 匹配替换。

### 评估指标
| 指标 | 描述 |
|-----|------|
| **Task Performance** | 如准确率（accuracy）、成功率（success rate）等任务相关指标 |
| **Action Token Reduction** | 相对于 vanilla 方法的动作 token 数量减少比例 |
| **Wall-clock Inference Time** | 实际推理耗时 |
| **Token Throughput (TT)** | tokens/sec |
| **Peak GPU Memory (PG)** | 最大显存占用 |
| **Reparameterization Rate $r$** | 压缩率：$\frac{\sum |\tilde{T}_i|}{\sum |T_i|}$ |

### 基线方法对比
| 类型 | 方法 |
|------|-------|
| Vanilla | 原始 LLM agent |
| Reasoning Pattern | CoT, ReAct |
| Token-Level Efficiency | TokenSkip, ConciseHint |
| Context/Memory Optimization | ACON |

所有方法在相同解码设置和硬件下运行，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Backbone | Method | TriviaQA ↑ | KodCode ↑ | Mind2Web ↑ | Action Tokens ↓ |
|---------|--------|------------|-----------|-------------|------------------|
| Qwen3-8B | Vanilla | 67.40 | 34.44 | 36.73 | — |
|          | ReAct | 77.84 | 53.64 | — | — |
|          | **LAR** | **80.09** (**+2.25**) | **54.30** (**+0.66**) | **39.84** (**+3.11**) | **-27.1%** |
| Llama-3.1-8B | Vanilla | 73.63 | 31.13 | 24.40 | — |
|             | ReAct | 59.88 | 33.11 | — | — |
|             | **LAR** | **72.46** (-1.17) | **35.10** (+1.99) | **28.30** (+3.90) | **-23.3%** |

> 🔍 **观察**：
> - LAR 在多数任务上 **保持或提升任务性能**，同时实现 **显著的 token 减少（~9–27%）**。
> - 特别是在结构化强的任务（如 KodCode、Mind2Web）中增益更大。

### 系统级效率提升（Table 8）
| Model | Method | TT (tokens/s) ↑ | PG (GB) ↓ |
|-------|--------|------------------|------------|
| Qwen | ReAct → LAR | 127.8 → **150.2** (+17.5%) | 75.3 → **73.1** |
| Llama | ReAct → LAR | 151.6 → **152.2** | 42.5 → **42.4** |

> ✅ **LAR 引入零额外开销**，latent action 作为普通 token 处理，直接带来：
> - 更高的 token 吞吐
> - 更低的 KV-cache 占用
> - 更短的 end-to-end 推理时间

### 消融实验结果

#### （1）Held-out 泛化性（Table 2）
| Backbone | Method | Musique | HumanEval | MBPP |
|--------|--------|--------|-----------|-------|
| Qwen3-8B | ReAct | 27.61 | 89.63 | 74.17 |
|           | **LAR** | **26.57** | **91.46** | **75.50** |

> 📌 LAR 在未参与训练的数据集上依然表现良好，说明 learned latent actions 具备 **跨任务泛化能力**，捕捉的是领域级结构规律（如代码模板、tool call 格式），而非数据特异性模式。

#### （2）渐进式抽象消融（Progressive Abstraction Ablation）
实验逐步增加 reparameterization rate，观察性能变化趋势，发现三阶段现象：

| 阶段 | 行为 | 性能趋势 |
|------|------|--------|
| **Phase I: Moderate Abstraction** | 抽象低熵结构（如 scaffolds、protocol） | ✅ 性能提升 + 效率提高 |
| **Phase II: Boundary** | 达到最优压缩点 | ⚖️ 性能达到峰值 |
| **Phase III: Collapse** | 开始抽象高熵参数内容（如 query、entity） | ❌ 性能骤降（catastrophic failure） |

> 🚨 **关键发现**：存在明确的“抽象边界”，一旦越过将导致 **semantic failure**（环境交互中断）。这验证了 LAR 设计中 entropy filter 的必要性。

#### （3）Action Equivalence 分析（Table 3）
引入 **LAR-PT**（padding 到原长度）以排除长度影响：
- LAR > LAR-PT > ReAct
> 表明性能提升不仅来自压缩，更源于 **latent action 带来的语义抽象优势**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **动作表示是影响 LLM agent 推理效率的关键因素**，不应被忽视。
2. ✅ **Latent Action Reparameterization (LAR)** 能有效压缩冗余结构动作，显著减少有效决策步数。
3. ✅ LAR 在保持甚至提升任务性能的同时，带来系统级效率增益（token throughput ↑, memory ↓）。
4. ✅ latent actions 具备良好的 **跨任务泛化能力**，适用于同领域的不同 benchmark。
5. ✅ 存在一个清晰的 **抽象边界**：只能安全压缩低熵、上下文无关的结构部分；高熵参数内容必须保留。

### 方法的局限性
- **依赖于结构性重复**：在自由形式推理密集的任务中（如开放对话），可压缩空间有限。
- **静态 vocabulary**：learned latent actions 固定，难以动态适应新出现的模式。
- **对 entropy threshold 敏感**：过高可能导致语义破坏，需谨慎调参。
- **目前为 offline 训练**：尚未支持 online incremental learning。

### 未来工作方向
- 探索 **dynamic latent action discovery**，支持在线扩展。
- 结合 **hierarchical planning**，构建多层级 action space。
- 应用于更大规模模型（已初步验证至 Qwen3-32B，见 Appendix A.10）。
- 扩展至多模态 agent，统一视觉与语言动作表示。
- 与 speculative decoding、MoE 等其他高效推理技术结合。

---

> 💡 **总体评价**：  
> LAR 提出了一种新颖且有效的视角——将“动作表示学习”作为提升 LLM agent 效率的一等公民（first-class modeling choice）。其实验充分、机制清晰，揭示了结构冗余与语义可执行性之间的平衡原则，为未来高效 agent 设计提供了重要启示。

</details>

---

### 7. [JanusPipe: Efficient Pipeline Parallel Training for Machine Learning Interatomic Potentials](https://arxiv.org/abs/2605.18404)

**Authors**: Hongyu Wang, Weijian Liu, Hongtao Xu, Yan Wang, Mingzhen Li, Weile Jia, Guangming Tan  
**Category**: cs.DC  
**Published**: 2026-05-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.18404v1  

#### Abstract
Discovering atom-level phenomena requires molecular dynamics (MD) simulations with ab initio accuracy. Machine learning interatomic potentials (MLIPs) enable stable, high-accuracy MD simulations, and their models exhibit scaling-law trends similar to large language models. However, the lack of scala...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# JanusPipe: Efficient Pipeline Parallel Training for Machine Learning Interatomic Potentials 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **保守型 MLIPs**（Machine Learning Interatomic Potentials）在训练时具有独特的 **double-backward 执行模式**，即在前向传播阶段需要计算力（F = -∇ₓE），这引入了四个执行阶段：Forward Energy (FE)、Forward Force (FF)、Backward Force (BF) 和 Backward Energy (BE)。
- 现有的 **Pipeline Parallelism (PP)** 调度系统（如 1F1B、Hanayo）是为一阶模型设计的（仅前向+后向两阶段），无法高效支持这种四阶段依赖关系，导致：
  - **冗余重计算**：FF 需要 FE 的激活值和计算图，但在不同设备上执行时需重新计算 FE；
  - **参数复制与同步开销**；
  - **流水线气泡增大**（pipeline bubbles），降低 GPU 利用率。

### 提出了什么新方法或新思路
作者提出 **JanusPipe**，一个专为保守型 MLIPs 设计的高效分布式训练系统，其核心组件包括：

#### （1）SymFold
- 将一阶 PP 调度转换为适用于二阶 MLIP 的四阶段调度。
- 引入 **指令列表**（instruction list）抽象表示每个设备上的执行流程。
- 通过“对称折叠”将 FE 和 FF 共置于同一物理设备，实现本地复用 FE 激活，避免跨设备通信和重计算。
- 同时优化梯度路径，确保数学等价性。

#### （2）WaveK
- 针对四阶段运行时间不均衡的问题（tFE < tFF < tBE < tBF），提出自适应调度策略。
- 将多个 micro-batch 组合成 **WaveK 单元**，组织成 forward wave (FE+FF) 和 backward wave (BF+BE)，并通过波间重叠减少单元边界处的流水线气泡。
- 支持离线调优选择最优 unit size `k`，在吞吐量与内存占用之间取得平衡。

#### （3）GARS（Graph-Aware Re-Scheduling）
- 微批次重打包模块，缓解因图大小分布长尾引起的负载不均。
- 基于原子数进行排序与最小负载分配，并标记 micro-batch 是否可免通信执行（comm-free vs. dist）。

#### （4）3D 并行集成
- 支持 **PP/DP/GP** 三维并行（Pipeline/Data/Graph Parallelism），形成完整的分布式训练框架。

### 相比现有方法的优势
- 显著提升端到端训练吞吐量；
- 减少峰值 GPU 内存使用；
- 消除冗余计算与通信；
- 可扩展性强，在更大规模下仍保持高效率；
- 成功运行原本会 OOM 的配置。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 混合数据集，来自以下三个来源，按相同比例采样：
  - **ODAC23**（Direct Air Capture 分子）
  - **OMat24**（无机材料）
  - **OMol25**（有机分子）
- 每个训练迭代处理全局 batch 包含 **12,800 个原子**，划分为 micro-batch（每批 400 原子）。

### 实验设置
- **硬件平台**：ARMv8 CPU + NVIDIA A100-40GB GPU 集群，共 32 GPUs。
- **软件环境**：CUDA 12.4, PyTorch 2.6。
- **并行维度组合**：测试多种 PP/GP/DP 设置，例如：
  - P=8, G=2, D=2
  - P=4, G=4, D=2
  - P=4, G=2, D=4
- **模型族**：
  - **UMA**（Universal Models for Atoms）：稀疏 MoLE 架构，参数量 1.2B / 2.3B；
  - **eSEN**：全稠密架构，参数量 100M / 220M。

### 评估指标
- **训练吞吐量**（throughput）：以 **atoms/sec** 衡量；
- **峰值 GPU 内存使用**（peak GPU memory）；
- **强/弱扩展性**（strong/weak scaling efficiency）；
- **消融实验**验证各组件贡献。

### 基线方法对比
由于没有原生支持二阶训练的 pipeline 调度器，作者基于现有方案进行了适配：
- **1F1B-2nd**：Megatron-LM 中的 1F1B 调度扩展至四阶段，FF 侧需重算 FE；
- **Hanayo-2nd**：源自 Hanayo 的 wave-style 调度，也进行二阶适配；
- 所有基线均采用贪婪打包 micro-batch，且未启用 SymFold/WaveK/GARS。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **平均吞吐提升**（vs. 1F1B-2nd） | **1.51×** |
| **平均吞吐提升**（vs. Hanayo-2nd） | **1.45×** |
| **峰值内存降低**（vs. 1F1B-2nd） | 最多 **20.56%** |
| **峰值内存降低**（vs. Hanayo-2nd） | 最多 **42.70%** |
| **最大吞吐提升**（特定配置） | 达 **1.64×** |

### 与基线方法的对比结果
- 在所有测试模型（UMA/eSEN）和并行配置下，**JanusPipe 均显著优于两个基线**。
- 特别是在 **eSEN-220M** 上，基线出现 OOM，而 JanusPipe 可成功运行。
- 内存方面，JanusPipe 更接近设备上限利用，同时避免溢出。

### 消融实验结果
逐步启用各组件后的性能增益（相对于 1F1B-2nd）：

| 组件 | 吞吐提升幅度 |
|------|-------------|
| **+SymFold** | 最高达 **23%**（消除重计算） |
| **+WaveK** | 进一步提升 **18%**（减少气泡） |
| **+GARS** | 额外增加 **6–23%**（缓解负载不均） |
| **全部启用** | 总体达 **1.5–1.6×** 加速 |

> 注：WaveK 的 unit size `k` 对性能影响明显，通过离线搜索选择最佳 `k` 可进一步优化吞吐。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **保守型 MLIPs 的 double-backward 模式与传统 PP 不兼容**，直接沿用会导致严重性能退化。
2. **SymFold 有效解决了 FE/FF 分离带来的冗余问题**，通过共置实现本地激活复用，是性能提升的基础。
3. **WaveK 利用四阶段的时间偏序关系（tFE < tFF < tBE < tBF）**，通过波状调度与边界重叠显著减少流水线气泡。
4. **GARS 缓解了真实数据集中图尺寸异构带来的负载失衡问题**，尤其在 GP 场景下减少 halo All-Gather 开销。
5. **JanusPipe 实现了高效的 3D 并行训练**，为大规模 MLIPs 的可扩展训练提供了可行路径。

### 方法的局限性
- 当前调度设计依赖于稳定的四阶段时间偏序（tFE < tFF < tBE < tBF），若某些模型打破该顺序可能影响 WaveK 效果。
- 离线调优 `k` 虽然有效，但仍需手动配置或搜索，尚未完全自动化。
- 当前实现聚焦于 pipeline 层面优化，未整合 kernel-level 加速技术（如 FlashTP）或编译器级优化。

### 未来工作方向
- 探索更智能的在线调度机制，动态感知 workload 变化调整 `k`；
- 结合 **state sharding**（如 ZeRO/FSDP）进一步降低内存压力；
- 扩展至更多类型的物理约束模型（如电荷预测、磁矩建模）；
- 推动 MLIP 社区建立统一的 scaling law benchmark，助力大模型发展。

---

> ✅ **总结一句话**：  
> **JanusPipe 是首个专为保守型 MLIPs 设计的高效 pipeline parallel 训练系统，通过 SymFold + WaveK + GARS 三重创新，在 32 GPU 上实现了最高 1.64× 的吞吐提升，并显著降低内存消耗，推动了 MLIP 大模型训练的可扩展性边界。**

</details>

---

### 8. [Scalable Knowledge Editing for Mixture-of-Experts LLMs via Tensor-Structured Updates](https://arxiv.org/abs/2605.16686)

**Authors**: Roman Maksimov, Vladimir Aletov, Dmitry Bylinkin, Daniil Medyakov, Vladimir Solodkin, Aleksandr Beznosikov  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.16686v1  

#### Abstract
Knowledge editing (KE) provides a lightweight alternative to repeated fine-tuning of LLMs. However, most existing KE methods target dense feed-forward layers, while modern LLMs increasingly adopt Mixture-of-Experts (MoE) architectures for their superior memory footprint and inference efficiency. Thi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Scalable Knowledge Editing for Mixture-of-Experts LLMs via Tensor-Structured Updates*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大型语言模型（LLMs）越来越多地采用 **Mixture-of-Experts (MoE)** 架构以提升参数规模与推理效率。然而，现有的 **Knowledge Editing (KE)** 方法大多针对传统的**dense FFN 层**设计，无法有效适配 MoE 结构。这导致在 MoE 模型上进行高效、精确的知识编辑成为一个未被充分解决的问题。

现有方法如 **MoE-Edit** 虽然尝试处理 MoE 编辑，但其依赖逐层激活收集和专家间的块坐标下降（Block Coordinate Descent, BCD），存在严重的**顺序计算瓶颈**，难以扩展到大规模场景。

### 提出了什么新方法或新思路
本文提出了一种名为 **MoTE (MoE Tucker Editor)** 的新型知识编辑框架，专为 MoE 架构设计，具有以下核心创新：

- **Tensor-Structured Update Formulation**  
  将 MoE 层中的专家权重视为一个三阶张量 $ \mathbf{W} \in \mathbb{R}^{E \times d_{\text{model}} \times d_{\text{hidden}}} $，并利用 **Tucker 分解** 对更新量 $\Delta \mathbf{W}$ 进行低秩结构化建模，从而捕捉专家之间的共享结构。

- **Woodbury Identity 加速求解**  
  利用 **Woodbury 矩阵恒等式** 将原本需要对 $(E \cdot d_{\text{hidden}})$ 维矩阵求逆的操作，转化为仅需对大小为 $T \times T$（$T$ 为编辑批次大小）的小矩阵求逆，极大降低了计算复杂度。

- **单次激活 + 多层传播机制**  
  只需在**最后一层**收集一次激活值，即可通过 MEMIT 风格的残差传播机制将编辑效果扩散至多个关键层，避免了 MoE-Edit 中每层重复前向计算的开销。

### 相比现有方法的优势
| 特性 | MoE-Edit | MoTE（本文） |
|------|---------|------------|
| 是否闭式解 | 否（迭代优化） | 是（closed-form） |
| 是否需要反向传播 | 是 | 否（backward-pass-free） |
| 每层是否需重新收集激活 | 是 | 否（仅最后一层） |
| 专家间是否考虑结构相关性 | 否（独立更新） | 是（Tucker 分解建模共变结构） |
| 计算复杂度 | 高（$O(E^3)$ 或更高） | 低（主导为 $O(T^3)$） |
| 编辑速度 | 慢 | **快达 6×** |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **COUNTERFACT**：标准的单跳反事实知识编辑评测集，用于衡量模型能否成功修改特定事实。
- **ZsRE (Zero-shot Relation Extraction)**：零样本关系抽取数据集，测试编辑后模型在自然语言提示下的泛化能力。

### 实验设置和评估指标
#### 模型
在三个主流 MoE LLM 上进行实验：
- **Qwen3-30B-A3B**（128 专家，top-8）
- **GPT-OSS-20B**（32 专家，top-4）
- **Qwen3.6-35B-A3B**（256 专家 + 1 共享专家，top-8）

#### 评估指标
沿用 KE 领域标准指标：
- **Efficacy (%)**：在原始编辑提示下是否成功输出目标对象。
- **Generalization (%)**：在同义改写（paraphrase）提示下是否仍能正确回答。
- **Specificity / Locality (%)**：在无关邻近提示下是否保持原预测不变（防止干扰）。
- **Utility (%)**：上述三项的平均值，综合评价编辑质量。

#### 基线方法对比
- **Fine-tuning (FT)** 和 **FT-L**（带 L∞ 约束）
- **AdaLoRA**：参数高效微调方法
- **UnKE**：基于外部记忆的编辑方法
- **MoE-Edit**：当前唯一专为 MoE 设计的 KE 方法（主要对比基准）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | Method | Eff. | Gen. | Spe. | Utility |
|-------|--------|------|------|------|---------|
| Qwen3-30B-A3B | MoE-Edit | **97.90** | **86.50** | **83.45** | **89.28** |
| | **MoTE** | 97.00 | 82.60 | 79.57 | 86.39 |
| GPT-OSS-20B | MoE-Edit | 96.90 | 38.80 | 80.93 | 72.21 |
| | **MoTE** | **94.60** | **37.55** | **81.85** | **71.33** |
| Qwen3.6-35B-A3B | MoE-Edit | 93.40 | 78.55 | 77.99 | 83.25 |
| | **MoTE** | **89.20** | **68.44** | **75.33** | **77.66** |

> ✅ **结论**：MoTE 在所有模型上均达到**第二优性能**，部分指标接近甚至略超 MoE-Edit，尤其在 **Specificity（局部性）** 上表现优异，说明其对原有知识破坏更小。

### 与基线方法的对比结果
- MoTE 性能显著优于 FT、AdaLoRA、UnKE 等通用方法。
- 相比 MoE-Edit，虽然 Efficacy 和 Generalization 略有下降（约 1–4 个百分点），但换来了巨大的**效率提升**。
- 在 **GPT-OSS-20B** 上，MoTE 的 **Specificity 明显高于 MoE-Edit**，表明其编辑更具局部性。

### 消融实验结果（Table 2 & Appendix C）

#### （1）Tucker 结构 vs 无结构（Table 2）
| Method | Utility (Qwen3-30B) | Time (s) |
|--------|---------------------|----------|
| MoE-Edit | 89.28 | 477.0 |
| Global + speedup (no Tucker) | 82.72 | 81.6 |
| **MoTE (with Tucker)** | **86.39** | **76.8** |

> 🔍 **发现**：即使使用 Woodbury 加速，若不引入 Tucker 结构先验（即 flat matrix 更新），编辑质量会显著下降。**Tucker 分解是保证高性能的关键结构假设**。

#### （2）Whitening 类型消融（Table 5）
- **In-whitening**（基于隐藏状态协方差白化输入因子）效果最好。
- 不使用 whitening 导致性能大幅退化。
- Out-whitening 几乎无增益，甚至轻微负作用。

#### （3）Null-space 投影消融（Table 6）
| Null-space | Utility (Qwen3-30B) |
|-----------|---------------------|
| Enabled (√) | 86.39 |
| Disabled (×) | 49.17 |

> ⚠️ **关键发现**：**Null-space projection 至关重要**。移除该组件后，模型退化为随机猜测（~50%），说明它有效防止了知识覆盖过程中的灾难性遗忘。

#### （4）运行时间对比（Table 2）
| Model | MoE-Edit (s) | MoTE (s) | Speedup |
|-------|---------------|----------|---------|
| Qwen3-30B | 477.0 | 76.8 | ~6.2× |
| GPT-OSS-20B | 469.2 | 88.9 | ~5.3× |
| Qwen3.6-35B | 717.4 | 112.5 | ~6.4× |

> 🚀 **最大优势**：MoTE 实现了高达 **6× 的加速**，主要得益于：
> - 单次激活收集
> - Woodbury 小矩阵求逆
> - Tucker 压缩降低优化维度

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **MoE 架构可以支持闭式知识编辑**：通过恰当的代数建模（Tucker + Woodbury），无需梯度优化也能实现高质量编辑。
2. ✅ **结构先验至关重要**：将 MoE 权重视为张量而非拼接矩阵，并利用 Tucker 分解建模专家间的共变结构，是维持编辑性能的核心。
3. ✅ **效率可大幅提升**：相比迭代式 BCD 方法，MoTE 通过单次求解 + 多层传播，实现了高达 **6× 的速度提升**，更适合实际部署。
4. ✅ **局部性更好**：MoTE 在 Specificity 指标上普遍优于或媲美 MoE-Edit，说明其更新更“干净”，不易破坏无关知识。

### 方法的局限性
1. ❗ **大批次编辑时仍昂贵**：当编辑数量 $T > 1000$ 时，$T^3$ 的求逆成本会上升，可能限制超大规模批量编辑。
2. ❗ **依赖 HOSVD 预处理**：需要预先对 MoE 权重做 Tucker 分解（HOSVD），增加了初始化开销。
3. ❗ **未完全超越 MoE-Edit 性能**：尽管非常接近，但在 Generalization 等指标上仍有小幅差距，可能因结构压缩损失部分表达力。

### 未来工作方向
- 探索更高效的 Tucker 因子学习方式（如在线估计）。
- 将 MoTE 扩展到多跳知识编辑或多关系联合编辑任务。
- 结合 memory-based KE 方法，构建 hybrid 编辑系统。
- 研究如何在不访问完整权重的情况下进行远程 KE（适用于 API 模型）。

---

> 💡 **总体评价**：  
> MoTE 成功将经典的 **MEMIT-style closed-form KE 范式** 扩展到了 **MoE 架构**，并通过 **Tucker 结构建模 + Woodbury 加速** 实现了**性能与效率的优秀平衡**。它是迈向**可扩展、实用化 MoE 模型编辑**的重要一步，也为未来研究提供了清晰的技术路径。

</details>

---

### 9. [TriAxialKV: Toward Extreme Low-Precision KV-Cache Quantization for Agentic Inference Tasks](https://arxiv.org/abs/2605.17170)

**Authors**: Hanzhang Shen, Haoran Wu, Yiren Zhao, Robert Mullins  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.17170v1  

#### Abstract
Agentic workloads have emerged as a major workload for LLM inference. They differ significantly from chat-only workloads, requiring long-context processing, the ability to handle multimodal inputs, and structured multi-turn interactions with tool calling capabilities. As a result, their context exhi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**TriAxialKV: Toward Extreme Low-Precision KV-Cache Quantization for Agentic Inference Tasks**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **Agentic workloads**（代理型任务）对 LLM 推理提出了更高要求：长上下文、多轮交互、多模态输入、工具调用等，导致 **KV cache 快速膨胀**，成为内存瓶颈。
- 现有 KV-cache 压缩方法（如 KIVI、PM-KVQ）大多采用**单一维度异构策略**（如仅考虑时间或模态），忽略了 token 在多个结构性维度上的敏感性差异，导致在极端低精度下性能下降。

### 提出的新方法：**TriAxialKV**
- 一种新型的混合精度 KV-cache 量化方案，首次系统性地识别并利用了三个正交的 token 敏感性轴：
  - **Temporal Axis**（时间轴）：token 距当前轮次的远近（older, turn_m2, turn_m1, current）
  - **Modal Axis**（模态轴）：文本 vs 图像 token
  - **Semantic Axis**（语义轴）：功能角色（如 `inst`, `user`, `reasoning`, `tool_call`, `obs` 等）
- 每个 token 被赋予一个三轴标签（triaxial tag），基于该标签进行**感知敏感性的 INT2/INT4 混合精度分配**。

### 相比现有方法的优势
- **更细粒度的压缩策略**：联合建模三轴结构，避免“一刀切”的压缩，显著提升压缩效率与准确性平衡。
- **无需模型推理即可打标**：标签完全由 chat-template 结构决定（如特殊 token、分隔符），实现快速预处理。
- **端到端系统集成**：实现了从校准、内存管理到定制 Triton 内核的完整推理系统，支持在线动态解码。

---

## 2. 核心实验方法和设置

### 数据集
- **BFCL (Berkeley Function Calling Leaderboard)**：聚焦纯文本函数调用任务，强调结构化工具使用。
- **OSWorld**：多模态计算机操作任务，包含截图观察、GUI 操作等，上下文更复杂且长度可达 ~100K tokens。

### 模型
- **Qwen3 系列**：Qwen3-14B, Qwen3-32B, Qwen3-235B-A22B-Instruct
- **Qwen3-VL 系列**：Qwen3-VL-8B-Thinking, Qwen3-VL-32B-Thinking（多模态）
- **InternVL3.5-38B**（多模态）
- **Falcon3-10B-Instruct**

### 实验设置与评估指标
- **评估任务**：任务准确率（Task Accuracy %）
- **吞吐量测试**：端到端推理吞吐量（tokens/sec），在单张 GPU 上测量最大并发请求下的性能
- **硬件平台**：
  - NVIDIA B200（180GB HBM3e）
  - NVIDIA H100（80GB HBM3e）
- **系统框架**：基于 **SGLang v0.5.10** 构建，集成 FlashInfer 进行 prefill，自定义 Triton decode kernel

### 基线方法对比
| 方法 | 类型 |
|------|------|
| **SGLang BF16** | 全精度 KV-cache，作为无损基准 |
| **SGLang FP4** | 统一浮点 4-bit 量化 |
| **KIVI [26]** | 非对称 2-bit KV-cache 量化（INT2） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **任务准确率（Accuracy）**
- 在 **BFCL Memory** 上：
  - TriAxialKV Mixed 平均仅比 BF16 低 **<1.1%**，而 KIVI 和 FP4 下降达 4–5%
  - 表明其能有效保留关键信息（如 `inst` 中的工具签名）

- 在 **OSWorld** 上：
  - TriAxialKV Mixed **匹配甚至略超 BF16 基线**（如 Qwen3-VL-32B-Thinking 达 40.59 vs 39.20）
  - 显著优于 FP4 和 KIVI，说明三轴建模对多模态任务至关重要

#### ✅ **端到端吞吐量（Throughput）**
- 在 **Qwen3-VL-32B-Thinking + OSWorld** 上：
  - **B200 GPU**：达到 **1.32× BF16 吞吐**
  - **H100 GPU**：高达 **1.52× BF16 吞吐**
- 支持 **4.5× 更大的 KV cache 容量**
- 并发请求数提升 **3.4–4.0×**

#### ✅ **消融实验结果**

##### 🔹 三轴消融（Ablation Study）
| 方法 | Qwen3-14B | Qwen3-32B |
|------|----------|----------|
| Full (三轴完整) | 24.22 | 25.11 |
| No Temporal | 22.00 | 24.00 |
| No Semantic | 18.00 | 20.89 |

- **语义轴影响最大**：移除后准确率下降超 6%，因其保护了 `inst` 等关键段落
- 时间轴也有稳定增益，体现“近者重、远者轻”的注意力衰减规律

##### 🔹 内存预算扫描
| 平均 bitwidth B | 2.5 | 2.6 | 2.7 |
|------------------|-----|-----|-----|
| 准确率（Qwen3-14B） | 16.22 | 19.56 | 24.22 |

- 每减少 0.1 bit，准确率下降约 5%，验证了**精细校准的必要性**

---

## 4. 关键结论和发现

### 主要发现
1. **Agentic workloads 的 KV 敏感性具有强结构性**，可被三个正交轴（时间、模态、语义）几乎完全解释。
2. **统一或单轴压缩策略不足以应对复杂代理任务**，会导致关键 token 被过度压缩而引发任务失败。
3. **TriAxialKV 实现了极端低精度下的高保真压缩**：在平均 <3-bit 的情况下保持 BF16 级准确率。
4. **系统级优化带来显著吞吐收益**：通过混合精度内存池 + fused Triton kernel，释放了压缩带来的内存红利。

### 方法的局限性
- **非零样本迁移**：需为每个 workload/model 执行一次离线校准（虽成本低但不可免）
- **依赖标准 chat-template**：若 prompt 结构不规范（如无 `<think>` 或 `tool_call` 分隔符），tagger 可能失效
- **仅支持 INT2/INT4**：未探索更细粒度（如 INT3）或多级量化，存在进一步优化空间

### 未来工作方向
- 扩展至更多 bitwidth（如 INT3、FP6）以逼近帕累托前沿
- 自动化 tagger 设计，适应多样化 prompt engineering 风格
- 将三轴思想应用于权重量化或其他模型组件压缩

---

> **总结一句话**：  
> TriAxialKV 通过联合建模 **Temporal、Modal、Semantic** 三轴结构，实现了面向 Agentic Inference 的极致低精度 KV-cache 量化，在几乎不损失准确率的前提下，将吞吐提升 **30–50%**，并支持 **4.5× 更大上下文**，是迈向高效代理系统的重要一步。

</details>

---

### 10. [InfoFlow: A Framework for Multi-Layer Transformer Analysis](https://arxiv.org/abs/2605.17930)

**Authors**: Penghao Yu, Haotian Jiang, Zeyu Bao, Qianxiao Li  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.17930v1  

#### Abstract
While the approximation properties of single-layer Transformer architectures have been studied in recent works, a rigorous theoretical understanding of the multi-layer setting remains limited. In this work, we establish that multi-layer Transformers possess fundamentally different approximation capa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# InfoFlow: A Framework for Multi-Layer Transformer Analysis 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文旨在解决**多层 Transformer 架构的逼近能力理论分析不足**的问题。尽管单层 Transformer 的逼近性质已有一定研究，但多层架构在深度上的优势缺乏系统性的理论解释。特别是，现有理论难以解释为何某些任务在多层结构下可以高效完成，而在单层结构下则需要指数级增长的参数。

### **提出了什么新方法或新思路**

作者提出了 **InfoFlow** ——一个用于分析多层 Transformer 的抽象框架。其核心思想是将复杂的隐藏状态传播过程简化为**信息集（information set）的演化**，并结合**参数代价律（parameter cost law）** 来量化不同信息传播模式的成本。

InfoFlow 的设计基于两个关键理论发现：
- **Softmax Attention 的检索限制**：只能高效检索注意力得分最高的 token（即 argmax 位置），对第 k 大（k ≥ 2）token 的检索需要至少 Ω(e⁻ᵏ) 参数，其中 k 随序列长度 T 线性增长。
- **解码耦合信息的代价**：从聚合后的 token 状态中解码来自多个输入 token 的信息时，参数成本随参与 token 数量呈指数增长。

InfoFlow 定义了三种主要的信息传播机制：
1. **Max-position retrieval**：通过残差连接保留自身位置，并通过 attention 聚焦于最大得分位置。
2. **Global information aggregation**：一个 head 聚合整个序列的信息，代价高且随 T 增长。
3. **Specific position aggregation**：依赖 positional encoding 实现固定位置的选择（如前三个 token）。

### **相比现有方法的优势**

| 方面 | InfoFlow 的优势 |
|------|----------------|
| **理论可解释性** | 提供了一个“有效理论”式的抽象，类似于物理中的热力学模型，无需追踪具体表示即可预测逼近效率。 |
| **适用范围广** | 不仅能复现已知的单层逼近界（如 Yu et al., 2026），还能预测当前无法直接分析的多层行为。 |
| **指导性强** | 可用于预训练前的任务可行性判断、架构设计建议（如 head 数选择）。 |
| **与实验一致** | 框架预测与实际训练网络的行为高度一致，验证了其现实意义。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

所有实验均基于**合成任务（synthetic tasks）**，因为这些任务具有明确的“信息需求”结构，便于验证 InfoFlow 的预测能力。主要包括两类目标函数：

1. **Intrinsic Dimension Task**（内在维度任务）  
   $$
   F(X_T) = \sum_{i=1}^D \max_{1 \leq s,t \leq T} x(s)^T A_i x(t)
   $$  
   其中 $ A_i $ 是随机生成的不同矩阵。此任务要求进行 D 组独立的 pairwise 比较，InfoFlow 预测其“内在维度”为 D。

2. **Triangle-Center Task**（三角中心任务）  
   $$
   F(X_T) = \min_{1 \leq t_1,t_2,t_3 \leq T} \|x(t_1) + x(t_2) + x(t_3)\|
   $$  
   此任务涉及三个 token 的联合最小化，属于高阶比较任务，InfoFlow 预测其逼近难度随 T 快速上升。

输入 token $ x(t) \sim \mathcal{N}(0, I_d) $，经过归一化处理。

### **实验设置和评估指标**

- **模型结构**：使用标准 two-layer Transformer，无 positional encoding（除非特别说明）。
- **配置对比**：
  - Intrinsic Dimension 实验测试三种 head 配置：(D,D), (D−1,D), (D,D−1)
  - Triangle-Center 实验测试三种规模：(2,2)/E=24, (4,4)/E=24, (4,4)/E=48
- **优化器**：AdamW，学习率 1e-3，weight decay 1e-2，cosine annealing
- **训练样本数**：数万至数十万量级（i.i.d. 采样）
- **评估指标**：**NMSE**（Normalized Mean Square Error），值越小越好；接近 1 表示模型退化为常数预测。

### **基线方法对比**

本工作并非提出新的训练算法或模型结构，因此不与其他模型进行性能对比。而是将 **InfoFlow 的预测结果与实际训练表现进行对照**，以验证其有效性。

此外，在理论层面，InfoFlow 成功复现了以下已有工作的结论作为“自洽性验证”：
- **Yu et al. [2026]**：关于单层 Transformer 中 head 数不足导致参数成本指数上升的结论。
- **Sanford et al. [2024a,b]**：关于 induction head 任务需两层才能高效解决的结论。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **Intrinsic Dimension 实验结果（T=64）**

| D | (D,D) NMSE | (D−1,D) NMSE | (D,D−1) NMSE |
|----|------------|--------------|---------------|
| 2 | 5.89×10⁻⁵ | 5.89×10⁻² | 3.67×10⁻² |
| 3 | 6.37×10⁻⁵ | 2.64×10⁻² | 2.07×10⁻² |
| 4 | 6.92×10⁻⁵ | 1.39×10⁻² | 1.30×10⁻² |
| 5 | 4.69×10⁻⁴ | 4.46×10⁻³ | 7.19×10⁻³ |
| 6 | 1.99×10⁻³ | 6.26×10⁻³ | 7.67×10⁻³ |

> 🔍 **观察**：当 head 数满足 $ h_1 = h_2 = D $ 时，NMSE 达到 ~10⁻⁵ 量级；一旦任一层 head 数减少 1，性能下降 2–3 个数量级。

#### ❌ **Triangle-Center 实验结果（随 T 增大）**

| T | 所有配置 NMSE（平均） |
|----|------------------------|
| 3 | ~10⁻³ |
| 8 | ~0.08 |
| 16 | ~0.85 |
| 32 | ~0.96 |
| 64 | ~0.985 |

> 🔍 **观察**：随着 T 增加，所有配置的 NMSE 迅速趋近于 1，表明模型几乎无法学习该任务。即使增加 head 数或 embedding dimension，也无法缓解这一趋势。

### **与基线方法的对比结果**

- InfoFlow **成功预测了内在维度现象中的相变点**（phase transition at $ h_1=h_2=D $），与实验完全吻合。
- 对于 triangle-center 任务，InfoFlow 预测其 Number of Comparison 为 Ω(T³)，远超任何固定架构的能力，实验结果证实了这一点。
- 相比之下，传统直觉可能认为“只要模型足够大就能学会”，但实验证明这种高阶任务存在**根本性瓶颈**。

### **消融实验结果**

虽然未显式命名“ablation study”，但以下实验起到了类似作用：

1. **Max-position retrieval 验证实验**  
   - 在训练好的模型上拟合线性映射 $ g(x^{(1)}(t)) \to x(\text{argmax}_s A(t,s)) $
   - 结果显示：训练后恢复误差显著降低（ratio 下降至 0.08~0.84），而对第二大的位置则无改善（ratio ≈1）
   - ✅ 支持 Theorem 2：softmax attention 仅有效聚焦于 argmax 位置。

2. **Specific position aggregation 实验**  
   - 训练单层单头 Transformer 学习 $ F(X_T)=x(1)+x(2)+x(3) $
   - 观察到 CLS token 的 attention 权重稳定集中在 position 1,2,3 上（各 1/3），不受 token 内容影响。
   - ✅ 验证了 positional encoding 可实现 content-independent 的位置选择。

---

## 4. 关键结论和发现

### **论文的主要发现**

1. **深度分离现象（Depth Separation）**  
   存在一类任务，单层 Transformer 需要参数随 T 指数增长才能精确逼近，而两层 Transformer 仅需 O(ε⁻¹) 参数即可完成。这揭示了**深度带来的本质性表达优势**。

2. **Softmax Attention 的结构性局限**  
   Softmax attention 本质上只能高效检索 argmax 位置的信息；对于其他位置（如第二大），解码所需参数成本呈指数增长。这是由 softmax 的集中特性决定的。

3. **信息传播的三大机制**  
   InfoFlow 抽象出 max-position retrieval、global aggregation 和 specific position aggregation 三种主导模式，构成了理解多层 Transformer 信息流的基础语言。

4. **Intrinsic Dimension 是关键瓶颈**  
   即使在多层设置下，head 数仍是一个关键资源。若第一层 head 数小于任务的 intrinsic dimension D，则无法有效逼近目标函数。

5. **高阶检索任务存在根本困难**  
   如 triangle-center 这类需要同时比较三个及以上 token 的任务，其计算复杂度随 T 呈立方增长，任何固定大小的 Transformer 都无法高效逼近。

### **方法的局限性**

| 局限性 | 说明 |
|-------|------|
| **仅适用于 retrieval-type 任务** | 当前框架假设 active index set 大小不随 T 增长，难以推广到输出依赖全局变换的任务。 |
| **未考虑 attention weight matrix 的秩** | 参数代价律未纳入低秩约束的影响，可能高估实际需求（见 Amsel et al., 2025）。 |
| **基于合成任务验证** | 尚未在真实 NLP/CV 任务上验证其预测能力。 |
| **抽象层级较高** | 丢失了具体表示细节，不适合用于解释特定神经元功能或中间特征。 |

### **未来工作方向**

1. **扩展到更广泛的 target 类型**：如函数逼近、生成任务等。
2. **整合更多信息传播机制**：例如 feed-forward network 的非线性变换路径。
3. **应用于实际架构搜索**：利用 InfoFlow 自动推荐最优 head 数、depth、embedding dimension。
4. **与 Transformer Circuits 框架融合**：将宏观的 InfoFlow 与微观的 circuit 分析结合，形成多层次理解体系。
5. **探索动态信息流建模**：允许信息集根据输入内容动态调整传播路径。

---

> 💡 **总结一句话**：  
> InfoFlow 提供了一个**原理性框架**，首次系统地解释了多层 Transformer “为什么比单层更强”，并通过信息集演化和参数代价律，实现了对逼近效率的**定量预测**，为模型设计提供了新的理论工具。

</details>

---

### 11. [TeleCom-Bench: How Far Are Large Language Models from Industrial Telecommunication Applications?](https://arxiv.org/abs/2605.18025)

**Authors**: Jieting Xiao, Yun Lin, Huizhen Qiu, Rui Ma, Chen Zhong, Dongyang Xu, Xiao Long, Chaoyu Zhang, Qiaobo Hao, Ding Zou, Zhiguo Yang, Yanqin Gao, Fang Tan  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.18025v1  

#### Abstract
While Large Language Models have achieved remarkable integration in various vertical scenarios, their deployment in the telecommunications domain remains exploratory due to the lack of a standardized evaluation framework. Current telecom benchmarks primarily focus on static, foundational knowledge a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*TeleCom-Bench: How Far Are Large Language Models from Industrial Telecommunication Applications?*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在电信领域（telecommunications）中，尽管 Large Language Models（LLMs）已在多个垂直行业取得进展，但其工业级部署仍处于探索阶段。主要原因在于：
- **缺乏标准化评估框架**：现有基准（如 SPEC5G、TeleOnA）主要关注基础通信标准和理论知识，忽视了设备特定文档（equipment-specific documentation）和端到端工业流程（end-to-end workflows）。
- **“原子化”技能评估**：多数任务仅测试孤立的“atomic”能力（如问答、数学推理），无法反映真实运维场景中的多步决策与工具调用过程。
- **脱离生产环境**：缺少对实际网络操作轨迹（live network agent workflows）的建模，导致模型评估与现实脱节。

因此，该论文旨在填补这一空白，系统评估LLMs在工业电信应用中的真实能力。

---

### 提出了什么新方法或新思路
作者提出了 **TeleCom-Bench** ——一个面向工业电信应用的综合性基准测试框架，具有两大核心支柱：

#### （1）Multi-dimensional Knowledge Comprehension（多维知识理解）
- 融合了 **3GPP协议、5G网络架构、电信基础理论** 和 **厂商专有产品知识**（如设备手册、专利、现场案例）。
- 创新地采用 **双合成策略**：
  - **Knowledge Distillation-driven 方法**：从技术标准、研究论文等非结构化文本中提取结构化评估样本（如模板抽取、渐进式知识挖掘）。
  - **Knowledge Graph-driven 方法**：构建领域知识图谱（KG），通过图遍历生成单跳、多跳及聚合型 QA 对，确保逻辑连贯性和抗幻觉能力。

#### （2）End-to-End Knowledge Application（端到端知识应用）
- 基于真实网络运维轨迹（来自EMS/UME系统）构建六项连续任务链：
  1. **Intent Recognition**
  2. **Entity Extraction**
  3. **Event Verification**
  4. **Tool Invocation**
  5. **Root Cause Diagnosis**
  6. **Solution Generation**

这些任务模拟了网络优化与故障维护的真实闭环流程，并引入 **Agent-in-the-Loop 数据合成机制**，即利用已有智能体执行真实任务并记录其完整推理路径（Input-Reasoning-Output），从而生成高质量、可验证的评估样本。

此外，所有生成的动作均经过 **现场验证（Field Verification）** 和 **专家审核（Expert Review）**，确保“Ground Truth”具备物理可执行性。

---

### 相比现有方法的优势
| 维度 | 现有基准（如 SPEC5G, TeleLogs） | TeleCom-Bench |
|------|-------------------------------|----------------|
| **知识覆盖** | 公共标准为主，忽略厂商私有知识 | 包含设备手册、配置指南、现场案例等专有信息 |
| **任务粒度** | 孤立任务（atomic skills） | 多步骤、端到端流程建模 |
| **执行真实性** | 缺乏工具交互与命令生成 | 支持 Tool Invocation 与 Solution Generation |
| **数据来源** | 合成或公开数据集 | 来自商用现网的真实 telemetry 与工单轨迹 |
| **评估严谨性** | 静态 QA 准确率 | 引入闭环验证 + 专家审计 |

> ✅ **优势总结**：首次实现从“理论理解”到“工程执行”的全栈评估，推动LLM从“学术原型”向“生产就绪”演进。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **TeleCom-Bench 自建数据集**，包含 **12个子集**，总计 **22,678个样本**，分为两类：
  
  #### Knowledge Comprehension（共13,462样本）
  - **Basic Theory**（基础理论）：
    - 3GPP Protocols（4,043）
    - 5G Network（2,564）
    - Basic Knowledge（2,662）
  - **Product Knowledge**（产品知识）：
    - Wireless Network（3,725）
    - Wired Network（3,488）
    - Core Network（960）

  #### Knowledge Application（共9,216样本）
  - 全部基于真实运维流程采样：
    - Intent Recognition（2,174）
    - Entity Extraction（365）
    - Event Verification（146）
    - Tool Invocation（585）
    - Root Cause Diagnosis（983）
    - Solution Generation（983）

> 所有数据均来源于 ZTE 内部积累的 **1.52TB 异构原始资料**（PDF/CHM/DOCX/日志等），经统一预处理与结构化标注。

---

### 实验设置和评估指标

#### 模型选择（8个主流LLM）
| 模型 | 类型 |
|------|------|
| Qwen3-32B / Qwen3-235B | Dense 架构 |
| DeepSeek-V3.2 | MoE 架构 |
| Gemini 2.5 | Proprietary |
| Grok 4.1 | Proprietary |
| GLM-4.7 | Dense |
| Doubao-pro | Proprietary |
| Kimi K2 | Proprietary |

#### 推理参数
- 温度（temperature）= 0.7
- 启用 reasoning mode（支持 CoT 推理）
- 每样本独立采样3次，取多数投票为最终预测

#### 评估指标
| 任务类型 | 指标 |
|--------|------|
| Multiple-Select Questions | Macro-F1（解决类别不平衡） |
| Structured QA（JSON输出） | Exact Match（格式+内容完全匹配） |
| Subjective QA（开放回答） | **LLM-as-a-Judge** 三专家投票制，5分Likert量表评分，Krippendorff's α = 0.82（高一致性） |

---

### 基线方法对比
本文未直接对比传统NLP模型（如BERT系列），而是聚焦于 **state-of-the-art LLMs之间的横向比较**，揭示不同架构、训练策略对电信任务的影响。

特别分析了：
- 参数规模影响（Qwen3-32B vs Qwen3-235B）
- 架构差异（Dense vs MoE）
- 是否经过 function calling 微调（如 DeepSeek-V3.2 表现优异）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）

| 任务 | 最佳表现模型 | 性能（%） | 最差表现模型 | 性能（%） |
|------|-------------|----------|--------------|-----------|
| Intent Recognition | Qwen3-235B | 94.52 | GLM-4.7 | 92.69 |
| Entity Extraction | 多模型并列 | 99.72 | GLM-4.7 | 81.25 |
| Event Verification | Grok 4.1 | 81.85 | GLM-4.7 | 12.95 |
| Tool Invocation | DeepSeek-V3.2 | 94.06 | Qwen3-235B | 45.54 |
| Root Cause Diagnosis | Qwen3-235B | 71.49 | GLM-4.7 | 26.63 |
| **Solution Generation** | **Doubao-pro** | **30.72** | **GLM-4.7** | **9.64** |

> ⚠️ 注意：即使最佳模型在 **Solution Generation** 上也仅达 **30.72%**，远低于实用门槛（通常需 >95%）。

---

### 与基线方法的对比结果

#### （1）语言接口任务已趋近饱和
- **Intent Recognition** 和 **Entity Extraction** 平均准确率 >92%，表明 LLMs 已能可靠解析自然语言指令与提取结构化实体。
- ➜ “语言理解墙”基本突破。

#### （2）MoE 架构显著提升 Tool Invocation 能力
- DeepSeek-V3.2（MoE）在 Tool Invocation 达到 **94.06%**，而同源 dense 模型 Qwen3-235B 仅为 **45.54%**。
- Pearson相关性分析显示：**MoE 架构与 tool-calling 能力强正相关（r=0.87, p<0.01）**。
- ➜ 表明稀疏激活 + 针对性后训练 更适合动态工具编排。

#### （3）诊断与行动之间存在巨大鸿沟（Diagnosis-Action Paradox）
- Qwen3-235B 在 Root Cause Diagnosis 达 **71.49%**，但在 Solution Generation 暴跌至 **4.67%**（差距达 **66.82个百分点**）。
- DeepSeek-V3.2 同样呈现类似趋势（63.00% → 5.61%）。
- ➜ 揭示出“**Execution Wall**”现象：模型擅长因果推理，但难以转化为安全、合规、可执行的操作序列。

---

### 消融实验结果（隐含分析）

虽然未设显式消融实验，但以下对比揭示关键因素：

| 分析维度 | 发现 |
|--------|------|
| **参数规模 vs 数据分布** | Qwen3-32B 在 Wireless Network 上优于更大的 Qwen3-235B（73.04% vs 70.80%），说明 **vendor-specific 文档覆盖率比参数量更重要**。 |
| **产品知识掌握度** | 所有模型在 Product Knowledge 上平均得分 <70%，证实 **专有运营知识仍是公共LLM盲区**。 |
| **工具使用能力** | 多数通用模型输出自然语言建议而非调用工具函数，甚至生成不存在的命令（如 `defragmentIQNR`），暴露 **缺乏 procedural agency**。 |

---

## 4. 关键结论和发现

### 论文的主要发现

✅ **核心结论一：当前LLMs是优秀的“诊断员”，但不是合格的“现场工程师”**
- 在 Intent/Entity 等前端任务上表现接近人类水平；
- 在 Root Cause Diagnosis 上具备一定因果推理能力；
- 但在最终的 **Solution Generation** 上全面崩溃，暴露“**Execution Wall**”。

✅ **核心结论二：存在“诊断-行动悖论”（Diagnosis-Action Paradox）**
- 模型可以正确识别故障根源，却无法生成符合安全规范、语法正确、参数完整的修复脚本。
- 这反映了 LLMs 缺乏 **procedural synthesis**（程序综合）能力和 **tool-grounded interaction**（工具锚定交互）机制。

✅ **核心结论三：MoE 架构更适配电信自动化场景**
- 在 Tool Invocation 任务中，MoE 模型（如 DeepSeek-V3.2）明显优于 Dense 模型，提示未来应加强稀疏架构在工业AI中的应用。

✅ **核心结论四：厂商私有知识是关键瓶颈**
- 即使最大模型也无法充分掌握设备手册中的细节（如 attribute variation、scenario mapping），说明 **domain-specific alignment 不可替代**。

---

### 方法的局限性

⚠️ **1. 数据获取门槛高**
- TeleCom-Bench 依赖大量内部私有数据（设备文档、现网日志），限制其在其他厂商间的复现。

⚠️ **2. 评估成本高昂**
- 每条 Solution Generation 样本需经现场验证与专家审核，难以大规模扩展。

⚠️ **3. 当前LLM缺乏“执行力”基因**
- 现有训练范式（pretrain + SFT + RLHF）侧重语言流畅性，而非命令安全性与可执行性。

---

### 未来工作方向

🚀 **1. 构建 Telecom-LLM 专用预训练语料**
- 整合更多厂商设备文档、历史工单、运维SOP，打造垂直领域 corpus。

🚀 **2. 开展 Safety-Constrained Procedural Training**
- 设计专门的微调目标，强制模型生成符合 CLI 语法、权限控制、风险检查的安全脚本。

🚀 **3. 引入 Simulation-in-the-Loop 评估**
- 构建电信数字孪生环境，在虚拟网络中自动验证解决方案的有效性与副作用。

🚀 **4. 推动 TeleCom-Bench 成为行业标准**
- 开源数据集与评估代码（https://github.com/ZTE-AICloud/TeleCom-Bench），促进跨机构协作。

---

> 🔚 **总结一句话**：  
> *TeleCom-Bench* 揭示了 LLMs 在电信工业落地的最大障碍不是“不知道”，而是“做不到”。未来的突破点不在更大模型，而在更专业的 **领域对齐（domain alignment）**、更严格的 **过程约束（procedural constraint）** 和更真实的 **工具协同（tool-augmented execution）**。

</details>

---

### 12. [AutoVecCoder: Teaching LLMs to Generate Explicitly Vectorized Code](https://arxiv.org/abs/2605.17978)

**Authors**: Shangzhan Li, Xinyu Yin, Xuanyu Jin, Ye He, Yuxin Zhou, Yuxuan Li, Xu Han, Wanxiang Che, Qi Shi, Ting Liu, Maosong Sun  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.17978v1  

#### Abstract
Vectorization via Single Instruction, Multiple Data (SIMD) architectures is a cornerstone of high-performance computing. To fully exploit hardware potential, developers often resort to explicit vectorization using intrinsics, as compiler-based auto-vectorization frequently yields suboptimal results ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**AUTOVECCODER: Teaching LLMs to Generate Explicitly Vectorized Code**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代高性能计算依赖于 **SIMD（Single Instruction, Multiple Data）** 架构进行向量化加速，但高效生成显式向量化代码（explicit vectorization）极具挑战：
- **编译器自动向量化（auto-vectorization）** 常因保守的静态分析失败，尤其在存在循环依赖、条件分支等复杂控制流时。
- **手动使用 SIMD intrinsics 编程** 虽能保证性能，但开发成本高、可移植性差。
- 当前 **Large Language Models (LLMs)** 在通用代码生成上表现优异，但在显式向量化任务中表现不佳，主要受限于：
  - 高质量显式向量化训练数据稀缺；
  - 对低级硬件指令语义理解不足；
  - 缺乏对执行效率的优化目标。

---

### 🚀 提出的新方法与创新思路

作者提出 **AUTOVECCODER**，一个专为显式向量化任务设计的 LLM 训练框架，包含两大核心组件：

#### （1）**VECPROMPT：基于知识增强的数据合成管道**
- **自动化构建高质量标量→向量化平行语料库**。
- 利用 **Retrieval-Augmented Generation (RAG)** 技术，从官方 SIMD intrinsic 文档中检索相关知识，注入到生成过程中，提升模型对硬件指令的理解。
- 数据来源包括：
  - 合成模板（覆盖算子、数据类型、维度等组合）；
  - 真实世界代码片段（如 MBPP、XLCoST）；
  - 经过编译性、功能等价性和复杂度三重过滤。

#### （2）**VECRL：性能驱动的强化学习算法**
- 引入 **Correctness-Gated Performance Reward** 机制：
  - 只有功能正确的代码才能获得性能奖励；
  - 性能增益通过 `SpeedUp` 指标衡量，并使用 `tanh` 进行归一化以稳定训练。
- 采用 **Group Relative Policy Optimization (GRPO)** 进行策略优化，使模型不仅能模仿正确代码，还能探索超越传统编译器启发式的高性能实现。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 / 现有 LLM 方法 | AUTOVECCODER |
|------|--------------------------|---------------|
| **数据质量** | 依赖零样本提示或小规模人工标注 | 自动化合成大规模、高保真、领域特定数据 |
| **知识注入** | 无系统性硬件知识注入 | RAG 注入最新 SIMD intrinsic 文档 |
| **优化目标** | 仅关注语法/功能正确性 | 显式优化执行效率（performance-aware） |
| **训练范式** | 监督微调为主 | 结合 SFT + 强化学习（RL）双阶段训练 |
| **实际效果** | 多数无法超越 `-O3` 编译优化 | 在部分场景下**优于 `-O3` 编译器优化结果** |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **SimdBench**（He et al., 2025）作为主评测基准：
  - 支持多架构（x86, ARM, RISC-V）；
  - 包含标量函数及其对应的高性能向量化实现；
  - 本研究聚焦其 **SSE 和 AVX 子集**。
- **自建训练数据集**：
  - 来源于 VECPROMPT 流水线合成；
  - 共 **7,685 个高质量样本**，涵盖 6 类操作（算术、数学、逻辑、规约、类型转换、比较）、11 种数据类型、多种控制流结构；
  - 教师模型为 DeepSeek-R1-250528，且该模型发布早于 SimdBench，避免数据泄露。

---

### ⚙️ 实验设置

- **基础模型**：Qwen3-8B
- **训练流程**：
  1. **SFT 阶段**：基于 VECPROMPT 生成的数据进行监督微调；
  2. **VECRL 阶段**：使用 verl 框架进行 GRPO 强化学习，共 5 轮，学习率 $1 \times 10^{-6}$，batch size 64。
- **执行环境隔离**：
  - 使用 **ZeroMQ (ZMQ)** 构建轻量级沙箱；
  - 每个任务绑定独立 CPU 核心，确保性能测量稳定。
- **编译选项**：所有代码均使用 `-O3` 编译，公平比较。

---

### 📊 评估指标

| 指标 | 定义 | 说明 |
|------|------|------|
| **SpeedUp** | $\frac{T_{\text{scalar}}}{T_{\text{vector}}}$ | 向量化版本相对于标量版本的速度提升倍数 |
| **Corr (Correctness)** | 功能正确样本占比 | 必须通过所有测试用例 |
| **fast₁** | $ \frac{1}{N} \sum \mathbb{1}(\text{correct}_i \land \text{SpeedUp}_i > 1) $ | 正确且快于标量的样本比例 |
| **P50 / P75** | 在正确样本上的 SpeedUp 中位数 / 第75百分位数 | 衡量性能分布 |

---

### 🆚 基线方法对比

重新评估了多个前沿闭源与开源 LLM，在 **零样本（zero-shot）设置下** 对比：

- **Qwen 系列**：Qwen3-Coder-480B-A35B, Qwen3-Coder-Plus, Qwen3-8B
- **DeepSeek 系列**：DeepSeek-V3, DeepSeek-V3.2-Thinking, DeepSeek-R1
- **闭源模型**：GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet, Grok4-Fast

> 所有模型均未在推理时接入 RAG，仅 AUTOVECCODER 在训练阶段利用 RAG 构造数据。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| Model | AVX Corr | AVX fast₁ | AVX P50 | AVX P75 | SSE Corr | SSE fast₁ | SSE P50 | SSE P75 |
|-------|----------|-----------|---------|---------|----------|-----------|---------|---------|
| **AUTOVECCODER-8B (ours)** | **76.76** | **47.35** | **0.99** | **2.74** | **77.35** | **53.53** | **1.02** | **2.22** |
| DeepSeek-R1 | 73.53 | 44.12 | 0.97 | 2.83 | 69.85 | 46.32 | 0.99 | 1.95 |
| Gemini-2.5-Pro | 63.97 | 39.71 | 0.88 | **2.84** | 61.76 | 47.06 | 0.93 | 2.35 |
| GPT-5 | 62.50 | 36.76 | 0.82 | 1.18 | 55.88 | 33.82 | 0.60 | 1.15 |
| Qwen3-8B (w/o training) | 9.41 | 2.79 | — | — | 10.88 | 5.00 | — | — |

> ✅ **AUTOVECCODER-8B 在几乎所有指标上达到 SOTA，尤其在 fast₁ 上显著领先**。

---

### 🔬 消融实验结果

#### （1）**VECPROMPT 的作用（Table 2）**
- 若跳过 VECPROMPT 直接进行 RL：
  - 初始正确率极低（~10%），导致 reward 稀疏；
  - 收敛慢且不稳定。
- 经过 VECPROMPT SFT 初始化后：
  - 正确率提升至 ~63%，为 RL 提供坚实起点；
  - 显著提高训练效率与最终性能。

> ✔️ VECPROMPT 提供了“语义锚点”，缩小搜索空间，缓解 RL 冷启动问题。

#### （2）**奖励函数设计对比（Table 3）**

| Reward Type | AVX Corr | AVX fast₁ | SSE Corr | SSE fast₁ |
|-------------|----------|-----------|----------|-----------|
| Naive SpeedUp Reward (NSR) | 63.97 | 36.03 | 70.59 | 46.32 |
| **VECRL (ours)** | **69.85** | **41.18** | **72.06** | **47.79** |

- NSR 导致早期过度追求高风险模式，虽短暂提升性能，但后期出现**策略退化**（policy collapse），正确率下降。
- VECRL 的分层奖励机制更稳定，实现了**正确性与性能的协同增长**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **性能感知训练至关重要**：
   - 单纯模仿人类编写的向量化代码不足以产生最优实现；
   - 引入执行反馈（execution-in-the-loop）是突破编译器瓶颈的关键。

2. **小模型也能超越大模型**：
   - 尽管参数仅为 **8B**，AUTOVECCODER-8B 超越了包括 GPT-5、Gemini 等在内的大型闭源模型；
   - 表明**领域专用训练 > 参数规模**，特别是在低级系统编程任务中。

3. **可超越 `-O3` 编译优化**：
   - 在 SimdBench 的多个案例中，AUTOVECCODER 生成的代码性能**超过 GCC/Clang `-O3` 自动生成的结果**；
   - 成功处理了以下传统编译器难以向量化的场景：
     - **Mask-based 控制流**（将条件分支转为掩码操作）
     - **非确定性迭代结构**（无需静态证明即可并行化）
     - **指针别名/内存依赖误判**（语义理解规避保守分析）
     - **不规则内存访问重构**（如转置、strided 访问优化）

4. **VECRL 实现了从“模仿”到“创新”的跃迁**：
   - 图 3 显示训练分为两个阶段：
     - Phase I：快速提升正确性（探索合法解空间）；
     - Phase II：稳步提升性能（在正确前提下优化效率）。

---

### ⚠️ 局限性

1. **架构泛化能力有限**：
   - 当前主要验证于 **x86 平台的 SSE/AVX**；
   - 对 ARM NEON 或 RISC-V Vector 的支持尚未充分测试；
   - 不同架构的 intrinsic 差异可能影响 RAG 与 RL 的稳定性。

2. **向量化深度未被精细建模**：
   - 当前 reward 仅看端到端 speedup，不分“纯计算并行” vs “仅内存优化”；
   - 存在“浅层向量化”现象（如仅用于批量加载）；
   - 未来需引入更细粒度的 reward 设计。

3. **适用范围局限于循环密集型代码**：
   - 主要针对标量循环体的向量化；
   - 对复杂非结构化代码或高级 DSL（如 CUDA kernel）的支持仍待扩展。

---

### 🔮 未来工作方向

1. **跨架构统一向量化框架**：
   - 构建多平台 intrinsic 知识库，实现一次训练、多端部署。

2. **精细化向量化深度评估与奖励机制**：
   - 引入“向量化覆盖率”、“FLOPs 利用率”等指标指导 RL。

3. **扩展至其他高性能 DSL**：
   - 如 Triton、CUDA、SYCL 等 GPU 编程语言；
   - 探索 AUTOVECCODER 思路在 kernel fusion、tiling 等优化中的应用。

4. **结合形式验证保障安全性**：
   - 对生成的 intrinsics 代码增加形式化检查，防止边界错误或未定义行为。

---

> 💡 **总结一句话**：  
> **AUTOVECCODER 通过“知识注入 + 性能驱动 RL”双轮驱动，首次实现了 LLM 在显式 SIMD 向量化任务上的全面超越，展示了小模型在专业领域战胜大模型的可能性，为 LLM 赋能系统级编程开辟了新路径。**

</details>

---

### 13. [World Model-Enabled Causal Digital Twins for Semantic Communications in Physical AI Systems](https://arxiv.org/abs/2605.16547)

**Authors**: Lingyi Wang, Tingyu Shui, Walid Saad, Pascal Adjakple  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.16547v1  

#### Abstract
Semantic communication has emerged as a promising paradigm for enabling goal-oriented networking. However, most existing semantic communication solutions are tailored to one-shot tasks and optimize instantaneous performance. Hence, they cannot be used to support closed-loop dynamic systems with phys...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：World Model-Enabled Causal Digital Twins for Semantic Communications in Physical AI Systems

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**闭环物理人工智能（Physical AI）系统中的语义通信**（Semantic Communications）问题，解决了现有方法在以下两方面的不足：

- **信息价值评估不准确**：传统方法（如 VoI、AoI 或特征重要性评分）仅基于统计相关性或瞬时任务损失来评估语义信息的价值，无法捕捉其对**长期控制决策、状态演化和任务回报的因果影响**。
- **缺乏长视野优化能力**：现有强化学习（RL）方法依赖试错交互，数据效率低且难以进行长视野规划，尤其在资源受限的无线环境中表现不佳。

### 提出的新方法与创新思路
作者提出了一种名为 **World Model-Enabled Causal Digital Twin (WM-CDT)** 的新型框架，其核心创新包括：

#### （1）**因果信息价值（Causal Information Value, CIV）度量**
- 引入基于 **Pearl 的 do-operator** 的反事实推理机制，定义每个语义 token 对长期回报的**边际因果贡献**。
- CIV 通过比较“传输该 token”与“不传输该 token”的两种干预情景下的预期长期回报差异来计算，从而剥离历史依赖性，实现更准确的信息价值评估。

#### （2）**世界模型驱动的因果数字孪生（WM-CDT）**
- 构建一个基于 Recurrent State-Space Model (RSSM) 的数字孪生体，部署于边缘服务器，用于：
  - 学习闭环系统的动态演化（感知-通信-推理-控制）
  - 支持**反事实想象推演**（counterfactual imagined rollouts），以估计 CIV 和长期回报
  - 实现无需真实环境交互的高效策略训练

#### （3）**返回每比特最大化的目标函数**
- 将语义通信问题建模为 **return-per-bit maximization**，联合优化通信效率（比特成本）与控制效率（任务回报），而非单纯最小化重建误差或瞬时损失。

#### （4）**CIV引导的语义选择器设计**
- 设计基于门控网络与**反向剪枝**（reverse pruning）的选择策略，结合 CIV-per-bit 指标，在满足比特预算的前提下保留最具长期价值的语义 token。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **信息价值评估** | 超越相关性 → 实现因果性评估，更贴近真实长期收益 |
| **学习效率** | 利用世界模型进行想象推演，显著提升数据效率，减少真实试错 |
| **长视野规划** | 支持多步想象与反事实分析，优于仅依赖当前观测的短视策略 |
| **通信效率** | 显式优化 return-per-bit，避免冗余传输，适应带宽约束 |

---

## 2. 核心实验方法和设置

### 数据集与仿真平台
- **自研仿真器：AirSim-Sionna-based Simulator**
  - **AirSim + Unreal Engine**：构建高保真的3D无人机（UAV）导航环境，模拟视觉传感器（RGB相机）、IMU、LiDAR等。
  - **Sionna**：集成5G/6G 物理层与MAC层建模，支持射线追踪、路径损耗、衰落、噪声等真实无线信道效应。
  - 两大模块时间同步，形成完整的 **sensing-communication-inference-control 闭环**。

### 实验设置
- **任务**：UAV 在含随机障碍物的 200m × 200m 三维空间中导航至目标点。
- **成功条件**：进入目标半径 5m 内；失败条件：碰撞或超时（T=200 步）。
- **语义编码**：
  - 提取 N=32 个候选语义 token（如物体假设、空间关系、风险提示等）
  - 每个 token 编码为 64 维嵌入，经向量量化压缩为 **8 bits/token**
- **比特预算**：$ U_t \in \{64, 96, 128, 160, 192\} $ bits/slot
- **控制命令**：连续动作 $[v_x, v_z, w]$（前向速度、垂直速度、角速度）

### 评估指标
| 指标 | 定义 |
|------|------|
| **Return-per-kbit** | 单位通信开销（每千比特）带来的累计折扣回报，衡量通信效率 |
| **Navigation Success Rate** | 成功完成导航任务的比例 |
| **Data Efficiency** | 达到特定性能所需的环境交互步数 |
| **Robustness** | 在不同 Packet Error Rate 下的性能稳定性 |

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **AC-RRL** [20] | Model-free RL | 使用相同语义编码器，无世界模型，直接从真实交互中学习 |
| **MBPO** | Model-based RL | 学习潜动力学模型并进行短视想象推演 |
| **AC** | Feedforward AC | 不维护历史状态，仅基于当前接收 token 决策 |
| **PPO** | Model-free RL | 使用 PPO 算法更新策略，受比特预算约束 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（最大比特预算下）

| 方法 | Return-per-kbit ↑ | 导航成功率 ↑ |
|------|-------------------|-------------|
| **WM-CDT (Proposed)** | **1.60** | **80.0%** |
| AC-RRL | 1.36 (+17.3%) | 72.8% (+9.6%) |
| MBPO | 1.03 (+55.4%) | 63.2% (+26.3%) |
| AC | 1.22 (+30.7%) | 68.8% (+16.1%) |
| PPO | 1.20 (+33.7%) | 67.2% (+18.0%) |

> ✅ 所有改进均具有统计显著性。

### 数据效率对比（达到 1 return-per-kbit 所需步数）
| 方法 | 所需环境步数 | 相对提升 |
|------|---------------|----------|
| **WM-CDT** | **79k** | — |
| MBPO | 126k | +37.3% |
| AC-RRL | 144k | +45.1% |
| AC | 264k | +70.1% |

> 📈 WM-CDT 在极少数交互下即可收敛，展现出卓越的数据效率。

### 鲁棒性测试（Packet Error Rate = 20%）
| 方法 | Return-per-kbit |
|------|-----------------|
| **WM-CDT** | **1.12** |
| AC-RRL | 0.94 (+19.6%) |
| MBPO | 0.63 (+77.4%) |
| AC | 0.77 (+46.2%) |
| PPO | 0.84 (+32.5%) |

> 🔒 WM-CDT 凭借 RSSM 的信念预测能力和 CIV 的关键 token 优先级调度，在丢包环境下仍保持最优性能。

### 消融实验结果（Ablation Studies）

#### （1）动态建模消融（Fig. 9）
- 移除“随机潜变量”或“递归动态”导致 return-per-kbit 下降 13.9%~19.1%
- “一步预测”模型性能最差，说明**长视野想象推演至关重要**

#### （2）语义编码器设计消融（Fig. 10）
- 使用 reconstruction-based 编码器比 TD-based 下降约 9.2%
- 移除 TD 预测或控制条件化目标均显著降低性能
- 结论：**面向任务的、时间可预测的表示优于重建导向的表示**

#### （3）语义选择策略消融（Fig. 11）
- 移除 CIV 估计 → 性能下降 14.9%
- 移除 per-bit 归一化 → 忽视通信效率，偏好大 token
- 移除反向剪枝 → 无法有效去冗
- 结论：**CIV + per-bit norm + reverse pruning 是高效选择的关键组合**

---

## 4. 关键结论和发现

### 主要发现
1. **CIV 是更优的信息价值度量**  
   - 图 8 显示，CIV 与真实反事实回报增益的相关性远高于 myopic VoI、saliency score 和 confidence score。
   - 表明**因果推理机制能更准确识别对长期任务有实质性影响的语义 token**。

2. **WM-CDT 实现高效闭环控制**  
   - 通过世界模型支持反事实想象，实现了高数据效率的学习与安全的策略探索。
   - return-per-bit 最大化目标促使系统主动权衡通信开销与任务收益，避免“过度通信”或“信息缺失”。

3. **通信效率与任务性能可兼得**  
   - 在较低比特预算（如 96 bits/slot）下即能达到 return-per-kbit 峰值，表明**少量高质量语义即可支撑高性能控制**。
   - 过多 token 反而导致边际效益递减。

### 方法的局限性
- **训练复杂度较高**：WM-CDT 模型参数量大（8.6M），训练耗时较长（见 Table II），但此开销发生在离线阶段。
- **依赖预定义语义提取器**：token 提取过程未端到端学习，可能限制泛化能力。
- **数字孪生依赖建模精度**：若世界模型未能准确捕获真实系统动态，可能导致策略偏差。

### 未来工作方向
1. **端到端语义 token 学习**：将 token 提取与选择联合优化，摆脱手工设计先验。
2. **多智能体扩展**：应用于车联网、无人机群等多 agent 协同场景。
3. **在线自适应建模**：增强数字孪生对动态环境变化的实时适应能力。
4. **硬件部署验证**：在真实无人机平台上验证 WM-CDT 的可行性与延迟表现。

---

> 💡 **总结一句话**：  
> 本文提出的 **WM-CDT 框架**通过引入**因果信息价值（CIV）** 与**世界模型驱动的数字孪生**，首次实现了面向物理 AI 系统的**长视野、高效率、可解释的语义通信**，显著提升了 return-per-bit 与任务成功率，为下一代智能无线系统提供了新的范式。

</details>

---

### 14. [Privacy-Preserving Generation Fraud Detection for Distributed Photovoltaic Systems: A Solar Irradiance-Fused Federated Learning Framework](https://arxiv.org/abs/2605.17039)

**Authors**: Xiaolu Chen, Chenghao Huang, Yanru Zhang, Hao Wang  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.17039v1  

#### Abstract
The wide adoption of residential photovoltaic (PV) systems introduces new challenges for generation fraud detection (FD). Unlike traditional electricity theft detection, which focuses on electricity consumption-side behavior, PV generation fraud detection (PVG-FD) is complicated by the inherent inte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Privacy-Preserving Generation Fraud Detection for Distributed Photovoltaic Systems: A Solar Irradiance-Fused Federated Learning Framework

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**分布式光伏系统（Distributed Photovoltaic, PV）中的发电欺诈检测（Generation Fraud Detection, PVG-FD）**提出了一种新的解决方案。传统电力盗窃检测主要关注用电侧异常，而PVG-FD面临以下挑战：
- **光伏出力的间歇性和不确定性**导致正常与欺诈行为难以区分；
- **数据隐私问题**：集中式模型需要聚合各家庭的敏感数据，存在隐私泄露风险；
- **类别不平衡**：欺诈样本极少，模型容易偏向多数类（正常发电），影响检测性能。

### 提出的新方法与创新思路
作者提出了一种**基于联邦学习（Federated Learning, FL）的隐私保护型分布式PVG-FD框架**，其核心创新包括：

#### （1）多源数据融合架构（Solar Irradiance-Fused Model）
- 将**PV generation time series**与**天气数据（GHI/DNI/DHI）** 融合输入模型；
- 设计了一个**co-attention机制**，通过self-attention和cross-attention联合建模两种模态之间的动态依赖关系，有效捕捉“发电量 vs. 实际辐照”之间的物理不一致性，从而识别过报行为。

#### （2）原型对齐正则化（Prototype Alignment Regularization）
- 引入**class prototype**概念，在嵌入空间中为每个类别（如正常/欺诈）构建代表性向量；
- 在FL过程中，客户端上传本地prototype，服务器进行加权聚合生成global prototype；
- 客户端在训练时引入**余弦相似度损失项** $ \mathcal{L}_{reg} $，使本地prototype向全局prototype对齐，提升小样本欺诈类别的表征能力。

#### （3）隐私保护的分布式协作框架
- 采用**model splitting**策略：仅上传浅层base model参数，深层head model保留在本地；
- 所有通信均不涉及原始数据或标签，实现真正的**privacy-preserving**；
- 支持跨社区知识共享，同时缓解局部数据稀疏问题。

### 相比现有方法的优势
| 维度 | 本文方法优势 |
|------|---------------|
| **隐私性** | 显著优于集中式方法，无需共享原始数据；相比标准FL更轻量（仅传base model + prototypes） |
| **检测精度** | 在多种场景下F1-score、MCC等指标显著领先于FedAvg、BalanceFL、FedNH等基线 |
| **鲁棒性** | 对季节变化、噪声辐照数据、极端类别不平衡（低至0.5%欺诈率）具有更强适应性 |
| **可扩展性** | 可灵活适配不同规模社区，且在新增未见社区上表现出良好迁移能力 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用真实世界住宅光伏数据集：**Solar Home Electricity Data [38]**，由澳大利亚Ausgrid提供；
- 包含300个安装屋顶光伏的家庭，时间跨度为2010年7月1日至2013年6月30日，采样间隔为30分钟；
- 同步获取**National Solar Radiation Database (NSRDB)** 中的气象数据（GHI, DNI, DHI）；
- 总体覆盖约2.5年有效数据（前6个月无气象记录被剔除）。

### 实验设置
- **联邦划分方式**：将300个prosumer划分为 $ N=5 $ 个独立社区（每社区约60户），按时间顺序划分训练/测试集（第一年训练，第二年测试）；
- **欺诈样本合成**：基于已有研究 [5][13][40]，人工注入三种典型欺诈模式：
  - **Type 1**: 全天均匀膨胀（Uniform Daily Inflation）
  - **Type 2**: 高峰时段局部放大（Peak-Window Inflation）
  - **Type 3**: 电价驱动的时间转移攻击（Tariff-Driven Time-Shift）
- 默认欺诈比例设为15%，并在部分实验中降至0.5%以模拟极端不平衡场景。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (Acc.)** | 整体分类准确率 |
| **AUC** | ROC曲线下面积，衡量排序能力 |
| **F1-score** | 精确率与召回率的调和平均，尤其适用于不平衡数据 |
| **Matthews Correlation Coefficient (MCC)** | 综合考虑TP/TN/FP/FN的平衡性指标，适合严重不平衡任务 |
| **Precision@5/1000 & FPR@5/1000** | 预算感知指标，用于评估在极少量警报下的检测质量 |

### 基线方法对比
#### 局部模型对比（Local Models under FedAvg）
- LSTM
- CNN-LSTM
- Transformer
- Reformer
- DLinear
- **Proposed (ours)**

#### 联邦学习框架对比（Using Proposed Local Model）
- **Local-only**：各社区独立训练，无协作
- **FedAvg** [15]：经典FL聚合算法
- **BalanceFL** [32]：面向长尾分布的FL方法
- **FedNH** [33]：基于类原型的不平衡FL框架
- **Proposed (ours)**

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 表1：局部模型在FedAvg下的平均性能（Table III）
| Method | Acc. (%) | AUC (%) | F1 (%) | MCC (%) |
|--------|----------|---------|--------|---------|
| LSTM | 92.91 ± 0.50 | 92.52 ± 0.45 | 73.92 ± 2.35 | 70.83 ± 2.56 |
| CNN-LSTM | 93.36 ± 0.34 | 93.82 ± 0.43 | 74.91 ± 1.94 | 72.60 ± 1.91 |
| Transformer | 94.82 ± 0.40 | 95.52 ± 0.24 | 81.57 ± 1.83 | 79.13 ± 1.90 |
| **Proposed** | **95.57 ± 0.29** | **96.58 ± 0.23** | **84.17 ± 1.48** | **82.26 ± 1.46** |

> ✅ 提出的本地模型在所有指标上均取得最优表现。

#### 表2：不同FL框架性能比较（Table VI）
| FL Framework | Acc. (%) | AUC (%) | F1 (%) | MCC (%) |
|--------------|----------|---------|--------|---------|
| Local-only | 94.70 ± 0.48 | 95.63 ± 0.49 | 81.31 ± 2.01 | 78.69 ± 2.57 |
| FedAvg | 95.57 ± 0.33 | 96.58 ± 0.37 | 84.17 ± 1.58 | 82.26 ± 1.83 |
| BalanceFL | 96.13 ± 0.31 | 96.70 ± 0.20 | 86.32 ± 1.45 | 84.62 ± 1.60 |
| FedNH | 96.43 ± 0.30 | 97.09 ± 0.24 | 87.52 ± 1.47 | 85.88 ± 1.55 |
| **Proposed** | **96.90 ± 0.29** | **97.49 ± 0.23** | **89.27 ± 1.32** | **87.78 ± 1.46** |

> ✅ 所提FL框架全面超越所有基线，尤其在F1和MCC上有显著增益。

#### 极端不平衡场景（0.5%欺诈率）——预算感知指标（Table X）
| FL Framework | Precision@5/1000 (%) | FPR@5/1000 (%) |
|--------------|-----------------------|----------------|
| Local-only | 35.63 | 0.32 |
| FedAvg | 50.57 | 0.25 |
| BalanceFL | 55.17 | 0.23 |
| FedNH | 62.06 | 0.19 |
| **Proposed** | **65.51** | **0.17** |

> ✅ 即使在极低欺诈率下仍保持最高精度和最低误报率。

### 消融实验与关键分析
- **跨季节测试（Cross-season Evaluation）**：模型在春夏秋冬四个季节间泛化能力强，提出的模型性能下降最小（Table IV）；
- **噪声辐照条件测试**：当使用社区级平均辐照代替精细测量时，所有模型性能下降，但所提方法相对最稳健（Table V）；
- **不同社区数量的影响（Scalability）**：随着社区数增加（即每社区户数减少），Local-only性能急剧下降，而所提FL方法维持较高水平（Fig. 7）；
- **跨社区迁移能力（Cross-Community Setting）**：在held-out社区上微调后，所提方法性能下降最少，说明全局知识更具迁移价值（Table VIII）；
- **通信开销对比（Table VII）**：所提方法每轮通信参数量最低（7.60×10⁶），优于FedAvg（9.45×10⁶）和FedNH（8.52×10⁶），更适合边缘部署。

---

## 4. 关键结论和发现

### 主要发现
1. **多源数据融合 + co-attention机制能显著增强PVG-FD的判别能力**，特别是在复杂环境波动下仍可捕捉物理不一致信号；
2. **基于prototype的正则化机制有效缓解了FL中的类别不平衡问题**，通过global-local prototype alignment提升了少数类（欺诈）的特征表达；
3. **所提FL框架实现了高性能与强隐私性的统一**，既避免了原始数据暴露，又通过模型级协作获得优于集中式训练的效果；
4. 方法具备良好的**scalability**和**robustness**，适用于不同规模、不同数据分布的实际电网应用场景。

### 方法的局限性
- 当前框架采用**同步FL协议**，要求所有客户端每轮参与更新，缺乏对异构设备延迟的支持；
- 欠缺对**无监督或半监督学习**的支持，目前仍依赖人工标注的欺诈样本进行训练；
- 实验基于**合成欺诈数据**，尚未在真实攻击事件中验证；
- 未考虑**模型异构性（Model Heterogeneity）**，假设所有社区使用相同模型结构。

### 未来工作方向
- 探索**semi-supervised / unsupervised learning**策略，降低对标注数据的依赖；
- 引入**adaptive communication-efficient FL**机制（如梯度压缩、异步更新）以提升实用性；
- 结合**game-theoretic modeling**模拟攻防博弈过程，设计更具鲁棒性的detector；
- 将框架拓展至其他**renewable energy systems**（如风电、储能）的欺诈检测任务中。

--- 

> 📌 **总结一句话**：  
> 本论文提出了一种融合太阳能辐照信息的联邦学习框架（Solar Irradiance-Fused FL），通过co-attention多源融合与prototype alignment机制，在保障隐私的前提下显著提升了分布式光伏系统中发电欺诈检测的准确性与鲁棒性，尤其在极端类别不平衡场景下表现卓越。

</details>

---

### 15. [NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning](https://arxiv.org/abs/2605.16757)

**Authors**: Haoran Lu, Luyang Fang, Wenxuan Zhong, Ping Ma  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.16757v1  

#### Abstract
Multi-agent language systems are often built as hand-designed workflows, where agents are assigned semantic roles and communication protocols are specified in advance. We propose NeuroMAS, a method that first treats a multi-agent language system as a trainable and scalable neural-network-like archit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前基于 Large Language Model (LLM) 的 **Multi-Agent Systems (MAS)** 多依赖于**人工设计的工作流**（hand-designed workflows），例如为每个 agent 分配“规划者”、“验证者”、“批评者”等语义角色，并通过预定义协议进行通信。这种设计方式存在以下问题：

- **缺乏可扩展性**：组织结构是静态的，难以随任务复杂度增长而动态扩展。
- **设计成本高**：需要大量人力进行角色定义和流程编排。
- **优化割裂**：通常只优化提示（prompt）或拓扑结构，而不联合训练 agent 本身。

该论文提出，应将多智能体系统视为一个**可学习、可扩展的神经网络式架构**，而非固定流程。

---

### 🧠 提出的新方法：NeuroMAS

**NeuroMAS**（Neural Multi-Agent System）是一种将多智能体语言系统建模为**可训练文本神经网络架构**的新框架，其核心思想包括：

#### （1）**Role-Free but Structure-Aware Nodes**
- 所有 agent 节点**不预设语义角色**（如 planner/solver/critic）。
- 每个节点仅知道其在拓扑中的位置（layer 和 index）以及消息格式要求。
- 功能专业化（specialization）由强化学习在训练中自动涌现。

#### （2）**Text-Valued Neural Architecture**
- 将 LLM agents 视为“神经元”，中间传递的是**离散文本信号**（textual messages），而非连续向量。
- 构成类似 MLP 或 ResNet 的分层结构：输入 → 隐藏层 agents → 输出聚合 agent。
- 支持灵活调整深度（depth）、宽度（width）、连接方式（connectivity）。

#### （3）**Joint Training with Terminal Reward**
- 所有节点共享最终任务奖励（如答案正确性），使用 **REINFORCE 算法**进行端到端联合训练。
- 不引入中间监督或角色特定奖励，协调行为完全通过终端反馈驱动。

#### （4）**Progressive Growth Scaling Protocol**
- 类似于神经网络架构搜索中的渐进生长策略。
- 更大的系统不是从头训练，而是从小型已训练系统**逐步扩展并继承已有参数**，提升训练稳定性与效率。

---

### 🔍 相比现有方法的优势

| 维度 | 传统 MAS 方法 | NeuroMAS |
|------|----------------|----------|
| **角色设定** | 固定人工角色（planner, verifier 等） | 无角色，结构感知，功能自涌现 |
| **训练方式** | 冻结 LLM + 优化 prompt / topology | 联合训练所有 agent 的可调参数（如 LoRA） |
| **组织结构** | 固定拓扑或手动设计图 | 可扩展、模块化、类神经网络架构 |
| **优化目标** | 分离式优化（先搜 topology 再训 agent） | 统一优化：agent 行为与组织协同进化 |

> 💡 **核心理念转变**：  
> 从 “Workflow Engineering” → “Architecture Design”  
> 从 “How to design better roles?” → “How to grow smarter organizations?”

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

实验涵盖六项具有挑战性的推理与代码生成任务，覆盖多个领域：

| 数据集 | 任务类型 | 描述 |
|--------|--------|------|
| **ARC-Challenge** | 科学问答 | 多项选择题，考察常识与科学推理能力 |
| **BBH-Navigate** | 空间导航推理 | 判断是否能从起点到达终点 |
| **MMLU-Abstract Algebra** | 数学推理 | 抽象代数领域的多项选择题 |
| **MMLU-College Physics** | 物理推理 | 大学水平物理知识理解 |
| **MMLU-Professional Medicine** | 医学知识 | 专业医学考试题目 |
| **HumanEval** | 代码生成 | 函数级代码补全，执行测试通过率衡量 |

> ⚠️ 设置特点：使用**弱 backbone 模型**（Qwen3-0.6B 和 Gemma-3-1B-IT），确保 baseline 性能未饱和，以凸显组织优化的价值。

---

### ⚙️ 实验设置与评估指标

| 项目 | 设置说明 |
|------|---------|
| **Backbone 模型** | Qwen3-0.6B（主实验）、Gemma-3-1B-IT（鲁棒性验证） |
| **可训练参数** | 所有 agent 共享冻结 backbone，仅使用 **node-specific LoRA adapters** 进行微调 |
| **训练算法** | REINFORCE with baseline（score-function gradient） |
| **评估方式** | Greedy decoding，输出标准化后计算准确率 |
| **奖励函数** | 正确答案得分为 1，否则为 0（binary reward） |
| **Topology 编码** | NeuroMAS-c 表示每前向传播调用 c 次 LLM：<br>- NeuroMAS-3: [1,1] 结构（两隐藏层各1节点 + 输出）<br>- NeuroMAS-5: [2,2]<br>- NeuroMAS-7: [2,2,2] |

---

### 🆚 基线方法对比

分为两类基线：

#### （1）**Frozen-Backbone Methods**（不更新模型参数）
- Direct Prompting
- Self-Refine
- Self-Check
- MoA (Mixture-of-Agents)
- GoA (Graph of Agents)
- GPTSwarm
- AgentNet

> ➤ 测试：纯提示工程或多轮协作能否超越 NeuroMAS？

#### （2）**Trained-Backbone Methods**（允许参数更新）
- **Single-LLM RL**：单 agent 强化学习（NeuroMAS 的退化情形）
- **MALT**：post-train generator-verifier-refiner pipeline
- **CoLLM-CC**：去中心化协作 + centralized critic
- **NeuroMAS variants**：不同规模结构

> ➤ 关键控制变量：比较相同 LoRA 参数预算下的性能差异

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Method | ARC | Navigate | Algebra | Physics | Medicine | HumanEval | Trainable Params | LLM Calls |
|--------|-----|----------|---------|---------|----------|-----------|------------------|-----------|
| **Direct Prompting** | 24.5 | 40.5 | 22.0 | 17.6 | 24.0 | 11.0 | 0 | 1 |
| **Self-Refine** | 37.5 | 40.5 | 25.0 | 24.5 | 37.5 | 21.3 | 0 | 2 |
| **GPTSwarm** | 46.0 | 40.5 | 25.0 | 25.5 | 43.5 | 12.8 | 0 | 11 |
| **Single-LLM RL** | 46.0 | 42.2 | 29.0 | 24.5 | 43.0 | 17.7 | 6.9M | 1 |
| **NeuroMAS-3** | **56.5** | **45.5** | **39.0** | **44.1** | **48.0** | **30.5** | 6.9M | 3 |
| **NeuroMAS-5** | 54.0 | 48.0 | 39.0 | 39.0 | 41.5 | 29.9 | 11.5M | 5 |
| **NeuroMAS-7** | 53.5 | 51.0 | 42.0 | 39.5 | 43.5 | 31.7 | 16.1M | 7 |

> ✅ **NeuroMAS-3 在所有任务上均优于最强 fixed-backbone 基线**，平均提升显著（最大 +18.6 pts on Physics）。

> ✅ **相比 Single-LLM RL（同参数量 6.9M LoRA）**：
- 平均提升超过 **10 个百分点以上**
- HumanEval 上从 17.7% → 30.5%，翻倍增长
- 表明增益并非来自更多参数，而是**组织结构带来的协作优势**

> ✅ **最佳表现由不同规模 NeuroMAS 实现**：
- ARC/Physics/Medicine: NeuroMAS-3 最优
- Navigate/Algebra/HumanEval: NeuroMAS-7 最优  
→ 支持“可扩展组织”的潜力

---

### 🔬 消融实验结果：Progressive Growth 的作用（Table 3）

| Method | From Scratch | Progressive Growth |
|--------|--------------|--------------------|
| NeuroMAS-3 | 45.5 | 45.5 |
| NeuroMAS-5 | 41.0 | **48.0** (+7.0) |
| NeuroMAS-7 | 40.5 | **51.0** (+10.5) |

> ❗ **关键发现**：
- 更大拓扑 ≠ 更好性能！从头训练的大系统反而**性能下降**（可能因训练不稳定、协调困难）。
- **Progressive Growth 显著逆转趋势**，使大系统可达更高性能。
- 证明：**组织扩展是路径依赖的（path-dependent）**，训练过程本身至关重要。

---

### 🔄 Backbone 鲁棒性实验（Table 2，使用 Gemma-3-1B-IT）

| Method | ARC (%) |
|--------|--------|
| GPTSwarm | 43.0 |
| Single-LLM RL | 36.0 |
| **NeuroMAS-3** | **44.5** |
| NeuroMAS-5 | 44.0 |

> ✅ 在更强 backbone 上仍保持领先，说明方法具有跨模型泛化能力。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Learned Organization 是新的能力来源**  
   即使 backbone 固定且较弱，通过可训练的多 agent 组织结构也能显著提升性能，表明“如何组织计算”本身可以成为 scaling law 的新维度。

2. **Role-Free Design 是可行的**  
   无需人为指定“planner/solver/critic”，只要提供结构位置信息，agent 自然会在训练中发展出分工与协作行为。

3. **Parameter Efficiency 来自 Compositional Structure**  
   理论分析表明：当任务具备层次分解结构时，模块化 multi-agent 架构比单一 unstructured policy 更参数高效（error ∝ q⁻¹/ᵃ vs q⁻¹/ᵇ, a < b）。

4. **Organizational Scaling 是 Path-Dependent**  
   大系统难以从零训练成功，但可通过 progressive growth 稳定扩展，强调“成长路径”比“最终结构”更重要。

5. **NeuroMAS 是一种新型 Neural Architecture**  
   它模糊了 MAS 与 DNN 的边界：agents 是“神经元”，text 是“激活值”，topology 是“网络结构”，RL 是“学习规则”。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **推理开销高** | 每次前向需多次 LLM 调用（e.g., NeuroMAS-7 需 7 次），延迟和成本上升 |
| **依赖任务可分解性** | 若任务无法有效分治（如全局判断型任务），收益有限 |
| **当前规模较小** | 实验仅到 7 个节点，尚未验证超大规模系统的潜力 |
| **理论为容量分析** | 定理描述的是最优情况下的误差界，非实际收敛保证 |

---

### 🔮 未来工作方向

1. **自动化拓扑搜索（Neural Architecture Search for MAS）**  
   探索自动发现最优 depth/width/connectivity 的方法。

2. **异构 agent 与动态拓扑**  
   支持不同类型 agent（e.g., reasoning vs tool-use）混合部署，动态添加/删除节点。

3. **更高效的通信机制**  
   引入 attention、sparse routing、message compression 等机制降低通信负担。

4. **应用于长程任务与具身智能（embodied agents）**  
   如机器人规划、游戏 AI、自主科研助手等复杂场景。

5. **结合 inference-time scaling**  
   将 NeuroMAS 与 ToT、MCTS 等推理时搜索方法融合，实现训练期组织 + 推理期探索的双重增强。

---

> 🎯 **一句话总结**：  
> **NeuroMAS 提出了一种将 Multi-Agent System 视为“文本神经网络”的新范式，通过 role-free 结构设计与 joint RL 训练，实现了组织即架构（Organization as Architecture）的可扩展智能，揭示了“学会如何协作”可能是继“增大模型”之后的下一个关键 scaling axis。**

</details>

---

### 16. [Skim: Speculative Execution for Fast and Efficient Web Agents](https://arxiv.org/abs/2605.16565)

**Authors**: Mike Wong, Kevin Hsieh, Suman Nath, Ravi Netravali  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.16565v1  

#### Abstract
Skim is a speculative execution framework for web agents that exploits the predictable structure of purpose-built websites. Today's web-agent expense is not intrinsic to the tasks but a property of how agents are composed: frontier-model inference, browser rendering, and ReAct-style planning are app...

---

### 17. [ADR: An Agentic Detection System for Enterprise Agentic AI Security](https://arxiv.org/abs/2605.17380)

**Authors**: Chenning Li, Pan Hu, Justin Xu, Baris Ozbas, Olivia Liu, Caroline Van, Manxue Li, Wei Zhou, Mohammad Alizadeh, Pengyu Zhang, KK Sriramadhesikan, Ming Zhang  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.17380v1  

#### Abstract
We present the Agentic AI Detection and Response (ADR) system, the first large-scale, production-proven enterprise framework for securing AI agents operating through the Model Context Protocol (MCP). We identify three persistent challenges in this domain: (1) limited observability -- existing Endpoi...

---

### 18. [QQJ: Quantifying Qualitative Judgment for Scalable and Human-Aligned Evaluation of Generative AI](https://arxiv.org/abs/2605.17382)

**Authors**: Marjan Veysi, Pirooz Shamsinejadbabaki, Mohammad Zare, Mohammad Sabouri  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.17382v1  

#### Abstract
The rapid progress of generative artificial intelligence has exposed fundamental limitations in existing evaluation methodologies, particularly for open-ended, creative, and human-facing tasks. Traditional automatic metrics rely on surface-level statistical similarity and often fail to reflect human...

---

### 19. [Evidence-Grounded Frontier Mapping and Agentic Hypothesis Generation in Nanomedicine](https://arxiv.org/abs/2605.18144)

**Authors**: Christiaan G. A. Viviers, Koen de Bruin, Mirre M. Trines, Ayla M. Hokke, Roy van der Meel, Avi Schroeder, Twan Lammers, Willem J. M. Mulder, Fons van der Sommen  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.18144v1  

#### Abstract
Nanomedicine research spans delivery chemistry, immunology, imaging, biomaterials, and disease-specific translational science, yet its conceptual design space remains fragmented across a large and heterogeneous literature. To date, artificial intelligence in nanomedicine has focused primarily on pro...

---

### 20. [Learning Transferable Topology Priors for Multi-Agent LLM Collaboration Across Domains](https://arxiv.org/abs/2605.17359)

**Authors**: Taolin Zhang, Zijie Zhou, Jiuheng Wan, Tingyuan Hu, Chengyu Wang, Xiaofeng He, Richang Hong  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.17359v1  

#### Abstract
Large language model (LLM)-based multi-agent systems have shown strong potential for complex reasoning by coordinating specialized agents through structured communication. However, existing topology-evolution methods typically construct or optimize a collaboration topology for each query from scratc...

---

### 21. [S2Aligner: Pair-Efficient and Transferable Pre-Training for Sparse Text-Attributed Graphs](https://arxiv.org/abs/2605.18579)

**Authors**: Yuhan Wang, Haopeng Zhang, Yibo Ding, Jiaqi Yu, Xinyu Zhao, Yuhang Liu, Ziwei Zhang, Xiao Wang, Ruijie Wang  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.18579v1  

#### Abstract
Pre-training on text-attributed graphs (TAGs) is central to building transferable graph foundation models, where LLM-as-Aligner methods align graph and text representations through the semantic knowledge of large language models. However, these methods usually assume that node texts provide sufficie...

---

### 22. [NGM: A Plug-and-Play Training-Free Memory Module for LLMs](https://arxiv.org/abs/2605.16893)

**Authors**: Yuwen Qu, Wenhui Dong, Chenyang Si, Caifeng Shan  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.16893v1  

#### Abstract
Recent studies introduce conditional memory modules that decouple knowledge storage from neural computation, enabling more direct knowledge access. Compared to MoE, which relies on dynamic computation paths, explicit lookup provides a more efficient knowledge retrieval mechanism. However, these appr...

---

### 23. [DuIVRS-2: An LLM-based Interactive Voice Response System for Large-scale POI Attribute Acquisition](https://arxiv.org/abs/2605.17900)

**Authors**: Le Zhang, Shengming Zhang, Rui Zha, Yunpeng Wu, Jingbo Zhou, Jizhou Huang  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.17900v1  

#### Abstract
Accurate Point of Interest (POI) attribute acquisition is essential for location-based services, yet traditional modular Interactive Voice Response (IVR) systems suffer from error accumulation and high maintenance overhead. We present DuIVRS-2, a large language model (LLM)-based end-to-end framework...

---

### 24. [A Practical Noise2Noise Denoising Pipeline for High-Throughput Raman Spectroscopy](https://arxiv.org/abs/2605.18511)

**Authors**: David Martin-Calle (ILM,UCBL,CNRS), Cesar Alvarez Llamas (ILM,UCBL,CNRS), Vincent Motto- Ros (ILM,UCBL,CNRS), Christophe Dujardin (ILM,UCBL,CNRS,IUF), J\'er\'emie Margueritat (ILM,UCBL,CNRS), David Rodney (ILM,UCBL,CNRS)  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.18511v1  

#### Abstract
A lightweight and reproducible denoising pipeline for high-throughput Raman spectroscopy is presented. The approach relies on a one-dimensional convolutional autoencoder trained using a Noise2Noise strategy, requiring neither external spectral libraries nor high signal-to-noise reference spectra for...

---

### 25. [Democratizing Large-Scale Re-Optimization with LLM-Guided Model Patches](https://arxiv.org/abs/2605.18692)

**Authors**: Tinghan Ye, Arnaud Deza, Ved Mohan, El Mehdi Er Raqabi, Pascal Van Hentenryck  
**Category**: cs.AI  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.18692v1  

#### Abstract
Optimization models developed by operations research (OR) experts are often deployed as decision-support systems in industrial settings. However, real-world environments are dynamic, with evolving business rules, previously overlooked constraints, and unforeseen perturbations. In such contexts, end ...

---

### 26. [CompactAttention: Accelerating Chunked Prefill with Block-Union KV Selection](https://arxiv.org/abs/2605.16839)

**Authors**: Jiwon Song, Dongwon Jo, Beomseok Kang, Jae-Joon Kim  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.16839v1  

#### Abstract
Chunked prefill has become a widely adopted serving strategy for long-context large language models, but efficient attention computation in this regime remains challenging. Existing sparse attention methods are primarily designed for one-shot prefill and do not translate efficiently to chunked prefi...

---

### 27. [Taming "Zombie'' Agents: A Markov State-Aware Framework for Resilient Multi-Agent Evolution](https://arxiv.org/abs/2605.17348)

**Authors**: Taolin Zhang, Pukun Zhao, Qizhou Chen, Jiuheng Wan, Chen Chen, Xiaofeng He, Chengyu Wang, Richang Hong  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.17348v1  

#### Abstract
Recent advancements in LLM-based multi-agent systems have demonstrated remarkable collaborative capabilities across complex tasks. To improve overall efficiency, existing methods often rely on aggressive graph evolution among agents (e.g., node or edge pruning), which risks prematurely discarding va...

---

### 28. [NewsLens: A Multi-Agent Framework for Adversarial News Bias Navigation](https://arxiv.org/abs/2605.17364)

**Authors**: Joy Bose  
**Category**: cs.CL  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.17364v1  

#### Abstract
Media bias detection has predominantly been framed as a classification task: assign a political label to an article or outlet. We argue this framing is too shallow: it identifies that bias exists but not where, how, or crucially, what is structurally omitted. We present NewsLens, a five-agent advers...

---

### 29. [Geometric Asymmetry in MoE Specialization: Functional Decorrelation and Representational Overlap](https://arxiv.org/abs/2605.16349)

**Authors**: Feilong Liu  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.16349v1  

#### Abstract
Mixture-of-Experts (MoE) architectures achieve scalable capacity through sparse routing, yet the geometric structure of expert specialization remains poorly understood. We introduce a unified Jacobian-PCA-Grassmann framework for analyzing MoE layers in both function space and representation space. A...

---

### 30. [R2V Agent: Teaching SLMs When to Ask for Help](https://arxiv.org/abs/2605.16604)

**Authors**: Raghu Vamshi Hemadri, Humaira Firdowse Mohammed, Rishabh Maheshwary, Srivatsava Daruru, Sagar Davasam, Vikas Yadav, Srinivas Sunkara, Sai Rajeswar  
**Category**: cs.LG  
**Published**: 2026-05-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.16604v1  

#### Abstract
Efficient agentic systems should incur expensive frontier-model costs only on decisions where a cheaper local model is likely to fail. Existing LLM cascades usually route whole queries before execution, but task difficulty shifts mid-trajectory - after flaky tool calls, truncated observations, or co...

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
