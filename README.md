# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-14 07:16:52 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding](https://arxiv.org/abs/2604.10152)

**Authors**: Jehyeon Bang, Eunyeong Cho, Ranggi Hwang, Jinha Chung, Minsoo Rhu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 16.0  
**Type**: new  
**ArXiv ID**: 2604.10152v1  

#### Abstract
The Mixture-of-Experts (MoE) architecture has emerged as a promising approach to mitigate the rising computational costs of large language models (LLMs) by selectively activating parameters. However, its high memory requirements and sub-optimal parameter efficiency pose significant challenges for ef...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Mixture-of-Experts (MoE) 架构虽然通过稀疏激活降低了计算成本，但其巨大的内存需求导致在实际部署中面临严重挑战。尤其是当采用 **CPU-offloading** 策略将非活跃专家参数卸载到 CPU DRAM 时，频繁的专家迁移会因 PCIe 带宽限制而引入显著的通信开销，成为推理延迟的主要瓶颈。

现有方法如 **MoE-Overlap**（重叠计算与通信）和 **MoE-Caching**（缓存高频专家）在小批量场景下有效，但在大批量推理中效果有限，无法根本解决数据传输量大的问题。

---

### 提出的新方法与创新思路
本文提出 **SpecMoE** —— 一种基于 **Self-Assisted Speculative Decoding** 的高效 MoE 推理系统，其核心思想是：

- **算法层面：自辅助推测解码（Self-Assisted Speculative Decoding）**
  - 利用目标 MoE 模型自身的一部分（非专家层 + 少量热专家）构建一个无需额外训练的 **draft model**。
  - 在推测阶段（speculation），仅使用驻留在 GPU 上的“草稿专家”生成多个 draft tokens。
  - 在验证阶段（verification），由完整的目标模型并行验证这些 tokens，并生成最终输出。

- **系统层面：减少冗余专家迁移**
  - 所有专家加载请求被推迟至验证阶段统一处理，从而实现 **请求合并（coalescing）**，避免重复迁移相同专家。
  - 引入 **动态专家替换策略（HotTemporal）** 和 **基于亲和度的专家选择机制（affinity-based selection）** 来维持 draft model 的高质量。

---

### 相比现有方法的优势
| 特性 | SpecMoE | MoE-Overlap / MoE-Caching |
|------|--------|---------------------------|
| 是否需要额外 draft model | ❌ 否（复用原模型部分） | ✅ 是（需单独训练） |
| 是否引入额外内存开销 | ❌ 否 | ✅ 是（存储 draft 模型权重及 KV cache） |
| 对大批量推理的支持 | ✅ 高效扩展 | ⚠️ 效果随 batch size 下降 |
| 是否依赖模型修改 | ❌ 否 | ⚠️ MoE-Overlap 需架构改动 |
| 数据传输优化程度 | ✅ 显著降低（高达 76.73%） | ⚠️ 有限（缓存仅缓解部分） |

> ✅ **核心优势**：SpecMoE 是首个完全免训练、软硬件协同、适用于任意 MoE 架构的 speculative decoding 方案，在不增加资源的前提下大幅提升吞吐。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **NLLB-MoE**：在 `WMT-14` 英法翻译任务上进行主实验。
- **Mixtral-8x7B** 与 **Llama-4-Scout**：在 `CNN-DailyMail` 数据集上进行文本补全任务，用于评估低专家热度（low hotness）模型下的表现。

---

### 实验设置
- **硬件平台**：
  - 单块 NVIDIA H100 GPU（96GB HBM3）
  - 双 Intel Xeon Platinum 8558 CPU（共 1TB DDR5）
  - PCIe 5.0 连接（单向带宽 64 GB/s）

- **软件环境**：
  - PyTorch 2.5.0 + CUDA 12.6
  - Hugging Face Transformers 4.51.0 + Accelerate 1.6.0
  - 自定义实现 speculative decoding 支持动态 draft model 构建

- **评估指标**：
  - End-to-end inference latency（端到端延迟）
  - Throughput（吞吐量，单位：tokens/sec）
  - CPU-to-GPU data transfer size（专家迁移数据量）
  - Tokens generated per step $ T(\gamma) $（每步接受的 token 数）

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **MoE-OnDemand** | 每步按需从 CPU 加载所需专家，无优化 |
| **MoE-Overlap** | 尝试将通信与计算重叠（本文使用 oracle 版本以最大化收益） |
| **MoE-Caching** | 缓存前 10% 最常访问的专家于 GPU 内存 |

> 所有基线均采用相同的 CPU-offloading 设计，仅策略不同。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **最大吞吐提升** | 较 MoE-OnDemand 提升 **4.30×** |
| **CPU-GPU 数据传输减少** | 最多减少 **76.73%**（vs MoE-OnDemand / MoE-Overlap），较 MoE-Caching 减少 **71.89%** |
| **平均每步生成 token 数 $ T(\gamma) $** | 在 NLLB-MoE 上达 **7.265**（γ=10） |
| **大批次扩展性** | Batch=256 时，SpecMoE 延迟仅为 MoE-OnDemand 的 **23%**

---

### 与基线方法的对比结果
#### （1）吞吐量对比（图11）
- 随着 batch size 增加，SpecMoE 的吞吐增长最为显著。
- 在 batch=256 时，SpecMoE 达到约 **123 tokens/sec**，远超其他所有基线（普遍低于 50 tokens/sec）。

#### （2）数据传输量对比（图12）
- MoE-OnDemand / MoE-Overlap：传输量随 batch size 几乎线性上升（×45~50 倍）。
- MoE-Caching：适度降低（约 30% 节省）。
- **SpecMoE**：仅增加 **10.96×**，表现出极强的可扩展性。

#### （3）低专家热度模型表现（图14–15）
- 在 **Mixtral-8x7B**（skewness=0.32）上，SpecMoE 实现 **2.17×** 吞吐提升。
- 在 **Llama-4-Scout**（skewness=0.59）上，实现 **1.42×** 提升。
- 明显优于 MoE-Caching（最高仅 1.15×）和 MoE-Overlap（1.53×）。

---

### 消融实验结果
#### （1）专家选择策略对比（表 II）
| 策略 | Tokens/step | 性能加速（归一化） |
|------|------------|------------------|
| Random | 6.812 | 1.000 |
| HotGlobal（静态热专家） | 7.220 | 1.084 |
| **HotTemporal（动态更新）** | **7.265** | **1.143** |

> 表明动态捕捉时间局部性的专家替换策略对性能有显著增益（+6%）。

#### （2）draft 专家数量影响（表 III）
| Draft 专家数 N | Tokens/step | 吞吐（tokens/sec） |
|---------------|------------|--------------------|
| 2 | 6.929 | 116.40 |
| **4** | **7.265** | **122.97** ✅ |
| 8 | 7.620 | 121.31 |
| 16 | 7.819 | 117.16 |

> 并非越多越好！N=4 为最佳平衡点，更多专家反而因计算延迟上升而降低整体吞吐。

#### （3）亲和度机制有效性（图13）
- 使用 **affinity-based selection** 比随机选择平均多生成 **7.23%** 的 token。
- 证明该机制能有效逼近原始路由行为，提升 draft token 质量。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Speculative decoding 可成功应用于 MoE 模型**，且无需额外训练 draft model。
2. ✅ **利用 MoE 自身结构构建 draft model** 是高效且实用的设计范式。
3. ✅ **延迟合并 + 请求去重** 是应对 PCIe 带宽瓶颈的关键。
4. ✅ **即使在专家激活分布均匀的模型上**（如 Mixtral），SpecMoE 仍能取得显著加速。
5. ✅ **SpecMoE 在 SSD-offloading 场景下依然有效**（图16），平均提速 2.25×，说明其适用于高延迟存储环境。

---

### 方法的局限性
1. **小批量性能略逊于 MoE-Caching**：在 batch=1 时，由于 MoE-Caching 已命中常用专家，SpecMoE 的 speculative 开销略高。  
   → **解决方案**：可在运行时自动关闭 speculative decoding（软件级开关即可）。

2. **依赖专家激活的时间局部性**：若专家切换过于剧烈或无规律，HotTemporal 策略可能失效。

3. **当前未支持多头 speculative 或树状 speculative**：未来可结合 Medusa、SpecInfer 等进一步提升吞吐。

---

### 未来工作方向
1. 将 SpecMoE 扩展至 **multi-GPU + MoE parallelism** 场景。
2. 探索更智能的 **dynamic γ 控制机制**（根据上下文调整 draft token 数量）。
3. 结合 **KV cache reuse** 技术进一步压缩验证阶段开销。
4. 应用于 **fine-tuning 或 prefill 阶段** 的加速。
5. 探索在 **edge device** 上部署轻量化 SpecMoE 的可能性。

---

> 🔚 **总结一句话**：  
> **SpecMoE 提出了一种免训练、高兼容、强扩展的 MoE 推理加速框架，首次将 speculative decoding 成功适配于 MoE 架构，在大批量场景下实现了高达 4.30× 的吞吐提升，为现实世界中的 MoE 模型服务提供了极具前景的解决方案。**

</details>

---

### 2. [DERM-3R: A Resource-Efficient Multimodal Agents Framework for Dermatologic Diagnosis and Treatment in Real-World Clinical Settings](https://arxiv.org/abs/2604.09596)

**Authors**: Ziwen Chen, Zhendong Wang, Chongjing Wang, Yurui Dong, Luozhijie Jin, Jihao Gu, Kui Chen, Jiaxi Yang, Bingjie Lu, Zhou Zhang, Jirui Dai, Changyong Luo, Xiameng Gai, Haibing Lan, Zhi Liu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.09596v1  

#### Abstract
Dermatologic diseases impose a large and growing global burden, affecting billions and substantially reducing quality of life. While modern therapies can rapidly control acute symptoms, long-term outcomes are often limited by single-target paradigms, recurrent courses, and insufficient attention to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DERM-3R: A Resource-Efficient Multimodal Agents Framework for Dermatologic Diagnosis and Treatment in Real-World Clinical Settings

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

传统中医（TCM）在皮肤病诊疗中具有整体观和个体化治疗的优势，但其临床实践面临以下挑战：
- **知识体系非标准化**：辨证论治高度依赖医生经验，缺乏统一标准。
- **多模态记录不完整**：病历常缺失关键图像或文本信息，图文对应关系断裂。
- **专家推理难以规模化**：高质量病例积累困难，专家级诊断能力难以复制。

同时，现有的大型多模态大模型（Multimodal LLMs）在医疗领域存在：
- **资源消耗巨大**：百十亿参数模型部署成本高，不适合临床环境。
- **幻觉严重**：在专业领域生成内容偏离医学逻辑。
- **缺乏领域结构化建模**：通用模型无法有效模拟复杂的临床推理流程。

### 提出了什么新方法或新思路

提出 **DERM-3R** —— 一种**资源高效、基于多智能体协作的多模态框架**，用于真实临床场景下的皮肤科诊断与治疗决策。其核心思想是：

- **任务分解与重构**：将复杂的皮肤病诊疗过程解构为三个阶段：
  1. **细粒度皮损识别**（Fine-grained lesion recognition）
  2. **多视角皮损表征与病机建模**（Multi-view representation with specialist pathogenesis modeling）
  3. **整体辨证与治疗规划**（Holistic clinical reasoning for syndrome differentiation and treatment planning）

- **三智能体协同架构**：
  - **DERM-Rec**：单图皮损识别智能体，提取局部形态学特征。
  - **DERM-Rep**：多图整合与病机分析智能体，构建患者级综合描述与TCM病理机制。
  - **DERM-Reason**：多模态临床推理智能体，融合病史、症状、舌脉等信息进行辨证施治。

- **轻量化训练策略**：
  - 基于 **Qwen2.5-VL-7B** 构建基础模型，并集成 **Tianyi** 中医语言模型以增强领域知识。
  - 采用 **LoRA** 进行参数高效微调（Parameter-efficient fine-tuning），仅使用 **103例真实世界银屑病病例**（含多次复诊）完成训练。

### 相比现有方法的优势

| 维度 | DERM-3R | 通用大模型（如 GPT-5.1, Gemini-3-Flash） |
|------|---------|----------------------------------------|
| **资源效率** | ✅ 轻量模型 + 少样本训练 | ❌ 百亿参数 + 兆级token训练 |
| **专业准确性** | ✅ 结构化推理链符合TCM逻辑 | ❌ 输出泛化，缺乏因果一致性 |
| **临床适用性** | ✅ 可在有限算力下部署 | ❌ 部署门槛极高 |
| **性能表现** | ✅ 在多项任务上媲美甚至超越通用大模型 | ⚠️ 表面流畅但细节错误频发 |

---

## 2. 核心实验方法和设置

### 使用的数据集

所有数据均来自**真实世界中医皮肤科临床实践**，共包含 **103例银屑病患者**，每例含多次复诊记录，具体子数据集如下：

| 智能体 | 数据集名称 | 内容 | 规模 |
|-------|-----------|------|-----|
| DERM-Rec | `DRec` | 单张皮损图像 ↔ 细粒度形态描述 | 518 对 |
| DERM-Rep | `DRep` | 多图输入 ↔ 整体皮损描述 + TCM病机分析 | 148 个样本 |
| DERM-Reason | `DReason` | 完整病历（图文+病史+舌脉）↔ 辨证分型+治则+方剂 | 134 个样本 |

> 所有标注由多位中医皮肤科专家独立完成并交叉验证，确保质量和一致性。

### 实验设置和评估指标

#### 评估框架设计为双轨制：

| 类型 | 自动评估（Automatic Evaluation） | 人工评估（Human Evaluation） |
|------|-------------------------------|-----------------------------|
| 方法 | - BLEU-4, ROUGE-L<br>- **LLM-as-a-Judge**（结合RAG增强的专业判断） | 多中心交叉验证：<br>来自 **9家医院的15位资深皮肤科医师**匿名评分 |
| 目标 | 客观衡量生成质量 | 获取临床专家对推理逻辑、合理性的真实反馈 |
| 评分维度 | - 皮损描述准确率<br>- 病机分析正确性<br>- 治疗原则匹配度等 | 每项满分10分：<br>- 皮损描述<br>- 病因病机分析<br>- 辨证分型<br>- 治疗原则<br>- 处方用药<br>- 可读性（总分60） |

#### 基线方法对比

选取四类代表性多模态LLM作为基线：
- **大规模通用模型**：`GPT-5.1-instant`, `Gemini-3-Flash`
- **同规模开源模型**：`Qwen2.5-VL-7B`, `Qwen3-VL-8B`

> 特别说明：未引入更大规模模型（如Qwen3-72B）是出于**临床可部署性**的实际考量。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）自动评估结果（BLEU-4 / ROUGE-L）

##### 表1：DERM-Rep 性能对比（集成皮损描述 + TCM病机分析）

| 模型 | 描述 BLEU-4 | 病机 BLEU-4 | **平均 BLEU-4** | 描述 ROUGE-L | 病机 ROUGE-L | **平均 ROUGE-L** |
|------|-------------|-------------|----------------|---------------|---------------|------------------|
| GPT-5.1 | 0.0714 | 0.0105 | 0.0410 | 0.2331 | 0.1758 | 0.2044 |
| Gemini-3-Flash | 0.0584 | 0.0102 | 0.0343 | 0.2324 | 0.1423 | 0.1874 |
| Qwen2.5-VL-7B | 0.0543 | 0.0658 | 0.0600 | 0.3325 | 0.2682 | 0.3004 |
| Qwen3-VL-8B | 0.0354 | 0.0243 | 0.0298 | 0.2662 | 0.1767 | 0.2214 |
| **DERM-Rep** | **0.2298** | **0.1246** | **0.1772** | **0.4786** | **0.3763** | **0.4275** |

> ✅ **DERM-Rep全面领先**，平均得分约为通用模型的 **2–5倍**

##### 表2：DERM-Reason 性能对比（最终诊疗策略）

| 子任务 | GPT-5.1 | Gemini | Qwen2.5 | Qwen3 | **DERM-Reason** |
|--------|--------|--------|--------|--------|------------------|
| **平均 BLEU-4** | 0.1664 | 0.1322 | 0.1646 | 0.1861 | **0.2887** |
| **平均 ROUGE-L** | 0.3735 | 0.3048 | 0.3921 | 0.2809 | **0.5802** |

> ✅ 在处方生成外的所有任务上显著领先，尤其在**治疗原则选择**（BLEU-4: 0.4877）和**方剂匹配**（ROUGE-L: 0.7778）上优势巨大。

#### （2）LLM-as-a-Judge 评估结果

使用 **Gemini-3-Flash**, **GPT-5.2**, 和 **Deepseek-V3.2** 作为裁判模型，结合RAG检索增强评判专业性。

| 模型 | Judge平均总分 |
|------|--------------|
| GPT-5.1-instant | 23.6462 |
| Qwen2.5-VL-7B | 18.8866 |
| **DERM-Reason** | **34.4366** |

> ✅ **大幅领先第二名近11分**，且在各子维度（病因病机、辨证、治法、处方）均排名第一。

#### （3）人类专家评估结果（多中心临床评审）

| 模型 | 总分（60） | 方差 |
|------|----------|------|
| Qwen2.5-VL-7B | 35.54 | 2.06 |
| Qwen3-VL-8B | 39.42 | 1.91 |
| GPT-5.1-instant | 41.23 | 2.03 |
| Gemini-3-Flash | 41.17 | 2.05 |
| **DERM-3R** | **44.16** | **1.49** |

> ✅ **取得最高分且最低方差**，表明其不仅性能最优，输出也最稳定可靠。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **结构化多智能体优于暴力扩展**：  
   即使在极小样本（103例）、轻量模型（7B）条件下，通过合理的任务分解与智能体分工，**DERM-3R 的性能可媲美甚至超越百亿参数通用多模态模型**（如 GPT-5.1 和 Gemini-3-Flash）。

2. ✅ **领域感知建模至关重要**：  
   将中医理论系统（如“血热生风”、“阴虚燥热”）显式编码进模型架构与训练目标中，显著提升了病机解释与辨证推理的**临床一致性与逻辑连贯性**。

3. ✅ **多模态融合需面向临床工作流**：  
   图像识别必须服务于下游辨证，因此 DERM-Rec → DERM-Rep → DERM-Reason 的**级联推理架构更贴近真实医生思维路径**。

4. ✅ **少样本+高效微调可行**：  
   证明了在高质量标注支持下，**LoRA 微调可在极低数据量下实现强大的多模态医学推理能力**，为资源受限场景提供实用方案。

### 方法的局限性

1. **数据来源单一**：当前训练数据来自**单一中心、单一学派**，尚未涵盖不同TCM流派间的差异，可能影响泛化性。
2. **图像模态有限**：仅使用普通拍照图像，未引入**皮肤镜、组织病理**等高级影像，限制了细微特征捕捉能力。
3. **疾病谱较窄**：目前仅验证于**银屑病**，其他复杂皮肤病（如湿疹、红斑狼疮）仍需进一步测试。
4. **Remission期识别弱**：由于训练集中缓解期样本较少（仅占24.8%），模型对该阶段皮损描述准确性较低。

### 未来工作方向

1. **拓展疾病种类与多中心数据**：收集更多皮肤病种及跨区域、多流派TCM病例，提升模型普适性。
2. **引入多模态成像技术**：整合 dermoscopy、histopathology 图像，增强视觉语义理解。
3. **探索“广博学习”机制**：借鉴“师从多家”的中医传承方式，研究如何让模型安全地吸收多个TCM学派的知识而不产生内部冲突。
4. **推动人机协同闭环**：将 DERM-3R 部署至临床辅助系统，形成“AI建议 → 医生修正 → 反馈训练”的正向循环，持续优化模型性能。

---

> **总结**：DERM-3R 展示了一条不同于“大力出奇迹”的医疗AI发展路径——**以临床需求为导向，通过结构化建模与资源高效训练，在有限条件下实现媲美甚至超越通用大模型的专业性能**。这为中医智能化、基层医疗赋能以及可解释AI在医学中的应用提供了重要范式。

</details>

---

### 3. [Leveraging Mathematical Reasoning of LLMs for Efficient GPU Thread Mapping](https://arxiv.org/abs/2604.10387)

**Authors**: Jose Maureira, Crist\'obal A. Navarro, Hector Ferrada, Luis Veas-Castillo  
**Category**: cs.DC  
**Published**: 2026-04-14  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.10387v1  

#### Abstract
Mapping parallel threads onto non-box-shaped domains is a known challenge in GPU computing that, if done efficiently, can prevent severe performance penalties from allocating unnecessary computational resources. Currently, achieving this optimal efficiency requires significant analytical human time ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Leveraging Mathematical Reasoning of LLMs for Efficient GPU Thread Mapping

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **GPU computing** 中，将并行线程高效映射到非规则几何域（如三角形、四面体、分形等）是一个长期存在的挑战。传统的 **Bounding Box (BB)** 映射策略会分配大量无效线程块，导致严重的资源浪费（idle threads）、能耗增加和性能下降。

手动为每种复杂几何形状推导精确的 `O(1)` 或 `O(log N)` 映射函数需要大量的数学分析工作，效率低下且难以扩展。

### 提出的新方法
本文提出了一种全新的自动化框架，利用 **Large Language Models (LLMs)** 的符号推理能力，通过 **in-context learning** 自动推导出适用于复杂几何域的精确 GPU 线程映射函数。

该方法将线程映射问题从一个“连续数值拟合”任务（如传统 Symbolic Regression）转变为一个“离散模式识别与算法逆向工程”任务。

### 相比现有方法的优势
- **相比人工推导**：完全自动化，无需专家介入，显著降低人力成本。
- **相比传统 Symbolic Regression (SR)**：
  - SR 是基于数据拟合的方法，输出是近似解，在整数索引场景下不可接受；
  - 本文方法由 LLM 推理生成的是**精确的、可证明正确的解析表达式**，满足 GPU 内存寻址对绝对精度的要求。
- **开源可控**：使用 **open-weight LLMs** 在本地执行，实现完全主权（sovereign）的自动化流程，避免依赖闭源 API。

---

## 2. 核心实验方法和设置

### 使用的数据集 / 测试领域
研究在六种具有递增复杂度的计算域上进行评估，涵盖密集结构与分形结构：

| 域 | 类型 | 复杂度 |
|----|------|--------|
| 2D Triangular | Dense | `O(1)` |
| 3D Pyramid | Dense | `O(1)` |
| 2D Sierpinski Gasket | Fractal | `O(log₃N)` |
| 2D Sierpinski Carpet | Fractal | `O(log₈N)` |
| 3D Sierpinski Pyramid | Fractal | `O(log₄N)` |
| 3D Menger Sponge | Fractal | `O(log₂₀N)` |

这些领域的 Ground Truth 映射逻辑来自已有文献（见 Table I），作为验证标准。

### 实验设置
- **模型选择**：共评测 11 个 state-of-the-art open-weight LLMs，包括：
  - `deepseek-r1:70b`（强化推理型）
  - `gpt-oss:120b`, `gpt-oss:20b`
  - `llama3.3:70b`, `llama4:16x17b`
  - `qwen3:235b`, `qwen3:32b`
  - `gemma3`, `mistral-nemo`, `nemotron` 等
- **输入方式**：采用 **few-shot in-context learning**，提供前 N 个 `(index → coordinates)` 对作为上下文提示。
  - Stage 20：前 20 个样本
  - Stage 50：前 50 个样本
  - Stage 100：前 100 个样本
- **Prompt 设计**：严格约束输出格式（仅返回 Python 函数 `map_to_coordinates(n)`），禁止硬编码、查表、解释文本等，确保生成的是通用算法。

### 评估指标
1. **Ordered Accuracy (%)**  
   输出序列是否与 Ground Truth 完全一致（顺序 + 坐标都匹配）。
2. **Any-order Accuracy (%)**  
   是否覆盖所有正确坐标（允许顺序不同），用于识别“银级”解决方案。
3. **Big-O Efficiency**  
   分析生成代码的时间复杂度（如 `O(1)`, `O(log N)`, `O(N^{1/3})`），判断是否达到最优。
4. **Energy Efficiency (Points/Joule)**  
   衡量 LLM 推理阶段的能量效率。
5. **CUDA Block-Level 性能**  
   将生成的映射函数集成进 CUDA kernel，测量实际运行时间与能耗。

### 基线方法对比
- **Bounding Box (BB)**：朴素方法，用包围盒启动所有线程，再过滤无效区域。
- **Paper / Analytical (Gold Standard)**：人工推导的最优映射函数（来自文献）。
- **Traditional Symbolic Regression (SR)**：文中指出其系统性失败于此类离散任务，未列出具体数据。

---

## 3. 主要实验结果和性能指标

### 符号推理准确性（Tables II–VII）
- ✅ **2D Triangular**：多个顶级模型（如 `OSS:120b`, `R1:70b`, `Qw3:32b`）在各 stage 下均达到 **100% Ordered Accuracy**。
- ✅ **2D Sierpinski Gasket**：`OSS:120b` 和 `OSS:20b` 成功推导出正确逻辑。
- ✅ **2D Sierpinski Carpet**：`OSS:120b`（stage 100）和 `Qw3:235b`（stage 20/50）实现 **100% Ordered Accuracy**。
- ✅ **3D Triangular / Pyramid**：`OSS:120b`, `Qw3` 系列表现优异；`R1:70b` 虽初期 Any-order 高但 Ordered 低，显示其具备部分理解能力。
- ❌ **3D Menger Sponge**：所有 open-weight 模型表现极差，Any-order 最高仅 **~0.36%**，Ordered 全为 0%，揭示当前“**reasoning ceiling**”。

> 注：`R1:70b` 因其 Chain-of-Thought 推理机制耗时更长，能力建立较慢但潜力大。

### 能效分析（Figure 5）
- **Reasoning-driven penalty**：`deepseek-r1:70b` 能效低于同规模模型，因其内部推理链更长。
- **Parameter-driven penalty**：超大规模模型（如 `qwen3:235b`）因参数搬运开销大，能效较低。
- **上下文增益**：提供更多 in-context 示例（如从 20→100）通常提升能效，减少无效生成。

### CUDA 执行性能与节能效果（Tables VIII–IX）
| 场景 | 方法 | 时间 (ms) | 能耗 (J) | 加速比 | 节能倍数 |
|------|------|----------|---------|--------|----------|
| **3D Pyramid** | BB Baseline | 2530.65 | 282.67 | — | — |
|             | LLM Optimal (`OSS:120b`) | 3.84 | 0.92 | **~659×** | **~307×** |
| **3D Sierpinski** | BB (Projected) | ~15949 | ~1591 | — | — |
|                | LLM Bitwise (`OSS:120b`) | 3.30 | 0.55 | **~4833×** | **~2890×** |

> ⚡️ 结论：尽管 LLM 推理有一次性高能耗，但一经部署，其生成的优化 kernel 可在**首次执行中就完全摊销（amortize）前期成本**。

### 消融实验（隐含在结果中）
- **Stage 影响**：更多 in-context 示例有助于提高准确率与能效（如 `OSS:120b` 在 Sierpinski Carpet 上仅在 stage 100 成功）。
- **模型架构影响**：
  - 参数数量不等于成功（`qwen3:235b` 在某些任务失败）；
  - 推理机制（如 CoT）可能牺牲效率换取深度理解。
- **代码质量差异**：
  - 同一逻辑下，有的模型生成 `Analytical (O(1))`，有的却生成 `Linear Search (O(N^{1/3}))`，导致性能相差 24× 以上。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **现代 open-weight LLMs 能够自动推导出复杂的 GPU thread mapping 方程**，涵盖 2D/3D 密集域与多种分形结构。
2. ✅ **LLM 方法远胜传统 Symbolic Regression**，因其本质是**算法归纳（algorithm induction）** 而非数值拟合。
3. ⚖️ **存在显著的“能量权衡”（energy trade-off）**：
   - 前期 LLM 推理能耗高（尤其对 reasoning-heavy 模型）；
   - 但一旦集成，节省的 GPU 执行能耗巨大，**一次推理即可永久受益**。
4. 🧱 **存在“Menger Limit”**：当前 open-weight 模型无法处理高度递归的 3D 分形（如 Menger Sponge），构成当前能力边界。

### 局限性
- 仅适用于具有确定性数学规律的几何结构（如算术序列、自相似分形）。
- 不适用于任意非结构化网格或依赖外部数据的动态拓扑。
- 当前最强模型仍受限于深层递归三维结构的理解能力。
- LLM 推理过程本身能耗较高，不适合频繁变更几何类型的场景。

### 未来工作方向
1. **轻量化 fine-tuning**：训练小型（<10B）专用模型用于离散空间推理，降低推理能耗。
2. **突破 Menger Boundary**：随着 MoE 架构、强化学习训练（如 CoT RL）的发展，预期下一代 open-weight 模型将攻克 3D 递归分形。
3. **扩展应用范围**：将框架推广至非结构化网格、自适应细化网格（AMR）等更广泛的 HPC 场景。
4. **持续监控 open-weight 生态演进**，建立自动化 benchmark 追踪数学推理能力进展。

---

> 🔗 **代码与数据公开**：  
> GitHub 仓库：[https://github.com/aspiadevs/llm-gpu-thread-mapping](https://github.com/aspiadevs/llm-gpu-thread-mapping)  
> 支持完整复现。

</details>

---

### 4. [CodeComp: Structural KV Cache Compression for Agentic Coding](https://arxiv.org/abs/2604.10235)

**Authors**: Qiujiang Chen, Jing Xiong, Chenyang Zhao, Sidi Yang, Ngai Wong  
**Category**: cs.CL  
**Published**: 2026-04-14  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.10235v1  

#### Abstract
Agentic code tasks such as fault localization and patch generation require processing long codebases under tight memory constraints, where the Key-Value (KV) cache becomes the primary inference bottleneck. Existing compression methods rely exclusively on attention signals to estimate token importanc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CodeComp: Structural KV Cache Compression for Agentic Coding

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**Agentic coding**任务（如故障定位、补丁生成）中，模型需要处理超长代码库（repository-level context），导致 **KV Cache** 成为推理阶段的主要内存瓶颈。现有 KV Cache 压缩方法（如 SnapKV、ParallelComp）依赖 **attention scores** 来判断 token 重要性，但在代码场景下存在严重缺陷：

- **结构性盲区（Structural Blind Spot）**：attention 往往低估对程序语义至关重要的结构化元素（如函数调用 `callsite`、分支条件 `branch`、返回语句 `return`、赋值 `assignment`），导致这些关键 token 被错误剪枝。
- **预算分配不合理**：现有方法采用统一或 attention 驱动的压缩策略，无法根据代码的**结构重要性**动态分配压缩预算。

### 🚀 提出的新方法：CodeComp
提出 **CodeComp** —— 一种**无需训练**的结构感知 KV Cache 压缩框架，首次将**静态程序分析**（Static Program Analysis）引入 KV Cache 压缩流程，通过 **Code Property Graph (CPG)** 提供结构先验。

#### 核心创新机制：
1. **Span-Level Structural Protection（结构化跨度保护）**
   - 利用 Joern 提取 CPG，识别语义关键的局部代码区域（如函数调用、控制流谓词、返回语句等）。
   - 将这些区域标记为“保护跨度”（protected spans），**强制保留**在 KV Cache 中，无论其 attention 分数多低。

2. **Structure-Aware Budget Allocation（结构感知预算分配）**
   - 基于 CPG 特征（如调用节点数、控制流边数）计算每个 chunk 的结构重要性得分。
   - 更重要的 chunk 获得更高的 KV Cache 容量配额，实现更合理的资源分配。

3. **两阶段压缩流程**
   - 先由结构信号决定必须保留的内容（硬约束）；
   - 再用 attention-based 压缩填充剩余容量（软选择），形成互补机制。

### 🔍 相比现有方法的优势
| 维度 | 传统方法（SnapKV / ParallelComp） | CodeComp |
|------|-------------------------------|--------|
| 重要性判断依据 | 仅依赖 attention scores | 结合 CPG 结构先验 + attention |
| 关键 token 保留能力 | 弱，常误删 callsite、branch 等 | 强，显式保护结构关键区域 |
| 压缩效率与准确性平衡 | 差，在高压缩比下性能崩溃 | 优，高压缩比下仍接近全上下文性能 |
| 是否需模型微调 | 否 | 否（training-free） |
| 可集成性 | 多基于 Transformers 库 | 原生支持 SGLang，无缝接入 agent 流程 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **SWE-bench Lite** | 补丁生成（Patch Generation） | 包含真实 GitHub issue 和对应修复，评估模型能否生成可应用的补丁 |
| **LCA (Library-based Code Generation)** | API 正确性代码生成 | 要求模型调用正确的库 API，评估 API 级语义正确性 |
| **InfiniteBench-CodeDebug** | 长上下文 bug 定位 | 多文件、深层依赖的调试任务 |
| **DebugBench** | Bug 定位 | 评估模型在复杂代码中定位错误位置的能力 |
| **LongCodeQA** | 长代码问答 | 涉及跨文件逻辑推理的问题 |

### ⚙️ 实验设置
- **模型**：
  - `DS-Coder`, `Qwen-Coder`, `Llama3-8B-Instruct`, `Qwen3-8B`
- **压缩比例（KV Cache Capacity Ratio）**：
  - 设置为 `0.4` 和 `0.6`，即仅保留 40%~60% 的 KV 缓存
- **Chunk 划分**：
  - 按函数/方法边界划分，最长 4096 tokens，最小 128 tokens
- **结构特征提取工具**：
  - 使用 **Joern** 提取 **CPG**，识别 callsite、control-flow、return、assignment 等结构单元

### 📊 评估指标
| 指标 | 含义 |
|-----|------|
| **GF Prec/Rec/F1/Jac** | Ground-truth File-level 指标，衡量是否定位到正确文件 |
| **Patch Valid** | 生成的补丁是否可通过编译并修复问题 |
| **Edit Distance** | 生成补丁与参考补丁之间的编辑距离，越小越好 |
| **API Prec/Rec/F1/Jac** | API 调用级别的精确率、召回率、F1 和 Jaccard 相似度 |
| **Str. Score** | 结构保留分数，表示被保留的结构关键 token 占比 |
| **End-to-End Latency** | 整体推理延迟，评估实际部署效率 |

### 🆚 基线方法对比
- **No Compression**：完整上下文，作为上限基准
- **SnapKV**：基于 observation window 的 attention 驱动 token 保留
- **ParallelComp**：chunk-level 并行压缩，attention 驱动
- **CodeComp (Ours)**：结构感知压缩（span protection + structure-aware budget）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 2 & 3）

#### 在 **SWE-bench Lite** 上的表现（GF F1）
| Model | Cap. | SnapKV | ParallelComp | **CodeComp** |
|-------|------|--------|--------------|-------------|
| DS-Coder | 0.4 | 0.200 | 0.021 | **0.250** |
| DS-Coder | 0.6 | 0.350 | 0.038 | **0.250** |
| Qwen-Coder | 0.4 | 0.600 | 0.372 | **0.547** |
| Qwen-Coder | 0.6 | 0.700 | 0.297 | **0.613** |

> 💡 **CodeComp 在高压缩比下显著优于 ParallelComp**，甚至在 DS-Coder 上超过 SnapKV；在 Qwen-Coder 上虽略低于 SnapKV 的 F1，但**edit distance 更低（0.683 vs 0.718）**，说明生成质量更高。

#### 在 **DebugBench** 上的平均准确率（Table 3）
| Model | Cap. | SnapKV | ParallelComp | **CodeComp** |
|-------|------|--------|--------------|-------------|
| Llama3-8B | 0.4 | 0.41 | 0.03 | **0.43** |
| Qwen3-8B | 0.4 | 0.71 | 0.25 | **0.72** |
| Qwen3-8B | 0.6 | 0.66 | 0.48 | **0.79** |

> ✅ **CodeComp 在所有设置下均达到最优或次优**，尤其在 ParallelComp 几乎失效时仍保持高精度。

#### 结构保留能力（Str. Score）
| 方法 | Qwen3-8B @ Cap=0.6 | Str. Score |
|------|------------------|-----------|
| SnapKV | – | 0.57 |
| **CodeComp** | – | **0.77** |

> 🔥 **尽管保留 token 数量相近，CodeComp 保留了更多语义关键 token**，验证了结构先验的有效性。

### 🔬 消融实验结果（Table 4）

| 设置 | Cap=0.4 GF F1 | Cap=0.6 GF F1 | 发现 |
|------|---------------|---------------|------|
| **Span-only** | 0.633 | 0.783 | 仅保留结构跨度即可大幅提升性能 |
| **Capacity-only** | 0.283 | 0.450 | 仅调整预算分配效果有限 |
| **Capacity-Span (Full)** | **0.667** | **0.800** | 两者结合最佳，span protection 是主导因素 |

> 📌 **核心发现**：**span-level protection 是性能提升的主要来源**，而 budget allocation 提供轻量级增强。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Attention 不是代码中 token 重要性的可靠代理**  
   - 图 2 显示 attention 排名与 CPG 结构重要性排名的 Jaccard 重叠仅为 **0.0944**，表明两者几乎不相关。

2. **结构关键 token 被系统性误删**  
   - 在 attention-only 压缩下，高达 **50% 以上的 callsite、branch、return token 被丢弃**。

3. **引入 CPG 先验可显著恢复性能**  
   - CodeComp 在 **40% KV Cache 容量下恢复了 80% 以上全上下文性能**，在某些任务上 patch valid 达到 **1.000**。

4. **结构保留分数（Str. Score）是良好代理指标**  
   - 该指标与任务准确率高度正相关，可用于快速评估压缩策略的质量。

5. **端到端延迟无显著增加**  
   - 如图 4b 所示，CodeComp 的推理延迟稳定在 112–118 秒之间，**结构分析开销可忽略**。

### ⚠️ 方法的局限性
- **依赖外部工具 Joern**：增加了部署复杂性，可能影响跨语言兼容性。
- **CPG 提取成本**：虽然推理时不重复运行，但仍需预处理步骤。
- **对非结构化脚本语言（如 Bash）效果未知**：目前主要针对 Python/Java/C++ 等结构化语言设计。
- **未探索动态分析信号**：当前仅使用静态分析，未来可融合运行时信息。

### 🔮 未来工作方向
1. **支持更多编程语言和 CPG 工具链**，提升通用性。
2. **将结构先验反向用于训练阶段**，构建原生结构感知的 LLM。
3. **结合动态执行轨迹**（execution traces）进一步优化关键路径保留。
4. **自动化 span 保护规则学习**，减少人工定义。
5. **扩展至其他结构化数据模态**，如 JSON Schema、数据库模式等。

---

## 总结

> **CodeComp 是首个将静态程序分析（via CPG + Joern）融入 KV Cache 压缩的框架，解决了 attention-only 方法在代码任务中的结构性盲区问题。它通过 span-level protection 和 structure-aware budget allocation，在极低 KV Cache 容量下仍能保持高性能，且无需模型修改，可无缝集成至 SGLang 等 agent 流水线中。实验证明其在 bug localization 和 code generation 多项任务上显著优于 SnapKV 和 ParallelComp，是迈向高效、可信 Agentic Coding 的重要一步。**

</details>

---

### 5. [Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale](https://arxiv.org/abs/2604.11554)

**Authors**: Liujie Zhang, Benzhe Ning, Rui Yang, Xiaoyan Yu, Jiaxing Li, Lumeng Wu, Jia Liu, Minghao Li, Weihang Chen, Weiqi Hu, Lei Zhang  
**Category**: cs.CL  
**Published**: 2026-04-14  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.11554v1  

#### Abstract
Reinforcement learning (RL) post-training has proven effective at unlocking reasoning, self-reflection, and tool-use capabilities in large language models. As models extend to omni-modal inputs and agentic multi-turn workflows, RL training systems face three interdependent challenges: heterogeneous ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大规模语言模型（LLM）的 **Reinforcement Learning (RL)** 后训练面临三大挑战：
1. **异构模态数据流**：从纯文本扩展到图像、音频、视频等多模态输入，导致数据预处理、并行策略和推理后端复杂化。
2. **大规模下的运行鲁棒性**：长尾延迟、OOM（Out-of-Memory）、NCCL 超时等问题频发，传统单体架构难以实现细粒度容错。
3. **吞吐量与策略新鲜度（staleness）的权衡**：同步训练效率低，异步训练虽提升吞吐但可能因策略过期影响收敛。

现有框架如 veRL、OpenRLHF 等在这些方面存在局限，尤其对 **omni-modal** 和 **agentic multi-turn** 工作流支持不足。

---

### 提出的新方法与架构设计
作者提出 **Relax** ——一个开源的大规模异步强化学习引擎，通过三层协同设计解决上述问题：

#### ✅ 1) **Role-Isolated Service Architecture (S3.2)**
- 每个 RL 角色（Actor、Critic、Rollout、Reward Model 等）作为独立的 **Ray Serve Deployment** 运行。
- 实现：
  - **故障隔离**：单个服务崩溃不影响全局训练。
  - **弹性伸缩**：可独立扩缩容 Rollout 或 Trainer 集群。
  - **生命周期管理**：支持 per-role 快照恢复。
- 引入 **Distributed Checkpoint Service (DCS)**，将权重同步解耦为专用服务，支持跨集群 NCCL/TCP 高效传输。

#### ✅ 2) **Staleness-Unified Asynchronous Training (S3.3)**
- 基于 **TransferQueue** 构建异步数据总线，所有角色仅通过该总线通信。
- 核心机制：
  - `max_staleness` 参数统一控制训练模式：
    - `max_staleness=0` → on-policy
    - `max_staleness>0` → off-policy
  - 支持 **streaming micro-batch pipeline**，避免全局 batch 同步造成的长尾阻塞。
- 结果：**同一代码库、同一入口脚本即可切换训练模式**，无需重写逻辑。

#### ✅ 3) **Omni-Modal Agentic RL Pipeline (S4)**
- 全栈原生支持多模态（image/text/audio/video），包括：
  - 统一的数据流水线（modality-aware loader + mask 标注）
  - 并行策略优化：
    - **ViT Tensor Parallelism**：视觉编码器在 TP 组内复制而非切分。
    - **Encoder-Aware Pipeline Parallelism**：视觉/音频编码器置于 PP0，减少跨阶段通信。
- 支持 agentic 多轮交互：
  - 可插拔的 **custom reward services**
  - 工具调用（tool calling）与沙箱集成（sandbox integration）

---

### 相比现有方法的优势
| 特性 | Relax | veRL / OpenRLHF |
|------|-------|----------------|
| 多模态原生支持 | ✅ 全栈设计 | ❌ 文本优先，多模态为补丁 |
| 服务级容错 | ✅ 独立部署 + DCS | ❌ 单体架构，失败即重启 |
| 异步灵活性 | ✅ 单参数切换 on/off-policy | ⚠️ 需不同配置模板 |
| MoE 模型稳定性 | ✅ R3 开销仅 +1.9% | ❌ veRL 中 R3 开销达 32% |
| 吞吐性能 | 最高 **2.00× speedup** | 基准 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 描述 |
|--------|------|------|
| **Echo Ink (AVQA-R1-6K)** | 图像+音频问答 | 用于验证 omni-modal RL 收敛性 |
| **NextQA** | 视频理解任务 | 包含 0–30 秒短视频片段 |
| **DAPO-MATH-17k** | 数学推理 | 纯文本 Chain-of-Thought 任务 |
| **Deepeyes** | 多轮工具调用 | 视觉 QA，需执行 crop/zoom 等操作 |

---

### 实验设置与评估指标

#### 模型规模
- **Qwen3-4B**（dense）
- **Qwen3-Omni-30B**（omni-modal）
- **Qwen3-30B-A3B**（MoE）

#### 硬件环境
- **16×H800 GPUs**（用于 Qwen3-4B / DAPO-MATH）
- **16×H20 GPUs**（用于 Qwen3-Omni-30B / Echo Ink）

#### 基线方法对比
- **veRL**：主流高性能 RL 框架，采用 HybridFlow 编程模型。
- 对比维度：
  - 端到端 step time
  - steps/hour
  - 收敛曲线一致性
  - R3 开销
  - 多模态稳定性

#### 评估指标
- **Per-step time (s)**：每步耗时
- **Throughput (steps/hour)**：训练吞吐
- **Speedup vs Colocate**：相对加速比
- **Reward Convergence**：奖励值随 step 变化
- **Routing Mismatch**：MoE 模型中 rollout 与 training 的专家选择差异
- **Idle Ratio**：Trainer GPU 空闲比例

---

## 3. 主要实验结果和性能指标

### 🔢 关键性能数据汇总

| 实验场景 | 方法 | Step Time | Steps/Hour | Speedup vs Colocate |
|---------|------|-----------|------------|---------------------|
| Qwen3-4B / DAPO-MATH | **Relax (Async Off-Policy)** | 128.6s | 28.0 | **1.76×** |
| | veRL (Async) | 150.5s | 23.9 | 1.00× |
| | Relax (Colocate) | 225.9s | 15.9 | 1.00× |
| Qwen3-Omni-30B / Echo Ink | **Relax (Fully Async)** | 133.6s | 26.9 | **2.00×** |
| | Relax (Colocate) | 267.4s | 13.5 | 1.00× |

> ✅ Relax 在 omni-modal 场景下加速优势更显著（2.00× > 1.76×），说明其架构优势随模型规模放大。

---

### 与基线方法对比结果

#### 📈 端到端性能 vs veRL
- **Relax 比 veRL 快 1.20×**（125.6s vs 150.5s/step）。
- 原因：
  - veRL 的 reference log-prob 和 rollout residual 仍在关键路径上（共 +65.5s）。
  - Relax 利用 micro-batch streaming + 资源分离，使这些前向计算不计入 step time。
- Trainer Idle Ratio：<0.1%，接近满载运行。

#### 🔄 R3（Rollout Routing Replay）开销对比
| 方法 | R3 开启后 Step Time 增加 |
|------|--------------------------|
| **Relax** | **+1.9%** |
| veRL | **+32%** |

> ✅ Relax 将 R3 数据通过 NCCL zero-copy 广播，并保持 GPU resident，彻底移出关键路径。

#### 📉 收敛质量
- 所有模式（colocate/on-policy/off-policy）均收敛至相同 reward 水平。
- Omni-modal 任务中：
  - Echo Ink：reward 从 0.72 → 0.93（450 steps）
  - NextQA（video）：持续训练 2,000 steps 无退化，reward 从 0.75 → 0.93，方差稳定（std ~0.04–0.06）

---

### 消融实验结果

#### ✅ R3 消融（Qwen3-30B-A3B）
- R3 显著降低 routing mismatch（↓38%），且不影响最终 reward。
- Relax 几乎零开销实现 R3，而 veRL 代价高昂。

#### ✅ FP16 vs BF16 精度对比（Qwen3-4B）
- FP16 下 train-rollout log-prob 差异仅为 BF16 的 **1/7.7**（0.0016 vs 0.0122）。
- FP16 更早进入正 reward 区域（~15 steps vs 30 steps）。
- 表明 **FP16 可有效缓解数值漂移问题**，是 R3 之外的重要补充手段。

---

## 4. 关键结论和发现

### 主要发现
1. **三者耦合设计形成正反馈闭环**：
   - omni-modal → 异构性 → 服务隔离 → 异步总线 → 字段级解耦 → 更好支持多模态。
2. **异步不是牺牲收敛换速度**：
   - 通过 `max_staleness` 控制，可在保证收敛的同时大幅提升吞吐。
3. **微批流式调度消除长尾瓶颈**：
   - micro-batch + streaming loader 使 trainer 几乎无等待。
4. **Relax 是首个支持全模态 RL 收敛的系统**：
   - 成功在 image/text/audio/video 上完成超过 2,000 步的稳定训练。

---

### 局限性
1. **暂不支持生成式多模态 RL**：
   - 当前聚焦于“感知”类任务（如看图说话），尚未支持 text-to-image 等生成式 reward 优化。
2. **部署复杂度较高**：
   - 需要运维 Ray Serve、TransferQueue、DCS 等多个组件，不适合短期实验。
3. **超大模型（>100B）验证仍在进行**：
   - 当前最大测试为 30B–35B，397B+ 规模仍在验证中。

---

### 未来工作方向
1. **支持生成式多模态 RL**（如 image/audio generation）
2. **开放弹性扩缩容能力**（Autoscaler via REST API）
3. **丰富 agentic 场景支持**：
   - SWE-Agent（软件工程）
   - CodeAgent（交互式调试）
   - Search-Augmented Reasoning
4. **进一步降低服务启动开销**，提升小规模实验友好性。

---

> 🔗 **项目地址**：[https://github.com/rednote-ai/Relax](https://github.com/rednote-ai/Relax)  
> Relax 已完全开源，旨在成为 omni-modal 与 agentic RL 训练的基础设施标准。

</details>

---

### 6. [Tessera: Unlocking Heterogeneous GPUs through Kernel-Granularity Disaggregation](https://arxiv.org/abs/2604.10180)

**Authors**: Tiancheng Hu, Jin Qin, Zheng Wang, Junhao Hu, Yuzheng Wang, Lei Chen, Yizhou Shan, Mingxing Zhang, Ting Cao, Chunwei Xia, Huimin Cui, Tao Xie, Chenxi Wang  
**Category**: cs.DC  
**Published**: 2026-04-14  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.10180v1  

#### Abstract
Disaggregation maps parts of an AI workload to different types of GPUs, offering a path to utilize modern heterogeneous GPU clusters. However, existing solutions operate at a coarse granularity and are tightly coupled to specific model architectures, leaving much room for performance improvement. Th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Tessera: Unlocking Heterogeneous GPUs through Kernel-Granularity Disaggregation**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代 GPU 数据中心日益**异构化**（heterogeneous），包含多种型号、代际和性能特征的 GPU（如 A100、H100、L40s、RTX Pro 6000 等）。然而，现有的 AI 推理调度方法在利用这种硬件多样性时存在以下问题：
- **粒度过粗**：现有 disaggregation 方法（如 prefill-decode 或 attention-FFN）以**执行阶段**或**模型块**为单位进行任务划分，忽略了更细粒度的资源需求差异。
- **架构耦合性强**：这些方法依赖特定模型结构（如 Transformer 的 prefill/decode 分离），难以推广到非标准架构（如 Mamba、Diffusion Models）。

这导致大量性能和成本效率潜力未被挖掘。

---

### **提出了什么新方法或新思路**
本文提出 **Tessera**，是首个基于 **kernel-granularity disaggregation**（内核级解聚）的异构 GPU 推理系统。

#### **核心思想**
- 将 AI 工作负载中的每个 **kernel** 视为独立调度单元，根据其实际资源需求（compute-bound / memory-bound）动态分配到最适合的 GPU 上。
- 利用 **kernel-level heterogeneity**（内核级异质性）匹配 **hardware heterogeneity**（硬件异质性），实现更精细的计算-硬件对齐。

#### **关键技术设计**
1. **PTX-Level Kernel Analyzer**
   - 在 PTX（Parallel Thread Execution）汇编层面插桩，精确分析 opaque kernels 的内存访问模式。
   - 提取每个 kernel 的读写缓冲区地址与大小，构建精确的 **Data Dependency Graph (DDG)**，确保跨 GPU 执行的功能正确性。

2. **Pipelined Execution Model**
   - 采用多请求流水线机制，在不同 CUDA stream 上并发处理多个请求。
   - 利用优先级感知的 stream 调度，错开通信相位，有效隐藏跨 GPU 通信延迟。

3. **Workload-Aware Scheduling Policies**
   - **吞吐量导向策略**：建模为混合整数线性规划（MILP），联合优化 kernel 放置、通信开销和负载均衡。
   - **延迟导向策略**：针对在线服务场景，最小化单个请求的关键路径延迟。
   - **在线监控器（Online Monitor）**：动态检测队列压力，自适应切换调度策略。

4. **轻量级运行时适配**
   - 与主流框架（vLLM、PyTorch）集成，支持 CUDA Graph，通过子图分解实现低开销重放。

---

### **相比现有方法的优势**
| 维度 | 现有方法（PD/AF Disaggregation） | Tessera |
|------|-------------------------------|--------|
| **调度粒度** | Phase / Block-level | ✅ **Kernel-level**（更细粒度） |
| **通用性** | 仅适用于特定架构（如 Transformer） | ✅ **Model-agnostic**（支持 LLM、SSM、MLLM、Diffusion） |
| **性能提升空间** | 忽略 kernel 内部异质性 | ✅ 充分利用 kernel 级性能差异 |
| **成本效率** | 固定分工，利用率受限 | ✅ 动态调度，最大化 Perf/$ |

---

## **2. 核心实验方法和设置**

### **使用的数据集与模型**
评估覆盖四类典型 AI 模型，验证通用性：
1. **LLMs**: Llama-3 8B, GPT-oss 20B
2. **SSMs**: Mamba-Codestral 7B（替代注意力机制）
3. **MLLMs**: Qwen2.5-VL 7B（图文多模态）
4. **Diffusion Models**: Stable Diffusion 3.5（图像生成）

输入数据来源：
- 文本：Splitwise 对话数据集（median input 1020 tokens）
- 图像：COCO captioning（512×512）
- Prompt：PartiPrompts（文本到图像）

---

### **实验设置**
#### **硬件平台**
- **本地节点配置**：
  - A100 + L40s
  - H100 + RTX Pro 6000
  - B200 + H100
- **集群扩展配置**：
  - 2×A100 + 1×L40s
  - 8×B200 + 8×H100
- 所有 GPU 配备 200Gbps RDMA NIC，NVLink 用于节点内连接。

#### **评估指标**
| 指标 | 定义 |
|------|------|
| **Throughput** | tokens/s（LLM/SSM/MLLM）、images/min（Diffusion） |
| **Latency** | normalized latency（end-to-end 延迟 / 输出 token 数） |
| **Cost Efficiency (Perf/$)** | 平均吞吐量 / GPU 租赁成本（归一化） |
| **SLO Compliance** | 在 50ms/token SLO 下可承受的最大请求率 |

---

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Homogeneous (左/右 GPU)** | 单一类型 GPU 运行，无 disaggregation |
| **PD Disaggregation** | Prefill-Decode 解聚（如 DistServe） |
| **AF Disaggregation** | Attention-FFN 解聚（如 MegaScale-Infer） |

> 注：PD 不适用于 Diffusion（无 prefill/decode 分离）；AF 不适用于 Mamba 和 Diffusion（非 Transformer 架构）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 离线吞吐量（Offline Throughput）**
- **平均提升**：
  - 相比 PD Disaggregation：**1.5×**（最高达 **2.3×**）
  - 相比 AF Disaggregation：**1.35×**（最高达 **1.7×**）
- **典型案例**：在 H100 + RTX Pro 6000 上运行 GPT-oss 20B，Tessera 吞吐量达到 PD 的 **2.3×**。

#### **(2) 成本效率（Cost Efficiency, Perf/$）**
| GPU Pair | Tessera vs PD | Tessera vs AF |
|---------|---------------|----------------|
| A100+L40s | **1.5×** | **1.4×** |
| H100+RTX Pro 6000 | **1.6×** | **1.5×** |
| B200+H100 | **1.01×** | **1.01×** |

> 💡 **惊人发现**：在 H100 + RTX Pro 6000 组合下，Tessera 的成本效率是单独使用 H100 的 **1.64×**，且吞吐量超过两个 H100！

#### **(3) 在线延迟表现（Online Latency）**
- 在 GPT-oss 20B 上测试 Poisson 请求流：
  - 低负载下，Tessera 的 normalized latency 比 PD 和 AF 分别降低 **1.3×** 和 **1.2×**。
  - 在 **50ms/token SLO** 下，Tessera 可承载的请求率比最佳基线高 **1.3×**。

#### **(4) 集群规模扩展性**
- 在 2×A100 + 1×L40s 和 8×B200 + 8×H100 集群上：
  - Tessera 仍保持 **1.5× 吞吐优势**于 PD，**1.4×** 于 AF。
  - 支持与 Tensor Parallelism（TP）自然组合，无需修改通信拓扑。

---

### **消融实验结果**

#### **(1) 流水线有效性（Pipelined Request Processing）**
- 无流水线 → 仅 50% 最优吞吐；
- 加入流水线 → 提升 **1.47×**；
- 加入优先级调度 → 达到最优吞吐的 **96.6%**。

> 说明：priority-aware scheduling 显著减少所有 stream 同时等待通信的现象。

#### **(2) 在线监控器敏感性**
- 默认参数：窗口 `W = 300ms`，阈值 `β = 1.5`
- 若过于激进（W=30ms），频繁切换策略导致额外 **30ms stall**，延迟上升 **1.32×**；
- 若过于保守（W=1.5s），响应滞后，延迟恶化 **1.55×**。

#### **(3) 网络带宽鲁棒性**
- 将 A100-L40s 间带宽从 200Gbps 降至 25Gbps：
  - 离线吞吐下降 < **6%**（得益于流水线掩盖延迟）；
  - 在线延迟仍优于同构 A100（因 latency-oriented policy 减少跨 GPU 传输）；
  - 最坏情况自动退化为单 GPU 执行，无性能悬崖。

#### **(4) MILP 求解可扩展性**
- 对含 1500 kernels 的 DeepSeek-V3 671B 模型，求解时间 < **1秒**；
- 利用层间重复性进一步压缩问题规模，加速求解。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Kernel-level heterogeneity 是真实存在的**：
   - 即使在同一 phase 或 block 中，也有大量 kernel 在不同 GPU 上表现迥异（如 FlashAttention 在 L40s 上快 2.1×）。
   - 粗粒度 disaggregation 必然“错配”大量 kernel。

2. ✅ **Kernel Granularity Disaggregation 显著提升性能与成本效率**：
   - 最高实现 **2.3× 吞吐提升** 和 **1.6× 成本效率增益**。
   - 异构 GPU 组合甚至能超越两个高端 GPU 的性能。

3. ✅ **通用性强，不受限于模型架构**：
   - 成功应用于 LLM、SSM、MLLM、Diffusion 等多种模型，而现有方法无法支持部分模型。

4. ✅ **系统设计高效可靠**：
   - PTX 分析保证功能正确性；
   - 流水线 + 优先级调度有效隐藏通信开销；
   - 动态策略切换适应负载变化。

---

### **方法的局限性**
1. **依赖 CUDA 生态**：目前仅支持 NVIDIA GPU 和 CUDA，尚未扩展至 AMD 或 NPU。
2. **不支持间接内存访问**：若 kernel 通过全局变量访问 buffer（非常见情况），分析器无法识别，会保守地不进行 disaggregation。
3. **MILP 求解为离线过程**：虽无运行时开销，但需预分析，不适合极端动态模型（但可通过 CUDA Graph 缓解）。
4. **内存复制开销**：当前默认全量复制权重，未做 selective unloading，限制了大 batch size 的潜力。

---

### **未来工作方向**
1. **跨厂商异构支持**：扩展至 AMD ROCm 或 Intel oneAPI 生态。
2. **更智能的内存管理**：结合运行时分析，按需加载权重分片，释放显存。
3. **与 MoE 架构深度整合**：将 kernel disaggregation 与 expert routing 结合。
4. **自动化 PTX 分析增强**：引入 ML 模型预测 opaque kernel 行为，提升覆盖率。
5. **边缘端部署探索**：在终端设备间实现 kernel 级任务迁移。

---

> **总结一句话**：  
> **Tessera 首次实现了 kernel-level 的异构 GPU 解聚，通过细粒度调度将“硬件碎片”转化为“性能红利”，在吞吐、延迟、成本和通用性上全面超越现有方法，为下一代 AI 推理系统提供了新范式。**

</details>

---

### 7. [Beyond Compliance: A Resistance-Informed Motivation Reasoning Framework for Challenging Psychological Client Simulation](https://arxiv.org/abs/2604.10507)

**Authors**: Danni Liu, Bo Liu, Yuxin Hu, Hantao Zhao, Yan Liu, Ding Ding, Jiahui Jin, Jiuxin Cao  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.10507v1  

#### Abstract
Psychological client simulators have emerged as a scalable solution for training and evaluating counselor trainees and psychological LLMs. Yet existing simulators exhibit unrealistic over-compliance, leaving counselors underprepared for the challenging behaviors common in real-world practice. To bri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有基于LLM的心理学**client simulator**普遍存在“过度顺从”（over-compliance）问题，即模拟的来访者过于开放、合作、情绪稳定，无法真实再现现实中常见的**client resistance**（客户阻抗）行为。这种偏差导致：
- 心理咨询培训场景失真；
- 对心理LLM的评估不够严格；
- 忽视了治疗中最具挑战性的互动动态。

### 提出了什么新方法或新思路
本文提出 **ResistClient**，一种基于**Client Resistance Theory**的挑战性来访者模拟框架，并引入 **Resistance-Informed Motivation Reasoning (RIMR)** 两阶段训练框架：

1. **Resistance-Informed Supervised Fine-Tuning (SFT)**  
   构建大规模阻抗导向的心理对话数据集 **RPC (Resistance-Informed Psychological Conversations)**，通过监督微调缓解预训练LLM的顺从性偏见。

2. **Motivation Reasoning Reinforcement Learning (MRRL)**  
   引入显式的**动机推理链**（Motivation Reasoning Chain），在生成响应前进行三步心理推理：
   - **Profile Reflection**：基于5P profile反思潜在阻抗倾向；
   - **Situation Awareness**：分析当前对话情境与咨询师话语；
   - **Reaction Decision**：决定反应类型与行为特征。  
   并采用**过程监督的强化学习**（process-supervised RL）优化推理过程的真实性与响应一致性。

### 相比现有方法的优势
- **超越表面模仿**：不仅模拟阻抗行为，还建模其背后的**心理动机机制**，提升行为的连贯性与真实性；
- **系统性建模多样性**：覆盖5种阻抗类型（Controlling, Emotional, Defensive, Avoidant, Compliant）与2种合作反应；
- **可解释性强**：显式推理步骤为行为提供透明解释，适用于教学与评估。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **RPC Dataset**：本文构建的新数据集，包含1,849个完整咨询会话，其中1,761个包含阻抗行为。
  - 数据来源：基于真实心理咨询语料库 **ProPsyC** 重写；
  - 结构：每个会话配有一个经验证的**5P client profile**（Presenting Problems, Predisposing/Precipitating/Perpetuating/Protective Factors）；
  - 标注：每条来访者回应标注**反应类型**与**动机解释**，由LLM辅助+专家验证。

### 实验设置和评估指标

#### 自动化评估指标
| 指标 | 含义 |
|------|------|
| **Precision / Recall / F1** | 衡量阻抗生成的准确性与时效性 |
| **RTF (Resistance Trigger Frequency)** | 阻抗触发频率，衡量全局挑战强度 |
| **CCR (Client Cooperation Rate)** | 客户合作率，越低表示挑战性越强 |
| **Turns** | 平均对话轮次，越长表示交互难度越高 |
| **Coh. (Coherence)** | 基于句嵌入的语义一致性得分 |

#### 人工评估指标（0–3分）
| 指标 | 含义 |
|------|------|
| **Fid. (Fidelity)** | 阻抗行为与目标类型的匹配度 |
| **Rat. (Rationality)** | 阻抗出现的情境合理性 |
| **Qua. (Quality)** | 动机推理的质量与逻辑连贯性 |
| **Real. (Realism)** | 整体行为的真实感 |
| **Cons. (Consistency)** | 与profile的一致性 |
| **Eff. (Effectiveness)** | 咨询策略有效性（用于评估counselor LLM） |
| **CDD / CPD** | 咨询偏离度 / 咨询进展度 |

### 基线方法对比
- **大型闭源模型**：GPT-5.1, DeepSeek-V3.2, Kimi-K2-thinking, GLM-4.6
- **开源小模型**：Qwen3-8B, DeepSeek-R1-8B
- **消融变体**：Qwen3-8B-SFT（仅SFT）
- **其他挑战性模拟器**：
  - **PATIENT-V**：基础profile条件生成
  - **AnnaAgent**：随机注入情绪标签
  - **Yang et al. (2025b)**：低接受度控制

---

## 3. 主要实验结果和性能指标

### 关键性能数据（RQ1: 阻抗模拟能力）

| 模型 | Precision (%) | Recall (%) | F1 (%) | RTF (%) | Fid. | Rat. | Qua. |
|------|---------------|------------|--------|---------|------|------|------|
| GPT-5.1 | 59.31 | 62.88 | 61.04 | 35.94 | 1.42 | 1.35 | 2.52 |
| DeepSeek-R1-8B | 34.67 | 46.26 | 39.64 | 45.23 | 0.98 | 1.08 | 1.72 |
| Qwen3-8B-SFT | 63.54 | 73.90 | 68.33 | 39.43 | 1.46 | 1.41 | 2.39 |
| **ResistClient** | **70.38** | **78.95** | **74.42** | **38.03** | **1.63** | **1.58** | **2.61** |

> ✅ **结论**：ResistClient在所有指标上显著优于基线，尤其在precision和fidelity上表现最佳，表明其能更准确、合理地触发阻抗。

### 挑战性行为质量对比（RQ3）

| 模拟器 | CCR (%) | Turns | Coh. | Real. | Cons. |
|--------|--------|-------|------|-------|-------|
| PATIENT-V | 87.94 | 11.24 | 0.51 | 1.87 | 1.32 |
| AnnaAgent | 78.62 | 12.65 | 0.62 | 1.95 | 1.60 |
| Yang et al. | 62.33 | 16.67 | 0.68 | 2.01 | 1.83 |
| **ResistClient** | **60.84** | **17.88** | **0.73** | **2.39** | **1.75** |

> ✅ **结论**：ResistClient实现了最强挑战性（最低CCR + 最长turns）与最高真实感（最高Real.与Coh.），优于仅靠情绪扰动或低接受度控制的方法。

### 消融实验结果（RQ2）
- **Prompt-only (Qwen3-8B)**：严重偏向合作反应，混淆矩阵显示几乎不生成阻抗；
- **SFT-only**：显著提升阻抗生成能力，减少顺从偏见；
- **Full RIMR (ResistClient)**：进一步提升各类阻抗区分度，尤其减少不同类型间的混淆（如Defensive vs. Emotional），证明MRRL对心理一致性的增益。

---

## 4. 关键结论和发现

### 主要发现
1. **现有client simulator严重低估挑战性**：主流方法因预训练顺从性偏见，难以生成真实阻抗；
2. **RIMR框架有效提升挑战保真度**：通过**RPC数据集+SFT+MRRL**，ResistClient能系统性生成多样化、情境合理的阻抗行为；
3. **动机推理增强行为连贯性**：显式建模心理机制使行为更具解释性与稳定性，避免机械重复；
4. **ResistClient可用于压力测试心理LLM**：实验显示多数心理LLM在面对阻抗时仍会**偏离治疗轨道**（高CDD）、**进展有限**（低CPD），暴露其临床鲁棒性不足。

### 方法的局限性
1. **文化局限性**：RPC基于中国心理咨询语料构建，阻抗分布（如Compliant Resistance最常见）反映中国文化偏好（和谐、尊重权威），跨文化泛化需验证；
2. **评估依赖少数专家**：人工评估由4位持证咨询师完成，虽具专业性但视角有限；
3. **仅限客户端模拟**：未涉及咨询师如何识别与应对阻抗，未来可扩展为双向对抗训练环境。

### 未来工作方向
- 将ResistClient部署为**可扩展的虚拟培训平台**，用于大规模心理咨询师训练；
- 开发具备**阻抗管理能力的counselor agent**，实现闭环对抗训练；
- 推广至**跨文化心理咨询模拟**，构建多语言、多文化的阻抗数据集；
- 探索**个性化阻抗演化建模**，模拟阻抗随疗程动态变化的过程。

> 🔍 **总体评价**：该工作首次系统性地将**Client Resistance Theory**融入LLM-based client simulation，提出兼具**理论深度**与**工程可行性**的RIMR框架，为构建更真实、更具挑战性的心理健康对话系统提供了新范式。

</details>

---

### 8. [Frugal Knowledge Graph Construction with Local LLMs: A Zero-Shot Pipeline, Self-Consistency and Wisdom of Artificial Crowds](https://arxiv.org/abs/2604.11104)

**Authors**: Pierre Jourlin (LIA)  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.11104v1  

#### Abstract
This paper presents an empirical study of a multi-model zero-shot pipeline for knowledge graph construction and exploitation, executed entirely through local inference on consumer-grade hardware. We propose a reproducible evaluation framework integrating two external benchmarks (DocRED, HotpotQA), W...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Frugal Knowledge Graph Construction with Local LLMs: A Zero-Shot Pipeline, Self-Consistency and Wisdom of Artificial Crowds

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文致力于解决**在消费级硬件上以零训练、本地推理方式构建和利用高质量知识图谱（Knowledge Graph, KG）的可行性问题**。当前主流方法依赖大规模监督训练和云端API调用，存在高成本、高能耗和幻觉（hallucination）风险。本文探索如何通过量化后的本地LLM（quantized LLMs）实现“节俭AI”（frugal AI），即低成本、低资源消耗的知识图谱构建与问答。

### 提出的新方法与创新思路
1. **SYNSYNTH 多模型零样本管道**  
   构建了一个端到端可复现的自动化 pipeline，将四个任务模块化并分配给不同的本地运行的 quantized LLM：
   - Relation Extraction（Gemma-4）
   - Text-to-Query（Qwen3-Deep）
   - Multi-hop Reasoning（Phi-4）
   - Conversational RAG（Mistral-Small）

2. **基于软匹配的关系抽取优化**
   - 引入 **relation synonyms dictionary** 和语义规则提示工程（prompt engineering），显著提升 relation matching 准确率。
   - 使用 **soft matching** 策略容忍表达变体（如同义词、子串匹配等）。

3. **自洽性（Self-Consistency）与置信度路由级联机制（Confidence-Routing Cascade）**
   - 在 multi-hop reasoning 中引入 self-consistency（多采样投票）来增强推理鲁棒性。
   - 设计了一种 **confidence-routing cascade**：当主模型内部一致性较低时，自动将问题重定向至第二模型（如 Phi-4 → GPT-OSS），结合架构多样性（architectural diversity）与随机多样性（stochastic diversity）。

4. **揭示“共识悖论”（Agreement Paradox）**
   发现：**高样本间一致性往往对应集体幻觉而非正确答案**，而中等一致性的样本反而更可能包含正确答案。这一现象呼应了群体智慧研究中的社会影响效应（wisdom of crowds）。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **成本与可持续性** | 全流程可在单张 RTX 3090 上完成，无需训练，总耗时约 5 小时，碳足迹仅 ~0.09kg CO₂eq |
| **部署模式** | 完全本地执行，不依赖云服务，保障隐私与可控性 |
| **性能表现** | 零样本下达到接近监督模型的性能水平（如 F1 达 0.70 vs DREEAM 的 0.80） |
| **可复现性** | 开源代码与完整评估框架（集成 DocRED、HotpotQA、RAGAS） |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 用途 | 规模 | 特点 |
|--------|------|-------|------|
| **DocRED** | Document-level relation extraction | 500 samples | 外部基准，含 96 种关系类型 |
| **HotpotQA** | Multi-hop reasoning | 500 samples | 至少两跳推理，标准指标为 EM 和 Token-F1 |
| **WebQuestionsSP-style synthetic data** | Text-to-Query 转换 | 200 samples | 自生成数据，目标为 Neo4j 的 Cypher 查询语言 |
| **RAGAS + 自合成数据** | 反事实问答评估 | 50 samples | 自动生成问题用于评估 faithfulness、relevance 等维度 |

> ⚠️ 注意：Text-to-Query 和 RAG 数据为 LLM 自动生成（seed 手动编写后扩展），存在潜在循环偏差（circular bias），需谨慎解读其绝对分数。

### 实验设置
- **硬件环境**：
  - GPU: NVIDIA RTX 3090 (24GB VRAM)
  - CPU: Intel Core i9-12900HK
  - 内存: 32GB DDR5
  - 系统: Ubuntu 24.04 LTS
- **推理引擎**：Ollama v0.20.0，支持原生 JSON Schema 输出约束
- **量化格式**：GGUF Q4KM，降低显存占用
- **超参数**：
  - `temperature=0.3`, `top_p=0.9`, `num_ctx=8192`
  - 所有结果报告 **95% Bootstrap 置信区间**

### 评估指标
| 任务 | 主要指标 |
|------|----------|
| Relation Extraction | Precision, Recall, F1 |
| Text-to-Query | Accuracy, Valid Cypher Rate |
| Multi-hop Reasoning | Exact Match (EM), Token-F1 |
| RAG Evaluation | Faithfulness, Relevance, Context Precision (via RAGAS) |

### 基线方法对比
| 方法 | 类型 | 来源 | 性能（F1/EM） |
|------|------|------|-------------|
| DREEAM [9] | Supervised (fine-tuned) | Cloud/Multi-GPU | F1=0.802 |
| ATLOP [8] | Supervised | Multi-GPU | F1=0.778 |
| GPT-3 (few-shot) [10] | Few-shot (API) | Cloud API | ~0.72 |
| GPT-3 (zero-shot) [10] | Zero-shot (API) | Cloud API | ~0.30 |
| ChatGPT (zero-shot) [11] | Zero-shot (API) | Cloud API | ~0.25 |
| SYNSYNTH (本工作) | Zero-shot (local) | Local RTX 3090 | F1=0.702 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总
| 任务 | 指标 | 结果（95% CI） |
|------|-----|----------------|
| Relation Extraction (DocRED) | F1 | **0.702 ± 0.04** |
| Text-to-Query (Synthetic) | Accuracy | **0.795 ± 0.06** |
| Multi-hop Reasoning (HotpotQA) | EM | **0.458 ± 0.04** |
| RAGAS Faithfulness | Mean | **0.957 ± 0.04** |

### 与基线方法的对比结果
- **Relation Extraction**：
  - 超越所有已发表的 zero-shot 方法（GPT-3: ~0.30, ChatGPT: ~0.25）
  - 接近监督模型 DREEAM（0.802），差距仅约 10 个百分点
- **Multi-hop Reasoning**：
  - 零样本 EM=0.46，低于最佳监督系统（~0.70），但在无训练前提下表现优异
- **Text-to-Query**：
  - 语法有效率（Valid Cypher）达 **100%**，得益于 constrained decoding
  - 准确率达 0.80，表明对自然语言到图查询的理解能力较强

### 消融实验结果
#### （1）Pipeline 版本演进分析（Table 13）
| 版本 | 描述 | EM / F1 提升来源 |
|------|------|----------------|
| V1 | Baseline（自由输出） | F1=0.263 |
| V2 | + Constrained Decoding (JSON Schema) | F1≈0.26 → 无明显提升 |
| V3 | + Prompt Engineering + Synonyms | F1→**0.702**（↑+0.44） |
| V4 | + QLoRA 微调（7B 模型） | EM=0.406（低于零样本 Phi-4 的 0.462） |
| V5a | + Self-Consistency (k=3) | EM→**0.482**（↑+2pt） |
| V5b | + Confidence-Routing Cascade | EM→**0.552**（↑+9pt vs zero-shot） |

> ✅ **核心发现**：性能跃迁主要来自 **prompt engineering 和 synonym soft matching**，而非模型本身或 constrained decoding。

#### （2）跨模型迁移性测试（Table 20）
将 V3 的 prompt 工程应用于其他模型（Mistral-Small、Phi-4、GPT-OSS）：
- 对 Gemma-4 提升巨大（F1 从 0.039 → 0.702）
- 对其他模型几乎无效甚至轻微下降（ΔF1 ≤ ±0.011）
> 🔍 表明：**prompt 优化效果具有高度模型特异性（model-specific interaction）**

#### （3）Self-Consistency 与 Cross-Model Oracle 分析（Table 15）
在 181 个难解问题上（T=0 时全部失败）：
| 方法 | EM 提升 |
|------|--------|
| GPT-OSS (k=5, T=0.7) | ↑+23.2 pts |
| Phi-4-Reasoning+ | ↑+17.1 pts |
| Phi-4 | ↑+14.4 pts |
| Cross-model Oracle (3×k=5) | ↑+46.4 pts |

> 📌 显示：**架构多样性（architectural diversity）远优于单一模型内的随机多样性**

---

## 4. 关键结论和发现

### 主要发现
1. **Prompt Engineering 是决定性因素**  
   在零样本 setting 下，精心设计的 prompt（含 relation 列表、禁止项、语义指导）和 synonym 匹配策略带来了超过 60 个 F1 点的提升，远超模型选择的影响。

2. **共识悖论（Agreement Paradox）的存在**  
   - 高一致性（≥80%）常指向共同错误（collective hallucination）
   - 中等一致性（40%-80%）区域才是最有价值的信息源
   - 这类现象与人类群体决策中的“社会影响力导致多样性丧失”高度相似（Moussaid et al., 2013）

3. **Confidence-Routing Cascade 实现最优性价比**
   - 通过检测主模型（Phi-4）的响应一致性，动态路由至次模型（GPT-OSS）
   - 在仅使用两个模型的情况下，达到 EM=0.552，优于八模型投票（EM=0.446）
   - 成本仅为八模型方案的 1/10，形成帕累托前沿优势

4. **本地零样本系统具备实用潜力**
   - 在消费级硬件上实现接近监督系统的性能
   - 碳足迹极低（~0.09kg CO₂eq），符合绿色AI趋势
   - 支持完全离线运行，适合隐私敏感场景

### 方法的局限性
1. **部分数据为自生成，存在循环偏差**
   - Text-to-Query 和 RAG 评估数据由 LLM 自产，可能导致性能被高估
2. **语言单一性**
   - 所有实验均在英文环境下进行，未验证多语言能力
3. **任务天花板受限于知识缺失**
   - 分析显示 51.6% 的 multi-hop 错误源于训练数据中缺乏冷门事实（knowledge gap）
4. **confidence 字段不可靠**
   - 自报 confidence 值无法区分正确与否（AUC=0.5），不能作为过滤信号
5. **prompt 优化不可迁移**
   - 当前 prompt 设计对 Gemma-4 效果显著，但难以泛化到其他架构

### 未来工作方向
1. **拓展至垂直领域**（医疗、法律）和非英语语言
2. **引入 retrieval-augmented 机制** 以补充缺失上下文，突破知识瓶颈
3. **改进 confidence routing**：
   - 引入外部校准器（Platt scaling）
   - 图谱验证模块（graph-based verification）
   - 动态阈值调整
4. **尝试 QLoRA 应用于 relation extraction**
5. **研究 prompt 在不同架构间的可迁移性**，构建通用优化模板

--- 

> 💡 **总体评价**：该论文展示了在严格资源限制下，通过精巧的工程设计（prompt + pipeline + diversity control）也能逼近监督学习性能，是“节俭AI”（frugal AI）范式的有力实践，同时提出了关于 LLM 群体行为的新洞见，具有理论与应用双重意义。

</details>

---

### 9. [OOWM: Structuring Embodied Reasoning and Planning via Object-Oriented Programmatic World Modeling](https://arxiv.org/abs/2604.09580)

**Authors**: Hongyu Chen, Liang Lin, Guangrun Wang  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.09580v1  

#### Abstract
Standard Chain-of-Thought (CoT) prompting empowers Large Language Models (LLMs) with reasoning capabilities, yet its reliance on linear natural language is inherently insufficient for effective world modeling in embodied tasks. While text offers flexibility, it fails to explicitly represent the stat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*OOWM: Structuring Embodied Reasoning and Planning via Object-Oriented Programmatic World Modeling*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于文本的 **Chain-of-Thought (CoT)** 推理在具身智能（Embodied AI）任务中存在严重缺陷：
- **表示能力不足**：自然语言是线性的，难以显式建模物理环境中的对象层次、状态空间和因果依赖。
- **缺乏可执行性**：生成的推理链多为“意识流”，无法直接转化为机器人可执行的控制策略。
- **浅层世界模型（Shallow World Models）**：无法区分对象的静态属性与动态状态，导致长期规划逻辑不一致。

这些问题限制了 LLM 在真实场景（如房间清理、烹饪等）中的可靠决策与执行能力。

---

### 🚀 提出的新方法：Object-Oriented World Modeling (OOWM)

作者提出 **OOWM** 框架，将具身推理从“自由文本生成”转变为“面向对象的程序化世界建模”。

#### 核心思想
将世界模型 $ W $ 定义为一个符号元组：
$$
W = (S, T)
$$
- **$ S $：State Abstraction（状态抽象）**
  - 使用 **UML Class Diagram** 显式定义环境中对象的类、属性、继承与聚合关系。
  - 将视觉感知映射为结构化的对象系统（例如：`Bed`, `Desk`, `Clothing` 及其 `isDirty`, `location` 属性）。
- **$ T $：Transition Logic / Control Policy（控制策略）**
  - 使用 **UML Activity Diagram** 编码动作流程，包含顺序、分支、循环等控制结构。
  - 输出是可解析、可验证、可执行的计划。

> 💡 类比软件工程：不是写一篇作文描述怎么打扫房间，而是设计一个类图 + 流程图来构建一个“清洁机器人控制系统”。

---

### 🔍 相比现有方法的优势

| 方法 | 缺陷 | OOWM 改进 |
|------|------|-----------|
| **Text-based CoT** | 线性、模糊、无状态管理 | 引入结构化状态与行为分离 |
| **Scene Graphs / Logic Graphs** | 表达力弱，难建模复杂流程（如循环） | 利用 UML 的标准化语义支持高级抽象 |
| **Graph of Thoughts (GoT), Tree of Thoughts (ToT)** | 仍是非标准图结构，缺乏统一语法 | 使用工业级标准 **UML**，具备严格语法与可视化能力 |

> ✅ OOWM 实现了：
> - 更强的 **结构保真度（Structural Fidelity）**
> - 更高的 **计划一致性（Plan Coherence）**
> - 更好的 **可执行性（Executability）**

---

## 2. 核心实验方法和设置

### 📚 数据集：**MRoom-30k**
- 新构建的大规模基准数据集，共 **30,792 张杂乱室内图像**。
- 来源：Google/Bing/Baidu/Rednote + Messy Rooms Dataset。
- 场景多样，涵盖卧室、客厅等真实家庭环境。
- 注释分为两个子集：
  - **Reasoning-Enhanced Subset (1k samples)**  
    包含完整的 $ G_{\text{state}} $(Class Diagram) 和 $ G_{\text{control}} $(Activity Diagram)，用于监督训练。
  - **Base Planning Set (29k samples)**  
    仅提供最终的 $ G_{\text{control}} $，用于大规模强化学习。

同时提供两种格式标注：
- **UML 格式**（PlantUML 序列化代码）
- **Unstructured Text 格式**（用于对比基线）

---

### ⚙️ 模型架构与输入输出

- **Backbone**: InternVL 2.5（多模态大模型，结合 InternViT-300M 视觉编码器 + InternLM 2.5 语言解码器）
- **输入**：一张杂乱房间图像 + 清洁指令
- **输出**：序列化的 PlantUML 代码，包含两个部分：
  ```xml
  <think>
  @startuml
  class Bed {
    +isMade: Boolean
    +hasLaundry: Boolean
  }
  ...
  @enduml
  </think>
  <answer>
  @startuml
  start
  :Identify messy areas;
  :Prioritize bed > desk > floor;
  if (bed is messy?) then
    :Strip sheets;
    :Make bed;
  endif
  ...
  stop
  @enduml
  </answer>
  ```

---

### 🧪 三阶段训练策略

| 阶段 | 名称 | 方法 | 目标 |
|------|------|------|------|
| **Stage 1** | OOWM Initialization via SFT | 监督微调 | 学习生成合法的 PlantUML 语法 |
| **Stage 2** | Structural Alignment via RLFT | Group Relative Policy Optimization (GRPO) | 优化 $ G_{\text{control}} $ 质量，反向提升 $ G_{\text{state}} $ |
| **Stage 3** | Scale-Up via Outcome-Based GRPO | 在 29k 数据上进行 RL 微调 | 利用稀疏奖励隐式学习高质量世界模型 |

> ✅ 关键机制：通过最终计划的成功率作为奖励信号，反向传播优化中间的状态建模过程（Latent Reward Propagation）

---

### 📊 评估指标

由于传统 ROUGE 不适用于结构化计划评估，本文采用：

#### 结构感知语义评估（Structure-Aware Semantic Evaluation）
1. **功能分区分解**：
   - 将 Activity Diagram 分解为三个模块：
     - Messy Areas Identification
     - Priority Order
     - Specific Cleaning Steps
2. **节点对齐匹配**：
   - 使用 `all-MiniLM-L12-v2` 编码每个动作节点
   - 基于余弦相似度进行贪心匹配
3. **计算指标**：
   - **Semantic Similarity (Regression)**：平均匹配得分
   - **Precision, Recall, F1 (Classification)**  
     - Threshold: sim ≥ 0.5 才视为正确
     - **Recall ≈ Task Execution Success Rate**（覆盖了多少必要步骤）

---

### 🆚 基线方法对比

| 基线方法 | 描述 |
|--------|------|
| **Text → Text (Unstructured CoT)** | 标准 VLM-R1 风格，纯文本推理与计划 |
| **Tree of Thoughts (ToT)** | 多路径搜索推理树 |
| **Graph of Thoughts (GoT)** | 图结构推理，但仍为非标准图 |
| **Hybrid Strategy (Text → OOWM)** | 文本推理 + 结构化输出（UML） |
| **OOWM 2-Stage** | 全结构化推理与计划，无 RL 优化 |
| **OOWM 3-Stage (Full Pipeline)** | 完整三阶段训练流程 |

---

## 3. 主要实验结果和性能指标

### 📈 定量结果（MRoom-30k 测试集）

| Method | Similarity | Precision | **Recall (Success Rate)** | F1 |
|-------|------------|-----------|-----------------------------|-----|
| Tree of Thoughts [31] | 0.4209 | 0.4854 | 0.4639 | 0.4695 |
| Graph of Thoughts [2] | 0.5383 | 0.5263 | 0.5579 | 0.5371 |
| Unstructured Baseline (Text→Text) | 0.5498 | **0.5489** | 0.6280 | 0.5811 |
| Hybrid Strategy (Text→OOWM) | 0.5617 | 0.5304 | 0.6536 | 0.5803 |
| OOWM 2-Stage (OOWM→OOWM) | 0.5562 | 0.5384 | 0.6438 | 0.5812 |
| **OOWM 3-Stage (Full Pipeline)** | **0.5694** | 0.5326 | **0.6744** | **0.5904** |

> ✅ **OOWM 3-Stage 在 Recall 和 F1 上全面领先**，说明其能更完整地恢复专家计划中的关键步骤。

---

### 🔬 消融实验（Ablation Studies）

#### （1）SFT 初始化是否必要？
- ❌ 若跳过 Stage 1（SFT），直接在 29k 上做 GRPO：
  - 模型完全无法收敛
  - 因为随机生成几乎不可能产出合法 PlantUML 语法
- ✅ **结论**：SFT 是必须的“结构引导启动”（Structural Bootstrapping）

#### （2）GRPO 是否优于持续 SFT？
- 继续 SFT 训练 → 所有指标在第 5 轮后饱和
- 切换到 GRPO → 所有指标继续上升
- ✅ **结论**：Outcome-based RL 能突破模仿学习瓶颈，实现真正的能力跃迁

#### （3）显式建模 State Abstraction 的价值
- OOWM 2-Stage 明显优于 Hybrid Strategy
- 证明：**先构建 $ G_{\text{state}} $ 再生成 $ G_{\text{control}} $ 更有利于逻辑一致性**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **结构化胜于非结构化**
   - 即使是最复杂的文本推理（ToT/GoT），也无法超越简单的结构化输出（Hybrid Strategy）
   - **引入 UML 形式的 $ G_{\text{control}} $ 显著提高 Recall**

2. **状态抽象至关重要**
   - 显式建模 $ S $（via Class Diagram）使模型具备持久的对象记忆，避免遗漏重要操作

3. **Outcome-based RL 可优化隐式推理结构**
   - 即使没有 $ G_{\text{state}} $ 的标注，也能通过最终计划质量反向优化内部世界模型

4. **标准化形式主义带来可靠性**
   - UML 提供了严格的语法与通用语义，使得模型输出更具解释性、可调试性和跨任务迁移潜力

---

### ⚠️ 方法的局限性

1. **依赖强大的视觉基础模型**
   - 对细小物体或遮挡严重的物品识别仍有限
   - 如 Painting 任务中因未见过某些工具而失败

2. **跨域泛化仍有挑战**
   - 在 Cooking 和 Painting 上虽表现尚可，但 Recall 提升有限
   - 表明当前框架对新对象的零样本建模能力有待加强

3. **UML 学习门槛较高**
   - 需要模型掌握较复杂的语法结构，训练成本高

---

### 🔮 未来工作方向

1. **扩展至更多具身任务**
   - 如装配家具、厨房料理、家庭护理等
2. **结合神经符号系统**
   - 将 OOWM 输出连接到实际机器人控制器（如 ROS）
3. **自动修复错误的世界模型**
   - 引入反馈机制，在执行失败后修正 $ G_{\text{state}} $ 或 $ G_{\text{control}} $
4. **轻量化与部署优化**
   - 设计更高效的 UML 编码方式，便于边缘设备运行

---

## 总结

> **OOWM 开辟了一条全新的路径：将具身智能的推理过程视为“软件系统设计”而非“语言叙述”。**

它通过 **UML Class & Activity Diagrams** 构建了兼具语义深度与执行能力的结构化世界模型，并借助 **GRPO + 三阶段训练** 实现了从模仿到自主优化的跨越。实验证明，这种 **Object-Oriented Paradigm** 在计划完整性、逻辑一致性和任务成功率方面显著优于现有方法，为下一代具身 AI 提供了坚实的设计范式。

</details>

---

### 10. [Hubble: An LLM-Driven Agentic Framework for Safe and Automated Alpha Factor Discovery](https://arxiv.org/abs/2604.09601)

**Authors**: Runze Shi, Shengyu Yan, Yuecheng Cai, Chengxi Lv  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.09601v1  

#### Abstract
Discovering predictive alpha factors in quantitative finance remains a formidable challenge due to the vast combinatorial search space and inherently low signal-to-noise ratios in financial data. Existing automated methods, particularly genetic programming, often produce complex, uninterpretable for...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Hubble: An LLM-Driven Agentic Framework for Safe and Automated Alpha Factor Discovery

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在量化金融中，**alpha factor 发现**面临两大挑战：
- **组合爆炸**：候选因子表达式的搜索空间巨大。
- **低信噪比与过拟合风险**：金融数据噪声高、非平稳性强，传统自动化方法（如遗传编程 GP）容易生成复杂且不可解释的公式，泛化能力差。

此外，直接执行 LLM 生成的代码存在严重的**计算安全风险**，如代码注入、无限循环等。

---

### 🚀 提出的新方法：Hubble 框架
Hubble 是一个基于 **LLM 驱动的智能体框架（agentic framework）**，用于安全、自动化的 alpha factor 发现。其核心创新在于将 LLM 作为“智能搜索启发式”，并引入三重确定性约束机制：

#### 主要创新点：
1. **Domain-Specific Language (DSL)**  
   - 定义了一套金融领域专用的操作符语言，包括数学（Math）、时序（TS）、横截面（CS）和逻辑（Logic）四类操作符。
   - 限制 LLM 只能使用预定义、可解释、有经济含义的基本算子，避免生成无意义或非法表达式。

2. **AST-Based Execution Sandbox（抽象语法树沙箱）**  
   - 三层验证机制确保计算安全性：
     - **Security Whitelist**：仅允许安全的 AST 节点类型（如 Call, Name），阻止 `_import_` 等危险操作。
     - **Complexity Bounds**：限制树深度（≤30）和节点总数（≤1000），防止组合爆炸。
     - **Semantic Verification**：检查函数参数数量是否匹配（arity）、变量是否合法。
   - 实现 **100% 运行时稳定性**，零崩溃。

3. **Evolutionary Feedback Loop（进化反馈机制）**
   - 将每轮表现最好的因子和错误诊断信息反馈给 LLM，引导其进行“定向突变”式迭代优化。
   - 类似于进化算法中的选择与变异，但由 LLM 扮演“智能搜索引擎”。

4. **闭环评估流水线**
   - 因子通过严格的统计指标评估：RankIC、Information Ratio（IR）、Turnover、Coverage。
   - 综合得分（composite score）用于排序和反馈。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | Hubble 的优势 |
|------|--------|----------------|
| 遗传编程（Genetic Programming, GP） | 表达式复杂、不可读、易过拟合 | 生成更简洁、可解释、符合金融直觉的因子 |
| Alpha-GPT / Chain-of-Alpha 等 LLM 方法 | 缺乏执行安全保障，可能产生运行时错误 | 引入 AST 沙箱，实现生产级稳定性 |
| 手工构造因子库（如 Kakushadze [7]） | 依赖专家经验，效率低 | 自动化 + LLM 创造力，提升探索效率 |

> ✅ **核心优势总结**：结合 LLM 的创造性与确定性系统的安全性，实现了 **可复现、可解释、稳定可靠** 的自动化因子挖掘。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **资产范围**：30 只美国股票（U.S. equities）
- **时间跨度**：752 个交易日（2022年1月 – 2024年12月）
- **数据频率**：日频 OHLCV 数据（Open, High, Low, Close, Volume）
- **数据组织形式**：MultiIndex 面板数据 $ X \in \mathbb{R}^{T \times N \times 5} $

---

### ⚙️ 实验设置
- **总轮数（Rounds）**：3 轮
- **每轮生成数量（Batch Size）**：B = 40
- **反馈机制**：每轮返回 top-5 最优因子 + 错误分布摘要给 LLM
- **评分权重（Composite Score）**：
  ```plaintext
  score = 2.0 × IR_annual + 1.0 × IC − 2.0 × turnover − 3.0 × ddrop
  ```
  （鼓励高 IR 和 IC，惩罚高换手率和数据缺失）

- **黄金因子阈值**：score > 1.0（本实验未达到）

---

### 🎯 评估指标
| 指标 | 定义 | 目的 |
|------|------|------|
| **RankIC** | 截面因子值与未来收益的 Spearman 秩相关系数 | 衡量预测能力 |
| **IR_daily / IR_annual** | 日均 IC / 年化信息比率（IR_daily × √252） | 衡量风险调整后收益 |
| **Top-K Turnover** | 前 K 名股票集合的平均变动比例 | 衡量交易成本影响 |
| **Data Coverage** | 因子输出有效值的比例 | 衡量鲁棒性（避免 NaN 或常数） |
| **Composite Score** | 加权综合得分 | 多目标优化依据 |

---

### 🔁 基线方法对比
本文未直接与其他 LLM 或 GP 方法进行端到端性能比较，而是强调以下几点差异：
- 与传统 GP 相比：Hubble 生成因子更具可解释性，且受 DSL 约束。
- 与现有 LLM 方法（如 Alpha-GPT, Chain-of-Alpha）相比：Hubble 具备完整的 **安全执行保障机制**（AST sandbox），是首个实现 **100% 计算稳定性** 的 LLM-based factor mining 框架。

> ❗ 注：作者指出当前研究重点在于框架设计与可行性验证，而非横向 benchmark。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & Table 2）

#### 整体流程效率（Table 1）
| Round | Candidates (Unique) | Evaluated | OK (Valid) | Errors | Best Score |
|-------|---------------------|---------|----------|--------|------------|
| R1    | 41                  | 76      | 71       | 5 PARSE + 5 DUP | **0.827** |
| R2    | 41                  | 60      | 60       | 21 DUP         | -0.279     |
| R3    | 40                  | 50      | 50       | 31 DUP         | 0.717      |
| **Total** | **122**           | **186** | **181**  | **62**         | **0.827** |

- **有效性**：181/186 ≈ **97.3%** 的表达式成功通过验证并被评估。
- **稳定性**：**零运行时崩溃**，验证 AST 沙箱的有效性。
- **重复率上升**：DUPLICATE 错误从 R1 的 5 上升至 R3 的 31，表明搜索空间逐渐收敛。

#### Top-5 因子表现（Table 2）
| Rank | Score | IC     | IR_annual | Hit%  | Turnover (T̄) | Cov. |
|------|-------|--------|-----------|-------|---------------|------|
| 1    | 0.827 | 0.0185 | **1.310** | 52.7% | 0.902         | 100% |
| 2    | 0.717 | 0.0185 | **1.298** | 51.7% | 0.945         | 100% |
| 3    | 0.506 | 0.0044 | 0.319     | 53.1% | 0.028         | 100% |
| 4    | 0.218 | 0.0026 | 0.187     | 53.7% | 0.039         | 100% |
| 5    | 0.161 | 0.0021 | 0.151     | 53.8% | 0.052         | 100% |

- **最高 IR_annual 达到 1.31**，远超行业公认的 1.0 门槛，显示强预测能力。
- f1 和 f2 换手率较高（>0.9），可能是捕捉短期动量信号；f3–f5 更稳定。
- 所有 top-5 因子均实现 **100% 数据覆盖率**，说明 DSL 设计有效避免退化输出。

---

### 🔍 消融实验与分析（Implicit Ablation）

虽然没有显式消融实验，但文中通过多个维度展示了各组件的作用：

| 分析维度 | 发现 | 对应机制 |
|--------|------|----------|
| **PARSE_ERROR 减少** | R1 有 5 例语法错误，R2/R3 归零 | 反馈机制帮助 LLM 学习 DSL 规则 |
| **DUPLICATE 增加** | 后续轮次重复表达式增多 | 反馈使 LLM 收敛于已有成功模式 |
| **OK Rate 提升** | 从 R1 的 93% → R2/R3 的 100% | 反馈 + 沙箱共同提升生成质量 |
| **Score 分布收敛** | R3 得分方差减小（见 Figure 3） | 探索趋于稳定，进入局部优化阶段 |

> 💡 这些观察间接证明了 **evolutionary feedback** 和 **AST sandbox** 的协同作用。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 可作为高效的因子搜索代理**：在 DSL 约束下，LLM 能够生成具有经济意义的、高性能的 alpha factor。
2. **AST 沙箱保障了系统稳定性**：实现了 **100% computational stability**，解决了 LLM 自动生成代码的安全隐患。
3. **进化反馈机制有效引导搜索方向**：通过反馈 top factors 和 error diagnostics，LLM 逐步“学习”哪些结构更优，减少无效尝试。
4. **生成因子具备实际价值**：top 因子年化 IR 超过 1.3，达到可应用于实盘的标准。

---

### ⚠️ 局限性
1. **数据规模有限**：仅测试于 30 只美股、3 年时间窗口，泛化性有待验证。
2. **探索深度不足**：所有 top 因子均为单层操作符组合，未发现深层复合结构，可能受限于搜索策略或 DSL 表达力。
3. **缺乏 out-of-sample walk-forward validation**：尚未验证因子在市场 regime change 下的稳健性。
4. **固定 LLM 后端**：未比较不同 LLM 对生成效果的影响。
5. **仅三轮迭代**：更大规模运行可能揭示不同的收敛行为。

---

### 🔮 未来工作方向
1. **加入 Walk-Forward Testing**：评估因子在不同市场周期下的鲁棒性。
2. **扩展 DSL 功能**：
   - 引入更高阶算子（如跨资产协方差项）
   - 支持非线性嵌套结构
3. **改进搜索策略**：
   - 结合 Tree-of-Thought prompting 实现多步推理
   - 引入 formal evolutionary operators（如交叉、变异）增强多样性
4. **多 LLM 协同架构**：构建 multi-agent system，分工负责生成、评估、优化。
5. **部署到真实交易环境**：进行 paper trading 或 live trading 验证。

---

## 总结

> Hubble 成功地将 **LLM 的创造力** 与 **量化金融的严谨性** 相结合，在保证 **安全性、可解释性和可复现性** 的前提下，实现了高质量 alpha factor 的自动化发现。它是迈向“AI-native quant research infrastructure”的重要一步。

</details>

---

### 11. [UniToolCall: Unifying Tool-Use Representation, Data, and Evaluation for LLM Agents](https://arxiv.org/abs/2604.11557)

**Authors**: Yijuan Liang, Xinghao Chen, Yifan Ge, Ziyi Wu, Hao Wu, Changyu Zeng, Wei Xing, Xiaoyu Shen  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.11557v1  

#### Abstract
Tool-use capability is a fundamental component of LLM agents, enabling them to interact with external systems through structured function calls. However, existing research exhibits inconsistent interaction representations, largely overlooks the structural distribution of tool-use trajectories, and r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# UniToolCall: Unifying Tool-Use Representation, Data, and Evaluation for LLM Agents — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大语言模型（LLM）在 **tool-use（工具调用）** 能力上的研究存在三大碎片化问题：
1. **表示不一致（Representation Inconsistency）**：不同数据集采用不同的格式编码函数调用、参数和观察结果，导致难以联合训练。
2. **结构建模不足（Structural Under-modeling）**：现有方法大多忽略工具调用中的执行结构多样性，如串行（serial）与并行（parallel）调用模式。
3. **评估不统一（Evaluation Mismatch）**：各基准测试使用异构的协议、工具定义和评估脚本，无法进行公平比较。

这些问题阻碍了可扩展的训练和系统性的评估。

---

### 🚀 提出的新方法与创新
作者提出 **UniToolCall**，一个统一的框架，标准化了从工具集构建、数据生成到评估的全流程：

#### 主要创新点：
- **统一的数据表示形式：QAOA 格式**
  - 将所有交互标准化为 **Query-Action-Observation-Answer (QAOA)** 序列，支持跨数据集的一致处理。
  
- **大规模、结构可控的混合训练数据集**
  - 构建了一个包含 **22,606+ 工具** 的大型工具池（Tool Pool），覆盖 13 个领域和 6 种功能类别。
  - 合成生成了 **390,060 条训练实例**，涵盖单跳/多跳、单轮/多轮、串行/并行等多种交互结构。
  - 引入 **Anchor Linkage 机制**，强制实现多轮对话中跨回合的状态依赖，提升长程推理能力。

- **统一的评估基准与细粒度指标**
  - 将 7 个公开 benchmark 统一转换为 QAOA 格式，并设计了多层次评估体系：
    - 函数调用级（Function-call level）
    - 对话轮次级（Turn level）
    - 完整对话级（Conversation level）
  - 支持 **Strict** 和 **Flexible** 两种评分策略，兼顾严格性和容错性。

---

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | UniToolCall |
|------|--------|-----------|
| 数据来源 | 多为真实 API 执行，规模小且不稳定 | 混合公共数据 + 结构化合成数据，规模大、可控性强 |
| 执行结构 | 忽视串行/并行差异 | 显式建模多种执行路径 |
| 多轮一致性 | 缺乏状态追踪机制 | 引入 Anchor Linkage 保证上下文连贯 |
| 评估方式 | 各自为政，不可比 | 统一格式 + 统一指标，支持公平对比 |

> ✅ **优势总结**：UniToolCall 实现了 **representation、data、evaluation 的三重统一**，显著提升了 tool-use 的可复现性、可扩展性和可评估性。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）训练数据 $D_{\text{train}}$（共 390,060 条）
- **公共数据集成 $D_{\text{pub}}$**（387,123 条）：
  - 整合了 10 个标准公开数据集，包括：
    - `BFCL v3`, `ACEBench`, `Seal-Tools`, `HammerBench`, `API-Bank`, `ToolAlpaca`, `ToolHop`, `APIGen`, `Toucan` 等。
  - 经过统一格式化为 QAOA 并进行质量过滤（见 Table 5）。
- **合成数据增强 $D_{\text{syn}}$**（2,937 条）：
  - 基于工具池自动生成，分为：
    - 单跳（Single-hop, 979）
    - 多跳（Multi-hop, 979）
    - 多轮（Multi-turn, 979）
  - 使用 Qwen3-32B 生成并通过 LLM 自评机制筛选高质量样本。

#### （2）评估基准 $D_{\text{test}}$（共 6,163 条）
- 包含 7 个公开 benchmark 的高质量子集，用于统一评估。

---

### ⚙️ 实验设置

#### 模型架构
- **主干模型**：Qwen3-8B（开源轻量级模型）
- **微调方式**：LoRA 微调（rank=8, α=16）
- **训练配置**：
  - 使用 DeepSpeed + LLaMA-Factory 框架
  - Batch size: 8, Max seq length: 8192
  - 学习率：1e-5，Warmup ratio: 0.03
  - 训练 1 轮

#### 推理设置
- 所有模型在 `thinking=False` 模式下评估（禁用思维链输出以提高效率）
- 商业模型使用其非推理版本（如 GPT-5.2 Instant, Gemini 3 Flash）

---

### 🎯 评估指标

定义四个核心宏平均指标：

| 指标 | 公式 | 描述 |
|------|------|------|
| **SP (Strict Precision)** | $\frac{1}{N}\sum_i \mathbb{1}(|P|=|G| \land \forall p \in P, m_n(p)=1)$ | 所有预测工具名称必须完全匹配真值 |
| **FP (Flexible Precision)** | $\frac{1}{N}\sum_i \frac{1}{|P|}\sum_{p \in P} m_n(p)$ | 正确命名的工具比例 |
| **SPA (Strict Parameter Accuracy)** | $\frac{1}{N}\sum_i \frac{1}{|P|}\sum_{p \in P} m_s(p)$ | 参数完全正确的调用占比 |
| **FPA (Flexible Parameter Accuracy)** | $\frac{1}{N}\sum_i \frac{1}{|P|}\sum_{p \in P} m_f(p)$ | 参数语义相似（ROUGE-L > 0.7）即视为正确 |

> ✅ 评估场景：Hybrid-20 设置（候选工具列表含 20 个：1 个真值 + 19 个干扰项），模拟真实检索环境。

---

### 🆚 基线方法对比
| 类型 | 模型列表 |
|------|---------|
| **商业闭源模型** | GPT-5.2 Instant, Gemini 3 Flash Preview, Claude 4.6 Sonnet |
| **开源大模型** | Qwen3-32B, DeepSeek-V3.2 |
| **其他开源模型** | Kimi-K2-Instruct |
| **基线对照** | Vanilla Qwen3-8B（未微调） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Hybrid-20 设置下）

| 模型 | SH SP (%) | MH SP (%) | ST SP (%) | MT FP (%) | SH SPA (%) | ST FPA (%) |
|------|----------|----------|----------|----------|------------|-------------|
| **UniToolCall (Ours)** | **92.9** | **80.7** | **93.0** | 37.8 | **39.4** | **52.8** |
| Qwen3-32B | 72.6 | 64.0 | 72.7 | 27.1 | 38.2 | 43.8 |
| Claude 4.6 Sonnet | 58.8 | 83.3 | 62.1 | 30.3 | 34.8 | 43.1 |
| GPT-5.2 Instant | 50.9 | 39.0 | 50.5 | 23.2 | 26.3 | 39.2 |
| Vanilla Qwen3-8B | 66.9 | 22.7 | 63.3 | 25.8 | 22.1 | 19.8 |

> ✅ **亮点结果**：
- 在 **单跳 Strict Precision** 上达到 **92.9%**，远超所有基线。
- 在 **单轮 Strict Precision** 上达到 **93.0%**，是目前 state-of-the-art 表现。
- 即使在干扰严重的 Hybrid-20 设置下，仍显著优于 GPT、Gemini、Claude 等商业模型。

---

### 🔍 与 GT Setting 的对比
- 在 **Ground Truth（仅保留目标工具）** 设置下，UniToolCall 接近甚至超越 vanilla 模型的表现。
- 表明其提升不仅来自抗干扰能力，更源于对工具选择和参数生成的内在能力增强。

---

### 🔬 消融实验结果（Ablation Studies）

#### Ablation I：公共数据 vs 合成数据
- 公共数据天然具有广域覆盖，但在 **串行任务上表现弱**（因缺乏深层依赖）。
- 合成数据通过控制结构（如串行调用）显著提升复杂逻辑下的性能。
- 在高串行比例任务中，**Synthetic Mixed** 明显优于 Public Mixed。

#### Ablation II：是否混合多种结构
- 单一结构训练（如纯 multi-hop）在特定任务上有优势，但泛化差。
- **混合结构训练（Mixed）** 提供正向知识迁移，在各类任务中均取得最佳平衡。

#### Ablation III：Anchor Linkage 机制有效性
- 移除该机制后，多轮对话的 **state continuity** 显著下降（见雷达图 Figure 6）。
- 验证了其在维持跨轮状态引用方面的必要性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **结构化合成数据至关重要**：显式建模串行/并行、单跳/多跳等结构能有效提升模型对复杂工具流的理解。
2. **统一表示带来公平评估**：QAOA 格式使得跨数据集、跨任务的 tool-use 能力评估成为可能。
3. **轻量模型也能超越商用大模型**：基于 Qwen3-8B 的 UniToolCall 在多个指标上超过 GPT、Gemini、Claude，证明高质量数据的价值大于单纯扩大模型规模。
4. **多轮一致性需显式约束**：Anchor Linkage 机制有效解决了多轮对话中断层问题。

---

### ⚠️ 局限性
1. **上下文长度限制**：最大支持 8192 tokens，难以处理极长工具输出或超长交互。
2. **未探索更大 backbone**：仅在 Qwen3-8B 上验证，未系统研究在 30B+ 或 70B 模型上的 scaling behavior。
3. **潜在评估偏差**：数据清洗和格式转换可能导致原始任务特征丢失。

---

### 🔮 未来工作方向
1. 扩展至更长 horizon 的 agent interaction。
2. 在真实环境中进行 live tool execution 测试。
3. 进一步优化合成数据的质量与多样性。
4. 探索在更大规模模型上的迁移效果。

---

> 💡 **一句话总结**：  
> **UniToolCall 通过“统一表示 + 结构化合成数据 + 统一评估”，实现了 LLM agent 工具调用能力的全面升级，在轻量模型上达到了 SOTA 性能，为 tool learning 提供了一个可复现、可扩展的新范式。**

</details>

---

### 12. [Tracing the Roots: A Multi-Agent Framework for Uncovering Data Lineage in Post-Training LLMs](https://arxiv.org/abs/2604.10480)

**Authors**: Yu Li, Xiaoran Shang, Qizhi Pei, Yun Zhu, Xin Gao, Honglin Lin, Zhanping Zhong, Zhuoshi Pan, Zheng Liu, Xiaoyang Wang, Conghui He, Dahua Lin, Feng Zhao, Lijun Wu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.10480v1  

#### Abstract
Post-training data plays a pivotal role in shaping the capabilities of Large Language Models (LLMs), yet datasets are often treated as isolated artifacts, overlooking the systemic connections that underlie their evolution. To disentangle these complex relationships, we introduce the concept of \text...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Tracing the Roots: A Multi-Agent Framework for Uncovering Data Lineage in Post-Training LLMs》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在大型语言模型（LLM）的 **post-training** 阶段，尽管高质量数据至关重要，但社区普遍缺乏对数据来源（provenance）的系统追踪机制。这导致两个严重风险：
- **结构性冗余（Structural Redundancy）**：多个看似独立的数据集可能隐式继承自相同的上游源，造成语义同质化，削弱数据规模增长的实际价值。
- **基准污染传播（Benchmark Contamination Propagation）**：测试集样本被无意中包含在训练数据中，并通过衍生链条向下传播，破坏模型评估的可信度。

现有方法依赖于基于样本级别的去重或语义匹配（如 N-gram、embedding 检索），效率低且难以追溯污染路径。

### 提出了什么新方法或新思路
本文提出以下核心创新：

- **引入“数据谱系”（Data Lineage）概念**  
  将 post-training 数据集视为一个有向图 $ G = (V, E) $，其中节点 $ V $ 表示数据集，边 $ E $ 表示继承关系（如 CoT distillation、question reformulation 等）。该图揭示了数据演化的拓扑结构。

- **构建多智能体协同框架（Multi-Agent Collaborative Framework）**  
  设计了一个自动化系统，通过多个 LLM Agent 协作完成以下任务：
  - **Sourcing Agent**：从 HuggingFace README 中提取外部资源链接。
  - **Extracting Agent**：抓取并摘要论文、博客、GitHub 内容。
  - **Tracing Agent**：推理出具体的源-目标依赖关系及衍生方式。
  - **Aggregation Agent**：标准化名称、消歧、验证时间顺序与置信度。

- **提出“溯源采样”策略（Provenance-based Sampling）**  
  在构建新数据集时，优先从谱系图中的根节点（leaf nodes, $ d=0 $）采样，以最大化指令级多样性，避免因重复使用中间产物而导致的隐性冗余。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **分析粒度** | 样本级（Sample-level） | 谱系级（Topological-level） |
| **效率** | 高计算成本（需扫描百万级样本） | 高效（仅比较祖先路径） |
| **鲁棒性** | 易受语义漂移影响（rewrite 后无法识别） | 对改写、扩展等操作鲁棒 |
| **可解释性** | 黑箱匹配 | 可追溯污染/冗余源头 |
| **生态洞察力** | 局部视图 | 揭示全局演化模式 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **种子数据集（Seed Datasets）**：共选取 **83 个高影响力文本数据集**，覆盖四个领域：
  - General（通用）
  - Math（数学）
  - Code（代码）
  - Science（科学）
- 最终构建的谱系图包含 **430 个唯一数据集节点** 和 **971 条继承边**。

### 实验设置和评估指标
#### （1）谱系重建设置
- **工具链**：LangChain + GPT-5.1 / Gemini-2.5-flash
- **流程**：
  1. 候选验证（时间 ≥ 2020，可用性检查）
  2. 多源信息检索（HF README、arXiv、GitHub、Blog）
  3. 语义源推断（Tracing Agents 并行抽取 `<Source, Relation, Confidence, Evidence>`）
  4. 聚合与递归扩展（DFS 遍历，终止于无祖先或发布时间 < 2020）

#### （2）多样性评估指标
用于评估提出的 **lineage-aware dataset** 性能：
- **Vendi Score**：衡量嵌入空间中有效独立簇的数量，越高表示多样性越强。
- **Centroid Distance**：样本到全局中心的平均余弦距离补值，越高表示分布越分散。

#### （3）污染与冗余检测
- **Redundancy Rate**：基于 `(instruction, input, output)` 三元组精确匹配计算重复率。
- **Contamination Ratio**：下游数据集中与基准测试集完全匹配的比例。

### 基线方法对比
- **多样性对比基线**：
  - OpenHermes-2.5
  - Tulu-3-sft-mixture
  - MegaScience
  - OpenThoughts3
  - OmniThought-0528
  - Hercules v1
- 所有基线均为广泛使用的高质量 post-training 数据集。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）谱系图统计特征（Table 1）
| Domain | Nodes | Avg Depth | In-Deg. | Out-Deg. | Leaf % |
|-------|-------|-----------|--------|----------|--------|
| Math | 99 | **2.92** | 3.30 | 1.54 | 38.38% |
| Code | 98 | 2.12 | 3.78 | 1.36 | 43.88% |
| General | 285 | 1.05 | 2.51 | 1.29 | **68.42%** |
| Science | 44 | 2.82 | 3.98 | 1.25 | 47.73% |

> 发现：**数学领域呈现深度递归精炼（deep refinement）**，而**通用领域为广度积累（broad accumulation）**。

#### （2）结构性冗余检测（Table 3）
| Dataset | Redundancy Rate (%) |
|--------|---------------------|
| open-instruct-v1 | **46.48%** |
| opc-sft-stage2 | 27.96% |
| codeforces-cots | 23.12% |
| Fast-Math-R1-SFT | 5.30% |

> 示例：`open-instruct-v1` 包含其自身的子集 `self_instruct`，导致近半数数据冗余。

#### （3）基准污染传播（Figure 6 & Table 10）
| Benchmark | Dataset | Contamination Ratio |
|----------|--------|----------------------|
| Omni-MATH | DeepScaleR-Preview-Dataset | **79.48%** |
| Omni-MATH | Big-Math-RL-Verified | 57.97% |
| Omni-MATH | Caco-1.3M | **37.95%**（间接继承） |
| TruthfulQA | UltraFeedback | **99.27%** |
| SciBench | Open-Platypus | 76.23% |

> 说明污染可通过谱系链路自动追踪，例如 Caco-1.3M 虽未直接引用 Omni-MATH，但仍继承了 37.95% 的样本。

#### （4）多样性对比结果（Table 4）
| Dataset | Size | Vendi Score ↑ | Centroid Dist. ↑ |
|--------|------|---------------|------------------|
| OpenHermes-2.5 | 615K | 437.76 | 0.6271 |
| MegaScience | 1.2M | 373.78 | 0.6150 |
| OpenThoughts3 | 1.2M | 133.26 | 0.4970 |
| **Ours (Provenance-based)** | **570K** | **452.44** | **0.6385** |

> 结论：**即使数据量仅为一半，本文方法在两项多样性指标上均超越所有更大规模的基线**。

### 消融实验结果（隐含分析）
虽然未明确列出消融表，但文中强调：
- 仅使用 **root nodes ($d=0$)** 构建数据集即可实现最优多样性，证明了“溯源采样”的有效性。
- 若进一步加入内部节点（如经过 CoT 改写的版本），仍有提升空间（high ceiling）。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Post-training 数据并非孤立存在，而是构成复杂的演化网络**，具有明显的领域特异性结构：
   - 数学：垂直深化（deep refinement），依赖少数核心锚点（如 `hendrycks_math`, `gsm8k`）进行多代迭代。
   - 通用：水平扩展（horizontal aggregation），倾向于聚合大量浅层来源。
   - 代码：作为连接通用与数学的桥梁，融合两者输入。
   - 科学：极度稀缺原生数据，高度依赖其他领域资源。

2. **结构性冗余普遍存在**，部分热门数据集冗余率超过 40%，严重稀释实际数据价值。

3. **基准污染广泛传播**，即使未显式引入 benchmark，也可能通过间接谱系路径继承污染样本。

4. **基于谱系的分析是高效且稳健的替代范式**，相比样本级方法更适合大规模生态系统治理。

5. **溯源采样显著提升数据多样性**，验证了将谱系意识融入数据构建的价值。

### 方法的局限性
1. **LLM 幻觉风险**：尽管采用 confidence-aware verification，低置信度提取仍需人工审核。
2. **文档透明度依赖**：若创建者未记录上游来源，则无法恢复真实依赖关系。
3. **格式限制**：目前主要处理文本模态，未涵盖多模态数据。

### 未来工作方向
- 构建开放的 **Data Lineage Registry**，推动社区共享谱系信息。
- 开发基于谱系的 **自动去重与防污染工具链**。
- 探索如何利用谱系结构指导 **multi-domain curriculum learning**。
- 将谱系分析应用于 **model editing** 与 **capability attribution**。

--- 

> ✅ **一句话总结**：本文首次系统性地将“数据谱系”概念引入 LLM 生态，提出一个多智能体框架自动重建 post-training 数据的演化图谱，并揭示了结构性冗余与基准污染的深层机制；同时证明，基于谱系的溯源采样能以更小规模构建出更高多样性的训练数据，为数据工程提供了新的顶层设计视角。

</details>

---

### 13. [Introspective Diffusion Language Models](https://arxiv.org/abs/2604.11035)

**Authors**: Yifan Yu, Yuqing Jian, Junxiong Wang, Zhongzhu Zhou, Donglin Zhuang, Xinyu Fang, Sri Yanamandra, Xiaoxia Wu, Qingyang Wu, Shuaiwen Leon Song, Tri Dao, Ben Athiwaratkun, James Zou, Fan Lai, Chenfeng Xu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.11035v1  

#### Abstract
Diffusion language models promise parallel generation, yet still lag behind autoregressive (AR) models in quality. We stem this gap to a failure of introspective consistency: AR models agree with their own generations, while DLMs often do not. We define the introspective acceptance rate, which measu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Introspective Diffusion Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

**Diffusion Language Models (DLMs)** 虽然理论上支持并行生成，但在实际应用中仍显著落后于 **Autoregressive (AR) 模型**，主要体现在两个方面：

- **质量差距**：DLMs 在推理、数学、代码等复杂任务上表现远不如同规模 AR 模型。
- **效率瓶颈**：尽管 DLMs 支持多步并行解码，但由于计算开销高、系统不兼容等问题，难以在真实部署中实现吞吐优势。

作者指出，根本原因在于 DLMs 缺乏 **introspective consistency（内省一致性）**：即模型无法“认同”自己生成的内容。相比之下，AR 模型通过因果掩码（causal masking）和 logit shifting，在训练过程中隐式地实现了生成与验证的一致性。

---

### 提出了什么新方法或新思路

本文提出 **Introspective Diffusion Language Model (I-DLM)**，其核心思想是：

> **从 AR 模型出发，保留其内在一致性机制，并将其迁移到扩散范式中，从而构建高质量且高效的并行生成模型。**

#### 主要创新点：

- **Introspective-Consistency Training（内省一致性训练）**
  - 使用 **全掩码训练目标**（all-masked objective），确保每个位置都参与监督信号。
  - 引入 **因果注意力 + logit shift**，使干净区域（clean positions）输出因果锚定分布 $p$（用于验证），掩码区域输出生成分布 $q$（用于解码）。
  - 设计 **自平衡损失函数**（auto-balanced loss），动态调整生成路径与验证路径的梯度强度，避免一方主导。

- **Introspective Strided Decoding (ISD)**
  - 单次前向传播同时完成：
    - **生成新 token（decode）**
    - **验证已生成 token（introspect）**
  - 利用 $p/q$ 接受准则进行采样控制，保证输出分布严格匹配 AR 分布。
  - 支持自适应步长（adaptive stride），简单 token 并行生成，困难 token 回退到 AR 模式。

- **AR 兼容的 Serving Stack**
  - 构建基于 **SGLang** 的推理引擎，直接继承 AR 系统优化（如 paged KV cache、continuous batching）。
  - 提出 **stationary-batch scheduler** 和 **kernel fusion** 技术，减少 CPU-GPU 同步开销。
  - 支持 **lossless 模式**：通过 gated LoRA 机制，实现 bit-for-bit 与原 AR 模型一致的输出。

---

### 相比现有方法的优势

| 维度 | I-DLM | 传统 DLMs（如 SDAR、LLaDA） |
|------|-------|-----------------------------|
| **质量** | 匹配甚至超越同规模 AR 模型 | 显著低于 AR 模型 |
| **效率** | 高并发下吞吐提升 3–4× | 受限于 KV commit、多步迭代等开销 |
| **系统兼容性** | 完全兼容 AR serving 栈 | 需定制化推理流程 |
| **训练成本** | 仅需 4.5B token 微调 | 如 SDAR 需 54B token |

---

## 2. 核心实验方法和设置

### 使用的数据集

共评估 **15 个基准**，涵盖四大领域：

| 类别 | 数据集 |
|------|--------|
| **知识与推理** | ARC-C, MMLU, MMLU-Pro, GPQA-D, GPQA |
| **数学推理** | GSM8K, MATH-500, MathBench, AIME-24, AIME-25 |
| **代码生成** | HumanEval, MBPP, LiveCodeBench-v6 (LCB-v6) |
| **指令遵循** | IFEval |

所有任务均启用 `thinking` 模式（允许模型进行链式思考）。

---

### 实验设置和评估指标

#### 模型配置
- **主干模型**：基于 Qwen3-8B 和 Qwen3-32B 进行转换。
- **训练数据量**：4.5B tokens。
- **训练方式**：全参数微调 + LoRA（lossless 模式）。
- **解码策略**：ISD (N=3~4)，temperature=1.0, top-k=50, top-p=0.95。

#### 评估指标
- **质量指标**：
  - 准确率（Accuracy）、pass@1（代码）、\boxed{} 提取（数学）
- **效率指标**：
  - **Per-request TPS**（每请求 token/s，反映延迟）
  - **Server-level TPS**（服务器总吞吐，反映并发能力）
- **一致性指标**：
  - **Introspective Acceptance Rate (α)**：衡量模型是否“认同”自己的生成

#### 基线方法对比
- **DLM 类**：
  - LLaDA-2.1-mini (16B), LLaDA-2.0/2.1-flash (100B)
  - SDAR (8B), NBDiff (7B), WeDLM (8B), TiDAR (8B), DREAM (7B), Fast-dLLM (7B)
  - Mercury Coder Small, Gemini Diffusion
- **Speculative Decoding 类**：
  - EAGLE-3（基于辅助 draft model 的 AR 加速）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 模型 | AIME-24 | AIME-25 | LiveCodeBench-v6 | HumanEval | MATH-500 |
|------|---------|---------|------------------|-----------|----------|
| **I-DLM-8B** | **69.6** | **60.8** | **45.7** | **93.3** | **96.8** |
| Qwen3-8B (AR) | 73.1 | 65.4 | 50.3 | 95.1 | 95.8 |
| LLaDA-2.1-mini (16B) | 43.3 | 43.3 | 30.4 | 86.0 | 85.0 |
| SDAR (8B) | 10.0 | 10.0 | 16.6 | 78.7 | 78.6 |

> ✅ I-DLM-8B 在多个任务上接近甚至超过原 AR 模型，远超所有现有 DLM。

---

### 与基线方法的对比结果

#### 质量方面
- **全面超越同规模 DLMs**：
  - AIME-24 上比 LLaDA-2.1-mini 提升 **+26.3 pts**
  - LiveCodeBench-v6 提升 **+15.3 pts**
  - HumanEval 提升 **+7.3 pts**
- **首次达到 AR 水平**：
  - I-DLM-8B 与 Qwen3-8B 表现几乎一致（平均差值 <1 pt）
  - 在 MATH-500 上反而略优（96.8 vs 95.8）

#### 效率方面（Concurrency=32）
| 模型 | Throughput (tok/s) | 相对提升 |
|------|--------------------|----------|
| **I-DLM-8B** | **~125** | — |
| LLaDA-2.1-mini | ~35 | **3.6× 更低** |
| SDAR | ~28 | **4.5× 更低** |
| EAGLE-3 | ~70 | **1.8× 更低** |

> 🔥 I-DLM 在高并发下实现 **2.9–4.1× 吞吐提升**，且优于 speculative decoding 方法。

#### 内省一致性（α）
| 模型 | Introspective Acceptance Rate (α) |
|------|------------------------------------|
| **I-DLM-8B** | **0.984** |
| Qwen3-8B (AR) | 1.0 |
| SDAR (8B) | 0.699 |
| LLaDA 2.0-flash | 0.568 |

> 💡 α 指标揭示了质量差距的本质：I-DLM 成功闭合了生成与验证之间的“认知鸿沟”。

---

### 消融实验结果

#### （1）训练设计消融（Figure 6a）
移除任一组件均导致严重性能下降：
- 移除 causal attention + logit shift → HumanEval 下降 **32.4 pts**（92.7 → 60.3）
- 数学任务（MathBench）下降 **17.5 pts**
- 表明 **introspective consistency 是长程推理的关键**

#### （2）系统优化消融（Figure 6b）
在 C=32 下累计增益达 **2.1–2.5×**：
- **CUDA graph capture**：+42–76%
- **Stationary-batch decode loop**：+11–21%
- **Argmax proposals**：+11–15%
- **Paged-only attention**：+10–14%

#### （3）步长（Stride）影响（Table 3）
| Stride N | TPF | MATH-500 | MBPP |
|---------|-----|----------|------|
| 2 | 1.80 | 96.8 | 93.4 |
| 3 | 2.48 | 95.8 | 92.8 |
| 4 | 2.96 | 96.8 | 92.2 |
| 8 | 4.01 | 94.6 | 88.3 |

> ⚖️ 展示了良好的 **parallelism-quality trade-off**：即使 N=8，精度仍保持高位。

#### （4）松弛接受阈值（Relaxed Acceptance）
| τ | HumanEval | TPF |
|---|-----------|-----|
| 0 (strict) | 93.3 | 2.63 |
| 1.0 | 91.2 | 2.73 |

> 📈 小幅牺牲质量即可进一步提升 TPF，说明提案质量本身已高度对齐。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Introspective consistency 是 AR 模型强大的本质原因**，而这是大多数 DLMs 所缺失的。
2. ✅ **I-DLM 是首个在质量上匹配同规模 AR 模型的 DLM**，打破了“并行必牺牲质量”的固有印象。
3. ✅ **单模型 + 单前向 pass 实现生成与验证统一**，无需额外 draft model 或复杂控制逻辑。
4. ✅ **系统级优化至关重要**：I-DLM 不仅算法先进，更实现了与主流 AR serving 栈的无缝集成。
5. ✅ **高并发场景下的吞吐优势显著**：相比现有 DLMs 提升 **3–4×**，具备实际部署价值。

---

### 方法的局限性

- **依赖高质量 AR 预训练模型**：目前主要用于将 AR 模型转化为 DLM，尚未验证从零训练的效果。
- **训练数据需求仍较高**：虽仅需 4.5B token，但仍需大量推理类数据以维持一致性。
- **极端长序列生成未充分测试**：当前最大长度为 4096，超长上下文场景有待验证。
- **硬件依赖较强**：最佳性能依赖 H100/B200 及 CUDA graph 等高级特性。

---

### 未来工作方向

- **扩展至更大模型与多模态**：探索 I-DLM 在 MoE、vision-language 模型中的应用。
- **降低训练成本**：研究更轻量化的 introspective-consistency 微调策略。
- **动态自适应 stride 控制**：根据输入难度自动调节并行粒度。
- **开放生态建设**：作者承诺开源模型与系统，推动社区共建 introspective AI 范式。

---

> 🔚 **总结一句话**：  
> I-DLM 通过“向 AR 学习”的哲学，将 **introspective consistency** 注入 DLM，首次实现了 **高质量 + 高效率 + 高兼容性** 的三位一体，为下一代并行语言模型提供了全新范式。

</details>

---

### 14. [MADQRL: Distributed Quantum Reinforcement Learning Framework for Multi-Agent Environments](https://arxiv.org/abs/2604.11131)

**Authors**: Abhishek Sawaika, Samuel Yen-Chi Chen, Udaya Parampalli, Rajkumar Buyya  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.11131v1  

#### Abstract
Reinforcement learning (RL) is one of the most practical ways to learn from real-life use-cases. Motivated from the cognitive methods used by humans makes it a widely acceptable strategy in the field of artificial intelligence. Most of the environments used for RL are often high-dimensional, and tra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MADQRL: Distributed Quantum Reinforcement Learning Framework for Multi-Agent Environments 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Quantum Reinforcement Learning (QRL)** 在面对高维、复杂的 **multi-agent environments (MAE)** 时面临严重挑战，主要原因包括：
- 现有量子硬件受限于噪声、相干时间短和规模小（NISQ 设备限制）；
- 多智能体联合训练对量子资源需求极高，难以在单台设备上实现；
- 传统集中式训练策略（如 CTCE/CTDE）在量子场景下扩展性差。

### 提出了什么新方法或新思路
作者提出 **MADQRL**（Multi-Agent Distributed Quantum Reinforcement Learning），一种面向多智能体环境的分布式量子强化学习框架，其核心思想是：
- 属于 **DTDE**（Distributed Training, Decentralized Execution）范式；
- 每个 agent 使用独立的 **hybrid quantum-classical model** 进行本地训练；
- 联合策略通过局部策略乘积近似：  
  $$
  \pi(\mathbf{a}|\mathbf{s}) \approx \prod_i \pi_i(a_i|s_i)
  $$
- 利用 **Variational Quantum Circuit (VQC)** 构建 **Quantum Neural Network (QNN)** 来表示策略函数。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可扩展性** | 分布式架构有效分摊计算负载，适配当前 NISQ 设备能力 |
| **通信开销低** | 不依赖全局状态共享，适用于 observation space disjoint 的场景 |
| **参数效率高** | QNN 模型仅需约一半可训练参数即能达到更优性能 |
| **灵活性强** | 可集成任意经典 DRL 算法（文中使用 PPO），易于迁移 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Cooperative Pong** 环境（来自 PettingZoo 库）
  - 2 个 agent 协作防止球出界；
  - 动作为离散集合：{-1, 0, 1}（左移、不动、右移）；
  - 状态空间为 RGB 图像 `(560 × 960 × 3)`，实验中缩放至 `64 × 64`；
  - 每个 agent 仅观察屏幕的一半（observation space disjoint）；
  - 奖励机制基于球持续在场时间。

### 实验设置
- **算法**：采用 **PPO**（Proximal Policy Optimization）进行策略更新；
- **训练平台**：使用 **Ray framework** 实现分布式训练；
- **训练轮数**：约 15K iterations；
- **超参数**：
  - Batch size: 512
  - Learning rate α: 1e-4
  - Discount factor γ: 0.95
  - Clip range ε: 0.3
  - KL coefficient: 0.2
  - Entropy coefficient: 0.5

### 评估指标
| 指标 | 描述 |
|------|------|
| **Test Episodes** | 测试阶段平均 episode 长度（越长越好） |
| **Total Runtime (s)** | 总训练耗时（秒） |
| **Trainable Weights** | 可训练参数数量 |
| **Sampling Saturation (%)** | 达到性能饱和所需的迭代比例（反映收敛速度） |
| **Mean Episodic Reward** | 平均每回合奖励值 |

### 基线方法对比
比较三种训练策略：
- **Joint**（CTCE）：联合训练单一全局策略
- **Shared**（CTDE）：共享策略但去中心化执行
- **Independent**（DTDE）：本文所提方式，各 agent 独立训练

模型类型分为两类：
- **Classical Model**：CNN + RELU 架构
- **Quantum Model**：QNN with VQC（13 qubits, 9 layers）

此外还进行了消融实验，比较不同 entanglement 结构：
- **Basic Entangling (BE)**
- **Strongly Entangling (SE)**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table I 和 Figure 4）

| Model-Distribution | Test Episodes | Trainable Weights | Sampling Saturation (%) | Total Runtime (s) |
|--------------------|---------------|-------------------|--------------------------|-------------------|
| Classical-Shared   | 193           | 40,074            | 40.0                     | 377               |
| Classical-Joint    | 254           | 20,560            | 35.8                     | 291               |
| Classical-Independent | 264        | 18,752            | 45.0                     | 391               |
| Quantum-Shared     | 176           | 20,140            | 37.9                     | 2,753             |
| Quantum-Joint SE   | 259           | 10,889            | 37.8                     | 1,529             |
| Quantum-Joint BE   | 245           | 10,893            | 37.5                     | 1,795             |
| **Quantum-Independent** | **288** | **10,691**        | **48.0**                 | **2,835**         |

> 注：所有量子模型均在模拟器上运行（非真实量子硬件）

### 与基线方法的对比结果
- **性能提升**：
  - **相比其他分布策略**：Quantum-Independent 比 Joint 和 Shared 提升约 **10%** 的测试 episodes；
  - **相比经典模型**：在相同任务下，QNN 表示的策略比 CNN 提升约 **5%** 的平均 episodic reward；
- **参数效率显著更高**：
  - Quantum 模型平均参数量仅为 classical 的 **~50%**，却实现了更优性能；
- **独立训练最优**：
  - 在 disjoint observation 场景下，“Independent” 策略表现最佳，尤其适合 MADQRL；
  - “Shared” 策略因引入冗余信息反而导致性能下降（confusion effect）；

### 消融实验结果
- **Entanglement 影响**：
  - 使用 **Strongly Entangling (SE)** 的量子电路优于 Basic Entangling (BE)；
  - SE 版本具有更高的 reward、更快的收敛速度和更低 runtime（相对 BE）；
  - 原因：更强纠缠补偿了因输入压缩造成的特征损失，增强表达能力（expressivity）；

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **DTDE 是当前 NISQ 时代 MA-QRL 的可行路径**：将复杂多智能体任务分解为独立子任务，大幅降低量子资源压力；
2. ✅ **QNN 具备高参数效率和强表达能力**：尽管运行在模拟器上较慢，但以更少参数实现更优策略；
3. ✅ **disjoint observation space 是 MADQRL 成功的关键前提**：允许策略解耦，使 $\pi \approx \prod \pi_i$ 成立；
4. ✅ **Strong entanglement 提升模型鲁棒性和学习效率**：有助于克服低维编码带来的信息瓶颈。

### 方法的局限性
- ⚠️ **依赖模拟器，尚未部署于真实量子硬件**：当前实验基于 classical simulation of VQC，存在指数级计算开销；
- ⚠️ **仅适用于 low-interference 协作场景**：若 agent 间高度依赖对方动作或状态，则独立训练失效；
- ⚠️ **图像预处理仍由经典网络完成**：未实现端到端量子化（如量子编码尚不成熟）；
- ⚠️ **训练时间较长**：由于量子模拟成本高，总 runtime 显著高于 classical 模型。

### 未来工作方向
- 🔮 将 MADQRL 部署于真实量子处理器（如 IBM Quantum 或 Rigetti）；
- 🔮 扩展至 **continuous action/state spaces** 和 **competitive environments**（COMP-MAE）；
- 🔮 探索 **quantum communication protocols** 支持有限信息交换的 semi-distributed 架构；
- 🔮 结合 **PO-MDP** 框架处理部分可观测环境；
- 🔮 引入 **federated learning + quantum** 实现隐私保护下的分布式训练；
- 🔮 开发轻量化量子编码方案，减少模拟开销并提升训练效率。

--- 

> 📌 **一句话总结**：  
> MADQRL 提出了一种实用且高效的分布式量子强化学习框架，在 disjoint observation 的协作环境中，利用独立训练的 QNN 实现了比经典模型高约 5–10% 的性能增益，同时参数量减半，为 NISQ 时代的 MA-QRL 提供了可扩展的新范式。

</details>

---

### 15. [Vestibular reservoir computing](https://arxiv.org/abs/2604.09943)

**Authors**: Smita Deb, Shirin Panahi, Mulugeta Haile, Ying-Cheng Lai  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.09943v1  

#### Abstract
Reservoir computing (RC) is a computational framework known for its training efficiency, making it ideal for physical hardware implementations. However, realizing the complex interconnectivity of traditional reservoirs in physical systems remains a significant challenge. This paper proposes a physic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Vestibular reservoir computing》核心总结

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
传统 **reservoir computing (RC)** 虽然在时间序列预测等任务中表现出色，但在物理硬件实现时面临两大挑战：
- **复杂的互连拓扑**：全耦合（fully coupled）网络需要大量节点间连接，难以在物理系统中高效构建；
- **高调参复杂度**：耦合参数需精细校准，限制了可扩展性和鲁棒性。

本文旨在解决如何在**降低硬件复杂度**的前提下，仍能保持高性能的物理 RC 实现。

---

### 🚀 提出的新方法与新思路
提出了一种受生物启发的新型物理 RC 架构 —— **Vestibular Reservoir Computing**（前庭储层计算），其核心创新包括：

- **生物启发架构设计**：灵感来源于人类内耳中的**vestibular system**（前庭系统），该系统天然具备感知运动、维持平衡的能力，是高效的生物传感与信息处理系统。
- **解耦拓扑（uncoupled topology）**：所有 reservoir 节点独立运行，无相互连接（即 $ A $ 为对角矩阵），极大简化了物理实现。
- **数学建模抽象**：将前庭系统的机械动力学（semicircular canals 和 otolith organs）与神经响应（hair cells 的 FHN 模型）结合，构建了一个可计算的非线性动力学模型作为 reservoir。

---

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **硬件可行性** | 无需复杂互联，适合电子、光子、MEMS 等物理平台部署 |
| **训练效率** | 仅优化 readout 权重，保留 RC 的快速训练特性 |
| **性能表现** | 在多个混沌系统预测任务中，**uncoupled 架构性能媲美甚至优于 coupled 架构** |
| **理论支撑** | 首次从 memory capacity 角度证明：当 eigenvalue spectrum 一致时，uncoupled 与 coupled 线性 reservoir 具有等效记忆能力 |

> 💡 特别指出：传统观点认为 recurrent connectivity 是实现“echo state property”和 memory 的必要条件，而本工作挑战这一认知，展示了**无耦合也能实现强记忆与预测能力**。

---

## 2. **核心实验方法和设置**

### 📊 使用的数据集
两个典型的**混沌动力系统**用于生成时间序列输入：
- **Lorenz system**：经典三维混沌系统，常用于测试非线性预测能力。
- **Chaotic food-chain system**：三物种食物链模型，具有复杂生态动力学行为。

> 输入维度均为 $ D=3 $，经归一化后送入 reservoir。

---

### ⚙️ 实验设置
#### Reservoir 模型结构
- **状态演化方程**：基于前庭系统建模，包含两部分：
  1. **机械模块**：用二阶线性系统描述半规管（semicircular canals）流体动力学；
  2. **神经模块**：采用 **FitzHugh–Nagumo (FHN)** 模型模拟毛细胞（hair cells）电信号输出。
- **两种配置对比**：
  - **Coupled reservoir**：$ A $ 为稀疏随机对称矩阵（密度 0.4），特征值可控；
  - **Uncoupled reservoir**：$ A $ 为对角阵，各节点独立演化。

#### 训练流程（Open-loop + Closed-loop）
1. **Open-loop phase**（训练与验证）：
   - 外部输入驱动 reservoir；
   - 收集 reservoir 状态并使用 ridge regression 学习输出权重 $ W_{\text{out}} $；
   - 评估 one-step prediction 错误（NRMSE）。
2. **Closed-loop phase**（测试）：
   - 将预测输出反馈作为下一时刻输入，形成自持预测；
   - 评估长期动态重建能力。

---

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **NRMSE**（Normalized Root-Mean-Square Error） | 衡量短期预测精度（训练/验证误差） |
| **Deviation Value (DV)** | 吸引子轨迹偏离程度，越小越好 |
| **KL Divergence** | 预测与真实吸引子分布的信息损失，越小越好 |
| **Largest Lyapunov Exponent (LLE)** | 判断是否保留原系统的混沌特性（应接近真值） |
| **Memory Capacity (MC)** | 定量衡量系统重构历史输入的能力 |

---

### 🔁 基线方法对比
- 主要对比对象为标准的 **coupled reservoir computing**；
- 所有超参数通过贝叶斯优化调优（见 Supplementary Table S1）；
- 在相同 reservoir size（如 N=30）、相同 spectral radius 或 eigenvalue spectrum 下进行公平比较。

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（N=30）

#### ✅ Lorenz 系统预测结果
| 指标 | Coupled | Uncoupled |
|------|--------|----------|
| Training Error (NRMSE) | 0.013 | 0.018 |
| Validation Error (NRMSE) | 0.015 | 0.019 |
| Deviation Value (DV) | 0.330 | 0.318 |
| KL Divergence | 0.0006 | 0.0027 |
| Predicted LLE | 0.030 | 0.030 |
| True LLE | 0.030 | — |

> ➤ Uncoupled 架构在长期动态重建方面表现更稳定（DV 更低），KL 差异略大但仍极小。

#### ✅ Food-chain 系统预测结果
| 指标 | Coupled | Uncoupled |
|------|--------|----------|
| Training Error | 0.006 | 0.009 |
| Validation Error | 0.007 | 0.009 |
| Deviation Value | 0.364 | 0.355 |
| KL Divergence | 0.0007 | 0.0004 |
| Predicted LLE | 0.021 | 0.021 |
| True LLE | 0.023 | — |

> ➤ Uncoupled 架构 KL 更优，LLE 匹配良好，说明其能准确捕捉系统混沌本质。

> ✅ 补充结果见 Supplementary Tables S2-S3：当 uncoupled 与 coupled 共享相同 **eigenvalue spectrum** 时，二者性能几乎完全一致。

---

### 🔬 消融实验结果

#### （1）Reservoir Size 影响（Fig. 3–4）
- 当 $ N < 30 $ 时，divergence 概率上升；
- $ N > 30 $ 后，DV 和 KL 收敛，LLE 接近真实值；
- **uncoupled 与 coupled 随规模增长趋势一致**，表明 scalability 相当。

#### （2）Memory Capacity 分析（Fig. 6–7）
- 对于线性 reservoir，推导出 memory function 公式：
  $$
  M_F(\tau) = \mathbf{H}_\tau^\top (\mathbf{H}\mathbf{H}^\top)^{-1} \mathbf{H}_\tau
  $$
  其中 $ \mathbf{H} $ 由 $ A $ 的特征值构成。
- **关键发现**：memory capacity 只依赖于 $ A $ 的 eigenvalue spectrum，而非具体拓扑结构。
- 数值验证显示：只要 eigenvalues 相同，uncoupled 与 coupled 的 $ M_F(\tau) $ 和总 memory capacity 几乎重合。

#### （3）Spectral Radius vs Eigenvalue Spectrum
- 若仅共享 spectral radius $ \rho(A) $，但 eigenvalues 不同 → memory capacity 明显下降；
- 强调：**不能只看 $ \rho $，必须控制整个 spectrum 才能保证性能匹配**。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **解耦 reservoir 可以达到与耦合 reservoir 相当甚至更好的预测性能**，打破了“必须 recurrent connectivity”的固有认知。
2. **memory capacity 决定了预测能力**，且对于线性系统，它**仅取决于 reservoir 矩阵 $ A $ 的特征值谱（eigenvalue spectrum）**，与是否耦合无关。
3. 在非线性 vestibular reservoir 中，若 uncoupled 与 coupled 具有相同的 eigenvalue spectrum，则 memory capacity 和预测统计高度相似。
4. **vestibular system 的生物机制本身就是一个天然高效的 reservoir**，其多尺度动力学（机械+神经）提供了丰富的非线性变换能力。

---

### ⚠️ 局限性
- 当前模型设定为 fixed-point steady state，尚未探索 spiking regime 下的表现；
- 所有分析基于理想化数学模型，实际物理实现可能引入噪声和失配；
- eigenvalue spectrum 匹配在物理系统中较难精确控制，需进一步工程适配。

---

### 🔮 未来工作方向
1. **引入 spiking dynamics**：使模型更具生物学真实性，并研究 spike-based RC 的潜力；
2. **跨平台物理实现**：尝试在 memristor、微流控、磁性材料等平台上构建 vestibular-inspired reservoir；
3. **扩展至高维任务**：应用于图像识别、语音处理等非时间序列任务；
4. **探索 adaptive eigenvalue tuning**：开发自动调节机制以维持最优 spectrum 匹配；
5. **与其他生物系统融合**：如结合 cochlea 或视觉路径，构建多功能 bio-hybrid computing 架构。

---

## ✅ 总结一句话
> 本文提出 **vestibular reservoir computing**，首次证明**解耦 reservoir 在适当设计下可媲美全耦合架构**，并通过理论与实验证明 **memory capacity 由 eigenvalue spectrum 决定**，为低复杂度、高效率的物理 RC 提供了全新可行路径。

</details>

---

### 16. [Energy-Efficient Federated Edge Learning For Small-Scale Datasets in Large IoT Networks](https://arxiv.org/abs/2604.10662)

**Authors**: Haihui Xie, Wenkun Wen, Shuwu Chen, Zhaogang Shu, Minghua Xia  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.10662v1  

#### Abstract
Large-scale Internet of Things (IoT) networks enable intelligent services such as smart cities and autonomous driving, but often face resource constraints. Collecting heterogeneous sensory data, especially in small-scale datasets, is challenging, and independent edge nodes can lead to inefficient re...

---

### 17. [Flow-Controlled Scheduling for LLM Inference with Provable Stability Guarantees](https://arxiv.org/abs/2604.11001)

**Authors**: Zhuolun Dong, Junyu Cao  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.11001v1  

#### Abstract
Large language models (LLMs) have been widely adopted due to their great performance across a wide range of applications. ChatGPT and Gemini now serve hundreds of millions of active users and handle billions of user requests per day, which puts optimizing LLM inference into the spotlight. A key chal...

---

### 18. [MobiFlow: Real-World Mobile Agent Benchmarking through Trajectory Fusion](https://arxiv.org/abs/2604.09587)

**Authors**: Yunfei Feng, Xi Zhao, Cheng Zhang, Dahu Feng, Daolin Cheng, Jianqi Yu, Yubin Xia, Erhu Feng  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.09587v1  

#### Abstract
Mobile agents can autonomously complete user-assigned tasks through GUI interactions. However, existing mainstream evaluation benchmarks, such as AndroidWorld, operate by connecting to a system-level Android emulator and provide evaluation signals based on the state of system resources. In real-worl...

---

### 19. [EE-MCP: Self-Evolving MCP-GUI Agents via Automated Environment Generation and Experience Learning](https://arxiv.org/abs/2604.09815)

**Authors**: Tiantian He, Yihang Chen, Keyue Jiang, Ka Yiu Lee, Kaiwen Zhou, Kun Shao, Shuai Wang  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.09815v1  

#### Abstract
Computer-use agents that combine GUI interaction with structured API calls via the Model Context Protocol (MCP) show promise for automating software tasks. However, existing approaches lack a principled understanding of how agents should balance these two modalities and how to enable iterative self-...

---

### 20. [TrajOnco: a multi-agent framework for temporal reasoning over longitudinal EHR for multi-cancer early detection](https://arxiv.org/abs/2604.10386)

**Authors**: Sihang Zeng, Young Won Kim, Wilson Lau, Ehsan Alipour, Ruth Etzioni, Meliha Yetisgen, Anand Oka  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.10386v1  

#### Abstract
Accurate estimation of cancer risk from longitudinal electronic health records (EHRs) could support earlier detection and improved care, but modeling such complex patient trajectories remains challenging. We present TrajOnco, a training-free, multi-agent large language model (LLM) framework designed...

---

### 21. [ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval](https://arxiv.org/abs/2604.10898)

**Authors**: David H. Yang, Yuxuan Zhu, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, Subhajit Chaudhury, Pin-Yu Chen  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.10898v1  

#### Abstract
Large language models (LLMs) have shown great performance on complex reasoning tasks but often require generating long intermediate thoughts before reaching a final answer. During generation, LLMs rely on a key-value (KV) cache for autoregressive decoding. However, the memory footprint of the KV cac...

---

### 22. [MAFIG: Multi-agent Driven Formal Instruction Generation Framework](https://arxiv.org/abs/2604.10989)

**Authors**: Shixing Zhao, Zheng Si, Pengpeng Ouyang, Zhengqing Hu, Wanqi Zhu, Dong Chen, Yibo Guo, Mingliang Xu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.10989v1  

#### Abstract
Emergency situations in scheduling systems often trigger local functional failures that undermine system stability and even cause system collapse. Existing methods primarily rely on robust scheduling or reactive scheduling, handling emergencies through predefined rules or rescheduling strategies. Ho...

---

### 23. [Dynamic Summary Generation for Interpretable Multimodal Depression Detection](https://arxiv.org/abs/2604.11334)

**Authors**: Shiyu Teng, Jiaqing Liu, Hao Sun, Yu Li, Shurong Chai, Ruibo Hou, Tomoko Tateyama, Lanfen Lin, Yen-Wei Chen  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.11334v1  

#### Abstract
Depression remains widely underdiagnosed and undertreated because stigma and subjective symptom ratings hinder reliable screening. To address this challenge, we propose a coarse-to-fine, multi-stage framework that leverages large language models (LLMs) for accurate and interpretable detection. The p...

---

### 24. [Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](https://arxiv.org/abs/2604.10072)

**Authors**: Chao Xue, Yao Wang, Mengqiao Liu, Di Liang, Xingsheng Han, Peiyang Liu, Xianjie Wu, Chenyao Lu, Lei Jiang, Yu Lu, Haibo Shi, Shuang Liang, Minlong Peng, Flora D. Salim  
**Category**: cs.CL  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.10072v1  

#### Abstract
Recent advancements in the Generative Reward Model (GRM) have demonstrated its potential to enhance the reasoning abilities of LLMs through Chain-of-Thought (CoT) prompting. Despite these gains, existing implementations of GRM suffer from two critical limitations. First, CoT prompting is applied ind...

---

### 25. [LLM-assisted Agentic Edge Intelligence Framework](https://arxiv.org/abs/2604.09607)

**Authors**: Chinmaya Kumar Dehury, Siddharth Singh Kushwaha, Qiyang Zhang, Alaa Saleh, Praveen Kumar Donta  
**Category**: cs.DC  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.09607v1  

#### Abstract
Edge intelligence delivers low-latency inference, yet most edge analytics remain hard-coded and must be redeployed as conditions change. When data patterns shift or new questions arise, engineers often need to write new scripts and push updates to devices, which slows iteration and raises operating ...

---

### 26. [Exact Certification of Neural Networks and Partition Aggregation Ensembles against Label Poisoning](https://arxiv.org/abs/2604.11416)

**Authors**: Ajinkya Mohgaonkar, Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan G\"unnemann  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.11416v1  

#### Abstract
Label-flipping attacks, which corrupt training labels to induce misclassifications at inference, remain a major threat to supervised learning models. This drives the need for robustness certificates that provide formal guarantees about a model's robustness under adversarially corrupted labels. Exist...

---

### 27. [Learning How Much to Think: Difficulty-Aware Dynamic MoEs for Graph Node Classification](https://arxiv.org/abs/2604.11473)

**Authors**: Jiajun Zhou, Yadong Li, Xuanze Chen, Chen Ma, Chuang Zhao, Shanqing Yu, Qi Xuan  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.11473v1  

#### Abstract
Mixture-of-Experts (MoE) architectures offer a scalable path for Graph Neural Networks (GNNs) in node classification tasks but typically rely on static and rigid routing strategies that enforce a uniform expert budget or coarse-grained expert toggles on all nodes. This limitation overlooks the varyi...

---

### 28. [Competing with AI Scientists: Agent-Driven Approach to Astrophysics Research](https://arxiv.org/abs/2604.09621)

**Authors**: Thomas Borrett, Licong Xu, Andy Nilipour, Boris Bolliet, Sebastien Pierre, Erwan Allys, Celia Lecat, Biwei Dai, Po-Wen Chang, Wahid Bhimji  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.09621v1  

#### Abstract
We present an agent-driven approach to the construction of parameter inference pipelines for scientific data analysis. Our method leverages a multi-agent system, Cmbagent (the analysis system of the AI scientist Denario), in which specialized agents collaborate to generate research ideas, write and ...

---

### 29. [CWCD: Category-Wise Contrastive Decoding for Structured Medical Report Generation](https://arxiv.org/abs/2604.10410)

**Authors**: Shantam Srivastava, Mahesh Bhosale, David Doermann, Mingchen Gao  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.10410v1  

#### Abstract
Interpreting chest X-rays is inherently challenging due to the overlap between anatomical structures and the subtle presentation of many clinically significant pathologies, making accurate diagnosis time-consuming even for experienced radiologists. Recent radiology-focused foundation models, such as...

---

### 30. [From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](https://arxiv.org/abs/2604.11137)

**Authors**: Chen Zhan, Xiaoyu Tan, Gengchen Ma, Yu-Jie Xiong, Xiaoyan Jiang, Xihe Qiu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.11137v1  

#### Abstract
The integration of Large Language Models (LLMs) into clinical decision support is critically obstructed by their opaque and often unreliable reasoning. In the high-stakes domain of healthcare, correct answers alone are insufficient; clinical practice demands full transparency to ensure patient safet...

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
