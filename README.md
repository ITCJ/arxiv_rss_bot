# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-19 06:42:30 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression](https://arxiv.org/abs/2603.17435)

**Authors**: Ruibo Fan, Xiangrui Yu, Xinglin Pan, Zeyu Li, Weile Luo, Qiang Wang, Wei Wang, Xiaowen Chu  
**Category**: cs.DC  
**Published**: 2026-03-19  
**Score**: 14.5  
**Type**: new  
**ArXiv ID**: 2603.17435v1  

#### Abstract
Lossless model compression holds tremendous promise for alleviating the memory and bandwidth bottlenecks in bit-exact Large Language Model (LLM) serving. However, existing approaches often result in substantial inference slowdowns due to fundamental design mismatches with GPU architectures: at the k...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ZIPSERV: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **lossless compression** 方法虽然能减少模型存储大小并保证数值精确性（bit-exact），但在实际 **LLM 推理** 中引入显著的运行时开销，导致推理速度大幅下降。其根本原因在于：
- **Kernel-level 问题**：传统熵编码（如 Huffman、ANS）产生变长比特流，解码过程依赖数据且串行化，破坏 GPU 的 **SIMT 并行执行模型**，造成严重的控制流发散（control-flow divergence）。
- **System-level 问题**：主流框架采用“先解压再计算”的 **decoupled pipeline**，需将解压后的权重写入全局内存，造成冗余的数据搬运和带宽浪费，降低了 **compute intensity**。

这使得 lossless compression 在推理中面临“节省内存却牺牲速度”的两难困境。

---

### **提出了什么新方法或新思路**
论文提出 **ZIPSERV**，首个为 GPU 上高效 LLM 推理而协同设计的 **硬件感知无损压缩框架**，核心创新如下：

#### **(1) Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE)**
- 利用 LLM 中 BFloat16 权重的 **指数位（exponent）分布高度偏斜且连续（contiguous）** 的特性（前7个高频指数覆盖 >95% 的权重）。
- 设计一种 **固定长度、基于位图（bitmap）的编码格式**，避免变长编码带来的串行解码。
- 将每个 8×8 权重块编码为三个 64-bit 位图（bit-planes）和两个紧凑值缓冲区（高频值仅存 sign + mantissa，异常值存全精度 BF16）。
- 引入 **隐式查找表（implicit lookup）**：通过 `base_exp + codeword` 算术重建指数，避免共享内存查表。

#### **(2) 融合解压-GEMM 内核：ZipGEMM**
- 提出 **“load-compressed, compute-decompressed”** 执行范式。
- 设计 **ZipGEMM** 内核，在加载压缩权重的同时，直接在寄存器中进行解码，并将结果送入 **Tensor Core** 进行矩阵乘法。
- 完全消除中间解压缓冲区，最大化 **compute intensity** 和 **SIMT 并行度**。

#### **(3) 阶段感知推理策略（Stage-Aware Inference Strategy）**
- **Prefill 阶段**（计算密集）：使用解耦的解压内核，因高算力可摊销解压开销。
- **Decode 阶段**（内存密集）：启用融合的 ZipGEMM，最大化带宽效率。

---

### **相比现有方法的优势**
| 维度 | ZIPSERV | 传统方法（如 DFloat11, DietGPU） |
|------|--------|-------------------------------|
| **解码并行性** | ✅ 固定长度，支持常数时间并行解码 | ❌ 变长编码，串行解析，线程发散严重 |
| **内存访问** | ✅ 解压直接进寄存器，无中间缓冲 | ❌ 解压到全局内存，冗余读写 |
| **系统效率** | ✅ 融合内核，提升 compute intensity | ❌ 解耦流水线，降低 roofline 性能上限 |
| **性能影响** | ✅ 同时实现压缩和加速 | ❌ 压缩带来显著推理延迟 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **模型家族**：LLaMA-3.1（8B, 70B, 405B）、Qwen2.5（7B–72B）、Gemma-3（12B, 27B）、Mistral（24B, 123B）
- **关注层**：Transformer 块中的线性层，包括：
  - `QKV_proj`, `O_proj`, `GateUp_proj`, `Down_proj`, `LM Head`

### **实验设置和评估指标**
#### **硬件平台**
1. **消费级**：4× NVIDIA RTX 4090（24GB, CC 8.9）
2. **数据中心级**：4× NVIDIA L40S（48GB, CC 8.9）
3. **前沿测试**：RTX 5090（Blackwell, CC 12.0）

#### **评估指标**
- **Kernel-level**：
  - GEMM 执行时间（normalized speedup vs. cuBLAS_TC）
  - 解压时间
  - DRAM 读取量、bank conflict、ALU/Tensor Core 利用率
- **End-to-end**：
  - 端到端请求延迟（latency）
  - 输出吞吐量（tokens/sec）
  - 内存占用（weight footprint, KV cache size）

#### **批处理大小（Batch Size）**
- Prefill：N = 8, 16, 32
- Decode：N = 1–128（模拟自回归生成）

---

### **基线方法对比**
| 基线 | 类型 | 说明 |
|------|------|------|
| **cuBLAS_TC** | 基准 | NVIDIA 官方 BF16 Tensor Core GEMM |
| **DietGPU** | Lossless | Facebook 开源的 rANS 解压框架 |
| **nvCOMP (rANS)** | Lossless | NVIDIA 通用解压库 |
| **DFloat11** | Lossless | 最先进的 Huffman 编码 LLM 推理框架 |
| **vLLM** | 推理引擎 | 当前最先进的 LLM 服务框架（集成对比） |
| **Transformers** | 推理引擎 | Hugging Face 标准库 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) Kernel-level 加速**
- **ZipGEMM vs. cuBLAS_TC**：
  - RTX 4090：平均 **1.31×**，最高 **1.71×**
  - L40S：平均 **1.36×**，最高 **2.21×**
- **ZipGEMM vs. DFloat11**：最高 **5.53×** 加速
- **小层（如 O_proj）**：可能略慢（如 0.79×），但占总 FLOPs 比例小，不影响整体收益。

#### **(2) 解压内核性能**
- **ZIPSERV-Decomp** vs. DietGPU / nvCOMP / DFloat11：
  - 平均提速 **2.14× / 1.83× / 1.10×**
- 解压时间远低于 GEMM 时间，prefill 阶段开销仅 **~4%**（N=8192）

#### **(3) 端到端推理性能**
- **平均延迟降低**：
  - vs. vLLM：**17.6%**
  - vs. Transformers：**60.79%**
  - vs. DFloat11：**82.13%**
- **平均吞吐提升**：
  - vs. vLLM：**1.22×**
  - vs. Transformers：**3.18×**
  - vs. DFloat11：**8.52×**
- **长序列生成优势更明显**：2048 tokens 时，LLaMA3.1-8B 吞吐达 **1105 tokens/sec**（vs. vLLM 的 666 tokens/sec，**1.66×**）

#### **(4) 模型压缩率**
- **平均压缩 30%**（压缩比 ~1.51×）
  - LLaMA3.1-8B：14.96 GB → **10.83 GB**（72.4%）
  - Mistral-24B：43.92 GB → **31.30 GB**（71.3%）
  - LLaMA3.1-70B：131.56 GB → **93.52 GB**（71.1%）

#### **(5) 内存利用优化**
- 释放的权重内存被用于扩展 **KV Cache**：
  - LLaMA3.1-8B：KV Cache 从 5.07 GB → **8.60 GB**（**1.70×** 增加）
  - 支持更大 batch 和更长上下文。

---

### **消融实验结果**
- **TCA-TBE 固定长度 vs. 变长编码**：固定长度使解码完全并行，避免发散。
- **融合 vs. 解耦流水线**：
  - 解耦方案 compute intensity 下降 **>60%**
  - 融合方案 compute intensity **反超未压缩 GEMM**
- **ZipGEMM 微架构分析**：
  - DRAM 读取减少 **29.3%**
  - 共享内存 bank conflict 几乎为零（~4.7K vs. DietGPU 百万级）
  - Tensor Core 利用率达 cuBLAS 的 **71.6%**，证明解码开销被有效隐藏。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Lossless compression 可以同时实现压缩和加速**：ZIPSERV 是首个在 GPU 上实现这一目标的系统。
2. **算法-硬件协同设计是关键**：传统熵编码与 GPU 架构不匹配；TCA-TBE + ZipGEMM 充分利用了 SIMT 和 Tensor Core 特性。
3. **decode 阶段是瓶颈，也是最大受益者**：融合执行在内存受限场景下效果最显著。
4. **静态压缩带来动态收益**：压缩的权重空间被转化为更大的 KV Cache，进一步提升吞吐。

---

### **方法的局限性**
1. **对小尺寸 GEMM 效果有限**：如 `O_proj` 层，因难以充分调度硬件资源，可能出现轻微性能回退。
2. **主要针对推理优化**：训练场景未覆盖。
3. **当前聚焦 NVIDIA GPU**：虽理论上可移植，但尚未在 Intel AMX 或 AMD Matrix Cores 上验证。
4. **依赖 BFloat16 指数分布特性**：若未来模型权重分布变化，压缩率可能下降。

---

### **未来工作方向**
1. **扩展至 KV Cache 压缩**：解决长上下文下的内存瓶颈。
2. **跨硬件平台适配**：支持 Intel AMX、AMD Matrix Cores 等。
3. **与 lossy 方法结合**：在量化模型上应用 TCA-TBE，进一步挖掘残余冗余。
4. **应用于分布式训练通信压缩**：减少梯度同步开销。
5. **探索动态压缩窗口**：适应不同层或不同模型的指数分布差异。

---

> **总结**：ZIPSERV 成功将 **lossless compression** 从一个“只省空间”的工具，转变为“既省空间又提速度”的实用技术，首次实现了无损压缩在 LLM 推理中的 **双赢（win-win）**，为在消费级 GPU 上高效部署大模型提供了强有力的新范式。

</details>

---

### 2. [SENSE: Efficient EEG-to-Text via Privacy-Preserving Semantic Retrieval](https://arxiv.org/abs/2603.17109)

**Authors**: Akshaj Murhekar, Christina Liu, Abhijit Mishra, Shounak Roychowdhury, Jacek Gwizdka  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2603.17109v1  

#### Abstract
Decoding brain activity into natural language is a major challenge in AI with important applications in assistive communication, neurotechnology, and human-computer interaction. Most existing Brain-Computer Interface (BCI) approaches rely on memory-intensive fine-tuning of Large Language Models (LLM...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SENSE: Efficient EEG-to-Text via Privacy-Preserving Semantic Retrieval

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Brain-Computer Interface (BCI)** 系统在将脑电图（EEG）信号解码为自然语言时，通常依赖对 **Large Language Models (LLMs)** 或编码器-解码器模型进行端到端的微调。这类方法存在以下问题：
- **计算开销大**：需要大量资源进行训练，难以部署。
- **隐私风险高**：原始EEG信号需上传至云端，暴露敏感神经数据。
- **缺乏可扩展性**：模型与特定LLM耦合，难以适应快速发展的生成式AI生态。

### 提出的新方法：SENSE框架
作者提出 **SENSE (SEmantic Neural Sparse Extraction)** ——一种轻量级、隐私保护的EEG-to-Text框架，其核心思想是**将解码过程解耦为两个阶段**：
1. **On-device 语义检索**：在本地设备上将EEG信号映射为离散的文本关键词（Bag-of-Words, BoW）。
2. **Prompt-based 语言生成**：将提取的关键词作为提示（prompt），输入现成的LLM生成最终文本。

该方法**完全避免了对LLM的微调**，仅通过提示工程实现高质量文本合成。

### 相比现有方法的优势
| 维度 | SENSE | 传统方法（如THOUGHT2TEXT） |
|------|-------|-----------------------------|
| **计算效率** | 轻量级（~6M参数），可在边缘设备运行 | 需要大规模微调，资源密集 |
| **隐私保护** | 原始EEG保留在本地，仅共享抽象语义词 | 原始神经数据外传，隐私泄露风险高 |
| **模块化设计** | 可灵活接入任意LLM，无需重新训练 | 模型与LLM强绑定，迁移成本高 |
| **训练成本** | 几分钟内完成训练 | 数小时甚至数天的训练周期 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用公开的 **CVPR2017EEG-ImageNet** 数据集。
- 包含6名受试者观看2,000张图像时记录的同步EEG信号。
- 每个图像配有自然语言描述（caption），共9,940个样本（训练：7,959；验证：1,994；测试：1,987）。
- EEG通道数为128，采样频率为200Hz。

### 实验设置
- **EEG编码器**：采用预训练的 **ChannelNet** 架构（冻结），提取视觉对齐的EEG嵌入。
- **跨模态投影器（Similarity Refiner）**：一个多层感知机（MLP），将EEG嵌入映射到 **CLIP文本嵌入空间**，以便基于余弦相似度进行关键词检索。
- **关键词提取**：从CLIP词汇表中选取top-15最相关的词构成BoW。
- **LLM生成**：使用多种LLM进行zero-shot生成，包括：
  - GPT-4o-mini
  - Gemini 2.5 Flash Lite
  - LLaMA-3-8B
  - Qwen2.5-7B

### 评估指标
- **自动评估指标**：
  - BLEU-1 / BLEU-4
  - ROUGE-1 / ROUGE-2 / ROUGE-L
  - METEOR
  - BERTScore
- **人工评估（LLM-based）**：使用GPT-5对生成文本进行评分：
  - Fluency（流畅性）
  - Adequacy（充分性）
- 所有指标均在测试集上平均。

### 基线方法对比
- **Naive Baseline**：直接使用未优化的EEG嵌入进行关键词检索。
- **THOUGHT2TEXT (Mishra et al., 2025)**：当前最先进的端到端微调方法，作为主要对比基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1）
在最优配置下（Focal Loss + Gemini 2.5 Flash Lite + WithObj）：
- **ROUGE-1**: **31.5**
- **ROUGE-2**: **8.5**
- **ROUGE-L**: **28.7**
- **BLEU-1**: **25.2**
- **BLEU-4**: **5.6**
- **METEOR**: **26.1**
- **BERTScore**: **0.897**
- **GPT-5 Fluency**: **4.77**
- **GPT-5 Adequacy**: **1.40**

> 注：数值越高越好，加粗表示最佳结果。

### 与基线方法的对比
- SENSE在多个指标上**匹配甚至超越**了完全微调的 **THOUGHT2TEXT** 基线。
  - 例如，在ROUGE-1上，SENSE（31.5） > THOUGHT2TEXT（30.0）。
- 即使使用较小的LLM（如LLaMA-3-8B），性能也接近或优于基线。
- **闭源LLM（Gemini、GPT-4o-mini）显著优于开源模型**，说明它们更擅长利用结构化BoW提示。

### 消融实验结果
#### （1）不同损失函数的影响
- **Focal Loss** 表现最佳，因其能有效应对极端类别不平衡（平均仅激活5个词 / 1210个）。
- 对比：
  - Focal Loss: ROUGE-1 = 31.5
  - BCE: ROUGE-1 = 30.2
  - Contrastive Multi-Label: ROUGE-1 = 30.6
  - Naive: ROUGE-1 = 29.6

#### （2）是否提供对象标签（WithObj vs WithoutObj）
- 提供预测的对象标签（object label）作为锚点可显著提升性能。
  - 例如，Gemini在WithObj下ROUGE-1为31.5，WithoutObj为30.5。
- 但在WithoutObj情况下，LLM仍能基于BoW重建大致场景，显示BoW本身具有较强语义表达能力。

#### （3）跨被试泛化能力分析（Figure 2）
- 在所有6名受试者上表现一致，**无需个体校准（per-subject calibration）**。
- 性能排序稳定（Focal > BCE > Naive），表明模型具有良好的跨被试泛化能力。

---

## 4. 关键结论和发现

### 主要发现
1. **轻量级语义检索可替代LLM微调**：  
   通过将EEG信号映射到CLIP语义空间并提取关键词，即可引导LLM生成高质量文本，**无需昂贵的端到端训练**。

2. **隐私与性能可兼得**：  
   将原始EEG保留在本地，仅共享抽象关键词，既保护了用户隐私，又实现了与微调方法相当甚至更优的生成质量。

3. **模块化架构更具可持续性**：  
   随着LLM不断进化，SENSE只需更换下游LLM即可受益，而无需重新训练整个系统。

4. **Focal Loss对稀疏语义建模至关重要**：  
   在高度稀疏的多标签分类任务中，Focal Loss能有效缓解类别不平衡问题，提升关键词检索准确性。

### 局限性
1. **非侵入式EEG分辨率有限**：  
   由于EEG空间分辨率低、噪声大，仍会出现语义混淆（如“mushroom”误判为“flower”，“piano”误判为“billiard table”）。

2. **固定词汇限制表达能力**：  
   当前使用ImageNet caption构建的1210词词汇表，限制了开放词汇（open-vocabulary）思维到文本的生成。

3. **对外部LLM API的依赖**：  
   若不部署本地高性能LLM，则仍需信任第三方服务，存在潜在安全风险。

4. **生成可能脱离真实神经状态**：  
   若关键词检索失败，LLM可能基于先验知识“幻觉”生成合理但错误的内容（即“scene maker”而非“neural decoder”）。

### 未来工作方向
- 引入动态阈值或置信度机制改进BoW提取。
- 在概念选择阶段引入句法先验以增强连贯性。
- 扩展语义词汇表，迈向开放词汇生成。
- 加强跨模态对齐（如结合视觉上下文）。
- 实现实时多传感器融合，支持更丰富的脑驱动语言交互。

---

> ✅ **总结一句话**：  
> SENSE提出了一种高效、隐私友好的EEG-to-Text新范式——**用语义检索代替模型微调**，在保持原始神经数据本地化的同时，实现了与最先进方法相媲美甚至更优的文本生成性能。

</details>

---

### 3. [PlotTwist: A Creative Plot Generation Framework with Small Language Models](https://arxiv.org/abs/2603.16410)

**Authors**: Abhinav Thorat, Ravi Kolla, Jyotin Goel, Niranjan Pedanekar  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.16410v1  

#### Abstract
Creative plot generation presents a fundamental challenge for language models: transforming a concise premise into a coherent narrative that sustains global structure, character development, and emotional resonance. Although recent Large Language Models (LLMs) demonstrate strong fluency across gener...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《PlotTwist: A Creative Plot Generation Framework with Small Language Models》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **创意情节生成**（Creative Plot Generation）是语言模型面临的一项根本挑战：如何将一个简短的前提（premise）扩展为具有**全局结构一致性、角色发展、情感共鸣和节奏控制**的连贯叙事。
- 尽管前沿的 **Large Language Models (LLMs)** 在通用任务上表现出色，但在创意写作等特定领域仍需进行**偏好对齐**（preference alignment），而对大规模 LLM 进行此类对齐在计算上成本极高，限制了其可访问性和实际部署。

### **提出的新方法与新思路**
作者提出了 **PLOTTWIST**，一个专为 **Small Language Models (SLMs)** 设计的三模块结构化框架，仅用 ≤3B 活跃参数即可生成媲美 200 倍更大模型的情节质量。其核心创新在于：

1. **结构化解耦设计**：
   - 将生成过程分解为三个独立组件，避免单一模型隐式学习所有叙事维度。
   
2. **三大核心组件**：
   - **Aspect Rating Reward Model**：基于五项 **Narrative Quality Dimensions (NQDs)** 构建细粒度奖励信号。
   - **Mixture-of-Experts (MoE) Plot Generator**：采用 **Direct Preference Optimization (DPO)** 对 SLM 进行高效对齐。
   - **Agentic Evaluation Module**：独立于训练流程，模拟人类批判性判断，实现无偏后评估。

3. **新型 Positive-Negative Prompting 策略**：
   - 通过分别提示 LLM 输出“正面评价”和“负面批评”，缓解 LLM 评估中的**正向偏差**（positivity bias），提升评分可靠性。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **资源效率** | 使用仅 3B 活跃参数的 SLM，远低于主流 LLM（如 GPT-4.1、Claude Sonnet 4）。 |
| **性能表现** | 在多个 NQDs 上超越包括 GPT-4.1 在内的前沿闭源模型。 |
| **架构解耦** | 各模块可独立部署，支持灵活迭代与模块替换。 |
| **抗奖励黑客攻击** | Agentic Evaluation 与训练奖励模型分离，防止过拟合或循环验证。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **MovieLens 数据集**（5,000 部电影）
   - 用于构建初始情节池，覆盖广泛的 IMDb 评分范围。
2. **维基百科电影剧情摘要**
   - 抓取长度 ≤4000 单词的剧情作为原始文本输入。
3. **外部高质量/低质量基准集**：
   - **高质参考**：*101 Greatest Screenplays of All Time*（GSAT）
   - **低质参考**：*Golden Raspberry Awards*（Razzies）获奖剧本
   - 用于验证评估模块是否能可靠区分叙事质量。

### **实验设置与评估指标**

#### **五大 Narrative Quality Dimensions (NQDs)**
| 维度 | 描述 |
|------|------|
| **Character Development** | 角色是否有有意义的成长弧线？动机是否清晰？关系是否真实？ |
| **Tone Consistency** | 故事氛围、风格和情绪是否一致？有无突兀转变？ |
| **Pacing** | 叙事节奏是否合理？事件分布是否均衡？紧张感是否得当？ |
| **Narrative Coherence** | 情节逻辑是否连贯？是否存在漏洞或矛盾？因果链是否成立？ |
| **Emotional Turning Points** | 是否存在有力的情感转折点？是否引发观众共鸣？ |

#### **评估方式**
- 所有生成情节由 **Agentic Evaluation Module** 进行打分（非训练所用 Reward Model），确保评估独立性。
- 采用 **bootstrap 重采样法**计算 95% 置信区间，并报告 Cohen’s d 效应量以衡量差异显著性。

### **基线方法对比**
涵盖三大正交维度的全面基线：

| 类别 | 基线模型 |
|------|--------|
| **模型规模** | GPT-4.1, Claude Sonnet 4, Gemini 2.0 Flash, Llama-3-70B, Qwen-3-32B |
| **架构设计** | DeepSeek-R1 14B (MoE), Phi-4 Mini, Mistral Small 2501 |
| **生成范式** | Agents'Room (多智能体协作), WizardLM-StoryTelling (指令微调) |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**
在 160 个测试前提下，各模型平均得分（越高越好）：

| Model | Char Dev | Tone Cons | Pacing | Narr Coher | Emo Turn | **Avg** |
|-------|----------|-----------|--------|------------|-----------|--------|
| **PLOTTWIST** | 8.64±0.62 | **8.70±0.30** | **8.85±0.29** | **8.89±0.39** | **8.98±0.21** | **8.81** ✅ |
| Claude Sonnet 4 | **8.67±0.74** | 8.64±0.32 | 8.65±0.43 | 8.81±0.44 | 8.95±0.16 | 8.73 |
| GPT-4.1 | 8.29±0.76 | 8.61±0.37 | 8.74±0.30 | 8.74±0.46 | 8.88±0.22 | 8.65 |
| Agents'Room | 8.55±0.48 | 8.59±0.34 | 8.68±0.39 | 8.81±0.42 | 8.94±0.19 | 8.74 |
| Llama-3-70B | 7.84±0.95 | 8.27±0.48 | 8.22±0.55 | 8.11±0.67 | 8.68±0.36 | 8.22 |

> 📌 **结论**：PLOTTWIST 在 **4/5 个 NQDs 上排名第一**，总体平均得分最高（8.81），尤其在 **Pacing 和 Emotional Turning Points** 上领先明显。

---

### **与基线方法的对比结果**

| 对比维度 | 结果 |
|--------|------|
| **vs. 前沿大模型**（GPT-4.1, Claude 等） | PLOTTWIST 以 **3B 活跃参数** 超越高达 **~600B 参数** 的闭源模型（如 GPT-4.1: 8.65 → PLOTTWIST: 8.81） |
| **vs. 大型开源模型**（Llama-3-70B） | 显著优于未专门对齐的大模型（8.22 vs. 8.81） |
| **vs. 多智能体系统**（Agents'Room） | 性能更优（8.74 → 8.81），且无需复杂协调机制，仅需单次推理 |
| **vs. 指令微调模型**（WizardLM） | 明显胜出（8.03 vs. 8.81），说明 token-level 监督不足以捕捉长程叙事结构 |

---

### **消融实验结果（Ablation Studies）**

| 实验方向 | 发现 |
|--------|------|
| **模型规模影响** | PLOTTWIST（3B） > GPT-4.1（~600B） → 表明**结构化对齐比单纯扩大模型更有效** |
| **架构设计影响** | 基础 MoE 模型（8.03）→ 加入 DPO 后提升至 8.81（+0.78）→ **DPO 是主要增益来源** |
| **生成范式比较** | 单模型 + DPO > 多智能体协作（Agents'Room）→ **偏好优化可内化协作推理能力** |
| **MoE 效率优势** | 仅激活 3B 参数即达到接近 32B 密集模型的效果 → **稀疏激活兼顾容量与效率** |

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **小模型也能讲好故事**：通过结构化偏好对齐，**SLMs 完全可以在创意写作任务上匹敌甚至超越 LLMs**。
2. ✅ **Positive-Negative Prompting 更可靠**：该策略显著提升了自动评分系统的判别力，能稳定识别高质量与低质量剧本。
3. ✅ **Agentic Evaluation 具备人类级判别能力**：在 GSAT vs. Razzies 测试中，评估模块始终给优质剧本更高分（Δ=+1.15, p<1e-27）。
4. ✅ **质量自适应生成行为**：PLOTTWIST 能根据输入质量动态调整干预强度：
   - **优秀剧本**：轻微润色（refinement）
   - **中等剧本**：结构性增强（enhancement）
   - **劣质剧本**：近乎完全重写（regeneration）

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **依赖合成数据** | 训练数据由 LLM 自动生成，可能存在噪声或偏见传播风险。 |
| **人工验证缺失** | 缺乏真实人类作家对生成情节的质量打分，依赖代理指标（如 GSAT/Razzies）。 |
| **领域泛化待验证** | 当前聚焦影视情节，是否适用于小说、游戏叙事等其他形式尚不明确。 |
| **实时性未强调** | 虽然使用 SLM，但 MoE + DPO + 多轮评估可能仍有一定延迟。 |

### **未来工作方向**
1. **引入人类反馈闭环**：结合真实编剧评审进行迭代优化，进一步提升真实性。
2. **跨媒介扩展**：应用于小说、动画、互动叙事（interactive fiction）等场景。
3. **可控性增强**：允许用户指定角色性格、结局倾向、主题深度等约束条件。
4. **轻量化部署**：探索在移动端或边缘设备上的本地运行方案，推动普惠创作辅助工具落地。

---

> 🔚 **总结一句话**：  
> **PLOTTWIST 证明了“结构优于规模”——通过将叙事结构外显化并辅以精准的偏好优化，小模型也能创造出媲美顶级大模型的动人故事。**

</details>

---

### 4. [SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation](https://arxiv.org/abs/2603.16219)

**Authors**: Hang Lv, Sheng Liang, Hao Wang, Yongyue Zhang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.16219v1  

#### Abstract
Realizing personalized intelligence faces a core dilemma: sending user history to centralized large language models raises privacy concerns, while on-device small language models lack the reasoning capacity required for high-quality generation. Our pilot study shows that purely local enhancements re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对个性化生成中的**核心矛盾**：  
- **集中式大模型（LLM）** 虽具备强大的推理能力，但需要上传用户历史数据，存在**隐私泄露风险**和延迟瓶颈；  
- **本地小模型（SLM）** 可保护隐私，但受限于容量，推理能力弱，难以生成高质量、逻辑连贯的个性化内容。

作者通过一项**先导研究（pilot study）** 发现：即使采用 RAG、LoRA 等增强技术，本地小模型仍无法在生成质量上超越未接触用户数据的通用大模型，说明**本地优化无法弥补推理能力的鸿沟**。

### **提出的新方法：SPECSTEER**
提出一种**非对称协同推理框架 SPECSTEER**，将个性化生成建模为 **Bayesian Knowledge Fusion** 问题，并重新利用 **Speculative Decoding** 作为分布式对齐协议，实现“**Draft-Verify-Recover**”三阶段流程：

1. **Draft（草拟）**：本地 Specialist 小模型基于私有上下文生成个性化候选序列；
2. **Verify（验证）**：云端 Generalist 大模型通过**比率验证机制（ratio-based verification）** 判断逻辑合理性，无需访问原始用户数据；
3. **Recover（恢复）**：若草案被拒绝，通过**引导恢复（steering recovery）** 在纠正时注入本地意图，确保输出仍符合用户偏好。

### **相比现有方法的优势**
| 维度 | SPECSTEER | 传统方法 |
|------|----------|---------|
| **隐私保护** | ✅ 用户上下文始终保留在设备端 | ❌ 需上传上下文或模型参数 |
| **推理质量** | ✅ 结合小模型的个性化与大模型的逻辑能力 | ❌ 单一模型能力受限 |
| **效率** | ✅ 实现 **2.36× 推理加速** | ❌ 同步融合方法通信开销高 |
| **通信开销** | ✅ 仅传输 token ID 和稀疏 logit 向量 | ❌ 需频繁同步完整分布或上下文 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LaMP (Personalized Generation Benchmark)**：包含新闻标题生成、学术标题生成、推文改写等任务。
- **LongLaMP**：扩展版长文本个性化生成基准，涵盖摘要生成（Abstract）、评论写作（Review）、主题写作（Writing）等复杂任务。

### **实验设置与评估指标**
- **模型组合**：
  - 小模型（Specialist）：Qwen3-0.6B、Qwen2.5-1.5B、Llama-3.1-8B
  - 大模型（Generalist）：Qwen3-32B、Llama-3.1-32B
- **评估指标**：
  - **ROUGE-1 (R1)** 和 **ROUGE-L (RL)**：衡量生成文本与参考文本的重叠度。
  - **Speedup**：推理吞吐量提升倍数（tokens/s）。
  - **Acceptance Rate (α)**：草案被接受的比例，反映效率。

### **基线方法对比**
- **本地增强方法**：
  - `Direct`：零样本生成
  - `LoRA`：参数高效微调
  - `RAG`：检索增强生成
  - `RAFT`：结合 RAG 与微调
- **云端大模型**：`32B Direct`（无用户上下文）
- **协同方法**：
  - `CoSteer`：逐 token 融合
  - `LightCoSteer`：轻量级融合
  - `Standard SD`：标准推测解码

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **表1：容量缺陷实证（Capacity Deficit）**
在 LaMP 六项任务中，**即使是经过 LoRA+RAG 增强的小模型，在 22/24 项指标上仍劣于未接触用户数据的 32B 大模型**，证明本地增强不足以弥补推理能力差距。

#### **表2 & 表3：LongLaMP 上的性能对比**
| 方法 | Abstract (R1/RL) | Review (R1/RL) | Writing (R1/RL) |
|------|------------------|----------------|-----------------|
| SLM+ (RAG) | 39.89 / 21.65 | 23.18 / 12.84 | 25.50 / 12.36 |
| LLM (32B) | 40.18 / 22.17 | 31.18 / 14.78 | 29.46 / 12.64 |
| **SPECSTEER** | **41.35 / 23.57** | **33.03 / 17.26** | **30.79 / 12.88** |

✅ SPECSTEER 在所有任务上均优于 SLM+ 和 LLM，尤其在**Review**任务上提升显著（+1.85 R1 vs LLM）。

#### **表4：效率与质量权衡**
| 方法 | Speed (tokens/s) | Speedup | Acceptance Rate (α) |
|------|------------------|---------|---------------------|
| Vanilla LLM | 22.58 | 1.00× | — |
| CoSteer | 9.71 | 0.43× | — |
| Standard SD | 22.13 | 0.98× | ~35% |
| **SPECSTEER (λ=0.1)** | **53.29** | **2.36×** | **86.16%** |

✅ SPECSTEER 在保持高质量的同时，实现 **2.36× 加速**，远超其他协同方法。

### **消融实验结果**
- **噪声鲁棒性（Appendix A.3）**：即使 Specialist 输入噪声或使用弱检索器（BM25），SPECSTEER 仍能通过云端验证修复错误，维持高性能。
- **跨架构部署（Appendix A.4）**：Qwen Specialist + Llama Generalist 仍有效，证明框架**不依赖特定模型家族**。
- **超参数敏感性**：
  - **β（steering strength）**：在 [0.5, 2.0] 范围内稳定，过大导致逻辑断裂。
  - **λ（verification threshold）**：λ ∈ [0.1, 0.5] 为最优区间，平衡质量与效率。

---

## **4. 关键结论和发现**

### **主要发现**
1. **本地模型存在“容量缺陷”**：即使增强，也无法弥补推理能力不足。
2. **协同优于孤立优化**：SPECSTEER 成功融合本地个性化与云端推理，实现“1+1 > 2”的效果。
3. **效率与隐私可兼得**：通过比率验证与稀疏恢复，实现低通信开销下的高效协同。
4. **框架具有鲁棒性和泛化性**：对噪声、弱检索、跨架构均表现稳健。

### **方法的局限性**
- 当本地 Specialist 完全失效（如严重过拟合）时，引导信号消失，收益下降。
- 依赖云端大模型的可用性，不适合完全离线场景。
- 恢复阶段需设计合理的 β 参数以避免过度偏移。

### **未来工作方向**
- 集成更高级的隐私保护机制（如差分隐私、联邦学习）到输出层。
- 扩展至多模态个性化生成（如图文生成）。
- 探索动态调整 λ 和 β 的自适应策略，进一步优化质量-效率曲线。
- 研究在边缘集群上的分布式 SPECSTEER 架构。

---

> **总结一句话**：  
> SPECSTEER 通过将 **Speculative Decoding** 重构为 **隐私保护的协同推理协议**，首次实现了**高质量、高效率、高隐私**的个性化生成，为边缘-云协同智能提供了可扩展的新范式。

</details>

---

### 5. [MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild](https://arxiv.org/abs/2603.17187)

**Authors**: Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu Zheng, Cihang Xie, Huaxiu Yao  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.17187v1  

#### Abstract
Large language model (LLM) agents are increasingly used for complex tasks, yet deployed agents often remain static, failing to adapt as user needs evolve. This creates a tension between the need for continuous service and the necessity of updating capabilities to match shifting task distributions. O...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild  
**论文核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前部署在真实环境中的 **LLM Agent** 存在一个根本矛盾：它们需要持续服务用户，但其能力通常是静态的，无法随着任务分布的漂移而动态演化。这导致模型在面对新任务时反复失败，尤其是在多通道、多场景的复杂工作流中（如 OpenClaw 平台连接 20+ 消息渠道）。现有方法存在以下局限：
- **Memory-based 方法**：仅存储原始对话轨迹，冗余且难以提取可迁移的行为模式。
- **Skill-based 方法**：构建技能库但与模型权重优化脱节，无法实现联合演进。
- **RL-based 方法**：依赖梯度更新，常需离线训练并忽略“陈旧奖励污染”（stale reward contamination）问题——即旧技能下的轨迹若用于新策略训练，会引入错误信号。

### 提出了什么新方法或新思路
本文提出 **MetaClaw**，一个面向真实世界部署的 **continual meta-learning 框架**，通过两个互补机制实现 LLM Agent 的自主演化：

1. **Skill-driven Fast Adaptation（技能驱动快速适应）**
   - 利用 LLM Evolver 分析失败轨迹，自动生成新的 **behavioral instruction**（行为指令），即时注入到系统提示中。
   - 零停机、零参数更新，立即生效，提升后续任务表现。

2. **Opportunistic Policy Optimization（机会主义策略优化）**
   - 在用户非活跃窗口（由 OMLS 调度器检测）触发基于 **Cloud LoRA** 的 **RL 微调**。
   - 使用 **Process Reward Model (PRM)** 进行梯度更新，优化模型底层策略 `θ`。

二者形成**正向循环**：
- 更好的策略产生更有信息量的失败 → 更高质量的技能合成；
- 更丰富的技能库生成更高回报的轨迹 → 更有效的策略优化。

此外，引入 **Skill Generation Versioning Mechanism**，严格区分：
- **Support Data**：用于技能演化的失败轨迹（旧技能上下文）
- **Query Data**：技能更新后的新轨迹（新技能上下文）

确保 RL 更新只使用有效数据，防止陈旧奖励污染。

### 相比现有方法的优势
| 维度 | MetaClaw | 现有方法 |
|------|---------|--------|
| **适应速度** | 秒级技能注入 | 无或延迟高 |
| **服务连续性** | 零停机 | RL 更新常中断服务 |
| **知识沉淀** | 技能库持续积累，跨任务泛化 | 记忆冗余或技能静态 |
| **联合优化** | 技能 + 权重协同进化 | 单独优化一方 |
| **部署成本** | Proxy 架构，无需本地 GPU | 多需本地算力 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
#### （1）MetaClaw-Bench（新构建的持续代理基准）
- 包含 **934 个问题**，跨越 **44 个模拟工作日**
- 分为两部分：
  - **Part I**：30 天，346 题，侧重文件操作类任务（file-check）和选择题（multi-choice），测试端到端执行可靠性。
  - **Part II**：14 天，588 题，规则密集型任务流，强调对隐式行为规范的学习（如时间格式、备份协议等）。

#### （2）AutoResearchClaw（下游验证）
- 一个 **23 阶段全自动研究流水线**，从想法生成论文，涵盖文献检索、假设提出、代码生成、沙盒运行、同行评审等。
- 测试 MetaClaw 是否能在开放、长周期任务中泛化。

### 实验设置和评估指标
#### 主要评估指标：
| 指标 | 定义 |
|------|------|
| **Accuracy (%)** | 所有问题平均得分 |
| **File-check Completion Rate (%)** | 文件检查类任务完全通过自动化校验的比例 |
| **Composite Robustness Score**（AutoResearchClaw） | 加权综合评分：阶段完成率 (40%) + 重试减少 (30%) + 优化循环效率 (30%) |

#### 基线方法对比：
| 条件 | 描述 |
|------|------|
| **Baseline** | 原始模型，无任何适应机制 |
| **MetaClaw (Skills)** | 仅启用技能驱动快速适应 |
| **MetaClaw (Full)** | 完整框架：技能注入 + 机会主义 RL 微调（LoRA） |

使用的骨干模型：
- **GPT-5.2**
- **Kimi-K2.5**

所有条件使用相同 prompt 和工具集，以隔离 MetaClaw 组件的影响。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | Condition | Part I Acc. (%) | Part I Compl. (%) | Part II Acc. (%) | Part II Compl. (%) |
|-------|-----------|------------------|--------------------|------------------|---------------------|
| GPT-5.2 | Baseline | 41.1 | 14.7 | 44.9 | 58.4 |
| GPT-5.2 | MetaClaw (Skills) | 44.0 (+7.1%) | 17.1 | 49.1 (+9.4%) | 67.5 |
| Kimi-K2.5 | Baseline | 21.4 | 2.0 | 21.1 | 18.2 |
| Kimi-K2.5 | MetaClaw (Skills) | 28.3 (+32.2%) | 2.0 | 26.9 (+27.5%) | 33.8 |
| Kimi-K2.5 | MetaClaw (Full) | **40.6** | **16.5** (**8.25×**) | **39.6** | **51.9** (**+185%**) |

### 与基线方法的对比结果
- **技能注入显著提升准确率**：
  - 对较弱模型 Kimi-K2.5，相对提升高达 **32.2%**（Part I）。
  - 表明技能库能有效弥补模型先验知识不足。
- **完整框架解锁端到端完成能力**：
  - Kimi-K2.5 的 file-check completion 从 **2.0% → 16.5%**（**8.25 倍增益**）。
  - 接近 GPT-5.2 基线水平（41.1% vs 40.6%），说明 MetaClaw 可缩小不同模型间的实际差距。
- **AutoResearchClaw 上的表现（Table 2）**
  - 仅技能注入即可带来显著改进：
    - 阶段重试率 ↓ **24.8%**
    - 优化循环数 ↓ **40.0%**
    - 综合鲁棒性分数 ↑ **18.3%**
  - 证明技能泛化能力强，适用于非结构化、长流程任务。

### 消融实验结果
- **技能 vs 策略的作用分离**（Figure 3）：
  - **Skills-only**：显著提升 multi-choice 准确率，但对 file-check completion 几乎无改善。
  - **MetaClaw (Full)**：大幅提升 file-check completion，但 multi-choice 略降 → 表明策略已偏向执行可靠性。
- **RL 训练动态分析**：
  - 第 8 天出现明显拐点，之后 file-check 成功率迅速上升 → 验证了“技能先行，RL 后续”的双阶段学习模式。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **技能与策略应协同演化**：单一机制无法解决所有瓶颈；技能处理“知道怎么做”，策略处理“能否做好”。
2. ✅ **快速适应与慢速优化天然互补**：秒级技能注入 + 小时级 RL 微调构成正向反馈闭环。
3. ✅ **支持-查询数据分离至关重要**：防止 stale reward 污染是在线 meta-learning 成功的关键。
4. ✅ **轻量级架构可扩展至生产级模型**：基于 proxy 的设计无需本地 GPU，易于集成现有系统（如 OpenClaw）。
5. ✅ **方法具有强泛化性**：不仅在 CLI 任务中有效，在 AutoResearchClaw 这类复杂科研流程中也显著提升鲁棒性。

### 方法的局限性
- **依赖用户配置空闲信号**：OMLS 当前依赖睡眠时间、键盘不活动、Google Calendar 等信号，可能不适用于所有部署环境（如共享设备或隐私敏感场景）。
- **技能演化质量受限于 LLM Evolver**：若 evolver 自身能力不足，可能导致低质技能注入。
- **未处理任务遗忘问题**：虽然 focus 在持续学习，但未明确讨论 catastrophic forgetting 的缓解机制。

### 未来工作方向
- 探索更通用的空闲检测机制（如行为建模预测）。
- 引入技能评估与淘汰机制，避免技能库膨胀。
- 扩展至多模态 Agent（如视觉、语音交互）。
- 研究去中心化的联邦式 MetaClaw 架构，允许多个 Agent 共享技能经验。

---

> **总结一句话**：  
> **MetaClaw 实现了一个真正“越用越聪明”的 LLM Agent** —— 用户只需正常使用，系统就能自动从失败中提炼技能，并在后台悄悄变强，最终实现从“辅助工具”到“成长伙伴”的跃迁。

</details>

---

### 6. [WINFlowNets: Warm-up Integrated Networks Training of Generative Flow Networks for Robotics and Machine Fault Adaptation](https://arxiv.org/abs/2603.17301)

**Authors**: Zahin Sufiyan, Shadan Golestan, Yoshihiro Mitsuka, Shotaro Miwa, Osmar Zaiane  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.17301v1  

#### Abstract
Generative Flow Networks for continuous scenarios (CFlowNets) have shown promise in solving sequential decision-making tasks by learning stochastic policies using a flow and a retrieval network. Despite their demonstrated efficiency compared to state-of-the-art Reinforcement Learning (RL) algorithms...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**WINFlowNets: Warm-up Integrated Networks Training of Generative Flow Networks for Robotics and Machine Fault Adaptation**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **传统 CFlowNets 在动态机器人控制任务中的应用受限**，因其依赖对 **retrieval network (Gφ)** 的预训练（pre-training），而预训练数据在真实动态环境中往往不可得或不具代表性。
- 在 **Out-of-Distribution (OOD)** 场景下（如硬件故障、环境突变），预训练模型难以适应，导致 inflow/outflow 估计错误，影响策略学习效率和稳定性。

### 🚀 提出的新方法：**WINFlowNets**
提出一种新型 CFlowNets 框架 —— **WINFlowNets**，实现 **flow network (Fθ)** 和 **retrieval network (Gφ)** 的联合训练（co-training），无需预训练 Gφ。

#### 核心创新点：
1. **Warm-Up Phase（预热阶段）**  
   - 在训练初期，仅激活 Gφ 与环境交互，收集初始经验并训练其预测前驱状态的能力。
   - 学习率从低开始逐步提升，确保训练稳定。

2. **Dual-Training Phase（双训阶段）**  
   - Warm-Up 后，Fθ 与 Gφ 联合优化，共享一个 replay buffer。
   - Gφ 不断更新 inflow 估计，Fθ 利用这些估计进行 flow matching loss 优化，形成协同学习机制。

3. **Shared Replay Buffer（共享回放缓冲区）**  
   - 所有环境交互数据统一存储，供两个网络共同采样使用，增强数据利用率和一致性。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **无需预训练** | 消除对独立预训练数据集的依赖，适用于数据稀缺或快速变化的场景 |
| **更强的 OOD 适应能力** | 可持续从新环境中学习，适应执行器损坏（AD）、运动范围受限（ROM）等故障 |
| **更高的最终性能** | 在标准与故障环境下均取得更高平均奖励 |
| **更稳定的长期学习** | 尽管初期波动较大，但随训练推进逐渐收敛，不确定性降低 |

---

## 2. 核心实验方法和设置

### 🧪 数据集与环境
- 使用 **MuJoCo** 中的 `Reacher-v2` 机器人仿真环境（来自 Gymnasium）：
  - 两自由度机械臂，目标是让“指尖”触达空间中某一点。
  - 状态空间：11维（关节角度、速度、目标位置等）
  - 动作空间：连续二维向量（施加于两个关节的扭矩）

### ⚙️ 实验设置
- **总训练步数**：100万 timestep（1M）
- **评估频率**：每110k timestep评估一次（Warm-Up 阶段为100k）
- **评估方式**：10次 rollout 取平均奖励 ± 标准差
- **关键指标**：
  - **Average Reward**：反映学习过程的表现趋势
  - **Final Performance**：最后20次评估的平均奖励 ± std
  - **Sample Efficiency**：达到渐近性能所需的 timesteps（百万级）

### 📊 基线方法对比
| 类别 | 对比模型 |
|------|---------|
| **FlowNet 类** | CFlowNets（原始方法） |
| **RL 基线** | PPO, SAC, DDPG |
| **消融实验版本** | WINFlowNets-v1（无 Warm-Up）、WINFlowNets-v2（无共享 buffer） |

---

## 3. 主要实验结果和性能指标

### 📈 正常环境下的性能表现（Normal Environment）

| Model | Final Performance | Sample Efficiency (M) |
|-------|-------------------|------------------------|
| SAC | -7.89 ± 0.16 | 0.67 |
| PPO | -9.50 ± 0.37 | 3.39 |
| DDPG | -9.55 ± 0.44 | 5.20 |
| CFlowNets | **-3.70 ± 0.05** | **0.10** |
| **WINFlowNets** | **-2.39 ± 0.17** | 0.72 |

> ✅ **结论**：
> - WINFlowNets 达到 **最优最终性能**（reward 最高，负值越小越好）
> - CFlowNets 最快收敛（sample efficiency 最佳），但 WINFlowNets 稍慢但仍优于多数 RL 方法
> - RL 方法中 SAC 表现最稳，PPO 和 DDPG 收敛慢且不稳定

### ⚠️ 故障环境下的适应能力（Faulty Environments）

#### （1）Actuator Damage (AD)：执行器输出力矩降至 1/4
| Model | Final Performance | Sample Efficiency |
|-------|-------------------|-------------------|
| SAC | -9.69 ± 0.19 | **0.11** |
| CFlowNets | -9.01 ± 0.17 | 0.32 |
| **WINFlowNets** | **-8.25 ± 0.19** | 0.24 |

#### （2）Reduced ROM：Joint1 角度范围由 [-3.0, 3.0] 缩至 [-1.5, 1.5]
| Model | Final Performance | Sample Efficiency |
|-------|-------------------|-------------------|
| SAC | **-6.16 ± 0.03** | 0.37 |
| CFlowNets | -8.50 ± 0.08 | **0.11** |
| **WINFlowNets** | **-5.25 ± 0.12** | 0.12 |

> ✅ **结论**：
> - 在两种 OOD 故障场景下，**WINFlowNets 均取得最佳最终性能**
> - 虽然 SAC 在 AD 下收敛更快，但 WINFlowNets 最终超越
> - CFlowNets 在 ROM 故障下表现较差，说明其对预训练依赖性强，在分布偏移时泛化弱

### 🔬 消融实验（Ablation Study）

| 版本 | 描述 | 性能（asymptotic reward） |
|------|------|----------------------------|
| WINFlowNets-v1 | 无 Warm-Up，有共享 buffer | ~ -8.2 |
| WINFlowNets-v2 | 有 Warm-Up，无共享 buffer | ~ -4.6 |
| **Original WINFlowNets** | 完整设计（Warm-Up + Dual-Training + Shared Buffer） | **-2.39** |

> ✅ **结论**：
> - **Warm-Up 阶段显著提升性能稳定性**
> - **共享 replay buffer 极大促进协同学习**
> - 三者结合带来最大增益，验证了架构设计的有效性

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **WINFlowNets 成功实现了 Gφ 与 Fθ 的端到端联合训练**，摆脱了对预训练的依赖，更适合动态、开放世界任务。
2. 在 **正常与故障环境** 下，**WINFlowNets 均优于 CFlowNets 和主流 RL 算法（PPO/SAC/DDPG）**，尤其在最终策略质量上领先明显。
3. **通过 Warm-Up + Dual-Training + Shared Replay Buffer 的组合机制**，系统能够持续探索并适应 OOD 状态，避免陷入局部最优。
4. **尽管初期学习较慢且方差较高**，但随着训练推进，性能稳步上升，最终趋于稳定。

### ⚠️ 局限性
1. **计算资源消耗高**：
   - 双网络联合训练 + 共享大容量 replay buffer 导致 GPU 内存占用高于 CFlowNets 和 RL 方法。
2. **样本效率相对较低**：
   - 虽然最终性能好，但需要更多 timestep 才能达到渐近性能，不适合要求快速收敛的任务。
3. **超参数敏感**：
   - Warm-Up 时长、学习率调度、buffer 大小等对性能影响显著，需精细调参。
4. **安全性风险**：
   - 初始阶段策略不稳定，可能不适合人机共处（human-robot interaction）等安全关键场景。

### 🔮 未来工作方向
1. **开发轻量化版本**：
   - 引入模型压缩、分布式训练等技术降低资源消耗。
2. **改进 replay buffer 策略**：
   - 探索 **Prioritized Experience Replay**，优先采样高 TD-error 经验以加速学习。
3. **自适应 Warm-Up 机制**：
   - 设计动态判断 Warm-Up 结束时机的方法，减少不必要的延迟。
4. **迁移到真实机器人平台**：
   - 当前实验全部基于仿真，下一步将在物理机器人上验证泛化性和鲁棒性。
5. **与其他元学习框架融合**：
   - 结合 Meta-RL 思想，进一步提升跨任务迁移与少样本适应能力。

---

## ✅ 总结

**WINFlowNets** 是一项面向 **动态、容错机器人控制** 的重要进展。它通过引入 **Warm-Up 与 Dual-Training 双阶段机制** 和 **共享 replay buffer**，解决了 CFlowNets 依赖预训练的瓶颈，显著提升了在 OOD 和故障场景下的适应能力和最终性能。虽然存在资源开销大、收敛慢等问题，但其展现出的强大探索与持续学习潜力，使其成为未来智能机器人自主适应复杂环境的重要候选框架。

</details>

---

### 7. [BATQuant: Outlier-resilient MXFP4 Quantization via Learnable Block-wise Optimization](https://arxiv.org/abs/2603.16590)

**Authors**: Ji-Fu Li, Manyi Zhang, Xiaobo Xia, Han Bao, Haoli Bai, Zhenhua Dong, Xianzhi Yu  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.16590v1  

#### Abstract
Microscaling floating-point (MXFP) formats have emerged as a promising standard for deploying Multi-modal Large Language Models (MLLMs) and Large Language Models (LLMs) on modern accelerator architectures. However, existing Post-Training Quantization (PTQ) methods, particularly rotation-based techni...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BATQuant: Outlier-resilient MXFP4 Quantization via Learnable Block-wise Optimization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Post-Training Quantization (PTQ)** 方法（尤其是基于全局正交旋转的技术，如 QuaRot 和 SpinQuant）在应用于 **MXFP4** 格式时表现严重退化。根本原因在于：
- 全局旋转会将异常值（outliers）的能量跨量化块传播，破坏局部动态范围的缩放。
- 引入双峰分布（bimodal distribution），导致有限的量化位宽利用率低下。

这些问题使得当前方法难以实现高效的 W4A4 低比特量化，尤其是在多模态大模型（MLLMs）上。

### 提出了什么新方法或新思路
本文提出 **BATQuant**（Block-wise Affine Transformation Quantization），其核心思想是：
- **Block-wise Affine Transformation (BAT)**：将仿射变换限制在与 MXFP 量化粒度对齐的块内（例如每 32 个元素一个块），避免跨块能量转移。
- 放弃正交性约束，学习最优的仿射矩阵以优化分布形态。
- 引入 **Global and Private Kronecker (GPK) 分解** 来压缩参数量，提升存储和计算效率。
- 设计 **Block-wise Learnable Clipping** 动态抑制残余异常值。

### 相比现有方法的优势
| 特性 | BATQuant | 传统方法（如 QuaRot, BRQ） |
|------|---------|-----------------------------|
| 变换粒度 | Block-wise（对齐 MXFP） | Global 或 Block-wise Hadamard |
| 能量传播 | 阻止跨块传播 | 易引起跨块干扰 |
| 分布控制 | 单峰紧凑分布 | 易产生双峰分布 |
| 参数效率 | GPK 显著降低参数量 | 存储开销高 |
| 性能稳定性 | 在 W4A4 下仍保持高性能 | 在 W4A4 下性能崩溃 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
#### 多模态基准（用于 MLLM Qwen3-VL-8B-Instruct）
- **MME**：综合多模态理解能力评测
- **OCR-Bench**：OCR 文本识别能力
- **DocVQA**：文档图像问答
- **RealWorldQA**：现实场景空间推理
- **VLMBlind**：几何图形感知任务（线、圆等基本图元）

#### 语言模型基准（用于 LLM Qwen3-8B）
- **非推理任务**：
  - PIQA, Winogrande, Hellaswag, ARC-Easy, ARC-Challenge
- **复杂推理任务**：
  - GSM8K（小学数学题）
  - MATH-500, AIME24/AIME25（高级数学竞赛题）
  - GPQA-D（博士级“谷歌无法查到”的难题）

### 实验设置和评估指标
- **量化配置**：采用 `W{bits}A{bits}KV{bits}` 表示法，重点测试：
  - `W4A8KV16`（轻度压缩）
  - `W4A4KV16`（激进压缩，权重和激活均为4bit）
  - `W4A8KV8/KV4`（KV缓存压缩）
- **评估指标**：
  - 各任务准确率（Accuracy）
  - 相对于 BF16 精度模型的恢复率（Recovery Rate %）
- **校准集**：
  - LLM：Numina-Math-1.5 上自生成文本（128条 × 2048长度）
  - MLLM：GQA 数据集中采样 128 图文对
- **训练细节**：
  - 使用 AdamW 优化器，初始学习率 2e-3，cosine 衰减
  - 训练 5 轮，batch size=4
  - GPK 中 $g_1=8$, $g_2=4$

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **RTN** | 基线 | 直接舍入，无优化 |
| **QuaRot / SpinQuant** | Rotation-based | 全局正交旋转，适用于 INT4 |
| **BRQ** | Block-wise Rotation | 局部 Hadamard 变换 |
| **FlatQuant** | Affine-based | 全局仿射变换平滑分布 |
| **SmoothQuant** | Scale Migration | 将激活异常迁移到权重 |
| **GPTQ** | Hessian-based | 利用近似二阶梯度进行权重量化 |

所有方法均集成 GPTQ 进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 配置 | 方法 | 平均恢复率 (%) |
|------|------|--------|----------------|
| Qwen3-VL-8B-Instruct | W4A8KV16 | BATQuant | **99.29%** |
| Qwen3-VL-8B-Instruct | W4A4KV16 | BATQuant | **96.43%** |
| Qwen3-8B | W4A8KV16 | BATQuant | **99.34%** |
| Qwen3-8B | W4A4KV16 | BATQuant | **95.84%** |

> ✅ 在 `W4A8KV16` 下接近无损；在极端 `W4A4KV16` 下仍恢复超过 95% 性能。

### 与基线方法的对比结果
#### 在 MLLM 上（Table 2）
- 在 `W4A4KV16` 下，BATQuant 比最强基线 **FlatQuant** 高出 **1.64% 绝对值**（96.43% vs 94.79%）。
- 在 `W4A8KV16` 下唯一达到 <1% 性能损失的方法。

#### 在 LLM 推理任务上（Table 3）
| 配置 | 方法 | Avg. Recovery (%) |
|-------|------|--------------------|
| W4A8KV16 | BATQuant | **97.46%**（优于 GPTQ 0.92%） |
| W4A4KV16 | BATQuant | **92.45%**（大幅领先第二名 2.35%） |
| W4A8KV8 | BATQuant | **96.22%**（领先 GPTQ 1.68%） |

> 🔺 在复杂推理任务中优势更明显，说明其对误差累积具有更强鲁棒性。

### 消融实验结果（Table 4 & Figure 6–7）
#### 组件消融（W4A4KV16）
| 组件组合 | Qwen3-8B 平均精度 | Qwen3-VL-8B Recovery |
|----------|---------------------|------------------------|
| 仅全局变换 | 68.24% | 95.59% |
| + Block-wise Affine | → 68.70% | → 96.43% |
| + Block-wise Clipping | → 68.70% | → 96.43% |

✅ 两个模块均有显著增益。

#### 块大小影响（Figure 6）
- 最优性能出现在 **affine block size = 32**（即等于 MXFP 量化粒度）。
- 若小于 32 → 局部平滑不足；
- 若大于 32 → 跨块能量泄露，性能下降。

#### GPK 参数敏感性分析（Figure 7）
- 当 $g_1=8$（共享矩阵大小）时取得最佳平衡：
  - 参数数仅为 2,112（相比原始 131,072 减少 >98%）
  - 性能达到峰值
- $g_1$ 过大或过小都会导致性能下降。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **格式对齐至关重要**：只有当变换粒度与硬件量化块（如 MXFP 的 32-element block）严格对齐时，才能有效防止异常值污染。
2. **仿射优于旋转**：放弃正交性后，可自由优化分布形状，避免双峰问题，更适合浮点格式。
3. **GPK 实现高效部署**：通过共享基础 + 私有适配的设计，在极低参数下实现高性能。
4. **跨模态泛化能力强**：在同一框架下同时在 LLM 和 MLLM 上取得 SOTA，证明方法通用性强。

### 方法的局限性
- 当前设计依赖于固定的 block size（如 32），可能不适应所有硬件架构。
- GPK 中 $g_1$ 和 $g_2$ 的选择需要调参，缺乏自动化搜索机制。
- 未探索训练后微调（PTQ+FT）联合优化路径。

### 未来工作方向
- 扩展至其他微缩放格式（如 MXFP8, NVFP）。
- 结合量化感知训练（QAT）进一步提升极限压缩性能。
- 自动化 GPK 结构搜索（NAS for GPK）。
- 探索动态块划分策略以应对不同层的统计特性差异。

---

> 💡 **总结一句话**：  
> BATQuant 通过 **block-wise 仿射变换 + GPK 参数压缩 + learnable clipping**，首次实现了在 **MXFP4** 下稳定高效的 W4A4 量化，在 MLLM 和 LLM 上均达到新的 SOTA，为下一代 AI 加速器上的大模型部署提供了实用解决方案。

</details>

---

### 8. [Variational Rectification Inference for Learning with Noisy Labels](https://arxiv.org/abs/2603.17255)

**Authors**: Haoliang Sun, Qi Wei, Lei Feng, Yupeng Hu, Fan Liu, Hehe Fan, Yilong Yin  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.17255v1  

#### Abstract
Label noise has been broadly observed in real-world datasets. To mitigate the negative impact of overfitting to label noise for deep models, effective strategies (\textit{e.g.}, re-weighting, or loss rectification) have been broadly applied in prevailing approaches, which have been generally learned...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Variational Rectification Inference for Learning with Noisy Labels》总结**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对**Learning with Noisy Labels (LNL)** 中的两个关键挑战：
- **模型坍缩（model collapse）**：在基于 Monte Carlo (MC) 近似的概率元学习方法中，由于采样数有限，后验分布容易退化为狄拉克函数（Dirac delta），导致模型失去不确定性建模能力，泛化性能下降。
- **忽略平滑性假设（smoothness assumption）**：现有方法过度依赖可能错误的标签进行损失修正，而未充分利用特征空间中的判别性信息。

### **提出了什么新方法或新思路**
提出 **Variational Rectification Inference (VRI)**，一种将损失修正过程建模为**变分推断（variational inference）问题**的新框架：
- 将**修正向量（rectifying vector）** 视为隐变量，构建**分层贝叶斯模型（hierarchical Bayes）**。
- 引入一个**amortization meta-network** 来近似其条件后验分布 $ q_\phi(\mathbf{v}|x,y) $。
- 设计一个**先验网络（prior network）** $ H_\omega $，以特征 $ x $ 为输入，生成先验分布 $ p(\mathbf{v}|x) $，通过 KL 正则项约束后验，避免模型坍缩。
- 在元学习框架下，采用**双层优化（bi-level optimization）** 学习所有参数。

### **相比现有方法的优势**
- ✅ **避免模型坍缩**：通过显式的 KL 正则项防止后验退化，保持模型的随机性和鲁棒性。
- ✅ **增强泛化能力**：利用先验网络强制模型从特征中学习平滑的修正策略，减少对噪声标签的依赖。
- ✅ **高效且可扩展**：仅需少量采样（如 $ k=1 $ 或 $ 2 $）即可达到甚至超越 MC 方法的性能，训练开销接近确定性方法。
- ✅ **适用于开放集噪声（open-set noise）**：在包含分布外样本的场景下表现优异。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **合成噪声数据集**：
  - **CIFAR-10**, **CIFAR-100**：用于测试不同噪声类型下的性能。
- **真实世界噪声数据集**：
  - **Clothing1M**：约 100 万张图像，标签噪声率 ~38.46%。
  - **Food-101N**：约 31 万张图像，噪声率 ~20%。
- **开放集噪声数据集**：
  - **ANIMAL-10N**：人类标注的混淆动物图像，噪声率 ~8%。
  - **CIFAR-80N**：人工构造的开放集噪声数据集（CIFAR-100 中前 80 类为 in-distribution，后 20 类为 out-of-distribution）。

### **实验设置和评估指标**
- **噪声类型**：
  - Flip noise（类别间翻转）
  - Uniform noise（均匀随机翻转）
  - Instance-dependent (ID) noise（实例相关噪声）
  - Real-world noise（真实世界噪声）
  - Open-set noise（含分布外样本）
- **评估指标**：**Top-1 测试准确率（Test Accuracy %）**
- **元数据集（meta-data）**：
  - CIFAR-10/100：随机选取 1,000 个干净样本作为元数据。
  - Clothing1M 和 Food-101N：使用验证集作为元数据。
- **骨干网络**：ResNet-18, ResNet-32, ResNet-34, ResNet-50, Wide ResNet-28-10, VGG19。

### **基线方法对比**
- **非元学习方法**：
  - `Baseline`（直接训练）
  - `DivideMix`, `ELR`, `Co-teaching`, `Peer Loss`, `CDR`, `Late Stopping`
- **元学习方法**：
  - `MW-Net`, `PMW-Net`, `MSLC`, `FSR`, `FasTEN`, `EMLC`, `FaMUS`
- **MC 近似方法**：
  - `WarPI`（MC-based rectification）

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### **表：CIFAR-10 上 Flip 40% 噪声下的测试准确率**
| 方法 | 准确率 (%) |
|------|-----------|
| MW-Net | 87.54 |
| MLC | 88.97 |
| FSR | 90.20 |
| WarPI (MC) | 89.87 |
| **VRI (Ours, ResNet-32)** | **91.21** |
| **VRI (Ours, ResNet-34)** | **93.27** |

> ✅ **VRI 超越所有基线，在 ResNet-34 上达到 93.27%，显著领先。**

#### **表：CIFAR-100 上 Uniform 40% 噪声下的测试准确率**
| 方法 | 准确率 (%) |
|------|-----------|
| ELR | 60.05 |
| MSLC | 60.25 |
| FSR | 62.79 |
| FasTEN | 63.82 |
| **VRI (Ours)** | **68.92** |

> ✅ **VRI 在 CIFAR-100 上提升明显，较次优方法高出近 5%。**

#### **表：真实世界噪声下的性能**
| 数据集 | 方法 | 准确率 (%) |
|--------|------|-----------|
| Clothing1M | MW-Net | 73.72 |
| | WarPI | 74.98 |
| | **VRI (Ours)** | **75.19** |
| Food-101N | WarPI | 85.91 |
| | **VRI (Ours)** | **86.24** |

> ✅ **在真实噪声下仍取得 SOTA 性能，分别提升 0.21% 和 0.33%。**

#### **表：开放集噪声（CIFAR-80N）下的平均准确率**
| 方法 | Flip 40% |
|------|---------|
| MoPro | 60.22 |
| Jo-SRC | 53.03 |
| PNP-soft | 61.23 |
| USDNL | — |
| **VRI (Ours)** | **64.71** |

> ✅ **在 Flip 40% 下比当前最优方法高 3.48%。**

---

### **消融实验结果**

#### **采样数 $ k $ 对性能的影响（CIFAR-10 Flip 40%）**
| 方法 | $ k $ | 准确率 (%) | 时间 (min/epoch) |
|------|-------|------------|------------------|
| MC | 1 | 88.23 | 2.17 |
| MC | 3 | 89.45 | 4.32 |
| MC | 5 | 89.87 | 7.04 |
| **VRI** | **1** | **90.20** | **2.20** |

> ✅ **VRI 仅用 $ k=1 $ 即超越 MC 方法 $ k=5 $ 的性能，效率更高。**

#### **是否使用贝叶斯建模（VRI vs Non-Bayesian VRI）**
| 噪声设置 | 方法 | 准确率 (%) |
|----------|------|-----------|
| CIFAR-10 Unif. 40% | VRI | 91.29 |
| | Non-Bayesian VRI | 89.27 |
| CIFAR-100 Inst. 40% | VRI | 68.17 |
| | Non-Bayesian VRI | 64.92 |

> ✅ **贝叶斯建模带来显著增益，证明变分推断的有效性。**

#### **元数据数量敏感性分析**
- 当元数据仅为 100 个时，VRI 在 CIFAR-10 Flip 40% 下仍能达到 **91.07%** 的高精度。
- 性能随元数据增加而提升，尤其在 Flip 噪声下更明显。

#### **超参数 $ \lambda $ 敏感性**
- 最佳值为 $ \lambda = 0.001 $，过大或过小均会导致性能下降。
- 验证了 KL 正则项对稳定训练的重要性。

---

## 4. **关键结论和发现**

### **主要发现**
1. **变分建模有效缓解模型坍缩**：通过引入先验网络和 KL 正则项，VRI 成功避免了 MC 方法中常见的后验退化问题。
2. **特征驱动的修正更鲁棒**：先验网络迫使模型从特征中学习平滑的修正向量，减少对噪声标签的依赖，符合平滑性假设。
3. **高效且高性能**：即使只采样一次（$ k=1 $），VRI 也能达到甚至超越需要多次采样的 MC 方法，训练效率高。
4. **在开放集噪声下表现卓越**：VRI 是少数能在开放集噪声下持续取得领先的方法之一。
5. **无需手动调参**：整个修正策略由数据驱动学习，避免了传统方法中复杂的超参数调优。

### **方法的局限性**
- **依赖元数据**：虽然可通过样本选择构造伪元数据，但在极端噪声或无任何干净样本的情况下性能会下降。
- **额外网络开销**：尽管训练成本可控，但仍需维护 meta-network 和 prior network，增加了模型复杂度。
- **对元数据质量敏感**：若元数据本身存在噪声或偏差，可能影响整个学习过程。

### **未来工作方向**
- 探索**完全无监督的元数据构建策略**，例如结合自监督学习或聚类方法。
- 将 VRI 扩展到**多标签学习、弱监督学习、联邦学习**等更广泛场景。
- 研究如何将 VRI 与 **contrastive learning** 或 **semi-supervised learning** 更深度结合，进一步提升鲁棒性。
- 探索**动态调整采样数 $ k $** 或 **自适应 KL 权重 $ \lambda $** 的机制。

---

> 🔚 **总结**：  
> VRI 通过将损失修正建模为**变分推断问题**，成功解决了现有概率元学习方法中的**模型坍缩**难题，并在多种噪声类型下实现了**SOTA 性能**。其实验充分、理论严谨，是 LNL 领域的一项重要进展。代码已开源：[https://github.com/haolsun/VRI](https://github.com/haolsun/VRI)。

</details>

---

### 9. [Fanar 2.0: Arabic Generative AI Stack](https://arxiv.org/abs/2603.16397)

**Authors**: FANAR TEAM, Ummar Abbas, Mohammad Shahmeer Ahmad, Minhaj Ahmad, Abdulaziz Al-Homaid, Anas Al-Nuaimi, Enes Altinisik, Ehsaneddin Asgari, Sanjay Chawla, Shammur Chowdhury, Fahim Dalvi, Kareem Darwish, Nadir Durrani, Mohamed Elfeky, Ahmed Elmagarmid, Mohamed Eltabakh, Asim Ersoy, Masoomali Fatehkia, Mohammed Qusay Hashim, Majd Hawasly, Mohamed Hefeeda, Mus'ab Husaini, Keivin Isufaj, Soon-Gyo Jung, Houssam Lachemat, Ji Kim Lucas, Abubakr Mohamed, Tasnim Mohiuddin, Basel Mousi, Hamdy Mubarak, Ahmad Musleh, Mourad Ouzzani, Amin Sadeghi, Husrev Taha Sencar, Mohammed Shinoy, Omar Sinan, Yifan Zhang  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.16397v1  

#### Abstract
We present Fanar 2.0, the second generation of Qatar's Arabic-centric Generative AI platform. Sovereignty is a first-class design principle: every component, from data pipelines to deployment infrastructure, was designed and operated entirely at QCRI, Hamad Bin Khalifa University. Fanar 2.0 is a sto...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fanar 2.0: Arabic Generative AI Stack

## 1. 论文的主要贡献和创新点

### 解决的问题
Fanar 2.0 旨在解决阿拉伯语在生成式 AI 领域面临的多重挑战：
- **数据稀缺性**：尽管阿拉伯语拥有超过 4 亿母语者，但其在互联网文本中的占比仅为约 0.5%，导致高质量训练数据严重不足。
- **语言复杂性**：阿拉伯语具有根词模式形态学、广泛的方言差异以及作为伊斯兰教礼拜语言的文化宗教敏感性，这些都对通用模型提出了特殊要求。
- **主权与控制**：依赖外部 AI 提供商存在访问风险、文化价值观不匹配和技术差距扩大的隐患。

### 提出的新方法与新思路
Fanar 2.0 是一个完全自主设计、构建和运营的第二代阿拉伯语中心生成式 AI 平台，其核心创新在于“资源受限下的卓越”（resource-constrained excellence）策略：

- **主权优先的设计原则**：整个平台的所有组件（从数据管道到部署基础设施）均在卡塔尔计算研究所（QCRI）内部完成，不依赖任何外部 AI 提供商，确保了数据治理和文化对齐的完全控制。
- **质量优于数量的数据策略**：放弃单纯扩大数据规模，转而采用精心策划的高质量数据（仅约 1200 亿 token），并通过三种不同的预训练配方（data recipes）、基于配方的退火（recipe-based annealing）和模型合并（model merging）来最大化收益。
- **全栈开放权重模型**：所有组件均基于公开发布的模型权重进行构建，而非从零开始训练，这极大地降低了计算成本，并将团队精力集中在无法继承的阿拉伯语文化和语言适应上。
- **丰富的专用能力堆栈**：超越单一 LLM，构建了一个包含多个专用模型家族的生态系统，以应对不同模态和任务的需求。

### 相比现有方法的优势
- **高效性**：在仅使用 256 块 NVIDIA H100 GPU 的有限算力下，实现了与更大规模系统相媲美的性能。
- **文化对齐**：通过专门设计的组件（如 FanarGuard, Oryx-IG, Fanar-Sadiq）深度嵌入了阿拉伯和伊斯兰文化规范，解决了通用模型在此方面的不足。
- **架构先进性**：引入了多层协调器（orchestrator）、代理工具调用框架（agentic tool-calling）等现代架构，提升了系统的灵活性和功能性。

## 2. 核心实验方法和设置

### 使用的数据集
- **预训练数据**：由约 1200 亿高质量 token 组成的精选语料库，包含三个独立的“配方”：
  1. **配方1**：来自 Fanar 1.0 数据集的手动策划高质量子集。
  2. **配方2**：上述高质量数据与 FineWeb-EDU 和 ArabicWeb-EDU 的结合。
  3. **配方3**：高质量数据与 FineWeb-EDU 及其由内部翻译系统生成的阿拉伯语译文的平行数据。
- **后训练数据**：通过选择性过滤公共数据集和受控合成生成的方式构建，特别加强了阿拉伯语推理痕迹、文化对齐和长上下文适应。
- **专用模型数据**：
  - **FanarGuard**：在 46.8 万标注的阿拉伯语和英语提示-响应对上训练。
  - **Aura-STT-LF**：在公开的阿拉伯语（QASR, MGB3, MGB5, GALE, Common Voice）和英语（GigaSpeech, LibriSpeech, Common Voice）语料库上训练，并进行了数据增强。
  - **Oryx-IG**：通过基于分类法的爬取，从 Google Images 和 Flickr 收集了超过 200 万张原始图像，经过严格过滤后保留了约 48 万张高质量图像-文本对。
  - **Oryx-IVU**：构建了约 6200 万训练样本的多模态语料库，涵盖阿拉伯/伊斯兰文化内容、字体/书法识别、物体检测与定位、通用字幕和纯文本指令。
  - **Fanar-Diwan**：在从 AlDiwan.net 爬取并清洗后的 11.8 万首古典阿拉伯诗歌上微调。

### 实验设置和评估指标
- **模型基础**：核心 LLM Fanar-27B 基于 Gemma-3-27B 进行持续预训练。
- **训练策略**：采用分阶段的 recipe-based annealing 和模型合并（linear interpolation）。
- **评估指标**：
  - **世界知识**：MMMLU/Ar, ArabicMMLU, MMLU
  - **阿拉伯语能力**：Nahw-MCQ, AraLingBench, Al-Mieyar
  - **方言理解**：Belebele, AraDiCE
  - **文化意识**：ACVA, PalmX
  - **数学推理**：GSM8K, MATH500, AIME24, AMC23
  - **指令遵循**：MT-Bench, IFEval
  - **安全性**：aiXamine（涵盖 9 个维度）
  - **语音识别**：WER (Word Error Rate)
  - **机器翻译**：BLEU
  - **图像生成**：基于多模态 Judge 模型（Gemini）的自动化评分。

### 基线方法对比
论文将 Fanar-27B 与一系列代表性的阿拉伯语中心和多语言模型进行了比较，包括：
- **阿拉伯语中心模型**：Fanar-1-9b-instruct, ALLaM-7B-Instruct, Karnak, AceGPT-v2, Jais-2-70B-Chat
- **多语言模型**：Gemma-3-27b-it, Qwen3-32b, Llama-3.3-70B-Instruct

## 3. 主要实验结果和性能指标

### 关键性能数据
- **基准测试提升**（相较于 Fanar Prime 9B）：
  - 阿拉伯世界知识 (MMMLU/Ar)：+9.1 分
  - 一般阿拉伯语 (ArabicMMLU)：+7.3 分
  - 英语能力 (MMLU)：+7.6 分
  - 方言理解 (Belebele)：+3.5 分
- **效率**：在使用比 Fanar 1.0 少约 8 倍的预训练 token 的情况下，取得了显著的性能提升。

### 与基线方法的对比结果
- **阿拉伯语任务**：Fanar-27B 在大多数阿拉伯语、方言和文化知识任务中，在同等规模（27B）的模型中表现最佳。
- **文化对齐**：在 Al-Mieyar 基准测试中，Fanar-27B 在所有子类别上均取得最高分。
- **方言理解**：在 Belebele 和 AraDiCE 等方言基准测试中，Fanar-27B 表现优于所有其他被评估的模型。
- **安全性**：FanarGuard 在阿拉伯语安全基准测试上达到了最先进的性能，其平均 F1 分数为 0.82，同时参数量（4B）远小于竞争对手（7-8B）。
- **机器翻译**：FanarShaheen 在 AraBench 基准测试上实现了最高的平均 BLEU 分数，尤其在 TED 演讲、医学、新闻、教育和联合国文件等领域优势明显。

### 消融实验结果
- **退火（Annealing）的重要性**：实验证明，退火阶段对阿拉伯语性能至关重要。例如，配方3 在退火后，其 OALL 性能从 57.33% 提升至 65.59%（+8.26 分）。
- **模型合并的有效性**：最终合并的模型（66.62%）性能超过了任何单个经过退火的检查点（最高为 65.59%），证明了模型合并可以带来超越单一训练路径的增益。
- **后训练数据质量**：更严格的基于评分标准的过滤和文化适应调整，显著提升了模型的对齐度和性能。

## 4. 关键结论和发现

### 主要发现
1.  **质量优于数量是可行的**：在资源受限的情况下，通过精心策划高质量数据、采用先进的训练策略（如配方退火和模型合并），可以构建出与大规模系统竞争的高质量 LLM。
2.  **主权 AI 是可实现的**：一个国家可以在不依赖外部巨头的情况下，建立一个全面、高性能且文化对齐的主权 AI 平台。
3.  **专用化是关键**：针对特定需求（如文化、宗教、方言）构建专用模型（如 Fanar-Sadiq, Oryx-IG）比依赖单一通用模型更能有效解决问题。
4.  **开放权重是主权的赋能器**：利用开放权重的基础模型可以大幅降低进入门槛，使资源有限的团队能够专注于最具价值的文化和语言适应工作。

### 方法的局限性
- **持续预训练的瓶颈**：随着模型容量的增加，持续预训练在小规模 token 预算下的收益正在显现天花板。
- **多轮对话安全**：当前的安全和对齐评估主要集中在单轮交互上，对于多轮对话中的对齐漂移和渐进式越狱攻击的防御仍需加强。
- **架构限制**：当前架构继承了基础模型的密集 Transformer 设计，可能在扩展性和效率上存在固有瓶颈。

### 未来工作方向 (Fanar 3.0)
- **探索 MoE 架构**：研究从头开始训练的混合专家（Mixture-of-Experts, MoE）架构，以在可控的推理成本下实现更大的参数容量。
- **大规模高质量语料库建设**：系统性地投资于收集和策划一个覆盖更多领域、语域和方言的超大规模、高质量阿拉伯语语料库。
- **强化多轮安全**：将多轮安全作为重点研究方向，开发更丰富的对齐数据集和评估框架，以确保模型在长时间、对抗性对话中保持对齐。

</details>

---

### 10. [The 1/W Law: An Analytical Study of Context-Length Routing Topology and GPU Generation Gains for LLM Inference Energy Efficiency](https://arxiv.org/abs/2603.17280)

**Authors**: Huamin Chen, Xunzhuo Liu, Yuhan Liu, Junchen Jiang, Bowei He, Xue Liu  
**Category**: cs.DC  
**Published**: 2026-03-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.17280v1  

#### Abstract
How many tokens can a GPU inference cluster deliver per watt? Across deployments of identical hardware, the answer varies by 40x -- not because of software inefficiency, but because of the serving context window. We derive the 1/W law: tokens per watt halves every time the context window doubles. A ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The 1/W Law: An Analytical Study of Context-Length Routing Topology and GPU Generation Gains for LLM Inference Energy Efficiency

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文系统性地研究了**大语言模型（LLM）推理过程中的能源效率问题**，特别是探究了在相同硬件条件下，为何不同部署场景下的 `tokens per watt`（tok/W）差异可达 **40×**。作者指出，这一巨大差异并非源于软件效率低下，而是由**上下文长度（context window）** 和 **路由拓扑（routing topology）** 所主导。

### 提出的新方法与新思路
- **提出“1/W Law”**：  
  核心发现是：**每瓦特处理的 token 数量（tok/W）随上下文窗口加倍而减半**。即：  
  > **tok/W ∝ 1 / W**  
  这是因为更大的 context window 占用更多 KV-cache 内存，导致并发序列数（concurrent sequences）下降，从而降低吞吐量；而 GPU 功耗基本保持不变。

- **引入 FleetOpt 路由策略**：  
  提出一种基于上下文长度的双池路由架构（two-pool context-length routing），将短请求路由到小 context pool，长请求保留给大 context pool。该策略能显著提升 fleet-level 的能源效率。

- **揭示三个独立的能效杠杆**：
  1. **Routing Topology**（路由拓扑）
  2. **GPU Generation**（硬件代际升级）
  3. **Model Architecture**（模型架构，如 MoE）

  并证明这三者的影响是**正交且可乘的**，而非简单叠加。

### 相比现有方法的优势
- 不依赖新硬件实验，而是通过分析建模（analytical modeling）整合多个已有工具（logistic power model、roofline model、fleet simulator），得出普适性规律。
- 首次量化了 **topology 对能效的影响超过 GPU 升级本身**（2.5× vs 1.7×），挑战了“买新卡就能省电”的直觉。
- 强调 **context-length routing 是低成本高回报的优化手段**，无需更换硬件即可实现显著节能。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Azure Azi LLM Inference Trace** [Patel et al., 2024]：真实生产负载，89% 请求 ≤ 4K tokens。
- **LMSYS-Chat-1M** [Zheng et al., 2023]：公开对话数据集，用于模拟分散型负载。

### 实验设置
- **目标 SLO**：P99 TTFT（Time to First Token）≤ 500ms，请求速率 λ = 1,000 req/s。
- **模型**：主要使用 `Llama-3.1-70B`（TP=8, fp16），部分实验涉及 `Qwen3-235B-A22B`、`DeepSeek-V3` 等。
- **GPU 类型**：
  - H100-SXM5（80GB）
  - B200-SXM（156GB）
  - H200-SXM
  - GB200-NVL
- **KV-cache 设置**：采用 tensor-parallel sharding（TP=8），每个 GPU 存储一个 KV head。

### 评估指标
- **tok/W**（tokens per watt）：核心能效指标，衡量每焦耳能量产生的输出 token 数。
- **Fleet-level tok/W**：加权平均，考虑不同 pool 的 GPU 数量、利用率和功耗。
- **vs H100 Homo baseline**：相对于 H100 同构集群的相对增益。

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Homogeneous (Homo)** | 所有请求统一发送至 64K context pool，作为基准 |
| **Pool Routing** | 两池结构，按 prompt 长度分流（如 <4K → short pool） |
| **FleetOpt** | 最优分割边界（y*）下的 context-length routing，最大化 tok/W |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| 工作负载 | 架构 | GPU | tok/W | 相对提升 |
|--------|-------|-----|--------|----------|
| Azure | Homogeneous | H100 | 5.58 | — |
| Azure | FleetOpt | H100 | 14.08 | **+152%** |
| Azure | Homogeneous | B200 | 9.74 | +75% |
| Azure | FleetOpt | B200 | **23.71** | **+325%** |

> ✅ **FleetOpt 在 H100 上带来约 2.5× tok/W 提升**  
> ✅ **B200 相比 H100 提供约 1.7× 提升**  
> ✅ **组合使用可达 2.5 × 1.7 ≈ 4.25× 提升**

### 与其他方法的对比
- **Semantic Routing vs Context Routing**（Table 4）：
  - 语义路由（small model for simple tasks）在 short pool 效率略低（6.24 vs 8.77 tok/W）；
  - long pool 性能完全受限于 KV-cache，并无优势；
  - 结论：**context-length routing 更高效且物理上更可预测**。

- **MoE 模型表现**（Table 2）：
  - `Qwen3-235B-A22B`（22B active）在 H100 上达 **37.8 tok/W**（8K context），是 Llama-70B 的 **5.1×**；
  - 原因：weight-streaming 时间仅与激活参数相关，远低于 dense 模型；
  - 注意：此为**理论上限**，未计入 MoE dispatch 开销（可能增加 10ms 延迟）。

### 消融实验与敏感性分析
- **GPU generation 影响独立于 topology**：
  - topology gain: H100 上 2.52×，B200 上 2.44× → 几乎一致
  - generation gain: Homogeneous 下 1.75×，FleetOpt 下 1.68× → 几乎一致
  - 表明两个维度**互不干扰，收益相乘**

- **B200 在长 context 下优势缩小**：
  - 4K context：B200 比 H100 快 1.75×
  - 64K context：仅快 1.49×
  - 原因：idle power 占比上升（B200 P_idle=430W），低并发时浪费更大

---

## 4. 关键结论和发现

### 主要发现
1. **1/W Law 成立**：  
   tok/W 随 context window 加倍而减半，根本原因是 KV-cache 容量限制了并发度，而功耗几乎不变。

2. **Routing Topology 是最强能效杠杆**：  
   - FleetOpt 可带来 **~2.5× tok/W 提升**
   - 超过从 H100 升级到 B200 的收益（~1.7×）

3. **Topological Gain 与 Hardware Gain 正交**：  
   两者可**相乘**，联合优化可达 **4.25×** 能效提升。

4. **MoE 架构天然具备能效优势**：  
   因为 weight-streaming 时间只取决于 active parameters，可在所有 context 长度下受益。

5. **工作负载分布决定最优策略**（Table 6）：
   - **Short-dominant**（>80% ≤8K）：推荐 FleetOpt + B200
   - **Mixed**：Pool routing + H200/B200
   - **Long-dominant**：直接用 B200/GB200 支持大 context
   - **MoE-capable**：优先启用 MoE + short pool

### 方法的局限性
- **B200/H200 数据为预测值**（FAIR quality），缺乏实测 P(b) 曲线验证（±20% 不确定性）。
- 所有分析基于**稳态流量假设**，未考虑突发流量或 diurnal patterns。
- 使用 **continuous batching** 模型，忽略 prefill-decode 干扰、head-of-line blocking 等现实影响。
- **仅考虑 decode 阶段能耗**，prefill 能耗被排除在外（对长 prompt 场景会高估效率）。
- 当前仅支持 **two-pool topology**，未探索多级分层或动态调整机制。

### 未来工作方向
1. **实测 B200/H200 的 P(b) 曲线**，校准 power model 至 HIGH quality。
2. **结合 Splitwise 式 prefill-decode disaggregation**，进一步解耦能耗。
3. **扩展为 K≥3 的 multi-pool topology optimization**，构建混合整数规划模型。
4. **开发 adaptive topology 控制器**，根据实时请求分布动态调整 split boundary。
5. **纳入碳感知优化**（carbon-aware），最小化 gCO₂/token 或 $/token。
6. **研究 speculative decoding 对 batch 分布和 tok/W 的影响**。

---

> 📌 **一句话总结**：  
> **控制 context window 比购买新 GPU 更能提升 LLM 推理能效——通过 FleetOpt 类型的 context-length routing，可实现高达 2.5× 的节能，与硬件升级效果正交且可乘。**

</details>

---

### 11. [Complementary Reinforcement Learning](https://arxiv.org/abs/2603.17621)

**Authors**: Dilxat Muhtar, Jiashun Liu, Wei Gao, Weixun Wang, Shaopan Xiong, Ju Huang, Siran Yang, Wenbo Su, Jiamang Wang, Ling Pan, Bo Zheng  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.17621v1  

#### Abstract
Reinforcement Learning (RL) has emerged as a powerful paradigm for training LLM-based agents, yet remains limited by low sample efficiency, stemming not only from sparse outcome feedback but also from the agent's inability to leverage prior experience across episodes. While augmenting agents with hi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Complementary Reinforcement Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **样本效率低（Sample Inefficiency）**：基于结果反馈（outcome-based rewards）的强化学习（RL）在训练 LLM-based agents 时，仅依赖稀疏的二元奖励信号（成功/失败），无法利用轨迹中的丰富过程信息（如有效行为、可恢复的失败模式等），导致学习效率低下。
- **经验利用不充分**：现有方法虽然尝试通过“经验提取”（experience distillation）来复用历史轨迹，但通常将经验视为静态资源（static experience bank），或使用固定/非自适应的经验提取器（non-adaptive extractor）。这导致随着策略演员（actor）能力提升，所提取的经验逐渐与其当前能力脱节（distributional misalignment），反而降低学习效果。

### **提出了什么新方法或新思路**
提出 **Complementary Reinforcement Learning (Complementary RL)**，一个实现策略演员（policy actor）与经验提取器（experience extractor）**协同进化**（co-evolution）的强化学习框架，其核心思想受神经科学中的**互补学习系统**（Complementary Learning Systems, CLS）启发：
- **Neocortex（类比 actor）**：缓慢形成结构化长期知识（策略）。
- **Hippocampus（类比 extractor）**：快速存储具体记忆（经验），并通过反馈巩固有价值的经验。

该框架满足三个设计原则：
1. **Actor-Extractor Co-Evolution**：actor 和 extractor 在训练中相互塑造，动态对齐。
2. **Experience Consolidation**：自动从轨迹中提炼、合并、去重，构建高质量经验库（experience bank）。
3. **Training-Distillation Coordination**：通过异步架构解耦训练与蒸馏，避免阻塞延迟。

### **相比现有方法的优势**
- **动态对齐**：经验提取器通过 actor 的实际表现反馈进行优化，确保经验始终匹配 actor 的当前能力水平。
- **闭环互惠**：actor 越强 → 生成更高质量轨迹 → 提炼出更有用经验 → 反哺更强 actor。
- **高效可扩展**：异步训练框架支持大规模并行，不影响训练吞吐量。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个开放域环境上进行评估：
- **MiniHack**：基于文本的网格探索游戏，测试导航与规划能力。
- **WebShop**：模拟网页购物任务，需搜索与点击操作。
- **ALFWorld**：文本版具身智能环境，执行家庭日常任务（如“把加热的盘子放进冰箱”）。
- **SWE-Bench**：真实世界软件工程任务，修复 GitHub issue 并通过单元测试。

### **实验设置和评估指标**
- **主干模型**：
  - Actor：`Qwen2.5-7B-Instruct` 或 `Qwen3-4B-Instruct-2507`
  - Experience Extractor：`Qwen3-4B-Thinking-2507`
- **训练目标**：最大化任务成功率（success rate）或累积奖励。
- **关键机制**：
  - 将 rollout 分为两组：**experience-guided** 与 **experience-free**，用于稳定优势估计（condition-wise advantage estimation）。
  - 经验提取器通过 **CISPO** 目标优化，奖励信号来自其所提供经验是否最终引导任务成功。
- **基础设施**：基于 `ROLL` 框架，采用异步双环设计，由中央 `ExperienceManager` 协调检索与蒸馏。

### **基线方法对比**
- **Baseline**：无经验指导的标准 RL。
- **Offline Exp.**：预构建的静态经验库。
- **Static Online Exp.**：在线维护但 extractor 不更新。
- **Exp. Only**：只训练 extractor，冻结 actor。
- **Complementary RL (Ours)**：actor 与 extractor 联合训练。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **单任务训练（Single-Task）**
- 在所有四个任务上均显著优于 baseline：
  - **MiniHack Room**：成功率提升 **1.3×**，动作数减少 **1.5×**。
  - **ALFWorld**：动作数减少 **2×**，学习更高效。
  - **SWE-Bench**：最终性能 **+3.0%**，且完成更多完整任务（虽动作数略增，但因更彻底而合理）。

#### **多任务训练（Multi-Task）**
- **平均性能**（MiniHack + WebShop + ALFWorld）：
  - Complementary RL 达到 **0.82**（w/ exp.）和 **0.78**（w/o exp.），显著高于 baseline（0.75）。
  - 相比 baseline 提升 **+7%（with exp.）** 和 **+2%（without exp.）**，表明经验已被内化。
- **跨任务泛化**：经验提取器能提炼出通用策略（见 Table 6），如“如何识别停滞并切换策略”。

| Method | Avg. Score |
|--------|----------|
| Baseline | 0.75 |
| Static Online Exp. (w/ exp.) | 0.59 |
| **Complementary RL (w/ exp.)** | **0.82** |
| **Complementary RL (w/o exp.)** | **0.78** |

> 注：Static Online Exp. 表现甚至低于 baseline，验证了“静态经验”的危害。

### **消融实验结果**
- **w/o Merge**：定期合并经验条目对性能至关重要，否则冗余经验损害检索质量。
- **w/o search_and_ask**：允许 actor 在决策中主动查询经验（context-aware query）可进一步提升效率。
- **Extractor 容量影响**：使用更大的 `Qwen3-30B-A3B` 作为 extractor，性能再提升 **+5%**，说明更强的归纳能力有助于提炼更通用经验。
- **任务规模扩展**：在 6-task 混合任务中，性能增益从 **+6.6%（3-task）** 提升至 **+8.1%（6-task）**，显示方法具有良好的可扩展性。
- **延迟分析**：引入的检索与蒸馏开销极小，rollout 收集时间与 baseline **基本持平**（<1秒），无显著延迟。

---

## **4. 关键结论和发现**

### **主要发现**
1. **协同进化是关键**：actor 与 experience extractor 必须共同演化，静态或孤立优化无法发挥经验驱动学习的最大潜力。
2. **经验内化而非依赖**：Complementary RL 不仅提升了推理时使用经验的表现，更重要的是将有用经验**内化**到 actor 自身参数中，使其即使在无经验输入时也更强。
3. **通用经验可被提炼**：在多任务场景下，extractor 能够抽象出跨任务的通用问题解决原则（如“检测停滞并切换策略”），实现知识迁移。
4. **异步架构保障效率**：提出的异步训练框架有效解耦了 rollout、distillation 与 update，实现了高吞吐、低延迟的大规模训练。

### **方法的局限性**
- **实现复杂度高**：需要维护两个模型、一个经验库、异步调度系统，工程复杂度较高。
- **Extractor 训练不稳定**：由于严重 off-policy 特性（经验可能很久后才被使用），需引入 retrieval diversification 和 advantage reweighting 等技巧稳定训练。
- **Actor-Critic 机制有代价**：虽然引入 actor 对经验的“批判”（accept/refine/reject）可进一步提升性能，但会引入额外延迟，牺牲训练速度。

### **未来工作方向**
- **Self-Distillation 集成**：初步尝试将 self-distillation 引入 Complementary RL，虽初期有效但后期崩溃，未来可探索更稳定的集成方式。
- **更高效的检索机制**：探索基于语义聚类或图结构的经验组织方式，提升检索精度与速度。
- **应用于更大规模工业场景**：在更复杂的多模态或多 agent 场景中验证其可扩展性与鲁棒性。

---

> **一句话总结**：  
> Complementary RL 通过构建 actor 与 experience extractor 的**协同进化闭环**，实现了经验的动态提炼与高效利用，在多个复杂任务上显著提升了样本效率与最终性能，为构建可持续自我进化的 LLM agents 提供了一种可扩展的新范式。

</details>

---

### 12. [InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning](https://arxiv.org/abs/2603.17310)

**Authors**: Chengwei Wei, Jung-jae Kim, Longyin Zhang, Shengkai Chen, Nancy F. Chen  
**Category**: cs.AI  
**Published**: 2026-03-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.17310v1  

#### Abstract
Large Language Models (LLMs) with extended reasoning capabilities often generate verbose and redundant reasoning traces, incurring unnecessary computational cost. While existing reinforcement learning approaches address this by optimizing final response length, they neglect the quality of intermedia...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **Large Reasoning Models (LRMs)** 在执行复杂推理任务时倾向于生成**冗长且重复的 Chain-of-Thought (CoT) 推理轨迹**，导致不必要的 token 开销和计算成本。虽然已有基于强化学习（RL）的方法通过引入长度惩罚来鼓励简洁推理，但这些方法通常只优化最终答案的正确性和响应长度，**忽略了中间推理步骤的质量**，从而容易引发 **reward hacking**——模型学会生成看似简短但实际上逻辑跳跃或错误的推理路径。

作者指出：**verbosity（冗长）不仅是长度问题，更是推理质量低下的症状**。

---

### **提出了什么新方法或新思路**
提出 **InfoDensity**，一种全新的 **RL训练奖励框架**，其核心思想是：
> 高质量的推理轨迹应具备“**信息密度高**”（informationally dense）的特点，即每一步都对减少关于最终答案的不确定性（conditional entropy）做出有意义的贡献。

InfoDensity 的设计基于两个从实证分析中发现的关键轨迹级属性：

- **Low Uncertainty Convergence（低不确定性收敛）**：高质量推理在结束时应显著降低对答案的不确定性（熵值趋近于零）。
- **Monotonic Progress（单调进展）**：每一步都应持续减少不确定性，避免反复或停滞。

由此构建的奖励函数由三部分组成：

1. **AUC Reward**：衡量整个推理过程中不确定性的累积程度，越小越好。
2. **Monotonicity Reward**：鼓励每一步都严格降低熵值。
3. **Length Scaling Term**：偏好更短的响应，但以不牺牲推理质量为前提。

最终奖励为：
$$
R_{\text{InfoDensity}}(T) = R_{\text{quality}}(T) \cdot R_L(T)
$$
其中 $ R_{\text{quality}} = \alpha \cdot R_{\text{AUC}} + (1-\alpha) \cdot R_{\text{mono}} $，仅应用于答案正确的样本。

---

### **相比现有方法的优势**
| 对比维度 | 现有方法（如 GRPO-LP, PEAR） | InfoDensity |
|--------|----------------------------|-----------|
| **监督信号粒度** | 只关注最终答案 + 长度 | 引入**轨迹级熵动态**作为中间质量信号 |
| **抗 Reward Hacking 能力** | 弱，易出现“跳步”或虚假简洁 | 更强，要求每步都有信息增益 |
| **是否依赖人工标注** | PRM 类需大量标注 | 完全无监督，基于模型自身概率估计 |
| **效率-准确性权衡** | 往往牺牲准确率换长度缩短 | 实现**更强的 accuracy-efficiency trade-off** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练集**：
  - `GSM8K`
  - `MATH`
- **测试集（涵盖不同难度与分布）**：
  - **In-Domain**：
    - `GSM8K`
    - `MATH500`（MATH 子集）
  - **Out-of-Distribution**：
    - `AIME 24`
    - `OlympiadBench`

---

### **实验设置和评估指标**
- **Base Models**：
  - `Qwen3-0.6B`
  - `DeepSeek-R1-Distill-Qwen-1.5B`
- **Judge Model**（用于计算 entropy 和打分）：
  - 固定外部模型 `Qwen3-4B-Instruct`
- **训练协议**：
  - 使用 GRPO 框架进行 RL 训练
  - 最大生成长度：32,768 tokens
  - 温度采样：0.6，top-p: 0.95
- **评估指标**：
  - **Accuracy (Acc)**：最终答案正确率
  - **Token Usage (Tok)**：平均生成 token 数量
  - 综合考量 **accuracy-efficiency trade-off**

---

### **基线方法对比**
1. **GRPO-Acc**：仅用 accuracy 做奖励
2. **GRPO-LP**：加入长度惩罚（length penalty）
3. **PEAR**：使用 phase-level entropy 作为奖励信号，SOTA 效率推理方法
4. **Direct-Scoring (DS)**：让 judge model 显式评分推理质量（控制变量实验，验证 entropy 是否优于显式评分）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2）**

#### 在 `DeepSeek-R1-Distill-Qwen-1.5B` 上：
| Method | Avg Acc (%) | Avg Tok | 相对原始模型变化 |
|-------|------------|--------|----------------|
| Original | 61.5 | 9217 | — |
| GRPO-Acc | 63.9 (+2.4) | 7248 (-21%) | ✅ 准确率↑，长度↓ |
| PEAR | 61.1 (-0.4) | 6136 (-33%) | ❌ 准确率↓，长度↓ |
| **InfoDensity** | **64.0 (+2.5)** | **6443 (-30%)** | ✅✅ **准确率最高，长度大幅下降** |

#### 在 `Qwen3-0.6B` 上：
| Method | Avg Acc (%) | Avg Tok | 相对原始模型变化 |
|-------|------------|--------|----------------|
| Original | 49.5 | 8291 | — |
| GRPO-Acc | 51.9 (+2.4) | 8819 (+6%) | ⚠️ 准确率↑但更耗 token |
| PEAR | 50.2 (+0.7) | 6811 (-18%) | ✅ 小幅提升，长度降 |
| **InfoDensity** | **49.2 (-0.3)** | **6014 (-27%)** | ✅✅ **最省 token，准确率几乎持平** |

> 💡 结论：**InfoDensity 在保持甚至提升 accuracy 的同时，实现了最显著的 token 压缩效果。**

---

### **消融实验结果（Ablation Studies）**

#### （1）**AUC 与 Monotonicity 奖励的协同作用（Figure 5）**
- 当 $\alpha = 1.0$（仅 AUC）：
  - 模型学会早期锁定一个答案并不断重复推导（如 “let me double-check”），造成**冗余而非高效**
  - 准确率迅速崩溃
- 当 $\alpha = 0.0$（仅 Monotonicity）：
  - 模型可逐步减熵但永不收敛到低值 → 推理不完备
  - 准确率降至 ~70%
- 当 $\alpha = 0.5$：
  - 两者结合实现稳定训练和高性能
  - 验证了两个组件的**互补性**

#### （2）**Length Scaling 强度的影响（Figure 4）**
- 适度的长度强度（$\lambda = 0.01 \sim 0.05$）带来良好平衡
- 过强（$\lambda = 0.5$）会导致模型崩溃，准确率跌破 60%
- 表明：**必须在质量和长度之间取得精细平衡**

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **高质量推理具有可量化的信息动力学特征**：
   - 正确的推理轨迹表现出 **low uncertainty convergence** 和 **monotonic progress**
   - 错误轨迹在首次出错后熵值停滞，方差高
2. ✅ **基于 entropy trajectory 的奖励信号比显式评分更稳定可靠**：
   - Direct-Scoring 因 judge model 缺乏专门训练而波动大
   - InfoDensity 利用概率输出提供连续、细粒度反馈
3. ✅ **InfoDensity 实现了当前最优的 accuracy-efficiency trade-off**：
   - 在多个 benchmark 上匹配或超越 SOTA 方法
   - 显著减少 token 使用（最高达 -30%），无明显准确率损失

---

### **局限性**
1. **领域限制**：目前仅在数学推理任务上验证，其轨迹性质是否适用于代码生成、开放域问答等尚待研究。
2. **依赖外部 Judge Model**：
   - 使用固定的大模型（如 Qwen3-4B）增加推理开销
   - 若 judge 模型能力不足，可能引入噪声
3. **未解决“过度压缩”风险**：极端 length scaling 仍可能导致 collapse

---

### **未来工作方向**
1. **扩展至其他任务领域**：探索 InfoDensity 在 code reasoning、multi-hop QA 中的有效性。
2. **自判别机制**：尝试让训练模型自身估计 entropy，消除对外部 judge 的依赖。
3. **动态长度调节**：结合问题难度自动调整 length scaling 强度。
4. **多模态推理中的应用**：将 entropy 分析拓展到图像+文本联合推理场景。

---

> 🔗 **开源信息**：作者已公开 InfoDensity 的模型、训练与评估 pipeline  
> GitHub 地址：[https://github.com/anonymous/InfoDensity](https://github.com/anonymous/InfoDensity)（匿名提交）

</details>

---

### 13. [Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies](https://arxiv.org/abs/2603.15903)

**Authors**: Nathaniel Imel, Richard Futrell, Michael Franke, Noga Zaslavsky  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.15903v1  

#### Abstract
Natural languages have been argued to evolve under pressure to efficiently compress meanings into words by optimizing the Information Bottleneck (IB) complexity-accuracy tradeoff. However, the underlying social dynamics that could drive the optimization of a language's vocabulary towards efficiency ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在解决语言演化中的两个核心开放问题：
1. **效率的机制来源**：自然语言为何在语义系统中表现出信息论意义上的高效压缩（即在复杂度与准确性之间取得最优权衡）？这种效率是如何通过文化演化机制实现的？
2. **EGT 与 IB 的关系**：基于进化博弈论（Evolutionary Game Theory, EGT）的成功通信策略是否等同于信息瓶颈（Information Bottleneck, IB）框架下的信息论最优？

尽管已有研究分别从功能主义、认知科学和多智能体学习角度探讨语言效率，但缺乏一个将**微观个体行为动力学**与**宏观群体层面的信息论效率**统一起来的理论框架。

---

### 提出了什么新方法或新思路
论文提出了一种**统一的理论与计算模型**，将以下三个关键元素整合：
- **Information Bottleneck (IB)**：用于形式化定义语义系统的“高效压缩”目标（最小化 $ \mathcal{F}_\beta = I(M;W) - \beta I(W;U) $）。
- **Noisy Sim-Max Signaling Games**：一种受感知模糊性和相似性驱动的信号博弈，由 Franke & Correia (2018) 提出，具有独立动机（如解释模糊性起源）。
- **Imprecise Conditional Imitation Dynamic**：一种基于频率依赖选择的社会模仿动态，允许个体在感知噪声下模仿他人的策略。

这一整合实现了从**局部互动**（agent-level signaling behavior）到**全局效率**（population-level IB optimality）的桥梁构建。

---

### 相比现有方法的优势
| 方面 | 本文方法优势 |
|------|---------------|
| **理论统一性** | 首次明确连接 EGT 中的“沟通成功”与 IB 中的“信息论效率”，揭示二者可能内在一致。 |
| **机制简洁性** | 不依赖深度神经网络架构、超参数调优或多层认知假设，仅需简单的模仿规则即可涌现高效系统。 |
| **可扩展性与解释力** | 模型抽象程度高，适用于任何语义域；同时能解释跨语言语义系统的变异（如不同精度标准导致不同 trade-off）。 |
| **生态合理性** | 引入感知混淆（state confusion）和不完美模仿，更贴近真实人类学习与文化传播过程。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
本研究为**合成实验**（synthetic domain），未使用真实自然语言数据集。设计了一个理想化的**数值性（numerosity）语义空间**：
- **状态空间 $ X $**：$ \{0, 1, ..., 99\} $，共 100 个连续数量状态。
- **词汇表 $ W $**：同样有 100 个可用信号（词），理论上支持一一映射（exact numerosity）或粗略编码（approximate system）。

此设定便于控制变量并可视化类别边界演化。

---

### 实验设置和评估指标

#### 动态过程
采用 **discrete-time imprecise imitation dynamic** 模拟文化演化，运行最多 $10^5$ 步直至收敛。

#### 关键参数调节
- **$ \gamma $**：pragmatic standard of precision，控制博弈奖励对误差的敏感度（$ \gamma \to 0 $：宽松；$ \gamma \to \infty $：严格）。
- **$ \sigma $**：perceptual certainty parameter，控制状态混淆概率（固定为 0.5）。
- 初始条件：8 个随机种子，$ \gamma $ 在 $[10^{-8}, 10]$ 范围内取 100 个对数间隔值。

#### 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **Complexity** | $ I(M_o; W) $：编码所需比特数 | 衡量压缩程度 |
| **Accuracy** | $ I(W; X_a) $：信号与真实状态间互信息 | 衡量信息保真度 |
| **Efficiency Loss ($ \epsilon $)** | $ \min_\beta [\mathcal{F}_\beta[S] - \mathcal{F}_\beta^*] $ | 衡量偏离 IB 最优界的程度 |
| **Population Fitness** | $ \mathbb{E}[\text{sim}(x_a, \hat{x}_a)] $：期望相似性得分 | 衡量博弈表现 |

所有结果均在“信息平面”（information plane）上绘制并与 IB 理论边界比较。

---

### 基线方法对比
| 基线 | 描述 | 目的 |
|------|------|------|
| **Permuted Systems** | 对最终系统进行随机打乱（meaning-word 映射重排） | 验证效率非偶然产生 |
| **NK99 Dynamic** | Nowak & Krakauer (1999) 提出的有限群体复制-突变模型 | 对比传统 EGT 模型能否达到 IB 效率 |
| **Random Controls** | 完全随机生成的通信系统 | 提供性能下限参考 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **接近 IB 边界**：绝大多数演化出的系统落在 IB 理论边界附近（见 Figure 2A），表明实现了近似最优压缩。
- **低效率损失**：平均 $ \epsilon < 0.1 $ bit，在多数 $ \gamma $ 设置下显著低于基线。
- **最大可达精度受限**：即使在高 $ \gamma $ 下，最高 Accuracy 仍低于理论最大值（约 4.61 bits），实际最高仅约 **4 bits**，显示噪声模仿存在根本限制。

---

### 与基线方法的对比结果
| 比较项 | 结果 |
|--------|------|
| vs. **Permuted Systems** | 打乱后的系统严重偏离 IB 边界，复杂度高而准确率低，验证当前结构非随机产物。 |
| vs. **NK99 Dynamic** | NK99 模型虽能发展出稳定词汇，但其系统普遍远离 IB 边界，表现为“高复杂度、低效率”，说明传统 EGT 动态不一定导向信息论最优。 |
| vs. **Random Systems** | 随机系统聚集于原点附近，远不如本文模型高效。 |

> ✅ **结论**：只有引入感知噪声与不完美模仿的动态才能持续逼近 IB 最优解。

---

### 消融实验结果（隐含分析）
虽然未设显式“消融”模块，但通过参数扫描实现了类似效果：
- **$ \gamma $ 影响 trade-off 位置**：随着 $ \gamma $ 增大，系统沿 IB 曲线向高复杂度-高准确性区域移动（$ \rho_{\text{Spearman}} \approx 0.99 $），说明**语用精度标准是调控效率权衡的关键机制参数**。
- **噪声限制上限**：当 $ \gamma > 1 $ 时，效率损失 $ \epsilon $ 反而上升（Figure 2B），表明在高精度要求下，噪声模仿难以逼近最优解。
- **收敛速度差异**：在 $ \gamma \sim 10^{-6} $ 和 $ \gamma \sim 10^{-2} $ 附近收敛缓慢，暗示可能存在临界相变行为。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **局部模仿可导向全局效率**：即使个体仅基于局部收益模仿他人策略（无全局优化目标），群体仍能自发演化出接近 IB 最优的语义系统。
2. ✅ **EGT 成功 ≠ IB 最优，但在特定动态下可等价**：传统的 EGT 模型（如 NK99）未必高效，但当引入**感知噪声**与**不完美模仿**后，演化路径被正则化，反而趋向 IB 最优。
3. ✅ **语用标准塑造语义结构变异**：参数 $ \gamma $（pragmatic slack）系统地决定了最终系统的复杂度-准确性权衡位置，为跨语言语义类型学提供潜在解释机制。
4. ✅ **噪声既是约束也是正则器**：状态混淆和模仿误差虽限制了最大精度，但也防止过拟合，促使系统形成平滑、泛化的类别边界，间接促进压缩。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **静态语义空间** | 当前模型假设固定状态集和词汇量，未考虑递归、组合性或新词创造。 |
| **均匀先验** | 假设所有状态同等重要（uniform $ p(x) $），忽略了现实中频率偏倚（如幂律分布）的影响。 |
| **两群体分离假设** | Sender 与 Receiver 分属不同群体，简化了现实中的双向交互。 |
| **缺乏心理真实性细节** | 虽机制简洁，但未建模具体认知机制（如注意力、记忆衰减）。 |

---

### 未来工作方向
1. **引入非均匀先验**：模拟真实 communicative need 分布（如 power-law），检验是否再现自然语言频率模式。
2. **扩展至组合系统**：探索如何从简单信号博弈中演化出递归与组合语法。
3. **结合神经网络代理**：在 RL 框架中实现相同动态，验证抽象模型预测。
4. **实证验证**：将模型应用于儿童语言习得或人工语言实验数据，检验其对真实演化轨迹的拟合能力。
5. **数学关系深化**：进一步探究 replicator dynamics 与 IB optimization 的形式联系，例如是否可视为某种变分推断过程。

---

> 🔚 **总体评价**：  
> 本文是一项理论驱动的开创性工作，成功搭建了 **Evolutionary Game Theory** 与 **Information-Theoretic Efficiency** 之间的桥梁。它不仅回答了“语言为何高效”的机制问题，还指出：“看似低效的噪声模仿”，恰恰可能是通向高效文化的必要路径。

</details>

---

### 14. [AI Scientist via Synthetic Task Scaling](https://arxiv.org/abs/2603.17216)

**Authors**: Ziyang Cai, Harkirat Behl  
**Category**: cs.AI  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.17216v1  

#### Abstract
With the advent of AI agents, automatic scientific discovery has become a tenable goal. Many recent works scaffold agentic systems that can perform machine learning research, but don't offer a principled way to train such agents -- and current LLMs often generate plausible-looking but ineffective id...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AI Scientist via Synthetic Task Scaling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前的 **LLM-based AI agents** 虽然具备丰富的机器学习理论知识，但在实际科研任务中往往生成“看似合理但无效”的想法，缺乏从真实研究过程中获得的经验。现有方法通常仅在静态输出（如论文、代码）上进行训练，忽略了**迭代探索、调试失败和逐步优化**等关键科研行为。

该论文旨在解决：
- 如何让 AI 科学家通过“做研究”来学习研究？
- 如何规模化地生成高质量、可执行的机器学习研究任务以供代理训练？

---

### 🚀 提出的新方法与新思路
提出了一套**全自动、可扩展的合成环境生成流水线（synthetic environment generation pipeline）**，用于训练能够自主完成机器学习研究任务的 AI Agent。

#### 核心创新点包括：

1. **端到端任务自动生成**
   - 自动采样 ML 主题（如 CV、Graph NN）
   - 自动生成任务描述、推荐 HuggingFace 数据集并验证其存在性
   - 合成完整的可运行环境：配置文件、起始代码（starter code）、评估脚本

2. **基于 SWE-Agent 框架的任务兼容性**
   - 所有任务均适配于通用的 **SWE-Agent** 框架，支持工具调用（文件读写、命令行执行等），实现跨领域统一交互接口。

3. **自我调试循环（self-debugging loop）提升任务质量**
   - 在任务验证阶段引入 GPT-5 作为教师模型运行任务
   - 若出现编译错误或执行失败，则反馈错误信息回生成系统，尝试修复而非直接丢弃
   - 显著提高最终可用任务的比例和质量

4. **无监督、高可扩展的合成流程**
   - 整个流程无需人工干预，可通过 HPC 集群并行化处理，支持大规模轨迹采集

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 数据来源 | 人类撰写论文/代码 | 全自动合成任务 |
| 训练信号 | 静态输出（结果导向） | 动态轨迹（过程导向） |
| 任务多样性 | 受限于公开基准数量 | 千级主题采样，高度多样化 |
| 任务有效性 | 依赖人工设计与验证 | 自动验证 + 自我调试保障可执行性 |
| 可扩展性 | 低（人力瓶颈） | 高（完全自动化） |

> ✅ 优势总结：首次实现了**从零开始构建大量真实、多样且可执行的 ML 研究任务**，为训练“会做实验”的 AI 科学家提供了有效路径。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **合成任务基础数据源**：
  - 从模型中采样约 **1000 个 ML 主题**
  - 对每个主题提议使用一个 **HuggingFace 数据集**，并通过 HuggingFace API 验证其存在性和可用性
  - 最终保留约 **500 个有效合成任务**

- **评估基准**：
  - **MLGym [Nathani et al., 2025]**：包含 13 个复杂度各异的机器学习挑战任务，涵盖：
    - 图像分类（CIFAR-10, Fashion MNIST）
    - 自然语言理解（MNLI）
    - 强化学习（MountainCar, Meta Maze）
    - 回归预测（House Price）
    - 游戏博弈（Prisoner’s Dilemma, Blotto）

---

### ⚙️ 实验设置

| 设置项 | 描述 |
|-------|------|
| **任务生成器** | GPT-5（teacher model） |
| **轨迹生成方式** | 在 HPC 集群上并行运行每个任务，每任务目标收集 256 条 agent 轨迹 |
| **轨迹总数** | 收集约 56,210 条原始轨迹 → 过滤后保留 **~34,000 条训练轨迹** |
| **过滤标准** | - 成功提交至少一次<br>- 总长度 ≤ 48K tokens<br>- 训练时进一步截断至 32K tokens |
| **训练范式** | Supervised Fine-Tuning (**SFT**) |
| **学生模型** | Qwen3-4B 和 Qwen3-8B |
| **训练数据** | 上述合成任务中的 agent 轨迹（由 GPT-5 生成） |
| **交互格式** | SWE-Agent 格式：每轮输出 reasoning + action（如 edit file, run command） |

---

### 📊 评估指标

- **主指标**：**AUP（Area Under the Performance Curve）**
  - 综合衡量在整个任务周期内性能随时间提升的趋势
  - 不同子任务得分经过归一化后聚合，避免尺度差异影响
- **辅助指标**：
  - 各子任务上的最终得分（accuracy, loss, win rate 等）
  - 成功提交率
  - 平均 token 数、平均回合数（turns）

---

### 🆚 基线方法对比

| 模型 | 类型 | 是否微调 | 备注 |
|------|-----|----------|------|
| GPT-4o | Closed-source LLM | 否 | 商业强基线 |
| GPT-5 | Teacher Model | 否 | 用于生成轨迹 |
| Qwen3-4B / Qwen3-8B | Base Models | 否 | 未微调原始版本 |
| **SFT-Qwen3-4B / SFT-Qwen3-8B** | ✅ 本文方法 | 是 | 在合成轨迹上 SFT 微调 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 模型 | AUP Score ↑ | 提升幅度 |
|------|-------------|---------|
| Qwen3-4B (base) | 0.833 | — |
| **SFT-Qwen3-4B** | **0.908** | **+9%** |
| Qwen3-8B (base) | 0.881 | — |
| **SFT-Qwen3-8B** | **0.988** | **+12%** |

> 💡 注：AUP 提升表明学生模型不仅最终表现更好，而且在整个探索过程中更高效地逼近最优解。

---

### 📊 子任务层面表现（见 Figure 4）

- 在 **13 个 MLGym 子任务中的 9 个**，SFT 模型优于 base Qwen3 模型
- 特别是在以下任务中增益显著：
  - CIFAR-10 图像分类
  - House Price 回归
  - Breakout 强化学习
  - Meta Maze 探索任务
- 少数任务（如 MS-COCO）未见明显提升，作者分析可能因这些任务涉及更复杂的 starter code 结构，当前合成流程未能充分覆盖

---

### ❌ 消融实验（文中未明确提供完整消融，但有讨论）

尽管没有系统的 ablation study，作者在 **Limitations 和 Discussion 中指出了多个潜在影响因素**：

| 组件 | 分析结论 |
|------|--------|
| **HuggingFace 数据验证** | 确保任务接地于真实世界数据，增强泛化能力 |
| **Self-debugging Loop** | 显著减少无效任务数量，提升任务成功率 |
| **Success-based Filtering** | 过滤掉陷入死循环或无法提交的轨迹，保证训练数据质量 |
| **Teacher Model Quality (GPT-5)** | 决定了任务上限；若 teacher 无法解决某类任务，则学生也无法学到相关技能 |

> ⚠️ 作者承认：目前尚不清楚各组件对最终性能的具体贡献比例，需未来开展消融研究。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **合成任务可以成为有效的训练信号**
   - 通过大规模合成可执行的 ML 研究任务，并利用 teacher model 生成 agent 轨迹，能显著提升学生模型在真实科研基准上的表现。

2. **过程经验比静态知识更重要**
   - 单纯的知识灌输不足以支撑有效科研行为；**经历假设→实验→失败→调试→改进的全过程**是培养 AI 科学家的关键。

3. **可扩展性是通往自主科学发现的必经之路**
   - 本文提出的 pipeline 完全自动化，可在超算集群上并行扩展，为未来更大规模训练奠定基础。

4. **结构一致性有助于迁移学习**
   - 所有任务共享 SWE-Agent 的交互框架（turn-based reasoning + action），使得 agent 能够将在某一任务中学到的行为模式迁移到其他任务中。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **评估局限于单一基准（MLGym）** | 缺乏在 MLE-Bench、PaperBench 等其他科研基准上的测试，泛化能力有待验证 |
| **格式对齐 vs. 能力提升难以分离** | 性能提升可能是由于熟悉了 SWE-Agent 的交互格式，而非真正提升了科研能力 |
| **依赖 teacher model 的能力边界** | GPT-5 无法解决的任务不会出现在训练集中，限制了学生的潜力 |
| **缺乏探索与创新机制** | SFT 训练偏向模仿已有策略，不鼓励新颖想法；未引入 RL 或 curiosity-driven learning |
| **无完整消融实验** | 无法量化各个模块（如 self-debugging、dataset grounding）的独立贡献 |

---

### 🔮 未来工作方向

1. **引入 Reinforcement Learning**
   - 利用任务最终得分作为 reward signal，训练 agent 主动探索更优策略
   - 挑战：训练耗时长、reward 稀疏、scale 困难

2. **扩展至更多元化的基准**
   - 应用于 **MLE-Bench (Kaggle-style)**、**PaperBench (复现论文)**、**NanoGPT Speedrunning** 等任务

3. **融合文献检索与知识更新机制**
   - 允许 agent 在 trajectory 生成过程中主动搜索最新研究成果，促进真正意义上的“新发现”

4. **基于高质量代码库的任务生成**
   - 当前 starter code 较简单；未来可基于 NanoGPT、HuggingFace Transformers 等真实项目生成更复杂的任务

5. **多智能体协作科研框架**
   - 构建“AI Co-Scientist”团队，分工完成 idea generation、implementation、debugging、writing 等环节

---

## 总结

✅ **一句话总结**：  
本论文提出了一种通过**合成任务规模化训练 AI 科学家**的新范式，借助自动化的任务生成与自我调试机制，成功提升了 Qwen3 模型在 MLGym 基准上的科研能力，AUP 指标最高提升 **12%**，为实现**自主、迭代式科学发现**提供了切实可行的技术路径。

🎯 **长远意义**：  
推动 AI 从“知道怎么做”向“真的去做并学会”转变，迈向真正的 **AI Scientist** 时代。

</details>

---

### 15. [AgentFactory: A Self-Evolving Framework Through Executable Subagent Accumulation and Reuse](https://arxiv.org/abs/2603.18000)

**Authors**: Zhang Zhang, Shuqi Lu, Hongjin Qian, Di He, Zheng Liu  
**Category**: cs.AI  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.18000v1  

#### Abstract
Building LLM-based agents has become increasingly important. Recent works on LLM-based agent self-evolution primarily record successful experiences as textual prompts or reflections, which cannot reliably guarantee efficient task re-execution in complex scenarios. We propose AgentFactory, a new self...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*AgentFactory: A Self-Evolving Framework Through Executable Subagent Accumulation and Reuse*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前基于 **LLM-based agents** 的自进化研究主要依赖于将成功经验记录为文本形式的 **prompts、reflections 或 reasoning traces**。然而，这种文本化的经验存储方式存在以下问题：

- **不可靠复用**：在复杂任务中，仅靠文本提示无法保证高效、准确地重执行任务。
- **缺乏可移植性**：文本经验难以直接迁移到其他系统或框架中。
- **静态行为**：大多数 agent 框架（如 LangChain、AutoGPT）不具备持续积累和改进能力。

因此，如何实现 **可持续、可靠且可迁移的能力积累机制** 成为关键挑战。

---

### ✅ 提出了什么新方法或新思路

作者提出 **AgentFactory** —— 一种全新的 **self-evolving framework**，其核心思想是：

> 将成功的任务解决方案保存为 **可执行的 subagent 代码**（而非文本经验），并通过执行反馈不断优化这些 subagent。

该框架遵循一个三阶段生命周期：

1. **Install（安装）**  
   - 面对新任务时，Meta-Agent 将任务分解为子任务，并动态创建专用的 subagent。
   - 成功执行后，将 subagent 以纯 Python 脚本形式保存至技能库。

2. **Self-Evolve（自进化）**  
   - 当遇到相似任务时，优先检索并复用已有 subagent。
   - 若失败，则分析执行反馈，自动调用 `modify_subagent` 改进代码（如增强错误处理、支持边缘情况等）。
   - 经验证后更新到技能库，形成“越用越强”的正向循环。

3. **Deploy（部署）**  
   - 成熟的 subagent 可导出为独立 Python 模块，供其他 AI 系统（如 LangChain、AutoGen、Claude Code）直接使用。
   - 所有 subagent 均附带标准化文档 `SKILL.md`，便于跨平台集成。

此外，系统架构包含三大组件：
- **Meta-Agent**：负责任务分解与调度。
- **Skill System**：统一管理 meta skills、tool skills 和 subagent skills。
- **Workspace Manager**：提供隔离环境，确保安全演化。

---

### ✅ 相比现有方法的优势

| 对比维度 | 传统方法（如 Reflexion） | AgentFactory |
|--------|--------------------------|-------------|
| 经验表示 | 文本 prompt/reflection | **可执行 Python 代码** |
| 复用可靠性 | 低（依赖 LLM 解析文本） | 高（直接运行脚本） |
| 进化粒度 | 输出级 refine | **agent 级 code 修改** |
| 可移植性 | 差（绑定特定框架） | 强（纯 Python + 文档） |
| 自动化程度 | 手动设计模板 | 完全自主生成与优化 |

> 🔑 **核心创新**：从“记录经验”转向“构建可执行智能体”，实现了 **executable knowledge accumulation**。

---

## 2. 核心实验方法和设置

### 📦 数据集与任务设计

共设计两批任务（每批 15 项），用于评估初始学习与迁移复用能力：

- **Batch 1**：涵盖多个领域的真实世界任务，包括：
  - Web 信息检索（如课程页面抓取）
  - 数据可视化（matplotlib 绘图）
  - 浏览器自动化（Tencent Meeting 预约）
  - 音频处理（语音转写与播放）
  - 编程小游戏（Tetris/Snake）
  > ➤ 用于构建初始 subagent 库。

- **Batch 2**：结构上与 Batch 1 类似，但具体需求不同（例如：从 Bitcoin 改为 Ethereum；中国人口改为日本人口），用于测试 **subagent 的泛化与复用能力**。

> 示例：  
> - Batch1-T3: 预约明天 4PM 的腾讯会议  
> - Batch2-T3: 预约明天 7PM 的腾讯会议 → 可复用同一 subagent 并微调时间参数

完整任务列表见附录 B（Tables 2 & 3）。

---

### 📊 实验设置与评估指标

#### ✅ 模型选择
使用两个主流 LLM 作为 backbone：
- **Claude Opus 4.6**
- **Claude Sonnet 4.6**

Meta-Agent 与 subagents 均采用相同模型，保持一致性。

#### ✅ 基线方法（Baselines）
1. **ReAct**  
   - 每次任务都从零开始解决，无任何知识积累。
2. **Self-Evolving Agent (with Textual Experience)**  
   - 类似 Reflexion，将成功/失败经验总结为文本摘要，后续任务中检索参考。

#### ✅ 主要评估指标
- **Average Output Tokens per Task**（排除 subagent 内部调用）
  - 衡量 **orchestrator（Meta-Agent）的决策开销**
  - 数值越低，说明 subagent 复用越高效，主控 agent 越“省力”

> 💡 注意：不统计 subagent 内部 LLM 调用 token，聚焦于“协调成本”。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| Method | Task Setting | Opus 4.6 (avg tokens) | Sonnet 4.6 (avg tokens) |
|-------|--------------|------------------------|-------------------------|
| ReAct | Batch 1 | 8,298 | 6,893 |
| ReAct | Batch 2 | 7,022 | 7,029 |
| Self-Evolving Agents | Batch 1 (from scratch) | 8,608 | 8,163 |
| Self-Evolving Agents | Batch 2 (w/ saved) | 6,210 | 8,223 |
| **AgentFactory** | **Batch 1 (from scratch)** | **4,324** | **9,199** |
| **AgentFactory** | **Batch 2 (w/ saved)** | **2,971** | **3,862** |

---

### 🔍 结果分析

#### ✅ 显著降低 orchestration 开销
- 在 **Batch 2（迁移任务）** 中：
  - AgentFactory 相比 ReAct：
    - Opus 下减少 **~58%** token（2,971 vs 7,022）
    - Sonnet 下减少 **~45%** token（3,862 vs 7,029）
  - 相比文本经验方法（6,210 / 8,223）也有明显优势。

> 表明：**可执行 subagent 的复用效率远高于文本经验引导**

#### ✅ 即使在 Batch 1 也展现早期收益
- 尽管 Batch 1 是“从零开始”，但 Opus 版本已达到 **4,324 tokens**，显著低于 ReAct 的 8,298。
- 说明：**更强的 LLM 更善于识别 subtask 共性，提前实现 subagent 复用**。

#### ✅ 子代理演化实例（Figure 2）
以 README 生成 subagent 为例，经历三次迭代：
1. Run1：硬编码路径（hardcoded path）
2. Run2：尝试 LLM 解析 JSON，失败后 fallback 到硬编码（脆弱）
3. Run3：引入 regex fallback，提升鲁棒性

> 展示了 **自主检测缺陷 → 修改代码 → 增强健壮性** 的完整 self-evolve 过程。

#### ✅ 跨系统复用演示（Figure 3）
- 在 AgentFactory 中训练出的 `Audio Transcriber` 和 `Document Creator` subagent，
- 被迁移到 **Claude Code** 系统中，
- 新 agent 通过阅读 `SKILL.md` 文档理解接口，
- 成功组合调用两个 subagent 完成音频驱动的文档创建任务。

> 验证了 **portability 与 interoperability** 的可行性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Executable subagent 是更高效的长期记忆形式**  
   相比文本经验，可执行代码能被精确复用，避免 LLM 解释偏差，显著降低协调成本。

2. **AgentFactory 实现了真正的“越用越聪明”**  
   通过 feedback-driven 自我修改机制，subagent 不断变得更通用、更鲁棒。

3. **具备生态系统级潜力**  
   导出的 subagent 可作为“插件”被其他 agent 框架使用，推动 **agent skill marketplace** 的可能。

4. **强 LLM 更能发挥早期复用优势**  
   如 Opus 在 Batch 1 就表现出更低开销，预示随着 foundation model 提升，self-evolution 效益将进一步放大。

---

### ⚠️ 方法的局限性

1. **依赖高质量代码生成能力**  
   若 LLM 生成的 subagent 初始质量差，可能导致错误积累或难以修复。

2. **演化过程缺乏形式化验证**  
   当前修改基于启发式反馈，可能存在逻辑退化风险（虽未观察到，但理论上存在）。

3. **仅限 web-based 工具交互**  
   当前主要通过 browser_automation 和 web_search 实现外部交互，尚未覆盖桌面应用或本地 GUI。

4. **安全与权限控制需人工介入**  
   尽管 `shell_command` 有安全检查，但仍需用户审计导出代码，不适合完全无人监管场景。

---

### 🔮 未来工作方向

1. **集成 Vision-Language Models (VLMs)**  
   支持 GUI-based interaction，扩展至非网页类应用程序（如 Windows/Mac 软件）。

2. **建立 subagent 版本控制系统**  
   支持回滚、diff 分析、自动化测试，防止劣化。

3. **构建开放的 Agent Skills Marketplace**  
   推动 subagent 社区共享与协作演进。

4. **探索分布式 agent 协同演化**  
   多个 AgentFactory 实例之间交换 subagent，加速全局能力积累。

---

## 总结

> **AgentFactory 开创了一种“构建即进化”的新范式**：它不只是一个 agent，更是一个 **生产智能体的工厂（agent factory）**。通过将成功经验固化为可执行、可演化、可迁移的 subagent 代码，实现了真正意义上的 **cumulative capability growth**，为构建长期自治、持续进化的 AI 系统提供了坚实基础。

</details>

---

### 16. [Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability](https://arxiv.org/abs/2603.16017)

**Authors**: Fan Huang, Haewoon Kwak, Jisun An  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16017v1  

#### Abstract
Large language models (LLMs) increasingly participate in morally sensitive decision-making, yet how they organize ethical frameworks across reasoning steps remains underexplored. We introduce \textit{moral reasoning trajectories}, sequences of ethical framework invocations across intermediate reason...

---

### 17. [ASDA: Automated Skill Distillation and Adaptation for Financial Reasoning](https://arxiv.org/abs/2603.16112)

**Authors**: Tik Yu Yim, Wenting Tan, Sum Yee Chan, Tak-Wah Lam, Siu Ming Yiu  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16112v1  

#### Abstract
Adapting large language models (LLMs) to specialized financial reasoning typically requires expensive fine-tuning that produces model-locked expertise. Training-free alternatives have emerged, yet our experiments show that leading methods (GEPA and ACE) achieve only marginal gains on the FAMMA finan...

---

### 18. [AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents](https://arxiv.org/abs/2603.16496)

**Authors**: Shannan Yan, Jingchen Ni, Leqi Zheng, Jiajun Zhang, Peixi Wu, Dacheng Yin, Jing Lyu, Chun Yuan, Fengyun Rao  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16496v1  

#### Abstract
Large language model (LLM) agents increasingly rely on external memory to support long-horizon interaction, personalized assistance, and multi-step reasoning. However, existing memory systems still face three core challenges: they often rely too heavily on semantic similarity, which can miss evidenc...

---

### 19. [Omnilingual SONAR: Cross-Lingual and Cross-Modal Sentence Embeddings Bridging Massively Multilingual Text and Speech](https://arxiv.org/abs/2603.16606)

**Authors**: Omnilingual SONAR Team, Jo\~ao Maria Janeiro, Pere-Llu\'is Huguet Cabot, Ioannis Tsiamas, Yen Meng, Vivek Iyer, Guillem Ram\'irez, Loic Barrault, Belen Alastruey, Yu-An Chung, Marta R. Costa-Jussa, David Dale, Kevin Heffernan, Jaehyeong Jo, Artyom Kozhevnikov, Alexandre Mourachko, Christophe Ropers, Holger Schwenk, Paul-Ambroise Duquenne  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16606v1  

#### Abstract
Cross-lingual sentence encoders typically cover only a few hundred languages and often trade downstream quality for stronger alignment, limiting their adoption. We introduce OmniSONAR, a new family of omnilingual, cross-lingual and cross-modal sentence embedding models that natively embed text, spee...

---

### 20. [HoloByte: Continuous Hyperspherical Distillation for Tokenizer-Free Modeling](https://arxiv.org/abs/2603.16917)

**Authors**: Vladimer Khasia  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16917v1  

#### Abstract
Sequence modeling universally relies on discrete subword tokenization to circumvent the $\mathcal{O}(N^2)$ computational intractability of native byte-level attention. However, this heuristic quantization imposes artificial morphological boundaries, enforces vocabulary dependence, and fractures the ...

---

### 21. [SCE-LITE-HQ: Smooth visual counterfactual explanations with generative foundation models](https://arxiv.org/abs/2603.17048)

**Authors**: Ahmed Zeid, Sidney Bender  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.17048v1  

#### Abstract
Modern neural networks achieve strong performance but remain difficult to interpret in high-dimensional visual domains. Counterfactual explanations (CFEs) provide a principled approach to interpreting black-box predictions by identifying minimal input changes that alter model outputs. However, exist...

---

### 22. [Translation Invariance of Neural Operators for the FitzHugh-Nagumo Model](https://arxiv.org/abs/2603.17523)

**Authors**: Luca Pellegrini  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.17523v1  

#### Abstract
Neural Operators (NOs) are a powerful deep learning framework designed to learn the solution operator that arise from partial differential equations. This study investigates NOs ability to capture the stiff spatio-temporal dynamics of the FitzHugh-Nagumo model, which describes excitable cells. A key...

---

### 23. [Flow Matching Policy with Entropy Regularization](https://arxiv.org/abs/2603.17685)

**Authors**: Ting Gao, Stavros Orfanoudakis, Nan Lin, Elvin Isufi, Winnie Daamen, Serge Hoogendoorn  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.17685v1  

#### Abstract
Diffusion-based policies have gained significant popularity in Reinforcement Learning (RL) due to their ability to represent complex, non-Gaussian distributions. Stochastic Differential Equation (SDE)-based diffusion policies often rely on indirect entropy control due to the intractability of the ex...

---

### 24. [Symmetry-Reduced Physics-Informed Learning of Tensegrity Dynamics](https://arxiv.org/abs/2603.17824)

**Authors**: Jing Qin, Muhao Chen  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.17824v1  

#### Abstract
Tensegrity structures possess intrinsic geometric symmetries that govern their dynamic behavior. However, most existing physics-informed neural network (PINN) approaches for tensegrity dynamics do not explicitly exploit these symmetries, leading to high computational complexity and unstable optimiza...

---

### 25. [MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment](https://arxiv.org/abs/2603.17694)

**Authors**: Yusen Wu, Yiran Liu, Xiaotie Deng  
**Category**: cs.AI  
**Published**: 2026-03-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.17694v1  

#### Abstract
In the real economy, modern decision-making is fundamentally challenged by high-dimensional, multimodal environments, which are further complicated by agent heterogeneity and combinatorial data sparsity. This paper introduces a Multi-Agent Large Language Model-based Economic Sandbox (MALLES), levera...

---

### 26. [SIA: A Synthesize-Inject-Align Framework for Knowledge-Grounded and Secure E-commerce Search LLMs with Industrial Deployment](https://arxiv.org/abs/2603.16137)

**Authors**: Zhouwei Zhai, Mengxiang Chen, Anmeng Zhang  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.16137v1  

#### Abstract
Large language models offer transformative potential for e-commerce search by enabling intent-aware recommendations. However, their industrial deployment is hindered by two critical challenges: (1) knowledge hallucination due to insufficient encoding of dynamic, fine-grained product knowledge, and (...

---

### 27. [Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR](https://arxiv.org/abs/2603.16184)

**Authors**: Quy-Anh Dang, Chris Ngo  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.16184v1  

#### Abstract
We present Polyglot-Lion, a family of compact multilingual automatic speech recognition (ASR) models tailored for the linguistic landscape of Singapore, covering English, Mandarin, Tamil, and Malay. Our models are obtained by fine-tuning Qwen3-ASR-0.6B and Qwen3-ASR-1.7B exclusively on publicly avai...

---

### 28. [Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning](https://arxiv.org/abs/2603.15981)

**Authors**: Jingxiang Chen, Minseok Kim, Seong-Gyun Leem, Yin Huang, Rashi Rungta, Zhicheng Ouyang, Haibin Wu, Surya Teja Appini, Ankur Bansal, Yang Bai, Yue Liu, Florian Metze, Ahmed A Aly, Anuj Kumar, Ariya Rastrow, Zhaojiang Lin  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.15981v1  

#### Abstract
Speech large language models (LLMs) observe paralinguistic cues such as prosody, emotion, and non-verbal sounds--crucial for intent understanding. However, leveraging these cues faces challenges: limited training data, annotation difficulty, and models exploiting lexical shortcuts over paralinguisti...

---

### 29. [Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization](https://arxiv.org/abs/2603.16105)

**Authors**: Francesco Pio Monaco, Elia Cunegatti, Flavio Vella, Giovanni Iacca  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.16105v1  

#### Abstract
Post-training model compression is essential for enhancing the portability of Large Language Models (LLMs) while preserving their performance. While several compression approaches have been proposed, less emphasis has been placed on selecting the most suitable set of data (the so-called \emph{calibr...

---

### 30. [Structured Semantic Cloaking for Jailbreak Attacks on Large Language Models](https://arxiv.org/abs/2603.16192)

**Authors**: Xiaobing Sun, Perry Lam, Shaohua Li, Zizhou Wang, Rick Siow Mong Goh, Yong Liu, Liangli Zhen  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.16192v1  

#### Abstract
Modern LLMs employ safety mechanisms that extend beyond surface-level input filtering to latent semantic representations and generation-time reasoning, enabling them to recover obfuscated malicious intent during inference and refuse accordingly, and rendering many surface-level obfuscation jailbreak...

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
