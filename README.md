# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-19 06:43:00 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression](https://arxiv.org/abs/2603.17435)

**Authors**: Ruibo Fan, Xiangrui Yu, Xinglin Pan, Zeyu Li, Weile Luo, Qiang Wang, Wei Wang, Xiaowen Chu  
**Category**: cs.DC  
**Published**: 2026-03-19  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2603.17435v1  

#### Abstract
Lossless model compression holds tremendous promise for alleviating the memory and bandwidth bottlenecks in bit-exact Large Language Model (LLM) serving. However, existing approaches often result in substantial inference slowdowns due to fundamental design mismatches with GPU architectures: at the k...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《ZIPSERV: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **lossless compression** 方法在 LLM 推理中面临严重性能瓶颈，尽管能节省存储空间，但在 GPU 上推理时通常导致显著的运行时开销。其根本原因在于：
- **Kernel-level 问题**：传统熵编码（如 Huffman、ANS）产生变长比特流，破坏了 GPU 的 **SIMT 并行执行模型**，导致控制流发散和计算资源利用率低下。
- **System-level 问题**：主流框架采用“先解压再计算”的解耦流水线，导致中间缓冲区冗余，增加了内存带宽压力，降低了 **compute intensity**。

### **提出的新方法与创新思路**
ZIPSERV 是首个为高效 LLM 推理而协同设计的 **硬件感知无损压缩框架**，核心创新包括：

#### **(1) Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE)**
- 利用 LLM 中 BFloat16 权重的指数位分布高度偏斜且连续的特性（top-7 指数覆盖 >95% 权重），设计了一种**固定长度、基于位图（bitmap）的编码格式**。
- 将每个 8×8 权重块编码为三个 64-bit 位图（分别表示 3-bit 编码的每一位），以及两个紧凑值缓冲区（高频值的符号+尾数、异常值的完整 BF16）。
- **优势**：支持常数时间、并行解码，避免控制流发散，完美契合 GPU 的 SIMT 执行模型。

#### **(2) 融合解压-GEMM 内核（ZipGEMM）**
- 设计了一个全新的 **fused kernel**，在 Tensor Core 执行 GEMM 时**即时解压权重**，直接将解压后的数据送入寄存器。
- 实现“**load-compressed, compute-decompressed**”的设计，消除中间全局内存缓冲区，最大化计算强度（compute intensity）。

#### **(3) 阶段感知推理策略（Stage-Aware Inference Strategy）**
- 在 **decode 阶段**（内存受限）使用融合的 ZipGEMM，最大化带宽效率。
- 在 **prefill 阶段**（计算密集）使用解耦的解压 + cuBLAS GEMM，以摊销解压开销。

---

### **相比现有方法的优势**
| 维度 | 传统方法（如 DFloat11, DietGPU） | ZIPSERV |
|------|-------------------------------|--------|
| **解码方式** | 变长熵编码，串行依赖强 | 固定长度位图，完全并行 |
| **执行模式** | 解压 → 存全局内存 → GEMM | 解压与 GEMM 融合，直接进寄存器 |
| **SIMT 兼容性** | 差，线程发散严重 | 优，零分支，对齐 warp 执行 |
| **内存访问** | 多次冗余读写 | 显著减少，提升 compute intensity |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **模型家族**：LLaMA3.1（8B, 70B, 405B）、Qwen2.5（7B–72B）、Gemma3（12B, 27B）、Mistral（24B, 123B）
- **关注层**：Transformer 块中的关键线性层，包括：
  - `QKV_proj`, `O_proj`, `GateUp_proj`, `Down_proj`, `LM Head`

### **实验设置**
- **硬件平台**：
  - **消费级**：4× RTX 4090（24GB, CC 8.9）
  - **数据中心级**：4× L40S（48GB, CC 8.9）
  - **前瞻性测试**：RTX 5090（Blackwell 架构, CC 12.0）
- **软件环境**：GCC 11.3, NVCC 12.4（RTX 5090 使用 12.8）
- **批大小（Batch Size）**：8, 16, 32（end-to-end）；N=1~2048（分析不同序列长度影响）

### **评估指标**
- **Kernel-level**：
  - GEMM 执行时间（normalized speedup）
  - 内存带宽（DRAM read）
  - ALU/Tensor Core 利用率
- **End-to-end**：
  - **端到端延迟（latency）**
  - **吞吐量（throughput, tokens/sec）**
  - **内存占用（weight footprint, KV cache size）**

### **基线方法对比**
- **cuBLAS_TC**：NVIDIA 官方 BF16 Tensor Core GEMM（黄金标准）
- **DietGPU**：基于 rANS 的开源 GPU 解压方案
- **nvCOMP (rANS)**：NVIDIA 通用解压库
- **DFloat11**：最先进的 Huffman 编码 LLM 推理框架
- **vLLM**：当前最流行的 LLM 服务框架（用于 end-to-end 对比）
- **Transformers**：HuggingFace 标准库

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) Kernel-level 性能（ZipGEMM vs cuBLAS_TC）**
- **平均加速比**：
  - RTX 4090：**1.31×**
  - L40S：**1.36×**
- **峰值加速比**：
  - L40S 上达 **2.21×**（`GateUp_proj` 层）
- **对比其他 lossless 方法**：
  - ZipGEMM 比 DFloat11 快 **5.53×**
  - 比 DietGPU 和 nvCOMP 快 **4–5×**

#### **(2) 端到端推理性能（vs vLLM）**
- **平均延迟降低**：**17.6%**
- **平均吞吐量提升**：**1.22×**
- **长序列生成（2048 tokens, BS=32）**：
  - 吞吐量达 **1105 tokens/sec**，比 vLLM 快 **1.66×**

#### **(3) 模型压缩率**
- **平均模型尺寸缩减**：**30%**
  - LLaMA3.1-8B：14.96 GB → **10.83 GB**（压缩至 72.4%）
  - Mistral-24B：43.92 GB → **31.30 GB**（71.3%）
  - LLaMA3.1-70B：131.56 GB → **93.52 GB**（71.1%）

#### **(4) 内存利用优化**
- 压缩释放的内存被自动用于扩展 **KV Cache**：
  - LLaMA3.1-8B 上 KV Cache 从 5.07 GB → **8.60 GB**（+70%）
  - 支持更长上下文和更大 batch size

#### **(5) 微观分析（Nsight Compute）**
- **DRAM 读取减少 29.3%**
- **共享内存 bank conflict 几乎消除**（~4.7K vs DietGPU 的百万级）
- **Tensor Core 利用率达 cuBLAS 的 71.6%**，说明解压开销被有效隐藏

#### **(6) 消融实验（Ablation Study）**
- **TCA-TBE 编码本身**：即使单独作为解压内核（ZIPSERV-Decomp），也比 DietGPU 快 **2.14×**，证明其高效性。
- **阶段感知策略**：
  - Decode 阶段使用 ZipGEMM：显著加速
  - Prefill 阶段切换回 cuBLAS：仅引入 **~4%** 额外开销，性价比高

---

## **4. 关键结论和发现**

### **主要发现**
1. **Lossless compression 不应牺牲性能**：通过硬件协同设计，ZIPSERV 首次证明无损压缩不仅能节省存储，还能**直接加速 LLM 推理**。
2. **指数位的统计特性是突破口**：LLM 权重中 BFloat16 指数位的高度偏斜和连续性（top-K contiguity）是实现高效压缩的关键。
3. **融合执行是关键**：“解压-计算”融合消除了中间内存瓶颈，将理论压缩收益转化为实际性能增益。
4. **消费级 GPU 可逼近数据中心性能**：
   - 在 RTX 4090 上，ZIPSERV 比 A100 上的 cuBLAS 快 **9.3%**
   - 在 RTX 5090 上，与 H800 的差距从 53.3% 缩小至 **14.1%**

### **方法的局限性**
- **对小矩阵层加速有限**：如 `O_proj` 等小形状层因难以充分调度硬件，可能略有降速（如 0.79×）。
- **依赖特定硬件特性**：目前针对 NVIDIA Tensor Core 优化，移植到其他架构需适配。
- **训练期不适用**：当前聚焦推理，未考虑训练中的动态权重更新。

### **未来工作方向**
1. **扩展至 KV Cache 压缩**：将 TCA-TBE 应用于 key/value 缓存，缓解长上下文内存瓶颈。
2. **跨架构支持**：适配 Intel AMX、AMD Matrix Cores 等其他矩阵加速单元。
3. **与 lossy 方法结合**：在量化模型上进一步应用无损压缩，实现双重压缩。
4. **系统级集成**：用于模型检查点压缩、分布式训练通信压缩等场景。

---

> **总结**：ZIPSERV 重新定义了 lossless compression 在 LLM 推理中的角色——从“存储工具”变为“性能引擎”。它通过 **TCA-TBE + ZipGEMM** 的协同设计，解决了算法与 GPU 架构的根本错配，首次实现了**无损压缩下的推理加速**，为在资源受限设备上部署大模型提供了全新路径。

</details>

---

### 2. [SENSE: Efficient EEG-to-Text via Privacy-Preserving Semantic Retrieval](https://arxiv.org/abs/2603.17109)

**Authors**: Akshaj Murhekar, Christina Liu, Abhijit Mishra, Shounak Roychowdhury, Jacek Gwizdka  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.17109v1  

#### Abstract
Decoding brain activity into natural language is a major challenge in AI with important applications in assistive communication, neurotechnology, and human-computer interaction. Most existing Brain-Computer Interface (BCI) approaches rely on memory-intensive fine-tuning of Large Language Models (LLM...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SENSE: Efficient EEG-to-Text via Privacy-Preserving Semantic Retrieval 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **Brain-Computer Interface (BCI)** 系统在将脑电图（**EEG**）信号解码为自然语言时，通常依赖于对 **Large Language Models (LLMs)** 或编码器-解码器模型进行端到端的微调（fine-tuning）。这种方法存在以下问题：
- **计算开销大**：需要大量计算资源进行训练。
- **隐私风险高**：原始神经数据需上传至云端参与训练，暴露敏感信息。
- **可扩展性差**：模型与特定 LLM 耦合紧密，难以适应快速演进的生成式 AI。

### 提出的新方法
本文提出 **SENSE (SEmantic Neural Sparse Extraction)**，一种轻量级、隐私保护的 EEG-to-Text 框架，其核心思想是**解耦神经解码与语言生成过程**，采用“**语义检索 + 提示生成**”（retrieval-augmented prompting）范式。

#### 方法流程如下：
1. **On-device 语义检索**：在本地设备上将 EEG 信号映射为离散的 **Bag-of-Words (BoW)** 表示（即一组关键词），不涉及任何 LLM 微调。
2. **Prompt-based 文本生成**：将提取的关键词作为提示（prompt）输入现成的 LLM（如 GPT-4o-mini、Gemini），由其合成流畅文本。

### 相比现有方法的优势
| 维度 | SENSE | 传统方法（如 THOUGHT2TEXT） |
|------|-------|-----------------------------|
| **是否微调 LLM** | ❌ 否（仅提示） | ✅ 是（端到端微调） |
| **参数规模** | ~6M（仅 EEG-to-keyword 模块） | 数亿至数十亿 |
| **计算效率** | 高，可在边缘设备运行 | 低，需高性能 GPU |
| **隐私保护** | 强，原始 EEG 不离开本地 | 弱，需上传原始数据 |
| **模块化与可扩展性** | 高，可灵活更换 LLM | 低，与特定模型绑定 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用公开的 **CVPR2017EEG-ImageNet** 数据集。
- 包含 **6 名受试者** 对 **2,000 张图像** 的同步 EEG 记录与自然语言描述。
- 总样本数：9,940（训练：7,959，验证：1,994，测试：1,987）。
- 每个样本为 `(EEG, caption)` 对，用于模拟视觉感知下的脑信号到文本任务。

### 实验设置
- **EEG 编码器**：采用预训练的 **ChannelNet** 架构（冻结），输出 512D 向量。
- **跨模态投影器（Similarity Refiner）**：一个多层感知机（MLP），将 EEG 特征对齐到 **CLIP 文本嵌入空间**。
- **关键词提取**：通过余弦相似度从固定词汇表（V=1,210）中检索 Top-15 最相关的词，形成 BoW。
- **LLM 接口**：使用 API 调用多种 LLM 进行零样本生成：
  - GPT-4o-mini
  - Gemini 2.5 Flash Lite
  - LLaMA-3-8B
  - Qwen2.5-7B

### 评估指标
- **自动评估指标**：
  - **BLEU-1/4**, **ROUGE-1/2/L**, **METEOR**, **BERTScore**
- **人工评估替代方案**：
  - 使用 **GPT-5** 对生成文本进行双维度评分（1–5 分）：
    - **Fluency**（流畅性）
    - **Adequacy**（充分性，即语义忠实度）

### 基线方法对比
- 主要对比 **THOUGHT2TEXT (Mishra et al., 2025)** —— 当前最先进的端到端微调方法。
- 同时设置了多个消融变体：
  - **Naive Baseline**：直接使用原始 EEG 嵌入进行检索
  - 不同损失函数版本：**BCE**, **Contrastive Multi-Label Loss**, **Focal Loss**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 ROUGE-1 和 GPT-5 Adequacy 为代表）

| 方法 | LLM | ROUGE-1 | ROUGE-2 | BERTScore | GPT-5 Flu. | GPT-5 Ade. |
|------|-----|---------|---------|-----------|------------|------------|
| **SENSE (Focal Loss)** | Gemini 2.5 Flash Lite | **31.5** | **8.5** | 0.897 | **4.77** | **1.40** |
| **SENSE (Focal Loss)** | GPT-4o-mini | 30.6 | 8.2 | 0.898 | 4.75 | 1.40 |
| **THOUGHT2TEXT** | LLaMA-3-8B | 30.0 | 8.1 | 0.890 | 4.82 | 1.58 |
| **THOUGHT2TEXT** | Qwen2.5-7B | 26.4 | 4.6 | 0.880 | 4.75 | 1.28 |

> ✅ **关键发现**：  
> - SENSE 在 **ROUGE-1** 上 **超过 THOUGHT2TEXT**（31.5 vs. 30.0），且在多数指标上表现相当甚至更优。
> - 尽管未微调 LLM，**闭源模型（Gemini/GPT）能有效利用 BoW 提示**，生成高质量文本。
> - 开源模型（LLaMA/Qwen）在无对象锚点时表现较差，说明其对结构化提示的依赖更强。

### 消融实验结果
#### （1）不同损失函数的影响
- **Focal Loss** 效果最佳，显著优于 BCE 和 Contrastive Loss。
- 原因：Focal Loss 显式缓解了标签极端不平衡问题（平均仅激活 5 个词 / 1210），提升稀有概念的召回率。

#### （2）是否提供主对象标签（withObj vs. withoutObj）
| 设置 | LLM | ROUGE-1（Focal Loss） | GPT-5 Ade. |
|------|-----|------------------------|------------|
| withObj | Gemini | **31.5** | **1.40** |
| withoutObj | Gemini | 30.5 | 1.30 |
| withObj | GPT-4o-mini | 30.6 | 1.40 |
| withoutObj | GPT-4o-mini | 30.3 | 1.27 |

> 🔍 发现：主对象标签作为“语义锚点”有助于稳定生成，但即使移除，LLM 仍能基于 BoW 恢复大致场景。

#### （3）定性分析（见 Table 2）
- SENSE 能生成包含丰富细节的描述（如 “A yellow mushroom growing on a log surrounded by grass”）。
- 错误主要来源于 Stage-1 对象预测错误（如将 mushroom 误判为 flower），导致后续生成偏差（hallucination）。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **无需微调也能实现高质量 EEG-to-Text**：通过将连续 EEG 映射为离散语义关键词，并结合 prompt-based LLM 生成，即可达到甚至超越全模型微调的效果。
2. ✅ **语义对齐至关重要**：将 EEG 嵌入对齐到 **CLIP 文本空间** 可实现跨模态语义检索，支持有效的关键词推断。
3. ✅ **隐私与效率兼得**：整个 EEG 解码过程可在本地完成，仅共享抽象语义词，大幅降低隐私泄露风险。
4. ✅ **框架具有强泛化能力**：单一模型在 **6 名不同受试者** 上均表现一致，无需个体校准（subject-agnostic）。

### 局限性
- **EEG 本身分辨率低**：非侵入式 EEG 信噪比低，易混淆视觉或语义相近类别（如 mushroom vs. flower）。
- **词汇受限**：当前仅使用 1,210 词的封闭词汇表，限制了表达多样性，尚不能实现开放词汇生成。
- **依赖外部 LLM API**：虽然 EEG 处理本地化，但最终生成仍需信任第三方服务（除非部署本地 LLM）。
- **错误传播风险**：若关键词提取失败，LLM 可能完全脱离真实神经状态，“自由发挥”生成无关内容（risk of ungrounded generation）。

### 未来工作方向
- 改进关键词提取机制：
  - 从静态 Top-k 检索转向动态置信阈值选择。
  - 引入句法先验指导概念组合。
- 扩展语义词汇空间，迈向 open-vocabulary thought-to-text。
- 加强跨模态对齐（如融合视觉上下文）。
- 实现实时多传感器融合，构建更丰富的脑驱动语言接口。
- 探索完全本地化的高效小规模 LLM 部署，实现端到端私密推理。

---

> 📌 **总结一句话**：  
> **SENSE 证明了“轻量级语义检索 + 提示生成”的架构可以在不牺牲性能的前提下，显著提升 EEG-to-Text 系统的隐私性、效率和可扩展性，为下一代 BCI 提供了一种可行的新范式。**

</details>

---

### 3. [BATQuant: Outlier-resilient MXFP4 Quantization via Learnable Block-wise Optimization](https://arxiv.org/abs/2603.16590)

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
当前主流的 Post-Training Quantization (PTQ) 方法（如 QuaRot、SpinQuant）在应用于 **MXFP4**（Microscaling Floating-Point 4-bit）格式时表现严重退化，甚至不如基础的 RTN 方法。其根本原因在于：

- **全局正交变换**（global orthogonal rotations）会将异常值（outliers）的能量跨量化块传播，破坏 MXFP 的局部块级缩放机制。
- 这些变换还会导致激活分布出现 **双峰分布**（bimodal distribution），浪费有限的量化范围，降低精度。

### 提出了什么新方法或新思路
本文提出 **BATQuant**（Block-wise Affine Transformation Quantization），一种专为 MXFP4 设计的鲁棒量化框架，核心创新包括：

- **Block-wise Affine Transformation (BAT)**  
  将仿射变换限制在与 MXFP 量化粒度对齐的块内（如每 32 个元素一个块），避免跨块能量转移，保留每个块的独立统计特性。

- **Global and Private Kronecker (GPK) 分解**  
  为解决可学习块仿射矩阵带来的参数开销，提出 GPK 分解：共享一个全局变换矩阵 $ A \in \mathbb{R}^{g_1 \times g_1} $，同时每个块拥有私有小矩阵 $ B_i \in \mathbb{R}^{g_2 \times g_2} $，显著减少存储和计算开销。

- **Block-wise Learnable Clipping**  
  引入可学习的逐块裁剪阈值，动态抑制残余异常值，进一步提升量化稳定性。

### 相比现有方法的优势
| 特性 | BATQuant | 传统方法（如 QuaRot, BRQ） |
|------|----------|-----------------------------|
| 变换粒度 | 块级（block-wise） | 全局或块级旋转 |
| 能量传播 | 阻止跨块传播 | 易造成跨块干扰 |
| 分布形态 | 单峰紧凑分布 | 易产生双峰分布 |
| 参数效率 | GPK 极大压缩参数量 | 参数量高或无优化 |
| 对 MXFP 适配性 | 完全对齐硬件量化粒度 | 存在格式不匹配 |

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 多模态任务（MLLM）
- **MME**：综合多模态理解评测
- **OCRBench**：OCR 能力评测
- **DocVQA**：文档图像问答
- **RealWorldQA**：现实场景空间推理
- **VLMBlind**：低层视觉几何识别任务

#### 语言模型任务（LLM）
- **非推理任务**：PIQA（物理常识）、Winogrande（共指消解）、Hellaswag（常识推理）、ARC-Easy/Challenge（科学题）
- **复杂推理任务**：GSM8K（小学数学）、MATH-500、AIME24/AIME25（奥数题）、GPQA-D（博士级问题）

### 实验设置和评估指标
- **模型**：Qwen3-VL-8B-Instruct（MLLM）、Qwen3-8B（LLM）
- **量化配置**：采用 `W{w}A{a}KV{k}` 表示法，例如：
  - `W4A4KV16`：权重 4-bit，激活 4-bit，KV Cache 16-bit
  - `W4A8KV16`：更宽松设置下的近无损目标
- **评估指标**：
  - 多模态：各基准平均得分及相对于 BF16 的 **恢复率（Recovery Rate）**
  - 推理任务：准确率（Accuracy @1 或 Avg@N）

### 基线方法对比
- **RTN**：直接舍入量化
- **QuaRot / SpinQuant**：基于旋转的 INT4 成功方法
- **BRQ**：针对 MXFP 的块级旋转方法
- **FlatQuant**：全局仿射平坦化方法
- **SmoothQuant**：通过迁移难度到权重来平滑激活
- **GPTQ**：基于 Hessian 的权重量化方法

所有方法均测试其与 GPTQ 结合后的变体以公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 量化配置 | 方法 | 平均恢复率（MLLM） | 平均恢复率（LLM 推理） |
|---------|-------|--------------------|------------------------|
| W4A8KV16 | BATQuant | **99.29%** | **97.46%** |
| W4A4KV16 | BATQuant | **96.43%** | **92.45%** |
| W4A8KV8 | BATQuant | **98.89%** | — |
| W4A8KV4 | BATQuant | **97.51%** | — |

> ✅ 在 `W4A8KV16` 下接近无损；在极具挑战性的 `W4A4KV16` 下仍能恢复超过 **96%** 的原始性能。

### 与基线方法的对比结果
- 在 **所有量化配置下**，BATQuant 均取得 **state-of-the-art** 性能。
- 在 `W4A4KV16` 设置中：
  - 比最强基线 **FlatQuant** 高出 **1.64%**（MLLM）
  - 比 **GPTQ** 高出 **1.35%**（LLM 推理）
- 在复杂推理任务（如 GSM8K、AIME）上优势尤为明显，说明其对误差累积具有更强鲁棒性。
- 可视化显示：BATQuant 成功消除双峰分布，保持单峰紧凑结构，而 BRQ 和 FlatQuant 仍存在分布畸变。

### 消融实验结果
#### （1）模块有效性（Ablation Study）
| 组件组合 | MLLM Recovery (%) | LLM Avg Accuracy |
|--------|-------------------|------------------|
| 无 BAT + 无 Clipping | 95.59% | 68.24% |
| 仅 BAT | 96.18% | 68.51% |
| 仅 Clipping | 95.59% | 68.24% |
| BAT + Clipping (**完整版**) | **96.43%** | **68.70%** |

✅ 表明 **块级仿射变换** 和 **可学习裁剪** 均为关键组件。

#### （2）GPK 参数敏感性分析
- 最优设置为：$ g_1 = 8, g_2 = 4 $
- 当 $ g_1 $ 过大 → 私有部分太小 → 缺乏灵活性
- 当 $ g_1 $ 过小 → 参数过多 → 优化困难
- 推荐默认配置可在 **精度与效率之间取得最佳平衡**

#### （3）变换块大小影响
- 最佳性能出现在 **变换块大小 = MXFP 量化块大小 = 32**
- 若小于 32 → 无法充分平滑异常值
- 若大于 32 → 引发跨块能量泄漏 → 性能下降

---

## 4. 关键结论和发现

### 主要发现
1. **MXFP4 量化失败的根本原因是“格式不匹配”**：传统旋转方法设计用于整数量化，强行用于 MXFP 会导致跨块干扰和分布畸变。
2. **块级仿射变换是适配 MXFP 的理想选择**：它既能灵活重塑分布，又能防止异常值扩散。
3. **GPK 分解实现了高效参数共享**：在几乎不影响性能的前提下大幅降低内存占用（相比 FlatQuant 减少 >74% 参数）。
4. **方法具备强跨模态泛化能力**：在 MLLM 和 LLM 上均表现出色，适用于从 OCR 到复杂数学推理的广泛任务。

### 方法的局限性
- 当前 GPK 的最优维度需手动调参（如 $ g_1=8 $），尚未实现完全自动化搜索。
- 在极端低比特（如 W2）下未验证，可能面临新的挑战。
- 训练阶段需要少量校准数据（约 128 条），虽成本低但仍非完全零样本。

### 未来工作方向
- 将 BATQuant 扩展至 **训练中量化**（Quantization-Aware Training, QAT）
- 探索 **自动结构搜索** 以确定最优 GPK 维度
- 应用于更多硬件平台（如 Ascend NPU、TPU）验证通用性
- 结合 **稀疏化** 与 **混合精度** 进一步提升端侧部署效率

---

> 🔚 **总结**：BATQuant 是首个真正适配 MXFP4 格式的高性能 PTQ 框架，通过 **块级仿射变换 + GPK 分解 + 可学习裁剪** 的组合，在保持高参数效率的同时解决了跨块异常值传播和双峰分布问题，为大模型在新一代浮点量化硬件上的高效部署提供了实用解决方案。

</details>

---

### 4. [SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation](https://arxiv.org/abs/2603.16219)

**Authors**: Hang Lv, Sheng Liang, Hao Wang, Yongyue Zhang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 7.5  
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
该论文旨在解决**个性化生成中的核心困境**：  
- **集中式大模型（LLM）** 虽具备强大的推理能力，但需要上传用户历史数据，存在**隐私泄露风险**和延迟瓶颈。  
- **本地小模型（SLM）** 可在设备端运行以保护隐私，但受限于容量，缺乏复杂推理能力，生成质量低，易出现逻辑错误或幻觉。

现有方法（如 RAG、LoRA、Sequential Fusion）无法有效平衡**隐私保护**与**高质量生成**之间的矛盾。

---

### **提出的新方法：SPECSTEER**
作者提出 **SPECSTEER**，一种**非对称的边缘-云协同推理框架**，通过“**Draft-Verify-Recover**”三阶段流程，实现本地上下文与云端推理的高效融合。

#### **核心创新点：**
1. **将协作推理建模为贝叶斯知识融合（Bayesian Knowledge Fusion）问题**  
   - 形式化定义了一个目标分布 $ \pi^*(y) \propto P_{LLM}(y) \cdot \frac{P_{SLM}(y)}{P_{\bar{SLM}}(y)} $，其中：
     - $ P_{LLM} $：云端通用模型的先验（保证逻辑一致性）
     - $ \frac{P_{SLM}}{P_{\bar{SLM}}} $：基于 Pointwise Mutual Information (PMI) 的个性化意图度量
   - 该公式实现了**逻辑稳健性**与**个性化保真度**的数学统一。

2. **重构 Speculative Decoding 作为分布式对齐协议**  
   - 将传统的加速机制（Speculative Decoding）重新设计为**跨隐私边界的协作范式**：
     - **Drafting**：本地 Specialist 生成个性化候选序列。
     - **Verification**：云端 Generalist 执行**比率验证（Ratio-Based Verification）**，即判断 $ \alpha = \min\left(1, \frac{P_{LLM}(y)}{\lambda \cdot P_{\bar{SLM}}(y)}\right) $，避免因未知私有实体而误拒。
     - **Recovery**：若拒绝，则执行**Steering Recovery**，注入本地 logit 差值 $ h_{SLM} - h_{\bar{SLM}} $ 进行修正，保留个性化信号。

3. **通信高效的设计**
   - 验证仅需传输 token IDs，无需发送 logits 或原始上下文。
   - 恢复阶段采用 top-k 稀疏向量回传，大幅降低带宽开销。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | SPECSTEER |
|------|--------|----------|
| **隐私保护** | 需上传 context 或参数 | 用户 context 始终留在本地 |
| **推理能力** | 依赖本地模型能力 | 利用云端 LLM 强大推理 |
| **对齐精度** | 粗粒度文本交换或全量 logits 同步 | 细粒度、分布级对齐 |
| **实时效率** | 高频同步导致高延迟 | 仅在拒绝时通信，支持高吞吐 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LaMP (Personalized Language Model Performance)** (Salemi et al., 2023)
- **LongLaMP** (Kumar et al., 2024)：更长文本、更强个性化需求的任务集合

选取以下生成任务进行评估：
- **LongLaMP-2**: 个性化摘要生成（Pers. Abstract Gen）
- **LongLaMP-3**: 个性化评论写作（Pers. Review Writing）
- **LongLaMP-4**: 个性化话题撰写（Pers. Topic Writing）

---

### **实验设置**
- **模型组合**（共四组）：
  1. Qwen3-0.6B (SLM) / Qwen3-32B (LLM)
  2. Qwen2.5-1.5B / Qwen2.5-32B
  3. Qwen3-8B / Qwen3-32B
  4. Llama-3.1-2.1B / Llama-3.1-8B （跨架构测试）

- **本地增强策略**：
  - **SLM+**：基于 BM25/BGE 的 RAG 增强
  - **LoRA** 微调（部分消融实验中使用）

- **评估指标**：
  - **ROUGE-1 (R1)** 和 **ROUGE-L (RL)**：衡量生成内容与参考摘要的重叠度
  - **Speedup**：推理速度提升倍数（tokens/sec）
  - **Acceptance Rate (α%)**：草案被接受的比例

- **默认超参**：
  - 验证阈值 $ \lambda = 0.5 $
  - 恢强强度 $ \beta = 1.0 $

---

### **基线方法对比**
| 类型 | 方法 | 描述 |
|------|------|------|
| **本地增强** | SLM (Direct), SLM+ (RAG), LoRA, LoRA+RAG | 在设备上增强小模型 |
| **纯云端模型** | LLM (32B Direct) | 不访问用户 context 的零样本大模型 |
| **协作推理** | CoSteer, LightCoSteer, Standard SD | Token-level 融合或标准投机解码 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2 & 3）**

| 方法 | Abs. R1 | Abs. RL | Rev. R1 | Rev. RL | Wri. R1 | Wri. RL |
|------|--------|--------|--------|--------|--------|--------|
| SLM (0.6B) | 36.58 | 20.34 | 24.15 | 12.95 | 26.32 | 12.72 |
| SLM+ (RAG) | 39.89 | 21.65 | 23.18 | 12.84 | 25.50 | 12.36 |
| LLM (32B) | 40.18 | 22.17 | 31.18 | 14.78 | 29.46 | 12.64 |
| **SPECSTEER** | **41.35** | **23.57** | **33.03** | **17.26** | **30.79** | **12.88** |

✅ **SPECSTEER 在所有任务上均优于最强基线（LLM）**，尤其在 **Review 写作任务上提升显著（+1.85 R1）**。

---

### **与更广泛基线对比（Table 3）**
- SPECSTEER 显著优于各类方法：
  - **检索类**：BM25, BGE, PAG
  - **PEFT类**：TAM, OPPU, PAD, CoPE
- 在 **Review 任务** 上，SPECSTEER 达到 **33.03 R1**，远超第二名 CoPE（28.54 R1），表明其在复杂推理任务上的优势。

---

### **效率表现（Table 4）**
| 方法 | Speed (tok/s) | Speedup | Acceptance (%) |
|------|--------------|---------|----------------|
| Vanilla LLM | 22.58 | 1.00× | — |
| CoSteer | 9.71 | 0.43× | — |
| LightCoSteer | 16.03 | 0.71× | — |
| Standard SD | 22.13 | 0.98× | ~35% |
| **SPECSTEER ($\lambda=0.1$)** | **53.29** | **2.36×** | **~86%** |

🚀 **SPECSTEER 实现了 2.36× 的推理加速**，同时保持高质量输出，显著优于其他协作方法（多数反而变慢）。

---

### **消融实验结果**
#### **(1) 验证阈值 $\lambda$ 的影响（Table 10）**
- $\lambda = 0.1$：接受率高达 86%，速度最快（2.36×），质量略有下降但仍优于 LLM。
- $\lambda = 1.0$：质量最高，但接受率仅 ~37%，速度仅 1.12×。
- ✅ 最佳权衡区间：$\lambda \in [0.1, 0.5]$

#### **(2) 恢强系数 $\beta$ 的影响（Table 9）**
- $\beta \in [0.5, 2.0]$：性能稳定，优于基线。
- $\beta > 2.5$：过度强调个性化，导致逻辑断裂，性能下降。
- ✅ 推荐值：$\beta = 1.0$

#### **(3) 跨架构部署（Table 8）**
- 使用 **Qwen3-0.6B (Specialist)** + **Llama-3.1-8B (Generalist)**：
  - Review R1 达到 **32.03**，优于单独 SLM+（23.18）和 LLM（31.71）
- 表明 SPECSTEER **不依赖特定模型家族**，具有良好的泛化性。

#### **(4) 对噪声 Specialist 的鲁棒性（Table 7）**
- 即使本地模型输入被注入噪声或使用弱检索（BM25），SPECSTEER 仍能通过云端验证修复错误，性能接近甚至超过纯净设置下的 LLM。
- 证明其作为“**逻辑过滤器**”的有效性。

#### **(5) 计算成本分析（Table 11）**
- 在长上下文（10k tokens）下，SPECSTEER 的总 FLOPs 比 LLM-RAG 降低 **近 3.5 倍**。
- 因为 LLM 不处理原始长 context，计算负担轻。

---

## **4. 关键结论和发现**

### **主要发现**
1. **本地模型存在“能力鸿沟”（Capacity Deficit）**  
   - 即使使用 RAG、LoRA 等增强手段，小型本地模型也无法匹敌未接触用户数据的大模型，说明**推理能力是制约因素而非信息缺失**。

2. **SPECSTEER 成功弥合了这一差距**  
   - 通过将 **Speculative Decoding 重构为协作协议**，实现了：
     - **隐私保护**：用户 context 不出设备
     - **高质量生成**：利用云端 LLM 的强大推理
     - **高效率**：2.36× 加速，通信开销极低

3. **框架具有强鲁棒性和泛化性**  
   - 对噪声、弱检索、不同模型架构均表现稳定
   - 可与 RAG、LoRA 等本地优化方法正交结合

---

### **局限性**
- 当本地 Specialist 完全失效（如微调崩溃）时，个性化增益会减弱，尽管 LLM 仍可维持基本连贯性。
- 恢复阶段依赖 logit 差值，假设两个 Specialist 模型结构一致（本地 vs 云端副本）。
- 当前未集成更复杂的隐私保护机制（如差分隐私、同态加密），但论文指出其可模块化扩展。

---

### **未来工作方向**
- 探索动态调整 $\lambda$ 和 $\beta$ 以适应不同任务难度。
- 结合安全多方计算（MPC）或联邦学习进一步强化隐私保障。
- 扩展至多轮对话、Agent 规划等更复杂场景。
- 支持更多异构模型组合与 tokenizer 映射策略。

---

> **总结一句话**：  
> **SPECSTEER 提供了一种优雅且高效的解决方案，在不牺牲隐私的前提下，让小模型“借力”大模型完成高质量个性化生成，并实现显著加速，为现实世界中的 edge-cloud 个性化智能代理铺平了道路。**

</details>

---

### 5. [MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild](https://arxiv.org/abs/2603.17187)

**Authors**: Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu Zheng, Cihang Xie, Huaxiu Yao  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.17187v1  

#### Abstract
Large language model (LLM) agents are increasingly used for complex tasks, yet deployed agents often remain static, failing to adapt as user needs evolve. This creates a tension between the need for continuous service and the necessity of updating capabilities to match shifting task distributions. O...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前部署在真实环境中的 **LLM agents** 虽然具备处理复杂多步任务的能力，但普遍存在“静态部署”问题：模型一旦训练完成便不再更新，无法适应用户需求和任务分布的动态变化。这导致：

- 模型能力随时间“过时”（capability staleness）；
- 在跨平台、多通道（如 OpenClaw 连接 20+ 消息渠道）场景下，任务类型频繁漂移，固定模型反复失败于预训练中少见的任务；
- 现有方法存在以下局限：
  - **Memory-based methods**：仅存储原始对话轨迹，冗余且无法提炼可迁移的行为模式；
  - **Skill-based methods**：将经验压缩为技能指令，但技能库与模型权重优化脱节；
  - **RL-based methods**：通过梯度更新优化策略，但忽略“旧技能上下文下的轨迹奖励已失效”的问题，导致**陈旧奖励污染**（stale reward contamination）。

### 提出了什么新方法或新思路

提出 **MetaClaw** —— 一个**持续元学习**（continual meta-learning）框架，使 LLM agent 能够在真实环境中通过使用不断自我进化。其核心是两个互补机制的协同：

1. **Skill-driven fast adaptation**（技能驱动快速适应）
   - 分析失败轨迹，由 LLM evolver 自动生成新的行为技能（behavioral instructions），立即注入系统提示（system prompt）；
   - **零服务中断**（zero service downtime），无需修改模型参数；
   - 实现秒级快速适应。

2. **Opportunistic policy optimization**（机会主义策略优化）
   - 利用成功执行后的轨迹，在用户空闲时段（idle windows）触发基于 **RL + PRM**（Process Reward Model）的 **Cloud LoRA fine-tuning**；
   - 通过 **Opportunistic Meta-Learning Scheduler (OMLS)** 监控三种空闲信号（睡眠时间、键盘无活动、Google Calendar 占用）来决定何时训练；
   - 避免服务中断。

3. **Skill generation versioning mechanism**（技能生成版本控制）
   - 严格区分 **support data**（用于技能演化的失败轨迹）和 **query data**（技能生效后的新轨迹）；
   - 只有 `query data` 才可用于 RL 更新，防止陈旧奖励污染。

### 相比现有方法的优势

| 维度 | MetaClaw 的优势 |
|------|----------------|
| **适应性** | 同时支持快速（技能级）和慢速（权重级）双路径适应，形成正向循环：更好的策略产生更有价值的失败，更丰富的技能带来更高回报的轨迹。 |
| **实用性** | 无需本地 GPU，基于代理架构（proxy-based），可无缝集成到现有个人 agent 和 LLM 平台。 |
| **鲁棒性** | 通过版本控制确保 RL 数据有效性，避免因技能演化导致的策略退化。 |
| **通用性** | 技能以自然语言形式存在，具有跨任务泛化能力，适用于 CLI、研究自动化等多种场景。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

1. **MetaClaw-Bench**（新构建的持续代理基准）
   - 包含 **934 个问题**，跨越 **44 个模拟工作日**；
   - 分为两部分：
     - **Part I**：30 天，346 题，侧重文件操作类任务（file-check）和多选题（multi-choice），强调端到端执行可靠性；
     - **Part II**：14 天，588 题，规则密集型任务流，测试程序性规则内化速度。

2. **AutoResearchClaw**（下游评估）
   - 一个 **23 阶段全自动研究流水线**，从研究想法自动生成会议论文；
   - 用于验证 MetaClaw 方法在开放、长周期、多智能体协作任务中的泛化能力。

### 实验设置和评估指标

#### 主要评估指标

| 指标 | 定义 |
|------|------|
| **Accuracy (%)** | 平均每题得分（overall accuracy） |
| **File-check Completion Rate (%)** | 文件检查任务中所有自动校验器同时通过的比例 |
| **Composite Robustness Score**（AutoResearchClaw） | 加权平均指标：阶段完成率（40%）、重试减少（30%）、精炼效率（30%） |

#### 基线方法对比

| 条件 | 描述 |
|------|------|
| **Baseline** | 原始模型，无任何适应机制 |
| **MetaClaw (Skills)** | 仅启用技能驱动快速适应（技能注入） |
| **MetaClaw (Full)** | 完整流程：技能注入 + 机会主义策略优化（RL + Cloud LoRA） |

#### 测试模型

- **GPT-5.2**（强模型）
- **Kimi-K2.5**（弱模型）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | Condition | Part I Acc. (%) | Part I Compl. (%) | Part II Acc. (%) | Part II Compl. (%) |
|-------|-----------|------------------|--------------------|------------------|---------------------|
| GPT-5.2 | Baseline | 41.1 | 14.7 | 44.9 | 58.4 |
| GPT-5.2 | MetaClaw (Skills) | 44.0 (+7.1%) | 17.1 | 49.1 (+9.4%) | 67.5 |
| Kimi-K2.5 | Baseline | 21.4 | 2.0 | 21.1 | 18.2 |
| Kimi-K2.5 | MetaClaw (Skills) | 28.3 (+32.2%) | 2.0 | 26.9 (+27.5%) | 33.8 |
| Kimi-K2.5 | **MetaClaw (Full)** | **40.6** | **16.5** (**8.25×**) | **39.6** | **51.9** (**+185%**) |

### 与基线方法的对比结果

- **技能注入显著提升准确率**：
  - 对 Kimi-K2.5，技能注入使 Part I 准确率相对提升 **32.2%**；
  - 表明技能库有效弥补了弱模型缺乏的隐式程序性知识。

- **完整流程大幅提升端到端完成率**：
  - Kimi-K2.5 的 file-check 完成率从 **2.0% → 16.5%**（**8.25 倍**）；
  - 在 Part II 达到 **+185%** 提升，接近 GPT-5.2 基线水平（40.6% vs 41.1%）；
  - 说明 **仅靠技能不足以保证零缺陷输出**，需要策略层面的权重优化。

- **强模型增益较小，弱模型受益更多**：
  - GPT-5.2 本身能力强，技能注入收益有限；
  - MetaClaw 特别适合部署**非最先进但成本更低的模型**，通过持续学习逼近顶级模型表现。

### 消融实验结果（关键发现）

- **技能注入 vs 策略优化的作用分离**（Figure 3）：
  - **Skills-only**：显著提升 multi-choice 准确率（推理类任务），但对 file-check 完成率几乎无影响；
  - **Full pipeline**：大幅提高 file-check 完成率，但 multi-choice 准确率略有下降 → 表明策略向执行行为偏移。

- **RL 训练动态分析**：
  - MetaClaw (Full) 在 Part II 中，file-check 完成率在第 8 天出现明显拐点，之后迅速上升；
  - 符合 MAML 内循环结构：前期积累支持数据，后期策略发生质变。

- **技能库分析**：
  - 提炼出三类高频通用技能：
    1. 时间格式标准化（ISO 8601 + TZ）
    2. 修改前备份（`.bak` 协议）
    3. 命名规范遵循（日期前缀 + snake_case）
  - 这些技能具有强跨任务泛化能力。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **双路径适应机制相辅相成**：
   - 快速技能注入提供即时修复；
   - 缓慢策略优化实现根本性能力升级；
   - 二者形成“更好策略 → 更好失败 → 更好技能 → 更好策略”的**良性循环**。

2. ✅ **技能注入即可显著提升鲁棒性**：
   - 在 AutoResearchClaw 上，仅技能注入就使：
     - 阶段重试率 ↓ **24.8%**
     - 精炼循环数 ↓ **40.0%**
     - 复合鲁棒性分数 ↑ **18.3%**
   - 证明该方法可**零成本迁移到复杂长周期任务**。

3. ✅ **版本控制至关重要**：
   - 显式区分 `support` 与 `query` 数据，是防止 RL 训练被陈旧奖励误导的关键设计。

4. ✅ **MetaClaw 是通用持续学习层**：
   - 不依赖特定模型或任务，可通过 prompt 注入方式适配多种 agent 系统。

### 方法的局限性

- **依赖用户配置空闲信号**：OMLS 依赖睡眠时间、日历等手动配置，在某些部署环境可能不适用；
- **技能演化质量依赖 LLM evolver**：若 evolver 生成低质技能，可能引入噪声；
- **未解决灾难性遗忘**：虽然通过版本控制缓解，但在极长期运行中仍需关注。

### 未来工作方向

- 自动化空闲检测（如基于行为模式预测）；
- 引入技能评估与淘汰机制，提升技能库质量；
- 探索多 agent 协同进化场景；
- 将 MetaClaw 架构扩展至视觉、具身智能等多模态 agent。

---

> **GitHub 开源地址**：[https://github.com/aiming-lab/MetaClaw](https://github.com/aiming-lab/MetaClaw)

</details>

---

### 6. [WINFlowNets: Warm-up Integrated Networks Training of Generative Flow Networks for Robotics and Machine Fault Adaptation](https://arxiv.org/abs/2603.17301)

**Authors**: Zahin Sufiyan, Shadan Golestan, Yoshihiro Mitsuka, Shotaro Miwa, Osmar Zaiane  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 7.0  
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
- **传统 CFlowNets 在动态机器人控制任务中的局限性**：
  - CFlowNets（Continuous Flow Networks）依赖于对 **retrieval network $ G_\phi $** 的预训练（pre-training），这在实际机器人环境中不现实。
  - 动态环境（如硬件故障、环境变化）导致状态分布偏移（Out-of-Distribution, OOD），使得预训练模型失效。
  - 预训练需要额外的数据收集和计算资源，限制了其在实时适应场景中的应用。

### 🚀 提出的新方法：**WINFlowNets**
- **核心思想**：提出一种新型 CFlowNets 框架 **WINFlowNets**，实现 **flow network $ F_\theta $** 和 **retrieval network $ G_\phi $** 的联合训练（co-training），无需预训练。
- **关键创新设计**：
  1. **Warm-Up Phase（热身阶段）**：
     - 初始阶段仅训练 $ G_\phi $，使其通过与环境交互积累经验并初步学习状态转移关系。
     - 学习率从低开始逐步提升，以稳定早期高不确定性的学习过程。
  2. **Dual-Training Phase（双训阶段）**：
     - $ F_\theta $ 和 $ G_\phi $ 同时参与环境交互与参数更新。
     - 引入 **Shared Replay Buffer**，统一存储经验，支持两个网络协同学习。
  3. **共享架构与联合优化机制**：
     - 两网络基于同一 replay buffer 进行 inflow/outflow 匹配，增强一致性与稳定性。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **无需预训练** | 消除对独立预训练数据集的依赖，适用于数据稀缺或快速变化的场景 |
| **更强的 OOD 适应能力** | 能持续从新环境中学习，适应执行器损坏（AD）、运动范围受限（ROM）等故障 |
| **更高的最终性能** | 在标准与故障环境下均取得更高平均奖励（higher average reward） |
| **更稳定的训练趋势** | 尽管初期波动较大，但后期收敛稳定，不确定性逐渐降低 |

---

## 2. 核心实验方法和设置

### 🧪 数据集与仿真环境
- 使用 **MuJoCo 物理引擎** 中的 `Reacher-v2` 环境进行实验：
  - 一个具有两个自由度的机械臂（joint0 和 joint1）
  - **State Space**: 11维连续向量（关节角度、速度、目标位置等）
  - **Action Space**: 2维连续动作（施加于关节的扭矩，范围 [-1.0, 1.0]）
  - **Reward Function**:  
    $$
    R(s,a) = -\|p_{\text{fingertip}} - p_{\text{target}}\| - \alpha \|a\|^2
    $$
    （距离惩罚 + 控制代价惩罚）

### ⚙️ 实验设置
- **总训练步数**：100万 timestep（1M）
- **Warm-Up Duration**：前 100k 步用于 $ G_\phi $ 的 warm-up 训练
- **评估方式**：
  - 每隔一定时间评估一次策略性能
  - 报告 **average reward ± standard deviation**（10次 rollout 平均）
  - **Final Performance**：最后20次评估的平均值
  - **Sample Efficiency**：达到渐近性能所需的 timesteps 数量

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Average Reward** | 衡量策略质量的核心指标 |
| **Final Performance** | 反映最终学到策略的优劣 |
| **Sample Efficiency** | 衡量学习效率，越小越好 |
| **Confidence Interval Width** | 衡量训练稳定性与方差 |

### 🆚 基线方法对比
- **CFlowNets**：原始版本，需预训练 $ G_\phi $
- **RL Baselines**：
  - **SAC**（Soft Actor-Critic）：主流 off-policy 算法，样本效率高
  - **PPO**（Proximal Policy Optimization）：on-policy 方法，稳定性好但效率较低
  - **DDPG**（Deep Deterministic Policy Gradient）：易受超参影响，存在过估计问题

---

## 3. 主要实验结果和性能指标

### 📈 正常环境下的性能表现（Normal Environment）

#### 图表结果（Figure 4 & Table 1）
| Model | Final Performance ↓ | Sample Efficiency ↓ |
|-------|---------------------|---------------------|
| SAC | -7.89 ± 0.16 | 0.67 |
| PPO | -9.50 ± 0.37 | 3.39 |
| DDPG | -9.55 ± 0.44 | 5.20 |
| CFlowNets | **-3.70 ± 0.05** | **0.10** |
| **WINFlowNets** | **-2.39 ± 0.17** | 0.72 |

> 注：负值越小表示性能越好（reward 更接近零，即更优）

#### 关键观察：
- **WINFlowNets 达到最高最终性能**（-2.39），显著优于所有基线。
- **CFlowNets 最早收敛**（sample efficiency 最佳），但最终性能低于 WINFlowNets。
- **WINFlowNets 初期增长较慢**（因无预训练），但在约 240k 步后反超 CFlowNets。
- **SAC 表现稳定**，优于 PPO 和 DDPG；而 PPO 出现早期性能下降。

---

### ⚠️ 故障环境下的适应能力（Faulty Environments）

引入两种常见机器人故障：
1. **Actuator Damage (AD)**：执行器输出力矩降至 1/4
2. **Reduced Range of Motion (ROM)**：joint1 角度限制由 [-3.0, 3.0] 缩减为 [-1.5, 1.5]

#### 性能对比（Table 2）

| Fault Type | Model | Final Performance ↓ | Sample Efficiency ↓ |
|----------|--------|----------------------|----------------------|
| **AD** | SAC | -9.69 ± 0.19 | **0.11** |
|        | CFlowNets | -9.01 ± 0.17 | 0.32 |
|        | **WINFlowNets** | **-8.25 ± 0.19** | 0.24 |
| **ROM** | SAC | **-6.16 ± 0.03** | 0.37 |
|         | CFlowNets | -8.50 ± 0.08 | **0.11** |
|         | **WINFlowNets** | **-5.25 ± 0.12** | 0.12 |

#### 关键发现：
- **WINFlowNets 在两种故障下均取得最佳最终性能**，尤其在 ROM 场景中大幅领先。
- **SAC 在 ROM 下表现较好且高效**，但仍不如 WINFlowNets。
- **CFlowNets 在 AD 下表现尚可，但在 ROM 下严重退化**，说明其对结构变化敏感。
- WINFlowNets 初期波动大，但随训练推进迅速稳定，体现强适应性。

---

### 🔬 消融实验（Ablation Study）

比较三种变体（Figure 6）：
1. **WINFlowNets-v1**：有 Dual-Training，无 Warm-Up，共享 buffer
2. **WINFlowNets-v2**：有 Warm-Up 和 Dual-Training，但使用独立 replay buffer
3. **Original WINFlowNets**：完整设计（Warm-Up + Dual-Training + Shared Buffer）

#### 结果分析：
- **v1（无 Warm-Up）**：性能最差，训练不稳定 → 表明 Warm-Up 对初始化 $ G_\phi $ 至关重要
- **v2（独立 buffer）**：性能中等，初期滞后 → 共享 buffer 有助于知识迁移与协同学习
- **Original WINFlowNets**：全面胜出 → 证明三者结合是关键

> ✅ **结论**：Warm-Up + Dual-Training + Shared Replay Buffer 构成有效闭环，缺一不可。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **WINFlowNets 显著提升了 CFlowNets 在动态环境下的实用性**：
   - 成功去除对 retrieval network 预训练的依赖。
   - 实现了在 OOD 场景下的持续自适应学习。
2. **在正常与故障环境中均超越 SOTA RL 方法（SAC/PPO/DDPG）及原版 CFlowNets**：
   - 更高的最终 reward，更强的鲁棒性。
3. **训练机制带来更好的探索能力**：
   - 基于 flow distribution 的生成式探索减少陷入局部最优的风险。
4. **共享 replay buffer 和双阶段训练促进协同进化**：
   - $ F_\theta $ 和 $ G_\phi $ 相互增强，形成正反馈循环。

---

### ⚠️ 局限性
| 问题 | 描述 |
|------|------|
| **较高的样本消耗** | 虽然最终性能更好，但 **sample efficiency 不如 CFlowNets 和 SAC**，需更多时间达到稳定 |
| **初始阶段不稳定** | Warm-Up 后初期 variance 较大，在安全关键场景（如人机交互）可能带来风险 |
| **计算资源需求高** | 双网络联合训练 + 共享 buffer 导致 GPU 内存占用增加，不适合边缘设备部署 |
| **超参数敏感性强** | Warm-Up 时长、学习率调度、buffer 大小等需精细调参，泛化性受限 |

---

### 🔮 未来工作方向
1. **提升 sample efficiency**：
   - 引入 **Prioritized Shared Replay Buffer**，优先回放 TD error 高的经验。
2. **轻量化模型设计**：
   - 探索模型压缩、蒸馏或分布式训练，降低推理成本。
3. **自动化超参数调节**：
   - 结合 NAS 或 BO 方法自动搜索最优配置。
4. **扩展至真实机器人平台**：
   - 当前实验全在模拟环境中完成，下一步应在物理机器人上验证迁移能力。
5. **多任务与元学习结合**：
   - 将 WINFlowNets 与 Meta-RL 结合，进一步加速跨任务适应。

---

## ✅ 总结一句话
> **WINFlowNets 通过 Warm-Up 与 Dual-Training 阶段的联合设计，实现了无需预训练的 CFlowNets 端到端训练，在机器人控制任务中展现出卓越的性能与故障适应能力，尽管牺牲了一定的样本效率，但为动态环境下的智能决策提供了新范式。**

</details>

---

### 7. [Complementary Reinforcement Learning](https://arxiv.org/abs/2603.17621)

**Authors**: Dilxat Muhtar, Jiashun Liu, Wei Gao, Weixun Wang, Shaopan Xiong, Ju Huang, Siran Yang, Wenbo Su, Jiamang Wang, Ling Pan, Bo Zheng  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.17621v1  

#### Abstract
Reinforcement Learning (RL) has emerged as a powerful paradigm for training LLM-based agents, yet remains limited by low sample efficiency, stemming not only from sparse outcome feedback but also from the agent's inability to leverage prior experience across episodes. While augmenting agents with hi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Complementary Reinforcement Learning**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前基于 **Reinforcement Learning (RL)** 的 **LLM-based agent** 学习面临两大挑战：
- **低样本效率 (low sample efficiency)**：依赖稀疏的 outcome-based rewards，无法利用轨迹中的丰富过程信息（如成功策略、失败模式）。
- **经验利用不充分**：已有方法虽尝试从历史轨迹中提取经验（experience），但通常将经验存储为静态资源，或使用固定的经验提取器（experience extractor），导致经验与不断进化的 agent 能力之间出现 **分布错配 (distributional misalignment)**，经验逐渐失效。

### **提出的新方法**
作者受神经科学中的 **Complementary Learning Systems (CLS)** 启发，提出了 **Complementary RL** 框架，其核心是构建一个 **policy actor** 和 **experience extractor** 的**协同进化闭环 (co-evolutionary loop)**：
- **Actor**：与环境交互，通过 outcome-based rewards 优化策略。
- **Extractor**：从轨迹中提炼结构化经验（如策略、规则、模式），并维护动态演化的经验库（experience bank）。
- **协同机制**：Extractor 的训练信号来自其提炼的经验是否真正帮助了 Actor 成功，从而实现两者在训练过程中**相互塑造、同步演化**。

### **相比现有方法的优势**
- **动态对齐**：避免了静态经验库或非自适应提取器带来的“经验过时”问题。
- **高效学习**：通过持续提炼高质量、高相关性的经验，显著提升样本利用率。
- **可扩展架构**：设计了异步训练框架，解耦 actor 与 extractor 的训练流程，避免阻塞延迟，支持大规模部署。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个开放域任务环境中进行评估：
- **MiniHack**：基于文本的网格探索游戏，测试导航与规划能力。
- **WebShop**：模拟电商购物场景，需通过搜索与点击完成购买。
- **ALFWorld**：文本版家庭任务执行环境，结合自然语言理解与物体操作。
- **SWE-Bench**：真实 GitHub 代码修复任务，衡量软件工程推理能力。

### **实验设置与评估指标**
- **模型配置**：
  - Actor：默认使用 `Qwen2.5-7B-Instruct` 或 `Qwen3-4B-Instruct-2507`。
  - Extractor：使用 `Qwen3-4B-Thinking-2507`。
- **训练目标**：最大化跨任务的成功率（success rate）或奖励（reward）。
- **评估方式**：
  - 单任务与多任务联合训练。
  - 固定测试集上报告最终性能。
  - 对比是否在推理时检索经验（eval w/ exp. vs w/o exp.）。

### **基线方法对比**
- **Baseline**：无经验引导的标准 RL 训练。
- **Offline Exp.**：预构建静态经验库，训练中不更新。
- **Static Online Exp.**：在线动态构建经验库，但 extractor 不参与训练。
- **Exp. Only**：仅训练 extractor，冻结 actor 参数。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **单任务表现（图6）**
- 在所有四个任务上，**Complementary RL** 均显著优于 baseline：
  - **MiniHack Room**：成功率提升约 **1.3×**。
  - **SWE-Bench**：绝对成功率提升 **+3.0%**。
- 更高的行动效率（图7）：
  - MiniHack Room 平均减少 **1.5× 动作数**。
  - ALFWorld 减少 **2× 动作数**。
  > 注：SWE-Bench 上动作数增加是因为 agent 更完整地执行任务流程以提高成功率。

#### ✅ **多任务表现（表1 & 图8）**
| Method | Avg. Score |
|--------|----------|
| Baseline | 0.75 |
| Static Online Exp. (w/ exp.) | 0.59 |
| **Complementary RL (w/ exp.)** | **0.82** |
| **Complementary RL (w/o exp.)** | **0.78** |

- **+7% 绝对增益**（平均），即使不在推理时使用经验（w/o exp.），actor 自身能力也更强，说明经验已被内化。
- Static Online Exp. 表现甚至低于 baseline，验证了**静态 extractor 的局限性**。

#### ✅ **消融实验结果**
- **Periodic Merge（定期合并经验条目）**（图5a）  
  → 移除后性能下降，表明去冗余对保持经验质量至关重要。
- **Search_and_ask（主动查询机制）**（图5b）  
  → 支持 agent 在决策关键时刻主动提问，显著提升学习效率。
- **Extractor 容量影响**（图9a）  
  → 使用更大的 extractor（如 30B-A3B）带来额外 **+5% 平均增益**，说明更强的归纳能力有助于提炼通用经验。
- **任务规模扩展性**（图9d）  
  → 在 3-task 和 6-task 场景下分别取得 **+6.6%** 和 **+8.1%** 增益，显示方法具有良好的可扩展性。
- **Rollout 延迟测试**（图9c）  
  → 引入的额外延迟可忽略不计，证明异步架构高效稳定。

---

## **4. 关键结论和发现**

### **主要发现**
1. **协同进化机制有效解决了经验错配问题**：通过让 experience extractor 接收来自 actor 成功与否的反馈信号，实现了经验与策略的同步演化。
2. **经验不仅能用于推理引导，更能被内化为 agent 的内在能力**：即使关闭推理时的经验检索，actor 本身性能仍显著优于 baseline。
3. **高质量经验提炼需要专用模型与独立优化路径**：共享参数或相对奖励策略均导致训练不稳定或性能下降。
4. **异步架构保障了系统吞吐量**：ExperienceManager 实现了高效的并发管理，不影响 rollout 效率。

### **方法的局限性**
- **Extractor 训练高度 off-policy**：由于经验可能在生成很久后才被检索使用，导致训练信号延迟严重。
- **数据冗余问题**：某些高频任务可能导致同一经验被反复训练，引发 extractor 过拟合。
- **Actor-Critic 机制引入延迟**：虽然能提升性能，但因需等待 critic 决策而牺牲训练速度，未作为默认组件。

### **未来工作方向**
- 探索更鲁棒的 **self-distillation** 机制，进一步加速经验内化（当前尝试出现后期崩溃）。
- 设计更智能的 **retrieval diversification** 策略，缓解低多样性任务下的经验重复问题。
- 将 Complementary RL 扩展至更大规模工业级 post-training 流水线，验证其在复杂混合任务流中的长期稳定性。
- 研究如何将通用经验（如 Table 6 中的“停滞检测协议”）形式化为可迁移的认知模块。

---

> 🔗 **开源声明**：作者已公开训练框架与演示代码（文中提及 “We release our training framework and training demo at here”）。

</details>

---

### 8. [InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning](https://arxiv.org/abs/2603.17310)

**Authors**: Chengwei Wei, Jung-jae Kim, Longyin Zhang, Shengkai Chen, Nancy F. Chen  
**Category**: cs.AI  
**Published**: 2026-03-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.17310v1  

#### Abstract
Large Language Models (LLMs) with extended reasoning capabilities often generate verbose and redundant reasoning traces, incurring unnecessary computational cost. While existing reinforcement learning approaches address this by optimizing final response length, they neglect the quality of intermedia...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 **Large Reasoning Models (LRMs)** 在执行多步推理（如 Chain-of-Thought, CoT）时往往生成**冗长且重复的推理轨迹（reasoning traces）**，导致不必要的 token 开销和计算成本。现有的基于强化学习（RL）的方法通常仅通过优化最终答案的正确性和响应长度来鼓励简洁性，但忽略了对**中间推理步骤质量的监督**，从而容易引发 **reward hacking**——模型学会生成看似简短但实际上逻辑跳跃或错误的推理。

作者指出：**verbosity（冗长）不仅是长度问题，更是推理质量低下的症状**。

---

### **提出的新方法与新思路**
论文提出了 **InfoDensity**，一种全新的 **RL 训练奖励框架**，其核心思想是：  
> **高质量的推理轨迹应具备“信息密度高”（informationally dense）的特点——每一步都有效减少对最终答案的不确定性（entropy）。**

为此，InfoDensity 引入了两个基于**信息论信号**（information-theoretic signals）的轨迹级（trajectory-level）奖励：

1. **AUC Reward**  
   - 衡量整个推理过程中**条件熵曲线下的面积（Area Under Curve）**。
   - 鼓励模型快速收敛到低不确定性状态（low uncertainty convergence）。

2. **Monotonicity Reward**  
   - 衡量推理过程中每一步是否持续降低不确定性。
   - 鼓励**单调递减的熵变化趋势**（monotonic progress），防止模型在某步后停滞不前。

此外，结合一个**长度缩放项（length scaling term）**，使得在相同质量下更短的推理获得更高奖励。

最终奖励公式为：
$$
R_{\text{InfoDensity}}(T) = R_{\text{quality}}(T) \cdot R_L(T)
$$
其中 $ R_{\text{quality}} = \alpha \cdot R_{\text{AUC}} + (1-\alpha) \cdot R_{\text{mono}} $，$ R_L $ 是长度相对缩放因子。

---

### **相比现有方法的优势**
| 对比维度 | 现有方法（如 GRPO-LP, PEAR） | InfoDensity |
|--------|----------------------------|-----------|
| **监督粒度** | 仅监督最终答案正确性 + 长度惩罚 | 引入中间步骤的**不确定性动态**作为质量信号 |
| **抗 Reward Hacking** | 易被绕过（如跳步、虚假简洁） | 更难欺骗，因需真实逐步减少 entropy |
| **无需人工标注** | PRM 类方法依赖人工标注或 MCTS | 完全无监督，基于模型自身概率估计 |
| **综合权衡** | 常牺牲准确率换长度（如 GRPO-LP） | 实现更强的 **accuracy-efficiency trade-off** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练集**：
  - `GSM8K`
  - `MATH`
- **测试集（涵盖不同难度）**：
  - **In-Domain**：`GSM8K`, `MATH500`
  - **Out-of-Domain**：`AIME 24`, `OlympiadBench`

这些数据集覆盖从小学数学到奥数级别的挑战性数学推理任务。

---

### **实验设置与评估指标**
- **Base Models**：
  - `Qwen3-0.6B`
  - `DeepSeek-R1-Distill-Qwen-1.5B`
- **Judge Model**（用于计算 entropy 和打分）：
  - 固定使用 `Qwen3-4B-Instruct` 作为外部裁判模型（external judge model）
- **训练协议**：
  - 使用 GRPO 框架进行 RL 训练
  - 最大生成长度：32,768 tokens
  - 推理参数：temperature=0.6, top_p=0.95
- **评估指标**：
  - **Accuracy (Acc)**：最终答案正确率
  - **Token Usage (Tok)**：平均生成 token 数量
  - 综合考量 **accuracy 与 token 效率之间的 trade-off**

---

### **基线方法对比**
1. **GRPO-Acc**：仅以 accuracy 为奖励
2. **GRPO-LP**：加入长度惩罚（length penalty）
3. **PEAR**：当前 SOTA 方法，利用 phase-level entropy 作为奖励信号
4. **Direct-Scoring (DS)**：控制变量法，让 judge model 直接对推理轨迹打分（显式评分 vs. InfoDensity 的隐式 entropy 分析）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2）**

#### 在 `DeepSeek-R1-Distill-Qwen-1.5B` 上：
| Method | Avg Acc (%) | Avg Tok |
|-------|-------------|---------|
| Original | 61.5 | 9217 |
| GRPO-Acc | 63.9 | 7248 |
| GRPO-LP | 60.9 | 7436 |
| PEAR | 61.1 | 6136 |
| **InfoDensity** | **64.0** ✅ | **6443** |

✅ InfoDensity 同时实现**最高准确率**和**显著更低 token 使用量**（↓30%）。

#### 在 `Qwen3-0.6B` 上：
| Method | Avg Acc (%) | Avg Tok |
|-------|-------------|---------|
| Original | 49.5 | 8291 |
| GRPO-Acc | 51.9 | 8819 ↑
| PEAR | 50.2 | 6811
| **InfoDensity** | 49.2 | **6014** ✅

✅ InfoDensity 实现**最低 token 消耗**（↓27%），同时保持接近原始模型的 accuracy，优于 GRPO-LP 和 PEAR 的激进压缩策略。

---

### **与基线方法的对比结果**
- **优于 PEAR**：虽然 PEAR 也能压缩长度，但在 DeepSeek 模型上导致 accuracy 下降；而 InfoDensity 在提升 accuracy 的同时仍大幅缩短长度。
- **优于 GRPO-LP**：后者在 Qwen3-0.6B 上 accuracy 明显下降（↓1.2%），说明其存在严重 reward hacking。
- **优于 Direct-Scoring**：DS 方法因 judge model 缺乏专门训练，在过程评价中不稳定，无法有效引导压缩。

---

### **消融实验结果**
#### （1）AUC 与 Monotonicity 奖励的协同作用（Figure 5）
- 当 $\alpha = 1.0$（仅用 AUC）：模型学会早期锁定答案并重复推导（“let me double-check”类冗余），accuracy 快速崩溃。
- 当 $\alpha = 0.0$（仅用 Monotonicity）：模型可做微小熵减但不真正收敛，最终 accuracy 跌至 ~70%。
- 当 $\alpha = 0.5$：两者平衡，稳定收敛，验证了两个组件的互补性。

#### （2）长度缩放强度 $\lambda$ 的影响（Figure 4）
- $\lambda = 0.01 \sim 0.05$：合理压缩，性能稳定
- $\lambda = 0.5$：过度压缩导致模型崩溃（accuracy < 60%）

表明：**适度的长度压力 + 高质量监督 = 成功压缩的关键**

---

## **4. 关键结论和发现**

### **主要发现**
1. **高质量推理轨迹具有两种信息论特征**：
   - **Low Uncertainty Convergence**：总熵低，快速收敛
   - **Monotonic Progress**：每步几乎都减少不确定性
2. **单步 entropy gain 不足以可靠检测错误**（step-level IG 分布重叠严重），但**整条轨迹的 entropy 动态能有效区分正误路径**。
3. **InfoDensity 可在不依赖人工标注的情况下，自动识别并奖励高质量、高信息密度的推理过程**。
4. **实现了当前最优的 accuracy-efficiency trade-off**：既不过度牺牲精度，又能显著减少 token 使用。

---

### **局限性**
1. **领域限制**：目前实验集中在**数学推理**任务，因其答案可验证、entropy 易建模。是否适用于代码生成、开放域问答等尚待研究。
2. **依赖外部 Judge Model**：使用固定 judge model（如 Qwen3-4B）会引入额外推理开销；若 judge 模型较弱，可能带来噪声。
3. **未探索 self-judge 机制**：当前未尝试让训练模型自己估计 entropy，限制了可扩展性。

---

### **未来工作方向**
1. **将 InfoDensity 扩展至其他推理密集型任务**：如程序合成、科学推理、规划等。
2. **探索 self-supervised entropy estimation**：让训练模型自身成为 entropy 估计器，消除对外部 judge 的依赖。
3. **动态调整 $\alpha$ 和 $\lambda$**：根据问题难度自适应调节奖励权重，进一步优化 trade-off。
4. **结合 PRM 或 MCTS 提供混合监督信号**：融合显式人类偏好与隐式信息论信号。

---

> 🔗 **开源信息**：作者已公开 InfoDensity 的模型、训练与评估 pipeline：  
> [https://github.com/anonymous/InfoDensity](https://github.com/anonymous/InfoDensity)

</details>

---

### 9. [Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies](https://arxiv.org/abs/2603.15903)

**Authors**: Nathaniel Imel, Richard Futrell, Michael Franke, Noga Zaslavsky  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.15903v1  

#### Abstract
Natural languages have been argued to evolve under pressure to efficiently compress meanings into words by optimizing the Information Bottleneck (IB) complexity-accuracy tradeoff. However, the underlying social dynamics that could drive the optimization of a language's vocabulary towards efficiency ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决语言演化中的两个核心开放问题：
- **机制解释缺失**：尽管已有研究指出自然语言倾向于在**信息瓶颈**（Information Bottleneck, IB）框架下实现语义系统的高效压缩（即在表达准确性和编码简洁性之间取得最优权衡），但尚不清楚这种效率是如何通过**基于个体的、文化演化的动态过程**涌现出来的。
- **理论脱节**：进化博弈论**（Evolutionary Game Theory, EGT）** 被广泛用于建模信号游戏（signaling games）中意义系统的从无到有，但这些模型的成功标准（如纳什均衡）是否对应于信息论意义上的**全局最优压缩**（IB最优）仍未知。

### 提出的新方法与新思路
作者提出了一个**统一的理论框架**，将两大领域结合：
- **整合IB与EGT**：首次将Zaslavsky等人提出的**IB效率框架**与Franke和Correia的**含噪声的sim-max信号游戏**及其**不精确模仿动态**（imprecise imitation dynamic）进行形式化整合。
- **提出新模型**：构建了一个基于**种群动力学**（population dynamics）的模型，其中语言通过**社会模仿**而非复杂的认知学习或深层神经网络演化而来。该模型抽象掉了具体的认知架构或学习算法，仅依赖于局部互动和频率依赖的选择。

### 相比现有方法的优势
| 方面 | 本文方法优势 |
|------|--------------|
| **理论基础** | 统一了功能主义（IB效率）与演化博弈论，为语言效率提供了**机制性解释**。 |
| **模型普适性** | 不依赖深度学习架构或超参数调优，更具**可解释性和泛化能力**。 |
| **动态真实性** | 引入感知混淆（state confusion）和不精确模仿，更贴近人类语言传播中的**认知限制和社会学习特性**。 |
| **关注层面** | 聚焦于**群体层面**（population-level）的语言演化，而不仅是成对交互，更符合真实语言作为共享系统的本质。 |

---

## 2. 核心实验方法和设置

### 数据集
本研究未使用真实自然语言数据集，而是采用**合成语义域**（synthetic domain）进行模拟实验：
- **领域设定**：理想化的**数量感**（numerosity）空间，世界状态 $ X = \{0, 1, ..., 99\} $，共100个数值。
- **词汇表**：提供100个可用词语 $ W $，理论上允许建立一一映射（exact numerosity）或近似系统（approximate systems）。

### 实验设置
- **动态模型**：采用 **imprecise conditional imitation dynamic**（源自Franke & Correia, 2018），模拟发送者（Sender）和接收者（Receiver）两个独立种群的策略演化。
- **关键参数**：
  - $ \gamma $：**语用精度标准**（pragmatic standard of precision），控制奖励函数对误差的敏感度（$ \text{sim}(x,x') = \exp(-\gamma(x-x')^2) $）。$ \gamma $ 越大，要求越精确。
  - $ \sigma $：**感知不确定性参数**，控制状态混淆概率（固定为0.5）。
- **初始条件**：随机初始化策略分布，运行至收敛（最多 $10^5$ 步）。

### 评估指标
- **复杂度**（Complexity）：$ I(M_o; W) $，即编码器的信息率，衡量词汇使用的简洁性。
- **准确性**（Accuracy）：$ I(W; X_a) $，即词与真实状态间的互信息，反映传达的保真度。
- **效率损失**（Efficiency Loss）：$ \epsilon = \min_\beta \left( \mathcal{F}_\beta[S] - \mathcal{F}_\beta^* \right) $，衡量当前系统偏离IB理论最优边界的程度。
- **团队适应度**（Population Fitness）：期望相似性 $ \mathbb{E}[\text{sim}(x_a, \hat{x}_a)] $。

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Random Permutation** | 对最终收敛系统的编码矩阵进行行置换，破坏其结构，检验效率是否偶然达成。 |
| **NK99 Dynamic**（Nowak & Krakauer, 1999） | 经典有限种群复制-变异动态，作为主流EGT基线。区别在于：<br>• 单一群体 vs 双群体<br>• 无感知噪声<br>• 包含突变机制 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **接近IB边界**：绝大多数由模仿动态生成的系统都**非常接近IB理论最优边界**（见Figure 2A），表明群体可通过简单模仿实现近乎最优的压缩。
- **高相关性**：
  - $ \gamma $（语用标准）与系统达到的**复杂度和准确性**高度正相关（$ \rho \approx 0.99 $）。
  - $ \gamma $ 与拟合出的IB参数 $ \beta $ 呈**单调递增关系**（Spearman $ \rho = 0.99 $），说明局部语用压力能系统性地调节全局效率权衡。

### 与基线方法的对比结果
| 指标 | 本文模型（FC18） | 随机置换（Permuted） | NK99动态 |
|------|------------------|---------------------|---------|
| **平均效率损失** $ \epsilon $ | 极低（接近0） | 显著更高 | 更高，且远离IB边界 |
| **信息平面位置** | 紧贴IB边界 | 分散于低效区域 | 多数位于低准确区，未有效最小化复杂度 |
| **最大可达准确性** | < 4 bits（理论最大 ~4.61 bits） | — | — |

> ✅ **结论**：本文模型显著优于两种基线，证明其演化路径具有**强导向性**，能自发趋向高效通信。

### 消融实验与关键发现
- **不精确模仿的限制作用**：
  - 即使当 $ \gamma \to \infty $（理论上应趋近完全精确的一一映射），系统**也无法达到全双射**（bijective mapping）。
  - 最大准确性始终低于理论上限，表明**模仿过程中的噪声本身构成了一种根本性的精度天花板**。
- **效率非先验保证**：模型中并无直接优化IB目标的机制，因此系统能达到近优是**涌现现象**，而非设计结果。
- **动态轨迹分析**（Figure 4）：
  - 在演化过程中，系统始终**紧贴IB边界前进**，说明每一步改进都在维持当前准确度下的最小复杂度。
  - 效率损失虽总体下降，但在早期（前100步）会出现短暂波动，尤其在高 $ \gamma $ 下更明显，反映探索与稳定之间的张力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **局部模仿可驱动全局效率**：即使个体仅基于局部成功进行**不精确的社会模仿**，整个群体的语言也能自发演化出**信息论意义上近最优的压缩系统**。
2. ✅ **EGT成功蕴含IB效率**：在适当建模下（引入感知噪声与模仿机制），**进化博弈论中的“成功”策略确实可以通向IB框架下的“高效”系统**，弥合了两大理论范式之间的鸿沟。
3. ✅ **语用标准是关键调控参数**：游戏中的**语用精度要求**（$ \gamma $）是决定最终系统位于IB曲线上哪一点的核心机制性变量，为跨语言语义差异提供了潜在解释。
4. ✅ **认知限制塑造语言结构**：**感知混淆**和**模仿噪声**不仅不可避免，反而可能作为一种**正则化机制**，防止过度复杂化，并促成模糊范畴的形成。

### 方法的局限性
- **合成域限制**：实验基于理想化的一维数量空间，尚未扩展到更复杂的语义结构（如分类、层级、递归）。
- **静态环境假设**：未考虑语义需求分布 $ p(u) $ 的动态变化（如幂律分布、语境依赖）。
- **简化奖励机制**：使用指数相似函数建模效用，虽有心理学依据，但仍为理想化设定。
- **忽略语言生成结构**：仅建模词汇映射，未涉及句法组合性。

### 未来工作方向
- 将模型应用于**真实语言类型学数据**（如颜色词、亲属称谓、数词系统），验证其预测能力。
- 引入**非均匀先验**（如幂律分布）以模拟高频词压缩现象。
- 探索**多维语义空间**中的范畴化，研究几何结构如何影响IB解的形态。
- 结合**迭代学习**（iterated learning）范式，研究代际传递中的效率累积。
- 进一步探究**个体认知偏差**（如IB偏好）与群体动态之间的相互作用。

--- 

> 📌 **一句话总结**：  
> 该论文证明，通过简单的**含噪模仿动态**，语言群体能够在无需全局优化的情况下，自发演化出**接近信息瓶颈理论极限的高效语义系统**，为“为何人类语言如此高效”这一经典问题提供了有力的**基于主体的机制解释**。

</details>

---

### 10. [The 1/W Law: An Analytical Study of Context-Length Routing Topology and GPU Generation Gains for LLM Inference Energy Efficiency](https://arxiv.org/abs/2603.17280)

**Authors**: Huamin Chen, Xunzhuo Liu, Yuhan Liu, Junchen Jiang, Bowei He, Xue Liu  
**Category**: cs.DC  
**Published**: 2026-03-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.17280v1  

#### Abstract
How many tokens can a GPU inference cluster deliver per watt? Across deployments of identical hardware, the answer varies by 40x -- not because of software inefficiency, but because of the serving context window. We derive the 1/W law: tokens per watt halves every time the context window doubles. A ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The 1/W Law: An Analytical Study of Context-Length Routing Topology and GPU Generation Gains for LLM Inference Energy Efficiency

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**大语言模型（LLM）推理过程中的能源效率问题**，特别是：
- 为什么在相同硬件上部署 LLM 时，不同上下文长度（context length）会导致高达 **40× 的 tokens per watt（tok/W）差异**？
- 如何通过系统级设计（而非仅依赖更先进硬件）显著提升能效？

传统观点认为升级到新一代 GPU（如从 H100 到 B200）是提高能效的关键。本文指出，**实际中更强大的杠杆是“routing topology”——即请求如何被路由到不同的服务池**。

---

### 🚀 提出的新方法与新思路

#### （1）提出 **1/W Law**
- **核心公式**：`tok/W ∝ 1 / W`，其中 `W` 是上下文窗口大小。
- 含义：每当上下文窗口翻倍，tokens per watt 几乎减半。
- 原因：更大的 context 导致 KV-cache 可容纳的并发序列数下降，而 GPU 功耗基本不变 → 能效降低。

> 🔍 这是一个**解析推导得出的规律**，基于 KV-cache 内存限制、roofline 模型 和 logistic GPU power model。

#### （2）强调 **Routing Topology 是比硬件升级更强的能量杠杆**
- 提出并分析了 **FleetOpt**（一种两池 context-length routing 架构）：
  - 将短请求路由至小 context 池（高并发、高效）
  - 长请求进入大 context 池（低并发、低效但必要）
- 发现：**拓扑优化带来的增益独立于硬件代际升级，且二者可乘性叠加**。

#### （3）揭示三大正交能效杠杆
| 杠杆 | 效果来源 |
|------|--------|
| **GPU Generation** | 更高的内存带宽、更大 VRAM、更低延迟 |
| **Routing Topology** | 控制每个 GPU 实际服务的 context 长度分布 |
| **Model Architecture (MoE)** | 激活参数少 → 权重流式传输时间短 |

这三者作用维度不同，**收益相乘而非相加**。

---

### ⚖️ 相比现有方法的优势

| 对比项 | 本文方法 | 现有主流做法 |
|-------|---------|-------------|
| 能效优化重点 | Routing topology + Hardware + Architecture | 单纯依赖硬件升级或软件优化 |
| 成本可行性 | 不需购买新硬件即可重构路由实现 2.5× 提升 | 高成本采购新卡 |
| 分析方式 | 完全解析建模（analytical study），无需实测新硬件 | 多依赖经验测量或仿真 |
| 杠杆识别 | 明确分离 topology 与 generation 的独立增益 | 忽视 topology 的结构性影响 |

> 💡 例如：运营商若只买 B200 而不改 topology，只能获得 1.7× 提升；而采用 FleetOpt on H100 就可达 2.5×；两者结合达 **4.25×**，远超任一单独手段。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **Azure Azi LLM Inference Trace** [Patel et al., 2024]
  - 特征：89% 请求 ≤ 4K tokens → 属于 **short-dominant workload**
- **LMSYS-Chat-1M** [Zheng et al., 2023]
  - 特征：约 50–80% 请求较短，其余较长 → 属于 mixed workload

这些真实 trace 用于模拟请求到达率、context 分布和利用率。

---

### ⚙️ 实验设置

#### 模型配置
- 主要模型：`Llama-3.1-70B`（TP=8, fp16）
- MoE 对照模型：`Qwen3-235B-A22B`（激活 22B 参数）、`DeepSeek-V3`

#### 硬件平台
| GPU | TDP | VRAM | Memory Bandwidth |
|-----|-----|------|------------------|
| H100-SXM5 | 700W | 80GB | 3.35TB/s |
| B200-SXM | 1000W | 156GB | 8.0TB/s |
| H200-SXM | 700W | ~144GB | 4.8TB/s |
| GB200-NVL | 1200W | 200GiB | 8.0TB/s |

> 注：B200/H200 数据为 **first-principles projection**（基于 H100 标定的比例外推），不确定性 ±20%

#### 并发控制与 KV-cache 假设
- 使用 **tensor-parallel sharding**（TP=8）管理 KV heads
- 每 token KV 存储开销约为 55KB（H100 实测校准）

#### 评估框架
- 工具：`inference-fleet-sim` [Chen et al., 2026b]
- 输入：request arrival rate (`λ = 1,000 req/s`)，SLO（P99 TTFT ≤ 500ms）
- 输出：所需 GPU 数量、总功耗（kW）、fleet-level tok/W

---

### 🎯 评估指标

| 指标 | 定义 | 用途 |
|------|-----|------|
| **tok/W** | Tokens delivered per joule of energy consumed | 核心能效指标 |
| **Fleet-level tok/W** | $\frac{\sum_i \lambda_i \cdot L_{\text{out},i}}{\sum_j n_j \cdot P(n_{\text{act},j})}$ | 衡量整体集群效率 |
| **vs H100 Homo baseline** | 相对于 H100 均质 64K 集群的相对提升 | 归一化比较基准 |
| **$/hr & tok/$M** | 每小时成本与每百万美元产出 token 数 | 经济性分析 |

---

### 🔁 基线方法对比

| 基线 | 描述 |
|------|------|
| **Homogeneous 64K fleet** | 所有请求都走 64K context 池（当前常见部署模式） |
| **Two-pool context routing** | 按 prompt length 分流（如 <4K → short pool） |
| **FleetOpt** | 最优分割边界 $y^*$ 下的 context-length routing（理论最优） |
| **Semantic routing** | 按任务复杂度路由到不同规模模型（如 8B vs 70B） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ 单 GPU 层面（Table 1 & 2）

| Context Length | H100 tok/W | B200 tok/W | 衰减趋势 |
|---------------|------------|-----------|----------|
| 4K            | 17.6       | 30.8      | —        |
| 8K            | 8.97       | 15.5      | ↓ ~2×    |
| 16K           | 4.69       | 7.87      | ↓ ~2×    |
| 64K           | 1.50       | 2.24      | ↓ ~2×    |

➡️ **验证 1/W Law：context 翻倍 ⇒ tok/W 几乎减半**

| Model (8K context) | H100 tok/W | B200 tok/W |
|--------------------|------------|-----------|
| Llama-3.1-70B      | 7.41       | 20.93     |
| Qwen3-235B-A22B (MoE) | **37.82** | **177.73** |

➡️ MoE 模型因 active-parameter streaming 时间短，在能效上具有天然优势（理论上可达 5× 以上）

---

#### ✅ 集群层面（Table 3）——Azure Workload 示例

| Topology | GPU | GPUs Needed | Power (kW) | tok/W | vs H100 Homo |
|---------|-----|--------------|------------|--------|----------------|
| Homogeneous 64K | H100 | 141 | 58.3 | 5.58 | — |
| FleetOpt (4K/64K) | H100 | 40 | 23.1 | **14.08** | **+152%** |
| Homogeneous 64K | B200 | 47 | 33.4 | 9.74 | +75% |
| FleetOpt (4K/64K) | B200 | 17 | 13.7 | **23.71** | **+325%** |

> ✅ **组合增益 = topology × generation = 2.52 × 1.75 ≈ 4.4×**

---

#### ✅ 拓扑与硬件增益的独立性验证

| Gain Type | H100 上 topology 提升 | B200 上 topology 提升 |
|----------|------------------------|------------------------|
| △topo | 14.08 / 5.58 = **2.52×** | 23.71 / 9.74 = **2.44×** |

| Gain Type | Homogeneous 下 gen 提升 | FleetOpt 下 gen 提升 |
|----------|----------------------------|------------------------|
| △gen | 9.74 / 5.58 = **1.75×** | 23.71 / 14.08 = **1.68×** |

➡️ 证明两个杠杆**几乎完全正交**，可乘性叠加。

---

#### ✅ MoE 模型潜力（Table 2）
- `Qwen3-235B-A22B` 在 H100 上达到 **37.8 tok/W @ 8K context**
- 是 `Llama-3.1-70B` 的 **5.1×**
- 但这是**上界估计**（未计入 MoE dispatch overhead）
  - 若 dispatch 增加 10ms，则优势降至约 1.5×

---

## 4. 关键结论和发现

### 🧩 主要发现

1. **1/W Law 成立**：
   - tok/W 与 context length 成反比，源于 KV-cache 并发容量下降而功耗不变。
   - 是物理瓶颈，非软件缺陷。

2. **Routing Topology > Hardware Upgrade**：
   - 两池 context-length routing（FleetOpt）带来 **~2.5× tok/W 提升**
   - 硬件从 H100 → B200 仅带来 **~1.7× 提升**
   - **拓扑优化性价比更高，且无需额外支出**

3. **三大杠杆正交可乘**：
   - Topology × Generation × Architecture → 总体可达 **4.25× 能效提升**
   - 任一单一手段都无法接近此水平

4. **MoE 架构具备天然能效优势**：
   - 权重流式传输时间取决于激活参数数量，而非总参数量
   - 但在现实中受 dispatch 开销制约，需实测验证

5. **工作负载类型决定最佳策略**（Table 6）
| Archetype | 推荐 topology | 推荐 GPU |
|----------|----------------|----------|
| Short-dominant (>80% ≤8K) | FleetOpt two-pool | B200 |
| Mixed (50–80%) | Pool routing | H200 / B200 |
| Long-dominant (<50%) | Homogeneous long-only | B200 / GB200 |
| MoE-capable | Short pool + MoE | B200 / GB200 |

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **B200/H200 数据为预测值** | 缺乏实测 P(b) 曲线，tok/W 有 ±20% 不确定性 |
| **假设连续批处理理想状态** | 忽略 head-of-line blocking、KV eviction 等现实干扰 |
| **steady-state 流量假设** | 实际存在突发流量，需预留 capacity，降低平均 util 和 tok/W |
| **仅考虑 decode 阶段能耗** | Prefill 能耗未计入（对长 prompt 场景可能占主导） |
| **仅分析 two-pool topology** | 更多分池或语义路由未深入探索 |
| **MoE dispatch overhead 缺失** | 当前 MoE 结果为上界，实际差距可能大幅缩小 |

---

### 🔮 未来工作方向

1. **Empirical Calibration on B200/H200**
   - 使用 ML.ENERGY 方法测量真实 P(b) 曲线，更新模型准确性。

2. **Prefill-Decode Disaggregation**
   - 结合 Splitwise 架构，将 prefill 与 decode 分离，进一步优化能效。

3. **Multi-pool Topology Optimization**
   - 扩展 FleetOpt 至 K≥3 个 context 池，构建混合整数规划求解最优划分。

4. **Carbon-aware Joint Optimization**
   - 将 tok/W 扩展为 `$ / token` 或 `gCO₂ / token`，结合电价与电网碳强度进行调度。

5. **Adaptive Topology Control**
   - 设计在线控制器动态调整 split boundary，适应 workload drift。

6. **Speculative Decoding Interaction**
   - 分析 speculative decoding 如何改变 batch size 分布，进而影响 tok/W。

---

## ✅ 总结一句话

> **在 LLM 推理能效优化中，“让短请求走快车道”（context-length routing）比“换更快的车”（升级 GPU）更有效，而两者结合可实现 4.25× 的能效飞跃。**

</details>

---

### 11. [Variational Rectification Inference for Learning with Noisy Labels](https://arxiv.org/abs/2603.17255)

**Authors**: Haoliang Sun, Qi Wei, Lei Feng, Yupeng Hu, Fan Liu, Hehe Fan, Yilong Yin  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.17255v1  

#### Abstract
Label noise has been broadly observed in real-world datasets. To mitigate the negative impact of overfitting to label noise for deep models, effective strategies (\textit{e.g.}, re-weighting, or loss rectification) have been broadly applied in prevailing approaches, which have been generally learned...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Variational Rectification Inference for Learning with Noisy Labels

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**Learning with Noisy Labels (LNL)** 中存在的两个关键挑战：
- **模型坍缩 (model collapse)**：在基于 Monte Carlo (MC) 近似的概率元学习方法中，后验分布可能退化为狄拉克函数（Dirac delta），导致模型退化为确定性网络，损害泛化能力。
- **忽略平滑性假设 (smoothness assumption)**：现有方法过度依赖噪声标签进行样本权重或校正向量估计，而未能充分利用特征空间中的判别信息。

### 提出的新方法与思路
作者提出 **Variational Rectification Inference (VRI)**，一种将损失校正过程建模为**变分推断 (variational inference)** 问题的新框架。其核心思想包括：

- 将**校正向量 (rectifying vector)** 视为隐变量，构建一个层次贝叶斯 (hierarchical Bayes) 模型。
- 引入一个**先验网络 (prior network)** 和一个**摊销元网络 (amortization meta-network)** 来分别建模先验 $p(v|x)$ 和近似后验 $q_\phi(v|x,y)$。
- 推导出基于证据下界 (ELBO) 的目标函数，其中包含一个由 KL 散度构成的**正则化项**，用于约束后验与先验的一致性。

### 相比现有方法的优势
- ✅ **避免模型坍缩**：通过引入 KL 正则项，防止后验方差趋近于零，从而维持模型的随机性和鲁棒性。
- ✅ **增强泛化能力**：鼓励元网络更多地从特征 $x$ 而非潜在噪声标签 $y$ 中提取信息，符合平滑性假设。
- ✅ **高效推理**：采用 reparameterization trick 实现可微采样，支持端到端训练；仅需少量采样 (如 k=1 或 2) 即可达到高性能。
- ✅ **理论保证**：提供了算法收敛性的理论分析，证明其可在 $O(1/\epsilon)$ 步内找到 $\epsilon$-一阶平稳点。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖五种基准数据集，覆盖合成噪声与真实世界噪声：
- **CIFAR-10 & CIFAR-100**：用于测试合成噪声下的性能。
- **Clothing1M**：大规模真实世界图像分类数据集，约 38.46% 标签噪声。
- **Food-101N**：来自网络搜索的真实食品图像，约 20% 标签噪声。
- **ANIMAL-10N**：动物图像数据集，约 8% 标签噪声。
- **CIFAR-80N**：人工构造的**开集噪声 (open-set noise)** 数据集，包含分布外 (OOD) 样本。

### 实验设置与评估指标
- **元数据集 (meta-data)**：
  - CIFAR-10/100：随机选取 1000 个干净样本作为元数据。
  - Clothing1M / Food-101N：使用官方验证集作为元数据。
- **噪声类型**：
  - Flip noise（类别间翻转）
  - Uniform noise（均匀随机翻转）
  - Instance-dependent (ID) noise（实例相关噪声）
  - Real-world noise
  - Open-set noise
- **评估指标**：**Top-1 测试准确率 (%)**

### 基线方法对比
对比了多种主流方法，包括：
- **基础方法**：Baseline (直接训练)
- **重加权 (re-weighting)**：MW-Net, PMW-Net
- **标签/损失校正 (correction)**：MLC, MSLC, FSR, EMLC
- **半监督风格方法**：DivideMix, ELR, Co-teaching
- **其他元学习方法**：FaMUS, FasTEN
- **开集噪声专用方法**：Jo-SRC, PNP, USDNL
- **MC 近似方法**：WarPI（作为同源对比）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（部分摘录）

| 数据集 | 噪声类型 | 噪声比例 | VRI 准确率 | 最佳基线 | 提升幅度 |
|--------|----------|-----------|------------|----------|----------|
| CIFAR-10 | Flip | 20% | **93.79%** | MSLC (94.11%) | -0.32% |
| CIFAR-10 | Flip | 40% | **93.27%** | MSLC (92.48%) | **+0.79%** |
| CIFAR-100 | Flip | 40% | **67.81%** | ELR (73.73%) | — |
| CIFAR-10 | Uniform | 40% | **91.29%** | FasTEN (91.94%) | -0.65% |
| CIFAR-10 | ID | 40% | **90.60%** | PADDLES (89.87%) | **+0.73%** |
| Clothing1M | Real-world | ~38.5% | **75.19%** | WarPI (74.98%) | **+0.21%** |
| Food-101N | Real-world | ~20% | **86.24%** | WarPI (85.91%) | **+0.33%** |
| CIFAR-80N | Flip 40% | — | **64.71%** | PNP-soft (61.23%) | **+3.48%** |

> 注：在 ResNet-34 架构下，VRI 在多数 Flip 和 ID 噪声场景中表现最优。

### 与基线方法的对比结果
- 在 **Flip 和 Instance-dependent 噪声** 下，VRI 显著优于大多数元学习方法（如 MW-Net, MSLC）及非元学习方法（如 ELR, DivideMix）。
- 在 **Uniform 噪声** 下略逊于最强对手（如 FasTEN），但仍保持竞争力。
- 在 **真实世界噪声** 上，VRI 在 Clothing1M 和 Food-101N 上均取得**最高准确率**，超越 WarPI 等 MC 方法。
- 在 **Open-set noise (CIFAR-80N)** 上，VRI 表现尤为突出，在 Flip 40% 下比当前 SOTA 方法高出 **3.48%**，显示其对复杂噪声的强大适应能力。

### 消融实验结果
#### (1) 采样数量 $k$ 影响
| 方法 | $k$ | 时间 (min/epoch) | 准确率 (%) |
|------|-----|------------------|-------------|
| MC | 1 | 2.17 | 88.23 |
| MC | 3 | 4.32 | 89.45 |
| MC | 5 | 7.04 | 89.87 |
| **VRI** | **1** | **2.20** | **90.20** |

> ✅ **VRI 仅用一次采样即超越多次采样的 MC 方法**，且效率更高。

#### (2) 是否使用贝叶斯形式（vs 非贝叶斯 VRI）
| 噪声设置 | VRI (贝叶斯) | 非贝叶斯 VRI | 提升 |
|---------|--------------|----------------|-------|
| CIFAR-10 Unif. 40% | **91.29%** | 89.27% | +2.02% |
| CIFAR-10 Inst. 40% | **90.60%** | 87.01% | +3.59% |

> ✅ 贝叶斯建模显著提升性能，验证了变分框架的有效性。

#### (3) 超参数敏感性
- KL 正则系数 $\lambda = 0.001$ 时效果最佳。
- 元数据规模增加有助于性能提升，即使仅有 100 个元样本也能取得较好结果（如 91.07% @ CIFAR-10 Flip 40%）。

---

## 4. 关键结论和发现

### 主要发现
1. **变分建模范式有效缓解模型坍缩**：通过引入先验网络和 KL 正则项，VRI 成功避免了 MC 方法常见的后验坍缩问题，提升了模型的稳定性和泛化能力。
2. **特征驱动的校正更鲁棒**：VRI 鼓励元网络关注特征而非噪声标签，符合平滑性假设，在高噪声比（如 60%-70%）下仍能保持良好性能。
3. **高效且低采样需求**：得益于变分学习机制，VRI 在仅使用 $k=1$ 或 $2$ 次采样的情况下即可达到甚至超过传统 MC 方法多次采样的性能，训练效率高。
4. **对开集噪声具有强适应性**：在包含 OOD 样本的 CIFAR-80N 上大幅领先，表明该方法能有效处理更复杂的现实噪声场景。

### 方法的局限性
- **依赖元数据**：虽然可通过伪标签缓解，但在完全无干净元数据的情况下性能会下降（见 Table 12）。
- **额外网络开销**：相比确定性方法，需维护额外的先验网络和元网络，尽管计算成本可控。
- **超参数调优**：KL 系数 $\lambda$ 对性能有一定影响，需通过轻量级验证选择。

### 未来工作方向
- 探索无需元数据的自监督式变分校正机制。
- 将 VRI 扩展至其他弱监督任务（如 partial labels, complementary labels）。
- 结合 contrastive learning 或 prompt learning 提升表示质量。
- 应用于更大规模视觉模型（如 ViT）和多模态场景。

> 🔗 **代码已开源**：https://github.com/haolsun/VRI

</details>

---

### 12. [ASDA: Automated Skill Distillation and Adaptation for Financial Reasoning](https://arxiv.org/abs/2603.16112)

**Authors**: Tik Yu Yim, Wenting Tan, Sum Yee Chan, Tak-Wah Lam, Siu Ming Yiu  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.16112v1  

#### Abstract
Adapting large language models (LLMs) to specialized financial reasoning typically requires expensive fine-tuning that produces model-locked expertise. Training-free alternatives have emerged, yet our experiments show that leading methods (GEPA and ACE) achieve only marginal gains on the FAMMA finan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ASDA: Automated Skill Distillation and Adaptation for Financial Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在通用领域表现优异，但在**金融推理**（financial reasoning）这类需要多步定量计算与深度领域判断的任务上仍存在显著瓶颈。传统解决方案如**domain-specific fine-tuning**成本高昂、产生“模型锁定”（model-locked）的知识，且不适用于通过API调用的黑盒商业LLM。

此外，现有的训练免费（training-free）适配方法（如GEPA、ACE）仅优化扁平文本提示（flat text prompts），缺乏模块化和可执行性，难以支持复杂、多步骤的金融子领域推理。

### 🚀 提出的新方法：ASDA框架
作者提出 **Automated Skill Distillation and Adaptation (ASDA)** ——一种无需修改模型权重的自动化技能蒸馏与适应框架，其核心思想是：
> 将错误分析转化为**可执行的skill artifacts**（技能文件），并在推理时动态注入。

#### 核心机制：
- **Teacher-Student 架构**：一个更强的teacher model分析student model在金融任务上的失败案例。
- **错误聚类与技能生成**：按`subfield × error_type`对错误进行聚类，并自动生成结构化的`.md`技能文件，包含：
  - 领域特定的推理流程（reasoning procedures）
  - 代码模板（code templates）
  - 工作示例（worked examples）
- **动态技能注入**：基于问题内容，由LLM-based selector选择相关技能文件并注入prompt中指导推理。

### 🔍 相比现有方法的优势
| 方面 | 传统Fine-tuning | GEPA / ACE等Prompt优化 | ASDA |
|------|------------------|----------------------------|-------|
| 是否需训练 | 是（高成本） | 否 | 否 ✅ |
| 是否依赖权重访问 | 是 ❌ | 否 ✅ | 否 ✅ |
| 输出形式 | 模型参数变化 | 扁平文本提示 | **结构化、可执行技能文件** ✅ |
| 可审计性 | 差 | 差 | **人类可读、版本控制、符合Agent Skills标准** ✅ |
| 跨模型迁移能力 | 弱 | 中等 | 明确指出应为每个模型单独生成（避免负迁移）✅ |

> ✅ **创新本质**：不是更好的prompt，而是引入了一个新的**表示层**（representational layer）——介于原始模型与部署场景之间的**可维护、可验证的技能库**。

---

## 2. 核心实验方法和设置

### 📚 数据集：FAMMA-Basic-Txt
- 来源：大学教材与专业金融考试题
- 规模：共1,378个英文问题（过滤后）
- 子领域：涵盖8个金融子领域（corporate finance, derivatives, portfolio management等）
- 类型划分：
  - **Arithmetic**（算术推理）：需程序化计算（PoT）
  - **Non-Arithmetic**（非算术推理）：概念判断、逻辑推理
- 分割方式：分层60/40划分训练集与测试集（按难度与题型）

### ⚙️ 实验设置
- **学生模型（Student Models）**：
  - `Haiku 3.5`（较弱基础模型）
  - `Haiku 4.5`（较强同系列模型）
- **教师模型（Teacher Model）**：
  - `Sonnet 4.5`
- **评估指标**：
  - 多选题：rule-based exact match
  - 开放式问题：LLM judge（Qwen-Max）
- **推理配置**：
  - 温度设为0以保证可复现性
  - 使用Program-of-Thought（PoT）处理算术问题

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Baseline** | 无增强 | 标准任务提示 |
| **GEPA** [2] | training-free | 反思式提示进化（reflective prompt evolution） |
| **ACE** [1] | training-free | 测试时知识积累（test-time knowledge accumulation） |
| **ASDA (WU)** | ours | Warm-up阶段初始技能库 |
| **ASDA (E2)** | ours | 经过两轮迭代精炼后的最终版本 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Haiku 3.5 结果）

| 方法 | Arithmetic Acc (%) | Δ (pp) | Non-Arithmetic Acc (%) | Δ (pp) |
|------|------------------------|--------|----------------------------|--------|
| Baseline | 41.00 | — | 49.21 | — |
| GEPA | 42.33 | +1.33 | 50.79 | +1.58 |
| ACE | 44.30 | +3.30 | 49.60 | +0.39 |
| **ASDA (WU)** | **49.67** | **+8.67** | **51.98** | **+2.78** |
| **ASDA (E2)** | **58.33** | **+17.33** | **55.16** | **+5.95** |

> ✅ **ASDA实现最大提升：**
> - 算术推理：**+17.33个百分点**
> - 非算术推理：**+5.95个百分点**
>
> 远超所有training-free基线方法。

### 🔁 迭代精炼效果
- **Warm-up阶段**已带来显著增益（+8.67pp arithmetic）
- **Iterative Refinement**进一步提升至峰值（epoch 2）
- 第3轮出现性能回落（overfitting），表明**2轮为最优操作点**

### 🧪 消融实验：Self-Teaching Ablation（表3）
- 当使用**同一模型作为teacher和student**（即无更强teacher）时：
  - Haiku 3.5 arithmetic：从+8.67pp降至**+6.33pp**（达到全收益的73%）
  - 表明：**大部分增益来自训练数据中的错误模式识别本身，而非teacher的知识优势**

> 💡 结论：即使没有更强大的teacher，组织也可仅凭自身标注数据运行ASDA，具备现实部署可行性。

### 🔁 技能跨模型迁移实验（Cross-Transfer）
| 技能来源 | Arithmetic Acc (%) | Δ (pp) | MC Δ | Open Δ |
|---------|------------------------|--------|------|--------|
| 自身失败生成（Own skills） | 69.67 | +5.00 | +7.91 | +2.48 |
| Haiku 3.5迁移（Cross-transfer） | 62.33 | **-2.33** | +2.16 | **-6.21** |

> ❌ **反向迁移有害**：将弱模型的技能用于强模型会导致整体退化，尤其影响开放式问答。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **失败驱动的蒸馏能有效外化隐性知识**  
   即使没有外部监督信号，模型也能通过分析自身错误，提炼出可重用的结构化技能。

2. **技能是模型特异性的“故障修复包”**  
   技能并非通用领域知识，而是针对特定模型失败分布的补丁。跨模型复用可能导致性能下降。

3. **结构优于文本：executable skills > flat prompts**  
   相比于GEPA/ACE等扁平提示优化，模块化、带代码模板的skill files更能支撑复杂推理。

4. **实用性强：低成本、可审计、易更新**  
   - 成本约 **$13 + 6小时墙钟时间**
   - 技能库可版本控制、合规审查
   - 模型升级后可快速重新生成新技能库

### ⚠️ 局限性
- 当前实验局限于 **FAMMA数据集** 和 **Claude模型家族**
- 错误分类依赖预定义的10类taxonomy，泛化性待验证
- OCR提取文本可能引入噪声，导致技能学习到数据修正启发式规则（data-correction heuristics）
- 在非算术任务中增益较小，因错误分散、信号弱

### 🔮 未来工作方向
1. **跨领域扩展**：应用于法律、税务等同样需要可审计推理的领域
2. **技能压缩（Skill Compression）**：合并窄范围技能文件，减少冗余与回归风险
3. **自教机制深化**：探索完全无需更强teacher的自我改进路径
4. **诊断质量预测**：研究错误聚类清晰度是否可作为adaptation成功的关键指标

---

## 总结一句话
> **ASDA通过从错误中自动蒸馏出结构化、可执行的agent skills，在无需微调的前提下显著提升了LLM在金融推理任务上的表现，同时提供了可审计、可维护、低成本的领域适配新范式。**

</details>

---

### 13. [Flow Matching Policy with Entropy Regularization](https://arxiv.org/abs/2603.17685)

**Authors**: Ting Gao, Stavros Orfanoudakis, Nan Lin, Elvin Isufi, Winnie Daamen, Serge Hoogendoorn  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.17685v1  

#### Abstract
Diffusion-based policies have gained significant popularity in Reinforcement Learning (RL) due to their ability to represent complex, non-Gaussian distributions. Stochastic Differential Equation (SDE)-based diffusion policies often rely on indirect entropy control due to the intractability of the ex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Flow Matching Policy with Entropy Regularization 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Stochastic Differential Equation (SDE)** 的扩散策略（diffusion policies）在在线强化学习（online RL）中面临两大挑战：
- **熵不可计算**：由于反向采样过程是随机的，导致策略熵（policy entropy）无法精确计算，阻碍了最大熵优化（maximum-entropy optimization），从而影响探索能力。
- **训练效率低**：依赖多步迭代去噪链进行梯度回传，计算开销大且梯度方差高。

### 🚀 提出的新方法：FMER
作者提出 **Flow Matching Policy with Entropy Regularization (FMER)**，一种基于 **Ordinary Differential Equation (ODE)** 的在线 RL 框架，其核心思想包括：
- 使用 **Conditional Flow Matching (CFM)** 构建确定性生成路径，通过回归一个目标速度场来参数化策略。
- 引入 **优势加权 Flow Matching 损失（Weighted-CFM）**，利用 critic 提供的优势函数对候选动作赋权，引导策略向高价值区域更新。
- 推导出 **可解析计算的策略熵表达式**，实现对策略熵的显式正则化，增强探索。

### 🔍 相比现有方法的优势
| 维度 | 传统扩散策略（如 DIPO, QVPO） | FMER |
|------|-------------------------------|-------|
| 采样机制 | 随机 SDE 反向过程，需大量步骤 | 确定性 ODE 路径，仅需少量积分步 |
| 熵控制 | 不可计算，常使用启发式近似（如高斯混合） | 显式闭式解，支持原则性最大熵优化 |
| 训练效率 | 梯度需穿越长采样链，计算昂贵 | 无需反向传播采样路径，训练更快更稳定 |
| 更新方式 | 多采用 Top-1 硬选择，易受 critic 误差影响 | 软加权机制，保留分布信息，提升鲁棒性 |

---

## 2. 核心实验方法和设置

### 📚 数据集与环境
实验在以下三大类环境中进行：
- **FrankaKitchen**：稀疏奖励、多任务、真实机器人厨房操作环境，具有显著的多峰最优行为分布。
- **MuJoCo**：标准连续控制基准（HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5），奖励密集，通常为单峰动作分布。
- **2D Multi-Goal Task**：二维平面中多个目标点的导航任务，用于可视化策略的多模态行为。

### ⚙️ 实验设置
- **训练步数**：1百万环境交互步（1M env steps）
- **评估频率**：每 10,000 步评估一次
- **随机种子**：5 次独立运行取均值 ± 标准差
- **硬件配置**：单张 NVIDIA V100 GPU + 5 CPU 核心
- **ODE 求解器**：Euler 方法，固定 10 步积分

### 🎯 评估指标
- **FrankaKitchen**：
  - 平均完成子任务数（Accomplished Tasks）
  - 成功率（Success Rate %）
- **MuJoCo**：
  - 最终平均回报（Maximum Average Return）
- **其他**：
  - 训练时间（Training Time）
  - GPU 内存占用（GPU Memory Usage）
  - 消融实验分析（Ablation Studies）

### 🆚 基线方法对比
涵盖三类共 **7 种基线算法**：
1. **经典 RL 方法**：
   - PPO, SAC, TD3
2. **扩散策略（Diffusion-based）**：
   - DIPO, QVPO, DPMD
3. **流匹配策略（Flow Matching-based）**：
   - FPO

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ FrankaKitchen 结果（Table 1）
| 方法 | N=1 Success (%) | N=4 Success (%) | N=7 Success (%) |
|------|------------------|------------------|------------------|
| SAC | 80.00 | 31.00 | 17.43 |
| DIPO | 22.00 | 23.50 | 25.43 |
| QVPO | 2.00 | 2.00 | 21.14 |
| **FMER** | **100.00** | **50.00** | **45.71** |

> FMER 在所有复杂度下均取得最佳表现，尤其在高维多任务场景（N=7）上成功率接近翻倍。

#### ✅ MuJoCo 结果（Table 2）
| 方法 | HalfCheetah | Walker2d | Humanoid |
|------|-------------|----------|----------|
| SAC | 10295.33 | 4453.24 | **5350.96** |
| DIPO | 9516.47 | 5028.66 | 5123.38 |
| DPMD | 10706.06 | 4910.31 | 5100.60 |
| QVPO | 10309.60 | 4846.39 | 5078.00 |
| **FMER** | **12332.38** | **5285.08** | **5286.12** |

> FMER 在 HalfCheetah 和 Walker2d 上达到 SOTA，在 Humanoid 上仅次于 SAC，但在生成式策略中排名第一。

### 🔁 与基线方法的对比结果
- **优于所有扩散策略**：在 FrankaKitchen 上大幅领先 QVPO、DPMD 等，说明其更强的多模态建模与探索能力。
- **超越传统 RL 方法**：在稀疏奖励环境下（如 FrankaKitchen），SAC 因单峰限制性能下降，而 FMER 表现稳健。
- **优于流匹配基线 FPO**：FPO 基于 PPO 框架，缺乏显式熵控制，导致探索不足；FMER 利用熵正则化实现了更优性能。

### 🔍 消融实验结果（Section 5.3）

#### (a) 优势加权策略比较（Figure 6a）
- **Hard Top-1 Selection**：性能早衰，因过度依赖 critic 最佳估计，缺乏鲁棒性。
- **Unnormalized Weighting**：权重总量不一致，导致学习信号不稳定。
- **Soft Weighting（FMER 默认）**：平滑收敛，性能最优，验证软加权的有效性。

#### (b) 熵正则化消融（Figure 6b）
- **早期关闭熵损失（1e4 步）**：立即崩溃，表明初始阶段熵对探索至关重要。
- **中期关闭（1e5 步）**：仍能收敛，但波动更大。
- **晚期关闭（5e5 步）**：曲线更平滑，说明后期可减少探索噪声以精调策略。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ODE-based Flow Matching 更适合在线 RL**：
   - 确定性路径避免了 SDE 的随机性带来的熵不可计算问题。
   - 支持直接、可微分的熵建模，实现原则性的最大熵优化。

2. **显式熵控制对稀疏奖励环境至关重要**：
   - 尤其在 FrankaKitchen 这类多模态任务中，良好的探索机制决定了能否发现多个可行解路径。

3. **软加权优于硬选择**：
   - 利用整个候选集的优势信息进行加权回归，提升了策略更新的稳定性与质量。

4. **高效且可扩展**：
   - FMER 在 Humanoid 等高维任务上仍保持良好性能，得益于 Hutchinson trace estimator 对散度的高效估计。

### ⚠️ 方法的局限性
1. **固定步长 Euler 求解器**：
   - 积分精度受限，可能引入轨迹偏差。
2. **固定目标熵 H**：
   - 缺乏自适应调节机制，在不同训练阶段可能不是最优。
3. **温度参数 T 需手动调参**：
   - 敏感性分析显示不同环境需要不同的 `T` 值（见 Appendix C），缺乏自动化调度。

### 🔮 未来工作方向
- 探索更高阶的 ODE 求解器（如 Runge-Kutta）以提高采样精度。
- 设计 **adaptive entropy scheduling** 机制，动态调整 Lagrange multiplier α 或目标熵 H。
- 研究自动调节优势温度 T 的方法，降低超参数敏感性。
- 扩展到离线 RL 和多智能体场景。

---

> 💡 **总结一句话**：  
> **FMER 通过将 Flow Matching 与显式熵正则化结合，构建了一个高效、稳定、可解释的生成式策略框架，在复杂多模态任务中实现了 SOTA 性能，同时显著降低了训练成本。**

</details>

---

### 14. [MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment](https://arxiv.org/abs/2603.17694)

**Authors**: Yusen Wu, Yiran Liu, Xiaotie Deng  
**Category**: cs.AI  
**Published**: 2026-03-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.17694v1  

#### Abstract
In the real economy, modern decision-making is fundamentally challenged by high-dimensional, multimodal environments, which are further complicated by agent heterogeneity and combinatorial data sparsity. This paper introduces a Multi-Agent Large Language Model-based Economic Sandbox (MALLES), levera...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代真实经济（real economy）中的决策面临三大核心挑战：
- **类别级数据稀疏性**（category-level data sparsity）：单一产品类别的交易记录不足以覆盖高维特征空间；
- **分布外泛化能力差**（Poor OOD generalization）：难以对新类别或长尾商品进行准确预测；
- **高维多模态特征处理困难**（high-dimensional multimodal features）：传统模型在处理文本、图像等非结构化信息时效率低下且易丢失语义。

现有 LLM 经济模拟框架（如 EconAgent、FinCon）虽能实现语义交互，但在**数值敏感性**（numerical sensitivity，如价格弹性）、**跨域泛化**和**多模态对齐**方面表现不足。

---

### 🚀 提出的新方法与创新思路

本文提出 **MALLES**（Multi-Agent Large Language Model-based Economic Sandbox），一个基于多智能体 LLM 的统一经济沙盒框架，其核心创新包括：

#### （1）**跨类别后训练实现消费者偏好对齐**（Cross-category Post-training for Preference Alignment）
- 利用大规模异构交易数据对 LLM 进行 post-training，使其内化并迁移潜在的消费偏好模式；
- 通过跨类别知识转移缓解单个品类的数据稀疏问题，提升 OOD 泛化能力。

#### （2）**多智能体讨论机制**（Multi-agent Discussion Framework）
- 引入角色分工的多 agent 架构（如经销商、客服、制造商），通过结构化对话协同推理；
- 分布认知负荷，避免单 agent 注意力瓶颈，增强可解释性和策略多样性。

#### （3）**均值场稳定机制 + 注意力控制**（Mean-field Stabilization & Attention Control）
- 均值场机制建模客户群体与产品环境的动态交互，提升采样稳定性；
- 输入增强与注意力先验（attention priors）确保价格、折扣等关键经济变量被合理加权。

---

### 🔍 相比现有方法的优势
| 方面 | 现有方法局限 | MALLES 改进 |
|------|----------------|-------------|
| 数据利用 | 单品类训练，受限于数据稀疏 | 跨品类联合训练，知识迁移 |
| 数值敏感性 | 缺乏对价格弹性的建模 | 显式注入经济规律，支持 quantity 预测 |
| 可解释性 | 黑箱输出 | 多 agent 对话生成 human-readable 决策规则 |
| 稳定性 | 输出方差大 | 均值场 + 一致性正则化降低波动 |
| 泛化能力 | OOD 表现差 | 全类别训练显著提升新类别适应 |

---

## 2. 核心实验方法和设置

### 📊 数据集构建
- **来源**：真实工业销售数据
- **规模**：119,252 名顾客，3,361 个商品类别
- **数据类型**：
  - 交易记录（timestamp, customer ID, product ID, quantity, unit price, discount, channel, review）
  - 商品信息（ID, category, base price, 图像 embedding, attributes, 销售序列）
  - 客户画像（income bracket, buyer type, purchase history, brand loyalty, 促销敏感度）
  - 对话日志（role, turn, negotiated outcome）

> 图1展示了从原始交易数据到结构化 prompt 的构建流程。

---

### ⚙️ 实验设置

#### 模型选择
使用四种代表性 LLM 进行推理测试（每次随机选取一种）：
- GPT-5.2 (OpenAI 2025)
- Gemini-3 (Team 2025)
- DeepSeek-V2 (DeepSeek-AI 2024)
- Llama-4 (70B)

#### 实验配置对比
| 配置维度 | 设置选项 |
|--------|---------|
| 模型版本 | Base LLM vs. Post-trained LLM |
| 决策架构 | Single-agent vs. Multi-agent discussion |
| 采样方式 | Standard sampling vs. Mean-field sampling |
| 测试对象 | Bottom 50% customers（普通消费者） |

> 多 agent 场景中，agents 轮流发言多轮后汇总决策；single-agent 直接输出。

---

### 📈 评估指标
1. **SKU 选择命中率**（Hit Rate）：预测产品是否购买的准确率
2. **购买数量相对误差**（Relative Quantity Error）
3. **稳定性**（Stability）：多次采样下预测数量的方差（越低越好）
4. **OOD 泛化能力**：在未见品牌/品类上的表现
5. **时间成本**（Time Cost）：归一化计算开销

#### 基线方法对比
- **EconAgent** (Li et al. 2024b)
- **ABIDES-Economist** (Dwarakanath et al. 2025)
- **FinCon** (Yu et al. 2024)
- **LLM Economist** (Horton 2023)
- **Base Sandbox**（无增强的 baseline）

---

## 3. 主要实验结果和性能指标

### 📊 性能对比（图3）

| 方法 | Hit Rate | Quantity Error | Stability | Time Cost |
|------|----------|----------------|-----------|------------|
| **MALLES (base)** | **0.700** | 0.825 | 0.68 | ~1.0 |
| **MALLES (enhanced)** | **0.775** | 0.790 | **0.72** | ~1.3 |
| FinCon | 0.800 | **1.325** | 0.55 | ~1.1 |
| ABIDES-Economist | 0.580 | 0.650 | 0.60 | ~1.2 |
| LLM Economist | 0.520 | 0.710 | 0.58 | ~1.0 |

> 💡 尽管 FinCon 在 hit rate 上略高，但 quantity error 显著偏高，说明存在系统性偏差；而 MALLES 在准确性与稳定性之间取得最佳平衡。

---

### 🔬 消融实验结果（图4）

#### （1）后训练对齐消融（Post-training Alignment）
| 训练设置 | Industry Hit Rate | Quantity Error | OOD Error |
|--------|------------------|----------------|------------|
| None | 0.20 | 0.96 | >1.0 |
| Epoch=10 | 0.58 | 0.62 | 0.78 |
| Epoch=50 | **0.73** | **0.41** | **0.52** |

✅ 结论：post-training 显著提升所有指标，尤其改善数值推理和 OOD 泛化。

#### （2）多智能体 vs 单智能体
- 多 agent 在 **批发场景** 中优势明显（策略更全面，避免局部最优）；
- 随着讨论轮次增加，收益递减，但 **3–4 轮对话即可带来显著提升**；
- 成本随 agent 数线性增长，适合复杂决策，零售轻量场景仍推荐 single-agent。

#### （3）均值场机制效果
| 观察窗口长度（月） | Hit Rate | Quantity Error | Variance ↓ |
|--------------------|--------|----------------|------------|
| None | 0.68 | 0.85 | 0.71 |
| Window=5 | 0.71 | 0.80 | 0.65 |
| Window=10 | **0.73** | **0.77** | **0.62** |

✅ 中等长度窗口（5–10个月）效果最佳，过长反而引入噪声。

#### （4）全类别训练 vs 其他数据组合（表1）
| 数据策略 | 平均 Hit Rate | 平均 Quantity Error |
|--------|--------------|---------------------|
| Background-only | 0.21 | 0.98 |
| Single-category | 0.59 | 0.40 |
| Similar-category | 0.54 | 0.51 |
| **Full-category** | **0.55** | **0.50** |

📌 发现：**full-category 训练在数据稀缺新品类上表现优于 specialized 模型**，验证其作为通用 fallback 的价值。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 可通过 post-training 实现“货币意识”**（monetary awareness），学会价格弹性、折扣敏感等经济行为；
2. **跨类别训练是解决数据稀疏的关键路径**，能有效迁移偏好表示；
3. **多 agent 协作显著提升决策质量与可解释性**，尤其适用于批发、供应链等复杂谈判场景；
4. **均值场机制 + 注意力控制可大幅提升模拟稳定性**，使输出更贴近真实市场分布；
5. MALLES 在 **accuracy、stability、generalization** 三方面均优于现有 LLM 经济模拟器。

---

### ⚠️ 局限性
1. **跨模态对齐尚不充分**：视觉、数值信号融合仍弱于文本理解；
2. **微观个体与宏观动态耦合不足**：当前 mean-field 是简化近似，缺乏复杂网络效应建模；
3. **计算开销较高**：multi-agent 和 full-context 推理限制实时部署；
4. **伦理风险**：精准行为预测可能被用于操纵性营销，威胁消费者自主性。

---

### 🔮 未来工作方向
1. **深化多模态经济表征学习**：融合图像、价格曲线、评论情感等多源信息；
2. **构建多层次模拟系统**：连接 agent-level 决策与 market-level emergence（如供需震荡、价格战）；
3. **引入因果推理机制**：识别隐藏变量影响，提升反事实分析能力；
4. **开发轻量化推理方案**：如 agent pruning、early exit、蒸馏技术；
5. **制定伦理准则**：防止滥用个性化预测，保障公平透明的商业实践。

---

> ✅ **总体评价**：  
> MALLES 成功将 LLM 的语义泛化能力与经济系统的结构性要求相结合，为高保真、可扩展的真实经济决策模拟提供了坚实基础，代表了 LLM-based economic simulation 的重要进展。

</details>

---

### 15. [A Family of LLMs Liberated from Static Vocabularies](https://arxiv.org/abs/2603.15953)

**Authors**: Aleph Alpha,  :, Adnen Abdessaied, Artur Baranowski, Lukas Balles, Michael Barlow, Fabien C. Y. Benureau, Felix Berkenkamp, Lukas Bluebaum, Bastian Boll, Thomas F. Burns, Bj\"orn Deiseroth, Constantin Eichenberg, David Friede, Pablo Iyu Guerrero, Ahmed Hammam, Bastian Harren, Johann Higl, Yasser Jadidi, Carina Kauf, Johannes Messner, Jan Hendrik Metzen, Max Meuer, Vedant Nanda, Pit Neitemeier, Koen Oostermeijer, Letitia Parcalabescu, Markus Pernpointner, Felix Reinfurt, Dylan Rodriquez, Gr\'egory Schott, Philipp Siedler, Martin Simonovsky, Till Speicher, Volker Stampa, Stephan W\"aldchen, Samuel Weinbach, Gregor Ziegltrum  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.15953v1  

#### Abstract
Tokenization is a central component of natural language processing in current large language models (LLMs), enabling models to convert raw text into processable units. Although learned tokenizers are widely adopted, they exhibit notable limitations, including their large, fixed vocabulary sizes and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Family of LLMs Liberated from Static Vocabularies

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 **Large Language Models (LLMs)** 严重依赖于固定的、静态的 **vocabulary**（如通过 BPE 或 SentencePiece 学习的子词表），这带来了以下问题：
- **固定词汇大小**：无法灵活适应新领域、新语言或罕见词。
- **跨语言/领域泛化能力差**：在低资源语言（如德语）或特定领域文本上表现不佳。
- **对拼写变体鲁棒性弱**：例如 “color” 和 “colour” 被视为完全不同的 token。
- **参数冗余**：embedding matrix 和 LM head 占用大量参数，且未利用 token 间的相似性。

### 提出的新方法或新思路
提出了一种名为 **Hierarchical Autoregressive Transformer (HAT)** 的架构，并引入 **Tokenizer-Free (T-Free)** 范式，其核心思想是：
- **直接处理字节 (byte-level processing)**：输入文本以 UTF-8 字节序列形式进入模型。
- **动态分词机制**：采用基于规则的 **word splitter**（遵循 UAX#29）将字节流切分为“词”单元，而非依赖预训练的静态 tokenizer。
- **三层级结构**：
  1. **Encoder**：将字节序列编码为字节嵌入，并通过 cross-attention 将其聚合成词级嵌入（word embeddings）。
  2. **Backbone**：标准因果 Transformer，处理词级表示并生成下一个词的预测。
  3. **Decoder**：结合字节上下文与 backbone 输出的词级信息，逐字生成输出。

该方法被称为 **HATification** —— 可将已有 LLM（如 Llama 3.1）的 backbone 适配到此框架中，替换原有 tokenizer。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **适应性** | 不受固定 vocab 限制，天然支持多语言、新词、拼写变异等。 |
| **压缩效率** | 显著提升 **text compression ratio**（backbone 序列长度更短）。 |
| **参数效率** | 移除大型 embedding matrix 和 LM head，减少非 backbone 参数占比（从 13% → <3%）。 |
| **可复现性与开放性** | 公开全部模型（含 200 个 pre-training checkpoints）、evaluation framework 和推理优化代码。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
预训练数据来自多源混合语料库，总计约 **4T words**，具体构成如下：

| 类别 | 来源 | 比例 |
|------|------|------|
| 英语文本 | 高质量网页、公共书籍、合成数据 | 70% |
| 德语文本 | 同上 + 合成翻译数据 | 7% |
| 数学内容 | 数学证明、方程、编程中的数学逻辑 | 5% |
| 编程代码 | Python 及通用编程语言代码 | 18% |

> 注：使用 fastText 进行语言识别，Resiliparse 提取 HTML 文本，进行去重、过滤低质内容等清洗流程。

### 实验设置和评估指标

#### 模型配置
构建了三个主要模型：
- **Llama-TFree-HAT-Pretrained**：7B 参数，从头预训练。
- **Llama-3.1-8B-TFree-HAT**：基于 Llama 3.1 8B 的 backbone 进行 HATification。
- **Llama-3.1-70B-TFree-HAT**：同上扩展至 70B 规模。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Model Quality** | 多项基准任务上的平均得分（如 MMLU、ARC、HellaSwag 等）。 |
| **Compression** | 定义为 **bytes per sequence position**（越高越好），衡量 backbone 输入的压缩率。 |
| **Efficiency** | 推理时的吞吐量与延迟（受限于 vLLM 实现）。 |

#### 基线方法对比
- **Llama 3.1 8B / 70B Base / Instruct**
- **Tulu 3.1 8B SFT / DPO**

所有模型均在同一 evaluation framework 下测试，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（DPO 模型）

| Benchmark | Llama-3.1-8B-Instruct | Llama-3.1-8B-TFree-HAT-DPO | 提升情况 |
|----------|------------------------|-------------------------------|---------|
| **MMLU 5-shot** | 0.697 | **0.657** | - |
| **GPQA 0-shot** | 0.330 | **0.286** | - |
| **ARC Challenge 25-shot** | 0.654 | **0.668** | ✅ |
| **HellaSwag 10-shot** | 0.764 | **0.775** | ✅ |
| **German MMMLU 5-shot** | 0.605 | **0.594** | ≈ |
| **German ARC Challenge DE** | 0.515 | **0.594** | ✅✅ |
| **WMT16 BLEU (DE)** | 34.350 | **34.778** | ✅ |
| **AlpacaEval CS** | 0.186 | **0.420** | ✅✅✅ |
| **MT-Bench Win Rate (EN)** | — | **71.0% vs Llama-Instruct** | ✅✅✅ |
| **MT-Bench Win Rate (DE)** | — | **73.9% vs Llama-Instruct** | ✅✅✅ |

> 注：“-” 表示原模型更强，“✅” 表示 HAT 模型更优。

### 与基线方法的对比结果
- 在大多数英文任务上，**Llama-3.1-8B-TFree-HAT-DPO** 与 Llama-Instruct 性能相当，部分推理任务（如 ARC）略有领先。
- 在 **德语任务上全面超越**，尤其在 MMMLU、ARC、HellaSwag 上显著优于原始 Llama 模型。
- **AlpacaEval 上指令遵循能力大幅提升**，CS（Chatbot Style）得分翻倍以上。
- **压缩效率显著提高**：
  - 英文平均 compression 达 **5.18 bytes/position**（vs Llama 的 4.28）
  - 德语高达 **6.03 bytes/position**（vs Llama 的 3.33），说明 HAT 架构在低资源语言上有更强的信息压缩能力。

### 消融实验结果（隐含分析）
虽然未设独立消融章节，但从设计中可推断关键发现：
- **Backbone 冻结策略有效**：先冻结 backbone 训练 encoder/decoder，再联合微调，有助于稳定 HATification 过程。
- **QK-Norm 与 Softcapping 对训练稳定性重要**：仅在从头训练的 7B 模型中使用，提升了收敛性。
- **Dual KV Cache 设计必要但复杂**：需分别管理 byte-level 与 word-level KV cache，带来实现挑战但保证了正确性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **Tokenizer-Free 是可行且高效的替代路径**：无需学习型 tokenizer，也能达到甚至超越传统 tokenized LLM 的性能。
2. ✅ **HAT 架构具备更强的语言适应性和鲁棒性**：尤其在德语等非英语任务上表现突出，验证了其对新语言的良好泛化能力。
3. ✅ **HATification 成功迁移预训练知识**：可以将 Llama 3.1 的 backbone 成功“嫁接”到 HAT 架构中，在不重新预训练的情况下获得高性能。
4. ✅ **更高的文本压缩率带来潜在推理效率优势**：尽管当前 vLLM 实现尚未完全释放潜力，但理论上有更低的 backbone 计算负担。
5. ✅ **参数更高效**：Llama-3.1-8B-TFree-HAT 实际参数仅为 7.19B（接近 7B），比原版 8.03B 更小，同时性能持平或更好。

### 方法的局限性
- ⚠️ **推理效率仍受限**：由于 dual-sequence processing 和 variable-length byte generation，当前 vLLM 实现的吞吐低于同等规模的 tokenized 模型。
- ⚠️ **缺乏对 code/math 的专项优化**：虽包含相关数据，但架构未针对这些任务优化，故未在 CodeGen 或 GSM8K 上取得 SOTA。
- ⚠️ **训练成本高**：需要开发专用基础设施（如修改 FlashAttention 支持长 query 序列）。
- ⚠️ **context parallelism 支持不足**：现有 ring attention 实现不兼容 sliding window attention，导致长上下文训练困难。

### 未来工作方向
- 🔮 开发更高效的 **inference engine**，专门优化 HAT 架构的双路缓存与异步生成。
- 🔮 探索 **更大规模的 HAT 模型**（>70B）及其 scaling laws。
- 🔮 将 HAT 扩展至 **multimodal 或 programming language modeling** 场景。
- 🔮 利用公开的 **200 个 pre-training checkpoints** 研究 learning dynamics、capability emergence 和 grokking 现象。
- 🔮 改进 **word splitter** 的智能程度，探索可学习的 segmentation policy。

---

> 📢 **开源贡献亮点**：
> - 所有模型发布于 Hugging Face：[Aleph-Alpha org](https://huggingface.co/Aleph-Alpha)
> - 包含 **200 个 pre-training checkpoints**，远超 Pythia 系列，极大促进 developmental interpretability 研究。
> - 开源 **evaluation framework** 与 **vLLM 修改版本**，推动社区共建 tokenizer-free 生态。

--- 

📌 **总结一句话**：  
本文提出了 **HAT 架构** 与 **Tokenizer-Free 范式**，成功摆脱了 LLM 对静态 vocab 的依赖，在保持甚至提升性能的同时增强了语言适应性、压缩效率和参数利用率，是一次对现代 LLM 架构基础组件的重要革新。

</details>

---

### 16. [Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR](https://arxiv.org/abs/2603.16184)

**Authors**: Quy-Anh Dang, Chris Ngo  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.16184v1  

#### Abstract
We present Polyglot-Lion, a family of compact multilingual automatic speech recognition (ASR) models tailored for the linguistic landscape of Singapore, covering English, Mandarin, Tamil, and Malay. Our models are obtained by fine-tuning Qwen3-ASR-0.6B and Qwen3-ASR-1.7B exclusively on publicly avai...

---

### 17. [PlotTwist: A Creative Plot Generation Framework with Small Language Models](https://arxiv.org/abs/2603.16410)

**Authors**: Abhinav Thorat, Ravi Kolla, Jyotin Goel, Niranjan Pedanekar  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.16410v1  

#### Abstract
Creative plot generation presents a fundamental challenge for language models: transforming a concise premise into a coherent narrative that sustains global structure, character development, and emotional resonance. Although recent Large Language Models (LLMs) demonstrate strong fluency across gener...

---

### 18. [Can Linguistically Related Languages Guide LLM Translation in Low-Resource Settings?](https://arxiv.org/abs/2603.16660)

**Authors**: Aishwarya Ramasethu, Niyathi Allu, Rohin Garg, Harshwardhan Fartale, Dun Li Chan  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.16660v1  

#### Abstract
Large Language Models (LLMs) have achieved strong performance across many downstream tasks, yet their effectiveness in extremely low-resource machine translation remains limited. Standard adaptation techniques typically rely on large-scale parallel data or extensive fine-tuning, which are infeasible...

---

### 19. [PRISM: Demystifying Retention and Interaction in Mid-Training](https://arxiv.org/abs/2603.17074)

**Authors**: Bharat Runwal, Ashish Agrawal, Anurag Roy, Rameswar Panda  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.17074v1  

#### Abstract
We present PRISM, a comprehensive empirical study of mid-training design choices for large language models. Through controlled experiments across seven base models spanning four families (Granite, LLaMA, Mistral, Nemotron-H), two architecture types (dense Transformer and attention-Mamba hybrid), and...

---

### 20. [Frequency Matters: Fast Model-Agnostic Data Curation for Pruning and Quantization](https://arxiv.org/abs/2603.16105)

**Authors**: Francesco Pio Monaco, Elia Cunegatti, Flavio Vella, Giovanni Iacca  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.16105v1  

#### Abstract
Post-training model compression is essential for enhancing the portability of Large Language Models (LLMs) while preserving their performance. While several compression approaches have been proposed, less emphasis has been placed on selecting the most suitable set of data (the so-called \emph{calibr...

---

### 21. [SIA: A Synthesize-Inject-Align Framework for Knowledge-Grounded and Secure E-commerce Search LLMs with Industrial Deployment](https://arxiv.org/abs/2603.16137)

**Authors**: Zhouwei Zhai, Mengxiang Chen, Anmeng Zhang  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.16137v1  

#### Abstract
Large language models offer transformative potential for e-commerce search by enabling intent-aware recommendations. However, their industrial deployment is hindered by two critical challenges: (1) knowledge hallucination due to insufficient encoding of dynamic, fine-grained product knowledge, and (...

---

### 22. [Omnilingual SONAR: Cross-Lingual and Cross-Modal Sentence Embeddings Bridging Massively Multilingual Text and Speech](https://arxiv.org/abs/2603.16606)

**Authors**: Omnilingual SONAR Team, Jo\~ao Maria Janeiro, Pere-Llu\'is Huguet Cabot, Ioannis Tsiamas, Yen Meng, Vivek Iyer, Guillem Ram\'irez, Loic Barrault, Belen Alastruey, Yu-An Chung, Marta R. Costa-Jussa, David Dale, Kevin Heffernan, Jaehyeong Jo, Artyom Kozhevnikov, Alexandre Mourachko, Christophe Ropers, Holger Schwenk, Paul-Ambroise Duquenne  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.16606v1  

#### Abstract
Cross-lingual sentence encoders typically cover only a few hundred languages and often trade downstream quality for stronger alignment, limiting their adoption. We introduce OmniSONAR, a new family of omnilingual, cross-lingual and cross-modal sentence embedding models that natively embed text, spee...

---

### 23. [HoloByte: Continuous Hyperspherical Distillation for Tokenizer-Free Modeling](https://arxiv.org/abs/2603.16917)

**Authors**: Vladimer Khasia  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.16917v1  

#### Abstract
Sequence modeling universally relies on discrete subword tokenization to circumvent the $\mathcal{O}(N^2)$ computational intractability of native byte-level attention. However, this heuristic quantization imposes artificial morphological boundaries, enforces vocabulary dependence, and fractures the ...

---

### 24. [SCE-LITE-HQ: Smooth visual counterfactual explanations with generative foundation models](https://arxiv.org/abs/2603.17048)

**Authors**: Ahmed Zeid, Sidney Bender  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.17048v1  

#### Abstract
Modern neural networks achieve strong performance but remain difficult to interpret in high-dimensional visual domains. Counterfactual explanations (CFEs) provide a principled approach to interpreting black-box predictions by identifying minimal input changes that alter model outputs. However, exist...

---

### 25. [Efficient Soft Actor-Critic with LLM-Based Action-Level Guidance for Continuous Control](https://arxiv.org/abs/2603.17468)

**Authors**: Hao Ma, Zhiqiang Pu, Xiaolin Ai, Huimu Wang  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.17468v1  

#### Abstract
We present GuidedSAC, a novel reinforcement learning (RL) algorithm that facilitates efficient exploration in vast state-action spaces. GuidedSAC leverages large language models (LLMs) as intelligent supervisors that provide action-level guidance for the Soft Actor-Critic (SAC) algorithm. The LLM-ba...

---

### 26. [Translation Invariance of Neural Operators for the FitzHugh-Nagumo Model](https://arxiv.org/abs/2603.17523)

**Authors**: Luca Pellegrini  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.17523v1  

#### Abstract
Neural Operators (NOs) are a powerful deep learning framework designed to learn the solution operator that arise from partial differential equations. This study investigates NOs ability to capture the stiff spatio-temporal dynamics of the FitzHugh-Nagumo model, which describes excitable cells. A key...

---

### 27. [Symmetry-Reduced Physics-Informed Learning of Tensegrity Dynamics](https://arxiv.org/abs/2603.17824)

**Authors**: Jing Qin, Muhao Chen  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.17824v1  

#### Abstract
Tensegrity structures possess intrinsic geometric symmetries that govern their dynamic behavior. However, most existing physics-informed neural network (PINN) approaches for tensegrity dynamics do not explicitly exploit these symmetries, leading to high computational complexity and unstable optimiza...

---

### 28. [Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training](https://arxiv.org/abs/2603.17970)

**Authors**: Ben S. Southworth, Stephen Thomas  
**Category**: cs.LG  
**Published**: 2026-03-19  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.17970v1  

#### Abstract
Orthogonalized-momentum optimizers such as Muon improve transformer training by approximately whitening/orthogonalizing matrix-valued momentum updates via a short polar-decomposition iteration. However, polar-factor approximations typically require multiple large matrix multiplications, and the resu...

---

### 29. [MedArena: Comparing LLMs for Medicine-in-the-Wild Clinician Preferences](https://arxiv.org/abs/2603.15677)

**Authors**: Eric Wu, Kevin Wu, Jason Hom, Paul H. Yi, Angela Zhang, Alejandro Lozano, Jeff Nirschl, Jeff Tangney, Kevin Byram, Braydon Dymm, Narender Annapureddy, Eric Topol, David Ouyang, James Zou  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.15677v1  

#### Abstract
Large language models (LLMs) are increasingly central to clinician workflows, spanning clinical decision support, medical education, and patient communication. However, current evaluation methods for medical LLMs rely heavily on static, templated benchmarks that fail to capture the complexity and dy...

---

### 30. [Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning](https://arxiv.org/abs/2603.15981)

**Authors**: Jingxiang Chen, Minseok Kim, Seong-Gyun Leem, Yin Huang, Rashi Rungta, Zhicheng Ouyang, Haibin Wu, Surya Teja Appini, Ankur Bansal, Yang Bai, Yue Liu, Florian Metze, Ahmed A Aly, Anuj Kumar, Ariya Rastrow, Zhaojiang Lin  
**Category**: cs.CL  
**Published**: 2026-03-19  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.15981v1  

#### Abstract
Speech large language models (LLMs) observe paralinguistic cues such as prosody, emotion, and non-verbal sounds--crucial for intent understanding. However, leveraging these cues faces challenges: limited training data, annotation difficulty, and models exploiting lexical shortcuts over paralinguisti...

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
