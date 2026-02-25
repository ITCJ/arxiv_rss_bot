# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-25 06:47:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training](https://arxiv.org/abs/2602.20656)

**Authors**: Guanbin Xu, ZhenGuo Xu, Yuzhe Li, Youhui Bai, Ping Gong, Chaoyi Ruan, Cheng Li  
**Category**: cs.DC  
**Published**: 2026-02-25  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2602.20656v1  

#### Abstract
Overlapping communication with computation is crucial for distributed large-model training, yet optimizing it - especially when computation becomes the bottleneck-remains challenging. We present Lagom, a system that co-tunes communication parameters to balance resource usage between computation and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Lagom: Unleashing the Power of Communication and Computation Overlapping for Distributed LLM Training**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
在分布式大模型训练中，**通信与计算重叠（communication-computation overlapping）** 是提升训练效率的关键手段。然而，当计算成为瓶颈时，通信操作仍会因争夺 GPU 资源（如 SMs、内存带宽）而干扰计算，导致性能下降。现有方法（如 AutoCCL）主要优化通信瓶颈场景，在计算受限情况下反而可能加剧资源争用，造成负优化。

Lagom 针对这一“双瓶颈”场景提出系统性解决方案，旨在通过**协同调优（co-tuning）通信参数**，平衡通信与计算之间的资源分配，从而最大化整体训练吞吐。

---

### 🚀 **提出了什么新方法或新思路**

1. **统一的重叠性能建模（Unified Overlap Performance Model）**  
   提出一个适用于**计算瓶颈**和**通信瓶颈**两种场景的统一成本模型，将总执行时间 $ Z = \max(X, Y) $ 建模为通信时间 $ X $ 和计算时间 $ Y $ 的最大值，并显式建模通信参数对计算性能的影响。

2. **资源争用建模（Contention Modeling）**  
   将通信-计算争用分为两类：
   - **SM 竞争（SM competition）**：由通信占用 Streaming Multiprocessors 导致计算波次（wave）增加。
   - **全局资源竞争（Global resource contention）**：通信对全局内存带宽和 L2 缓存的竞争，受 chunk size 和 number of channels 影响显著。

3. **优先级驱动的搜索算法（Priority-based Search Algorithm）**  
   引入 **priority metric H_j** 来衡量每个通信操作调优的性价比：
   $$
   H_j = \frac{Y' - Y}{x - x'}
   $$
   其中分子是计算开销的变化，分母是通信加速收益。越小的 $ H_j $ 表示单位计算代价下获得的通信增益越高，指导系统优先调优高价值通信。

4. **线性复杂度优化**  
   传统联合调优空间随通信数量呈指数增长（$ O(r^N) $），Lagom 通过启发式顺序调优 + 动态反馈机制，将搜索复杂度从**指数级降至线性级**，实现高效收敛。

---

### 🔍 **相比现有方法的优势**

| 方法 | 局限性 | Lagom 的改进 |
|------|--------|-------------|
| **NCCL** | 固定配置，默认激进使用资源（如大 NC），在计算瓶颈下引发严重争用 | 自动选择轻量资源配置，缓解干扰 |
| **AutoCCL** | 仅关注通信性能最优化，忽视其对计算的负面影响 | 显式建模通信对计算的拖慢效应，追求整体最优 |
| **Libra / Liger** | 仅考虑部分参数或静态分配，缺乏动态适应能力 | 支持多参数联合调优，适用于多种并行范式（FSDP、TP、EP） |

> ✅ **核心优势**：Lagom 在计算成为瓶颈时仍能有效优化，避免“为了加速通信而拖慢整体”的反效果。

---

## 2. **核心实验方法和设置**

### 🧪 **使用的模型与 workload**
在多个主流大规模语言模型上进行测试，涵盖密集模型与 MoE 架构：

- **Dense Models**:
  - Phi-2-2B
  - Llama-3-8B
  - MPT-7B
- **MoE Models**:
  - DeepSeek-MoE-16B
  - OLMoE-1B-7B

训练框架基于 **Megatron-LM** 和 **PyTorch 2.3.0**，支持以下并行策略：
- Fully Sharded Data Parallelism (**FSDP**)
- Tensor Parallelism (**TP**)（结合 DeepSpeed-Domino）
- Expert Parallelism (**EP**)（采用 dual-batch overlapping）

---

### ⚙️ **硬件基础设施**
使用两个不同带宽特性的 GPU 集群验证通用性：

| 集群 | 节点数 | GPU 数量 | 内部连接 | 节点间连接 |
|------|--------|----------|-----------|--------------|
| **Cluster A** | 2 | 8×A40/node | NVLink (400Gbps) | InfiniBand (2×400Gbps) |
| **Cluster B** | 2 | 8×A40/node | PCIe 4.0 | InfiniBand (100Gbps) |

> 注：CUDA 版本为 12.8，Driver 570.13。

---

### 📊 **评估指标**
- **Training iteration time**（训练迭代耗时）：主性能指标
- **Speedup over baselines**：相对于 NCCL 和 AutoCCL 的加速比
- **Convergence behavior**：调优过程中的性能演化路径
- **Tuning efficiency**：搜索收敛所需的迭代次数

---

### 🆚 **基线方法对比**
- **NCCL v2.18.3-1**：工业标准通信库，使用默认参数
- **AutoCCL**：当前最先进的自动通信调优系统，作为强基线

Lagom 基于 AutoCCL 的 divide-and-conquer 框架构建，复用其实现基础，专注于优化 resource-related 参数（NC, NT, C）。

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据**

#### ✅ **端到端训练加速比（Speedup）**

| 场景 | 对比 NCCL | 对比 AutoCCL |
|------|---------|------------|
| **FSDP（高/低带宽集群）** | **1.10–1.33×** | **1.07–1.33×** |
| **TP / EP** | **1.08–1.16×** | **1.03–1.27×** |

> 图7显示，在 FSDP 下 Lagom 显著优于 NCCL 和 AutoCCL；尤其在 NVLink 高带宽环境下，NCCL 默认使用较大 NC 导致严重争用，Lagom 通过降低资源配置反而取得更优表现。

---

#### 🔍 **典型模式分析（Pattern Breakdown）**

##### **Pattern 1（计算瓶颈）**
- **操作**：AllGather（FSDP 中）
- **NCCL 配置**：NC=8, C=2MB
- **AutoCCL 调整**：NC=61 → 通信更快，但计算被严重拖慢 → 整体性能仅为 **0.87× NCCL**
- **Lagom 配置**：NC=2, C=684KB → 通信略慢，但计算提速明显 → 实现 **1.35× 加速**

> 💡 发现：过度优化通信会导致“得不偿失”，Lagom 成功识别并规避此陷阱。

##### **Pattern 2（多通信耦合）**
- 包含多个通信阶段（如 ReduceScatter）
- Lagom 根据 priority metric $ H $ 优先调优影响最大的通信
- 最终配置从 (NC=8, C=2MB) → (NC=4, C=1366KB)
- 实现 **1.43× 加速**

---

#### ⏱️ **调优效率（Tuning Efficiency）**

| 方法 | 收敛所需迭代数（两通信任务） |
|------|--------------------------|
| **AutoCCL** | ~16 次 |
| **Lagom** | ~33 次 |

> 虽然 Lagom 迭代更多，但其搜索复杂度为**线性增长**（vs. 指数），且每次迭代开销可控。在整个百万级迭代训练中，调优开销可忽略不计（<0.003%）。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **通信即使在非瓶颈状态下也会影响整体性能**  
   即使通信时间小于计算时间，不当的通信资源配置仍可通过资源争用**拖慢计算最多达 35%**。

2. **“少即是多”原则成立**  
   在计算瓶颈场景下，**减少通信资源投入（如减小 NC 和 C）反而能加快整体训练速度**，因为释放了更多资源给计算。

3. **现有调优器存在“目标错配”问题**  
   AutoCCL 以最小化通信时间为优化目标，但在实际训练中可能导致整体性能下降，说明需转向**以 end-to-end makespan 为目标**的协同优化。

4. **priority metric $ H $ 有效指导搜索方向**  
   该指标能够准确反映不同通信调优的性价比，支持高效的贪心式搜索策略。

---

### ⚠️ **方法的局限性**

1. **依赖于 AutoCCL 的 subspace 划分框架**  
   Lagom 主要在 resource-related 参数层面进行增强，未重新设计 algorithm-level 的搜索逻辑。

2. **假设通信串行化执行**  
   当前模型基于 collective communication 的串行依赖链，若未来引入并发通信机制，需扩展模型。

3. **未覆盖所有硬件平台**  
   实验集中在 A40 + NVLink/PCIe 架构，是否泛化至 H100、MI300 等新型硬件有待验证。

---

### 🔮 **未来工作方向**

1. **扩展至动态 workload 场景**  
   当前调优针对静态模型结构，未来可探索运行时自适应调优（online adaptation）。

2. **集成至完整训练栈**  
   将 Lagom 与调度器、编译器（如 TorchInductor）结合，实现全栈协同优化。

3. **支持异构设备环境**  
   探索在 CPU-GPU 混合、跨数据中心等复杂部署下的参数调优策略。

4. **建模更细粒度的资源争用行为**  
   如 warp scheduling 干扰、TLB 压力等底层细节，进一步提升模型精度。

---

> 📢 **总结一句话**：  
> **Lagom 通过建立通信-计算协同优化框架，首次实现了在计算瓶颈下仍能安全、高效地重叠通信与计算，推动分布式 LLM 训练进入“智能资源调度”新阶段。**

</details>

---

### 2. [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2602.20309)

**Authors**: Jingxuan Zhang, Yunta Hsieh, Zhongwei Wang, Haokun Lin, Xin Wang, Ziqi Wang, Yingtie Lei, Mi Zhang  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2602.20309v1  

#### Abstract
Vision-language-action (VLA) models unify perception, language, and control for embodied agents but face significant challenges in practical deployment due to rapidly increasing compute and memory demands, especially as models scale to longer horizons and larger backbones. To address these bottlenec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
Vision-Language-Action (VLA) 模型在机器人控制中表现出色，能够统一感知、语言推理与动作生成。然而，随着模型规模扩大（尤其是采用 Diffusion Transformer (DiT) 作为动作头），其计算和内存开销急剧上升，严重制约了在嵌入式设备或移动机器人上的部署。

现有效率优化方法（如 TinyVLA、EfficientVLA、VLA-Cache）主要聚焦于架构设计、层剪枝或缓存机制，**普遍忽略对 DiT 动作头进行低精度量化**，因为该模块对激活分布变化极为敏感，直接量化会导致性能崩溃。此外，主流的 Post-Training Quantization (PTQ) 方法（如 SmoothQuant、DuQuant）为单模态或松耦合系统设计，无法应对 VLA 中跨模态、紧耦合带来的 scale drift 问题。

### **提出了什么新方法或新思路**
本文提出 **QuantVLA** —— 首个专为 VLA 模型设计的 **训练免费（training-free）、尺度校准的 PTQ 框架**，并首次成功实现了对 **Diffusion Transformer (DiT) 动作头** 的低比特量化。

其核心创新在于三个 **scale-calibrated 组件**：

1. **Selective Quantization Layout（选择性量化布局）**  
   - 对所有 **Linear 层进行整数量化**（包括 LLM 和 DiT 的 MLP 块）
   - 但保留注意力投影 $Q, K, V, O$ 在 **浮点格式**，以避免放大上游量化误差，维持原始算子调度不变。

2. **Attention Temperature Matching (ATM)**  
   - 引入 **每头标量 $\alpha$** 来匹配教师模型与量化学生模型的 logits 分布标准差。
   - 通过调整 $Q/K$ 的去量化尺度，稳定 softmax 温度，防止注意力分布过锐或过平。
   - 该标量从无标签校准缓冲区估计，并折叠进去量化过程，不增加推理开销。

3. **Output Head Balancing (OHB)**  
   - 引入 **每层标量 $\beta$** 来匹配残差连接前输出头的能量（RMS）。
   - 校正因量化导致的残差流能量漂移，恢复 LayerNorm 的操作点。
   - 同样通过标量折叠实现，无需新增算子。

整个框架 **无需额外训练**，仅需一个小的无标签校准集，支持整数核执行 W4A8 推理，且完全保持原模型架构。

### **相比现有方法的优势**
| 特性 | QuantVLA | 现有方法（如 DuQuant / SmoothQuant） |
|------|----------|-------------------------------|
| 是否支持 DiT 动作头量化 | ✅ 是（首个） | ❌ 否或失败 |
| 是否需要重新训练 | ✅ 否（PTQ） | ✅ 否 |
| 是否改变模型结构 | ✅ 否 | ✅ 否 |
| 是否引入额外推理延迟 | ✅ 否（标量折叠） | ⚠️ 可能（如旋转） |
| 性能是否超越 FP16 基线 | ✅ 是 | ❌ 多数下降 |
| 内存节省 | ~70% | ~60–70%，但性能损失大 |

> **优势总结**：QuantVLA 在不牺牲甚至提升性能的前提下，显著降低内存占用，是目前唯一能在强耦合 VLA 系统上稳定运行的 PTQ 方案。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LIBERO**：主流的 VLA 仿真基准，包含四类任务套件：
  - **Spatial**：空间关系与精确放置
  - **Object**：物体为中心的抓取操作
  - **Goal**：指令到目标对齐
  - **Long**：长视野任务（累计误差挑战）
- **Pick-and-Can（扩展评估）**：用于验证跨任务鲁棒性（见 Appendix F）

### **实验设置和评估指标**
- **模型**：
  - **OpenPI 0.5**：高效型 VLA，DiT 动作头
  - **GR00T N1.5**：高容量通用人形机器人基础模型，DiT + Flow Matching
- **量化配置**：
  - 主要使用 **W4A8**（权重4bit，激活8bit）
  - 校准使用小批量无标签数据（32 batch，128 steps）
  - ATM 和 OHB 参数通过 clip 和 neutrality band 控制稳定性
- **硬件平台**：NVIDIA A100 GPU
- **评估指标**：
  - **Task Success Rate (%)**：标准 LIBERO 协议下的成功率
  - **Memory Usage (GB)**：量化组件的内存消耗
  - **Relative Savings (%)**：相对于 FP16 基线的内存节省比例

### **基线方法对比**
- **FP16 Baseline**：全精度浮点模型
- **DuQuant (LLM+DiT)**：当前最先进的 PTQ 方法，应用于完整堆栈
- **SmoothQuant**：经典通道平滑 PTQ 方法
- **QuantVLA (ablations)**：消融版本（如仅量化 LLM、关闭 ATM/OHB）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2）**

| Model | 方法 | Avg. Succ. Rate (%) | Memory (GB) | Relative Savings |
|-------|------|------------------------|-------------|------------------|
| OpenPI 0.5 | FP16 | 97.1% | 4.27 | 0.0% |
| + DuQuant (LLM+DiT) | W4A8 | 76.3% | 1.17 | 72.6% |
| + **QuantVLA** | **W4A8** | **97.6%** | **1.28** | **70.0%** |
| GR00T N1.5 | FP16 | 86.5% | 2.02 | 0.0% |
| + DuQuant (LLM+DiT) | W4A8 | 70.0% | 0.74 | 63.4% |
| + **QuantVLA** | **W4A8** | **88.0%** | **0.91** | **55.0%** |

> ✅ **QuantVLA 不仅将内存减少约 70%，还略微超过了 FP16 基线的成功率！**

### **与基线方法的对比结果**
- **vs. DuQuant**：尽管 DuQuant 能应用但性能暴跌（如 OpenPI 0.5 平均下降 20.8%），说明传统 PTQ 无法处理 VLA 的紧耦合特性。
- **vs. SmoothQuant（Table 5）**：
  - 在 W8A8 下表现尚可（97.0%）
  - 但在更激进的 W4A8 下未报告结果，而 QuantVLA 在 W4A8 下仍达 97.6%
  - 表明 QuantVLA 更适合低比特部署场景

### **消融实验结果（Table 1 & Figure 3）**
#### **Layer Selection 消融（Table 1）**
- 仅量化 LLM：性能轻微下降（96.5%）
- 仅量化 DiT：性能大幅下降（71.6%）
- 全量量化（LLM+DiT）：严重崩溃（76.3%）
- **LLM + DiT(MLP only)**：最佳折衷（95.4%），验证了“保护注意力投影”的必要性

#### **ATM 与 OHB 效果（Figure 3）**
- **ATM** 显著缩小了 logits 标准差与教师模型之间的差距，尤其在深层 block 更明显
- **OHB** 成功对齐了输出头 RMS，缓解了残差能量漂移
- 两者共同作用使 QuantVLA 曲线几乎贴合教师模型

#### **不同精度设置（Table 3）**
| 方法 | Avg. Success Rate |
|------|-------------------|
| FP16 | 97.1% |
| W4A8 | 97.6% ✅ |
| W4A4 | 95.3% |

> 即使在极端压缩的 W4A4 设置下，仍保持 95.3% 的高成功率，显示极强鲁棒性。

#### **不同去噪步数（Table 4）**
- 在 8 步和 16 步去噪下，QuantVLA 均达到或超过基线（88.0% vs 86.5%），表明其泛化能力强。

---

## **4. 关键结论和发现**

### **主要发现**
1. **DiT 动作头对 scale drift 极度敏感**：量化引起的 logits 温度偏移和残差能量漂移是性能崩溃的根本原因。
2. **选择性量化 + 尺度校准可破解难题**：保留 $Q/K/V/O$ 浮点 + ATM/OHB 标量校正是稳定低比特推理的关键。
3. **QuantVLA 实现“负成本增益”**：在减少 ~70% 内存的同时，**反而提升了任务成功率**，打破了“压缩必损性能”的固有认知。
4. **适用于多种 VLA 架构**：即使在非 DiT 模型（如 OpenVLA）上也能取得良好效果（Table 7），显示一定通用性。

### **方法的局限性**
- 当前 ATM 和 OHB 主要针对 **DiT 动作头接口处的跨模块漂移**，若模型内部存在更多复杂依赖（如多阶段扩散），可能需扩展校准机制。
- 所有实验基于仿真环境（LIBERO/Simulation），尚未在真实机器人上验证实时性和鲁棒性。
- W4A8 是当前最优平衡点，更低比特（如 W3A4）尚未探索。

### **未来工作方向**
- 将 QuantVLA 扩展至 **端到端训练感知量化（QAT）**，进一步压榨性能边界。
- 探索 **动态比特分配**，根据不同层敏感度自动配置 bit-width。
- 结合其他效率技术（如 MoLe-VLA 的 routing 或 VLA-Cache）构建 **复合加速管道**。
- 在 **真实机器人平台** 上部署，测试长期控制稳定性与能耗表现。

---

> **总结一句话**：  
> **QuantVLA 是首个成功实现 DiT-based VLA 模型低比特 PTQ 的框架，在无需重训练的情况下，以 ~70% 内存节省实现了超越全精度模型的任务成功率，为具身智能的轻量化部署开辟了新路径。**

</details>

---

### 3. [Scaling Vision Transformers: Evaluating DeepSpeed for Image-Centric Workloads](https://arxiv.org/abs/2602.21081)

**Authors**: Huy Trinh, Rebecca Ma, Zeqi Yu, Tahsin Reza  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2602.21081v1  

#### Abstract
Vision Transformers (ViTs) have demonstrated remarkable potential in image processing tasks by utilizing self-attention mechanisms to capture global relationships within data. However, their scalability is hindered by significant computational and memory demands, especially for large-scale models wi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Scaling Vision Transformers: Evaluating DeepSpeed for Image-Centric Workloads》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **Vision Transformers (ViTs)** 在图像任务中表现出色，但由于其自注意力机制带来的高计算复杂度和内存开销，在大规模训练时面临严重的**可扩展性瓶颈**。
- 虽然 **DeepSpeed** 已被广泛用于语言模型（如 BERT、GPT）的大规模分布式训练，但在 **image-centric workloads**（尤其是 ViTs）上的应用尚未系统研究。

> 本文首次系统性地探索将 DeepSpeed 应用于 Vision Transformers 的分布式训练，填补了该领域的空白。

### ✅ 提出的新方法/新思路
- 将 **DeepSpeed 框架适配到 Vision Transformer 架构**，构建支持 intra-node 和 inter-node 分布式训练的完整 pipeline。
- 利用 **Data Parallelism** 实现跨多 GPU 和多节点的高效训练，并结合 DeepSpeed 的优化能力（如梯度同步、通信优化）提升训练效率。
- 系统评估不同软件参数（如 `batch size`、`gradient accumulation`）对训练速度、通信开销和准确率的影响。

### ✅ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **框架适用性拓展** | 首次验证 DeepSpeed 在 ViT 类视觉任务中的可行性，为后续大规模视觉模型训练提供基础 |
| **训练效率提升潜力** | 通过合理配置 batch size 和硬件资源，显著降低单位 epoch 的训练时间 |
| **可复现性强** | 开源代码在 GitHub 可查（`trinhgiahuy/Scalable_ViT_DT`），便于社区复现与改进 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类别数 | 图像数量 | 分辨率 | 备注 |
|--------|-------|----------|--------|------|
| **CIFAR-10** | 10 | 60,000 | 32×32 | 主要用于初步验证 |
| **CIFAR-100** | 100 | 60,000 | 32×32 | 更具挑战性的分类任务 |
| **ImageNet-100\*** | 100 | ~100,000 | 224×224 | 因时间限制未完成训练 |

> \* 注：选择 ImageNet-100 是为了引入更高分辨率输入以测试极限情况。

### ⚙️ 实验设置
- **模型架构**：采用标准 **ViT_B_16** 模型
- **训练模式**：
  - **Strong Scaling**：固定总数据量，增加 GPU 数量 → 观察训练时间是否随 GPU 增加而减少
  - **Weak Scaling**：按比例扩大数据分区，使每 GPU 负载恒定 → 观察吞吐量是否线性增长
- **硬件平台**：
  - **Nebula Cluster**（intra-node）：单节点内多 GPU 测试（T4 GPUs）
  - **Vector Cluster**（intra- & inter-node）：最多扩展至 **32 nodes × 1 GPU/node**
  - **Tesla Cluster**（inter-node）：异构 GPU 环境（含 RTX 3070 / GTX 1070 / Tesla P4），用于分析非均匀硬件影响
- **软件栈**：
  - DeepSpeed + PyTorch + NCCL + OpenMPI
  - 使用 `torch.distributed.barrier()` 同步各进程

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Training Time per Epoch** | 衡量训练效率的核心指标 |
| **Speedup Ratio** | 实际加速比 vs 理想线性加速 |
| **Communication Overhead** | 同步梯度所花费的时间占比 |
| **Accuracy & Loss** | 模型收敛性和最终性能表现 |
| **Scalability Trends** | 强/弱缩放下的性能变化趋势 |

### 🔁 基线方法对比
- 本文**没有直接对比其他分布式框架**（如 Megatron-LM 或 HuggingFace Accelerate），而是以 **单 GPU 训练为基准**，衡量 DeepSpeed 下多 GPU 扩展带来的加速效果。
- 未来计划将其作为 benchmark 进行横向比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ Vector 集群（T4V2 GPUs）——良好扩展性
| GPU 数量 | Strong Scaling 平均训练时间（CIFAR-10） | Speedup（相对 1 GPU） |
|---------|----------------------------------------|------------------------|
| 1       | 1817.65 s                              | 1.00×                 |
| 2       | 941.75 s                               | ~1.93×                |
| 4       | 621.72 s                               | ~2.92×                |
| 8       | 489.64 s                               | ~3.71×                |

> ➤ 显示接近理想的强缩放趋势，尤其从 1→2 GPU 减少近一半时间。

#### ✅ Weak Scaling 结果（Vector, CIFAR-10）
- 训练时间基本保持稳定（约 170–200 秒），表明系统能有效维持负载均衡。

#### ❌ Tesla 集群（异构 GPU）——扩展失败
- 加入较弱 GPU（GTX 1070 / Tesla P4）后：
  - 小 batch size（如 16）导致频繁同步 → **通信开销剧增**
  - 第四、第五个 GPU 加入反而**延长训练时间**
  - 最终训练时间不降反升（见 Fig. 4）

> ➤ 揭示了 **GPU 异构性是分布式训练的重大瓶颈**

#### ✅ Batch Size 影响（Nebula 集群）
| Batch Size | Synchronization Cost（2-GPU） | 总体训练时间趋势 |
|------------|-------------------------------|------------------|
| 16         | 极高（主导耗时）               | 缩放差           |
| 64         | 显著下降                      | 明显改善         |
| 128        | 接近最优                      | 改善趋于平缓     |
| 256        | 再上升                        | 出现新瓶颈（CPU→GPU 数据加载） |

> ➤ **推荐 batch size = 64~128** 作为最佳折中点

#### ✅ Accuracy vs Batch Size（Fig. 7）
- 准确率先升后降：
  - batch=64 时达到峰值（~73.65%）
  - batch=256 时下降明显（~67.5%）
- 原因推测：过大 batch size 导致泛化能力下降（overfitting 或更新频率过低）

#### ✅ Gradient Accumulation 的潜在价值
- 对于内存受限 GPU，可通过 `gradient_accumulation_steps > 1` 模拟大 batch 效果，避免小 batch 带来的高频同步开销。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DeepSpeed 可成功应用于 Vision Transformers 的分布式训练**，尤其在同构硬件环境下展现出良好的 strong 和 weak scaling 特性。
2. **Batch size 是影响通信开销的关键因素**：
   - 过小 → 高频同步 → 通信成为瓶颈
   - 过大 → 内存压力 & 数据加载延迟 → 新瓶颈出现
   - **最优值在 64~128 之间**
3. **GPU 硬件一致性至关重要**：
   - 异构环境（如 Tesla 集群）严重破坏扩展性，甚至导致负加速
4. **Intra-node 与 Inter-node 性能差异不大**（Vector 集群）：
   - 表明现代集群网络（如 NFS + NCCL）已足够支撑高效的跨节点训练

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **未启用 ZeRO 优化** | 当前仅使用基础 Data Parallelism，未利用 DeepSpeed 的 ZeRO-1/2/3 来进一步节省内存 |
| **缺乏与其他框架对比** | 无法判断 DeepSpeed 是否优于 Megatron-LM 或 HuggingFace Accelerate |
| **未覆盖超大规模模型** | 实验集中在中小规模 ViT 和 CIFAR 数据集，尚未验证在 billion-parameter 级别的表现 |
| **ImageNet 训练未完成** | 缺少高分辨率图像下的实证结果 |

### 🔮 未来工作方向
1. **深入研究 ZeRO 各阶段对 ViT 的影响**：
   - 测量 memory saving 与通信 overhead 的权衡
   - 探索 ZeRO-Infinity（CPU/NVMe offloading）应对 activation 内存爆炸问题
2. **引入 Sequence Parallelism 思路处理长序列图像 patch**：
   - 借鉴 DeepSpeed-Ulysses 中的 all-to-all 通信策略
   - 结合 SparseViT、Long-Sequence-Segmentation 等技术优化 attention 计算
3. **扩展至多模态与科学成像领域**：
   - 如 fastMRI（医学影像）、CoSTAR（机器人视觉）、GTDB（基因组图像）等长序列或多通道图像任务
4. **横向 benchmarking**：
   - 对比 DeepSpeed vs Megatron-LM vs HuggingFace Accelerate 在 ViT 上的训练效率

---

> 💡 **一句话总结**：  
> 本论文证明了 **DeepSpeed 在 Vision Transformers 上具备良好的可扩展潜力**，但其性能高度依赖于 **batch size 设置** 和 **硬件同构性**；合理的软件调优可显著提升训练效率，为未来大规模视觉模型训练提供了重要实践参考。

</details>

---

### 4. [ReviveMoE: Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments](https://arxiv.org/abs/2602.21140)

**Authors**: Haley Li, Xinglu Wang, Cong Feng, Chunxu Zuo, Yanan Wang, Hei Lo, Yufei Cui, Bingji Wang, Duo Cui, Shuming Jing, Yizhou Shan, Ying Xiong, Jiannan Wang, Yong Zhang, Zhenan Fan  
**Category**: cs.DC  
**Published**: 2026-02-25  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2602.21140v1  

#### Abstract
As LLM deployments scale over more hardware, the probability of a single failure in a system increases significantly, and cloud operators must consider robust countermeasures to handle these inevitable failures. A common recovery approach is to simply restart the LLM serving instance; however, this ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ReviveMoE: Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments

---

## 1. 论文的主要贡献和创新点

### 解决的问题
随着大规模 **LLM** 部署规模扩大，硬件故障概率显著上升（如 1024 GPU 集群每 7.9 小时即发生一次故障）。传统恢复方式是重启整个服务实例，但在 **Model-as-a-Service (MaaS)** 场景中，重新加载模型权重、重建通信域和重编译计算图会引入数分钟延迟，严重影响 **SLO (Service Level Objectives)**。

此外，**MoE (Mixture of Experts)** 模型对通信高度依赖，单个硬件故障可能导致整个系统中断。现有研究多集中于训练阶段的容错或故障预防，缺乏针对推理场景的快速恢复机制。

### 提出的新方法：ReviveMoE
作者提出 **ReviveMoE**，一种专为大规模 **MoE LLM** 推理部署设计的快速故障恢复系统，集成于华为云的 **xDeepServe** 服务平台和 **XCCL** 通信库中。其核心思想是：**避免全量重启，仅隔离并修复受影响组件**。

### 创新点与优势
- ✅ **无需重启实例即可恢复服务**：通过状态迁移、通信重建和缓存化图编译实现秒级恢复。
- ✅ 支持两种部署架构：
  - **MA-collocated**（Attention 与 MoE 共置）
  - **MA-disaggregated**（Attention 与 MoE 分离）
- ✅ 多层次恢复策略：
  - **Failure Detection**：基于 Kubernetes + Ray 的心跳检测机制。
  - **Sequence State Recovery**：保留已生成 Token 并迁移至健康节点，进行部分重计算（prefill）。
  - **Block Table Recovery**：采用日志机制回滚 KV Cache 分配状态，保证一致性。
  - **Weight Integrity**：支持三种专家丢失处理策略：
    1. 使用冗余专家（redundant experts）
    2. 角色切换（role switching）：将 Attention 节点转为 MoE 节点
    3. 容忍专家丢失（missing experts），通过 gating mask 屏蔽失效专家
  - **Communication Reinitialization**：动态重建 XCCL 通信域（如 A2E/E2A）。
  - **Cached Graph Compilation**：利用预编译图缓存，将图编译时间从 **12.9 分钟降至 6 秒内**。

> 相比现有方法，ReviveMoE 更适用于客户侧推理场景，强调低延迟恢复而非检查点回滚，且不依赖硬件快速恢复。

---

## 2. 核心实验方法和设置

### 实验平台与模型
- **模型**：DeepSeek V3（671B 参数，MoE 架构）
- **硬件平台**：Huawei CloudMatrix384，使用 **80 张 Ascend 64GB NPU**
- **软件栈**：
  - CANN 8.2.1
  - Torch NPU 2.1.0
  - xDeepServe + XCCL

### 实验设置
模拟单张 NPU 故障场景，测试不同模块（Attention / MoE）失败下的恢复表现。

#### 恢复场景分类：
| 场景 | 描述 |
|------|------|
| Attention Failure | 迁移序列到其他 DPExecutor，恢复 Block Table |
| MoE Failure + Redundant Experts | 直接启用备份专家 |
| MoE Failure + Missing Experts | 允许部分专家丢失，gating mask 屏蔽 |
| MoE Failure + Role Switching | 将一个 DPExecutor 转换为 MoEExecutor，并加载权重 |

### 评估指标
- **Recovery Time**：从故障检测到服务恢复的时间（代表停机时间）
- **Model Accuracy**：在专家丢失情况下的任务性能变化（使用 Language Model Evaluation Harness）
- **Ablation Study**：分析各组件（如 role switching、cached compile）对总恢复时间的影响

### 基线方法
- **Baseline**：完整实例重启 + 缓存化图编译（cached reinitialization）
- 对比项包括是否启用 cached compilation、是否需要 weight loading 等

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Figure 5）

| 恢复场景 | 恢复时间 | 相比 Baseline 提升 |
|--------|---------|------------------|
| Baseline（完全重启） | 83.1 秒 | — |
| ReviveMoE（Attention Failure） | **10.2 秒** | ↓ **87.8%** |
| ReviveMoE（MoE Redundant） | 10.7 秒 | ↓ 87.1% |
| ReviveMoE（MoE Missing） | 10.8 秒 | ↓ 87.0% |
| ReviveMoE（MoE Role Switch） | 12.8 秒 | ↓ **84.6%**（仍快 36.6%） |

> 即使最坏情况（需 role switching + 权重加载），也比 baseline 快 **36.6%**

#### 时间构成分析（Table 1）：
- **Generator 初始化 + 权重加载** 是最大开销（~40–50 秒）
- **Graph Compile** 若无缓存可达 **12.9 分钟**，而 ReviveMoE 使用 **cached compile** 仅需 **6 秒**
- Engine 和 Executor Process 不需重启，节省约 10+ 秒

### 模型准确性评估（S4.2）

#### 实验设置：
- 在 DeepSeek V3 上模拟不同比例专家失效
- 两种模式：
  - **Task-based**：关闭每个任务中最活跃的专家（最差情况）
  - **Every nth**：均匀关闭专家（平均情况）
- 测试任务：ARC, HellaSwag, MMLU, GSM8k 等共 10 项

#### 结果（Table 2 & Figure 6）：
- 当 **≤ 1/32 的专家丢失**（即 EP ≥ 32），平均准确率下降 < **0.05**
- 在 **EP32 配置下单个 MoE NPU 失效**（即丢失 1/32 专家），性能影响可忽略
- 最大降幅出现在 GSM8k（逻辑密集任务），但仍可在容忍范围内

> 结论：**在高 EP 设置下，“missing experts” 是可行的轻量恢复选项**

### 消融实验（S4.3）：Role Switching 的必要性
- 虽然 missing experts 在 EP≥32 时有效，但在以下情况仍需 role switching：
  1. **EP < 32** → 专家丢失比例过高，精度不可接受
  2. **冗余专家本身发生最终副本丢失**（即使有冗余，冷门专家可能未被复制）

> 因此，**role switching 是保障最终一致性的兜底机制**，可与 missing experts 组合使用（先快速恢复，后台逐步补全权重）

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **重启不是唯一选择**：在 MoE 推理中，可通过局部状态恢复实现在 **10 秒级完成故障恢复**（相比 83 秒提升 87.8%）。
2. ✅ **cached graph compilation 至关重要**：图编译耗时远超权重加载，必须通过预编译缓存规避。
3. ✅ **专家冗余 + 角色切换 + 容忍丢失** 三者结合，形成灵活恢复策略组合拳。
4. ✅ **MA-disaggregated 架构同样适用**：ReviveMoE 同时支持 Attention/MoE 分离部署，适应未来 disaggregated serving 趋势。

### 方法局限性
- ❌ 仅处理 **单 NPU 故障**，不支持网络分区或多节点大规模宕机。
- ❌ 未考虑 **慢速设备（straggler）问题**：性能退化类故障无法检测和恢复。
- ❌ **Role switching 需要磁盘读取权重**，若存储带宽受限会影响恢复速度。
- ❌ 当前冗余专家主要用于负载均衡，非专为容错设计，可能导致关键专家无备份。

### 未来工作方向
- 🔮 扩展至 **多节点故障恢复**
- 🔮 引入 **straggler detection 与 preemptive migration**
- 🔮 设计 **兼顾性能与容错的冗余专家放置策略**
- 🔮 探索 **跨集群容灾恢复机制**
- 🔮 优化 **weight loading pipeline** 以进一步压缩 role switching 时间

---

> **总结**：ReviveMoE 是首个面向 **大规模 MoE LLM 推理场景** 的快速故障恢复系统，通过细粒度状态管理、通信重构与图缓存技术，在不影响用户体验的前提下实现了近实时恢复，已在华为云 MaaS 平台落地应用，具有重要工程价值。

</details>

---

### 5. [Deep unfolding of MCMC kernels: scalable, modular & explainable GANs for high-dimensional posterior sampling](https://arxiv.org/abs/2602.20758)

**Authors**: Jonathan Spence, Tob\'ias I. Liaudat, Konstantinos Zygalakis, Marcelo Pereyra  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.20758v1  

#### Abstract
Markov chain Monte Carlo (MCMC) methods are fundamental to Bayesian computation, but can be computationally intensive, especially in high-dimensional settings. Push-forward generative models, such as generative adversarial networks (GANs), variational auto-encoders and normalising flows offer a comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Deep unfolding of MCMC kernels: scalable, modular & explainable GANs for high-dimensional posterior sampling**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统的 **Markov Chain Monte Carlo (MCMC)** 方法在贝叶斯反问题中能提供可靠的后验采样和不确定性估计，但计算成本高，尤其在高维场景下效率低下。  
另一方面，**生成对抗网络 (GANs)** 等黑盒生成模型虽高效，但缺乏可解释性，且对似然函数（likelihood）变化不鲁棒，难以适应新的观测模型。

本文旨在**结合 MCMC 的可解释性与 GAN 的高效性**，提出一种既能快速采样、又具备物理一致性和适应性的新型生成模型架构。

---

### **提出的新方法与新思路**
作者提出了 **“深度展开 MCMC 核心” (deep unfolding of MCMC kernels)** 的新框架，其核心思想是：

- 将迭代式的 MCMC 算法（如 Langevin 动力学）的每一步视为一个神经网络层。
- 将固定步数的 MCMC 链（例如 $L$ 步）**展开为一个 $L$ 层的递归神经网络 (RNN)**。
- 通过端到端训练，联合优化先验参数 $\theta$ 和 MCMC 超参数（如步长 $\gamma_l$），使最终输出逼近真实后验分布 $p(x|y)$。

该方法本质上是将 MCMC 的**物理过程嵌入神经网络结构中**，从而保留了贝叶斯建模的模块化和可解释性。

---

### **相比现有方法的优势**
| 特性 | 传统 MCMC | 黑盒 GAN / Score-based Model | 本文方法 (Unfolded MCMC) |
|------|-----------|-------------------------------|--------------------------|
| **采样速度** | 慢（需数千步） | 快（单次前向传播） | 快（仅需 $L=8\sim64$ 步） |
| **可解释性** | 高（显式建模 likelihood） | 低（黑盒映射） | 高（每层对应 MCMC 更新） |
| **对 likelihood 变化的鲁棒性** | 高（直接使用 $ \nabla \log p(y|x) $） | 低（依赖训练时固定的 $A$） | **高（支持 inference-time 参数调整）** |
| **不确定性量化能力** | 强 | 弱（易模式坍塌） | 强（生成相关但多样样本） |

> ✅ **关键优势**：兼具 **scalability（可扩展性）、modularity（模块性）和 explainability（可解释性）**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **MNIST**  
   - 用于图像去模糊任务（image deblurring）
   - 输入维度较低，便于进行消融研究和精确指标比较。

2. **PROBES 数据集**  
   - 包含超过 2000 张星系图像，用于射电干涉成像（radio interferometry）任务。
   - 更具现实意义，挑战更高（高维、稀疏傅里叶采样）。

---

### **实验设置**
#### **任务设计**
- **MNIST 去模糊**：  
  观测模型为 $ y = k * x + \epsilon $，其中模糊核 $k$ 从高斯过程（GP）随机采样，模拟不同运动轨迹。
- **射电干涉成像 (RI)**：  
  观测模型为 $ y = m \mathcal{F} P x^* + \epsilon $，其中 $m$ 是可见度掩码（visibility mask），$\mathcal{F}$ 是傅里叶变换，$P$ 是天线主波束。

#### **模型实现**
- **Unfolded-Split Gibbs Sampler (U-SGS)**：用于 MNIST 实验，基于分裂吉布斯采样器展开。
- **Unfolded LATINO (U-LATINO)**：用于 RI 实验，结合 score-based denoiser 与 proximal data-fidelity 步骤。

#### **训练策略**
- 使用 **正则化 Wasserstein GAN 框架** 进行端到端训练。
- 损失函数包含三部分：
  ```math
  \mathcal{L}(\theta) = \underbrace{\mathcal{L}_{\text{adv}}}_{\text{对抗损失}} + w_1 \underbrace{\mathcal{L}_{\text{L1}}}_{\text{数据一致性}} - w_{\text{SD}} \underbrace{\mathcal{L}_{\text{SD}}}_{\text{样本多样性}}
  ```
- 引入 **Robbins-Monro 自动调参机制** 动态调整 $w_{\text{SD}}$，防止过拟合并提升稳定性。

---

### **评估指标**
| 指标 | 含义 |
|------|------|
| **PSNR** | 峰值信噪比，衡量重建精度 |
| **SSIM** | 结构相似性，感知质量 |
| **LPIPS** | 学习型感知图像块相似度，更符合人类视觉 |
| **SW (Sliced Wasserstein)** | 衡量后验分布近似质量 |
| **CFID / W-latent** | 在隐空间中的 Fréchet 距离，评估分布匹配程度 |
| **CMMD** | 基于 CLIP 嵌入的最大均值差异 |
| **NFEs (Neural Function Evaluations)** | 每样本所需网络前向次数，反映计算效率 |

---

### **基线方法对比**
| 方法 | 类型 | 是否需要训练 | 特点 |
|------|------|---------------|------|
| **VAE-SGS / IRIS / LATINO** | Zero-shot MCMC / SBM | ❌ | 不依赖训练数据，通用性强但慢 |
| **RCGAN / RIGAN** | End-to-end Conditional GAN | ✅ | 快速但黑盒，泛化差 |
| **本文方法 (U-SGS / U-LATINO)** | Unfolded MCMC | ✅ | 快速 + 可解释 + 分布建模能力强 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 & Table 2）**

#### **MNIST 去模糊任务（Table 1）**
| 方法 (NFEs) | PSNR (mean) | LPIPS ↓ | SW ↓ | W-latent ↓ | Time (ms) |
|-------------|--------------|--------|-------|------------|------------|
| U-SGS (64) | **28.46** | **0.003** | 6.95 | **0.35** | 1.89 |
| RCGAN (1) | 26.69 | 0.009 | 9.21 | 9.08 | **0.10** |
| VAE-SGS (10⁴) | 23.49 | 0.015 | 11.66 | 0.92 | 120 |

> 🔹 **结论**：U-SGS 在仅 64 步内显著优于 RCGAN 的分布建模能力（SW、W-latent 更优），同时保持良好 PSNR；尽管推理时间略长，但远快于传统 MCMC。

#### **射电干涉成像任务（Table 2）**
| 方法 (NFEs) | PSNR ↑ | LPIPS ↓ | CFID ↓ | Time (s) |
|-------------|--------|--------|--------|----------|
| U-LATINO (8) | **45.61** | **0.01** | **0.06** | 0.08 |
| RIGAN (1) | 43.77 | 0.07 | 0.52 | **0.04** |
| IRIS (1000) | 48.99 | 0.07 | 2.24 | **55.93** |
| LATINO (8) | 37.90 | 0.08 | 0.19 | 0.48 |

> 🔹 **结论**：
> - U-LATINO 以 **8 NFEs** 实现接近 IRIS (1000 NFEs) 的 PSNR，但速度快 **~700 倍**。
> - LPIPS 和 CFID 显著优于所有基线，说明其生成样本更具真实感且分布更准确。
> - 相比 RIGAN，虽多耗 2 倍时间，但在 PSNR 和不确定性建模上全面领先。

---

### **消融实验结果**
- **迭代步数 $L$ 影响**（U-SGS）：
  - 随 $L$ 增加，PSNR、LPIPS、SW 持续改善，表明更多展开层有助于逼近真实后验。
  - 推荐 $L=32\sim64$ 在性能与效率间取得平衡。

- **burn-in 设置**：
  - 设定 $L_o = L/4$ 可有效去除初始偏差。

- **感知损失加入（LPIPS）**：
  - 加入 LPIPS 正则项后，LPIPS 和 CFID 显著下降，验证其有效性（见 Table 4）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **深度展开 MCMC 架构成功融合了物理建模与深度学习的优势**：
   - 保留了 MCMC 对 likelihood 的显式建模，支持 inference-time 参数调整（如模糊核 $k$ 或噪声水平 $\sigma$）。
   - 通过端到端训练加速收敛，仅需少量迭代即可获得高质量后验样本。

2. **在多个任务上实现了“快而准”的后验采样**：
   - 在 MNIST 上，U-SGS 优于 RCGAN 的分布建模能力；
   - 在 PROBES 上，U-LATINO 以极低 NFEs 实现媲美数千步 SBM 的性能。

3. **具有出色的 out-of-distribution 泛化能力**：
   - 在测试使用未见过的模糊核或 visibility mask 时，U-SGS 和 U-LATINO 表现稳定，而 RCGAN 性能明显下降（见 Table 3 & Table 5）。

4. **不确定性估计更可靠**：
   - U-LATINO 生成的样本标准差与残差图高度相关（Pearson > 0.4），说明其能准确识别不确定区域。

---

### **方法的局限性**
- **依赖训练数据**：目前方法需监督训练，无法完全脱离 ground truth。
- **内存开销增加**：由于展开多层，参数量略高于零样本方法（如 LATINO 多 ~3×10⁵ 参数）。
- **仍为 few-shot 而非 zero-shot**：虽然比纯 MCMC 快，但仍需离线训练阶段。

---

### **未来工作方向**
1. **自监督训练**：探索无需 ground truth 的训练方式（如利用噪声建模或 consistency learning）。
2. **与其他采样器结合**：将该框架推广至 HMC、MALA 等更复杂 MCMC 核心。
3. **模型压缩与蒸馏**：进一步减少 NFEs，逼近 GAN 的实时性能。
4. **应用于其他科学成像领域**：如 MRI、CT、天文光谱重建等。

---

> ✅ **总结一句话**：  
> 本文提出的 **unfolded MCMC** 框架为高维贝叶斯反问题提供了一种**兼具速度、准确性、可解释性与适应性的新一代生成采样器**，是连接经典统计推断与现代深度生成模型的重要桥梁。

</details>

---

### 6. [WeirNet: A Large-Scale 3D CFD Benchmark for Geometric Surrogate Modeling of Piano Key Weirs](https://arxiv.org/abs/2602.20714)

**Authors**: Lisa L\"uddecke, Michael Hohmann, Sebastian Eilermann, Jan Tillmann-Mumm, Pezhman Pourabdollah, Mario Oertel, Oliver Niggemann  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.20714v1  

#### Abstract
Reliable prediction of hydraulic performance is challenging for Piano Key Weir (PKW) design because discharge capacity depends on three-dimensional geometry and operating conditions. Surrogate models can accelerate hydraulic-structure design, but progress is limited by scarce large, well-documented ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：WeirNet: A Large-Scale 3D CFD Benchmark for Geometric Surrogate Modeling of Piano Key Weirs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在水工结构设计中，**Piano Key Weir (PKW)** 的水力性能（如泄流能力）高度依赖于其复杂的三维几何形状和运行条件。传统的高保真 **CFD** 模拟虽然精确，但计算成本极高，难以支持快速的设计迭代。此外，目前缺乏公开、大规模、多模态的 PKW 数据集，严重限制了数据驱动的代理模型（surrogate model）研究和跨方法公平比较。

### 🚀 提出的新方法与创新
本研究提出了 **WeirNet** —— 一个面向 PKW 几何代理建模的大规模 3D CFD 基准数据集，主要贡献如下：

- **大规模多模态数据集构建**  
  包含 **3,794 种** 参数化生成的矩形与梯形 PKW 几何体，每种进行 **19 个流量工况** 的 CFD 模拟，共完成 **71,387 次成功模拟**，提供完整的泄流系数 $ c_p $ 标签。

- **扩展的几何参数化体系**  
  在 Pralong 等人 [56] 的基础上，引入了新的参数（如 $ R_{B.i}, R_{B.o}, T_{s.2}, T_{s.3} $），以更精确描述梯形 PKW 的侧壁倾斜与厚度变化，提升参数化建模的鲁棒性和通用性。

- **多表示形式输出**  
  每个几何体提供三种表示形式：
  - 参数向量（parametric descriptors）
  - 封闭表面网格（watertight surface meshes）
  - 高分辨率点云（high-resolution point clouds）

- **标准化任务与划分协议**  
  定义了标准回归任务（如 $ c_p $ 预测、曲线重建）、**in-distribution (ID)** 和 **out-of-distribution (OOD)** 划分（基于几何或流量外推），便于公平评估泛化能力。

- **开源可复现框架**  
  公开发布数据集（CC BY-NC 4.0）、CAD 生成脚本、CFD 设置、后处理流程及训练代码，推动可复现研究。

### 🔍 相比现有方法的优势
| 维度 | 现有工作局限 | WeirNet 改进 |
|------|----------------|-------------|
| 数据规模 | 多为小样本（<100 几何体）或二维简化 | 超过 3,700 个 3D 几何体，7 万+ CFD 模拟 |
| 几何多样性 | 固定多数参数，仅变单一比例（如 key-width ratio） | 多参数联合变化（key width, overhang, wall thickness, inclination） |
| 表示形式 | 通常仅提供参数或网格之一 | 同时提供参数、网格、点云三模态 |
| 开放性 | 多数未公开或仅发布表格数据 | 完整开放多模态数据 + 仿真流程 |
| 泛化评估 | 缺乏 OOD 协议 | 明确定义 OOD-Geom 和 OOD-Head 分割 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **WeirNet** 是唯一使用的主数据集。
- 所有几何体通过 **Rhino + Grasshopper** 参数化建模生成，采用 **拉丁超立方采样 (LHS)** 在物理可行范围内采样。
- CFD 模拟使用 **OpenFOAM v2212**，求解器为 `interFoAM`（VOF + k-ω SST 湍流模型），边界条件参考 Thorenz [68]。
- 流量范围：**50–250 L/s**，共 19 个离散值（见 Table 6）。
- 输出：上游总水头 $ H_t $、泄流系数 $ c_p $，由 Du Buat 公式计算。

### ⚙️ 实验设置
- **任务定义**：
  - **Task 1**: 给定几何参数/表示 + 流量 $ Q $，预测 $ c_p $
  - **Task 2**: 重构完整 $ H_t/P $-$ c_p $ 曲线（rating curve）
- **数据划分**：
  - **ID Split**: 80%/10%/10% 按 geometry-discharge pair 划分
  - **OOD-Geom**: 按侧壁倾角 $ \alpha $ 分割（如训练陡坡，测试平直）
  - **OOD-Head**: 按流量 $ Q $ 分割（如训练高低，测试中间）
- **评估指标**：
  - **MAE**, **MSE**, **Max AE**, **R²**
  - 推理时间（inference time / sample）
  - 模型参数量（#Params）

### 🆚 基线方法对比
| 类型 | 模型 | 输入表示 |
|------|------|----------|
| **几何深度学习模型** | PointNet | 点云（5,000 pts）+ $ Q $ |
| | RegDGCNN | 动态图点云 |
| | Mesh-GCN | 三角网格顶点坐标 + $ Q $ |
| **参数回归模型** | Random Forest | 参数向量 + $ Q $ |
| | XGBoost | 参数向量 + $ Q $ |
| | LightGBM | 参数向量 + $ Q $ |
| | Gradient Boosting | 参数向量 + $ Q $ |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Task 1，ID 测试集）

#### 表：参数模型性能对比（ID）
| Model | MSE ↓ | R² ↑ | MAE ↓ | Max AE ↓ |
|-------|--------|--------|--------|-----------|
| **Random Forest** | **5.20×10⁻⁵** | **99.67%** | **3.94×10⁻³** | 2.90×10⁻¹ |
| XGBoost | 6.80×10⁻⁵ | 99.58% | 5.53×10⁻³ | 1.44×10⁻¹ |
| LightGBM | 7.50×10⁻⁵ | 99.54% | 6.00×10⁻³ | **1.05×10⁻¹** |
| Gradient Boosting | 119.50×10⁻⁵ | 92.63% | 19.61×10⁻³ | 3.54×10⁻¹ |

> ✅ **Random Forest 表现最佳**，R² > 99.6%，误差远低于典型实验测量不确定性。

#### 表：几何模型性能对比（ID）
| Model | MSE ↓ | R² ↑ | MAE ↓ | Inference Time ↓ |
|--------|--------|--------|--------|------------------|
| **PointNet** | **15.30×10⁻⁵** | **98.96%** | **9.06×10⁻³** | **0.49 ms** |
| RegDGCNN | 97.40×10⁻⁵ | 83.11% | 22.01×10⁻³ | 53.15 ms |
| Mesh-GCN | 241.60×10⁻⁵ | 64.33% | 36.88×10⁻³ | 0.90 ms |

> ✅ **PointNet 在几何模型中表现最优**，且推理极快；Mesh-GCN 效果最差，可能因网格复杂度高导致训练困难。

### 🔁 与基线方法对比结果
- **参数模型显著优于几何模型**：Random Forest 的 MSE 比 PointNet 低近 **3 倍**，R² 更高。
- **所有代理模型推理时间均在毫秒级**，相比 CFD（数小时至数天）实现 **>10⁴ 倍加速**。
- **PointNet 是唯一具有竞争力的几何模型**，表明简单全局池化在本任务上优于局部动态图建模。

### 🔍 消融实验与关键分析

#### （1）OOD 泛化能力（Table 9）
| 场景 | 发现 |
|------|------|
| **OOD-Head**（外推流量） | 性能下降轻微，R² 仍 >90%，说明对未知流量泛化能力强 |
| **OOD-Geom**（外推几何，如 $ \alpha < 2^\circ $） | 性能急剧下降，Random Forest 的 R² 从 99.67% → **-19.43%**，MSE 增加百倍以上 |

> ❗ **几何偏移是主导失败模式**，表明当前模型对几何结构变化极为敏感。

#### （2）数据效率实验（Fig. 14）
- 当训练数据达到约 **60%** 时，R² 增长趋于饱和。
- 超过 80% 后提升微弱，表明 **WeirNet 已接近“收益递减”阶段**。
- Gradient Boosting 在更多数据下反而性能下降，显示其容量有限。

#### （3）特征重要性分析（SHAP）
- 正向影响 $ c_p $：$ \alpha $（倾角）、$ W_{o.a} $（出口下游宽度）
- 负向影响 $ c_p $：$ Q $、$ T_{s.3} $、$ B_i $
- 与流体力学直觉一致，验证模型可解释性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **高质量参数化 + 树模型 = 最佳实践**  
   基于领域知识设计的紧凑参数向量，配合 Random Forest/XGBoost，可在精度、速度、可解释性上全面超越复杂的几何深度学习模型。

2. **几何代理模型具备实用价值**  
   即使是最简单的 PointNet，也能达到足够工程可用的精度（R² > 98%），适用于无参数化信息时的 fallback 方案。

3. **几何外推是最大挑战**  
   模型在训练分布内的插值表现优异，但在新几何形态（如从梯形到矩形）上严重失效，凸显 **OOD 泛化仍是瓶颈**。

4. **数据效率存在上限**  
   约 60% 数据即可获得接近饱和的性能，暗示未来可通过主动学习或合成采样优化数据利用率。

### ⚠️ 方法的局限性
- **设计空间受限**：仅涵盖 Type A PKW，固定全局尺寸，排除带导墙（parapet wall）等变体。
- **单一流体模型**：全部使用 RANS k-ω 模型，未考虑 LES 或实验验证偏差。
- **目标单一**：仅预测标量 $ c_p $，未提供全场流动信息（velocity, pressure）或不确定性估计。
- **尺度效应**：基于实验室尺度模拟，可能存在雷诺数缩放问题。

### 🔮 未来工作方向
1. **扩展设计空间**：纳入其他 PKW 类型（B/C/D）、变高宽比、非对称结构。
2. **多保真融合**：结合低精度模拟、高精度 CFD 与真实实验数据，构建 multi-fidelity surrogate。
3. **逆向设计与生成模型**：利用 WeirNet 训练 conditional VAE/GAN，实现“给定性能需求 → 自动生成几何”。
4. **不确定性量化**：开发贝叶斯代理模型，在 OOD 区域输出置信度，辅助安全决策。
5. **点云生成与反演**：探索 diffusion models 或 flow-based models 用于几何生成。
6. **开放原始 CFD 数据**：释放 70TB 原始仿真数据，支持更高分辨率重算或多任务学习。

---

> 💡 **总体评价**：WeirNet 不仅是一个数据集，更是一个推动 **AI for Hydraulic Engineering** 可复现研究的基础设施。它揭示了“**domain-informed parameterization + lightweight ML**”在工程代理建模中的强大潜力，同时为几何深度学习提供了真实而具挑战性的测试平台。

</details>

---

### 7. [CHESS: Context-aware Hierarchical Efficient Semantic Selection for Long-Context LLM Inference](https://arxiv.org/abs/2602.20732)

**Authors**: Chao Fei, Guozhong Li, Chenxi Liu, Panos Kalnis  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.20732v1  

#### Abstract
Long-context LLMs demand accurate inference at low latency, yet decoding becomes primarily constrained by KV cache as context grows. Prior pruning methods are largely context-agnostic: their token selection ignores step-wise relevance and local semantics, which undermines quality. Moreover, their ir...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《CHESS: Context-aware Hierarchical Efficient Semantic Selection for Long-Context LLM Inference》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
长上下文 LLM 推理中，随着上下文长度增长，**KV Cache** 成为推理延迟的主要瓶颈。传统方法在解码时需从显存读取 $O(L)$ 的 KV 数据，导致内存带宽受限、延迟随长度线性上升。

现有 KV 缓存压缩方法存在以下问题：
- **Context-agnostic selection**：基于全局注意力分数选择重要 token，忽略当前生成步骤的语义相关性。
- **细粒度 token 级操作**：引发昂贵的数据移动和不规则内存访问，难以与现代硬件优化（如 PagedAttention）集成。
- **系统开销高**：选择逻辑本身带来显著延迟，削弱了稀疏性带来的理论加速。

---

### 提出的新方法：CHESS
CHESS 是一个 **算法-系统协同设计** 的 KV Cache 管理框架，核心思想是：

#### （1）Context-aware 动态语义选择
- 不再静态保留“高频”token，而是每一步动态识别与当前查询语义相关的上下文块。
- 利用 **Key-Key Semantic Affinity**（查询 Key 与历史 Key 向量的点积相似度）作为选择依据，确保检索内容与当前生成意图一致。

#### （2）Hierarchical Page-aligned Selection
- 将 KV Cache 组织为三级结构：**Page → Chunk → Grid**，实现粗到细的选择流程。
- 所有操作以 **Page 为单位**（如 32 tokens/page），避免 token 级别的数据搬移，支持零拷贝（zero-copy）逻辑索引操作。

#### （3）Quality-aware Backtracking 自适应回溯机制
- 监控生成过程中的不确定性指标（**Entropy** 和 **Varentropy**）。
- 当检测到生成不稳定时，触发完整上下文重建，防止因错误剪枝导致质量崩溃。

#### （4）系统级高效实现
- 使用 **Tensor Coalescing + 单次 GEMM** 实现跨层级批量相似度计算。
- 集成 **CUDA Graphs** 和 **FlashInfer** 内核，最大化 GPU 利用率。
- 通过操纵逻辑页索引而非物理数据移动，实现真正的零拷贝选择。

---

### 相比现有方法的优势
| 维度 | CHESS | 其他方法（H2O, SnapKV 等） |
|------|-------|--------------------------|
| **选择策略** | Context-aware，动态适配当前生成 | Context-agnostic，依赖全局统计 |
| **处理粒度** | Page-level，对齐 PagedAttention | Token-level，破坏内存连续性 |
| **系统效率** | 零拷贝 + 批量 GEMM，开销极低 | 数据搬移频繁，调度复杂 |
| **鲁棒性** | 支持自纠错（backtracking） | 一旦误删不可恢复 |

---

## 2. 核心实验方法和设置

### 数据集
- **LongBenchV2**：用于评估生成质量，涵盖多种长文本任务（QA、对话、代码、多跳推理等），按难度和长度分组。
- **合成数据集**：用于吞吐量和延迟测试，控制输入长度（4k–32k tokens）和 batch size（1–192）。

---

### 实验设置
- **硬件环境**：单节点 4× H20 GPU / A800 GPU，CUDA 12.4，PyTorch 2.5.1。
- **模型**：基于 Qwen3-8B（bfloat16）进行模拟，KV Cache 占用随长度线性增长。
- **Page 大小**：统一设为 32 tokens。
- **Retention Ratios**：Grid ($p_g$), Chunk ($p_c$), Page ($p_p$) 可调，例如 Aggressive 设置为 (0.5, 0.2, 0.1)。

---

### 评估指标
| 类别 | 指标 |
|------|------|
| **生成质量** | LongBenchV2 上的 overall / easy / hard / short/medium/long 分类得分 |
| **系统性能** | End-to-end throughput（tokens/s）、TPOT（Time Per Output Token）、Latency 增长趋势 |
| **开销分析** | Selection 模块的延迟占比、OOM 情况 |

---

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Full-KV** | 全量缓存 | 性能基准，无压缩 |
| **H2O** | Token-level, Attn-score-based | 保留 attention “heavy hitters” |
| **KeyDiff** | Token-level, Key-similarity-based | 基于 key 分布差异筛选 |
| **SnapKV** | Token-level, Cluster-based | 聚类注意力模式保留关键段落 |
| **Quest** | Block-level, Query-aware | 支持动态重构，但仅限单 batch |

> 注：除 Quest 外，其余基线均集成至同一评测栈；Quest 使用其官方实现。

---

## 3. 主要实验结果和性能指标

### 生成质量（LongBenchV2）
| 方法 | KV Cache Budget | Overall Score |
|------|------------------|---------------|
| Full-KV | 100% | 30.2 |
| H2O (best) | 20% | 34.0 |
| SnapKV (best) | ~10% (~4k tokens) | 30.2 |
| **CHESS (Aggressive)** | **1%** | **33.2** ✅ |

- **关键发现**：即使只保留 **1% 的 KV Cache**，CHESS 仍优于 Full-KV，并接近使用 20× 更多缓存的 H2O。
- 在 **Single-QA、Multi-QA、L-ICL、Structured Tasks** 上表现尤为突出，因其受益于连贯的语义上下文。
- 图表显示，CHESS 在 **Long (>128K)** 输入上优势更明显，说明其对长程依赖建模更强。

> 📊 表格见原文 Table 1 & E.3，CHESS 在 1% 预算下达到最高分，甚至超过 Full-KV。

---

### 系统性能（Throughput & Latency）

#### 吞吐量提升（vs Full-KV）
- **峰值吞吐加速达 `4.56×`**（32k context, large batch）。
- 随着 batch size 和 context length 增加，CHESS 的优势持续扩大。
- 其他方法在大 batch 下出现 OOM 或收益递减，而 CHESS 保持稳定扩展。

> 📈 见 Figure 6 & D.10，CHESS 曲线始终位于顶部。

#### 解码延迟稳定性
- **TPOT 几乎恒定**，不受生成步数增加影响（Figure 7b）。
- 对比之下，Full-KV 和 SnapKV 的 per-token latency 随序列增长呈线性上升。
- CHESS 成功将 **单 token 推理延迟与上下文长度解耦**。

#### 选择模块开销
- Selection 开销占总延迟比例极低：
  - 8k context: **0.72%**
  - 16k context: **0.98%**
  - 32k context: **1.49%**
- 得益于 **amortized execution + GEMM 优化 + CUDA Graphs**。

---

### 消融实验与关键发现（Appendix C & Figure 9）
研究不同缓存预算下的性能变化趋势，揭示三个阶段：
1. **Distraction Phase**（中等预算）：引入低秩 token 但缺乏上下文 → 注意力被干扰 → 质量下降。
2. **Contextual Recovery Phase**（足够预算）：恢复局部语境 → 质量回升。
3. **Attention Dilution Phase**（过高预算）：注意力过度分散 → 性能趋近 Full-KV。

> 💡 **洞见**：最优并非“越多越好”，而是“最相关”。CHESS 在 1% 极端压缩下仍胜出，说明它有效过滤了冗余噪声。

---

## 4. 关键结论和发现

### 主要发现
1. **Token 重要性是动态且上下文相关的**，静态或全局重要性指标不足以支撑高质量长上下文推理。
2. **Page-level + zero-copy selection** 是实现实际加速的关键，解决了算法稀疏性无法转化为 wall-clock speedup 的系统鸿沟。
3. **Context-aware selection + adaptive backtracking** 显著提升了压缩下的鲁棒性，避免不可逆的信息丢失。
4. **适度压缩可能优于全量缓存**：去除无关上下文反而有助于模型聚焦，提升推理准确性（anti-distraction effect）。

---

### 方法局限性
- **依赖 Key 向量语义完整性**：若 Key 层表达能力弱，会影响选择精度。
- **超参数敏感性**：虽然整体鲁棒，但在极端设置下需仔细调参（如 $p_g, p_c, p_p$）。
- **目前未结合 Speculative Decoding 或 RAG**：未来可进一步增强端到端效率。

---

### 未来工作方向
- 结合 **Speculative Decoding** 实现更快预填充与草稿生成。
- 集成至 **RAG-augmented pipelines**，联合优化外部知识检索与内部 KV 管理。
- 探索更丰富的 **uncertainty signals**（如 gradient norm, attention entropy variance）用于动态控制。
- 扩展至 **multi-modal LLMs** 中的 cross-modal context selection。

---

## 总结
✅ **CHESS 成功实现了“高质量 + 高效率 + 高稳定”的长上下文 LLM 推理新范式**。  
它不仅在 **仅用 1% KV Cache 的情况下超越 Full-KV 的生成质量**，还实现了高达 **4.56× 的吞吐提升**，并保持平坦的延迟曲线。其成功源于深刻的 **algorithm-system co-design** 思想：既提出 context-aware 的语义选择机制，又通过 page-aligned、zero-copy、batched GEMM 等技术将其高效落地，真正打通了从理论稀疏性到实际性能增益的“最后一公里”。

</details>

---

### 8. [PyVision-RL: Forging Open Agentic Vision Models via RL](https://arxiv.org/abs/2602.20739)

**Authors**: Shitian Zhao, Shaoheng Lin, Ming Li, Haoquan Zhang, Wenshuo Peng, Kaipeng Zhang, Chen Wei  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.20739v1  

#### Abstract
Reinforcement learning for agentic multimodal models often suffers from interaction collapse, where models learn to reduce tool usage and multi-turn reasoning, limiting the benefits of agentic behavior. We introduce PyVision-RL, a reinforcement learning framework for open-weight multimodal models th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PyVision-RL: Forging Open Agentic Vision Models via RL

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **交互崩溃（interaction collapse）**：在基于强化学习（RL）训练的 agentic 多模态模型中，模型倾向于减少工具调用和多轮推理，导致“短视”行为，限制了 agentic 能力的发挥。
- **视频理解中的 token 效率低下**：传统多模态大模型（MLLMs）通常采用均匀帧采样处理视频，造成大量冗余视觉 token，效率低且不利于长时序推理。

### 🛠️ 提出的新方法与创新
1. **PyVision-RL 框架**  
   一个面向开源权重（open-weight）多模态模型的统一 agentic 强化学习框架，支持图像与视频理解任务。

2. **动态工具调用（Dynamic Tooling）**  
   将 Python 作为基础工具（primitive tool），允许模型在推理过程中动态生成代码以执行任务特定操作（如裁剪、绘图、统计分析等），相比静态工具集更具灵活性。

3. **防止交互崩溃的两大机制**：
   - **累积工具奖励（accumulative tool reward）**  
     在最终奖励中加入与工具调用次数成正比的项（`0.1 * ntc`），仅当答案正确时生效，从而鼓励持续使用工具而不惩罚无效尝试。
   - **过采样-过滤-排序 Rollout 策略（oversampling-filtering-ranking rollout strategy）**  
     通过标准差排序选择具有适度难度和高奖励方差的 rollout 组，提升训练信号质量，避免零优势问题。

4. **按需上下文构建（on-demand context construction）——专为视频设计**  
   视频不直接注入 MLLM 上下文，而是加载到 Python 运行环境中，由模型通过代码按需采样并绘制关键帧。显著降低视觉 token 消耗。

---

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | PyVision-RL |
|------|--------|------------|
| 工具范式 | 静态工具集（如 crop/zoom） | 动态 Python 工具，灵活组合 |
| 训练稳定性 | 易出现交互崩溃 | 引入累积奖励 + rollout 排序，稳定多轮交互 |
| 视频处理效率 | 均匀帧采样 → 高 token 开销 | 按需采样 → token 减少约 90% |
| 模型开放性 | 多依赖闭源 API 或系统 | 支持 open-weight 模型（基于 Qwen2.5-VL-7B） |

---

## 2. 核心实验方法和设置

### 📚 数据集

#### 图像任务（PyVision-Image）
- **Visual Search**: `V*`, `HRBench-4K`, `HRBench-8K`
- **Multimodal Reasoning**: `MathVerse`, `MathVision`, `WeMath`, `DynaMath`
- **Agentic Reasoning**: `TIR-Bench`

#### 视频任务（PyVision-Video）
- **Spatial Reasoning**: `VSI-Bench`（含多个子任务：对象计数、距离估计、路线规划等）
- SFT 数据还使用了 `SpaceR`, `LongVILA`

### ⚙️ 实验设置
- **基础模型**：`Qwen2.5-VL-7B`
- **训练流程**：
  1. 先进行 SFT（Supervised Fine-Tuning），使用合成数据（GPT-4.1 生成）
  2. 再进行 RL 微调，共 700 步
- **RL 参数**：
  - Batch size: 16（训练） / 32（过采样）
  - Group size: 8
  - Learning rate: 1e-6
  - 使用 GRPO 算法，移除标准差归一化
- **最大回合数（max turn budget）**：默认设为 4

### 🎯 评估指标
- 主要指标：准确率（Accuracy / Pass@1）
- Token 效率：平均每样本视觉 token 数量
- 辅助指标：响应长度、工具调用次数、消融实验性能变化

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **静态工具集** | Pixel-Reasoner, Mini-o3, DeepEyes, VITAL |
| **动态工具调用** | Thyme, CodeV, CodeDance, CodeVision, DeepEyes-v2 |
| **纯文本推理** | Video-R1, Qwen2.5-VL-7B（无工具） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

#### PyVision-Image 性能（vs. Qwen2.5-VL-7B）
| Benchmark | PyVision-Image | 提升幅度 |
|---------|----------------|----------|
| V* (avg@32) | **88.7** | +10.2% |
| HRBench-4K | **78.1** | +6.5% |
| HRBench-8K | **74.3** | +6.4% |
| DynaMath | **61.6** | +8.3% |
| MathVerse | **55.8** | +10.2% |
| WeMath | **47.7** | +9.6% |
| TIR-Bench | **19.8** | +3.8% |

> 💡 在所有类别上达到 SOTA，尤其在数学与 agentic 推理上有显著提升。

#### PyVision-Video 性能（VSI-Bench）
| 方法 | 平均得分 | 视觉 token / 样本 |
|------|--------|------------------|
| Qwen2.5-VL-7B | 36.7 | ~45K |
| VITAL | 41.8 | — |
| **PyVision-Video** | **44.0** | **~5K** |

> ✅ **绝对提升 +7.3%**，同时 token 消耗仅为基线的 **1/9**，实现更优的 accuracy-efficiency trade-off。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）累积工具奖励（Accumulative Tool Reward）
- 移除后，早期性能略好，但后期停滞。
- 工具调用次数迅速下降 → 表明该奖励对维持长期交互至关重要。
- 在 step 600 时，有奖励比无奖励高出约 **+1.9%**。

#### （2）最大回合预算（Max Turn Budget）
- 设为 4 vs 2，在训练后期带来明显增益（step 600 时 +1.93%）。
- 表明更大的推理深度可解锁更高性能上限。

#### （3）标准差排序（Standard Deviation Sorting）
- 显著减少“正确但优势为负”的样本比例（见 Fig. 6）。
- 提升训练稳定性，尤其在早期阶段。

#### （4）移除标准差归一化（Remove Std Normalization）
- 保留归一化会导致训练波动加剧。
- 移除后梯度更稳定，收敛更好。

> ✅ 所有组件共同作用，实现了稳定高效的 agentic RL 训练。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **交互崩溃可通过激励机制缓解**  
   交互行为的退化并非本质缺陷，而是训练目标与 rollout 选择不当所致。引入 **accumulative tool reward** 可有效维持多轮工具使用。

2. **动态工具调用优于静态工具集**  
   Python 作为通用工具接口，赋予模型更强的表达能力与适应性，在复杂推理任务中表现更优。

3. **按需上下文构建大幅提升视频效率**  
   PyVision-Video 通过 on-demand frame fetching，将平均 token 从 45K 降至 5K，同时提升准确率，验证了“智能采样 > 均匀采样”。

4. **rollout 质量控制是 agentic RL 成功的关键**  
   过采样 + 过滤 + 排序策略提升了训练信号的信息量，避免无效或破坏性轨迹干扰优化过程。

---

### ⚠️ 局限性
- **安全性风险**：由于使用 Python 执行环境，模型可能访问主机文件系统，存在潜在安全漏洞，部署需谨慎隔离。
- **依赖高质量 SFT 初始化**：SFT 数据全部由 GPT-4.1 合成，若合成质量不高，可能影响后续 RL 效果。
- **当前仅支持单任务独立训练**：尚未探索跨任务泛化或多任务联合优化。

---

### 🔮 未来工作方向
- 构建更安全的沙箱执行环境，防止恶意代码执行。
- 扩展至更多模态（如音频、3D 场景）与工具（如浏览器控制、API 调用）。
- 探索 test-time scaling 下的 agentic 推理能力边界。
- 将 on-demand context construction 应用于其他长序列输入（如文档、语音流）。

---

## 📌 总结一句话
> **PyVision-RL 通过动态 Python 工具 + 累积奖励 + 按需上下文机制，成功解决了 agentic 多模态模型中的交互崩溃与 token 效率难题，在图像与视频理解任务上实现了高性能与高效率的统一。**

</details>

---

### 9. [GauS: Differentiable Scheduling Optimization via Gaussian Reparameterization](https://arxiv.org/abs/2602.20427)

**Authors**: Yaohui Cai, Vesal Bakhtazad, Cunxi Yu, Zhiru Zhang  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.20427v1  

#### Abstract
Efficient operator scheduling is a fundamental challenge in software compilation and hardware synthesis. While recent differentiable approaches have sought to replace traditional ones like exact solvers or heuristics with gradient-based search, they typically rely on categorical distributions that f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GauS: Differentiable Scheduling Optimization via Gaussian Reparameterization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **operator scheduling** 是一个 NP-hard 问题，广泛存在于软件编译和硬件综合（如 HLS）中。现有方法存在以下瓶颈：
- **Exact solvers**（如 ILP、SMT）虽然能提供最优解，但计算复杂度高，在大规模图上难以收敛。
- **Heuristic 方法**（如 List Scheduling、FDS）速度快但缺乏全局优化能力，常导致次优解。
- 最近的 **differentiable combinatorial optimization** 方法（如 GS-Schedule）通过 Gumbel-Softmax 进行离散松弛，但仍存在两个关键缺陷：
  - 使用 **categorical distribution** 忽略了时间步长之间的**序数关系**（ordinal nature），即相邻时间步应具有更强相关性；
  - 参数空间为 $O(D \cdot |V|)$，随图深度 $D$ 和节点数 $|V|$ 线性增长，导致内存爆炸。

### 🚀 提出的新方法：GauS
本文提出 **GauS**（Gaussian Reparameterization Scheduling），一种基于 **Gaussian 分布建模** 的可微调度框架。

#### 核心思想
将每个操作符 $v_i$ 的执行时间建模为一个连续随机变量：
$$
X_i \sim \mathcal{N}(\mu_i, \sigma_i)
$$
其中：
- $\mu_i$ 表示期望的调度时间步；
- $\sigma_i$ 控制不确定性（训练初期较大以探索，后期收缩以“硬化”为确定值）。

通过 **Gaussian Reparameterization** 将原离散组合优化问题转化为对 $(\mu, \sigma)$ 的连续优化问题，利用梯度下降求解。

### 🔍 创新优势
| 方面 | 优势说明 |
|------|---------|
| **参数效率** | 参数量从 $O(D \cdot |V|)$ 降至 $O(|V|)$，仅需学习每个节点的均值和标准差，极大提升可扩展性。 |
| **序数感知（Temporal Inductive Bias）** | Gaussian PDF 天然具备平滑性和局部连续性，使得 $\mu$ 更新时能获得跨邻近时间步的清晰梯度信号，避免 categorical 方法中的“梯度消失”。 |
| **灵活性强** | 支持多种目标函数（资源使用、memory footprint、communication overhead）和约束（dependency, latency, recurrence），首次实现了 **pipelined scheduling** 的可微形式化。 |
| **GPU 友好** | 完全向量化实现，充分利用现代 GPU 并行能力，平均 GPU 利用率接近 100%。 |

---

## 2. 核心实验方法和设置

### 📊 数据集
由于缺乏公开的大规模调度数据集，作者采用并扩展了已有基准：

| 类型 | 名称 | 描述 |
|------|------|------|
| **现实工作负载** | EPFL Benchmark Suite | 包括 `ctrl`, `router`, `bar`, `div` 等真实电路设计 DAG，节点数从 182 到 57,375 不等。 |
| **合成随机图** | Random Workloads (RW) | 随机生成的 DAG，用于测试可扩展性，最大达 9,447 节点。 |
| **增强版用于流水线** | Augmented EPFL/RW | 在原始图基础上添加 **recurrence constraints**（模拟循环依赖），用于 modulo scheduling 测试。 |

> 图表显示最大图深度达 4373 步（div），验证了方法在极端情况下的可行性。

### ⚙️ 实验设置
- **时间限制**：所有方法统一设定 **15 分钟超时**。
- **硬件环境**：NVIDIA A100 GPU（80GB）、PyTorch + CUDA。
- **优化器**：Adam（lr=1e-2），使用 **Augmented Lagrangian Method (ALM)** 动态调整约束权重，无需手动调参。
- **初始化策略**：$\mu$ 初始化为 ASAP 与 ALAP 中点；$\sigma = K \cdot (s_{ALAP} - s_{ASAP})$，鼓励初始探索。

### 🎯 评估指标
定义三种典型 scheduling formulation：

| Formulation | 目标函数 | 约束条件 |
|------------|----------|--------|
| **A**: Latency-constrained, resource & communication opt. | $\min L_{Res} + \alpha L_{com}$ | Dep, Lat |
| **B**: Latency-constrained, memory footprint opt. | $\min L_{Mem}$ | Dep, Lat |
| **C**: Modulo scheduling (pipelined) | $\min L_{MMem}$ | MRes ≤ Cres, Dep, Lat, Rec |

> 性能以 **相对质量比**（GauS 为 1.0）表示，越低越好。

### 🆚 基线方法对比
| 类别 | 方法 | 说明 |
|------|------|------|
| **Exact Solvers** | CPLEX, Gurobi | 商业级 ILP/SMT 求解器，代表最优性上限 |
| **Heuristics** | List Scheduling, FDS | 经典快速启发式算法 |
| **Differentiable Baseline** | GS-Schedule | 当前唯一可微调度方法，基于 categorical + Gumbel-Softmax |

> 注意：GS-Schedule 仅在 Formulation A 上可比较，因其未支持 memory 或 modulo scheduling。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **参数压缩比** | 从 $O(D \cdot |V|)$ → $O(|V|)$，减少 **1–2 个数量级** |
| **内存占用** | GauS 内存增长近乎线性；GS-Schedule 在大图（如 `div`）直接 OOM |
| **GPU 利用率** | GauS 平均 >95%，GS-Schedule <40%（复杂图下） |
| **收敛速度** | 达到同等质量快 **1–2 个数量级** |
| **解质量** | 几何平均优于 GS-Schedule **71.8%**，优于 heuristics 20%-60% |

### 🔬 详细对比结果

#### ▶️ Formulation A（常规调度，通信敏感）
- 如图 3 所示：
  - GauS 在所有可解实例上均达到或接近 Pareto-optimal。
  - GS-Schedule 解质量差约 30%~300%，且在多个大图上因 **CUDA out-of-memory (OOM)** 失败。
  - CPLEX/Gurobi 在小图表现良好，但在大图无法在 15 分钟内返回可行解（标记为 "inf"）。

> **结论**：GauS 在质量与速度之间取得最佳平衡，尤其适合大规模场景。

#### ▶️ Formulation B & C（内存/流水线优化）
- 如图 5、6 所示：
  - GauS 显著优于 List/FDS（memory footprint 高出 20%-60%）；
  - CPLEX/Gurobi 仍能在中小图找到高质量解，但随着规模上升迅速失效；
  - 在 modulo scheduling（Formulation C）中，GauS 是**首个支持该任务的可微方法**，展示了独特灵活性。

#### ▶️ Trade-off 分析（图 7、附录 E）
- GauS 在极短时间内（几十秒）即可逼近高质量解；
- 即使最终目标相近，GauS 收敛速度快 **10–100 倍**；
- 在 RW_5 等案例中，虽最终略逊于 Gurobi，但其解已足够实用且速度快得多。

#### ❌ 消融实验（文中未明确列出表格，但从分析可推断）
- **Gaussian vs Categorical**：消除了“ordinal blindness”，显著改善梯度传播；
- **ALM vs 固定加权**：动态惩罚机制有效引导约束满足，避免人工调参；
- **Legalization Heuristic**：轻微修正非法调度，不影响整体优化路径。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Gaussian reparameterization 是 operator scheduling 的高效可微建模范式**：
   - 成功捕捉时间的序数特性；
   - 极大降低参数维度，实现 $O(|V|)$ 可扩展性。
2. **GauS 实现了 Pareto-optimal 权衡**：
   - 在解质量、运行时间和内存消耗之间全面领先；
   - 特别适用于 **ten-thousands-node 级别的现代硬件设计**。
3. **首次实现可微的 modulo scheduling**：
   - 支持 recurrence constraint 和周期性资源建模；
   - 为 AI-driven HLS 工具链开辟新路径。

### ⚠️ 局限性
- **独立 Gaussian 假设忽略节点间相关性**：
  - 当前模型假设各操作符独立，未建模共享边或资源竞争带来的联合分布效应；
  - 虽可通过 Gaussian Process 建模相关性，但会引入 $O(|V|^3)$ 计算开销，不可行。
- **偶尔收敛至局部最优**：
  - 尽管 ALM 和初始化策略缓解了此问题，但在高度耦合的约束下仍可能发生。

### 🔮 未来工作方向
1. 探索更高效的 **node correlation modeling** 方法（如低秩近似、attention-based covariance）；
2. 引入 **advanced optimization strategies**（如 curriculum learning、momentum-based exploration）提升稳定性；
3. 扩展至 **multi-objective scheduling** 与 **learning-based cost models** 融合；
4. 应用于真实工业流程，如 **AI 编译器**（TensorRT, TVM）或 **PIM 架构调度**。

---

## 💡 总结一句话
> **GauS 通过 Gaussian Reparameterization 实现了高效、灵活、可扩展的可微调度框架，在大规模 operator scheduling 任务中实现了质量与速度的 Pareto 最优，是迈向自动化高性能系统设计的重要一步。**

</details>

---

### 10. [Physics-based phenomenological characterization of cross-modal bias in multimodal models](https://arxiv.org/abs/2602.20624)

**Authors**: Hyeongmo Kim, Sohyun Kang, Yerin Choi, Seungyeon Ji, Junhyuk Woo, Hyunsuk Chung, Soyeon Caren Han, Kyungreem Han  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.20624v1  

#### Abstract
The term 'algorithmic fairness' is used to evaluate whether AI models operate fairly in both comparative (where fairness is understood as formal equality, such as "treat like cases as like") and non-comparative (where unfairness arises from the model's inaccuracy, arbitrariness, or inscrutability) c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Physics-based phenomenological characterization of cross-modal bias in multimodal models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**多模态大语言模型（MLLMs）中的跨模态偏差（cross-modal bias）问题**，即在融合多种输入模态（如文本、图像、音频）时，模型往往过度依赖某一主导模态（如文本），而其他模态的信息被忽略甚至干扰决策过程。这种偏差不仅导致性能下降，还引发算法公平性（algorithmic fairness）问题，尤其是在非比较性语境下（non-comparative unfairness），表现为任意性（arbitrariness）和不可解释性（inscrutability）。

传统基于嵌入空间或表征层面的分析方法难以捕捉此类系统性偏差的动态机制。

### 🚀 提出的新方法与新思路
1. **提出一种基于物理的替代模型（surrogate physics-based model）**  
   构建了一个**multi-oscillator dynamical model**，将Transformer中的self-attention和cross-attention机制映射为耦合振荡器系统的动力学行为，从而从**物理现象学（phenomenological）角度**刻画跨模态交互的动力学特性。

2. **引入“现象学可解释性”框架**  
   摒弃传统的认知主义符号解释范式（cognitivist symbolic account），转而关注机器在训练/推理过程中所经历的“内部物理实体”的演化过程，强调对系统内在动态的经验描述而非对外部现实的表征假设。

3. **结合实证诊断与理论建模的双重路径**  
   - 在真实MLLM上进行**标签扰动实验（label perturbation）**，揭示错误吸引子结构；
   - 利用物理代理模型模拟**Lorenz混沌时间序列预测任务**，量化不同注意力强度下的模态偏好。

### ⚖️ 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 分析层次 | 静态embedding/representation level | 动态dynamics-level建模 |
| 可解释性视角 | 符号主义、表征主义（cognitivism） | 现象学、物理动力学（physics-based phenomenology） |
| 偏差检测能力 | 依赖aggregate accuracy等宏观指标 | 揭示失败状态下的隐含偏好层级（error-attractor hierarchies） |
| 模型通用性 | 特定架构依赖性强 | 可推广至不同MLLM架构 |

> ✅ **优势总结**：能够揭示传统评估无法发现的**结构性偏见动态机制**，提供更深层、更具因果性的理解路径。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **CREMA-D**（Crowdsourced Emotional Multimodal Actors Dataset）
  - 包含7,442个视频样本，来自91位演员；
  - 每个样本包含面部表情（video/face）和语音情感（audio/voice）；
  - 情感类别：`happy`, `neutral`, `sad`, `angry`, `disgust`, `fear`；
  - 提供三种标注：意图情绪（intended）、视觉感知情绪（visual-perceived）、听觉感知情绪（audio-perceived）。

### 🔬 实验设置
#### （1）零样本情感分类任务（Zero-shot Emotion Classification）
- **模型**：
  - `Qwen2.5-Omni`
  - `Gemma 3n`
- **输入条件对比**：
  1. **Video + Audio**（完整多模态）
  2. **Video-only**（音频替换为静音）
  3. **Audio-only**（视频帧替换为空白占位符）
- **提示模板统一**（见Table 1）：
  ```json
  [{"emotion": "<predicted_emotion>"}]
  ```
  输出严格限制为JSON格式，禁止额外说明。

#### （2）标签扰动策略（Prompt-based Label Perturbation）
- 系统性地从候选集中移除部分emotion标签（每次移除1–4类）；
- 观察模型在受限输出空间中的fallback行为；
- 目标：识别**错误吸引子层级结构**（error-attractor hierarchy）。

#### （3）物理代理模型实验：Lorenz混沌时间序列预测
- **任务设计**：
  - 子系统X受Lorenz系统的$x(t)$驱动；
  - 子系统Y受$y(t)$驱动；
  - 目标是通过振荡器相位观测值$\{\sin\theta_i^{(o)}\}$线性回归预测$z(t)$。
- **评价指标**：
  - **NMSE**（Normalized Mean Squared Error）：衡量预测精度；
  - **Dynamical SHAP Value**：量化各模态对预测的贡献度；
    $$
    \phi_o = \frac{1}{2}[(NMSE_{\neg o} - NMSE_\emptyset) - (NMSE_{all} - NMSE_{\neg o})]
    $$
    其中$\phi(Y) - \phi(X)$反映模态偏好方向。

- **参数配置**：
  - $N_x = N_y = 50$ 个振荡器；
  - 连接结构采用Watts-Strogatz小世界网络（rewiring prob. $p=0.01$, degree $k=10$）；
  - 注意力强度$(B_{\text{self}}, B_{\text{cross}})$作为控制变量扫描。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据与发现

#### （1）情感分类中的层级偏差（Hierarchical Bias）
- **Qwen2.5-Omni**：
  - 在所有输入条件下，`neutral` 是最强的错误吸引子；
  - 当`neutral`被排除后，模型主要fallback到`happy`；
  - 层级稳定：`neutral > happy > sad/angry`。
- **Gemma 3n**：
  - 同样以`neutral`为主导吸引子；
  - 但在**Audio-only**条件下表现出极端的`neutral`坍缩倾向；
  - 加入video信息后该偏差显著抑制，显示更强的模态切换敏感性。

> 🔍 图3显示：无论移除多少标签，错误分布均非均匀，而是呈现清晰的层级回退模式。

#### （2）模态依赖性与不对称融合
- **核心发现**：
  - 多模态输入（Face+Voice）的错误结构更接近**Face-only**，而非Audio-only；
  - 表明**视觉模态占据主导地位**，且其存在会压制音频带来的偏差；
  - 并未实现真正的“互补融合”，而是**强化了主导模态的支配地位**。

> ❗ 结论：当前多模态融合机制不是纠偏，而是**锁定主导模态的偏差模式**。

#### （3）物理代理模型的结果
- **低注意力水平**（$B_{\text{self}}, B_{\text{cross}} = 10^{-4}$）：
  - 模态X（对应$x(t)$）主导预测；
  - NMSE较高 → 推理不准确；
- **高注意力水平**（$B_{\text{self}}, B_{\text{cross}} = 100$）：
  - $ \phi(X) \approx \phi(Y) $，两模态贡献均衡；
  - NMSE最低，预测完美复现Lorenz吸引子结构（图5c）；
- **中间区域**：随着注意力增强，系统从单模态主导平滑过渡到平衡状态。

> ✅ 支持假设：**足够的self-和cross-attention强度是防止跨模态偏差的关键**。

---

## 4. 关键结论和发现

### 🔑 主要发现
1. **跨模态偏差具有结构性和动力学根源**  
   错误并非随机发生，而是由Transformer内部的attention dynamics导致的**系统性吸引子结构**。

2. **多模态输入可能加剧而非缓解模态偏见**  
   添加次要模态并不会自动提升鲁棒性或公平性；相反，在当前融合机制下，它常被主导模态“覆盖”或“抑制”。

3. **neutral 类别是普遍的语义吸引子**  
   在情绪识别任务中，`neutral`成为默认fallback选项，反映出模型在不确定性下的保守策略，构成一种**输出空间中的 reinforcement bias**。

4. **物理代理模型能有效模拟并解释真实MLLM的行为**  
   multi-oscillator model成功再现了注意力调节下的模态平衡现象，验证了其作为**可解释性工具**的有效性。

### ⚠️ 方法的局限性
- **代理模型仍为简化抽象**：虽捕捉了attention机制的核心动力学，但未完全还原实际MLLM的复杂结构（如FFN非线性、LayerNorm细节等）；
- **实验集中在特定任务**：情绪识别与时间序列预测，是否适用于其他领域（如医疗诊断、自动驾驶）需进一步验证；
- **缺乏干预性实验**：目前主要是观察与建模，尚未提出具体的去偏算法。

### 🔮 未来工作方向
1. **发展基于动力学调控的去偏机制**  
   利用物理模型指导attention权重初始化或正则化，主动平衡模态贡献。

2. **构建标准化的“偏差动力学测试平台”**  
   将multi-oscillator model作为benchmark环境，用于评估不同MLLM架构的内在公平性倾向。

3. **扩展至更多模态与任务场景**  
   引入text、vision、audio三模态交互，并应用于医疗、教育等高风险领域。

4. **推动“计算现象学”（computational phenomenology）成为AI可解释性新范式**  
   倡导从机器“主观经验”出发的研究路径，超越传统symbolic representation框架。

---

> 💡 **总体评价**：本论文开创性地将**物理学思想与现象学哲学**引入多模态AI研究，提出了一种全新的理解跨模态偏差的视角——不仅是“学到什么”，更是“如何动态地处理”。这一范式转变有望为解决深度学习中的黑箱问题提供根本性突破路径。

</details>

---

### 11. [Exploring the Impact of Parameter Update Magnitude on Forgetting and Generalization of Continual Learning](https://arxiv.org/abs/2602.20796)

**Authors**: JinLi He, Liang Bai, Xian Yang  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.20796v1  

#### Abstract
The magnitude of parameter updates are considered a key factor in continual learning. However, most existing studies focus on designing diverse update strategies, while a theoretical understanding of the underlying mechanisms remains limited. Therefore, we characterize model's forgetting from the pe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Exploring the Impact of Parameter Update Magnitude on Forgetting and Generalization of Continual Learning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于 **Continual Learning** 中一个核心挑战：**catastrophic forgetting**（灾难性遗忘）。尽管已有大量方法尝试缓解遗忘，但多数研究集中于设计复杂的更新策略（如正则化、回放、模块扩展），而对参数更新机制本身的理论理解仍不足。

本文提出并回答以下关键问题：
- 遗忘是否仅由部分参数变化引起？
- 在何种条件下，冻结大部分参数（frozen training）优于全量微调（initialized training）？
- 多少参数更新是“足够”的？是否存在最优的 **parameter update magnitude**？

### 🧠 提出的新方法与新思路
1. **从参数空间漂移角度建模遗忘机制**  
   将遗忘形式化为因任务特定漂移（task-specific drift）导致的知识退化，突破了传统假设中统一参数空间的限制。

2. **推导最小化遗忘的最优参数更新幅度**  
   在固定模型容量下，通过数学分析推导出使遗忘最小化的最优可训练参数数量 $ p^{(2)} $，建立了一个统一的优化框架，将 **frozen training** 和 **initialized training** 统一起来。

3. **提出自适应混合更新框架（Adaptive Hybrid Update Framework）**  
   基于梯度方向动态调整更新幅度：
   - 若当前任务与前一任务梯度方向相似（高 cosine similarity），说明任务接近 → 减少更新（偏向 frozen training）
   - 否则增加可训练参数数量 → 更多更新（偏向 initialized training）

### ⚖️ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **理论深度** | 首次系统分析参数更新幅度对遗忘与泛化的影响，提供可解释的数学依据 |
| **效率与可扩展性** | 不依赖额外存储（如 replay buffer）或复杂架构设计，适合大规模部署 |
| **灵活性** | 自适应机制能根据任务相关性自动调节稳定性（stability）与可塑性（plasticity）之间的权衡 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在四个标准 Continual Learning 基准上进行验证：
- **Split CIFAR-10**：10类图像分类，分为多个增量任务
- **Split CIFAR-100**：100类细粒度图像分类
- **Split CUB-200**：200种鸟类细粒度分类（fine-grained）
- **Split Permuted MNIST**：手写数字经随机像素排列生成的任务序列

此外还构造了两个变体用于验证理论预测：
- **Correlated Split CIFAR-100**：增强任务间相似性（共享类别）
- **Corrupted Split CUB-200**：引入图像噪声以降低任务相关性

### 🔬 实验设置与评估指标
- **网络架构**：基于 ResNet 构建，共9个卷积块，最后接平均池化与 FC 层
- **训练方式**：每个任务依次训练，不保存旧数据（无 replay）
- **评估阶段**：完成所有任务后测试模型在各历史任务上的表现

#### 主要评估指标：
| 指标 | 定义 | 意义 |
|------|------|------|
| **Avg.Acc (%)** | $\frac{1}{M}\sum_{k=1}^M a_{M,k}$ | 平均准确率，反映整体性能 |
| **Task.Acc (%)** | $\frac{1}{M}\sum_{k=1}^M a_{k,k}$ | 当前任务准确率，衡量学习能力 |
| **Avg.Forgetting (%)** | $\frac{1}{M-1}\sum_{k=1}^{M-1}(a_{k,k} - a_{M,k})$ | 忘记率，越低越好 |

所有结果均为三次运行的均值 ± 标准差。

### 🔁 对比的基线方法
| 方法 | 描述 |
|------|------|
| **Initialized Training** | 每个任务从头初始化并训练全部参数（即独立训练） |
| **Frozen Training** | 冻结主干网络，仅训练少量新增的 task-specific 参数 |
| **Adaptive Training (Ours)** | 本文提出的混合策略，根据梯度一致性动态调整可训练参数比例 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table I）

| 方法 | Split CIFAR-100 | Split CUB-200 | Split CIFAR-10 | Split PMNIST |
|------|------------------|---------------|----------------|--------------|
| | Avg.Acc / Forget | Avg.Acc / Forget | Avg.Acc / Forget | Avg.Acc / Forget |
| **Initialized Training** | 42.27 / 24.84 | 37.98 / 21.28 | 75.74 / 23.30 | 75.97 / 30.76 |
| **Frozen Training** | 43.38 / 23.96 | 37.15 / 21.18 | 75.98 / 21.10 | 76.78 / 29.21 |
| **Adaptive Training (Ours)** | **47.19** / **24.15** | **39.01** / **20.76** | **80.55** / **19.51** | **77.99** / **28.05** |
| **Improvement vs Init.** | **+4.92↓** / **-0.69↓** | **+1.03↓** / **-0.52↓** | **+4.81↓** / **-3.79↓** | **+2.02↓** / **-2.71↓** |

> 注：“↓”表示提升（数值越大越好或越小越好）

### 🔍 进一步长序列实验（Table III，20-task setting）

| 方法 | Task.Acc | Avg.Acc | Forgetting |
|------|----------|---------|------------|
| **Initialized Training** | 79.36 | 37.79 | 46.68 |
| **Frozen Training** | 78.28 | 37.59 | 45.88 |
| **Adaptive Training (Ours)** | **80.81** | **39.09** | **45.08** |
| **Improvement** | **+1.45** | **+1.30** | **-1.60** |

> 在更长任务序列下，本方法依然显著优于两种极端策略。

### 📈 当前任务学习能力（Table II）
| 方法 | Split CIFAR-100 | Split CUB-200 |
|------|------------------|---------------|
| **Initialized Training** | 60.72 | 57.32 |
| **Frozen Training** | 57.37 | 56.27 |
| **Adaptive Training** | **67.30** | **60.69** |
| **Improvement** | **+6.58** | **+3.37** |

> 表明所提方法不仅能减少遗忘，还能更好掌握新任务。

### 🔍 消融实验与理论验证
- 在 **Correlated Split CIFAR-100** 上，Frozen Training 明显优于 Initialized Training → 支持理论预测：当任务在参数空间中相近时，冻结更有利。
- 在 **Corrupted Split CUB-200** 上，Frozen Training 优势减弱 → 说明任务差异大会削弱其效果。
- Adaptive Training 在两类场景下均取得最佳性能，证明其具备良好的适应性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **参数更新幅度直接影响遗忘程度**  
   存在一个最优的更新规模，在此之下既能有效学习新任务，又能最大限度保留旧知识。

2. **frozen training 并非总是劣于 full fine-tuning**  
   当连续任务在参数空间中距离较近时，**frozen training 能实现更低的遗忘和更好的泛化性能**。

3. **任务间的梯度一致性可用于判断参数空间距离**  
   通过计算前后任务梯度的 cosine similarity，可以有效估计任务相关性，从而指导更新策略选择。

4. **自适应混合策略优于固定策略**  
   动态切换 frozen 与 initialized training 可兼顾 stability 与 plasticity，显著超越单一模式。

### ⚠️ 方法的局限性
- **依赖梯度方向作为任务相关性的代理信号**：可能受 batch variance 或 optimizer dynamics 影响。
- **未考虑任务边界未知的情况（task-free CL）**：目前假定任务身份已知。
- **理论分析基于简化模型设定**（如 Gaussian features），实际 DNN 中可能存在非线性效应。

### 🔮 未来工作方向
- 扩展至 **class-incremental** 和 **task-free** 场景
- 探索更鲁棒的任务相关性度量方式（如 Hessian-based metrics）
- 结合 parameter-efficient tuning 方法（如 LoRA, Adapter）进一步提升效率
- 将理论推广到 Transformer 架构与 LLMs 的持续学习场景

---

## 总结
> 本文从理论出发，揭示了 **parameter update magnitude** 是影响 Continual Learning 中遗忘与泛化的关键因素。通过统一分析 frozen 与 initialized training，并提出一种基于梯度方向的自适应混合更新策略，实验证明其在多个基准上显著优于传统方法。这项工作不仅提供了新的理论视角，也为高效、可扩展的持续学习算法设计指明了方向。

</details>

---

### 12. [HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG](https://arxiv.org/abs/2602.20926)

**Authors**: Yuqi Huang, Ning Liao, Kai Yang, Anning Hu, Shengchao Hu, Xiaoxing Wang, Junchi Yan  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.20926v1  

#### Abstract
Large Language Models (LLMs) often struggle with inherent knowledge boundaries and hallucinations, limiting their reliability in knowledge-intensive tasks. While Retrieval-Augmented Generation (RAG) mitigates these issues, it frequently overlooks structural interdependencies essential for multi-hop ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在知识密集型任务中容易产生**幻觉**（hallucination）并受限于其固有的知识边界。传统的 **Retrieval-Augmented Generation (RAG)** 虽然通过检索外部知识缓解了这一问题，但通常将文本视为扁平化的段落集合，忽略了文档间的结构性依赖关系，导致在需要多跳推理（multi-hop reasoning）的任务中表现不佳。

现有的 **Graph-based RAG** 方法尝试引入知识图谱（Knowledge Graph, KG）来建模实体间的关系，但仍面临显著的**准确率与效率之间的权衡**：
- 一些方法（如 HippoRAG、ToG）依赖复杂的图遍历算法（如 Personalized PageRank），计算开销大、延迟高；
- 另一些轻量级方法（如 LinearRAG）为提升效率而简化图结构（忽略边信息），牺牲了对复杂逻辑链的建模能力；
- LLM生成的社区摘要可能引入语义噪声和二次幻觉。

---

### 提出的新方法：HELP 框架
本文提出 **HELP**（HyperNode Expansion and Logical Path-Guided Evidence Localization for GraphRAG），一个兼顾准确性与效率的新型 GraphRAG 框架，包含两个核心策略：

#### （1）HyperNode 扩展（HyperNode Expansion）
- 引入 **HyperNode** 作为基本检索单元，每个 HyperNode 是一组连贯的知识三元组（triplets）的聚合体。
- 采用迭代方式从查询相关的种子三元组出发，逐步扩展形成多跳推理路径。
- 利用确定性线性化函数对三元组集合进行排序并序列化，确保嵌入一致性。
- 在每一步使用基于语义距离的剪枝机制（beam search），控制搜索空间规模，避免组合爆炸。

> ✅ **优势**：显式建模跨文档的事实依赖关系，支持复杂多跳推理，同时通过剪枝保持高效。

#### （2）逻辑路径引导的证据定位（Logical Path-Guided Evidence Localization）
- 利用最终构建的 HyperNodes 作为“逻辑锚点”，结合预构建的 **Triple-to-Passage Index**，直接映射回原始语料库中的支持性段落。
- 设计加权评分机制，综合考虑多个推理路径对同一段落的支持程度，实现共识驱动的证据定位。
- 采用**混合检索策略**（hybrid retrieval）：优先保留由逻辑路径定位的高质量段落，辅以 Dense Passage Retrieval（DPR）补充广度覆盖。

> ✅ **优势**：避免昂贵的实时图遍历，大幅降低延迟；利用结构共识增强鲁棒性，减少噪声影响。

---

### 相比现有方法的优势
| 维度 | 传统 GraphRAG | HELP |
|------|----------------|-------|
| 准确性 | 高（依赖完整图结构） | 更高（显式路径建模 + 共识机制） |
| 效率 | 低（图遍历耗时） | 极高（无需在线图操作） |
| 结构利用 | 完整但冗余 | 精准聚焦相关路径 |
| 噪声抑制 | 易受 LLM 摘要污染 | 通过剪枝与共识过滤 |

> 🔥 **核心突破**：首次实现了在不牺牲多跳推理精度的前提下，将 GraphRAG 的检索速度提升近 **28.8×**。

---

## 2. 核心实验方法和设置

### 数据集
涵盖单跳与多跳问答任务，确保全面评估：
- **Simple QA**（单跳）：
  - `NaturalQuestions` (NQ)
  - `PopQA`
- **Multi-Hop QA**（多跳）：
  - `MuSiQue`
  - `2WikiMultiHopQA` (2Wiki)
  - `HotpotQA`
  - `LV-Eval`（长上下文基准，最长达 256k tokens）

所有实验沿用 HippoRAG2 的数据划分、语料库和评估协议，保证公平比较。

---

### 实验设置与评估指标

#### 主干模型配置
- **LLM 生成器**：`Llama-3.3-70B-Instruct`
- **嵌入模型**：`NV-Embed-v2`
- **知识提取**：基于 LLM 的 OpenIE 模块抽取三元组
- **上下文长度**：Top-5 段落用于生成答案

#### 评估指标
- **F1 Score**（token-level）：衡量预测答案与真实答案的重叠度
- **Exact Match (EM)**：完全匹配的比例
- **Recall@5**：前5个检索段落中是否包含黄金证据
- **检索时间**（秒/千条查询）：评估效率

---

### 基线方法对比
分为三类进行系统比较：
1. **经典与稠密检索器**：
   - BM25, Contriever, GTR
2. **先进嵌入模型**（7B级别）：
   - GTE-Qwen2-7B, GritLM-7B, NV-Embed-v2
3. **Graph-based RAG 方法**：
   - RAPTOR, GraphRAG, LightRAG
   - HippoRAG / HippoRAG2（当前最优）
   - LinearRAG*（本文复现）
   - HyperGraphRAG*（本文复现）

> 注：带 * 表示作者自行复现以确保实验一致性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | Avg F1 (%) | 备注 |
|------|------------|------|
| NV-Embed-v2 | 51.7 | 最强纯嵌入模型 |
| HippoRAG2 | **54.6** | 当前 SOTA GraphRAG |
| **HELP (Ours)** | **55.3** | ✅ 新 SOTA |

#### 性能亮点：
- 在 **平均 F1 上超越 HippoRAG2 1.3%**，相对提升约 **2.4%**
- 相比最强嵌入模型（NV-Embed-v2）有 **7.0% 的相对增益**
- 在 **2Wiki** 多跳任务上，F1 达到 **73.9 vs. 71.0（HippoRAG2）**，提升显著
- 成功处理 **LV-Eval** 大规模语料（HyperGraphRAG 因索引失败无法运行）

---

### 与基线方法的对比结果

| 对比维度 | 结果 |
|--------|------|
| **vs. Dense Retrievers** | 超越 GTR 达 **21.3% F1 提升**，证明结构信息的关键作用 |
| **vs. Lightweight GraphRAG** | 比 LinearRAG 在 2Wiki 上高出 **34.6% F1**，说明边信息不可忽视 |
| **vs. State-of-the-art GraphRAG** | 略胜 HippoRAG2，且在更短时间完成检索 |

---

### 消融实验结果（Ablation Studies）

#### （1）混合检索策略分析（Table 2）
| 逻辑路径配额 M | F1 (%) | Recall@5 (%) |
|---------------|--------|-------------|
| 0（仅 DPR） | 61.55 | 76.25 |
| 4 | **73.90** | **92.15** |
| 5（全逻辑） | 73.09 | 91.65 |

> 📌 发现：当 M=4 时性能达到峰值，表明 **保留部分 DPR 结果可有效弥补图不完整性**，是最佳平衡点。

#### （2）扩展跳数 N 的影响（Figure 3）
- **F1 表现稳健**：即使 N=1 或 N=2，F1 均 >74.5%
- **N=3 时达到峰值（76.18%）**
- **N=4 时性能下降**：因过度扩展引入噪声
- **时间成本指数增长**：N=3 时已达 577s，N=4 不可行

> ✅ 推荐设置：**N=2**，可在 <100s 内获得接近最优性能。

#### （3）超参数敏感性分析（Figure 4）
- 在不同初始三元组数量 `n ∈ {1,...,5}` 和 beam size `k ∈ {30,50,70,100}` 下，F1 波动范围仅为 **46.7% ~ 48.4%**
- 表明 HELP 对超参数选择**高度鲁棒**，适合实际部署。

#### （4）跨主干模型泛化性（Table 4）
使用 `Qwen3-30B-A3B-Instruct` 替代 Llama：
- HELP 平均 EM/F1 达 **42.4%/52.6%**
- 显著优于 HippoRAG2（40.5%/51.4%）
> ✅ 证明性能增益源于方法本身，而非特定 LLM 特性。

---

## 4. 关键结论和发现

### 主要发现
1. **结构化推理不必牺牲效率**：HELP 成功打破了 GraphRAG 中“高精度 ↔ 高延迟”的固有矛盾。
2. **HyperNode 是有效的高阶推理单元**：它能自然地将孤立事实组织成逻辑链条，显著提升 multi-hop QA 表现。
3. **路径引导的证据定位极为高效**：通过预建索引将结构路径映射回原文，避免运行时图遍历，实现 **高达 28.8× 的速度提升**。
4. **混合检索策略最优**：结合逻辑路径的精准性与 DPR 的广泛性，达到深度与广度的平衡。
5. **方法具有强泛化性和鲁棒性**：在多种 LLM 主干、数据集和超参数下均稳定领先。

---

### 方法的局限性
1. **依赖高质量三元组抽取**：OpenIE 模块的性能直接影响 HyperNode 初始化质量。
2. **固定跳数限制深层推理**：虽然 N=2 已足够强大，但对于极深（>4 hop）问题仍可能存在瓶颈。
3. **Triple-to-Passage Index 占用额外存储空间**：需维护三元组到段落的映射表，增加离线索引负担。
4. **无法动态修正错误三元组**：一旦错误 triplet 进入路径，后续难以纠正（缺乏反馈机制）。

---

### 未来工作方向
1. **引入自校正机制**：在推理过程中检测并修正错误的三元组或路径分支。
2. **动态决定扩展跳数**：根据问题复杂度自动调整 N，而非固定值。
3. **端到端联合优化**：将 HyperNode 构建与 LLM 生成过程联合训练，进一步提升协同效果。
4. **应用于其他结构化任务**：如事件因果推理、法律条款追溯、科学文献溯源等场景。
5. **探索更紧凑的索引结构**：减少 Triple-to-Passage Index 的存储开销，提升可扩展性。

---

> ✅ **总体评价**：HELP 是一项兼具理论创新与工程实用性的杰出工作，为构建**准确、高效、可扩展的 GraphRAG 系统**树立了新的标杆。

</details>

---

### 13. [LogicGraph : Benchmarking Multi-Path Logical Reasoning via Neuro-Symbolic Generation and Verification](https://arxiv.org/abs/2602.21044)

**Authors**: Yanrui Wu, Lingling Zhang, Xinyu Zhang, Jiayu Chang, Pengyu Li, Xu Jiang, Jingtao Hu, Jun Liu  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.21044v1  

#### Abstract
Evaluations of large language models (LLMs) primarily emphasize convergent logical reasoning, where success is defined by producing a single correct proof. However, many real-world reasoning problems admit multiple valid derivations, requiring models to explore diverse logical paths rather than comm...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LogicGraph: Benchmarking Multi-Path Logical Reasoning via Neuro-Symbolic Generation and Verification**

---

## **1. 论文的主要贡献和创新点**

### ✅ **解决了什么问题**
当前对 **Large Language Models (LLMs)** 的逻辑推理能力评估主要集中在**收敛性推理（convergent reasoning）**，即模型只需找到一条正确路径得出最终结论即可。然而，现实世界中的复杂推理问题往往存在**多条合法的推导路径（multi-path reasoning）**，要求模型具备探索不同逻辑路线的能力（即发散性思维，divergent thinking）。现有基准（如 ProofWriter、FOLIO）大多只关注单一正确答案，忽略了对多路径推理能力的系统性评估。

### ✅ **提出了什么新方法或新思路**
作者提出：
- **LogicGraph**：首个专门用于系统评估**多路径逻辑推理能力**的基准（benchmark），每个实例都具有多个最小证明路径（minimal proofs）。
- 一个**神经符号生成与验证框架（neuro-symbolic generation and verification pipeline）**，通过以下方式构建高质量数据：
  - **反向逻辑生成（backward logic generation）**：从目标结论出发，自顶向下构造满足多种推导路径的 **Logic DAG（Directed Acyclic Graph）**。
  - **语义实例化（semantic instantiation）**：利用 LLM 将抽象逻辑公式转化为自然语言叙述，保持逻辑一致性的同时增强现实感。
- 一种**无参考评估框架（reference-free evaluation framework）**，结合：
  - **自动形式化（auto-formalization）**：将模型输出的自然语言步骤转换为 Prover9 形式逻辑。
  - **符号验证（symbolic verification）**：使用 Prover9 验证每一步的局部有效性（Local Validity）和全局可达性（Global Validity）。

### ✅ **相比现有方法的优势**
| 特性 | LogicGraph | 其他基准（如 ProofWriter, FOLIO） |
|------|-----------|-------------------------------|
| 多路径支持 | ✅ 支持 2–19 条有效路径 | ❌ 通常仅 1 条路径 |
| 推理深度 | 平均 6.01 步 | 多数 < 4 步 |
| 节点复用 | ✅ 允许中间节点跨路径共享（Reuse Ratio: 1.0–1.9） | ❌ 路径独立 |
| 逻辑干扰项 | ✅ 结构级干扰（某些前提在一条路径中关键，在另一条中是干扰） | ❌ 主要是语义噪声或死胡同事实 |
| 评估方式 | ✅ 支持开放式的多路径生成评估（stepwise + proof-level） | ❌ 多为最终答案匹配 |

---

## **2. 核心实验方法和设置**

### 📚 **使用的数据集**
- **LogicGraph 自建数据集**：共 900 个实例，按难度分为三档：
  - Small: 2–4 条有效路径（300 例）
  - Medium: 5–7 条路径（300 例）
  - Large: ≥8 条路径（300 例）
- 数据来源于神经符号生成流程，确保每条路径经 Prover9 验证且互不冗余。

### ⚙️ **实验设置和评估指标**
#### **评估流程三阶段**：
1. **预处理与自动形式化（Pre-processing & Auto-Formalization）**
   - 使用 LLM 提取模型输出中的推理链，并显式标注依赖关系。
   - 将自然语言步骤翻译成 Prover9 语法，借助上下文示例保证保真度。

2. **符号验证（Symbolic Verification）**
   - **Local Validity**：检查每步是否由引用前提逻辑推出。
   - **Global Validity**：确认最终结论能否仅从所用前提子集推得，防止“幻觉前提”。

3. **分层错误分类（Hierarchical Error Taxonomy）**
   - **Semantic Comprehension Errors**：
     - Semantic Misinterpretation（误解事实/规则）
     - Information Omission（遗漏必要前提）
     - Fact Hallucination（虚构外部前提）
   - **Logical Execution Errors**：
     - Invalid Deduction（非逻辑后承）
     - Rule Misapplication（错误应用推理规则）

#### **核心评估指标**：
- **Solution Coverage**：模型生成的有效路径占所有最小证明的比例（衡量发散能力）。
- **Convergent Accuracy**：是否能至少找到一条正确路径。
- **Divergence Gap**：Coverage 与 Convergent Accuracy 的差距，反映模型探索多样性路径的能力缺陷。

### 🔁 **基线方法对比**
测试了多种主流 LLM，包括：
- **通用模型**：GPT-4, GPT-5.1 (OpenAI), GLM-4.6 (ZhipuAI)
- **推理优化模型**：Gemini-3-Pro (DeepMind), Deepseek-V3.2-Thinking
- 所有模型均以 zero-shot 或 few-shot 方式运行。

---

## **3. 主要实验结果和性能指标**

### 📊 **关键性能数据**
| 模型 | Convergent Accuracy | Average Solution Coverage | Divergence Gap |
|------|------------------------|------------------------------|----------------|
| GPT-5.1 | 92.1% | 38.7% | 53.4% |
| Gemini-3-Pro | 90.3% | 41.2% | 49.1% |
| Deepseek-V3.2-Thinking | 88.6% | 36.5% | 52.1% |
| GLM-4.6 | 85.4% | 30.1% | 55.3% |

> 💡 **观察**：尽管顶尖模型在收敛任务上表现良好（>85%），但在**多路径覆盖**方面严重不足（平均 < 42%），说明它们倾向于“早早就锁定一条路径”，缺乏探索备选方案的能力。

### 🔍 **与基线方法的对比结果**
- 在相同条件下，**推理专用模型略优于通用模型**，但差距有限。
- 所有模型的 **solution coverage 随着推理深度增加显著下降**：
  - 当路径长度 > 6 步时，coverage 下降超过 60%。
- 存在明显的 **“结果导向型失败”（result-oriented failure）**：模型常虚构中间引理强行连接目标结论，导致无效路径。

### 🧪 **消融实验结果（隐含分析）**
虽然未明确列出消融实验表，但从设计可看出关键组件作用：
- 若去除 **backward DAG 构造** → 无法保证路径穷尽性和无遗漏。
- 若跳过 **Prover9 符号验证** → 评估易受 LLM judge 幻觉影响，可靠性下降。
- 若不用 **语义实例化** → 数据过于人工化，降低对真实场景的泛化能力。

---

## **4. 关键结论和发现**

### ✅ **主要发现**
1. **当前 LLM 普遍存在“发散性推理鸿沟”（Divergence Gap）**：
   - 即使能正确回答问题，也极少能枚举出多种合法推导路径。
   - 表现出强烈的“单路径执念”（single-route fixation），尤其在深层推理中更明显。

2. **推理深度加剧覆盖率下降**：
   - 随着推理链变长，模型探索替代路径的能力急剧退化。

3. **错误类型集中于“结果驱动幻觉”**：
   - 模型为了达成目标结论，频繁引入未经支持的前提或错误推理规则。

4. **神经符号验证高度可靠**：
   - 与人类专家标注相比，Prover9 + LLM 形式化流程达到 **98.80% Step Accuracy 和 95.22% Proof Accuracy**，验证了该评估框架的有效性。

### ⚠️ **方法的局限性**
1. **合成到现实的差距（Synthetic-to-Real Gap）**：
   - 数据基于离散逻辑生成，缺乏现实世界的模糊性、概率性和不确定性。
   - 实际推理常涉及不完整信息或信念更新，本框架尚未涵盖。

2. **计算成本高**：
   - 每个样本需进行多次 Prover9 验证，评估效率远低于传统多项选择题。

3. **潜在的社会偏见风险**：
   - 尽管使用中立模板，LLM 在语义实例化过程中仍可能引入隐性偏见。

4. **环境开销大**：
   - 多路径生成与符号验证过程能耗较高，碳足迹较大。

### 🔮 **未来工作方向**
- 开发更高效的神经符号验证流水线，降低评估资源消耗。
- 引入**概率逻辑**或**模糊推理**扩展至不确定环境下的多路径推理。
- 探索**主动探索机制**（如 Tree of Thoughts、Best-of-N sampling）提升模型路径多样性。
- 将 LogicGraph 应用于训练阶段，引导模型学习真正的发散性思维策略。

---

> 🔗 **代码与数据开源**：  
> 项目已公开在 GitHub：[https://github.com/kkkkarry/LogicGraph](https://github.com/kkkkarry/LogicGraph)  
> 包括数据集、生成脚本、评估工具及完整文档。

</details>

---

### 14. [Nonparametric Teaching of Attention Learners](https://arxiv.org/abs/2602.20461)

**Authors**: Chen Zhang, Jianghui Wang, Bingyang Cheng, Zhongtao Chen, Wendong XU, Cong Wang, Marco Canini, Francesco Orabona, Yik Chung WU, Ngai Wong  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.20461v1  

#### Abstract
Attention learners, neural networks built on the attention mechanism, e.g., transformers, excel at learning the implicit relationships that relate sequences to their corresponding properties, e.g., mapping a given sequence of tokens to the probability of the next token. However, the learning process...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Nonparametric Teaching of Attention Learners**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **Attention Learners（如Transformer）训练成本高**：尽管在 NLP 和 CV 领域表现卓越，但其训练过程通常需要大量计算资源和时间。
- **传统机器教学（Machine Teaching）不适用于非参数化模型**：已有方法多针对参数化目标函数设计教学集，难以直接用于隐式定义的复杂映射（如大模型学到的功能）。

### 🚀 提出的新方法与新思路
- **提出 AtteNT（Attention Neural Teaching）范式**：
  - 将 Attention Learner 的训练重新解释为一种**非参数教学（nonparametric teaching）**过程。
  - 教师通过选择“最具教学价值”的序列样本（即预测误差最大的样本），加速学生模型收敛。
- **理论桥梁构建**：
  - 首次证明：基于参数空间梯度下降的 Attention Learner 演化过程，在功能空间上等价于**函数梯度下降（functional gradient descent）**。
  - 动态 Attention Neural Tangent Kernel (ANTK) 收敛至重要性自适应的规范核（importance-adaptive canonical kernel），从而将 Attention 机制与非参数教学理论统一起来。

### 🔍 相比现有方法的优势
| 方面 | AtteNT 的优势 |
|------|----------------|
| **通用性** | 可应用于任意基于 Attention 的模型（LLM、ViT 等），不限于特定架构 |
| **效率提升** | 显著减少训练时间，无需额外超参调优 |
| **性能保持甚至增强** | 不仅提速，还提升了下游任务准确率 |
| **理论支撑强** | 建立了 Attention 学习与非参数教学之间的形式化联系，提供可解释性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### 自然语言处理（NLP）
| 任务 | 数据集 |
|------|--------|
| 数学推理 | MetaMathQA（训练）、GSM8K、MATH（测试） |
| 编程能力 | CodeFeedback（训练）、HumanEval、MBPP（测试） |
| 对话理解 | WizardLM-Evol-Instruct（训练）、MT-Bench（测试） |

#### 计算机视觉（CV）
| 任务 | 数据集 |
|------|--------|
| 图像分类 | ImageNetS50（从 ImageNet-1k 中提取的 50 类子集） |
| 语义分割 | NYUv2(S) |
| 深度估计 | NYUv2(D) |

> 所有数据均通过 HuggingFace 获取，并生成伪标签用于多模态预训练。

### ⚙️ 实验设置与评估指标

| 组件 | 设置详情 |
|------|----------|
| **模型** | LLaMA 2-7B, Mistral-7B, Gemma-7B（LLM）；ViT-B/16（CV） |
| **训练方式** | LoRA 微调（LLM）、端到端训练（ViT） |
| **优化器** | AdamW，学习率调度采用 cosine decay |
| **Batch Size** | LLM: 128；ViT: 2048（分布式） |
| **精度** | Float32（LLM），AMP 混合精度（ViT） |
| **评估指标** | Accuracy / mIoU / δ1（Depth Estimation） / Pass@1（代码生成） |
| **主要对比维度** | 平均微调时间（Avg. Time）、各项任务得分提升（↑） |

### 🆚 基线方法对比
- **标准训练（Standard Fine-tuning）**：随机采样全部数据进行训练。
- **Class Weight Sampling**：按类别频率反比加权采样。
- **Fixed Weight Sampling**：固定不同任务模态的采样比例（如 RGB:SemSeg:Depth = 1:2:2）。
- **GradNorm Sampling (Chen et al., 2018)**：根据各任务梯度范数动态调整采样权重。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ 大语言模型（LLM）微调结果（Table 1）
| 模型 | AtteNT 时间 ↓ | GSM8K ↑ | MATH ↑ | HumanEval ↑ | MBPP ↑ |
|------|---------------|---------|--------|-------------|--------|
| LLaMA 2-7B | 213±2m (-13.4%) | +0.49 | +1.42 | +3.45 | +1.96 |
| Mistral-7B | 180±2m (-11.8%) | +2.13 | +3.06 | +3.13 | +3.22 |
| Gemma-7B | 201±2m (-11.9%) | +2.51 | +0.88 | +0.43 | +0.59 |
| **平均加速** | **12.78%** | — | — | — | — |

> 注：原始宣称 **LLM 微调时间减少 13.01%**，实测接近该值。

#### ✅ 视觉 Transformer（ViT）训练结果（Table 2）
| 方法 | 预训练时间 ↓ | ImageNetS50 ↑ | NYUv2(S) ↑ | NYUv2(D) ↑ |
|------|--------------|---------------|------------|------------|
| Baseline | 1234m | 92.2 | 51.9 | 52.1 |
| **AtteNT (Ours)** | **980m (-20.58%)** | **92.3** | **52.6** | **57.2** |

> 在 **ViT 从零训练** 场景下实现 **20.58% 的时间节省**，且所有下游任务性能均有提升，尤其深度估计 δ1 提升达 **5.1%**。

### 🔬 消融实验结果（Ablation Study, Table 3）

| 配置 | 训练时间 | ImageNetS50 | NYUv2(S) | NYUv2(D) |
|------|----------|-------------|----------|----------|
| Random | 966m | 88.6 | 45.3 | 49.6 |
| Hard (确定性难样本) | 963m | 91.4 | 48.4 | 57.2 |
| **Soft (Gumbel-Top-k)** | **980m** | **92.3** | **52.6** | **57.2** |
| Fixed Interval | 1301m | 93.2 | 53.6 | 61.4 |

#### 发现：
- **Soft 采样策略最优**：基于损失分数的概率采样（Gumbel-Top-k）比 Hard 或 Random 更稳定有效。
- **增量式比率调度更佳**：采用 `Incremental` ratio 调度（从低到高增加采样比例）优于固定或余弦调度。
- 最终配置：`(Incremental Ratio, Incremental Interval, Soft Selection)` 实现最佳平衡。

### 🔄 与其他方法对比（Table 8）
| 方法 | 训练时间 | ImageNetS50 | NYUv2(S) | NYUv2(D) |
|------|----------|-------------|----------|----------|
| Class Weight | 1108m | 90.4 | 48.2 | 52.0 |
| Fixed Weight | 1065m | 89.6 | 49.7 | 54.6 |
| GradNorm | 1112m | 91.9 | 52.4 | 55.8 |
| **AtteNT (Ours)** | **980m** | **92.3** | **52.6** | **57.2** |

> AtteNT 在**更低耗时**的同时实现了**全面性能领先**，说明其增益来自原理性机制而非简单采样启发式。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Attention Learner 的训练可被形式化为非参数教学问题**：
   - 教师通过选择“最大误差”样本，能最有效地引导模型进化。
2. **参数空间更新 ≈ 函数空间梯度下降**：
   - Attention 引入的重要性加权使参数梯度具有结构一致性，支持向非参数框架迁移。
3. **动态 ANTK 收敛至规范核**：
   - 实验证明 NTK 在前 200 轮内快速稳定（见 Figure 5 & 6），验证理论假设成立。
4. **AtteNT 具有普适高效性**：
   - 在 LLM 和 ViT 上均取得显著加速（13–20.58%），且性能不降反升。

### ⚠️ 局限性
- **依赖高质量损失信号**：若 loss 不能准确反映样本难度（如噪声标签），可能选错教学样本。
- **当前为单阶段教学策略**：未考虑更复杂的课程设计或多轮交互式教学。
- **对极小数据场景适用性未知**：目前实验集中在大规模训练，小样本场景有待验证。

### 🔮 未来工作方向
1. **扩展至图注意力网络（Graph Attention Networks）**：探索 GAT/GNN 场景下的 AtteNT 应用。
2. **鲁棒性研究**：结合 noise-robust learning 技术，提升在真实噪声环境下的稳定性。
3. **在线/持续教学机制**：构建动态反馈循环，实现教师与学生的协同演化。
4. **应用于世界模型（World Models）等数据驱动系统**：提高模拟与规划的学习效率。

---

> 💡 **一句话总结**：  
> 本论文提出了 **AtteNT**——首个将 **Attention Learner 训练** 与 **非参数教学理论** 统一起来的新范式，通过理论分析与实验证明，该方法可在 **不牺牲性能的前提下，将 LLM 微调时间减少 13.01%，ViT 训练时间减少 20.58%**，为高效训练大模型提供了新的理论视角与实用工具。

</details>

---

### 15. [Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty](https://arxiv.org/abs/2602.20729)

**Authors**: Xu Wan, Chao Yang, Cheng Yang, Jie Song, Mingyang Sun  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.20729v1  

#### Abstract
Safe Reinforcement Learning (RL) is crucial for achieving high performance while ensuring safety in real-world applications. However, the complex interplay of multiple uncertainty sources in real environments poses significant challenges for interpretable risk assessment and robust decision-making. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在现实世界中部署 **Safe Reinforcement Learning (Safe RL)** 面临多重不确定性挑战，包括：
- **Observation uncertainty**（观测噪声）
- **Action uncertainty**（执行延迟或扰动）
- **Dynamics uncertainty**（系统参数漂移）

传统方法如 **min-max robust RL** 过于保守且计算昂贵；**distributionally robust RL** 假设不确定性独立同分布，难以建模耦合效应；**risk-sensitive RL**（如 CVaR）依赖敏感超参，对多源不确定性鲁棒性不足。

本文旨在解决：  
> 如何在 **多源、非独立、耦合不确定性** 下实现 **可解释、高效、鲁棒的安全决策**？

---

### **提出了什么新方法或新思路**
作者提出 **Fuz-RL** —— 一种基于 **模糊测度理论（fuzzy measure theory）** 的新型鲁棒安全强化学习框架，其核心创新如下：

#### ✅ **(1) 模糊贝尔曼算子（Fuzzy Bellman Operator）**
- 引入 **Choquet 积分** 对价值函数进行非加性聚合，替代传统的期望或最坏情况估计。
- 利用 **λ-fuzzy measure** 建模不同不确定性水平之间的交互作用（super-additive 或 sub-additive），更真实反映复杂系统的复合风险。

#### ✅ **(2) 理论等价性证明：Fuz-RL ≡ Distributionally Robust CMDP**
- 从理论上证明：求解 Fuz-RL（CMDP 形式）等价于求解一个 **distributionally robust safe RL 问题**（robust CMDP 形式）。
- **避免了 min-max 优化**，无需显式构造对抗性转移核，显著降低计算复杂度。

#### ✅ **(3) 模型无关集成能力**
- Fuz-RL 是一个 **plug-and-play 框架**，可无缝嵌入任意现有的 model-free Safe RL 算法（如 PPOL, CUP, CPPO），提升其鲁棒性。

---

### **相比现有方法的优势**
| 维度 | 传统方法（如 RAMU, min-max） | Fuz-RL |
|------|-------------------------------|--------|
| **保守性** | 极端保守，牺牲性能保安全 | 动态权衡，允许安全探索 |
| **计算效率** | 需要内层对抗优化，训练慢 | 无 min-max loop，训练稳定快速 |
| **不确定性建模** | 假设独立、高斯扰动 | 支持耦合、非线性、多源扰动 |
| **可解释性** | 黑箱策略，难分析风险来源 | 模糊密度参数揭示各扰动层级影响 |

---

## 2. **核心实验方法和设置**

### **使用的数据集与环境**
实验在两个主流安全控制基准上进行：

- **safe-control-gym**：物理控制任务，含精确动力学模型
  - `Cartpole-Stab`, `Cartpole-Track`
  - `Quadrotor-Stab`, `Quadrotor-Track`

- **safety-gymnasium**：高维状态动作空间的安全导航任务
  - `Safety-PointGoal1-v0`, `Safety-PointButton1-v0`
  - `Safety-PointCircle1-v0`, `Safety-PointPush1-v0`

---

### **实验设置与评估指标**

#### 🔧 **不确定性注入方式**
| 类型 | 扰动形式 |
|------|----------|
| **Observation Uncertainty** | 加性高斯噪声：`ε·N(0, I)`，`ε ∈ [-0.1, 0.1]` |
| **Action Uncertainty** | 冲击噪声模型：持续时间 D=80 步，衰减率 γ=0.9 |
| **Dynamics Uncertainty** | 修改系统参数（如杆长、质量）并添加噪声 |
| **Multi-source Uncertainty** | 同时施加上述三种扰动 |

#### 📊 **评估指标**
- **AvgRet ↑**：平均累积回报（越高越好）
- **AvgRisk ↓**：约束违反比例（越低越好）
- 所有结果基于 **10 episodes × 10 seeds** 报告均值 ± 标准差

---

### **基线方法对比**
- **Safe RL Baselines**：
  - **PPOL**（PPO-Lagrangian）
  - **CUP**（Conservative Update Policy）
  - **CPPO**（CVaR-Proximal Policy Optimization）

- **Robust Safe RL SOTA**：
  - **RAMU**（Risk-Averse Model Uncertainty）—— 当前最先进的鲁棒安全 RL 方法

- **Fuz-RL 变体**：
  - Fuz-PPOL, Fuz-CUP, Fuz-CPPO（将 Fuz 框架集成到上述算法）

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 和附录）**

| Task | Method | AvgRet | AvgRisk |
|------|--------|--------|---------|
| Cartpole-Stab | Fuz-CPPO | **59±25** | **0.32±0.10** |
| Cartpole-Stab | CPPO | 41±19 | 0.47±0.10 |
| Quadrotor-Track | Fuz-CUP | **175±9** | **0.04±0.02** |
| Quadrotor-Track | CUP | 151±14 | 0.04±0.03 |

> ✅ **Fuz-RL 在 94.9% 的测试场景中实现了更低的风险，在 88.9% 中获得更高回报**

---

### **与基线方法的对比结果**

#### 📈 **vs. Safe RL 基线**
- 平均提升：
  - **AvgRet 提升 15–60%**
  - **AvgRisk 降低 10–25%**
- 方差显著下降（稳定性增强）：
  - AvgRet 方差减少 **20.7%**
  - AvgRisk 方差减少 **22.6%**

> 示例：在 `CartPole-Stab` 上，**Fuz-CUP** 相比 CUP 实现 **61.4% 更高的 AvgRet** 与 **16.7% 更低的 AvgRisk**

#### ⚔️ **vs. RAMU（SOTA robust RL）**
- 在 **36 项对比实验中**：
  - **83.3% 场景下 Fuz-RL 获得更高 AvgRet**
  - **52.8% 场景下实现更低 AvgRisk**
- 特别是在 **Action Uncertainty** 设置下，Fuz-RL **全面超越 RAMU**

> RAMU 往往通过极度保守策略压低风险，但严重牺牲性能；而 Fuz-RL 实现了更好的 **性能-安全平衡**

---

### **消融实验结果（Ablation Study）**

#### 🔍 不确定性层级数 $ K $ 的影响（见 Figure 3）
| $ K $ | 表现 |
|-------|------|
| $ K=1 $ | 性能差，风险高（建模能力不足） |
| $ K=5–15 $ | **最优区间**，奖励高且风险稳定 |
| $ K=25 $ | 训练困难，性能下降（过拟合扰动组合） |

> 结论：适度划分不确定性层级（建议 $ K=10 $）可在表达力与训练稳定性间取得最佳平衡

#### 🧪 验证于电力系统频率控制（IEEE 39-Bus）
| 方法 | 观测噪声 | 动作噪声 | 动态噪声 |
|------|--------|--------|--------|
| PPOL | -5456.30 / 0.17 | -6357.81 / 0.16 | -7471.96 / 0.52 |
| **Fuz-PPOL** | **-4822.03 / 0.14** | **-5789.19 / 0.13** | **-7363.20 / 0.47** |

> ✅ 在真实工业级任务中仍保持一致优势

---

## 4. **关键结论和发现**

### **主要发现**
1. **模糊测度能有效建模耦合不确定性**：现实系统中的多源扰动具有非加性叠加效应，传统概率模型无法捕捉，而 λ-fuzzy measure 成功建模此类交互。
2. **Choquet 积分提供自然的鲁棒视角**：通过核心集（core）隐式编码 worst-case 期望，无需显式对抗训练。
3. **Fuz-RL 显著提升现有 Safe RL 的鲁棒性**：在多种扰动下均优于 PPOL/CUP/CPPO/RAMU，尤其擅长处理 **action impulse noise** 和 **multi-source coupling**。
4. **理论与实践统一**：Fuz-RL 的 CMDP 形式等价于 distributionally robust CMDP，为模糊方法提供了坚实的理论基础。

---

### **方法的局限性**
- **高维状态空间扩展性受限**：当前模糊网络以状态为输入估计 fuzzy density，当状态维度极高时可能失效。
- **静态模糊测度假设**：目前未考虑非平稳不确定性分布（如随时间变化的噪声强度）。
- **计算开销增加**：需采样多个扰动轨迹并计算 Choquet 积分，单步推理成本略高于标准 RL。

---

### **未来工作方向**
1. **设计更高效的模糊测度参数化方法**，适配高维视觉输入（如图像观测）。
2. **引入自适应机制**，动态调整 fuzzy measure 以应对非平稳环境。
3. **扩展至离线 RL 与模仿学习场景**，利用历史数据预训练模糊风险感知模块。
4. **应用于更多现实系统**：自动驾驶、医疗机器人、电网调度等高风险领域。

---

> 🔗 **代码开源地址**：[https://github.com/waunx/FuzRL](https://github.com/waunx/FuzRL)

</details>

---

### 16. [PromptCD: Test-Time Behavior Enhancement via Polarity-Prompt Contrastive Decoding](https://arxiv.org/abs/2602.20696)

**Authors**: Baolong Bi, Yuyao Ge, Shenghua Liu, Yuchen He, Siqian Tong, Lizhe Chen, Lingrui Mei, Zehao Li, Yiwei Wang, Yujun Cai, Ming-Hsuan Yang, Xueqi Cheng  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.20696v1  

#### Abstract
Reliable AI systems require large language models (LLMs) to exhibit behaviors aligned with human preferences and values. However, most existing alignment approaches operate at training time and rely on additional high-quality data, incurring significant computational and annotation costs. While rece...

---

### 17. [ICON: Indirect Prompt Injection Defense for Agents based on Inference-Time Correction](https://arxiv.org/abs/2602.20708)

**Authors**: Che Wang, Fuyao Zhang, Jiaming Zhang, Ziqi Zhang, Yinghui Wang, Longtao Huang, Jianbo Gao, Zhong Chen, Wei Yang Bryan Lim  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.20708v1  

#### Abstract
Large Language Model (LLM) agents are susceptible to Indirect Prompt Injection (IPI) attacks, where malicious instructions in retrieved content hijack the agent's execution. Existing defenses typically rely on strict filtering or refusal mechanisms, which suffer from a critical limitation: over-refu...

---

### 18. [NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](https://arxiv.org/abs/2602.21172)

**Authors**: Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21172v1  

#### Abstract
Vision-Language-Action (VLA) models are advancing autonomous driving by replacing modular pipelines with unified end-to-end architectures. However, current VLAs face two expensive requirements: (1) massive dataset collection, and (2) dense reasoning annotations. In this work, we address both challen...

---

### 19. [ID-LoRA: Efficient Low-Rank Adaptation Inspired by Matrix Interpolative Decomposition](https://arxiv.org/abs/2602.20727)

**Authors**: Xindian Ma, Rundong Kong, Peng Zhang, Ruoxiang Huang, Yongyu Jiang  
**Category**: cs.CL  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.20727v1  

#### Abstract
LoRA has become a universal Parameter-Efficient Fine-Tuning (PEFT) technique that equips Large Language Models (LLMs) to adapt quickly to new tasks. However, when these models are scaled up, even the latest LoRA variants still introduce considerable overhead in trainable parameters. Conversely, aggr...

---

### 20. [FedAvg-Based CTMC Hazard Model for Federated Bridge Deterioration Assessment](https://arxiv.org/abs/2602.20194)

**Authors**: Takato Yasuno  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.20194v1  

#### Abstract
Bridge periodic inspection records contain sensitive information about public infrastructure, making cross-organizational data sharing impractical under existing data governance constraints. We propose a federated framework for estimating a Continuous-Time Markov Chain (CTMC) hazard model of bridge ...

---

### 21. [Discrete Diffusion with Sample-Efficient Estimators for Conditionals](https://arxiv.org/abs/2602.20293)

**Authors**: Karthik Elamvazhuthi, Abhijith Jayakumar, Andrey Y. Lokhov  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.20293v1  

#### Abstract
We study a discrete denoising diffusion framework that integrates a sample-efficient estimator of single-site conditionals with round-robin noising and denoising dynamics for generative modeling over discrete state spaces. Rather than approximating a discrete analog of a score function, our formulat...

---

### 22. [Stability and Generalization of Push-Sum Based Decentralized Optimization over Directed Graphs](https://arxiv.org/abs/2602.20567)

**Authors**: Yifei Liang, Yan Sun, Xiaochun Cao, Li Shen  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.20567v1  

#### Abstract
Push-Sum-based decentralized learning enables optimization over directed communication networks, where information exchange may be asymmetric. While convergence properties of such methods are well understood, their finite-iteration stability and generalization behavior remain unclear due to structur...

---

### 23. [PIME: Prototype-based Interpretable MCTS-Enhanced Brain Network Analysis for Disorder Diagnosis](https://arxiv.org/abs/2602.21046)

**Authors**: Kunyu Zhang, Yanwu Yang, Jing Zhang, Xiangjie Shi, Shujian Yu  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21046v1  

#### Abstract
Recent deep learning methods for fMRI-based diagnosis have achieved promising accuracy by modeling functional connectivity networks. However, standard approaches often struggle with noisy interactions, and conventional post-hoc attribution methods may lack reliability, potentially highlighting datas...

---

### 24. [Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning](https://arxiv.org/abs/2602.20722)

**Authors**: Xu Wan, Yansheng Wang, Wenqi Huang, Mingyang Sun  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.20722v1  

#### Abstract
Traditional on-policy Reinforcement Learning with Verifiable Rewards (RLVR) frameworks suffer from experience waste and reward homogeneity, which directly hinders learning efficiency on difficult samples during large language models post-training. In this paper, we introduce Batch Adaptation Policy ...

---

### 25. [Tool Building as a Path to "Superintelligence"](https://arxiv.org/abs/2602.21061)

**Authors**: David Koplow, Tomer Galanti, Tomaso Poggio  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.21061v1  

#### Abstract
The Diligent Learner framework suggests LLMs can achieve superintelligence via test-time search, provided a sufficient step-success probability $\gamma$. In this work, we design a benchmark to measure $\gamma$ on logical out-of-distribution inference. We construct a class of tasks involving GF(2) ci...

---

### 26. [Motivation is Something You Need](https://arxiv.org/abs/2602.21064)

**Authors**: Mehdi Acheli, Walid Gaaloul  
**Category**: cs.AI  
**Published**: 2026-02-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.21064v1  

#### Abstract
This work introduces a novel training paradigm that draws from affective neuroscience. Inspired by the interplay of emotions and cognition in the human brain and more specifically the SEEKING motivational state, we design a dual-model framework where a smaller base model is trained continuously, whi...

---

### 27. [Overton Pluralistic Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2602.20759)

**Authors**: Yu Fu, Seongho Son, Ilija Bogunovic  
**Category**: cs.CL  
**Published**: 2026-02-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.20759v1  

#### Abstract
Existing alignment paradigms remain limited in capturing the pluralistic nature of human values. Overton Pluralism addresses this gap by generating responses with diverse perspectives from a single query. This paper introduces OP-GRPO (Overton Pluralistic Group Relative Policy Optimization), a reinf...

---

### 28. [FinAnchor: Aligned Multi-Model Representations for Financial Prediction](https://arxiv.org/abs/2602.20859)

**Authors**: Zirui He, Huopu Zhang, Yanguang Liu, Sirui Wu, Mengnan Du  
**Category**: cs.CL  
**Published**: 2026-02-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.20859v1  

#### Abstract
Financial prediction from long documents involves significant challenges, as actionable signals are often sparse and obscured by noise, and the optimal LLM for generating embeddings varies across tasks and time periods. In this paper, we propose FinAnchor(Financial Anchored Representations), a light...

---

### 29. [Tensor Network Generator-Enhanced Optimization for Traveling Salesman Problem](https://arxiv.org/abs/2602.20175)

**Authors**: Ryo Sakai, Chen-Yu Liu  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.20175v1  

#### Abstract
We present an application of the tensor network generator-enhanced optimization (TN-GEO) framework to address the traveling salesman problem (TSP), a fundamental combinatorial optimization challenge. Our approach employs a tensor network Born machine based on automatically differentiable matrix prod...

---

### 30. [CITED: A Decision Boundary-Aware Signature for GNNs Towards Model Extraction Defense](https://arxiv.org/abs/2602.20418)

**Authors**: Bolin Shen, Md Shamim Seraj, Zhan Cheng, Shayok Chakraborty, Yushun Dong  
**Category**: cs.LG  
**Published**: 2026-02-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.20418v1  

#### Abstract
Graph neural networks (GNNs) have demonstrated superior performance in various applications, such as recommendation systems and financial risk management. However, deploying large-scale GNN models locally is particularly challenging for users, as it requires significant computational resources and e...

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
