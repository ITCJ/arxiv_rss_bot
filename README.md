# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-11 06:15:29 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [RSH-SpMM: A Row-Structured Hybrid Kernel for Sparse Matrix-Matrix Multiplication on GPUs](https://arxiv.org/abs/2603.08734)

**Authors**: Aiying Li, Jingwei Sun, Han Li, Wence Ji, Guangzhong Sun  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.08734v1  

#### Abstract
Sparse Matrix-Matrix Multiplication (SpMM) is a fundamental computation in graph analytics, scientific simulation, and sparse deep learning workloads. However, the extreme irregularity of real-world sparse matrices prevents existing GPU-based methods from maintaining high Tensor Core utilization and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RSH-SpMM: A Row-Structured Hybrid Kernel for Sparse Matrix-Matrix Multiplication on GPUs  
**论文核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
稀疏矩阵-矩阵乘法（SpMM）是图神经网络（GNN）、科学计算和稀疏深度学习中的核心算子。然而，现实世界中的稀疏矩阵具有高度不规则的结构特征（如行长度分布长尾、局部密度快速变化、非零元模式碎片化），导致现有基于GPU的方法难以有效利用 **Tensor Core**（TC），从而限制了性能提升。

具体挑战包括：
- **Tensor Core利用率低**：TC要求密集且对齐的tile输入，而稀疏矩阵常产生大量填充（padding）和低密度tile。
- **负载不均衡**：固定窗口大小或粗粒度划分策略无法适应行间和行内的动态稀疏性变化。
- **执行路径不协调**：混合方法（hybrid）通常在矩阵级或块级进行CUDA/TC路径划分，忽略了细粒度结构一致性。

---

### 🚀 提出的新方法与创新思路

作者提出 **RSH-SpMM** —— 一种面向现代GPU架构的**行结构化混合SpMM框架**，其核心思想是：**以行为单位，自适应地将适合TC处理的“结构一致”行组分配给Tensor Core，而将短小或孤立的行交由轻量级CUDA核处理**。

#### 主要创新点如下：

| 创新模块 | 核心设计 |
|--------|--------|
| **RS-Tile 表示法** | 将稀疏矩阵划分为两个部分：<br>• **TC部分**：连续行被聚合为TC友好的8×8 tile，采用bitmap压缩列索引；<br>• **CUDA残差部分**：极短或结构不兼容的行单独存储，避免污染TC tile密度。 |
| **细粒度混合执行策略** | 引入自适应行划分机制，在构造row window前判断每行是否应进入TC路径，依据两个标准：<br>• 非零元数量（`nnz`）<br>• 加入该行后是否会显著增加列覆盖范围（影响tile密度） |
| **负载均衡流水线内核** | 设计双缓冲流水线，重叠内存加载、解码与MMA计算，隐藏访存延迟；同时支持灵活的adaptive load balancing，仅对“超级长行”进行拆分，其余保持聚合。 |
| **局部性感知重排序（Locality-aware Reordering）** | 在格式转换前引入基于加权Jaccard相似度的kNN图+最小生成树（MST）遍历的重排序方法，增强相邻行之间的结构相似性，减少tile碎片化。 |

---

### 🔍 相比现有方法的优势

| 对比维度 | 传统方法缺陷 | RSH-SpMM改进 |
|--------|-------------|--------------|
| **表示效率** | 固定窗口（如DTC-SpMM）易造成tile稀疏；平衡版本需额外元数据 | RS-Tile天然融合load balancing，无需重复索引结构，元数据更紧凑 |
| **执行灵活性** | 混合方法（如HC-SpMM）划分粒度粗，仍存在大量低效TC tile | 细粒度决策确保只有高质量tile送入TC，残差行走高效CUDA路径 |
| **资源利用率** | TC路径因padding和空转利用率不足（如Acc-SpMM平均仅5.6% MMA周期有用） | 通过选择性剔除破坏性行，TC利用率提升至8.8%，SM吞吐提高18% |
| **通用性与稳定性** | 多数方法对特定稀疏模式敏感，性能波动大 | 在多样化的稀疏分布下均表现稳定，尤其在高异构性矩阵中优势明显 |

---

## 2. 核心实验方法和设置

### 📊 数据集

- **9个代表性真实世界图数据集**，广泛用于GNN研究：
  - `com-amazon`, `ddi`, `DD`, `amazon0505`, `amazon0601`, `Yeast`, `OVCAR-8H`, `YeastH`, `web-BerkStan`
  - 来源：TC-GNN、SNAP、DGL等公开图基准
- **SuiteSparse Matrix Collection** 中的 **512个稀疏矩阵**
  - 筛选条件：≥5K行/列，≥100K非零元
  - 覆盖多个应用领域（工程、物理模拟、网络分析等），用于评估泛化能力

---

### ⚙️ 实验设置与评估指标

| 项目 | 设置说明 |
|------|---------|
| **硬件平台** | • NVIDIA RTX 4090（Ada Lovelace, CC 8.9）<br>• NVIDIA RTX 3090（Ampere, CC 8.6）<br>• 所有测试取预热后的kernel运行时间 |
| **问题配置** | 计算 $ C = A \times B $，其中：<br>• $ A $：稀疏矩阵（CSR输入）<br>• $ B $：稠密特征矩阵 $ \in \mathbb{R}^{n \times d} $<br>• 特征维度 $ d \in \{64, 128, 256, 512\} $ |
| **精度模式** | FP16输入用于TC MMA运算，FP32累加，输出为FP32，保证数值一致性 |
| **评估指标** | • Kernel-level吞吐加速比（vs cuSPARSE）<br>• SM利用率<br>• Tensor Core利用率<br>• 端到端训练时间（GCN案例） |

---

### 🆚 基线方法对比

| 类别 | 基线方法 |
|------|--------|
| **CUDA-core 方法** | • cuSPARSE（厂商库）<br>• Sputnik（warp协作）<br>• RoDe（基于行分布优化） |
| **Tensor Core 方法** | • TC-GNN（GNN专用TC映射）<br>• DTC-SpMM（row-window TC tile）<br>• Acc-SpMM（bitmap导向的8×8 tile） |
| **Hybrid / SpTC 方法** | • HC-SpMM（CUDA/TC双路径）<br>• MP-SpMM（匹配+填充激活Sparse Tensor Cores） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Kernel级SpMM性能（RTX 4090 / 3090）

| 指标 | 结果 |
|------|------|
| **平均加速比（vs cuSPARSE）** | • RTX 4090: **2.35×**<br>• RTX 3090: **2.86×** |
| **在SuiteSparse上的平均加速比** | **3.21×**（超过80%矩阵达到1.24×–8.2×） |
| **最大加速比（vs TC-GNN）** | 达到 **6.13×**（排除崩溃情况） |
| **相比Acc-SpMM/DTC-SpMM平均提升** | 分别达 **1.61×** 和 **1.91×** |
| **相比HC-SpMM平均提速** | **2.10×** |
| **相比MP-SpMM平均优势** | **1.28×**（避免结构约束开销） |

> 💡 图形化结果显示：RSH-SpMM在所有数据集上始终处于性能领先位置，尤其在稀疏结构复杂（如web-BerkStan）时优势更为显著。

---

#### （2）消融实验与组件分析

##### ✅ 存储效率对比（Normalized Space Usage）
- RS-Tile相比平衡版ME-TCF和BitTCF，**元数据体积减少约15.05%**
- 原因：无需为load balancing维护额外的row remapping表或offset数组

##### ✅ 混合核敏感性分析（Row-nnz Threshold）
- 当阈值从0增至4时，TC block平均nnz从16.7升至19.5，row window平均nnz从174升至233
- 超过6后收益饱和，且CUDA路径负载急剧上升 → 性能下降
- **结论**：只需移除少量“异常短行”即可大幅提升tile质量，RSH-SpMM采用自适应小阈值策略最优

##### ✅ GPU Profiling（vs Acc-SpMM）
| 指标 | Acc-SpMM | RSH-SpMM | 提升 |
|------|----------|-----------|-------|
| **SM Throughput (%)** | 28% (median) | 33% | ↑18% |
| **Tensor Core Utilization (%)** | 5.6% (avg) | 8.8% | ↑57% |

> 显示RSH-SpMM更有效地维持了TC流水线的持续占用，减少了idle cycle。

---

##### ✅ 重排序效果比较（Speedup vs 原始顺序）

| 方法 | 平均加速比 |
|------|------------|
| Rabbit Order | 0.87× |
| TCA Reorder (DTC-SpMM) | 1.15× |
| **RSH-SpMM（本文）** | **1.25×**（最高达1.7×） |

- 消融显示：MST排序贡献最大（1.19×），2-opt微调+隔离调整进一步提升至1.25×

---

### 🧪 端到端案例：GCN训练性能（6层GCN，500 epoch）

| 后端 | 平均训练时间（秒） | 相对加速比 |
|------|------------------|------------|
| cuSPARSE | 65.82 | 1.00× |
| PyG | 54.05 | 1.22× |
| TC-GNN | 48.23 | 1.36× |
| DTC-SpMM | 47.10 | 1.40× |
| **RSH-SpMM** | **44.32** | **1.49×** |

> 在`web-BerkStan`等大数据集上，RSH-SpMM实现最快收敛，验证了其在实际模型中的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **细粒度行结构感知是解锁TC潜力的关键**  
   传统的粗粒度划分无法应对局部稀疏性突变，而RSH-SpMM通过逐行评估结构影响，精准分离“可向量化”与“需降级处理”的部分，实现了更高的TC利用率。

2. **RS-Tile是一种高效且自洽的混合表示法**  
   它将load balancing内建于格式构建过程，避免了后期重映射带来的元数据膨胀，提升了整体存储与执行效率。

3. **局部性重排序能显著改善tile连贯性**  
   基于加权Jaccard + MST的排序策略优于Rabbit Order和TCA，能有效暴露潜在的结构连续性，降低fragmentation。

4. **混合执行必须兼顾“质量”与“代价”**  
   不是越多用TC越好，而是要用TC处理真正高效的区域；将低强度行交给轻量CUDA路径反而提升整体并发性和稳定性。

---

### ⚠️ 方法的局限性

- **依赖静态稀疏模式**：当前方法假设稀疏结构不变，若在动态图或训练过程中频繁改变拓扑，则需重新构建RS-Tile格式，带来预处理开销。
- **不适合极高稀疏度极端情况**：当绝大多数行为单非零元时，TC部分可能退化，优势减弱。
- **目前仅支持SpMM（A稀疏，B稠密）**：未扩展至SDDMM或其他稀疏算子组合。

---

### 🔮 未来工作方向

1. **动态SpMM支持**：结合增量编码技术，支持在线更新的稀疏结构。
2. **多算子融合**：将RSH-SpMM与SDDMM、Reduce等操作融合，构建端到端稀疏GNN流水线。
3. **跨设备扩展**：适配多GPU或多节点环境下的分布式SpMM。
4. **支持更多数据类型与稀疏模式**：如block-sparsity、structured pruning等。

---

## ✅ 总结

**RSH-SpMM** 是一项针对现代GPU异构架构设计的高性能SpMM解决方案。它通过 **RS-Tile表示法 + 自适应行划分 + 局部性重排序 + 流水线混合执行** 的协同设计，成功解决了稀疏矩阵不规则性与Tensor Core高效执行需求之间的根本矛盾。

实验表明，RSH-SpMM在多种真实图和通用稀疏矩阵上实现了 **1.27×–6.13× 的加速比**，显著优于现有CUDA-core、Tensor-core及混合方法，并在端到端GNN训练中也展现出卓越的实际效益。

> **一句话总结**：  
> RSH-SpMM通过“以行为本”的细粒度结构感知调度，让每一行都走上最适合它的计算路径，从而最大化GPU异构计算潜能。

</details>

---

### 2. [ConFu: Contemplate the Future for Better Speculative Sampling](https://arxiv.org/abs/2603.08899)

**Authors**: Zongyue Qin, Raghavv Goel, Mukul Gagrani, Risheek Garrepalli, Mingu Lee, Yizhou Sun  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.08899v1  

#### Abstract
Speculative decoding has emerged as a powerful approach to accelerate large language model (LLM) inference by employing lightweight draft models to propose candidate tokens that are subsequently verified by the target model. The effectiveness of this paradigm critically depends on the quality of the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ConFu: Contemplate the Future for Better Speculative Sampling**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
现有的 **speculative decoding** 方法（如 EAGLE 系列）虽然能加速大语言模型（LLM）推理，但其 **draft model** 在生成候选 token 时仅依赖当前前缀（prefix），缺乏对目标模型未来生成方向的感知。这导致随着解码步数增加，预测误差不断累积（error accumulation），draft token 与目标模型分布逐渐偏离，从而降低 token 接受率（acceptance rate），削弱加速效果。

### 🚀 **提出了什么新方法或新思路**
本文提出 **ConFu**（Contemplate the Future），一种全新的 speculative decoding 框架，核心思想是让 draft model “思考未来”——即利用目标模型的中间“思维”信号来指导 draft token 的生成。

#### 主要创新点包括：
1. **Contemplate Tokens 与 Soft Prompts**  
   - 引入 **contemplate token**（也称 pause token）附加到输入序列末尾，促使目标模型在不显著增加计算开销的前提下，输出反映其“当前思考”的连续向量（future prediction vector `f`）。
   - 使用可学习的 **soft prompt tokens** 插入 KV Cache 中，引导目标模型生成更有意义的 future signal，而不改变其原始行为。

2. **基于 MoE 的动态 Contemplate Token 机制**  
   - 不再使用固定的 contemplate token embedding，而是通过 **Mixture-of-Experts (MoE)** 架构，根据上下文动态生成 contemplate token。
   - MoE 以最近被接受 token 的隐藏状态为输入，选择最合适的“专家”embedding，实现对多样化任务（如数学推理、写作等）的自适应 future 预测。

3. **鲁棒训练框架：Anchor Token Sampling + Future Prediction Replication**  
   - **Anchor Token Sampling**：只在部分关键位置插入 contemplate token，减少训练时内存消耗。
   - **Future Prediction Replication**：将某个 anchor token 对应的 future prediction 复用于邻近 token，增强 future signal 的稳定性与泛化能力。

### 🔍 **相比现有方法的优势**
- **更高质量的 draft token**：通过引入 future-oriented 信号，draft model 能更好地对齐目标模型的语义轨迹，减少漂移。
- **更高的接受率和速度提升**：实验证明，在多种任务和设置下，ConFu 显著优于当前最优的 EAGLE-3。
- **无需微调目标模型**：保持目标模型冻结，符合实际部署要求。
- **轻量级开销**：contemplate token 可并行处理，推理成本极低。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
- **训练数据**：ShareGPT 和 UltraChat-200K 指令数据集。
- **评估基准**：**SpecBench**（Xia et al., 2024），涵盖多个下游任务：
  - Writing (WRIT)
  - Question Answering (QA)
  - Summarization (SUMM)
  - Translation (TRANS)
  - Coding (CODE)
  - Mathematical Reasoning (M/R)

### ⚙️ **实验设置**
- **目标模型（Target Model）**：
  - Llama-3.2-3B-Instruct
  - Llama-3.1-8B-Instruct
- **Draft Model 架构**：基于 EAGLE-3 的单层 Transformer 结构，并集成 future token 输入。
- **初始化策略**：从预训练的 EAGLE-3 checkpoint 初始化，确保公平比较。
- **硬件配置**：训练使用 8×NVIDIA H100 GPU；测试使用单张 H100 GPU，batch size = 1。

### 📊 **评估指标**
| 指标 | 含义 |
|------|------|
| **Average Accepted Draft Length (T)** | 每次验证步骤平均接受的 draft token 数量，越高越好 |
| **Speed-up Ratio (SR)** | 相比标准自回归解码的速度提升倍数，越高越好 |

### 🔁 **对比的基线方法**
- **EAGLE-3**（Li et al., 2025）：当前最先进的 speculative decoding 方法，作为主要对比 baseline。
- 其他早期方法如 Medusa、HASS 等已被 EAGLE-3 超越，故未直接比较。

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据（来自 Tables 1 & 2）**

#### 在 **Llama-3.2-3B-Instruct** 上的表现（平均 across 所有任务）：
| 方法 | 平均 Accept Length (T) | 平均 Speed-up Ratio (SR) |
|------|------------------------|--------------------------|
| EAGLE-3 | ~3.8–4.3 | ~1.8–2.0 |
| **ConFu** | **~4.2–4.7** | **~2.0–2.2** |

> ➜ **提升约 8–11% 的接受长度和速度比**

#### 在 **Llama-3.1-8B-Instruct** 上的表现：
| 方法 | 平均 Accept Length (T) | 平均 Speed-up Ratio (SR) |
|------|------------------------|--------------------------|
| EAGLE-3 | ~4.0–4.7 | ~2.3–2.6 |
| **ConFu** | **~4.5–5.2** | **~2.5–2.7** |

> ➜ **同样实现 8–11% 的一致增益**

#### 温度敏感性分析：
- 在 **greedy decoding (T=0)** 下优势最明显（+9–13% 接受长度）
- 高温（T=1.0）下仍有稳定增益，表明 future signal 在随机采样中依然有效

#### Draft Tree 规模影响：
- 在 **30-node** 和 **60-node** draft tree 设置下均表现优越
- 表明 ConFu 可适配不同计算预算场景

---

### 🔍 **消融实验结果（Table 3）**

| 方法变体 | Avg T (8B) | Avg SR (8B) | 分析 |
|---------|------------|-------------|------|
| EAGLE-3 | 4.59 | 2.36 | 基线 |
| ConFu (完整) | **5.01** | **2.69** | 完整模型最佳 |
| w/o MoE | 4.97 | 2.67 | MoE 贡献 +0.05 T, +0.02 SR |
| w/o MoE & Replication | 4.81 | — | 移除两项后性能下降明显 |

> ✅ **结论**：
> - **Dynamic Contemplate Token (MoE)** 提升 context-awareness，带来小幅但稳定的增益。
> - **Future Prediction Replication** 显著增强 future signal 的鲁棒性，是性能提升的关键。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **未来感知显著提升 draft quality**  
   首次将 speculative decoding 与 latent reasoning 中的“连续思维表示”结合，证明让 draft model “预见未来”可有效缓解 error accumulation。

2. **Contemplate Token 是高效 future signal 载体**  
   利用 pause token + soft prompt 的组合，在几乎无额外延迟的情况下提取目标模型的 high-level intent。

3. **ConFu 在多尺度、多任务上全面超越 EAGLE-3**  
   平均提升 **8–11%** 的 token 接受率和推理速度，且在各种温度、draft budget 下保持稳健。

4. **MoE 与 future replication 是关键设计**  
   动态 contemplate token 和鲁棒训练策略共同支撑了高性能。

---

### ⚠️ **方法的局限性**
- **额外计算开销随 draft tree 增大而线性增长**  
  每个 draft node 都需对应一个 contemplate token，当 draft tree 很大时可能影响扩展性。
- **依赖目标模型支持 KV Cache 操作**  
  soft prompt 的注入需要访问内部缓存，可能限制在某些封闭 API 场景中的应用。
- **目前仅限于 decoder-only LLMs**  
  尚未验证在 encoder-decoder 或其他架构上的通用性。

---

### 🔮 **未来工作方向**
1. **优化 contemplate token 的数量**  
   探索稀疏化或共享机制，减少每个 draft node 都插入 contemplate token 的开销。
2. **探索更高效的 future representation 学习方式**  
   如蒸馏、压缩 latent thought 表示。
3. **扩展至多模态 LLM 推理加速**  
   将 future-contemplation 思路应用于图像、音频等生成任务。
4. **与硬件协同设计**  
   开发专用 kernel 支持 contemplate token 的高效并行处理。

---

## ✅ **总结一句话**
> **ConFu 首次将 speculative decoding 与 latent reasoning 融合，通过让 draft model “思考未来”，实现了比 EAGLE-3 高出 8–11% 的推理加速，为 LLM 高效推理开辟了新路径。**

</details>

---

### 3. [LooComp: Leverage Leave-One-Out Strategy to Encoder-only Transformer for Efficient Query-aware Context Compression](https://arxiv.org/abs/2603.09222)

**Authors**: Thao Do, Dinh Phu Tran, An Vo, Seon Kwon Kim, Daeyoung Kim  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.09222v1  

#### Abstract
Efficient context compression is crucial for improving the accuracy and scalability of question answering. For the efficiency of Retrieval Augmented Generation, context should be delivered fast, compact, and precise to ensure clue sufficiency and budget-friendly LLM reader cost. We propose a margin-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**LooComp: Leverage Leave-One-Out Strategy to Encoder-only Transformer for Efficient Query-aware Context Compression**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
在 **Retrieval-Augmented Generation (RAG)** 系统中，随着检索到的上下文长度增加，虽然信息覆盖更全，但也带来了以下挑战：
- **计算开销大**：长上下文显著增加 LLM 的推理成本和延迟；
- **信息冗余与干扰**：无关或低价值内容可能分散模型注意力，降低回答准确性；
- **压缩效率与保真度难以平衡**：现有压缩方法要么速度慢（如生成式），要么精度不足（如简单提取）。

本文旨在解决 **高效且精准的 query-aware 上下文压缩** 问题，在保留关键信息的同时，实现高速、低内存消耗的压缩。

---

### **提出了什么新方法或新思路**

作者提出 **LooComp**，一种基于 **leave-one-out delta (LOO-Δ) 策略** 和 **encoder-only Transformer** 的轻量级上下文压缩框架，其核心创新如下：

#### ✅ **LOO-Δ-based Clue Richness Scoring**
- 不直接对句子进行二分类（相关/不相关），而是通过 **移除单个句子后答案可答性（answerability）的变化量 Δ** 来衡量其重要性。
- 公式定义为：  
  $$
  \Delta_k = f_\theta(q, P) - f_\theta(q, P \setminus \{s_k\})
  $$
  其中 $ f_\theta $ 是一个 encoder-only 模型输出的“线索丰富度”（clue richness）得分。
- **Δ 越大，说明该句越关键**，移除后对理解问题影响越大。

#### ✅ **Margin-based Composite Ranking Loss**
- 设计了一个复合损失函数，包含：
  - **Ranking Loss**：拉大关键句与非关键句之间的 Δ 差距；
  - **Classification Loss (BCE)**：确保完整上下文得分高，空上下文得分低；
  - **Critical Drop Constraint**：强制关键句被删除时 Δ 必须超过某个阈值 $ m_2 $；
  - **Non-critical Stability Constraint**：防止非关键句删除引起过大波动。
- 引入多个 margin 超参（$ m_1, m_2, m_3 $）控制不同类别间的分离程度。

#### ✅ **Adaptive Gap-based Threshold Selection**
- 在推理阶段，不使用固定阈值，而是根据 Δ 分布中的 **最大间隔（largest gap）** 动态确定保留哪些句子。
- 若所有 Δ 都很高，则默认保留更多；若存在明显断层，则在断层处截断。
- 实现了 **自适应压缩率**，无需手动配置压缩比例。

#### ✅ **基于 ModernBERT 的轻量 encoder-only 架构**
- 使用 **ModernBERT**（支持 flash-attention）作为 backbone，相比 decoder-based LLM 更适合此分类任务。
- 显著降低内存占用和推理延迟，适用于实时系统。

---

### **相比现有方法的优势**

| 维度 | LooComp 优势 |
|------|--------------|
| **效率** | 推理速度快（<0.2s @ top-20），远快于 CompAct / Refiner（>4s） |
| **性能保持** | 在多个 QA 数据集上达到 **SOTA 或第二优的 EM/F1** |
| **压缩质量** | 压缩后上下文更紧凑（Rate ~12–14%），优于 EXIT / LongLLMLingua |
| **通用性** | 支持多种 LLM reader（Llama, GPT, Gemini, Kimi-K2），具备良好零样本迁移能力 |
| **架构合理性** | 避免使用 decoder-based LLM 做分类任务，减少不必要的计算开销 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **多跳问答（Multi-hop QA）**：
  - **HotpotQA (HQA)**
  - **2WikiMultihopQA (2Wiki)**
  - **Musique**
- **单跳问答（Single-hop QA）**：
  - **Natural Questions (NQ)**
  - **TriviaQA (TQA)**

训练仅使用 **HotpotQA 的训练子集（约 2.9 万样本）**，其余用于测试，验证泛化能力。

---

### **实验设置和评估指标**

#### 🔧 **RAG Pipeline 设置**
1. **Retriever**：Contriever-MSMARCO + Wikipedia 2018 corpus
2. **Compressor**：LooComp（ModernBERT-large/base）
3. **Reader**：
   - 开源：`Llama-3.1-8B-Instruct`, `Llama-3.3-70B-Instruct`
   - 商业 API：`Gemini-2.5-flash`, `GPT-5-mini`, `Kimi-K2`

#### 📊 **评估指标**
| 类别 | 指标 | 说明 |
|------|------|------|
| **有效性** | **EM (Exact Match)**, **F1** | 衡量最终问答准确率 |
| **压缩效率** | **Compression Ratio (Rate↓)** | 压缩后 token 数 / 原始 token 数 |
| **响应速度** | **Compression Latency (Time↓)** | 端到端压缩耗时（秒） |
| **吞吐能力** | **Questions per Second (QpS)** | 单位时间内可处理的问题数 |
| **资源消耗** | **Peak Memory Usage** | 压缩过程峰值显存 |

测试场景包括：top-5 和 top-20 检索块（chunks）

---

### **基线方法对比**

共比较 **7 种主流压缩器**：

| 方法 | 类型 | 模型 | 特点 |
|------|------|------|------|
| **RECOMP-abs** | Abstractive | T5 | 生成摘要，压缩强但慢 |
| **RECOMP-ext** | Extractive | Dual Contriever | 句子选择，速度快但精度一般 |
| **CompAct** | Iterative Extractive | Mistral-7B | 多轮迭代压缩，效果好但极慢 |
| **Refiner** | Extractive | Llama2-7B | 基于 LLM 的重写优化 |
| **LongLLMLingua (LongLLMLin)** | Prompt Compression | Llama2-7B | 查询感知，动态压缩率 |
| **EXIT** | Context-aware Extractive | Gemma-2B | 利用全文上下文并行判断 |
| **Provence** | Token-level Pruning | Custom | 基于 token 的重要性传播 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据（见 Table 1 & Table 2）**

#### 📈 **综合性能汇总（Table 2 平均值）**

| Method | EM ↑ | F1 ↑ | Time ↓ (s) | Rate ↓ (%) |
|--------|-------|-------|------------|-------------|
| **Ours (LooComp)** | **34.0** | **43.6** | **0.098** | **12.8** |
| LongLLMLin | 32.3 | 41.7 | 0.853 | 41.8 |
| Provence | 32.4 | 42.0 | 0.126 | 17.9 |
| EXIT | 32.0 | 41.3 | 0.921 | 46.2 |
| RECOMP-ext | 29.1 | 38.0 | 0.023 | 30.4 |
| CompAct | 32.2 | 41.6 | 3.670 | 7.7 |

> 💡 **结论**：LooComp 在 **EM/F1 上全面领先**，同时拥有 **第二快的速度** 和 **非常紧凑的压缩率**。

---

#### 🔁 **不同 top-k 下的表现趋势（Fig. 3）**

- 当检索块从 5 增加到 30：
  - LooComp 的 EM 持续提升（30.97 → 32.82），表明能有效利用更多信息；
  - 多数 baseline（如 RECOMP, Refiner）出现性能下降，说明无法处理噪声增长；
- 压缩延迟随 k 几乎线性增长（0.037s → 0.241s），扩展性良好。

---

#### ⚙️ **消融实验结果（Ablation Study）**

##### （1）**Loss Components 消融（Table 3）**

| Variant | EM ↓ | F1 ↓ | 说明 |
|--------|------|------|------|
| Full (完整损失) | 34.0 | 43.6 | 最优 |
| -BCE | 32.6 | 41.2 | 性能显著下降，说明 BCE 对整体建模至关重要 |
| -crit | 33.8 | 42.8 | 次要影响，但仍关键 |
| -BCE-crit | 31.5 | 40.1 | 完全失效，证明各组件协同作用 |

> ✅ **发现**：BCE 和 Ccrit 组件共同保障了模型对“有无线索”的敏感性和稳定性。

---

##### （2）**Backbone 规模对比（Table 4）**

| Backbone | EM ↑ | F1 ↑ | Latency ↓ (ms) | Rate ↓ (%) |
|---------|------|------|----------------|-------------|
| ModernBERT-large | **36.8** | **51.8** | 78.4 | 11.4 |
| ModernBERT-base | 33.2 | 46.5 | **38.1** | 13.8 |

> ✅ **权衡明确**：large 版本精度更高，base 版本速度快近两倍，适合不同部署需求。

---

##### （3）**推理策略对比（Table 5）**

| Strategy | EM ↑ | F1 ↑ | 说明 |
|----------|------|------|------|
| **Adaptive-gap (默认)** | 36.8 | 51.8 | 利用 Δ 分布间隙自动设定阈值 |
| Margin-based | 36.5 | 50.4 | 使用训练时固定的 margin 判断 |

> ✅ **结论**：**adaptive threshold 更鲁棒、泛化更好**，优于硬编码 margin。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Encoder-only 模型足以胜任上下文压缩任务**，无需使用昂贵的 decoder-based LLM。
2. ✅ **LOO-Δ 是一种更合理的重要性度量方式**，比简单的 relevance classification 更贴近实际语义影响。
3. ✅ **动态阈值选择机制（gap-based）能自适应调整压缩强度**，兼顾覆盖率与简洁性。
4. ✅ **LooComp 实现了性能与效率的最佳平衡**：
   - 回答准确率 SOTA；
   - 压缩速度快（接近 RECOMP-ext）；
   - 内存占用小（Fig. 4 显示处于左下角最优区域）；
   - 支持多种 reader 和 retrieval depth。

---

### **方法的局限性**

1. ❗ **依赖 sentence-level 标注**：
   - 当前训练依赖 HQA 中的人工标注关键句；
   - 若使用 LLM 自动生成标签，会引入“LLM-as-judge”的可靠性与成本问题。

2. ❗ **粒度限制在 sentence 层面**：
   - 无法处理长而杂乱的句子内部冗余；
   - 未来可探索 clause 或 phrase-level 剪枝。

3. ❗ **未完全摆脱预训练偏差**：
   - 虽然泛化能力强，但在某些 domain shift 场景下仍有提升空间。

---

### **未来工作方向**

1. 🔄 探索 **finer-grained pruning**（如短语级、子句级）以进一步压缩复杂句子；
2. 🤖 研究 **self-supervised 或 weakly-supervised 训练范式**，减少对人工标注的依赖；
3. 🧠 将 LooComp 与 **multi-hop reasoning pipeline** 更深度集成，实现端到端优化；
4. 📦 推出 **轻量化部署版本**（如蒸馏 + quantization），适配边缘设备或移动端应用。

---

> 🔗 **代码与模型已开源**：https://github.com/thaodod/LooComp

--- 

✅ **总结一句话**：  
**LooComp 通过 LOO-Δ + encoder-only 架构 + 自适应阈值，在保持最高问答准确率的同时，实现了当前最高效的 query-aware 上下文压缩方案，是 RAG 系统中极具实用价值的轻量级替代选择。**

</details>

---

### 4. [Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control](https://arxiv.org/abs/2603.09221)

**Authors**: Peihao Wang, Shan Yang, Xijun Wang, Tesi Xiao, Xin Liu, Changlong Yu, Yu Lou, Pan Li, Zhangyang Wang, Ming Lin, Ren\'e Vidal  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.09221v1  

#### Abstract
Associative memory has long underpinned the design of sequential models. Beyond recall, humans reason by projecting future states and selecting goal-directed actions, a capability that modern language models increasingly require but do not natively encode. While prior work uses reinforcement learnin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control

## 1. 论文的主要贡献和创新点

### 解决的问题
现代语言模型（LLMs）在推理任务中表现受限，其架构本质上依赖于**associative memory**（联想记忆），即通过检索历史上下文来预测下一个 token。这种“System 1”式的快速直觉模式缺乏“System 2”式的**多步规划、长期推理和目标导向决策能力**。尽管强化学习（RL）等方法尝试引入目标导向行为，但这些优化通常作为外部训练过程，**并未内化到模型的前向推理机制中**。

因此，当前 LLM 架构存在一个根本性瓶颈：它们知道要优化什么（what to optimize），但不知道如何在推理时进行规划（how to reason）。

### 提出的新方法与思路
本文提出了一种全新的架构范式——将**推理（reasoning）建模为最优控制问题（optimal control）**，并将其直接嵌入到模型架构中。

核心创新是提出了 **Test-Time Control (TTC) 层**：
- **TTC 层**在推理时对隐状态执行有限时间范围的 **LQR（Linear-Quadratic Regulator）规划**。
- 给定由上下文编码的初始隐状态 $h_0$，TTC 层求解一个马尔可夫决策过程（MDP），其状态转移为线性，成本函数为二次型。
- 将第一步的最优控制动作 $u^*_1$ 解码为下一个 token 的表示，从而实现“先规划，再预测”。

该方法将以下功能统一在一个架构中：
- **Test-Time Memorization**（测试时记忆）
- **World Modeling**（世界建模）
- **Model-Based RL**（基于模型的强化学习）
- **Planning**（规划）

### 相比现有方法的优势
| 方面 | 现有方法（如 TTT） | 本文方法（TTC） |
|------|---------------------|------------------|
| **本质** | 测试时自监督回归（Test-Time Training） | 测试时决策制定（Test-Time Decision Making） |
| **目标** | 更好地编码过去上下文 | 优化未来轨迹和长期目标 |
| **机制** | 参数适应（slow/fast weights） | 隐空间中的显式规划 |
| **可扩展性** | 传统求解器效率低 | 硬件高效的对称辛求解器 |

此外，作者设计了**硬件算法协同设计**的解决方案，确保 TTC 层高效可扩展。

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **Sudoku**  
   - 来源：Palm et al. (2018) 的 10k 个 9×9 数独谜题
   - 划分：9k 训练，1k 测试
   - 任务：从空白数字开始，逐步填满整个棋盘

2. **数学推理任务**
   - **MATH-500**：标准数学问题数据集
   - **AMC**（American Mathematics Competitions）
   - **AIME 2024 & AIME 2025**：更具挑战性的数学竞赛题
   - **OpenThoughts-114K + 自建数据集（800K）**：用于微调，包含多领域可验证的问答对，并生成经过严格验证的 CoT 数据

### 实验设置与评估指标
- **基础模型**：Llama-3-Instruct-7B
- **TTC-Net 架构**：在每 8 个 Transformer block 中插入一个 TTC 层（位于 Attention 和 MLP 之间）
- **训练方式**：持续学习（Continual Learning），在预训练模型上进行监督微调（SFT）
- **TTC 参数初始化**：输出投影矩阵 $W_{\text{out}} = 0$，保证初始行为与原模型一致
- **推理时规划范围（Test-Time Scaling）**：可在推理时动态调整 $T_{\text{test}}$（如 8, 16, 32, 64），以换取更高计算量和更强推理能力

#### 评估指标
- **Sudoku**：
  - 单步准确率（Single-Step Board/Cell Acc）
  - 多步完成准确率（Multi-Step Board/Cell Acc）
- **数学推理**：
  - **Acc@8**：8 次采样中至少有一次正确的比例
  - **Pass@8**：8 次采样中通过自动评分的比例（更严格）

### 基线方法对比
- **纯记忆型架构**：
  - Transformer
  - Mamba / Mamba2
  - GDN
  - Samba
- **混合架构（Memory-based Adapter）**：
  - + Attention
  - + RetNet
  - + Mamba
  - + GDN
  - + MesaNet
- **TTC-Net（本文方法）**

所有对比模型均保持参数量相近，公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 表 1：Sudoku 推理准确率（%）

| 方法 | 单步 Board | 单步 Cell | 多步 Board | 多步 Cell |
|------|------------|-----------|------------|-----------|
| Transformer | 58.50 | 86.54 | 90.10 | 94.08 |
| Mamba | 54.60 | 85.50 | 88.60 | 91.29 |
| GDN | 57.30 | 87.19 | 89.80 | 93.70 |
| **TTC-Net (Ours)** | **61.30** | **90.17** | **93.40** | **97.33** |

> ✅ 在单步和多步任务上全面领先，尤其在多步推理中表现出更强的连贯性和全局规划能力。

#### 表 2：数学推理任务性能（%）

| 模型 | MATH-500 | AMC Acc@8 | AMC Pass@8 | AIME24 Pass@8 | AIME25 Pass@8 |
|------|----------|------------|-------------|----------------|----------------|
| Base model | 25.00 | 6.63 | 31.32 | 0.00 | 0.00 |
| Full Finetuning + MLP | 46.80 | 20.78 | 46.98 | 6.67 | 0.00 |
| + Mamba | 44.80 | 22.29 | 44.58 | 3.33 | 3.33 |
| + GDN | 47.80 | 17.77 | 37.35 | 3.33 | 6.67 |
| **TTC-Net** | **52.80** | **23.34** | **54.22** | **20.00** | **20.00** |

> ✅ 在 MATH-500 上提升高达 **+27.8%**（相比 base model）  
> ✅ 在 AMC 和 AIME 上实现 **2–3× 的 Pass@8 提升**  
> ✅ 基础模型在 AIME 上完全失败（0%），而 TTC-Net 显著激活复杂推理能力

### 消融实验结果（表 3）

| 变体 | MATH-500 (T=8) | MATH-500 (T=16) |
|------|---------------|----------------|
| Time-Homogeneous | 48.40 | 45.70 |
| Fixed Horizon Training | 50.60 | 31.50 |
| Uniform Horizon Sampling | 50.80 | 51.00 |
| **Full Model (Time-Heterogeneous + PLN + 8:1)** | **52.80** | **53.60** |

- **Time-Heterogeneous 参数化至关重要**：允许 $A_t, B_t, Q_t, R_t$ 随时间变化，显著优于共享参数的时间齐次版本。
- **Poisson Log-Normal (PLN) 训练策略更优**：相比固定或均匀采样，PLN 能更好泛化到不同推理时长，且训练成本更低。
- **8:1 Attention:TTC 插入比例最佳**：过多 TTC 层不经济，不如增加测试时规划长度。

---

## 4. 关键结论和发现

### 主要发现
1. **推理可以被形式化为最优控制问题**，并通过 TTC 层内化到 LLM 架构中，形成“System 1 + System 2”的统一架构。
2. **TTC 层实现了真正的“测试时规划”**，而非仅仅是“测试时训练”，使模型能在隐空间中模拟未来轨迹并做出目标导向决策。
3. **硬件高效的对称辛求解器（symplectic LQR solver）** 成功解决了传统 Riccati 迭代无法并行化的难题，支持高吞吐、低内存开销的大规模部署。
4. **TTC 支持原生的 Test-Time Scaling**：通过增大推理时规划范围 $T_{\text{test}}$，可动态分配更多计算资源以换取更强推理能力（见图 5）。
5. **TTC-Net 在多个硬推理任务上显著超越现有方法**，尤其是在需要长期约束传播的任务（如 Sudoku）和复杂数学竞赛题（如 AIME）上表现突出。

### 方法的局限性
- **理论解释不足**：多个 TTC 层在深层网络中如何协同建模动态系统尚不清楚。
- **表达能力限制**：目前仅使用线性动力学和二次成本函数，虽高效但可能不足以捕捉所有复杂推理路径。
- **尚未在超大规模模型上全面验证**：实验主要基于 Llama-3-7B，更大模型上的效果有待探索。

### 未来工作方向
- 探索更丰富的隐空间动态建模范式（如非线性 SSM 或神经 ODE）。
- 将 TTC 与其他推理机制（如 Tree-of-Thought, Search）结合。
- 在全阶段训练（pretraining, SFT, RLHF）中统一整合 TTC 目标。
- 扩展至视觉、机器人等跨模态决策任务。

---

> 🔚 **总结一句话**：  
> 本文提出了 **TTC-Net**，首次将**最优控制**作为可学习、可扩展的模块内化到 LLM 架构中，实现了从“记忆驱动预测”到“规划驱动推理”的范式跃迁，在 Sudoku 和数学竞赛题上取得了显著突破，并为 LLM 的原生推理能力提供了新的设计蓝图。

</details>

---

### 5. [PPO-Based Hybrid Optimization for RIS-Assisted Semantic Vehicular Edge Computing](https://arxiv.org/abs/2603.09082)

**Authors**: Wei Feng, Jingbo Zhang, Qiong Wu, Pingyi Fan, Qiang Fan  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.09082v1  

#### Abstract
To support latency-sensitive Internet of Vehicles (IoV) applications amidst dynamic environments and intermittent links, this paper proposes a Reconfigurable Intelligent Surface (RIS)-aided semantic-aware Vehicle Edge Computing (VEC) framework. This approach integrates RIS to optimize wireless conne...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# PPO-Based Hybrid Optimization for RIS-Assisted Semantic Vehicular Edge Computing 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**动态环境下延迟敏感的车联网（Internet of Vehicles, IoV）应用**所面临的挑战，解决以下关键问题：
- **无线链路不稳定**：城市环境中障碍物和多径衰落导致车辆与路边单元（RSU）之间的通信质量下降，影响服务质量（QoS）。
- **高延迟瓶颈**：传统云计算架构难以满足自动驾驶等任务对超低时延和高可靠性的要求。
- **资源联合优化困难**：在语义通信（Semantic Communication）、可重构智能表面（RIS）辅助传输与边缘计算协同场景下，存在**非凸、高维、耦合性强的联合优化难题**。

### 提出的新方法与新思路
提出了一种**RIS辅助的语义感知车载边缘计算（RIS-aided Semantic-aware VEC）框架**，并设计了一个**两层混合优化方案（two-tier hybrid scheme）**：
- **上层采用 Proximal Policy Optimization (PPO)**：用于决策离散变量——RIS相移矩阵（phase shifts）和语义符号数量（number of semantic symbols）。
- **下层采用 Linear Programming (LP)**：在给定RIS配置和语义压缩参数的前提下，求解最优的任务卸载比例（offloading ratios），实现连续变量的高效优化。

该方法实现了**跨层联合优化**，将物理层（RIS调控信道）、语义层（语义特征提取与压缩）与应用层（任务卸载调度）深度融合。

### 相比现有方法的优势
| 维度 | 本文优势 |
|------|--------|
| **系统架构** | 首次将 RIS、语义通信与 VEC 在动态 IoV 场景中统一建模，支持三路径任务分割（本地执行、V2I 到 RSU、V2V 到服务车辆 SV）。 |
| **优化机制** | 提出 PPO-LP 混合架构，避免直接用 DRL 处理高维连续-离散混合动作空间带来的训练不稳定性与“维度灾难”。 |
| **灵活性与适应性** | 将语义符号数 $v$ 作为显式决策变量，结合实时 SINR 和语义相似度进行自适应语义压缩，提升抗干扰能力。 |
| **性能表现** | 显著优于 GA、QPSO 等启发式算法，在多种负载条件下保持稳定低延迟。 |

---

## 2. 核心实验方法和设置

### 数据集
本研究为**仿真驱动型工作**，未使用真实世界公开数据集。所有任务生成基于以下设定：
- 车辆任务到达服从 **Poisson 分布**；
- 每个时间槽生成 **0.4 Mbit 文本任务**；
- 使用预训练的 **DeepSC 模型** 进行语义编码与解码，复用其语义参数映射表（来自文献 [57]）。

### 实验设置
| 参数 | 设置值 |
|------|-------|
| 车辆数量 $K$ | 15 → 最多测试至 30 辆 |
| RIS 规模 | $6\times6$ 元素（默认），扩展至 $4\times4$ 至 $10\times10$ |
| RSU 位置 | (-10, 150, 25)，配备大规模 MIMO |
| RIS 位置 | (10, 175, 25)，部署于建筑墙面 |
| 车速 | 固定为 20 m/s |
| 带宽 $W$ | 360 kHz |
| 发射功率 $P_{k,j}$ | 可变，范围 0.1–0.3 W |
| 语义知识库 | 所有节点共享静态 DeepSC 模型，暂不考虑动态更新开销 |

### 评估指标
- **平均端到端延迟（Average End-to-End Latency）**：主要性能指标。
- **通信延迟分解**：分别统计 V2I 与 V2V 链路的传输延迟。
- **延迟分布稳定性**：通过箱线图（boxplot）分析不同密度下的延迟波动情况。
- **算法收敛性**：观察 PPO 的累计奖励变化趋势。

### 基线方法对比
- **Genetic Algorithm (GA)**：经典进化算法，用于全局搜索。
- **Quantum-behaved Particle Swarm Optimization (QPSO)**：量子行为粒子群优化，具备更强全局探索能力。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 场景 | 本文方法（PPO-LP） | 对比基线（GA / QPSO） | 性能增益 |
|------|------------------|--------------------|---------|
| 默认设置（15车，6×6 RIS） | ~0.076 s | ~0.127–0.130 s | **降低约 40%–50% 延迟** |
| 高密度场景（30辆车） | 延迟增长平缓 | GA/QPSO 性能急剧恶化 | 展现出强**可扩展性** |
| 不同发射功率（0.1W→0.3W） | 延迟持续下降且曲线平稳 | 存在局部震荡 | 表明**更强鲁棒性** |
| 不同 RIS 规模（4×4→10×10） | 随规模增大延迟显著下降 | GA/QPSO 在大尺寸时陷入局部最优 | 体现对“维度灾难”的克服能力 |

### 与基线方法的对比结果
- 如 **Figure 4** 所示，在整个发射功率范围内，PPO 方案始终领先 GA 和 QPSO 约 **40%-50% 的延迟减少**。
- **Figure 5** 显示，PPO 在 V2V 链路上尤其出色，延迟稳定在 **0.030 秒左右**，相比基线提升超过 **45%**。
- **Figure 6 与 Figure 9** 的箱线图表明：
  - GA 和 QPSO 在车辆数增加或 RIS 规模扩大时出现大量异常值（outliers），说明易陷入局部最优；
  - PPO 分布紧凑，上下界差距小，表现出优异的**稳定性与公平性**。

### 消融实验结果（隐含分析）
虽然文中未明确列出消融实验表格，但从多组对比中可得出以下结论：
- **RIS 的作用**：随着反射单元数量增加，系统延迟下降，验证了 RIS 对信道重构的有效性。
- **语义通信的作用**：通过控制语义符号数 $v$ 实现自适应压缩，是实现 **40%-50% 性能提升的关键驱动力之一**（见第5节总结）。
- **PPO-LP 架构的作用**：将连续卸载问题交给 LP 求解器处理，使 PPO 能专注于复杂的离散决策空间，有效规避了“维度灾难”，保证了训练效率与最终性能。

---

## 4. 关键结论和发现

### 主要发现
1. **RIS + Semantic Communication 是降低 VEC 延迟的有效组合**：
   - RIS 改善信道质量，提供虚拟直视路径（virtual LoS）；
   - 语义通信减少冗余数据传输，提高有效信息吞吐率（semantic rate）。

2. **PPO-LP 两层混合架构具有优越的实用性与可扩展性**：
   - 上层 DRL（PPO）擅长处理复杂环境下的策略学习；
   - 下层 LP 精确求解瞬时最优卸载比例，形成闭环反馈。

3. **系统在高密度交通场景下仍能维持低延迟**：
   - 即使在 **30 辆车**的拥堵场景中，PPO 方案依然保持较低且稳定的延迟增长速率，展现出强大的**实际部署潜力**。

4. **语义符号数 $v$ 作为决策变量增强了系统弹性**：
   - 在信道条件差时可通过调整 $v$ 维持较高语义相似度 $\delta$，从而保障任务完整性。

### 方法的局限性
- **静态语义知识库假设**：当前模型依赖固定的 DeepSC 编码器，未考虑动态任务导致的知识库更新延迟与同步开销。
- **单 RIS 设置**：仅考虑单一 RIS 覆盖区域，尚未涉及多 RIS 协同覆盖广域道路网络。
- **理想化移动模型**：车辆速度恒定，未引入加减速、变道等更真实的驾驶行为。
- **集中式训练架构**：依赖 RSU 作为中心代理进行训练，可能带来通信开销与隐私问题。

### 未来工作方向
1. **研究动态语义知识库的更新机制**，支持在线语义模型演化与轻量化同步。
2. **探索多 RIS 协同部署与联合波束成形策略**，以增强广域覆盖能力。
3. **向分布式/联邦式 DRL 架构演进**，提升系统的去中心化程度与隐私保护能力。
4. **集成更多模态任务**（如图像、语音），构建多模态语义通信框架，进一步拓展应用场景。

--- 

> ✅ **总结一句话**：  
> 本文提出了一种融合 RIS、语义通信与 VEC 的新型智能车联网架构，并通过 PPO-LP 混合优化框架实现了高达 **40%-50% 的端到端延迟降低**，在高密度、动态环境下展现出卓越的稳定性与可扩展性，为下一代低时延高可靠 IoV 系统提供了重要技术路径。

</details>

---

### 6. [A Multi-Prototype-Guided Federated Knowledge Distillation Approach in AI-RAN Enabled Multi-Access Edge Computing System](https://arxiv.org/abs/2603.09727)

**Authors**: Luyao Zou, Hayoung Oh, Chu Myaet Thwal, Apurba Adhikary, Seohyeon Hong, Zhu Han  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.09727v1  

#### Abstract
With the development of wireless network, Multi-Access Edge Computing (MEC) and Artificial Intelligence (AI)-native Radio Access Network (RAN) have attracted significant attention. Particularly, the integration of AI-RAN and MEC is envisioned to transform network efficiency and responsiveness. There...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对在 **AI-RAN enabled MEC system** 中部署 **Federated Learning (FL)** 所面临的关键挑战——**非独立同分布 (non-IID) 数据** 导致的模型性能下降问题。传统 FL 在客户端数据分布差异显著时，本地更新容易发散，导致全局模型收敛缓慢且精度降低。

### 提出的新方法：MP-FedKD
为解决上述问题，作者提出了一种名为 **Multi-Prototype-Guided Federated Knowledge Distillation (MP-FedKD)** 的新方法，其核心创新点包括：

- **多原型生成机制 (Multi-Prototype Strategy)**：
  - **问题**：现有基于单原型的方法（如 FedProto）通过平均同类样本的嵌入向量来生成一个单一原型，这会丢失类内多样性信息。
  - **创新**：采用 **Conditional Hierarchical Agglomerative Clustering (CHAC)** 方法，在每个客户端为每个类别生成多个原型（multi-prototypes），从而更全面地捕捉类内特征的复杂结构。

- **自知识蒸馏 (Self-Knowledge Distillation, SKD)**：
  - **问题**：传统的知识蒸馏 (KD) 需要预先训练一个强大的教师模型，增加了系统开销和实现难度。
  - **创新**：利用前一轮的本地模型作为“教师”来指导当前轮次的“学生”模型训练，无需额外的教师网络，有效缓解了 non-IID 问题并促进了知识迁移。

- **原型对齐机制 (Prototype Alignment, PA)**：
  - **问题**：在聚合生成全局原型时，简单的平均操作同样会造成信息损失。
  - **创新**：设计了一个新的 **PA loss**，让本轮的全局原型学习从上一轮本地模型产生的局部嵌入向量中学习，保留了历史信息，减少了因平均操作带来的信息损失。

- **新型损失函数 (LEMGP Loss)**：
  - **创新**：设计了一个结合 **COREL loss** 思想的 **LEMGP loss**，包含吸引项 (attractive) 和排斥项 (repulsive)。该损失函数显式地拉近本地嵌入与同类全局原型的距离，同时推远与其他类全局原型的距离，增强了模型的判别能力。

### 相比现有方法的优势
- **信息更丰富**：多原型策略相比单原型能更好地建模类内异质性。
- **无需外部教师**：SKD 机制避免了复杂的教师模型设计和训练。
- **减少聚合损失**：PA 机制和 LEMGP loss 有效缓解了因平均聚合导致的信息损失。
- **端到端优化**：所有组件（SKD, CHAC, PA, LEMGP）协同工作，形成一个统一的优化框架。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在六个数据集上进行，涵盖同构和异构场景：
- **同构数据集 (Same Domain)**：
  - `CIFAR-10` (32x32 彩色图像，10类)
  - `MNIST` (28x28 灰度图像，10类)
  - `Fashion-MNIST` (28x28 灰度图像，10类)
  - `EuroSAT` (64x64 卫星图像，10类)
- **异构数据集 (Distinct Domain)**：
  - `M+F`：`MNIST` 和 `Fashion-MNIST` 的混合。
  - `C+E`：`CIFAR-10` 和 `EuroSAT` 的混合。

### 实验设置和评估指标
- **Non-IID 设置**：
  - **Type 1 (Same Domain)**：使用 **Dirichlet 分布 (Dir={0.3, 0.5, 0.7, 0.9})** 在同一域内的客户端间划分数据，Dir 值越小，数据异质性越强。
  - **Type 2 (Distinct Domain)**：将客户端分为两组，每组分别分配来自不同域（如 M+F 或 C+E）的数据，并在组内应用 Dirichlet 分布。
- **模型架构**：在客户端使用 `S-CNN`, `ResNet-8`, `ResNet-10` 进行测试，最终选择 `ResNet-10` 作为主干网络。
- **超参数**：`batch_size=32`, `learning_rate=0.001`, `local_epochs=5`, `communication_rounds=50`。
- **评估指标**：
  - 准确率 (`Accuracy`)
  - 平均准确率 (`Average Accuracy`)
  - 均方根误差 (`RMSE`)
  - 平均绝对误差 (`MAE`)
  - 宏平均 F1 分数 (`Macro F1 Score`)

### 基线方法对比
- **主流 FL 方法**：`FedProx`, `FedProto`, `MOON`, `FedAS`, `FedALA`, `E-FPKD`。
- **消融对比**：将提出的 `CHAC` 替换为 `K-Means` 聚类，以验证 CHAC 的优越性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **总体表现**：MP-FedKD 在所有数据集和 Non-IID 设置下均取得了最佳性能。
- **具体提升示例**：
  - 在 `EuroSAT` 数据集上（10个客户端），相比基线方法，准确率提升了 **1.98% ~ 28.70%**。
  - 在 `CIFAR-10` 数据集上（20个客户端），准确率是 `FedProx` 的 **2.01倍**，是 `MOON` 的 **1.48倍**。
  - 在 `M+F` 异构数据集上，MP-FedKD 的准确率达到了 **93.92%**。

### 与基线方法的对比结果
- **误差分析**：MP-FedKD 在 `RMSE` 和 `MAE` 上显著低于 `FedProx` 和 `FedProto`。例如，在 `EuroSAT` 上，其 `RMSE` 比 `FedProx` 低约 **1.62倍**，`MAE` 低约 **2.54倍**。
- **可扩展性 (Scalability)**：在不同客户端数量（10, 20, 50）下，MP-FedKD 始终保持最高准确率，证明了其良好的可扩展性。
- **鲁棒性 (Robustness)**：在训练过程中，MP-FedKD 展现出更快的收敛速度和更高的最终准确率，波动更小，尤其是在收敛阶段优势明显。

### 消融实验结果
- **CHAC vs. K-Means**：使用 `CHAC` 的 MP-FedKD 在 `M+F` 和 `Fashion-MNIST` 上的准确率分别比使用 `K-Means` 的版本高出约 **1.03倍** 和 **1.02倍**，验证了层次聚类能提供更多信息。
- **PA 和 LEMGP Loss 的作用**：
  - 移除 `PA` (w/o PA)：在 `CIFAR-10` 上平均准确率下降 **0.72%**。
  - 移除 `LEMGP` (w/o LEMGP)：在 `CIFAR-10` 上平均准确率下降 **1.58%**。
  - 结果表明，`PA` 和 `LEMGP` loss 对于模型性能至关重要。
- **不同超参数**：在不同的 `learning_rate` 和 `batch_size` 下，MP-FedKD 始终优于所有基线方法，表现出强大的鲁棒性。

---

## 4. 关键结论和发现

### 主要发现
1.  **多原型优于单原型**：为每个类别生成多个原型（通过 CHAC）能够有效捕获类内多样性，显著优于仅使用单一平均原型的方法。
2.  **自知识蒸馏高效可行**：利用历史模型作为教师进行 SKD 是一种简单而有效的处理 non-IID 数据的方式，无需复杂的外部教师网络。
3.  **减少聚合信息损失至关重要**：无论是本地原型生成还是全局原型聚合，简单的平均操作都会造成信息损失。通过 `PA` 机制和 `LEMGP` loss 可以有效缓解这一问题。
4.  **MP-FedKD 综合性能卓越**：所提出的 MP-FedKD 框架在准确性、鲁棒性和可扩展性方面全面超越了现有的先进基线方法。

### 方法的局限性
- **计算开销增加**：CHAC 聚类算法的时间复杂度较高（O(|D_m,c|^3)），相比简单的平均操作，会增加客户端的本地计算负担。
- **超参数敏感性**：需要设定聚类数量 `ξ` 等超参数，虽然实验中设为3效果较好，但在不同任务上可能需要调整。
- **通信开销**：传输多个原型（而非单个）可能会略微增加客户端与服务器之间的通信负载。

### 未来工作方向
- **优化聚类效率**：探索更高效的聚类算法或近似方法，以降低 CHAC 的计算复杂度。
- **动态原型管理**：研究如何动态决定每个类别的最优原型数量，而不是固定为常数。
- **理论分析**：为 MP-FedKD 框架提供更深入的收敛性理论分析。
- **实际部署**：在真实的 AI-RAN 和 MEC 环境中进行原型部署和性能测试。

</details>

---

### 7. [A Unified Hierarchical Multi-Task Multi-Fidelity Framework for Data-Efficient Surrogate Modeling in Manufacturing](https://arxiv.org/abs/2603.09842)

**Authors**: Manan Mehta, Zhiqiao Dong, Yuhang Yang, Chenhui Shao  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.09842v1  

#### Abstract
Surrogate modeling is an essential data-driven technique for quantifying relationships between input variables and system responses in manufacturing and engineering systems. Two major challenges limit its effectiveness: (1) large data requirements for learning complex nonlinear relationships, and (2...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Unified Hierarchical Multi-Task Multi-Fidelity Framework for Data-Efficient Surrogate Modeling in Manufacturing

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对制造系统中**代理模型（surrogate modeling）**面临的两大挑战：
1. **数据需求大**：学习复杂的非线性输入-输出关系需要大量高质量数据，而高保真（high-fidelity）实验或仿真成本高昂、耗时长。
2. **异构数据源**：实际制造过程中数据来自多种不同精度和分辨率的来源（如高低分辨率传感器、模拟器、测量设备），导致数据具有多保真度（multi-fidelity）特性。

现有方法通常将**多任务学习（MTL）** 和 **多保真建模（multi-fidelity modeling）** 分开处理：
- MTL 能共享相似任务间的信息以减少单个任务的数据需求，但常假设所有数据质量一致（homogeneous noise）；
- 多保真方法（如 Stochastic Kriging, SK）能融合不同保真度数据，但通常只用于单一任务。

因此，缺乏一个统一框架同时利用**跨任务相似性**与**保真度相关的不确定性**。

---

### 🆕 提出的新方法：H-MT-MF 框架
作者提出了一种全新的 **Hierarchical Multi-Task Multi-Fidelity (H-MT-MF)** 框架，基于 **Gaussian Process (GP)** 构建代理模型，其核心思想是：

> 将每个任务的响应分解为两个部分：
> 1. **任务特定的全局趋势（task-specific global trend）**
> 2. **跨任务联合学习的局部残差变异性（residual local variability）**

该框架采用**分层贝叶斯建模（hierarchical Bayesian formulation）**，并引入**异方差随机克里金（heteroscedastic Stochastic Kriging, SK）** 来显式建模不同保真度下的内在噪声（intrinsic uncertainty）。

---

### 🔍 相比现有方法的优势
| 特性 | H-MT-MF | 传统MTL | 传统SK |
|------|--------|---------|--------|
| 支持多任务学习（MTL） | ✅ | ✅ | ❌ |
| 支持多保真数据融合 | ✅ | ❌ | ✅ |
| 统一建模交叉任务相关性与保真度噪声 | ✅ | ❌ | ❌ |
| 提供预测不确定性量化 | ✅ | ✅ | ✅ |
| 可扩展至任意数量的任务、设计点和保真层级 | ✅ | 有限制 | 单任务 |

> **首次实现了 MTL 与 multi-fidelity modeling 在统一概率框架下的耦合建模**。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
1. **合成 1D 示例（Synthetic 1D Example）**
   - 包含 3 个任务（tasks），每个任务定义在区间 $[0, 20]$ 上。
   - 函数形式为：$y_l = a + bx + \sin(x/5) + c\sin(4x/5)$，其中 $(a,b,c)$ 不同体现“相似但不相同”属性。
   - 数据采集方式：
     - 每个任务采样 10 个位置，每处有 3 次重复测量。
     - 使用两种测量工具：
       - 高分辨率 gauge：$\sigma_{\text{high-res}} = 0.05$
       - 低分辨率 gauge：$\sigma_{\text{low-res}} = 0.2$

2. **真实案例研究：发动机表面形状预测（Engine Surface Shape Prediction）**
   - 数据来源：福特汽车工厂的三个相似发动机缸体表面。
   - 输入：空间坐标 $(x, y)$；输出：表面高度 $Z(x)$。
   - 测量方式：
     - 每个表面随机选取 50 个点进行测量。
     - 其中 25 个用高精度 gauge（repeatability $\sigma^2 = 0.1\%\sim2.5\%$）
     - 另外 25 个用低精度 gauge（$\sigma^2 = 0.5\%\sim12.5\%$）
   - 预测目标：对每个表面上的 15,000 个未观测点进行插值预测。

---

### 🧪 实验设置与评估指标

#### ✅ 评估指标
- **RMSE（Root Mean Squared Error）**：衡量预测准确性。
- **ΔRMSE**：相对于基线方法的百分比提升：
  $$
  \Delta\text{RMSE}_{\text{baseline}} = \frac{\text{RMSE}_{\text{baseline}} - \text{RMSE}_{\text{H-MT-MF}}}{\text{RMSE}_{\text{baseline}}} \times 100\%
  $$

#### ✅ 基线方法对比
| 方法 | 是否支持MTL | 是否支持Multi-Fidelity |
|------|-------------|-----------------------|
| **H-MT-MF**（本文） | ✅ | ✅ |
| **EG-MTL** [12] | ✅ | ❌（假设同方差噪声） |
| **Stochastic Kriging (SK)** [44] | ❌ | ✅（独立训练各任务） |

> 所有方法均使用相同的基核函数（squared exponential kernel）和参数估计策略（maximum likelihood / EM algorithm）。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 合成1D实验结果
- H-MT-MF 成功捕捉到复杂非线性函数形态。
- 利用 MTL 实现了**跨任务信息迁移**：
  - 例如 Task 2 在 $[0,5]$ 区域无观测点，但仍能通过 Task 1 和 Task 3 的数据实现准确预测。
- **后验方差（prediction uncertainty）也实现跨任务传播**：在一个任务中观测某区域可降低其他任务在同一区域的预测不确定性。

#### ✅ 发动机表面预测结果（见 Table 2 和 Figure 4）
- 在所有 9 种不同的高低保真度组合下，**H-MT-MF 均优于 EG-MTL 和 SK**。
- 平均性能提升显著：
  - 相比 **EG-MTL**：平均提升 **10.76% ~ 20.44%**
  - 相比 **SK**：平均提升 **13.09% ~ 24.50%**

| 高保真 $\sigma^2$ (%) | 低保真 $\sigma^2$ (%) | ΔRMSE vs EG-MTL (%) | ΔRMSE vs SK (%) |
|------------------------|------------------------|----------------------|------------------|
| 0.1                    | 0.5                   | 10.76               | 14.17           |
| 0.1                    | 12.5                  | 16.79               | 19.08           |
| 2.5                    | 12.5                  | 15.00               | 12.74           |

> ⚠️ 注意：随着测量噪声增大，EG-MTL 性能急剧下降（因其无法区分不同保真度），而 H-MT-MF 和 SK 更鲁棒。

#### ✅ 消融分析（隐含）
虽然没有明确命名“ablation study”，但从对比实验可得出以下结论：
- 若仅启用 MTL（如 EG-MTL）而不考虑保真度差异 → 在高噪声场景下失效；
- 若仅启用 multi-fidelity（如 SK）而不共享任务信息 → 无法利用跨任务相似性，预测效率低；
- **只有同时结合 MTL + multi-fidelity（即 H-MT-MF）才能实现最优且稳健的表现**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **统一建模范式有效**：H-MT-MF 成功地将 MTL 与 multi-fidelity modeling 统一起来，在理论上和实践中都优于分离建模的方法。
2. **信息迁移能力强大**：即使某些任务在特定区域内没有观测数据，也能借助其他任务的数据进行可靠预测。
3. **不确定性建模更真实**：通过 heteroscedastic noise modeling，能够更准确反映不同测量设备带来的误差水平，提升模型可信度。
4. **工程实用性高**：适用于多分辨率 metrology、多源传感、混合仿真等典型制造场景。

---

### ⚠️ 局限性
1. **计算复杂度较高**：EM 算法的复杂度约为 $O(k_1 k_2 m n^3)$，当任务数 $m$ 或样本量 $n$ 很大时可能受限。
2. **依赖预估超参数**：需通过初步实验设定 NIW 分布的 hyperparameters（如 $\nu, \lambda$），全贝叶斯推断尚未实现。
3. **静态建模**：当前框架为静态空间建模，未考虑时间演化过程（如工具磨损、动态退化）。

---

### 🔮 未来工作方向
1. **拓展至 Spatiotemporal 过程建模**：
   - 应用于刀具状态监测、生产性能预测等随时间变化的系统。
2. **智能采样策略设计（Adaptive Sampling）**：
   - 结合 **Bayesian Optimization** 或 **Hierarchical Genetic Algorithms**，优化以下决策：
     - 选择哪个任务采样？
     - 在该任务中选择哪个位置？
     - 使用何种保真度（分辨率/成本）进行测量？
3. **在线更新与增量学习机制**：
   - 支持新任务动态加入，实现实时模型更新。
4. **与其他深度模型集成**：
   - 探索 Deep Gaussian Processes 或 Neural Processes 与 H-MT-MF 的结合潜力。

---

## ✅ 总结
本论文提出的 **H-MT-MF 框架** 是制造领域代理建模的一项重要进展。它首次实现了 **multi-task learning** 与 **multi-fidelity modeling** 的深度融合，不仅提升了预测精度（最高提升达 **23%**），还增强了模型对异构数据的适应性和不确定性量化能力。该方法为智能制造中的数字孪生、工艺优化、质量控制等应用提供了强有力的统计建模基础。

</details>

---

### 8. [MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games](https://arxiv.org/abs/2603.09022)

**Authors**: Yunfei Xie, Kevin Wang, Bobby Cheng, Jianzhu Yao, Zhizhou Sha, Alexander Duffy, Yihan Xi, Hongyuan Mei, Cheston Tan, Chen Wei, Pramod Viswanath, Zhangyang Wang  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.09022v1  

#### Abstract
Multi-turn, multi-agent LLM game evaluations often exhibit substantial run-to-run variance. In long-horizon interactions, small early deviations compound across turns and are amplified by multi-agent coupling. This biases win rate estimates and makes rankings unreliable across repeated tournaments. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MEMO: Memory-Augmented Model Context Optimization for Robust Multi-Turn Multi-Agent LLM Games

## 1. 论文的主要贡献和创新点

### 解决了什么问题
多轮、多智能体 LLM 游戏评估存在显著的**运行间方差**（run-to-run variance）。由于早期微小偏差在长视野交互中不断累积，并通过多智能体耦合被放大，导致胜率估计有偏、排名不可靠。此外，**提示词选择**（prompt choice）会进一步加剧该问题，产生不同的有效策略，使得模型比较缺乏可重复性和公平性。

### 提出了什么新方法或新思路
提出 **MEMO**（Memory-augmented MOdel context optimization），一种无需更新模型权重的自对弈（self-play）框架，通过优化推理时上下文来提升鲁棒性。其核心是将**保留**（Retention）与**探索**（Exploration）相结合：
- **Retention（保留）**：维护一个持久的**记忆库**（memory bank），以 CRUD（创建、读取、更新、删除）操作存储从自对弈轨迹中提炼出的结构化洞察，并将其作为先验注入后续游戏。
- **Exploration（探索）**：运行锦标赛式的提示词进化，利用 **TRUESKILL** 进行不确定性感知的选择，并使用**优先级重放**（prioritized replay）机制重新访问稀有且决定性的状态。

### 相比现有方法的优势
- **稳定性更高**：相比固定提示词或仅进行提示优化的方法，MEMO 显著降低了运行间的方差（RSE 降低至 6.4%），使排名更稳定。
- **效率更高**：相比强化学习（RL）方法，MEMO 在达到相当甚至更好性能的同时，所需的游戏数量少 19 倍（2,000 vs 38,000）。
- **通用性强**：在谈判和不完全信息游戏中提升最大，而这些正是传统 RL 难以处理的场景。
- **无梯度优化**：整个过程不依赖于模型权重更新，是一种纯粹的上下文优化方法。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验基于五个文本游戏，来自 **TextArena** 和 **SPIN-Bench** 基准套件，分为三类：
- **谈判游戏**（Negotiation）：`SimpleNegotiation`, `TwoDollar`
- **不完全信息游戏**（Imperfect Information）：`KuhnPoker`, `Briscola`
- **完全信息游戏**（Perfect Information）：`SimpleTak`

### 实验设置和评估指标
- **基础模型**：`GPT-4o-mini` 和 `Qwen-2.5-7B-Instruct`。
- **优化流程**：运行 5 代（generations），每代 8 个候选上下文，每个候选进行 50 场自对弈，总计 **2,000 场自对弈游戏**。
- **评估协议**：进行 3 次独立的优化运行。最终的上下文在固定的对手池（`Grok-4-Fast-Non-Reasoning`, `Gemini-2.5-Flash-Lite`, `Qwen3-235B-A22B-Instruct-2507`）上进行评估，每轮 50 场游戏。
- **核心评估指标**：
  - **平均胜率**（Mean Win Rate）
  - **相对标准误差**（Relative Standard Error, RSE），用于衡量运行间稳定性。

### 基线方法对比
- **静态提示**（STATIC）：`baseline`, `CoT` (Chain-of-Thought), `ToT` (Tree-of-Thought)
- **提示优化**（PROMPT）：`TextGrad`, `MIPRO`, `GEPA`
- **强化学习**（RL）：`UnstableBaseline`, `SPIRAL`

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 `GPT-4o-mini` 上，MEMO 将平均胜率从基线的 **25.1%** 提升至 **49.5%**；在 `Qwen-2.5-7B-Instruct` 上，从 **20.9%** 提升至 **44.3%**。

### 与基线方法的对比结果
| 模型 | 方法 | 平均胜率 | 平均 RSE |
| :--- | :--- | :--- | :--- |
| GPT-4o-mini | baseline | 25.1% | 44.9% |
| GPT-4o-mini | CoT | 31.1% | 28.7% |
| GPT-4o-mini | TextGrad | 34.6% | 18.4% |
| GPT-4o-mini | MIPRO | 36.7% | 12.4% |
| GPT-4o-mini | GEPA | 32.0% | 11.3% |
| GPT-4o-mini | **MEMO (Ours)** | **49.5%** | **6.4%** |
| Qwen2.5-7B-Instruct | UnstableBaseline | 45.0% | 43.3% |
| Qwen2.5-7B-Instruct | **MEMO (Ours)** | **44.3%** | **6.1%** |

- MEMO 在所有提示优化方法中表现最佳，且**稳定性远超所有方法**（RSE 最低）。
- 在 `KuhnPoker` 上，MEMO 仅用 2,000 场游戏就达到了 60% 胜率，而 RL 基线需要 38,000 场（**19倍更高效**）。
- MEMO 在**谈判和不完全信息游戏**中优势最明显，在完全信息游戏中 RL 仍更具优势。

### 消融实验结果（Ablation Study）
消融实验证明了各模块的必要性（Table 3）：
- **仅锦标赛**（Tournament-only）：胜率 27.1% （+3.3%）
- **仅记忆库**（Memory-only）：胜率 34.2% （+10.4%）
- **锦标赛 + 重放**（Tournament + Replay）：胜率 41.6% （+17.8%）
- **锦标赛 + 记忆库**（Tournament + Mem）：胜率 48.1% （+24.3%）
- **完整 MEMO**（Tournament + Mem + Replay）：胜率 50.2% （+26.4%）

**结论**：**记忆库**（Retention）是性能提升的主导因素，而**探索**（Exploration）与**保留**（Retention）的结合带来了最大的增益。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **上下文敏感性**：多轮多智能体 LLM 游戏的结果对上下文选择高度敏感，微小的提示变化可能导致排名反转，因此需要鲁棒的评估实践。
2. **保留优于纯探索**：单纯的随机探索收益有限，**持久的记忆库**是将上下文优化从无记忆搜索转变为累积学习过程的关键。
3. **效率与稳定性并存**：MEMO 在固定预算下大幅提升了胜率，同时显著降低了方差，实现了性能和稳定性的双重提升。
4. **跨任务泛化**：从一个游戏中学到的上下文可以零样本迁移到其他游戏中，并带来性能提升，尤其是在具有相似结构的游戏中。
5. **跨模型迁移有限**：为 `GPT-4o-mini` 学到的上下文在较弱模型（如 `Gemini-2.5-Flash-Lite`）上能普遍提升性能，但在较强模型上可能因策略冲突而导致负迁移。

### 方法的局限性
- 在**完全信息游戏**（如 `SimpleTak`）中，传统的 RL 方法仍然更有效。
- 方法的有效性依赖于自对弈能够生成多样且有价值的轨迹。
- 跨模型迁移的效果不稳定，可能干扰目标模型自身的优秀策略。

### 未来工作方向
- 探索如何将 MEMO 与轻量级参数微调（如 LoRA）相结合，以追求更高的性能上限。
- 研究更有效的跨模型迁移机制，避免负迁移。
- 将 MEMO 应用于更复杂的现实世界多智能体交互场景。

</details>

---

### 9. [Robust Regularized Policy Iteration under Transition Uncertainty](https://arxiv.org/abs/2603.09344)

**Authors**: Hongqiang Lin, Zhenghui Fu, Weihao Tang, Pengfei Wang, Yiding Sun, Qixian Huang, Dongxu Zhang  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.09344v1  

#### Abstract
Offline reinforcement learning (RL) enables data-efficient and safe policy learning without online exploration, but its performance often degrades under distribution shift. The learned policy may visit out-of-distribution state-action pairs where value estimates and learned dynamics are unreliable. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Robust Regularized Policy Iteration under Transition Uncertainty**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
- **Distribution Shift** 和 **Transition Uncertainty** 是离线强化学习（offline RL）中的核心挑战。当策略访问训练数据中未覆盖的 state-action 对时，价值函数估计容易产生外推误差（extrapolation error），导致性能下降。
- 现有方法通常采用保守的价值学习（conservative value learning）或基于不确定性的正则化来缓解该问题，但这些方法往往依赖启发式设计，且未显式建模**转移动力学本身的不确定性**。

### **提出了什么新方法或新思路**
- 提出 **Robust Regularized Policy Iteration (RRPI)**，将 offline RL 建模为一个**鲁棒策略优化问题**（robust policy optimization）：
  - 将转移模型 $ p(\cdot|s,a) $ 视为在**不确定性集合 $ \mathcal{P} $** 内的决策变量，而非固定估计值。
  - 优化目标是最大化在最坏情况转移动态下的期望回报，即求解：
    $$
    \pi^* = \arg\max_\pi \min_{p \in \mathcal{P}} J(\pi, p)
    $$
- 为解决上述 max-min 双层优化问题的计算困难，引入一个 **KL 正则化的代理目标函数**（surrogate objective），并定义了对应的 **Robust Regularized Bellman Operator**。
- 推导出一种高效的迭代算法，结合策略评估与策略改进步骤，避免直接求解复杂的双层优化。

### **相比现有方法的优势**
- **统一处理策略诱导偏差与转移不确定性**：通过鲁棒优化框架自然地抑制对高不确定性区域的动作选择。
- **理论保障强**：
  - 所提出的 Bellman 算子是 $ \gamma $-收缩映射，保证收敛到唯一不动点。
  - 迭代过程能实现原始鲁棒目标的单调提升，并最终收敛至最优鲁棒策略。
- **无需显式不确定性惩罚项**：相比基于置信区间或集成方差的方法，RRPI 的保守行为由最坏情况动态选择内生驱动，更具原则性。
- **稳定性高**：KL 正则化防止策略剧烈变化，提升训练稳定性。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- 在 **D4RL benchmark** 上进行广泛验证，涵盖多种环境与数据分布：
  - **Environments**: `HalfCheetah`, `Hopper`, `Walker2d`
  - **Dataset Types**:
    - `random`: 随机策略收集的数据
    - `medium`: 中等性能策略数据
    - `expert`: 专家策略数据
    - `medium-expert`, `medium-replay`, `full-replay`: 混合数据集

### **实验设置和评估指标**
- **评估方式**：
  - 报告归一化得分：  
    $$
    \text{Normalized Score} = \frac{\text{Agent Score} - \text{Random Score}}{\text{Expert Score} - \text{Random Score}} \times 100
    $$
  - 所有结果取 **4 个随机种子的均值 ± 标准差**
- **模型实现细节**：
  - 使用神经网络参数化 Q 函数和策略。
  - 不确定性集合 $ \mathcal{P} $ 通过 **transition model ensemble**（N 个动力学模型）近似。
  - 最坏情况动态通过选择使 Bellman 目标最小的模型实现。
  - 使用蒙特卡洛采样与重要性采样估计期望。

### **基线方法对比**
- 包括主流 model-free 与 model-based 方法：
  - **Model-Free Baselines**:
    - CQL (Conservative Q-Learning)
    - DMG, EPQ
  - **Model-Based Baselines**:
    - MOReL
    - RAMBO
    - PMDB (Percentile-based Model-based Offline RL)
    - ADM

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
- 如表 1 所示，RRPI 在多数任务上取得**最佳平均表现**：
  - 在 `HalfCheetah-Medium` 上达到 **75.2 ± 0.7**，优于 PMDB 的 75.6±1.3，接近最优。
  - 在 `HalfCheetah-Medium-Replay` 上显著领先：**74.4 ± 0.7 vs. 第二名 71.7 ± 1.1**
  - 在 `Walker2d-Full-Replay` 达到 **107.3 ± 0.4**，为所有方法中最高。
  - 在 `Hopper-Full-Replay` 达到 **108.6 ± 0.2**，略高于 PMDB 的 108.5 ± 0.7。

### **与基线方法的对比结果**
- **总体表现**：
  - RRPI 在 **18 个环境中击败 PMDB（当前 SOTA）中的 11 个**，其余 7 个保持竞争力。
  - 显著优于 CQL、MOReL、RAMBO 等经典方法。
- **鲁棒性体现**：
  - 学习到的 Q 值在高 **epistemic uncertainty** 区域明显降低（见图 2），表明策略主动规避不可靠动作。
  - 表现出更强的抗外推能力，在 medium 和 replay 类型数据集中优势尤为明显。

### **消融实验结果（Ablation Study）**
- 移除“最坏情况动态选择”机制后性能大幅下降（见表 2）：
  - 在 `HalfCheetah-Medium` 上得分下降 **12.4%**，标准差增加 **3.9x**
  - 在 `Hopper-Medium-Replay` 上得分下降 **9.1%**，方差飙升 **25.3x**
- 结论：**worst-case dynamic selection 是性能增益的关键组件**，随机采样无法提供足够的鲁棒性。

---

## 4. **关键结论和发现**

### **论文的主要发现**
1. **鲁棒优化框架有效缓解 offline RL 中的外推风险**：通过显式考虑转移不确定性，RRPI 能自动避免高不确定性区域的动作。
2. **隐式的不确定性感知机制更优**：无需手动设计 uncertainty penalty，其保守行为源于最坏情况建模，更具理论一致性。
3. **KL 正则化 + 鲁棒 Bellman 算子 = 高效稳定训练**：所提算法兼具理论可解释性和工程实用性。
4. **RRPI 实现 SOTA 性能与强鲁棒性平衡**：不仅平均得分高，且在不同数据分布下表现稳健。

### **方法的局限性**
- 依赖于动力学模型的准确性与多样性：若 model ensemble 未能充分覆盖真实不确定性，则鲁棒性受限。
- 当前实现仍基于矩形不确定性假设（ensemble level），可能在高维状态空间中过于保守。
- 计算开销相对较高（需维护多个动力学模型及最坏情况搜索）。

### **未来工作方向**
- 改进不确定性集合建模方式，探索非矩形、结构化 $ \mathcal{P} $
- 引入多模态输入（如视觉观测），拓展至复杂感知任务（vision-based offline RL）
- 加强理论与实践之间的联系，例如更精确地量化 epistemic uncertainty 并用于自适应正则化
- 应用于现实世界高风险场景，如医疗决策、能源管理、自动驾驶等（见附录 B）

--- 

> ✅ **总结一句话**：  
> RRPI 通过将 offline RL 建模为鲁棒马尔可夫决策过程，并设计可高效求解的 KL 正则化策略迭代算法，在理论严谨性与实际性能之间取得了良好平衡，成为当前 model-based offline RL 的有力竞争者。

</details>

---

### 10. [AutoAgent: Evolving Cognition and Elastic Memory Orchestration for Adaptive Agents](https://arxiv.org/abs/2603.09716)

**Authors**: Xiaoxing Wang, Ning Liao, Shikun Wei, Chen Tang, Feiyu Xiong  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.09716v1  

#### Abstract
Autonomous agent frameworks still struggle to reconcile long-term experiential learning with real-time, context-sensitive decision-making. In practice, this gap appears as static cognition, rigid workflow dependence, and inefficient context usage, which jointly limit adaptability in open-ended and n...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AutoAgent: Evolving Cognition and Elastic Memory Orchestration for Adaptive Agents**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 autonomous agent 框架在以下三方面存在显著局限：
- **静态认知（Static Cognition）**：工具、自身能力、同伴专长等描述依赖于人工编写的固定 prompt，无法通过经验动态更新，导致决策僵化。
- **刚性工作流（Rigid Workflow Dependence）**：依赖预定义的推理循环或计划，在面对非预期情况时适应性差。
- **低效上下文管理（Inefficient Context Usage）**：历史交互以原始文本形式累积，造成 token 浪费、推理变慢，且缺乏对经验的结构化组织。

这些问题共同限制了 agent 在开放、动态环境中的长期学习与自适应能力。

---

### **提出的新方法与核心思想**
作者提出了 **AutoAgent**，一个基于 **自我演化（self-evolution）** 范式的多智能体框架，其核心由三大支柱构成：

#### **(1) Evolving Cognition（演进式认知）**
将 agent 的认知建模为可更新的显式状态，分为两个维度：
- **Internal Cognition**：关于自身能力的知识，包括工具配置文件（tool profiles）、技能库（skill library），随使用反馈不断优化。
- **External Cognition**：关于外部世界的知识，如其他 agent 的专长模型、环境响应模式，支持更精准的协作请求。

> ✅ 创新：从“静态 prompt”转向“可演化的知识库”，实现基于实践的认知闭环更新。

#### **(2) On-the-fly Contextual Decision-Making（即时情境决策）**
引入统一的动作空间（Unified Action Space），融合两类动作：
- **Emic Actions**：依靠内部资源完成的任务（如 LLM 生成、调用已知工具）。
- **Etic Actions**：寻求外部帮助的行为（如向特定 peer 发起咨询）。

决策过程采用原子化的 **Select-Execute-Update 循环**，结合当前 context 与 cognition 动态选择最优动作，摆脱预设流程束缚。

> ✅ 创新：将单 agent 工具使用与 multi-agent 协作统一到同一决策框架下，提升灵活性。

#### **(3) Elastic Memory Orchestration（弹性记忆协调）**
设计了一个 **Elastic Memory Orchestrator (EMO)** 来高效管理交互历史：
- **Step-wise Compression**：每步保留原始记录（Raw Info）和摘要（Abstract），并根据需要动态选择加载粒度。
- **Episodic Memory Construction**：将连续成功步骤聚合成 episode，用于高层级回顾与技能提炼。
- **Selective Retrieval & MemFold**：通过 selector 决定是否保留某步信息，并可折叠多个步骤为高阶抽象。

> ✅ 创新：实现“有选择地压缩 + 按需恢复细节”的弹性机制，兼顾效率与保真度。

#### **(4) Closed-loop Cognitive Evolution（闭合式认知进化）**
所有组件集成在一个 **Self-Evolution Loop** 中：
1. 执行产生轨迹（Trajectory）
2. 分析意图与结果差异
3. 更新 cognition 和 memory 策略
4. 改进后的认知指导后续行动

无需外部重训练即可持续自我改进。

> ✅ 核心优势：形成“行动 → 经验 → 学习 → 更好行动”的正向循环。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | AutoAgent |
|------|--------|----------|
| 认知表示 | 静态 prompt / 固定 schema | 可演化的 structured cognition |
| 决策机制 | 固定 workflow（如 ReAct 循环） | 动态、情境感知的选择机制 |
| 上下文管理 | 全量拼接历史 | 弹性压缩 + 多层级抽象 |
| 自我改进 | 无 / 依赖人工干预 | 闭环演化，自动提炼技能 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖两大类任务场景：

#### **(1) Retrieval-Augmented Generation (RAG) 问答任务**
- **HotpotQA**：多跳推理，需跨段落整合信息
- **2WikiMultihopQA**：维基百科基础上的复杂推理
- **Bamboogle**：测试抗干扰能力（含误导性文档）
- **Musique**：实体密集型多跳推理

#### **(2) Tool-Augmented Agent 与 Embodied Task**
- **GAIA**：真实世界多步任务（网页操作、API 调用等）
- **HLE-Bench**：分层语言嵌入任务，强调子目标规划
- **ALFWorld**：文本模拟环境中执行指令（如“把苹果放进冰箱”）

---

### **实验设置与评估指标**

| 类别 | 指标说明 |
|------|---------|
| **RAG 任务** |  
| - Acc (Exact Match) | 答案完全匹配准确率 |
| - LLM-as-a-judge Score | 使用独立 LLM 对答案忠实性与完整性打分 |
| **GAIA / HLE** |  
| - Pass@1 | 输出是否满足任务要求（由 o3-mini 判断） |
| - Step Efficiency | 平均执行步数，越少越好 |
| **ALFWorld** |  
| - Success Rate | 成功完成任务的比例 |
| - Path Similarity | 行动序列与近优路径的相似度（衡量效率） |

---

### **基线方法对比**
#### **RAG 基线**
- Naive Generation
- Standard RAG
- Self-Ask, IRCoT, SuRe, REPLUG 等先进迭代检索方法

#### **Tool-use 基线**
- **Direct Answer**：直接生成答案（无动作）
- **ReAct**：经典“思考-行动”交替范式
- **DeepAgent**：近期先进的 agent 架构，具备复杂工具调用能力

#### **模型后端**
涵盖多种 LLM：
- **Closed-source**: `GPT-4o`, `Gemini-3-Pro`, `Gemini-3-Pro-Thinking`
- **Open-source**: `DeepSeek-R1`, `QwQ-32B`, `Qwen3-30B-A3B`

所有 agent 使用相同工具接口与初始描述，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### **(1) RAG 多跳问答表现（Table 1）**
| 方法 | Avg Acc | Avg LLM Judge |
|------|--------|--------------|
| IRCoT（第二名） | 0.3475 | 0.2050 |
| **AutoAgent** | **0.3965** | **0.4315** |

- 在 **HotpotQA**, **2Wiki**, **Bamboogle** 上均达到 SOTA。
- 在 **HotpotQA** 上准确率达 **53.0%**，显著优于 IRCoT 的 43.4%。
- 在 **Musique** 上表现一般（仅 13.4%），表明当前认知建模对该类实体组合推理仍有不足。

> 📌 结论：AutoAgent 在多数多跳 QA 场景中表现出更强的信息整合与推理稳定性。

---

#### **(2) 工具增强任务表现（Tables 2 & 3）**

##### **GAIA 总体成功率（closed-source）**
| Model | DeepAgent | AutoAgent |
|-------|-----------|------------|
| `gemini-3-pro` | 27.3% | **54.5%** (+27.2pp) |
| `gemini-3-pro-thinking` | 23.0% | **52.7%** (+29.7pp) |

> 💥 提升超过一倍，尤其在强推理模型上放大优势。

##### **GAIA（open-source）**
| Model | ReAct | AutoAgent |
|-------|-------|-----------|
| `DeepSeek-R1` | 41.2% | **52.1%** |
| `QwQ-32B` | 29.1% | **33.9%** |

> ✅ 显著领先，验证框架普适性。

##### **ALFWorld 成功率**
| Model | DeepAgent | AutoAgent |
|-------|-----------|------------|
| `gemini-3-pro` | 97.8% | **99.3%** |
| `QwQ-32B` | 79.9% | **82.8%** |
| `Qwen3-30B-A3B` | 48.5% | **58.2%** |

> ✅ 在多个模型上刷新 SOTA。

---

### **消融实验结果（Ablation Studies）**

#### **(1) Elastic Memory Orchestration (EMO) 模块影响**
| 设置 | GAIA (All) | HLE (All) |
|------|------------|----------|
| w/o EMO | 45.5% | 19.2% |
| w/ EMO | **52.1%** | **28.2%** |

> 🔺 +6.6pp 和 +9.0pp 提升，证明 EMO 对长程任务至关重要。

#### **(2) Evolving Cognition 在工具不稳定下的适应能力**
在人为注入 50% 故障率的工具环境下进行测试：

| 工具类型 | F1 (初始) → (演化后) | EM (初始) → (演化后) |
|--------|---------------------|---------------------|
| Web Search | 24.07% → **29.67%** (+5.6) | 22.00% → **28.00%** (+6.0) |
| Python Code Execution | 25.55% → **37.31%** (+11.76) | 24.00% → **36.00%** (+12.00) |

> ✅ 明确验证：agent 能通过执行反馈识别不可靠工具并调整策略。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Evolving Cognition 是实现长期适应的关键**  
   将认知作为可更新的状态，使 agent 能够从失败中学习、修正工具理解、优化协作对象选择。

2. **Elastic Memory 显著提升长程推理效率**  
   动态压缩 + 多粒度检索机制有效控制 token 开销，同时保留关键证据，避免信息丢失。

3. **Contextual Decision-Making 提供更高灵活性**  
   统一 Emic/Etic 动作空间使得 agent 可灵活切换“自主解决”与“求助协作”，适应动态需求。

4. **Self-Evolution Loop 实现免外部训练的持续进步**  
   无需额外标注或微调，仅凭交互反馈即可提炼新技能、修正错误认知。

5. **框架具有良好的模型泛化性**  
   在 open-source 与 closed-source LLM 上均稳定超越基线，尤其在强推理模型（如 Gemini）上增益最大。

---

### **局限性**
- 当前认知更新依赖 LLM-based analyzer，可能存在误判风险。
- 在高度结构化或多模态环境中的扩展尚未充分验证。
- Musique 数据集上的表现不佳，提示对某些复杂语义组合推理仍需改进。
- 大规模 agent 社会中的去中心化认知同步机制未涉及。

---

### **未来工作方向**
1. 引入更多学习信号：如 verifier 反馈、环境奖励函数，增强进化精度。
2. 探索去中心化的分布式认知更新机制，适用于大规模 agent 社群。
3. 在企业级工作流、科研工具链等现实生态系统中部署并评估鲁棒性。
4. 结合神经符号系统，进一步提升认知建模的形式化程度与可靠性。

---

> 🔗 **代码开源地址**：[https://github.com/vicFigure/AutoAgent](https://github.com/vicFigure/AutoAgent)

</details>

---

### 11. [Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning](https://arxiv.org/abs/2603.08999)

**Authors**: Juming Xiong, Kevin Guo, Congning Ni, Chao Yan, Katherine Brown, Avinash Baidya, Xiang Gao, Bradley Marlin, Zhijun Yin  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.08999v1  

#### Abstract
Large language models (LLMs) achieve strong reasoning performance through chain-of-thought (CoT) reasoning, yet often generate unnecessarily long reasoning paths that incur high inference cost. Recent self-consistency-based approaches further improve accuracy but require sampling and aggregating mul...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在执行 **Chain-of-Thought (CoT)** 推理时虽然表现出强大的推理能力，但通常会生成过长且不必要的推理路径，导致高昂的 **inference cost**（推理开销）。  
此外，现有的 **self-consistency (SC)** 方法通过采样多条推理路径并进行聚合（如多数投票），显著提升了准确性，但也带来了更高的计算资源消耗。

因此，本文旨在解决以下核心矛盾：
> 如何在保持高准确率的同时，大幅降低多路径推理带来的 token 开销？

---

### 提出了什么新方法或新思路
作者提出了一种 **confidence-aware 决策框架**，其核心思想是：

- **仅分析一条完整的贪婪推理轨迹（greedy CoT path）**
- 从中提取 **sentence-level 的数值特征（numeric features）和语言学特征（linguistic features）**
- 使用一个轻量级的 **attention-based RNN 模型** 来预测该推理路径是否可靠（即最终答案是否正确）
- 根据预测的置信度 $ p \in [0,1] $ 和预设阈值 $ \tau $ 动态决策：
  - 若 $ p \geq \tau $：接受当前单路径输出（节省计算）
  - 若 $ p < \tau $：启动多路径推理（如 dynamic voting）以提升可靠性

这一机制实现了“按需增强”（selective enhanced reasoning），避免对所有样本都进行昂贵的多路径采样。

---

### 相比现有方法的优势
| 方法 | 是否需要多路径采样 | 是否可提前终止 | 效率优势 | 准确性保障 |
|------|------------------|---------------|----------|------------|
| Greedy Decoding | 否 | 是 | 高 | 低 |
| Self-Consistency (SC) | 是 | 否 | 低 | 高 |
| Dynamic Voting (DV) | 是（动态终止） | 是 | 中等 | 高 |
| **本文方法（Ours）** | **仅部分样本需要** | **基于置信度判断** | ✅ **最高（最多减少80% token）** | ✅ **与SC/DV相当** |

> ✅ **核心优势**：用极少的额外训练成本，实现跨任务、跨模型的高效推理控制，无需 fine-tuning 或修改 LLM 本身。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖医学与通用领域共四个主流多选问答数据集：
- **MedQA**：USMLE风格的医学考试题，测试临床推理能力
- **MedMCQA**：印度医学入学考试题，规模更大（>20万题）
- **MathQA**：数学应用题，需符号推理
- **MMLU**：涵盖57个学科的综合知识基准，评估泛化能力

> 所有模型均在 **MedQA 上训练**，其余三个数据集上进行 **zero-shot 迁移评估**

---

### 实验设置和评估指标

#### 模型列表（均为开源 LLMs）：
- GPT-OSS 20B（主模型）
- LLaMA3.1 8B
- Qwen2.5 7B
- Qwen3 14B
- Qwen3 32B

#### 主要评估指标：
- **Accuracy**：最终答案准确率
- **Token Usage**：平均每个样本使用的 token 数量（衡量效率）
- **Statistical Significance**：采用 paired bootstrap（2000次重采样）检验差异显著性

#### 其他设置：
- 多路径采样数量：10条
- 温度（temperature）：1.0
- 数据划分：train/val/test = 8500/500/1000
- 所有实验在单张 NVIDIA H100 80GB GPU 上完成

---

### 基线方法对比
| 基线方法 | 简称 | 描述 |
|--------|-----|------|
| **Greedy Sampling** | —— | 单路径贪婪解码，最低开销 |
| **Self-Consistency** | SC | 采样多条路径，取最频繁答案 |
| **Confidence Enhanced Reasoning** | CER | 对不同路径加权聚合，权重来自中间步骤置信度 |
| **Dynamic Voting** | DV | 动态采样路径，达成共识后立即停止 |

> 本文方法（Ours）的目标是在不显著损失 accuracy 的前提下，大幅优于上述方法的 token efficiency。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 GPT-OSS 20B 为例）

| 数据集 | 最优置信阈值 $ \tau $ | Accuracy（vs. DV） | Token Reduction（vs. SC/CER） | Token Reduction（vs. DV） |
|-------|------------------------|--------------------|-------------------------------|----------------------------|
| MedQA | 0.65 | ≈ +0.00%（n.s.） | ~79% | ~48% |
| MathQA | 0.20 | ≈ +0.00%（n.s.） | ~69% | ~27% |
| MedMCQA | 0.60 | ≈ -0.25%（n.s.） | ~72% | ~37% |
| MMLU | 0.80 | ≈ -0.47%（n.s.） | ~73% | ~53% |

> ✅ **关键结论**：在几乎所有数据集上，accuracy 与 SC/DV 无统计显著差异（n.s.），但 **token 使用量最多减少达 80%**。

---

### 与基线方法的对比结果
从图3（Figure 3）可以看出：
- 在 accuracy 方面，本文方法（Ours）与 SC、CER、DV **没有显著差异（n.s.）**
- 在 token usage 上，本文方法显著更低（$ p < 0.05 $）
- 相比 SC 和 CER：平均减少 **69–79%** token
- 相比更高效的 DV：仍能减少 **27–53%** token

> 表明本方法在维持高准确率的同时，实现了最优的 **accuracy-efficiency trade-off**

---

### 消融实验结果（Ablation Study）

#### （1）注意力模块消融（Table 3）
| 变体 | Accuracy ↑ | Token Usage ↓ |
|------|-----------|--------------|
| 无 FA & 无 MHSA | 0.818 | 947 |
| 有 FA | 0.819 | 888 |
| 有 MHSA | 0.819 | 900 |
| **有 FA + 有 MHSA（完整模型）** | **0.820** | **794（↓16.16%）** |

> ✅ **双注意力机制协同有效**：feature attention（FA）用于通道重加权，multi-head self-attention（MHSA）建模句子间依赖，二者结合效果最佳。

#### （2）输入特征消融（Table 4）
| 特征类型 | Accuracy ↑ | Token Usage ↓ |
|--------|-----------|--------------|
| Numeric only | 0.818 | 884 |
| Linguistic only | 0.818 | 911 |
| **Numeric + Linguistic** | **0.820** | **805（↓8.94%）** |

> ✅ **两类特征互补**：数值特征捕捉概率趋势，语言学特征反映表达风格（如 hedge words、punctuation），联合使用提升判别能力。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **单条推理轨迹蕴含丰富的不确定性信号**  
   即使只看一条 CoT 路径中的 sentence-level 特征（如概率变化、熵、句法模式），也能有效估计其最终正确性。

2. ✅ **推理模式具有跨任务可迁移性**  
   在 MedQA 上训练的决策模型，可以直接 zero-shot 应用于 MathQA、MedMCQA、MMLU，并保持良好性能，说明 LLM 的推理行为存在通用规律。

3. ✅ **大模型具有更清晰的轨迹区分度**  
   更大规模的 LLM（如 Qwen3 32B）其正确与错误路径的概率/熵演化更具可分性，使得 confidence estimation 更准确。

4. ✅ **无需修改 LLM 架构即可实现高效推理控制**  
   本方法为“外挂式”轻量决策器，适用于任意支持输出 token-level logits 的 LLM，部署灵活。

---

### 方法的局限性
1. ❌ **无法在线 early-exit**  
   当前框架分析的是 **已完成的 CoT 轨迹**，不能在生成过程中实时中断，未来可探索 causal 模型实现实时判断。

2. ❌ **依赖内部信号访问权限**  
   需要获取 LLM 的 token-level logits 和生成文本，目前主要适用于开源模型；闭源 API 可能受限。

3. ❌ **集中在多选问答任务**  
   实验集中于 structured reasoning 场景（如选择题），在开放生成、对话、长文本推理中有效性待验证。

4. ❌ **阈值需轻微调参**  
   虽然无需重新训练，但在新数据集上仍需通过 validation set 校准 confidence threshold $ \tau $。

---

### 未来工作方向
- 将框架扩展至 **causal setting**，实现生成过程中的动态 early-exit
- 探索 **zero-shot threshold adaptation**，完全免去阈值校准
- 应用于 **open-ended generation** 和 **agent planning** 等复杂场景
- 结合 **process reward models (PRMs)** 或 **self-refinement**，构建端到端自适应推理系统

---

> 🔚 **总结一句话**：  
> 本文提出了一种轻量、通用、高效的 confidence-aware 决策机制，利用单条 CoT 轨迹中的浅层特征预测其可靠性，从而智能决定是否启用多路径推理，在几乎不牺牲 accuracy 的前提下，将 token 消耗最多降低 **80%**，为 LLM 推理的 **efficiency-accuracy trade-off** 提供了一个极具实用价值的新范式。

</details>

---

### 12. [Serving Compound Inference Systems on Datacenter GPUs](https://arxiv.org/abs/2603.08797)

**Authors**: Sriram Devata, Rahul Singh, Sarita Adve  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.08797v1  

#### Abstract
Applications in emerging domains such as XR are being built as compound inference systems, where multiple ML models are composed in the form of a task graph to service each request. Serving these compound systems efficiently raises two questions: how to apportion end-to-end latency and accuracy budg...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Serving Compound Inference Systems on Datacenter GPUs》核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

随着机器学习应用向**Compound Inference Systems**（复合推理系统）演进——即多个ML模型以任务图（DAG）形式组合处理请求（如XR、多智能体系统），传统单模型推理服务框架已无法满足效率需求。这类系统面临两大挑战：

- 如何在端到端的**Latency SLO**和**Accuracy SLO**约束下，合理分配各子任务的延迟与精度预算；
- 如何高效地为资源需求各异的模型分配异构GPU资源。

现有系统大多忽略任务间的依赖关系、缺乏对精度可伸缩性的支持，或未充分利用现代GPU的**Spatial Partitioning**能力。

---

### **提出了什么新方法或新思路**

本文提出 **JIGSAWSERVE**，是首个联合优化**Latency、Accuracy 和 GPU 资源成本**的数据中心级复合推理服务框架。其核心创新在于将以下三种技术进行**协同设计与联合优化**：

1. **Per-task Accuracy Scaling**  
   允许每个任务选择不同**Model Variants**（如ResNet-18/34/50等），在保证端到端Accuracy SLO的前提下，动态调整模型大小以节省资源。

2. **GPU Spatial Partitioning**  
   利用NVIDIA GPU的**MIG**（Multi-Instance GPU）和**MPS**（Multi-Process Service）机制，实现细粒度的GPU资源划分。通过“GPU Segment”抽象（MIG实例 + MPS并发数），允许多个任务共享同一物理GPU且干扰最小。

3. **Task-Graph-Informed Resource Budgeting**  
   基于任务图结构（包括路径、乘法因子 Multiplicative Factor）进行全局资源与SLO预算分配，而非孤立地为每个任务设定局部限制。

> 名称“JIGSAW”寓意：像拼图一样灵活组合多种小块模型变体与GPU分区，形成最优整体配置。

---

### **相比现有方法的优势**

| 特性 | JIGSAWSERVE | Prior Works（如Loki, IPA, DREAM） |
|------|-------------|-------------------------------|
| 支持复合推理系统 | ✅ | 部分支持（但功能受限） |
| 联合优化Accuracy + Latency + Cost | ✅ | ❌ 多数仅优化其中一项 |
| 使用GPU空间分区（MIG/MPS） | ✅ | ❌ 或仅使用整卡 |
| 任务图感知的预算分配 | ✅ | ❌ 多为静态/独立分配 |
| 支持动态批处理与早期丢弃 | ✅ | ⚠️ 少数支持 |

JIGSAWSERVE首次实现了三者融合，在相同GPU资源下显著提升服务能力。

---

## 2. 核心实验方法和设置

### **使用的数据集与工作负载**

构建了三个典型复合推理应用作为测试用例，覆盖不同任务图深度：

| 应用 | 任务图深度 | 主要模型 | 输入数据集 |
|------|------------|----------|------------|
| **Social Media**（社交媒体） | 1 | ResNet + GIT | COCO dataset |
| **Traffic Analysis**（交通分析） | 2 | YOLO → EfficientNet/VGG | Bellevue Traffic Dataset |
| **AR Assistant**（增强现实助手） | 3 | YOLO → GIT → TTS | COCO dataset |

每个任务提供多个**Model Variants**供选择（如YOLOv5n/s/m/l/x），具有不同的Latency、Throughput、Accuracy指标。

---

### **实验设置**

- **硬件平台**：Dell PowerEdge XE9640服务器，配备4×NVIDIA H100 SXM 80GB GPU（共28个MIG slice等价资源）
- **软件环境**：Ubuntu 22.04, Python 3.10, CUDA 12.4, PyTorch 2.6.0
- **MIG配置工具**：`nvidia-mig-parted`
- **求解器**：Gurobi用于解决MILP优化问题
- **调度频率**：每5分钟根据需求变化重新运行MILP并重配置系统

---

### **评估指标**

| 指标 | 定义 |
|------|------|
| **Max Serviceable Demand**（最大可服务请求率） | 在满足SLO前提下能处理的最大req/s |
| **GPU Resource Usage** | 占用的GPU slices百分比 |
| **Accuracy Drop** | 相对于最高精度系统的准确率下降比例 |
| **SLO Violation Rate** | 请求超时的比例（含Early Drop也算违反） |
| **Ablation Study** | 分析各组件（A/S/T）单独及组合效果 |

---

### **基线方法对比**

| 基线 | 特征组合 | 对应Prior Work |
|------|---------|----------------|
| Unopt | 无任何优化 | — |
| T | Task-graph-informed budgeting only | — |
| S | Spatial partitioning only | ParvaGPU-like |
| A | Accuracy scaling only | INFaaS/Loki-like |
| A+T | Accuracy + Task-graph budgeting | **Loki** |
| S+T | Spatial + Task-graph budgeting | ParvaGPU+T |
| A+S | Accuracy + Spatial (task-unaware) | Clover+MPS |
| **JIGSAWSERVE** | **A+S+T**（全功能） | — |

> 注意：**A+S+T 是 JIGSAWSERVE 的完整配置搜索空间，也是所有prior work的超集**。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **最大服务需求提升：21.6× vs Unopt，11.3× vs 最接近基线**

- 在一个假设拥有120 GPUs（840 slices）的大规模测试环境中：
  - **Unopt**（无优化）：基准吞吐量
  - **S alone**：5.25× 提升 → 表明**Spatial Partitioning 是最大增益来源**
  - **A alone**：1.6× 提升
  - **T alone**：1.1× 提升
  - **A+T**（≈Loki）：1.9× 提升
  - **S+T**：7.8× 提升
  - **A+S**：12.1× 提升
  - **JIGSAWSERVE (A+S+T)**：**21.6× 提升**

> 🔥 **JIGSAWSERVE 比最接近的 prior work（A+T ≈ Loki）高 11.3× 的最大服务容量**

---

#### ✅ **实证评估：平均仅用43.3% GPU资源，SLO违规 < 0.6%**

在真实流量模式（基于Twitter trace建模的日级请求波动）下的端到端系统评估显示：

| 方法 | 平均GPU使用 | Accuracy Drop | SLO Violation Rate |
|------|--------------|----------------|--------------------|
| **JIGSAWSERVE** | **43.3%** | ≤10%阈值内 | **< 0.6%** |
| S+T / A+T | >80% | 可接受 | **>10%**（严重超标） |
| A+S | ~58% | 可接受 | **6.7%**（仍偏高） |

> 📉 所有非完整版本均出现明显退化，尤其在高负载时SLO违规激增。

---

### **消融实验结果（Ablation Study）**

从图3和表1可得以下结论：

| 组件 | 贡献度 | 观察 |
|------|--------|------|
| **Spatial Partitioning (S)** | ⭐⭐⭐⭐⭐ | 单独带来**5.25×**增益，是最关键因素 |
| **Accuracy Scaling (A)** | ⭐⭐⭐☆ | 单独1.6×，但在高并发中价值凸显 |
| **Task-Graph Budgeting (T)** | ⭐⭐ | 单独仅1.1×，但与其他结合后显著放大收益 |
| **三者联合 (A+S+T)** | ✅✅✅ | 效果远超加总（非线性增益），体现协同效应 |

> 💡 关键发现：**缺少任一组件都会导致效率大幅下降**，证明三者缺一不可。

---

## 4. 关键结论和发现

### **主要发现**

1. **GPU Spatial Partitioning 是提升数据中心推理效率的关键突破口**  
   相比传统的整卡分配，利用MIG/MPS实现细粒度资源共享可大幅提升GPU利用率。

2. **必须联合优化 Accuracy、Resource、Latency**  
   孤立优化某一方面（如只调模型大小或只分GPU）无法达到最优；只有将三者纳入统一MILP框架才能实现全局最优。

3. **任务图结构信息至关重要**  
   忽视任务间依赖关系（如乘法因子、路径长度）会导致资源错配，进而引发SLO大规模违反。

4. **JIGSAWSERVE 实现了极高的资源效率**  
   平均仅需**43.3% GPU资源**即可满足SLO，意味着可在同集群上部署更多服务或降低成本。

---

### **方法的局限性**

1. **依赖离线Profiling**：需预先测量所有(model variant, GPU segment, batch size)组合的Latency/Throughput，耗时约7–12小时。
2. **MIG Repartitioning 开销**：更改MIG配置会中断服务，虽可通过预留GPU缓解，但仍影响可用性。
3. **当前基于MILP的控制器非实时在线决策**：依赖周期性重配置，难以应对突发流量尖峰。
4. **假设任务图为DAG**：不支持循环或动态变更拓扑的复杂Agent系统。

---

### **未来工作方向**

1. **扩展至更复杂的Compound AI Systems**  
   支持包含LLM Agent交互、反馈回路、动态路由的系统。

2. **适配其他厂商GPU**  
   AMD GPU也提供类似Chiplet-based spatial partitioning机制，思想可迁移。

3. **集成先进预测与弹性调度**  
   引入更精准的需求预测模型，提前触发资源配置变更，隐藏repartitioning延迟。

4. **优化MIG Placement以减少碎片化**  
   当前使用贪心Bin-Packing算法可能导致资源浪费，未来可将其纳入MILP目标函数。

5. **探索碳感知（Carbon-Aware）调度**  
   修改MILP目标函数以最小化碳排放强度，支持绿色AI。

6. **多层级资源池化架构**  
   构建“资源池 + 运行时弹性分配”机制，提升响应速度与灵活性。

---

## 总结

> 🔷 **JIGSAWSERVE 是首个面向数据中心GPU的、支持Accuracy-Resource-Latency联合优化的复合推理服务框架。**
>
> 🔷 它通过整合 **Model Variant Selection、GPU Spatial Partitioning 和 Task-Graph-Informed Budgeting**，实现了高达 **11.3× 的服务容量提升**，并在实际部署中**平均仅消耗43.3% GPU资源**，同时保持低于0.6%的SLO违规率。
>
> 🔷 实验证明：三大特性相辅相成，任意缺失都将导致性能急剧下降。
>
> 🔷 本工作呼吁：**ML社区应持续发布多变体模型，GPU厂商应进一步强化空间分区能力**，共同推动高效AI服务生态发展。

</details>

---

### 13. [Accelerating High-Order Finite Element Simulations at Extreme Scale with FP64 Tensor Cores](https://arxiv.org/abs/2603.09038)

**Authors**: Jiqun Tu, Ian Karlin, John Camier, Veselin Dobrev, Tzanio Kolev, Stefan Henneking, Omar Ghattas  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.09038v1  

#### Abstract
Finite element simulations play a critical role in a wide range of applications, from automotive design to tsunami modeling and computational electromagnetics. Performing these simulations efficiently at the high resolutions needed for practical applications and scientific insights necessitates the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accelerating High-Order Finite Element Simulations at Extreme Scale with FP64 Tensor Cores*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
高阶有限元模拟（high-order finite element simulations）在科学计算中至关重要，如海啸建模、电磁场仿真等。然而，这类模拟在大规模 GPU 系统上运行时，其核心计算内核（kernels）常受限于内存带宽，尤其是共享内存（shared memory）的 bank conflict 和低效的数据重用。尽管已有研究利用 FP16/FP32 Tensor Cores 加速矩阵运算，但**需要双精度（FP64）精度的科学应用**（如反问题、多尺度物理）难以从中受益。

本文旨在解决：**如何有效利用 NVIDIA 新一代支持 FP64 的 Tensor Cores 来加速高阶有限元中的小规模密集张量收缩（small dense tensor contractions），从而提升大规模科学仿真的性能与能效**。

### 提出的新方法与新思路
- **直接编程 FP64 Tensor Cores（DMMA）**：首次将 FP64 Tensor Cores 直接集成到复杂的 PDE 求解器中，用于执行小规模矩阵乘法（如 $25 \times 5$ 与 $5 \times 4$ 类型的 DGEMM），而非依赖 cuBLAS 等库函数。
- **优化内存访问模式以避免 bank conflict**：
  - 设计了针对不规则矩阵形状（如 m25n5k4）的 `fm/fn/fk` 映射策略，确保 warp 内线程访问共享内存时不发生 bank 冲突。
  - 引入**张量索引重排序技术**（tensor index reordering），使求和索引成为最快变化维度，从而适配 Tensor Core 的高效访存模式。
- **Kernel Fusion 优化**：
  - 将 Partial Assembly (PA) 和 Matrix-Free (MF) 中的多个算子融合为单一 kernel，减少全局内存访问次数和中间数据搬运。
  - 特别提出 `Fused PA` 和 `Fused MF` 架构，在保留精度的同时最大化数据局部性。

### 相比现有方法的优势
| 方面 | 传统方法 | 本工作 |
|------|--------|-------|
| **Tensor Core 利用** | 多用于大矩阵 GEMM 或低精度训练 | 首次用于复杂 HPC 应用中的小规模 FP64 张量操作 |
| **编程方式** | 调用 cuBLAS 等黑盒 API | 直接使用 PTX 指令进行细粒度控制 |
| **适用场景** | 通用线性代数 | 高阶有限元离散化中的特定算子（如 $B^T D B$） |
| **能效提升** | 通常关注吞吐 | 实现高达 **83% 的能量效率增益** |

---

## 2. 核心实验方法和设置

### 使用的数据集与应用背景
- **应用案例**：基于 MFEM 库构建的“海啸早期预警数字孪生”系统（digital twin for tsunami early warning），该应用荣获 2025 年 Gordon Bell Prize。
- **PDE 模型**：声-重力波方程组（acoustic-gravity wave equations），描述地震引发的海洋波动传播。
- **离散方法**：
  - 压力场：四阶连续有限元（H1-conforming, 4th order）
  - 速度场：三阶间断有限元（L2-conforming, 3rd order）
- **时间推进**：显式四阶 Runge-Kutta (RK4)

### 实验平台与硬件配置
| 组件 | 规格 |
|------|-----|
| **GPU 架构** | NVIDIA Grace Hopper GH200 Superchip, Grace Blackwell GB200 Superchip |
| **Tensor Core 支持** | FP64 DMMA (m8n8k4 指令) |
| **超算系统** | Alps 系统（Swiss National Supercomputing Centre, CSCS）<br>• 2,688 节点，每节点 4× GH200<br>• 总计最多使用 **9,216 块 GH200 GPU** |
| **互联网络** | HPE Slingshot-11 dragonfly topology |

### 评估指标
- **单卡性能**：
  - 吞吐率（Throughput）：GDOF/s（十亿自由度每秒）
  - 能效：MDOF/W（百万自由度每瓦特）
  - 功耗（Average Power）
- **多卡扩展性**：
  - 强扩展性（Strong Scaling Efficiency）
  - 弱扩展性（Weak Scaling Efficiency）
- **底层分析工具**：
  - NVIDIA Nsight Compute：分析 SM 利用率、共享内存带宽、bank conflict 等

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Original PA / MF** | 原始 CUDA Core 实现，未使用 Tensor Core |
| **DMMA PA / MF** | 使用 FP64 Tensor Cores 替换部分 GEMM 运算 |
| **Fused PA / Fused MF** | 结合 kernel fusion 的优化版本 |
| **DMMA + Fused PA/MF** | 融合 Tensor Core 与 kernel fusion 的最终方案 |

---

## 3. 主要实验结果和性能指标

### 单 GPU 性能（5.4 亿 DOF 问题规模）

#### 表：GH200 与 GB200 上的关键性能指标（摘自 Table VI）

| Kernel | GDOF/s (GB200) | GDOF/s (GH200) | MDOF/W (GB200) | MDOF/W (GH200) |
|--------|----------------|----------------|----------------|----------------|
| PA | 23.78 | 18.73 | 26.60 | 28.65 |
| Fused PA | 29.28 | 24.04 | 36.41 | 40.14 |
| DMMA PA | 33.72 | 25.27 | 31.37 | 36.51 |
| **DMMA Fused PA** | **46.60** | **36.15** | **45.70** | **52.42** |

> ✅ **最高实现 2× 的端到端性能提升**（相比原始 PA kernel）

#### 能效提升
- 在 GH200 上：
  - DMMA PA 相比原生 PA：**+27% 能效提升**
  - DMMA Fused PA 相比原生 PA：**+83% 能效提升**
- 在 GB200 上：
  - DMMA Fused PA 相比原生 PA：**+72% 能效提升**

> 💡 尽管 GB200 峰值更高，但由于更高的空闲功耗和时钟频率，其单位功耗表现略低于 GH200。

### 多 GPU 扩展性（Alps 系统，最大 9,216 GPUs）

#### 强扩展性（Strong Scaling）
- 问题总规模固定为 ~1450 亿 DOF
- 从 144 到 9,216 GPUs（64× 扩展）
- **并行效率保持在 86–91%**（见 Figure 4）
- 所有六种 kernel 变体均表现出优异的可扩展性

#### 弱扩展性（Weak Scaling）
- 每个 GPU 固定 ~1.01 亿 DOF
- 最大问题规模达 **~9.28 万亿 DOF**
- **弱扩展效率接近完美线性（≈100%）**（见 Figure 5）

> 🚀 这是目前公开报道中**最大规模的 FP64 Tensor Core 科学计算验证之一**。

### 消融实验分析
- **仅启用 DMMA** → 吞吐提升 35–59%，瓶颈由 shared memory bandwidth 转向 compute-bound
- **仅启用 kernel fusion** → 减少数据移动，但受限于 CUDA core 计算能力
- **两者结合** → 实现协同效应，达到 **2× 加速比**
- **性能瓶颈转移**：
  - 原始 kernel：97% 共享内存带宽利用率（表 I）
  - DMMA kernel：共享内存降至 84%，DMMA pipe 利用率达 54%（表 II）

---

## 4. 关键结论和发现

### 主要发现
1. **FP64 Tensor Cores 可有效加速小规模科学计算内核**：
   - 即使是非标准尺寸的小矩阵乘法（如 $25\times5$），通过合理映射也能高效利用 DMMA 指令。
2. **直接编程优于调用库函数**：
   - 对于嵌入在复杂 PDE 求解流程中的算子，手动调度 Tensor Core 更灵活且性能更优。
3. **Kernel Fusion 是释放潜力的关键**：
   - 与 DMMA 结合后，显著降低内存流量，提高数据复用率。
4. **卓越的大规模扩展性**：
   - 在近万 GPU 上实现了近乎理想的弱/强扩展效率，证明了方法在 exascale 级系统的可行性。

### 方法的局限性
- **对矩阵形状敏感**：当前 DMMA 优化主要针对特定形状（如 m25n5k4）。若元素自由度不同，需重新设计映射逻辑。
- **开发复杂度高**：需深入理解 PTX、warp-level matrix fragment、bank indexing 等底层机制，不利于快速移植。
- **GB200 能效未完全发挥**：虽然峰值更高，但在 FP64 密集负载下，其优势不如 GH200 明显，可能与其设计偏向 AI 工作负载有关。

### 未来工作方向
- **自动代码生成框架**：开发 DSL 或编译器 pass，自动将高阶 FE 算子分解并映射至 DMMA 指令。
- **支持更多 Tensor Core 指令**：探索 m16n8k4、m8n4k4 等变体以适配更多算子。
- **跨厂商兼容性**：研究 AMD Matrix Cores 或其他架构上的类似优化路径。
- **集成进主流库**：推动相关优化合并至 [MFEM](https://mfem.org) 开源主干，供社区广泛使用。

---

> 🔚 **总结一句话**：  
> 本文首次成功将 **FP64 Tensor Cores** 直接应用于大规模高阶有限元模拟，在 **GH200/GB200** 平台上实现了 **最高 2× 性能提升** 和 **83% 能效增益**，并在 **近万 GPU 上展示了近乎完美的扩展性**，为 exascale 科学计算提供了新的加速范式。

</details>

---

### 14. [PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies](https://arxiv.org/abs/2603.09216)

**Authors**: Sunjung Lee, Sanghoon Cha, Hyeonsu Kim, Seungwoo Seo, Yuhwan Ro, Sukhan Lee, Byeongho Kim, Yongjun Park, Kyomin Sohn, Seungwon Lee, Jaehoon Yu  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.09216v1  

#### Abstract
On-device deployments of large language models (LLMs) are rapidly proliferating across mobile and edge platforms. LLM inference comprises a compute-intensive prefill phase and a memory bandwidth-intensive decode phase, and the decode phase has been widely recognized as well-suited to processing-in-m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

该论文首次系统性地识别并解决了在 **PIM-enabled**（Processing-in-Memory）设备上部署大语言模型（LLM）时存在的两个关键系统级挑战：

- **Memory Attribute Inconsistency（内存属性不一致）**  
  - **Prefill 阶段**：计算密集型，依赖缓存重用，因此权重应放在 **cacheable 区域**。  
  - **Decode 阶段**：内存带宽受限，需触发 PIM 执行，要求权重位于 **non-cacheable 区域**，以确保 DRAM 请求能到达控制器。  
  → 两者对内存属性的要求冲突。

- **Weight Layout Inconsistency（权重布局不一致）**  
  - **Host 友好布局**：适合 GEMM 运算，利用 channel/bank 交错提升带宽。  
  - **PIM-aware 布局**：适合 GEMV 运算，要求矩阵行连续存储于单个 DRAM bank 内，以最大化 in-bank SIMD 利用率。  
  → 两种阶段需要不同的物理内存排布。

现有方法如 HBM-PIM 和 PAISE 采用 **权重复制（weight duplication）** 来解决，但这几乎使内存占用翻倍，在移动设备上不可行。

---

### 🚀 提出的新方法

作者提出 **PIM-SHERPA** —— 一种纯软件解决方案，无需硬件修改，即可在运行时动态解决上述两种不一致性。

#### 主要技术包括：

1. **DRAM Double Buffering (DDB)**  
   - 在 cacheable 区域分配两个小缓冲区。
   - 当前层执行 GEMM 时，并行将下一层的 PIM-aware 权重从 non-cacheable 区域预取到 cacheable 缓冲区，并进行 **swizzled memory copy** 转换为 host-friendly 布局。
   - 利用双缓冲机制实现 **计算与数据搬运的重叠（overlap）**，隐藏延迟。

2. **Online Weight Rearrangement with Swizzled Memory Copy (OWR)**  
   - 不使用预取，而是在每层 GEMM 前立即执行 swizzled memory copy。
   - 更简单，无需复杂同步，适用于输入序列较长、GEMM 时间占主导的场景。

3. **Swizzled Memory Copy (SMC)**  
   - 核心转换操作：将 non-cacheable 区域中的 PIM-aware 权重复制到 cacheable 区域，并重新排列为 host-friendly 布局。
   - 允许标准 GEMM 库直接使用，无需修改底层 kernel。

---

### 🔍 相比现有方法的优势

| 方法 | 是否需硬件改动 | 是否复制权重 | 内存开销 | 实现复杂度 |
|------|----------------|---------------|-----------|-------------|
| Weight Duplication (WD) | ❌ 否 | ✅ 是 | ⬆️⬆️ 极高（~2x） | ✅ 低 |
| FACIL | ✅ 是（需改 MC） | ❌ 否 | ✅ 低 | ⬆️ 高 |
| **PIM-SHERPA (DDB/OWR)** | ❌ 否 | ❌ 否 | ✅✅ 极低（仅 +1~2 层 buffer） | 中等 |

- **优势总结**：
  - **纯软件方案**，兼容现有 PIM 架构（如 LPDDR-PIM）。
  - **避免权重复制**，显著降低 DRAM 容量需求。
  - **动态解决属性与布局矛盾**，支持完整的 Prefill + Decode 流程。
  - **可扩展性强**，适用于 GPU/NPU 等多种加速器架构。

---

## 2. 核心实验方法和设置

### 🧪 实验平台

- **设备**：Samsung Galaxy S24+
- **SoC**：Exynos 2400（含 Cortex-X4 + A720 核心）
- **内存**：LPDDR5X-8533，4通道，带宽 68.264 GB/s
- **数据格式**：BF16
- **FLOP/B** ≈ 4.7
- **PIM 模拟**：基于 `PIMLibrary` 改造，模拟 LPDDR5X-PIM 行为，并通过 HBM-PIM 硬件验证其准确性（误差 < 3.6%）

### 📚 使用模型

- **Llama 3.2 1B** 和 **Llama 3.2 3B**（官方 on-device 版本）
- Batch Size = 1
- Hidden Size：2048（1B）、3072（3B）

### 🎯 评估指标

- **Required DRAM Capacity**：总内存占用
- **TTFT（Time to First Token）**：衡量 prefill 性能
- **End-to-end Inference Speedup**：相对于纯 host-only 系统的加速比
- **Execution Time / TPS（Tokens Per Second）**

### 🔀 对比基线方法

| 方法缩写 | 描述 |
|--------|------|
| **WD** | Weight Duplication：同时保留 cacheable host-friendly 和 non-cacheable PIM-aware 权重副本 |
| **FACIL-O** | FACIL under Oracle 假设（理想化版本，假设已解决 memory attribute 问题） |
| **S-DDB** | PIM-SHERPA with DRAM Double Buffering |
| **S-OWR** | PIM-SHERPA with Online Weight Rearrangement |

---

## 3. 主要实验结果和性能指标

### 💾 内存容量节省（DRAM Capacity Saving）

| 模型 | WD（复制） | S-DDB | S-OWR | 节省比例 |
|------|------------|--------|--------|----------|
| Llama 3.2 1B | ~4.94 GB | ~2.58 GB | ~2.56 GB | **47.8% ~ 48.5%** |
| Llama 3.2 3B | ~12.8 GB | ~6.48 GB | ~6.44 GB | **49.4% ~ 49.7%** |

> ✅ **结论**：相比权重复制，PIM-SHERPA 减少近一半的 DRAM 占用，接近 FACIL-O 水平，且无需硬件改动。

---

### ⏱️ TTFT 性能表现（Time to First Token）

- **输入序列长度（Input Sequence Length, SL）从 64 到 192 变化**
- **S-DDB** 在 SL ≥ 128 时基本追平 FACIL-O，实现有效重叠。
- **S-OWR** 因串行执行 SMC + GEMM，始终有固定延迟，但在长序列下逐渐逼近。

| 输入 SL | S-DDB vs FACIL-O | S-OWR vs FACIL-O |
|--------|------------------|------------------|
| 128    | 基本持平         | 略慢             |
| 192    | 几乎无差异       | 接近（差 ~5%）   |

> ✅ **关键发现**：随着 SL 增加，SMC 开销被 GEMM 时间摊销，**online rearrangement 成为可行选择**。

---

### 🚄 端到端推理速度提升（End-to-End Speedup）

使用 **LPDDR5X-PIM** 加速 Decode 阶段，比较整体吞吐：

- **S-DDB 和 FACIL-O 表现几乎一致**，最大可达 **3.3× 以上加速**。
- **S-OWR 在短序列略逊，但在长序列差距缩小**。
- 当输入 SL ≥ 128，三者性能趋同。

> 图 12 显示：在大多数输入/输出组合下，S-DDB 与 FACIL-O 的加速曲线高度重合。

---

### 🔬 消融分析（Ablation Insights）

- **SMC 延迟约为 0.6–2.5 秒**（取决于模型大小），主要受制于非缓存区域读取速度较慢。
- **DDB 中仅使用 2 个 copy thread**，未充分利用内存带宽（仅达峰值 25%），仍有优化空间。
- **调度延迟影响**：某些层因线程排队导致完成时间延迟（如 FF1 层中 T2 等待 X4 核心释放）。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **首次揭示了 PIM 上 LLM 推理中的 memory attribute inconsistency 问题**，这是此前研究忽略的关键瓶颈。
2. **PIM-SHERPA 是首个纯软件方案**，成功在不修改硬件的前提下，统一解决 memory attribute 与 weight layout 不一致问题。
3. **通过引入 small cacheable buffer + online swizzled copy**，实现了接近理论最优的性能，同时节省 **约 47.8–49.7% 的 DRAM 容量**。
4. **在真实移动端平台（Galaxy S24+）上验证可行性**，TTFT 与 FACIL-O 相当，证明其工程实用价值。

---

### ⚠️ 方法的局限性

1. **依赖足够长的输入序列**才能充分摊销 SMC 开销，在极短输入（SL < 64）时性能损失明显。
2. **当前实现未完全压榨内存带宽**，copy thread 数量有限，存在进一步优化空间。
3. **需要操作系统支持 large physically contiguous allocation（如 CMA）**，在部分系统上可能受限。
4. **SMC 实现增加了软件栈复杂性**，尤其对闭源 runtime 集成有一定挑战。

---

### 🔮 未来工作方向

1. **结合编译器优化自动插入 SMC 操作**，提升自动化程度。
2. **探索更高效的 swizzling 算法或硬件辅助轻量转换**。
3. **扩展至多设备协同推理场景**，如手机 + 边缘服务器联合调度。
4. **适配更高 FLOP/B 平台（如 NVIDIA GPU）**，研究其在 server 级 PIM 中的应用潜力。
5. **支持动态批处理（dynamic batching）和持续生成（streaming generation）** 场景。

---

## 总结

> **PIM-SHERPA 是一项面向现实 PIM 设备的重要软件创新**。它以极低的额外内存代价，解决了制约 on-device LLM 推理落地的核心矛盾——**prefill 与 decode 对内存属性和布局的根本冲突**。其实验结果表明，在主流移动平台上即可实现媲美硬件增强方案（如 FACIL）的性能，为 PIM 技术的商业化铺平了道路。

</details>

---

### 15. [Flash-KMeans: Fast and Memory-Efficient Exact K-Means](https://arxiv.org/abs/2603.09229)

**Authors**: Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Xiaoze Fan, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Kurt Keutzer, Song Han, Chenfeng Xu, Ion Stoica  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.09229v1  

#### Abstract
$k$-means has historically been positioned primarily as an offline processing primitive, typically used for dataset organization or embedding preprocessing rather than as a first-class component in online systems. In this work, we revisit this classical algorithm under the lens of modern AI system d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Flash-KMeans: Fast and Memory-Efficient Exact K-Means**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统的 **k-means** 实现（尤其是在 GPU 上）面临严重的性能瓶颈，主要源于以下两个方面：

- **IO 瓶颈**：在 assignment 阶段，标准实现需要显式构建一个 $N \times K$ 的距离矩阵 $D$ 并将其写入 High Bandwidth Memory (HBM)，造成巨大的内存读写开销。
- **原子写冲突（Atomic Write Contention）**：在 centroid update 阶段，多个线程并发对同一聚类中心进行 `atomic_add` 操作，导致严重的硬件级序列化和带宽浪费。

此外，在现代 AI 工作负载中，k-means 被越来越多地用作在线组件（如 LLM 中的 token routing、KV cache 压缩等），要求低延迟、高吞吐、支持动态形状和超大规模数据，而传统方法难以满足这些需求。

---

### **提出了什么新方法或新思路**
作者提出 **flash-kmeans**，一种面向现代 GPU 架构优化的、数学上精确的 k-means 实现，其核心是通过系统-算法协同设计（algorithm-system co-design）来规避底层硬件瓶颈。主要包括三大技术创新：

#### **(1) FlashAssign：消除中间距离矩阵的在线 argmin**
- 将 distance computation 与 row-wise argmin 融合为一个流式核函数。
- 在片上（on-chip）维护每个点的当前最小距离和对应 centroid index，逐块加载 centroids 进行比较更新。
- 完全避免了 $N \times K$ 距离矩阵 $D$ 的显式 materialization，大幅减少 HBM 访问。

#### **(2) Sort-Inverse Update：低冲突的 centroid 更新机制**
- 引入 **inverse mapping**：先对 assignment vector $a$ 执行 `argsort` 得到排序索引 `sorted_idx`，使得相同 cluster ID 的 token 在逻辑上连续排列。
- 每个 CTA 处理一段连续 cluster-id segment，利用 `sorted_idx` 从原始数据中 gather 特征，在片上完成局部累加。
- 只在 segment 边界处执行一次 global `atomic_add`，极大减少了原子操作数量。

#### **(3) 系统级协同优化**
- **Chunked Stream Overlap**：对于超出 VRAM 的大规模数据，采用异步 CUDA Streams 实现 host-to-device 传输与计算重叠，隐藏 PCIe 延迟。
- **Cache-Aware Compile Heuristic**：基于 L1/L2 缓存大小和问题规模直接推导最优 kernel 配置，避免耗时的 exhaustive auto-tuning，显著降低 time-to-first-run。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **性能** | 端到端加速最高达 **17.9×**，远超 cuML (**33×**) 和 FAISS (**>200×**) |
| **内存效率** | 不显式存储 $N \times K$ 距离矩阵，支持更大规模任务（如 $N=10^9$） |
| **部署友好性** | 支持 out-of-core 和动态 shape，配置开销降低 **175×**，几乎无性能损失 |
| **数学正确性** | 不引入任何近似，输出结果与标准 k-means 完全一致 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
论文未使用特定公开数据集，而是生成合成数据以全面测试不同维度组合下的性能表现：
- 数据点数 $N$: 从 $16K$ 到 $1B$
- 聚类数 $K$: 从 $1K$ 到 $64K$
- 特征维度 $d$: 128 或 512
- 批次大小 $B$: 最高达 32

覆盖了多种典型 workload regime：
- 内存密集型（large N, large K）
- 计算密集型（large N, small K）
- 小批量高频调用场景（small N, small K）

---

### **实验设置和评估指标**

#### **硬件平台**
- **GPU**: NVIDIA H200
- **CUDA 版本**: 12.8
- 自定义 CUDA kernel 实现，结合 Triton 进行部分优化

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **End-to-end latency per iteration** | 单轮 k-means 的总运行时间 |
| **Kernel-level latency** | 分别测量 assignment 和 update 阶段耗时 |
| **Throughput** | 每秒处理的数据量或迭代次数 |
| **Memory footprint** | GPU 显存峰值占用 |
| **Time-to-first-run** | 首次编译 + 配置搜索的时间 |
| **Effective bandwidth** | 实测内存带宽利用率 |

---

### **基线方法对比**
与以下四个高度优化的 baseline 对比：
1. **fast_pytorch_kmeans**：基于 PyTorch 的快速实现
2. **fastkmeans** (Clavié and Warner, 2025)：Triton 加速版本
3. **cuML** (NVIDIA RAPIDS)：工业级机器学习库
4. **FAISS** (Facebook AI Similarity Search)：广泛用于向量检索的库

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 场景 | 性能提升 |
|------|---------|
| **端到端速度** | 最高 **17.9×** 超过 best baseline (`fast_pytorch_kmeans`) |
| **vs cuML** | 最高 **33×** 加速 |
| **vs FAISS** | 超过 **200×** 加速 |
| **FlashAssign 内核** | assignment 阶段最高 **21.2×** 加速 |
| **Sort-Inverse Update 内核** | update 阶段最高 **6.3×** 加速 |
| **Out-of-Core 扩展性** | 支持 $N = 10^9$，最大 **10.5×** 端到端加速 |
| **配置开销** | 编译+调优时间减少 **175×**，性能差距 < **0.3%** |

---

### **与基线方法的对比结果**

#### **Figure 3: End-to-End Latency Comparison**
- 在 large N, large K 场景下（如 $N=1M, K=64K$），PyTorch 实现因 OOM 失败，而 flash-kmeans 成功运行并实现 >5.4× 加速。
- 在 compute-intensive 场景（$N=8M, K=1K$），仍实现 **17.9×** 加速。
- 在小批量批处理（$B=32$）下也达到 **15.3×** 加速，说明框架开销控制优秀。

#### **Figure 4: Kernel-Level Breakdown**
- FlashAssign 在 $N=1M, K=8192$ 下将 assignment 时间从 **122.5ms → 5.8ms**（**21.2×**）。
- Sort-Inverse Update 在 $N=33M, K=4096$ 下实现 **6.3×** 加速，有效带宽接近理论上限。

#### **Figure 5: Cache-Aware Heuristic 效果**
- exhaustive tuning 需要超过 **325 秒**进行配置搜索。
- 本文 heuristic 仅需 **<2.5 秒**，提速 **175×**。
- 运行时性能与最优配置相差不到 **0.3%**，真正做到“即插即用”。

---

### **消融实验结果（Ablation Study）**
虽然没有单独列出 ablation 表格，但从模块化分析可看出：
- **FlashAssign 单独作用**：消除 $O(NK)$ HBM 流量，assignment 阶段成为轻量操作。
- **Sort-Inverse Update 单独作用**：将原子操作从 $O(Nd)$ 降至 $O((K + N/B_N)d)$，彻底缓解 contention。
- **Chunked Stream Overlap**：在 $N=400M$ 场景下实现 **10.5×** 加速，证明通信-计算重叠有效性。
- **Compile Heuristic**：在动态 shape 下保持高性能，验证其泛化能力。

---

## **4. 关键结论和发现**

### **论文的主要发现**
1. **k-means 的性能瓶颈不在算法复杂度，而在实现层面的 IO 与同步问题**  
   即使理论 FLOPs 很低，传统实现仍受制于 HBM 访问和 atomic contention。

2. **借鉴 FlashAttention 的 IO-aware 设计哲学可成功迁移至其他经典算法**  
   FlashAssign 的思想与 FlashAttention 一脉相承——通过融合操作避免中间结果 materialization。

3. **排序 + 分段归约（sort + segmented reduction）是解决 irregular scatter 的通用范式**  
   类似技术已在 MoE、Attention 中被验证，本工作进一步推广至 k-means。

4. **系统级优化对实际部署至关重要**  
   Chunking + streaming + 编译启发式共同保障了 flash-kmeans 在真实场景中的可用性和鲁棒性。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **依赖 GPU 架构特性** | 如 shared memory、async prefetch、high-bandwidth sorting primitives，在 CPU 或低端 GPU 上可能无法发挥全部优势 |
| **排序带来额外开销** | `argsort(a)` 虽只作用于整数向量，但在极端不平衡 clustering 下可能影响性能 |
| **不适用于 streaming k-means 或 online variant** | 当前聚焦 batched Lloyd’s iteration，未考虑增量更新场景 |

---

### **未来工作方向**
1. **扩展至其他 clustering 算法**  
   如 k-medoids、DBSCAN 等同样存在 IO 和 irregular access 模式的问题。

2. **支持更多 distance metric**  
   当前基于 Euclidean distance，可拓展至 cosine similarity 或 Mahalanobis distance。

3. **集成进主流框架**  
   将 flash-kmeans 作为 backend 集成进 PyTorch、RAPIDS、HuggingFace 等生态。

4. **探索稀疏化与量化版本**  
   结合 low-bit arithmetic 和 structured sparsity，进一步提升能效比。

5. **适配多 GPU / 分布式环境**  
   当前为单卡优化，未来可研究 AllReduce-aware centroid update 和数据分片策略。

---

> ✅ **总结一句话**：  
> **flash-kmeans 通过 IO-aware kernel fusion 与 contention-free aggregation，实现了数学精确、极致高效且易于部署的 GPU k-means，为现代生成式 AI 基础设施提供了关键支撑。**

</details>

---

### 16. [Multi-DNN Inference of Sparse Models on Edge SoCs](https://arxiv.org/abs/2603.09642)

**Authors**: Jiawei Luo, Di Wu, Simon Dobson, Blesson Varghese  
**Category**: cs.DC  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.09642v1  

#### Abstract
Modern edge applications increasingly require multi-DNN inference systems to execute tasks on heterogeneous processors, gaining performance from both concurrent execution and from matching each model to the most suited accelerator. However, existing systems support only a single model (or a few spar...

---

### 17. [Quantifying Memorization and Privacy Risks in Genomic Language Models](https://arxiv.org/abs/2603.08913)

**Authors**: Alexander Nemecek, Wenbiao Li, Xiaoqian Jiang, Jaideep Vaidya, Erman Ayday  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.08913v1  

#### Abstract
Genomic language models (GLMs) have emerged as powerful tools for learning representations of DNA sequences, enabling advances in variant prediction, regulatory element identification, and cross-task transfer learning. However, as these models are increasingly trained or fine-tuned on sensitive geno...

---

### 18. [MAPLE: Elevating Medical Reasoning from Statistical Consensus to Process-Led Alignment](https://arxiv.org/abs/2603.08987)

**Authors**: Kailong Fan, Anqi Pu, Yichen Wu, Wanhua Li, Yicong Li, Hanspeter Pfister, Huafeng Liu, Xiang Li, Quanzheng Li, Ning Guo  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.08987v1  

#### Abstract
Recent advances in medical large language models have explored Test-Time Reinforcement Learning (TTRL) to enhance reasoning. However, standard TTRL often relies on majority voting (MV) as a heuristic supervision signal, which can be unreliable in complex medical scenarios where the most frequent rea...

---

### 19. [Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards](https://arxiv.org/abs/2603.09117)

**Authors**: Zhengzhao Ma, Xueru Wen, Boxi Cao, Yaojie Lu, Hongyu Lin, Jinglin Yang, Min He, Xianpei Han, Le Sun  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.09117v1  

#### Abstract
Reinforcement Learning from Verifiable Rewards (RLVR) significantly enhances large language models (LLMs) reasoning but severely suffers from calibration degeneration, where models become excessively over-confident in incorrect answers. Previous studies devote to directly incorporating calibration o...

---

### 20. [AgentOS: From Application Silos to a Natural Language-Driven Data Ecosystem](https://arxiv.org/abs/2603.08938)

**Authors**: Rui Liu, Tao Zhe, Dongjie Wang, Zijun Yao, Kunpeng Liu, Yanjie Fu, Huan Liu, Jian Pei  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.08938v1  

#### Abstract
The rapid emergence of open-source, locally hosted intelligent agents marks a critical inflection point in human-computer interaction. Systems such as OpenClaw demonstrate that Large Language Model (LLM)-based agents can autonomously operate local computing environments, orchestrate workflows, and i...

---

### 21. [Social-R1: Towards Human-like Social Reasoning in LLMs](https://arxiv.org/abs/2603.09249)

**Authors**: Jincenzi Wu, Yuxuan Lei, Jianxun Lian, Yitian Huang, Lexin Zhou, Haotian Li, Xing Xie, Helen Meng  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.09249v1  

#### Abstract
While large language models demonstrate remarkable capabilities across numerous domains, social intelligence - the capacity to perceive social cues, infer mental states, and generate appropriate responses - remains a critical challenge, particularly for enabling effective human-AI collaboration and ...

---

### 22. [MiniAppBench: Evaluating the Shift from Text to Interactive HTML Responses in LLM-Powered Assistants](https://arxiv.org/abs/2603.09652)

**Authors**: Zuhao Zhang, Chengyue Yu, Yuante Li, Chenyi Zhuang, Linjian Mo, Shuai Li  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.09652v1  

#### Abstract
With the rapid advancement of Large Language Models (LLMs) in code generation, human-AI interaction is evolving from static text responses to dynamic, interactive HTML-based applications, which we term MiniApps. These applications require models to not only render visual interfaces but also construc...

---

### 23. [MedMASLab: A Unified Orchestration Framework for Benchmarking Multimodal Medical Multi-Agent Systems](https://arxiv.org/abs/2603.09909)

**Authors**: Yunhang Qian, Xiaobin Hu, Jiaquan Yu, Siyang Xin, Xiaokun Chen, Jiangning Zhang, Peng-Tao Jiang, Jiawei Liu, Hongwei Bran Li  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.09909v1  

#### Abstract
While Multi-Agent Systems (MAS) show potential for complex clinical decision support, the field remains hindered by architectural fragmentation and the lack of standardized multimodal integration. Current medical MAS research suffers from non-uniform data ingestion pipelines, inconsistent visual-rea...

---

### 24. [SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models](https://arxiv.org/abs/2603.09215)

**Authors**: Hsiao-Ying Huang, Cheng-Han Chiang, Hung-yi Lee  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.09215v1  

#### Abstract
Interleaved spoken language models (SLMs) alternately generate text and speech tokens, but decoding at full transformer depth for every step becomes costly, especially due to long speech sequences. We propose SPAR-K, a modality-aware early exit framework designed to accelerate interleaved SLM infere...

---

### 25. [Efficient Reasoning at Fixed Test-Time Cost via Length-Aware Attention Priors and Gain-Aware Training](https://arxiv.org/abs/2603.09253)

**Authors**: Rian Atri  
**Category**: cs.LG  
**Published**: 2026-03-11  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.09253v1  

#### Abstract
We study efficient reasoning under tight compute. We ask how to make structured, correct decisions without increasing test time cost. We add two training only components to small and medium Transformers that also transfer to broader differentiable optimizers. First, a length aware attention prior bu...

---

### 26. [The Reasoning Trap -- Logical Reasoning as a Mechanistic Pathway to Situational Awareness](https://arxiv.org/abs/2603.09200)

**Authors**: Subramanyam Sahoo, Aman Chadha, Vinija Jain, Divya Chaudhary  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09200v1  

#### Abstract
Situational awareness, the capacity of an AI system to recognize its own nature, understand its training and deployment context, and reason strategically about its circumstances, is widely considered among the most dangerous emergent capabilities in advanced AI systems. Separately, a growing researc...

---

### 27. [PrivPRISM: Automatically Detecting Discrepancies Between Google Play Data Safety Declarations and Developer Privacy Policies](https://arxiv.org/abs/2603.09214)

**Authors**: Bhanuka Silva, Dishanika Denipitiyage, Anirban Mahanti, Aruna Seneviratne, Suranga Seneviratne  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09214v1  

#### Abstract
End-users seldom read verbose privacy policies, leading app stores like Google Play to mandate simplified data safety declarations as a user-friendly alternative. However, these self-declared disclosures often contradict the full privacy policies, deceiving users about actual data practices and viol...

---

### 28. [Logos: An evolvable reasoning engine for rational molecular design](https://arxiv.org/abs/2603.09268)

**Authors**: Haibin Wen, Zhe Zhao, Fanfu Wang, Tianyi Xu, Hao Zhang, Chao Yang, Ye Wei  
**Category**: cs.AI  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09268v1  

#### Abstract
The discovery and design of functional molecules remain central challenges across chemistry,biology, and materials science. While recent advances in machine learning have accelerated molecular property prediction and candidate generation, existing models tend to excel either in physical fidelity wit...

---

### 29. [Bioalignment: Measuring and Improving LLM Disposition Toward Biological Systems for AI Safety](https://arxiv.org/abs/2603.09154)

**Authors**: Trent R Northen, Mingxun Wang  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09154v1  

#### Abstract
Large language models (LLMs) trained on internet-scale corpora can exhibit systematic biases that increase the probability of unwanted behavior. In this study, we examined potential biases towards synthetic vs. biological technological solutions across four domains (materials, energy, manufacturing,...

---

### 30. [DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval](https://arxiv.org/abs/2603.09185)

**Authors**: Taegyeong Lee, Jiwon Park, Seunghyun Hwang, JooYoung Jang  
**Category**: cs.CL  
**Published**: 2026-03-11  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.09185v1  

#### Abstract
Recent advances in Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) have enabled diverse retrieval methods. However, existing retrieval methods often fail to accurately retrieve results for negation and exclusion queries. To address this limitation, prior approaches rely on embe...

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
