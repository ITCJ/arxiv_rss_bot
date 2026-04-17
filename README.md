# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-17 07:16:29 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Material-Agnostic Zero-Shot Thermal Inference for Metal Additive Manufacturing via a Parametric PINN Framework](https://arxiv.org/abs/2604.14562)

**Authors**: Hyeonsu Lee, Jihoon Jeong  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.14562v1  

#### Abstract
Accurate thermal modeling in metal additive manufacturing (AM) is essential for understanding the process-structure-performance relationship. While prior studies have explored generalization across unseen process conditions, they often require extensive datasets, costly retraining, or pre-training. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **跨材料零样本热场预测难题**：传统金属增材制造（metal AM）中的热建模方法（如FEM、data-driven模型）通常针对特定材料进行训练或仿真，难以泛化到未见过的新材料。现有Physics-Informed Neural Networks（PINNs）多为非参数化（non-parametric），在材料变化时需重新训练，限制了其在多材料场景下的实用性。
- **缺乏物理一致性和训练稳定性**：纯数据驱动模型缺乏物理可解释性；而标准PINNs在处理不同材料时因热物性差异大导致梯度失衡，引发训练不稳定。

### **提出的新方法与创新思路**
本文提出了一种**参数化PINN框架（parametric PINN framework）**，实现**无需标签数据、无需重训练或预训练的跨材料零样本（zero-shot）热推断**。其三大核心创新如下：

#### **(1) 解耦式参数化PINN架构（Decoupled Parametric PINN Architecture）**
- 将**空间-时间坐标** $(x,t)$ 和**材料属性向量** $\lambda$ 分别输入两个独立分支网络（spatiotemporal branch & material branch），通过**条件调制机制**（conditional modulation，受FiLM启发）融合。
- 优势：更符合物理规律——材料属性以乘法形式影响导热方程中的系数项（如热导率 $k$、比热容 $C_p$）。相比传统的单体式（monolithic）拼接输入方式，该设计提升了表示能力与泛化性。

#### **(2) 物理引导的输出缩放（Physics-Guided Output Scaling）**
- 引入基于**Rosenthal解析解**估算峰值温度 $T_{\text{max}}(\lambda)$ 的机制，并结合修正因子 $K$ 构造输出变换：
  $$
  T_{\text{phys}} = T_\infty + K \cdot T_{\text{max}}(\lambda) \cdot \text{Softplus}(T_e)
  $$
- 作用：自动适应不同材料的温度尺度，缓解因材料间温差巨大导致的梯度爆炸/消失问题，显著提升训练稳定性和收敛速度。

#### **(3) 混合优化策略（Hybrid Optimization Strategy）**
- 结合**Adam**（用于全局探索）与**随机小批量L-BFGS**（用于局部精细收敛）。
- 创新点：在L-BFGS阶段引入**动态重采样机制**——当曲率信息停滞时重新采样collocation点，避免过拟合特定配置并增强优化鲁棒性。

### **相比现有方法的优势**
| 维度 | 本文方法 | 传统方法（如N-PINN、P-PINN） |
|------|----------|-----------------------------|
| 泛化能力 | ✅ 支持任意新材料的zero-shot推理 | ❌ 需对每种新材料单独训练 |
| 训练效率 | ⏱️ 仅需约4.4%的epoch即可超越基线 | 🐢 动辄数万轮迭代 |
| 参数量 | 🔽 更少可学习参数（9,641 vs >11k） | 🔺 参数更多 |
| 物理一致性 | ✅ 显式嵌入物理边界约束 | ✅ 有但易受数值不稳破坏 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **无真实实验数据**，采用高保真**有限元模拟（FEM）生成的合成数据**作为ground truth。
- 使用开源库 **JAX-AM** 进行FEM仿真，模拟**裸板激光粉末床熔融（bare plate LPBF）过程**。
- 材料属性空间 $\mathcal{M}$ 定义为连续区间：
  $$
  \rho \in [3000,10000],\quad C_p \in [300,1000],\quad k \in [3,50]\ \text{W/(m·K)}
  $$

### **实验设置**
- **任务**：预测三维空间+时间域内的瞬态温度场 $T(x,t;\lambda)$。
- **输入**：归一化的$(x,y,z,t)$ 和材料属性 $(\rho, C_p, k)$。
- **Collocation Sampling**：
  - 时间步长0.1s，共30个时间片；
  - 多分辨率网格：靠近激光区域加密至0.25mm，表面细化至0.5mm。
- **训练配置**：
  - 总epoch数：10,000（Adam前2,000 + L-BFGS后8,000）
  - GPU：NVIDIA RTX 5090
  - 开源代码地址：[https://github.com/hsleecri/MaterialAgnosticTempPred](https://github.com/hsleecri/MaterialAgnosticTempPred)

### **评估指标**
- **相对L2误差（Relative L2 Error）**：
  $$
  \text{L2 error} = \frac{\sqrt{\sum_n (T_{\text{pred}}^{(n)} - T_{\text{FEM}}^{(n)})^2}}{\sqrt{\sum_n (T_{\text{FEM}}^{(n)})^2}} \times 100\%
  $$
- 所有结果报告**五次随机种子下的均值±标准差**。

### **基线方法对比**
| 基线模型 | 类型 | 描述 |
|--------|-----|------|
| **N-PINN** | Non-parametric PINN | 固定材料参数，每个材料需单独训练 |
| **P-PINN** | Monolithic Parametric PINN | 将$(x,t,\lambda)$直接拼接输入单一网络 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 在分布内材料上的表现（ID: Ti-6Al-4V, Inconel 718, SS 316L）**

| 材料 | 方法 | #Params | L2 Error (%) |
|------|------|---------|------------|
| Ti-6Al-4V | N-PINN | 11,341 | 6.19 ± 1.00 |
| | **Proposed** | **9,641** | **2.71 ± 0.34** (**↓56.2%**) |
| Inconel 718 | N-PINN | — | 6.09 ± 0.83 |
| | **Proposed** | — | **2.18 ± 0.45** (**↓64.2%**) |
| SS 316L | N-PINN | — | 5.94 ± 0.63 |
| | **Proposed** | — | **2.17 ± 0.53** (**↓63.4%**) |

> ✅ **结论**：在更少参数下，L2误差降低**56–64%**，且方差更低，表明更强的鲁棒性。

#### **(2) 与P-PINN对比（相同训练设置）**

| 材料 | 方法 | L2 Error (%) |
|------|------|------------|
| Ti-6Al-4V | P-PINN | 7.59 ± 3.86 |
| | **Proposed** | **2.71 ± 0.34** (**↓64.3%**) |
| Inconel 718 | P-PINN | 6.12 ± 3.22 |
| | **Proposed** | **2.18 ± 0.45** (**↓64.4%**) |
| SS 316L | P-PINN | 5.84 ± 2.59 |
| | **Proposed** | **2.17 ± 0.53** (**↓62.8%**) |

> ✅ 即使P-PINN失败的最佳情况也劣于本文最差情况，说明**解耦架构极大增强了训练稳定性**。

#### **(3) 训练效率对比**
- **本文方法达到N-PINN水平精度所需epoch**：约 **2,200**
- **N-PINN完成训练所需epoch**：50,000
- ➡️ **仅用4.4%的训练轮次即超越基线**

#### **(4) Out-of-Distribution（OOD）材料测试**

| 材料（属性） | 方法 | L2 Error (%) |
|------------|------|-------------|
| **AlSi10Mg**<br>($k=150$) | N-PINN | 4.25 ± 0.35 |
| | P-PINN | 3.25 ± 0.39 |
| | **Proposed** | **1.75 ± 0.28** |
| **Copper**<br>($k=401$) | N-PINN | 1.63 ± 0.13 |
| | P-PINN | 14.38 ± 23.92 (**严重发散！**) |
| | **Proposed** | **0.69 ± 0.14** |

> ✅ 在极端高导热铜上仍保持<1%误差，验证了强大的**外推能力（extrapolation capability）**。

---

### **消融实验结果（Ablation Study）**

#### **(1) 输出缩放策略对比（Table 6）**
| 缩放方式 | Ti-6Al-4V L2 Error (%) |
|--------|-----------------------|
| 无缩放（Raw output） | 77.81 ± 31.91 |
| $T_\infty + \text{Softplus}(T_e)$ | 38.59 ± 0.00 |
| 手动固定$T_{\text{max}}=3000$ | 5.66 ± 1.45 |
| 学习网络预测$T_{\text{max}}$ | 38.50 ± 0.09 |
| **物理引导 $T_{\text{max}}(\lambda)$ + $K=1.5$** | **2.71 ± 0.34** |

> 🔍 发现：手动设定$T_{\text{max}}$敏感且次优；学习方式反而退化；**物理先验是最有效稳定的方案**。

#### **(2) 混合优化策略有效性（Table 8 & 9）**
- 使用**随机mini-batch L-BFGS**比全批量L-BFGS进一步降低误差和方差。
- 该策略不仅对本文模型有效，也能提升**N-PINN和P-PINN的性能**，说明具有**广泛适用性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **参数化解耦架构优于传统拼接式输入**：将材料属性作为条件变量进行调制，能更好捕捉其在PDE中“乘法系数”的角色，提升表示能力和泛化性。
2. **物理引导的输出缩放至关重要**：利用Rosenthal解估计$T_{\text{max}}(\lambda)$ 可大幅缓解跨材料训练中的梯度失衡问题，是实现稳定zero-shot推理的关键。
3. **混合优化显著加速收敛**：Adam + 随机L-BFGS组合可在极短时间内达到高精度，解决了PINNs普遍存在的“长训练周期”瓶颈。
4. **真正实现了material-agnostic建模**：一个模型通用于任意材料（含OOD），无需再训练，具备高度实用价值。

### **方法的局限性**
1. **假设材料属性为常数**：未考虑温度依赖性（temperature-dependent properties），实际应用中可能需要校准。
2. **当前仅限热传导模型**：未包含流体动力学、相变等复杂物理，尤其在熔池区精度受限。
3. **统一采样策略非最优**：不同材料的热扩散速率差异大，固定collocation分布可能无法充分解析所有时空尺度。
4. **仅适用于单材料系统**：尚未扩展至multi-material AM场景。

### **未来工作方向**
1. **开发自适应/材料感知的collocation采样策略**，根据$\lambda$动态调整点密度。
2. **集成更多物理机制**：引入thermo-fluid coupling、phase transition modeling，提升熔池区域预测精度。
3. **扩展至多工艺参数联合泛化**：将激光功率、扫描速度等也纳入参数化输入，构建全工艺-材料联合zero-shot模型。
4. **应用于多材料增材制造系统**，探索异质材料界面处的热行为建模。
5. **结合在线传感数据进行实时修正与不确定性量化**。

---

> 💡 **总体评价**：本研究提出了一套高效、稳定、可扩展的**material-agnostic zero-shot thermal inference框架**，为金属AM中的快速虚拟工艺设计与缺陷预测提供了强有力的工具，推动了PINNs在工业级多材料场景中的落地应用。

</details>

---

### 2. [Rethinking AI Hardware: A Three-Layer Cognitive Architecture for Autonomous Agents](https://arxiv.org/abs/2604.13757)

**Authors**: Li Chen  
**Category**: cs.AI  
**Published**: 2026-04-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.13757v1  

#### Abstract
The next generation of autonomous AI systems will be constrained not only by model capability, but by how intelligence is structured across heterogeneous hardware. Current paradigms -- cloud-centric AI, on-device inference, and edge-cloud pipelines -- treat planning, reasoning, and execution as a mo...

---

### 3. [Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter](https://arxiv.org/abs/2604.15039)

**Authors**: Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang  
**Category**: cs.DC  
**Published**: 2026-04-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.15039v1  

#### Abstract
Prefill-decode (PD) disaggregation has become the standard architecture for large-scale LLM serving, but in practice its deployment boundary is still determined by KVCache transfer. In conventional dense-attention models, prefill generates huge KVCache traffics that keep prefill and decode tightly c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大规模 LLM 推理服务广泛采用 **Prefill-Decode (PD) disaggregation** 架构，将计算密集型的 *prefill* 阶段与内存带宽敏感的 *decode* 阶段分离。然而，这种架构在实践中仍受限于 **KVCache 跨节点传输开销**，导致 prefill 和 decode 必须部署在同一高带宽、低延迟的 RDMA 网络域内（通常为单个数据中心），限制了以下能力：
- **异构硬件部署**：无法灵活使用专用于 prefill 或 decode 的不同加速器（如 Rubin CPX vs LPU）；
- **资源弹性扩展**：prefill 与 decode 的硬件比例固定，难以随流量动态调整；
- **跨数据中心部署**：因跨中心网络带宽有限，KVCache 传输成为瓶颈。

尽管新一代混合注意力模型（hybrid-attention models）显著降低了 KVCache 大小，但仅靠模型优化不足以实现跨数据中心的实用化部署——真实场景中的请求长度偏斜、突发流量、前缀缓存分布不均等问题依然存在。

---

### 🚀 提出的新方法：**Prefill-as-a-Service (PrfaaS)**

作者提出 **PrfaaS** ——一种支持跨数据中心部署的新型 LLM 服务架构，其核心思想是：

> **Selective Offloading + Cross-Datacenter KVCache Transfer**

#### 主要创新点：
1. **选择性卸载长上下文请求**  
   并非所有请求都适合远程处理。PrfaaS 只将“长且未命中前缀缓存”的 prefill 请求卸载到专用的 **compute-dense prefill 集群**，短请求仍在本地 PD 集群处理，避免不必要的网络开销。

2. **系统级协同设计（System-Model Co-design）**  
   不仅依赖模型侧的 KVCache 压缩（如 KDA、SWA 等机制），还结合了：
   - **带宽感知调度器（Bandwidth-aware Scheduler）**：实时监控链路利用率与队列深度，动态调整路由策略以防止拥塞；
   - **缓存感知请求放置（Cache-aware Request Placement）**：利用全局 KVCache 管理器追踪前缀缓存位置，优先复用已有状态；
   - **混合前缀缓存池（Hybrid Prefix Cache Pool）**：统一管理线性注意力的状态与全注意力的 KVCache 块，提升缓存效率。

3. **解耦异构集群间的网络依赖**  
   允许 prefill 集群与 decode 集群通过普通 **Ethernet（如 VPC Peering）** 连接，无需共享昂贵的 RDMA fabric，从而实现：
   - 跨数据中心部署
   - 异构硬件独立扩缩容
   - 更低成本的资源利用（例如使用性价比更高的 prefill 专用芯片）

---

### 🔍 相比现有方法的优势

| 维度 | 传统 PD Disaggregation | Naive Heterogeneous PD | PrfaaS |
|------|------------------------|------------------------|-------|
| 部署范围 | 单数据中心内（RDMA） | 同一紧耦合集群 | **跨数据中心** |
| 网络要求 | RDMA 级高带宽互联 | RDMA 或高性能网络 | **Commodity Ethernet** |
| 是否支持异构硬件 | 困难（需共置） | 是，但缺乏调度优化 | **是，且可独立扩展** |
| KVCache 传输量 | 高（dense attention） | 高 | **大幅降低（hybrid + selective）** |
| 资源利用率 | 中等（静态配比） | 差（易失衡） | **高（动态负载均衡）** |

---

## 2. 核心实验方法和设置

### 📊 实验模型
使用 Moonshot AI 内部开发的一个 **1T 参数 hybrid-attention 模型**，架构基于 **Kimi Linear [26]**，采用 KDA:MLA = 3:1 的层堆叠结构，具备显著的 KVCache 压缩特性。

### ⚙️ 实验设置
- **PrfaaS 集群**：32 × H200 GPU，专用于处理长上下文 prefill 请求；
- **本地 PD 集群**：64 × H20 GPU，运行标准 PD disaggregation，支持 prefill 和 decode；
- **连接方式**：两个集群通过 **VPC 网络互联**，提供约 **100 Gbps 跨数据中心带宽**；
- **对比基线**：
  1. **Homogeneous PD**：96 × H20 GPU 的纯同构集群；
  2. **Naive Heterogeneous PD**：所有 prefill 分配给 H200，所有 decode 分配给 H20，无智能调度或选择性卸载；

### 📈 评估指标
| 指标 | 定义 |
|------|------|
| `Amax` | 最大稳态吞吐量（requests/sec） |
| `TTFT`（Time to First Token） | 用户发出请求到收到首个 token 的延迟 |
| `KV Throughput (Pkv)` | 单实例单位时间生成的 KVCache 数据量（Gbps） |
| `Bout` | PrfaaS 集群出口带宽需求 |
| `Throughput Gain` | 相对于基线的吞吐提升倍数 |

此外，构建了一个 **分析型吞吐模型**（analytical throughput model），用于指导最优路由阈值 `t` 和 prefill/decode 实例比例 `Np/Nd` 的配置。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 6）

| 指标 | PrfaaS-PD | Homogeneous PD | Naive Heterogeneous PD |
|------|-----------|----------------|-------------------------|
| **最大吞吐 `Amax` (req/s)** | **3.24** | 2.11 | 2.45 |
| **相对吞吐增益** | **1.54×** | 1.00× | 1.16× |
| **平均 TTFT (s)** | 2.22 | 4.44 | 1.74 |
| **P90 TTFT (s)** | **3.51** | 9.73 | 3.51 |
| **PrfaaS 出口带宽（均值）** | **13 Gbps** | — | — |

---

### 🔬 对比结果分析

#### ✅ vs Homogeneous PD
- 吞吐量提升 **54%**（3.24 → 2.11 req/s）
- P90 TTFT 降低 **64%**（从 9.73s 降至 3.51s）
- 原因：长请求被卸载至高性能 H200 集群并行处理，缓解了本地 prefill 资源竞争

#### ✅ vs Naive Heterogeneous PD
- 吞吐量高出 **32%**（3.24 vs 2.45 req/s）
- 尽管 naive 方案也用了异构硬件，但由于缺乏调度优化，出现严重负载失衡（decode 成为瓶颈）
- PrfaaS 通过 **选择性卸载 + 动态资源调配** 显著提升了整体利用率

#### 💡 跨数据中心带宽消耗极低
- 在 100 Gbps 总带宽下，PrfaaS 集群平均仅占用 **13 Gbps（13%）**
- 峰值也不超过带宽容量，验证了 **commodity Ethernet 支持跨数据中心 KVCache 传输的可行性**

---

### 🔍 消融实验与关键参数分析（Grid Search 结果）

- **最优路由阈值 `t = 19.4K tokens`**  
  表明只有当增量输入长度超过 ~19K 时才应卸载至 PrfaaS，平衡了计算收益与网络成本。

- **最优 prefill/decode 实例比 `Np:Nd = 3:5`（在本地 PD 集群）**  
  显示 decode 资源更稀缺，需更多实例支撑；而 prefill 可部分由外部集群承担。

- **约 50% 的请求被卸载至 PrfaaS**  
  主要是长尾请求，体现了“精准卸载”策略的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **KVCache 减少 ≠ 可直接跨数据中心部署**  
   尽管 hybrid-attention 模型将 KVCache 规模压缩数十倍（如 Ring-2.5-1T 达 36× 压缩），但仍需 **系统层面的设计**（如 selective offloading, bandwidth-aware scheduling）才能真正实现高效跨数据中心服务。

2. **PrfaaS 实现了“实用化的跨数据中心 PD disaggregation”**  
   在仅消耗 **13 Gbps Ethernet 带宽** 的前提下，获得 **54% 吞吐提升** 和 **64% P90 TTFT 下降**，证明该架构兼具高性能与低成本优势。

3. **异构部署必须配合智能调度**  
   “naive heterogeneous” 方案表现不佳，说明单纯拆分硬件角色而不考虑请求特征、缓存分布和带宽约束会导致资源浪费和性能下降。

4. **未来 LLM 服务将走向“地理分布式 + 功能专业化”架构**  
   PrfaaS 支持将 prefill 部署在算力强但带宽弱的数据中心，decode 部署在靠近用户的边缘节点，推动 LLM 服务基础设施的进一步解耦与优化。

---

### ⚠️ 局限性

1. **依赖 hybrid-attention 模型普及**  
   当前大多数主流模型仍为 dense attention，PrfaaS 的优势在传统模型上会减弱。

2. **对前缀缓存管理复杂度较高**  
   全局 KVCache 管理器需要维护多集群间的缓存元数据一致性，增加了系统复杂性。

3. **未考虑跨区域延迟对用户体验的影响**  
   虽然 TTFT 得到改善，但在广域网环境下，decode 阶段的响应延迟可能受地理位置影响。

4. **实验规模较小（最多 96 GPU）**  
   大规模 IDC 场景下的稳定性、故障恢复、多租户隔离等问题尚未充分验证。

---

### 🔮 未来工作方向

1. **与 KVCache 压缩技术结合**  
   如集成 **KIVI（2-bit quantization）**、**H2O（heavy-hitter oracle）**、**CacheGen（压缩流式传输）** 等算法，进一步降低传输体积。

2. **支持多级 PrfaaS 层级架构**  
   构建 regional → global 的 prefill 服务层级，按请求热度与长度分级调度。

3. **自动化弹性扩缩容机制**  
   根据实时流量预测自动启停 PrfaaS 实例，提升能效比与成本效益。

4. **面向碳效率与绿色计算的调度**  
   利用跨数据中心灵活性，将 prefill 任务调度至电价更低或清洁能源更丰富的地区（类似 FREESH [9] 的理念）。

---

## ✅ 总结

本论文提出了 **Prefill-as-a-Service (PrfaaS)** 架构，首次系统性地论证了 **下一代 LLM 模型的 KVCache 可以跨越数据中心边界进行传输**。它不是简单依赖模型进步，而是通过 **模型-系统协同设计**，实现了：
- **跨数据中心部署**
- **异构硬件独立扩展**
- **低带宽消耗下的高性能推理**

实验证明，在合理调度下，PrfaaS 可带来 **54% 吞吐提升** 且仅占用 **13% 的 100G Ethernet 带宽**，为未来大规模、低成本、高弹性的 LLM 服务体系提供了可行路径。

</details>

---

### 4. [Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery](https://arxiv.org/abs/2604.14375)

**Authors**: Noureddine Kermiche  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.14375v1  

#### Abstract
Catastrophic forgetting remains a primary hurdle in sequential task learning for artificial neural networks. We propose a silicon-native modular architecture that achieves structural parameter isolation using Task-Specific Experts and a distributed, outlier-based Gatekeeper. Moving beyond traditiona...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题：** *Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文致力于解决**持续学习**（Continual Learning, CL）中的三大核心挑战：
- **灾难性遗忘**（Catastrophic Forgetting, CF）：模型在学习新任务时遗忘旧知识。
- **隐私合规性**（Zero-Leakage）：现代工业场景（如医疗、金融）受GDPR等法规约束，禁止长期存储原始用户数据。
- **任务边界自主识别**：传统方法依赖人工标注的任务切换信号，缺乏对未知领域输入的自动判别能力。

此外，作者指出当前主流的“单体AI”（monolithic AI）范式存在计算不可扩展、容量饱和和法律风险等问题，提出向模块化架构转型的必要性。

---

### 🚀 提出的新方法与核心思想

#### （1）**Simultaneous Pipeline（同步流水线）**
- 在**原始数据流活跃期间**，并行执行三项操作：
  - **Teacher Learning**：高容量教师网络快速拟合当前任务。
  - **Student Distillation**：将知识蒸馏至紧凑的学生专家（Student Expert）。
  - **Router Manifold Acquisition**：训练基于重构误差的路由模块以捕获任务流形特征。
- 数据一旦完成处理即被永久删除，实现 **Zero-Leakage Compliance**。

#### （2）**Tight-Bottleneck Autoencoder (TB-AE)**
- 针对标准VAE在高维稀疏LLM嵌入空间中易发生**后验坍缩**（posterior collapse）的问题，提出确定性的极窄瓶颈自编码器。
- 通过结构压缩强制网络提取语义拓扑签名，在4096-D LLM embeddings中有效区分密集流形。

#### （3）**Decentralized Reconstruction Routers**
- 每个任务配备独立的、仅在其自身数据上训练的**异常检测型路由器**（Outlier Detector），无需全局分类器。
- 路由机制完全分布式且局部化，从根本上避免了共享门控机制导致的灾难性遗忘。

#### （4）**Autonomous Task Discovery**
- 利用路由模块的重构误差作为**无监督新颖性信号**，系统可自主判断是否需要创建新专家模块。
- 引入**Minimum Viable Manifold (MVM)** 和动态阈值校准防止冗余模块爆炸。

#### （5）**Semi-Frozen Backbone 设计**
- 冻结底层预训练骨干网络（F）以生成稳定的中间表示 $ h = F(x) $，确保路由信号不变。
- 上层作为可塑的Persistent Teacher（G）用于快速适应新任务，兼顾前向迁移与稳定性。

---

### 🔍 相比现有方法的优势

| 方法类别 | 局限性 | 本文改进 |
|--------|-------|---------|
| **正则化方法**（EWC/SI） | 容量饱和，权重冲突累积 | 物理隔离参数，0% 后向干扰 |
| **经验回放**（ER） | 违反GDPR，需存储历史数据 | Simultaneous Pipeline + 即时清除，满足Zero-Leakage |
| **生成式回放**（Generative Replay） | 模式崩溃，噪声累积 | 放弃“做梦”，直接实时蒸馏 |
| **LwF类蒸馏** | 共享表征漂移，软标签失真 | Live Distillation on true manifold + 学生冻结 |
| **MoE/Mixture Models** | 全局路由器易遗忘 | 分布式、去中心化的重建路由 |

> ✅ **核心优势总结**：  
> - 实现 **零泄漏**（Zero-Leakage）合规；
> - 达到接近完美的**知识保留**（>99%）；
> - 支持**盲推理**（Domain-Incremental Learning），无需任务ID；
> - 架构天然支持**终身扩展**，无容量上限。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 类型 | 数据集 | 描述 |
|------|--------|------|
| **计算机视觉** | Split-MNIST | 将MNIST分为两个任务：0–4 vs 5–9 |
| **自然语言处理** | Synthetic 4096-D Crowded Manifold | 模拟LLaMA-3级别的高维嵌入空间，构造语义相近但分布不同的任务流形（如Amazon Reviews vs Yelp Reviews），中心间距仅±0.15单位，内在维度d=12 |

> 注：合成数据用于精确控制流形结构，验证TB-AE在极端拥挤空间下的表现。

---

### ⚙️ 实验设置与评估指标

#### 主要评估目标：
- **Task Retention**：旧任务准确率保持情况（衡量稳定性）
- **Forward Transfer Speedup**：新任务训练加速程度（衡量可塑性）
- **Routing Accuracy**：能否正确识别已知/未知任务
- **Autonomous Retrieval SNR**：返回任务时的重构误差信噪比
- **Discrimination Ratio**：已知 vs 未知任务的MSE比值

#### 模块实现细节：
- **Student Expert**：采用LoRA适配器形式，参数高效。
- **Router**：
  - 视觉任务使用VAE（ELBO损失）
  - NLP任务使用TB-AE（MSE重构损失，瓶颈k=12）
- **Inference Routing**：Contrastive Soft Routing加权融合多专家输出。

#### 对比基线方法：
| 基线方法 | 类型 |
|--------|-----|
| Naive Sequential | 顺序训练，无防御 |
| EWC [2] | 正则化方法 |
| LwF [4] | 蒸馏方法 |
| Experience Replay [3] | 回放方法（非zero-leakage） |
| Ours (Simultaneous) | 本文方法 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### 表1：Split-MNIST 上的任务保留能力（Task A Retention after Task B）

| 方法 | Task A Retention | CF风险 | 泄露风险 |
|------|------------------|--------|----------|
| Naive Sequential | 19.40% | 严重 | 无缓冲 |
| LwF | 79.80% | 中等 | 无缓冲 |
| EWC | 84.00% | 中等 | 权重惩罚 |
| Experience Replay | 95.10% | 低 | **违反Zero-Leakage** |
| **Ours (Simultaneous)** | **99.42%** | **零干扰** | **即时清除，合规** |

> ✅ **结论**：本文方法不仅超越所有基线，还实现了接近完美的保留率，并满足隐私要求。

---

#### 表2：TB-AE在4096-D LLM嵌入空间中的消融研究（不同瓶颈尺寸）

| Bottleneck (k) | MSE Task A | MSE Task B | Discrimination Ratio |
|----------------|------------|------------|------------------------|
| k=4 (Narrow) | 0.0674 | 0.2476 | 3.67x（欠拟合） |
| **k=12 (Tight)** | **0.0010** | **0.2109** | **203.78x ✅ 最优分离** |
| k=32 (Relaxed) | 0.0012 | 0.2104 | 174.23x |
| k=64 (Wide) | 0.0011 | 0.2104 | 167.47x |

> ✅ **关键发现**：当瓶颈宽度匹配任务内在维度（k≈12）时，TB-AE达到最佳判别能力（超200倍差异），证明**结构性压缩优于概率正则化**。

---

#### 表3：自主任务检索信噪比（Autonomous Retrieval SNR）

| 输入流形 | 路由器 | 重构MSE | 自主动作 |
|----------|--------|---------|-----------|
| 返回Task A | Task A Router | **0.0014** | RECOGNIZED → 路由至Expert A |
| 新Task B | Task A Router | **0.2105** | NOVELTY → 创建Expert B |

> ✅ **信噪比达145.34x**，系统能以高置信度判断任务归属，避免重复实例化。

---

#### 其他重要结果：
- **Live Distillation产生负保真度差距**（-0.31%）：学生精度略高于教师，说明蒸馏本身起到了正则化作用。
- **前向迁移提速16.2%**：得益于Persistent Teacher的warm-start机制。
- **端到端盲流准确率约95.54%**：结合99.42%专家保留 + 96.10%路由准确率。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **Simultaneous Pipeline 是实现 zero-leakage continual learning 的可行路径**：
   - 知识在数据存在期间完成“即时固化”，无需事后回放或生成。

2. **Tight-Bottleneck Autoencoder 可破解高维LLM嵌入空间中的流形拥挤问题**：
   - 确定性结构瓶颈比B-VAE等概率方法更鲁棒，适用于现代大模型场景。

3. **去中心化路由机制从根本上规避了全局门控遗忘问题**：
   - 每个router是独立的异常检测器，彼此无耦合。

4. **Live Distillation 是一种强效正则化手段**：
   - 学生模型在压缩过程中反而提升泛化能力，出现“负保真度差距”。

5. **semi-frozen backbone 是稳定持续学习的基础架构前提**：
   - 类似生物大脑的进化硬编码感知通路，为高层认知提供不变基础。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **不适用于Class-Incremental Learning** | 当前框架基于协变量偏移（covariate shift）检测任务边界，难以处理同一输入域内的细粒度类别增长。 |
| **并非绝对隐私安全** | 虽然消除replay buffer，但蒸馏后的weights仍可能遭受Model Inversion攻击，需结合DP-SGD等防御机制。 |
| **前向迁移受限于upper Teacher plasticity** | 下层冻结意味着跨任务的底层特征复用有限，true forward transfer主要发生在上层。 |
| **Block-Sequential 流假设** | 若任务高度交错（interleaved），Commitment Gate难以稳定触发。 |
| **O(N) 路由复杂度** | 随着任务数量增加，需遍历所有router进行熟悉度评分，影响推理效率。 |
| **瓶颈维度k需调参** | k=12为特定任务设定，实际部署中应动态调整以适应不同语义密度。 |

---

### 🔮 未来工作方向

1. **Hierarchical Routing**：
   - 引入K-Means聚类或taxonomy树结构，减少O(N)搜索开销，降低误匹配概率。

2. **Decoupled Caching Mechanism**：
   - 解决高度交错数据流下的模块初始化中断问题，允许异步收敛。

3. **Dynamic Bottleneck Tuning**：
   - 开发自动调节TB-AE瓶颈大小的元控制器，适应不同任务复杂度。

4. **Integration with Differential Privacy**：
   - 结合DP-SGD或Private Aggregation，进一步增强模型抗逆向攻击能力。

5. **Extension to Class-Incremental Setting**：
   - 探索结合prototype learning或contrastive routing来支持细粒度增量分类。

---

## 总结

> 本论文提出了一种面向**企业级AI生态系统**的模块化持续学习架构，成功解决了**灾难性遗忘、隐私泄露、任务自主发现**三大难题。其核心创新——**Simultaneous Pipeline + TB-AE + Decentralized Routers**——不仅在性能上超越现有方法，更在工程落地层面提供了符合GDPR要求的可行方案。实验充分验证了其在CV与NLP领域的有效性，尤其在高维LLM嵌入空间中展现出前所未有的判别能力和稳定性，标志着向“后单体AI时代”迈进的关键一步。

</details>

---

### 5. [SeaAlert: Critical Information Extraction From Maritime Distress Communications with Large Language Models](https://arxiv.org/abs/2604.14163)

**Authors**: Tomer Atia, Yehudit Aperstein, Alexander Apartsin  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.14163v1  

#### Abstract
Maritime distress communications transmitted over very high frequency (VHF) radio are safety-critical voice messages used to report emergencies at sea. Under the Global Maritime Distress and Safety System (GMDSS), such messages follow standardized procedures and are expected to convey essential deta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SeaAlert: Critical Information Extraction From Maritime Distress Communications with Large Language Models》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于**海上遇险通信（maritime distress communications）中的关键信息提取与严重性分类**。这类通信通过VHF无线电传输，属于安全关键型语音消息，需在紧急情况下快速准确地传达船只身份、位置、遇险性质和所需援助等信息。

然而，现实中的VHF通信面临诸多挑战：
- 通话简短、噪声大、说话者处于压力状态；
- 自动语音识别（ASR）系统因信道干扰和情绪影响产生大量转录错误；
- 消息可能偏离标准格式，省略或替换关键GMDSS codewords（如MAYDAY、PAN-PAN）；
- 缺乏大规模标注的真实数据用于训练和评估。

因此，如何从**嘈杂且非标准化的ASR转录文本中鲁棒地进行信息提取与分类**，是当前研究的空白。

---

### 提出了什么新方法或新思路
作者提出了 **SeaAlert** ——一个基于大语言模型（LLM）的端到端评估框架，包含以下核心创新：

1. **合成数据生成管道（Synthetic Data Generation Pipeline）**
   - 使用 GPT-4 生成多样化的海上无线电消息，涵盖四种严重等级（Distress, Urgency, Safety, Routine）、三种通信风格（正式、非正式、第三方转发）及12种事故场景。
   - 引入“掩码版本”：将标准GMDSS codewords（如MAYDAY）替换为通用标记 `[SIGNAL]`，以测试模型对显式关键词的依赖程度。

2. **逼真的音频模拟与噪声注入流程**
   - 将合成文本使用 Coqui TTS 转换为语音；
   - 注入真实VHF信道噪声（SNR分别为12dB和6dB）；
   - 使用 Whisper ASR 进行转录，模拟实际ASR错误（WER达29.6%~36.2%），获得“现实世界风格”的噪声文本。

3. **联合任务评估框架**
   - 同时评估两个任务：
     - **严重性分类（Severity Classification）**：判断消息属于哪一类紧急情况；
     - **结构化信息提取（Structured Information Extraction）**：抽取Vessel Name、MMSI、Position、POB等关键字段。

4. **引入对抗性陷阱测试集（Adversarial Trap Messages）**
   - 手工构造15条具有迷惑性的消息，涵盖否定句、演习指令（drill）、中继通信（relay）、模糊表达等，检验模型是否仅依赖关键词匹配。

---

### 相比现有方法的优势
| 方面 | 传统方法局限 | SeaAlert改进 |
|------|--------------|-------------|
| 数据来源 | 依赖稀缺的真实数据 | 构建可控、平衡、多样化的合成数据集 |
| 鲁棒性评估 | 忽视ASR噪声和协议偏差 | 显式模拟ASR退化与关键词缺失 |
| 模型比较 | 多采用BoW或规则方法 | 对比BoW与Transformer架构在多种条件下的表现 |
| 信息提取 | 依赖Regex规则 | 探索GPT-4零样本提取能力 |

> ✅ **总体优势**：首次在一个统一、可复现的框架下系统评估NLP模型在**噪声、掩码、对抗**条件下的鲁棒性，填补了安全关键领域中低资源、高噪声文本理解的研究空白。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **自建合成数据集**：共1,872条消息，按70%/15%/15%划分为训练/验证/测试集。
  - 四类严重等级均衡分布；
  - 包含原始文本与掩码文本（codeword → `[SIGNAL]`）；
  - 每条附带结构化元数据（Vessel, MMSI, Position等），作为信息提取真值。

### 实验设置
#### （1）严重性分类任务
- **输入**：clean text / ASR Medium / ASR High 转录文本
- **输出**：四分类标签（Distress, Urgency, Safety, Routine）
- **评估指标**：
  - Accuracy
  - Macro-F1（主指标）
  - 类别级 Precision, Recall, F1（特别关注Distress召回率）

#### （2）结构化信息提取任务
- **抽取字段**：Vessel Name, Call Sign, MMSI, Position, POB, Nature of Incident
- **评估方式**：field-level accuracy
- **对比方法**：
  - Regex-based extraction（基于正则表达式的规则抽取）
  - Zero-shot GPT-4-based extraction（无需微调，直接提示GPT-4解析）

#### （3）对比模型
| 模型类别 | 具体模型 | 说明 |
|--------|---------|------|
| Bag-of-Words Baseline | Logistic Regression + TF-IDF | 经典文本分类基线 |
| Transformer Models | DistilBERT, RoBERTa-base | 选择RoBERTa为最终代表 |

#### （4）四项核心实验设计
1. **Baseline & ASR Robustness Test**：干净文本 vs. 噪声ASR文本性能下降分析  
2. **Adversarial Scenario Test**：在15条人工构造的陷阱消息上测试抗欺骗能力  
3. **Keyword Ablation Test**：比较原始文本 vs. codeword掩码文本的表现差异  
4. **Structured Information Extraction Comparison**：Regex vs. GPT-4 在噪声条件下的提取效果对比  

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）严重性分类性能（Macro-F1）

| 条件 | Logistic Regression | RoBERTa |
|------|---------------------|--------|
| Clean Text | 0.674 | **0.679** |
| ASR Medium | 0.521 | **0.645** |
| **ASR High** | **0.423** | **0.608** |

> 🔍 **关键发现**：在高噪声条件下，RoBERTa的Macro-F1仅下降约10%，而Logistic Regression下降高达37%。

#### （2）性能退化对比（Clean → ASR High）

| 模型 | Macro-F1 下降绝对值 | 相对降幅 |
|------|--------------------|----------|
| Logistic Regression | -0.251 | ≈37% |
| RoBERTa | -0.071 | ≈10% |

> ✅ RoBERTa展现出显著更优的**噪声鲁棒性**。

#### （3）关键词掩码实验（Codeword Masking）

| 设置 | 描述 | RoBERTa 表现 | LR 表现 |
|------|------|------------|--------|
| A（原词训练+测试） | 正常使用codeword | 高性能 | 高性能 |
| B（掩码训练+测试） | 完全不见codeword | 性能下降但稳定 | 明显下降 |
| C（原词训练 + 掩码测试） | 训练见词，测试不见 | **仍保持稳定** | **性能进一步恶化，出现负gap** |

> 📌 结论：RoBERTa能更好地泛化到未见过的无关键词场景；而LR严重依赖训练时看到的lexical pattern。

#### （4）对抗性陷阱测试（15条手写消息）

| Trap Type | Logistic Regression | RoBERTa |
|----------|----------------------|--------|
| Negation | 0.25 | 0.25 |
| Drill | 0.00 | 0.00 |
| Relay | 0.50 | 0.50 |
| Ambiguous | 0.333 | **0.667** |
| Real_no_codeword | 0.333 | **0.667** |
| **Overall Accuracy** | **0.267** | **0.400** |

> ⚠️ 两者均无法处理“Drill”类消息（全部误判为真实事件），存在重大安全隐患。

#### （5）结构化信息提取性能（ASR High 条件下）

| 字段 | Regex 准确率 | GPT-4 准确率 | 提升幅度 |
|------|-------------|--------------|---------|
| Vessel | 中等 | 显著更高 | ↑↑ |
| Call Sign | 极低（字母数字易错） | 明显恢复 | ↑↑↑ |
| MMSI | 极低 | **接近完美恢复** | ↑↑↑↑ |
| Position | 中等偏低 | 更好语义理解 | ↑↑ |
| POB | 较稳定 | 略优 | ↑ |
| Nature | 中等 | 更自然表达还原 | ↑↑ |

> ✅ **GPT-4在所有字段上全面超越Regex**，尤其在MMSI、Call Sign等格式敏感字段上优势巨大。

---

## 4. 关键结论和发现

### 主要发现
1. **Transformer模型（RoBERTa）在噪声环境下显著优于BoW方法**
   - 尽管在clean text上性能相近，但在ASR High噪声下，RoBERTa的性能下降远小于Logistic Regression（仅-10% vs -37%）。
   - 得益于其上下文建模能力，能够容忍拼写错误、词语缺失和语法变形。

2. **两种模型都高度依赖GMDSS codewords**
   - 当消息包含MAYDAY/PAN-PAN等词时，分类准确率明显提升；
   - 但当这些词被移除或损坏时，BoW模型性能急剧下滑，甚至不如从未见过这些词的情况（负gap现象）；
   - RoBERTa则表现出更强的**上下文推理能力**，能在缺少关键词时利用其他线索做出判断。

3. **GPT-4在结构化信息提取中完胜Regex**
   - Regex严重依赖固定格式和精确字符串匹配，在ASR噪声下几乎失效；
   - GPT-4凭借语义理解和纠错能力，即使面对严重转录错误也能恢复关键信息（如MMSI、Position）；
   - 特别适用于**alphanumeric identifiers** 和 **semantically variable free-text fields**。

4. **当前系统仍难以应对对抗性/程序性消息**
   - 无论是BoW还是RoBERTa，在“Drill”（演习）类消息上**准确率为0**；
   - 表明模型容易将带有紧急词汇但非真实事件的消息误判为真实危机；
   - 存在引发**虚假警报**的重大风险。

---

### 方法的局限性
1. **数据为合成生成，未必完全反映真实语言多样性**
   - 尽管使用GPT-4生成，但仍受限于prompt设计和分布控制；
   - 缺少真实船员在极端压力下的语言行为建模。

2. **ASR噪声为模拟环境，未覆盖所有现实复杂性**
   - 如多人重叠讲话、方言口音、突发强干扰等情况尚未考虑。

3. **对抗测试集规模小（仅15条），代表性有限**
   - 虽具诊断价值，但不足以支撑全面鲁棒性评估。

4. **未探索端到端音频输入模型**
   - 当前框架为“ASR → 文本 → 分析”，存在误差累积问题；
   - 未来可尝试audio-to-semantics的联合建模。

---

### 未来工作方向
1. **在真实VHF录音数据上验证模型性能**
   - 收集并标注真实的海上遇险通信记录，提升外部有效性。

2. **构建更大规模的对抗性基准测试集**
   - 包括更多类型的陷阱消息（negation, speculation, indirect requests等）。

3. **开发减少对codeword依赖的训练策略**
   - 如对抗训练、数据增强、知识蒸馏等方式提升模型语义理解能力。

4. **探索audio-based end-to-end模型**
   - 跳过ASR中间环节，直接从音频波形中提取语义信息，避免错误传播。

5. **集成人类监督机制**
   - 设计人机协同决策流程，特别是在边界案例或疑似演习消息中引入人工审核。

---

## 总结

> 🎯 **SeaAlert的核心贡献在于建立了一个面向安全关键系统的、系统化且可复现的评估范式**，揭示了：
>
> - **RoBERTa在噪声和关键词缺失条件下显著优于传统BoW模型**；
> - **GPT-4在结构化信息提取任务中全面超越Regex规则方法**；
> - **当前AI系统仍无法可靠区分演习与真实遇险，必须保留人工监督**。

该研究为未来海上搜救系统的智能化升级提供了重要技术参考和实践指导，推动从“关键词匹配”向“语义理解+鲁棒推理”的转变。

</details>

---

### 6. [From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning](https://arxiv.org/abs/2604.15244)

**Authors**: Kiran Purohit, Ramasuri Narayanam, Soumyabrata Pal  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.15244v1  

#### Abstract
Speculative decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose outputs that a stronger target model verifies. However, its token-centric nature allows erroneous steps to propagate. Prior approaches mitigate this using external reward models, but...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Speculative Decoding (SD)** 虽然通过轻量级 draft model 加速大模型推理，但其**token-centric**（以词元为中心）的设计在多步推理任务中存在严重缺陷：
- 错误的中间步骤一旦被接受，会**持续传播（error propagation）**，导致最终答案错误。
- 严格依赖目标模型的概率分布，常拒绝语义正确但概率较低的 draft token，造成计算浪费。
- 现有改进方法如 **Reward-guided Speculative Decoding (RSD)** 引入外部 **Process Reward Model (PRM)** 进行验证，但带来额外延迟、计算开销，并且泛化能力差。

### 🚀 提出的新方法：SPECGUARD
提出一种**验证感知的推测解码框架 SPECGUARD**，实现高效且可靠的多步推理。其核心思想是：
> 将验证从“token级”提升到“step级”，并完全基于模型内部信号进行轻量级验证，避免依赖外部模型。

#### 主要创新点：
1. **Step-Level Verification（步骤级验证）**
   - 不再逐个 token 验证，而是将每个完整的 reasoning step 作为一个单元进行验证，更符合推理任务的本质。

2. **双信号集成验证器（Ensemble Verifier）**
   - 结合两个**模型内部（model-internal）** 的轻量级信号：
     - **Attention-Based Grounding Verification (ABGV)**：通过 attention rollout 分析生成步骤是否充分 grounding 到输入或已验证的前序步骤。
     - **Log-Probability-Based Verification (LPBV)**：衡量生成 token 的置信度（log probability），确保高可信输出。
   - 两者联合决策，只有同时满足“高置信”和“强归因”的步骤才被接受。

3. **自一致性选择器（Self-Consistency Selector）**
   - 在每一步对 draft model 采样多个候选步骤，选择**语义上最一致**的那个作为待验证对象，提升鲁棒性。

4. **无需外部奖励模型**
   - 完全摒弃 PRM，仅用模型自身注意力和概率信号，显著降低延迟和部署复杂度。

### 🔍 相比现有方法的优势
| 特性 | SD | RSD | SPECGUARD |
|------|----|-----|-----------|
| 是否依赖外部 PRM | ❌ | ✅ | ❌ |
| 支持 step-level 验证 | ❌ | ✅ | ✅ |
| 验证信号来源 | 目标模型重打分 | 外部 PRM | 模型内部信号（ABGV + LPBV） |
| 推理效率 | 高 | 中（PRM 开销大） | 高 |
| 准确率稳定性 | 易受错误传播影响 | 受 PRM 质量限制 | 更可靠（双重保障） |

---

## 2. 核心实验方法和设置

### 📚 数据集
在四个具有挑战性的多步推理基准上评估：
- **MATH500**：500道竞赛级别数学题，涵盖代数、几何、组合等。
- **GSM8K**：8.5K小学数学应用题，测试多步数值推理。
- **Gaokao-2023-En**：中国高考英语翻译版题目，强调逻辑与函数推理。
- **OlympiadBench**：国际奥赛级别的科学与数学难题，极具创造性要求。

### 📊 评估指标
- **Exact Match (EM)**：预测答案与标准答案完全匹配的比例。

### ⚙️ 实验设置
- **模型配置**：
  - 主要使用 `Qwen2.5-Math-Instruct`（7B target + 1.5B draft）
  - 对比也测试 `Llama-3` 系列（8B target + 1B draft）
- **推理设置**：
  - 每个 reasoning step 由 `\n\n` 分隔。
  - 温度 `temperature=0.7`，`top_p=0.8`，每步采样 `n=16` 个候选。
  - 使用 **vLLM** 作为推理后端，在 **NVIDIA A100 GPU** 上运行。
- **超参数**：
  - 验证阈值 `T = 0.7`
  - 权重系数 `β = 0.3`（平衡 ABGV 和 LPBV）

### 🆚 基线方法对比
| 类别 | 方法 |
|------|------|
| 单模型 | Target-only, Draft-only |
| 推理时计算增强 | Best-of-N, Majority Voting |
| 推测解码 | SD, RSD, RSD Majority |
| 搜索类方法 | Beam Search, Process Best-of-N |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 1 & 2）

| 方法 | MATH500 ↑ | GSM8K ↑ | Gaokao ↑ | Olympiad ↑ | Latency ↓ |
|------|----------|--------|--------|----------|---------|
| Target-only | 83.0 | 94.7 | 66.8 | 40.6 | 高 |
| SD | 82.4 | 94.2 | 66.3 | 39.4 | 较低 |
| RSD | 82.4 | 94.4 | 68.5 | 39.6 | 中等 |
| **SPECGUARD** | **85.4** | **95.8** | **69.4** | **41.2** | **~11% 低于 RSD** |

> ✅ **SPECGUARD 在所有基准上均优于 state-of-the-art 方法**：
> - 平均准确率提升 **+3.6%**（相比 RSD）
> - 推理延迟降低 **约 11%**

### 🔁 与搜索类方法对比（Table 2）
| 方法 | MATH500 | GSM8K |
|------|--------|-------|
| Beam Search (bs=8) | 78.2 | 88.4 |
| Process Best-of-N (N=16) | 76.0 | 87.9 |
| **SPECGUARD** | **85.4** | **95.8** |

> 💡 表明：在复杂推理空间中，**基于大模型生成 + 轻量反馈机制** 比纯搜索策略更有效。

### 🔍 消融实验结果（Ablation Studies）

#### （1）组件消融（SC + LPBV vs SPECGUARD）
- **SC + LPBV**（无 ABGV）：83.2 (MATH500)，说明仅靠置信度无法过滤“看似合理但未归因”的错误步骤。
- **完整 SPECGUARD**：85.4 → 证明 **ABGV 对防止 ungrounded steps 至关重要**。

#### （2）层数选择（Figure 3 & 4）
- 使用最后 **3 层 attention** 即可达到接近使用全部层的效果，且显著降低内存和延迟。
- 使用前3层效果最差，表明深层注意力更适合 grounding 判断。

#### （3）注意力稀疏化（Sparsification）
- 过滤掉 <0.01 的 attention 权重，反而提升准确率和速度。
- 原因：聚焦于关键注意力路径，减少噪声干扰。

#### （4）样本数量影响（Figure 2）
- 随着每步采样数增加，SPECGUARD 准确率稳步上升并在 `k=16` 左右饱和。
- RSD Majority 在样本增多时出现性能下降，可能因 PRM 噪声累积。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Step-level 验证优于 Token-level**：将验证单位从 token 提升为 reasoning step 更契合推理任务结构。
2. **模型内部信号足以支撑高质量验证**：ABGV 和 LPBV 的结合能有效识别错误推理链，无需依赖外部 PRM。
3. **SPECGUARD 实现精度与效率双赢**：
   - 准确率高于 target model 和 RSD；
   - 延迟低于 RSD，接近甚至优于普通 SD。
4. **自一致性 + 双信号验证 是关键**：多候选采样 + 自一致性选择 + 双重验证机制共同提升了鲁棒性和可靠性。

### ⚠️ 方法的局限性
1. **评估集中在结构化推理任务**：未在开放生成、长文本对话、创意写作等场景中验证 step-level 验证的有效性。
2. **单实例推理优化**：未考虑大规模 batch 推理或硬件级优化（如 TensorRT 部署），实际生产系统需进一步工程适配。
3. **仍可能产生幻觉**：虽然降低了错误传播风险，但不能完全消除 LLM 固有的 hallucination 问题，仍需人工监督。

### 🔮 未来工作方向
1. **引入更多内部信号**：如熵（entropy）、不确定性校准（uncertainty calibration）等，进一步增强验证器可靠性。
2. **扩展至非推理任务**：探索在摘要、对话、代码生成等任务中应用 step-level 验证的可能性。
3. **动态调整采样数**：根据问题难度自适应调节每步候选数 `k`，进一步优化性价比。
4. **结合 test-time scaling**：与 Best-of-N、Tree of Thoughts 等方法融合，构建更强的推理系统。

---

> **总结一句话**：  
> **SPECGUARD 通过“自一致性选择 + 注意力归因 + 概率置信”的三重机制，在不依赖外部奖励模型的前提下，实现了更准确、更高效的多步推理，为低成本高可靠 LLM 推理提供了新范式。**

</details>

---

### 7. [Generative Augmented Inference](https://arxiv.org/abs/2604.14575)

**Authors**: Cheng Lu, Mengxin Wang, Dennis J. Zhang, Heng Zhang  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.14575v1  

#### Abstract
Data-driven operations management often relies on parameters estimated from costly human-generated labels. Recent advances in large language models (LLMs) and other AI systems offer inexpensive auxiliary data, but introduce a new challenge: AI outputs are not direct observations of the target outcom...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Generative Augmented Inference 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **data-driven operations management** 中一个核心挑战：高质量的人类标注数据（如购买决策、专家注释、调查响应）成本高昂且难以大规模获取，而当前基于 **Large Language Models (LLMs)** 和其他 **AI systems** 生成的辅助数据虽然廉价，但存在以下问题：
1.  **非直接观测**：AI 输出并非目标结果的真实标签，而是可能带有复杂未知关系的高维表示。
2.  **传统方法失效**：现有方法（如 **Prediction-Powered Inference, PPI**）将 AI 预测作为真实标签的直接代理（surrogate），当这种代理关系弱或被错误指定时，会导致估计效率低下或不可靠。

### 提出的新方法和新思路
作者提出了 **Generative Augmented Inference (GAI)** 框架，其核心创新在于一次**概念上的根本转变**：
- **从“代理”到“特征”**：不再将 AI 输出视为人类标签的替代品，而是将其视为**有助于预测人类标签的丰富信息特征**。
- **理论基础**：利用 **Neyman orthogonality** 构造正交矩（orthogonal moment），构建了一个偏差校正的估计量。这使得 GAI 能够在不依赖 AI 输出是准确代理的前提下，稳健地整合辅助信息。

### 相比现有方法的优势
1.  **普适性强**：能够处理各种形式的 AI 输出，包括离散预测、连续分数、高维嵌入（embeddings）甚至非结构化文本。
2.  **“安全默认”属性 (Safe Default Property)**：在随机标注条件下，GAI 相对于仅使用人类数据的估计器（primary-only estimator）**永远不会更差**（弱占优），并且只要 AI 输出具有任何预测性，就能实现严格增益。
3.  **效率提升来源明确**：通过方差分解，明确了 GAI 效率提升的三个独立来源：样本扩展、AI 的表征能力、以及 AI 提供的额外预测信息。
4.  **稳健性高**：对 AI 输出中的系统性偏差和不准确性具有鲁棒性。

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在三个不同的实际商业应用中进行了验证，覆盖了三种截然不同的辅助数据场景：
1.  **疫苗联合分析 (Vaccine Conjoint Analysis)**：使用 Kreps et al. (2020) 的疫苗选择调查数据，共 1,971 名受访者。AI 辅助数据来自 LLM 的链式思维（Chain-of-Thought, CoT）推理，包括：
    -   **离散选择标签 (Labels)**：LLM 对疫苗偏好的预测。
    -   **高维文本嵌入 (Embeddings)**：使用 `text-embedding-3-large` 模型将 LLM 的完整推理文本转换为 3072 维向量。
2.  **零售定价 (Retail Pricing)**：使用 Toubia et al. (2025) 的“数字孪生”（digital twins）数据集，包含 2,058 名参与者对产品价格的购买决策。AI 辅助数据是基于参与者个人资料生成的二元购买预测。
3.  **健康保险选择 (Health Insurance Choice)**：使用美国人口普查数据，研究收入与私人医疗保险覆盖率的关系。AI 辅助数据由 Angelopoulos et al. (2023a) 使用梯度提升分类器生成的预测概率。

### 实验设置和评估指标
- **实验设计**：采用重复抽样实验，固定主样本大小 `np` 和辅助样本大小 `nA`，进行多次独立试验（通常为 50 次），报告平均结果。
- **评估指标**：
    1.  **点估计精度**：使用 **MAPE (Mean Absolute Percentage Error)** 衡量参数估计的平均百分比误差。
    2.  **推断质量**：
        -   **置信区间覆盖率 (CI Coverage)**：95% 置信区间包含真实参数的比例，理想值为 95%。
        -   **置信区间宽度 (CI Width)**：衡量推断的精确度。
        -   **决策错误率 (Decision Errors)**：因置信区间导致错误管理决策的比例。

### 基线方法对比
与四种主流方法进行比较：
1.  **Primary**：仅使用人类标注数据的最大似然估计。
2.  **Naive**：简单地将人类和 AI 标签混合后进行最大似然估计。
3.  **PPI**：原始的 Prediction-Powered Inference 方法。
4.  **PPI++**：PPI 的改进版本，通过调整收缩参数来优化区间宽度。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
GAI 在所有三个应用中均表现出显著且一致的优势：

| 应用场景 | 关键性能提升 | 与最佳基线对比 |
| :--- | :--- | :--- |
| **疫苗联合分析** | - MAPE 降低约 **50%** <br> - 所需人类标签减少 **>75%** <br> - 决策错误率从 6.9% 降至 **2.6%** | GAI (50 标签) 的表现优于 Primary (200 标签) |
| **零售定价** | - MAPE 从 10-22% 降至 **7-12%** <br> - 所需人类标签减少 **67%** <br> - 决策错误率接近 **0%** | GAI (100 标签) 的精度匹配 Primary (300 标签) |
| **健康保险选择** | - MAPE 从 290-980% 降至 **140-160%** <br> - 所需人类标签减少 **>90%** <br> - 决策错误率为 **0%** | GAI (100 标签) 的表现远超 Primary (1000 标签) |

**总体趋势**：
- **点估计**：GAI 在所有设置下都实现了最低的 MAPE。
- **推断质量**：GAI 的置信区间覆盖率始终达到或超过 95%，而 PPI++ 等方法经常出现**欠覆盖 (undercover)**。同时，GAI 的置信区间宽度与最优基线相当，没有为了提高覆盖率而牺牲精度。
- **消融实验**：在疫苗联合分析中，比较了 GAI 使用 **Labels** 和 **Embeddings** 两种输入。结果显示，**Embeddings** 在降低 MAPE 和提高覆盖率方面表现更好，而 **Labels** 能产生更窄的置信区间，证明了不同形式的 AI 特征各有优势，均可被有效利用。

---

## 4. 关键结论和发现

### 主要发现
1.  **范式转变的有效性**：将 AI 输出视为**特征**而非**代理**是一种强大且稳健的范式，能有效解决 AI 辅助数据集成中的核心挑战。
2.  **GAI 的普适性和优越性**：GAI 在从近乎随机的低质量预测到高度校准的高质量预测等各种辅助数据场景下，均能稳定地提供显著的统计效率增益。
3.  **“安全默认”属性的实践价值**：这一特性意味着研究人员可以无风险地尝试使用 GAI，因为它保证不会比只用人类数据更差，从而极大地降低了采用新技术的门槛。
4.  **效率增益的多源性**：GAI 的成功不仅源于利用了 AI 的额外信息，也得益于其强大的表征能力和有效的样本扩展机制。

### 方法的局限性
1.  **随机标注假设**：论文的“安全默认”优势依赖于**随机选择标注**的假设（即标注概率 `e(X,z)` 是常数）。如果标注过程是有偏的（例如，优先标注容易预测的样本），该优势可能不成立。
2.  **计算复杂性**：GAI 涉及两步估计和交叉拟合（cross-fitting），相比简单方法计算开销更大。

### 未来工作方向
1.  **拓展至非随机标注场景**：将 GAI 的优势推广到战略性或有偏的标注设计中。
2.  **开发在线版本**：将 GAI 框架集成到实时决策系统中，以应对计算挑战。
3.  **探索更复杂的模型**：将 GAI 的思想应用于更广泛的机器学习和因果推断模型中。

</details>

---

### 8. [xFODE: An Explainable Fuzzy Additive ODE Framework for System Identification](https://arxiv.org/abs/2604.14883)

**Authors**: Ertugrul Kececi, Tufan Kumbasar  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.14883v1  

#### Abstract
Recent advances in Deep Learning (DL) have strengthened data-driven System Identification (SysID), with Neural and Fuzzy Ordinary Differential Equation (NODE/FODE) models achieving high accuracy in nonlinear dynamic modeling. Yet, system states in these frameworks are often reconstructed without cle...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# xFODE: An Explainable Fuzzy Additive ODE Framework for System Identification 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Neural ODE (NODE)** 和 **Fuzzy ODE (FODE)** 虽然在非线性系统建模中表现出色，但在以下方面存在显著缺陷：
- **状态表示缺乏物理意义**：系统状态通常通过隐变量或滞后输出构建（如 $ x_k = [y_k, y_{k-1}, \dots] $），难以解释其动态含义。
- **黑箱特性强**：无论是 NN 还是 FLS，其对状态导数 $ \dot{x} $ 的映射均为高维、复杂且不可解释的函数，无法提供输入维度对动态演化的具体影响。

这些问题限制了模型在安全关键系统（如工业控制、机器人、电池管理）中的可信部署。

---

### 提出了什么新方法或新思路
本文提出 **xFODE (Explainable FODE)**，一种可解释的数据驱动系统辨识（SysID）框架，结合了深度学习训练能力与模糊系统的透明性。主要创新包括：

#### （1）**增量式状态表示（Incremental State Representation, SR2）**
- 将状态定义为输出及其差分形式：  
  $$
  x_k = [y_k, \Delta y_k, \dots, \Delta^m y_k]
  $$
- 各分量具有明确物理意义（如位置、速度、加速度），增强了状态的可解释性。

#### （2）**Fuzzy Additive Model for Derivatives (FAME)**
- 使用 **Fuzzy Additive Models (FAMs)** 替代传统单个高维 FLS 来逼近状态导数：
  $$
  \dot{x}_k = \sum_{i=1}^{n_z} f_i(z_{i,k}; \theta)
  $$
- 每个 $ f_i(\cdot) $ 是一个单输入多输出的 FLS，独立建模每个输入 $ z_i $ 对所有状态变化的贡献，实现 **input-wise interpretability**。

#### （3）**可解释性增强的划分策略（Partitioning Strategies, PSs）**
设计三种 PS 策略，在训练过程中“雕刻”隶属函数（MFs）空间，确保任意输入仅激活两个相邻规则，从而提升模糊推理的局部简洁性和语义清晰度：
- **PS1**: 使用 Triangular MFs（TriMFs），通过参数耦合强制连续覆盖。
- **PS2**: 使用双侧高斯 MFs（Gauss2MFs），允许不对称形状并控制重叠。
- **PS3**: 构造互补型 Gauss2MFs，进一步增强相邻 MF 间的平滑过渡与重叠。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **可解释性** | 显著优于 NODE 和标准 FODE；状态有物理意义，输入贡献可分解，规则激活稀疏且语义清晰（如 N, Z, P）。 |
| **模型复杂度** | 参数数量（#LP）远低于 NODE，接近 FODE，适合嵌入式部署。 |
| **灵活性与表达力** | 保留 FODE 的非线性拟合能力，同时通过 additive 结构解耦输入影响。 |
| **端到端训练** | 支持基于梯度的 DL 优化（如 Adam），自动学习 MF 参数，无需手动调参。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共测试五个基准 SysID 数据集，涵盖单输入单输出（SISO）与多输入多输出（MIMO）场景：

| Dataset       | $ n_u $ | $ n_y $ | Train Samples | Test Samples | 特点 |
|---------------|----------|----------|----------------|---------------|------|
| Two-Tank      | 1        | 1        | 1500           | 1500          | 非线性液位系统 |
| Hair Dryer    | 1        | 1        | 500            | 500           | 温度响应延迟 |
| MR Damper     | 1        | 1        | 3000           | 499           | 磁流变阻尼器 |
| Steam Engine  | 2        | 2        | 250            | 201           | 多变量动力系统 |
| EV Battery    | 2        | 1        | 15001          | 14351         | 电动汽车电池电压预测 |

所有数据已归一化处理。

---

### 实验设置和评估指标

#### 模型配置
- **NODE**: 两层隐藏层（每层128 unit），tanh 激活。
- **FODE**: 多输入 FLS，P=5 规则，使用 GaussMFs，无 PS。
- **AFODE**: 加法式 FODE，使用 GaussMFs，无 PS。
- **xFODE**: 每个 $ f_i(\cdot) $ 使用 P=5 规则，并应用 PS1–PS3。
- **NLARX 基线**：使用 MATLAB `nlarx` 函数训练，包括 SN（Sigmoid Network）、TE（Tree Ensemble）、SVM、NN 四种结构。

#### 状态表示方式
- 比较两种状态重构方式：
  - **SR1 (Lagged Form)**: $ x_k = [y_k, y_{k-1}, \dots] $
  - **SR2 (Incremental Form)**: $ x_k = [y_k, \Delta y_k, \dots] $
- 差分阶数 $ m $ 通过交叉验证选择（Two-Tank 等设为 2，Steam Engine 和 EV Battery 设为 1）。

#### 评估任务与指标
- **任务类型**：multi-step prediction（仿真模式）
- **Roll-out 步长**：N = 20
- **评价指标**：
  - **RMSE**（均方根误差）：报告平均值 ± 标准误（20次独立运行）
  - **#LP**：可学习参数总数（衡量模型复杂度）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table II）

| Model         | Two-Tank RMSE ↓ | Hair Dryer RMSE ↓ | MR Damper RMSE ↓ | EV Battery RMSE ↓ | Steam Engine RMSE ↓ | #LP ↓ |
|---------------|------------------|--------------------|-------------------|--------------------|------------------------|--------|
| NODE-SR2      | 0.0197±0.0043   | 0.1173±0.0038     | 9.6809±0.3299    | 0.1757±0.0038     | y1:0.0909 / y2:0.0873 | ~17.5K |
| FODE-SR2      | 0.0166±0.0014   | 0.1352±0.0107     | 9.2819±0.8725    | 0.3814±0.2994*    | y1:0.1014 / y2:0.1129 | ~115–200 |
| AFODE-SR2     | 0.0163±0.0008   | 0.1233±0.0021     | 9.5588±0.4112    | 0.1721±0.0258     | y1:0.0762 / y2:0.0713 | ~160–300 |
| **xFODE-SR2-PS1** | **0.0197±0.0015** | **0.1271±0.0032** | **9.6841±0.3818** | **0.1845±0.0222** | **y1:0.0790 / y2:0.0729** | **148–282** |
| **xFODE-SR2-PS3** | **0.0180±0.0012** | 0.1293±0.0043     | 10.0630±0.3803   | 0.1880±0.0394     | y1:0.0862 / y2:0.0768 | 148–282 |

> *注：FODE 在 EV Battery 上出现 NaN，稳定性较差*

---

### 与基线方法的对比结果

- ✅ **精度媲美甚至超越主流方法**：
  - 在 Two-Tank 和 Steam Engine 上，xFODE 表现优于 NODE 和 FODE。
  - 在 Hair Dryer、EV Battery 上虽略逊于 AFODE，但仍优于或持平于 NODE/FODE。
- ✅ **SR2 显著提升性能**：
  - 所有模型采用 SR2 后 RMSE 更低、方差更小（见 Fig. 3），说明增量状态更具动态表征能力。
- ✅ **xFODE 实现“零代价”可解释性**：
  - 性能损失极小（相比 AFODE 仅略高），但换来巨大可解释收益（规则稀疏、语义清晰）。
- ✅ **参数效率高**：
  - #LP 远少于 NODE（约 1%），与 FODE 相当，适合资源受限场景。

---

### 消融实验结果（隐含分析）

- **PS 策略的影响**：
  - 不同 PS 在不同数据集上表现最优（e.g., PS3 在 Two-Tank 最佳，PS1 在 Steam Engine 最佳），表明 **最佳 PS 具有数据依赖性**。
  - PS 强制稀疏规则激活（仅两个连续规则生效），显著提高 antecedent space 的可读性（见 Fig. 2）。
- **AFODE vs xFODE**：
  - AFODE 性能稍优，但由于多个规则同时激活（Fig. 2a），导致模糊系统“纠缠”，降低可解释性。
  - xFODE 以轻微性能下降换取更强的规则语义分离。

---

## 4. 关键结论和发现

### 主要发现
1. **增量状态表示（SR2）优于滞后表示（SR1）**：不仅提升建模精度，也赋予状态明确物理意义。
2. **Additive + Fuzzy + PS = 可解释性飞跃**：
   - FAME 结构实现了输入维度级别的贡献分解；
   - PS 策略有效约束模糊推理过程，使规则激活路径清晰、易于理解。
3. **xFODE 在保持高性能的同时实现高度可解释性**：
   - RMSE 与 NODE/FODE 相当，部分任务领先；
   - 参数量极少，具备工程落地潜力；
   - 模糊规则可用自然语言描述（如 “若输入为负，则状态减小”）。

---

### 方法的局限性
- **规则后件（consequents）仍不透明**：当前研究聚焦于前件空间（antecedent）的可解释性，但规则输出 $ d_p \in \mathbb{R}^n $ 仍是数值向量，缺乏语义标签。
- **PS 设计依赖先验知识**：需预设规则数 $ P $ 和划分方式，自动化程度有待提升。
- **尚未引入不确定性量化**：尽管 FLS 天然适合概率扩展，但本文未涉及置信区间估计。

---

### 未来工作方向
- **提升规则后件的可解释性**：探索将 consequent 输出映射为符号化动作或趋势（如“加速”、“减速”）。
- **自适应 PS 设计**：开发数据驱动的自动划分机制，减少人工干预。
- **集成不确定性建模**：结合 Type-2 FLS 或贝叶斯训练，支持鲁棒预测与决策。
- **实际系统部署验证**：在真实工业控制系统中测试 xFODE 的在线辨识与控制性能。

--- 

> 📌 **一句话总结**：  
> xFODE 成功地在 **NODE/FODE 的高精度** 与 **模糊系统的可解释性** 之间取得平衡，通过 **增量状态 + 加法模糊模型 + 划分策略** 三大设计，构建了一个既准确又透明的 SysID 新范式。

</details>

---

### 9. [CROP: Token-Efficient Reasoning in Large Language Models via Regularized Prompt Optimization](https://arxiv.org/abs/2604.14214)

**Authors**: Deep Shah, Sanket Badhe, Nehal Kathrotia, Priyanka Tiwari  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.14214v1  

#### Abstract
Large Language Models utilizing reasoning techniques improve task performance but incur significant latency and token costs due to verbose generation. Existing automatic prompt optimization(APO) frameworks target task accuracy exclusively at the expense of generating long reasoning traces. We propos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CROP: Token-Efficient Reasoning in Large Language Models via Regularized Prompt Optimization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Large Language Models (LLMs)** 在复杂推理任务中广泛采用 **Chain-of-Thought (CoT)** 等逐步生成策略以提升准确率，但这类方法会产生大量冗长的中间推理文本，导致：
- 显著增加 **token consumption** 和 **inference latency**
- 推理成本高昂，限制了在生产环境中的实际部署

现有的 **Automatic Prompt Optimization (APO)** 框架（如 TextGrad）虽然能自动优化提示以提高任务准确率，但通常忽视输出长度，反而加剧了“**verbosity tax**”——即为了修复边缘错误而不断添加指令，导致推理链越来越长。

### 提出了什么新方法或新思路
作者提出 **CROP (Cost-Regularized Optimization of Prompts)**，一种新型的多目标 APO 框架，其核心思想是：
- 将 **response length** 作为显式的优化目标，引入一个基于自然语言的 **textual regularization gradient**
- 在 prompt 优化过程中同时考虑两个反馈信号：
  - `gtask`：来自任务准确性的标准反馈（如逻辑错误分析）
  - `greg`：来自响应简洁性的正则化反馈（鼓励压缩、去除冗余）

通过将这两个梯度进行字符串拼接（`gtotal = gtask ⊕ "\n" ⊕ greg`），引导 meta-optimizer 在提升准确率的同时主动压缩输出。

### 相比现有方法的优势
| 维度 | CROP | 传统 APO（如 TextGrad） |
|------|------|------------------------|
| 优化目标 | 多目标：accuracy + token efficiency | 单目标：仅 accuracy |
| 输出长度控制 | 显式、连续的 textual 正则项 | 无，甚至加剧冗长 |
| 是否需要模型微调 | 否，纯 prompt-level 优化 | 否 |
| 是否依赖特殊解码机制 | 否 | 否 |
| 实际部署友好性 | 高，适用于 agentic AI pipelines | 中，因高延迟和成本受限 |

> ✅ **关键优势**：CROP 在几乎不牺牲准确率的前提下，实现了高达 **80.6% 的 token 节省**，为低成本、低延迟的推理系统提供了实用解决方案。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验在三个具有挑战性的复杂推理基准上进行：
- **GSM8K**：小学数学应用题，测试多步算术推理能力
- **LogiQA**：源自专业逻辑考试的阅读理解数据集，要求类别、条件和析取推理
- **BIG-Bench Hard (BBH) – Object Counting**：算法性推理任务，需从自然语言描述中统计特定类别的对象数量

这些任务天然诱导 LLM 产生冗长的 CoT 迹象，适合验证压缩效果。

### 实验设置和评估指标
#### 模型角色分配
| 角色 | 模型 |
|------|------|
| **Target LLM (M)** | Gemini 2.0 Flash / Qwen 2.5 7B（用于前向推理） |
| **Meta-Optimizer (O)** | Gemini 3.1 Pro（启用 Thinking 功能，用于更新 prompt） |
| **Evaluator LLMs** | Gemini 2.0 Flash（双用途：计算 `gtask` 和 `greg`） |

#### 优化流程概览
- 初始 prompt → 批量采样输入 → 前向生成输出 → 双重反馈生成（accuracy + brevity）→ 梯度聚合 → meta-optimizer 更新 prompt
- 最终选择依据复合得分：  
  $$
  S(p) = \text{Accuracy}(p) - \lambda \cdot L_{\text{norm}}(p)
  $$
  其中 $\lambda=0.2$ 为最优正则系数（经网格搜索确定）

#### 评估指标
- **Accuracy (%)**：Exact match 准确率
- **Average Token Count**：输出 token 数量（衡量效率）
- **Token Reduction Rate**：相对于 CoT 的节省比例

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Zero-Shot (Direct Prompting)** | 不允许中间推理，直接输出答案（token 下限） |
| **Chain-of-Thought (CoT)** | 标准“Think step by step”，代表高准确率但高开销上限 |
| **BeConcise / Only Number / Abbreviate Words** | 各种启发式简洁提示 |
| **TextGrad** | 当前最先进的 APO 方法，仅优化 accuracy |
| **CROP (Ours)** | 提出的方法，联合优化 accuracy 与 length |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Gemini 2.0 Flash 结果）

| Dataset | Method | Accuracy (%) | Avg Token Count | Token Reduction vs CoT |
|--------|--------|---------------|------------------|-------------------------|
| **GSM8K** | CoT | 95.3 | 198.4 | — |
|          | CROP | **93.4** | **50.0** | **↓74.8%** |
|          | TextGrad | 95.5 | 185.5 | ↓6.5% |
|          | BeConcise | 96.2 | 120.8 | ↓39.1% |
| **Object Counting** | CoT | 96.0 | 101.2 | — |
|                     | CROP | **94.8** | **19.6** | **↓80.6%** |
|                     | TextGrad | 90.5 | 24.5 | ↓75.8% |
| **LogiQA** | CoT | 65.2 | 540.5 | — |
|           | CROP | **64.2** | **181.0** | **↓66.5%** |
|           | TextGrad | 66.4 | 663.0 | ↑22.7%（更长！） |

> 🔥 **最高达 80.6% 的 token 消耗减少**，且准确率下降极小（通常 <2%），显著优于所有 baseline。

### 与基线方法的对比结果
- CROP 在 **token 效率方面全面碾压 CoT 和 TextGrad**
- 相比人工设计的简洁提示（如 BeConcise），CROP 自动发现更优压缩策略，避免“一刀切”截断导致的信息丢失
- 在 **Qwen 2.5 7B** 上也观察到类似趋势，证明方法具有良好泛化性

### 消融实验结果
#### （1）Meta-Optimizer 容量影响
| Meta-Optimizer | 发现的 Prompt 特征 | 效果 |
|----------------|--------------------|------|
| Gemini 2.0 Flash（低容量） | “Only provide the answer” 类粗暴截断指令 | 导致 **reasoning collapse**，无法保留必要中间步骤 |
| Gemini 3.1 Pro（高容量） | “Extreme conciseness… avoid filler… focus on chained calculations” | 成功平衡 brevity 与 reasoning integrity |

> ✅ 表明：**高容量 meta-optimizer 是实现稳定多目标优化的关键**

#### （2）Batch Size 影响
| 设置 | Batch Size | 发现的 Prompt 行为 |
|------|------------|---------------------|
| Stochastic (B=1) | 1 | 过度关注单个样本的长度惩罚，频繁删除关键推理指令 |
| Full-batch (B=128) | 128 | 稳定收敛，有效抑制 regularization-induced collapse |

> ✅ 表明：**大 batch 提供统计代表性样本，防止梯度方差过大导致优化崩溃**

---

## 4. 关键结论和发现

### 论文的主要发现
1. **中间推理高度可压缩**：尽管 CoT 推理通常冗长，但其中存在大量“叙事填充”（conversational filler）和重复解释，可通过正则化有效剔除。
2. **CROP 能自主发现高效符号化推理模式**：例如在 GSM8K 上，模型自发采用类似 **"Chain-of-Draft"** 的数学简写风格（如 `8*5+8*(5*0.6)=64`），无需人工设计。
3. **多目标 textual optimization 可行且强大**：通过将 length penalty 编码为自然语言反馈，可在不修改模型权重的情况下实现端到端 prompt 压缩。
4. **生产部署友好**：最终得到的是静态 system prompt，可完美利用现代 LLM 服务的 **prefix caching** 机制进一步降低输入成本。

### 方法的局限性
1. **未显式优化 input token length**：虽然 output token 得到极大压缩，但优化后的 system prompt 本身可能较长。不过由于可缓存，实际影响较小。
2. **依赖高容量 meta-optimizer**：目前仅验证了 Gemini 3.1 Pro 的有效性，尚不清楚中小规模模型是否也能胜任此类多目标 prompt 编辑任务。
3. **正则化梯度稳定性依赖大 batch**：小 batch 易引发 prompt collapse，对资源有限场景构成挑战。

### 未来工作方向
- 探索轻量化 meta-optimizer 设计，使 CROP 更易于普及
- 将 input prompt 长度也纳入联合优化目标
- 扩展至更多模态（如 code generation、plan execution）中的 token 效率优化
- 结合 early exiting 或 speculative decoding，构建全栈式高效推理 pipeline

---

> 📌 **总体评价**：  
> CROP 是一项极具实践价值的工作，首次将 **regularization 思想系统引入 textual prompt optimization**，填补了“准确率 vs. 成本”之间长期存在的鸿沟。它不仅大幅降低了推理开销，还揭示了 LLM 推理过程中的“**语义密度提升空间**”，为下一代高效 agentic AI 系统的设计提供了重要范式。

</details>

---

### 10. [SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](https://arxiv.org/abs/2604.14672)

**Authors**: Binxian Su, Haoye Lou, Shucheng Zhu, Weikang Wang, Ying Liu, Dong Yu, Pengyuan Liu  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.14672v1  

#### Abstract
Large language models (LLMs) are being increasingly used in urban planning, but since gendered space theory highlights how gender hierarchies are embedded in spatial organization, there is concern that LLMs may reproduce or amplify such biases. We introduce SPAGBias - the first systematic framework ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本论文首次系统地研究了**Large Language Models (LLMs)** 在**城市空间语境下的结构性性别偏见（spatial gender bias）**。尽管已有大量研究关注 LLMs 中的职业、代词等维度的性别偏见，但**空间维度的性别偏见长期被忽视**。

该问题具有重要现实意义：
- LLMs 正越来越多地应用于城市规划、导航、灾害响应等依赖空间推理的任务；
- 若模型在“谁更可能出现在图书馆/夜店/厨房”这类判断中存在系统性偏见，可能导致资源分配不公、服务设计失衡（如医疗设施布局偏向男性活动模式）。

### ✅ 提出了什么新方法或新思路

作者提出了 **SPAGBias** —— 首个用于评估 LLMs 中空间性别偏见的多层级分析框架，其核心创新在于：

#### （1）构建了结构化的 Urban Space Taxonomy
- 包含 **62 个城市微空间**（43 个公共空间 + 19 个私人空间），涵盖交通、休闲、商业、医疗、家庭等多个场景。
- 基于城市地理学理论（如 feminist geographies）进行分类，确保社会文化相关性。

#### （2）设计了三种 Prompt 类型以从不同层面探测偏见
| Prompt 类型 | 功能 |
|-----------|------|
| **FCPrompt (Forced-Choice Prompt)** | 强制二选一：“Who is more likely to be found in the [SPACE]?” → 测量显性偏好 |
| **SGPrompt (Solo-Gender Prompt)** | 单人叙事生成：“Write a story about a man/woman in the [SPACE]…” → 分析词汇与情感倾向 |
| **CGPrompt (Co-Gender Prompt)** | 双人互动叙事：“Write a story about a man and a woman interacting in the [SPACE]…” → 分析角色权力关系 |

#### （3）提出三层诊断机制（Three Diagnostic Layers）
| 层级 | 方法 | 目标 |
|------|------|------|
| **Explicit Bias** | 多次采样 + Binomial Hypothesis Testing + EDI 指标 | 检测模型是否表现出显著的空间性别偏好 |
| **Probability Bias** | Log-probability 分析 | 揭示模型内部对性别词的概率不对称（即使输出中立） |
| **Construction Bias** | Semantic Role Labeling (SRL) + Interactional Positioning Theory + GPT-4o 自动标注 | 分析生成文本中的语义角色（Agent/Patient）、叙事角色（Leader/Supporter/Observer/Dependent） |

---

### ✅ 相比现有方法的优势

| 维度 | 传统方法局限 | SPAGBias 改进 |
|------|-------------|----------------|
| **分析粒度** | 多聚焦宏观职业或代词替换 | 聚焦微观城市空间，揭示细粒度映射（如“游戏室→男”，“步行衣帽间→女”） |
| **偏见检测深度** | 仅看表面输出 | 结合概率层与叙事层，区分“策略性拒绝”与“真实中立” |
| **理论基础** | 缺乏社会学支撑 | 融合 feminist geography 理论，赋予偏见发现更强解释力 |
| **可扩展性** | 多为单任务测试 | 框架模块化，支持跨模型、跨语言、跨文化比较 |

---

## 2. 核心实验方法和设置

### ✅ 使用的模型（LLMs）

共评测 **6 个代表性 LLMs**：
- GPT-3.5-turbo
- GPT-4
- Llama3-8B-instruct
- Qwen2-7B-instruct
- Phi-3-mini-4k-instruct
- Deepseek-llm-7b-chat

覆盖开源/闭源、不同规模、架构与训练策略，增强结论普适性。

---

### ✅ 数据集与资源

- **自建空间分类体系**：62 个 urban micro-spaces（见 Figure 8）
- **Prompt Library**：包含 FCPrompt、SGPrompt、CGPrompt 的模板集合
- **Pre-training Corpus 对照**：使用 C4 corpus 查询性别-空间共现频率（via WIMBD 工具）
- **Reward Model 对照**：FsfairX-LLaMA3-RM 和 Skywork-Reward-Llama-3.1-8B
- **Real-world Statistics**：来自 Mintel、NCHS、Morningstar、UNIDO 等机构的实际统计数据（用于下游任务验证）

---

### ✅ 实验设置与评估指标

| 任务 | 设置 | 指标 |
|------|------|-------|
| **Explicit Bias** | 每个空间重复采样 30 次（temperature=1） | Binomial test p-value, **Entropy Deviation Index (EDI)** |
| **Probability Bias** | 提取 gender token 的 log-probabilities | t-test, **Normalized Spatial Gender Co-occurrence (NSGC)** |
| **Construction Bias** | 每个空间生成 30 条 solo/co-story（temperature=1） | Adjective Odds Ratio (OR), Sentiment Polarity, **SRL Agent Rate**, **Narrative Role Score** |
| **Robustness Test** | 变换 prompt wording / temperature / 模型大小 | MAE, Direction Consistency (DC) |
| **Downstream Tasks** | City Planning Task (CP), User Profiling Task (UP) | OR (odds ratio), Match Rate |

---

### ✅ 基线方法对比

本文无传统“基线模型”，而是将多个主流 LLMs 视为横向对比对象，并通过以下方式体现差异：
- 各模型在同一 prompt 下的表现对比（如 Table 1, Figure 3）
- 不同发展阶段模型对比（如 Llama3-8b vs Llama3-8b-instruct）
- Reward model 与 base model 对比

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 🔹 显著偏见空间数量（Finding 1）
所有模型均表现出显著的空间性别偏见：

| Model | 显著偏见空间数（共62） |
|--------|------------------|
| Phi-3 | **62**（100%） |
| Llama3-8b-instruct | 59 |
| Qwen2-7b-instruct | 58 |
| GPT-3.5-turbo | 57 |
| GPT-4 | 55 |
| Deepseek-llm-7b-chat | 32（最少但仍广泛存在） |

> 即使是被认为对齐较好的 GPT-4，在回应时也表现出强烈偏好（尽管有 24.78% 拒绝率）。

#### 🔹 偏见强度（EDI）排名（Figure 3）
- **Phi-3** 偏见最强且最稳定（高 EDI + 低方差）
- **GPT-4** 偏见表达受 alignment 抑制，但一旦作答则仍具强偏向性

#### 🔹 典型高偏见空间（Heatmap）
- **女性关联强**：Kitchen, Beauty Salon, Nursing Home, Walk-in Closet, Children’s Room
- **男性关联强**：Garage, Game Room, Industrial Park, Sports Field, Mosque

#### 🔹 概率偏见（Log-probabilities）
- 除 Phi-3 外，其余模型在私有空间中均未表现出“女性主导”的统计趋势（Table 2）
- 表明传统“公域属男、私域属女”的简单划分已被更复杂的细粒度偏见取代

#### 🔹 叙事层偏见（Finding 4–6）
- **词汇层面**：男性故事更多使用 cold, lonely, gray, aged；女性则为 vibrant, warm, soft, resilient
- **语义角色**：GPT-4 中男性作为 **Agent (ARG0)** 的比例远高于女性（private: 0.81 vs 0.51；public: 0.80 vs 0.50）
- **叙事角色**：
  - 私人空间：男性多为 **Leader**，女性为 **Supporter**
  - 公共空间：出现反转，女性更常为 **Leader**，男性沦为 **Observer**（占比达 50.4%）

> 这表明模型捕捉到了现代话语中“女性公共可见性提升”的现象，但并未消除深层权力不平等。

---

### ✅ 与基线方法的对比结果

由于缺乏专门针对 spatial bias 的 prior work，本文通过以下方式凸显优势：

| 对比维度 | 本文发现 | 以往研究盲区 |
|---------|----------|--------------|
| 偏见结构 | 细粒度映射（非仅公/私二分） | 多停留在“职业-性别”刻板印象 |
| 偏见来源追踪 | 贯穿 pre-training → instruction tuning → RLHF 全流程 | 多归因于训练数据 |
| 下游影响 | 导致 normative 与 descriptive 任务双重失败 | 少有实证验证实际危害 |

---

### ✅ 消融实验结果（Robustness Analysis）

| 因素 | 影响程度 | 发现 |
|------|--------|------|
| **Prompt 变体** | 中等 | “选项顺序改变”影响最大（Prompt 2），尤其在中立空间易导致结果翻转 |
| **Temperature** | 较小 | Phi-3 几乎无变化；Deepseek 在低温下输出趋于极端 |
| **Model Scale** | 显著 | 更大模型（如 Llama3-70b）偏见更强，说明 scaling law 可能放大偏见 |

> 尽管敏感，但总体偏见模式保持一致（Direction Consistency > 76%），证明结论稳健。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLMs 存在系统性、结构化的空间性别偏见**，超越传统的“公共-私人”二元对立，形成精细的 micro-level 映射。
2. 偏见不仅体现在输出选择上，还深植于 **token-level 概率分布** 和 **叙事建构机制** 中。
3. **GPT-4 等对齐模型虽会拒绝部分 prompt，但其内部偏好依然强烈**，说明 alignment 未能根除偏见。
4. 偏见贯穿整个模型 pipeline：
   - **Pre-training data**：C4 中已存在性别-空间共现不平衡
   - **Instruction Tuning**：略有修正但核心配对不变
   - **Reward Modeling**：进一步强化既有偏见（reward models 本身即具强偏见）
5. **下游应用双失败**：
   - **Normative Task (CP Task)**：模型基于性别刻板印象做决策（OR << 1）
   - **Descriptive Task (UP Task)**：过度去偏，无法反映真实分布（准确率仅 5%-20%）

> 模型既不能公平决策，也不能如实描述现实。

---

### ✅ 方法的局限性

| 局限 | 说明 |
|------|------|
| **Binary Gender Paradigm** | 仅考虑 men/women 二元框架，忽略 non-binary 群体，可能加剧边缘化 |
| **英语中心主义** | 所有实验基于英文 prompt 和 corpus，未涵盖多语言情境 |
| **空间粒度有限** | 未细分子空间（如 CEO office vs 普通办公室） |
| **真实数据不足** | 缺乏全球统一、细粒度的空间性别使用统计数据，难以精确校准 |
| **因果推断受限** | 无法完全控制同一模型各阶段变量，结论多为趋势观察而非因果证明 |

---

### ✅ 未来工作方向

1. **扩展空间覆盖范围**：纳入农村、郊区、宗教场所等多元环境
2. **推进多语言与跨文化比较**：检验不同语言中 spatial gender bias 是否受文化调节
3. **引入非二元性别视角**：构建 inclusive 的 prompt 与评估体系
4. **开发干预机制**：探索如何在保留有用常识的同时削弱有害偏见
5. **建立标准 benchmark**：推动 SPAGBias 成为 LLM fairness 评估的标准组件之一

---

## 总结

> 📌 **SPAGBias 是首个将 feminist geography 理论与 computational linguistics 深度结合的工作，揭示了 LLMs 如何通过语言编码并再生产社会空间中的性别秩序。它不仅是 bias detection 工具，更是连接社会科学与 AI ethics 的桥梁。**

该研究表明：当前 LLMs 不仅“知道”性别刻板印象，还能“讲述”符合这些偏见的故事，并在实际应用中造成双重失败 —— **既不公平，也不真实**。这警示我们：若要在城市治理、公共服务等领域安全部署 LLM，必须首先解决其深层的社会认知偏差问题。

</details>

---

### 11. [RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](https://arxiv.org/abs/2604.14885)

**Authors**: Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.14885v1  

#### Abstract
Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but existing training-free variants face trade-offs: retrieval-based drafts break when no exact match ex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）采用自回归解码（autoregressive decoding），逐个生成 token，导致推理延迟高、效率低。**Speculative Decoding (SD)** 是一种通过“猜测-验证”策略加速推理的方法，但现有训练无关（training-free）方法存在以下瓶颈：

- **基于检索的方法**（如 PLD、REST）：依赖精确的 token 序列匹配，当上下文中无完全匹配时失效，泛化能力差。
- **基于 logits 的方法**（如 Token Recycling、LogitSpec）：缺乏外部结构引导，推测范围受限，难以生成高质量的长序列草案。

### 🚀 提出的新方法：RACER
提出 **RACER**（Retrieval-Augmented Contextual Rapid Speculative Decoding），一种轻量级、无需训练的 speculative decoding 方法，其核心思想是：

> **将检索到的精确模式作为“可靠锚点”，将 logits 驱动的预测作为“动态外推线索”，统一构建更丰富、更准确的草案树（draft tree）。**

#### 创新点：
1. **双信号融合机制**：
   - **Retrieval Tree**：利用 Aho-Corasick 自动机维护 n-gram 模式，提供结构化、可靠的草案候选。
   - **Logits Tree**：基于“copy-logit”策略复用历史 logits，实现对未见未来的动态推测。
2. **LRU 缓存淘汰机制**：在检索自动机中引入 **Least Recently Used (LRU)** 策略，动态管理有限节点容量，保留高频、近期使用的 n-gram，提升内存效率。
3. **智能集成策略**：优先使用检索候选，不足部分由 logits 扩展补足，实现“结构引导 + 动态探索”的协同。

### 🔍 相比现有方法的优势
| 方法类型 | 代表 | 局限 | RACER 的优势 |
|--------|------|------|-------------|
| 检索型 | PLD, REST | 依赖精确匹配，稀疏且不稳定 | 引入 logits 补全，增强泛化能力 |
| logits 型 | TR, LogitSpec | 缺乏结构引导，推测质量低 | 引入检索锚点，提升草案可靠性 |
| 模型型 | EAGLE-3 | 需额外训练模型，开销大 | 无需训练，即插即用，轻量高效 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个基准上进行评估，覆盖多种任务类型：
- **Spec-Bench**：通用任务集合，包括多轮对话（MT）、翻译（Trans）、摘要（Sum）、问答（QA）、数学推理（Math）、检索增强生成（RAG）。
- **HumanEval**：代码生成任务，评估函数实现能力。
- **MGSM-ZH**：中文数学推理任务（GSM8K 中文版），用于测试非英语场景下的表现。

### ⚙️ 实验设置
- **目标模型**：Vicuna（7B, 13B, 33B）、LLaMA3.1-8B、OpenPangu-7B、Qwen3（8B, 14B, 32B）等。
- **解码方式**：greedy decoding，batch size = 1，最大输出长度 1024。
- **硬件环境**：NVIDIA RTX4090 / A800 GPU，PyTorch + HuggingFace Transformers。
- **超参数**：
  - 草案大小（draft size）：64
  - Logits Tree 最大宽度：8
  - Retrieval Tree 容量：最多 10,000 节点，n-gram 长度为 10

### 📊 评估指标
- **Mean Accepted Tokens (MAT)**：每步 speculative decoding 平均接受的 token 数量，衡量草案质量。
- **Speedup Ratio**：相对于标准自回归解码的速度提升倍数。

### 🆚 基线方法对比
| 类型 | 方法 | 简介 |
|------|------|------|
| 检索型 | **PLD**, **REST** | 基于 n-gram 或后缀数组的检索 |
| logits 型 | **Token Recycling (TR)**, **LogitSpec** | 复用历史 logits 构建草案 |
| 模型型 | **EAGLE-3** | 使用额外训练的小模型生成草案 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）
| 模型 | 方法 | MAT (平均) | Speedup (平均) |
|------|------|------------|----------------|
| Vicuna 7B | RACER | **3.27** | **2.42×** |
| Vicuna 13B | RACER | **3.23** | **2.50×** |
| Vicuna 33B | RACER | **3.09** | **2.52×** |
| LLaMA3.1-8B | RACER | **3.12** | **2.72×** |
| Qwen3-8B | RACER | **2.82** | **2.21×** |

> ✅ **RACER 在所有模型和任务上均取得最高或接近最高的 speedup，且 MAT 显著优于其他 training-free 方法。**

### 🔁 与基线方法对比
- **相比 PLD/REST**：RACER 的 speedup 提升超过 2×，MAT 提升 50% 以上。
- **相比 TR/LogitSpec**：RACER 在 MAT 上高出 0.3–0.6，在 MGSM-ZH 等推理任务上优势尤为明显。
- **相比 EAGLE-3（模型型）**：
  - EAGLE-3 在 MAT 上有时更高（因训练优化），但 **RACER 的 speedup 更高**，因其无需额外模型推理，验证成本更低。
  - 特别是在中文任务（MGSM-ZH）上，EAGLE-3 因训练数据偏差表现下降，而 RACER 保持稳定。

### 🔍 消融实验结果（Ablation Study）
#### （1）移除组件的影响（Table 3）
| 设置 | MAT ↓ | Speedup ↓ | 结论 |
|------|-------|-----------|------|
| w/o logits | 1.5–1.8 | 1.3–1.4× | logits 是推测主干 |
| w/o retrieval | 2.5–2.7 | 2.0–2.2× | retrieval 提供关键结构引导 |
| **RACER（完整）** | **3.0–3.2** | **2.4–2.5×** | 双信号协同增效 |

> ✅ 移除任一组件均显著降低性能，证明两者互补。

#### （2）集成策略对比（Table 5）
| 策略 | MAT | Speedup |
|------|-----|---------|
| Merge（RACER） | **3.00** | **2.18×** |
| Half（固定分配） | 2.69 | 1.97× |
| Hard（回退切换） | 2.77 | 2.11× |

> ✅ “Merge” 策略最优，能灵活协调检索与 logits 的贡献。

#### （3）参数鲁棒性（Figure 8）
- RACER 在不同 draft size（16–64）、node capacity（2.5K–20K）、n-gram depth 和 top-k breadth 下性能稳定。
- 最佳参数集中在 n-gram depth ≈ 9–11，top-k ≈ 8–10，符合 copy-logit 的 85% 分位排名观察。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **检索与 logits 具有互补性**：
   - 检索提供“可靠锚点”，适用于重复模式；
   - logits 提供“动态外推”，适用于新颖内容；
   - 二者结合可构建更丰富、更准确的草案。
2. **RACER 是轻量高效的通用方案**：
   - 无需训练，即插即用；
   - 内存占用可控（O(n)），适合边缘部署；
   - 在多种模型、任务、语言下均实现 **>2× speedup**。
3. **copy-logit 优于 last-logit**：
   - 复用相同 token 上一次出现时的 logits（copy-logit）比复用当前 logits（last-logit）效果更好，因其保留了语义一致性。

### ⚠️ 局限性
- **未测试多模态任务**：当前仅针对文本任务，是否适用于 vision/audio token 尚不明确。
- **依赖上下文重复性**：在高度原创、无重复模式的任务中，检索增益可能减弱。
- **长程依赖挑战**：随着输出变长，logits 的预测能力受注意力衰减影响。

### 🔮 未来工作方向
1. 探索多模态 token 的检索与融合机制。
2. 引入更高级的检索结构（如向量索引）以捕捉语义相似性而非精确匹配。
3. 结合 parallel decoding 或 distributed inference 框架，进一步提升吞吐。
4. 研究多语言检索提示（multilingual retrieval cues）以增强跨语言泛化。

---

> **总结**：RACER 通过巧妙融合 retrieval 与 logits 两种信号，在无需训练的前提下实现了高效、稳定、通用的 speculative decoding，为 LLM 推理加速提供了新的范式。

</details>

---

### 12. [Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving](https://arxiv.org/abs/2604.14993)

**Authors**: Tingyang Sun, Ting He, I-Hong Hou  
**Category**: cs.DC  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.14993v1  

#### Abstract
As a current trend in Artificial Intelligence (AI), large foundation models are increasingly employed as the core of AI services. However, even after training, serving such models at scale remains a challenging task due to their heavy resource footprints, particularly in terms of GPU memory. While r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

本文针对**大规模基础模型（Large Foundation Models）在推理服务中的资源分配瓶颈**问题展开研究，特别是当采用 **pipeline parallelism**（流水线并行）进行分布式部署时，如何高效地组合服务器链（server chain）以优化系统性能。

传统方法将模型分块（如Transformer层）放置到不同GPU上，并通过链式调用处理请求。然而，这种模式面临两个核心挑战：
- **GPU内存是瓶颈资源**：不仅要存储模型参数（shared），还需为每个请求分配KV Cache（dedicated），导致严重的内存竞争。
- **缺乏对“可组合服务器链”（composable server chains）的系统化建模与优化**：即如何联合设计 block placement（块放置）、cache allocation（缓存分配）和 load balancing（负载均衡）。

### 提出了什么新方法或新思路

作者提出了一种**两阶段优化框架**，从排队论和服务功能链（SFC）等角度抽象出一个全新的问题——“**Server Chain Composition via Block Placement and Cache Allocation**”。

#### 主要创新点包括：

1. **问题建模创新**：
   - 将 LLM 推理任务抽象为“chain-structured, memory-bound jobs”，明确区分了模型参数内存与KV Cache内存的使用方式。
   - 引入“job server”概念：由一组物理服务器构成的逻辑服务单元，具备特定的服务速率 $ \mu_k $ 和并发容量 $ c_k $。

2. **算法设计创新**：
   - **GBP-CR (Greedy Block Placement with Cache Reservation)**：一种高效的贪心块放置算法，在预设最小并发容量 $ c $ 下，优先将快速服务器组成短链，平衡服务时间与等待时间。
   - **GCA (Greedy Cache Allocation)**：基于最短路径动态构建可行 server chain 并最大化缓存分配，仅需考虑多项式数量级的链集合（而非指数级），适用于 JFFS 类策略。
   - **JFFC (Join-the-Fastest-Free-Chain)**：适配于多任务并行场景的负载均衡策略，扩展了经典的 JFFS 策略。

3. **理论分析贡献**：
   - 证明了最优 cache allocation 是 NP-hard。
   - 对 GBP-CR 在同构内存环境下的最优性进行了证明（Theorem 3.4）。
   - 给出了 JFFC 策略下稳态平均响应时间的闭式上下界（Theorem 3.7），可用于指导超参调优。

### 相比现有方法的优势

| 方面 | 优势说明 |
|------|----------|
| **系统视角更全面** | 联合优化 block placement + cache allocation + load balancing，而多数现有系统（如 vLLM, PETALS）仅关注实现细节或局部调度。 |
| **性能更高** | 实验显示相比 PETALS 和 BPRR，平均响应时间降低 **63–77%**。 |
| **更具鲁棒性** | 即使真实流量偏离泊松假设，仍能保持优异性能。 |
| **可解释性强** | 提供显式的性能边界分析，支持参数自动调优（如选择最佳 $ c^* $）。 |

---

## 2. 核心实验方法和设置

### 使用的数据集与模型

- **模拟实验**：
  - 模型：BLOOM-176B（70 层 Transformer）
  - 参数大小：每 block $ s_m = 1.32\,\text{GB} $，KV Cache $ s_c = 0.11\,\text{GB} $
  - 请求长度：平均输入 2000 tokens，输出 20 tokens
  - 到达过程：Poisson 过程（默认到达率 $ \lambda = 0.2\,\text{req/s} $）

- **真实系统实验**：
  - 系统平台：修改版 **PETALS** [6]
  - 部署环境：单台机器上的 9 个 MIG 实例（3×3g.40gb + 6×2g.20gb）
  - 模型：LLaMA-2-7B（32 层）
  - 流量来源：Azure LLM inference trace（2023年11月11日采集）
    - 平均到达率：2.57 req/s
    - 输入长度：2048 tokens
    - 输出长度：28 tokens
  - 网络延迟：基于 RIPE Atlas European network 的 RTT 数据模拟

### 实验设置和评估指标

| 设置项 | 描述 |
|-------|------|
| **对比方法** | 
| • `PETALS` [6] | 原始启发式块放置与路由 |
| • `BPRR` [29] | 最近提出的两阶段资源分配算法 |
| • `JFFC only` | 所有服务器独立运行完整模型实例，仅使用 JFFC 负载均衡 |
| • `Proposed`: GBP-CR + GCA + JFFC | 本文完整方案 |
| **评估指标** | 
| • Mean/Median/P95/P99 Response Time | 总响应时间（含排队+处理） |
| • Waiting Time / Service Time 分解 | 分析性能提升来源 |
| • Improvement vs. baseline | 相对性能增益 |
| • 参数敏感性测试 | 如 $ c $ 对性能的影响 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | Mean RT (s) | P95 RT (s) | Mean Wait (s) | Mean Serv (s) |
|------|-------------|------------|----------------|----------------|
| PETALS [6] | 31.4 | 68.5 | 24.2 | 7.2 |
| BPRR [29] | 19.8 | 44.2 | 12.6 | 7.2 |
| JFFC only | 10.0 | 22.1 | 1.5 | 8.5 |
| **Proposed** | **7.3** | **15.2** | **0.6** | **6.7** |

> ✅ **性能提升显著**：
- 相比 PETALS：**平均响应时间下降 76.8%**
- 相比 BPRR：**平均响应时间再降 63.1%**
- P95 响应时间下降 **77.8%**

### 与基线方法的对比结果

| 对比维度 | 结果分析 |
|--------|---------|
| **vs. PETALS/BPRR** | 改进主要来自 **waiting time 的大幅压缩**（减少 >97%），表明本文方法能更有效地利用缓存空间实现高并发。 |
| **vs. JFFC only** | 尽管后者也用了先进负载均衡，但由于未拆分模型，无法充分预留 cache，导致并发能力受限；本文通过智能 block placement 实现了更好的资源利用率。 |
| **在低资源环境下优势更大** | 图8显示，在服务器数少（J小）或高性能GPU比例低（η小）时，性能增益可达 **83%**，说明该方法特别适合边缘/去中心化部署场景。 |

### 消融实验结果（Ablation Study）

- **参数 $ c $ 的影响（图6）**：
  - 存在一个最优 $ c^* $（本实验中约为7），过小则并发不足，过大则服务链变长、服务时间上升。
  - 使用 Theorem 3.7 中的 **lower bound on response time** 可准确预测 $ c^* $，优于代理目标函数 $ c \cdot K(c)/\lambda $。

- **最优 $ c^* $ 随负载变化的趋势（图7）**：
  - 低负载时偏好较小 $ c $（优先缩短服务链）
  - 高负载时偏好较大 $ c $（优先提高并发度）
  - 本文提出的 bound 能正确捕捉这一趋势，验证其用于在线调参的有效性。

---

## 4. 关键结论和发现

### 主要发现

1. **内存是 LLM 推理服务的核心瓶颈**，必须联合考虑模型放置与KV Cache分配。
2. **Server chain composition 是一个可优化的关键环节**，不能仅依赖静态部署或简单启发式。
3. **GBP-CR + GCA + JFFC 构成的两阶段框架显著优于现有方法**，尤其在去中心化、异构环境中表现突出。
4. **理论推导的性能边界具有实际指导意义**，可用于自动化参数调优（如选择最优 $ c $）。

### 方法的局限性

- **假设静态缓存分配**：未考虑 dynamic KV caching 或 paging 技术（如 vLLM 的 PagedAttention）。
- **集中式调度器依赖**：目前假设有全局协调者（orchestrator），不适用于完全去中心化的 P2P 场景。
- **未建模动态负载变化**：当前优化基于固定到达率 $ \lambda $，难以应对突发流量。
- **简化了通信开销模型**：假设通信时间为常量，忽略了带宽波动和拥塞效应。

### 未来工作方向

1. **支持动态请求长度与弹性缓存管理**，结合现代推理引擎特性（如 vLLM）。
2. **开发分布式版本的 server chain composition 算法**，适应无中心节点的 Internet-scale 部署。
3. **引入时间维度优化**：针对周期性或突增型负载进行自适应 reconfiguration。
4. **集成更多控制维度**：如 tensor parallelism、量化、offloading 等，形成统一的 multi-granularity resource orchestration 框架。

--- 

> 📌 **一句话总结**：  
> 本文首次系统性地提出了“server chain composition”这一新型资源管理问题，通过联合优化 block placement、cache allocation 与 load balancing，在理论与实践中均实现了对大规模模型推理服务性能的显著提升，为构建高效、可扩展的 LLM serving 系统提供了坚实基础。

</details>

---

### 13. [TOPCELL: Topology Optimization of Standard Cell via LLMs](https://arxiv.org/abs/2604.14237)

**Authors**: Zhan Song, Yu-Tung Liu, Chen Chen, Guoheng Sun, Jiaqi Yin, Chia-tung Ho, Ang Li, Haoxing Ren, Cunxi Yu  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.14237v1  

#### Abstract
Transistor topology optimization is a critical step in standard cell design, directly dictating diffusion sharing efficiency and downstream routability. However, identifying optimal topologies remains a persistent bottleneck, as conventional exhaustive search methods become computationally intractab...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TOPCELL: Topology Optimization of Standard Cell via LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
标准单元（standard cell）设计中的**晶体管拓扑优化**是影响布局布线质量（routability）、面积（area）和寄生参数的关键环节。传统方法如递归搜索或穷举探索（如 SO3-Cell）在晶体管数量增加时面临**指数级复杂度**，导致计算成本过高，难以扩展至先进工艺节点（如 7nm、2nm）。

此外，现有自动化方案缺乏对物理实现（physical awareness）的充分建模，且无法高效探索高质量拓扑结构。

---

### 🚀 提出的新方法与创新思路
本文提出 **TOPCELL** —— 一种基于 **Large Language Models (LLMs)** 的新型可扩展框架，用于标准单元的拓扑优化。其核心思想是将高维拓扑空间探索问题重构为一个**生成式任务**，由 LLM 驱动策略进行优化。

#### 主要创新点：
1. **LLM 驱动的拓扑优化框架**
   - 将标准单元的 SPICE netlist 输入给 LLM，模型输出建议的“pivot net”以触发局部拓扑变换。
   - 利用确定性算法（LLM-Guided Topology Permutation）执行连接重布线，保证功能等价性。

2. **引入 GRPO 进行强化学习微调**
   - 使用 **Group Relative Policy Optimization (GRPO)** 对 LLM 进行 post-training，使其策略与下游 P&R 反馈对齐。
   - GRPO 不依赖显式的 critic 网络，稳定性高、内存效率好，适合 EDA 中复杂的奖励信号学习。

3. **端到端的奖励建模机制**
   - 构建了一个基于 **Graph Neural Network (GNN)** 的轻量级奖励模型，替代耗时的完整 P&R 流程提供 routability 评分，极大加速训练过程。

4. **强大的零样本泛化能力（Zero-shot Generalization）**
   - 模型仅在 **3-input、2nm 工艺**的数据上训练，却能成功应用于更复杂的 **4–6 输入、7nm 工艺的标准单元**，展现出卓越迁移能力。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 SO3-Cell） | TOPCELL |
|------|--------------------------|--------|
| **可扩展性** | 指数级搜索，随晶体管数增长不可行 | 多项式时间推理，支持大规模单元 |
| **运行速度** | 数小时甚至十小时以上 | 秒级完成拓扑优化 |
| **物理感知能力** | 弱，依赖预定义规则 | 强，通过 GNN + P&R 反馈联合优化 |
| **自动化程度** | 需人工干预或启发式剪枝 | 完全自动化的生成式探索 |

---

## 2. 核心实验方法和设置

### 📊 数据集构建
- **布尔函数覆盖**：枚举所有非平凡的 **3-input 单输出 Boolean functions**（共 254 个）。
- **逻辑综合**：使用 ABC 工具生成优化后的 factored form，并合成为初始 SPICE netlist。
- **拓扑枚举引擎**：利用 LLM-Guided Topology Permutation 自动生成每个函数最多 100 种拓扑变体，总计生成 **7,918 个 unique netlists**。
- **筛选训练集**：从中提取 **2,039 个 unroutable 设计**作为训练/验证集，专注于修复差拓扑。

> ⚠️ 注：未使用 4-input 函数因组合爆炸（65536 种），无法穷举处理。

---

### ⚙️ 实验设置
- **基础模型**：
  - `Qwen2.5-Coder-3B` 和 `Qwen2.5-Coder-7B`
- **训练配置**：
  - 使用 Verl 库 + SGLang 框架实现 GRPO
  - Batch size: 256；Epochs: 15；Learning rate: 1e-6
  - 平台：NVIDIA DGX Station（4×A100 80GB）
- **GNN 奖励模型**：
  - 基于 PyTorch Geometric 构建
  - 输入为 netlist 图结构（nets 为节点，transistors 分解为双有向边）
  - 输出 routability 得分（routable/unroutable 分类 + margin loss）

---

### 🎯 评估指标
| 指标 | 含义 |
|------|------|
| **Routable Rate (%)** | 原本 unroutable 的单元经优化后变得可布线的比例 |
| **PDA Congestion (Mean)** | NVCell2 提出的局部拥塞度量，值越低表示拓扑越利于布线 |
| **Optimal Total Cost (OTC)** | SO3-Cell 框架中的综合 DTCO 成本目标 |
| **End-to-End Runtime / Speedup** | 总耗时对比（含 tLLM + tp&R），衡量效率提升倍数 |

---

### 🆚 基线方法对比
- **Foundation Models（零样本）**：
  - Qwen 系列（3B–32B）、CodeLlama（7B/34B）、Llama-3.3-70B、GPT-5、Gemini、Claude Opus 等
- **SOTA 自动化流程**：
  - **SO3-Cell**：当前最先进的标准单元自动化设计框架，采用联合优化 topology/placement/routing
- **消融实验基线**：
  - **SFT（Supervised Fine-Tuning）**：监督式微调，以最优单步交换为标签

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）

#### ✅ 在 2nm 节点上的 routability 优化表现（Table 1）
| 模型 | Routable Rate | PDA Congestion |
|------|---------------|----------------|
| **TOPCELL-7B** | **77.3%** (+24.3pp) | **3.90** (-8.02%) |
| **TOPCELL-3B** | **69.7%** (+14.1pp) | **4.06** (-6.02%) |
| Qwen2.5-Coder-7B (base) | 53.0% | 4.24 |

> 💡 即使是小模型 TOPCELL-3B，也显著优于更大的通用模型（如 GPT-5: 56.1%, DeepSeek-V3.2-Exp: 56.6%）

#### ✅ 与 SO3-Cell 集成后的性能对比（Table 2）
在 **7nm 工艺、≥4 输入单元** 上测试，**平均加速达 85.91×**

| 单元 | #Transistors | SO3-Cell Runtime (s) | TOPCELL Runtime (s) | Speedup × |
|------|--------------|-----------------------|------------------------|-----------|
| AOI222_X1_SH | 12 | 35,488 | 63.14 | **562×** |
| OAI222_X1_SH | 12 | 8,066 | 65.77 | **122.65×** |
| NAND4_X2 | 8 | 3,259 | 189.59 | **17.19×** |

✅ 所有案例中 OTC 成本基本一致（除 OAI211_X1 外），说明 **布局质量相当，但速度快两个数量级**

---

### 🔬 消融实验结果（Figure 4）
比较 **GRPO vs SFT** 的训练效果：

| 方法 | 最终 Routable Rate | 是否饱和 |
|------|--------------------|----------|
| **GRPO** | >77% | 否，持续上升 |
| **SFT** | ~57% | 是，在早期即饱和 |

> ❗ 结论：SFT 只能模仿已有“黄金答案”，无法探索新解；而 GRPO 通过奖励驱动主动探索，突破上限。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLMs 可有效用于标准单元拓扑优化**  
   LLM 能理解 netlist 结构并提出有意义的拓扑修改建议，尤其在结合 GRPO 后具备强物理感知能力。

2. **GRPO 是比 SFT 更适合该任务的学习范式**  
   因为不存在唯一的“最优拓扑”，多解共存，GRPO 的相对优势估计机制更适合此类任务。

3. **TOPCELL 实现了前所未有的效率-质量平衡**  
   在保持与 SOTA 方法相当布局质量的前提下，实现 **平均 85.91× 加速**，解决了传统方法的可扩展瓶颈。

4. **具备强大零样本泛化能力**  
   从 3-input → 6-input、2nm → 7nm 的跨域迁移成功，表明模型学到了可迁移的设计原则。

---

### ⚠️ 局限性
1. **依赖高质量 GNN 奖励模型**  
   若 GNN 不能准确预测 routability，则 LLM 学习会偏离真实目标。
   
2. **拓扑变换操作受限于局部 swap**  
   当前仅支持围绕 pivot net 的局部重连，可能错过全局结构性跃迁。

3. **训练数据局限于 3-input 函数**  
   虽然泛化能力强，但仍缺乏对更高输入维度的直接监督信号。

---

### 🔮 未来工作方向
1. **扩展到 multi-row standard cell design**  
   当前聚焦于单行单元，未来可拓展至多行布局场景。

2. **引入更多物理约束建模**  
   如 parasitics、timing、power 等，构建更全面的 reward model。

3. **探索多轮迭代优化机制**  
   当前为单步优化，未来可通过 chain-of-thought 或 multi-step RL 实现渐进式改进。

4. **构建专用 LLM 架构 for EDA**  
   开发专用于电路设计的稀疏化、图增强型 LLM，进一步提升效率与精度。

---

> ✅ **总体评价**：TOPCELL 开辟了 **AI-driven DTCO（Design-Technology Co-Optimization）** 在标准单元层级的新路径，标志着 LLM 在 EDA 领域从“代码生成”迈向“物理感知设计优化”的重要一步。

</details>

---

### 14. [Physics-Informed Machine Learning for Pouch Cell Temperature Estimation](https://arxiv.org/abs/2604.14566)

**Authors**: Zheng Liu  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.14566v1  

#### Abstract
Accurate temperature estimation of pouch cells with indirect liquid cooling is essential for optimizing battery thermal management systems for transportation electrification. However, it is challenging due to the computational expense of finite element simulations and the limitations of data-driven ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Physics-Informed Machine Learning for Pouch Cell Temperature Estimation**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对**间接液冷软包电池（pouch cell）的温度分布估计难题**展开研究。准确的温度估计对电池热管理系统（BTMS）至关重要，但传统高保真有限元仿真（FE）计算成本高昂，难以用于实时优化；而纯数据驱动模型在训练数据稀疏时泛化能力差，且可能违背物理规律，导致不合理预测。

### **提出了什么新方法或新思路**
提出了一种**基于物理信息机器学习（Physics-Informed Machine Learning, PIML）的框架**，用于高效、可靠地估计稳态温度场。其核心思想是将控制热传导的偏微分方程（PDE）直接嵌入神经网络的损失函数中，构建一个**Physics-Informed Neural Network (PINN)**。

具体创新包括：
- 将二维稳态热扩散方程 $ k \cdot t \cdot \nabla^2 T + q_{\text{gen}} - h(x,y)(T - T_{\text{coolant}}) = 0 $ 显式编码为 PDE 残差项。
- 设计复合损失函数，联合优化数据拟合误差（MSE）、PDE 残差和边界条件（BC）满足度。
- 利用连续坐标 $(x, y)$ 作为输入，输出温度标量 $T(x, y)$，实现网格无关建模潜力。

### **相比现有方法的优势**
- **更高的准确性**：尤其在远离冷却通道的区域表现更优。
- **更快的收敛速度**：仅需少量训练样本即可快速收敛。
- **更强的泛化能力**：即使在数据稀疏场景下仍能保持合理预测，因物理约束提供了先验知识。
- **减少对大量仿真/实验数据的依赖**：适用于设计探索阶段的数据稀缺情况。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 数据由**有限差分法（Finite Difference Method, FDM）模拟生成**。
- 共构建 **100 组不同冷却通道几何形状**的样本，每组包含：
  - 输入：二值化的冷却通道掩码 $M(x,y)$
  - 输出：对应的稳态温度场 $T$
- 所有样本均来自同一尺寸（154×203 mm²）的简化二维软包电池表面模型。

### **实验设置和评估指标**
- **空间离散化**：采用与 FDM 相同分辨率（$154 \times 203$ 网格），步长 1 mm。
- **训练/测试划分**：随机划分 80%/20%。
- **目标变量处理**：温度场标准化（zero-mean, unit-variance）以提升数值稳定性。
- **评估指标**：
  - **Mean Squared Error (MSE)**：主评价指标。
  - 可视化对比真实温度场、预测结果及误差图。

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Data-Driven Model** | 基于 Fully Convolutional Network (FCN) 的纯数据驱动代理模型，仅最小化 MSE 损失。 |
| **PIML Model (PINN)** | 提出的方法，最小化总损失 $ \mathcal{L}_{\text{total}} = w_1 \cdot \mathcal{L}_{\text{MSE}} + w_2 \cdot \mathcal{L}_{\text{PDE}} + w_3 \cdot \mathcal{L}_{\text{BC}} $ |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在训练 **10 个 epoch 后**：
  - **PIML 模型的 MSE 下降至 5.66**
  - **Data-Driven 模型的 MSE 为 11.12**
  - → **相对降低 49.1%**，显著提升收敛效率与精度。

### **与基线方法的对比结果**
- 在独立验证集上，PIML 模型在所有通道设计中均表现出更高精度。
- 特别是在**远离冷却通道的区域（如电池中心或边缘）**，PIML 明显优于数据驱动模型，后者常出现过平滑或偏差较大的现象。
- 可视化结果显示 PIML 更好地捕捉了非均匀温度梯度和局部热点。

### **消融实验结果**
- 文中未明确进行系统性消融实验（ablation study），但通过损失项的设计隐含说明各部分作用：
  - $\mathcal{L}_{\text{MSE}}$：确保与已知数据一致；
  - $\mathcal{L}_{\text{PDE}}$：强制满足物理规律；
  - $\mathcal{L}_{\text{BC}}$：软约束实现绝热边界条件。
- 实验表明，引入 PDE 和 BC 损失显著提升了模型外推能力和物理一致性。

---

## **4. 关键结论和发现**

### **论文的主要发现**
- **PIML 能有效融合物理先验与数据驱动优势**，在少量样本下实现高保真温度场预测。
- 相比纯 FCN 模型，PIML 收敛更快、误差更低（MSE ↓49.1%），且在复杂几何配置下更具鲁棒性。
- 物理约束的引入显著改善了模型在**数据未覆盖区域**（如远离冷却通道处）的表现，增强了可信度。
- 该方法可作为高效的**代理模型（surrogate model）**，支持 BTMS 快速设计迭代与优化。

### **方法的局限性**
- 当前模型基于**稳态假设**，尚未考虑动态充放电过程中的瞬态热行为。
- 依赖精确的物性参数（如 $k$, $h_{\text{coeff}}$）和边界条件设定，若参数不准会影响预测效果。
- 训练过程中需计算高阶导数（通过自动微分），可能导致数值不稳定，需 careful loss weighting。

### **未来工作方向**
- 扩展至**瞬态热传导问题**，建立时间-空间联合 PIML 模型。
- 应用于**冷却通道拓扑优化设计**，结合 PIML 与优化算法实现自动结构搜索。
- 开展**实验验证**，利用红外测温等手段获取实测温度场，进一步检验模型泛化能力。
- 探索**多保真度建模（multi-fidelity modeling）**，融合低精度模拟与高精度实测数据。

--- 

> ✅ **总结一句话**：  
> 本文提出的 PIML 框架通过将热传导 PDE 嵌入神经网络训练过程，在仅使用 100 个仿真样本的情况下，实现了比传统数据驱动方法快近两倍、精度提高 49.1% 的软包电池温度场估计，展现出在电池系统设计优化中的巨大潜力。

</details>

---

### 15. [SOLIS: Physics-Informed Learning of Interpretable Neural Surrogates for Nonlinear Systems](https://arxiv.org/abs/2604.14879)

**Authors**: Murat Furkan Mansur, Tufan Kumbasar  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.14879v1  

#### Abstract
Nonlinear system identification must balance physical interpretability with model flexibility. Classical methods yield structured, control-relevant models but rely on rigid parametric forms that often miss complex nonlinearities, whereas Neural ODEs are expressive yet largely black-box. Physics-Info...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SOLIS: Physics-Informed Learning of Interpretable Neural Surrogates for Nonlinear Systems

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统系统辨识（SysID）方法在**物理可解释性**与**模型灵活性**之间存在矛盾：
- 经典方法（如参数化ODE建模）具有良好的控制相关性和物理解释性，但依赖固定的刚性结构，难以捕捉复杂非线性。
- Neural ODE 和 PINN 类方法表达能力强，但多为“黑箱”，缺乏对自然频率、阻尼比等经典物理量的显式建模。
- 逆向 PINN（IPINN）假设全局恒定参数，当真实动力学为状态依赖时易出现**不可识别性（identifiability failure）** 和训练崩溃。

### 提出的新方法：SOLIS
提出 **SOLIS**（Second-Order Local Identification of Surrogates），一种基于物理信息学习的框架，用于构建**可解释的神经代理模型**（interpretable neural surrogates）。

#### 核心思想
将非线性系统的局部轨迹几何近似为一个**状态条件下的二阶质量-弹簧-阻尼器系统**，即构建一个 **Quasi-Linear Parameter-Varying (Quasi-LPV)** 表示：
$$
\ddot{y} + d(x)\dot{y} + k(x)y = g(x)u
$$
其中 $k(x), d(x), g(x)$ 是状态 $x=[y,\dot{y}]$ 的函数，分别对应**刚度、阻尼、增益**，并可映射到物理意义明确的量：
- 自然频率 $\omega_n(x) = \sqrt{k(x)}$
- 阻尼比 $\zeta(x) = d(x)/(2\sqrt{k(x)})$
- DC gain $K(x) = g(x)/k(x)$

### 创新点
1. ✅ **Quasi-LPV 替代模型公式化**  
   将系统辨识重构为学习状态相关的系数场，而非全局方程，提升对强非线性系统的适应能力。

2. ✅ **解-参数分离架构（Solution-Parameter Decomposition）**  
   采用双网络结构：
   - **Solution Network (Nsol)**：重建连续时间状态轨迹。
   - **Parameter Network (Nparam)**：输出状态相关的物理参数 $[k,d,g]$。
   解耦优化过程，避免联合非凸优化中的梯度冲突。

3. ✅ **课程学习策略 + Local Physics Hints**  
   引入循环课程训练（cyclic curriculum）：
   - **Phase 1**：固定 Nparam，仅训练 Nsol 进行轨迹拟合。
   - **Phase 2**：固定 Nsol，训练 Nparam，并引入 **Local Physics Hints**——通过滑动窗口岭回归提供局部解析估计作为正则化锚点，防止早期训练坍缩。

### 相比现有方法的优势
| 方法 | 局限性 | SOLIS 改进 |
|------|--------|-----------|
| IPINN | 假设全局常数参数，无法处理状态依赖动态 | 支持 state-varying 参数，增强表达力 |
| Black-box NODE | 缺乏物理解释性 | 输出可解释的 $\omega_n, \zeta, K$ 字段 |
| 传统SysID | 固定结构限制灵活性 | 数据驱动发现参数流形 |
| 普通PINN/IPINN | 训练不稳定、loss balancing困难 | 通过 curriculum 和 hints 稳定优化 |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **Duffing Oscillator**（仿真）
   - 动力学：$\ddot{y} + \delta \dot{y} + \alpha y + \beta y^3 = u$，体现立方刚度效应。
2. **Van der Pol Oscillator**（仿真）
   - 动力学：$\ddot{y} - \mu(1-y^2)\dot{y} + y = u$，具有自持振荡特性。
3. **Two-Tank System**（真实实验数据）
   - 双罐液位系统，存在非线性流体关系和测量噪声。

### 实验设置
- 输入信号 $u(t)$ 已知，观测数据稀疏且含噪。
- 多条轨迹训练（$J$ 条），每条有初始条件和输入变化。
- 使用密集的 collocation points $T_c$ 计算物理一致性损失。
- Solution Network 使用 GRU 上下文编码 + FiLM 调制以支持多轨迹建模。

### 评估指标
1. **Accuracy** = $(1 - \text{NRMSE}) \times 100\%$，NRMSE 归一化为信号峰峰值。
2. **Phase Portrait Similarity**：平均余弦相似度（cosine similarity）于 ground truth 向量场。
3. **Rollout Accuracy**：开环前向积分预测测试轨迹的准确性。

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **IPINN** | 标准逆PINN，假设全局常数参数 $(k,d,g)$ |
| **IPINN-M** | 多轨迹版本的IPINN |
| **TF** | 使用MATLAB `tfest` 函数估计的二阶传递函数 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables I–III）

#### 表 I：训练轨迹上的解重建精度（Accuracy %）
| System       | IPINN   | IPINN-M | **SOLIS** |
|--------------|---------|---------|-----------|
| Van der Pol  | 95.94%  | 98.07%  | **98.89%** |
| Duffing      | 95.43%  | 97.56%  | **99.01%** |
| Two-Tank     | 88.64%  | 91.25%  | **98.62%** |

> ✅ SOLIS 在所有系统上均取得最高重建精度，尤其在真实 Two-Tank 数据上优势显著。

#### 表 II：相图相似度（Average Cosine Similarity）
| System       | IPINN | IPINN-M | **SOLIS** |
|--------------|-------|---------|-----------|
| Van der Pol  | 0.70  | 0.67    | **0.86**  |
| Duffing      | 0.81  | 0.82    | **0.96**  |

> ✅ 显示 SOLIS 学习到的向量场更接近真实物理动态，具备更强的泛化能力。

#### 表 III：测试轨迹上的 rollout 预测精度（Accuracy %）
| System       | IPINN   | IPINN-M | TF     | **SOLIS** |
|--------------|---------|---------|--------|-----------|
| Van der Pol  | 83.72%  | 81.84%  | 70.22% | **90.54%** |
| Duffing      | 76.60%  | 75.27%  | 74.60% | **83.07%** |
| Two-Tank     | 82.74%  | 82.90%  | 77.74% | **84.29%** |

> ✅ 在未见测试轨迹上前向积分仍保持高精度，说明 learned surrogate 具备良好外推能力和稳定性。

### 消融实验（隐含分析）
虽然文中未明确列出消融表，但从设计中可推断以下关键组件的作用：
- **Local Physics Hints**：防止 Phase 2 中参数网络训练初期因梯度病态导致坍缩（如 $k\to0$）。
- **Cyclic Curriculum**：解耦优化路径，使两网络交替收敛，提升整体稳定性。
- **Ridge Regression Anchors**：提供 convex warm-start，缓解非凸优化难题。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **状态依赖的 Quasi-LPV 结构能有效逼近复杂非线性系统**，即使未知真实方程形式也能恢复物理一致的动力学。
2. 🧩 **解-参数分解 + Curriculum Learning 显著提升了训练稳定性和辨识准确性**，解决了 IPINN 中常见的优化失败问题。
3. ⚙️ **Local Physics Hints 是关键创新机制**，利用局部线性回归提供可靠初始化，引导网络学习全局一致的参数流形。
4. 📈 **SOLIS 不仅拟合好，还能生成物理合理的 rollout**，适用于控制设计（如 gain scheduling）和稳定性分析。

### 方法的局限性
1. 当前主要验证于**低维单输入单输出（SISO）系统**，扩展至 MIMO 或高维系统尚需研究。
2. 对**高频振荡或极端非光滑动态**可能需要更多特征工程或更高频编码（如 RFF）辅助。
3. 参数网络的学习仍依赖足够覆盖的状态空间采样，若训练轨迹分布狭窄，则外推区域可靠性下降。

### 未来工作方向
1. 扩展至 **multi-input multi-output (MIMO) systems**。
2. 应用于 **predictive control** 场景，利用 learned Quasi-LPV 模型进行实时增益调度控制。
3. 探索 **higher-dimensional nonlinear systems**，如柔性机械臂、气候模型等。
4. 结合 **uncertainty quantification** 方法（如贝叶斯NN）提升鲁棒性。

---

> ✅ 总结：SOLIS 成功弥合了**数据驱动灵活性**与**物理可解释性**之间的鸿沟，是面向控制的非线性系统辨识的一项重要进展。

</details>

---

### 16. [RL-STPA: Adapting System-Theoretic Hazard Analysis for Safety-Critical Reinforcement Learning](https://arxiv.org/abs/2604.15201)

**Authors**: Steven A. Senczyszyn, Timothy C. Havens, Nathaniel Rice, Jason E. Summers, Benjamin D. Werner, Benjamin J. Schumeg  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.15201v1  

#### Abstract
As reinforcement learning (RL) deployments expand into safety-critical domains, existing evaluation methods fail to systematically identify hazards arising from the black-box nature of neural network enabled policies and distributional shift between training and deployment. This paper introduces Rei...

---

### 17. [Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](https://arxiv.org/abs/2604.13488)

**Authors**: Ziwei Wang, Junjie Zheng, Leyang Yang, Sheng Zhou, Xiaoxuan Tang, Zhouhua Fang, Zhiwei Liu, Dajun Chen, Yong Li, Jiajun Bu  
**Category**: cs.AI  
**Published**: 2026-04-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13488v1  

#### Abstract
Autonomous Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs) enable digital automation on end-user devices. While scaling both parameters and data has yielded substantial gains, advanced methods still suffer from prohibitive deployment costs on resource-constr...

---

### 18. [Hierarchical Retrieval Augmented Generation for Adversarial Technique Annotation in Cyber Threat Intelligence Text](https://arxiv.org/abs/2604.14166)

**Authors**: Filippo Morbiato, Markus Keller, Priya Nair, Luca Romano  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.14166v1  

#### Abstract
Mapping Cyber Threat Intelligence (CTI) text to MITRE ATT\&amp;CK technique IDs is a critical task for understanding adversary behaviors and automating threat defense. While recent Retrieval-Augmented Generation (RAG) approaches have demonstrated promising capabilities in this domain, they fundament...

---

### 19. [Attention to Mamba: A Recipe for Cross-Architecture Distillation](https://arxiv.org/abs/2604.14191)

**Authors**: Abhinav Moudgil, Ningyuan Huang, Eeshan Gunesh Dhekane, Pau Rodr\'iguez, Luca Zappella, Federico Danieli  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.14191v1  

#### Abstract
State Space Models (SSMs) such as Mamba have become a popular alternative to Transformer models, due to their reduced memory consumption and higher throughput at generation compared to their Attention-based counterparts. On the other hand, the community has built up a considerable body of knowledge ...

---

### 20. [Explainable Graph Neural Networks for Interbank Contagion Surveillance: A Regulatory-Aligned Framework for the U.S. Banking Sector](https://arxiv.org/abs/2604.14232)

**Authors**: Mohammad Nasir Uddin  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.14232v1  

#### Abstract
The Spatial-Temporal Graph Attention Network (ST-GAT) framework was created to serve as an explainable GNN-based solution for detecting bank distress early warning signs and for conducting macro-prudential surveillance of the interbank system in the United States. The ST-GAT framework models 8,103 F...

---

### 21. [Awakening Dormant Experts:Counterfactual Routing to Mitigate MoE Hallucinations](https://arxiv.org/abs/2604.14246)

**Authors**: Wentao Hu, Yanbo Zhai, Xiaohui Hu, Mingkuan Zhao, Shanhong yu, Xue Liu, Kaidong Yu, Shuangyong Song, Xuelong Li  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.14246v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models have achieved remarkable scalability, yet they remain vulnerable to hallucinations, particularly when processing long-tail knowledge. We identify that this fragility stems from static Top-$k$ routing: routers tend to favor high-frequency patterns over rare fact...

---

### 22. [Mean Flow Policy Optimization](https://arxiv.org/abs/2604.14698)

**Authors**: Xiaoyi Dong, Xi Sheryl Zhang, Jian Cheng  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.14698v1  

#### Abstract
Diffusion models have recently emerged as expressive policy representations for online reinforcement learning (RL). However, their iterative generative processes introduce substantial training and inference overhead. To overcome this limitation, we propose to represent policies using MeanFlow models...

---

### 23. [GeoAgentBench: A Dynamic Execution Benchmark for Tool-Augmented Agents in Spatial Analysis](https://arxiv.org/abs/2604.13888)

**Authors**: Bo Yu, Cheng Yang, Dongyang Hou, Chengfu Liu, Jiayao Liu, Chi Wang, Zhiming Zhang, Haifeng Li, Wentao Yang  
**Category**: cs.AI  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.13888v1  

#### Abstract
The integration of Large Language Models (LLMs) into Geographic Information Systems (GIS) marks a paradigm shift toward autonomous spatial analysis. However, evaluating these LLM-based agents remains challenging due to the complex, multi-step nature of geospatial workflows. Existing benchmarks prima...

---

### 24. [Reward Design for Physical Reasoning in Vision-Language Models](https://arxiv.org/abs/2604.13993)

**Authors**: Derek Lilienthal, Manisha Mukherjee, Sameera Horawalavithana  
**Category**: cs.AI  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.13993v1  

#### Abstract
Physical reasoning over visual inputs demands tight integration of visual perception, domain knowledge, and multi-step symbolic inference. Yet even state-of-the-art Vision Language Models (VLMs) fall far short of human performance on physics benchmarks. While post-training algorithms such as Supervi...

---

### 25. [The PICCO Framework for Large Language Model Prompting: A Taxonomy and Reference Architecture for Prompt Structure](https://arxiv.org/abs/2604.14197)

**Authors**: David A. Cook  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.14197v1  

#### Abstract
Large language model (LLM) performance depends heavily on prompt design, yet prompt construction is often described and applied inconsistently. Our purpose was to derive a reference framework for structuring LLM prompts. This paper presents PICCO, a framework derived through a rigorous synthesis of ...

---

### 26. [CoCoDiff: Optimizing Collective Communications for Distributed Diffusion Transformer Inference Under Ulysses Sequence Parallelism](https://arxiv.org/abs/2604.14561)

**Authors**: Bin Ma, Xingjian Ding, Tekin Bicer, Pengfei Su, Dong Li  
**Category**: cs.DC  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.14561v1  

#### Abstract
Diffusion Transformers (DiTs) are increasingly adopted in scientific computing, yet growing model sizes and resolutions make distributed multi-GPU inference essential. Ulysses sequence parallelism scales DiT inference but introduces frequent all-to-all collectives that dominate latency. Overlapping ...

---

### 27. [Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](https://arxiv.org/abs/2604.14267)

**Authors**: Junzhe Wang, Zhiheng Xi, yajie yang, Hao Luo, Shihan Dou, Tao Gui, Qi Zhang  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.14267v1  

#### Abstract
Search agents extend Large Language Models (LLMs) beyond static parametric knowledge by enabling access to up-to-date and long-tail information unavailable during pretraining. While reinforcement learning has been widely adopted for training such agents, existing approaches face key limitations: pro...

---

### 28. [Non-intrusive Learning of Physics-Informed Spatio-temporal Surrogate for Accelerating Design](https://arxiv.org/abs/2604.14424)

**Authors**: Sudeepta Mondal, Soumalya Sarkar  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.14424v1  

#### Abstract
Most practical engineering design problems involve nonlinear spatio-temporal dynamical systems. Multi-physics simulations are often performed to capture the fine spatio-temporal scales which govern the evolution of these systems. However, these simulations are often high-fidelity in nature, and can ...

---

### 29. [Quantization of Spiking Neural Networks Beyond Accuracy](https://arxiv.org/abs/2604.14487)

**Authors**: Evan Gibson Smith, Jacob Whitehill, Fatemeh Ganji  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.14487v1  

#### Abstract
Quantization is a natural complement to the sparse, event-driven computation of Spiking Neural Networks, reducing memory bandwidth and arithmetic cost for deployment on resource-constrained hardware. However, existing SNN quantization evaluation focuses almost exclusively on accuracy, overlooking wh...

---

### 30. [Constraint-based Pre-training: From Structured Constraints to Scalable Model Initialization](https://arxiv.org/abs/2604.14769)

**Authors**: Fu Feng, Yucheng Xie, Ruixiao Shi, Jing Wang, Xin Geng  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.14769v1  

#### Abstract
The pre-training and fine-tuning paradigm has become the dominant approach for model adaptation. However, conventional pre-training typically yields models at a fixed scale, whereas practical deployment often requires models of varying sizes, exposing its limitations when target model scales differ ...

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
