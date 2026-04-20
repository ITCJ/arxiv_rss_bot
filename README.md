# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-20 07:36:56 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Breaking the Training Barrier of Billion-Parameter Universal Machine Learning Interatomic Potentials](https://arxiv.org/abs/2604.15821)

**Authors**: Yuanchang Zhou, Hongyu Wang, Yiming Du, Yan Wang, Mingzhen Li, Siyu Hu, Xiangyu Zhang, Weijian Liu, Chen Wang, Zhuoqiang Guo, Long Wang, Jingde Bu, Yutong Lu, Guangming Tan, Weile Jia  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.15821v1  

#### Abstract
Universal Machine Learning Interatomic Potentials (uMLIPs), pre-trained on massively diverse datasets encompassing inorganic materials and organic molecules across the entire periodic table, serve as foundational models for quantum-accurate physical simulations. However, uMLIP training requires seco...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Breaking the Training Barrier of Billion-Parameter Universal Machine Learning Interatomic Potentials

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文致力于解决 **billion-parameter 通用机器学习原子间势能模型（uMLIPs）训练中的可扩展性瓶颈**。当前 uMLIPs 面临三大挑战：
- **高计算成本**：训练依赖于二阶导数（double-backward），导致计算和内存开销翻倍；
- **精度要求高**：必须使用 FP32 单精度以保证量子级精度，限制了 Tensor Core 利用率；
- **通信与并行框架缺失**：缺乏支持大规模、高维图结构和 MoE 架构的分布式训练框架。

这些问题使得训练十亿参数级别的 uMLIPs 在传统系统上变得极其低效甚至不可行。

---

### 提出的新方法与新思路

#### （1）**MatRIS-MoE：首个基于不变架构的十亿参数 MoE 模型**
- 在原有高效不变模型 MatRIS 基础上引入 **Sparse Mixture-of-Experts (MoE)** 架构，实现跨异质化学域（分子、晶体、催化界面等）的多任务学习。
- 引入 **task-aware 特征嵌入**，统一不同 DFT 泛函下的数据表示。
- 将原始的 separable attention 替换为 **multi-head self-attention**，提升表达能力，并更好地适配现代 GPU 的密集矩阵运算。

#### （2）**Janus：首个面向 uMLIPs 的高维分布式训练框架**
- 提出 **FS-3D（Fully Sharded 3 Dimensions）执行单元**，融合三种并行策略：
  - **FSDP**（Fully Sharded Data Parallelism）
  - **FSGP**（Fully Sharded Graph Parallelism）
  - **FSEP**（Fully Sharded Expert Parallelism）
- 支持对 **double-backward 自动微分流程** 和 **MoE 层稀疏路由** 的细粒度分片与调度。

#### （3）硬件感知优化
- **Just-in-time 稀疏专家规划机制**：动态负载均衡，避免冗余参数加载。
- **Atom-type-aware FP16 通信压缩**：按元素类型进行量化，降低 All-to-All 通信量达 50%。
- **SDMA 引擎驱动的 HBM 内存优化**（LineShine 平台）：通过专用 DMA 引擎隐藏 DDR-HBM 数据传输延迟，带宽提升最高达 1.4×。

---

### 相比现有方法的优势

| 维度 | 本文方法优势 |
|------|---------------|
| **模型容量** | 实现 **11.5B 参数** 的 uMLIP 模型训练，远超此前 SOTA（如 UMA: 1.4B） |
| **训练效率** | 达到 **1.2 EFLOPS 单精度峰值性能**，维持 >90% 并行效率，将训练时间从“周”压缩至“小时”级 |
| **吞吐量** | 归一化吞吐量相比当前最优（UMA）提升 **653–3201×** |
| **通用性与泛化能力** | 支持跨分子、材料、MOFs、催化剂等多领域零样本预测，准确率接近 DFT 水平 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 构建了一个包含 **473 million 原子构型** 的大规模多域数据集，涵盖：
  - 孤立分子（isolated molecules）
  - 周期性晶体（periodic crystals）
  - 催化表面（catalytic surfaces）
  - 分子晶体（molecular crystals）
  - 金属有机框架（MOFs）
- 数据来源包括：OMat24、OMol25、Opoly26、OC25、ODAC25 等开放数据库。

---

### 实验设置与评估指标

#### 模型配置
训练了三个 MatRIS-MoE 变体：

| 模型 | 总参数量 | 激活参数量 | MoE Top-k 路由 |
|------|----------|------------|----------------|
| (S) Small | 1.09B | 0.19B | 4 |
| (M) Medium | 2.47B | 0.56B | 8 |
| (L) Large | 11.5B | 2.89B | 16 |

所有模型均采用 6 层 Interaction Block，含 multi-head self-attention 和 MoE 模块。

---

#### HPC 系统平台
在两台 Exascale 超算上完成实验：

| 系统 | CNIS（中国新一代智能超算） | LineShine（深圳国家超算中心） |
|------|----------------------------|------------------------------|
| 节点数 | 5,632 | 20,480 |
| 加速器 | GPGPU（每节点8卡） | ARMv9-based LX2 CPU（每节点2颗） |
| 内存 | 64GB HBM2e/GPU | 每 DIE 配备 4GB HBM + 128GB DDR |
| 网络 | Proprietary InfiniBand-like RDMA | LingQi 多轨胖树网络（1.6 Tb/s/node） |

---

#### 评估指标
- **Peak Performance**: `总 FLOPs / 训练循环耗时`（FP32）
- **Sustained Performance**: 包含 IO、初始化等端到端时间
- **Normalized Throughput**:  
  $$
  \text{Throughput} = \frac{\text{Active Params} \times \text{Dataset Size} \times \text{Epochs}}{\text{Training Days}}
  $$
  以 UMA 模型为基准归一化为 1.0
- **Parallel Efficiency**: 实际性能 / 理论峰值比例
- **Accuracy Benchmarks**: 在 Matcalc、Wiggle150、GMTKN55、X23、OC20NEB、MOFSim 等多个下游任务中测试能量、力、应力等预测误差。

---

### 基线方法对比
| 方法 | 类别 | 参数规模 | 是否支持多任务 | 硬件需求 | 归一化吞吐量（UMA=1） |
|------|------|-----------|----------------|-----------|------------------------|
| CHGNet [9] | Invariant | 0.41M | 否 | 1×A100 | 0.0022 |
| eqV2 [24] | Equivariant | 86M | 否 | 64×A100 | 3.09 |
| PET [22] | Unconstrained | 730M | 否 | 512×H100 | — |
| UMA [7] | Equivariant | 1.4B | 是 | 256×H200 | 1.00 |
| **This Work (L)** | **Invariant + MoE** | **11.5B** | **是** | **45K GPGPU / 12.4M ARMv9 cores** | **2795.9 – 3201.8** |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）**峰值性能与并行效率**
- 在 **LineShine** 上达到 **1.2 EFLOPS（FP32）** 峰值性能，占理论峰值 **24.41%**
- 在 **CNIS** 上达到 **1.0 EFLOPS（FP32）**，占理论峰值 **35.52%**
- **弱扩展下并行效率超过 90%**

#### （2）**归一化吞吐量**
- 对比 UMA 模型（归一化为 1.0）：
  - MatRIS-MoE (M): **653–749×**
  - MatRIS-MoE (L): **2795–3201×**

> 这意味着相同时间内可处理数千倍更多的训练样本。

#### （3）**强扩展性能**
- 固定全局 batch size 下扩展至全系统：
  - MatRIS-MoE (L) 在 CNIS 上效率达 **53.93%**，LineShine 上 **50.60%**
  - 吞吐量从 ~0.8M edges/sec 扩展至 ~1.77M edges/sec

#### （4）**弱扩展性能**
- 成近线性扩展：
  - CNIS 上效率 **93.78%**（L 模型）
  - LineShine 上效率 **90.3%**（L 模型）
- Sustained Efficiency（考虑初始化与 IO）仍达 **72.73%（CNIS）** 和 **86.61%（LineShine）**

#### （5）**单步训练加速**
| 平台 | 模型 | 基线时间 | 优化后时间 | 加速比 |
|------|------|----------|------------|--------|
| CNIS | M | 5.96s | 2.21s | **2.7×** |
| CNIS | L | 7.71s | 2.66s | **2.9×** |
| LineShine | M | 33.1s | 8.08s | **4.1×** |
| LineShine | L | 40.7s | 8.14s | **5.0×** |

> 显著得益于通信重叠、MoE 压缩与 SDMA 内存优化。

---

### 消融实验结果（隐含分析）

虽然未明确列出消融表，但从性能分解可见各优化贡献：
- **异步梯度同步 + 参数更新流水线**：减少双反向传播阻塞
- **Atom-type-aware FP16 压缩**：MoE All-to-All 通信体积减少 50%
- **高性能核函数优化**（neighbor gather, attention, MoE dispatch）：提升单设备吞吐
- **SDMA 数据搬运优化**（LineShine）：内存带宽提升 1.4×，是其更高加速比的关键

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **十亿参数 uMLIPs 的训练壁垒已被打破**：
   - 通过 **算法-框架-系统协同设计**，首次实现了 11.5B 参数 uMLIP 的高效训练。
   - 达到 **Exascale 级别的持续性能（>1 EFLOPS）**，且保持高并行效率。

2. ✅ **不变模型（Invariant）也能胜任超高精度模拟**：
   - 尽管传统认为 equivariant 模型更优，但通过 MoE + self-attention + 大规模数据，MatRIS-MoE 在多种任务上达到甚至超越 equivariant 模型的精度。

3. ✅ **MoE 架构适合多任务原子建模**：
   - 不同专家可专注于特定元素或化学环境，显著增强表达能力。
   - 推理时仅激活 Top-K 专家，实现“大模型、小推理”。

4. ✅ **Janus 框架具有高度可移植性**：
   - 在 GPGPU 和 ARMv9 多核平台上均表现出优异扩展性，验证了其跨架构通用性。

---

### 方法的局限性

1. ❗ **严重依赖 Exascale 级硬件资源**：
   - 最大模型需 45K GPGPU 或 12.4M ARMv9 核心，普通实验室难以复现。

2. ❗ **初始化与 IO 开销显著影响 sustained performance**：
   - 初始化时间长达数百秒，在小规模运行中占比过高。

3. ❗ **MoE 动态负载均衡仍有挑战**：
   - 尽管 JIT 规划有效，但在极端不均匀任务分布下可能仍存在热点问题。

4. ❗ **目前仅支持静态图划分**：
   - 图分区在训练前确定，无法适应动态变化的局部密度。

---

### 未来工作方向

1. 🔮 **进一步扩展至 100B+ 参数模型**
   - 探索更高效的 MoE 路由机制与专家共享策略。

2. 🔮 **开发轻量化版本用于桌面级部署**
   - 结合知识蒸馏或参数剪枝技术，使 MatRIS-MoE 可在消费级 GPU 上运行。

3. 🔮 **构建闭环 AI-driven 科学发现流程**
   - 将 MatRIS-MoE 与 DFT solver、MD engine 耦合，形成自动化的材料筛选-模拟-验证 pipeline。

4. 🔮 **推动 HPC-AI 融合编程框架标准化**
   - 提出统一接口支持 GNN、MoE、double-backward 等科学 AI 工作负载，促进跨平台迁移。

---

> **总结一句话**：  
> 本论文通过 **MatRIS-MoE + Janus + Exascale 协同优化**，首次实现了 **EFLOPS 级、十亿参数 uMLIP 的高效训练**，标志着 AI for Science 在原子尺度模拟领域迈入真正的“基础模型时代”。

</details>

---

### 2. [CroSatFL: Energy-Efficient Federated Learning with Cross-Aggregation for Satellite Edge Computing](https://arxiv.org/abs/2604.15779)

**Authors**: Nan Yang, Bahman Javadi, Rodrigo Neves Calheiros, David Boland, Philip Leong  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.15779v1  

#### Abstract
Low Earth Orbit (LEO) mega-constellations extend the cloud-to-edge continuum into space, enabling satellite edge computing. However, Federated Learning (FL) in this environment is fundamentally energy-constrained due to dynamic inter-satellite connectivity, heterogeneous onboard computing hardware, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CroSatFL: Energy-Efficient Federated Learning with Cross-Aggregation for Satellite Edge Computing

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在 **Low Earth Orbit (LEO)** 卫星 mega-constellations（如 Starlink）中，**Federated Learning (FL)** 面临以下关键挑战：

- **能源受限**：卫星依赖太阳能，功耗预算严格。
- **通信瓶颈**：地面站（Ground Station, GS）可见窗口短、带宽有限、传输能耗高。
- **动态连接性**：激光星间链路（Laser Inter-Satellite Links, LISLs）受轨道运动影响，拓扑时变。
- **硬件异构性**：卫星搭载 CPU 或 GPU 类型不一，计算能力差异大，导致训练同步延迟（straggler problem）。

传统 FL 方法（如 FedSyn、FedLEO）通常依赖频繁的 GS 通信进行全局聚合，导致训练过程受制于地面站可见性，效率低下且能耗高。

---

### **提出了什么新方法或新思路**

作者提出 **CroSatFL** —— 一种**完全在轨运行的分层联邦学习框架**，其核心创新包括三个机制：

#### ✅ **1. 完全在轨训练（On-Orbit-Only Training）**
- 所有本地训练和中间模型聚合均在卫星上完成。
- 地面站仅用于**初始化模型广播**和**最终模型收集**，共两次通信。
- 彻底避免将 GS 通信置于训练迭代的关键路径上。

#### ✅ **2. StarMask：基于强化学习的资源感知聚类**
- 利用 **Reinforcement Learning (RL)** 动态形成 LISL 可达、计算能力均衡的卫星集群。
- 考虑因素：数据量、硬件类型（CPU/GPU）、每轮计算时间、能量消耗、LISL 扇出限制。
- 引入 **Action Masking** 确保聚类满足物理约束（如连接性、容量），提升可行性。

#### ✅ **3. Skip-One：轻量级同步优化机制**
- 允许每个集群在每轮训练中最多跳过一个“临时拖后腿”的卫星（straggler）。
- 减少等待时间，降低能耗，同时通过冷却计数器（cooldown）保证长期公平性。

#### ✅ **4. Random-k Cross-Aggregation：拓扑感知的跨集群混合**
- 在边缘轮次中，各集群主节点利用瞬时可达的跨轨道 LISL，随机选择 `k` 个邻居进行模型混合。
- 实现全局一致性，无需全连通图或额外同步开销。

---

### **相比现有方法的优势**

| 维度 | CroSatFL 优势 |
|------|---------------|
| **通信效率** | GS 通信次数减少 **两个数量级以上**（从 ~3200 次降至 18 次） |
| **能量效率** | GS 传输能耗降低约 **6×**，总训练能耗显著下降 |
| **训练速度** | 端到端训练时间大幅缩短，等待时间从数百小时降至 <8 小时 |
| **可扩展性** | 支持大规模 LEO 星座下的可持续在轨学习 |
| **鲁棒性** | 对非独立同分布（non-IID）数据和硬件异构性具有更强适应性 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **MNIST**：手写数字图像分类（IID 与 non-IID 设置）
- **CIFAR-10**：自然图像分类
- **EuroSAT**：遥感图像土地利用分类（更贴近卫星应用场景）

所有任务使用 **ResNet-18** 作为基础模型。

---

### **实验设置**

| 参数 | 值 |
|------|----|
| 卫星星座 | Walker-Delta 构型（类 Starlink） |
| 卫星总数 | 720 颗 |
| 轨道平面数 | 36 |
| 每轨道卫星数 | 20 |
| 高度 | 570 km |
| 倾角 | 70° |
| 地面站位置 | 澳大利亚堪培拉 |
| LISL 范围 | 659–1700 km（对应最大集群大小 2–10） |
| FL 主轮数（main rounds） | 1 |
| FL 边缘轮数（edge rounds） | 40 |
| 本地训练轮数（local epochs） | 10 |
| 客户端数量 | 随机选取 40 颗卫星参与 FL |
| 聚类数量 | 9 个集群，由 StarMask 生成 |

---

### **评估指标**

- **模型准确率（Accuracy）**：收敛速度与最终性能
- **总能耗（Total Energy）**：包括计算与通信能耗
- **端到端训练时间（End-to-End Training Time）**
- **GS 通信次数与能耗**
- **等待时间（Waiting Time）**：因无通信机会而空转的时间
- **LISL 使用统计**

---

### **基线方法对比**

| 基线方法 | 特点 |
|---------|------|
| **FedSyn** | 标准同步 FedAvg，频繁依赖 GS 聚合 |
| **FedLEO** | 针对 LEO 优化，但仍需多次 GS 交互 |
| **FELLO** | 使用光学 LISL 聚类，减少通信，但忽略异构性 |
| **FedSCS** | 能量感知客户端选择 |
| **FedOrbit** | 使用 block minifloat 算术降低计算开销，考虑硬件异构性 |

> 所有基线均按原设计实现，未强制采用 CroSatFL 的 on-orbit-only 通信模式。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔹 **模型准确性（Accuracy）**

- 在 **IID 设置下**：
  - CroSatFL 在 **CIFAR-10 上达到 80.09%**，优于所有基线。
  - 在 **EuroSAT 上达到 89.49%**，表现最佳或并列最优。
- 在 **non-IID 设置下（α=0.5）**：
  - 各基线性能普遍下降，但 CroSatFL 保持稳定收敛，接近最强方法。
  - 表明其对数据异构性具有较强鲁棒性。

> ➤ 图 2 和图 3 显示 CroSatFL 收敛更快且更平稳。

---

#### 🔹 **能耗与训练时间（图 4）**

| 指标 | CroSatFL 表现 |
|------|----------------|
| **总能耗** | 在所有数据集上均为最低 |
| **训练时间** | 显著短于其他方法（尤其在复杂数据集上） |
| **GS 通信次数** | 从 FedSyn 的 3200 次降至 **18 次**（↓ > 2 个数量级） |
| **GS 传输能耗** | 从 601.60 kJ（FedSyn）降至 **99.70 kJ**（↓ ~6×） |
| **等待时间** | 从 FedSyn 的 936 小时降至 **7.89 小时** |

> ✅ 表明 CroSatFL 极大缓解了 GS 成为瓶颈的问题。

---

#### 🔹 **LISL 使用情况（Table II）**

| 项目 | CroSatFL |
|------|----------|
| **Intra-cluster LISLs** | 1760 次 |
| **Inter-cluster LISLs** | 1440 次（引入跨集群通信） |
| **GS Communication** | 仅 18 次 |

> ➤ 说明 CroSatFL 成功将通信负载从昂贵的 GS 链路转移到高效的 LISL 上。

---

#### 🔹 **硬件异构性影响（图 5）**

- 在 **Half-Mixed（50% CPU + 50% GPU）** 设置下：
  - CroSatFL 比 FedOrbit 更好地利用 GPU 资源。
  - 通过 Skip-One 机制规避慢速 CPU 卫星，降低单轮时间和能耗。
- 随着 GPU 比例上升，CroSatFL 性能增益更加明显。

> ➤ 验证了 Skip-One 在异构环境中的有效性。

---

#### 🔹 **消融实验（Ablation Study）**

虽然文中未明确列出独立的消融表格，但从机制设计和结果分析可推断：

- **移除 Skip-One** → 同步延迟增加，等待时间上升。
- **移除 Random-k Cross-Aggregation** → 全局一致性下降，收敛变慢。
- **使用静态聚类代替 StarMask** → 资源不平衡，部分集群成为瓶颈。

> ➤ 三大组件协同作用，共同实现高效稳定的在轨 FL。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **完全在轨的 FL 是可行且高效的**：通过消除对 GS 的频繁依赖，CroSatFL 显著提升了训练效率和能源利用率。
2. ✅ **LISL 是构建空间智能网络的关键基础设施**：应充分利用其低延迟、高带宽特性进行模型交换。
3. ✅ **必须联合优化计算、通信与调度**：单一维度优化（如压缩、客户端选择）不足以应对 LEO 复杂环境。
4. ✅ **硬件异构性和动态拓扑需被显式建模**：StarMask 和 Skip-One 成功应对了这些现实挑战。

---

### **方法的局限性**

- 当前实验假设 LISL 连接是可靠的，未考虑链路中断或误码率。
- Skip-One 仅允许跳过一个客户端，可能不足以应对多个 stragglers 的场景。
- StarMask 的 RL 模型训练成本较高，可能不适合实时快速重构。
- 实验基于模拟轨迹，尚未在真实卫星平台上验证。

---

### **未来工作方向**

1. **引入自适应压缩机制**：结合 FedOrbit 的思想，在通信中进一步节省带宽与能量。
2. **容错与故障恢复机制**：处理卫星失效、链路断开等异常情况。
3. **支持更多类型的 onboard accelerators**：如 FPGA、NPU，拓展至更丰富的 AI 推理场景。
4. **多任务或多模型协同学习**：探索在轨多任务联邦学习的可能性。
5. **更大规模星座与更复杂轨道动力学验证**。

---

## 总结

CroSatFL 是首个将 **完全在轨分层联邦学习** 与 **能量效率、硬件异构性、动态 LISL 拓扑** 深度结合的系统性解决方案。它通过 **StarMask**、**Skip-One** 和 **Random-k Cross-Aggregation** 三大机制，在不牺牲模型性能的前提下，实现了：

- GS 通信减少 **>100 倍**
- GS 传输能耗降低 **~6×**
- 等待时间从 **百小时级降至小时级**

为未来大规模 **Satellite Edge Computing** 与 **Space-AI** 的发展提供了重要技术路径。

</details>

---

### 3. [PINNACLE: An Open-Source Computational Framework for Classical and Quantum PINNs](https://arxiv.org/abs/2604.15645)

**Authors**: Shimon Pisnoy, Hemanth Chandravamsi, Ziv Chen, Aaron Goldgewert, Gal Shaviner, Boris Shragner, Steven H. Frankel  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.15645v1  

#### Abstract
We present PINNACLE, an open-source computational framework for physics-informed neural networks (PINNs) that integrates modern training strategies, multi-GPU acceleration, and hybrid quantum-classical architectures within a unified modular workflow. The framework enables systematic evaluation of PI...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PINNACLE: An Open-Source Computational Framework for Classical and Quantum PINNs

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文旨在解决 **Physics-Informed Neural Networks (PINNs)** 在实际应用中的以下核心挑战：

- **训练困难与收敛不稳定**：PINNs 的优化过程常因非凸损失景观、梯度不平衡、频谱偏置（spectral bias）等问题而难以收敛。
- **计算成本高昂**：相比传统数值求解器，PINNs 需要大量自动微分和参数更新，导致训练时间长、资源消耗大。
- **缺乏统一、可扩展的开源框架**：现有工具（如 DeepXDE、Modulus）在支持量子-经典混合架构、多GPU加速和系统性消融研究方面存在不足。

### **提出了什么新方法或新思路**

作者提出 **PINNACLE** ——一个**模块化、开源的计算框架**，集成了一系列增强训练性能的技术，并首次将 **hybrid quantum-classical PINNs (QPINNs)** 与多GPU并行训练统一于同一平台。

#### 主要创新点包括：

- **统一的模块化架构**：支持从 vanilla PINNs 到复杂 QPINNs 的渐进式开发，便于复现和比较不同技术组合的效果。
- **全面集成先进训练策略**：
  - **Random Fourier Features (RFF)** 和 **Periodic Activation Functions**：缓解 spectral bias，提升高频特征捕捉能力。
  - **Random Weight Factorization (RWF)**：改善初始化，平衡激活与梯度分布。
  - **Strict Boundary Conditions**：通过网络结构设计强制满足周期性边界条件，消除边界损失项。
  - **Dynamic Loss Balancing**：基于梯度范数动态调整 PDE、初始和边界损失权重，防止梯度竞争。
  - **Curriculum Training**：逐步增加雷诺数等物理参数，引导模型稳定收敛至正确解。
  - **Adam + L-BFGS 两阶段优化**：先用 Adam 探索，再用 L-BFGS 精细收敛。
- **支持 Hybrid Quantum-Classical PINNs (QPINNs)**：
  - 将 Parametrized Quantum Circuit (PQC) 作为神经网络层嵌入经典网络。
  - 引入 **能量守恒正则项（energy regularization）** 以避免“黑洞”（black-hole）退化解。
- **多GPU分布式训练支持**：
  - 基于 PyTorch 的 **Distributed Data Parallel (DDP)** 实现大规模并行训练。
  - 提供完整的多GPU实现教程。

### **相比现有方法的优势**

| 方面 | PINNACLE 的优势 |
|------|----------------|
| **功能完整性** | 同时支持 classical PINNs、QPINNs、multi-GPU、多种优化策略，是目前最全面的开源框架之一。 |
| **可复现性与透明度** | 提供详细的模块化代码示例和消融实验，便于研究者理解各组件作用。 |
| **性能提升** | 在多个基准问题上实现了比文献中更高精度的解，尤其在高Re流场和电磁波传播中表现优异。 |
| **量子集成** | 是少数公开支持 QPINNs 并提供复杂度分析的框架，推动量子机器学习在科学计算中的探索。 |

---

## 2. 核心实验方法和设置

### **使用的数据集 / 测试问题**

论文并未使用传统意义上的“数据集”，而是选取了多个具有代表性的 **PDE 基准问题**，涵盖不同物理机制和数值挑战：

| 问题 | 类型 | 特点 |
|------|------|------|
| **Advection Equation** | 双曲型 | 线性平流，测试相位保持与周期性边界处理 |
| **Allen-Cahn Equation** | 反应扩散 | 多尺度界面演化，测试非线性稳定性 |
| **Inviscid Burgers Equation** | 双曲型 | 激波形成，测试间断捕捉能力 |
| **Lid-Driven Cavity** | 不可压缩Navier-Stokes | 耦合速度-压力场，测试高Re流动 |
| **2D Blood Flow in a Stenosis** | 医学流体力学 | 稀疏观测下的壁面剪切应力（WSS）重建 |
| **Sod Shock Tube** | Euler方程 | 一维激波管，测试非线性守恒律求解 |
| **2D Riemann Problem** | Euler方程 | 二维多波相互作用，测试复杂波系解析 |
| **Maxwell’s Equations (2D Gaussian Pulse)** | 电磁波 | 高频振荡传播，测试长时间积分稳定性 |

### **实验设置和评估指标**

#### **通用设置**
- 使用 **PyTorch** 实现，支持 CPU/GPU/A100/A6000/L40S 等硬件。
- 网络结构：MLP，层数与宽度依任务调整（如 5×128）。
- 优化器：Adam (lr=1e-3) + L-BFGS 微调。
- Collocation Points：数量从数千到数十万不等，部分采用 Latin Hypercube Sampling (LHS)。

#### **评估指标**
- **Relative L2 Error**：预测解与参考解之间的相对误差。
- **Mean Absolute Error (MAE)** 和 **Mean Relative Error (MRE)**：用于评估壁面剪切应力（WSS）。
- **Wall-clock Time** 和 **VRAM Usage**：评估计算效率与内存占用。
- **Loss History**：观察 PDE、IC、BC 损失的演化趋势。
- **中心线速度剖面图**：与 DNS 数据（如 Ghia et al.）对比。

### **基线方法对比**

- **Classical Baselines**：
  - 文献中的标准 PINNs 结果（如 Wang et al. [10], [56]）
  - 高分辨率有限差分法（FDM）、有限体积法（FVM）作为真值参考
- **Quantum Baselines**：
  - PennyLane 默认模拟器 `.default.qubit` 和 `lightning.qubit`
  - 自研量子模拟库 **TorQ** 进行对比

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 任务 | 方法 | 相对 L2 错误 | 备注 |
|------|------|---------------|------|
| **Advection (c=80)** | PINNACLE (RFF + Loss Balancing + Strict BC) | ~1.15×10⁻⁵ | 比 Wang et al. 低近两个数量级 |
| **Allen-Cahn** | PINNACLE (Uniform Grid) | 2.58×10⁻³ | 优于 LHS 采样 |
| **Lid-Driven Cavity (Re=3200)** | PINNACLE (Curriculum + LHS + 256×256) | — | 中心线速度与 DNS 高度一致，优于 Wang et al. [10] |
| **Stenosis Flow (WSS)** | PINNACLE (Swish) | MAE ≈ 3.58×10⁻², MRE ≈ 28.5% | 仅用 5 个内部观测点即可重建 WSS 趋势 |
| **Sod Shock Tube** | PINNACLE (RFF + Adam→L-BFGS) | — | 准确捕捉激波、接触间断和稀疏波，无震荡 |
| **Maxwell (2D Pulse)** | 完整配置（RFF + Periodicity + Causality） | 0.04218 (128×128) | 明显优于任一组件缺失版本 |

### **与基线方法的对比结果**

- **在 Advection 问题上**，PINNACLE 的误差比 Wang et al. 报告的结果低约 **100倍**。
- **在 Lid-Driven Cavity 上**，对于 Re=3200，本文方法在近壁区的速度预测更接近 DNS 数据，而 Wang et al. 的结果出现明显偏差。
- **在 Maxwell 方程求解中**，移除 RFF 导致误差上升超过 **4倍**，证明其对高频电磁波建模至关重要。
- **在 QPINNs 实验中**，最佳配置相比经典 PINN 实现了约 **19% 的误差降低**，且可减少约 **19% 的可学习参数**。

### **消融实验结果**

论文进行了系统的消融研究，验证各组件的有效性：

| 组件 | 移除后影响 |
|------|----------|
| **RFF** | Allen-Cahn 和 Maxwell 问题误差显著上升，无法捕捉高频细节 |
| **Strict Periodic BC** | 对 Advection 和 Maxwell 至关重要，否则解会衰减为零 |
| **Loss Balancing** | 若不使用，边界或 PDE 损失可能主导训练，导致欠拟合 |
| **Curriculum Training** | 对高 Re 流动至关重要，直接训练易陷入错误解分支 |
| **Energy Regularization (QPINNs)** | 缺失时多数运行坍缩为“黑洞”解；加入后可稳定训练并超越经典模型 |
| **Temporal Causality** | 在 Burgers 和 Maxwell 中未见显著改进，可能因问题本身较简单 |

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **成功训练 PINNs 需要多种技术协同作用**：单一方法不足以保证收敛，必须结合 RFF/RWF、loss balancing、curriculum learning 和两阶段优化。
2. ✅ **RFF 和 Strict BC 是提升精度的关键**：前者缓解 spectral bias，后者简化优化目标，二者对高频或多尺度问题不可或缺。
3. ✅ **QPINNs 具备潜力但代价高昂**：
   - 在特定问题（如 Maxwell）上可实现更高的参数效率和更低误差。
   - 但需引入额外正则项（如能量守恒）才能避免失败模式。
   - **电路评估复杂度呈指数增长**（见下文），限制其实用性。
4. ✅ **多GPU DDP 可有效扩展训练规模**：
   - 支持更大 collocation point 数量，突破单卡内存限制。
   - 在 1–4 GPU 范围内接近线性加速，但通信开销随 GPU 数增加而饱和。

### **方法的局限性**

| 局限性 | 说明 |
|--------|------|
| **计算成本远高于传统求解器** | 即使使用 GPU 加速，达到同等精度所需 FLOPs 仍远超 FDM/FVM。 |
| **QPINNs 的电路评估复杂度极高** | 基于 parameter-shift rule，每步更新需 $ O(P \cdot 2^K \cdot Q) $ 次电路执行（$P$: 参数数, $K$: 导数阶数, $Q$: 输入维度）。例如 Maxwell 示例中单点需 **2535 次电路运行**。 |
| **缺乏严格的误差界保证** | 与经典方法不同，PINNs 的误差是非确定性的，依赖随机初始化和超参选择。 |
| **Temporal Causality 效果有限** | 在所测试问题中未能显著提升性能，可能需要更复杂的因果建模。 |
| **当前 QPINN 模拟为理想态向量仿真** | 不支持噪声模型或 shot-based sampling，离真实量子硬件仍有距离。 |

### **未来工作方向**

1. **发展更高效的量子梯度估计方法**：探索替代 parameter-shift 的技术（如 implicit differentiation、quantum backpropagation）以降低电路评估次数。
2. **探索其他并行策略**：尝试 **Tensor Parallelism** 或 **Pipeline Parallelism** 来进一步扩展模型容量。
3. **改进时间积分策略**：设计更有效的 temporal causality 或 time-marching 方法，以支持长期动态预测。
4. **构建通用接口**：开发能接受任意 PDE 表达式、几何定义和边界条件的前端解析器，提升易用性。
5. **结合稀疏性和自适应采样**：利用 residual-based refinement 动态聚焦难拟合区域，提高计算效率。

---

> 📌 **总结一句话**：  
> **PINNACLE 是一个功能强大、高度模块化的开源框架，系统性地整合了现代 PINNs 训练的最佳实践，并首次将 QPINNs 与多GPU扩展纳入统一平台，为科学机器学习的研究提供了坚实基础，但也揭示了其在计算效率和理论保障方面的根本挑战。**

</details>

---

### 4. [Zero-Shot Scalable Resilience in UAV Swarms: A Decentralized Imitation Learning Framework with Physics-Informed Graph Interactions](https://arxiv.org/abs/2604.15762)

**Authors**: Huan Lin, Lianghui Ding  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.15762v1  

#### Abstract
Large-scale Unmanned Aerial Vehicle (UAV) failures can split an unmanned aerial vehicle swarm network into disconnected sub-networks, making decentralized recovery both urgent and difficult. Centralized recovery methods depend on global topology information and become communication-heavy after sever...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Zero-Shot Scalable Resilience in UAV Swarms: A Decentralized Imitation Learning Framework with Physics-Informed Graph Interactions*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**大规模无人机（UAV）集群在遭遇严重节点失效后产生的通信网络分裂（Communication Network Split, CNS）问题**，提出了一种去中心化的恢复机制。传统方法面临以下挑战：
- **集中式方法**依赖全局拓扑信息，在网络严重碎片化时通信开销大、难以部署；
- **启发式去中心化方法**适应性差，面对动态变化的拓扑表现不稳定；
- **现有的多智能体强化学习（MARL）方法**在扩展到更大规模或不同损伤程度时泛化能力弱，且缺乏对物理交互（如吸引、排斥、避碰）的有效建模。

### 提出的新方法与创新思路
作者提出了 **Physics-informed Graph Adversarial Imitation Learning (PhyGAIL)** 框架，其核心创新包括：

#### （1）**可扩展的去中心化恢复框架（CTDE架构）**
- 采用 **Centralized Training with Decentralized Execution (CTDE)** 范式。
- 在训练阶段利用全局信息进行优化，在执行阶段每个 UAV 仅基于局部观测决策，实现零样本迁移（zero-shot transfer）。

#### （2）**物理感知图神经网络（PhyGNN）**
- 设计了 **physics-informed graph neural network** 来显式建模局部方向性交互。
- 引入 **gated message passing 机制**，将每条边的消息分解为：
  - 吸引门（attraction gate）
  - 排斥门（repulsion gate）
  - 可学习的作用力强度（force strength）
- 这种设计赋予策略以**物理意义明确的协调偏置**，提升了运动安全性和稳定性。

#### （3）**场景自适应模仿学习策略（Scenario-Adaptive Imitation Learning）**
- 结合 **Generative Adversarial Imitation Learning (GAIL)** 提供密集的步级奖励指导。
- 设计 **expert-normalized temporal reward**，通过归一化专家完成时间来动态调整速度奖励，缓解因损伤程度不同导致的episode长度差异带来的训练不稳定性。

#### （4）**有界异构局部图感知机制**
- 构建包含三类节点的局部图：活跃 UAV、损毁 UAV 和虚拟中心（virtual center），增强对碎片区域的理解。
- 使用 **K-Nearest Neighbor (KNN) masking** 保证局部感知图的输入维度与全局规模无关，支持零样本扩展。

### 相比现有方法的优势
| 维度 | PhyGAIL优势 |
|------|-------------|
| **可扩展性** | 政策在一个20-UAV环境中训练，可直接迁移到500-UAV系统而无需微调 |
| **鲁棒性** | 在高损伤比（up to 95%）下仍保持完美收敛率 |
| **安全性** | 显著降低碰撞次数，优于大多数基线 |
| **效率** | 恢复速度快，runtime 开销低，适合实时应用 |
| **去中心化兼容性** | 单机计算、通信、内存复杂度均为 $O(1)$，不随总规模增长 |

---

## 2. 核心实验方法和设置

### 实验环境
- 自研二维 UAV swarm 仿真器
- 参数设定：
  - 通信半径 $D_{\text{comm}} = 120m$
  - 最大速度 $v_{\text{max}} = 10m/s$
  - 控制周期 $\Delta t = 0.1s$
  - 安全距离 $D_{\text{safe}} = 15m$（训练），检测碰撞阈值为10m（评估）

### 测试规模与损伤设置
- **UAV 数量**：$N \in \{20, 50, 100, 200, 500\}$
- **地图尺寸**：从 $320m \times 320m$ 到 $1600m \times 1600m$
- **损伤比例**：$p = N_{\text{dmg}} / N$，范围从 0.05 到 0.95（除 N=20 外上限为 0.9）
- 每组配置运行 50 次独立实验取平均

### 评估指标
| 指标 | 描述 |
|------|------|
| **Convergence Rate** | 成功恢复连通性的比率 |
| **Average Recovery Time (s)** | 平均恢复所需时间 |
| **Collisions per UAV** | 每架无人机在整个任务中发生的平均碰撞次数 |
| **Runtime Overhead** | 包括首次动作延迟（response time）、总求解时间、推理耗时等 |

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **集中式方法** | CR-MGC, DEMD, GDR-TS |
| **去中心化启发式** | center-fly, HERO, SIDR |
| **MARL 方法** | MADDPG-APF |
| **消融变体** | 替换为 GCN/GAT/SAGE 编码器、移除虚拟中心、KNN、GAIL奖励等 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table I & Fig. 4）

| 方法 | Convergence Rate | Avg. Recovery Time (s) | Collisions/UAV | Overall Rank |
|------|------------------|------------------------|----------------|---------------|
| **PhyGAIL** | **1.000** | **20.14** | **0.161** | **2.061** ✅ |
| DEMD | 0.990 | 26.55 | 1.851 | 3.956 |
| CR-MGC | 0.949 | 33.84 | 1.324 | 4.462 |
| GDR-TS | 0.975 | 36.48 | 1.664 | 4.471 |
| SIDR | 0.472 | 49.36 | 0.131 | 4.731 |
| center-fly | 0.998 | 28.95 | **18.994** ❌ | 4.798 |
| MADDPG-APF | 0.606 | 49.39 | 0.399 | 4.904 |
| HERO | 0.276 | 67.56 | 1.287 | 6.617 |

> 数据来源：Table I（汇总 N=100, 200, 500 表现）

#### 性能亮点：
- **完美收敛率**：在所有测试条件下达到 100%，显著优于其他方法；
- **最快恢复速度**：平均恢复时间最短（20.14s），尤其在高损伤情况下优势明显；
- **极低碰撞风险**：远低于 center-fly 等方法（后者高达近19次/机）；
- **最佳综合排名**：整体性能排名第一。

### 与基线方法对比结果
- **vs 集中式方法（CR-MGC, DEMD, GDR-TS）**：
  - 尽管这些方法在低损伤时表现尚可，但在大规模下通信成本极高（>10s 求解时间），无法满足实时需求；
  - PhyGAIL 的 per-step 推理时间仅为 **~17.61ms @ N=500**，远低于 GDR-TS 的 **69241.05ms**。
- **vs 去中心化启发式（HERO, SIDR）**：
  - SIDR 虽然碰撞略少，但收敛率极低（仅47.2%），恢复效率差；
  - HERO 收敛率最低（27.6%），且恢复时间最长。
- **vs MARL 方法（MADDPG-APF）**：
  - MADDPG-APF 泛化能力差，在大尺度下迅速退化。

### 消融实验结果（Ablation Study）

| 变体 | 影响说明 |
|------|---------|
| **替换为 GCN/GAT/SAGE** | 所有替代编码器在 N=500 时性能急剧下降，尤其是 GCN，收敛率下降达 0.673；表明标准图编码器缺乏方向敏感性，不利于稳定协调 |
| **移除虚拟中心（w/o virtual center）** | 导致严重性能退化（N=500 时收敛率↓0.640），说明共享参考方向对子网合并至关重要 |
| **移除损毁邻居观测（w/o damaged neighbor）** | 显著影响恢复效率，特别是在高损伤场景下，失去对“空洞”区域的空间感知 |
| **移除 KNN 选择机制** | 对小规模影响有限，但在 N=500 时开始显现问题，验证了有界感受野的重要性 |
| **移除专家时间奖励（w/o expert time reward）** | 恢复时间大幅增加，证明该设计对提升恢复效率的关键作用 |
| **移除 GAIL 奖励** | 不影响最终性能，但训练过程更慢、更不稳定，说明其主要用于加速收敛而非决定最终质量 |

> 图5显示：完整 PhyGAIL 训练稳定且快速收敛；而替换 PhyGNN 或移除虚拟中心会导致剧烈震荡甚至崩溃。

---

## 4. 关键结论和发现

### 主要发现
1. **PhyGNN 的物理门控消息传递机制是实现安全、稳定协调的核心**，它提供了强归纳偏置，使模型能够理解连续空间中的吸引与排斥行为。
2. **零样本扩展可行**：一个在20-UAV上训练的策略可以直接应用于500-UAV系统，并取得最优性能，验证了局部有界感知 + 物理建模的有效性。
3. **虚拟中心作为共享方向锚点极为重要**，尤其在高度碎片化场景中引导子网聚合。
4. **场景自适应奖励机制有效解决了稀疏奖励与变长episode带来的训练难题**，特别是 expert-normalized 时间奖励显著提升了恢复效率。
5. **去中心化执行下的资源消耗恒定**：理论分析与实测均表明，单机通信、计算、内存开销与全局规模无关，具备实际部署潜力。

### 方法的局限性
- 当前研究基于理想 LoS 信道假设，未考虑障碍物、风扰、非视距（NLOS）等现实因素；
- 所有 UAV 被视为同质个体，未涉及异构能力协同；
- 损伤模式为一次性静态失效，尚未处理持续性故障或动态攻击；
- 虚拟中心需预设位置，可能限制在完全未知环境中的适用性。

### 未来工作方向
- 扩展至 **三维空间与动态障碍环境** 下的恢复任务；
- 引入 **在线学习机制** 应对连续性故障；
- 探索 **无虚拟中心的自组织方向生成机制**（如共识算法）；
- 将 PhyGNN 模型推广至其他连续空间多智能体协调任务（如编队控制、覆盖搜索）；
- 在真实硬件平台上进行原型验证。

--- 

> ✅ **总结一句话**：  
> PhyGAIL 通过融合**物理感知图神经网络**、**有界局部感知**与**场景自适应模仿学习**，实现了首个可在严重碎片化下实现**零样本扩展、高可靠、高效且安全**的去中心化 UAV swarm 恢复框架，为大规模自主系统韧性提供了新范式。

</details>

---

### 5. [LACE: Lattice Attention for Cross-thread Exploration](https://arxiv.org/abs/2604.15529)

**Authors**: Yang Li, Zirui Zhang, Yang Liu, Chengzhi Mao  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.15529v1  

#### Abstract
Current large language models reason in isolation. Although it is common to sample multiple reasoning paths in parallel, these trajectories do not interact, and often fail in the same redundant ways. We introduce LACE, a framework that transforms reasoning from a collection of independent trials int...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LACE: Lattice Attention for Cross-thread Exploration

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前的 **Large Language Models (LLMs)** 在推理时通常采用**独立并行采样**（independent parallel sampling）策略，例如通过生成多个 Chain-of-Thought (CoT) 路径并选择最优解（如 Best-of-N 或 Self-Consistency）。然而，这些路径在生成过程中完全隔离，无法共享中间洞察，导致：

- **冗余失败**（redundant failures）：多个路径因相同错误而失败。
- **相关性误差**（correlated errors）：缺乏多样性，难以探索真正不同的解法。
- **后验选择低效**：依赖外部投票或验证器进行事后筛选，计算成本高且易受模型偏见影响。

该论文提出的问题是：**能否让多个推理路径在生成过程中实时通信、协作纠错，从而实现更高效、多样化的联合探索？**

---

### 提出了什么新方法或新思路

作者提出了 **LACE (Lattice Attention for Cross-thread Exploration)**，一个全新的框架，将传统的孤立推理转变为**协同并行推理**。其核心创新包括：

#### （1）Lattice Attention 架构
- 将标准的 1D 因果注意力（causal attention）推广为 **2D Lattice Attention**，引入“宽度维度”（thread dimension），允许信息不仅沿时间步（token）流动，也跨推理线程（thread）传播。
- 在标准注意力输出上添加轻量级的跨线程注意力分支，并通过**门控融合机制**（gated fusion）动态控制跨线程信息的注入强度。
- 仅增加 **<1% 的额外参数**，对预训练模型扰动极小。

#### （2）合成多线程训练数据管道
- 针对缺乏真实协作推理数据的问题，设计了一套**合成数据生成流程**，确保每道题有多个逻辑上不同的解法路径。
- 关键技术：
  - **Solution Cache 机制**：记录已生成的解法摘要，要求后续路径避免重复策略（`DO NOT USE: [Solutions Cache]`）。
  - **Step Decomposition**：压缩长推理链以提升训练效率。
  - **Self-Selection 标注任务**：由 LLM Judge 对多条路径进行比较评分，标注 `[best]`, `[success]`, `[fail]`，强制模型学习跨路径评估能力。

#### （3）多线程强化学习训练范式（Lattice GRPO）
- 扩展 GRPO 到多线程场景，在一次 rollout 中同时生成 T 条路径。
- 设计复合奖励函数：
  - **准确性奖励**（Accuracy Reward）：鼓励正确识别最佳路径（`[best]` 标签）。
  - **多样性奖励**（Diversity Reward）：基于嵌入差异性（embedding dissimilarity）激励不同推理路径。
- 共享优势信号（shared advantage）促进线程间协同优化。

---

### 相比现有方法的优势

| 维度 | 现有方法（如 Self-Consistency） | LACE |
|------|-------------------------------|------|
| 推理模式 | 孤立生成 → 事后聚合 | 并发交互 → 实时协作 |
| 多样性保障 | 依赖随机采样或提示工程 | 显式抑制重复路径（via cache + diversity reward） |
| 最优解选择 | 后验投票或外部验证器 | 内生自我选择（in-situ self-selection） |
| 效率 | 高延迟（需多次 decode + judge） | 单次前向传播完成探索与选择 |
| 可扩展性 | 随线程数线性增长延迟 | 微小内存带宽开销，支持高并发 |

> ✅ **核心优势**：LACE 实现了**统一的、协作式的推理过程**，从“多个盲人摸象”变为“团队共同解谜”。

---

## 2. 核心实验方法和设置

### 使用的数据集

- **主评测集**：
  - **AIME 24 / AIME 25**：美国数学邀请赛题目，用于评估复杂数学推理能力。
  - **LiveBench**：无污染、高难度的 LLM 推理基准。
- **训练数据来源**：
  - 基于 **DAPO dataset** 构建，经过本文提出的合成数据管道处理。
- **代理任务**：
  - **TextWorldCookAgent (TALES)**：交互式文本游戏环境，测试长期规划与反馈适应能力。

---

### 实验设置和评估指标

#### 模型基础
- 基于 **Qwen3-1.7B** 和 **Qwen3-4B** 模型构建。
- 插入 Lattice Attention 层于中后层（middle-to-last layers），每两层插入一次。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy (Acc)** | 正确率，基于 `[best]` 标签选出的答案是否正确（等价于 Pass@1） |
| **Exploration Diversity (Expl.)** | 所有线程输出之间的平均嵌入余弦距离，衡量路径多样性 |
| **Format Adherence (Fmt)** | 成功生成合法 `[best]` 标签的比例，反映协议遵循能力 |
| **Win Rate / Best Score** | 在 TextWorld 任务中的胜率与最高得分 |

#### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Independent Sampling** | 单线程模型 + 投票选择（majority voting） |
| **Isolated Parallel** | 多线程格式训练，但无跨线程注意力，线程仍独立 |
| **Judge-Based Selection** | 使用外部 LLM Judge 进行后验打分 |
| **Sequential Refinement** | 如 Self-Refine，迭代改进单一线程 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | AIME25 Acc | AIME24 Acc | LiveBench Acc |
|------|------------|------------|----------------|
| Independent + Voting | 10.0 | 10.0 | 16.0 |
| Isolated Parallel + SFT & RL + Voting | 3.3 | 3.3 | 15.5 |
| **LACE (Ours) + SFT & RL** | **13.3 (+3.3)** | **16.7 (+6.7)** | **16.5 (+0.5)** |

> 🔺 在 AIME24 上提升 **+6.7个百分点**，显著优于所有基线。

#### 更大规模（Qwen3-4B）表现更佳：
| 方法 | AIME25 Acc | AIME24 Acc | LiveBench Acc |
|------|-----------|-----------|----------------|
| Independent + Voting | 13.3 | 10.0 | 28.0 |
| Isolated Parallel + Voting | 3.3 | 10.0 | 15.5 |
| **LACE (Ours) + SFT & RL** | **16.7 (+3.4)** | **20.0 (+7.5)** | **33.0 (+5.0)** |

> ✅ LACE 在更大模型上依然保持强劲增益。

---

### 与后验选择机制对比（Table 12）

| 方法 | AIME24 Acc | Latency / query | Tokens / query |
|------|-----------|------------------|------------------|
| **LACE (Ours)** | **20.0** | **61.2s** | **7,120** |
| Gen-Judge (oracle) | 16.7–20.0 | 98.6s | 10,041 |
| Sequential Refinement | 16.7 | ~245s | 15,446 |

> ⚡ LACE 在准确率不逊色的前提下，**端到端延迟降低 60%+，token 消耗减少一半以上**。

---

### 消融实验结果

#### （1）数据质量至关重要（Table 2）
| 数据来源 | Edit Dist. | Emb. Dissim. | Cross-Attn Ratio |
|--------|------------|--------------|------------------|
| Parallel-R1 Data | 0.13 | 0.00 | 0.307 |
| **Our Data (w/ cache)** | **0.77** | **0.12** | **0.353** |

> 使用本文的数据管道显著提升了路径多样性与跨线程注意力强度。

#### （2）完整训练流程必要（Table 3）
| 训练方式 | AIME25 Acc | LiveBench Acc |
|---------|------------|----------------|
| w/o Pretrain & SFT | 0.0 | 1.0 |
| w/ Parallel-R1 Data | 0.0 | 3.0 |
| **w/ Our Data + Full Pipeline** | **13.3** | **11.5** |

> 缺少连续预训练或多阶段训练会导致性能崩溃。

#### （3）自我评估至关重要（Table 13）
| 设置 | AIME24 Acc | AIME25 Acc |
|-----|------------|------------|
| w/ Self-Assessment (`[best]`) | **20.0** | **16.7** |
| w/o SA → mean@4 | 7.5 | 5.8 |
| w/o SA → voting@4 | 6.7 | 6.7 |

> 自我选择机制本身带来超过 **+10点** 的准确率提升。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **跨线程注意力能有效促进协同推理**  
   LACE 成功实现了推理路径间的实时信息交换，使模型能够：
   - 动态吸收其他线程的中间洞察；
   - 主动规避已被证明无效的路径；
   - 在生成过程中完成“自我评判”与“最优解锁定”。

2. ✅ **多样性是性能提升的关键驱动力**  
   通过 `Solution Cache` 和 `Diversity Reward` 强制模型探索不同解法空间，避免陷入局部最优或重复错误。

3. ✅ **内生自我选择优于外部判断**  
   LACE 学会了在没有外部干预的情况下，准确识别出最优雅高效的解决方案，且效率远高于 judge-based 方法。

4. ✅ **协作行为可泛化至真实任务**  
   尽管训练数据为合成，LACE 在 AIME、LiveBench 和 TextWorld 等真实任务上均表现出良好泛化能力。

---

### 方法的局限性

1. 🚧 **依赖高质量合成数据构建**  
   当前方法严重依赖精心设计的数据管道，尚未解决如何从自然语料中自动提取多路径协作样本的问题。

2. 🚧 **超参敏感性**  
   多样性奖励权重需谨慎调节，过大可能导致路径过于分散，难以稳定选出 `[best]`。

3. 🚧 **可解释性挑战**  
   虽然可视化了 gate score 和 attention flow，但具体“何时为何借鉴某一线程”仍缺乏细粒度解释工具。

4. 🚧 **扩展性边界待验证**  
   当前线程数限制在 4–8，更多线程下的收益是否持续增长尚需进一步研究。

---

### 未来工作方向

1. 🔮 **构建真实世界协作推理数据集**  
   探索人类协作解题记录、多人辩论日志等作为天然的多路径推理数据源。

2. 🔮 **动态线程管理机制**  
   引入 early stopping、adaptive branching 等机制，根据推理进展动态调整活跃线程数量。

3. 🔮 **跨模态协作推理**  
   将 LACE 思想拓展至视觉-语言或多智能体系统，实现跨模态或多角色协同思考。

4. 🔮 **理论分析深化**  
   进一步建立形式化模型，量化协作推理相对于独立搜索的理论优势边界（如附录 Lemma 1 & Theorem 1 的延伸）。

---

> 💡 **总体评价**：LACE 是迈向“集体智能型语言模型”的重要一步，它不再将 LLM 视为单一思维流，而是将其重构为一个**内部可协作的认知网络**，为下一代推理架构提供了全新范式。

</details>

---

### 6. [Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning](https://arxiv.org/abs/2604.16029)

**Authors**: Jiaxi Bi, Tongxu Luo, Wenyu Du, Zhengyang Tang, Benyou Wang  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.16029v1  

#### Abstract
Parallel reasoning enhances Large Reasoning Models (LRMs) but incurs prohibitive costs due to futile paths caused by early errors. To mitigate this, path pruning at the prefix level is essential, yet existing research remains fragmented without a standardized framework. In this work, we propose the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Large Reasoning Models (LRMs)** 中，**Parallel Reasoning**（并行推理）通过生成多条独立的推理路径并聚合结果来提升准确性。然而，这种方法计算成本极高，因为许多路径从早期就已出错（称为“futile paths”），但仍被完整生成，造成大量算力浪费。

现有路径剪枝（path pruning）方法缺乏统一框架，研究分散，且多数方法存在以下问题：
- **External 方法**（如奖励模型）引入额外计算开销；
- **Non-learnable 方法**（如基于困惑度或相似性的启发式）无法适应任务特定错误模式。

因此，亟需一种高效、可学习、能利用内部状态进行早期剪枝的方法。

---

### 提出了什么新方法或新思路
本文提出两大核心贡献：

#### （1）首个系统化的 **Path Pruning 统一分类法（Unified Taxonomy）**
将现有方法按两个维度分类：
- **信号来源（Signal Source）**：External vs. Internal
- **是否可学习（Learnability）**：Learnable vs. Non-learnable

由此划分出四类方法，其中 **Type IV（Internal + Learnable）** 是理想但未被探索的方向。

#### （2）提出 **STOP (Super TOken for Pruning)** —— 首个 Type IV 实例化方法
- 在 LRM 内部插入一个轻量级模块（含 `[STOP]` token、LoRA adapter 和分类头）；
- 利用 LRM 的 **internal states（如 KV Cache）** 进行评分；
- 可训练以适应不同任务，实现高精度早期路径筛选。

---

### 相比现有方法的优势
| 维度 | STOP | 其他方法 |
|------|------|---------|
| **效率** | 极低验证延迟（仅处理几个 `[STOP]` token） | 外部模型需重新编码，延迟高 |
| **有效性** | 更早识别失败路径，保留更高比例正确答案 | 表面特征或固定规则难以捕捉复杂错误 |
| **通用性** | 基于 LoRA，易于迁移至不同模型 | 外部奖励模型需单独训练 |
| **部署友好** | “Plug-and-play”，无需额外服务 | 需维护双模型架构 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **数学类基准**：
  - AIME24, AIME25
  - BRUMO25
  - HMMT25
- **科学类基准**：
  - GPQA-D (GPQA Diamond subset)
- **逻辑推理**：
  - ZebraLogic（用于泛化能力测试）
- **竞赛场景**：
  - AIMO3（带工具使用的现实场景）

> 所有训练数据均经过去污染处理，确保无评测集泄露。

---

### 实验设置和评估指标

#### 模型规模
涵盖从 **1.5B 到 20B 参数** 的多种 LRM：
- DS-Qwen-2.5-1.5B / 7B
- DS-Qwen-3-8B
- GPT-OSS-20B

#### 推理流程（三阶段 Pipeline）
1. **Launch**：生成 N 条短前缀（如 64 条 × 2048 tokens），缓存 KV Cache；
2. **Check**：附加 `[STOP]` token 并打分；
3. **Resume**：保留 top-k 路径继续生成完整答案。

#### 评估指标
- **avg@mlk**：从 m 条中选出 k 条后，平均准确率；
- **Total Tokens**：衡量计算成本；
- **Token Reduction (%)**：相比不剪枝节省的 token 数量。

#### 固定配置
- 初始路径数 $N=64$，保留 $k=8$
- 剪枝检查点：2048 tokens
- 温度=0.6，top-p=0.95

---

### 基线方法对比
| 类型 | 方法 | 描述 |
|------|------|------|
| Type I (External + Non-learnable) | SlimSC | 基于语义冗余（Jaccard 相似性）剪枝 |
| Type II (External + Learnable) | LaBoR, DeepPrune | 使用外部 Process Reward Model (PRM) 打分 |
| Type III (Internal + Non-learnable) | DeepConf, AdaDec | 使用困惑度、熵等内部统计量 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 模型 | 数据集 | 不剪枝 avg@64 | STOP avg@8l64 | Token ↓ |
|------|--------|----------------|---------------|----------|
| DS-Qwen-1.5B | AIME24 | 30.10% | **37.92%** | **-73.88%** |
| DS-Qwen-7B | AIME25 | 39.67% | **42.50%** | **-71.91%** |
| GPT-OSS-20B | AIME25 | 70.99% | **75.42%** | **-71.62%** |
| GPT-OSS-20B | GPQA-D | 65.55% | **77.46%** | **-48.26%** |

> ✅ STOP 在所有模型和任务上均显著优于基线，同时减少 **超过 70% 的 token 消耗**。

---

### 与基线方法的对比结果
- **Type IV (STOP)** 在 avg@8l64 上全面超越其他类型：
  - 比 Type II（外部奖励模型）最高提升 **+5.84%**（AIME24 @ 1.5B）
  - 比 Type III（原始置信度）最高提升 **+9.00%**
- **计算效率优势明显**：
  - STOP 验证延迟仅 **0.20s**（相对开销 0.59%）
  - Type II 达 **1.13s**（3.37%），因需重编码整个序列

---

### 消融实验结果

#### （1）监督信号质量（Table 3）
| 监督方式 | AIME24 avg@8l64 | Cons@N |
|--------|------------------|--------|
| Hard Label ($K=1$) | 35.42% | 46.67% |
| Soft Label ($K=32$, MC) | **36.67%** | **53.33%** |

> ✅ **软标签（soft labels）更稳定**，降低方差，有助于学习可靠边界。

#### （2）Critique Adapter 必要性（Table 4）
移除 LoRA adapter 后性能大幅下降：
- AIME24 上 avg@8l64 从 **36.67% → 31.67%**

> ✅ 表明不能仅靠探针（probe）内部状态，需要专用转换模块进行“反思”。

#### （3）设计敏感性分析（Tables 5 & 6）
- 最优 `[STOP]` token 数量为 **4–6**
- LoRA rank 在 **128–256** 即可达到最佳性能，过大反而略降

> ✅ STOP 对超参选择鲁棒，无需大容量适配器即可有效运行。

---

## 4. 关键结论和发现

### 主要发现
1. **Type IV 范式最优**：结合 **Internal 信号** 与 **Learnability** 的方法在效率与效果上取得最佳平衡。
2. **STOP 显著提升性价比**：在保持甚至提高准确率的同时，**节省超 70% 的推理 token**。
3. **可扩展性强**：在不同模型大小（1.5B–20B）、任务类型（数学/科学/逻辑）和计算预算下表现稳健。
4. **过程导向评估机制**：注意力分析显示，STOP 关注的是 **逻辑转折点**（如 "don't"），而非直接跳向答案，说明其评价的是推理完整性。
5. **提出实用经验法则**：推导出缩放律公式预测最优保留率 $\gamma$：
   $$
   \gamma^{-1} = a C^b L_{\text{prefix}}^c L_{\text{task}}^{-d}
   $$
   可指导实际部署中的参数选择。

---

### 方法的局限性
- **极端规模未验证**：尚未在 >70B 模型或 $N>1000$ 的采样下测试；
- **结构灵活性不足**：目前只支持单阶段固定位置剪枝，未探索多阶段动态剪枝；
- **训练成本较高**：构建 MC 监督信号需大量采样（每前缀 $K=32$），前期计算开销大（见 Table 11）。

---

### 未来工作方向
1. **Progressive Multi-Stage Pruning**：级联式剪枝（64→32→16），逐步缩小搜索空间；
2. **加速 RL 训练**：在 PPO 或 GRPO 中作为 rollout 阶段的在线拒绝机制，提升高质量样本密度；
3. **动态剪枝点选择**：根据内容进展自动决定何时检查，而非固定 token 步长；
4. **跨领域预训练 STOP 模块**：构建通用的 early rejection head，减少 per-task 微调需求。

---

> 🔗 **代码、数据与模型开源地址**：https://bijiaxihh.github.io/STOP

</details>

---

### 7. [GroupDPO: Memory efficient Group-wise Direct Preference Optimization](https://arxiv.org/abs/2604.15602)

**Authors**: Jixuan Leng, Si Si, Hsiang-Fu Yu, Vinod Raman, Inderjit S. Dhillon  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.15602v1  

#### Abstract
Preference optimization is widely used to align Large Language Models (LLMs) with preference feedback. However, most existing methods train on a single positive-negative pair per prompt, discarding additional supervision available in preference datasets that typically contain multiple candidate resp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GroupDPO: Memory efficient Group-wise Direct Preference Optimization 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Direct Preference Optimization (DPO)** 及其变体通常仅在单个正负样本对上进行训练，而现实中的偏好数据集（如 Dolci-Instruct-DPO）往往为每个提示（prompt）生成多个候选响应，并通过人工或自动反馈评分。传统方法将这些多响应组降级为单一正负对，**丢弃了组内其他响应所蕴含的丰富相对质量信号**。

此外，虽然已有研究探索 **group-wise preference learning**（如 MPO、LiPO），但这些方法在实现时需要对整个响应组执行联合前向和反向传播，导致梯度在样本间耦合（gradient coupling），从而造成**显存占用随组大小呈指数增长**，限制了可扩展性。

### 提出了什么新方法或新思路
本文提出 **GroupDPO**，一种**内存高效的 group-wise DPO 算法**，其核心思想是：

- 设计一个**样本级别的代理损失函数（sample-level surrogate loss）**，该损失在**一阶梯度上等价于原始的 group-coupled 损失**。
- 具体实现分为两步：
  1. **无梯度前向传递（no-grad forward pass）**：计算所有响应的偏好分数 $u_i$，并基于此预计算每个样本的梯度系数 $c_i$。
  2. **标准反向传播**：使用这些预计算的 $c_i$ 作为常量权重，对每个样本独立计算损失 $L_{\text{sur}} = \sum_i c_i u_i(\theta)$ 并进行反向传播。

这种方法**解耦了样本间的梯度依赖**，避免了构建联合计算图，从而大幅降低峰值显存。

### 相比现有方法的优势
- **内存效率高**：峰值显存几乎不随组大小（group size）增长，解决了大组训练的 OOM 问题。
- **可扩展性强**：支持更大的组大小，充分利用组内多响应提供的监督信号。
- **保持性能**：在训练延迟（latency）上优于“展平”（flatten）实现，在性能上优于单对 DPO。
- **通用性好**：适用于多种 group-wise 目标（如 Margin、MPO、All-Pairs、Softmax）。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **离线设置（Offline Setting）**：
  - Prompt 来源：`Dolci-Instruct-DPO`
  - 响应生成模型：`gemma`, `olmo`, `gpt-oss`, `phi-4`, `qwen3`, `yi-1.5` 等多个模型
  - 分组方式：使用 `Skywork-Reward-V2-Qwen3-8B` 对响应打分，取 top-k 为正例，bottom-k 为负例，形成无序组（unordered group）。
- **在线设置（Online Setting）**：
  - Prompt 来源：`DAPO-Math-17k`
  - 响应生成：由策略模型自身采样
  - 分组方式：基于规则奖励（如数学答案正确性）划分正负组。

### 实验设置和评估指标
- **模型**：
  - 离线：`gemma-3-4b-sft`, `olmo-3-7b-it-sft`, `olmo-3.1-32b-it-sft`
  - 在线：`qwen3-4b-base`
- **组大小（Group Size）**：
  - 4B/7B 模型：16（8 正 + 8 负）
  - 32B 模型：8（受限于算力）
- **评估基准**：
  - 数学：`AIME24`, `AIME25`, `AMC23`, `MATH500`, `Minerva`, `Olympiad`
  - 推理：`AGIEval Eng`, `GPQA-D`, `MMLU-PRO`, `ZebraLogic`, `BBH`
  - 编程：`HumanEval+`, `LiveCode v6`
  - 指令跟随：`IFBench`, `IFEval`
- **超参数**：
  - 学习率：1e-6
  - β（DPO 温度）：2.0 ~ 15.0
  - 包含 NLL 正则项，系数默认为 1.0

### 基线方法对比
- **RFT**（Rejection Fine-Tuning）：仅在正响应上微调。
- **DPO**：标准单对偏好优化。
- **Group-wise 方法**（均使用本文提出的 surrogate 实现）：
  - **Margin**
  - **MPO**（Multi-Preference Optimization）
  - **All-Pairs**：对组内所有正负对应用 DPO 损失。
  - **Softmax**：一个正例 vs. softmax 聚合的负例。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 `olmo-3-7b-it-sft` 为例，Avg 为所有任务平均得分）

| 方法 | Math Avg | General Avg | Avg |
|------|----------|-------------|-----|
| DPO | 48.5 | 51.3 | 50.9 |
| Margin+ | 51.1 | 51.6 | 52.0 |
| MPO+ | 51.2 | 51.8 | 52.0 |
| All-Pairs+ | 51.2 | 51.7 | 51.7 |
| Softmax+ | 51.2 | 52.0 | 51.8 |

> ✅ 所有 group-wise 方法均显著优于 DPO 基线。

### 与基线方法的对比结果
- **Group-wise > Pairwise**：在离线和在线设置下，所有 group-wise 方法均一致优于 DPO 和 RFT。
- **大组更优**：如图 2 所示，随着组大小从 2 增加到 8，性能持续提升；但在 16 时趋于饱和，可能因噪声增加或收益递减。
- **不同 group 目标性能相近**：Margin、MPO、All-Pairs、Softmax 性能差异较小，表明 group-wise 结构本身是增益主因。

### 消融实验结果
#### （1）NLL 正则项的重要性（见 Figure 3 & 4）
- **是否加入 NLL** 是影响性能的关键因素。
- 加入 NLL 后，`General Avg` 平均提升约 **2~3 个百分点**。
- **训练稳定性**：无 NLL 时训练易崩溃（collapse），而有 NLL 时稳定收敛。
- 结论：**NLL 正则化对防止正响应似然下降至关重要**。

#### （2）内存与延迟分析（见 Figure 5 & Table 4）
| 实现方式 | 显存开销 | 时间复杂度 | 特点 |
|---------|----------|------------|------|
| **Vanilla** | 极高，随组大小↑ | $O(G^2 C(T))$ | 易 OOM |
| **Flatten** | 低 | $O(G^2 C(T))$ | 速度慢 |
| **Surrogate (本文)** | **低且稳定** | $O(G C(T))$ | **最优权衡** |

> ✅ Surrogate 实现在保持低显存的同时，训练速度远快于 Flatten，接近 Vanilla。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Group-wise 训练显著优于单对训练**：利用每 prompt 多个响应能提供更丰富的监督信号，提升模型性能。
2. **NLL 正则化不可或缺**：在 group-wise 设置中，必须加入对正响应的 NLL 损失，否则训练不稳定且性能差。
3. **内存瓶颈可通过梯度解耦解决**：本文提出的 surrogate 方法在**一阶梯度等价**的前提下，实现了样本级反向传播，极大降低了显存需求。
4. **方法高效且可扩展**：Surrogate 实现使大组训练成为可能，且训练效率优于现有替代方案。

### 方法的局限性
- **引入额外前向传递**：需要一次 no-grad 前向 pass 预计算系数，带来轻微延迟开销。
- **仅保证一阶等价**：代理损失与原损失在二阶及以上结构上不一致，可能影响依赖高阶信息的优化器行为。
- **未探索排序结构**：当前假设组内响应无序，未利用潜在的组内排名信息。

### 未来工作方向
- 探索结合组内排序信息的更精细 group-wise 目标。
- 将该内存优化范式推广至其他 group-based 或 listwise 学习任务（如推荐系统、检索）。
- 研究如何进一步减少 no-grad pass 的计算开销，例如缓存或近似计算。
- 探索在在线强化学习框架中动态调整组大小和分组策略。

</details>

---

### 8. [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)

**Authors**: Qwen Team  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.15804v1  

#### Abstract
In this work, we present Qwen3.5-Omni, the latest advancement in the Qwen-Omni model family. Representing a significant evolution over its predecessor, Qwen3.5-Omni scales to hundreds of billions of parameters and supports a 256k context length. By leveraging a massive dataset comprising heterogeneo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Qwen3.5-Omni Technical Report 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Qwen3.5-Omni 致力于构建一个**真正端到端、原生多模态（omnimodal）的大模型系统**，解决以下关键挑战：
- **多模态输入输出的统一建模**：如何在单一模型中高效处理文本、图像、音频、视频等异构模态，并支持实时交互。
- **长上下文理解能力不足**：现有模型难以有效处理长达数小时的音频或数百秒的视频内容。
- **流式语音合成中的不稳定性**：传统TTS系统在流式生成时存在跳词、发音错误、语调不自然等问题，尤其在跨语言场景下更严重。
- **真实世界代理行为缺失**：多数模型停留在“感知-响应”范式，缺乏自主工具调用、实时对话控制和跨模态推理能力。

### 提出的新方法与创新
1. **Hybrid MoE Thinker-Talker 架构**
   - 在 Thinker 和 Talker 两个模块均采用 **Hybrid Mixture-of-Experts (MoE)** 设计，提升长序列建模效率与容量平衡。
   - 支持高达 **256k token 的上下文长度**，可处理超过10小时音频或400秒720P视频（1FPS）。

2. **ARIA (Adaptive Rate Interleave Alignment)**
   - 创新性地提出一种动态对齐机制，在流式解码过程中自适应协调文本与语音token的生成节奏。
   - 避免因编码效率差异导致的错位问题，显著提升语音流畅度与自然度，且仅引入极低延迟。

3. **可控的多模态字幕生成（Controllable Audio-Visual Captioning）**
   - 能够生成结构化、带时间戳、自动分段的剧本级描述，精确同步音视频事件。

4. **原生多模态代理行为（Native Omnimodal Agentic Behavior）**
   - 支持自主 WebSearch、复杂 FunctionCall 调用。
   - 发现并命名了一种新兴能力：**Audio-Visual Vibe Coding** —— 模型能直接根据音视频指令生成可执行代码。

5. **零样本语音克隆与多语言扩展**
   - 支持基于用户提供的语音样例进行零样本声音定制。
   - 扩展至 **113种语言/方言的语音识别** 和 **36种语言的语音合成**，覆盖广泛语种。

### 相比现有方法的优势
| 维度 | Qwen3.5-Omni 优势 |
|------|------------------|
| **架构设计** | Thinker-Talker 分离 + Hybrid MoE，兼顾推理效率与表达能力 |
| **上下文长度** | 256k token，远超多数单模态模型 |
| **语音生成质量** | ARIA 显著改善流式语音稳定性与韵律自然性 |
| **交互能力** | 支持语义中断、音量/语速/情感控制、实时语音克隆 |
| **多语言支持** | 语音识别支持113种语言，远超 Gemini/GPT 等主流系统 |
| **代理能力** | 原生支持工具调用与跨模态编程（Vibe Coding），无需外部编排 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 模态 | 数据集 |
|------|-------|
| **预训练数据** | 超过 **100 million 小时的音视频内容**，以及大规模图文、音文、视文、纯文本语料 |
| **音频理解** | `MMAU`, `MMAR`, `MMSU`, `RUL-MuchoMusic`, `SongFormBench` |
| **语音对话** | `VoiceBench`, `URO-Bench-pro`, `SpeechRole`, `WildSpeech-Bench` |
| **语音识别 (ASR)** | `FLEURS`, `Common Voice`, `LibriSpeech`, `WenetSpeech`, `KeSpeech`, `Opencpop`, `MIR-1K` |
| **视觉理解** | `MMMU`, `MathVista`, `Video-MME`, `MLVU`, `MVBench`, `LVBench`, `MedXpertQA-MM` |
| **音视频理解** | `DailyOmni`, `WorldSense`, `AVUT`, `Qualcomm IVD`, `OmniCloze`, `OmniGAIA` |
| **语音生成 (TTS)** | `SEED-TTS`, `TTS Multilingual Test Set`, `CV3-Eval` |

### 实验设置与评估指标
| 任务类型 | 主要指标 |
|--------|---------|
| **X→Text（理解任务）** | 准确率（Accuracy）、BLEU、WER（越低越好） |
| **X→Speech（生成任务）** | WER（内容一致性）、SIM（说话人相似度，余弦相似性） |
| **长上下文能力** | `AA-LCR`, `LongBench v2` |
| **医学视觉问答** | `SLAKE`, `PMC-VQA`, `MedXpertQA-MM` |
| **工具使用能力** | `OmniGAIA`（无思考提示下的工具调用成功率） |

### 基线方法对比
- **主要对比对象**：
  - `Gemini-3.1 Pro`
  - `Qwen3.5-Plus-NoThinking`（同规模纯文本模型）
  - `Qwen3-Omni`（前代版本）
  - 商业TTS系统如 `ElevenLabs`, `MiniMax-Speech`, `GPT-Audio`

- **变体对比**：
  - `Qwen3.5-Omni-Plus`：高性能版本
  - `Qwen3.5-Omni-Flash`：轻量高速版本

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ 音频理解（Audio → Text）
| 指标 | Qwen3.5-Omni-Plus | Gemini-3.1 Pro | 优势 |
|------|--------------------|----------------|------|
| MMAU | **82.2** | 81.1 | ✔️ |
| MMSU | **82.8** | 81.3 | ✔️ |
| RUL-MuchoMusic | **72.4** | 59.6 | ✔️ |
| VoiceBench | **93.1** | 88.9 | ✔️ |
| 平均 ASR WER | **6.6%** | 7.3% | ✔️ |
| Cantonese ASR WER | **2.2%** | 6.3% | ✔️✔️ |

> 💡 在多个音频理解基准上超越 Gemini-3.1 Pro，尤其在粤语等复杂声调语言上表现突出。

#### ✅ 多语言语音生成（Zero-Shot TTS）
| 指标 | Qwen3.5-Omni-Plus | ElevenLabs | MiniMax |
|------|--------------------|------------|---------|
| 中文 WER | **0.695** | 2.252 | 16.026 |
| 英文 WER | **0.631** | 2.164 | 0.756 |
| 日语 WER | **3.479** | 3.519 | 10.046 |
| 平均 SIM | **0.788** | 0.756 | 0.677 |

> 📈 在29种语言中，有22种取得最低WER；在说话人相似度上全面领先。

#### ✅ 跨语言语音克隆（Cross-Lingual Voice Cloning）
| 方向 | Qwen3.5-Omni-Plus | CosyVoice3 |
|------|--------------------|-----------|
| zh → ko | **4.03** | 14.4 |
| zh → en | **2.18** | 2.98 |
| en → zh | **4.86** | 5.09 |

> 🔥 在 zh→ko 上实现 **72% 的相对错误率下降**，显示强大的跨语言泛化能力。

#### ✅ 工具使用与代理能力
| 指标 | Qwen3.5-Omni-Plus |
|------|------------------|
| OmniGAIA（工具调用） | **57.2%** |
| Qualcomm IVD（真实音视频交互） | **68.5**（vs Gemini 66.2） |

> 🤖 展现出较强的原生工具调用与现实场景交互能力。

#### ✅ 流式首包延迟（First-Packet Latency）
| 输入类型 | Qwen3.5-Omni-Plus | Qwen3.5-Omni-Flash |
|--------|--------------------|---------------------|
| 音频输入 | 435ms | **235ms** |
| 视频输入 | 651ms | **426ms** |

> ⚡ Flash 版本具备极低延迟，适合高并发实时服务部署。

---

## 4. 关键结论和发现

### 主要发现
1. **原生多模态训练可行且有效**  
   Qwen3.5-Omni 在保持与同规模纯文本模型相当的语言能力的同时，实现了卓越的多模态理解与生成能力，验证了“统一训练优于后期融合”的路径。

2. **ARIA 显著提升语音自然性**  
   通过动态对齐文本与语音token，解决了传统双通道生成中的错位问题，使流式语音更加稳定、自然，适用于长时间对话。

3. **长上下文建模能力突破**  
   支持256k token上下文，结合 Chunked Prefilling 和 GDN 模块，有效降低KV缓存开销，实现高效长序列推理。

4. **Emergent Capability：Audio-Visual Vibe Coding**  
   模型能够直接从音视频指令中提取意图并生成可运行代码，标志着多模态模型已具备初步的“感知即行动”能力。

5. **多语言语音能力达到SOTA**  
   在ASR、S2TT、TTS等多个维度全面超越 Gemini 和其他商业系统，特别是在亚洲语言（如粤语、日语、韩语）上优势明显。

### 方法的局限性
- **计算资源需求较高**：尤其是 Plus 版本，需要大量GPU显存支持。
- **部分低资源语言仍存在识别误差**：尽管覆盖广，但在某些小语种上的鲁棒性有待加强。
- **极端噪声环境下的语音理解仍有挑战**：未专门针对嘈杂场景优化。
- **视频帧率依赖采样策略**：目前为动态抽帧，可能丢失细节动作信息。

### 未来工作方向
- 进一步扩大语音支持语言数量，特别是非洲、南美等地的小语种。
- 探索更高效的 MoE 路由机制以降低推理成本。
- 引入物理仿真与机器人接口，推动模型走向具身智能（Embodied AI）。
- 加强对非标准口音、儿童语音、病理语音的理解能力。
- 开发更精细的情感控制与角色扮演能力，用于虚拟助手与娱乐场景。

---

> ✅ **总体评价**：Qwen3.5-Omni 是当前最先进、功能最完整的原生多模态大模型之一，不仅在技术架构上有重要创新（如 ARIA、Hybrid MoE），而且在实际性能上全面对标甚至超越 Gemini-3.1 Pro，是迈向通用多模态智能体（General Omnimodal Agent）的重要一步。

</details>

---

### 9. [T-RBFT: A Scalable and Efficient Byzantine Consensus Based on Trusted Execution Environment for Consortium Blockchain](https://arxiv.org/abs/2604.16053)

**Authors**: Wen Gao, Xinhong Hei, Yichuan Wang  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.16053v1  

#### Abstract
With the continuous expansion of blockchain application scenarios, consortium chains have raised higher performance and security requirements for consensus mechanisms. Unlike public blockchains, consortium chains typically implement an admission mechanism that restricts participation to trusted enti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：T-RBFT: A Scalable and Efficient Byzantine Consensus Based on Trusted Execution Environment for Consortium Blockchain

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Byzantine Fault Tolerance (BFT)** 协议（如 PBFT）在联盟链（Consortium Blockchain）场景中存在显著的**通信开销高、延迟大、可扩展性差**等问题。其 $O(N^2)$ 的通信复杂度在节点数量增加时导致性能急剧下降。尽管已有两层共识机制尝试结合 CFT（如 Raft）与 BFT 来提升效率，但往往以牺牲安全性为代价，容忍的拜占庭节点数减少。

此外，现有方案多依赖全局串行共识流程，在高并发或大规模部署下仍面临瓶颈。

### 🚀 提出的新方法：T-RBFT
本文提出 **T-RBFT**（TEE-based Raft cluster Byzantine Fault Tolerance），一种基于 **可信执行环境**（Trusted Execution Environment, TEE）的两层共识机制，专为联盟链设计。

#### 核心架构：
- **分组结构**：采用类似网络分片（network sharding）的思想，将共识节点动态划分为多个组（group）。
- **双层共识**：
  - **组内共识**（Intra-group）：各组内部使用改进的 **Raft 算法** 实现高效本地共识。
  - **组间共识**（Inter-group）：各组领导者通过基于 **TEE 的 BFT 协议** 达成全局一致性。
- **TEE 支持**：利用 TEE 提供的 **USIG 服务**（Unique Sequential Identifier Generator）实现消息唯一标识与顺序保障，增强安全性和效率。

### 🔍 创新点
1. **基于一致哈希与动态权重的节点分组策略**
   - 引入虚拟节点缓解数据倾斜。
   - 动态权重综合考虑**信用值**（consensus success rate, transaction impact rate）和**负载**（CPU、内存、网络），支持节点动态加入/退出时的均衡调整。
   - 设置**重组阈值**控制资源消耗。

2. **TEE 辅助的轻量级组间 BFT 共识**
   - 利用 TEE 中的 USIG 服务，将传统 PBFT 的三阶段简化为**两个通信步骤**。
   - 将所需最小共识节点从 $3f+1$ 降低至 $2f+1$，提升容错效率。

3. **具备拜占庭容错能力的组内共识强化机制**
   - 在 Raft 基础上引入 TEE 和 BLS 聚合签名，从以下三方面增强安全性：
     - **日志复制**：附加 f+1 个来自组领导者的 UI 证明，确保块内容已被全局认可。
     - **领导者选举**：候选人需提供“已提交日志项”的哈希值，并通过远程认证（remote attestation）证明其完整性（称为 *Committed Proof*）。
     - **提交确认**：要求超过 $3n/4$ 节点响应才视为提交成功，防止拜占庭节点伪造确认。

4. **天然支持跨组交易一致性**
   - 所有涉及多组的交易由组间 BFT 层协调，无需额外协议即可保证系统级原子性和一致性。

### ⚖️ 相比现有方法的优势
| 方面 | T-RBFT | 传统 PBFT | 其他两层协议（如 R-PBFT, WRBFT） |
|------|--------|----------|-------------------------------|
| 通信复杂度 | $O(N + k^2)$ | $O(N^2)$ | 通常仍较高 |
| 安全性 | 支持完整 BFT，组内也具 BFT 能力 | 高 | 多牺牲安全性换取性能 |
| 可扩展性 | 显著优于 PBFT，随节点增长更稳定 | 差 | 一般 |
| 延迟 | 更低，尤其在网络延迟明显时 | 高 | 中等 |
| 吞吐量 | 更高 | 低 | 较高但受限于安全模型 |

---

## 2. 核心实验方法和设置

### 🧪 实验平台
- **硬件**：Intel(R) Core(TM) i7-9700 CPU (8核)，36GB RAM
- **软件**：Ubuntu 20.04，Go 1.16
- **部署方式**：容器化模拟多节点环境，配合端口映射
- **TEE 实现**：基于 Intel SGX 构建 enclave，实现 USIG 服务；使用 HMAC-SHA256 进行消息认证，BLS 签名用于聚合签名。

### 📊 评估指标
1. **通信次数**（Communication Times）：完成一次共识所需的总消息交换次数。
2. **容错能力**（Fault Tolerance）：系统能容忍的最大拜占庭节点数。
3. **吞吐量**（Throughput）
4. **共识延迟**（Consensus Latency）

### 🔁 对比基线方法
- **MinBFT**：基于 TEE 的经典 BFT 协议，高容错但通信开销大。
- **R-PBFT**：Li et al. 提出的两层共识，含监督节点。
- **WRBFT**：基于主从架构的两层共识，使用 BLS 签名优化。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）通信次数对比（图4）
- 当总节点数 $N=42$ 时：
  - **R-PBFT**：需 246 次通信
  - **T-RBFT**：仅需 **188 次**，降低约 **23.6%**
- 随着节点数增加，T-RBFT 的通信增长趋势远低于 PBFT 类协议。
- **MinBFT** 通信开销最大，因其未采用分层结构。

#### （2）不同分组数下的通信开销（表1，N=60）
| 分组数 $k$ | 通信次数 |
|------------|---------|
| 20         | 559     |
| 15         | 404     |
| 10         | 299     |
| 6          | 255     |
| **3**      | **236**（最低）|

👉 表明适当减少分组数有助于降低组间通信开销。

#### （3）容错能力对比（图5）
- **MinBFT**：最高容错（如 N=36 时可达 17 个故障节点）
- **T-RBFT**：次之（N=36 时为 6 个：组间 2 个，每组内 1 个）
- **R-PBFT**：较低（同条件下仅为 1 个）
- **WRBFT**：被认为**不安全**，因无法检测主节点对不同请求分配相同序列号的行为，**不能容忍拜占庭节点**

#### （4）吞吐量与延迟
- 实验表明 T-RBFT 在相同容错阈值下具有更低的共识延迟。
- 随着节点规模扩大，T-RBFT 吞吐量表现更稳定，展现出良好可扩展性。
- 在非忽略通信延迟的网络中，T-RBFT 延迟优势更加明显。

> ❗ 注：文中未报告消融实验（ablation study），但通过模块化设计分析了各组件作用（如 USIG、TEE、动态分组等）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **T-RBFT 成功解决了联盟链中 BFT 协议性能与安全难以兼顾的问题**：
   - 通过**分组 + 双层共识 + TEE 辅助**，实现了高性能与强安全性的统一。
2. **通信复杂度显著降低**：
   - 从 $O(N^2)$ 降至 $O(N + k^2)$，大幅提升可扩展性。
3. **组内共识具备拜占庭容错能力**：
   - 通过 TEE 和 BLS 签名强化 Raft，使其可在存在拜占庭节点环境下安全运行。
4. **动态分组策略有效维持系统平衡**：
   - 结合信用与负载的复合评分机制提升了分组合理性与适应性。
5. **TEE 的引入虽带来一定开销，但整体性能仍优于现有方案**：
   - USIG 服务带来的效率增益抵消了 TEE 访问成本。

### ⚠️ 方法的局限性
1. **依赖 TEE 硬件支持**：
   - 要求所有共识节点配备 TEE（如 SGX），限制了部署灵活性。
   - 存在侧信道攻击风险（如 SGX 已知漏洞），需持续更新防护措施。
2. **信任假设较强**：
   - 假设存在可信管理员负责初始分组，可能影响去中心化程度。
3. **未考虑跨链场景**：
   - 当前设计聚焦单条联盟链内部优化。

### 🔮 未来工作方向
1. **多平台 TEE 实现**：
   - 将方案移植到更多标准 TEE 平台（如 GlobalPlatform TEE）。
2. **结合跨链技术**：
   - 探索与跨链协议集成，解决单链可扩展性与隐私隔离不足问题。
3. **进一步优化分组策略**：
   - 引入机器学习预测负载变化，实现更智能的动态重分组。
4. **增强抗共谋攻击能力**：
   - 研究如何防范多个恶意组联合发起攻击。

---

## 总结
T-RBFT 是一项面向联盟链的高性能、高安全性共识机制创新。它巧妙融合 **network sharding 思想、Raft 高效性与 TEE 安全性**，构建了一个**可扩展、低延迟、具备完整拜占庭容错能力**的两层共识框架。实验证明其在通信开销、容错能力和吞吐量方面均优于当前主流方案，为高性能联盟链提供了坚实的底层共识基础。

</details>

---

### 10. [Weak-Link Optimization for Multi-Agent Reasoning and Collaboration](https://arxiv.org/abs/2604.15972)

**Authors**: Haoyu Bian, Chaoning Zhang, Jiaquan Zhang, Xingyao Li, Yuanfang Guo, Wei Dong, Yang Yang  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.15972v1  

#### Abstract
LLM-driven multi-agent frameworks address complex reasoning tasks through multi-role collaboration. However, existing approaches often suffer from reasoning instability, where individual agent errors are amplified through collaboration, undermining overall performance. Current research mainly focuse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Weak-Link Optimization for Multi-Agent Reasoning and Collaboration

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
当前的 **LLM-driven multi-agent** 框架虽然通过多角色协作提升了复杂推理任务的表现，但仍面临以下关键挑战：
- **Reasoning instability**：个体 agent 的错误在协作过程中被放大，导致系统整体性能下降。
- **弱代理（weak agent）影响大**：系统可靠性受最薄弱环节制约，传统方法如 majority voting 或 debate 无法有效抑制弱 agent 的负面影响。
- **缺乏系统性优化机制**：现有研究多聚焦于增强高能力 agent 或抑制不可靠输出，而对识别并强化性能瓶颈（即“弱链路”）关注不足。

### 🚀 提出的新方法与思路  
作者提出 **WORC**（Weak-Link Optimization for Reasoning Cooperation），一个基于“弱链原则”（weak-link principle）的 multi-agent 推理优化框架。其核心思想是：**系统的整体性能由最弱的一环决定，应优先补偿弱 agent 而非一味加强强项**。

WORC 包含两个阶段：
1. **Weak Agent Localization（弱代理定位）**
   - 利用 **Swarm Intelligence Algorithms (SIAs)** 在少量样本上搜索最优 agent 权重配置，构建 **weight knowledge base**。
   - 构造 **task signature**（融合语义 embedding 和结构统计特征），用于表征任务特性。
   - 设计 **meta-learning-based weight predictor**，实现从 task signature 到 agent 权重的零样本映射，从而快速识别新任务中的 weak agent。

2. **Weak-Link Optimization（弱链优化）**
   - 基于预测权重，采用 **uncertainty-driven allocation strategy** 动态分配额外推理预算（reasoning budget）。
   - 预测权重越低的 agent，获得越多重复生成机会（repeated-sampling quota），以弥补其可靠性缺陷。

### 🔍 相比现有方法的优势  
| 方面 | WORC 的优势 |
|------|-------------|
| **稳定性** | 显著降低 multi-agent 系统的性能波动，提升推理一致性 |
| **通用性** | 支持跨任务、跨架构迁移，可集成到 MetaGPT、HIMA 等多种 MAS 框架中 |
| **无需监督训练** | 基于 SIAs 自动学习 agent 贡献度，避免依赖人工标注或显式监督信号 |
| **动态适应性强** | 根据任务特征自动调整资源分配，优于静态策略（如 uniform 或 rule-based 分配） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集  
共使用 **6 个标准推理 benchmark 数据集**，覆盖多种推理类型：
- **MATH**：高等数学题求解
- **GSM8K**：小学数学应用题
- **BBH**（Big-Bench Hard）：逻辑与算法推理
- **MMLU-CF**：常识与事实知识问答（无污染版本）
- **HotpotQA**：多跳问答（multi-hop QA）
- **LongBench**：长上下文理解任务

### ⚙️ 实验设置与评估指标  
- **主干框架**：基于 **AgentChain (AC)** 实现 WORC，所有 agent 使用 **GPT-4o**。
- **Swarm Intelligence Algorithms (SIAs)**：测试了三种 SIA —— **HO**（Hippopotamus Optimization）、**PSO**、**GWO**，用于构建权重知识库。
- **评估指标**：
  - 对单一答案任务（如 MATH、GSM8K）：使用 **Exact Match Accuracy**
  - 对部分正确答案任务（如 HotpotQA、LongBench）：使用 **F1 Score**

### 🔁 基线方法对比  
#### 推理级基线（inference-level baselines）：
- **CoT**（Chain-of-Thought）
- **CoT-SC**(n=5)（Self-Consistency）
- **Self-Refine**
- **Analogical Prompting**
- **FoT**(n=8)（Forest-of-Thought）
- **AoT**（Atom of Thoughts）
- **AFlow**

#### 框架级基线（framework-level optimization strategies）：
- **Majority Voting**
- **AFlow**
- 各 multi-agent system（MAS）原始版本（如 MetaGPT、HIMA、MAS2、AgentChain）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table I）

| Method | Avg. Accuracy/F1 (%) |
|--------|------------------------|
| CoT | 73.6 |
| CoT-SC | 75.5 |
| AFlow | 76.1 |
| AoT | 80.8 |
| **WORC + AC (Ours)** | **82.2 ± 0.4** |

- WORC 在所有任务上均取得最佳表现，平均准确率达 **82.2%**，显著优于最强基线 AoT（+1.4 pp）和 FoT（+6.3 pp）。
- 在复杂任务上增益更明显：
  - **HotpotQA**：83.2 vs. 80.6（AoT）
  - **LongBench**：68.4 vs. 68.5（AoT）——虽绝对值接近，但 WORC 更稳定

### 🆚 与其他优化策略对比（Table II）
将 WORC 作为插件模块应用于不同 MAS 架构，结果表明：
| MAS Architecture | WORC 提升幅度（vs. None） |
|------------------|----------------------------|
| MetaGPT | +4.0% |
| HIMA | +3.3% |
| MAS2 | +3.0% |
| AgentChain | +6.6% |

> ✅ 表明 WORC 具有良好的 **cross-architecture generalization** 能力。

### 🔍 不同 LLM 下的表现（Table V）
| LLM | MATH | GSM8K | Avg. |
|-----|------|-------|------|
| **DeepSeek-V3** | **89.3** | **98.2** | **~90.5** |
| Qwen-Turbo | 84.9 | 95.3 | ~87.0 |
| GPT-4.1-nano | 85.3 | 95.6 | ~87.5 |

> 💡 显示 WORC 可适配不同 backbone LLM，并在更强模型上释放更大潜力。

### 🔪 消融实验结果（Ablation Studies）

#### （1）Task Signature 组件分析（Table III）
| 配置 | 平均精度 |
|------|----------|
| 仅语义 embedding | 79.9% |
| 仅统计特征 | 80.0% |
| **完整 task signature** | **82.2%** |

> 结果证明：**语义 + 结构特征联合建模** 是实现跨任务泛化的关键。

#### （2）预算分配策略比较（Table IV）
| 策略 | 平均精度 |
|------|----------|
| Uniform Allocation | 80.0% |
| Rule-Based Allocation | 80.9% |
| **WORC Allocation (ours)** | **82.2%** |

> 表明 **基于权重的自适应分配机制** 明显优于静态策略。

#### （3）Cross-Task Generalization（Table VI）
| 训练集 → 测试集 | 准确率 |
|----------------|--------|
| GSM8K → MATH | 86.3% |
| MATH → GSM8K | 95.6% |
| HotpotQA → LongBench | 67.4% |

> 显示 meta-learning predictor 具备较强的 **zero-shot 迁移能力**。

#### （4）不同 SIA 的影响（Figure 3 & Table VII）
- 尽管 PSO、GWO、HO 得到的权重数值略有差异，但 **agent 排名趋势高度一致**。
- 所有 SIA 版本的 WORC 均显著优于 baseline AC，且性能波动更小（variation ↓）。

> 表明 WORC 的有效性不依赖特定 SIA，具有较强鲁棒性。

---

## 4. 关键结论和发现

### ✅ 主要发现  
1. **“补短板”优于“拉长板”**：针对 weak agent 进行资源补偿能显著提升 multi-agent 系统的整体鲁棒性和准确性。
2. **task signature + meta-learning 可实现 zero-shot 弱代理识别**：无需重新训练即可泛化至新任务。
3. **WORC 是一种通用优化范式**：不仅适用于自定义 AgentChain，也能提升 MetaGPT、HIMA 等主流框架性能。
4. **系统稳定性大幅提升**：相比 baseline，WORC 的多次运行结果方差显著减小（Figure 4），更适合实际部署。

### ⚠️ 局限性  
- **计算开销增加**：由于引入重复采样与 SIA 搜索过程，推理成本约为 baseline 的 **1.8 倍**（见 Table IX）。
- **依赖高质量 EvalAgent**：虽然人类评测显示 EvalAgent 与专家评分一致性高（Kappa > 0.72），但在极端模糊任务中仍可能存在偏差。
- **当前为离线优化**：weight knowledge base 需预先构建，尚未支持完全在线自适应更新。

### 🔮 未来工作方向  
- 降低计算开销，探索轻量化 SIA 或增量学习机制。
- 提升在线适应能力，实现实时 agent 性能监测与动态调参。
- 扩展至更大规模、异构 agent 环境（如混合开源/闭源模型）。
- 探索与其他 reasoning enhancement 技术（如 CoT-SC、Self-Refine）的融合。

---

> **一句话总结**：  
> WORC 提出了一种“以弱制强”的新视角，通过 **识别并补偿 multi-agent 系统中的 weak link**，实现了推理性能、稳定性和泛化性的全面提升，为构建可靠、高效的协同智能系统提供了新范式。

</details>

---

### 11. [BlockRaFT: A Distributed Framework for Fault-Tolerant and Scalable Blockchain Nodes](https://arxiv.org/abs/2604.15731)

**Authors**: Manaswini Piduguralla, Souvik Sarkar, Arunmoezhi Ramachandran, Sathya Peri  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.15731v1  

#### Abstract
Blockchain technology enhances transparency by maintaining a distributed ledger among mutually untrusting parties. Despite its advantages, scalability and availability remain critical bottlenecks that hinder widespread adoption. The increasing complexity of blockchain nodes further necessitates robu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《BlockRaFT: A Distributed Framework for Fault-Tolerant and Scalable Blockchain Nodes》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

区块链节点在**可扩展性**（scalability）和**可用性**（availability）方面面临严重瓶颈，尤其是在**许可链**（permissioned blockchain）环境中。尽管区块链网络本身具备容错能力，但单个组织运行的节点若发生崩溃，将导致该组织无法提交交易、查询状态或验证区块，形成“组织级单点故障”。

此外，传统区块链节点架构中，**Merkle Tree 更新开销大**，限制了智能合约的并行执行效率。

### 提出了什么新方法或新思路

本文提出了 **BlockRaFT** —— 一个基于 **RAFT 共识协议** 的分布式、容错且可扩展的区块链节点框架，其核心创新包括：

#### （1）**分布式节点内架构（Intra-node Distribution）**

- 将单个逻辑区块链节点实现为一个由多个物理系统组成的 **RAFT 领导者-跟随者集群**（leader-follower cluster）。
- 集群对外表现为单一网络实体，不增加网络消息复杂度，也不破坏各组织平等代表权。
- 利用 RAFT 实现领导者选举，支持 **[(n-1)/2] 个节点崩溃容忍**。

#### （2）**任务划分与负载均衡**

- 将节点内部模块划分为 **stateless**（无状态）和 **stateful**（有状态）两类：
  - **Stateless 操作**（如交易收集、DAG 构建）集中在 leader 处理；
  - **Stateful 操作**（如状态更新、Merkle Tree 维护）在所有节点上复制并协调。
- Leader 负责将 DAG 分解后的独立组件分发给 followers 并行执行，实现负载均衡。

#### （3）**三阶段并发 Merkle Tree 优化**

提出一种新型并发 Merkle Tree 设计，**解耦智能合约执行与状态树更新**：

1. **并发执行 + 局部记录**：执行期间读取全局状态，修改写入本地 `concurrent hash map`，避免直接更新 Merkle Tree；
2. **并行叶子节点更新**：所有交易执行完成后，将最终状态变更并行写入 Merkle Tree 叶子节点；
3. **顺序父节点重计算**：仅对受影响路径上的内部节点进行串行哈希回溯，生成新的 Merkle Root。

此设计显著降低状态更新开销，提升并行度。

---

### 相比现有方法的优势

| 特性 | BlockRaFT | DiPETrans | PilotFish | Sharding |
|------|----------|-----------|-----------|---------|
| 可扩展性 | ✅ | ✅ | ✅ | ✅ |
| 容错性（crash tolerance） | ✅ | ❌ | ❌ | ❌ |
| 工作负载分布 | ✅（全栈） | ⚠️（仅执行） | ⚠️（部分） | ✅ |
| 支持分布式 SCT 执行 | ✅ | ✅ | ✅ | ✅ |
| 无需修改共识机制 | ✅ | ✅ | ✅ | ❌ |
| 与现有系统兼容 | ✅ | ❌（需社区信任） | ✅ | ❌ |

> ✅ 表示支持；❌ 表示不支持；⚠️ 表示有限支持

**优势总结**：
- 不改变外部区块链网络结构，兼容性强；
- 实现真正的**节点级容错**，而非依赖多节点冗余；
- 同时优化执行层与存储层，综合性能更优；
- 支持动态 leader 故障恢复与任务再分配。

---

## 2. 核心实验方法和设置

### 实验平台配置

- **操作系统**：Ubuntu 64位
- **主服务器硬件**：
  - CPU：AMD EPYC 7452 32核（共128逻辑CPU）
  - 内存：251 GB
- **单节点资源配置**：
  - CPU：16逻辑核
  - 内存：9.7 GB

### 实现技术栈

- 编程语言：C++
- 分布式协调服务：**ETCD**（内置 RAFT 协议），用于 leader 选举与共享状态管理
- 消息队列：**Redpanda**，构建高吞吐异步交易池（multi-producer, single-consumer queue）
- 通信模型：支持 ETCD 共享内存 和 直接消息传递两种模式

### 数据集与工作负载生成

- 使用 **YCSB**（Yahoo! Cloud Serving Benchmark）作为基准测试工具
- 基于 **BenchBase** 框架生成可控读写比例的工作负载
- 智能合约类型：
  - **Voting Contract**：模拟投票系统，含注册、投票、转账等操作
  - **Wallet Contract**：模拟钱包系统，含 deposit、withdraw、transfer、balance 查询

### 评估指标

- **Execution Time**（执行时间，ms）
- **Throughput**（吞吐量，tx/s）
- **Scalability**：随线程数、事务数、冲突率变化的趋势
- **Fault Tolerance**：在不同数量节点崩溃下的性能表现
- **Breakdown Analysis**：各阶段耗时分析（如 DAG 构建、组件分配、执行、通信等）

### 基线方法对比

1. **Single-core Execution Model**  
   - 单线程顺序执行，作为传统区块链节点的基线

2. **Multi-core Execution Model**  
   - 在单机内多核并行执行，使用相同 DAG 和 Merkle Tree 优化，但无分布式容错机制

3. **BlockRaFT-Msg**  
   - 替换 ETCD 为直接消息传递的变体版本，用于评估通信开销影响

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）并发 Merkle Tree 性能提升（图3）

| 场景 | 提升幅度 |
|------|--------|
| **100% 写密集型负载** | > **两个数量级**（>100x） |
| **读密集型负载** | ~5x 加速 |
| **100K 操作规模** | 近线性扩展，远超串行版本 |

> 结论：分离执行与状态更新极大减少了 Merkle Tree 更新开销。

#### （2）BlockRaFT 与基线对比（图4、图11）

| 配置 | 4000 tx/block, 16 threads, 5% conflict |
|------|-------------------------------|
| **Single-core Voting** | ~8.5 s |
| **Multi-core Voting** | ~3.4 s |
| **BlockRaFT Voting** | ~3.1 s |

- BlockRaFT 比 single-core 快 **约2.7倍**
- 比 multi-core 略快（**~10% 提升**），同时提供完整容错能力

#### （3）线程扩展性（图4c & 图8）

| 线程数 | Multi-core Wallet (ms) | BlockRaFT Wallet (ms) |
|-------|------------------------|------------------------|
| 2     | 7671.2                 | 10744                  |
| 16    | 1883.6                 | 3272.8                 |
| 64    | 1779.6                 | 2944                   |

- 多核模型在低线程下更快（无通信开销）
- 但 BlockRaFT 在高并发下仍保持良好扩展性，且具备 fault tolerance

#### （4）事务量扩展性（图11）

- 当每块事务从 1000 增至 5000：
  - Single-core 执行时间增长最快（非线性）
  - Multi-core 与 BlockRaFT 接近线性增长
  - BlockRaFT 在大负载下仍优于 single-core 显著

#### （5）容错性能（图5 & 表IX）

| 节点数 | 0 crash (ms) | 1 crash (ms) | 2 crashes (ms) |
|--------|--------------|---------------|----------------|
| 3-node | 3934.4       | 7890.8        | N/A            |
| 5-node | 3621.6       | 7524.25       | 7412.5         |
| 7-node | 3817.8       | 7539.2        | 7546 → 7423    |

- 发生首次崩溃后执行时间翻倍（因负载重新分配）
- 后续崩溃影响较小，系统趋于稳定
- 证明具有**优雅降级**（graceful degradation）能力

#### （6）消融实验（Breakdown Analysis，图6）

- 主要瓶颈在于：
  - **Execution Phase**（执行阶段）
  - **Component Detection**（连通分量检测）
- ETCD 开销相对较小，表明协调机制高效
- 0% 冲突时性能不佳是由于大量地址导致跨节点通信成为瓶颈

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **BlockRaFT 成功实现了节点级容错**，解决了组织因单节点宕机而离线的问题；
2. ✅ **通过 leader-follower 架构实现了工作负载的有效分发与并行化**，提升了吞吐量；
3. ✅ **三阶段并发 Merkle Tree 显著降低了状态更新开销**，尤其在写密集场景下效果突出；
4. ✅ **相比 single-core 和 multi-core 模型，在保证容错的前提下实现了接近最优的性能**；
5. ✅ **系统具备良好的可扩展性和故障恢复能力**，支持动态 leader 切换与任务重调度。

### 方法的局限性

1. **通信开销成为瓶颈**，特别是在低冲突、高地址分散场景下（如 0% 冲突），跨节点数据同步限制了性能；
2. **当前实现依赖 ETCD 或类似协调服务**，引入额外运维复杂性；
3. **组件检测与 DAG 构建阶段存在计算开销**，尚未完全优化；
4. **未解决拜占庭容错**（Byzantine Fault Tolerance），仅支持崩溃容错（crash fault tolerance）；
5. **集群规模受限于 RAFT 协议本身的可扩展性边界**。

### 未来工作方向

1. **优化通信机制**：引入更高效的数据共享策略，减少跨节点传输开销；
2. **去中心化架构探索**：研究 P2P 形式的分布式节点架构，进一步降低对中心协调者的依赖；
3. **集成更先进的 Merkle Tree 结构**：如 Jellyfish Merkle Tree 或 Angela，结合批处理与稀疏更新特性；
4. **支持 BFT 扩展**：将框架扩展至容忍恶意节点行为；
5. **自动化调参与资源调度**：基于负载动态调整线程数、分片粒度等参数。

---

> **总结一句话**：  
> BlockRaFT 提出了一种新颖的“**节点即集群**”思想，通过在单个逻辑节点内部构建容错分布式架构，并辅以并发 Merkle Tree 优化，在不牺牲公平性和兼容性的前提下，有效提升了区块链系统的**可扩展性**与**可用性**，为下一代高性能许可链基础设施提供了可行路径。

</details>

---

### 12. [Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models](https://arxiv.org/abs/2604.15609)

**Authors**: Yunbei Zhang, Shuaicheng Niu, Chengyi Cai, Feng Liu, Jihun Hamm  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.15609v1  

#### Abstract
Test-Time Adaptation (TTA) for black-box models accessible only via APIs remains a largely unexplored challenge. Existing approaches such as post-hoc output refinement offer limited adaptive capacity, while Zeroth-Order Optimization (ZOO) enables input-space adaptation but faces high query costs and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文研究的是**严格黑盒环境下的 Test-Time Adaptation (TTA)** 问题。在现实场景中，许多先进的模型（如通过 API 提供服务的 CLIP、ViT 或商业视觉系统）仅允许用户提交输入并接收输出预测，而无法访问其内部参数、梯度、中间特征或架构细节。这种“黑盒”限制使得传统的白盒 TTA 方法（如 TENT、CoTTA）无法应用。

现有黑盒方法存在以下缺陷：
- **Output refinement**（如 LAME）只能对最终预测进行后处理，适应能力有限。
- **Input modification** 方法如 ZOO（Zeroth-Order Optimization）虽然能学习输入提示（prompt），但需要大量 API 查询，成本高昂且在无监督信号下容易优化不稳定（例如将图像扭曲为高置信度错误预测）。
- **Augmentation-based** 和 **diffusion-based purification** 方法则面临查询效率低或推理延迟高的问题。

因此，如何在**不增加额外 API 调用**的前提下实现高效、稳定、强适应性的黑盒 TTA 是一个未被充分探索的关键挑战。

---

### 提出的新方法：BETA
作者提出 **BETA (Black-box Efficient Test-time Adaptation)**，一种新颖的黑盒 TTA 框架，其核心思想是：

> 利用一个轻量级的本地白盒“steering model”作为梯度通路（gradient pathway），通过**prediction harmonization**技术构建共享优化目标，在无需访问目标模型内部的情况下实现稳定高效的适应。

#### 核心创新点：
1. **Prediction Harmonization（预测融合）**
   - 将黑盒模型 $f_B$ 和本地 steering model $f_s$ 的输出以加权方式融合成一个混合分布 $p_H = \alpha p_s + (1-\alpha)p_B$。
   - 仅通过反向传播更新 $f_s$ 的路径来优化该混合目标，从而绕过对 $f_B$ 的梯度需求。
   - 理论分析表明，该目标等价于最小化 $H(p_s)$ 和 $JS(p_s \| p_B)$ 的组合，其中 JS 散度项起到了“锚定”作用，确保 prompt 学习不会偏离黑盒模型的信念。

2. **Stabilization Mechanisms（稳定性机制）**
   - **Consistency Regularization**：引入 KL 散度约束 $\mathcal{L}_{\text{consist}} = D_{KL}(p_s(x)\|p_s(x'))$，防止 prompt 过度修改输入导致语义破坏。
   - **Prompt Learning-oriented Filtering**：只使用低熵样本更新 prompt，避免噪声梯度干扰训练过程。

3. **零样本 steering model 构建**
   - 即使没有合适的预训练 steering model，也可利用公开 CLIP 的 text encoder 将任务标签映射为分类权重，构建一个弱但可用的梯度提供者。

---

### 相比现有方法的优势
| 维度 | BETA | ZOO | LAME | Augmentation |
|------|------|-----|------|-------------|
| API 调用次数 | **1次/样本** | 高达数十次 | 1次 | 数十次 |
| 推理延迟 | 几乎无增加 | 极高 | 极低 | 极高 |
| 适应能力 | 强（输入空间优化） | 强但不稳定 | 弱（仅输出调整） | 中等 |
| 成本效益 | **极高（250×优于 ZOO）** | 极低 | 高 | 低 |

BETA 在保持标准推理速度的同时实现了接近白盒方法的性能提升，是首个真正适用于真实世界 API 部署的实用型黑盒 TTA 方案。

---

## 2. 核心实验方法和设置

### 数据集
- **ImageNet-C**（severity 5）：用于评估模型在常见图像损坏下的鲁棒性。
- **ImageNet-Sketch (IN-S)** 和 **ImageNet-Rendition (IN-R)**：评估跨域泛化能力。
- **EuroSAT**：卫星遥感图像分类，测试领域差异大的零样本迁移。
- **Derm7pt**：皮肤病灶分类，验证医学图像中的适用性。
- **Clarifai 商业 API**：真实部署环境测试，每请求收费 \$0.0032。

### 实验设置与评估指标
- **任务类型**：Fully Test-Time Adaptation（源域数据不可见，无标签在线流式适应）
- **评估指标**：Top-1 Accuracy (%)
- **运行模式**：strictly online, one-pass, single API call per sample
- **steering model**：ViT-S/16（22M 参数），远小于目标模型（ViT-B/16: 87M, ViT-L/16: 304M）
- **超参数**：$\alpha=0.4$, $\lambda=50$, entropy threshold $\epsilon=0.9 \times \ln(1000)$

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **White-box** | TENT, SAR, CoTTA, T3A |
| **Gray-box** | FOA, TPT, B2TPT, BCA, RA-TTA |
| **Black-box** | LAME, ZOO-CMA/SPSA/RGB, TT-Aug, DDA |

特别地，ZOO 方法被实现为三种变体（CMA-ES, RGF, SPSA-GC），均配置为每样本 16 次查询以保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 ImageNet-C 上的表现（ViT-B/16 黑盒目标）
| 方法 | 平均准确率 (%) | 增益 (%) |
|------|----------------|----------|
| Source | 55.5 | 0.0 |
| TENT (white-box) | 59.6 | +4.1 |
| LAME (black-box) | 54.1 | -1.4 |
| ZOO-SPSA-GC | 55.1 | -0.4 |
| **BETA (ours)** | **62.6** | **+7.1** |

> ✅ BETA 不仅显著超越所有 black-box 方法，还优于多个 white-box 方法（如 TENT、CoTTA），且仅需一次 API 调用。

#### 在 CLIP 上的表现（Vision-Language Model）
| 方法 | IN-S / IN-R 平均准确率 (%) | 增益 (%) |
|------|----------------------------|----------|
| Source | 60.0 | 0.0 |
| TPT (white-box) | 62.5 | +2.5 |
| DPE (white-box) | 66.3 | +6.3 |
| **BETA (ours)** | **63.4** | **+3.4** |

> ✅ BETA 是首个在严格黑盒条件下成功提升 CLIP 性能的方法，并超过多个专为 VLM 设计的 white/gray-box 方法。

#### 在 Clarifai 商业 API 上的成本效益
| 方法 | 预算 (\$) | 准确率增益 (%) | 成本优势 |
|------|-----------|----------------|---------|
| ZOO | >\$100 | ~+5.2 | — |
| **BETA** | **\$0.4** | **+5.2** | **250× 更便宜** |

> 💡 在相同预算下，BETA 可获得高达 +17.1% 的增益，证明其极高的实用性。

---

### 消融实验结果（Ablation Study）

#### 组件分析（Table 8）
| 方法 | 输入适应 | KL 正则 | 数据过滤 | 输出融合 | 准确率 (%) | 增益 (%) |
|------|----------|--------|----------|----------|------------|----------|
| Source | – | – | – | – | 55.5 | 0.0 |
| LAME | – | – | – | ✓ | 54.1 | -1.4 |
| ZOO | ✓ | – | – | – | 56.0 | +0.5 |
| Exp-3 (仅 KL) | ✓ | ✓ | – | ✓ | 59.7 | +4.3 |
| Exp-4 (仅过滤) | ✓ | – | ✓ | ✓ | 60.2 | +4.7 |
| **BETA (全框架)** | ✓ | ✓ | ✓ | ✓ | **62.6** | **+7.1** |

> 🔍 结果显示：**KL 正则化和数据过滤都至关重要**，单独使用任一组件都无法达到最佳效果。

#### Steering Model 选择的影响（Table 7）
即使使用更小的 ViT-Tiny 或 ResNet50 作为 steering model，BETA 仍能有效提升大模型性能：
- 使用 ViT-Small 时平均增益达 +7.1%
- 使用 ViT-Tiny 时仍有 +2.7% 增益

> 📌 表明 BETA 的有效性不依赖于 steering model 自身的强大性能，而是它能否提供有效的梯度信号。

---

## 4. 关键结论和发现

### 主要发现
1. **黑盒 TTA 可行且高效**  
   BETA 首次证明：即使完全无法访问模型内部，也能通过一个轻量本地模型实现强大而稳定的 test-time 适应。

2. **prediction harmonization 是关键机制**  
   通过融合两个模型的输出并仅从本地模型回传梯度，构建了一个可优化且与目标一致的目标函数，理论上有 Jensen-Shannon 对齐项支撑。

3. **稳定性机制不可或缺**  
   在无监督、随机初始化 prompt 的设定下，缺乏 KL 正则或数据过滤会导致性能崩溃。

4. **成本与性能的帕累托最优**  
   BETA 实现了“单次 API 调用 + 零额外延迟”的极限效率，同时取得最强性能，打破了以往“高成本换性能”的范式。

5. **通用性强**  
   BETA 可应用于纯视觉模型（ViT）、视觉语言模型（CLIP）、甚至医疗图像（Derm7pt）和遥感图像（EuroSAT），展现出广泛适用性。

---

### 方法的局限性
1. **依赖 steering model 的质量**  
   若 steering model 输出过于混乱（如标签嵌入无效），JS 项失效，则退化为单纯的 entropy minimization，可能导致失败。

2. **极端计算资源受限时性能下降**  
   如使用极度量化或百万以下参数的 steering model，梯度粗糙，难以有效引导像素空间优化。

3. **无法纠正严重误判的黑盒模型**  
   如果黑盒模型对某个类别过度自信（如对抗性偏移），JS 项会将其“锚定”，导致 prompt 向错误方向优化。

4. **当前仅支持分类任务**  
   扩展到 detection、segmentation 或 generation 任务需重新设计 harmonized objective。

---

### 未来工作方向
1. **扩展至多模态 agent 系统**  
   论文指出 BETA 可视为“advisor-executor”架构的实例：小型可控 advisor（steering model）指导大型黑盒 executor（API）行为。这一思想可推广至 LLM agent 中的 plan 调整、tool use 优化等。

2. **适配 retrieval、reward、tool-use API**  
   在测试时动态调整外部模块的行为，仅需一次远程调用即可完成适应。

3. **自动选择或生成 steering model**  
   开发自动化流程，根据目标任务自动生成最合适的轻量级 steering model。

4. **增强对 adversarial shift 的鲁棒性**  
   引入去偏机制或不确定性估计，避免被误导性高置信度预测锁定。

---

> ✅ **总结一句话**：  
> **BETA 开创了一种全新的黑盒 TTA 范式——用一个廉价、可控的小模型“驾驶”一个强大但封闭的大模型，在几乎零成本的条件下实现了卓越的适应性能，为现实世界中 API 化 AI 系统的持续鲁棒化提供了切实可行的解决方案。**

</details>

---

### 13. [Bilevel Optimization of Agent Skills via Monte Carlo Tree Search](https://arxiv.org/abs/2604.15709)

**Authors**: Chenyi Huang, Haoting Zhang, Jingxu Xu, Zeyu Zheng, Yunduan Lin  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.15709v1  

#### Abstract
Agent \texttt{skills} are structured collections of instructions, tools, and supporting resources that help large language model (LLM) agents perform particular classes of tasks. Empirical evidence shows that the design of \texttt{skills} can materially affect agent task performance, yet systematica...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Bilevel Optimization of Agent Skills via Monte Carlo Tree Search

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于 **LLM Agent Skills** 的优化问题。Agent Skills 是由指令、工具和辅助资源组成的结构化软件包，用于增强大型语言模型（LLM）代理在复杂任务中的表现。尽管已有研究表明技能设计对代理性能有显著影响，但由于其异构性（包含自然语言、代码、文档等）、组件间的强依赖性以及离散组合的设计空间，系统性地优化技能仍极具挑战。

传统方法难以建模这种“结构+内容”的联合优化问题，而本文提出了一种新的框架来解决这一难题。

---

### 提出了什么新方法或新思路
作者提出了一个 **双层优化（Bilevel Optimization）框架**，将技能优化分解为两个层次：
- **外层循环（Outer Loop）**：使用 **Monte Carlo Tree Search (MCTS)** 搜索最优的技能结构配置（如文件组织、模块划分等）。
- **内层循环（Inner Loop）**：在固定结构下，通过 LLM 驱动的内容精炼机制优化各组件的具体内容（如指令文本、脚本逻辑等）。

该框架的关键创新在于：
- 将技能表示为元组 $ S = (\theta, \phi) $，其中 $\theta$ 表示结构，$\phi$ 表示内容，从而形式化双层优化问题。
- 外层采用 MCTS 进行路径依赖的结构搜索，利用延迟反馈进行探索与利用权衡。
- 内层引入 **family-specific refinement** 策略，根据不同类型的结构变更调用不同的内容优化策略（如 metadata 编辑、instruction 改写、script 生成等），并结合保守选择规则（conservative selection rule）防止过拟合噪声评估信号。

此外，整个流程由 LLM 实现自动化，无需人工干预即可完成从种子技能到优化技能的演化。

---

### 相比现有方法的优势
| 对比维度 | 本文方法 | 现有方法（如 AFlow [9]） |
|--------|--------|----------------------|
| 优化对象 | Agent Skills（异构组件：instructions, scripts, references, assets） | Code-represented workflows（以代码为主） |
| 优化架构 | 双层结构：明确分离 structure search 与 content refinement | 单一层级的 workflow 修改 |
| 内容优化 | 家族感知的 refinement 策略 + 保守选择机制（LCB） | 统一的代码修改策略 |
| 探索机制 | MCTS + LLM-guided 分析/诊断/提案三阶段推理 | MCTS 直接作用于代码变更 |
| 反馈归因 | 明确区分结构与内容的影响，支持可解释性优化 | 结构与内容耦合，归因困难 |

因此，该方法更适用于真实世界中复杂的、非标准化的 Agent Skills 优化场景。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验基于 **Operations Research Question Answering (ORQA)** 数据集 [21]：
- 类型：多选题问答任务，涉及运筹学建模理解（识别决策变量、约束、目标函数等）。
- 规模：共 1,513 个实例，覆盖 20 个应用领域。
- 输入长度：平均 231 词。
- 示例见原文 Table 1。

---

### 实验设置和评估指标
#### 数据划分
为了控制优化成本并避免过拟合，原始 ORQA 被划分为三个子集：
- **Search Split**：用于双层优化过程中的候选技能评估与搜索引导（训练集角色）。
- **Confirm Split**：用于不同搜索配置之间的模型选择（验证集角色）。
- **Test Split**：最终评估所选最优技能的泛化性能（测试集角色）。
- 每次运行采样共计约 120 道题目。

#### LLM 设置
- **运行时执行（Runtime Execution）**：`openai/gpt-5.2-codex` 在 Harbor 沙箱中执行技能。
- **优化过程协调（Optimization Orchestration）**：`openai/gpt-5.4` 通过 DSPy 框架驱动 MCTS 流程。
- 各阶段 token 预算差异化分配，尤其提案阶段给予高达 20,000 tokens 的预算，体现多保真度思想（multi-fidelity simulation optimization）。

#### 评估指标
- 主要指标：**Exact Match Score**（精确匹配率），即正确选择答案选项的比例。
- 辅助信号：改进幅度、置信度、诊断信息（用于内层保守选择）。

---

### 基线方法对比
- **Baseline**：初始的 AI 生成种子技能（seed skill），包含 `SKILL.md` 和一个 reference 文件。
- **对比配置**：
  - **Configuration A**：保守型 MCTS 设置（较少轮次、UCB1 选择、早收敛）。
  - **Configuration B**：探索型设置（更多轮次、mixed-probability selection、更大动作空间）。
- 最终胜出者在 Confirm Split 上选出，并与 Baseline 在 Test Split 上比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | Search Split Reward | Confirm Split Score | Test Split Exact Match |
|-----|----------------------|-----------------------|----------------------------|
| Seed Skill (Baseline) | — | — | **0.90625** |
| Optimized Skill (Config B) | 0.9434 | 0.8857 | **0.9375** |

> ✅ **绝对提升：+0.03125（相对提升约 3.45%）**

---

### 与基线方法的对比结果
- 在 Search Split 上，两种配置均达到峰值奖励 0.9434，表明优化潜力一致。
- 在 Confirm Split 上，Configuration B 显著优于 A（0.8857 vs. 0.8571），说明更强的探索能力有助于找到泛化更好的结构。
- 最终选定 Config B 的优化技能，在独立 Test Split 上实现了 **93.75%** 的准确率，显著超越基线。

---

### 消融分析与路径可视化（见 Figure 2）
- MCTS 搜索树显示多个分支被探索后放弃，证明算法并非简单线性改进，而是主动比较多种结构可能性。
- 成功路径包括：
  1. 将原 reference 文件中的关键 question-type guidance 内联至 `SKILL.md`；
  2. 新增 **Question-Type Triage Checklist** 模块，强制代理先分类问题类型再进入主流程。
- 这些结构性调整配合内容层面的精细化改写（如步骤显式化、输入契约强化、输出格式严格限定），共同促成性能提升。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **双层优化框架有效分离了结构与内容优化**，使得复杂技能的系统性改进成为可能。
2. ✅ **MCTS 是处理路径依赖、离散组合结构搜索的有效工具**，尤其适合结合 LLM 进行语义感知的动作提议。
3. ✅ **结构优化与内容优化具有协同效应**：外层结构重组提升了信息可访问性，内层内容精炼增强了执行确定性。
4. ✅ **中心化与显式化设计提升性能**：将分散的知识集中到主指令文件，并增加检查清单类结构，显著改善代理行为一致性。
5. ✅ **LLM 可作为优化器本身（LLMs as optimizers）**，不仅用于推理，还可驱动自身系统的自我进化。

---

### 方法的局限性
1. 🔒 **计算开销高**：每轮 MCTS 扩展需多次 LLM 调用与下游任务执行，不适合实时或低资源场景。
2. 🧩 **依赖高质量种子技能与 profile 构建**：若 comprehension stage 提取错误或 profile 不准，可能导致搜索方向偏差。
3. ⚠️ **评估噪声敏感**：虽然使用 LCB 提供保守估计，但在极小样本下仍可能出现误选。
4. 📏 **扩展性待验证**：目前仅在一个特定领域（ORQA）验证，是否适用于更广泛的任务类型尚需进一步研究。

---

### 未来工作方向
1. ➕ **引入学习型价值函数**：用 learned value model 替代随机 rollout，加速 MCTS 收敛。
2. 🔁 **闭环自迭代优化**：让优化后的技能反哺生成新的种子技能，实现持续演进。
3. 🤝 **多人协作式技能工程**：结合人类反馈（human-in-the-loop）指导搜索先验 $P$ 的构建。
4. 📊 **建立通用 Skill Optimization Benchmark**：推动跨任务、跨平台的技能质量评估标准建设。

---

> 💡 **总结一句话**：  
> 本文首次将 **bilevel optimization + MCTS + LLM-as-optimizer** 融合应用于 Agent Skills 自动优化，展示了结构与内容协同进化的可行性，在 ORQA 上实现了显著性能提升，为下一代智能代理的自我完善提供了新范式。

</details>

---

### 14. [CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](https://arxiv.org/abs/2604.15840)

**Authors**: Shidong Yang, Ziyu Ma, Tongwen Huang, Yiming Hu, Yong Wang, Xiangxiang Chu  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.15840v1  

#### Abstract
Reinforcement learning for LLM agents is typically conducted on a static data distribution, which fails to adapt to the agent's evolving behavior and leads to poor coverage of complex environment interactions. To address these challenges, we propose CoEvolve, an agent-data mutual evolution framework...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前基于 **Reinforcement Learning (RL)** 的 **LLM Agent** 训练通常依赖于静态的、人工标注的交互轨迹数据集（如专家演示），存在以下关键瓶颈：

- **高成本**：人工收集高质量交互轨迹耗时且昂贵。
- **低覆盖性**：静态数据无法适应 agent 行为的动态演化，难以覆盖真实环境中长尾、复杂或边界情况的交互模式。
- **缺乏自适应性**：传统合成数据生成（如 LLM 自动生成任务）多为开环、无反馈驱动，探索浅层，无法针对 agent 的弱点进行定向增强。

### ✅ 提出了什么新方法或新思路

本文提出 **CoEvolve** —— 一种 **Agent-Data Mutual Evolution（代理-数据共进化）框架**，实现无需人类监督的闭环训练：

- **核心思想**：从 agent 的 rollout 轨迹中提取 **failure-prone 信号**（如遗忘、不稳定性、稀有行为），并利用这些信号指导 LLM 进行 **反馈驱动的任务再探索与合成**。
- **闭环机制**：
  1. **Agent 训练** → 2. **信号提取** → 3. **LLM 引导的环境再探索** → 4. **新任务抽象与验证** → 5. **更新训练数据分布** → 回到第1步。

该过程实现了 **agent 能力** 与 **训练数据分布** 的协同演化。

### ✅ 相比现有方法的优势

| 对比维度 | 传统方法（如专家标注 / 静态合成） | CoEvolve |
|--------|-------------------------------|---------|
| 数据来源 | 人工标注 或 无反馈 LLM 合成 | 反馈驱动、动态演化 |
| 探索方式 | 静态、一次性 | 动态、闭环、针对性强 |
| 成本 | 高（人力）或低效（随机探索） | 低（全自动、无监督） |
| 泛化能力 | 易过拟合固定分布 | 更好覆盖长尾与边界场景 |
| 可扩展性 | 有限 | 可持续自我改进 |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集

- **AppWorld**：模拟现实数字服务交互环境（日历、邮件、音乐等），通过 Python API 调用完成多步推理任务。
  - 评估指标：**Task Goal Completion (TGC)** 和 **Scenario Goal Completion (SGC)**
- **BFCL-V3 (Berkeley Function Calling Leaderboard)**：评估多轮、嵌套、并行工具调用能力。
  - 评估指标：**Multi-turn Success Rate**（所有步骤均正确才计为成功）
- （附录补充）**WebShop** 和 **ALFWorld**：用于跨域迁移分析。

### ✅ 实验设置和评估指标

- **Backbone 模型**：
  - Qwen2.5-7B-Instruct
  - Qwen3-4B-Instruct
  - Qwen3-30B-A3B-Instruct
- **训练算法**：基于 **Group Relative Policy Optimization (GRPO)** 进行 RL 训练。
- **反馈信号类型**：
  - **Forgetting Signal**：曾成功但现在失败的任务。
  - **Boundary Signal**：同一任务下策略输出高度不稳定（部分成功、部分失败）。
  - **Rare Signal**：出现频率极低但反复出现的行为模式。
- **任务合成与验证**：使用更强的 LLM（如 Qwen-Max）进行信号引导的探索，并通过环境执行验证任务有效性后加入训练集。

### ✅ 基线方法对比

- **Zero-shot**：直接在未训练模型上测试。
- **GRPO**：标准强化学习训练，使用初始合成数据。
- **Static Synthetic Data**：仅使用离线 LLM 生成的任务训练。
- **Closed-source LLMs**：Claude-Sonnet-4.5、GPT-4、Gemini-2.5-Flash。
- **Open-source Baselines**：DeepSeek-V3.2、LLaMA-3.3-70B 等。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table 1）

| Model | Avg. Score (原始) | Avg. Score (w/ CoEvolve) | 绝对增益 |
|-------|------------------|--------------------------|--------|
| Qwen2.5-7B | 3.08 | 22.51 | **+19.43** |
| Qwen3-4B | 11.72 | 27.30 | **+15.58** |
| Qwen3-30B-A3B | 22.64 | 40.78 | **+18.14** |

> 所有 backbone 均取得显著提升，最大绝对增益达 **19.43%**。

### ✅ 与基线方法的对比结果

- 在 **BFCL-V3** 上，**Qwen3-4B-w/CoEvolve (63.00)** 超越了 **GPT-4 (54.00)** 和 **Gemini-2.5-Flash (41.50)**。
- 在 **AppWorld-Challenge Split** 上，CoEvolve 显著提升了对复杂、非标准任务的处理能力（如按钮标签变化、流程分支等）。
- 即使是中等规模模型（如 Qwen3-4B），经 CoEvolve 训练后性能接近甚至超过更大规模开源模型。

### ✅ 消融实验结果（Ablation Studies）

#### 🔹 不同训练阶段的影响（Table 3）

| 阶段 | AppWorld (TGC) | BFCL | 平均得分 |
|------|---------------|------|--------|
| Zero-shot | 16.67 | 26.50 | 21.59 |
| + Synthetic Data | 28.57 | 58.00 | 43.29 |
| + Random Exploration | 30.36 | 60.50 | 45.43 |
| + Feedback (CoEvolve) | **35.71** | **63.00** | **49.36** |

> 反馈机制带来最大增益，证明其核心作用。

#### 🔹 移除不同反馈信号的影响（Table 5）

| 设置 | AppWorld | BFCL | 平均 |
|------|---------|------|-----|
| 完整 CoEvolve | 35.71 | 63.00 | 49.36 |
| w/o Forgetting | 30.36 | 60.00 | 45.18 |
| w/o Boundary | 33.33 | 61.00 | 47.17 |
| w/o Rare | 33.92 | 60.50 | 47.21 |

> **Forgetting Signal 贡献最大**，说明纠正“退化”行为至关重要；三种信号互补。

#### 🔹 是否移除任务验证（Table 11）

| 设置 | BFCL | AppWorld |
|------|------|---------|
| 有验证 | 63.00 | 35.71 |
| 无验证 | 58.50 | 27.38 |

> 验证机制显著提升数据质量，防止噪声任务污染训练。

#### 🔹 计算成本分析（Table 7）

| Benchmark | 额外计算时间占比 | 性能相对提升 |
|----------|------------------|-------------|
| AppWorld | ~9.67% | +22.92% |
| BFCL | ~12.76% | +8.62% |

> CoEvolve 仅引入约 **10% 的额外计算开销**，却带来显著收益，性价比极高。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **反馈驱动的数据演化有效**：利用 agent 自身的失败信号（forgetting, boundary, rare）可精准定位薄弱环节，并引导生成有针对性的新任务。
2. **数据分布动态演化优于静态训练**：CoEvolve 能持续扩展训练数据的多样性与复杂度，尤其在长尾和边界场景表现更鲁棒。
3. **无需人工干预即可实现自我提升**：整个框架完全自动化，无需 human-in-the-loop，具备强可扩展性。
4. **合成任务更具挑战性**：分析显示，合成任务普遍具有更长的交互步数、更强的步骤依赖性和更高的逻辑复杂度（见 Fig. 6, 7, 8）。
5. **跨域泛化能力初现**：在 AppWorld 上训练的 agent，在 BFCL 上也表现出零样本性能提升（26.50 → 45.00），反之亦然。

### ⚠️ 方法的局限性

1. **早期信号可能噪声大**：在训练初期，agent 行为不稳定，提取的信号可能不可靠或误导。
2. **依赖外部探索 LLM 质量**：任务发现能力受限于用于 re-exploration 的 LLM 的推理与世界知识水平。
3. **潜在安全风险**：自动合成任务可能导致生成有害或对抗性指令，需引入安全过滤机制。
4. **反馈信号种类有限**：目前仅使用三种信号，未来可探索更多细粒度反馈（如认知偏差、冗余动作等）。

### 🔮 未来工作方向

- 引入 **安全审查机制** 和 **风险触发的人工审计**，确保合成任务可控。
- 探索更丰富的反馈信号类型（如 attention drift、reasoning inconsistency）。
- 将 CoEvolve 应用于 **multi-agent** 或 **real-world tool use** 场景。
- 结合 **curriculum learning** 机制，按难度逐步引入新任务。
- 研究如何降低对外部强 LLM 的依赖，实现端到端自进化。

---

> **总结一句话**：  
> **CoEvolve 开创了一种无需人类监督的 agent 自我进化范式，通过 agent 与 data 的闭环共演，实现了高效、持续、针对性的能力提升，在多个复杂交互 benchmark 上取得了突破性进展。**

</details>

---

### 15. [Accuracy Is Speed: Towards Long-Context-Aware Routing for Distributed LLM Serving](https://arxiv.org/abs/2604.15732)

**Authors**: Takeshi Yoshimura, Valentijn Dymphnus van de Beek, Tatsuhiro Chiba  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.15732v1  

#### Abstract
Distributed LLM serving systems optimize per-request latency and throughput. However, under long-context workloads, inference accuracy becomes more variable. When incorrect responses trigger retries, accuracy directly translates into cumulative user-visible delay that is not captured by single-shot ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Accuracy Is Speed: Towards Long-Context-Aware Routing for Distributed LLM Serving**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Distributed LLM Serving** 系统中，传统调度策略（如负载均衡、会话亲和性）主要优化单次请求延迟（latency）和吞吐量，假设模型输出始终正确。然而，在 **long-context workloads** 下，模型推理准确性随输入长度、语言等因素显著波动，导致错误响应触发重试（retries），从而累积用户可见延迟。

现有系统忽略了 **accuracy variability** 对端到端延迟的影响，仅用单次延迟（single-shot latency）无法准确反映用户体验。

> **核心问题**：当错误答案引发重试时，accuracy 实际上影响了系统的“速度”。

---

### 🚀 提出的新方法与新思路

#### （1）提出新度量指标：**Time-to-Correct-Answer (TTCA)**
- 定义为从首次请求开始，直到获得第一个正确答案所经历的 **wall-clock time**。
- 显式建模 retry 动态，更真实地反映用户感知延迟。
- 将 **accuracy** 视为系统级性能因素，而非孤立的模型能力问题。

#### （2）提出轻量级路由机制：**Lightweight Accuracy-Aware Routing (LAAR)**
- 一种 **capability-based routing** 设计，结合成功率预测与延迟估计进行调度决策。
- 使用轻量特征（如 prompt length、language）构建 success probability 模型，避免语义分析开销。
- 在控制平面保持低复杂度（O(|M|)，M 为候选模型数），适用于高并发场景。

#### （3）引入 retry-aware 路由机制
- 对已失败的模型施加惩罚，防止重复选择同一模型造成无效重试。
- 鼓励探索不同模型路径，提升多轮尝试下的成功概率。

---

### 🔍 相比现有方法的优势

| 方面 | 现有方法（Load-aware / Session-affinity） | LAAR |
|------|----------------------------------------|------|
| 优化目标 | 单次延迟、资源利用率 | **TTCA（综合考虑 correctness + latency）** |
| 准确性建模 | 忽略 accuracy 变化 | 显式建模 success probability |
| 控制平面开销 | 低 | **仍保持低开销（无额外模型调用）** |
| Retry 利用效率 | 通常复用原模型（prefix reuse），但可能持续失败 | 主动切换模型，提高最终成功率 |
| 多语言/长上下文适应性 | 固定策略，不感知 context 特征 | 感知 context length 和 language |

> ✅ **优势总结**：LAAR 在不增加语义解析负担的前提下，通过轻量建模实现 accuracy-aware 调度，有效降低 TTCA。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 基于 **SCBench** 修改的 KV 查找任务（key-value retrieval）。
- 包含 100 个查询，分为两个 50 查询集合（一个用于训练 capability model，另一个用于测试）。
- 输入格式：
  - 上下文为包含随机 UUID 键值对的大规模 JSON 字符串（前缀 `"JSON data: "`）。
  - 查询要求返回特定 key 对应的 value。
- 上下文长度：**4K, 8K, 16K, 32K, 64K tokens**。
- 支持三种语言：**English, Japanese, Chinese**（通过翻译生成）。

---

### ⚙️ 实验设置

| 组件 | 配置 |
|------|------|
| 推理引擎 | **vLLM v0.16.0** |
| 硬件 | **NVIDIA A100 GPU (80GB)** ×5，10Gbps 网络互联 |
| 模型池 | 5 个异构模型：<br>- Granite3.1-2B<br>- Granite3.1-8B<br>- Phi3-mini<br>- Phi3-medium<br>- Llama3.1-Swallow-8B |
| 并发请求数 | 8 |
| 解码方式 | Deterministic decoding (`temperature=0`)，减少输出方差 |
| 重试上限 | 最多 10 次重试（retry cap R=10） |
| 路由实现 | 基于 **Envoy Endpoint Picker (EPP)** 实现 LAAR 策略 |

---

### 📈 评估指标

| 指标 | 描述 |
|------|------|
| **TTCA** | Time-to-Correct-Answer，主指标 |
| **Success Rate** | 成功获取正确答案的比例（随重试次数变化） |
| **Latency per attempt** | 单次推理延迟 |
| **Control-plane overhead** | 路由计算耗时（毫秒级） |

---

### 🆚 基线方法对比

| 基线方法 | 描述 |
|---------|------|
| **Load-aware routing** | 根据当前队列负载选择最优延迟路径，忽略准确性 |
| **Session-affinity routing** | 同一会话内始终路由到相同模型，利于 KV-cache 复用，但缺乏灵活性 |

> 所有方法共享相同的 gateway 和转发逻辑，仅替换评分函数（scoring logic），确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据

#### （1）**TTCA 改进幅度**
- LAAR 相比基线平均降低 TTCA：
  - 较 **load-aware routing** 最高提升 **31%**
  - 较 **session-affinity routing** 最高提升 **49%**
- 在多数 context length 和 language 设置下均表现更优。

#### （2）**最终成功率（Final Success Rate）**
- LAAR 在大多数设置中达到最高的最终成功率。
- 特别是在 **32K 和 64K** 场景下，session-affinity 表现较差，因其固守初始模型可能导致连续失败。
- LAAR 通过模型切换实现了更广的覆盖。

#### （3）**首答成功率（First-attempt success rate）**
- LAAR 有时首答成功率低于其他方法，但由于后续 retry 更高效，**整体 TTCA 更低**。
- 表明：**序列选择策略比单一最优更重要**。

#### （4）**高负载场景下的表现**
- 在 **64K token** 场景中，load-aware routing 因良好负载均衡表现出较低绝对延迟，甚至在某些情况下 TTCA 略优于 LAAR。
- 但其最终成功率更低，说明存在 **latency-correctness trade-off**。

---

### 🔍 消融实验（隐含分析）

虽然未明确列出消融实验表格，但从设计可推断以下关键组件作用：

| 组件 | 作用验证 |
|------|--------|
| **Success probability model (logistic regression)** | 若移除，则无法感知 accuracy 差异，退化为纯延迟驱动 |
| **Retry penalty mechanism** | 若无此机制，LAAR 将趋向重复尝试同模型，失去探索优势 |
| **轻量特征提取（length + language）** | 实验表明这些特征足以捕捉 accuracy 变化趋势，无需复杂 embedding 分析 |

> ✅ 结论：LAAR 的有效性依赖于 **capability modeling + retry-aware exploration** 的协同。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Accuracy is Speed**  
   在 long-context 场景下，更高的 accuracy 能减少 retry 次数，从而直接缩短用户等待时间 —— **accuracy 成为一种“速度”形式**。

2. **Model Rankings Are Not Stationary**  
   不同模型在不同 context length 和 language 下的表现排名会发生变化：
   - 小模型（如 Phi3-mini）在中等长度（8K–16K）可能优于大模型。
   - Llama3.1-Swallow-8B 在超过 32K 后出现“阈值崩溃”现象。
   > ❗ 固定优先级策略不再适用。

3. **轻量特征即可有效建模 capability**  
   prompt length 和 language 是 strong predictors of accuracy degradation，无需复杂语义理解即可构建有效的 capability model。

4. **Retry-awareness 是关键**  
   单纯优化首次响应延迟不足以最小化 TTCA；必须主动避免重复失败路径。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **依赖可判定 correctness** | 当前框架假设 response 正确性可通过自动指标判断（如 exact match），难以扩展至开放生成任务（如对话、摘要）。 |
| **静态 capability model** | 使用离线训练的 logistic regression，无法动态适应模型更新或漂移。 |
| **未整合 cache affinity** | 严格切换模型牺牲了 prefix reuse 带来的 prefill 加速，未来需权衡 correctness 与 cache 效率。 |
| **任务范围受限** | 目前仅验证于 retryable, task-oriented workloads（如 KV lookup），尚未推广至 multi-step agent 或 conversation flow。 |

---

### 🔮 未来工作方向

1. **Multi-objective Router Design**  
   联合优化 TTCA、cache locality 和 load balancing，例如引入权重调节机制。

2. **动态 capability adaptation**  
   在线学习模型表现变化，实时更新 success probability 估计。

3. **扩展至开放任务**  
   探索使用 LLM-as-a-judge 或 human feedback 构建 proxy success signal，将 TTCA 应用于更广泛场景。

4. **Hybrid Routing Strategies**  
   结合 LAAR 与 load-aware/session-affinity，在高负载或交互式场景中取得更好平衡。

5. **支持 multi-turn 和 stateful workloads**  
   将 TTCA 应用于整个 conversation flow，衡量 time-to-goal 而非单轮正确性。

---

> 💡 **总体评价**：该论文提出了一个深刻且实用的观点——**在长上下文服务中，accuracy 应被视为系统性能的一等公民**。LAAR 提供了一种低成本、高效益的实现路径，推动了 LLM serving 系统从“快”向“既准又快”的演进。

</details>

---

### 16. [Impact of Nonlinear Power Amplifier on Massive MIMO: Machine Learning Prediction Under Realistic Radio Channel](https://arxiv.org/abs/2604.15977)

**Authors**: Marcin Hoffmann, Pawe{\l} Kryszkiewicz  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.15977v1  

#### Abstract
M-MIMO is one of the crucial technologies for increasing spectral and energy efficiency of wireless networks. Most of the current works assume that M-MIMO arrays are equipped with a linear front end. However, ongoing efforts to make wireless networks more energy-efficient push the hardware to the li...

---

### 17. [Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents](https://arxiv.org/abs/2604.15877)

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.15877v1  

#### Abstract
As LLM agents scale to long-horizon, multi-session deployments, efficiently managing accumulated experience becomes a critical bottleneck. Agent memory systems and agent skill discovery both address this challenge -- extracting reusable knowledge from interaction traces -- yet a citation analysis of...

---

### 18. [Target-Oriented Pretraining Data Selection via Neuron-Activated Graph](https://arxiv.org/abs/2604.15706)

**Authors**: Zijun Wang, Haoqin Tu, Weidong Zhou, Yiyang Zhou, Xiaohuan Zhou, Bingni Zhang, Weiguo Feng, Taifeng Wang, Cihang Xie, Fengze Liu  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.15706v1  

#### Abstract
Everyday tasks come with a target, and pretraining models around this target is what turns them into experts. In this paper, we study target-oriented language model (LM) pretraining by introducing Neuron-Activated Graph Ranking (NAG-based Ranking), a training-free and interpretable framework for tar...

---

### 19. [A Systematic Study of Training-Free Methods for Trustworthy Large Language Models](https://arxiv.org/abs/2604.15789)

**Authors**: Wai Man Si, Mingjie Li, Michael Backes, Yang Zhang  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.15789v1  

#### Abstract
As Large Language Models (LLMs) receive increasing attention and are being deployed across various domains, their potential risks, including generating harmful or biased content, producing unsupported claims, and exhibiting vulnerabilities to adversarial attacks, have drawn significant attention. To...

---

### 20. [Disentangling Mathematical Reasoning in LLMs: A Methodological Investigation of Internal Mechanisms](https://arxiv.org/abs/2604.15842)

**Authors**: Tanja Baeumel, Josef van Genabith, Simon Ostermann  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.15842v1  

#### Abstract
Large language models (LLMs) have demonstrated impressive capabilities, yet their internal mechanisms for handling reasoning-intensive tasks remain underexplored. To advance the understanding of model-internal processing mechanisms, we present an investigation of how LLMs perform arithmetic operatio...

---

### 21. [Fusing Cellular Network Data and Tollbooth Counts for Urban Traffic Flow Estimation](https://arxiv.org/abs/2604.15782)

**Authors**: Oluwaleke Yusuf, Shaira Tabassum  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.15782v1  

#### Abstract
Traffic simulations, essential for planning urban transit infrastructure interventions, require vehicle-category-specific origin-destination (OD) data. Existing data sources are imperfect: sparse tollbooth sensors provide accurate vehicle counts by category, while extensive mobility data from cellul...

---

### 22. [Modern Structure-Aware Simplicial Spatiotemporal Neural Network](https://arxiv.org/abs/2604.15833)

**Authors**: Zhaobo Hu, Vincent Gauthier, Mehdi Naima  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.15833v1  

#### Abstract
Spatiotemporal modeling has evolved beyond simple time series analysis to become fundamental in structural time series analysis. While current research extensively employs graph neural networks (GNNs) for spatial feature extraction with notable success, these networks are limited to capturing only p...

---

### 23. [Multi-Objective Bayesian Optimization via Adaptive \varepsilon-Constraints Decomposition](https://arxiv.org/abs/2604.15959)

**Authors**: Yaohong Yang, Sammie Katt, Samuel Kaski  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.15959v1  

#### Abstract
Multi-objective Bayesian optimization (MOBO) provides a principled framework for optimizing expensive black-box functions with multiple objectives. However, existing MOBO methods often struggle with coverage, scalability with respect to the number of objectives, and integrating constraints and prefe...

---

### 24. [Think Multilingual, Not Harder: A Data-Efficient Framework for Teaching Reasoning Models to Code-Switch](https://arxiv.org/abs/2604.15490)

**Authors**: Eleanor M. Lin, David Jurgens  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.15490v1  

#### Abstract
Recent developments in reasoning capabilities have enabled large language models to solve increasingly complex mathematical, symbolic, and logical tasks. Interestingly, while reasoning models are often trained to generate monolingual text, these models have also been observed to code-switch (i.e., m...

---

### 25. [Skill-RAG: Failure-State-Aware Retrieval Augmentation via Hidden-State Probing and Skill Routing](https://arxiv.org/abs/2604.15771)

**Authors**: Kai Wei, Raymond Li, Xi Zhu, Zhaoqian Xue, Jiaojiao Han, Jingcheng Niu, Fan Yang  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.15771v1  

#### Abstract
Retrieval-Augmented Generation (RAG) has emerged as a foundational paradigm for grounding large language models in external knowledge. While adaptive retrieval mechanisms have improved retrieval efficiency, existing approaches treat post-retrieval failure as a signal to retry rather than to diagnose...

---

### 26. [SwanNLP at SemEval-2026 Task 5: An LLM-based Framework for Plausibility Scoring in Narrative Word Sense Disambiguation](https://arxiv.org/abs/2604.16262)

**Authors**: Deshan Sumanathilaka, Nicholas Micallef, Julian Hough, Saman Jayasinghe  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.16262v1  

#### Abstract
Recent advances in language models have substantially improved Natural Language Understanding (NLU). Although widely used benchmarks suggest that Large Language Models (LLMs) can effectively disambiguate, their practical applicability in real-world narrative contexts remains underexplored. SemEval-2...

---

### 27. [Evaluating SYCL as a Unified Programming Model for Heterogeneous Systems](https://arxiv.org/abs/2604.16043)

**Authors**: Ami Marowka  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.16043v1  

#### Abstract
High-performance computing (HPC) applications are increasingly executed in heterogeneous environments, introducing new challenges for programming and software portability. SYCL has emerged as a leading model designed to simplify heterogeneous programming and make it more accessible to developers. In...

---

### 28. [ProtoTTA: Prototype-Guided Test-Time Adaptation](https://arxiv.org/abs/2604.15494)

**Authors**: Mohammad Mahdi Abootorabi, Parvin Mousavi, Purang Abolmaesumi, Evan Shelhamer  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.15494v1  

#### Abstract
Deep networks that rely on prototypes-interpretable representations that can be related to the model input-have gained significant attention for balancing high accuracy with inherent interpretability, which makes them suitable for critical domains such as healthcare. However, these models are limite...

---

### 29. [Similarity-Based Bike Station Expansion via Hybrid Denoising Autoencoders](https://arxiv.org/abs/2604.15783)

**Authors**: Oluwaleke Yusuf, M. Tsaqif Wismadi, Adil Rasheed  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.15783v1  

#### Abstract
Urban bike-sharing systems require strategic station expansion to meet growing demand. Traditional allocation approaches rely on explicit demand modelling that may not capture the urban characteristics distinguishing successful stations. This study addresses the need to exploit patterns from existin...

---

### 30. [Univariate Channel Fusion for Multivariate Time Series Classification](https://arxiv.org/abs/2604.16119)

**Authors**: Fernando Moro, Vinicius M. A. Souza  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.16119v1  

#### Abstract
Multivariate time series classification (MTSC) plays a crucial role in various domains, including biomedical signal analysis and motion monitoring. However, existing approaches, particularly deep learning models, often require high computational resources, making them unsuitable for real-time applic...

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
