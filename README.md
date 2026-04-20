# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-20 07:35:59 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PINNACLE: An Open-Source Computational Framework for Classical and Quantum PINNs](https://arxiv.org/abs/2604.15645)

**Authors**: Shimon Pisnoy, Hemanth Chandravamsi, Ziv Chen, Aaron Goldgewert, Gal Shaviner, Boris Shragner, Steven H. Frankel  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.15645v1  

#### Abstract
We present PINNACLE, an open-source computational framework for physics-informed neural networks (PINNs) that integrates modern training strategies, multi-GPU acceleration, and hybrid quantum-classical architectures within a unified modular workflow. The framework enables systematic evaluation of PI...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PINNACLE: An Open-Source Computational Framework for Classical and Quantum PINNs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在解决 **Physics-Informed Neural Networks (PINNs)** 在训练过程中面临的以下核心挑战：
- **收敛困难**：由于非凸优化景观、梯度不平衡和频谱偏置（spectral bias），PINNs 难以收敛到高精度解。
- **计算成本高昂**：相比传统数值求解器，PINNs 的自动微分和多轮迭代导致浮点运算量巨大。
- **缺乏统一框架**：现有工具（如 DeepXDE、Modulus）在支持量子-经典混合架构和系统性基准测试方面存在不足。
- **量子PINNs (QPINNs) 的可扩展性瓶颈**：参数移位规则（parameter-shift rule）导致电路评估次数随参数数量线性增长，限制了实际应用。

### 提出了什么新方法或新思路
作者提出了 **PINNACLE**，一个开源的、模块化的计算框架，其核心创新包括：

- **集成多种增强策略**：将 **Random Fourier Features (RFF)**、**Random Weight Factorization (RWF)**、**严格边界条件约束**、**动态损失平衡 (loss balancing)**、**课程学习 (curriculum training)** 和 **二阶优化器 (L-BFGS)** 统一整合于一个可复现的工作流中。
- **支持 Hybrid Quantum-Classical 架构**：通过自研库 **TorQ** 实现量子层与经典神经网络的无缝集成，支持在模拟量子硬件上运行 QPINNs。
- **形式化分析 QPINNs 复杂度**：首次推导出基于参数移位规则的 **电路评估复杂度公式**，揭示其对导数阶数呈指数依赖、对可训练参数呈线性依赖。
- **多GPU分布式训练支持**：采用 **Distributed Data Parallel (DDP)** 实现跨GPU的数据并行，提升大规模问题的训练效率。

### 相比现有方法的优势
| 特性 | PINNACLE | 其他框架（如 DeepXDE, Modulus） |
|------|----------|-------------------------------|
| 开源与可复现性 | ✅ 完整代码公开，提供详细教程 | ⚠️ 部分闭源或配置复杂 |
| 量子-经典混合支持 | ✅ 支持 QPINNs 并提供复杂度分析 | ❌ 不支持或仅限简单接口 |
| 模块化设计 | ✅ 分级模块便于教学与扩展 | ⚠️ 结构较固定，定制成本高 |
| 性能分析深度 | ✅ 包含内存、时间、通信开销量化 | ⚠️ 多为功能实现，缺乏系统评估 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集 / 问题
实验基于一系列具有代表性的 PDE 基准问题，涵盖不同物理类别和数值挑战：

| 问题 | 类型 | 主要挑战 |
|------|------|--------|
| Advection Equation | 双曲守恒律 | 波传播、周期性边界、高频成分 |
| Allen-Cahn Equation | 反应扩散方程 | 尖锐界面演化、刚性动力学 |
| Inviscid Burgers Equation | 非线性双曲方程 | 梯度爆炸、激波形成 |
| Lid-driven Cavity | 不可压缩Navier-Stokes | 多尺度涡旋、高雷诺数流动 |
| Sod Shock Tube | 欧拉方程 | 激波、接触间断、稀疏波 |
| 2D Riemann Problem | 多维双曲系统 | 冲击交互、剪切层卷起 |
| 2D Gaussian Pulse in Periodic Box | 麦克斯韦方程 | 电磁波传播、长期积分稳定性 |
| Blood Flow in Stenosis | 生物流体力学 | 稀疏观测、壁面剪应力估计 |

### 实验设置和评估指标

#### 实验设置
- **硬件平台**：Apple Silicon, x86 CPU, NVIDIA A100/A6000/L40S GPU
- **软件栈**：PyTorch + 自研 TorQ 库（用于量子模拟）
- **训练流程**：Adam 初步训练 → L-BFGS 精细优化
- **并行策略**：DDP 支持最多 8-GPU 并行
- **量子模拟**：使用 TorQ 在单GPU上模拟参数化量子电路（PQC）

#### 评估指标
- **相对 L2 误差**：$\|u_{pred} - u_{ref}\|_2 / \|u_{ref}\|_2$
- **MAE / MRE**（壁面剪应力）：Mean Absolute Error / Mean Relative Error
- **Wall-clock time**：每 epoch 运行时间
- **VRAM 占用**：显存消耗
- **消融研究**：逐项移除增强技术观察性能变化

### 基线方法对比
- **经典基线**：未使用任何增强技术的 vanilla PINNs
- **文献对比**：与 Wang et al. [10, 56] 报告的结果进行横向比较
- **量子对比**：QPINNs vs. 全经典 PINNs（相同参数量或更少）
- **模拟器对比**：TorQ vs. PennyLane 默认 `.qubit` 模拟器

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 任务 | 方法 | 相对 L2 误差 | 备注 |
|------|------|-------------|------|
| Advection (c=80) | RFF + Loss Balancing + Strict BC | $1.15 \times 10^{-5}$ | 比 Wang et al. 低两个数量级 |
| Allen-Cahn | RFF + RWF + Curriculum | $2.58 \times 10^{-3}$ | 接近参考解 |
| Lid-driven Cavity (Re=3200) | Curriculum + LHS + 256×256 | — | 速度剖面匹配 DNS 数据 |
| Sod Shock Tube | RFF + Adam+L-BFGS | — | 成功捕捉激波、接触面、稀疏波 |
| Maxwell Pulse | Full config (RFF+Periodicity+Causality) | 0.04218 (@128 units) | 消融显示 RFF 最关键 |

### 与基线方法的对比结果
- **相比 vanilla PINNs**：
  - 引入 RFF 后，Allen-Cahn 方程误差下降约 **60%**
  - 加入 RWF 后，高雷诺数腔流（Re=400）成功收敛，而 vanilla 模型失败
  - 使用严格周期边界后，边界损失项可完全移除，训练更稳定

- **相比 Wang et al. [10]**：
  - 在 advection 问题上达到更低的 L2 误差（$10^{-5}$ vs $10^{-3}$）
  - 在 Allen-Cahn 上误差略高但仍处于同一量级，且指出差异集中在后期相界区域

- **QPINNs vs Classical PINNs**：
  - 在麦克斯韦方程求解中，最佳 QPINN 配置实现了 **~19% 更低的 L2 误差**
  - 参数量减少 **19%**，表明更高的参数效率
  - 但需付出巨大计算代价：单次更新需 **2535 次电路评估**

### 消融实验结果
#### （1）Maxwell 脉冲消融实验（Table 5）
| 配置 | L2 Error (128 units) | 影响 |
|------|-----------------------|------|
| Full (RFF + Pxy + Pt + Causality) | 0.04218 | 最优 |
| 缺失 Causality | 0.04272 | 几乎无影响 |
| 缺失 RFF | 0.04895 | 显著上升 |
| 缺失 RFF & Periodicity | 0.49885 | 性能崩溃 |

> **结论**：RFF 是最关键组件；时空周期性约束显著提升鲁棒性。

#### （2）QPINN 消融实验（Fig. 19）
| 因素 | 影响 |
|------|------|
| 输入缩放（scaling） | `scaleasin` > `scale`，误差差超 60% |
| 缠绕结构（entanglement） | 中等深度纠缠架构表现最好 |
| 能量正则项 | 对 QPINN 至关重要，防止“黑洞”退化解 |

---

## 4. 关键结论和发现

### 论文的主要发现
1. **PINNs 极度敏感于架构与训练选择**：没有单一配置适用于所有问题，必须根据具体 PDE 特性组合多种增强技术。
2. **RFF 和 RWF 是缓解频谱偏置的关键**：它们显著提升了对高频、尖锐特征的建模能力。
3. **严格边界条件优于软约束**：直接在网络输出中嵌入边界条件可消除梯度竞争，提高训练稳定性。
4. **动态损失平衡有效管理优化过程**：避免某一残差项主导梯度更新。
5. **课程学习对高雷诺数流动至关重要**：从低 Re 开始逐步增加难度，能引导模型进入正确解分支。
6. **QPINNs 展现出参数效率优势**：尽管计算成本极高，但在某些任务上可用更少参数获得更高精度。
7. **DDP 可有效扩展至多GPU**：在合理规模下接近线性加速，显存占用随设备数线性下降。

### 方法的局限性
- **计算成本远高于传统求解器**：即使使用 GPU 加速，达到同等精度所需 FLOPs 数量级更高。
- **QPINNs 当前不可扩展**：参数移位规则导致电路评估次数随参数线性增长，难以应用于大模型。
- **缺乏确定性误差界**：PINNs 的误差仍为随机变量，无法像有限元那样提供先验误差估计。
- **长期时间域外推能力弱**：不编码因果关系时，超出训练窗口后预测迅速失效。
- **DDP 不兼容 Jupyter Notebook**：必须使用独立脚本启动多进程训练。

### 未来工作方向
- **改进 QPINNs 微分机制**：探索替代参数移位的方法（如反向模式自动微分）以降低电路评估次数。
- **引入更多并行范式**：尝试 **Model Parallelism** 或 **Pipeline Parallelism** 来突破 DDP 的容量限制。
- **发展混合精度训练方案**：结合 non-dimensionalization 和 loss balancing 实现安全的 FP16/BF16 训练。
- **构建通用接口**：开发支持任意 PDE 输入的前端解析器，提升易用性。
- **探索物理先验更强的激活函数**：如基于格林函数或本征模态的隐式表示。
- **加强时间因果建模**：进一步优化 temporal causality 模块以支持更长时序预测。

--- 

> **总体评价**：  
> PINNACLE 不仅是一个功能强大的开源框架，更是对当前 PINNs 方法论的一次系统性梳理与实证检验。它揭示了“成功训练”背后复杂的权衡关系，并为后续研究提供了可复现的基准平台。虽然距离取代传统数值方法尚远，但它为理解机器学习与科学计算的融合路径奠定了坚实基础。

</details>

---

### 2. [Breaking the Training Barrier of Billion-Parameter Universal Machine Learning Interatomic Potentials](https://arxiv.org/abs/2604.15821)

**Authors**: Yuanchang Zhou, Hongyu Wang, Yiming Du, Yan Wang, Mingzhen Li, Siyu Hu, Xiangyu Zhang, Weijian Liu, Chen Wang, Zhuoqiang Guo, Long Wang, Jingde Bu, Yutong Lu, Guangming Tan, Weile Jia  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.15821v1  

#### Abstract
Universal Machine Learning Interatomic Potentials (uMLIPs), pre-trained on massively diverse datasets encompassing inorganic materials and organic molecules across the entire periodic table, serve as foundational models for quantum-accurate physical simulations. However, uMLIP training requires seco...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文致力于解决**大规模通用机器学习原子间势能模型**（universal Machine Learning Interatomic Potentials, uMLIPs）在训练阶段面临的巨大挑战。具体问题包括：
- **计算复杂度高**：uMLIPs 需要进行**二阶自动微分**（double-backward）以实现力匹配（force-matching），导致计算和内存开销翻倍。
- **精度要求高**：为保持量子级精度，必须使用 **FP32 单精度浮点运算**，无法利用低精度加速（如 FP16/FP8），限制了 Tensor Core 利用率。
- **通信开销大**：随着参数规模扩展至**十亿级**（billion-parameter），纯数据并行已无法满足内存需求，且缺乏支持二阶导数的高效并行框架。
- **异构任务建模难**：传统模型难以统一处理分子、晶体、催化剂等多化学域任务。

### 提出的新方法与创新
作者提出了两个核心组件来突破上述瓶颈：

#### （1）**MatRIS-MoE**：首个基于不变架构的大规模 MoE 模型
- 在原有 **MatRIS** 模型基础上引入 **Mixture-of-Experts (MoE)** 架构，支持跨异构化学领域的多任务学习。
- 采用 **element-type aware 路由机制**：每个元素类型激活其专属的 top-K 专家，提升表达能力的同时保证能量面平滑。
- 引入 **multi-head self-attention** 替代原有的 separable attention，虽然算法复杂度上升，但更适配现代 GPU 的密集矩阵计算，提高硬件利用率。
- 设计 **task-aware 特征嵌入**，融合任务、电荷、自旋及全局成分信息，对齐不同 DFT 泛函下的标签差异。

#### （2）**Janus**：首个面向 uMLIP 的高维分布式训练框架
- 提出 **FS-3D**（Fully Sharded 3 Dimensions）执行单元，统一整合三种并行策略：
  - **FSDP**（Fully Sharded Data Parallelism）
  - **FSGP**（Fully Sharded Graph Parallelism）
  - **FSEP**（Fully Sharded Expert Parallelism）
- 支持 **double-backward 自动微分** 的完整生命周期管理，动态恢复与去碎片化参数。
- 实现 **just-in-time sparse expert planning**，根据每步 token 分布动态规划专家分配，显著降低冗余通信和负载不均。
- 设计 **pipelined gradient synchronization** 和 **atom-type-aware 通信压缩**，优化梯度同步效率。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **模型容量** | 成功将 invariant 模型扩展至 **11.5B 参数**，远超此前同类模型（如 MatRIS-L 仅 10.4M active params） |
| **训练效率** | 达到 **1.2 EFLOPS** 峰值性能（LineShine 系统），维持 **>90% 并行效率**，训练时间从“周”级压缩至“小时”级 |
| **吞吐量** | 对于 11.5B 模型，归一化吞吐量达 **3201×**，相较当前最优 UMA 模型提升超过 3000 倍 |
| **通用性** | 支持分子、材料、催化、MOFs、直接空气捕获等多种下游任务，具备强大零样本泛化能力 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用一个包含 **473 million 原子构型**的多领域大规模数据集，涵盖：
  - 孤立分子（isolated molecules）
  - 周期性晶体（periodic crystals）
  - 催化表面（catalytic surfaces）
  - 分子晶体（molecular crystals）
  - 金属有机框架（MOFs）
- 包含约 **3.6 trillion 条边**（edges），构成极端边-令牌吞吐压力场景。
- 主要来源包括：`OMat24`, `OMol25`, `Opoly26`, `OC25`, `ODAC25` 等开放数据集。

### 实验设置
- **模型变体**：训练三个 MatRIS-MoE 变体：
  - Small (S): 1.09B total params, 0.19B active
  - Medium (M): 2.47B total, 0.56B active
  - Large (L): 11.5B total, 2.89B active
- **硬件平台**：
  - **CNIS**（China New-generation Intelligent Supercomputer）：基于 GPGPU，共 45K GPU
  - **LineShine**：基于 ARMv9 + HBM，共 12.4M cores
- **并行配置**：
  - FS-3D 单元大小为 8
  - GP replica size = 8
  - 利用 DP replica 扩展至全系统
- **训练细节**：
  - 全局批量大小最大达 **15K**
  - 使用 AdamW 优化器，学习率随 batch size √scaling 调整
  - 评估周期为 1000 步以上，覆盖弱缩放与强缩放测试

### 评估指标
| 指标 | 定义 |
|------|------|
| **Peak Performance** | 总 FLOPs / 训练循环耗时（不含 IO 初始化） |
| **Sustained Performance** | 总 FLOPs / 总墙钟时间（含 IO、初始化） |
| **Normalized Throughput** | `#Active Params × Dataset Size × Epochs / Training Days`，以 UMA=1 归一化 |
| **Parallel Efficiency** | 实际性能 / 理论峰值比例 |
| **Accuracy Metrics** | RMSE, MAE, F1 Score, dE, K/G Moduli, Reaction Energy Error 等跨任务指标 |

### 基线方法对比
在 Table I 中与以下主流 uMLIP 模型对比：
- **CHGNet**, **eqV2**, **ORB**, **PET**, **UMA**, **MACE-mh**, **SevenNet-Omni** 等
- 关键维度包括：是否支持 multi-task、active parameter 数量、硬件资源消耗、归一化吞吐量等

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **峰值性能** | 
| - CNIS 上 MatRIS-MoE(L) | **1.0 EFLOPS** (35.5% of peak) |
| - LineShine 上 MatRIS-MoE(L) | **1.2 EFLOPS** (24.4% of peak) |
| **并行效率** | 弱缩放下 > **90%**（LineShine 达 90.3%，CNIS 达 93.78%） |
| **持续性能**（end-to-end） |
| - CNIS | 762.3 PFLOPS (25.84%) |
| - LineShine | 1.033 EFLOPS (21.02%) |
| **归一化吞吐量** |
| - MatRIS-MoE(M) | 653–750× vs UMA |
| - MatRIS-MoE(L) | **2796–3202× vs UMA** |
| **训练速度提升** |
| - CNIS 上优化后提速 | 2.7–2.9× |
| - LineShine 上优化后提速 | **4.1–5.0×** |

### 与基线方法的对比结果
- 在相同数据集上，**MatRIS-MoE(L)** 的归一化吞吐量是 **UMA** 的 **3201.8 倍**。
- 尽管 UMA 使用 256 H200 GPU 训练 21 天，本文方法可在数小时内完成同等规模训练。
- 在 **strong scaling** 测试中，即使扩展至全机（45K GPUs / 12.4M ARM cores），仍保持 **>50% 相对效率**，验证了系统的可扩展性。

### 消融实验结果
- **系统级优化贡献分析**（Table III）显示：
  - **异步梯度同步 + 参数更新流水线** 显著减少等待时间。
  - **atom-type-aware FP16 压缩** 将 MoE all-to-all 通信量减少 **50%**，无精度损失。
  - **高性能内核优化**（neighbor gather, attention, MoE dispatch）大幅提升单卡吞吐。
  - **SDMA 加速内存搬运**（LineShine）带来最高 **1.4× 内存带宽增益**。

---

## 4. 关键结论和发现

### 主要发现
1. **uMLIP 训练可以实现 Exascale 级高效运行**：通过算法-框架-系统协同设计，首次实现了百亿参数 uMLIP 在 Exascale 超算上的高效训练。
2. **invariant 模型也能胜任超大规模建模**：尽管此前认为 equivariant 更适合高阶相互作用，本工作证明通过 MoE + self-attention 设计，invariant 架构同样可达到 SOTA 表达能力。
3. **Janus 框架具有高度可移植性**：在同一套代码下，在 GPGPU 和 ARMv9 平台均取得近线性弱缩放，验证了其跨架构适应能力。
4. **MatRIS-MoE 具备强大 out-of-the-box 泛化能力**：在未微调情况下，在多个跨域基准（如 GMTKN55, MOFSim, OC20NEB）上达到或接近 SOTA 精度。

### 方法的局限性
- **初始化开销显著**：在 Exascale 规模下，RCCL/MPI 初始化时间长达数百秒，影响端到端效率（sustained performance 仅为 peak 的 ~20–25%）。
- **MoE 动态路由仍存在负载波动风险**：尽管 JIT planning 有效缓解，但在极端稀疏或非均匀分布的任务中可能引发再平衡延迟。
- **依赖高质量大规模标注数据**：模型性能受限于 DFT 数据的质量与多样性，尤其在罕见元素或极端条件下可能存在偏差。

### 未来工作方向
- **进一步优化初始化与 I/O 开销**：探索预加载、懒初始化、异步启动等技术降低冷启动代价。
- **向 100B+ 参数迈进**：推动模型进入百亿元素级别，需发展更高效的专家共享机制与稀疏训练策略。
- **构建闭环 AI4S 工作流**：将 MatRIS-MoE 与 MD 引擎、DFT solver 深度集成，形成“主动学习-模拟-反馈”闭环。
- **推广至其他物理模拟领域**：将 Janus 框架拓展至电子结构预测、相场模拟、量子动力学等需要高阶导数的科学计算场景。

> ✅ **总结一句话**：  
> 本文通过提出 **MatRIS-MoE + Janus** 协同体系，成功打破了十亿参数 uMLIP 的训练壁垒，在 Exascale 超算上实现了 **EFLOPS 级训练性能** 与 **千倍以上吞吐提升**，为 AI-for-Science 奠定了新一代基础模型基础设施。

</details>

---

### 3. [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)

**Authors**: Qwen Team  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 9.0  
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
Qwen3.5-Omni 致力于构建一个**原生全模态（native omni-modal）大模型**，解决当前多模态系统中存在的以下关键问题：
- **被动感知-响应范式**：多数模型仅能处理输入并生成输出，缺乏主动行为能力（agentic behavior）。
- **跨模态对齐不稳定**：在流式语音合成中，由于文本与语音token编码效率不一致，导致跳词、发音错误等问题。
- **长上下文建模能力不足**：难以支持超过数万token的超长文本、音频或视频理解。
- **实时交互延迟高**：缺乏低延迟的端到端流式推理架构。

### 提出的新方法与创新
1. **Hybrid Attention Mixture-of-Experts (MoE) 架构**
   - 在 Thinker 和 Talker 两个模块均采用 Hybrid MoE 设计，提升参数扩展性和计算效率，尤其适用于长序列建模。

2. **256k 超长上下文支持**
   - 支持高达 256k token 的输入长度，可处理 **10小时音频** 或 **400秒 720P 视频（1FPS）**，显著超越主流模型。

3. **ARIA (Adaptive Rate Interleave Alignment) 技术**
   - 创新性地动态对齐文本与语音单位，在流式解码过程中强制保持“累积语音-文本token比率不超过全局比例”，有效缓解因编码速率差异引起的不稳定问题。

4. **Thinker-Talker 双轨协同架构升级**
   - Thinker 负责多模态理解与文本生成；Talker 基于 Thinker 输出进行条件化语音生成。
   - 引入 Chunked Prefilling 和 Streaming Generation，实现端到端低延迟交互。

5. **零样本语音定制（Zero-Shot Voice Cloning）**
   - 支持通过用户提供语音样例进行个性化语音克隆，并可在多语言场景下迁移。

6. **新兴能力：Audio-Visual Vibe Coding**
   - 首次观察到模型可以直接根据音视频指令生成可执行代码，无需外部编排器，体现真正的原生智能体行为。

### 相比现有方法的优势
| 维度 | Qwen3.5-Omni 优势 |
|------|------------------|
| **模态统一性** | 原生联合训练 text, image, audio, video，非拼接式架构 |
| **交互实时性** | 支持语义中断（semantic interruption）、自然轮转对话 |
| **语音质量** | ARIA 显著提升语音流畅度与稳定性，WER 更低 |
| **多语言能力** | 支持 **113种语言/方言** 的语音识别，**36种语言** 的语音合成 |
| **工具调用与代理行为** | 自主调用 WebSearch、FunctionCall，具备闭环任务执行能力 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 模态类别 | 数据集名称 | 描述 |
|--------|-----------|------|
| **预训练数据** | - | 包含异构图文对、超 **1亿小时音视频内容** |
| **音频-文本对** | Qwen3-ASR 生成的 40M 小时数据 | 多语言监督数据，增强泛化能力 |
| **图像-文本对** | SigLIP 训练数据 | 来自 Qwen3.5 视觉编码器 |
| **语音合成训练** | >20M 小时多语言语音数据 | 支持指令跟随语音生成 |
| **评估基准** | 见下表 | 覆盖理解、生成、推理、交互等多维度 |

### 实验设置与评估指标

#### 主要变体
- `Qwen3.5-Omni-Plus`：高性能版本
- `Qwen3.5-Omni-Flash`：轻量高效版本

#### 评估维度与指标
| 任务类型 | 评估指标 | 典型数据集 |
|--------|---------|----------|
| **Audio → Text** | WER（越低越好） | Fleurs, Common Voice, LibriSpeech |
| **Speech Translation (S2TT)** | BLEU（越高越好） | Fleurs top59 |
| **Audio Understanding** | 准确率 Accuracy | MMAU, MMSU, RUL-MuchoMusic |
| **End-to-End Dialogue** | U/R/O 分数 | VoiceBench, URO-Bench-Pro |
| **Video Understanding** | 准确率 | Video-MME, MLVU, MVBench |
| **Zero-Shot TTS** | WER（内容一致性），SIM（说话人相似度） | SEED-TTS, TTS multilingual test set |
| **Cross-Lingual Voice Cloning** | Mixed Error Rate (WER/CER) | Cross-Lingual Benchmark |
| **Custom Voice Generation** | WER | Internal multilingual test set |

#### 基线方法对比
- **Gemini-3.1 Pro**：作为主要竞争者，尤其在音频与音视频任务上对比
- **GPT-4o-Transcribe**, **ElevenLabs**, **MiniMax-Speech**, **CosyVoice 系列**：用于语音生成质量比较
- **Qwen3-Omni**：作为前代模型，验证迭代改进效果

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 类别 | 指标 | Qwen3.5-Omni-Plus | 最优基线 | 结果说明 |
|------|-----|--------------------|----------|----------|
| **Audio → Text (ASR)** | Avg WER (FLEURS) | **6.6%** | Gemini-3.1 Pro (7.3%) | 显著领先，尤其粤语（2.2% vs 6.3%） |
| **S2TT (en→xx)** | Avg BLEU | **33.8** | Gemini-3.1 Pro (31.8) | 全面胜出 |
| **S2TT (xx→zh)** | Avg BLEU | **38.9** | Gemini-3.1 Pro (39.4) | 接近最优，部分亚洲语言反超 |
| **Audio Understanding** | MMAU | **82.2** | Gemini-3.1 Pro (81.1) | SOTA |
| | MMSU | **82.8** | Gemini-3.1 Pro (81.3) | SOTA |
| | RUL-MuchoMusic | **72.4** | Gemini-3.1 Pro (59.6) | 大幅领先 |
| **End-to-End Dialogue** | VoiceBench Score | **93.1** | Gemini-3.1 Pro (88.9) | 显著更优 |
| **Zero-Shot TTS (SEED-TTS)** | WER (test-en) | **1.26** | CosyVoice3 (1.45) | 当前最佳 |
| **Cross-Lingual Voice Cloning** | zh→ko 错误率 | **4.03** | CosyVoice3 (14.4) | 下降约 72% |
| **First-Packet Latency** | Audio Input (Plus) | **435ms** | — | 支持实时交互 |
| | Video Input (Plus) | **651ms** | — | — |

### 与基线方法对比结果
- 在 **215项音频与音视频子任务** 上达到 SOTA 或接近 SOTA。
- 在 **音频理解、语音识别、语音翻译** 方面全面超越 Gemini-3.1 Pro。
- 在 **综合音视频理解** 上与 Gemini-3.1 Pro 持平。
- 在 **多语言语音生成** 中，WER 在 22/29 种语言中取得最低值。
- 在 **跨语言语音克隆** 中，10/12 方向表现最优。

### 消融实验结果（隐含分析）
虽然未明确列出消融表格，但从文中可推断关键组件影响：
- **ARIA 的引入**：显著降低语音合成中的跳词与错读现象，提升自然度与鲁棒性。
- **Hybrid MoE + GDN 模块**：大幅减少 KV Cache I/O 开销，提高长序列吞吐量。
- **Timestamp 文本化插入**：相比直接使用 TM-RoPE，提升了长视频时间感知精度。
- **On-Policy Distillation**：缩小了音频输入与文本输入下的响应质量差距。
- **Interaction-Aligned RL**：改善多轮对话中的人设一致性、指令遵循退化问题。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **原生全模态训练可行且强大**  
   Qwen3.5-Omni 在不牺牲文本与视觉能力的前提下，实现了强大的音频与音视频理解与生成能力。

2. ✅ **ARIA 是稳定流式语音合成的关键**  
   动态对齐机制解决了传统 dual-path 架构中的同步难题，为高质量实时语音交互提供保障。

3. ✅ **长上下文 + 流式处理 = 真实世界可用性**  
   支持长达 10 小时音频的理解与互动，使模型可用于会议记录、教育讲解等真实场景。

4. ✅ **Emergent Capability：Audio-Visual Vibe Coding**  
   模型能直接从音视频指令中提取需求并生成代码，是迈向通用智能体的重要一步。

5. ✅ **多语言语音能力达到行业领先水平**  
   特别是在中文方言、日韩语、东南亚语言上的表现优于主流商业系统。

### 方法的局限性
- **计算资源消耗较高**：尤其是 Plus 版本，部署成本较大。
- **极端噪声环境下的鲁棒性未充分验证**：如嘈杂背景、多人重叠语音等。
- **情感表达仍依赖提示控制**：虽支持情绪调节，但细粒度情感建模仍有提升空间。
- **ARIA 对非常规语速适应性待测**：是否能在极快/极慢朗读中保持对齐尚需更多测试。

### 未来工作方向
- 进一步优化 **Flash 版本的性能-延迟权衡**，推动边缘设备部署。
- 扩展 **更多语言与方言** 的语音支持，覆盖低资源语言。
- 加强 **具身智能（Embodied AI）集成**，结合机器人动作执行。
- 探索 **双向语音交互中的意图预测与打断检测机制**。
- 深化 **Audio-Visual Vibe Coding** 能力，构建可视化编程接口。

---

> 📌 **总结一句话**：  
> Qwen3.5-Omni 是首个真正意义上的 **原生全模态智能体模型（native omni-agent）**，不仅在理解与生成上达到 SOTA，更展现出自主行为、实时交互与跨模态编程等高级能力，标志着多模态大模型向 AGI 迈出关键一步。

</details>

---

### 4. [Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning](https://arxiv.org/abs/2604.16029)

**Authors**: Jiaxi Bi, Tongxu Luo, Wenyu Du, Zhengyang Tang, Benyou Wang  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 8.5  
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
在 **Large Reasoning Models (LRMs)** 中，**Parallel Reasoning**（并行推理）通过生成多条独立的推理路径并聚合结果来提升准确性。然而，这种方法计算成本极高，因为许多路径从早期就已出错（early errors），却仍被完整生成，造成大量算力浪费。

现有研究尝试进行 **path pruning**（路径剪枝），即在前缀阶段识别并终止无望路径，但缺乏统一框架，方法碎片化，且多数依赖外部模型或非学习型启发式规则，效率与效果受限。

---

### 提出了什么新方法或新思路
本文提出以下核心创新：

#### （1）首个系统性的 **path pruning 分类法（taxonomy）**
基于两个维度对现有方法进行分类：
- **信号来源（Source）**：Internal vs. External
- **可学习性（Learnability）**：Learnable vs. Non-learnable

由此划分出四类方法：
| | Non-Learnable | Learnable |
|---|---|---|
| **External** | Type I: 表面启发式（如相似度） | Type II: 外部判别器（如 PRM） |
| **Internal** | Type III: 原生置信度（如 perplexity） | **Type IV: STOP（本文提出）** |

> **关键洞察**：Type IV（内部 + 可学习）是理想范式，但此前未被探索。

#### （2）提出 **STOP (Super TOken for Pruning)**
- 首个实现 **Type IV 范式** 的方法。
- 在 LRM 内部插入一个轻量级模块，利用 **内部状态（如 KV Cache）** 和 **可训练参数** 来预测路径潜力。
- 包含三个组件：
  - `[STOP]` 特殊 token
  - Critique Adapter（LoRA）
  - Classification Head

设计原则：**非侵入式（non-invasive）**，原模型冻结，仅微调少量参数。

---

### 相比现有方法的优势
| 维度 | STOP | 其他方法 |
|------|------|---------|
| **效率** | 极低延迟（复用 KV Cache，无需重编码） | Type II 需额外模型推理；Type I 有计算瓶颈 |
| **有效性** | 更早、更准地识别失败路径 | 外部方法信息受限；固定规则泛化差 |
| **通用性** | 支持数学、科学、逻辑谜题等多种任务 | Type II 多为领域专用（如数学 PRM） |
| **部署友好** | “Plug-and-play”，仅增加少量参数 | Type II 需维护双模型，VRAM 占用翻倍 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **数学类**：
  - AIME24, AIME25（美国数学邀请赛）
  - BRUMO25, HMMT25（高中数学竞赛）
- **科学类**：
  - GPQA-D (GPQA Diamond)，高难度 STEM 问答
- **逻辑推理类**：
  - ZebraLogic（逻辑网格谜题）
- **竞赛场景**：
  - AIMO3（带工具使用的数学竞赛）

> 所有训练数据均经过去污染处理，确保测试集无泄露。

---

### 实验设置和评估指标

#### 模型规模
覆盖 **1.5B 到 20B 参数** 的多种 LRM：
- DS-Qwen-2.5-1.5B / 7B
- DS-Qwen-3-8B
- GPT-OSS-20B

#### 标准化协议
- 每个 query 生成 **64 条初始路径**
- 在 **2048 token** 处检查并保留 top-8 进行完成
- 使用 **zero-shot CoT prompt**

#### 评估指标
1. **avg@mlk**：从 k 条中选 m 条完成后的平均准确率  
   > 若高于 baseline avg@k，则说明剪枝有效提升了答案密度。
2. **Total Tokens**：衡量计算开销，计算相对减少量 Δ%
3. **Throughput (tok/s)**：系统吞吐量
4. **Latency / Check**：每次剪枝判断的延迟

---

### 基线方法对比
| 类型 | 方法 | 描述 |
|------|------|------|
| Type I | SlimSC | 基于语义冗余（Jaccard 相似度）剪枝 |
| Type II | LaBoR, DeepPrune | 使用外部 Process Reward Model (PRM) 评分 |
| Type III | DeepConf, AdaDec | 使用内部统计量（如 perplexity, entropy）作为信心信号 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 方法 | AIME24 (1.5B) | AIME25 (7B) | GPQA-D (20B) |
|------|---------------|-------------|--------------|
| No Pruning | 30.10% @782.3k | 39.67% @703.0k | 65.55% @277.2k |
| Type I (SlimSC) | 32.50% @325.9k (**-58.3%**) | 39.17% @317.6k | — |
| Type II (LaBoR) | 32.92% @210.6k (**-73.1%**) | 41.67% @202.6k | 68.43% @145.9k |
| Type III (DeepConf) | 32.92% @210.6k | 41.67% @202.6k | 68.43% @145.9k |
| **STOP (Ours)** | **37.92% @204.3k (-73.9%)** | **42.50% @197.5k** | **77.46% @143.4k (-48.3%)** |

> ✅ STOP 在所有任务上均显著优于基线，**准确率最高提升近 8 个百分点**，同时 **节省超过 70% 的 token 开销**。

---

### 与其他方法的关键对比

#### （1）相比 Type II（外部 PRM）
- **性能更强**：即使将 PRM 在相同数据上重新训练（Type II-retrain），STOP 依然胜出（见 Table 13）
- **速度更快**：STOP 推理延迟仅 **0.20s**，而 Type II 达 **1.13s**（+465%）
- **内存更省**：无需部署第二个大模型

#### （2）相比 Type III（内部非学习）
- **适应性强**：STOP 学会捕捉复杂错误模式，而非依赖固定公式
- **泛化更好**：在 ZebraLogic 上提升 **+3.5%**，证明其超越数学领域

---

### 消融实验结果

#### （1）监督信号质量（Table 3）
| 监督方式 | AIME24 avg@8l64 | Cons@N |
|--------|----------------|--------|
| Hard Label (K=1) | 35.42% | 46.67% |
| **Soft Label (K=32, MC)** | **36.67%** | **53.33%** |

> ✅ **Monte Carlo 估计的软标签** 显著降低方差，提升训练稳定性。

#### （2）Critique Adapter 必要性（Table 4）
| 配置 | AIME24 avg@8l64 |
|------|----------------|
| 仅有 Linear Head | 31.67% |
| **+ LoRA Adapter** | **36.67%** |

> ✅ 原始隐藏状态不适合直接用于价值判断，需要适配器进行特征转换。

#### （3）设计敏感性分析
- 最佳 `[STOP]` token 数量：**4–6 个**
- 最佳 LoRA Rank：**128 左右即可，过大反而下降**

> ✅ STOP 对超参不敏感，易于部署。

---

## 4. 关键结论和发现

### 主要发现
1. **Type IV 是最优范式**：结合 **Internal + Learnable** 的方法能最高效地识别早期错误。
2. **STOP 实现了最佳权衡**：在多个模型和任务上，**一致性地提升 accuracy 并降低 >70% token 消耗**。
3. **STOP 具备强扩展性**：在不同 compute budget 下表现稳健，尤其适合资源受限场景。
4. **提出了实用的缩放定律**：
   $$
   \gamma^{-1} = a C^b L_{\text{prefix}}^c L_{\text{task}}^d
   $$
   可指导用户根据预算自动选择最优保留比例。

---

### 方法的局限性
- **极端规模验证不足**：目前最大只测到 20B，尚未在 70B+ 或 N>1000 场景下验证。
- **单阶段剪枝**：当前仅支持固定位置的一次性剪枝，未探索多阶段动态剪枝。
- **依赖高质量训练数据构建**：MC rollouts 构建监督信号有一定前期成本（约 40–75 GPU 小时）。

---

### 未来工作方向
1. **Progressive Multi-Stage Pruning**：级联式剪枝（64→32→16），逐步聚焦。
2. **加速 RL 训练**：在 PPO/GRPO rollout 阶段使用 STOP 提前终止低价值轨迹，提高训练信号密度。
3. **动态检查点机制**：让模型自主决定何时进行剪枝判断，而非固定长度。
4. **跨任务迁移能力增强**：探索更通用的 pruning policy，减少 per-task 微调需求。

---

> 🔗 **代码、数据与模型已开源**：https://bijiaxihh.github.io/STOP

</details>

---

### 5. [CroSatFL: Energy-Efficient Federated Learning with Cross-Aggregation for Satellite Edge Computing](https://arxiv.org/abs/2604.15779)

**Authors**: Nan Yang, Bahman Javadi, Rodrigo Neves Calheiros, David Boland, Philip Leong  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.15779v1  

#### Abstract
Low Earth Orbit (LEO) mega-constellations extend the cloud-to-edge continuum into space, enabling satellite edge computing. However, Federated Learning (FL) in this environment is fundamentally energy-constrained due to dynamic inter-satellite connectivity, heterogeneous onboard computing hardware, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**CroSatFL: Energy-Efficient Federated Learning with Cross-Aggregation for Satellite Edge Computing**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Low Earth Orbit (LEO)** 卫星 mega-constellations（如 Starlink）中部署 **Federated Learning (FL)** 面临以下核心挑战：
- **能源受限**：卫星供电有限，计算与通信能耗需严格控制。
- **地面站（GS）瓶颈**：GS 可见窗口短、带宽低、传输能耗高，频繁上下行会严重拖慢训练。
- **动态连接性**：激光星间链路（LISL）拓扑时变，且受轨道几何限制（fan-out 有限）。
- **硬件异构性**：卫星间存在 CPU-only 与 GPU-accelerated 的混合配置，导致训练速度差异大，易出现 **straggler**（拖后腿节点）。

传统 FL 框架（如 FedSyn、FedLEO）仍依赖 GS 进行全局聚合，或将卫星视为同质设备，无法适应真实 LEO 动态环境。

---

### 🚀 提出的新方法：CroSatFL
提出一种**完全在轨（fully on-orbit）的分层联邦学习框架**，三大核心机制：

#### （1）**StarMask：基于强化学习的资源感知聚类**
- 将卫星划分为多个 LISL 可达、硬件能力均衡的集群（cluster）。
- 使用 **RL + Action Masking** 确保聚类满足：LISL 连通性、主节点 fan-out 容量、硬件一致性等约束。
- 输出稳定、低延迟、能量高效的集群结构。

#### （2）**Skip-One：轻量级同步优化机制**
- 在每个边缘轮次（edge round），允许集群主节点最多跳过一个临时慢节点（straggler）。
- 通过效用函数权衡：延迟降低、能耗节省、公平性保障。
- 避免因单个慢节点阻塞整个集群，同时防止长期排除某卫星。

#### （3）**Random-k Cross-Aggregation：拓扑感知的跨集群模型融合**
- 利用瞬时存在的跨轨道 LISL，在集群主之间进行随机采样式的模型交换。
- 不增加每轮耗时，实现全局模型一致性传播（gossip-like）。
- 所有中间聚合均在轨完成，**仅在训练开始广播初始模型、结束时回传最终模型**，大幅减少对 GS 的依赖。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 FedSyn, FELLO） | CroSatFL |
|------|-------------------------------|--------|
| **GS 使用频率** | 每轮都需 GS 聚合 | 仅初始化 & 最终回传（2 次） |
| **通信开销** | 大量高成本 RF 上下行 | 主要使用低能耗 LISL |
| **能效** | 忽视异构硬件与等待能耗 | 联合优化计算、通信、等待时间 |
| **鲁棒性** | 易被 straggler 拖累 | Skip-One 缓解瞬时延迟 |
| **可扩展性** | 依赖 GS 同步 | 完全分布式、适应动态拓扑 |

---

## 2. 核心实验方法和设置

### 📊 数据集
在三种标准图像分类任务上验证：
- **MNIST**（手写数字）
- **CIFAR-10**（自然小图）
- **EuroSAT**（遥感土地覆盖分类）

考虑 **IID 与 non-IID** 两种数据分布，测试泛化能力。

---

### ⚙️ 实验设置
| 参数 | 设置 |
|------|------|
| **星座模型** | Walker-Delta 构型（类 Starlink） |
| **卫星数量** | 720 颗，分布在 36 个轨道面，每面 20 颗 |
| **高度 / 倾角** | 570 km / 70° |
| **GS 位置** | 澳大利亚堪培拉 |
| **LISL 距离范围** | 659–1700 km（对应最大 cluster size 2–10） |
| **硬件比例** | 50% CPU-only, 50% GPU-equipped（基于真实 Space Edge One 轨道实测 trace） |
| **参与客户端** | 随机选取 40 颗作为 FL 客户端 |
| **聚类数 K** | 9 个 cluster |
| **Main Rounds (G)** | 1 |
| **Edge Rounds (R)** | 40 |
| **本地训练** | ResNet-18，每轮 10 个 epoch |

---

### 📈 评估指标
- **模型准确率（Accuracy）**
- **总能耗（Total Energy）**：含训练（computation）与通信（transmission）
- **端到端训练时间（End-to-end Training Time）**
- **GS 通信次数与能耗**
- **等待时间（Waiting Time）**：因无通信机会而空转的时间
- **消融实验**：验证 StarMask、Skip-One、Cross-Aggregation 的独立贡献

---

### 🆚 基线方法对比
| 方法 | 类型 |
|------|------|
| **FedSyn** | 标准同步 FedAvg，GS 中心化聚合 |
| **FedLEO** | LEO 自适应，基于 sink 卫星调度 |
| **FELLO** | 光学 LISL 聚类 + 边缘选择 |
| **FedSCS** | 能量感知客户端选择 |
| **FedOrbit** | 使用 block minifloat 算术降低计算开销 |

---

## 3. 主要实验结果和性能指标

### 📈 准确率表现（Figures 2 & 3）
- 在 **IID 设置下**：
  - CroSatFL 在 **CIFAR-10（80.09%）和 EuroSAT（89.49%）** 上达到最高精度。
  - 与其他方法持平或略优。
- 在 **non-IID 设置下（α=0.5）**：
  - 多数基线性能下降明显。
  - CroSatFL 保持稳健收敛，接近最优水平，显示其对数据异构的鲁棒性。

> ✅ 结论：**未牺牲模型性能的前提下实现了能效提升。**

---

### 💡 能耗与效率对比（Figure 4 & Table II）

| 指标 | CroSatFL vs. 最佳基线 |
|------|------------------------|
| **GS 通信次数** | **从 3200 次 → 18 次**（↓ > 两个数量级） |
| **GS 传输能耗** | **601.60 kJ → 99.70 kJ**（↓ ~6×） |
| **训练能耗（计算）** | **1080 kJ → 179.18 kJ**（↓ ~6×） |
| **等待时间** | **从数百小时 → 7.89 小时**（↓ > 99%） |
| **端到端训练时间** | 显著缩短（见 Fig. 4） |

> ✅ 关键发现：**将通信重心从 GS 转向 LISL，并通过 Skip-One 减少同步等待，是能效提升的关键。**

---

### 🔬 消融与敏感性分析（Figure 5）
- 在不同硬件构成下（All-CPU / Half-Mixed / All-GPU）：
  - **FedOrbit** 因强制全参与，受最慢节点制约，性能波动大。
  - **CroSatFL** 利用 Skip-One 动态规避 straggler，更好利用 GPU 加速能力。
  - 随着 GPU 比例上升，CroSatFL 的能耗与时延优势进一步放大。

> ✅ 表明：**Skip-One 对硬件异构环境具有强适应性。**

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **完全在轨 FL 是可行且高效的**：通过将所有中间聚合移至空间，可彻底摆脱 GS 可见性瓶颈。
2. **联合优化策略至关重要**：StarMask（聚类）、Skip-One（调度）、Random-k（跨集群融合）三者协同，显著降低能耗与延迟。
3. **LISL 应成为 FL 主干通道**：相比高成本 GS 链路，LISL 更适合高频次、低功耗的模型交换。
4. **轻量级 straggler 缓解优于复杂容错机制**：Skip-One 以极低开销实现有效加速，兼顾公平性。

---

### ⚠️ 局限性
- 当前假设 **主节点迁移可控**，未深入建模主节点故障或链路中断恢复机制。
- **未集成自适应压缩或量化技术**（如 FedOrbit 的 block minifloat），未来可进一步压缩通信负载。
- 实验基于仿真轨道动力学，尚未在真实在轨系统中部署验证。

---

### 🔮 未来工作方向
- 引入 **adaptive compression** 与 **fault-tolerant scheduling** 提升鲁棒性。
- 探索 **multi-main-round** 场景下的长期稳定性。
- 在更复杂的 **onboard accelerators（如 FPGA, NPU）** 上验证性能。
- 扩展至 **multi-task learning** 与 **cross-domain satellite AI** 场景。

---

## 总结
> **CroSatFL 是首个实现“全程在轨、零轮级 GS 交互”的高效卫星联邦学习框架**。它通过 **StarMask + Skip-One + Random-k** 三位一体设计，在不损失精度的前提下，将 **GS 通信减少两个数量级以上，能耗降低约 6 倍，训练时间显著缩短**，为未来大规模 LEO 星座中的可持续 AI 训练提供了切实可行的技术路径。

</details>

---

### 6. [Zero-Shot Scalable Resilience in UAV Swarms: A Decentralized Imitation Learning Framework with Physics-Informed Graph Interactions](https://arxiv.org/abs/2604.15762)

**Authors**: Huan Lin, Lianghui Ding  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 8.5  
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
该论文针对大规模 **UAV swarm** 在遭遇严重节点失效后引发的 **Communication Network Split (CNS)** 问题，即网络被分割为多个不连通子网，导致全局通信中断。传统方法面临以下挑战：
- **Centralized recovery** 需要全局拓扑信息，在严重碎片化时通信开销大、难以部署；
- **Decentralized heuristics** 规则固定，适应性差，尤其在拓扑剧烈变化时表现不佳；
- **Multi-Agent Reinforcement Learning (MARL)** 方法存在**可扩展性差**、**方向敏感交互建模不足**、以及**不同损伤程度下训练不稳定**等问题。

### 提出的新方法与创新思路
作者提出 **Physics-informed Graph Adversarial Imitation Learning (PhyGAIL)**，一种基于 **CTDE (Centralized Training with Decentralized Execution)** 范式的去中心化恢复框架，其核心创新包括：

#### （1）**可扩展的去中心化恢复框架**
- 构建**有界局部交互图 (bounded local interaction graph)**，通过异构节点建模（active/damaged/virtual center）和空间 K-NN 掩码机制，确保每个 UAV 的观测维度与全局 swarm 规模 $N$ 无关。
- 支持**零样本迁移 (zero-shot transfer)**：在 20-UAV 上训练的策略可直接部署到高达 500-UAV 的 swarm，无需微调。

#### （2）**物理感知的图神经网络交互模型 (PhyGNN)**
- 引入 **physics-gated message passing**，显式编码**吸引力 (attraction)** 和**排斥力 (repulsion)**，将方向性物理交互（如吸引、避障）融入消息传递过程。
- 消息由三部分控制：**吸引力门 (attract gate)**、**排斥力门 (repulse gate)** 和**可学习作用力强度 (force strength)**，提供物理合理的归纳偏置 (inductive bias)，提升协调的安全性和稳定性。

#### （3）**场景自适应的模仿学习策略 (Scenario-Adaptive Imitation Learning)**
- 结合 **GAIL (Generative Adversarial Imitation Learning)** 提供密集的 step-level 模仿奖励，引导策略学习专家行为。
- 设计**专家归一化时间奖励 (expert-normalized temporal reward)**，动态调整成功奖励，避免因不同损伤程度导致的 episode 长度差异对价值函数造成偏差，提高训练鲁棒性。

### 相比现有方法的优势
- **更强的可扩展性**：局部有界图设计支持零样本迁移到更大规模 swarm。
- **更优的物理协调能力**：PhyGNN 显式建模方向性交互，优于传统方向不可知的 GNN。
- **更高的训练稳定性**：结合模仿学习与专家归一化奖励，缓解稀疏奖励和长周期任务的学习困难。
- **综合性能更优**：在重连可靠性、恢复速度、运动安全性和运行效率之间取得最佳平衡。

---

## 2. 核心实验方法和设置

### 数据集与仿真环境
- 所有实验均在自研的 **2D UAV swarm simulator** 中进行。
- **通信半径** $D_{\text{comm}} = 120 \text{m}$，**最大速度** $v_{\text{max}} = 10 \text{m/s}$，**控制步长** $\Delta t = 0.1 \text{s}$。
- 测试 swarm 规模 $N \in \{20, 50, 100, 200, 500\}$，对应地图边长分别为 320–1600 m。
- 损伤比例 $p = N_{\text{dmg}} / N$ 从 0.05 到 0.95 变化（$N=20$ 时至 0.90）。
- 每种配置重复 50 次独立实验取平均。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Convergence Rate** | 成功恢复连通性的比例 |
| **Average Recovery Time** | 从损坏到恢复连通的平均时间（秒） |
| **Average Collisions per UAV** | 恢复过程中每架 UAV 的平均碰撞次数（距离 < 10m） |
| **Runtime Overhead** | 响应延迟、总求解时间、推理耗时 |

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **Centralized** | CR-MGC, DEMD, GDR-TS |
| **Decentralized Heuristics** | center-fly, HERO, SIDR |
| **MARL-based** | MADDPG-APF |
| **Ablation Variants** | 替换 PhyGNN 为 GCN/GAT/GraphSAGE，移除虚拟中心、受损邻居、KNN、GAIL、专家时间奖励等 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（$N=100,200,500$ 平均）
| Algorithm | Conv. Rate | Rec. Time (s) | Collisions/UAV | Overall Rank |
|----------|------------|----------------|------------------|---------------|
| **PhyGAIL** | **1.000** | **20.14** | 0.161 | **2.061** |
| DEMD | 0.990 | 26.55 | 1.851 | 3.956 |
| CR-MGC | 0.949 | 33.84 | 1.324 | 4.462 |
| GDR-TS | 0.975 | 36.48 | 1.664 | 4.471 |
| SIDR | 0.472 | 49.36 | **0.131** | 4.731 |
| center-fly | 0.998 | 28.95 | 18.994 | 4.798 |
| MADDPG-APF | 0.606 | 49.39 | 0.399 | 4.904 |
| HERO | 0.276 | 67.56 | 1.287 | 6.617 |

> ✅ **PhyGAIL 是唯一实现 100% 收敛率的方法**，且恢复时间最短、安全性高。

### 与基线方法对比结果
- **收敛性**：PhyGAIL 在所有规模和损伤比例下均保持完美收敛；而 SIDR、HERO、MADDPG-APF 在大规模高损伤下迅速退化。
- **恢复速度**：PhyGAIL 恢复时间显著优于大多数方法，仅在极高损伤下略慢于 DEMD，但整体更稳健。
- **安全性**：碰撞数远低于 center-fly、CR-MGC 等集中式方法，与 SIDR 相当但收敛性更好。
- **运行效率**：每步响应时间低（$N=500$ 时为 17.61 ms），远低于 CR-MGC (4901.74 ms)、DEMD (16032.39 ms)、GDR-TS (69241.05 ms)。

### 消融实验结果（Ablation Study）
| Ablated Component | 影响 |
|--------------------|------|
| **Replace PhyGNN → GCN** | $N=500$ 时收敛率下降 0.673，恢复时间增加 64.90s |
| **Remove virtual center** | $N=500$ 时收敛率下降 0.640，恢复时间增加 65.69s |
| **Remove damaged neighbors** | $N=500$ 时收敛率下降 0.433，恢复时间增加 50.88s |
| **Remove expert time reward** | $N=500$ 时收敛率下降 0.607，恢复时间增加 57.84s |
| **Remove GAIL reward** | 对最终性能影响较小，但训练初期收敛变慢 |
| **Remove KNN selection** | 仅在 $N=500$ 时显现影响，说明其对可扩展性重要 |

> 🔍 **关键发现**：PhyGNN 和虚拟中心对大规模鲁棒性至关重要；专家时间奖励决定恢复效率；GAIL 加速训练但非决定性。

---

## 4. 关键结论和发现

### 主要发现
1. **零样本可扩展性可行**：通过**有界局部图感知**和**物理感知交互建模**，可在小规模训练后直接迁移到大规模 swarm。
2. **物理归纳偏置有效**：显式建模**吸引力/排斥力**的 PhyGNN 比标准 GNN 更适合连续空间协调任务，提升稳定性和安全性。
3. **训练策略决定上限**：**专家归一化时间奖励**是实现高效恢复的关键，避免了长周期任务中的奖励偏差。
4. **综合性能最优**：PhyGAIL 在**重连可靠性、恢复速度、运动安全、运行效率**四方面达到最佳权衡。

### 方法的局限性
- 当前基于 2D 平面假设，未考虑高度变化或复杂 3D 地形。
- 依赖预设的 **virtual center**，在无先验参考点的场景中需额外机制生成。
- 仿真环境中未模拟通信延迟、丢包或传感器噪声等现实因素。
- 训练仍依赖中央收集专家轨迹，虽执行去中心化，但训练阶段非完全分布式。

### 未来工作方向
- 扩展至 **continuous failure scenarios**（持续而非一次性故障）。
- 考虑 **obstacle-constrained environments** 和 **dynamic obstacles**。
- 进一步降低机载资源消耗，适配更严格的嵌入式平台。
- 将 **physics-informed graph interaction** 模型推广至其他连续空间多智能体任务（如编队飞行、协同搬运）。

--- 

> 📌 **总结**：PhyGAIL 通过融合**有界局部感知**、**物理驱动的消息传递**和**场景自适应模仿学习**，实现了高性能、高鲁棒、可扩展的去中心化 UAV swarm 恢复框架，为大规模自主系统韧性提供了新范式。

</details>

---

### 7. [LACE: Lattice Attention for Cross-thread Exploration](https://arxiv.org/abs/2604.15529)

**Authors**: Yang Li, Zirui Zhang, Yang Liu, Chengzhi Mao  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.15529v1  

#### Abstract
Current large language models reason in isolation. Although it is common to sample multiple reasoning paths in parallel, these trajectories do not interact, and often fail in the same redundant ways. We introduce LACE, a framework that transforms reasoning from a collection of independent trials int...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LACE: Lattice Attention for Cross-thread Exploration 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Large Language Models (LLMs)** 在进行推理时通常是“孤立”的。尽管可以通过并行采样多个推理路径（reasoning paths）来提高成功率，但这些路径之间互不通信，导致它们经常以相同的方式失败（correlated errors），造成计算资源的冗余浪费。

该论文指出，这种 **isolated parallel sampling** 存在以下核心问题：
- **缺乏多样性**：不同路径容易陷入相同的错误模式。
- **无法纠错**：一个路径的失败无法帮助其他路径避免同样的陷阱。
- **后处理低效**：依赖于事后（post-hoc）的投票或验证机制，效率低下且易受模型内部偏见影响。

### 提出的新方法
为解决上述问题，作者提出了 **LACE (Lattice Attention for Cross-thread Exploration)** 框架，其核心创新在于将传统的 **1-D causal attention** 扩展为 **2-D Lattice Attention**，从而实现跨推理线程（cross-thread）的实时交互。

#### 主要技术亮点：
- **Lattice Attention 机制**：
  - 在标准的自回归注意力之上，增加了一个沿“线程维度”（thread dimension）的注意力层。
  - 允许不同的推理路径在生成过程中共享中间见解（intermediate insights），实现动态的协同探索。
  - 采用 **轻量级门控融合（gated fusion）** 和 **降维投影**，确保新增参数不到原模型的1%，对预训练模型干扰极小。

- **合成数据流水线（Synthetic Data Pipeline）**：
  - 由于真实世界中缺乏多线程协作推理的数据，作者设计了一套数据生成流程。
  - 通过 **solution caching** 和 **explicit instruction to avoid** 已有方法，强制模型生成逻辑上多样化的解法。
  - 引入 **step decomposition** 压缩长文本，并通过 **LLM-as-a-judge** 进行跨线程比较和打标（`[[best]]`, `[[success]]`, `[[fail]]`），构建监督信号。

- **多阶段训练框架**：
  - 包括 **continuous pre-training**, **Supervised Fine-Tuning (SFT)** 和基于 **Lattice GRPO** 的强化学习。
  - 在 RL 阶段引入 **accuracy reward** 和 **diversity reward**，鼓励模型不仅找到正确答案，还要能自我选择最优路径。

### 相比现有方法的优势
| 方面 | 传统方法（如 Self-Consistency） | LACE |
|------|-------------------------------|------|
| 推理模式 | 外部聚合（post-hoc voting） | 内部协同（on-the-fly collaboration） |
| 路径关系 | 独立、无交互 | 可通信、可纠错 |
| 多样性 | 依赖随机性，难以保证 | 通过数据和奖励显式促进 |
| 效率 | 需额外验证成本 | 自我选择，端到端高效 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **数学推理基准**：
  - **AIME 24** 和 **AIME 25**：美国数学邀请赛题目，高难度数学推理。
  - **LiveBench**：污染自由（contamination-free）的挑战性 LLM 基准。
- **训练数据来源**：
  - 基于 **DAPO dataset** 构建的合成多线程推理数据。
  - 经过作者提出的 **data curation pipeline** 处理，确保逻辑多样性和跨线程依赖性。

### 实验设置
- **基础模型**：
  - **Qwen3-1.7B** 和 **Qwen3-4B**。
- **线程数**：所有实验均使用 **4 threads**。
- **Lattice Attention 插入位置**：从中间层开始，每隔一层插入一次。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy (Acc)** | 最优路径（`[[best]]`）的答案是否正确。等价于 Pass@1。 |
| **Exploration Diversity (Expl.)** | 不同线程间嵌入向量的余弦距离，衡量路径多样性。 |
| **Format Adherence (Fmt)** | 模型是否遵循输出格式（如正确使用 `[[best]]` 标签）。 |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Independent Sampling** | 单线程模型 + 投票机制（majority voting）。 |
| **Isolated Parallel** | 多线程格式训练，但无 Lattice Attention，线程间仍独立。 |
| **Judge-based Selection** | 使用外部更强的 LLM 进行事后评判。 |
| **Sequential Refinement** | 如 Self-Refine，迭代改进单条路径。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
在 **Qwen3-4B** 模型上的最终结果：

| 方法 | AIME 25 Acc | AIME 24 Acc | LiveBench Acc |
|------|-------------|-------------|----------------|
| Independent + Voting | 13.3 | 10.0 | 28.0 |
| Isolated Parallel + Voting | 3.3 | 10.0 | 15.5 |
| **LACE (Ours) + SFT & RL** | **16.7 (+3.4)** | **20.0 (+7.5)** | **33.0 (+5.0)** |

> ✅ **平均提升超过 7 个百分点**，尤其在 AIME 24 上提升显著。

### 与基线方法的对比结果
- **准确性全面领先**：LACE 在所有任务和模型规模上均优于最强基线。
- **格式遵从性接近完美**：Fmt 达到 100%，表明模型能稳定地执行自我选择协议。
- **多样性更高**：消融实验证明，作者的数据管道相比普通并行数据（如 Parallel-R1）能产生更高的 **embedding dissimilarity** 和 **cross-thread attention ratio**。

### 消融实验结果
#### （1）数据管道消融（Table 2）
| 数据来源 | Edit Dist. | Emb. Dissim. | Cross-Attn Ratio |
|----------|-----------|---------------|------------------|
| Parallel-R1 Data | 0.13 | 0.00 | 0.307 |
| **Our Data** | **0.77** | **0.12** | **0.353** |

> 表明作者的数据管道显著提升了路径多样性和跨线程交互强度。

#### （2）训练流程消融（Table 3）
| 训练方式 | AIME25 Acc | LiveBench Acc |
|----------|------------|----------------|
| 仅 RL（无预训练/SFT） | 0.0 | 1.0 |
| Pretrain+SFT+RL（用 Parallel-R1 数据） | 0.0 | 3.0 |
| **Pretrain+SFT+RL（用我们的数据）** | **13.3** | **11.5** |

> 表明 **完整的三阶段训练流程** 和 **高质量的合成数据** 对成功至关重要。

#### （3）自我评估作用（Table 13）
| 设置 | AIME24 Acc |
|------|-----------|
| 含自我评估（w/ SA） | **20.0** |
| 无自我评估（回退到投票） | 6.7 |

> 表明 **on-the-fly self-selection** 是性能提升的关键，远胜于事后聚合。

---

## 4. 关键结论和发现

### 主要发现
1. **跨线程协作是可行且有效的**：通过 Lattice Attention，LLMs 可以在推理过程中实现类似人类的“集体智能”（collective intelligence），不同路径可以互相借鉴、纠错和收敛。
2. **多样性是性能提升的基础**：冗余的路径探索无法带来收益，只有**逻辑上多样化**的路径才能通过协作真正提升成功率。
3. **合成数据是关键使能因素**：没有合适的训练数据，协作行为无法被有效学习。作者的数据管道成功模拟了“collateral thinking”过程。
4. **涌现能力（emergent behavior）**：
   - 模型学会在关键步骤（如探索、自我评估）增强跨线程注意力（见 Figure 6, 7）。
   - 出现“早期停止”现象：当某一线程找到最优解后，其他线程会识别并标记自己为 `[[success]]`，避免重复计算（见 Figure 10）。

### 方法的局限性
- **依赖特定训练流程**：需要完整的三阶段训练（pretrain → SFT → RL），直接应用到已有模型上不可行。
- **数据生成成本较高**：合成高质量多线程数据需要多次调用强模型进行生成和评判。
- **扩展性待验证**：虽然初步 8B 结果显示趋势一致，但大规模下的稳定性仍需进一步研究（见 Appendix E.4）。

### 未来工作方向
- 将 LACE 应用于更广泛的 **agent tasks** 和 **long-horizon planning** 场景。
- 探索更高效的 **data curation** 方法，降低合成成本。
- 研究如何将 LACE 与 **test-time scaling**（如更大宽度）结合，进一步释放推理潜力。
- 探索在 **非数学领域**（如创意写作、代码生成）中的协作推理模式。

> 🔚 **总体而言，LACE 证明了“让语言模型学会合作”是一条通往更强推理能力的有效路径，为下一代 LLM 推理范式提供了新的方向。**

</details>

---

### 8. [Weak-Link Optimization for Multi-Agent Reasoning and Collaboration](https://arxiv.org/abs/2604.15972)

**Authors**: Haoyu Bian, Chaoning Zhang, Jiaquan Zhang, Xingyao Li, Yuanfang Guo, Wei Dong, Yang Yang  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.15972v1  

#### Abstract
LLM-driven multi-agent frameworks address complex reasoning tasks through multi-role collaboration. However, existing approaches often suffer from reasoning instability, where individual agent errors are amplified through collaboration, undermining overall performance. Current research mainly focuse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Weak-Link Optimization for Multi-Agent Reasoning and Collaboration*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **LLM** 的 **multi-agent** 框架在复杂推理任务中面临以下挑战：
- **推理不稳定性（reasoning instability）**：单个能力较弱的 agent（称为 **weak agent**）产生的错误会在协作过程中被放大，影响最终输出。
- **共识机制失效**：如 majority voting 或 debate 等方法无法有效抑制弱 agent 的负面影响。
- **静态权重分配不合理**：多数方法对所有 agent 一视同仁，忽视了不同任务下 agent 表现的差异性。

### 🚀 提出的新方法：**WORC**
提出 **WORC**（**Weak-Link Optimization for Reasoning Cooperation**），一个基于“**弱链原则**”（weak-link principle）的多智能体推理优化框架。其核心思想是：**系统的整体性能由最薄弱环节决定，应优先补偿弱 agent 而非强化强 agent**。

#### 两阶段流程：
1. **Weak Agent Localization（弱 agent 定位）**
   - 利用 **Swarm Intelligence Algorithms**（SIAs，如 PSO、GWO、HO）在少量样本上搜索最优 agent 权重配置，构建 **weight knowledge base**。
   - 构建 **task signature**（融合语义 embedding 和结构统计特征）来表征任务。
   - 训练一个 **meta-learning-based weight predictor**，实现从 task signature 到 agent 权重的零样本映射，从而识别出当前任务下的 weak agent。

2. **Weak-Link Optimization（弱链优化）**
   - 基于预测的权重，采用 **uncertainty-driven allocation strategy** 动态分配额外推理预算（reasoning budget）。
   - **权重越低的 agent 分配越多重复采样机会**，以提升其输出可靠性。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | WORC |
|------|--------|------|
| **agent 权重处理** | 静态或均等（如 majority voting） | 动态、任务自适应 |
| **错误控制** | 依赖后处理（如 self-consistency） | 前置识别并补偿弱 agent |
| **泛化能力** | 通常针对特定架构 | 支持跨架构迁移（MetaGPT、HIMA 等） |
| **资源利用** | 固定或随机分配 | 不确定性驱动，更高效 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在六个主流推理 benchmark 上进行评估，覆盖多种任务类型：
- **MATH**：高等数学推理
- **GSM8K**：小学数学应用题
- **BBH**（Big-Bench Hard）：逻辑与算法推理
- **MMLU-CF**：常识与事实知识（无污染版本）
- **HotpotQA**：多跳问答（F1 指标）
- **LongBench**：长上下文推理（F1 指标）

### ⚙️ 实验设置
- **基础框架**：以自研的 **AgentChain**（AC）为演示平台，包含四个角色 agent（Data Collection, Problem Understanding, Step Reasoning, Problem Solving）。
- **LLM 后端**：主要使用 **GPT-4o**，部分实验测试了 **DeepSeek-V3**、**Qwen-Turbo**、**GPT-4.1-nano**。
- **SIAs 实现**：使用 **Hippopotamus Optimization (HO)**、**PSO**、**GWO** 进行权重搜索。
- **元学习器**：两层 MLP，输入为 task signature，输出为 agent 权重向量。

### 📊 评估指标
- **主要指标**：
  - 单一正确答案任务：**Exact Match Accuracy**
  - 多跳/部分正确任务（HotpotQA, LongBench）：**F1 Score**
- **辅助分析指标**：
  - 性能波动（standard deviation）
  - 跨架构泛化能力
  - 消融实验（ablation study）
  - 人类一致性验证（Cohen’s Kappa）

### 🆚 基线方法对比
#### 推理级基线（inference-level）：
- **CoT**, **CoT-SC**, **Self-Refine**, **Analogical Prompting**, **FoT**, **AoT**

#### 框架级基线（framework-level）：
- **Majority Voting**
- **AFlow**
- 原始 **AgentChain**（AC）
- 在多个 MAS 架构上集成 WORC：**MetaGPT**, **HIMA**, **MAS2**, **AgentChain**

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table I）
| 方法 | 平均准确率 |
|------|----------|
| CoT | 73.6% |
| CoT-SC | 75.5% |
| AFlow | 76.1% |
| AoT | 80.8% |
| **AgentChain (AC)** | **77.4%** |
| **WORC + AC (Ours)** | **82.2% ± 0.4%** ✅ |

- **绝对提升**：相比最强基线 **AoT** 提升 **1.4 pp**，相比 AC 提升 **4.8 pp**。
- **显著优势**：在 **HotpotQA** 达到 **83.2% F1**，在 **LongBench** 达到 **68.4% F1**，表明在复杂多跳任务中表现优异。

### 🔁 跨架构泛化能力（Table II）
WORC 作为通用优化模块，在不同 MAS 架构上均带来稳定增益：
| 架构 | WORC 提升幅度 |
|------|-------------|
| MetaGPT | +4.0% |
| HIMA | +3.3% |
| MAS2 | +3.0% |
| AgentChain | +6.6% |

👉 表明 WORC 具有良好的 **cross-architecture generalization** 能力。

### 🧪 消融实验结果

#### （1）Task Signature 组件消融（Table III）
| 配置 | 平均准确率 |
|------|----------|
| 仅语义 embedding | 79.9% |
| 仅统计特征 | 80.0% |
| **完整 task signature** | **82.2%** ✅ |

- 结果表明：**语义 + 结构特征联合表示** 对跨任务泛化至关重要。

#### （2）预算分配策略对比（Table IV）
| 策略 | 平均准确率 |
|------|----------|
| Uniform Allocation | 80.0% |
| Rule-Based Allocation | 80.9% |
| **WORC Allocation** | **82.2%** ✅ |

- 证明 **动态、权重感知的资源分配** 显著优于静态策略。

#### （3）跨任务泛化能力（Table VI）
| 训练→测试 | 准确率 |
|---------|-------|
| GSM8K → MATH | 95.6% |
| MATH → GSM8K | 86.3% |
| HotpotQA → LongBench | 67.4% |

- 尽管存在领域差异，性能仍保持稳定，说明 **task signature 和 meta-predictor 具备良好迁移性**。

#### （4）不同 SIA 的影响（Table VII）
| SIA 类型 | 平均准确率 | 性能波动（Variation） |
|--------|----------|------------------|
| AC（无优化） | 77.4% | 0.97 |
| WORC + GWO | 82.0% | 0.37 |
| WORC + PSO | 82.0% | 0.42 |
| WORC + HO | 82.2% | 0.32 ✅ |

- 所有 SIA 变体均带来显著提升，且 **性能波动大幅降低**，说明 WORC 的有效性不依赖特定 SIA。

### 💬 人类一致性验证（Table VIII）
| 数据集 | Cohen’s Kappa (Kw) | 一致性水平 |
|-------|------------------|----------|
| GSM8K | 0.78 | Substantial |
| HotpotQA | 0.72 | Substantial |

- 表明 **EvalAgent** 的评分与人类专家高度一致，验证了其可靠性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **弱链补偿优于强化强链**：通过识别并补偿 weak agent，系统整体鲁棒性和准确性显著提升。
2. **任务自适应权重可行**：结合 **SIA + meta-learning** 可实现跨任务的 zero-shot weak agent 识别。
3. **动态资源分配有效**：不确定性驱动的 budget allocation 显著优于均匀或规则分配。
4. **高稳定性与泛化性**：WORC 不仅提升性能，还大幅降低多 agent 系统的输出波动，并支持跨架构部署。

### ⚠️ 局限性
1. **计算开销增加**：由于引入额外推理步骤和 SIA 搜索，**test-time cost 增加约 1–2 倍**（见 Table IX）。
2. **依赖高质量 EvalAgent**：若中间评估不可靠，会影响权重学习和预算分配。
3. **SIA 初始化敏感**：虽然整体趋势稳定，但不同 SIA 的收敛路径略有差异。
4. **agent 角色固定**：当前假设 agent 功能不变，未考虑动态角色调整。

### 🔮 未来工作方向
1. **降低计算成本**：探索更高效的 SIA 替代方案或在线增量学习机制。
2. **增强在线适应性**：实现实时反馈更新权重，而非依赖离线 knowledge base。
3. **扩展至异构 agent 群体**：支持不同类型、能力、模型的 agent 协同。
4. **结合强化学习**：将 budget allocation 建模为 RL 决策过程，进一步优化资源利用效率。

---

> **总结一句话**：  
> **WORC 通过“识别弱链 + 动态补偿”的范式，实现了多 agent 推理系统的稳定性与准确性双重提升，为构建可靠、可解释的协作式 AI 提供了新思路。**

</details>

---

### 9. [Impact of Nonlinear Power Amplifier on Massive MIMO: Machine Learning Prediction Under Realistic Radio Channel](https://arxiv.org/abs/2604.15977)

**Authors**: Marcin Hoffmann, Pawe{\l} Kryszkiewicz  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.15977v1  

#### Abstract
M-MIMO is one of the crucial technologies for increasing spectral and energy efficiency of wireless networks. Most of the current works assume that M-MIMO arrays are equipped with a linear front end. However, ongoing efforts to make wireless networks more energy-efficient push the hardware to the li...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Impact of Nonlinear Power Amplifier on Massive MIMO: Machine Learning Prediction Under Realistic Radio Channel*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **Massive MIMO (M-MIMO)** 系统中因 **非线性功率放大器 (Nonlinear PA)** 引起的失真问题展开研究。传统研究多假设信道为理想化的 **Rayleigh** 或 **Line of Sight (LoS)** 模型，忽略了真实城市环境中由空间相关性和多径传播带来的复杂影响。这导致对 **信号到失真比 (SDR)** 的估计不准确，进而影响资源分配策略的有效性。

### 提出的新方法与新思路
1. **提出两种新型 SDR 建模方法**：
   - **统计模型（用于非调度用户/Victim UEs）**：基于 **Generalized Extreme Value (GEV) 分布** 对 Victim UE 的 SDR 进行建模，并结合空间去相关距离（约 26 m），适用于网络规划中的干扰估计。
   - **机器学习模型（用于调度用户/Scheduled UEs）**：提出一种基于 **VGG16 CNN 架构** 的 SDR 预测方法，输入为融合了信道相关矩阵和 IBO（Input Back-Off）信息的 **特征矩阵 (Feature Matrix)**，实现对 SDR 的高精度预测。

2. **引入基于 3D-Ray Tracing (3D-RT) 的真实信道仿真**：
   - 使用 **Wireless InSite 3D-RT 软件** 在 **Madrid Grid 城市场景** 中生成接近现实的信道数据，验证了传统 Rayleigh 和 LoS 模型在 SDR 预测上的不足。

3. **提出“失真感知”的 per-user 功率分配方案**：
   - 利用 VGG16 预测的 SDR 动态调整每个用户的 IBO，以最大化其 **Signal to Noise and Distortion Ratio (SNDR)**，从而提升吞吐量。

### 相比现有方法的优势
| 方面 | 本文方法优势 |
|------|---------------|
| **信道建模真实性** | 使用 3D-RT 而非简化模型，更贴近实际部署环境 |
| **SDR 预测准确性** | ML 模型能捕捉复杂的空间相关性，优于理论公式（如 (23)/(24)） |
| **适用性广** | 模型可适配不同 PA 特性（如 Soft-Limiter 和 Rapp 模型） |
| **实用性** | 提出的功率分配方案可直接集成到基站调度器中 |

---

## 2. 核心实验方法和设置

### 数据集
- **生成方式**：通过 **Wireless InSite 3D-RT** 在 **Madrid Grid 城市场景** 中模拟信道。
- **天线配置**：MISO BS 配备 **K=128 天线**（8×16 矩形阵列），高度 45 m。
- **用户分布**：**3542 个 UE** 均匀分布在 4 m 间距的网格上，高度 1.5 m。
- **PA 模型**：主要使用 **Soft-Limiter**，并额外测试 **Rapp 模型 (p=2)**。
- **调制与参数**：**OFDM**, **QPSK**, **69 子载波**, **子载波间隔 360 kHz**, **fc=3.6 GHz**。

### 实验设置
- **训练集**：3542 UE × 4 个 IBO 值（{-3, 0, 3, 6} dB） → 共 14,168 个样本。
- **验证集**：随机选取 20% 的训练数据。
- **测试集**：3542 UE × 3 个未见 IBO 值（{-1, 2, 5} dB） → 共 10,626 个样本。
- **特征矩阵构建**：  
  $$
  F_m = \frac{1}{N_u \cdot \gamma \cdot \beta_m} \sum_{n=1}^{N_u} h_{m,n} h_{m,n}^H
  $$
  包含天线间相关性和 IBO 信息。

### 评估指标
- **回归误差**：
  - **MAPE (Mean Absolute Percentage Error)**
  - **RMSE (Root Mean Squared Error)**
  - **MAE (Mean Absolute Error)**
- **系统级性能**：
  - **UE 吞吐量 (User Rate)**
  - **吞吐量增益（相对于基线）**

### 基线方法对比
| 基线方法 | 描述 |
|---------|------|
| **Fixed IBO = 6 dB** | 当前工业界常用固定 IBO 设置 |
| **Tavares et al. [4]** | 基于单天线 SISO 系统解析优化的动态 IBO 选择算法 |
| **Theoretical SDR (Rayleigh/LoS)** | 使用公式 (23)/(24) 计算的理想边界值 |

---

## 3. 主要实验结果和性能指标

### SDR 预测性能（VGG16）
| 指标 | 数值 |
|------|------|
| **MAPE** | **1.49%** |
| **RMSE** | **0.51 dB** |
| **MAE** | **0.37 dB** |
> ✅ 测试集使用未见过的 IBO 值（-1, 2, 5 dB），表明模型具有良好的泛化能力。

### 模型压缩效果（Pruning）
| 模型变体 | 参数量 | MACs | 推理时间 (CPU/GPU) | MAPE |
|----------|--------|-------|---------------------|-------|
| 原始 VGG16 | 65·10⁶ | 5·10⁹ | 18.4 ms | 1.49% |
| **40% Pruned** | **39·10⁶** | **3·10⁹** | **9.5 ms** | **1.51%** |
> ✅ 显著减少计算开销，性能几乎无损，适合边缘部署。

### 功率分配方案性能对比
| 指标 | VGG16 方案 | Fixed IBO=6dB | Tavares [4] |
|------|------------|----------------|-------------|
| **中位数吞吐量增益** | **+12.1%** | — | +1.8% |
| **90 百分位增益** | **>30.7%** | — | ~15% |
| **是否所有用户受益** | ✅ 是 | — | ❌ 约 50% 用户速率下降 |
| **最差情况降级** | 无 | — | >40% 降级 |

> 📊 图 19 显示，VGG16 方案的用户速率比固定 IBO 最高提升近 **70%**，且无用户受损。

### 消融实验与关键观察
- **相位信息无关性**：加入信道相关矩阵的相位信息后，MAPE 未改善（仍 ~1.5%），证明仅需幅度信息即可。
- **IBO 变异性影响**：MRT 预编码可能导致各 PA 的实际 IBO 差异显著（如平均 6 dB 时，个别 PA 可低至 1.66 dB），使 SDR **低于 LoS 理论下限**。
- **Rapp 模型兼容性**：VGG16 在 Rapp PA 上同样有效（MAPE=1.25%），证明方法通用性强。

---

## 4. 关键结论和发现

### 主要发现
1. **传统信道模型不适用于非线性失真分析**：
   - Rayleigh 和 LoS 模型无法准确反映真实城市环境中 **SDR 的空间分布特性**。
   - 实际 SDR 受 **空间相关性、路径损耗差异、预编码策略** 综合影响。

2. **SDR 具有高度空间变异性**：
   - 即使在同一 IBO 下，不同位置 UE 的 SDR 可相差 **超过 10 dB**。
   - **Victim UE 的 SDR 可用 GEV 分布建模**，去相关距离约为 **26 米**。

3. **LoS 并非最坏情况**：
   - 当 MRT 预编码导致某些 PA 工作在低 IBO 时，**SDR 可低于 LoS 理论值**，挑战了传统认知。

4. **ML 方法可高效预测 SDR**：
   - **VGG16 CNN** 能从信道特征矩阵中学习复杂映射关系，实现 **<0.5 dB 误差** 的 SDR 预测。

5. **动态 IBO 分配显著提升性能**：
   - 所提 **distortion-aware per-user power allocation** 方案相比固定 IBO 实现 **12.1% 中位数吞吐量增益**，且优于现有解析方法。

### 方法的局限性
- **依赖特定硬件配置**：模型需针对具体天线阵列几何和 PA 特性重新训练。
- **静态场景假设**：当前基于快照式仿真，未考虑高速移动下的时变信道。
- **集中式训练需求**：目前为离线训练，缺乏在线自适应能力。

### 未来工作方向
- 将 SDR 预测模型用于 **能量效率优化**（Energy Efficiency Maximization）。
- 结合 **location-awareness** 与 **Reinforcement Learning** 实现 IBO 的实时动态调整。
- 支持 **UE 上报 SDR**，实现 **online learning** 与模型持续更新。
- 扩展至 **cell-free Massive MIMO** 和 **multi-cell interference** 场景。

--- 

> **总结**：本文首次在 **真实 3D-RT 信道** 下系统分析了非线性 PA 对 M-MIMO 的影响，揭示了传统模型的局限性，并提出了基于 **GEV 统计模型** 和 **VGG16 ML 模型** 的 SDR 预测框架，最终实现了 **12% 吞吐量增益** 的实用化功率分配方案，为 6G 高能效网络设计提供了重要参考。

</details>

---

### 10. [DeepER-Med: Advancing Deep Evidence-Based Research in Medicine Through Agentic AI](https://arxiv.org/abs/2604.15456)

**Authors**: Zhizheng Wang, Chih-Hsuan Wei, Joey Chan, Robert Leaman, Chi-Ping Day, Chuan Wu, Mark A Knepper, Antolin Serrano Farias, Jordina Rincon-Torroella, Hasan Slika, Betty Tyler, Ryan Huu-Tuan Nguyen, Asmita Indurkar, M\'elanie H\'ebert, Shubo Tian, Lauren He, Noor Naffakh, Aseem Aseem, Nicholas Wan, Emily Y Chew, Tiarnan D L Keenan, Zhiyong Lu  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.15456v1  

#### Abstract
Trustworthiness and transparency are essential for the clinical adoption of artificial intelligence (AI) in healthcare and biomedical research. Recent deep research systems aim to accelerate evidence-grounded scientific discovery by integrating AI agents with multi-hop information retrieval, reasoni...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DeepER-Med: Advancing Deep Evidence-Based Research in Medicine Through Agentic AI》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 deep research 系统（如 OpenAI Deep Research、Google AI Mode）存在以下关键缺陷：
- **缺乏透明的证据评估标准**：系统在多轮检索与推理中容易累积错误，中间过程不透明，难以验证结论是否基于高质量证据。
- **引用幻觉（citation hallucination）风险高**：部分系统通过 LLM 生成参考文献，而非从真实数据库获取，导致引用不可追溯。
- **评估基准脱离现实需求**：现有 benchmark 多为简化选择题，无法反映真实医学研究中的复杂性与多维度分析要求。

### 🔧 提出的新方法与新思路
作者提出 **DeepER-Med**，一个基于 **Evidence-Based Generation (EBG)** 范式的 agentic AI 框架，其核心是将医学研究建模为可解释、可审计的知识提炼流程。该框架包含三个模块：

1. **Research Planning（研究规划）**
   - 将原始问题分解为层级化的子问题（hierarchical sub-questions），明确研究意图。
   - 示例：针对“MacTel 2 最早的影像学特征是什么？”先识别成像方式，再逐项分析各模态下的早期表现。

2. **Agentic Collaboration（智能体协作）**
   - 构建三层智能体网络（Worker → Manager → Director），协调多种 API 工具进行证据检索与筛选。
   - 支持从 PubMed、ClinicalTrials.gov、PrimeKG 等权威资源直接调用数据，确保引用来源真实可查。
   - 引入预设的证据质量标准（methodological quality, contextual relevance）对检索结果进行显式过滤。

3. **Evidence Synthesis（证据综合）**
   - 在用户约束下整合证据，生成简洁答案 + 结构化分析报告。
   - 所有引用均来自源数据库 API 查询，杜绝 LLM 伪造参考文献。

此外，构建了 **DeepER-MedQA** 基准数据集，由 11 名跨学科专家设计并评审，涵盖真实医学研究场景。

### ⚖️ 相比现有方法的优势
| 维度 | DeepER-Med | 现有系统（如 OpenAI Deep Research） |
|------|------------|-------------------------------|
| **证据可靠性** | ✅ 引用直接来自数据库 API，无幻觉 | ❌ 存在生成虚假引用的风险 |
| **推理透明性** | ✅ 显式展示子问题链与证据筛选逻辑 | ❌ 黑箱式推理，过程不可见 |
| **评估全面性** | ✅ 多维人工评估（准确性、连贯性、新颖性等） | ❌ 主要依赖自动准确率指标 |
| **适用性** | ✅ 面向真实临床与科研复杂问题 | ❌ 多用于开放问答或简单任务 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）自建基准：**DeepER-MedQA**
- 包含 **100 个专家级医学研究问题**，覆盖 11 个疾病领域（如 melanoma、glioblastoma、AMD 等）。
- 问题分类：
  - **基础研究（Basic Research）**：54%
  - **转化研究（Translational Research）**：22%
  - **临床研究（Clinical Research）**：24%
- 每个条目包含：问题类别、问题文本、参考答案、专家注释、支持引用（按时间排序）。

#### （2）公开数据集（用于自动化评估）
| 数据集 | 类型 | 用途 |
|--------|------|------|
| **PubMedQA** | 三选一问答（Yes/No/Maybe） | Intent Identification & Knowledge Synthesis |
| **BioMaze (Binary/Open QA)** | 开放式与二元问答 | Literature Retrieval & Knowledge Synthesis |
| **MedAESQA** | Attribution QA（归因问答） | Evidence Interpretation（多样性与一致性） |
| **BioDSA** | Hypothesis Verification（假设验证） | Evidence Interpretation（溯源能力） |

---

### 🧪 实验设置与评估指标

#### 人工评估维度（由 11 名专家盲评）
| 指标 | 描述 |
|------|------|
| **Answer Accuracy** | 回答是否正确 |
| **Analytical Quality** | 分析是否深入、逻辑是否严密 |
| **Reference Relevance** | 引用是否相关且支持论点 |
| **Novel Insight** | 是否提供新的科学洞见 |
| **Comprehensiveness** | 是否覆盖足够广的相关知识 |

#### 自动化评估指标
| 指标 | 定义 |
|------|------|
| **Semantic Similarity**（使用 MedCPT 编码） | 衡量检索文献与参考文献之间的语义接近程度 |
| **Information Entropy**（香农熵） | 衡量证据集合的信息多样性 |
| **Jensen–Shannon Divergence** | 衡量系统检索分布与专家标注分布的一致性 |
| **Hypothesis Coverage** | 系统引用是否覆盖真实支持研究 |

---

### 🆚 基线方法对比
- **OpenAI Deep Research**
- **OpenEvidence**
- **Google AI Mode (Deep Search)**

这些均为当前最先进的生产级 deep research 平台，广泛应用于科研与临床辅助决策。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（基于 DeepER-MedQA 上的人工评估）

| 指标 | DeepER-Med | 最强基线（OpenAI Deep Research） |
|------|------------|-----------------------------|
| **高准确回答数（out of 100）** | **77** | 69 |
| **高分析质量回答数** | **67** | 47 |
| **高引用相关性回答数** | **81** | 59 |
| **产生新见解的回答数** | **27** | — |
| **被专家选为最佳回答次数** | **60**（其中 27 次唯一胜出） | 43（19 次唯一） |

> 💡 **说明**：DeepER-Med 在所有维度上显著优于现有系统，尤其在引用质量和分析深度方面优势明显。

---

### 🔍 自动化评估结果

#### （1）Intent Identification（PubMedQA）
- 使用 sub-question decomposition 后准确率达 **79.2%**
- 移除子问题后下降至 **74.4%** → 证明分步拆解有效提升理解精度

#### （2）Literature Retrieval（BioMaze）
- 检索文献与参考文献的 **semantic similarity > 70%**
- 嵌入空间可视化显示：DeepER-Med 检索范围更广但仍聚焦主题簇内 → 实现“可控扩展”

#### （3）Evidence Interpretation（MedAESQA）
- **信息熵更高**（接近专家水平），表明证据来源更多样；
- **JS 散度低（0.04–0.05）**，说明分布高度一致 → 多样性与一致性兼得

#### （4）Hypothesis Verification（BioDSA）
- **96.5% 的真实支持研究被成功检索到**
- 正确验证“真”假设的比例达 **91.3%**

#### （5）Knowledge Synthesis（BioMaze & PubMedQA）
| 方法 | Binary QA 准确率 | Open QA 准确率 |
|------|------------------|---------------|
| DeepER-Med | **89.3%** | **90.6%** |
| GPT-4o + CoT | 82.3% | 83.7% |
| G-Retriever | ~80% | ~80% |
| MedPrompt / Flan-PaLM | ~79–82% | — |

> ✅ DeepER-Med 在开放问答中达到 SOTA 水平，且无需复杂提示工程。

---

### 🔁 消融实验结果
| 设置 | PubMedQA 准确率 | 影响 |
|------|------------------|------|
| 完整模型（含上下文 + 子问题） | 79.2% | ✅ 基准 |
| 移除上下文信息 | 78.2% | 影响较小 |
| 移除子问题分解 | 74.4% | ↓ 4.8%，显著退化 |
| 移除知识图谱查询扩展 | ↓ ~5–11% | 证明 KG 对 query expansion 至关重要 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **结构化研究流程显著提升可信度与性能**
   - 通过 **intent decomposition + criteria-driven evidence appraisal**，避免了传统 agent-loop 中的误差累积问题。
   
2. **证据透明性可量化且至关重要**
   - DeepER-Med 的引用全部可追溯，且在专家评估中获得最高信任度。
   - 其他系统存在引用幻觉或弱证据支撑问题。

3. **可控的证据扩展优于盲目检索**
   - 系统在保持主题对齐的同时扩大检索范围，既提升了覆盖率又未牺牲相关性。

4. **能生成超越已有认知的新科学洞见**
   - 在 27 个案例中，专家认为 DeepER-Med 发现了文献中隐含但未被强调的关系。

5. **在真实临床场景中具备实用价值**
   - 在 8 个 Precision Oncology Tumor Board 案例中，**7 个结论与专家推荐一致**，证据充分性获认可。

---

### ⚠️ 局限性
1. **仍可能出现碎片化整合**
   - 有时将多个研究并列陈述，未能建立生物学机制间的因果联系（见 Appendix Table 2）。

2. **受制于现有文献质量与覆盖度**
   - 若某领域缺乏高质量研究，系统难以弥补知识空白。

3. **效率较低**
   - 复杂查询耗时较长，与 OpenAI Deep Research 相当，尚不适合实时交互。

4. **人工评估成本高**
   - DeepER-MedQA 构建需大量专家投入，难以大规模复制。

---

### 🔮 未来工作方向
1. 扩展 **DeepER-MedQA** 至更大规模、更多临床导向的问题集。
2. 加强对 **多证据间生物关联建模**，实现真正意义上的机制整合。
3. 探索与电子病历（EMR）、基因组数据等结构化临床系统的集成。
4. 优化并行处理流程以提高响应速度。
5. 开发更可靠的自动化评估器（如 GPT-5.2 判官），减少人工依赖。

---

> 📌 **总体评价**：  
> DeepER-Med 不仅是一个更强的 deep research 系统，更是一种范式转变——从“快速出结果”转向“可信赖、可审计、可复现”的医学 AI 研究基础设施。它为未来 AI 辅助科研提供了可信路径，有望成为下一代医学知识发现平台的标准架构。

</details>

---

### 11. [Bilevel Optimization of Agent Skills via Monte Carlo Tree Search](https://arxiv.org/abs/2604.15709)

**Authors**: Chenyi Huang, Haoting Zhang, Jingxu Xu, Zeyu Zheng, Yunduan Lin  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.15709v1  

#### Abstract
Agent \texttt{skills} are structured collections of instructions, tools, and supporting resources that help large language model (LLM) agents perform particular classes of tasks. Empirical evidence shows that the design of \texttt{skills} can materially affect agent task performance, yet systematica...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Bilevel Optimization of Agent Skills via Monte Carlo Tree Search**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题  
该论文聚焦于 **LLM Agent Skills 的系统性优化**问题。Agent Skills 是由指令、工具和辅助资源组成的结构化软件包，用于增强 LLM 代理在特定任务上的表现。尽管已有研究表明 Skill 设计对性能有显著影响，但由于其异构性（heterogeneous）、组件间强依赖性和离散组合的设计空间，**缺乏有效的自动化优化框架**。

传统方法难以处理 Skill 中“结构”与“内容”的联合优化，而手动设计效率低且易次优。

---

### 🚀 提出了什么新方法或新思路  
作者提出了一种 **基于 Monte Carlo Tree Search (MCTS) 的双层优化框架（Bilevel Optimization Framework）**，将 Skill 优化解耦为两个层次：

- **外层循环（Outer Loop）**：使用 **MCTS** 搜索最优的 Skill **结构配置**（structure configuration），即组件的组织方式（如文件目录、章节划分等）。
- **内层循环（Inner Loop）**：在固定结构下，进行 **内容精炼（content refinement）**，通过 LLM 驱动的迭代改进生成高质量的指令、代码和参考资料。

该框架引入了以下关键机制：
- **结构-内容分离建模**：将 Skill 表示为元组 $ S = (\theta, \phi) $，其中 $\theta$ 为结构，$\phi$ 为内容。
- **LLM-Guided MCTS**：利用 LLM 进行分析、诊断与动作提议，指导树搜索方向。
- **Family-Specific Refinement**：根据结构变更类型匹配不同的内容优化策略（如 metadata 编辑、instruction 改写、script 生成等）。
- **保守选择规则（Conservative Selection Rule）**：采用 Lower Confidence Bound (LCB) 作为筛选标准，提升优化稳定性，避免噪声干扰下的过拟合。

---

### 🔍 相比现有方法的优势  

| 对比维度 | 本文方法 | 现有方法（如 AFlow [9]） |
|--------|---------|--------------------------|
| 优化对象 | Agent Skill Package（含指令、脚本、参考文档等异构内容） | Code-based Workflow（以可执行代码为主） |
| 架构设计 | 双层优化（bilevel），明确区分结构搜索与内容精炼 | 单一层级的流程修改 |
| 内容优化 | 家族感知（family-aware）的内容精炼策略 | 统一的代码修改模板 |
| 探索机制 | MCTS + LLM 推理引导 + 保守反馈回传 | MCTS + 执行反馈 |
| 抗噪能力 | 引入 LCB 和诊断信息过滤，提高鲁棒性 | 依赖原始奖励信号，易受评估噪声影响 |

> 因此，本方法更适用于复杂、非结构化的 Agent Skill 优化场景。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集  
- **Operations Research Question Answering (ORQA)** 数据集 [21]
  - 包含 1,513 道运筹学领域的多选题，涉及决策变量、约束、目标函数等建模理解。
  - 平均输入长度 231 词，测试代理对自然语言问题转化为数学模型的能力。
  - 示例任务：识别“广播节目安排”中的决策变量是“是否播出某节目”。

---

### ⚙️ 实验设置  

#### 分割策略：
- 将 ORQA 划分为三个子集：
  - **Search Split**：用于 bilevel 优化过程中的候选 Skill 评估。
  - **Confirm Split**：优化后用于超参配置选择。
  - **Test Split**：最终独立评估，不参与任何训练或调优。

> 各阶段使用采样子集（共 120 题），控制计算成本。

#### 模型与环境：
- **运行时 Agent**：`openai/gpt-5.2-codex`，在 Harbor 沙箱中执行 Skill。
- **优化控制器**：`openai/gpt-5.4` + DSPy 框架，负责 MCTS 的推理与调度。
- **Token 预算分配**：
  - 外层 Proposal 阶段：最多 20,000 tokens（支持复杂推理）
  - 内层每次 refinement：限制为 1,024 tokens
  - 符合 multi-fidelity simulation optimization 思想

#### 评估指标：
- **Exact Match Score**：完全正确回答的比例。
- **Reward Signal ($R_s$)**：基于下游任务表现的标量评分，驱动 MCTS 回传。

#### 基线方法对比：
- **Seed Skill**：初始由 skill-creator 自动生成的 Skill，作为 baseline。
- **Configuration A vs B**：两种 MCTS 设置对比（保守 vs 探索性强）

| 参数 | Configuration A（保守） | Configuration B（探索性） |
|------|------------------------|----------------------------|
| 最大轮数 | 3 | 6 |
| 选择策略 | UCB1 | Mixed-probability |
| 动作空间 | 白名单限制 | 更开放 |
| 收敛条件 | 更早停止 | 更耐心 |

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据  

| 方法 | Confirm Split Score | Test Split Exact Match |
|------|---------------------|-------------------------|
| Seed Skill (Baseline) | — | **0.90625** |
| Optimized Skill (Config B) | 0.8857 | **0.9375** |
| **绝对提升** | — | **+0.03125** |

- 在 Search Split 上，两种配置均达到峰值 Reward **0.9434**。
- 最终胜出的是 **Configuration B**，因其在 Confirm Split 上表现更好（0.8857 > 0.8571）。

---

### 🔬 消融分析与路径可视化（见 Figure 2）  

- **MCTS 搜索树显示有效探索**：
  - 成功路径：先合并 reference 内容到主文件 → 添加 “Question-Type Triage Checklist” 节。
  - 放弃分支：仅修改 frontmatter 或分散编辑未带来增益。
- 结构变更带来的收益需经内容精炼才能释放，验证了 bilevel 设计必要性。

---

### 🧩 内容与结构变化对比  

| 层面 | Seed Skill | Winning Skill |
|------|-----------|---------------|
| **结构** | 两个文件：<br>- `SKILL.md`<br>- `references/question_types.md` | 单一文件：<br>- 所有关键指引集中于 `SKILL.md`<br>- 新增 **Triage Checklist** 节 |
| **内容** | 通用流程描述<br>较弱的输出约束 | 明确执行顺序<br>强化分类前置判断<br>严格单 token 输出要求 |

> 结构集中化 + 内容显式化 → 减少歧义，提升一致性。

---

## 4. **关键结论和发现**

### ✅ 主要发现  

1. **Skill 结构与内容存在强耦合关系**，必须协同优化；单独优化任一部分效果有限。
2. **MCTS 能有效导航离散、组合式的结构设计空间**，尤其适合路径依赖性强的 Skill 修改。
3. **LLM 可作为优化引擎**，不仅用于推理，还可充当“世界模型”与“规划器”，实现自动 Skill 演化。
4. **结构简化与内容强化相结合能显著提升性能**：将 reference 内容内联 + 增加 checklist 提升了 agent 的任务遵循能力。
5. **保守的选择机制（LCB）有助于抵御评估噪声**，防止无效修改被错误采纳。

---

### ⚠️ 方法的局限性  

1. **计算开销高**：每轮 MCTS 扩展需多次 LLM 调用与下游任务执行，不适合实时在线优化。
2. **依赖高质量 Seed Skill**：若初始 Skill 完全偏离功能需求，搜索可能陷入局部最优。
3. **泛化性待验证**：当前仅在 ORQA 上验证，跨领域迁移能力未知。
4. **Token 预算敏感**：Proposal 阶段需要极大上下文窗口，限制了轻量模型的应用。

---

### 🔮 未来工作方向  

1. **引入 warm-start 或 curriculum learning** 加速搜索收敛。
2. **扩展至多 Skill 协同优化**，研究 skill library 层面的自进化机制。
3. **结合 process reward modeling** 提供细粒度反馈，替代单一 scalar reward。
4. **探索轻量化版本**：如使用 smaller LLMs + caching + early stopping 降低延迟。
5. **安全与可解释性增强**：监控 Skill 修改中的潜在漏洞注入风险（参考 [18]）。

---

> 💡 **总结一句话**：  
> 本文首次将 **bilevel optimization + MCTS + LLM-as-optimizer** 范式应用于 Agent Skill 自动优化，在 ORQA 上实现了 **+3.125% 的准确率提升**，为构建可自我进化的智能体系统提供了新路径。

</details>

---

### 12. [Skill-RAG: Failure-State-Aware Retrieval Augmentation via Hidden-State Probing and Skill Routing](https://arxiv.org/abs/2604.15771)

**Authors**: Kai Wei, Raymond Li, Xi Zhu, Zhaoqian Xue, Jiaojiao Han, Jingcheng Niu, Fan Yang  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.15771v1  

#### Abstract
Retrieval-Augmented Generation (RAG) has emerged as a foundational paradigm for grounding large language models in external knowledge. While adaptive retrieval mechanisms have improved retrieval efficiency, existing approaches treat post-retrieval failure as a signal to retry rather than to diagnose...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Skill-RAG: Failure-State-Aware Retrieval Augmentation via Hidden-State Probing and Skill Routing**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
当前的 **Retrieval-Augmented Generation (RAG)** 系统虽然能通过检索外部知识增强生成能力，但在面对**持续性检索失败**（persistent retrieval failures）时，通常仅简单地重复检索（re-retrieve），而未深入诊断失败的根本原因。  
作者指出，许多失败并非由于缺乏相关证据，而是存在 **query-evidence misalignment** ——即查询与证据空间之间存在结构性对齐差距（如表述不匹配、多跳逻辑纠缠等）。这种“对齐鸿沟”导致即使有正确信息，模型也无法有效利用。

### 🚀 提出的新方法与思路
提出 **Skill-RAG**，一个**失败感知型 RAG 框架**，其核心思想是：
- 将后检索失败视为可分类、可纠正的状态，而非简单的重试信号。
- 引入两个关键组件协同工作：
  1. **Hidden-State Prober（隐藏状态探测器）**：轻量级模块，基于 LLM 内部层的 hidden states 判断是否需要检索或已具备足够知识作答。
  2. **Prompt-based Skill Router（技能路由器）**：当检测到失败状态时，分析失败模式并选择合适的“检索技能”进行精准修复。

定义了四种**可迁移的 retrieval skills**：
| 技能 | 功能 |
|------|------|
| **Query Rewriting** | 改写表面形式不匹配的查询，提升与语料索引的一致性 |
| **Question Decomposition** | 分解复杂多跳问题为子问题，解决前提纠缠 |
| **Evidence Focusing** | 针对宽泛查询提取缺失的信息槽位，聚焦检索目标 |
| **Exit Skill** | 识别不可修复情况（如知识缺失），提前终止以节省计算 |

该框架将 post-retrieval recovery 视为一个**条件技能选择问题**（conditional skill selection），实现细粒度控制。

### 🔍 相比现有方法的优势
| 对比维度 | Skill-RAG vs. Prior Work |
|--------|--------------------------|
| **决策机制** | 不只是“是否检索”，而是“为何失败 + 如何修正” → 更智能的恢复路径 |
| **效率** | 避免盲目重试导致的 query drift 和推理开销浪费 |
| **泛化性** | 在 out-of-distribution (OOD) 数据上表现显著优于仅依赖 probing 或 adaptive retrieval 的方法 |
| **可解释性** | 失败状态在表示空间中呈现几何结构，支持基于表征的诊断 |

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
分为 **in-domain** 与 **out-of-distribution (OOD)** 两类：

| 类型 | 数据集 | 描述 |
|------|-------|------|
| **In-Domain** | HotpotQA, NQ (Natural Questions), TriviaQA | 单跳或多跳问答，用于训练 prober 和开发调优 |
| **OOD** | MuSiQue, 2WikiMultiHopQA | 更复杂的多跳推理任务，测试泛化能力 |

- Prober 训练：从 in-domain 数据集中采样 3,000 条样本
- 开发集：500 条
- 测试集：每个 OOD 数据集各 500 条

### ⚙️ 实验设置
- **Backbone Model**: Gemma2-9B
- **Retriever**: BM25（无学习型稀疏检索）
- **Prompting**: 所有方法均采用 4-shot prompting
- **评估指标**:
  - **Exact Match (EM)**
  - **Accuracy (ACC)**

### 🆚 基线方法对比
| 方法 | 特点 |
|------|------|
| **No Retrieval** | 仅依赖参数知识生成答案 |
| **Single-step RAG** | 检索一次后生成 |
| **FLARE** | 基于 token-level 生成置信度触发检索 |
| **DRAGIN** | 利用 attention 信号判断检索时机 |
| **Adaptive-RAG** | 根据问题复杂度路由不同检索策略 |
| **Probing-RAG** | 使用 hidden-state 探测决定是否检索（最接近的 baseline） |

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（见 Table 1）

| Method | Average EM | Average ACC |
|--------|-----------|------------|
| No Retrieval | 28.0 | 40.0 |
| Single-step RAG | 26.7 | 41.5 |
| FLARE | 26.7 | 34.5 |
| DRAGIN | 32.7 | 38.7 |
| Adaptive-RAG | 27.2 | 34.6 |
| **Probing-RAG** | **30.1** | **42.4** |
| **Skill-RAG (Ours)** | **31.0** | **46.8** ✅ |

> ✅ **Skill-RAG 在平均 ACC 上超越所有基线 4.4 个百分点**

#### 🔺 在 OOD 数据上的巨大优势：
| 方法 | MuSiQue ACC | 2WikiMultiHopQA ACC |
|------|-------------|---------------------|
| Probing-RAG | 13.9 | 38.9 |
| **Skill-RAG** | **20.0** (+6.1) | **52.5** (+13.6) ✅ |

> 表明：**failure-conditioned skill routing 显著提升了对分布外复杂问题的适应能力**

#### 💡 与其他方法的关键对比发现：
- 虽然 DRAGIN 在部分 EM 指标上略高，但 ACC 较低 → 可能产生更多错误自信输出
- Skill-RAG 在 hard cases 中更稳定，尤其擅长处理需**对齐修正**而非简单多次检索的任务

### 🔍 消融实验与分析（Section 4.3）

#### ➤ 表示空间可视化（t-SNE 图，Figure 2）
- 初始失败状态形成两个分离簇：
  - **Cluster 0（左）**：可通过对齐技能修复（alignment-fixable）
  - **Cluster 1（右）**：真正不可解（irreducible，如知识缺失）
- 应用技能后，Cluster 0 明显收缩 → 说明技能有效缓解对齐问题
- 若使用 LLM 自动生成 >6 种技能，则聚类结构消失 → **技能过多破坏表征几何结构**

> ✅ 支持结论：提出的 **四技能体系具有内在合理性与简洁性（parsimonious taxonomy）**

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Query-Evidence Misalignment 是一类结构化失败模式**，可在 hidden state space 中被探测和区分。
2. **Failure states 具备几何可分性**，支持基于表示的诊断与路由。
3. **引入 skill-based routing 显著优于单纯的“重试”机制**，特别是在复杂、OOD 场景下。
4. **Skill-RAG 实现早期终止与防 query drift**，避免无效迭代带来的资源浪费（见 Table 2 案例）。

### ⚠️ 局限性（Limitations）
1. **依赖 prompt-based routing**：技能诊断质量受限于底层 LLM 的指令遵循能力，在弱模型上可能退化。
2. **技能词典泛化有限**：目前基于开放域 QA 设计，尚未验证在科学文献、多语言等领域的适用性。
3. **模型单一性**：实验仅在 Gemma2-9B 上完成，跨模型/规模的鲁棒性有待进一步验证。

### 🔮 未来工作方向
- 构建**可学习的 skill router** 替代 prompt-based 方案，提高稳定性
- 扩展技能集至新领域（如数学推理、代码检索）
- 探索 **end-to-end jointly trained probing + routing** 架构
- 将 skill taxonomy 推广为通用的 RAG debugging toolkit

---

## ✅ 总结一句话
> **Skill-RAG 首次将“失败诊断”引入 RAG 流程，通过 hidden-state probing 发现失败状态，并借助 skill routing 实施针对性纠正，实现了从“盲目重试”到“智能修复”的跃迁，在 hard & OOD 问题上取得显著突破。**

</details>

---

### 13. [CHOP: Chunkwise Context-Preserving Framework for RAG on Multi Documents](https://arxiv.org/abs/2604.15802)

**Authors**: Hyunseok Park, Jihyeon Kim, Jongeun Kim, Dongsik Yoon  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.15802v1  

#### Abstract
Retrieval-Augmented Generation (RAG) systems lose retrieval accuracy when similar documents coexist in the vector database, causing unnecessary information, hallucinations, and factual errors. To alleviate this issue, we propose CHOP, a framework that iteratively evaluates chunk relevance with Large...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CHOP: Chunkwise Context-Preserving Framework for RAG on Multi Documents

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Retrieval-Augmented Generation (RAG)** 系统在处理多文档场景时面临以下挑战：
- 文档间存在大量语义或词汇重叠（如产品手册、法规文本），导致 **retrieval accuracy 下降**；
- 基于长度的固定分块策略（length-based chunking）破坏了跨段落的上下文连续性（如指代、公式引用等），造成 **coreference 断裂** 和 **局部引用丢失**；
- 相似但不同的文档片段容易引发 **semantic collision**，导致检索混淆、幻觉（hallucination）和事实错误。

### 提出的新方法：CHOP 框架
作者提出 **CHOP（Chunkwise Context-Preserving Framework）**，通过引入两个关键模块实现上下文感知的分块与表示增强：

#### （1）CNM-Extractor（Category-Noun-Model Extractor）
- 从每个 chunk 中提取紧凑的三元组签名 `CNM = {Category, Nouns, Model}`：
  - **Category**：所属产品类别（如 air conditioner）
  - **Nouns**：核心名词短语（如 air conditioner filter）
  - **Model**：具体型号/系列（如 225B）
- 输出为结构化 JSON，确保格式一致性，并作为前缀注入原始 chunk。

#### （2）Continuity Decision Module
- 判断相邻 chunk 是否属于同一文档流（topical flow）：
  - 若 `continuity = TRUE` → 继承前一个 chunk 的 CNM
  - 若 `continuity = FALSE` → 重新提取新的 CNM
- 使用 LLM 作为分类器，基于预定义规则进行决策（见 Listing 2），保持保守策略以避免过度分割。

最终，每个 chunk 被表示为 `[PFX(M); C_i]` 并嵌入向量空间，从而构建更具判别性的 vector database。

### 相比现有方法的优势
| 方面 | CHOP 的优势 |
|------|-------------|
| **上下文保留** | 显式建模 chunk 间的连续性，维持 discourse-level coherence |
| **语义区分度** | CNM 前缀缓解相似 chunk 的 embedding collision，提升 retriever discrimination |
| **无需训练** | 完全基于 prompt-driven LLM 推理，不依赖额外训练 |
| **可扩展性** | 适用于长文档、高重叠、多来源的技术文档集合 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **MRAMG-Bench** [Yu et al. 2025]，源自 ManualsLib 的产品说明书数据集。
- 对原始碎片化文档进行重构，合并为连续长文档，模拟真实使用场景。
- 移除显式的文档边界提示，迫使模型依赖上下文连续性判断。

### 实验设置
- **Embedding 模型**：OpenAI’s `text-embedding-3-large`（3072维）
- **向量数据库**：ChromaDB + HNSW 索引用于近似最近邻搜索
- **LLM 主干**：Gemma-12B（所有推理任务中 temperature=0，保证确定性输出）
- **检索方式**：余弦相似度匹配，返回 top-k 最相关 chunks

### 评估指标

#### （1）检索性能（Retrieval Evaluation）
| 指标 | 描述 |
|------|------|
| **Hit@K** | Top-K 结果中包含至少一个黄金证据的比例 |
| **MRR@K** | 第一个相关 chunk 的平均倒数排名 |
| **NDCG@K** | 归一化的折损累积增益，衡量排序质量 |

#### （2）生成性能（Generation Evaluation）
| 指标 | 描述 |
|------|------|
| **F1** | 预测与参考答案之间的 token 级精确率与召回率调和平均 |
| **ROUGE-L** | 基于最长公共子序列的顺序保留相似性 |
| **BERTScore** | 基于预训练 embedding 的语义匹配分数，对 paraphrase 更鲁棒 |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Naive-500T** | 固定大小分块（500 tokens），重叠 100 tokens |
| **Cosine-Chunking** [Singh et al. 2024] | 基于句子级 cosine similarity 检测主题切换点，动态分块（阈值=0.35） |
| **CHOP**（本文） | 提出的方法，结合 CNM 注入与 continuity-aware 分块 |

---

## 3. 主要实验结果和性能指标

### 检索性能（Table 1）

| Method | Hit@1 | MRR@1 | NDCG@1 | Hit@3 | MRR@3 | NDCG@3 | Hit@5 | MRR@5 | NDCG@5 | Hit@10 | MRR@10 | NDCG@10 |
|--------|-------|--------|--------|--------|--------|--------|--------|--------|--------|---------|---------|----------|
| Naive-500T | 0.8128 | 0.8551 | 0.8656 | 0.9103 | 0.8637 | 0.8716 | 0.9487 | 0.8676 | 0.8716 | 0.9744 | 0.8676 | 0.8698 |
| Cosine-Chunking | 0.7077 | 0.7944 | 0.8309 | 0.8974 | 0.8052 | 0.8442 | 0.9436 | 0.8110 | 0.8389 | 0.9846 | 0.8110 | 0.8389 |
| **CHOP** | **0.9077** | **0.9325** | **0.9380** | **0.9641** | **0.9368** | **0.9380** | **0.9821** | **0.9368** | **0.9380** | **0.9923** | **0.9381** | **0.9291** |

> ✅ **关键发现**：
- CHOP 在 **Hit@1 上显著领先**（+9.5% vs Naive-500T，+20% vs Cosine-Chunking）
- 所有 K 下的 **MRR@K 和 NDCG@K 均最优**，说明其不仅找到正确 chunk，还能将其排在更高位置
- 尽管 Cosine-Chunking 力图捕捉主题变化，但在实际中表现最差，可能因噪声敏感导致误切

### 生成性能（Table 2）

| Method | F1@1 | ROUGE-L@1 | BERTScore@1 | F1@3 | ROUGE-L@3 | BERTScore@3 | F1@5 | ROUGE-L@5 | BERTScore@5 | F1@10 | ROUGE-L@10 | BERTScore@10 |
|--------|------|-----------|--------------|------|------------|----------------|------|------------|----------------|--------|-------------|------------------|
| Naive-500T | 0.2763 | 0.2472 | 0.6992 | 0.3497 | 0.3144 | 0.7253 | 0.3792 | 0.3394 | 0.7338 | 0.4072 | 0.3573 | 0.7381 |
| Cosine-Chunking | 0.2399 | 0.2143 | 0.6849 | 0.3093 | 0.2730 | 0.7119 | 0.3351 | 0.2962 | 0.7206 | 0.3577 | 0.3107 | 0.7256 |
| **CHOP** | **0.2760** | **0.2475** | **0.6998** | **0.3513** | **0.3164** | **0.7257** | **0.3814** | **0.3412** | **0.7349** | **0.4080** | **0.3591** | **0.7396** |

> ✅ **关键发现**：
- CHOP 在几乎所有指标上均优于 Naive-500T，尤其在 **Top-3 及以上时优势更明显**
- 绝对 F1 提升达 **+0.0266 ~ +0.0753**，表明更准确的 retrieval 显著改善 generation 质量
- BERTScore 持续领先，说明生成内容在语义层面更贴近参考答案
- Cosine-Chunking 表现最差，反映其分块策略未能有效支持 downstream generation

> ⚠️ 注意：虽然绝对提升看似不大，但作者强调这是在 **相同 embedding 和 generation 模型下仅改变 chunking 策略的结果**，因此微小改进具有实际意义。

---

## 4. 关键结论和发现

### 主要发现
1. **chunk-level 独立处理会损害 retrieval 性能**，尤其是在多文档、高重叠场景中；
2. 引入 **context-aware metadata prefix（CNM）** 可有效缓解 semantic collision，提高 retriever 判别能力；
3. **continuity-aware 分块机制** 能够维持文档内部逻辑流，减少 topic drift 和 reference loss；
4. CHOP 不需要 retrain retriever 或 generator，即可在现有 RAG pipeline 中 plug-and-play 地提升整体性能；
5. 更高的 retrieval ranking quality（MRR/NDCG）直接转化为更好的 generation fidelity，验证了“better retrieval → better generation”的假设。

### 方法的局限性
- **依赖 LLM 进行 CNM 提取与 continuity 决策**，带来额外计算开销，尤其在大规模文档集中；
- 当前 CNM schema（category/noun/model）针对技术文档设计，通用性有待验证；
- 对非结构化或叙事性强的文本（如小说、新闻）适应性未知；
- 未提供轻量化版本，在资源受限环境下部署成本较高。

### 未来工作方向
1. **Adaptive prefixing**：根据 query 类型动态调整 CNM 内容或粒度；
2. **Dynamic continuity modeling**：支持 streaming input 和 evolving knowledge base；
3. **Lightweight strategies**：探索小型模型替代 LLM 进行 CNM extraction 与 continuity decision；
4. **Efficient inference techniques**：优化 prompt engineering 或缓存机制以降低延迟；
5. 扩展至 **multimodal RAG** 场景，结合图像、表格等富媒体信息。

---

> 🔚 **总结一句话**：  
> **CHOP 通过引入 context-preserving 的 chunk 表示机制，在无需训练的前提下显著提升了 RAG 系统在多文档环境下的 retrieval 与 generation 质量，为构建高质量 knowledge base 提供了一种实用且可扩展的新范式。**

</details>

---

### 14. [BlockRaFT: A Distributed Framework for Fault-Tolerant and Scalable Blockchain Nodes](https://arxiv.org/abs/2604.15731)

**Authors**: Manaswini Piduguralla, Souvik Sarkar, Arunmoezhi Ramachandran, Sathya Peri  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.15731v1  

#### Abstract
Blockchain technology enhances transparency by maintaining a distributed ledger among mutually untrusting parties. Despite its advantages, scalability and availability remain critical bottlenecks that hinder widespread adoption. The increasing complexity of blockchain nodes further necessitates robu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：BlockRaFT: A Distributed Framework for Fault-Tolerant and Scalable Blockchain Nodes**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
区块链节点在**可扩展性**（scalability）和**可用性**（availability）方面面临严重瓶颈，尤其是在**许可链**（permissioned blockchain）环境中：
- 单个组织通常只运行一个节点，一旦该节点崩溃，整个组织将失去对网络的访问能力（无法提交交易、查询状态等），形成“组织级单点故障”。
- 传统解决方案（如增加节点数量）会显著增加**消息复杂度**（message complexity），影响共识效率，并破坏各组织间的公平性。

此外，**Merkle Tree 更新开销大**是智能合约执行中的主要性能瓶颈，因其串行更新机制限制了并行执行能力。

---

### **提出的新方法与新思路**

#### **(1) BlockRaFT 分布式节点框架**
- 设计了一个基于 **RAFT 共识协议**的**领导者-跟随者**（leader-follower）集群模型，将一个逻辑上的“区块链节点”由多个物理系统组成。
- 集群内部通过 RAFT 选举 leader，对外表现为单一节点，避免破坏网络公平性。
- 工作负载按**有状态**（stateful）和**无状态**（stateless）划分：
  - **Stateless operations**（如交易收集、DAG 构建）由 leader 统一协调。
  - **Stateful operations**（如状态更新、Merkle Tree 维护）在所有节点上复制，确保一致性与容错。

#### **(2) 三阶段并发 Merkle Tree 优化**
为减少 Merkle Tree 更新带来的性能开销，提出一种解耦执行与更新的策略：
1. **并发执行 + 局部记录**：执行期间不直接修改 Merkle Tree，而是将变更写入本地 `concurrent hash map`。
2. **并行叶子节点更新**：所有交易执行完成后，将最终状态变更**并行地应用到叶子节点**。
3. **顺序父节点重计算**：仅在最后**顺序重建受影响路径的内部节点**，生成新的 Merkle Root。

此设计显著减少了树更新的同步开销，支持高并发智能合约执行。

---

### **相比现有方法的优势**

| 特性 | BlockRaFT | DiPETrans | PilotFish | Sharding |
|------|---------|----------|-----------|--------|
| 可扩展性 | ✅ | ✅ | ✅ | ✅ |
| 容错能力（crash tolerance） | ✅ | ❌ | ❌ | ❌ |
| 全流程工作负载分发 | ✅ | ❌ | ⚠️（部分） | ❌ |
| 不改变链上共识公平性 | ✅ | ❌（多节点=更多投票权） | ✅ | ❌（需架构改动） |
| 信任模型 | Trustless | Trusted Community | Trustless | Trustless/⚠️ |
| 与现有系统兼容 | ✅ | ❌ | ✅ | ❌ |

> ✅ 表示具备；❌ 表示不具备；⚠️ 表示有限支持

**核心优势总结**：
- 在不牺牲网络公平性和安全性的前提下，实现**组织级容错**。
- 支持**细粒度并行执行**，提升吞吐量。
- 引入轻量级协调机制，**额外开销可控**，性能收益远超代价。

---

## **2. 核心实验方法和设置**

### **实验环境**
- **操作系统**：Ubuntu 64-bit
- **硬件配置**：
  - 总体测试平台：AMD EPYC 7452，128核，251GB内存
  - 单节点配置：16核，9.7GB内存
- **实现语言**：C++
- **通信机制**：
  - 主要使用 **ETCD** 作为分布式共享内存（基于其内置 RAFT 协议）
  - 对比实现了基于消息传递（message-passing）的版本（BlockRaFT-Msg）

---

### **数据集与工作负载**
- 使用 **YCSB**（Yahoo! Cloud Serving Benchmark）生成交易负载。
- 智能合约类型：
  - **Voting Contract**：模拟去中心化投票系统（注册、投票、转账、查询）。
  - **Wallet Contract**：模拟钱包操作（存款、取款、转账、余额查询）。
- 负载参数可调：
  - 读写比例（0% ~ 100%）
  - 每区块交易数（1000 ~ 5000）
  - 冲突率（dependency percentage，0% ~ 5%）
  - 线程数（2 ~ 64）
  - 集群规模（3/5/7 节点）

---

### **评估指标**
- **Execution Time**（执行时间，ms）
- **Throughput**（吞吐量，tx/s）
- **Scalability**（随负载增长的表现）
- **Fault Tolerance**（节点崩溃下的性能退化）
- **Breakdown Analysis**（各模块耗时占比）

---

### **基线方法对比**
1. **Single-core Execution Model**  
   - 单线程顺序执行，代表传统区块链节点行为。
2. **Multi-core Execution Model**  
   - 多线程并行执行，共享内存，使用相同 DAG 和 Merkle Tree 优化，但**无容错机制**。
3. **BlockRaFT-Msg**  
   - 替代通信方式验证 ETCD 开销影响。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 并发 Merkle Tree 优化效果**
- **写密集型负载下（100% writes）**：
  - 执行时间比传统串行 Merkle Tree **降低两个数量级以上**。
- **大规模操作（100K ops）**：
  - 并发版本执行时间稳定在约 **0.35秒**，而串行版本超过 **80秒**。
- **线程扩展性**：
  - 最佳性能出现在 **32线程**，64线程时略有下降（因同步开销）。
- **读密集型负载**：
  - 仍保持约 **5倍加速**，得益于并行执行层。

> 图3 显示并发 Merkle Tree 在各类负载下均显著优于串行实现。

---

#### **(2) BlockRaFT 整体性能（以 Voting Contract 为例）**

| 参数 | BlockRaFT（ms） | Multi-core（ms） | Single-core（ms） |
|------|------------------|------------------|--------------------|
| 4000 txns, 16 threads, 5% conflict | **1626.6** | 1883.6 | 8452.2 |
| 5000 txns/block, 3-node cluster | **2436.6** | ~3000+ | >12000 |
| 64 threads | **2944** | 1779.6 | 8452.2 |

> 尽管在极高线程数下 multi-core 更快，但 BlockRaFT 提供了**容错能力**，这是不可替代的价值。

---

#### **(3) 容错能力测试（RQ4）**
- 在 **3/5/7 节点集群**中引入 1~3 个节点崩溃：
  - 第一次崩溃后执行时间上升（负载重新分配）。
  - 后续崩溃影响较小，系统仍维持运行。
  - 只要满足多数派（quorum），即可恢复 leader 并继续处理。
- 示例（Wallet Contract，3节点）：
  - 0 crash: 3934.4 ms
  - 1 crash: 7890.8 ms （≈ +100%）
  - 2 crash: N/A（低于 quorum，停止服务）

> 表明系统具有**优雅降级**（graceful degradation）特性。

---

#### **(4) 消融实验与瓶颈分析**
- **Breakdown 分析显示主要开销来源**：
  - **Execution Phase**（执行阶段）
  - **Component Detection**（连通分量检测）
  - **ETCD Communication Overhead**（尤其在低冲突场景下明显）
- 在 **0% 冲突**时性能反而较差 → 因大量独立交易导致跨节点数据传输增多，暴露通信瓶颈。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **BlockRaFT 能有效解决组织级单点故障问题**，通过内部集群化实现 crash tolerance，且不影响链上公平性。
2. ✅ **近线性可扩展性**：随着交易量增加，执行时间呈近似线性增长，优于单核模型。
3. ✅ **并发 Merkle Tree 显著降低状态更新开销**，尤其在写密集和大规模负载下表现优异。
4. ✅ **容错开销合理**：相比 multi-core 模型仅有轻微性能损失，但获得了宝贵的 fault tolerance 能力。
5. ✅ **系统具备优雅降级能力**：在部分节点失效时仍能维持基本功能。

---

### **方法的局限性**
1. **通信成为新瓶颈**：
   - 尤其在低冲突、高并行场景下，ETCD 或消息传递的延迟限制了进一步提速。
2. **组件检测与调度开销较高**：
   - DAG 构建与连通分量识别（DSU算法）在大数据量下耗时显著。
3. **未完全去中心化**：
   - 当前为 leader-centric 架构，存在潜在 leader 成为性能热点的风险。
4. **依赖外部系统（ETCD）**：
   - 增加部署复杂性，虽简化开发，但也引入外部依赖。

---

### **未来工作方向**
1. **优化数据共享机制**：
   - 减少跨节点状态同步开销，特别是在低冲突场景。
2. **探索更去中心化的 P2P 架构**：
   - 替代当前 leader-driven 模式，提升鲁棒性。
3. **改进组件检测算法**：
   - 加速 DAG 分解过程，提升高负载下的响应速度。
4. **集成更高效的存储引擎**：
   - 如结合 Jellyfish Merkle Tree 或 Angela 等稀疏 Merkle Tree 结构，进一步优化持久化性能。

---

> **代码开源地址**：[https://github.com/PDCRL/SCT-DistFramework.git](https://github.com/PDCRL/SCT-DistFramework.git)  
> **匿名仓库**：[https://anonymous.4open.science/r/BlockRAFT-00C1](https://anonymous.4open.science/r/BlockRAFT-00C1)

</details>

---

### 15. [Structured Abductive-Deductive-Inductive Reasoning for LLMs via Algebraic Invariants](https://arxiv.org/abs/2604.15727)

**Authors**: Sankalp Gilda, Shlok Gilda  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.15727v1  

#### Abstract
Large language models exhibit systematic limitations in structured logical reasoning: they conflate hypothesis generation with verification, cannot distinguish conjecture from validated knowledge, and allow weak reasoning steps to propagate unchecked through inference chains. We present a symbolic r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Structured Abductive-Deductive-Inductive Reasoning for LLMs via Algebraic Invariants*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在**结构化逻辑推理**任务中存在系统性缺陷，具体表现为：
- **混淆推理模式**：将假设生成（abduction）、逻辑验证（deduction）和经验验证（induction）混为一谈。
- **缺乏可靠性传播机制**：弱前提可能通过推理链不受控地传播，导致结论不可靠。
- **推理不一致**：chain-of-thought（CoT）解释仅25–39%忠实于模型真实计算过程（Anthropic, 2025），用户无法信任推理路径。
- **“复杂性诅咒”**（curse of complexity）：随着问题复杂度上升，准确率急剧下降。

### 🚀 提出的新方法与创新思路
作者提出一个**符号化推理框架**（symbolic reasoning scaffold），其核心是：
#### （1）**ADI 推理协议**（Abduction-Deduction-Induction Protocol）
将 LLM 推理分解为三个显式、可审计的阶段：
| 阶段 | 作用 | 输出层 | 可靠性上限 |
|------|------|--------|-----------|
| **Abduction**（假设生成） | 生成候选解释 | L0（conjecture） | ≤ 35% |
| **Deduction**（逻辑验证） | 检查是否与已有知识冲突 | L1（substantiated） | ≤ 75% |
| **Induction**（经验验证） | 用实证证据验证 | L2（corroborated） | ≤ 100% |

> 每个阶段有明确的**认知承诺**（epistemic commitment）和验证要求。

#### （2）**Gamma Quintet：五个代数不变量**（algebraic invariants）
用于约束推理链中可靠性（Reliability）的传播，确保逻辑一致性：
1. **IDEM**（Idempotence）：单一前提保留原可靠性。
2. **COMM**（Commutativity）：前提顺序不影响聚合结果。
3. **LOC**（Locality）：修改某前提只影响依赖它的结论。
4. **WLNK**（Weakest Link）：**结论可靠性 ≤ 最弱前提的可靠性**（核心约束）。
5. **MONO**（Monotonicity）：加强前提不会削弱结论。

> 其中 **WLNK 不变量** 是防止逻辑矛盾累积的关键机制。

#### （3）**外部推理支架架构**（External Reasoning Scaffold）
- LLM 负责自然语言假设生成；
- 外部符号系统维护**知识图谱**，跟踪每个声明的：
  - 形式性（Formality, F0–F3）
  - 范围（Scope）
  - 可靠性（Reliability）
  - 推理模式标签（ADI 阶段）
- 实现**认知状态分离**，避免自举循环（self-promotion loop）。

### 🔍 相比现有方法的优势
| 方法 | 缺陷 | 本论文改进 |
|------|------|------------|
| Chain-of-Thought (CoT) | 无形式保证，易出现“伪装演绎”的归纳跳跃 | 显式区分推理模式，强制结构化流程 |
| Self-Consistency | 多路径投票，但不验证路径正确性 | 引入外部实证验证（Induction） |
| Process Reward Models | 评分单步，无全局一致性约束 | 通过 WLNK 和图结构实现跨步一致性 |
| Neuro-symbolic Systems | 需完全形式化领域知识，成本高 | 轻量级符号结构，适用于任意领域 |

> ✅ **无需修改 LLM 内部结构**，作为外部系统即可增强推理可靠性。

---

## 2. 核心实验方法和设置

### 📚 数据集与测试方式
论文未使用传统 NLP 数据集进行端到端任务评测，而是采用：
#### **Property-Based Testing (PBT)** + **Fuzz Testing**
- **目标**：验证框架实现是否满足 Gamma Quintet 等代数不变量。
- **规模**：
  - **100+ 项属性测试**
  - **16 项模糊测试**
  - 每项测试运行 **超过 $10^5$** 随机生成案例
- **工具参考**：受 QuickCheck 启发，基于工业级 PBT 方法论（Arts et al., 2006）

### ⚙️ 实验设置
测试覆盖五大模块：
| 测试模块 | 测试内容 | 示例 |
|--------|---------|------|
| `Reff` 计算器 | 可靠性公式是否满足 WLNK、天花板约束等 | 验证 `min()` 是否始终主导 |
| Scope 代数 | 范围匹配、格结构（lattice）公理 | 验证上下文迁移惩罚 |
| Epistemic FSM | 推理状态机是否禁止跳层升级 | 如不能从 L0 直接到 L2 |
| 图拓扑 | 在深链、菱形结构中 WLNK 是否传播正确 | 多路径聚合行为 |
| Dependency Inspector | BFS 遍历、去重、层级保持 | 依赖追踪准确性 |

> 所有测试均通过随机生成输入来验证不变量成立。

### 📊 评估指标
- **不变量满足率**（Invariant Compliance Rate）：每项测试下所有生成案例中违反不变量的比例。
- **边界条件鲁棒性**：如 IEEE 754 浮点边界、并发访问、解析-序列化往返。
- **形式化正确性**：是否符合 t-norm 理论、possibilistic logic 等数学基础。

### 🆚 基线方法对比
本文不直接对比 accuracy 类指标，而是从**推理结构保真度**角度对比：
| 方法 | 是否支持 ADI 分离 | 是否满足 WLNK | 是否可审计 |
|------|------------------|---------------|-----------|
| Standard CoT | ❌ | ❌（mean 聚合违反） | ❌ |
| Self-Consistency | ❌ | ❌ | ❌ |
| PRM (Process Reward Model) | ❌ | ❌ | ⭕（部分） |
| **本文框架** | ✅ | ✅（强制 min 聚合） | ✅（完整审计轨迹） |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据
- **100% 不变量满足率**：在超过 $10^5$ 随机测试案例中，所有 100+ 属性测试和 16 项 fuzz 测试均通过。
- **WLNK 强制生效**：任何复合结论的 `Reff` 均 ≤ 其最弱前提，且差值平均被压缩 42%（模拟实验）。
- **双天花板机制有效**：
  - L0 声明即使形式为 F3，仍受限于 35% 层级上限。
  - F0 证据即使 L2 也无法超过 70% 形式性上限。

### 📉 与基线方法对比结果
| 聚合算子 | IDEM | COMM | WLNK | MONO | 推荐用途 |
|--------|------|------|------|------|----------|
| **min (Gödel t-norm)** | ✅ | ✅ | ✅ | ✅ | **推荐：串行推理链** |
| product | ✅ | ✅ | ~（近似满足） | ✅ | 独立证据融合 |
| mean | ❌ | ✅ | ❌ | ✅ | ❌ 不推荐 |
| max | ✅ | ✅ | ❌ | ✅ | ❌ 不推荐 |

> 使用 `mean` 会导致三个 R=0.4 的弱证据被误判为中等可信（0.4），掩盖质量缺陷。

### 🔍 消融实验（隐含分析）
虽然未设标准消融实验，但文中通过反例说明：
- 若移除 **WLNK**：矛盾前提（如 R=0.9 和 R=0.3）可同时支撑中等可靠结论（如 0.6），隐藏知识库不一致。
- 若允许 LLM 自我认证：会形成**信心自举循环**（confidence bootstrapping），违反 Transformer Mandate。
- 若取消外部验证：L1 升级无法保证逻辑一致性，退化为普通 CoT。

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **WLNK 是维持多步推理一致性的必要条件**：
   - 理论上源于 **possibilistic logic** 中的 weakest link resolution。
   - 经验上由 Jacovi et al. (2024) 在 CoT 中验证。
   - 数学上是唯一满足 **idempotent continuous t-norm** 的聚合函数（即 `min`）。

2. **三重三角验证支持 `min` 聚合**：
   - **代数规范**（Gamma Quintet）
   - **可能性理论**（Possibilistic Logic）
   - **经验测量**（CoT 步骤可靠性衰减）
   - **t-norm 理论**（唯一连续幂等范数）

3. **ADI 协议使推理过程可审计、可干预**：
   - 每个声明标注其来源、模式、可靠性、范围。
   - 支持自动检测跨步骤矛盾与过期知识。

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **非端到端训练集成** | 当前为外部系统，尚未作为可微模块嵌入训练 |
| **依赖多轮交互** | 无法在单次 autoregressive 生成中完成 ADI 循环 |
| **天花板值为策略默认值** | 如 CL₀=35% 尚未经大规模实证校准 |
| **验证依赖 PBT 而非定理证明** | 提供高置信度，但非形式化完备证明（如 Coq） |
| **初步评估集中于工程任务** | 在 ZebraLogic、FOLIO 等逻辑推理基准上的表现待验证 |

### 🔮 未来工作方向
1. **将 WLNK 作为可微训练目标**：探索在 RL 或 contrastive learning 中引入 weakest-link 损失。
2. **多智能体 ADI 架构**：不同 agent 分别负责 abduction、deduction、induction。
3. **分解 epistemic 与 aleatoric uncertainty**：在 `Reff` 中区分可减少与不可减少的不确定性。
4. **程序化工具调用集成**：让 LLM 通过 structured function calls 查询知识图谱。
5. **跨领域校准 ceiling 参数**：基于实际任务结果动态调整可靠性上限。

---

## 总结
该论文提出了一个**结构化、可验证、抗幻觉的 LLM 推理增强框架**，通过：
- **ADI 协议** 实现推理模式解耦，
- **Gamma Quintet**（尤其是 WLNK）保障逻辑一致性，
- **外部符号系统** 提供审计能力。

它不是对 LLM 的替代，而是一个**认知脚手架**（reasoning scaffold），有望成为未来**可信 AI 推理系统的基础组件**。

</details>

---

### 16. [Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents](https://arxiv.org/abs/2604.15877)

**Authors**: Xing Zhang, Guanghui Wang, Yanwei Cui, Wei Qiu, Ziyuan Li, Bing Zhu, Peiyang He  
**Category**: cs.AI  
**Published**: 2026-04-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.15877v1  

#### Abstract
As LLM agents scale to long-horizon, multi-session deployments, efficiently managing accumulated experience becomes a critical bottleneck. Agent memory systems and agent skill discovery both address this challenge -- extracting reusable knowledge from interaction traces -- yet a citation analysis of...

---

### 17. [Think Multilingual, Not Harder: A Data-Efficient Framework for Teaching Reasoning Models to Code-Switch](https://arxiv.org/abs/2604.15490)

**Authors**: Eleanor M. Lin, David Jurgens  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.15490v1  

#### Abstract
Recent developments in reasoning capabilities have enabled large language models to solve increasingly complex mathematical, symbolic, and logical tasks. Interestingly, while reasoning models are often trained to generate monolingual text, these models have also been observed to code-switch (i.e., m...

---

### 18. [Evaluating SYCL as a Unified Programming Model for Heterogeneous Systems](https://arxiv.org/abs/2604.16043)

**Authors**: Ami Marowka  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.16043v1  

#### Abstract
High-performance computing (HPC) applications are increasingly executed in heterogeneous environments, introducing new challenges for programming and software portability. SYCL has emerged as a leading model designed to simplify heterogeneous programming and make it more accessible to developers. In...

---

### 19. [Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models](https://arxiv.org/abs/2604.15609)

**Authors**: Yunbei Zhang, Shuaicheng Niu, Chengyi Cai, Feng Liu, Jihun Hamm  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.15609v1  

#### Abstract
Test-Time Adaptation (TTA) for black-box models accessible only via APIs remains a largely unexplored challenge. Existing approaches such as post-hoc output refinement offer limited adaptive capacity, while Zeroth-Order Optimization (ZOO) enables input-space adaptation but faces high query costs and...

---

### 20. [Fusing Cellular Network Data and Tollbooth Counts for Urban Traffic Flow Estimation](https://arxiv.org/abs/2604.15782)

**Authors**: Oluwaleke Yusuf, Shaira Tabassum  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.15782v1  

#### Abstract
Traffic simulations, essential for planning urban transit infrastructure interventions, require vehicle-category-specific origin-destination (OD) data. Existing data sources are imperfect: sparse tollbooth sensors provide accurate vehicle counts by category, while extensive mobility data from cellul...

---

### 21. [GroupDPO: Memory efficient Group-wise Direct Preference Optimization](https://arxiv.org/abs/2604.15602)

**Authors**: Jixuan Leng, Si Si, Hsiang-Fu Yu, Vinod Raman, Inderjit S. Dhillon  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.15602v1  

#### Abstract
Preference optimization is widely used to align Large Language Models (LLMs) with preference feedback. However, most existing methods train on a single positive-negative pair per prompt, discarding additional supervision available in preference datasets that typically contain multiple candidate resp...

---

### 22. [C-Mining: Unsupervised Discovery of Seeds for Cultural Data Synthesis via Geometric Misalignment](https://arxiv.org/abs/2604.15675)

**Authors**: Pufan Zeng, Yilun Liu, Mingchen Dai, Mengyao Piao, Chunguang Zhao, Lingqi Miao, Shimin Tao, Weibin Meng, Minggui He, Chenxin Liu, Zhenzhen Qin, Li Zhang, Hongxia Ma, Boxing Chen, Daimeng Wei  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.15675v1  

#### Abstract
Achieving cultural alignment in Large Language Models (LLMs) increasingly depends on synthetic data generation. For such synthesis, the most vital initial step is seed curation; however, current methods lack quantifiable standards for selecting these seeds. Existing approaches rely on unscalable man...

---

### 23. [A Systematic Study of Training-Free Methods for Trustworthy Large Language Models](https://arxiv.org/abs/2604.15789)

**Authors**: Wai Man Si, Mingjie Li, Michael Backes, Yang Zhang  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.15789v1  

#### Abstract
As Large Language Models (LLMs) receive increasing attention and are being deployed across various domains, their potential risks, including generating harmful or biased content, producing unsupported claims, and exhibiting vulnerabilities to adversarial attacks, have drawn significant attention. To...

---

### 24. [CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](https://arxiv.org/abs/2604.15840)

**Authors**: Shidong Yang, Ziyu Ma, Tongwen Huang, Yiming Hu, Yong Wang, Xiangxiang Chu  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.15840v1  

#### Abstract
Reinforcement learning for LLM agents is typically conducted on a static data distribution, which fails to adapt to the agent's evolving behavior and leads to poor coverage of complex environment interactions. To address these challenges, we propose CoEvolve, an agent-data mutual evolution framework...

---

### 25. [Accuracy Is Speed: Towards Long-Context-Aware Routing for Distributed LLM Serving](https://arxiv.org/abs/2604.15732)

**Authors**: Takeshi Yoshimura, Valentijn Dymphnus van de Beek, Tatsuhiro Chiba  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.15732v1  

#### Abstract
Distributed LLM serving systems optimize per-request latency and throughput. However, under long-context workloads, inference accuracy becomes more variable. When incorrect responses trigger retries, accuracy directly translates into cumulative user-visible delay that is not captured by single-shot ...

---

### 26. [T-RBFT: A Scalable and Efficient Byzantine Consensus Based on Trusted Execution Environment for Consortium Blockchain](https://arxiv.org/abs/2604.16053)

**Authors**: Wen Gao, Xinhong Hei, Yichuan Wang  
**Category**: cs.DC  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.16053v1  

#### Abstract
With the continuous expansion of blockchain application scenarios, consortium chains have raised higher performance and security requirements for consensus mechanisms. Unlike public blockchains, consortium chains typically implement an admission mechanism that restricts participation to trusted enti...

---

### 27. [Similarity-Based Bike Station Expansion via Hybrid Denoising Autoencoders](https://arxiv.org/abs/2604.15783)

**Authors**: Oluwaleke Yusuf, M. Tsaqif Wismadi, Adil Rasheed  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.15783v1  

#### Abstract
Urban bike-sharing systems require strategic station expansion to meet growing demand. Traditional allocation approaches rely on explicit demand modelling that may not capture the urban characteristics distinguishing successful stations. This study addresses the need to exploit patterns from existin...

---

### 28. [Geometric regularization of autoencoders via observed stochastic dynamics](https://arxiv.org/abs/2604.16282)

**Authors**: Sean Hill, Felix X. -F. Ye  
**Category**: cs.LG  
**Published**: 2026-04-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.16282v1  

#### Abstract
Stochastic dynamical systems with slow or metastable behavior evolve, on long time scales, on an unknown low-dimensional manifold in high-dimensional ambient space. Building a reduced simulator from short-burst ambient ensembles is a long-standing problem: local-chart methods like ATLAS suffer from ...

---

### 29. [Target-Oriented Pretraining Data Selection via Neuron-Activated Graph](https://arxiv.org/abs/2604.15706)

**Authors**: Zijun Wang, Haoqin Tu, Weidong Zhou, Yiyang Zhou, Xiaohuan Zhou, Bingni Zhang, Weiguo Feng, Taifeng Wang, Cihang Xie, Fengze Liu  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.15706v1  

#### Abstract
Everyday tasks come with a target, and pretraining models around this target is what turns them into experts. In this paper, we study target-oriented language model (LM) pretraining by introducing Neuron-Activated Graph Ranking (NAG-based Ranking), a training-free and interpretable framework for tar...

---

### 30. [Disentangling Mathematical Reasoning in LLMs: A Methodological Investigation of Internal Mechanisms](https://arxiv.org/abs/2604.15842)

**Authors**: Tanja Baeumel, Josef van Genabith, Simon Ostermann  
**Category**: cs.CL  
**Published**: 2026-04-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.15842v1  

#### Abstract
Large language models (LLMs) have demonstrated impressive capabilities, yet their internal mechanisms for handling reasoning-intensive tasks remain underexplored. To advance the understanding of model-internal processing mechanisms, we present an investigation of how LLMs perform arithmetic operatio...

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
