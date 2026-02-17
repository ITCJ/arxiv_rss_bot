# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-17 06:40:10 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [QuaRK: A Quantum Reservoir Kernel for Time Series Learning](https://arxiv.org/abs/2602.13531)

**Authors**: Abdallah Aaraba, Soumaya Cherkaoui, Ola Ahmad, Shengrui Wang  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2602.13531v1  

#### Abstract
Quantum reservoir computing offers a promising route for time series learning by modelling sequential data via rich quantum dynamics while the only training required happens at the level of a lightweight classical readout. However, studies featuring efficient and implementable quantum reservoir arch...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**QuaRK: A Quantum Reservoir Kernel for Time Series Learning**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Quantum Reservoir Computing (QRC)** 在时序学习中面临两大挑战：
1. **硬件实现困难**：许多 QRC 架构依赖昂贵或难以在真实量子设备上执行的测量方案（如全态层析）。
2. **缺乏理论保障**：缺少将可实现的 QRC 架构与明确的学习理论泛化界联系起来的端到端框架。

本文旨在填补这一空白，提出一个**既实用又具备理论保证**的量子时序学习框架。

---

### 🚀 提出的新方法：QUARK
作者提出了 **QUARK**（Quantum Reservoir Kernel），一个结合了硬件现实性和学习理论分析的端到端框架，其核心设计如下：

#### （1）硬件现实的量子储层特征提取器（Quantum Reservoir Featurizer）
- **Johnson-Lindenstrauss (JL) 投影**：将高维输入数据投影到与量子比特数 $n$ 匹配的低维空间，使模型资源（电路宽度/深度）独立于原始数据维度，提升可扩展性。
- **Contractive Encoding Quantum Channel (CEQC)** 架构：
  - 输入通过参数化旋转 $R_y(\theta)$ 编码。
  - 固定的硬件友好纠缠层（Ising-like unitary）用于动态演化。
  - 引入 **reset-rate channel** $\mathcal{E}_\lambda(\rho) = \lambda \rho + (1-\lambda)|+\rangle\langle+|^{\otimes n}$ 实现严格收缩，确保满足 **Echo State Property (ESP)** 和 **Fading Memory Property (FMP)**，从而稳定处理长序列。
- **Spatial Multiplexing**：使用多个具有不同参数（尤其是 $\lambda_r$）的子储层并行运行，增强表达能力。

#### （2）高效的测量与经典核读出（Kernel-based Readout）
- **Classical Shadows**：通过随机单量子比特 Pauli 测量，高效估计 $k$-局部可观测量（如 2-local Pauli）的期望值，生成紧凑特征向量。
- **Projected Quantum Kernel**：在经典特征空间上应用 **Matérn 核**，构建 Reproducing Kernel Hilbert Space (RKHS)，进行 **Kernel Ridge Regression (KRR)** 作为读出机制。
  - 优点：闭式解训练、正则化可控、避免复杂优化。

#### （3）学习理论保障
- **有效学习保证**（Theorem 1）：在理想条件下，当量子比特数 $n = \Omega(\log(wN))$ 时，存在足够大的 RKHS 范数约束 $A^*$，使得经验风险可降至接近零，表明模型具备**插值能力**。
- **泛化误差界**（Theorem 2）：针对弱依赖的 $\beta$-mixing 时间序列，给出了统一的高概率泛化界，分解为三项：
  1. **Rademacher 复杂度项**（$\mathcal{O}(1/\sqrt{N})$）——模型复杂度惩罚；
  2. **依赖性惩罚项**——由时间窗口间的 $\beta$-mixing 系数控制；
  3. **记忆衰减余项**（$\propto \lambda^w$）——由收缩因子 $\lambda$ 控制，随窗口长度 $w$ 几何衰减。

---

### ⚖️ 相比现有方法的优势
| 方面 | 现有 QRC 方法 | QUARK |
|------|----------------|--------|
| **可实现性** | 常需全态层析或复杂测量 | 使用 **Classical Shadows**，仅需 $k$-局部测量，适合近期设备 |
| **可扩展性** | 通常依赖数据维度决定量子资源 | 通过 **JL 投影** 解耦数据维与量子资源，支持高维输入 |
| **理论保障** | 多数缺乏有限样本泛化界 | 提供 **PAC 风格泛化界**，连接设计选择（$n, R, \lambda, m$）与性能 |
| **训练效率** | 读出常需梯度优化 | 使用 **KRR**，闭式求解，快速且正则化明确 |

---

## 2. 核心实验方法和设置

### 📊 数据集
使用合成的 **B-mixing 向量自回归滑动平均过程 (VARMA)**：
- 模型：$\text{VARMA}(3,3)$，即 $Z_t = \sum_{i=1}^3 \Phi_i Z_{t-i} + \Theta_0 e_t + \sum_{j=1}^3 \Theta_j e_{t-j}$，其中 $e_t \sim \mathcal{N}(0, I)$。
- 输入变换：$X_t = \tanh(Z_t)$，确保输入有界 $X_t \in (-1,1)^d$，$d=3$。
- 输出标签由三种不同复杂度的 **ground-truth fading-memory functionals** 生成：
  1. **One-step Forecasting**: $H^*(X) = u^\top X_{t+1}$
  2. **Exponential Fading Linear**: $H^*(X) = \sum_{k=0}^{w-1} \alpha^k u^\top X_{t-k}$
  3. **Volterra Order-2**: 加入二次交叉滞后项 $(v^\top X_{t-k})(v^\top X_{t-l})$

> 所有序列均为平稳 $\beta$-mixing 过程，符合理论假设。

---

### ⚙️ 实验设置
- **窗口大小**：$w = 25$
- **步幅（stride）**：$s = 100$（即 gap $g = 75$），以减少窗口间依赖
- **训练窗口数**：$N$ 从 100 到 8000 变化
- **测试集**：固定大小的 hold-out 测试集
- **量子设置**：
  - 每个子储层 $n = 5$ 量子比特
  - 子储层数 $R = 3$
  - 测量：2-local Pauli observables
  - 测量 shots：每电路 1000 次
- **读出**：
  - Matérn 核（超参通过轻量级验证集调优）
  - 正则化参数 $\lambda_{\text{reg}}$ 扫描以验证插值现象

---

### 📈 评估指标
- **训练 MSE**：验证插值能力
- **测试 MSE**：评估泛化性能
- **消融实验**：通过调节正则化强度和训练样本量，观察模型行为是否符合理论预测

---

### 🔁 基线方法对比
本文**未直接与其他 QRC 或经典模型进行横向对比**，而是聚焦于：
- 验证自身理论预测（插值相变、泛化衰减趋势）
- 展示不同任务难度下的相对性能排序

> 这种设计更侧重于“**理论-实验一致性**”而非绝对性能领先。

---

## 3. 主要实验结果和性能指标

### ✅ 实验一：有效学习验证（插值相变）
- **结果**：图3 显示，随着正则化减弱（$\lambda_{\text{reg}} \downarrow$），训练 MSE 经历**锐利相变**，在 $\lambda_{\text{reg}} \approx 10^{-1}$ 附近进入插值区域，MSE 下降至数值零（$10^{-12} \sim 10^{-14}$）。
- **意义**：实证支持 **Theorem 1**，证明 QUARK 具备强大的拟合能力，在适当放松正则化后可完美拟合训练数据。

> 图4 进一步可视化显示，在插值状态下，预测几乎完全贴合真实标签。

---

### ✅ 实验二：泛化能力验证
- **结果**：图5 显示，固定正则化下，**测试 MSE 随训练样本数 $N$ 增加而单调下降**，近似遵循 $1/\sqrt{N}$ 趋势。
- **任务难度排序**：测试误差大小顺序为  
  `One-step < Exponential Fading < Volterra`  
  符合任务复杂度预期。
- **意义**：支持 **Theorem 2** 的泛化界预测，表明增加数据量能有效提升泛化性能，且模型对任务复杂度敏感。

---

### 🔍 消融实验（隐含）
虽然未明确标注“消融”，但以下变量被系统研究：
- **正则化强度**：控制插值与否
- **训练样本量 $N$**：影响泛化误差
- **任务复杂度**：反映模型对不同函数类的适应性

> 结果均与理论一致，间接完成消融分析。

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **QUARK 是首个兼具硬件可行性与学习理论保障的端到端 QRC 框架**。
2. **插值现象存在**：在足够小的正则化下，模型可达到零训练误差，验证了高表达能力。
3. **泛化行为符合理论预测**：测试误差随样本量增加而下降，且受任务复杂度影响。
4. **设计选择直接影响性能**：量子比特数、子储层数、收缩因子 $\lambda$、测量预算等均可通过理论界进行权衡指导。

---

### ⚠️ 局限性
1. **基于模拟器**：所有实验在经典模拟器上完成，尚未在真实含噪量子设备上验证。
2. **未考虑测量噪声与有限 shot 效应**：Classical Shadows 的估计误差未纳入理论分析。
3. **合成数据集**：使用的是人工生成的 VARMA 数据，尚未在真实世界时间序列（如金融、医疗）上测试。
4. **无与其他 SOTA 方法对比**：缺乏与经典 RNN、Transformer 或其他 QML 模型的性能比较。

---

### 🔮 未来工作方向
1. **纳入噪声建模**：将有限 shot 误差、门噪声等纳入泛化界分析。
2. **优化资源分配**：研究测量预算 $m$、可观测量局部性 $k$、multiplexing 数 $R$ 之间的权衡。
3. **真实数据验证**：在更高维、非平稳的真实时间序列上测试 QUARK。
4. **自动超参调优**：开发针对量子核超参（如 Matérn 的 $\nu, \ell$）的自动化搜索策略。
5. **探索反馈机制**：引入输出反馈以增强长期依赖建模能力。

---

## 总结
**QUARK** 成功地将量子储层计算的表达力与经典核方法的稳定性、可解释性和理论保障相结合，为**近期量子设备上的时序学习**提供了一个兼具实用性与严谨性的新范式。其实验结果有力支持了其理论预测，标志着 QRC 从“黑箱启发式”向“可理解、可分析”的机器学习组件迈出了重要一步。

</details>

---

### 2. [Data-driven Bi-level Optimization of Thermal Power Systems with embedded Artificial Neural Networks](https://arxiv.org/abs/2602.13746)

**Authors**: Talha Ansar, Muhammad Mujtaba Abbas, Ramit Debnath, Vivek Dua, Waqar Muhammad Ashraf  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2602.13746v1  

#### Abstract
Industrial thermal power systems have coupled performance variables with hierarchical order of importance, making their simultaneous optimization computationally challenging or infeasible. This barrier limits the integrated and computationally scaleable operation optimization of industrial thermal p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Data-driven Bi-level Optimization of Thermal Power Systems with embedded Artificial Neural Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
工业热力发电系统（如燃煤电厂、燃气轮机）存在多层级、耦合性强且具有竞争关系的性能变量（如功率输出、燃料消耗、热效率等）。传统的多目标优化方法难以处理这种**分层决策结构**（hierarchical decision structure），导致计算复杂度高、求解困难甚至不可行。

具体挑战包括：
- 上层目标（如最大化功率输出）与下层目标（如最小化汽轮机热耗率 THR）相互依赖。
- 传统 bi-level optimization 方法在非凸、非线性场景下计算成本高昂，甚至无法收敛。
- 缺乏对运行不确定性（如传感器误差、工况波动）的鲁棒性建模能力。

---

### **提出了什么新方法或新思路**
本文提出了一种全新的 **ANN-KKT 框架**，将机器学习与数学规划深度融合，用于解决工业热力系统的 data-driven bi-level optimization 问题。

#### **核心创新点：**
1. **ANN 替代双层目标函数**  
   首次在 bi-level framework 中，**同时用 ANN 模型替代上下两层的目标函数**（FANN 和 fANN），实现从原始物理模型到数据驱动代理模型的完全转换。

2. **KKT 条件嵌入 + Fischer-Burmeister 改进**  
   将下层优化问题通过 **Karush-Kuhn-Tucker (KKT)** 条件转化为单层约束，并进一步引入 **Fischer-Burmeister (FB) 函数** 对互补松弛条件进行平滑重构，提升数值稳定性，避免求解器因 complementarity constraints 导致的不收敛。

3. **Mahalanobis Distance 约束保证域一致性**  
   引入基于历史数据协方差结构的 **Mahalanobis distance 约束**，确保优化解位于实际可行的操作包络内，防止生成物理上不合理或危险的操作点。

4. **扩展至鲁棒优化框架**  
   将 ANN-KKT 框架推广至 **robust optimization** 场景，构建“对抗性”下层问题，在不确定扰动集中寻找最坏情况下的性能下限，从而识别出能维持高效运行的**稳健操作空间**（robust operating envelope）。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | 本文 ANN-KKT |
|------|--------|--------------|
| **建模方式** | 物理机理模型或简化经验公式 | 完全数据驱动，ANN 捕捉高维非线性关系 |
| **求解效率** | 计算昂贵，常为 NP-hard | 单层化 + ANN 梯度可导 → 快速收敛（0.22–0.88 秒） |
| **适用性** | 多限于凸问题或小规模系统 | 可处理非凸、非线性、大规模工业系统 |
| **鲁棒性支持** | 较少集成不确定性建模 | 显式建模扰动集，输出稳健操作区间 |
| **工程落地性** | 难以实时响应控制需求 | 足够快的计算速度匹配动态过程控制 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **基准测试问题（Benchmark Problems）**
   - 三个标准 bi-level 测试案例：
     - Convex & Convex (C&C)
     - Convex & Non-convex (C&NC)
     - Non-convex & Non-convex (NC&NC)
   - 数据来源：合成采样（uniform sampling），共 10,000 样本，按 70%/15%/15% 划分为训练/验证/测试集。

2. **真实工业系统数据**
   - **660 MW 燃煤电厂**：1,279 条历史运行记录，输入变量包括 CFR、AFR、MSP、MST 等共 8 个操作参数。
   - **395 MW 燃气轮机系统**：579 条历史运行记录，输入变量包括 CDP、GFFR、AT、AP、AH 等共 9 个参数。
   - 所有输入均归一化至 [0,1] 区间。

---

### **实验设置和评估指标**

#### **模型架构**
- 使用三层前馈神经网络（shallow feed-forward ANN）
- 激活函数：隐藏层使用 SiLU，输出层为线性激活
- 超参数优化采用 **Bayesian Optimization (Hyperopt)**，搜索空间包括：
  - 隐藏层神经元数：[2, 16]
  - 学习率、L1/L2 正则化系数（log-uniform 分布）

#### **优化求解器**
- 主要使用 **IPOPT**（开源非线性优化求解器）
- 对比使用 **BARON**（全局优化求解器）验证局部最优解质量

#### **评估指标**
| 类别 | 指标 |
|------|------|
| **模型拟合性能** | RMSE, R²（训练/验证/测试集） |
| **优化性能** | 最优目标值（Power, THR）、CPU 时间（秒） |
| **可行性判断** | primal infeasibility < 1e-6 视为可行 |
| **鲁棒性分析** | 稳定半径（stability radius ρ）、最坏情况下 TE 是否高于目标阈值 |

---

### **基线方法对比**
- **Bi-level-KKT**：传统 KKT 单层化方法（无 ANN）
- **原始 bi-level 解**：来自文献的标准参考解
- 本文方法：**ANN-KKT (with/without FB reformulation)**

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 基准问题结果（Table 1 & Figure 1）**
| 问题类型 | 方法 | x | y | Objective (F) | Time (s) |
|---------|-------|----|----|---------------|----------|
| C&C | Bi-level-KKT | 1.0 | 3.0 | 5.0 | 0.14 |
| C&C | **ANN-KKT** | 0.981 | 2.962 | 4.976 | **1.27** |
| C&NC | Bi-level-KKT | 1.0 | 0.0 | 0.0 | 0.12 |
| C&NC | **ANN-KKT** | 1.014 | 0.0 | 0.0034 | **1.27** |
| NC&NC | Bi-level-KKT | -0.4191 | -1.0 | 0.1756 | 0.09 |
| NC&NC | **ANN-KKT** | -0.4191 | -1.0 | 0.1750 | **0.39** |

✅ **结论**：ANN-KKT 在所有基准问题中均能逼近真实 bi-level 解，相对误差极小，证明其有效性。

> ⚠️ 注：ANN-KKT 计算时间略长是由于需计算 ANN 梯度，但仍在可接受范围内。

---

#### **(2) 工业系统优化结果**

##### **660 MW 燃煤电厂**
- **最佳解**（T=94% Mahalanobis 容忍度）：
  - **最大功率输出：583 MW**
  - **最低汽轮机热耗率（THR）：7337 kJ/kWh**
  - **计算时间：0.052 秒**
- R² 表现：
  - Power 模型：R² ≈ 0.99
  - THR 模型：R² ≈ 0.72（测试集）

✅ 输出解集中在高频运行区域（kernel density 验证），具备工程可实施性。

---

##### **395 MW 燃气轮机系统**
- **最佳解**（T=89%）：
  - **最大功率输出：402 MW**
  - **THR：8159 kJ/kWh**
  - **计算时间：0.168 秒**
- R² 表现：
  - Power 模型：R² ≈ 0.99
  - THR 模型：R² ≈ 0.85

✅ 所有可行解均在 1 秒内完成，满足实时控制需求。

---

#### **(3) 鲁棒优化结果（Robust Optimization）**
针对燃气轮机系统，研究不同 **Target Efficiency Floor (TEtarget)** 下的最大稳定半径 ρ：

| TEtarget (%) | 稳定半径 ρ | 关键发现 |
|-------------|------------|----------|
| 38% | 较大（ρ=2.54） | 操作空间宽，允许较大扰动 |
| 42% | 很小（ρ=1.38） | 操作空间收缩，鲁棒性下降 |
| 43% | 无法满足 | 接近理论极限，无容错余地 |

✅ **发现**：随着效率目标提高，系统容忍扰动的能力显著降低，必须在“高效”与“鲁棒”之间权衡。

> 图表显示，名义效率（nominal TE）与最坏情况平均效率（mean TE at x+δ）之间的差距随目标升高而缩小。

---

### **消融实验结果（隐含分析）**
虽然未明确列出“ablation study”，但以下对比体现了设计选择的影响：

| 设计要素 | 影响表现 |
|--------|----------|
| **是否使用 FB 函数** | 未使用时出现 constraint violation；使用后显著改善收敛性和数值稳定性 |
| **Mahalanobis 约束 T 值调整** | 过低（如 T<80%）限制搜索空间；过高（如 T>90%）导致不可行解增多 |
| **IPOPT vs BARON** | IPOPT 更适合快速局部优化；BARON 验证了解的质量 |

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **ANN-KKT 是一种可扩展、高效的 bi-level 优化框架**，适用于复杂非凸工业系统。
2. ✅ 该方法能在 **0.22–0.88 秒内** 求得高质量解，满足工业级实时控制要求。
3. ✅ 结合 Mahalanobis distance 约束，可生成**领域一致**（domain-consistent）的操作策略，增强安全性。
4. ✅ 扩展至 robust optimization 后，能够量化“效率-鲁棒性” trade-off，指导调度决策。
5. ✅ 在燃煤和燃气机组上的成功应用表明，该方法具有良好的通用性和工程价值。

---

### **局限性**
1. **T 参数调优困难**  
   Mahalanobis distance 的容忍度 T 需手动设定，缺乏自动化调节机制，影响解的可行性分布。

2. **ANN 引入非凸性**  
   即使原问题是凸的，ANN 拟合可能引入局部极小和非凸区域，增加求解难度。

3. **依赖高质量历史数据**  
   若训练数据稀疏或噪声严重，ANN 模型泛化能力下降，进而影响优化结果可靠性。

4. **仅验证局部最优**  
   使用 IPOPT 求解不能保证全局最优，尤其在高度非凸空间中可能存在更优解未被发现。

---

### **未来工作方向**
1. **ANN 模型凸化技术**  
   探索 convex neural networks 或 piecewise-linear approximations，以提升优化问题的可解性。

2. **自适应 T 调节机制**  
   开发基于强化学习或贝叶斯优化的自动超参调节策略，动态平衡探索与可行性。

3. **集成不确定性传播分析**  
   将 dropout 或 Bayesian NN 引入框架，直接建模预测不确定性并反馈至优化过程。

4. **部署至数字孪生平台**  
   将 ANN-KKT 集成到电厂 DCS 或 MPC 控制层，实现实时闭环优化控制。

5. **拓展至多能耦合系统**  
   应用于综合能源系统（IES）、氢能混合电站等更复杂的工业场景。

---

> 📌 **最终定位**：本文提出的 ANN-KKT 框架不仅是算法创新，更是推动 **Industry 5.0** 和 **AI for Energy** 发展的重要一步——实现人机协同、智能、高效、低碳的下一代工业控制系统。

</details>

---

### 3. [Query as Anchor: Scenario-Adaptive User Representation via Large Language Model](https://arxiv.org/abs/2602.14492)

**Authors**: Jiahao Yuan, Yike Xu, Jinyong Wen, Baokun Wang, Ziyi Gao, Xiaotong Lin, Yun Liu, Xing Fu, Yu Cheng, Yongchao Liu, Weiqiang Wang, Zhongle Xie  
**Category**: cs.CL  
**Published**: 2026-02-17  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2602.14492v1  

#### Abstract
Industrial-scale user representation learning requires balancing robust universality with acute task-sensitivity. However, existing paradigms primarily yield static, task-agnostic embeddings that struggle to reconcile the divergent requirements of downstream scenarios within unified vector spaces. F...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Query as Anchor: Scenario-Adaptive User Representation via Large Language Model

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
工业级用户表示学习面临三大挑战：
1. **场景适应性差**：传统静态嵌入无法灵活适配不同下游任务（如推荐、风控、营销），导致需维护多个专用模型，系统复杂度高。
2. **模态与语义鸿沟**：真实用户行为数据稀疏、符号化、多源异构，与 LLM 预训练所用的密集文本数据存在显著差异，限制了 LLM 在用户建模中的泛化能力。
3. **噪声与冗余信号处理困难**：多源行为日志中存在大量无关或冲突信号，难以有效选择性关注场景相关特征。

### 🚀 提出的新方法与核心思想
提出 **Query-as-Anchor (Q-Anchor)** 框架，将用户建模从“静态编码”转变为“动态查询感知合成”，实现**一个统一模型支持多场景自适应表示**。

#### 主要创新点：
- **UserU 数据集构建**  
  构建首个面向用户表示学习的大规模预训练数据集 **UserU**，融合两类监督信号：
  - `D_future`：基于未来行为预测的结构化监督；
  - `D_uqa`：由 LLM 合成的 Query-Answer 对，增强高层语义理解。

- **Query-as-Anchor 动态表示机制**  
  将自然语言查询作为“锚点”（anchor），通过双塔架构引导 LLM 动态聚合用户行为序列，生成**任务感知的动态嵌入**，而非固定向量。

- **分层粗到细编码器（Hierarchical Coarse-to-Fine Encoder）**  
  设计三级编码结构（事件级 → 模态级 → 用户级），保留细粒度动作的同时捕捉宏观行为模式，提升对稀疏信号的鲁棒性。

- **联合对比-自回归优化目标**  
  结合：
  - **对比损失（Contrastive Loss）**：拉近 query-embedding 与 answer embedding；
  - **下一词预测损失（NTP）**：增强语义密度与局部建模能力。

- **Cluster-based Soft Prompt Tuning**  
  引入可学习软提示（soft prompt）进行轻量级后训练，无需微调整个模型即可对齐特定业务逻辑（如高风险 vs 低风险用户），避免灾难性遗忘。

- **KV-Cache 加速推理**  
  将用户历史编码为共享前缀并缓存 KV Cache，仅对查询部分增量计算，实现**多场景低延迟部署**。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | Q-Anchor |
|------|--------|---------|
| 表示类型 | 静态、任务无关 | 动态、查询条件化 |
| 场景适应 | 多模型或多头设计 | 单模型 + 软提示切换 |
| 推理效率 | 每场景独立编码 | 共享前缀 + KV-Cache 复用 |
| 泛化能力 | 依赖大规模参数 | 依赖高质量预训练与结构设计 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **UserU**：本文构建的工业级预训练数据集，包含约 1.024×10⁹ 条样本。
  - 来源于 Alipay 生态系统的多模态行为日志：
    - PayBill 交易
    - Mini Program 交互
    - SPM 导航路径
    - App 打开记录
    - 搜索 Query
    - 结构化 Tabular 特征（共 F=100+ 维）
- **下游测试基准 D_test**：10 个真实业务二分类任务，分为三类领域：
  | 领域 | 场景 |
  |------|------|
  | **User Engagement** | 社区活跃识别、演唱会点击预测、登录预测、蚂蚁森林互动 |
  | **Risk Control** | 欺诈检测、洗钱识别 |
  | **Marketing Sensitivity** | 外卖兴趣、品牌敏感度、大促响应、性价比偏好 |

### 🧪 实验设置
- **骨干模型**：Qwen2.5-0.5B-Instruct（Decoder-only LLM）
- **模态编码器**：gte-base 编码原始行为事件为稠密向量
- **训练配置**：
  - Batch size: 2048
  - 训练步数：50k 步
  - 优化器：AdamW，初始 LR=2e-4，cosine decay
  - 使用 LoRA（rank=64, α=32）进行高效微调
- **Embedding 维度**：固定为 128

### 📈 评估指标
- **AUC**（Area Under ROC Curve）：衡量分类判别性能
- **KS**（Kolmogorov-Smirnov Statistic）：衡量正负样本分布分离程度，尤其适用于风控等关键决策场景

### ⚔️ 基线方法对比
分为两类：
1. **通用文本嵌入模型**（General Embedding Models）：
   - Qwen2.5-0.5B-Instruct（无微调）
   - Qwen3-Embedding-0.6B
   - Llama-Embed-Nemotron-8B
   - KaLM-Embedding-Gemma3-12B

2. **用户表示模型**（User Embedding Models）：
   - MSDP、One4all、CPC（基于对比学习）
   - FOUND（当前 SOTA 用户基础模型）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（平均值）

| 方法 | Avg AUC | Avg KS |
|------|--------|-------|
| **Q-Anchor (Base)** | **0.8104** | **0.5044** |
| **Q-Anchor (Prompt Tuned)** | **0.8225** | **0.5267** |
| 最强 Baseline (Llama-Embed-Nemotron-8B) | 0.7488 | 0.3805 |
| FOUND (SOTA 用户模型) | 0.7832 | 0.4529 |

> ✅ **相对提升**：
> - AUC 提升 **+9.84%**（vs Llama-Embed）
> - KS 提升 **+38.4%**（vs Llama-Embed）

### 🔍 分项表现亮点
- 在 **Risk 领域**：
  - Money Laundering Detection AUC 达 **0.9439**，优于 FOUND 的 0.9235
- 在 **Marketing 领域**：
  - Brand Sensitivity AUC 从 Base 的 0.7979 提升至 Prompt Tuned 的 **0.8535**，说明软提示能精准捕捉细微偏好信号
- 在 **Engagement 领域**：
  - Forest Engagement AUC 高达 **0.9716**

### 🔧 消融实验结果（Ablation Study）

| 消融项 | Avg AUC ↓ | Avg KS ↓ | 说明 |
|--------|----------|---------|------|
| 移除 User/Modal Token | 0.8065 | 0.4966 | 显式结构标记有助于跨模态归因 |
| 移除 Contrastive Loss | 0.7667 | 0.4215 | 对比学习是主导驱动力 |
| 移除 NTP Loss | 0.8061 | 0.4961 | 自回归任务起正则化作用 |
| 移除 Margin Mask | 0.8047 | 0.4877 | 减少噪声负样本干扰 |
| **不进行预训练直接 Prompt Tuning** | 0.7782 | 0.4679 | **预训练提供必要行为先验** |

> 💡 发现：**预训练 ≠ 优化捷径，而是行为建模的基础前提**

### 📈 可扩展性分析
- **数据规模扩展**（20M → 102M 样本）：
  - AUC 从 0.8029 → 0.8105，持续增益
- **模型大小扩展**（0.5B → 3B）：
  - 性能非单调增长，**0.5B 表现最佳**
  - 更大模型梯度衰减严重（见 Fig. 11），优化更难
- **Prompt Tuning 扩展**：
  - 6 个可学习 token 即饱和，少量训练步（500 步）即可达到最优

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **“One Model for Many Scenarios” 成为可能**  
   Q-Anchor 实现了单一模型在用户参与、风控、营销三大异构领域均取得 SOTA，验证了其强大的跨域泛化能力。

2. **Query 是有效的语义控制器**  
   自然语言查询不仅能激活相关行为信号，还能抑制噪声，实现“语义聚焦”。

3. **轻量级 Prompt Tuning > 全量微调**  
   仅更新少量 soft prompt 即可实现领域专业化，兼顾精度与部署成本。

4. **KV-Cache 实现高效多场景服务**  
   用户前缀只需编码一次，新增场景仅增加极小推理开销，适合工业级部署。

5. **数据质量 > 模型规模**  
   在固定预算下，增加预训练数据比扩大模型更能提升嵌入质量，且小模型（0.5B）反而更具优势。

### ⚠️ 局限性
- 依赖高质量合成数据（D_uqa）的质量控制流程（如反思修正）
- 当前框架仍需人工设计 query 模板，尚未完全自动化
- 对极端长尾用户（行为极少者）建模能力有待加强

### 🔮 未来工作方向
- 探索 **gradient recovery 技术** 以突破大模型在 embedding 任务上的优化瓶颈
- 研究 **自动生成 query 指令** 的机制，减少人工干预
- 扩展至 **多兴趣解耦表示**，支持更复杂的个性化建模需求
- 推动 **开放数据集发布**（UserU 计划公开）

---

> 🏁 **总结一句话**：  
> Q-Anchor 通过“查询即锚点”的范式革新，实现了**动态、高效、可解释、可扩展**的工业级用户表示学习，在 Alipay 十大场景中全面超越现有方法，并已通过大规模线上 A/B 测试验证其商业价值。

</details>

---

### 4. [Federated Learning of Nonlinear Temporal Dynamics with Graph Attention-based Cross-Client Interpretability](https://arxiv.org/abs/2602.13485)

**Authors**: Ayse Tursucular, Ayush Mohanty, Nazal Mohamed, Nagi Gebraeel  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2602.13485v1  

#### Abstract
Networks of modern industrial systems are increasingly monitored by distributed sensors, where each system comprises multiple subsystems generating high dimensional time series data. These subsystems are often interdependent, making it important to understand how temporal patterns at one subsystem r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Federated Learning of Nonlinear Temporal Dynamics with Graph Attention-based Cross-Client Interpretability

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**去中心化非线性动态系统中的跨客户端时序依赖建模与可解释性**问题，具体挑战包括：
- **数据隐私与主权限制**：各客户端（如工业子系统）无法共享原始高维时间序列数据。
- **固定专有模型约束**：客户端运行由设备厂商提供的固定状态估计器（如 EKF），不能修改或重训练。
- **非线性动态复杂性**：系统动力学高度非线性，导致跨子系统的时序影响难以显式捕捉和解释。
- **缺乏全局可观测性**：传统集中式学习不可行，且现有联邦学习方法多关注静态任务或忽略动态演化。

### 提出的新方法与思路
提出了一种**基于图注意力网络（GAT）的联邦学习框架**，用于在保护隐私的前提下学习非线性系统的跨客户端时序依赖，并实现**可解释的动态交互分析**。其核心设计如下：

- **客户端侧**：保留原有 proprietary EKF 模型输出的预测状态 $(h_t^c)$ 和估计状态 $(h_t^c)_c$，引入一个可学习的非线性增强模块 $\Delta_m(\cdot)$ 构造**增强状态** $(h_t^a)$，使其能隐式编码跨客户端信息。
  
- **服务器侧**：构建一个基于 **Graph Attention Network (GAT)** 的全局状态转移模型，接收来自各客户端的增强状态 $(h_{t-1}^a)$，通过注意力机制聚合邻居信息，生成预测状态 $(h_t^s)$。

- **通信协议**：
  - 客户端 → 服务器：发送 $(h_t^c), (h_t^a)$。
  - 服务器 → 客户端：反向传播损失梯度 $ \nabla_{(h_t^a)} L_s $ 到客户端，指导 $\Delta_m$ 的更新。

- **可解释性机制**：首次将 GAT 中的 **attention coefficients** $\alpha_{mn}(t)$ 与状态转移函数的 **Jacobian 块** $J_{mn}(t) = \partial (h_t^s)_m / \partial (h_{t-1}^a)_n$ 联系起来，建立解析关系（见 Proposition 6.1），从而提供对“**当前状态下谁影响谁、如何影响**”的结构性与动态性双重解释。

### 相比现有方法的优势
| 维度 | 本文方法 | 现有方法局限 |
|------|----------|--------------|
| **隐私保护** | 仅交换低维 latent states 和 gradients，不暴露 raw data 或 server states | 多数 GNN 需要共享节点特征或图结构 |
| **兼容性** | 支持 fixed proprietary models，无需修改本地系统 | 多数 FL 方法要求客户端参与模型训练 |
| **动态建模能力** | 显式建模非线性 temporal dynamics via GAT | 多数 FL-GNN 工作处理静态图或忽略时间演化 |
| **可解释性** | 提供 Jacobian-based 动态影响 + attention-based 结构权重联合解释 | 多为黑箱模型，缺乏物理意义解读 |

---

## 2. 核心实验方法和设置

### 数据集
#### （1）合成数据集（Synthetic Dataset）
- **生成方式**：从已知 ground-truth 的非线性 SSM 生成，其中 state-transition 函数为一层 GAT，attention 系数预设。
- **参数配置**：
  - 客户端数量 $M=3$
  - 潜在状态维度 $p_m=1$，观测维度 $d_m=8$
  - 时间步长 $T=1000$（前800训练，后200验证）
- **目的**：验证方法能否恢复真实 interdependency 结构，检验收敛性和可解释性。

#### （2）真实世界工业数据集（HAI Dataset）
- **来源**：Hardware-in-the-Loop Augmented Industrial Control System (HAI) benchmark [21]
- **系统组成**：
  - P1: Water treatment process
  - P2: Chemical dosing process
  - P3: Heating process
  - P4: 控制层（排除在外）
- **预处理**：
  - 对每个客户端独立进行 SVD，提取 top-$p_m=3$ 右奇异向量作为 latent state。
  - 使用 Nominal 数据训练 proprietary LSTM 模型，在 Attack (AP04) 数据上冻结并用于联邦训练。
- **历史窗口长度**：$S=8$

### 实验设置与评估指标
| 类别 | 内容 |
|------|------|
| **训练策略** | Adam optimizer ($lr=10^{-3}$)，batch size=512，early stopping（patience=15） |
| **损失函数** | <br>• Client: Reconstruction loss $ \| y_t - g_m((h_t^a)) \|^2 $<br>• Server: MSE $ \sum_t \sum_m \| (h_t^s)_m - (h_t^a)_m \|^2 $ |
| **评估指标** | <br>• Server/client prediction loss<br>• Attention 与 ground truth 的相关性<br>• Jacobian 与 centralized oracle 的 cosine similarity / Pearson correlation<br>• 验证集 residual norm |
| **对比基线** | <br>1. **Proprietary Model**：仅本地 EKF/LSTM，无联邦学习<br>2. **Centralized Model**：全局 SVD + 共享 latent space + LSTM+GAT（上界）<br>3. **Pre-trained Federated Baseline** [15]：共识图对齐表示，但不建模动态<br>4. **NOTEARS-ADMM** [17]：联邦因果发现，假设 DAG，水平划分 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 表 2：客户机与服务器损失比较（越小越好）

| Baseline         | P1     | P2     | P3     | Avg    | Server |
|------------------|--------|--------|--------|--------|--------|
| Proprietary      | 0.8255 | 0.4817 | 1.0391 | 0.7821 | —      |
| Centralized      | 0.2190 | 0.2247 | 0.2440 | 0.2292 | —      |
| Pre-trained      | —      | —      | —      | —      | 0.0813 |
| NOTEARS-ADMM     | 1.1815 | 0.8640 | 1.1262 | 1.0572 | —      |
| **Our model**    | **0.6891** | **0.3721** | **0.3308** | **0.4640** | **0.0024** |

> ✅ 我们的模型显著优于 Proprietary 和 NOTEARS-ADMM，在 client loss 上接近 Centralized 模型（差距约 2.3×），远超 Pre-trained 方法。

#### 表 3：增强模型相对于专有模型的测试 MSE 下降率

| Client | Reduction in client loss (%) |
|--------|-------------------------------|
| P1     | +16.53                        |
| P2     | +22.74                        |
| P3     | +68.16                        |
| Average| **+40.67**                    |

> ✅ 所有客户端均获得显著提升，尤其 P3 达到近 70% 的误差下降，表明联邦学习有效增强了局部预测能力。

#### 表 4：Jacobian 相似性（vs Centralized Model）

| Metric               | Value     |
|----------------------|-----------|
| Cosine Similarity    | 0.7277    |
| Pearson Correlation  | 0.5820    |

> ✅ 高相似度说明所学动态与集中式模型高度一致，支持 Claim 5.5。

#### 图 7：Attention 系数相关性热力图
- 本方法学习的 attention 结构与 Centralized 模型高度相似，优于 Pre-trained baseline。
- 特别是在 P2→P3 和 P3→P1 方向表现出强关联，符合实际工业流程逻辑。

#### 图 5：Attention 与 Jacobian 幅度的相关性
- 报告了 strong empirical correlation between $\alpha_{mn}(t)$ 和 $|J_{mn}(t)|$。
- 验证了 Proposition 6.1 的理论推导：attention 控制信息流入强度，进而主导动态影响。

#### 图 10：隐私-效用权衡（加噪实验）
- 在 client-to-server 和 server-to-client 通信中加入 Gaussian noise。
- 发现小噪声下 $L_s$ 稳定，大噪声时急剧上升 → 存在实用的 privacy-utility trade-off。

#### 表 1：Scalability 分析
- Server loss $L_s$ 对 observation dimension 不敏感。
- 随 latent dimension $p_m$ 和 client 数量 $M$ 增加而上升，反映通信开销与模型复杂度增长。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **成功实现了在固定专有模型下的联邦动态建模**：即使客户端不能改变内部模型，也能通过 augmentation + gradient feedback 机制融入全局动态知识。
2. ✅ **GAT 是建模跨客户端非线性动态的理想选择**：其 attention 机制天然适合表达 directed, state-dependent interactions。
3. ✅ **Jacobian 与 attention 的耦合提供了新型可解释性**：首次建立了 attention weights 与实际动态影响之间的数学联系，使“谁影响谁”不仅可视，而且可量化。
4. ✅ **理论收敛保证成立**：证明了在合理假设下，联邦学习的 server-side dynamics 和 Jacobian 收敛至 centralized oracle。
5. ✅ **实验证明有效性与鲁棒性**：在 synthetic 和 real-world 数据上均表现优异，具备良好的 scalability 与 noise robustness。

### 方法的局限性
1. ❌ **依赖已知图骨架（graph skeleton）**：假设 adjacency matrix 已知（工程系统常见），但在未知连接场景需结合 causal discovery 方法。
2. ❌ **Jacobian 可解释性基于局部一阶近似**：在强非线性区域可能失效，无法捕捉高阶动态效应。
3. ❌ **未考虑异步更新或多速率采样**：假设所有客户端同步上报状态，现实系统可能存在延迟或不同频率。
4. ❌ **server-side GAT 架构固定**：未探索更复杂的 temporal GNN 结构（如 Temporal GAT, EvolveGCN）。

### 未来工作方向
1. 🔮 将框架扩展至 **unknown graph structure learning**，结合 NOTEARS 或 DYNOTEARS 进行联合结构发现与联邦训练。
2. 🔮 探索 **higher-order sensitivity analysis**（如 Hessian）以增强对强非线性的解释能力。
3. 🔮 引入 **asynchronous communication protocol** 以适应实际部署中的时延与丢包。
4. 🔮 应用于更大规模工业系统（如电网、智能制造流水线），验证在复杂拓扑下的泛化能力。
5. 🔮 探索 **privacy amplification techniques**（如 differential privacy, secure aggregation）进一步加强通信安全。

--- 

> **总结一句话**：本文提出了一种兼顾 **隐私保护、模型兼容性、动态建模能力与可解释性** 的联邦学习新范式，为现代分布式工业系统的智能监控与诊断提供了坚实的理论与实践基础。

</details>

---

### 5. [AllMem: A Memory-centric Recipe for Efficient Long-context Modeling](https://arxiv.org/abs/2602.13680)

**Authors**: Ziming Wang, Xiang Wang, Kailong Peng, Lang Qin, Juan Gabriel Kostelec, Christos Sourmpis, Axel Laborieux, Qinghai Guo  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.13680v1  

#### Abstract
Large Language Models (LLMs) encounter significant performance bottlenecks in long-sequence tasks due to the computational complexity and memory overhead inherent in the self-attention mechanism. To address these challenges, we introduce \textsc{AllMem}, a novel and efficient hybrid architecture tha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AllMem: A Memory-centric Recipe for Efficient Long-context Modeling 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在处理**长序列任务**时面临两大瓶颈：
- **计算复杂度高**：标准 Self-Attention 机制的时间和空间复杂度均为 $O(L)$，其中 $L$ 是输入长度，导致推理延迟显著增加。
- **内存开销大**：KV Cache 随序列增长线性膨胀，在边缘设备或超长上下文场景下难以部署。

此外，现有高效注意力机制（如滑动窗口、线性注意力）常因信息丢失而出现**性能下降**或**灾难性遗忘**。

---

### 提出的新方法与核心思想
本文提出 **ALLMEM**（A Large Language MEMory model），一种新型混合架构，融合了以下两种机制：

#### （1）Sliding Window Attention (SWA)
- 在局部窗口内执行标准 Softmax 注意力，保留对细粒度依赖的建模能力。
- 窗口大小固定为 $W$，显著降低计算负担。

#### （2）非线性 Test-Time Training (TTT) Memory Network
- 引入一个可学习的、参数化的长期记忆模块，在推理时通过在线优化动态更新。
- 将序列建模视为“在线压缩”问题，利用梯度下降最小化重建损失，实现知识的抽象与持久存储。
- 使用残差连接的 SwishGLU 构造**非线性记忆单元**，相比线性模型（如 Mamba）具有更强的信息压缩与表示能力。

#### 整体架构设计原则
- **并行双通路结构**：Token Mixer 分为 SWA（短期精确记忆）与 TTT-Memory（长期抽象记忆）两个分支。
- **可学习融合门控**：引入 channel-wise scaling gate $\alpha$ 动态加权两者的输出，实现上下文感知的融合：
  $$
  \text{TokenMixer}(x) = \text{SWA}(\text{RMSNorm}(x)) + \alpha \cdot \text{ALLMEM}(\text{RMSNorm}(x))
  $$
- **模块化微调策略**：仅微调 ALLMEM 模块的元参数（meta-parameters），冻结原始 SWA 和 MLP 权重，避免灾难性遗忘。

---

### 相比现有方法的优势
| 特性 | 全注意力 | 滑动窗口 + Sink Tokens | 线性注意力（如 Mamba） | **ALLMEM（本文）** |
|------|----------|------------------------|-------------------------|--------------------|
| 计算复杂度 | $O(L^2)$ | $O(LW)$ | $O(L)$ | $O(LW)$（Prefill）<br>$O(W)$（Decode） |
| 存储复杂度 | $O(L)$ | $O(L)$ | $O(1)$ | $O(1)$ ✅ |
| 是否需从头训练 | 否 | 否 | 是（通常） | ❌ 可基于预训练模型微调 ✅ |
| 是否支持非线性记忆 | 否 | 否 | 否 | ✅ 支持非线性压缩与学习 |
| 能否缓解遗忘 | 有限 | 有限 | 一般 | ✅ 显著提升长程建模稳定性 |

> ✅ **关键优势**：**无需从头训练即可将任意预训练 LLM 转换为高效长上下文模型**，同时保持甚至超越原模型性能。

---

## 2. 核心实验方法和设置

### 使用的数据集

#### 短序列基准测试（Short-Sequence Benchmarks）
用于验证基础语义理解、推理、数学与代码能力：
- **常识推理**：C-Eval（5-shot）、ARC-Easy/Challenge、HellaSwag、WinoGrande
- **综合推理**：MMLU-Redux、GPQA-Diamond
- **指令遵循**：IFEval
- **数学能力**：MATH-500
- **编程能力**：LiveCodeBench v5

#### 长序列基准测试（Long-Context Benchmarks）
评估超长上下文下的建模能力：
- **LongBench**：平均长度 ~37k tokens，选取6个子任务（如 narrativeqa, triviaqa）
- **InfiniteBench**：最长达 128k–200k+ tokens，涵盖中英文问答、事实回忆等
- **LV-Eval**：提供多层级长度任务，选择 128k context 子集进行测试

---

### 实验设置与评估指标

#### 模型基础
- 基于 **Qwen3-0.6B** 和 **Qwen3-1.7B** 进行转换与微调。
- 微调采用知识蒸馏框架，教师模型为原始 Qwen3，学生模型为 ALLMEM。
- 总损失函数为 KL 散度主导（$\mathcal{L}_{\text{total}} = \mathcal{L}_{KL}$），未使用额外 CE 损失。

#### 训练策略
- **随机配置训练**：每轮独立采样滑动窗口大小（512–8192）、sink 数量（0–256）、chunk size（512–4096），增强泛化性。
- **On-Policy Distillation**：先由学生生成响应，再由教师对其分布进行蒸馏，提升对齐精度（采用离线版本以提高效率）。
- **推理框架适配**：集成至 vLLM v1 框架，加速推理。

#### 评估指标
- 主要指标：**准确率（Accuracy）**
- 辅助分析：FLOPs（计算量）、Cache Size（KV 缓存占用，FP32）

---

### 基线方法对比
- **Full Attention**：完整注意力机制（作为性能上限）
- **SWA + Sink Tokens**：带 sink 的滑动窗口注意力
- **Mamba-enhanced Model**：用 Mamba 替换 ALLMEM 模块的变体（线性记忆对照）

---

## 3. 主要实验结果和性能指标

### 短序列任务表现（见 Table 1）

| 类别 | 指标 | Qwen3-0.6B | Qwen3-0.6B-ALLMEM | Qwen3-1.7B | Qwen3-1.7B-ALLMEM |
|------|------|-------------|---------------------|------------|---------------------|
| C-Eval | - | 40.6 | **42.2** | 58.0 | **58.9** |
| ARC-Easy | - | 70.0 | 70.2 | 84.5 | **84.7** |
| MMLU-Redux | - | 44.9 | **47.05** | 66.5 | **67.3** |
| MATH-500 | - | 48.8 | **49.8** | 73.6 | **74.4** |
| LiveCodeBench | - | 13.8 | **14.3** | 25.5 | 25.0 |

✅ **结论**：ALLMEM 在所有短序列任务上**持平或略优于原始模型**，证明其能有效继承预训练知识，无性能退化。

---

### 长序列任务表现（见 Table 2 & 3）

#### LongBench（平均长度 37k，窗口 4k）
| 模型 | Average Accuracy |
|------|------------------|
| Qwen3-0.6B (Full Attn) | 28.16 |
| SWA + Sink | 25.71 |
| Mamba-enhanced | 27.01 |
| **ALLMEM (Ours)** | **27.33** ✅ |

> ➤ 尽管仅使用 **4k 局部窗口**，ALLMEM 接近全注意力性能，且显著优于其他高效结构。

#### InfiniteBench & LV-Eval（128k context，窗口 8k）
| 模型 | Average Accuracy |
|------|------------------|
| Qwen3-1.7B (Full Attn) | ~5.29 |
| SWA + Sink | 3.76 |
| Mamba-enhanced | 4.36 |
| **ALLMEM (Ours)** | **5.56** ✅ |

> ✅ **突破性结果**：在 **128k 上下文** 下，**8k 窗口的 ALLMEM 模型反超全注意力模型**，表明其参数化记忆机制能有效补偿局部注意力的信息缺失。

---

### 计算成本与内存开销（见 Figure 3）

| 序列长度 | 方法 | FLOPs（相对） | Cache Size（相对） |
|--------|------|---------------|-------------------|
| 128k | Full Attention | 1× | 1× |
| 128k | ALLMEM | ↓ ~1/9 | ↓↓ **恒定 $O(1)$** ✅ |
| 128k | SWA + Sink | ↓ ~1/3 | 仍随 $L$ 增长 |

> ✅ **优势明显**：ALLMEM 实现了 **$O(W)$ 解码复杂度** 和 **常数级缓存占用**，非常适合边缘设备和实时系统。

---

### 消融实验（隐含分析）
虽然未单独列出消融表，但从设计中可推断关键组件作用：
- **非线性记忆单元（SwishGLU）**：相比线性模型（Mamba）表现更优 → 验证非线性压缩的重要性。
- **内部归一化机制（read-before-normalize）**：防止早期训练中记忆结构被破坏 → 提升训练稳定性。
- **动态学习率与动量系数**：输入依赖的元参数增强了适应性。
- **仅微调 ALLMEM 模块**：成功避免灾难性遗忘，保留原有能力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **高效的长上下文建模是可行的**：通过结合 **短视注意力 + 参数化长期记忆**，可在极低资源消耗下实现接近甚至超越全注意力的表现。
2. ✅ **非线性记忆优于线性记忆**：TTT 框架下的非线性更新规则（如 SwishGLU）比传统线性 SSM 更适合复杂语义压缩。
3. ✅ **无需从头训练也能获得强大性能**：通过**记忆增强微调**（Memory-Efficient Fine-Tuning），可直接将任何预训练 LLM 转换为高效长上下文模型。
4. ✅ **O(1) 存储复杂度真正解决扩展瓶颈**：KV Cache 不再随长度增长，彻底摆脱内存限制。

---

### 方法的局限性
1. **依赖高质量教师模型**：性能上限受限于蒸馏来源的 Qwen3 表现。
2. **初始化敏感**：TTT 内循环的稳定性依赖良好的归一化与学习率调度。
3. **硬件适配仍在演进**：尽管已接入 vLLM，但针对 ALLMEM 的专用推理引擎尚未普及。
4. **极端长度 (>256k) 验证不足**：当前实验集中在 128k，更长序列的效果有待进一步测试。

---

### 未来工作方向
1. **与外部记忆系统结合**：探索与 RAG、Engram 等持久化记忆系统的集成，构建**多级记忆体系**（短期感知 + 长期参数化 + 外部检索）。
2. **跨模态扩展**：将 ALLMEM 应用于视觉、语音等序列建模任务。
3. **自动化元参数搜索**：设计 NAS 或强化学习策略自动优化学习率、动量等超网络结构。
4. **端到端训练方案探索**：研究是否能在更大规模上直接训练原生 ALLMEM 架构，而非依赖蒸馏。

---

> 🧠 **结语引用**：  
> “Memory is the residue of thought.” — Daniel T. Willingham  
> ALLMEM 正是对这一理念的技术实践：不是简单地记住 token，而是让模型在运行时“思考”，从而形成真正的**可学习、可演化、可压缩的记忆系统**。

</details>

---

### 6. [REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents](https://arxiv.org/abs/2602.14234)

**Authors**: Zheng Chu, Xiao Wang, Jack Hong, Huiming Fan, Yuqi Huang, Yue Yang, Guohai Xu, Chenxiao Zhao, Cheng Xiang, Shengchao Hu, Dongdong Kuang, Ming Liu, Bing Qin, Xing Yu  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.14234v1  

#### Abstract
Large language models are transitioning from generalpurpose knowledge engines to realworld problem solvers, yet optimizing them for deep search tasks remains challenging. The central bottleneck lies in the extreme sparsity of highquality search trajectories and reward signals, arising from the diffi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）正从静态知识引擎向动态智能体演进，但在优化其用于**长周期、交互式搜索任务（long-horizon search tasks）**时面临两大瓶颈：
- **高质量搜索轨迹稀疏**：复杂推理任务难以大规模合成，监督信号极其稀疏。
- **训练成本高昂**：依赖真实环境中的工具调用（如 web search、maps 等）进行 rollouts 成本极高，限制了强化学习（RL）的可扩展性。

### 提出的新方法与创新思路
为解决上述挑战，作者提出 **REDSearcher**，一个统一的、可扩展且低成本的框架，涵盖任务合成、中段训练（mid-training）和后训练（post-training）全流程。其核心创新包括：

#### （1）Dual-Constrained Task Synthesis（双约束任务合成）
- 将任务生成建模为**图结构上的约束满足问题**，通过控制两个维度提升任务难度：
  - **拓扑逻辑复杂度（Topological Logical Complexity）**：使用 **treewidth** 作为衡量标准，构造具有环状、交织依赖关系的推理图（如 k=2 的环形、k=3 的四面体结构），迫使模型维护多个假设而非线性推理。
  - **信息源分散度（Information Dispersion）**：引入 **Minimum Source Dispersion (MSD)**，确保逻辑相关的事实分布在不同网页中，防止“单页捷径”解法，强制跨文档综合。

#### （2）Proactive Tool-Augmented Queries（主动式工具增强查询）
- 不再依赖被动召回，而是将关键实体替换为**工具可解析的约束条件**，例如：
  - 将地名替换为“距离某地两小时车程的城市”（需调用 Google Maps）；
  - 将学者名字替换为“拥有约 N 次引用的研究者”（需调用 Google Scholar）。
- 此设计使工具调用成为完成任务的必要条件，显著**稠密化了学习信号**。

#### （3）Cost-Efficient Mid-Training（低成本中段训练）
- 分两阶段进行中段训练，降低对真实环境交互的依赖：
  - **Stage I**：在无环境交互下，通过合成数据强化原子能力（atomic capabilities）：
    - **Intent-anchored Grounding**：从噪声中提取与当前意图相关的关键信息。
    - **Hierarchical Planning**：将模糊目标分解为具体子目标，支持多步规划。
  - **Stage II**：引入模拟环境中的工具调用循环和长周期轨迹，逐步过渡到真实行为。

#### （4）Functionally Equivalent Simulation Environment（功能等价模拟环境）
- 构建轻量级本地模拟环境，具备以下特性：
  - 接口与真实 API 一致（interface consistency）；
  - 包含所有必要证据（evidence completeness）；
  - 注入大量干扰文档以模拟现实噪声（environmental noise）。
- 支持高速、低成本的算法迭代和 RL 实验，避免外部 API 的延迟与不稳定性。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **文本搜索基准**：
  - **BrowseComp**（英文）
  - **BrowseComp-zh**（中文）
  - **GAIA**（通用 AI 助手能力评测）
  - **Humanity’s Last Exam (HLE)**（综合性难题测试）
- **多模态搜索基准**：
  - **MM-BrowseComp**, **BrowseComp-VL**, **MMSearch-Plus**, **LiveVQA**, **MMLiveSearch**

### 实验设置与评估指标
- **模型架构**：基于开源模型 **Qwen3-30B-A3B** 和 **Qwen3-VL-30B-A3B-Thinking** 微调。
- **训练流程**：
  1. **Mid-training**：分两阶段进行，使用大规模合成数据。
  2. **Supervised Fine-Tuning (SFT)**：在高质量合成轨迹上微调。
  3. **Agentic Reinforcement Learning (RL)**：采用 **GRPO** 算法，奖励由 LLM Judge 提供（基于答案正确性）。
- **上下文管理**：采用 **Discard-all** 策略，在接近上下文上限时清空历史交互记录，保留原始问题。
- **评估指标**：各基准的官方评分（pass@k, avg@k 等），总体平均得分。

### 基线方法对比
- **闭源代理模型**：
  - `Seed1.8`, `Gemini-3-Pro`, `GPT-5-Thinking-high`, `Claude-4.5-sonnet`
- **开源代理模型**：
  - `Kimi-K2.5`, `GLM-4.7`, `DeepSeek-V3.2`, `Tongyi DeepResearch`, `WebSailorV2`

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 模型 | BrowseComp | BrowseComp-zh | GAIA | HLE | **Overall** |
|------|------------|----------------|-------|------|-----------|
| **REDSearcher (30B)** | 57.4* | 58.2* | **80.1** | 34.3 | **51.6** |
| Tongyi DeepResearch-30B | 43.4 | 46.7 | 70.9 | 32.9 | 48.5 |
| WebSailorV2-30B | 35.3 | 44.1 | 74.1 | 30.6 | 46.0 |

> *注：`*` 表示使用了 Discard-all 上下文管理技术*

### 与基线方法的对比结果
- 在 **30B 参数级别**的开源模型中，**REDSearcher 取得 SOTA 性能**，整体得分 **51.6**，显著优于同类模型（如 Tongyi DeepResearch 的 48.5）。
- 在 **GAIA 基准上达到 80.1**，甚至**超过 GPT-5-Thinking-high（76.7）**，表明其强大的复杂任务处理能力。
- 多模态版本 **REDSearcher-MM-RL** 在 **BrowseComp-VL 达到 57.2**，在 **MM-BrowseComp 达到 26.6**，表现强劲。

### 消融实验结果（见 Table 2）

| 阶段 | BrowseComp | BrowseComp-zh | HLE | GAIA | **Average** |
|------|------------|----------------|------|-------|-------------|
| Base | 34.74 | 26.82 | 32.25 | 77.43 | 42.81 |
| + Stage I (Grounding) | 36.61 | 27.34 | 32.00 | 76.70 | 43.16 |
| + Stage I (Planning) | 36.97 | 29.84 | 31.37 | **80.83** | 44.75 |
| + Stage II (Agentic) | **40.44** | **38.75** | 31.25 | 79.13 | **47.39** |

- **Stage I** 显著提升 GAIA 分数（+4.13），验证了分层规划的重要性。
- **Stage II** 对 BrowseComp-zh 提升最大（+8.91），说明环境反馈和长周期交互对实际搜索能力至关重要。

### 强化学习效果（见 Figure 6）
- 经过 RL 后，**平均奖励从 47.4 提升至 51.3（+3.9）**，BrowseComp 从 39.4 → 42.1。
- **Rollout 长度下降 10.4%（100.6 → 90.1）**，而性能持续上升，表明模型学会了更高效的搜索策略。

---

## 4. 关键结论和发现

### 主要发现
1. **结构复杂性与信息分散是深度搜索的核心挑战**：仅靠多跳推理不足以驱动智能体进化，必须结合图结构复杂性和证据分布来制造真正困难的任务。
2. **工具使用应被显式激励**：通过将事实转化为工具可解的约束，可以有效引导模型主动调用工具，而非依赖参数记忆。
3. **中段训练是连接预训练与代理行为的关键桥梁**：分离原子能力训练与环境交互，大幅降低了高质量轨迹收集的成本。
4. **本地模拟环境可行且高效**：只要保证接口一致性、证据完备性和足够噪声，即可支撑高质量的 RL 迭代。

### 方法的局限性
- 当前框架仍依赖于高质量的合成数据管道，若合成过程存在偏差或错误，可能影响最终性能。
- 模拟环境虽功能等价，但仍无法完全复现真实网络的动态性和不确定性。
- 多模态任务中图像内容的设计仍较人工，自动构建更具挑战性的视觉推理任务仍有空间。

### 未来工作方向
- 扩展到更多工具类型（如数据库查询、API 编排等）。
- 探索更自动化的任务难度调控机制（如基于模型表现动态调整 treewidth 或 MSD）。
- 开放发布的 **10K 文本轨迹、5K 多模态轨迹、1K RL 查询集**，推动社区对长周期搜索智能体的研究。

---

> 🔗 **项目主页**：[redsearchagent.github.io](https://redsearchagent.github.io)  
> 📦 **资源公开**：作者承诺将发布 **10K 高质量文本搜索轨迹、5K 多模态轨迹、1K RL 查询集、代码与模型检查点**，极大促进后续研究。

</details>

---

### 7. [HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2602.13710)

**Authors**: Xin Yan, Zhenglin Wan, Feiyang Ye, Xingrui Yu, Hangyu Du, Yang You, Ivor Tsang  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.13710v1  

#### Abstract
Vision-Language-Action (VLA) models enable instruction-following embodied control, but their large compute and memory footprints hinder deployment on resource-constrained robots and edge platforms. While reducing weights to 1-bit precision through binarization can greatly improve efficiency, existin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Vision-Language-Action (VLA) 模型在机器人控制中表现出色，但其**高计算和内存开销**限制了在资源受限设备（如边缘机器人）上的部署。虽然 Post-Training Quantization (PTQ) 是一种高效的压缩手段，但现有的 1-bit 量化方法（如直接从 LLM 或 VLM 迁移的方法）在 VLA 上表现不佳，原因如下：
- **量化误差累积严重**：VLA 输出连续动作，在闭环执行中微小偏差会被动力学放大，导致任务失败。
- **标准 Hessian 估计不准确**：激活图受背景异常值和视觉 token 不平衡影响，导致重要权重识别错误。
- **跨模态权重混合破坏 Haar 变换效果**：不同模态的权重交错排列，造成变换后出现高频噪声。

### 提出了什么新方法或新思路
作者提出 **HBVLA** —— 一个专为 VLA 定制的 1-bit PTQ 框架，包含三个核心设计：

1. **Policy-Aware Enhanced Hessian**
   - 引入 token-level 的重要性矩阵 $ S_t $，通过反向传播梯度衡量每个 token 对动作生成的影响。
   - 构建修正的 Hessian 矩阵 $ H = \sum_t S_t X_t X_t^T $，更精准地识别对策略敏感的关键权重。

2. **Sparse Orthogonal Transform for Non-Salient Weights**
   - 对非显著权重应用稀疏正交置换矩阵 $ P $，将相似列聚类在一起，降低 Haar 变换后的高频能量。
   - 使用贪心配对链算法（Greedy Pairing-and-Chaining）实现高效重排序。

3. **Group-wise 1-Bit Quantization in Haar Domain**
   - 在 Haar 域进行分组量化，对显著和非显著权重分别处理：
     - 非显著权重采用行共享均值（shared-mean）以减少元数据开销。
     - 显著权重基于残差进行列方向 Haar 量化，保留关键信息。

### 相比现有方法的优势
- **行为保真度更高**：相比通用二值化方法（如 BiLLM、BiVLM），HBVLA 更好地保持了原始策略的行为一致性。
- **无需训练数据再参与**：作为 PTQ 方法，仅需少量校准轨迹，适合快速部署。
- **硬件友好**：1-bit 权重极大提升推理效率，适用于低功耗机器人平台。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LIBERO**：包含 100 个任务、5000 条轨迹的仿真操作基准，涵盖空间、对象、目标和长视野任务（LIBERO-Spatial/Object/Goal/Long）。
- **SIMPLER**：高保真模拟环境，贴近真实世界场景，测试两种设定：
  - *Visual Matching*：最小化环境差异；
  - *Variant Aggregation*：引入光照、背景等扰动以测试鲁棒性。
- **Real-world Evaluation Suite**：基于 Mobile ALOHA 双臂机器人的真实实验，包括：
  - Pick and Place（不规则物体）
  - Sequenced Instruction（汉诺塔）
  - Flexible Folding（毛巾折叠）

### 实验设置和评估指标
- **模型**：在 OpenVLA、OpenVLA-OFT 和 CogACT 上验证。
- **量化配置**：权重量化至平均 **1.08-bit**（接近纯 1-bit）。
- **评估指标**：
  - **Success Rate (SR)**：主要评价指标。
  - 相对下降率 △：相对于 Full-Precision (FP) 模型的表现损失。
- **硬件平台**：NVIDIA A800 GPU；真实实验使用 Magic 双臂协作机器人。

### 基线方法对比
选择当前最先进的 1-bit PTQ 方法作为基线：
- **BiLLM**：面向大语言模型的二值化方法。
- **BiVLM**：扩展到视觉-语言模型的 PTQ 方法。
- **HBLLM**：基于 Haar 变换的 LLM 二值化框架。

所有方法均只量化 vision 和 language 主干网络，其余部分保持全精度，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 LIBERO 上的结果（Table 2）
| 方法 | Avg SR (%) | △ vs FP |
|------|------------|--------|
| OpenVLA (FP) | 76.5 | – |
| BiLLM | 43.9 | -32.6 |
| BiVLM | 47.9 | -28.6 |
| HBLLM | 58.6 | -17.9 |
| **HBVLA (ours)** | **70.0** | **-6.5** |

> → HBVLA 比最佳基线（HBLLM）高出 **11.4% 绝对成功率**，仅比 FP 模型低 6.5%。

#### 在 SIMPLER 上的结果（Table 1）
| 方法 | Visual Matching SR (%) | Variant Aggregation SR (%) |
|------|--------------------------|----------------------------|
| CogACT (FP) | 74.8 | 61.3 |
| BiLLM | 28.8 | 14.9 |
| BiVLM | 57.1 | 47.4 |
| HBLLM | 62.3 | 47.9 |
| **HBVLA (ours)** | **70.0** | **51.0** |

> → 在两个设定下均取得最优结果，尤其在 Visual Matching 下接近 FP 模型。

#### 在 OpenVLA-OFT 上的进一步提升
| 方法 | Avg SR (%) | △ vs FP |
|------|------------|--------|
| OpenVLA-OFT (FP) | 97.1 | – |
| HBLLM | 79.2 | -17.9 |
| **HBVLA (ours)** | **90.3** | **-6.8** |

> → 保留了 **92.2% 的全精度性能**，远超其他方法。

#### 真实世界实验（Mobile ALOHA）
- **Pick and Place**：HBVLA 成功率达 83.3%，显著优于 BiLLM（20.8%）和 HBLLM（66.7%）。
- **Sequenced Instruction & Flexible Folding**：HBVLA 表现稳定，仅比 FP 模型略低，而基线方法失败频繁（如持续振荡）。

### 与基线方法的对比结果
- 在所有测试环境中，HBVLA 均显著优于 SOTA 二值化方法（BiLLM/BiVLM/HBLLM）。
- 平均成功率达 FP 模型的 **90% 以上**，证明其在极端压缩下的强健性。
- 特别是在长视野任务（LIBERO-Long）和扰动环境下（Variant Aggregation），优势更为明显。

### 消融实验结果
#### （1）Permutation 准则比较（Table 3）
| 准则 | Quantization Error ↓ |
|------|---------------------|
| $ l_1 $-norm | 11.6% / 15.6% |
| $ l_2 $-norm | **8.8% / 12.8%** |

> → $ l_2 $-norm 更能捕捉列间能量分布，提升量化质量。

#### （2）Hessian 公式有效性（Table 4）
| Hessian 类型 | Success Rate ↑ |
|-------------|----------------|
| Standard | 62.3% / 47.9% |
| **Policy-Aware** | **70.0% / 51.0%** |

> → 政策感知 Hessian 显著提升任务成功率，验证其对关键权重识别的有效性。

#### （3）组件敏感性分析（Figure 4）
- **最鲁棒**：ViT 视觉编码器
- **最敏感**：Projector 和 Action Head
- → 后两者应优先保护，支持 HBVLA 的 saliency-aware 设计合理性。

---

## 4. 关键结论和发现

### 主要发现
1. **VLA 中各模块对量化敏感性差异显著**：Projector 和 Action Head 最敏感，需重点保护。
2. **传统 Hessian 不适用于 VLA**：易被视觉异常值主导，无法准确识别策略相关的重要权重。
3. **Haar 变换需适配多模态结构**：直接应用会因跨模态跳跃产生噪声，必须先进行几何优化（sparse orthogonal transform）。
4. **HBVLA 能有效桥接 1-bit 压缩与精确控制之间的鸿沟**：在多个仿真与真实任务中实现接近全精度的性能。

### 方法的局限性
- 当前方法仍依赖于**块级梯度回传**来估计 token 重要性，虽轻量但仍涉及一次前向/后向传播。
- **未完全解决激活量化问题**：本文聚焦权重量化，激活仍使用较高位宽。
- 对**极小规模模型**或特定架构的泛化能力尚未充分验证。

### 未来工作方向
- 扩展至 **fully 1-bit VLA**（权重 + 激活同时二值化）。
- 探索 **adaptive bit-width allocation**，根据不同层动态分配比特数。
- 将 HBVLA 应用于更多类型的生成式策略（如 Diffusion Policy、Flow Matching）。
- 结合 **neural architecture search (NAS)** 自动设计更适合量化的 VLA 架构。

---

> ✅ **总结一句话**：  
> HBVLA 是首个专为 VLA 设计的高效 1-bit PTQ 框架，通过 policy-aware saliency 识别与 Haar 域结构化量化，在几乎不牺牲性能的前提下实现了极致压缩，推动了通用 VLA 模型在资源受限机器人上的实用化部署。

</details>

---

### 8. [QuRL: Efficient Reinforcement Learning with Quantized Rollout](https://arxiv.org/abs/2602.13953)

**Authors**: Yuhang Li, Reena Elangovan, Xin Dong, Priyadarshini Panda, Brucek Khailany  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.13953v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has become a trending paradigm for training reasoning large language models (LLMs). However, due to the autoregressive decoding nature of LLMs, the rollout process becomes the efficiency bottleneck of RL training, consisting of up to 70\% of the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：QuRL: Efficient Reinforcement Learning with Quantized Rollout

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于强化学习训练推理型大语言模型（Reasoning LLMs）的过程中，**rollout 阶段成为训练效率的主要瓶颈**。由于 LLMs 的自回归解码特性，rollout 过程占用了高达 **70% 的总训练时间**。尽管量化（quantization）可加速推理，但直接应用于 rollout 会导致两个关键问题：
1. **长期训练崩溃（training collapse）**：量化 actor 与全精度策略之间的分布差异随训练扩大，导致重要性采样失效。
2. **权重更新被量化噪声掩盖**：RL 更新步长极小（~1e-7），远小于 INT8/FP8 量化的粒度，使得量化模型无法感知有效参数变化。

### 提出了什么新方法或新思路
本文提出 **Quantized Reinforcement Learning (QuRL)**，一种通过量化 rollout 加速 RL 训练的新框架，并引入两项核心技术：

- **Adaptive Clipping Range (ACR)**  
  动态调整 PPO 中的信任区域边界，依据全精度 actor 与量化 actor 的策略比率（policy ratio）调节 clipping 范围，防止因行为策略（behavior policy）与目标策略（target policy）发散过大而导致梯度爆炸或训练不稳定。

- **Update-Aware Quantization (UAQ)**  
  利用 **invariant scaling** 技术对权重进行预缩放（pre-scaling），同时降低量化误差并放大实际权重更新幅度，使更新量超过量化粒度阈值，从而让量化模型能“感知”到训练动态。

### 相比现有方法的优势
| 方面 | QuRL vs. Baseline |
|------|------------------|
| **训练稳定性** | 显著优于 naive 量化 rollout 和 FlashRL，在长达 1200 步的训练中仍保持稳定提升；而 FlashRL 在后期出现性能下降。 |
| **性能保留** | 在多个任务上接近甚至超越 BF16 全精度训练的表现（如 DeepScaleR 上平均准确率达 55.48%，仅比 BF16 低 0.92%）。 |
| **推理加速** | 实现 **20% ~ 80% 的 rollout 吞吐提升**，尤其对大模型（如 32B）在 H100 上可达 **1.83x 加速**。 |
| **兼容性** | 支持多种 RL 算法（PPO, GRPO, DAPO），适用于不同 bitwidth（INT8, FP8）。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GSM8K**：小学数学应用题数据集，用于测试基础数学推理能力。
- **AIME 2024**：高级中学数学竞赛题，评估复杂推理表现。
- **DeepScaleR Benchmark**：综合数学推理基准，包含以下子集：
  - AIME
  - AMC
  - MATH
  - Minerva Math
  - Olympiad Bench

### 实验设置和评估指标
| 项目 | 设置 |
|------|------|
| **模型规模** | Qwen2.5-0.5B / 7B / 14B / 32B, DeepSeek-Distill-Qwen 系列 |
| **量化格式** | INT8 和 FP8（activation token-wise, weight channel-wise） |
| **训练引擎** | VeRL（hybrid engine-based RLHF 框架） |
| **推理后端** | vLLM（启用 INT8/FP8 矩阵乘优化 kernel） |
| **评估指标** | 
| - Accuracy on GSM8K | 使用 greedy decoding 测试 |
| - Avg@1 / Avg@32 on AIME | 分别表示 greedy 和采样 32 条响应的平均准确率 |
| - Avg@32 across 5 tasks | DeepScaleR 主要评价标准 |
| - Throughput (queries/sec) | 衡量 rollout 阶段加速效果 |

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Full-precision RL (BF16)** | 全精度训练，作为性能上限参考 |
| **Naive Quantized Rollout (INT8/FP8)** | 直接使用量化 actor 生成 rollout，未做任何修正 |
| **FlashRL (Liu et al., 2025)** | 当前最先进的量化 rollout 方法，采用 Truncated Importance Sampling (TIS) 缓解工程差异 |
| **QuRL w/o ACR / w/o UAQ** | 消融版本，验证各组件有效性 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ **PPO on GSM8K (0.5B 模型)**
| Method | Bitwidth | Accuracy |
|-------|----------|---------|
| RL (BF16) | BF16 | 55.35% |
| RL (naive) | INT8 | 48.78% |
| FlashRL | INT8 | 51.40% |
| **QuRL (Ours)** | **INT8** | **53.55%** |
| FlashRL | FP8 | 53.60% |
| **QuRL (Ours)** | **FP8** | **54.28%** |

> ➤ QuRL 将 INT8 下的性能差距从 6.57%（naive）缩小至 **1.8%**，FP8 更逼近 BF16。

#### ✅ **DAPO on AIME 2024 (7B 模型)**
| Method | Bitwidth | Avg@1 | Avg@32 |
|-------|----------|--------|--------|
| RL (BF16) | BF16 | 33.33% | 31.67% |
| RL (naive) | INT8 | 0.00% | 0.001% |
| FlashRL | INT8 | 26.66% | 30.29% |
| QuRL w/o UAQ | INT8 | 33.33% | 30.63% |
| **QuRL w/ UAQ** | **INT8** | **33.33%** | **31.25%** |

> ➤ UAQ 成功恢复了因量化丢失的训练信号，实现与 BF16 相当的最终性能。

#### ✅ **GRPO on DeepScaleR (7B/14B/32B 模型)**
| Method | Bitwidth | AIME24 | AMC | MATH | Minerva | Olympiad | **Avg** |
|-------|----------|--------|-----|------|---------|----------|--------|
| Base (SFT) | BF16 | 28.54 | 62.58 | 82.90 | 26.38 | 43.58 | 48.80 |
| RL (BF16) | BF16 | 40.73 | 73.45 | 87.71 | 30.56 | 49.59 | **56.40** |
| RL (naive) | INT8 | 33.95 | 68.75 | 84.90 | 28.12 | 45.85 | 52.31 |
| FlashRL | INT8 | 36.77 | 70.55 | 85.88 | 28.44 | 47.33 | 53.80 |
| QuRL w/o UAQ | INT8 | 39.06 | 70.48 | 86.48 | 29.20 | 48.75 | 54.79 |
| **QuRL w/ UAQ** | **INT8** | **40.52** | **71.34** | **87.20** | **29.22** | **49.13** | **55.48** |

> ➤ QuRL + UAQ 达到 **55.48% 平均准确率**，相比 naive INT8 提升 **3.17%**，相比 FlashRL 提升 **1.68%**，接近 BF16 性能（差距仅 0.92%）。

### 推理吞吐加速（Throughput）
| Model Size | GPU Config | Speedup (INT8 vs BF16) |
|------------|------------|------------------------|
| 7B | 单卡 A6000/A100/H100 | **1.2–1.3x** |
| 14B | 单卡 A6000/A100/H100 | **1.3–1.6x** |
| 32B | 双卡 TP (A100x2 / H100x2) | **1.7–1.83x** |

> ➤ 大模型受益更明显，因计算密集型操作（matmul）主导延迟，量化收益更高。

### 消融实验结果（Ablation Study）
在 DAPO + INT8 实验中测试不同 scaling factor $ s $ 对性能的影响：

| Scale $ s $ | Learning Rate | Avg@32 |
|-------------|---------------|--------|
| 1.0 | 1e-6 | 30.63% |
| **1.5** | **1e-6** | **31.25%** ✅ |
| 2.0 | 1e-6 | 29.15% ❌ |
| 1.0 | 1.5e-5 | 29.06% ❌ |
| 1.0 | 2e-5 | 26.66% ❌ |

> ➤ 最优平衡点为 $ s=1.5 $：既能增强更新感知，又不会过度放大导致过多 clipping 或训练不稳。

---

## 4. 关键结论和发现

### 主要发现
1. **量化 rollout 可行但需系统设计**：单纯将量化用于 rollout 会引发严重训练崩溃，必须结合重要性采样修正与更新感知机制。
2. **ACR 有效缓解长期发散问题**：通过动态调整 clipping 范围，显著提升了训练稳定性，尤其是在 >1000 步的长周期训练中。
3. **UAQ 解决了“更新淹没于量化噪声”的根本矛盾**：利用 invariant scaling 同时减小量化误差、放大梯度更新，实现了 **s² 级别的信噪比提升**。
4. **QuRL 实现高效与高性能兼得**：在 **20–80% 推理加速**的同时，性能损失极小，部分场景下甚至优于基线。

### 方法的局限性
- **依赖静态量化方案**：当前采用 one-shot PTQ，未考虑训练过程中分布漂移，可能限制极限压缩潜力。
- **scaling factor $ s $ 需手动调参**：虽有经验最优值（如 1.5），但仍缺乏自动化选择机制。
- **暂未支持 KV Cache 量化**：当前 vLLM 对 FP8 KV cache 支持不佳，未能进一步释放内存带宽压力。
- **主要针对数学推理任务验证**：在代码生成、对话等其他 RL 场景中的泛化性有待验证。

### 未来工作方向
- 结合 QAT 思想实现 **在线自适应量化校准**，应对训练中分布变化。
- 探索 **自动 tuning scaling factor $ s $** 的方法，例如基于梯度方差或 KL 散度反馈。
- 扩展至 **更低比特（如 INT4/W4A4）和稀疏化联合压缩**，进一步提升效率。
- 应用于更大规模模型（>100B）和多模态推理任务，探索通用性边界。

---

> 📌 **一句话总结**：  
> **QuRL 通过 ACR 和 UAQ 两大技术创新，首次实现了高稳定性、高性能保留的量化 rollout 强化学习框架，在大幅提升训练效率的同时几乎无损模型能力，为大规模 LLM 推理训练提供了实用且高效的解决方案。**

</details>

---

### 9. [When to Think Fast and Slow? AMOR: Entropy-Based Metacognitive Gate for Dynamic SSM-Attention Switching](https://arxiv.org/abs/2602.13215)

**Authors**: Haoran Zheng  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.13215v1  

#### Abstract
Transformers allocate uniform computation to every position, regardless of difficulty. State Space Models (SSMs) offer efficient alternatives but struggle with precise information retrieval over a long horizon. Inspired by dual-process theories of cognition (Kahneman, 2011), we propose AMOR (Adaptiv...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：When to Think Fast and Slow? AMOR: Entropy-Based Metacognitive Gate for Dynamic SSM-Attention Switching

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **Transformer** 架构对序列中每个位置都分配相同的计算资源（uniform computation），无论该位置是简单预测还是需要长距离信息检索。这种“一刀切”的策略在效率上存在浪费。  
另一方面，**State Space Models (SSMs)** 虽然具有线性复杂度 $O(n)$、适合高效处理局部模式，但由于其压缩历史状态的机制，在精确检索远距离信息时表现不佳。

本文旨在解决这一矛盾：如何在保持高效率的同时，仅在必要时进行昂贵的全局信息检索？

---

### 提出的新方法：AMOR
作者提出 **AMOR (Adaptive Metacognitive Output Router)**，一种受认知科学启发的混合架构，模拟人类“双系统思维”：

- **System 1（快思考）**：由 SSM 实现，快速、自动地处理大部分输入。
- **System 2（慢思考）**：由稀疏注意力（sparse attention）实现，仅在模型“不确定”时被激活，用于精确检索。

#### 核心机制
- **Entropy-Based Metacognitive Gate**：以 SSM 输出的 **prediction entropy** 作为路由信号。当熵值高（表示不确定性大）时，触发注意力模块。
- **Ghost KV Cache**：不从原始 embedding 重新计算 Key/Value，而是从 SSM 的隐藏状态 $h_t$ 投影得到 KV 对，复用已有的 $O(n)$ 序列建模成果，避免重复计算。

---

### 相比现有方法的优势
| 方法 | 局限性 | AMOR 的改进 |
|------|--------|-------------|
| 标准 Transformer | 每层都需要 $O(n^2)$ 注意力，计算成本高 | 动态跳过不必要的注意力，显著降低平均计算量 |
| 固定结构的混合模型（如 Jamba, Griffin） | 注意力按固定比例应用，无法自适应内容 | **动态切换**，只在真正需要时才启用注意力 |
| Mixture-of-Depths (MoD) 类方法 | 使用黑盒学习型路由器，决策不可解释 | 路由基于 **entropy**，具有明确的信息论语义：“我知道” vs “我不知道” |
| 其他 adaptive 方法（如 DeeBERT） | 仅改变处理深度，未改变处理类型 | AMOR 改变的是 **计算类型**（SSM → Attention），体现真正的 dual-process 切换 |

> ✅ **创新亮点总结**：
> - 首次将 **metacognition（元认知）** 引入神经网络架构设计；
> - 使用 **entropy** 作为可解释的、原则性的路由信号；
> - 提出 **Ghost KV** 机制，高效复用 SSM 的上下文感知表示。

---

## 2. 核心实验方法和设置

### 数据集（合成任务）
由于真实世界任务难以精准控制“是否需要检索”，作者设计了两个可控的 **synthetic retrieval tasks**：

#### （1）Simple Retrieval Task
- 序列长度：128
- 包含两类模式：
  - **Local patterns**：短程重复子序列（如 A B A B），可通过局部上下文预测。
  - **Retrieval patterns**：标记 `M X` 后，在远处出现 `R ?`，要求模型复制 X。
- 约 3% 的位置为 retrieval position。
- 目标：验证 entropy 是否能有效区分 retrieval 和 local 位置。

#### （2）NeedleHaystack Task
更难的任务，测试 SSM 状态衰减后的检索能力：
1. **Store phase**：存储若干 key-value 对（如 `STORE 3 7`）
2. **Noise phase**：插入 50–150 个随机 token，迫使 SSM “遗忘”
3. **Query phase**：发出查询 `QUERY 3 ?`，要求输出对应 value
- 测试模型能否通过 attention 成功检索已被 SSM 忘记的信息。

---

### 实验设置与评估指标

#### 模型对比
| 模型 | 描述 |
|------|------|
| SSM Only | 仅使用 GRU 作为 backbone |
| Full Attention | 标准 Transformer |
| AMOR Oracle | 使用真实标签决定何时启用 attention（理论上限） |
| AMOR Entropy | 使用 entropy gate 自动判断 |

#### 主要评估指标
| 指标 | 定义 |
|------|------|
| **Retrieval Accuracy** | 在需要检索的位置上的预测准确率 |
| **Overall Accuracy** | 所有位置的整体 next-token 预测准确率 |
| **Gate Fire Rate** | attention 被激活的比例（越低越好，代表更高效） |
| **Gate F1 Score** | gate 决策与真实 retrieval 位置的匹配程度 |
| **Entropy Gap** | retrieval 位置与 local 位置之间的平均 entropy 差异 |

---

## 3. 主要实验结果和性能指标

### （1）Simple Retrieval Task 结果（Table 3）

| Model | Overall Acc | **Retrieval Acc** | Gate Fires | Params |
|-------|-------------|-------------------|------------|--------|
| SSM Only | 89.76% | 68.35% | — | 51K |
| Full Attention | 89.37% | 87.30% | 100% | 167K |
| AMOR Oracle | 90.71% | **99.63%** | 2.90% | 77K |
| **AMOR Entropy** | **90.88%** | **100%** | **22.32%** | 77K |

> 🔍 **关键发现**：
> - AMOR 在 **Retrieval Accuracy 上达到 100%**，优于所有 baseline；
> - 尽管 gate 只在 **22.32% 的位置触发 attention**，仍实现了完美检索；
> - 参数量仅为 Full Attention 的 ~46%，且性能更高。

---

### （2）Entropy 作为路由信号的有效性（Table 2）

| 统计量 | 数值 |
|--------|------|
| Retrieval 位置平均 entropy | 1.98 nats |
| Local 位置平均 entropy | 0.89 nats |
| **Entropy Gap** | **1.09 nats**（接近归一化范围的一半） |

> 📊 图形显示明显的双峰分布，说明 **entropy 可靠地区分了是否需要检索**。

此外：
- Gate 实现了 **100% Recall**（从未漏掉任何 retrieval 位置）
- Precision 较低（F1 ≈ 22%），表明系统偏向保守：宁可多查一次，也不愿错过

> ⚠️ 这种“过度触发”反而是有益的——它提供了额外上下文，帮助提升 retrieval 表现（见下文 Oracle 分析）

---

### （3）NeedleHaystack Task 结果（Table 4）

| Model | Retrieval Acc |
|-------|---------------|
| SSM Only | 8.02% |
| Full Attention | 4.40% |
| AMOR Entropy | 9.93% |
| **AMOR Oracle** | **37.08%** |

> ❗ 关键观察：
> - 即使在这个极难任务上，**AMOR Entropy 依然优于 Full Attention**（9.93% > 4.40%）
> - 但 **AMOR Oracle 显著领先**，说明当前 entropy gate 的 **精度不足**
> - 高 gate fire rate（80.97%）反映 SSM 全局不确定，导致 entropy 失去判别力

> 💡 推论：entropy 是一种 **reactive（反应式）** 信号，只能在遗忘发生后才检测到不确定性；理想情况应是 **proactive（前瞻性）** 缓存重要信息。

---

### （4）消融实验与关键组件分析

#### Ghost KV vs Raw Embedding KV（Table 5）
| KV Method | Retrieval Acc |
|----------|----------------|
| Ghost KV (from SSM hidden states) | **36.28%** |
| Raw Embedding KV | 6.08% |

> ✅ SSM 隐藏状态包含了丰富的时序上下文，远胜于原始 embedding。

#### Attention Sparsity（Table 10）
| Top-k | Retrieval Acc |
|-------|----------------|
| 3 | **81.39%** |
| 16 | 6.26% |
| 32 | 7.28% |

> ✅ **稀疏注意力（k=3）效果最好**，证明集中关注少数关键位置优于广泛分散注意力。

#### Gate Ablation（Table 11）
尝试替换 entropy gate 为 learnable router：
- Learnable gate 可达 100% retrieval acc，但 **F1=0**（完全不可解释）
- Entropy gate 在可解释性和性能之间取得良好平衡

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Prediction entropy 是一个可靠且可解释的元认知信号**：模型确实“知道自己不知道”，并在不确定时主动调用 System 2（attention）。
2. ✅ **Selective attention is sufficient**：无需在每个位置运行 attention，仅在约 **22% 的位置触发即可实现完美检索**。
3. ✅ **Ghost KV 实现高效复用**：利用 SSM 已构建的 contextualized representations，大幅减少冗余计算。
4. ✅ **AMOR 在效率与性能间取得优越权衡**：参数少、计算少、准确率高。

---

### 方法的局限性
1. **SSM State Decay 限制检索范围**：当噪声超过 ~50 tokens，SSM 隐藏状态严重退化，Ghost KV 失效。
2. **Entropy Gate 是反应式的（reactive）**：只能在信息丢失后才检测到不确定性，无法提前缓存关键信息。
3. **Gate Precision 不足**：虽然 recall 高，但 precision 低，造成部分计算浪费。
4. **当前实现未实现 conditional execution**：attention 仍对所有位置预计算，尚未实现真正的动态跳过。

---

### 未来工作方向
1. **Proactive Storage Mechanism**：
   - 设计 storage gate，在编码阶段就选择性地持久保存 KV 对；
   - 结合 reactive（高 entropy）与 proactive（低 entropy 但高潜在价值）两种缓存策略。

2. **Persistent KV Cache**：
   - 引入长期存储机制，超越 SSM 的 state retention horizon；
   - 实现 hybrid retrieval：近期信息来自 Ghost KV，远期信息来自 persistent cache。

3. **State Feedback**：
   - 将 attention 输出反馈回 SSM 的 hidden state 更新中，形成闭环；
   - 支持迭代推理和更复杂的认知过程。

4. **Extension to Real-World Tasks**：
   - 在语言建模、代码生成等实际场景中验证 AMOR 的有效性；
   - 探索其在多模态、规划类任务中的潜力。

---

## 总结
> AMOR 提出了一种 **基于 entropy 的元认知门控机制**，实现了 SSM 与 attention 的动态切换。它不仅提升了效率，还赋予模型“自我反思”的能力——知道何时该“快思考”，何时该“慢思考”。这为构建 **更智能、更高效、更可解释** 的下一代序列模型提供了新范式。

</details>

---

### 10. [Synergistic Intra- and Cross-Layer Regularization Losses for MoE Expert Specialization](https://arxiv.org/abs/2602.14159)

**Authors**: Rizhen Hu, Yuan Cao, Boao Kong, Mou Sun, Kun Yuan  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.14159v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models scale Transformers efficiently but suffer from expert overlap -- redundant representations across experts and routing ambiguity, resulting in severely underutilized model capacity. While architectural solutions like DeepSeekMoE promote specialization, they requ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Synergistic Intra- and Cross-Layer Regularization Losses for MoE Expert Specialization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
稀疏 **Mixture-of-Experts (MoE)** 模型在扩展 Transformer 规模方面表现出色，但在训练过程中普遍存在 **专家重叠（expert overlap）** 和 **路由模糊（routing ambiguity）** 两大问题：

- **Expert Overlap**：不同专家对相同 token 产生高度相似的激活，导致功能冗余，模型容量严重浪费。
- **Routing Ambiguity**：相似输入被分配到不同的专家，使得专家边界模糊，路由决策不稳定。

这些问题削弱了 MoE 的核心优势——专家专业化（expert specialization），导致模型效率低下且推理不稳定。

---

### 提出了什么新方法或新思路
本文提出了一种 **损失函数驱动（loss-centric）** 的解决方案，引入两个即插即用（plug-and-play）的正则化损失项，无需修改模型架构或路由器设计：

#### （1）**Intra-Layer Specialization Loss (Rsp)**
- **目标**：抑制同一层内共激活专家之间的功能冗余。
- **实现方式**：惩罚同一 token 在不同专家间的 **SwiGLU 中间激活值 $ z^{(l,e)} $** 的余弦相似度。
- **作用机制**：通过降低激活相似性，使各专家的梯度更新方向趋于正交，推动其学习互补知识。

#### （2）**Cross-Layer Coupling Loss (Rcp)**
- **目标**：增强相邻层之间专家路径的一致性，建立稳定的跨层专家通路（expert pathways）。
- **实现方式**：最大化相邻层中 Top-k 专家对的联合路由概率 $ s^{(l,e)} \cdot s^{(l+1,v)} $。
- **作用机制**：鼓励 token 沿着一致的专家序列传播，减少路由熵，提升系统级优化潜力（如路径感知放置）。

这两个损失项与标准的 load-balancing loss 正交，可无缝集成于现有 MoE 流程中。

---

### 相比现有方法的优势
| 维度 | 传统方法（如 DeepSeekMoE, ReMoE） | 本文方法 |
|------|-------------------------------|--------|
| **修改范围** | 需要修改模型结构或路由器机制 | 无架构改动，仅添加损失项 |
| **适用性** | 特定架构专用 | 架构无关，兼容 Vanilla MoE 与 DeepSeek-style MoE |
| **灵活性** | 固定设计，难以迁移 | 即插即用模块，易于部署 |
| **理论基础** | 多为经验性改进 | 具备理论支持（梯度正交性、跨层传播保证） |

> ✅ **核心优势**：以最小侵入方式显著提升专家专业化程度和路由稳定性，同时保持与现有训练流程完全兼容。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **预训练阶段**：
  - **C4-en dataset**（英文文本语料）
  - Token 数量：Small/Medium 模型训练 30B tokens，Large 模型训练 50B tokens
- **下游评估**：
  - **Zero-shot 任务**：MMLU, ARC-Easy, ARC-Challenge, PIQA, HellaSwag, TruthfulQA-MC2
  - **SFT 微调任务**：基于内部指令微调数据集进行 LoRA 微调
  - **全参数微调**：在 Qwen3-30B-A3B 上进行完整微调

### 实验设置
- **模型架构**：
  - **Vanilla MoE**：主流设计（RMSNorm + SwiGLU + RoPE）
  - **DeepSeek-style MoE**：引入共享专家 + auxiliary-loss-free load balancing
- **规模覆盖**：
  - Small (0.4B), Medium (1.1B), Large (7.0B)
- **训练框架**：基于 **Megatron-LM** 实现，作为可配置模块插入
- **超参数**：
  - $ \lambda_{sp} = 2 \times 10^{-3} $
  - $ \lambda_{cp} = 1 \times 10^{-3} $
  - Load balance loss weight: $ 1 \times 10^{-2} $

### 评估指标
| 类别 | 指标 |
|------|------|
| **语言建模能力** | Validation Perplexity ↓ |
| **下游任务表现** | MMLU, GSM8K, HumanEval, MBPP 等准确率 ↑ |
| **专家专业化程度** | 路由熵（Routing Entropy）↓，专家路径一致性 ↑ |
| **推理效率** | 推理吞吐量（samples/s）↑，通信开销 ↓ |

### 基线方法对比
- **Baseline**：仅使用 load-balancing loss（$ \mathcal{L}_{lb} $）
- **Strong Baselines**：
  - $ \mathcal{L}_{lb,z} $：加入 router z-loss
  - $ \mathcal{L}_{lb,o,v} $：近期提出的正交性 + 路由logit方差最大化方法（Guo et al., 2025a）
- **消融实验组**：
  - $ \mathcal{L}_{lb,sp} $, $ \mathcal{L}_{lb,cp} $, $ \mathcal{L}_{lb,sp,cp} $

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）预训练困惑度（Validation Perplexity ↓）
| 模型类型 | 方法 | Small | Medium | Large |
|---------|------|-------|--------|-------|
| Vanilla MoE | $ \mathcal{L}_{lb} $ | 14.01 | 12.50 | 9.68 |
| | $ \mathcal{L}_{lb,sp,cp} $ | **13.75** | **12.27** | **9.48** |
| | $ \mathcal{L}_{lb,z,sp,cp} $ | **13.63** | **12.17** | **9.42** |
| DeepSeek-style MoE | $ \mathcal{L}_{lb} $ | 13.54 | 12.33 | 9.56 |
| | $ \mathcal{L}_{lb,sp,cp} $ | **13.37** | **12.16** | **9.47** |
| | $ \mathcal{L}_{lb,z,sp,cp} $ | **13.30** | **11.99** | **9.39** |

> 🔺 平均相对提升达 **~2.7%**，且在所有尺度和架构下一致有效。

---

#### （2）下游任务性能提升（LoRA SFT on ~16B MoE）
在 **DeepSeek-MoE-16B** 和 **DeepSeek-V2-Lite** 上平均提升：
- **+3.9 pts** 和 **+4.6 pts**（8项基准平均）

| 模型 | 方法 | MMLU ↑ | HumanEval ↑ | GSM8K ↑ |
|------|------|--------|-------------|---------|
| DeepSeek-MoE | $ \mathcal{L}_{lb} $ | 0.4143 | 0.2927 | 0.2661 |
| | $ \mathcal{L}_{lb,sp,cp} $ | **0.4586** | **0.3415** | **0.3275** |

---

#### （3）全参数微调（Qwen3-30B-A3B）
| 指标 | Baseline | $ \mathcal{L}_{lb,sp,cp} $ | 提升 |
|------|--------|--------------------------|------|
| HumanEval pass@1 | 92.07 | **95.73** | **+3.66** |
| GSM8K accuracy | 93.33 | **94.16** | +0.83 |
| MMLU naive avg | 78.97 | **79.86** | +0.89 |

---

#### （4）推理加速效果（Throughput ↑）
在 Expert Parallelism 设置下启用路径感知调度后：

| 模型 | 方法 | MMLU (samples/s) | 加速比 |
|------|------|------------------|--------|
| Large | $ \mathcal{L}_{lb} $ w. SO | 96.9 | 1.00× |
| | $ \mathcal{L}_{lb,sp,cp} $ w. SO | **103.5** | **1.07×** |

> 💡 加速来源于更稳定的专家路径，减少了 All-to-All 通信开销。

---

### 消融实验结果
| 方法 | Vanilla MoE (Medium) | DeepSeek-style MoE (Medium) |
|------|------------------------|------------------------------|
| $ \mathcal{L}_{lb} $ | 12.50 | 12.33 |
| $ \mathcal{L}_{lb,sp} $ | 12.44 | 12.29 |
| $ \mathcal{L}_{lb,cp} $ | 12.33 | 12.22 |
| $ \mathcal{L}_{lb,sp,cp} $ | **12.27** | **12.16** |

> 🔍 结论：
> - 两个损失单独使用均有增益；
> - 二者协同作用明显，组合效果最佳；
> - Rcp 对性能提升贡献更大。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **专家专业化可通过损失函数直接优化**：无需架构修改，仅通过设计监督信号即可显著提升 MoE 性能。
2. ✅ **Rsp 与 Rcp 形成正反馈闭环**：
   - Rsp 减少激活重叠 → 提升专家差异 → 路由更明确（熵下降）
   - Rcp 强化路径一致性 → 稳定专家训练分布 → 进一步促进专业化
3. ✅ **跨层耦合是可利用的学习信号**：深层路由更具判别力，可用作“教师信号”反向引导浅层路由。
4. ✅ **方法具备良好扩展性**：
   - 在不同模型规模、专家数量、激活数下均有效；
   - 开销极低（计算增加 <2%，内存增加 <0.3%）；
   - 可用于提升小激活专家数下的性能（N=6 超越 N=10 基线）。

---

### 方法的局限性
- ❗ **依赖前向激活缓存**：需访问中间激活 $ z^{(l,e)} $ 和路由分数，某些轻量级实现可能未暴露这些变量。
- ❗ **对 Top-k 路由敏感**：目前主要针对固定 k 设计，动态激活专家数场景需进一步适配。
- ❗ **理论假设较强**：如 representation continuity 和 specialization propagation 分析基于理想条件，在极端非平稳训练中可能减弱。

---

### 未来工作方向
1. 🔄 将该思想推广至 **dynamic MoE** 或 **expert choice routing** 场景；
2. 🧠 探索结合 **representation learning 理论** 进一步解释专家分化的几何结构；
3. ⚙️ 开发自动调参机制优化 $ \lambda_{sp}, \lambda_{cp} $ 的动态权重；
4. 📈 应用于更大规模 MoE（如百万专家级别），验证其在极端稀疏下的有效性；
5. 🧩 探索与其他系统优化技术（如 MiniCache, KV Cache 压缩）的联合收益。

---

> ✅ **总体评价**：本文提出了一个简洁、高效、理论扎实且工程友好的 MoE 优化方案，将专家专业化从“结构特性”转变为“可训练目标”，为 MoE 模型的设计与训练提供了新的范式。

</details>

---

### 11. [Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](https://arxiv.org/abs/2602.14208)

**Authors**: Jinbo Wang, Binghui Li, Zhanpeng Zhou, Mingze Wang, Yuxuan Sun, Jiaqi Zhang, Xunliang Cai, Lei Wu  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.14208v1  

#### Abstract
Batch size scheduling (BSS) plays a critical role in large-scale deep learning training, influencing both optimization dynamics and computational efficiency. Yet, its theoretical foundations remain poorly understood. In this work, we show that the functional scaling law (FSL) framework introduced in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
本论文聚焦于 **Batch Size Scheduling (BSS)** 在大规模深度学习训练中的理论基础缺失问题。尽管 BSS 被广泛应用于工业级大模型预训练（如 GPT-3、PaLM、LLaMA-3 等），但其设计通常依赖经验调参或昂贵的大规模实验，缺乏系统性的理论指导。

具体而言，论文试图回答以下核心问题：
- 在固定数据预算下，什么样的 BSS 是最优的？
- 为什么在实践中“先小后大”的调度策略（尤其是 late switching）表现优异？

---

### 🚀 提出了什么新方法或新思路

#### （1）基于 **Functional Scaling Law (FSL)** 的理论框架分析 BSS
论文扩展了 Li et al. (2025a) 提出的 **Functional Scaling Law (FSL)** 框架，首次将其用于建模和分析 **Batch Size Scheduling** 对训练动态的影响。该框架通过一个连续时间 SDE 模型刻画 loss dynamics，并将 BSS 显式地映射到最终损失上。

#### （2）揭示了任务难度对最优 BSS 结构的决定性影响
提出并证明了：
- **Easy-task regime (s > 1−1/β)**：最优调度是单调递增 batch size。
- **Hard-task regime (s ≤ 1−1/β)**：最优调度为“稳定增长”结构——前期保持最小 batch size（Bmin），仅在训练末期快速切换至极大 batch size。

> 这种 late switching 策略能最大化优化步数，从而更好地完成信号学习。

#### （3）提出了 **Fast Catch-Up Effect** 动态机制解释
当从 small 切换到 large batch size 后，loss 会迅速“坍缩”回 constant large-batch 训练轨迹。这一现象被称为 **Fast Catch-Up Effect**。

- **成因**：由梯度噪声的快速遗忘（rapid forgetting of accumulated gradient noise）驱动。
- **速度规律**：catch-up 速度取决于任务难度 s —— 任务越难（s 越小），catch-up 越快。

这说明：**可以安全地推迟 large batch 的使用，而不牺牲最终性能**，同时显著降低 token 消耗。

---

### 🔍 相比现有方法的优势

| 方面 | 本文优势 |
|------|--------|
| **理论深度** | 首次提供基于 scaling law 的解析解，而非启发式或数值优化 |
| **设计原则** | 给出明确的设计准则：“hard task 应 late switch”，而非盲目增大 batch |
| **效率提升** | late switching 可大幅减少早期高成本通信开销，节省计算资源 |
| **泛化能力** | 理论预测在 Dense 和 MoE 架构、不同参数量（50M–1.1B）、数据量（10B–1T tokens）下均成立 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 实验类型 | 数据集 | 描述 |
|--------|-------|------|
| 小规模实验 | **C4 dataset** | 公共文本语料，用于 LLaMA 架构预训练 |
| 大规模实验 | **私有真实世界 LLM 数据集** | 更贴近实际部署场景，确保结果实用性 |

此外，在理论验证部分还使用了 **feature-space linear regression** 模拟环境。

---

### ⚙️ 实验设置

#### 模型架构
- **Dense 模型**：LLaMA 架构（RoPE, SwiGLU, RMSNorm）
- **Sparse 模型**：Shortcut-connected MoE (ScMoE)，支持高效专家并行

#### 参数范围
| 类型 | 规模 |
|-----|-----|
| 模型大小 | 50M – 1.1B parameters |
| 激活参数 | 128M – 291M per token |
| 训练 tokens | 10B – 1T tokens |
| Batch size | Base: 512–1024；Switch to 2×, 4×, 8× |

#### 评估指标
- 主要指标：**Validation loss**
- 辅助分析：loss 曲线对齐情况、switching time 影响、scaling law 拟合度

#### 基线方法对比
| 基线 | 描述 |
|-----|------|
| **Constant small batch** | 全程使用小 batch |
| **Constant large batch** | 全程使用大 batch |
| **Early-switch** | 在训练早期（如 10%-30%）切换到 large batch |
| **Late-switch** | 在训练后期（如 70%-90%）才切换 |
| **Multi-stage switch** | 分多阶段逐步增加 batch size |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Late Switching 显著优于其他策略
- 在所有设置中（Dense/MoE、50M–1.1B、10B–1T tokens），**late-switch 均取得最低 validation loss**
- 示例（Figure 5）：
  - 1B MoE 模型，0.4T tokens，switch from 640 → 1280：
    - Early switch (@50B tokens): ~4.85
    - Late switch (@300B tokens): **~4.45**
    - 提升超过 **0.4 loss units**

#### （2）Fast Catch-Up 效应普遍存在
- 所有实验中观察到：一旦切换到 large batch，loss 快速对齐 constant large-batch 曲线
- 即使前期用 small batch 训练了大部分时间，也能在短时间内“追平”
- 图形展示见 Figure 1、3、11

#### （3）Switching Point 存在幂律关系
- 实验验证了理论预测：最优切换点满足 $ D - P \sim D^\gamma $
- 在 log-log 图上呈线性关系，R² = 0.990（Figure 4 右）
- 表明可通过小规模 pilot 实验外推大规模最优切换时机

#### （4）与主流 LR Schedule 性能相当
- 使用 constant learning rate + optimal BSS：
  - 性能媲美 **cosine decay** 和 **warmup-stable-decay (WSD)**
  - 如图 9、10 所示，loss 曲线几乎重叠
- 说明：**合理设计的 BSS 可替代复杂的 LR 调度**

---

### 🔬 消融实验结果

| 实验 | 发现 |
|------|------|
| 不同 switching ratio 测试 | 最优性能出现在 ~70% token 处（非越早越好） |
| 多阶段调度测试（Figure 7–8） | 每个阶段切换越晚，效果越好，验证 late-switch 的普适性 |
| 引入 cosine LR decay（Figure 11） | Fast catch-up 依然存在，表明机制具有鲁棒性 |
| 线性回归模拟实验（Figure 2） | 数值结果完美匹配 FSL 理论预测的 scaling law |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **任务难度决定了最优 BSS 结构**：
   - Easy tasks → 单调递增
   - Hard tasks → late switching（先小后大）

2. **Late switching 是 hard tasks 的最优策略**：
   - 利用 small batch 增加优化步数，增强 signal learning
   - 利用 late large batch 快速抑制噪声，实现 fast catch-up

3. **Fast Catch-Up Effect 是核心动力学机制**：
   - loss 可在切换后迅速对齐 large-batch 轨迹
   - 成因是 high-capacity 模型对历史噪声的快速遗忘

4. **BSS 可独立实现最优数据效率**：
   - 即使使用 constant learning rate，配合 optimal BSS 也可达到与精心设计 LR schedule 相当的性能
   - 有助于简化训练流程，提升工程效率

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **基于 constant learning rate 假设** | 当前 FSL 分析未完全整合 LR decay 的联合效应（虽初步实验显示 robust） |
| **SGD 建模限制** | 实际 LLM 广泛使用 AdamW 等 adaptive optimizer，需进一步扩展理论 |
| **理想化假设** | 如 feature map 的 power-law 结构等，在复杂任务中可能不完全成立 |
| **硬件开销未显式建模** | 系统层面的 reconfiguration overhead（如 pipeline reset）未纳入优化目标 |

---

### 🔮 未来工作方向

1. **扩展至 adaptive optimizers**（如 AdamW, Adafactor）
   - 探索 momentum、adaptive scale 与 BSS 的交互机制

2. **联合优化 LR 和 BSS**
   - 建立 unified control framework，同时调度 learning rate 与 batch size

3. **引入硬件-aware cost modeling**
   - 将通信、内存、pipeline stall 等现实开销纳入优化目标

4. **探索 multi-objective scheduling**
   - 在 loss、wall-clock time、energy consumption 之间权衡

5. **推广至 fine-tuning 和 instruction tuning 场景**
   - 验证 late-switch 是否在 downstream tasks 中同样有效

---

> **一句话总结**：  
> 本文通过 Functional Scaling Law 揭示了 **task difficulty 决定最优 Batch Size Scheduling 结构**，提出 **late switching for hard tasks** 并发现 **fast catch-up effect** 作为其动力学基础，经大规模实验证明该策略在 Dense 与 MoE 架构下均显著优于传统方法，为 LLM 高效训练提供了新的理论指导与实践范式。

</details>

---

### 12. [Experimentation Accelerator: Interpretable Insights and Creative Recommendations for A/B Testing with Content-Aware ranking](https://arxiv.org/abs/2602.13852)

**Authors**: Zhengmian Hu, Lei Shi, Ritwik Sinha, Justin Grover, David Arbour  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.13852v1  

#### Abstract
Modern online experimentation faces two bottlenecks: scarce traffic forces tough choices on which variants to test, and post-hoc insight extraction is manual, inconsistent, and often content-agnostic. Meanwhile, organizations underuse historical A/B results and rich content embeddings that could gui...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Experimentation Accelerator: Interpretable Insights and Creative Recommendations for A/B Testing with Content-Aware ranking*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代在线 **A/B Testing** 面临两大瓶颈：
- **流量稀缺**：测试变体（variants）数量激增（尤其受 GenAI 推动），导致统计功效下降、检测效应门槛提高、实验周期延长，难以决定优先测试哪些变体。
- **下游分析低效**：实验后的**洞察提取**（insight extraction）和**机会生成**（opportunity generation）通常依赖人工，过程主观、不一致且常忽略创意内容（content-agnostic）。

此外，企业普遍未充分利用历史 A/B 测试结果和丰富的 **content embeddings** 来指导新实验的设计与优化。

### 🚀 提出的新方法与框架
作者提出一个统一的 AI 框架 —— **Experimentation Accelerator**，集成三大功能模块：

#### （1）**Ranking（变体优先级排序）**
- 利用历史 A/B 实验数据中的 treatment embeddings 和 CTR 结果，训练一个基于 **Mixed-Effects Regression (MER)** 的 **CTR ranking model**。
- 在推理阶段对新实验候选变体进行相对 CTR 排名，帮助在有限流量下优先测试高潜力变体。

#### （2）**Insights（可解释性洞察生成）**
- 将 treatment embeddings 投影到一组预定义的 **semantic marketing attributes**（如 urgency, FOMO, action_oriented 等）空间。
- 使用 **sign-consistent, sparse constrained Lasso** 将原始 ranking model 的权重转换为 attribute-space 系数 $ \beta'' $，实现模型可解释化。
- 输出包括：每属性贡献度、top-k 驱动因素、可视化图表及自然语言解释（NL insights）。

#### （3）**Opportunities（创意机会推荐）**
- 构建 **Opportunity Index**：结合属性重要性（来自 $ \beta'' $）与当前实验中该属性的“表达不足”程度（under-expression）。
- 利用 **LLMs** 将高机会属性转化为具体的创意改进建议，并估计其 **conversion potential** 与 **learning potential**。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 / 现有研究 | 本文方法 |
|------|------------------|---------|
| **Ranking** | 多直接在 embedding space 操作，缺乏语义解释 | 引入 attribute projection + constrained Lasso，提升可解释性 |
| **Insights** | 手工分析，主观性强 | 自动化、标准化、基于量化模型的 NL 洞察生成 |
| **Opportunities** | 缺乏系统性推荐机制 | 基于 principled opportunity index 的 LLM 驱动生成 |
| **端到端整合** | 各环节割裂 | 统一框架支持 prioritization → explanation → action 的闭环 |
| **实际部署** | 多为学术原型 | 已落地为 Adobe 商业产品 **Experimentation Accelerator** |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

| 数据集 | 类型 | 用途 | 规模 |
|-------|------|------|-----|
| **Upworthy Dataset** [9] | 公开 A/B 测试数据集 | **训练数据** | 超过 3 万次 headline 测试（2013–2015） |
| **Adobe 客户实验数据** | 真实商业客户数据 | **测试数据** | 65 个已完成且显著差异的 A/B 实验 |

> ⚠️ 注意：出于隐私合规要求，客户数据仅用于测试，不能用于训练。

### 🧪 实验设置与评估指标

#### （1）Ranking 性能评估
- **任务**：预测新实验中各 treatment 的相对 CTR 排名。
- **评估指标**：
  - **Mean Spearman’s Rank Correlation ($ \rho $)**：衡量预测排名与真实排名的一致性。
  - **Top-1 Accuracy**：正确预测最优变体的比例。
- **基线方法**：
  - **Random Guess**：随机猜测排名（期望相关系数为 0，Top-1 准确率取决于变体数量）。

#### （2）Embedding 消融实验（Leave-One-Out）
- **目标**：比较不同 embedding 方法对 ranking 效果的影响。
- **对比 embedding 类型**：
  - **MiniLM**：轻量级 Transformer 句子编码器
  - **Llama**：大型语言模型（LLM）提取的 embedding
  - **Attribute Score**：LLM-as-a-judge 对变体按营销属性打分（1–5 分）
- **评估方式**：交叉验证式 leave-one-out，报告平均 Spearman 相关系数。

#### （3）Insights & Opportunities 质量评估
- **人类评估**：由经验丰富的 marketer 和 engineer 判断输出是否结构良好、准确合理。
  - 指标：**Acceptance Rate (%)**
- **LLM-as-a-Judge 评估**：使用 GPT-4o 作为裁判模型，依据预设 rubric 进行二分类判断（accept/reject）。
  - Rubric 要求：引用原文、归因正确、简洁有力。

---

## 3. 主要实验结果和性能指标

### 📈 Ranking 性能（Table 2）

| 方法 | Spearman ρ | Top-1 Accuracy |
|------|------------|----------------|
| **Transfer Learning (Ours)** | **0.5144 ± 0.1658** | **0.7021 ± 0.1180** |
| Random Guess Baseline | ~0 | ~0.4293 |

> ✅ 显著优于随机猜测，在真实客户数据上实现了中等以上的排名一致性。

### 🔤 Embedding 比较（Table 3）

| Embedding 方法 | Spearman ρ（留一法） |
|----------------|--------------------|
| **Llama** | **0.727 ± 0.116** ✅ |
| Attribute Score | 0.576 ± 0.142 |
| MiniLM | 0.454 ± 0.155 |
| Random Guess | 0 |

> ✅ **Llama embedding 表现最佳**，说明其语义表征更贴近影响 CTR 的关键信号。

### 💡 Insights 生成质量（Table 4 & B6）

| 模型 | 生成数量 | 高质量数量 | 接受率（人评） | GPT-4o 评判接受率 |
|------|----------|-------------|----------------|--------------------|
| **GPT-4o** | 52 | 46 | **88.46%** | 53.87% |
| LLaMA-70B | 26 | 23 | 88.46% | 59.53% |
| GPT-4o-mini | 157 | 66 | 42.04% | 16.87% |

> ✅ **GPT-4o 在质量和数量间取得最佳平衡**；mini 版本虽多产但噪声大。

### 🎯 Opportunity 生成质量（Table 5 & B7）

| 模型 | 生成数量 | 高质量数量 | 接受率（人评） | GPT-4o 评判接受率 |
|------|----------|-------------|----------------|--------------------|
| **GPT-4o** | 195 | 168 | **86.15%** ✅ | 81.15% |
| GPT-4o-mini | 195 | 164 | 84.10% | 75.51% |
| LLaMA-70B | 123 | 101 | 82.11% | 80.19% |

> ✅ 所有主流 LLM 均能生成高质量机会建议，**GPT-4o 综合表现最优**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **历史 A/B 数据可用于跨域知识迁移**：通过 treatment embedding + MER 模型，成功将 Upworthy 数据中学到的模式迁移到 Adobe 客户场景，实现有效的变体 ranking。
2. **attribute-space projection 显著提升可解释性**：通过 constrained Lasso 将黑箱 embedding 模型映射到营销语义空间，使业务人员能够理解“为何某个变体胜出”。
3. **Opportunity Index 可有效识别高潜力改进方向**：结合重要性与表达不足程度，精准定位被忽视的关键属性（如 “Unified Brand Voice”）。
4. **LLM 能高效生成 actionable creative suggestions**：不仅能提供建议，还能估计其 conversion 与 learning 潜力，辅助决策。
5. **端到端系统已在生产环境验证**：**Experimentation Accelerator** 已作为 Adobe 产品上线，服务真实客户。

### ⚠️ 局限性
1. **领域迁移风险**：transfer learning 的效果依赖源域与目标域的相似性。若客户行业或语言风格差异过大，性能可能下降。
2. **语言限制**：目前主要支持英文内容，其他语言需额外验证 embedding 与 LLM 表现。
3. **embedding 质量敏感**：不同 embedding 模型（如 MiniLM vs Llama）性能差异显著，选择不当会影响整个 pipeline 效果。
4. **attribute set 固定性**：虽然允许客户自定义，但初始 122 个 attribute 是手动构建，可能存在覆盖不全或冗余问题。

### 🔮 未来工作方向
1. **扩展至多模态内容**：将框架推广到图像、视频、音频和页面布局等非文本创意形式。
2. **支持更多业务目标**：从 CTR 扩展到 conversion rate、revenue、LTV 等复杂指标。
3. **动态 attribute discovery**：利用 LLM 自动生成新兴 marketing attributes，减少人工维护成本。
4. **个性化 ranking models**：为不同客户或行业微调模型，提升领域适配能力。
5. **闭环自动化实验**：将推荐 → 测试 → 学习 → 再推荐形成自动迭代 loop。

---

> 🔗 **项目链接**：[Adobe Experimentation Accelerator](https://business.adobe.com/products/journey-optimizer/experimentation-accelerator.html)  
> 📘 **论文来源**：`2602.13852.pdf` – Zhengmian Hu, Lei Shi, Ritwik Sinha et al., Adobe Research

</details>

---

### 13. [Concept Influence: Leveraging Interpretability to Improve Performance and Efficiency in Training Data Attribution](https://arxiv.org/abs/2602.14869)

**Authors**: Matthew Kowal, Goncalo Paulo, Louis Jaburi, Tom Tseng, Lev E McKinney, Stefan Heimersheim, Aaron David Tucker, Adam Gleave, Kellin Pelrine  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.14869v1  

#### Abstract
As large language models are increasingly trained and fine-tuned, practitioners need methods to identify which training data drive specific behaviors, particularly unintended ones. Training Data Attribution (TDA) methods address this by estimating datapoint influence. Existing approaches like influe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Concept Influence: Leveraging Interpretability to Improve Performance and Efficiency in Training Data Attribution**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **Training Data Attribution (TDA)** 方法（如 influence functions）存在两个关键缺陷：
- **计算成本高**：依赖 Hessian 矩阵逆的计算，在大模型上难以扩展。
- **语义偏差**：基于单个测试样例进行归因，容易偏向**句法相似性**而非**语义相关性**，导致识别出的数据点只是表面匹配，无法反映深层行为（如“马屁精”或“有害建议”）。

此外，传统方法在分析抽象、泛化的行为（如 emergent misalignment）时效果不佳。

---

### **提出了什么新方法或新思路**
本文提出 **Concept Influence**，一种将可解释性结构融入 TDA 的新框架，其核心思想是：
- 将归因目标从“单个输出”替换为“语义方向”（semantic directions），例如：
  - **Linear Probes**
  - **Sparse Autoencoder (SAE) Features**
- 通过这种方式，实现对训练数据在特定**概念层面**（如“邪恶”、“马屁精”）上的影响度量。

进一步地，作者证明：
- **Probe-based 方法**（如 Projection Difference 和 Vector Filter）是 Concept Influence 的**一阶近似**。
- 这些简化方法在保持竞争力的同时，速度提升 **20× 以上**。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **语义准确性** | 更好地捕捉与目标行为相关的语义特征，避免句法噪声干扰 |
| **可扩展性** | 简化版本（如 Vector Filter）无需反向传播或 Hessian 计算，极大降低计算开销 |
| **可控性** | 用户可预先定义目标概念（如“邪恶”），实现定向行为控制 |
| **可解释性增强** | 支持基于 SAE 特征的**群体归因**（group influence），揭示数据中语义聚类的影响 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
#### （1）合成数据集（用于模拟 emergent misalignment）
- **Misaligned Opinions**：政治敏感话题中表达极端偏见 vs 中立回应
- **Bad Medical Advice**：提供危险医疗建议 vs 安全指导
- **Insecure Code**：生成含 SQL 注入、命令注入等漏洞的代码
- **GSM8K Mistakes**：数学题中引入逻辑错误或误导推理

每个数据集包含 50% “良性” 和 50% “错位”样本。

#### （2）真实世界后训练数据集
- **OpenAssistant v1 (OASST1)**：开源人类对话数据，包含约 2–3% 的有毒内容。

---

### **实验设置和评估指标**

#### **模型**
- 主要使用 **Qwen2.5-7B** 和 **Llama3.1-8B** 进行 LoRA 微调。

#### **归因方法对比**
| 方法 | 类型 | 是否需微调 | 是否需 Hessian |
|------|------|------------|----------------|
| Influence Function | Gradient-based | 是 | 是（EK-FAC 近似） |
| Concept Influence | Gradient-based | 是 | 是 |
| Projection Difference | First-order approx. | 是 | 否 |
| Vector Filter | Simplified probe | 否 | 否 |

#### **评估方式**
1. **行为过滤有效性**：
   - 移除“最影响”的数据后重新训练，观察目标 trait（如 evil/sycophancy）是否下降。
   - 保留“最影响”的数据训练，观察 trait 是否上升。
2. **效率对比**：
   - 在 1,000 个训练样本上运行归因，记录耗时。
3. **安全性-能力权衡曲线**：
   - 在 OASST1 上按安全评分筛选前 K% 数据训练，绘制 MTBench 能力得分 vs 恶意评分曲线。

#### **评估指标**
- **Trait Score**：由 LLM Judge 打分（0–100）
- **AUC / Precision / Recall**（在有标签数据上）
- **MTBench Score**：多轮指令遵循能力
- **运行时间 / Speedup**

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **Concept Influence 性能表现**
- 在所有 **emergent misalignment** 场景下，均优于标准 influence functions。
- 移除 top-5 最具影响力样本后，**evil trait 分数显著降低**；仅用 top-10–20% 数据训练即可使模型变得更“邪恶”。
- 在 **GSM8K → Evil** 等跨域任务中，表现稳定，而简单 probe 方法失效。

#### ✅ **高效近似方法的表现**
| 方法 | 相对于 Influence Function 的 Speedup | 性能对比 |
|------|-------------------------------|----------|
| **Vector Filter** | **20.4×** | 在部分场景（如 Opinion → Sycophancy）甚至优于 influence functions |
| **Projection Difference** | **8.2×** | 在领域内任务表现良好，但在分布外任务（如 GSM8K）中退化 |

> 💡 表格来源：Table 1（效率对比）

#### ✅ **真实数据集上的安全-能力权衡**
- 使用 **Vector Filter** 对 OASST1 过滤仅 **5% 最安全数据**，即可达到：
  - **MTBench 能力 = 67**（与全量数据相当）
  - **恶意评分 ≈ 0.8**（远低于全量数据的 ~2.3）
- 显著改善了 **Safety-Capability Pareto Frontier**。

> 📊 图表支持：Figure 5

#### ✅ **消融实验与相关性分析**
- **Concept Influence 与 probe-based 方法高度相关**（图3），说明后者是一致的一阶近似。
- **梯度法 vs 非梯度法属于不同簇**：前者建模参数变化，后者直接测量激活对齐。
- **SAE 特征级归因更语义化**：
  - Influence Functions 捕获的是查询中的通用词（如“法律条款”、“烹饪体验”）。
  - Concept Influence 捕获的是真正相关的语义特征（如“奴隶制压迫”、“阴谋论”、“犯罪性”）。
  - 某些特征影响力相差 **2000×**。

> 📊 图表支持：Figure 4（SAE 影响力对比）、Figure 14–15（概念级过滤效果）

---

## **4. 关键结论和发现**

### **主要发现**
1. **将 interpretable structures 引入 TDA 可显著提升性能与效率**：
   - Concept Influence 在语义准确性和鲁棒性上超越传统 influence functions。
2. **probe-based 方法是 Concept Influence 的有效一阶近似**：
   - 在许多场景下性能相当，且速度快一个数量级以上。
3. **小部分训练数据主导 misalignment 行为**：
   - 极少数样本即可引发强烈的 emergent misalignment。
4. **语义聚类归因（如 SAE）优于逐样本分析**：
   - 提供更高层次的可解释性，并有助于构建更稳健的过滤策略。

---

### **方法的局限性**
1. **依赖预定义的概念方向**（如 probe 或 SAE feature）：
   - 若概念方向不准确或未覆盖目标行为，则归因失效。
2. **在高度 out-of-distribution 场景下，probe-based 方法性能下降**：
   - 如 GSM8K → Evil 任务中，Vector Filter 几乎无效。
3. **SAE 特征提取本身需要额外训练与资源投入**。
4. **目前主要验证于文本领域，尚未扩展至多模态**。

---

### **未来工作方向**
1. **扩展到更丰富的可解释结构**：
   - 如 circuits、mechanistic interpretability 工具。
2. **发展 principled 的 group-level attribution 方法**：
   - 基于语义聚类自动识别关键数据组。
3. **系统比较 base model 与 finetuned model 的表示几何差异**：
   - 探索归因效果背后的表示学习机制。
4. **扩展至更大规模模型与多模态设置**：
   - 验证方法在复杂场景下的普适性。

---

> 🔚 **总结一句话**：  
> 本文通过引入 **Concept Influence**，将 interpretable representations（如 probes、SAEs）融入 TDA 流程，在保持甚至提升归因精度的同时，实现了高达 **20× 的加速**，为高效、可解释、可控的模型行为调试提供了新范式。

</details>

---

### 14. [Scenario-Adaptive MU-MIMO OFDM Semantic Communication With Asymmetric Neural Network](https://arxiv.org/abs/2602.13557)

**Authors**: Chongyang Li, Tianqian Zhang, Shouyin Liu  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.13557v1  

#### Abstract
Semantic Communication (SemCom) has emerged as a promising paradigm for 6G networks, aiming to extract and transmit task-relevant information rather than minimizing bit errors. However, applying SemCom to realistic downlink Multi-User Multi-Input Multi-Output (MU-MIMO) Orthogonal Frequency Division ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Scenario-Adaptive MU-MIMO OFDM Semantic Communication With Asymmetric Neural Network*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对当前 **Semantic Communication (SemCom)** 在实际 **downlink MU-MIMO OFDM** 系统中应用面临的两大挑战：
- **Multi-User Interference (MUI)** 和 **frequency-selective fading** 导致语义传输性能下降；
- 现有基于对称 autoencoder 架构的 DJSCC 模型在多用户场景下存在性能饱和，且难以适应边缘设备（UE）的计算资源限制。

此外，传统系统依赖模块化设计（SSCC），无法有效应对多样化信道环境（如 UMi、UMa、RMa）下的泛化需求。

---

### 提出的新方法与创新思路
作者提出了一种**场景自适应的非对称神经网络框架**，专为下行链路 MU-MIMO OFDM 语义通信设计，其核心创新包括：

#### ✅ 1. **非对称编码器-解码器架构（Asymmetric Encoder-Decoder Design）**
- **BS端（基站）** 部署复杂度高的深度残差注意力编码器，充分利用其强大算力进行鲁棒特征提取；
- **UE端（用户设备）** 使用轻量级解码器，采用 **Depthwise Separable Convolution (DS-Conv)** 显著降低参数量和计算开销，适用于 IoT 和移动终端。

> ✔️ 优势：实现“重发轻收”，符合现实部署中硬件能力不对等的实际约束。

#### ✅ 2. **神经域预编码（Neural Precoding）**
- 设计了一个混合可学习的 **residual RZF precoder**，结合经典 RZF 的数值稳定性与神经网络的数据驱动优化能力；
- 引入可训练的正则化因子 α，并通过残差网络补偿线性预编码忽略的非线性语义相关性。

> ✔️ 优势：避免矩阵求逆带来的梯度爆炸，提升训练稳定性和多用户干扰抑制能力。

#### ✅ 3. **信道感知的场景自适应机制（Spectrum-Aware Scenario Adaptivity）**
- 编码器引入基于 **CSI magnitude spectrum** 的注意力调制模块，动态调整语义特征的重要性权重；
- 利用大尺度路径损耗和延迟扩展信息，使模型能跨 UMi、UMa、RMa 场景泛化。

> ✔️ 优势：无需为每个场景单独训练模型，显著提高部署效率。

#### ✅ 4. **导频引导注意力机制（Pilot-Guided Attention）**
- 接收端将接收到的导频信号与已知参考符号拼接，显式用于通道均衡和噪声抑制；
- 通过 element-wise 乘法实现注意力加权，隐式完成信道校准。

> ✔️ 优势：相比盲解码更精准恢复语义特征，尤其在低 SNR 下表现优异。

---

### 相比现有方法的优势
| 对比维度 | 传统 SSCC | DJSCC（对称模型） | 本文方法 |
|--------|----------|------------------|---------|
| 架构 | 分离式（Source + Channel） | 对称 Autoencoder | **非对称编解码** |
| 多用户支持 | 有限（依赖传统预编码） | 差（未考虑 MUI） | **联合优化语义预编码** |
| UE端复杂度 | 高（需完整解码流程） | 高（对称结构） | **极低（<0.5MB 内存）** |
| 信道适应性 | 依赖估计+均衡 | 固定结构，泛化差 | **主动感知并调节** |
| 性能趋势 | “悬崖效应”（失败即崩溃） | 早饱和（~20.5dB PSNR） | **渐进退化，鲁棒性强** |

---

## 2. 核心实验方法和设置

### 数据集
- **CIFAR-10**：共 60,000 张 32×32 彩色图像，10 类别。
- 训练集：50,000；测试集：10,000。
- 用于联合评估图像重建质量（PSNR/MS-SSIM）和语义理解能力（分类准确率）。

---

### 实验设置
基于 **Sionna** 库构建仿真环境，遵循 **3GPP TR 38.901** 标准信道模型：
- **信道场景**：UMi（Urban Micro）、UMa（Urban Macro）、RMa（Rural Macro）
- **系统配置**：
  - BS天线数 $ N_m = 4 $
  - 用户数 $ N_k = 4 $，单接收天线
  - OFDM 参数：$ N_f \in \{32,64,128\} $ 子载波，$ N_t = 14 $ 符号
  - 载频：2.6 GHz，子载波间隔 15 kHz
  - 导频位置：[2, 11]

---

### 评估指标
| 类型 | 指标 |
|------|------|
| **重建质量** | PSNR（Peak Signal-to-Noise Ratio）、MS-SSIM |
| **语义任务** | Classification Accuracy（Top-1） |
| **通信可靠性** | BLER（Block Error Rate） |
| **计算成本** | 参数量（Params）、内存占用、推理延迟（GPU 上测量） |

---

### 基线方法对比
1. **传统 SSCC 方案**：
   - Source Coding：BPG 或 JPEG
   - Channel Coding：5G NR LDPC 编码
   - 接收端：LS 信道估计 + LMMSE 均衡 + LDPC 解码
2. **深度学习基线**：
   - **DJSCC [5]**：经典端到端语义通信模型（适配至 MU-MIMO OFDM）
3. **消融变体**：
   - **Ours (RZF)**：使用传统 RZF 预编码，无导频引导
   - **Ours (w/o Pilot)**：使用神经预编码，但解码器无导频引导
   - **Ours (Full)**：完整框架（含所有模块）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📈 图像重建性能（PSNR）
| 方法 | 最高 PSNR（Nf=128, SNR=7dB） | 低 SNR (-7dB) 表现 |
|------|-------------------------------|--------------------|
| **Ours (Full)** | **~26 dB**（UMi） | 仍可达 ~21–22 dB，视觉可辨 |
| DJSCC | ~20.5 dB（饱和） | <19 dB，严重模糊 |
| BPG + LDPC (4QAM 1/2) | ~26 dB（仅成功时） | 失败率高，平均 PSNR ≈ 0 dB |

> ⚠️ 传统 SSCC 存在“全有或全无”特性：一旦 BLER 高，PSNR 直接归零。

#### 🧠 分类准确率（CIFAR-10, UMi, Nf=128）
| 方法 | SNR = 7 dB 时准确率 |
|------|---------------------|
| **Ours (Full)** | **72.3%** |
| DJSCC | 59.8% |
| Ours (w/o Pilot) | 72.1% |
| Ours (RZF) | 71.5% |

> 即使去掉导频引导，分类性能依然接近上限，说明高层语义更具抗噪性。

#### 📉 BLER 性能分析（Fig. 7）
- 在相同 Eb/No 下，**UMa > UMi > RMa** 的鲁棒性排序；
- 更多子载波（Nf=128）带来明显分集增益；
- 16QAM 比 4QAM 更脆弱，尤其在 RMa 场景；
- 不完美 CSI 导致约 **2–3 dB SNR 损失**。

---

### 与基线方法对比结果
| 维度 | 结果 |
|------|------|
| **PSNR 提升** | 相比 DJSCC 平均提升 **>3.5 dB**，在高带宽下达 5 dB |
| **鲁棒性** | 在低 SNR 区间（<-3dB）仍保持可用图像质量，而 SSCC 完全失效 |
| **跨场景一致性** | UMi/UMa/RMa 曲线高度重合，体现强泛化能力；SSCC 在 RMa 中性能骤降 |
| **带宽扩展性** | PSNR 随 Nf 增加持续上升（32→128），无平台期 |

---

### 消融实验结果
| 变体 | PSNR 影响 | 关键发现 |
|------|----------|----------|
| **Ours (RZF)** | 比 DJSCC ↑3.5dB，但比 Full ↓~1dB | 线性预编码不足以处理非线性干扰 |
| **Ours (w/o Pilot)** | 比 Full ↓~0.7dB（高带宽） | 导频引导对细节恢复至关重要 |
| **Ours (Full)** | 最优性能 | 所有模块协同作用，缺一不可 |

> 示例：在 3dB SNR 下，“cat” 和 “ship” 的纹理和边缘清晰度明显优于其他变体（见 Fig. 12）。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **非对称架构是面向边缘语义通信的关键**：将计算负担从 UE 转移到 BS 是可行且高效的。
2. ✅ **神经预编码优于传统线性方法**：不仅能规避数值不稳定问题，还能学习非线性空间-语义映射关系。
3. ✅ **显式利用导频信息可大幅提升解码精度**：导频引导注意力机制实现了隐式信道均衡，效果优于盲解码。
4. ✅ **语义通信天然具备更强的抗噪能力和渐进退化特性**：即使在极端低 SNR 下也能保留基本语义内容。
5. ✅ **端到端训练能自动平衡压缩与鲁棒性**：无需人工设定码率，在不同信道条件下自适应调节。

---

### 方法的局限性
- **依赖发射端已知 CSI 和 SNR**：虽然合理（TDD 系统可通过上行反馈获取），但在快速时变信道中可能滞后；
- **目前仅验证于小规模 MIMO（4×4）**：尚未拓展至大规模 MIMO 或毫米波场景；
- **仅支持静态图像传输**：未涉及视频、语音等动态模态；
- **训练开销较大**：尽管推理快，但端到端训练需要大量信道样本。

---

### 未来工作方向
1. **扩展至 Massive MIMO 场景**，研究波束成形与语义正交化的联合优化；
2. **动态语义资源分配**：根据不同用户的任务重要性（如自动驾驶 vs 普通浏览）差异化分配带宽；
3. **闭环反馈机制**：引入 ACK/NACK 或语义置信度反馈以实现自适应重传；
4. **多模态语义通信**：融合文本、语音、图像的统一语义表征与传输框架；
5. **真实硬件部署验证**：在 FPGA 或嵌入式平台上验证实时性与功耗表现。

---

> 🔗 **项目开源地址**：[https://github.com/Linkcy97/MUMIMOSC](https://github.com/Linkcy97/MUMIMOSC)

</details>

---

### 15. [Guided Collaboration in Heterogeneous LLM-Based Multi-Agent Systems via Entropy-Based Understanding Assessment and Experience Retrieval](https://arxiv.org/abs/2602.13639)

**Authors**: Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, Wanlei Zhou  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.13639v1  

#### Abstract
With recent breakthroughs in large language models (LLMs) for reasoning, planning, and complex task generation, artificial intelligence systems are transitioning from isolated single-agent architectures to multi-agent systems with collaborative intelligence. However, in heterogeneous multi-agent sys...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Guided Collaboration in Heterogeneous LLM-Based Multi-Agent Systems via Entropy-Based Understanding Assessment and Experience Retrieval

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文聚焦于**Heterogeneous Multi-Agent Systems (HMAS)** 中一个被忽视的关键瓶颈：**强弱模型之间的认知不匹配（cognitive mismatch）**。

尽管理论上强模型（如 GPT-4o）可以指导弱模型（如 Llama-3.2-1B）提升整体性能，但实验发现：
- 在某些任务中，**Strong-Weak 协作的表现甚至低于 Weak-Weak 组合**，即出现“负协同效应”（negative synergy effect）。
- 根本原因在于：弱代理在处理强代理生成的复杂推理链时面临**认知过载（cognitive overload）**，导致信息丢失、误解或简化，从而降低协作效率。

### ✅ 提出的新方法：Entropy-Based Adaptive Guidance Framework
为解决上述问题，作者提出了一种基于信息熵的理解评估与自适应引导框架，其核心创新包括：

#### （1）**多维熵理解评估机制（Understanding Assessment via Entropy）**
通过五维熵指标量化弱代理对任务的理解状态：
- `Hexpression`：词汇表达的离散程度
- `Huncertainty`：语义不确定性（如“可能”、“不确定”等词）
- `Hstructure`：响应结构化程度（是否含逻辑连接词）
- `Hcoherence`：推理连贯性
- `Hrelevance`：与问题的相关性

最终加权聚合为综合理解熵 $ H_u $，**熵越高表示理解越差，越低表示理解越好**。

#### （2）**三级自适应引导策略（Adaptive Guidance Strategies）**
根据理解熵动态调整引导强度：
- **Light Guidance**（$ H_u \leq T_1 $）：仅提供验证与修正建议
- **Moderate Guidance**（$ T_1 < H_u \leq T_2 $）：提供概念框架与执行建议
- **Intensive Guidance**（$ H_u > T_2 $）：进行细粒度步骤分解与示例引导

并引入**动态阈值调整机制**，随着弱代理能力提升逐步降低干预强度，避免过度干预。

#### （3）**基于 RAG 的经验保留机制（Experience Retention with RAG）**
- 成功协作后将问题、解法步骤、答案、难度、成功熵等打包存入经验库。
- 新任务启动前先检索相似历史案例，帮助弱代理快速建立上下文理解，降低初始熵值。
- 实现“即时引导 + 长期学习”的闭环。

### ✅ 相比现有方法的优势
| 方法 | 局限性 | 本文优势 |
|------|--------|----------|
| Chain-of-Thought (CoT) | 静态提示，无法适配不同能力代理 | 动态感知理解状态，按需调整引导强度 |
| Multi-Agent Debate / Reflection | 主要面向同构系统 | 明确建模异构系统中的认知差异 |
| 固定角色分工（如 Planner-Solver） | 忽视弱代理的认知负荷 | 引入熵驱动的自适应机制 |

> ✅ **核心优势**：首次系统识别并建模了 HMAS 中的“认知瓶颈”，并通过可量化的理解评估实现**个性化、渐进式知识传递**。

---

## 2. 核心实验方法和设置

### 📚 数据集
选用三个具有代表性的基准任务，覆盖多种推理类型：
- **GSM8K**：数学推理（symbolic reasoning），50题
- **MBPP**：Python 编程生成（program synthesis），50题
- **CVRP**：带容量约束的车辆路径规划（combinatorial optimization），20题

### 🧪 实验设置
#### 模型配置
- **Strong Agents**：GPT-4o, Claude-3.5-Sonnet
- **Weak Agents**：Qwen-2.5-0.5B, Llama-3.2-1B
- 构建多种 Strong-Strong、Strong-Weak、Weak-Weak 组合进行对比

#### 协作模式（按任务定制）
| 任务 | 协作模式 | 角色分工 |
|------|---------|--------|
| GSM8K | Framework-Solver | 强模型出框架，弱模型填步骤 |
| MBPP | Framework Provider + Implementer | 强模型设计伪代码，弱模型实现调试 |
| CVRP | Proposer-Validator | 弱模型提方案，强模型评估反馈 |

#### 评估指标
| 任务 | 主要指标 | 辅助指标 |
|------|--------|--------|
| GSM8K | Accuracy（正确率） | — |
| MBPP | Final Pass Rate（最终通过率） | Improvement Rate（错误转正确比例） |
| CVRP | Mean Accuracy（平均接近最优解的程度） | Solution Quality Score |

### 🔁 基线方法对比
1. **No-Guidance**：弱代理独立完成任务，无任何协作
2. **Chain-of-Thought (CoT)**：单代理使用 CoT 提示增强推理
3. **Strong-Strong / Weak-Weak Self-Collaboration**：作为性能上下界参考

---

## 3. 主要实验结果和性能指标

### 📊 性能对比（来自 Table II）

| 方法 | GSM8K (↑) | MBPP (↑) | CVRP (↑) | 平均增益 |
|------|----------|----------|----------|----------|
| No-Guidance | 45.0% | 36.5% | 65.0% | — |
| CoT | 51.0% | 43.5% | 66.6% | — |
| **Guided (Ours)** | **64.0%** | **43.5%** | **71.6%** | +13.0~19.0 pts |
| **Guided+RAG (Ours)** | **69.0%** | **46.0%** | **81.3%** | **+24.0 / +9.5 / +16.3 pts** |

> 💡 **关键发现**：
- 所有任务上均显著优于基线，尤其在 **GSM8K 和 CVRP** 上提升最大（最高达 **+24.0%**）
- RAG 的加入带来额外增益，说明**历史经验复用有效加速学习**

### 🔍 典型案例表现（Strong-Weak vs Weak-Weak）
| 配置 | GSM8K | MBPP | 发现 |
|------|-------|------|------|
| GPT-4o + Qwen-0.5B | 46.0% | 20.0% | **低于 Qwen 自协作（58.0%/26.0%）** → 负协同！ |
| Claude + Llama | 38.0% → **78.0% (with RAG)** | 54.0% → **54.0%** | 最大提升达 **+40%**，体现自适应引导价值 |

### 🔬 消融实验结果（Ablation Study）

#### （1）不同协作策略下的性能（Figure 5）
- 在三种协作模式（solve-review, code-check-improve, sequential）下，**Guided+RAG 均取得最佳性能**
- 特别是在 MBPP 的“code-check-improve”模式中，达到 **92.8%** 的惊人通过率

#### （2）RAG 机制的影响（Table III）
| 指标 | Guided | Guided+RAG | 变化 |
|------|--------|-----------|------|
| 平均 Accuracy | 71.6% | 81.3% | **+9.7%** |
| 平均 Quality Score | 6.4 | 7.1 | ↑ |
| 平均 Improvement Rate | 0.08 | 0.12 | ↑（收敛更快） |

> ⚠️ **有趣现象**：
- GPT 作为引导者时，RAG 使用频率低（~5%），但每次检索精准高效
- Claude 作为引导者时，RAG 使用频繁（~40%），反映其输出更抽象，需更多历史参照

#### （3）通信开销分析（Figure 7）
- 平均交互轮次控制在合理范围：
  - GSM8K：2.5–2.9 轮
  - MBPP：1.7–1.9 轮
  - CVRP：2.3–2.6 轮
- 表明该方法在**保持高性能的同时具备良好的通信效率**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **负协同效应真实存在**：
   - 在异构多智能体系统中，简单地让强模型指导弱模型**可能导致性能下降**。
   - 根本原因是**认知不匹配引发的信息损耗与认知过载**。

2. **性能瓶颈由最弱代理决定**：
   - 整体协作效果受限于弱代理的吸收能力，而非最强代理的能力上限。
   - “木桶原理”在 HMAS 中尤为明显。

3. **自适应引导至关重要**：
   - 固定强度的指导（如 CoT）无法应对动态变化的理解状态。
   - **基于熵的动态调节机制能有效匹配认知节奏**，实现“因材施教”。

4. **RAG 显著增强长期学习能力**：
   - 经验检索不仅提升当前任务成功率，还促进弱代理的持续成长。
   - 形成“实时引导 + 历史复用”的正向循环。

### ⚠️ 方法的局限性
1. **依赖高质量经验库初始化**：
   - 初始阶段若缺乏成功案例，RAG 效果受限。
2. **熵计算需要精细调参**：
   - 不同任务类型的权重矩阵 $ W_T $ 需人工设计或训练。
3. **未考虑多轮对抗或恶意行为**：
   - 当前框架假设合作是良性的，尚未扩展至博弈场景。

### 🔮 未来工作方向
1. **自动化熵权重学习**：利用强化学习自动优化各维度熵的权重。
2. **跨任务迁移经验**：构建通用经验编码器，支持跨领域知识迁移。
3. **引入情感与动机建模**：探索代理的“学习意愿”如何影响引导策略。
4. **部署到真实应用场景**：如智能制造、自动驾驶车队调度等复杂 HMAS 场景。

---

## 总结一句话
> 本文揭示了异构 LLM 多智能体系统中“强弱协作反不如弱弱协作”的反直觉现象，并提出首个基于**信息熵理解评估**与**RAG 经验复用**的自适应引导框架，在多个基准任务上实现了高达 **24% 的性能提升**，为构建高效、鲁棒的异构协作智能体提供了新范式。

</details>

---

### 16. [HyFunc: Accelerating LLM-based Function Calls for Agentic AI through Hybrid-Model Cascade and Dynamic Templating](https://arxiv.org/abs/2602.13665)

**Authors**: Weibin Liao, Jian-guang Lou, Haoyi Xiong  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.13665v1  

#### Abstract
While agentic AI systems rely on LLMs to translate user intent into structured function calls, this process is fraught with computational redundancy, leading to high inference latency that hinders real-time applications. This paper identifies and addresses three key redundancies: (1) the redundant p...

---

### 17. [On Theoretically-Driven LLM Agents for Multi-Dimensional Discourse Analysis](https://arxiv.org/abs/2602.13713)

**Authors**: Maciej Uberna, Micha{\l} Wawer, Jaros{\l}aw A. Chudziak, Marcin Koszowy  
**Category**: cs.CL  
**Published**: 2026-02-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.13713v1  

#### Abstract
Identifying the strategic uses of reformulation in discourse remains a key challenge for computational argumentation. While LLMs can detect surface-level similarity, they often fail to capture the pragmatic functions of rephrasing, such as its role within rhetorical discourse. This paper presents a ...

---

### 18. [LogitsCoder: Towards Efficient Chain-of-Thought Path Search via Logits Preference Decoding for Code Generation](https://arxiv.org/abs/2602.14054)

**Authors**: Jizheng Chen, Weiming Zhang, Xinyi Dai, Weiwen Liu, Kounianhua Du, Yasheng Wang, Ruiming Tang, Yong Yu, Weinan Zhang  
**Category**: cs.CL  
**Published**: 2026-02-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.14054v1  

#### Abstract
Code generation remains a challenging task that requires precise and structured reasoning. Existing Test Time Scaling (TTS) methods, including structured tree search, have made progress in exploring reasoning paths but still face two major challenges: (1) underthinking, where reasoning chains tend t...

---

### 19. [LLM-Guided Knowledge Distillation for Temporal Knowledge Graph Reasoning](https://arxiv.org/abs/2602.14428)

**Authors**: Wang Xing, Wei Song, Siyu Lin, Chen Wu, Man Wang  
**Category**: cs.CL  
**Published**: 2026-02-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.14428v1  

#### Abstract
Temporal knowledge graphs (TKGs) support reasoning over time-evolving facts, yet state-of-the-art models are often computationally heavy and costly to deploy. Existing compression and distillation techniques are largely designed for static graphs; directly applying them to temporal settings may over...

---

### 20. [Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/abs/2602.14516)

**Authors**: Wenhao He, Youhe Jiang, Penghao Zhao, Quanqing Xu, Eiko Yoneki, Bin Cui, Fangcheng Fu  
**Category**: cs.DC  
**Published**: 2026-02-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.14516v1  

#### Abstract
With the rapid evolution of Large Language Models (LLMs), multi-round workflows, such as autonomous agents and iterative retrieval, have become increasingly prevalent. However, this raises hurdles for serving LLMs under prefill-decode (PD) disaggregation, a widely adopted paradigm that separates the...

---

### 21. [RNM-TD3: N:M Semi-structured Sparse Reinforcement Learning From Scratch](https://arxiv.org/abs/2602.14578)

**Authors**: Isam Vrce, Andreas Kassler, G\"ok\c{c}e Aydos  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.14578v1  

#### Abstract
Sparsity is a well-studied technique for compressing deep neural networks (DNNs) without compromising performance. In deep reinforcement learning (DRL), neural networks with up to 5% of their original weights can still be trained with minimal performance loss compared to their dense counterparts. Ho...

---

### 22. [BotzoneBench: Scalable LLM Evaluation via Graded AI Anchors](https://arxiv.org/abs/2602.13214)

**Authors**: Lingfeng Li, Yunlong Lu, Yuefei Zhang, Jingyu Yao, Yixin Zhu, KeYuan Cheng, Yongyi Wang, Qirui Zheng, Xionghui Yang, Wenxin Li  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.13214v1  

#### Abstract
Large Language Models (LLMs) are increasingly deployed in interactive environments requiring strategic decision-making, yet systematic evaluation of these capabilities remains challenging. Existing benchmarks for LLMs primarily assess static reasoning through isolated tasks and fail to capture dynam...

---

### 23. [BEAGLE: Behavior-Enforced Agent for Grounded Learner Emulation](https://arxiv.org/abs/2602.13280)

**Authors**: Hanchen David Wang, Clayton Cohn, Zifan Xu, Siyuan Guo, Gautam Biswas, Meiyi Ma  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.13280v1  

#### Abstract
Simulating student learning behaviors in open-ended problem-solving environments holds potential for education research, from training adaptive tutoring systems to stress-testing pedagogical interventions. However, collecting authentic data is challenging due to privacy concerns and the high cost of...

---

### 24. [Detecting Jailbreak Attempts in Clinical Training LLMs Through Automated Linguistic Feature Extraction](https://arxiv.org/abs/2602.13321)

**Authors**: Tri Nguyen, Huy Hoang Bao Le, Lohith Srikanth Pentapalli, Laurah Turner, Kelly Cohen  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.13321v1  

#### Abstract
Detecting jailbreak attempts in clinical training large language models (LLMs) requires accurate modeling of linguistic deviations that signal unsafe or off-task user behavior. Prior work on the 2-Sigma clinical simulation platform showed that manually annotated linguistic features could support jai...

---

### 25. [Hippocampus: An Efficient and Scalable Memory Module for Agentic AI](https://arxiv.org/abs/2602.13594)

**Authors**: Yi Li, Lianjie Cao, Faraz Ahmed, Puneet Sharma, Bingzhe Li  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.13594v1  

#### Abstract
Agentic AI require persistent memory to store user-specific histories beyond the limited context window of LLMs. Existing memory systems use dense vector databases or knowledge-graph traversal (or hybrid), incurring high retrieval latency and poor storage scalability. We introduce Hippocampus, an ag...

---

### 26. [Prompt-Driven Low-Altitude Edge Intelligence: Modular Agents and Generative Reasoning](https://arxiv.org/abs/2602.14003)

**Authors**: Jiahao You, Ziye Jia, Chao Dong, Qihui Wu  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.14003v1  

#### Abstract
The large artificial intelligence models (LAMs) show strong capabilities in perception, reasoning, and multi-modal understanding, and can enable advanced capabilities in low-altitude edge intelligence. However, the deployment of LAMs at the edge remains constrained by some fundamental limitations. F...

---

### 27. [Learning User Interests via Reasoning and Distillation for Cross-Domain News Recommendation](https://arxiv.org/abs/2602.15005)

**Authors**: Mengdan Zhu, Yufan Zhao, Tao Di, Yulan Yan, Liang Zhao  
**Category**: cs.CL  
**Published**: 2026-02-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.15005v1  

#### Abstract
News recommendation plays a critical role in online news platforms by helping users discover relevant content. Cross-domain news recommendation further requires inferring user's underlying information needs from heterogeneous signals that often extend beyond direct news consumption. A key challenge ...

---

### 28. [Floe: Federated Specialization for Real-Time LLM-SLM Inference](https://arxiv.org/abs/2602.14302)

**Authors**: Chunlin Tian, Kahou Tam, Yebo Wu, Shuaihang Zhong, Li Li, Nicholas D. Lane, Chengzhong Xu  
**Category**: cs.DC  
**Published**: 2026-02-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.14302v1  

#### Abstract
Deploying large language models (LLMs) in real-time systems remains challenging due to their substantial computational demands and privacy concerns. We propose Floe, a hybrid federated learning framework designed for latency-sensitive, resource-constrained environments. Floe combines a cloud-based b...

---

### 29. [A Penalty Approach for Differentiation Through Black-Box Quadratic Programming Solvers](https://arxiv.org/abs/2602.14154)

**Authors**: Yuxuan Linghu, Zhiyuan Liu, Qi Deng  
**Category**: cs.LG  
**Published**: 2026-02-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.14154v1  

#### Abstract
Differentiating through the solution of a quadratic program (QP) is a central problem in differentiable optimization. Most existing approaches differentiate through the Karush--Kuhn--Tucker (KKT) system, but their computational cost and numerical robustness can degrade at scale. To address these lim...

---

### 30. [Attention in Constant Time: Vashista Sparse Attention for Long-Context Decoding with Exponential Guarantees](https://arxiv.org/abs/2602.13804)

**Authors**: Vashista Nobaub  
**Category**: cs.AI  
**Published**: 2026-02-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.13804v1  

#### Abstract
Large language models spend most of their inference cost on attention over long contexts, yet empirical behavior suggests that only a small subset of tokens meaningfully contributes to each query. We formalize this phenomenon by modeling attention as a projection onto the convex hull of key vectors ...

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
