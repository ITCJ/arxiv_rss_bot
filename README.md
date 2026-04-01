# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-01 07:08:56 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [A Framework for Hybrid Collective Inference in Distributed Sensor Networks](https://arxiv.org/abs/2603.28778)

**Authors**: Andrew Nash, Dirk Pesch, Krishnendu Guha  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.28778v1  

#### Abstract
With the ever-increasing range of applications of Internet in Things (IoT) and sensor networks, challenges are emerging in various categories of classification tasks. Applications such as vehicular networking, UAV swarm coordination and cyber-physical systems require global classification over distr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Framework for Hybrid Collective Inference in Distributed Sensor Networks

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**分布式传感器网络**（Distributed Sensor Networks）中的**集体推理**（Collective Inference）任务，解决在资源受限环境下（如通信带宽、计算能力和能耗限制）如何高效地实现全局分类或预测的问题。典型应用场景包括：
- UAV swarm coordination
- Vehicular networking
- Cyber-physical systems
- IoT 平台

这些场景要求在多个分布式节点之间进行协同决策，但受限于高成本的上行链路（如 NB-IoT）和有限的本地计算能力。

---

### 提出的新方法与新思路
作者提出了一种**混合式集体推理框架**（Hybrid Collective Inference Framework），其核心是将以下三种范式动态结合：
1. **Cloud/Edge Computing**：利用云端或边缘服务器进行集中式联合推理。
2. **Distributed P2P Inference**：通过设备间直接通信（如 LoRa）交换数据并达成共识。
3. **Split Computing / Early Exit**：允许设备基于局部置信度决定是否提前输出结果。

该框架的关键创新在于引入了一个**动态通信策略决策机制**，每个传感器独立执行如下三级判断逻辑（见图2）：

> **Policy Decision Flow**:
> 1. 若 $ P(Y_k|s_i) \geq \lambda $，则执行 **early-exit**（直接本地推断）
> 2. 否则，若从对等设备请求数据的**期望通信代价**低于上传至云，则发起 **peer data request**
> 3. 否则，将数据**offload 到 cloud/fog server**

其中 $\lambda$ 是用户定义的**置信阈值**，用于权衡准确率与通信开销。

---

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|--------|
| **通信效率** | 显著降低总体通信成本（理论和实验均显示可减少高达 50%-90% 的传输能量消耗） |
| **准确性** | 在多数参数配置下，准确率接近 centralized joint inference，远高于纯分布式独立分类器 |
| **灵活性与适应性** | 动态选择最优路径，能自适应不同数据分布、网络拓扑和通信成本环境 |
| **去中心化设计** | 不依赖中央协调器，具备良好的可扩展性和鲁棒性 |

> ✅ **首次实现了 cloud/edge + distributed P2P + split computing 的集成框架，并进行了系统建模与实证分析。**

---

## 2. 核心实验方法和设置

### 数据集与模拟环境
本研究未使用真实世界数据集，而是采用**合成高斯分布数据**来建模传感器观测值 $ S_i $，假设其服从以隐藏状态 $ Y $ 条件下的正态分布：
- 场景1: $ N=2 $, $ Y \sim \text{Binomial} $
- 场景2: $ N>2 $, $ Y \sim \text{Multinomial} $

具体形式为：
$$
S_i | Y=k \sim \mathcal{N}(k \cdot \delta_\mu, \sigma^2)
$$
通过调节 $\delta_\mu$ 控制类别间的分离程度，$\sigma=1.5$ 固定。

---

### 实验设置
- **通信成本模型**（基于 [27] 中 LoRaWAN vs NB-IoT 能耗对比）：
  - $ C_{S_iS_j} = 1\,\text{J} $ （LoRa 直连）
  - $ C_{S_iE} \in [1, 5]\,\text{J} $ （NB-IoT 上行，可变）
- **时间同步**：所有传感器按离散时间步同步操作
- **评估轮次**：每组参数重复采样 10,000 次进行统计平均

---

### 评估指标
| 指标 | 定义 |
|------|-----|
| **Accuracy (%)** | 正确预测全局状态 $ Y $ 的比例 |
| **Avg. Cost (J)** | 平均总通信能耗 |
| **Avg. # Direct Decisions** | 成功执行 early-exit 的次数 |
| **Avg. # Successful Requests** | 成功通过 peer 请求完成联合推断的次数 |
| **Cloud Baseline Cost** | 所有节点上传至 cloud 的总成本：$ N \times C_{S_iE} $ |
| **Independent Classifier Acc** | 各节点独立投票的准确率（无通信） |

---

### 基线方法对比
| 基线名称 | 描述 |
|--------|------|
| **Cloud/Fog Baseline** | 所有传感器上传数据到 cloud 进行 joint inference（最高准确率，最高成本） |
| **Independent Classifier** | 每个传感器仅基于本地数据做预测，最终取多数类（零通信成本，低准确率） |
| **Globally Optimal Baseline** | 理论上的最优通信调度方案（通过递归回溯求解），用于衡量冗余通信程度 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table VI 和 VII）

#### 场景：$ N=2, \delta_\mu=5, \lambda=0.95, C_{S_iE}=4\,\text{J} $
| 方法 | Accuracy (%) | Avg. Cost (J) |
|------|--------------|---------------|
| **Proposed Framework** | **97.6** | **0.75** |
| Cloud Baseline | 99.0 | 8.0 |
| Independent Classifier | 95.2 | 0.0 |

✅ **节省约 90.6% 的通信成本，同时保持接近中心化方法的准确率**

---

#### 场景：$ N=4, \delta_\mu=2, \lambda=0.85, C_{S_iE}=4\,\text{J} $
| 方法 | Accuracy (%) | Avg. Cost (J) |
|------|--------------|---------------|
| **Proposed Framework** | **80.7** | **4.71** |
| Cloud Baseline | 82.9 | 16.0 |
| Independent Classifier | 74.6 | 0.0 |

✅ **节省约 70.6% 的通信成本，准确率比独立分类器高 6.1%**

---

### 与基线方法的对比结果
- 在中等 $\delta_\mu$ 区间（1–5），提出的框架在**准确率-成本权衡**上显著优于两个极端基线。
- 当 $\delta_\mu$ 很大时（如 7），各传感器已能独立高置信推断 → 框架退化为独立分类器，成本趋近于 0。
- 当 $\delta_\mu$ 很小时（如 1），本地信息不足 → 更多依赖 cloud，但仍因 early-exit 和 peer-sharing 减少部分通信。

> 🔍 图7 显示：随着 $ C_{S_iE} $ 增加，本框架的成本增长速度明显慢于 cloud baseline，体现出更强的**抗高成本通信能力**。

---

### 消融实验与关键观察（Ablation Insights）
虽然没有显式的“消融实验”表格，但从参数敏感性分析中可得出以下结论：

| 参数变化 | 影响趋势 |
|--------|---------|
| ↑ $\lambda$（更高置信要求） | ➝ 更少 direct inference，更多 peer request 或 cloud offload；但在合适 $\delta_\mu$ 下仍能维持低成本高准确率 |
| ↑ $\delta_\mu$（更好类间分离） | ➝ 更多 early-exit，成本下降，准确率上升 |
| ↑ $ C_{S_iE} $（更贵的上行链路） | ➝ 框架更倾向于使用 peer-sharing，优势更加明显 |
| ↑ $ N $（更多传感器） | ➝ 成本线性增长但斜率小于 cloud baseline，具备良好可扩展性（见 Fig. 5） |
| ↑ $ |\text{dom}(Y)| $（更多类别） | ➝ 准确率下降，尤其当 $\delta_\mu$ 不足时；需更大分离度才能有效工作 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **混合推理框架可在保持高准确率的同时大幅降低通信开销**，特别适用于通信昂贵、电池受限的 LPWSN 场景。
2. ✅ **动态通信策略的有效性高度依赖于 $\delta_\mu$ 与 $ C_{S_iE}/C_{S_iS_j} $ 的相对关系**：
   - 存在一个“临界区域”（critical region），在此区域内框架最具价值。
3. ✅ **peer-to-peer data sharing 在中等不确定性条件下最为有用**，能够补充单一 early-exit 或 cloud-offload 的不足。
4. ✅ 即使在 $ N > 2 $ 多类分类任务中，该框架依然表现出稳健的性能提升。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖精确的通信代价估计** | 实际部署中 $ C_{S_iE}, C_{S_iS_j} $ 可能随时间波动，影响决策质量 |
| **当前仅验证于高斯数据** | 对非线性、复杂分布（如深度学习特征）尚未测试 |
| **未考虑延迟与带宽约束** | 目前只优化 energy/cost，未涉及 latency-sensitive 应用 |
| **缺乏真实硬件验证** | 所有实验基于仿真，尚未在真实 IoT 设备上实现 |

---

### 未来工作方向（Future Work）
1. **更复杂的分类器集成**  
   将框架扩展至支持 DNN、TinyML 模型，估算 $ P(Y|s), P(Y|s,s_j) $ 等概率输出。

2. **改进代价度量函数**  
   引入 computation cost、latency、bandwidth usage 等多维指标，构建综合 cost metric。

3. **真实应用验证**  
   在 real-world 场景（如空气质量监测、智能交通）中部署并评估性能。

4. **GMM 扩展与启发式优化**  
   探索使用 Gaussian Mixture Models 近似任意分布，并设计高效的 root-finding heuristic（如 IV-D.1 所提）。

5. **目标设备 $ T $ 的角色建模**  
   支持中间 aggregator 节点的存在，进一步分层优化通信路径（公式 10）。

---

> 📌 **总结一句话**：  
> 本文提出了首个融合 cloud/edge、split computing 与 distributed P2P 的**动态混合集体推理框架**，在理论和仿真实验中证明了其在**通信效率与推理精度之间的优越平衡能力**，为未来大规模、低功耗 IoT 系统提供了新的架构思路。

</details>

---

### 2. [Stochastic Dimension Implicit Functional Projections for Exact Integral Conservation in High-Dimensional PINNs](https://arxiv.org/abs/2603.29237)

**Authors**: Zhangyong Liang  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.29237v1  

#### Abstract
Enforcing exact macroscopic conservation laws, such as mass and energy, in neural partial differential equation (PDE) solvers is computationally challenging in high dimensions. Traditional discrete projections rely on deterministic quadrature that scales poorly and restricts mesh-free formulations l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Stochastic Dimension Implicit Functional Projections for Exact Integral Conservation in High-Dimensional PINNs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在高维物理系统建模中，**Physics-Informed Neural Networks (PINNs)** 虽然具有 mesh-free 和连续解表示的优点，但在强制执行**精确宏观守恒律**（如总质量、能量）方面面临三大挑战：

1. **积分维度灾难（Integral Curse of Dimensionality）**  
   显式离散投影依赖于确定性网格上的数值积分（如 Riemann 求和），其计算复杂度随空间维度指数增长，破坏了 PINNs 的 mesh-free 特性。

2. **非凸约束的优化不稳定性**  
   能量守恒通常对应二次型全局积分（如 $\int u^2 dx = C$），构成一个**非凸流形**（hyper-ellipsoid），通用隐式优化层（如 Inet）缺乏收敛保证。

3. **微分维度灾难（Differential Curse of Dimensionality）**  
   高阶空间导数的反向自动微分（reverse-mode AD）带来巨大的内存开销，限制了可扩展性。

---

### 提出的新方法：SDIFP 框架

作者提出 **Stochastic Dimension Implicit Functional Projection (SDIFP)** 框架，核心思想是将约束投影从“离散空间向量”提升到“连续函数空间”。

#### 主要创新点：

- ✅ **连续仿射函数投影（Continuous Affine Functional Projection）**  
  将网络输出 $ u_{\text{raw}}(x; \theta) $ 经过全局仿射变换：
  $$
  u(x) = \alpha(t) \cdot u_{\text{raw}}(x; \theta) + \beta(t)
  $$
  其中 $\alpha(t), \beta(t)$ 是动态标量参数，由守恒条件解析求解得到。

- ✅ **分离式蒙特卡洛积分（Detached MC Quadrature）**  
  使用大规模独立采样的 MC 点集 $ S_{\text{MC}} $（如 Sobol’ 序列）来估计全局积分，且该过程**脱离自动微分图**，避免内存爆炸。

- ✅ **闭式解析解处理非凸约束**  
  对线性和二次守恒量（质量与能量），推导出 $\alpha^*, \beta^*$ 的**闭式代数解**：
  $$
  \alpha^* = \sqrt{\frac{c_2 - c_1^2}{u_2 - u_1^2}}, \quad \beta^* = c_1 - \alpha^* u_1
  $$
  其中 $u_1, u_2$ 是 raw 输出的一阶、二阶矩估计。此解严格满足非凸能量约束。

- ✅ **双随机无偏梯度估计器（DS-UGE）**  
  引入双重解耦机制：
  - 空间 mini-batch 用于局部 PDE 残差计算；
  - 维度子采样（dimensional subsampling）用于高效计算高阶微分算子；
  - 两者正交解耦，实现内存复杂度从 $O(M \times N_c)$ 下降到 $O(N \times |Z|)$。

---

### 相比现有方法的优势

| 方面 | SDIFP | 现有方法（PINN-proj, PINN-SC, Inet 等） |
|------|-------|----------------------------------------|
| **是否 mesh-free** | ✅ 完全支持任意无结构点云 | ❌ 依赖均匀网格或固定体积元 |
| **能否处理非凸约束** | ✅ 支持二次能量守恒 | ❌ 多数仅适用于凸集或线性约束 |
| **内存效率** | ✅ 反向传播复杂度低 | ❌ 显式投影导致密集矩阵求逆，内存爆炸 |
| **守恒精度** | ✅ 数学上严格满足（machine epsilon 级别） | ❌ 软约束允许偏差；显式投影受采样方差影响 |
| **推理效率** | ✅ 推理阶段为标准 $O(1)$ 点级评估 | ⚠️ 投影层可能引入额外开销 |

---

## 2. 核心实验方法和设置

### 使用的 PDE 数据集（实验模型）
论文在四类典型守恒型 PDE 上进行验证，覆盖线性、非线性、色散与扩散行为：

| PDE 类型 | 方程形式 | 守恒量 |
|---------|--------|--------|
| **Advection Equation** | $\partial_t u + c \nabla u = 0$ | 线性积分 $c_1 = \int u dx$ |
| **Reaction-Diffusion Equation** | $\partial_t u = D \Delta u + ku$ | $c_1 = \int u dx$, $c_2 = \int u^2 dx$ |
| **Wave Equation** | $\partial_{tt} u = c^2 \Delta u$ | 动量 $c_1 = \int u dx$，能量近似 $c_2 = \int u^2 dx$ |
| **KdV Equation** | $\partial_t u + u \partial_x u + b \partial_{xxx} u = 0$ | 同时守恒 $c_1, c_2$，测试多约束能力 |

所有实验均拓展至 **1D–3D** 并最终扩展到 **高达 1000 维的空间域**。

---

### 实验设置

- **网络架构**：4 层 MLP，每层 128 单元
- **优化器**：Adam，初始学习率 $10^{-3}$，线性衰减至 0（共 10,000 epoch）
- **采样策略**：
  - 固定网格（Fixed Grid） vs. 随机配点（Random Collocation / MC）
  - SDIFP 使用 $M=10^5$ 个 detached MC 点估计积分
  - 残差 mini-batch 大小 $N=100$，维度采样子集 $|Z|=100$

---

### 评估指标

| 指标 | 描述 |
|------|------|
| **Error_u** | 预测解相对于真解的相对 $L^2$ 误差 |
| **Error_c1** | 线性守恒量绝对 $L^1$ 误差：$\left| c_{\text{pred}}(t) - c_{\text{true}}(t) \right|$ |
| **Error_c2** | 二次守恒量绝对 $L^1$ 误差 |
| **GPU Memory Usage** | 训练过程中峰值显存占用 |
| **Relative Compute Time** | 不同维度下约束执行时间的增长趋势 |

---

### 基线方法对比

| 方法 | 类型 | 是否硬约束 | 是否 mesh-free | 支持非凸？ |
|------|------|------------|---------------|-----------|
| **Vanilla PINN** | 软约束 | ❌ | ✅ | ❌ |
| **PINN-SC** | Soft Constraint | ❌ | ✅ | ❌ |
| **PINN-Proj** | 显式离散投影 | ✅ | ❌（需网格） | ⚠️ 近似有效 |
| **PINN-KTT** | KKT 投影（线性约束） | ✅ | ✅ | ❌（仅限线性） |
| **SDIFP (Ours)** | 连续函数投影 | ✅ | ✅ | ✅ |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 5.1 和 Figures）

#### ✅ 守恒误差显著降低（多个数量级）

| 方法 | 1D Wave Eq. (Error_c1) | 3D KdV Eq. (Error_c2) | 随机采样下表现 |
|------|------------------------|------------------------|----------------|
| PINN-SC | ~$10^{-1}$ | ~$10^{0}$ | 极差，严重漂移 |
| PINN-Proj | ~$10^{-4}$ | ~$10^{-1}$ | 在随机采样下崩溃 |
| **SDIFP** | **~$10^{-6}$** | **~$10^{-6}$** | **稳定保持 $10^{-5}-10^{-6}$** |

> 在所有维度和 PDE 类型中，SDIFP 的守恒误差比基线低 **3 到 7 个数量级**。

#### ✅ 高维可扩展性卓越

- **图 5.6(b)**：当维度 $D$ 增加到 1000 时，
  - PINN-SC / Proj / KTT 的相对守恒误差接近 **100%**
  - SDIFP 保持在 **$8.7 \times 10^{-3}\%$**（即约 $10^{-4}$），几乎平坦

- **图 5.8**：计算时间对比显示，
  - 固定网格方法在 $D \geq 7$ 时出现 OOM
  - SDIFP 在 $D=1000$ 仍可行，并实现 **最高达 126× 的加速**

#### ✅ 消融实验证明设计有效性（Section 5.3）

- 若使用传统固定 quadrature set 进行 shift（而非当前 batch），则会出现 batch-level 守恒残差（见 Fig 4.2a）
- SDIFP 使用 same-batch 积分估计 + detached large-M MC，实现了：
  - 正向传递中**每 batch 严格守恒**
  - 反向传播中**无过拟合采样噪声**
  - 解的正则性（smoothness）大幅提升，无高频伪振荡

---

## 4. 关键结论和发现

### 主要发现

1. 🔷 **连续函数空间投影优于离散向量投影**  
   将守恒约束嵌入到函数空间中的全局仿射变换，可将无限维非凸投影简化为二维代数求根问题，从根本上绕开了维度灾难。

2. 🔷 **detached MC + DS-UGE 实现高效无偏训练**  
   分离前向积分与反向梯度估计路径，在保证数学无偏性的前提下极大降低了内存消耗。

3. 🔷 **SDIFP 是真正 mesh-free 且 dimension-scalable 的框架**  
   在高达 1000 维的随机点云上仍能实现 machine-precision 级别的精确守恒，而其他方法在此场景下完全失效。

4. 🔷 **解决“双重维度灾难”**  
   成功克服了：
   - 积分维度灾难（via detached MC）
   - 微分维度灾难（via dimensional subsampling + DS-UGE）

---

### 方法的局限性

1. ❗ **不兼容 Dirichlet 边界条件**  
   当前仿射变换 $u = \alpha u_{\text{raw}} + \beta$ 中的常数项 $\beta(t)$ 会破坏边界值 $u|_{\partial X}=0$，除非特别构造修正函数（如乘以距离函数）。这是未来改进方向。

2. ❗ **初始化敏感性**  
   当网络初始输出接近常数场时，经验方差 $u_2 - u_1^2 \to 0$，可能导致 $\alpha^*$ 数值不稳定。实践中采用小扰动 $\max(\cdot, \epsilon)$ 缓解。

3. ❗ **目前仅支持全局积分型守恒律**  
   不直接适用于局部守恒律（如熵不等式）、通量匹配或多区域耦合约束。

---

### 未来工作方向

1. 🔄 扩展至 **局部硬约束**（如 entropy inequalities）和 **pointwise constraints**
2. 🧠 将 DS-UGE 整合进 **Neural Operator** 架构，学习无限维守恒映射
3. 🛠️ 设计兼容 **Dirichlet / Mixed BCs** 的增强 ansatz（例如 $u(x) = \phi(x)(\alpha u_{\text{raw}} + \beta)$，其中 $\phi|_{\partial X}=0$）
4. 🌐 探索在 **multi-physics 耦合系统** 中联合守恒多个物理量的应用

---

> **总结一句话**：  
> SDIFP 通过**连续函数空间投影 + 分离式蒙特卡洛积分 + 双随机梯度估计**，首次实现了在**任意高维、无结构点集上精确、稳定、可扩展地强制执行非凸守恒律**，为构建可信、长期稳定的物理神经网络提供了坚实基础。

</details>

---

### 3. [ARCS: Autoregressive Circuit Synthesis with Topology-Aware Graph Attention and Spec Conditioning](https://arxiv.org/abs/2603.29068)

**Authors**: Tushar Dhananjay Pathak  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.29068v1  

#### Abstract
I present ARCS, a system for amortized analog circuit generation that produces complete, SPICE-simulatable designs (topology and component values) in milliseconds rather than the minutes required by search-based methods. A hybrid pipeline combining two learned generators (a graph VAE and a flow-matc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ARCS: Autoregressive Circuit Synthesis with Topology-Aware Graph Attention and Spec Conditioning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统模拟电路设计依赖人工经验，耗时且效率低。现有机器学习方法存在以下三大局限：
- **仅生成拓扑**（如 AnalogGenie）：不输出元件值，需额外使用遗传算法（GA）进行 sizing，增加分钟级计算开销。
- **基于文本生成**（如 CircuitSynth）：在字符级别操作，缺乏对电路结构的语义理解，易产生语法错误。
- **无规格条件控制**（spec conditioning）：无法根据目标性能指标（如 $ V_{\text{out}} = 5V $）定向生成电路。

ARCS 针对上述问题，提出一个端到端可训练系统，实现**带规格条件的完整电路（拓扑 + 参数值）快速生成**。

---

### 🚀 提出的新方法与创新点

#### （1）**Group Relative Policy Optimization (GRPO)**  
- **问题识别**：标准 REINFORCE 在多拓扑训练中因不同拓扑奖励分布差异大（如电源转换器 reward 可达 8.0，放大器仅 ~4.0），导致“简单拓扑主导梯度更新”，困难拓扑被放弃。
- **解决方案**：引入 per-topology advantage normalization，即每个拓扑组内独立归一化优势函数：
  $$
  A^{(i)} = \frac{R_i - \mu_T}{\sigma_T}
  $$
  保证所有拓扑都能获得有效梯度信号。
- **效果**：仅用 500 步 RL 训练（相比 REINFORCE 的 5000 步），仿真有效性提升 **+9.6 pp**，训练成本降低 **10 倍**。

#### （2）**Grammar-Constrained Decoding**  
- 引入状态机驱动的 token masking，在自回归解码过程中强制遵守“组件-值交替”等结构规则。
- 实现 **100% 结构有效性**（structural validity），无需 RL 微调或后处理过滤。
- 支持三级约束：
  - `GRAMMAR`：基础语法（组件后必须接值）
  - `TOPOLOGY`：限制为当前拓扑所需的组件类型
  - `FULL`：进一步限制参数取值范围（如电感 ∈ [1pH, 10mH]）

#### （3）**Hybrid Multi-Source Ranking Pipeline**
- 融合两个互补生成器：
  - **VCG**（graph VAE）：图结构先验强，生成稳定
  - **CCFM**（Constrained Circuit Flow Matching）：非自回归并行生成，速度快
- 使用 SPICE 仿真结果作为排名依据，选择最优候选。
- 最终实现 **99.9% 仿真有效性**，仅需 **8 次 SPICE 评估**，比 GA 少 **40×**。

#### （4）**Topology-Aware Graph Transformer 架构**
- 注入电路图结构先验：
  - **Topology-aware attention bias**：通过预定义邻接矩阵增强连接元件间的注意力。
  - **Random-Walk Positional Encoding (RWPE)**：编码节点在网络中的结构性角色（如开关节点 vs 负载电阻）。
- 显著提升生成质量，尤其在复杂信号链路中表现优异。

---

### 🔍 相比现有方法的优势

| 方法 | 是否含元件值 | 规格条件 | 图结构偏置 | 结构有效性 | 速度 |
|------|----------------|-----------|--------------|---------------|--------|
| AnalogGenie | ❌ | ❌ | ⚠️（有限） | 93.2% | ~1s |
| CircuitSynth | ✅（字符级） | ❌ | ❌ | N/A | ~10s |
| ARCS（本文） | ✅ | ✅ | ✅ | **100%**（构造保证） | **~20ms** |

> ✅ **核心优势**：>1000× 于搜索方法的速度优势，支持快速原型设计与设计空间探索。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **自动化生成 62,000 个电路样本**，覆盖 **34 种拓扑模板**（32 用于主实验）。
- 包括三类电路：
  - **Tier 1**：7 种电源转换器（Buck, Boost, Cuk 等）
  - **Tier 2**：9 种信号电路（放大器、滤波器、振荡器）
  - **Tier 2b**：18 种扩展电路（BJT 放大器、稳压器、推挽等）
- 所有电路均通过 **ngspice** 进行瞬态/AC 仿真，并提取性能指标（效率、增益、带宽等）。

### 🧪 实验设置
- **输入格式**：`[START][TOPO][SEP][SPEC_VIN][VAL_x][SEP][COMP1][VAL_y]...[END]`
- **Tokenizer 设计**：706 个 token，涵盖组件类型、引脚、网络、数值（log-uniform 分桶，500 bins）
- **模型架构**：
  - GPT-style decoder，d_model=256，6 层，4 头
  - 支持三种变体：Baseline GPT、Two-Head Model、Graph Transformer

### 🎯 评估指标
| 指标 | 定义 |
|------|------|
| **Structural Validity** | 生成序列是否符合语法规范，能否解析为合法 netlist |
| **Sim. Success** | SPICE 是否收敛 |
| **Sim. Validity** | 仿真结果物理合理（如效率非负、输出正常） |
| **Reward** | 综合评分（满分 8.0），包含精度、效率、稳定性等 |
| **Wall Time** | 单次生成总耗时（含仿真） |

### 🆚 基线方法
| 基线 | 描述 |
|------|------|
| **Random Search (RS)** | 在参数空间均匀采样 200 组，选最佳 |
| **Genetic Algorithm (GA)** | BLX-α 交叉 + 高斯变异，种群 30，迭代 20 代（约 630 次评估） |
| **Supervised Learning Only** | 仅使用监督预训练，无 RL 微调 |
| **REINFORCE** | 标准策略梯度，全局 baseline |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table II & IV）

| 方法 | Params | Struct | SimValid | Reward | Wall Time |
|------|--------|--------|----------|--------|------------|
| **Random Search** | – | – | 81.2% | 7.28 | 58.8s |
| **Genetic Algorithm** | – | – | 80.0% | 7.48 | 271s |
| **ARCS (GT + GRPO, 500 steps)** | 6.8M | 96.6% | **53.1%** | 4.15 | ~30ms |
| **ARCS + Best-of-3** | – | – | **85.0%** | **5.48** | **97ms** |
| **Hybrid (VCG + CCFM)** | – | **100%** | **99.9%** | **6.43** | – |

> 💡 **亮点**：
> - Hybrid 方法以 **8 次 SPICE 评估** 达到接近 GA 的性能（6.43 vs 7.48），但节省 **40× 仿真次数**。
> - Best-of-3 推理策略使单模型达到 **85% 仿真有效性**，仍比随机搜索快 **600×**。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）**架构影响**（Table II）
- Two-Head Model → +10.3 pp SimValid over Baseline
- Graph Transformer → +5.3 pp，验证图结构先验的有效性

#### （2）**GRPO vs REINFORCE**
- GRPO（500 步）：SimValid **53.1%**
- REINFORCE（5000 步）：SimValid **43.5%**
- ➕ 提升 **+9.6 pp**，且训练时间减少 **10×**

#### （3）**Spec Conditioning 消融**
| 变体 | Struct | SimValid | Reward |
|------|--------|----------|--------|
| ARCS + RL | 98.1% | 52.5% | 3.49 |
| **No Spec Conditioning** | 58.8% | 38.8% | **2.60** |

> ❗ 规格条件至关重要，移除后性能大幅下降。

#### （4）**Topology Coverage Trade-off**
| 变体 | Topos | SimValid | Reward |
|------|-------|----------|--------|
| Tier 1 Only (7 topos) | 7 | 20.6% | **3.77** |
| Full Model (32 topos) | 32 | 45.4% | 3.49 |

> ✅ 虽然平均 reward 下降，但覆盖更多拓扑更具实用价值。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **GRPO 是多拓扑 RL 成功的关键**  
   - 解决了跨拓扑 reward distribution mismatch 问题，使得难拓扑也能有效学习。
   - 仅需少量 RL 步骤即可显著提升性能。

2. **结构正确性应由语法保障，而非 RL 学习**  
   - Grammar-constrained decoding 实现 **100% 结构有效性**，且适用于任意初始化模型。
   - 与 GRPO 形成正交分工：**结构由语法保证，语义由学习优化**。

3. **Amortized Inference 具有巨大工程价值**  
   - 虽然单次生成质量不及 GA，但其 **>1000× 速度优势** 支持：
     - 快速原型设计
     - 设计空间探索
     - 作为 GA 的 warm-start 初始化器

4. **Warm-Start 实验验证协同潜力**  
   - 使用 ARCS 初始化 GA：
     - 仅需 **49% 更少仿真次数**
     - 达到 **96.6% 冷启动 GA 性能**
     - 墙钟时间减少 **58%**

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **Per-design Quality 不及搜索方法** | Best-of-3 Reward 5.48 vs GA 7.48，仍有差距 |
| **数值分辨率受限** | Value tokenizer 使用 500 bins（~3.5% 相对误差），不适合高精度 RF 或精密模拟电路 |
| **拓扑固定，非发现式生成** | 需预先提供拓扑模板，不能像 AnalogGenie 一样生成全新结构 |
| **目前仅支持板级离散元件电路** | 尚未扩展至晶体管级 IC 设计 |

---

### 🔮 未来工作方向

1. **模型规模化**：扩展至 50–100M 参数，结合更大规模数据集。
2. **延长 GRPO 训练 + Curriculum Learning**：按拓扑难度逐步训练。
3. **End-to-End CCFM Fine-tuning**：解冻 VCG 编码器，联合优化 flow matching 与 SPICE 回馈。
4. **电气约束注入**：将 Kirchhoff 定律等物理规律纳入 constrained decoding。
5. **Multi-fidelity Training**：融合快速解析模型与 full SPICE 仿真进行 RL。

---

## 🏁 总结

ARCS 提出了首个真正意义上的 **amortized analog circuit generation** 框架，实现了从规格到完整 SPICE 网表的毫秒级生成。其核心思想是：

> **结构由语法保证（grammar-constrained decoding），语义由强化学习优化（GRPO），多样性由混合生成器提供（VCG + CCFM），最终通过极低成本的 SPICE 排名选出高质量设计。**

尽管尚未完全取代搜索方法，但 ARCS 开辟了一条全新的设计范式：**以极低成本生成大量“合理初稿”，再辅以轻量级优化**，极大提升了整体设计效率。

> 🔗 代码与数据已开源：[https://github.com/tusharpathaknyu/ARCS](https://github.com/tusharpathaknyu/ARCS)

</details>

---

### 4. [HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling](https://arxiv.org/abs/2603.29090)

**Authors**: Jaber Jaber, Osama Jaber  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.29090v1  

#### Abstract
World models that predict future states from video remain limited by flat latent representations that entangle objects, ignore causal structure, and collapse temporal dynamics into a single scale. We present HCLSM, a world model architecture that operates on three interconnected principles: object-c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **world model** 存在三大缺陷：
- **Flat latent representations**：将整个场景编码为单一向量，导致对象纠缠（object entanglement）。
- **缺乏多尺度时间建模**：无法同时处理连续物理运动、离散事件和长期目标等不同时间尺度的动态。
- **忽略因果结构**：模型无法识别“哪个物体影响了哪个”，限制了反事实推理与规划能力。

这些问题使得现有模型难以支持需要结构化知识的下游任务（如机器人规划、因果推断）。

---

### 🚀 提出的新方法与核心思想

作者提出 **HCLSM**（Hierarchical Causal Latent State Machines），一个统一的五层架构，整合以下三个关键维度：

#### （1）**Object-Centric 分解**
- 使用 **Slot Attention** 进行对象发现。
- 引入 **Spatial Broadcast Decoder (SBD)**，每个 slot 独立重建图像区域，强制其学习空间专属表示。
- 采用冻结的 **ViT 特征** 作为重建目标（借鉴 DINOSAUR），提升真实视频中的语义分解能力。

#### （2）**Hierarchical Temporal Dynamics**
构建三层时间层次结构：
- **Level 0: Selective SSM**  
  处理帧级连续物理动态（如物体滑动），参数共享但每对象独立追踪。
- **Level 1: Sparse Transformer**  
  只在检测到状态跃迁（event boundary）时激活，捕捉离散事件（如接触、碰撞）。
- **Level 2: Compressed Transformer**  
  对事件序列进行摘要压缩，用于高层目标推理。

#### （3）**Causal Structure Learning**
- 利用 **GNN** 建模对象间交互，边权重反映影响强度。
- 加入 **NOTEARS-style DAG 正则化**，鼓励学习无环因果图。

#### （4）**Two-Stage Training Protocol**
关键训练策略：
- **Stage 1（前40%训练步）**：仅优化 SBD 重建损失，迫使 slots 完成空间专业化（“先学会看，再学会预测”）。
- **Stage 2（后60%）**：开启完整 JEPA-style 动态预测损失，基于已分解的对象进行建模。

> 💡 类比视觉皮层发育：物体识别先于运动预测。

#### （5）工程优化亮点
- 开发 **自定义 Triton kernel** 实现 Selective SSM 的并行扫描，获得 **38× 速度提升**。
- 全 GPU 实现 Sinkhorn 匹配，消除 CPU-GPU 同步瓶颈。
- Chunked GNN 计算避免内存爆炸（对 $N>32$ slots）。

---

### 🔍 相比现有方法的优势

| 维度 | HCLSM | 其他方法（如 V-JEPA、DreamerV3、SlotFormer） |
|------|-------|---------------------------------------------|
| Object Decomposition | ✅ 显式 slot-based 分解 | ❌ 扁平 latent 或固定 slots |
| Hierarchical Time | ✅ 三层次动态建模（SSM + Sparse Tfm + Goal Tfm） | ❌ 单一尺度（通常为 Transformer） |
| Causal Structure | ✅ GNN 边权 + DAG 正则化 | ❌ 无显式因果建模 |
| 训练策略 | ✅ 两阶段训练防止 collapse | ❌ 所有损失同步优化易导致分布式表示 |
| 工程效率 | ✅ Triton kernel 加速 SSM | ❌ PyTorch 循环慢 |

> ✅ 表 1 显示 HCLSM 是首个同时具备 **object-centric、hierarchical dynamics 和 causal learning** 的可微分架构。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **PushT Task** 来自 **Open X-Embodiment Dataset**（通过 LeRobot 提供）
  - 包含 206 个 episodes，共 25,650 帧
  - 场景：机械臂推动 T 形块至目标位置
  - 输入：RGB 视频 + 动作指令（2D 末端执行器位移）
  - 视频片段长度：16 帧，分辨率 224×224

---

### ⚙️ 实验设置
- **模型规模**：HCLSM-Small（68M 参数）
- **硬件平台**：NVIDIA H100 80GB GPU
- **训练配置**：
  - Batch size: 4
  - Optimizer: AdamW
  - LR: 1.5e-4（cosine 调度，2K warmup）
  - Mixed Precision: bfloat16
  - 总训练步数：50K（约 6 小时/运行）
  - Stage 1: 前 20K 步（仅 SBD loss）
  - Stage 2: 后 30K 步（全 loss）

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Prediction Loss (MSE)** | 下一隐状态预测误差 |
| **SBD Loss** | Spatial Broadcast Decoder 重建损失 |
| **Tracking Loss** | slot 跨帧一致性 |
| **Diversity Loss ↓** | 衡量 slots 是否差异化（越低越好） |
| **Event Detection Accuracy** | 是否能准确识别状态跃迁时刻 |
| **Speed (sps)** | 每秒处理样本数 |
| **Visualizations** | alpha mask、slot 轨迹、event 概率曲线等定性分析 |

---

### 🔀 基线方法对比
- **HCLSM (no SBD)**：消融版本，不使用两阶段训练与 SBD
- **其他参考系统**（来自 Table 1）：
  - V-JEPA / V-JEPA2：flat latent + masking
  - DreamerV3：flat RNN-based world model
  - SlotFormer：object-centric + autoregressive Transformer
  - DINOSAUR：object-centric 但无动态建模

---

## 3. 主要实验结果和性能指标

### 📈 定量结果（见 Table 3）

| 方法 | Pred. Loss | Track. Loss | Diversity ↓ | SBD Loss | Total Loss | Speed (sps) |
|------|------------|-----------|------------|----------|------------|-------------|
| HCLSM (no SBD) | **0.002** | 0.001 | 0.154 | — | **0.100** | 2.3 |
| **HCLSM (two-stage)** | 0.008 | **0.016** | **0.132** | **0.008** | 0.262 | **2.9** |

> 🔍 关键观察：
> - “no SBD” 版本虽然预测误差更低，但这是以牺牲对象分解为代价的——slots 编码的是全局分布表示。
> - 两阶段训练显著提升了 **SBD loss 收敛** 和 **slot 多样性**，证明了结构优先的有效性。

---

### 🧪 消融实验与关键发现

#### （1）Two-Stage Training 至关重要
- 若从一开始就启用预测损失，梯度主导导致 slots 无法 specialization。
- Stage 1 强制空间重建，使每个 slot 学会“认领”特定区域（见 Figure 2）。

#### （2）Spatial Decomposition 成功涌现
- Figure 2 显示 alpha masks 出现明显空间划分，尽管尚未完全干净（32 slots 过多，每个物体被多个 slots 分割）。
- 无 SBD 的对照组显示均匀注意力，无空间结构。

#### （3）Event Detection 有效触发
- Figure 3 显示 event detector 在状态突变点（如接触瞬间）响应，平均每 16 帧检测 2–3 个事件。
- 使用多尺度差分特征 + 因果膨胀卷积实现高效检测。

#### （4）Latent Dynamics 结构清晰
- Figure 4 中 PCA 投影显示 slot 轨迹具有平滑演化路径，并在 event boundary 发生转向。
- 不同颜色代表不同 slots，轨迹差异表明它们捕捉了不同的动态模式。

#### （5）SSM Kernel 性能飞跃
| 配置 | Sequential (ms) | Triton (ms) | Speedup |
|------|------------------|------------|---------|
| Tiny | 6.22 | 0.16 | **39.3×** |
| Base | 69.64 | 1.83 | **38.0×** |

> Triton 实现将 SSM 扫描从正向传播的主要瓶颈降至仅占 **5% 时间**。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **结构必须先于预测**：通过两阶段训练，成功引导模型建立对象中心化的表示，避免陷入扁平分布式编码。
2. **多尺度动态建模可行**：SSM（细粒度）、Sparse Transformer（中观事件）、Goal Transformer（宏观目标）可协同工作。
3. **因果结构可通过 GNN 边权自然浮现**：无需强监督即可学习对象间的交互模式。
4. **真实机器人视频上可实现 unsupervised object discovery**：结合 SBD 与 ViT 特征目标，在 PushT 上实现了初步的空间分解。
5. **系统级工程优化至关重要**：Triton kernel、GPU-native tracking、chunked GNN 是支撑大规模训练的关键。

---

### ⚠️ 局限性
| 问题 | 描述 |
|------|------|
| **Slot 数量未自适应** | 所有 32 slots 始终存活，3-object 场景仍用 32 slots，造成冗余；存在头未学会“死亡”。 |
| **因果图学习失败** | 显式的 DAG 学习模块因稀疏正则化导致所有边权重归零；bf16 下联合训练不稳定。 |
| **事件检测泛化性未知** | 当前仅在 PushT 内部验证，跨任务或长期 episode 的表现未测试。 |
| **模型规模受限** | 仅成功训练 Small 版本（68M）；Base（262M）及以上出现 NaN，FSDP 多卡训练失败。 |
| **种子敏感性高** | ~40–60% 的训练 run 因 bf16 梯度溢出而崩溃（尤其在 Slot Attention GRU）。 |

---

### 🔮 未来工作方向
1. **引入 adaptive slot mechanism**  
   如 Adaptive Slot Attention 或 MetaSlot，动态调整 slot 数量。
   
2. **预训练 ViT 初始化**  
   使用 V-JEPA2 等大型视频模型初始化 backbone，提供更丰富的 patch features 作为 SBD 目标。

3. **扩展到复杂多物体场景**  
   在 ALOHA 双手机械臂任务等包含 5+ objects 的环境中验证 object-centric 的优势。

4. **闭环控制集成**  
   将 HCLSM 与 CEM/MPPI 规划器结合，实现基于世界模型的 robot control。

5. **解决数值稳定性问题**  
   探索 gradient scaling、fp32 关键路径计算等方式突破大模型训练障碍。

6. **因果评估基准建设**  
   构建具有 ground-truth 因果结构的仿真环境，使用干预实验（intervention-based metrics）评估 learned graph。

---

## 📦 开源信息
- **代码仓库**：[https://github.com/rightnow-ai/hclsm](https://github.com/rightnow-ai/hclsm)
- **完整发布内容**：
  - 8,478 行 Python 代码
  - 51 个模块
  - 171 个单元测试
  - 训练基础设施与评估套件

> 作者强调：“我们发布了完整的系统，包括瑕疵（warts and all）。”

</details>

---

### 5. [From Physics to Surrogate Intelligence: A Unified Electro-Thermo-Optimization Framework for TSV Networks](https://arxiv.org/abs/2603.29268)

**Authors**: Mohamed Gharib, Leonid Popryho, Inna Partin-Vaisband  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.29268v1  

#### Abstract
High-density through-substrate vias (TSVs) enable 2.5D/3D heterogeneous integration but introduce significant signal-integrity and thermal-reliability challenges due to electrical coupling, insertion loss, and self-heating. Conventional full-wave finite-element method (FEM) simulations provide high ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **高密度 Through-Substrate Vias (TSVs)** 在 2.5D/3D 异构集成中带来的**信号完整性**（如反射、插入损耗、串扰）和**热可靠性挑战**（如自加热、有效热导率 ETC 下降）。传统基于全波有限元法（FEM）的仿真工具（如 Ansys HFSS）虽然精度高，但在大规模设计空间探索时计算成本极高，难以支持快速迭代优化。

### 提出的新方法与创新思路
提出了一种**统一的电-热协同优化框架**，结合了物理建模、图神经网络（GNN）代理模型与多目标 Pareto 优化，具体包括以下四个核心贡献：

1. **物理信息驱动的解析电-热模型**  
   扩展了文献 [29] 中的金属-绝缘体-半导体（MIM）模型，支持多接地 TSV 结构和任意布局。该模型可高效计算宽带 S-参数 和各向异性等效热导率（$k_x, k_y, k_z$），为后续 ML 模型提供高质量训练数据。

2. **图神经网络代理模型 TSV-PhGNN**  
   设计了一个**物理信息图神经网络**（Physics-Informed GNN），将 TSV 阵列建模为图 $G=(V,E)$：
   - 节点表示 TSV，特征包含类型（信号/地）、几何尺寸（半径、间距、高度、氧化层厚度）和频率；
   - 边表示电磁耦合，特征为距离及其倒数；
   - 引入 **Feature-wise Linear Modulation (FiLM)** 机制以频率为条件调节网络行为；
   - 使用 **Graph Transformer** 进行注意力加权的消息传递；
   - 输出头强制对称化以满足互易性（$S_{ij}=S_{ji}$）；
   - 损失函数包含均方误差与被动性约束（passivity）。

3. **两阶段迁移学习策略（Sim-to-Sim Transfer Learning）**  
   - **预训练阶段**：在由解析模型生成的大规模数据集上训练 GNN（80,000 样本），实现快速且物理一致的初始化；
   - **微调阶段**：用少量 Ansys HFSS 高保真仿真数据（10,000 样本）进行微调，提升预测精度至接近 FEM 水平。

4. **多目标 Pareto 优化与自动化验证流程**  
   将 TSV-PhGNN 集成进优化框架，联合优化 TSV 布局与几何参数（半径、pitch、高度、oxide 厚度），构建反映 **反射系数、插入损耗、近端/远端串扰（NEXT/FEXT）、热导率** 权衡关系的 Pareto 前沿。最终设计方案通过 PyAEDT 自动导出并由 HFSS 和 Mechanical 完成签核验证。

### 相比现有方法的优势
| 维度 | 本文方法 | 现有方法（如 [22] GA + Q2D） |
|------|----------|-----------------------------|
| **可扩展性** | 支持任意大小阵列（测试达 22×22） | 通常限于小阵列（如 5×5） |
| **频率范围** | 全频段（最高 100 GHz） | 多数 < 20 GHz |
| **计算效率** | 千万级配置可在分钟内完成评估 | 每次仿真耗时数十分钟以上 |
| **准确性** | RFE < 2%，接近 HFSS | 依赖简化模型，误差较大 |
| **热电协同** | 显式建模 Joule 加热与温度反馈 | 多忽略自加热效应 |

---

## 2. 核心实验方法和设置

### 数据集
- **解析数据集**：使用提出的物理模型生成 **100,000 个样本**，涵盖 3×3 到 20×20 的 TSV 阵列，输入参数包括 TSV 半径（2–6 μm）、pitch（20–60 μm）、高度（60–100 μm）、oxide 厚度（0.5–3 μm）及频率（1–100 GHz）。
  - 训练集：80,000
  - 验证集：20,000
- **FEM 数据集**：使用 Ansys HFSS 对 **3×3 至 7×7** 的紧凑阵列进行全波仿真，生成 **10,000 个高保真样本**用于微调。

### 实验设置
- **硬件平台**：
  - ML 训练/推理：AMD Ryzen 97950X + NVIDIA RTX 4090（24GB VRAM）
  - FEM 仿真：Intel 14-core @ 2.6GHz + 32GB RAM
- **软件环境**：
  - ML 框架：PyTorch 2.8.0 + CUDA 12.9
  - 电路仿真：Synopsys PrimeSim HSPICE
  - 全波仿真：Ansys HFSS 2024 R2
- **评估指标**：
  - **Relative Frobenius Error (RFE)**：复数域下 S 参数矩阵的整体误差
    $$
    \text{RFE} = \frac{\|S_{\text{model}} - S_{\text{HFSS}}\|_F}{\|S_{\text{HFSS}}\|_F}
    $$
  - 插入损耗（Insertion Loss, $|S_{21}|$）
  - 反射系数（Return Loss, $|S_{11}|$）
  - 串扰（NEXT/FEXT）
  - 最大稳态温度（来自 Mechanical）
  - 单样本推理时间

### 基线方法对比
- **Analytical Model (HSPICE-based)**：作为低精度高速替代方案
- **Ansys HFSS**：作为黄金标准（ground truth）
- **State-of-the-Art ([22])**：基于遗传算法 + Ansys Q2D 的优化方法，仅适用于小阵列

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 数值 | 说明 |
|------|------|------|
| **解析模型 RFE** | 5.4% – 10.6% | 在 4×4 至 15×15 阵列上平均误差 |
| **TSV-PhGNN RFE** | **0.98% – 1.68%** | 微调后，在未见的大阵列（如 15×15）上仍保持低于 2% |
| **单样本推理时间** | **1.73 ms (15×15)** | 相比 HFSS 的 2,998 秒，加速约 **1.7×10⁶ 倍** |
| **设计空间探索速度** | 5.2×10⁶ 配置可在 **5 分钟内完成评估** | 若使用 HFSS，需数月时间 |
| **利用 D4 对称性后的搜索空间缩减** | 最多减少 **8×** | 使完整枚举成为可能 |

### 与基线方法的对比结果
#### （1）性能超越 SOTA [22]
- 在 **5×5 阵列含 12 个信号 TSV** 的基准问题中：
  - 本文最优布局最差受害者的总串扰为 **–45 dB**
  - 文献 [22] 报告为 –34 dB
  - **改善 11 dB（线性幅度降低 ~3.5×）**

- 在 **9 个信号 TSV** 场景中：
  - 本文最差受害者串扰为 **–48 dB**
  - [22] 报告为 –30 dB
  - **改善 18 dB（线性幅度降低 ~8×）**

#### （2）热分析加速显著
- 使用等效热导率（ETC）模型代替详细 TSV 几何建模：
  - **5×5 稀疏阵列**：提速 **35×**
  - **10×10 全填充阵列**：提速 **1300×**
- 温度预测误差：< 5%（最大偏差）

### 消融实验结果（隐含分析）
尽管文中未明确列出“消融实验”章节，但从设计选择中可推断关键组件的有效性：
- **FiLM 机制**：允许网络根据不同频率动态调整内部表征，适应从准静态到趋肤效应主导的行为转变；
- **Graph Transformer + Attention**：能自动学习屏蔽效应（如 ground TSV 阻挡耦合路径），优于固定权重聚合；
- **Symmetry Enforcement**：确保 $S_{ij}=S_{ji}$，提高物理一致性；
- **Homoscedastic Uncertainty Loss**：平衡不同任务（IL, RL, NEXT/FEXT）间的梯度，避免某一任务主导训练过程。

---

## 4. 关键结论和发现

### 主要发现
1. **物理引导的 ML 架构显著提升泛化能力**：TSV-PhGNN 能从仅 3×3–7×7 的 HFSS 数据中学习，并准确外推至 15×15 甚至 22×22 阵列，RFE 保持稳定（~1.7%），表明其具备强泛化性。
2. **百万倍加速使穷举式设计探索成为现实**：以往不可行的设计空间（如 5.2×10⁶ 种布局）现在可在几分钟内完成评估，极大提升了设计质量。
3. **多目标 Pareto 优化揭示内在权衡关系**：例如：
   - 最小化串扰的设计倾向于较小的 TSV 半径和较大的 pitch；
   - 最佳热性能设计则偏好更多铜体积（更大半径、更密排列），但会牺牲隔离度；
   - 插入损耗最优设计使用较薄 oxide 层；
   - 反射最小化要求更好的阻抗匹配结构。
4. **自动化签核验证闭环增强可信度**：所有 Pareto 最优解均通过 HFSS/Mechanical 验证，确认代理模型预测可靠。

### 方法的局限性
1. **VRAM 成为大规模阵列瓶颈**：由于图完全连接，边数随 $N^4$ 增长，当前模型受限于 GPU 显存，难以直接处理超大阵列（>22×22）。
2. **依赖解析模型的质量**：若解析模型在某些极端参数下失效，则会影响预训练数据的物理一致性。
3. **未考虑制造变异性和工艺偏差**：当前框架假设理想几何形状，尚未集成统计鲁棒性分析。

### 未来工作方向
1. **稀疏图表示与局部性剪枝**：引入基于距离的阈值或注意力掩码，降低内存复杂度至 $O(N^2)$ 或更低。
2. **支持非规则/异形阵列**：扩展图结构以处理非矩形排布或混合尺寸 TSV。
3. **集成不确定性量化（UQ）**：为代理模型输出提供置信区间，辅助风险敏感决策。
4. **向 3DIC 全系统级扩展**：将 TSV 子模块嵌入更大芯片-封装协同设计流程中。

---

> ✅ **总结一句话**：  
> 本文提出了一种融合物理建模、GNN 代理与 Pareto 优化的统一框架，实现了 TSV 网络的**百万倍加速电-热协同设计**，在保持接近 HFSS 精度的同时，使千万级配置的穷举优化成为可能，显著优于现有方法。

</details>

---

### 6. [Meteorology-Driven GPT4AP: A Multi-Task Forecasting LLM for Atmospheric Air Pollution in Data-Scarce Settings](https://arxiv.org/abs/2603.29974)

**Authors**: Prasanjit Dey, Soumyabrata Dev, Bianca Schoen-Phelan  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.29974v1  

#### Abstract
Accurate forecasting of air pollution is important for environmental monitoring and policy support, yet data-driven models often suffer from limited generalization in regions with sparse observations. This paper presents Meteorology-Driven GPT for Air Pollution (GPT4AP), a parameter-efficient multi-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Meteorology-Driven GPT4AP: A Multi-Task Forecasting LLM for Atmospheric Air Pollution in Data-Scarce Settings*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前空气污染预测模型严重依赖大规模、高质量的标注数据，在**数据稀缺（data-scarce）** 和**跨域迁移（cross-domain）** 场景下泛化能力差。这限制了其在监测基础设施薄弱地区（如新兴城市或偏远区域）的应用。

### 提出的新方法与新思路
本文提出了 **GPT4AP**（Meteorology-Driven GPT for Air Pollution），一种专为大气污染多任务预测设计的参数高效型大语言模型框架，具有以下核心创新：

- **基于预训练 GPT-2 的统一架构**：利用预训练 LLM 强大的表示学习能力，捕捉复杂的气象-污染物动态关系。
- **参数高效的微调机制 —— Gaussian rank-stabilized LoRA (rsLoRA)**：
  - 冻结 GPT-2 主干中的 self-attention 和 feed-forward 层，防止过拟合并降低计算开销。
  - 仅对轻量级模块进行训练：**可学习的位置编码（positional encoding）** 和 **输出预测头（prediction head）**。
  - 引入 **高斯初始化 + 秩相关缩放因子** $\beta_r = \sigma / \sqrt{r}$，稳定不同秩下的优化过程，提升训练稳定性与泛化性。
- 支持多种预测模式：在单一架构下实现 **few-shot**、**zero-shot cross-station transfer** 和 **long-term forecasting**。

### 相比现有方法的优势
| 维度 | GPT4AP 的优势 |
|------|----------------|
| **数据效率** | 在仅使用 10% 训练数据时仍表现优异，适合低资源场景 |
| **泛化能力** | 在未见过的站点上实现零样本迁移，显著优于传统模型 |
| **参数效率** | 仅需调整 <0.31% 的参数即可达到接近最优性能 |
| **部署友好性** | 极低的可训练参数量支持边缘设备部署和快速适应 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用来自中国六个空气质量监测站的真实世界数据（2013–2017 年）：
  - **Aoti Zhongxin (AZ)**, **Dongsi (DS)**, **Shunyicheng (SY)**, **Tiantan (TT)**, **Haidian Wanliu (HW)**, **Wanshou Xigong (WX)**
- 每个站点包含每小时测量的 **10 个变量**：
  - **6 种污染物**：PM2.5, PM10, SO₂, NO₂, CO, O₃
  - **4 项气象因素**：温度、气压、露点、降雨
- 数据划分：三年训练（~26,300 小时）、一年测试（8,760 小时）

### 实验设置与评估指标
#### 预测任务定义
- 输入历史窗口长度 $T=36$ 小时
- 预测未来 $H \in \{24, 36, 48, 60\}$ 小时的 PM2.5 浓度
- 多变量输入 → 单目标（PM2.5）多步预测

#### 三种实验设定
| 设定 | 描述 |
|------|------|
| **Few-shot** | 仅用 10% 的训练数据微调 rsLoRA 模块 |
| **Zero-shot** | 在一个源站点训练后，直接在目标站点测试，无任何微调 |
| **Long-term** | 使用全部训练数据进行完整训练 |

#### 评估指标
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

#### 基线方法对比
共比较六种先进的 time-series forecasting 模型：
- **DLinear**, **ETSformer**, **FiLM**, **Informer**, **Pyraformer**, **Transformer**

所有模型均在相同设置下复现以确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 实验设定 | GPT4AP (MSE/MAE) | 最优基线 (MSE/MAE) | 性能提升 |
|----------|------------------|--------------------|---------|
| **Few-shot** | **0.686 / 0.442** | DLinear (0.728 / 0.530) | MSE ↓7.2%, MAE ↓16.6% |
| **Zero-shot** | **0.529 / 0.403** | Pyraformer (0.565 / 0.557) | MSE ↓6.4%, MAE ↓27.6% |
| **Long-term** | 0.665 / 0.429 | Informer (0.598 / 0.416) | 略逊于专用模型，但仍具竞争力 |

> ✅ **GPT4AP 在数据受限场景下全面领先，在数据丰富场景中保持稳健。**

### 与基线方法的详细对比结果

#### Few-shot 结果亮点
- 在所有六个站点平均表现最佳：
  - Aotizhongxin: MSE 0.649 vs. DLinear 0.699 (**↓7.2%**)
  - Shunyi: MSE 0.634 vs. DLinear 0.675 (**↓6.1%**)
- 显著优于 ETSformer 和 FiLM，尤其在 MAE 上优势明显（反映更强的鲁棒性）
- 表明 **rsLoRA 能有效利用先验知识，在小样本下快速适应**

#### Zero-shot 跨站迁移结果
- 在五种转移路径中取得最佳性能：
  - DS → AZ: MSE 0.505 vs. Pyraformer 0.638 (**↓20.8%**)
  - TT → AZ: MSE 0.517 vs. DLinear 0.754 (**↓31.4%**)
- 平均 MSE 达到 **0.529**，远低于其他模型（最低为 Pyraformer 的 0.565）
- 说明 GPT4AP 成功捕获了**可迁移的气象-污染耦合规律**

#### Long-term 结果分析
- Informer 和 Pyraformer 表现最优（MSE ~0.598），因其专为长序列建模设计
- GPT4AP 平均 MAE 为 **0.429**，略高于最优值（0.416），但差距较小
- 表明该方法虽非专精于大数据长程依赖，但在统一框架下仍具备良好表现力

### 消融实验结果（Ablation Study: Rank Sensitivity of rsLoRA）

研究了不同 LoRA 秩 $r \in \{4,8,16,32,64\}$ 对性能的影响：

| 秩 $r$ | 可训练参数比例 | Few-shot MSE | Zero-shot MSE | Long-term MSE |
|--------|----------------|--------------|---------------|----------------|
| 4      | 0.038%         | 0.722        | 0.552         | 0.680          |
| 32     | **0.309%**     | **0.686**    | **0.529**     | **0.665**      |
| 64     | 0.617%         | 0.686        | 0.533         | 0.663          |

#### 关键发现：
- **性能在 $r=32$ 时趋于饱和**，继续增加秩几乎不带来收益
- $r=32$ 实现了 **99.9% 的峰值性能恢复率**，同时仅需 **0.309% 的可训练参数**
- **最佳性价比配置**：$r=32$ 是精度与效率之间的理想平衡点

---

## 4. 关键结论和发现

### 主要发现
1. **GPT4AP 是首个专为数据稀缺环境设计的气象驱动型 LLM 空气污染预测框架**，在 few-shot 和 zero-shot 场景下显著超越现有 SOTA 模型。
2. **rsLoRA 机制极大提升了参数效率与泛化能力**，冻结主干 + 轻量适配的设计特别适用于标签稀疏的实际应用场景。
3. **模型具备强跨站点迁移能力**，无需目标站点数据即可实现可靠预测，对发展中国家或新建监测网络极具价值。
4. **在 full-data 场景下虽略逊于专用时间序列模型，但差距有限且更具通用性和鲁棒性**。

### 方法的局限性
- 在充足数据条件下，尚未超越专门设计用于长序列建模的模型（如 Informer）。
- 当前评估局限于中国境内的六个城市站点，地理多样性有限。
- 仅预测单一污染物（PM2.5），未探索 multi-pollutant joint forecasting。
- 缺乏不确定性估计（uncertainty quantification）能力。

### 未来工作方向
1. **融合显式时间归纳偏置（temporal inductive biases）** 到 LLM 框架中，增强长期依赖建模能力。
2. 扩展至更广泛的地理区域和更多类型的污染物（如臭氧、NOx 等）。
3. 探索 **multi-task learning**，联合预测多种污染物及气象变量。
4. 引入 **probabilistic forecasting** 和 **uncertainty estimation** 模块，提高决策可靠性。
5. 推动 **edge deployment** 和实时系统集成，服务于智慧城市与公共健康预警。

---

> 🌍 **总体评价**：  
> GPT4AP 展示了 foundation model 在环境科学中的巨大潜力，特别是在解决“数据鸿沟”问题上的突破性进展。它不仅是一个高性能预测工具，更为构建**可扩展、可迁移、低成本的 AI-driven 环境监测系统**提供了新范式。

</details>

---

### 7. [CRAFT: Cost-aware Expert Replica Allocation with Fine-Grained Layerwise Estimations](https://arxiv.org/abs/2603.28768)

**Authors**: Adrian Zhao, Zhenkun Cai, Zhenyu Song, Lingfan Yu, Haozheng Fan, Jun Wu, Yida Wang, Nandita Vijaykumar  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.28768v1  

#### Abstract
Mixture-of-Experts (MoE) has recently emerged as the mainstream architecture for efficiently scaling large language models while maintaining near-constant computational cost. Expert parallelism distributes parameters by partitioning experts across devices, but this introduces token-level load imbala...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CRAFT: Cost-aware Expert Replica Allocation with Fine-Grained Layerwise Estimations 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模 **Mixture-of-Experts (MoE)** 模型部署中，**Expert Parallelism (EP)** 虽然能有效分布专家参数，但由于动态路由机制导致严重的 **token-level 负载不均衡（expert load imbalance）**。为缓解此问题，业界广泛采用 **expert replication** 技术复制“热点”专家以分散负载。

然而，现有主流方案如 **EPLB (Expert Parallelism Load Balancer)** 通常采用 **uniform replication**（每层每个 GPU 分配一个副本），存在以下问题：
- **过度复制（over-replication）**：许多副本带来的负载均衡收益极小；
- **高内存开销**：副本占用大量 GPU 显存，压缩 KV Cache 空间，降低并发能力；
- **资源浪费与吞吐下降**：显存争用导致系统整体吞吐反而可能低于不复制的情况。

### 🚀 提出的新方法与思路
本文提出 **CRAFT** —— 一种**成本感知、细粒度按层分配副本**的专家复制框架，其核心思想是：
- **按层估计复制收益（replication benefit）**：通过离线分析各 MoE 层的专家负载分布，量化不同副本数量下的负载均衡增益；
- **基于收益驱动的动态副本分配**：将副本优先分配给“高偏斜层”（high-skew layers），而对已平衡的层少复制甚至不复制；
- **建模为 Multiple-Choice Knapsack Problem (MCKP)**：在总显存预算约束下，使用动态规划求解最优的每层副本数，最大化整体均衡性增益。

### 🔍 相比现有方法的优势
| 维度 | EPLB (Uniform Replication) | CRAFT (Benefit-driven) |
|------|----------------------------|------------------------|
| 复制策略 | 固定每层每 GPU 一个副本 | 按层差异化分配，精细控制 |
| 内存效率 | 低，常造成 75%+ KV Cache 缩减 | 高，仅需 EPLB 的 ~1/7~1/8 副本数即可达到相近均衡性 |
| 吞吐表现 | 在小集群中可能劣于无复制（BASE） | 始终优于 BASE 和 EPLB |
| 可集成性 | 已集成于主流框架（如 SGLang） | 无缝替换 EPLB，无需训练或模型修改 |

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **模型**：
  - **DeepSeek-R1-671B**：58 个 MoE 层，256 个专家，top-8 路由；
  - **Kimi-K2-1000B**：60 个 MoE 层，384 个专家，top-8 路由。
- **数据集（workloads）**：
  - **FinePDFs**：包含德语（deu_Latn, E）、日语（jpn_Jpan, J）长文本；
  - **Lambada**（L）：常识推理任务；
  - **RedPajama-Data-1T** 中的 arXiv 子集（A）；
  - 所有输入截断至 4096 tokens，输出固定 256 tokens。

### ⚙️ 实验设置
- **硬件平台**：AWS EC2 `p4de.24xlarge` 实例，每台含 8× NVIDIA A100 (80GB)，NVLink 互联，跨节点使用 EFA；
- **软件栈**：CUDA 12.8 + NCCL 2.26.2，基于 **SGLang v0.4.8** 实现；
- **并行配置**：DP + TP-attention + EP，启用 mixed chunked prefill；
- **评估集群规模**：6、8、12 节点（即 48、64、96 GPU）。

### 📏 评估指标
- **Balancedness**：平均负载 / 最大负载，越高表示越均衡；
- **Goodput**：系统在请求延迟未显著上升前可维持的最大吞吐量（req/s）；
- **TTFT (Time-to-First-Token)**：首 token 延迟；
- **ITL (Inter-Token Latency)**：解码阶段 token 间延迟；
- **KV Cache Size**：反映可用显存容量，直接影响并发能力。

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **BASE** | 仅使用 expert placement，无复制 |
| **EPLB** | 默认最小 uniform replication（每层每 GPU 1 副本） |
| **CRAFT (CRA)** | 本文方法，支持手动或自动选择 replication factor R |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 吞吐提升（Goodput）
- 在 8 节点集群上，**CRAFT 平均提升 1.14× 吞吐**，最高达 **1.2×**；
  - DeepSeek-R1 上平均 **1.15×**；
  - Kimi-K2 上平均 **1.12×**，峰值 **1.17×**；
- 在高度偏斜数据集（如 E, J）上，CRAFT 提升可达 **1.42×**，远超 EPLB 的 1.24×；
- 在轻度偏斜数据集（如 L, A）上，EPLB 几乎无效（仅 1.02×），而 CRAFT 仍达 **1.14×**。

#### 💾 显存效率对比
- CRAFT 所需副本数仅为 EPLB 的 **1/7.25~1/7.5**；
- 在 6 节点集群中，EPLB 导致 **KV Cache 减少 75%**，严重限制并发；
- CRAFT 仅减少约 **6% KV Cache**，保留更高并发潜力。

#### 📊 负载均衡效果
- CRAFT 在远少于 EPLB 的副本数下，即可达到接近最优的 balancedness；
- 图 9 显示：增加副本后 balancedness 增益迅速饱和，**超过 16 副本后几乎无收益**；
- EPLB 因过度复制，在多个场景下 **goodput 低于 BASE**（如 KE6、KJ6）。

#### 🔁 可扩展性（Scalability）
- 随着集群规模从 6→8→12 节点，CRAFT 的 goodput 分别提升 **1.65× 和 1.6×**；
- 表明 CRAFT 在大集群中仍能高效扩展，而 BASE 和 EPLB 扩展性差。

#### ⏱️ 延迟表现
- **TTFT**：CRAFT 相比 BASE 平均降低 **29%**（最高 58%），与 EPLB 相当；
- **ITL**：所有方法在解码阶段差异不大，说明负载均衡主要影响 prefill 阶段。

### 🔬 消融实验（Ablation Study）
- **不同 R 值的影响**（图 11）：
  - R 过小 → 副本不足，无法缓解负载不均；
  - R 过大 → 显存开销过大，KV Cache 缩减抵消收益；
  - **R=8 是多数配置下的最优值**，实现最佳 trade-off；
- **自动 R 选择机制有效**：CRAFT 可根据收益曲线自动选择高效 R，避免人工调参。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **复制收益具有显著的层间差异性**：
   - 高偏斜层（如 top expert 占 >27× 平均负载）从复制中获益巨大；
   - 低偏斜层几乎不受影响，复制纯属浪费。
2. **复制收益随副本数呈亚线性增长（sublinear scaling）**：
   - 初始少量副本即可大幅改善均衡性；
   - 超过一定阈值后（如 16 副本），增益趋于饱和。
3. **现有 uniform replication 极其低效**：
   - EPLB 的默认策略在小集群中可能导致性能倒退；
   - “一刀切”的复制策略不适合多样化的 MoE 层结构。

### ⚠️ 方法的局限性
- **依赖离线负载分析**：需要预先收集 workload 的 expert load distribution；
- **静态分配**：当前为静态配置，未支持运行时动态 rebalancing；
- **假设负载分布稳定**：若 workload 分布剧烈变化，需重新 profiling；
- **未考虑通信开销细节**：如 replica 放置对 all-to-all 带宽的影响未深入建模。

### 🔮 未来工作方向
- **在线自适应复制（Online Adaptive Replication）**：结合 runtime 监控，动态调整副本；
- **与 expert placement 更深度协同**：将 CRAFT 与 topology-aware 或 affinity-based placement 结合；
- **支持多目标优化**：联合优化 latency、energy、cost 等指标；
- **扩展至其他 MoE 架构**：如 hierarchical MoE、grouped-query MoE 等。

---

## 总结
**CRAFT** 是首个系统性研究 **expert replication 成本效益** 的工作，揭示了现有 uniform replication 的低效本质，并提出了一种**基于细粒度层感知的收益驱动副本分配框架**。其实验充分证明：
> **更聪明地复制，比盲目复制更能提升 MoE 推理吞吐。**

该方法无需改动模型或训练流程，可直接集成进主流 serving 框架（如 SGLang），具备强实用性和推广价值，为大规模 MoE 模型的高效部署提供了新范式。

</details>

---

### 8. [Federated Inference for Heterogeneous LLM Communication and Collaboration](https://arxiv.org/abs/2603.28772)

**Authors**: Zihan Chen, Zeshen Li, Howard H. Yang, Tony Q. S. Quek, Jihong Park  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.28772v1  

#### Abstract
Given the limited performance and efficiency of on-device Large Language Models (LLMs), the collaborations between multiple LLMs enable desirable performance enhancements, in which data, tokens, and model weights could be shared across LLMs. This process is constrained by task-oriented QoS demands, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Federated Inference for Heterogeneous LLM Communication and Collaboration

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 on-device LLMs 面临以下挑战：
- **推理性能受限**：本地部署的小型 LLMs 在准确性和速度上远不如云端大型模型。
- **通信开销高**：传统的文本级通信（Text-to-Text, T2T）在多模型协作中引入显著延迟，尤其是由于每次交互都需要重建 Key-Value (KV) Cache 导致的 pre-fill 延迟。
- **隐私泄露风险**：原始输入/输出 tokens 可能暴露用户敏感信息。
- **系统异构性障碍**：不同架构、参数规模、tokenization 方式的 LLMs 难以直接共享知识。

现有协作机制（如联邦学习、知识蒸馏）通常依赖于梯度或权重传输，不适用于低延迟、保护隐私的推理场景。

---

### 🚀 提出的新方法：FedRefine
作者提出了一种全新的 **federated inference 框架——FedRefine**，其核心是基于 **Cache-to-Cache (C2C)** 的双向 KV Cache 共享机制，实现异构 LLMs 之间的高效协作推理。

#### 核心创新点：
1. **LLM-native Communication via KV Cache**
   - 不再传递可读的 tokens，而是直接交换模型内部的 KV Cache。
   - KV Cache 包含更丰富的上下文语义信息，避免了 T2T 中的信息损失和重复计算。

2. **Bidirectional Co-C2C（协同 C2C）**
   - 引入双向 fuser 网络（如 Fuser₁₂ 和 Fuser₂₁），允许两个 LLM 互为 receiver 和 transmitter，进行相互 refine。
   - 支持公平、激励兼容的协作范式，小模型也能帮助大模型提升性能。

3. **Privacy-Preserving via Input Rephrasing**
   - 输入问题由接收方模型自动 rephrase（语义改写），确保原始意图不被发送方知晓。
   - 所有私有 tokens 保留在本地，仅共享处理后的 KV Cache。

4. **Model-Agnostic Collaboration**
   - 利用预训练的 **C2C fuser**（如 MLP 投影网络）桥接不同架构的 LLMs，无需模型同构。
   - 支持灵活扩展至 N 个异构 LLM 协作。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（T2T / FL） | FedRefine |
|------|------------------------|-----------|
| **通信效率** | 高延迟（需重建 KV Cache） | 跳过 pre-fill，大幅降低延迟 |
| **隐私保护** | 明文 token 传输易泄密 | 本地 rephrase + KV Cache 加密语义 |
| **异构支持** | 多要求同构模型 | 支持任意架构组合（Qwen, Llama 等） |
| **更新成本** | 权重/梯度同步开销大 | 无需传输模型参数，仅传 KV Cache |
| **协作模式** | 单向辅助（大模型指导小模型） | 双向互惠（小模型也可 refine 大模型） |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据**：OpenHermes2.5 的前 50,000 个样本，用于训练 C2C fuser。
- **评估数据**：OpenBookQA，用于测试问答任务的推理准确性。

---

### ⚙️ 实验设置
- **接收模型（Receiver）**：Qwen3-0.6B
- **发送模型（Transmitters）**：
  - Qwen2.5-0.5B
  - Qwen2.5-0.5B-code
  - Qwen2.5-1.5B
  - Llama-3.2-1B
- **Fuser 架构**：三层 MLP，逐层对齐并投影 KV Cache。
- **协作方式**：
  - 单独模型推理（Baseline）
  - 多模型联合 FedRefine 推理（KV 通信 vs Token 通信）
  - 是否启用 rephrase 进行隐私保护

---

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy** | 在 OpenBookQA 上的任务正确率 |
| **Latency** | 平均推理时间（秒） |
| **Communication Load** | 每 token 传输的数据量（KB/byte） |
| **Privacy Protection** | 是否使用 rephrased input |

---

### 🔁 基线方法对比
| 方法 | 描述 |
|------|------|
| **Standalone Inference** | 接收模型独立完成推理 |
| **T2T Collaboration** | 多模型通过自然语言 token 交流协作 |
| **Non-private C2C** | 直接共享原始 KV Cache（无 rephrase） |
| **Private C2C (FedRefine)** | 使用 rephrased input + KV Cache 共享 |
| **Full Participation Setting** | 所有四个 transmitter 模型参与协作 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Figure 3）

#### (a) 联邦推理准确性（Fig. 3a）
- **单独模型 baseline**：约 48% 准确率。
- **FedRefine (Non-private KV)**：
  - 使用全部 4 个 sharer 模型时，准确率达到 **~69.2%**，**提升 21.2%**。
- **FedRefine (Private KV, rephrased)**：
  - 准确率为 **~67%**，相比非隐私版本仅下降 **3%**，说明隐私保护代价极小。
- **T2T 方法**：
  - 同样四模型参与下，准确率约为 **54%**，比 C2C 低约 **15%**。

> ✅ 结论：C2C 显著优于 T2T；隐私保护策略几乎不影响性能。

---

#### (b) 点对点通信性能（Fig. 3b）
- 性能增益与 sharer 模型自身能力正相关。
- 更强的 transmitter（如 Qwen2.5-1.5B）带来更大收益。
- 表明 FedRefine 能有效利用“互补优势”模型。

---

#### (c) 推理延迟（Fig. 3c）
- **T2T 方法**：延迟最高，因每次 token 交互都需重新 pre-fill。
- **C2C 方法**：
  - 尽管 rephrase 增加少量前端延迟，
  - **总延迟仍显著低于 T2T**（节省 ~40–60% 时间）。
- 原因：跳过了重复的 context 编码过程。

---

#### 📦 通信负载对比
| 方法 | 每 token 传输量 |
|------|----------------|
| **T2T** | 16 bytes |
| **C2C (KV Cache)** | 88 KB |

> ❗ 注意：C2C 通信开销远高于 T2T（高出约 5500 倍），这是主要 trade-off。

---

#### ✅ 消融实验发现（隐含分析）
- **是否启用 rephrase**：对 accuracy 影响小（仅降 3%），但极大增强 privacy。
- **参与模型数量**：accuracy 随 sharer 数量增加而单调上升，验证协作有效性。
- **双向 vs 单向 C2C**：bidirectional Co-C2C 支持更均衡的知识流动，提升整体鲁棒性。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **KV Cache 是高效的协作媒介**：
   - 相比 token 通信，C2C 显著减少延迟、提高 accuracy。
   - KV Cache 是 LLM-native 的知识表示，适合跨模型迁移。

2. **FedRefine 实现高效异构协作**：
   - 无需模型同构，通过 fuser 实现跨架构兼容。
   - 支持双向 refine，打破“只能大模型帮小模型”的固有假设。

3. **隐私与性能可兼得**：
   - 输入 rephrase 有效防止意图泄露。
   - 性能损失极小（<3%），实用性高。

4. **通信资源是瓶颈**：
   - C2C 的 high bandwidth demand 是当前最大限制。
   - 未来需结合压缩、稀疏化等技术优化传输效率。

---

### ⚠️ 局限性
| 问题 | 说明 |
|------|------|
| **高通信开销** | KV Cache 数据体积大（单 token 达 88KB），不适合带宽受限环境。 |
| **Fuser 训练成本** | 每对模型需独立训练 fuser，扩展性受制于 N² 对组合。 |
| **依赖高质量 rephrase** | 若 rephrase 不够忠实原意，可能导致语义偏移。 |
| **未考虑动态网络状态** | 当前框架未根据实时带宽/延迟自适应切换 C2C/T2T。 |

---

### 🔮 未来研究方向（作者建议）
1. **Iterative Local Refinement**  
   设计多轮迭代 refine 机制，结合 cache/token 动态通信。

2. **Continuous Global Federation Iterations**  
   类似联邦学习的多轮全局更新，在推理阶段持续优化整个系统。

3. **Cache Communication for Multi-modal LLMs**  
   扩展到视觉-语言等 multi-modal 场景，设计统一的 cross-modal cache fuser。

4. **Prompt Engineering for Federated Inference**  
   开发面向 cache communication 的 prompt 策略，协调角色分配与协作流程。

5. **Opportunistic Communication Switching**  
   根据 QoS 需求和网络状况，动态选择使用 C2C 或 T2T。

---

## 总结
FedRefine 提出了一种革命性的 **federated inference 范式**，将 LLM 协作从“文本对话”升级为“内存级协同”，实现了：
- ✅ 更快推理（低延迟）
- ✅ 更准结果（高性能）
- ✅ 更强隐私（rephrase + KV 本地化）
- ✅ 更广兼容（异构模型协作）

尽管面临通信开销大的挑战，该工作为下一代分布式智能系统的 **可持续 foundation model collaboration** 提供了重要思路。

</details>

---

### 9. [1.5 Million Messages Per Second on 3 Machines: Benchmarking and Latency Optimization of Apache Pulsar at Enterprise Scale](https://arxiv.org/abs/2603.29113)

**Authors**: Muhamed Ramees Cheriya Mukkolakkal  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.29113v1  

#### Abstract
This paper presents two independent contributions for Apache Pulsar practitioners. First, we validate 1,499,947 msg/s at 3.88 ms median publish latency on just three bare-metal Kubernetes nodes running Pulsar 4.0.8 with Java 21 and ZGC Generational garbage collection, and project a hardware-driven p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*1.5 Million Messages Per Second on 3 Machines: Benchmarking and Latency Optimization of Apache Pulsar at Enterprise Scale*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
该论文针对 **Apache Pulsar** 在企业级生产环境中出现的高发布延迟（median publish latency 达 13–18 ms）和间歇性延迟尖峰（spikes 高达 213 ms）的问题，进行了系统性的根因分析与优化。尽管集群负载较低（仅 700–9,000 msg/s），但性能表现不佳，影响实时事件流处理效率。

目标是：
- 定位并消除导致高延迟的根本原因；
- 在有限硬件资源下验证大规模吞吐能力；
- 构建可线性扩展的高性能架构。

---

### 🔧 提出的新方法或新思路

#### （1）基于 **JFR（Java Flight Recorder）** 的生产环境无损诊断
首次在 **不中断线上流量** 的情况下，对运行中的 Bookie 节点进行 JFR profiling，精准识别 JVM 和 OS 层面的性能瓶颈。

#### （2）发现一个此前未被记录的 **Linux 内核 page cache writeback 与 BookKeeper ForceWriteThread 的交互问题**
即使 journal 和 ledger 存储在物理独立的 NVMe 设备上，由于共享内核 block layer、bio pool 和 IRQ 处理机制，在 SyncThread 触发写缓存刷新时，会导致 `fdatasync` 延迟从 <1 ms 恶化至 15–22 ms。这是一个 **novel finding**，具有广泛适用性。

#### （3）提出一套完整的端到端优化方案
结合以下多个维度进行联合调优：
- JVM GC 替换为 **ZGC Generational**
- 使用专用 **NVMe journal disk**
- 调整 **write cache flush interval**
- 应用 **OS kernel 参数调优**（如 `dirty_ratio=2`）
- 利用 Pulsar 原生 **key-based partition routing** 实现横向扩展

---

### 🆚 相比现有方法的优势

| 方面 | 本文方法优势 |
|------|--------------|
| **诊断方式** | 不依赖压测或模拟，直接在生产节点上通过 JFR 实现“热诊断” |
| **优化深度** | 同时覆盖应用层（Pulsar/BookKeeper）、JVM 层（GC）、OS 层（kernel writeback） |
| **扩展模型** | 利用 Pulsar 原生分区路由实现 **无需外部负载均衡器** 的水平扩展 |
| **成本效益** | 多数优化无需新增机器，仅需合理配置已有资源即可提升数十倍性能 |

---

## 2. 核心实验方法和设置

### 📦 数据集与工作负载
- 并非传统意义上的“数据集”，而是采用 **可控的消息生成器** 发送固定大小消息（默认 1KB）。
- 工作负载模式：持续发布（producer-heavy），测试发布延迟和端到端吞吐。

---

### ⚙️ 实验设置

#### 硬件配置
- **3 台裸金属服务器（bare-metal nodes）**
  - 每台部署 2 个 Bookie + 2 个 Broker（共 6 bookies, 6 brokers）
  - 使用 Kubernetes `topologySpreadConstraints` 控制分布
  - **专用 NVMe 日志盘（journal drive）**：fsync 延迟低至 0.02 ms
  - **网络**：10 Gbps NIC（后续升级预测至 25 Gbps）

#### 软件栈
- Apache Pulsar 4.0.8 / 4.4.0
- Java 21 + **ZGC Generational** (`-XX:+UseZGC -XX:+ZGenerational`)
- BookKeeper journal 配置：E=3, Qw=2, Qa=2
- Topic 分区数：128

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Throughput** | 每秒消息数（msg/s） |
| **Publish Latency P50/P95/P99/P99.9** | 发布操作的延迟百分位 |
| **End-to-end Latency** | 从发布到消费的完整链路延迟 |
| **Failure Count** | 是否有消息失败或超时 |
| **Resource Headroom** | CPU、内存等资源利用率余量 |

---

### 🔁 基线方法对比

| 基线配置 | 描述 |
|--------|------|
| **原始生产环境（Baseline）** | G1GC + 32GB heap + SSD journal（5.1 ms fsync）+ 默认 flush interval（60s） |
| **不同 GC 对比** | G1GC vs ZGC（仅 bookie）vs ZGC（bookie & broker） |
| **flushInterval 对比** | 60s（默认）vs 30s vs 15s |
| **存储介质对比** | SSD vs 新 NVMe vs 磨损 NVMe（119% lifetime） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Benchmark 结果）

> 在 **3 台裸金属节点** 上连续运行 10 分钟，结果稳定且零故障：

| 指标 | 数值 |
|------|------|
| **总吞吐量（publish）** | **1,499,947 msg/s**（≈1.5 M msg/s） |
| **中位发布延迟（P50）** | **3.88 ms** |
| **P99 发布延迟** | 6.5 ms |
| **P99.9 发布延迟** | 8 ms |
| **端到端消费吞吐** | 1,065,780 msg/s |
| **端到端 P50 延迟** | 14.5 ms |
| **失败次数** | 0 |
| **网络带宽使用** | 8.4 Gbps（占 10Gbps 的 84%） |
| **计算资源余量** | CPU 和内存保留 **65–82% headroom** |

> 图表显示每 broker 发布速率稳定，P50 延迟维持在 1.44–1.49 ms，证明系统高度均衡。

---

### 🔍 与基线方法的对比结果

| 组件/阶段 | 优化前 | 优化后 | 改善幅度 |
|---------|-------|-------|----------|
| Journal fsync 延迟 | 5.1 ms（SSD） | 0.02 ms（NVMe） | ↓99% |
| groupWaitMSec | 2.0 ms | 1.0 ms | ↓50% |
| BookKeeper 处理延迟 | 6.5 ms | 1.86 ms | ↓71% |
| Broker + 网络延迟 | 4.5 ms | 1.0 ms | ↓78% |
| **总体 P50 延迟** | **18.1 ms** | **3.88 ms** | **↓79%** |
| **吞吐量** | **30k msg/s** | **1.5M msg/s** | **↑50×** |

> 表明优化不仅降低延迟，还极大提升了系统容量。

---

### 🔀 消融实验结果（Ablation Study）

#### （1）GC 算法对比（50k msg/s 下）

| 配置 | P50 | P99 发布延迟 | Consumer P99 |
|------|-----|----------------|---------------|
| A: G1GC（bookie + broker） | 2.9 ms | 17–55 ms | 15–197 ms |
| B: ZGC（仅 bookie） | 2.2 ms | 5.5–21 ms | 15–36 ms |
| C: ZGC（both） | **2.1 ms** | **7.7–38 ms** | **14–18 ms** |

✅ 结论：**ZGC Generational 显著减少 GC 停顿，尤其消除 213 ms 级别的 spike**

#### （2）flushInterval 调优对比

| 间隔 | P50 | P95 | P99.9 |
|------|-----|-----|--------|
| 60s（默认） | 2.20 ms | 6.09 ms | 118.8 ms |
| **30s（推荐）** | **1.42 ms** | **5.91 ms** | **104.5 ms** |
| 15s | 1.46 ms | 17.0 ms | 45.3 ms |

⚠️ 注意：虽然 15s 更快清空队列，但引发更频繁的突发 I/O，反而恶化 P99 性能。

✅ 推荐 **30s flushInterval**：平衡延迟与稳定性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **三大独立延迟根源被识别并解决：**
   - **G1GC 的长时间停顿** → 改用 **ZGC Generational** 彻底消除
   - **磨损 SSD 上 journal fdatasync 过慢（5.1 ms）** → 使用 **专用 NVMe journal disk** 可降至 0.02 ms
   - **ForceWriteThread 与 kernel page cache writeback 的隐式竞争** → 即使设备物理隔离仍受影响，属 **新型底层干扰现象**

2. **OS 内核参数调优效果显著：**
   ```bash
   vm.dirty_ratio = 2
   vm.dirty_background_ratio = 1
   ```
   有效抑制脏页累积，减轻 SyncThread 触发时的 I/O 突发压力。

3. **Pulsar 具备极强横向扩展潜力：**
   - 当前瓶颈仅为 **网络 I/O（10Gbps）**
   - 升级至 25Gbps NIC 可在相同 3 节点达到 **3M msg/s**
   - 通过 **partition federation** 扩展至 15 节点 → **15M msg/s（约 1.3 trillion/day）**

4. **无需外部 LB，利用 key-based routing + Key Shared Subscription 即可实现高可用与负载均衡**

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖高端硬件（如 NVMe、25Gbps NIC）** | 成本较高，不适合所有场景 |
| **JFR 分析需要 Java 21+ 和生产权限** | 某些受限环境难以实施 |
| **kernel tuning 可能影响其他服务共存** | 在多租户节点上需谨慎应用 |
| **未测试跨地域复制（geo-replication）场景** | 仅聚焦单区域性能优化 |

---

### 🔮 未来工作方向

1. **探索更多硬件组合下的最优配置**（如 Optane PMem 作为 journal）
2. **自动化动态 flushInterval 调节机制**，根据负载自适应调整
3. **将 ForceWriteThread 与 kernel writeback 的解耦机制反馈给社区**，推动 BookKeeper 或内核层面改进
4. **构建全自动的“Latency Observatory”监控平台**，集成 JFR + Prometheus + eBPF 实现根因自动定位
5. **验证更大规模（百节点级）下的扩展一致性与容错能力**

---

## ✅ 总结建议（来自论文 Key Recommendations）

1. **为 journal 专用 NVMe 磁盘**（带来 ~60% 延迟下降）
2. **升级至 25 Gbps NIC**（吞吐翻倍）
3. **切换到 ZGC Generational**（彻底消除 GC 尖峰）
4. **设置 `flushInterval=30s`**（P50 ↓35%）
5. **应用 OS 层 `dirty_ratio=2` 等调优参数**
6. **使用 Key Shared Subscription + `numBundles=256` + auto-split** 实现细粒度负载均衡

> 💡 **一句话总结**：  
> 通过 **JFR 驱动的全栈协同优化**，作者在 **3 台机器上实现了 1.5 百万 msg/s 的稳定吞吐与亚毫秒级中位延迟**，揭示了一个隐藏的内核级性能陷阱，并展示了通往千万级吞吐的清晰路径。

</details>

---

### 10. [PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression](https://arxiv.org/abs/2603.29078)

**Authors**: Caio Vicentino  
**Category**: cs.CL  
**Published**: 2026-04-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.29078v1  

#### Abstract
We present PolarQuant, a post-training weight quantization method for large language models (LLMs) that exploits the distributional structure of neural network weights to achieve near-lossless compression. PolarQuant operates in three stages: (1) block-wise normalization to the unit hypersphere, (2)...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLMs）在消费级硬件上的部署受限于显存容量。例如，一个9B参数的模型以FP16格式存储需要约18GB显存，远超大多数消费级GPU的能力。因此，**高效的权重量化**成为关键需求。

然而，传统的量化方法如 **absmax** 在非均匀分布（尤其是近似高斯分布）的权重上表现不佳，因为它将码本均匀分配在 $[-a, a]$ 区间内，导致：
- 将宝贵的码本资源浪费在稀有异常值上；
- 在高密度中心区域产生较大的量化误差。

### **提出了什么新方法或新思路**
本文提出 **PolarQuant**，一种无需校准数据的后训练权重量化方法，其核心思想是通过**块归一化 + Hadamard旋转**，将权重分布转换为近似独立同分布的标准正态变量（i.i.d. $ \mathcal{N}(0,1) $），从而可以应用**MSE最优的Lloyd-Max量化器**进行高效压缩。

#### **方法流程（四阶段）**
1. **Block-wise 归一化**：将权重划分为大小为 $ d=128 $ 的块，对每块进行 $ \ell^2 $ 归一化至单位超球面。
2. **Walsh-Hadamard 旋转**：用标准化的 Walsh-Hadamard 矩阵 $ H_d $ 对每个块进行线性变换。
3. **缩放与量化**：将旋转后的坐标乘以 $ \sqrt{d} $ 得到近似 $ \mathcal{N}(0,1) $ 分布，并使用预计算的 Lloyd-Max 质心进行最近邻量化。
4. **存储**：仅保存整数量化码（int8）、每块范数（fp16）和全局共享的质心表（fp32）。

### **相比现有方法的优势**
| 特性 | PolarQuant |
|------|----------|
| **无校准依赖** | 不需要任何 calibration data，简化部署 |
| **近似无损压缩** | Q5 量化后 PPL 仅比 FP16 高 +0.02～+0.03 |
| **极简设计** | 核心操作仅为一次矩阵乘法 $ H_{128} b_i $，无梯度、无迭代优化 |
| **可组合性强** | 可作为下游 INT4 量化的有效预处理步骤 |
| **零运行时开销** | Hadamard 矩阵自逆（$ H^{-1}=H $），反量化仅增加加载时间（~8秒），推理无额外代价 |

> ✅ **关键洞察**：Hadamard 旋转本身贡献了 **98% 的质量提升**，而 Lloyd-Max 质心仅贡献 2%，说明“分布对齐”比“最优质心设计”更重要。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **WikiText-2**：用于评估语言建模能力，采用滑动窗口（window=2048, stride=512），掩码前1536个token上下文。

### **实验设置**
- **模型**：Qwen3.5-9B（混合 DeltaNet+MoE 架构）
- **硬件平台**：
  - 主要：NVIDIA RTX PRO 6000 Blackwell（96GB VRAM）
  - 跨平台验证：Apple Mac mini M4（16GB 统一内存）
- **量化配置**：
  - 块大小 $ d = 128 $
  - 使用 Lloyd-Max 算法预先计算 $ \mathcal{N}(0,1) $ 下的最优质心（支持 2–5 bit）
- **速度测量**：平均3次生成100 token的结果，warm-up 后取 tokens/sec（tok/s）

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | 主要质量指标，越低越好 |
| **Throughput (tok/s)** | 推理吞吐量 |
| **VRAM / Memory Usage** | 显存或内存占用 |
| **Compression Ratio** | 相对于 FP16 的压缩倍数 |

### **基线方法对比**
| 方法 | 类型 | 是否需校准 |
|------|------|------------|
| FP16 | 全精度基准 | — |
| torchao INT4 (group-wise absmax) | 工业级 INT4 量化 | 否 |
| BitsAndBytes NF4 | NormalFloat 4-bit | 是（统计分布） |
| GPTQ | Hessian-based 逐层量化 | 是 |
| AWQ | Activation-aware 缩放 | 是 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（RTX PRO 6000）**

| 方法 | tok/s | VRAM | PPL | ΔPPL |
|------|--------|--------|------|-------|
| FP16 baseline | 45.7 | 17.9 GB | 6.37 | — |
| torchao INT4 (absmax) | 43.3 | 6.3 GB | 6.68 | +0.31 |
| BnB NF4 | 34.6 | 7.7 GB | ~6.7 | +0.33 |
| **PolarQuant Q5 + torchao INT4** | **43.1** | **6.5 GB** | **6.56** | **+0.19** |
| PolarQuant Q5 dequant (FP16) | 45.9 | 18.1 GB | 6.39 | +0.02 |
| PolarQuant + AWQ dequant | 45.8 | 17.9 GB | 6.43 | +0.06 |

> 🔥 **亮点**：PolarQuant Q5 + torchao INT4 在几乎不牺牲速度和显存的前提下，将 PPL 从 6.68 降至 **6.56**，缩小了与 FP16 的差距达 **39%**。

### **跨平台部署效果（Apple Silicon M4）**

| 方法 | tok/s | Memory | PPL |
|------|--------|---------|------|
| PolarQuant MLX Q4 | 19.7 | 4.8 GB | 6.90 |

✅ 成功在 16GB 内存设备上运行 9B 模型，接近实时推理水平。

### **消融实验结果（Ablation Study）**

| 配置 | PPL | ΔPPL | 改进占比 |
|------|------|--------|-----------|
| Absmax Q5（baseline） | 6.90 | +0.53 | — |
| + Hadamard rotation only | 6.40 | +0.03 | **98%** |
| + Lloyd-Max centroids only | 6.91 | +0.54 | -2%（恶化） |
| + Both（完整 PolarQuant Q5） | 6.39 | +0.02 | 100% |
| + AWQ scales | 6.43 | +0.06 | — |
| + torchao INT4 on top | 6.56 | +0.19 | — |

📌 **核心发现**：
- **Hadamard 旋转是决定性因素**，单独即可将 PPL 从 6.90 降到 6.40；
- Lloyd-Max 质心在 Q5 下作用微弱（仅再降 0.01），但在更低比特（如 Q3）下更关键（理论 MSE 减少 54%）；
- **PolarQuant Q5 单独优于 PolarQuant+AWQ**，表明**均匀比特分配 + 旋转 > 非均匀比特分配 + 校准**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Hadamard 旋转使权重服从近似高斯分布**，使得标准正态下的最优量化器（Lloyd-Max）得以适用。
2. ✅ **旋转本身贡献了 98% 的性能增益**，证明“分布对齐”比“精细质心设计”更重要。
3. ✅ **PolarQuant 可作为通用预处理模块**，显著提升下游 INT4 量化器（如 torchao）的质量。
4. ✅ **无需校准、无需图修改、零运行时开销**，适合工业部署。
5. ✅ **统一 Q5 比混合比特（mixed-bit）策略更优**，尤其当用于后续 re-quantization 时。

### **方法的局限性**
- 假设 Hadamard 旋转后权重块服从 i.i.d. 高斯分布，可能不适用于所有架构（如稀疏或高度结构化模型）；
- 当前未利用块间相关性（inter-block correlation）；
- 在极低位宽（如 Q2）下仍存在明显失真；
- 混合比特策略虽能进一步压缩，但不适合作为 INT4 的前置步骤（因 Q3 层信息损失过大）。

### **未来工作方向**
- 扩展至 **activation quantization**；
- 探索基于高斯分布的 **vector quantization** 和 lattice codebooks；
- 研究 **cascaded quantization pipelines** 的理论极限；
- 结合 learned rotation（如 SpinQuant）探索性能边界；
- 支持更多后端框架（GGUF, MLX, llama.cpp 等）。

---

> 📚 **代码与模型公开地址**：  
> GitHub: [https://github.com/caiovicentino/eoq-quantization](https://github.com/caiovicentino/eoq-quantization)  
> Hugging Face: [https://huggingface.co/caiovicentino](https://huggingface.co/caiovicentino)

</details>

---

### 11. [Mitigating Temporal Blindness in Kubernetes Autoscaling: An Attention-Double-LSTM Framework](https://arxiv.org/abs/2603.28790)

**Authors**: Faraz Shaikh, Gianluca Reali, Mauro Femminella  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.28790v1  

#### Abstract
In the emerging landscape of edge computing, the stochastic and bursty nature of serverless workloads presents a critical challenge for autonomous resource orchestration. Traditional reactive controllers, such as the Kubernetes Horizontal Pod Autoscaler (HPA), suffer from inherent reaction latency, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Mitigating Temporal Blindness in Kubernetes Autoscaling: An Attention-Double-LSTM Framework*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **边缘计算**（edge computing）环境中 **serverless 工作负载** 的 **随机性、突发性**（stochastic and bursty）特性，解决传统自动扩缩容机制在 **时序感知能力不足** 上的根本缺陷。

具体而言，现有方法存在以下问题：
- **Kubernetes HPA**（Horizontal Pod Autoscaler）等 **反应式控制器** 存在 **反应延迟**（reaction latency），导致流量突增时出现 **SLO 违规**（Service Level Objective violations），而在降级时则产生 **资源抖动**（resource flapping）。
- **标准 DRL**（Deep Reinforcement Learning）代理如 DQN 或单层 LSTM-PPO 虽具前瞻性，但仍受 **时序失明**（temporal blindness）困扰，即无法有效捕捉 **长期依赖关系** 和区分 **短期噪声** 与 **真实需求趋势变化**。

### 🚀 提出的新方法
作者提出一种 **稳定性感知的自动扩缩容框架**，其核心是将 **工作负载预测** 与 **控制决策** 统一在一个 **Attention-Enhanced Double-Stacked LSTM-PPO** 架构中。

#### 创新点包括：
1. **Attention-Enhanced Double-Stacked LSTM Policy Network**  
   - 使用 **双层堆叠 LSTM**（Double-Stacked LSTM）建模高阶时间依赖，同时引入 **软注意力机制**（soft-attention mechanism），使模型能动态加权历史状态，聚焦于关键事件（如并发请求激增），过滤高频噪声。
   - 该设计克服了浅层 RNN 的 **信息瓶颈**（information bottleneck），提升了对非马尔可夫环境的建模能力。

2. **统一预测与控制闭环**  
   - 不同于将预测模块与控制模块分离的设计（Separation of Concerns），本方法将深度时序模型直接嵌入 PPO 的策略网络中，实现端到端联合优化，避免误差传播。

3. **多维离散动作空间设计**  
   - 控制器操作维度更丰富，不仅调节 HPA CPU 目标值，还同时优化 **吞吐量乘数**（throughput multiplier）、**增强级别**（enhancement level）等参数，提升系统调控灵活性。

### 🔍 相比现有方法的优势
- 显著优于传统 HPA 和主流 DRL 基线，在 **延迟** 和 **稳定性** 之间取得更好平衡。
- 通过注意力机制实现了对 **关键历史时刻的选择性记忆**，增强了模型鲁棒性和泛化能力。
- 在真实世界 workload 下验证了其在生产级边缘环境中的可行性。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用 **Microsoft Azure Functions 2021 调用轨迹**（real-world Azure Functions invocation traces）作为 workload 输入。
- 随机选取 7 天数据，其中 **5 天用于训练**，**2 天用于测试**。
- 每天划分为 500 个控制周期（control interval），每个周期为 60 秒，共 2,500 训练步 + 1,000 测试步。

### ⚙️ 实验设置
- **平台**：基于 MicroK8s 搭建的两节点异构 Kubernetes 集群（master + worker）。
- **负载生成工具**：`hey` 工具回放 trace，注入至 OpenFaaS 网关。
- **目标函数**：一个 CPU 密集型的 `factorizator` serverless 函数，便于归因延迟变化。
- **硬件加速**：训练节点配备 NVIDIA L40S GPU，用于加速 LSTM/Attention 推理。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **平均延迟**（Avg Latency） | 请求响应时间均值 |
| **P90 延迟**（90th Percentile Latency） | 尾部延迟表现，反映用户体验一致性 |
| **SLO 合规率** | 达成目标延迟（20ms）和硬阈值（50ms）的比例 |
| **CPU 平均利用率** | 资源效率衡量 |
| **副本数量波动**（Replica Churn） | 定义为 $\sum |\Delta p_t|$，衡量控制稳定性 |
| **副本数均值 ± 标准差** | 反映资源配置的平稳性 |

### 🔄 对比的基线方法
| 方法 | 类型 | 特点 |
|------|------|------|
| **Static HPA (50%)** | 反应式规则控制 | 行业标准，固定 50% CPU 阈值，带冷却期 |
| **DDQN**（Double Deep Q-Network） | 值函数法 DRL | Stateless，无记忆机制，代表经典 DRL 扩缩容方案 |
| **Single-LSTM PPO** | 单层 LSTM-PPO | 当前先进方法（如 DRe-SCale），用于消融研究 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table II）

| Agent | Avg Latency (ms) | P90 Latency (ms) | Hard SLO (50ms) Compliance | Replica Churn |
|-------|------------------|------------------|----------------------------|---------------|
| Static HPA | 58.82 | >300 | 43.5% | 97 |
| DDQN | 77.33 | ~175–200 | 24.6% | 0 |
| Single-LSTM | 32.37 | ~100–130 | 92.0% | 716 |
| **Double-LSTM (Ours)** | **24.11** | **~60–70** | **96.9%** | **432** |

> 注：P90 数据未直接给出数值，由图 7 推断。

### 🔁 与基线对比结果
- **相比 Single-LSTM PPO 基线**：
  - **P90 延迟降低约 29%**
  - **副本抖动减少 39%**（716 → 432）
  - **SLO 合规率从 92.0% 提升至 96.9%**
- **相比 Static HPA**：
  - 平均延迟下降 **59%**（58.82 → 24.11 ms）
  - 尾部延迟大幅改善（>300ms → <70ms）
  - 更高效利用资源（CPU 利用率从 15.44% 提升至 38.22%）
- **相比 DDQN**：
  - 避免严重过载（DDQN 经常超 100% CPU）
  - 显著提升 SLO 合规率（24.6% → 96.9%）
  - 虽然 DDQN 不扩缩（churn=0），但属于“保守失效”，而非稳定

### 🔍 消融实验结果
- **Single-LSTM vs. Double-LSTM + Attention** 是核心消融对比。
- 结果显示：
  - 单纯增加 LSTM 层数（无 attention）仍易受噪声干扰，导致 **过度扩缩**（thrashing），表现为高 churn（716）和较大副本标准差（±3.54）。
  - 引入 **注意力机制后**，模型能够识别并关注真正的需求上升信号，从而做出更平滑、前瞻性的扩缩决策，显著降低 churn 至 432，并维持更稳定的副本数（2.83±3.03）。
- 图 8a 显示 Double-LSTM 的预测误差分布更集中于零附近，说明其 **预测精度更高且更一致**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **缓解“时序失明”是实现可靠自动扩缩的前提**  
   单纯使用 DRL 或浅层 LSTM 无法有效应对边缘环境下复杂的非马尔可夫 workload。必须通过 **深层记忆结构 + 注意力机制** 来捕获长期依赖。

2. **深度注意力机制显著提升控制稳定性**  
   Attention 使得模型可以 **选择性地关注关键历史状态**，有效滤除瞬时噪声，防止误触发扩缩，从而大幅降低 replica churn。

3. **预测与控制的紧耦合优于解耦架构**  
   将预测模块内置于策略网络中，避免了解耦系统中常见的 **预测误差放大** 和 **控制震荡** 问题。

4. **PPO + Attention + LSTM 架构适合连续控制场景**  
   相比 DDQN 等值函数方法，PPO 提供更稳定的策略更新，结合 LSTM 和 attention，在延迟与资源效率间取得了更优的帕累托前沿。

### ⚠️ 方法的局限性
1. **推理开销较高**  
   相比 HPA 的简单阈值判断，本方法需运行双层 LSTM 和 attention 计算，**推理延迟更高**。尽管在 GPU 上仅为 5–10ms，但在资源受限的 IoT 边缘设备上可能成为瓶颈。

2. **实验环境简化**  
   - 使用虚拟化集群和单一 microservice（factorizator），未考虑 **多租户干扰**（noisy neighbor）、**网络 I/O 竞争** 或 **缓存抖动**。
   - 未模拟真实 6G/RAN 动态对控制环路的影响。

3. **缺乏跨服务链协同能力**  
   当前为单智能体设计，无法处理微服务链中因上游扩缩引发的 **下游背压**（back-pressure）或资源饥饿问题。

### 🔮 未来工作方向
1. **扩展至 Multi-Agent RL**（MARL）框架，协调多个相关服务的扩缩行为，避免级联故障。
2. **集成能耗作为优化目标**，构建绿色、可持续的 autoscaling 策略。
3. **部署于物理 6G 测试床**，验证在真实无线接入网（RAN）动态下的鲁棒性，特别是在 AI-RAN 架构中的应用潜力。
4. **模型轻量化**：探索 **模型量化**（quantization）、**知识蒸馏**（knowledge distillation）等技术，降低模型体积与推理成本，适配嵌入式边缘节点。

---

> **总结一句话**：  
> 本文提出的 **Attention-Double-LSTM-PPO** 框架通过深度融合预测与时序建模，有效缓解了 Kubernetes 自动扩缩中的“时序失明”问题，在真实 workload 下实现了 **更低延迟、更高 SLO 合规率、更强控制稳定性** 的突破，为边缘 serverless 场景提供了新一代智能扩缩解决方案。

</details>

---

### 12. [Exploration of Energy and Throughput Tradeoffs for Dataflow Networks](https://arxiv.org/abs/2603.29367)

**Authors**: Abrarul Karim, Joachim Falk, J\"urgen Teich  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.29367v1  

#### Abstract
The introduction of dynamic power management strategies such as clock gating and power gating in dataflow networks has been shown to provide significant energy savings when applied during idle times. However, these strategies can also degrade throughput due to shutdown and wake-up delays. Such throu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

**论文标题**: *Exploration of Energy and Throughput Tradeoffs for Dataflow Networks*

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对 **dataflow networks** 中因引入动态功耗管理策略（如时钟门控 `clock gating` 和电源门控 `power gating`）所带来的能效与吞吐量之间的权衡问题。

- **核心矛盾**：允许硬件实现的 `actor` 在空闲时进入低功耗（睡眠）模式可以显著节省能量，但唤醒和关机延迟会增加执行时间，从而降低系统吞吐量（即增大周期 `P`）。
- 这对于需要保证吞吐量的信号处理系统（如 IoT 设备中的实时应用）尤为关键。

### **提出了什么新方法或新思路**
论文提出了三个层次的解决方案，构成了其主要贡献：

1. **最大吞吐量调度的线性规划（LP）模型**  
   - 针对给定的 `self-powering dataflow network`，提出一个 **Linear Program (LP)** 来寻找**周期性的最大吞吐量调度**。
   - 该模型能够分析在不同 `actor` 被配置为 `always-active` (AA) 或 `self-powered` (SP) 模式下的系统性能上限。

2. **最小能耗调度的混合整数线性规划（MILP）模型**  
   - 提出一个 **Mixed-Integer-Linear-Program (MILP)** 模型，在满足给定吞吐量约束（即固定周期 `P`）的前提下，**最小化每个周期的总能耗**。
   - 该模型通过决策变量自动确定哪些 `actor` 应被设为 `critical`（必须保持 AA 模式以避免延迟），哪些可以安全地进入 SP 模式以节能。

3. **高效的多目标设计空间探索（DSE）策略：Hop and Skip (H&S)**  
   - 提出一种名为 **Hop and Skip** 的新型 DSE 策略，用于高效地探索 `energy` 与 `throughput` 之间的帕累托前沿（Pareto front）。
   - **“Hop”**：利用 MILP 找到当前周期下能耗最低的配置。
   - **“Skip”**：利用 LP 快速跳转到该配置所能达到的最短周期，从而跳过中间大量无意义的周期点，极大减少搜索次数。

### **相比现有方法的优势**
- **效率极高**：相比于暴力穷举所有配置的 `decision-variable sweep` 和遍历所有周期的 `period sweep`，H&S 策略实现了数量级的加速（最高达 **2300×**）。
- **精度高**：在绝大多数情况下，H&S 找到的非支配解集与真实的帕累托前沿完全一致（`HV ratio = 1`）。
- **实用性更强**：提供了一种在工程实践中可行的、快速找到最优能效-吞吐量折衷点的方法。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **真实世界案例研究（Case Study）**：
   - **Acoustic Echo Cancellation (AEC) network**：一个实际的声学回声消除网络，作为贯穿全文的运行示例。
2. **基准测试集（Benchmarks）**：
   - **SDF3 Benchmarks**：来自工具集 "SDF3 - SDF For Free" 的一组已知应用，包括：
     - *H.263 Encoder*, *Modem*, *MP3 Playback*, *Satellite*, *Samplerate*, *MP3 Decoder* 等。
   - **随机生成图（Random Graphs）**：
     - 使用 SDF3 工具生成了 **100 个随机的 SDF 图**，每个图平均包含 15 个 `actor`，用于统计性评估。

> **注**：原始 SDF 图被转换为 `marked graph` 以适应本文的分析框架。

### **实验设置和评估指标**
- **参数提取**：基于 28nm 工艺的硬件综合（HLS & RTL），使用 QuestaSim 进行门级仿真，从网表中提取每个 `actor` 的 `execution time`, `power`, `shutdown delay`, `wake-up delay` 等真实参数。
- **评估指标**：
  1. **探索时间（Exploration Time）**：完成整个 DSE 所需的 CPU 时间。
  2. **加速比（Speedup）**：H&S 相对于 `x Sweep` 和 `P Sweep` 的速度提升倍数。
  3. **超体积比（Hypervolume Ratio, HV ratio）**：衡量所找到的非支配解集逼近真实帕累托前沿的程度。`HV ratio = 1` 表示完全匹配。

### **基线方法对比**
论文将提出的 **H&S** 策略与两种暴力搜索基线进行了对比：
1. **Decision Variable Sweep (`x Sweep`)**：
   - 穷举所有 `2^|A|` 种 `actor` 的 AA/SP 配置组合，对每种组合求解 LP 得到最优周期和能耗。
   - **优点**：能找到真实的帕累托前沿（由 Theorem 1 证明）。
   - **缺点**：指数级复杂度，不适用于大规模网络。
2. **Period Sweep (`P Sweep`)**：
   - 遍历从 `P_min` 到 `P_max` 的每一个整数周期，对每个周期 `P` 求解一次 MILP 以找到能耗最低的配置。
   - **优点**：比 `x Sweep` 更快，尤其当 `P_max - P_min` 较小时。
   - **缺点**：仍需求解大量 MILP，且可能错过具有有理数周期的帕累托点。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **AEC 网络案例结果**：
  - **全 AA 模式**：周期 `P = 23`，能耗 `E = 182 pJ`。
  - **全 SP 模式**：周期 `P = 27` (+17%)，能耗 `E = 82 pJ` (-55%)。
  - **H&S 找到的最优折衷**：存在一个**混合配置**，周期仍为 `P = 23`，但能耗降至 `E = 146 pJ`，实现了 **20% 的节能而无吞吐量损失**。

### **与基线方法的对比结果**
根据 **Table I** 的数据，H&S 策略表现卓越：

| 网络 | H&S vs `x Sweep` 加速比 | H&S vs `P Sweep` 加速比 | HV Ratio |
| :--- | :--- | :--- | :--- |
| **Satellite** | 348.93× | — | 1.0000 |
| **Modem** | 1452.39× | 8.30× | 1.0000 |
| **MP3 Decoder (Block)** | 1489.11× | 3.65× | 1.0000 |
| **Samplerate** | 2.67× | 27.83× | 1.0000 |

- **总体加速**：在 SDF3 基准上，H&S 平均比 `P Sweep` 快 **数十至数百倍**，比 `x Sweep` 快 **数百至数千倍**。
- **精度保证**：在所有 SDF3 基准和 98% 的随机图上，H&S 找到了完整的帕累托前沿（`HV ratio = 1`）。仅在极少数情况下（`ε=1` 时）因步长过大而遗漏，但将步长减小到 `ε=0.1` 后即可完美恢复。

### **消融实验结果**
- **步长 `ε` 的影响**：实验表明，`H&S` 算法中的步长 `ε` 是一个关键参数。较大的 `ε`（如 1）可能导致错过某些帕累托点；而较小的 `ε`（如 0.1）虽然计算量稍增，但能确保找到完整的前沿。
- **网络规模的影响**：随着网络中 `actor` 数量的增加，`x Sweep` 的计算时间呈指数爆炸，而 `H&S` 依然能保持高效。

---

## **4. 关键结论和发现**

### **主要发现**
1. **混合模式是关键**：并非所有 `actor` 都应进入睡眠模式。通过智能选择 `critical actors`（必须保持活跃），可以在几乎不牺牲吞吐量的情况下实现显著的节能（如 AEC 案例中的 20%）。
2. **H&S 策略极其高效**：所提出的 **Hop and Skip** DSE 策略能够在极短的时间内找到与暴力搜索相同的高质量帕累托前沿，是解决此类多目标优化问题的理想方案。
3. **理论与实践结合**：论文不仅提供了严谨的数学建模（LP/MILP），还通过真实的硬件参数验证了方法的有效性和实用性。

### **方法的局限性**
1. **依赖于精确的延迟模型**：方法的效果高度依赖于对 `shutdown delay` 和 `wake-up delay` 的准确估计。
2. **静态调度假设**：目前的工作基于周期性静态调度，未考虑更复杂的动态数据流行为。
3. **MILP 求解器开销**：尽管 H&S 大幅减少了调用次数，但每次 MILP 求解本身仍可能很耗时，尤其是在非常大的网络上。

### **未来工作方向**
1. **扩展到动态数据流模型**：将该方法应用于更灵活的动态数据流（Dynamic Dataflow）或场景感知数据流（Scenario-Aware Dataflow）模型。
2. **集成电压频率缩放（DVS）**：将 `Dynamic Voltage Scaling` 与 `DPM` 结合，进行更全面的能效优化。
3. **在线自适应策略**：开发能在运行时根据负载变化动态调整 `actor` 电源状态的轻量级算法。
4. **硬件原型验证**：将优化后的设计部署到 FPGA 或 ASIC 上，进行实际的功耗和性能测量。

</details>

---

### 13. [One-for-All: A Lightweight Stabilized and Parameter-Efficient Pre-trained LLM for Time Series Forecasting](https://arxiv.org/abs/2603.29756)

**Authors**: Prasanjit Dey, Soumyabrata Dev, Bianca Schoen-Phelan  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.29756v1  

#### Abstract
We address the challenge of adapting pre-trained Large Language Models (LLMs) for multivariate time-series analysis, where their deployment is often hindered by prohibitive computational and memory demands. Our solution, One-for-All, introduces Gaussian Rank-Stabilized Low-Rank Adapters (rsLoRA) to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：One-for-All: A Lightweight Stabilized and Parameter-Efficient Pre-trained LLM for Time Series Forecasting

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对将预训练的大型语言模型（**LLM**）应用于多变量时间序列分析时面临的三大挑战：
1. **数据稀缺**：时间序列数据集远小于NLP领域（通常 <10GB），难以支撑大规模训练。
2. **计算与内存开销巨大**：传统fine-tuning需要更新全部参数，导致部署成本高昂。
3. **现有PEFT方法不适应时间序列特性**：
   - **非平稳性**（Non-stationarity）：统计特性随时间变化，破坏标准适配器假设。
   - **不规则采样**（Irregular sampling）：缺失值和异构频率影响tokenization。
   - **多尺度依赖**（Multi-scale dependencies）：局部噪声与全局趋势并存，需自适应rank分配。

### 提出了什么新方法或新思路
提出 **One-for-All** 框架，其核心是 **Gaussian Rank-Stabilized Low-Rank Adapters (rsLoRA)**，一种轻量级、稳定且参数高效的微调机制。

#### 创新点包括：
- **rsLoRA**：在LoRA基础上引入数学上可证明的**秩稳定性机制**（rank-stabilization），通过高斯分布缩放因子 $ \beta_r = \alpha / \sqrt{r} $ 实现梯度稳定，首次为时间序列适配器提供理论保障。
- **统一架构设计**：仅在 **Positional Embeddings** 和 **Output Layers** 注入可训练的低秩矩阵（rank=16），冻结整个LLM主干（如GPT-2），实现跨任务通用性。
- **Patch-based Tokenization**：将原始时间序列分块（patching）后投影为token，有效处理局部趋势和不规则采样。
- **边缘设备部署能力**：模型仅需 **2.2MiB** 内存，可在资源受限设备（如医疗、金融、环境监测终端）运行。

### 相比现有方法的优势
| 维度 | One-for-All | 现有方法（如GPT4TS, TIME-LLM） |
|------|-------------|-------------------------------|
| **可训练参数量** | 0.55M | 3.9–24.0M (GPT4TS), 6.4–11.3M (其他) |
| **内存占用** | 2.2MiB | 340MiB – 4.18GiB |
| **参数效率 (Eff.*MSE)** | 5.50 | 0.26 (GPT4TS), 4.41 (TimesNet) |
| **跨任务支持** | ✅ 预测、分类、异常检测 | ❌ 多为单一任务优化 |
| **理论稳定性** | ✅ 证明梯度稳定 | ❌ 缺乏理论保证 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖六大类时间序列任务，共涉及多个标准基准：

| 任务 | 数据集 | 描述 |
|------|--------|------|
| **长期预测** | ETTh, ETTm, Weather | 高频电力消耗与气象数据，预测长度96–720步 |
| **少样本预测** | ETTh, ETTm, Weather | 使用10%/5%训练数据，测试泛化能力 |
| **零样本预测** | M3 → M4 | 跨数据集迁移，无fine-tuning |
| **短期预测** | M4 | 包含年/季/月/其他频率的商业经济序列 |
| **分类** | UEA Multivariate | 多变量时间序列分类任务（如Japanese Vowels, SCP1等） |
| **异常检测** | SMD, MSL, SMAP, SWaT, PSM | 工业系统与航天器传感器异常识别 |

### 实验设置和评估指标
- **主干模型**：GPT-2（默认），所有比较均基于相同配置以确保公平。
- **rsLoRA Rank**：固定为16（消融实验证明此为最优平衡点）。
- **训练细节**：
  - 使用PyTorch实现，在NVIDIA A100 80GB GPU上训练。
  - 每个实验重复3次取平均性能。
  - Patch大小=16，输入归一化采用Z-score标准化。

#### 评估指标按任务划分：
| 任务 | 主要指标 |
|------|---------|
| 长期/少样本预测 | **MSE**, **MAE** |
| 零样本预测 | **sMAPE**（对称平均绝对百分比误差） |
| 短期预测 | **sMAPE**, **MASE**, **OWA** |
| 分类 | **Accuracy** |
| 异常检测 | **Precision**, **Recall**, **F1-Score** |

### 基线方法对比
涵盖三类主流方法：
1. **现代LLM-based模型**：
   - GPT4TS, TIME-LLM, TEST, TEMPO
2. **Transformer变体**：
   - FEDformer, Non-Stationary Transformer
3. **经典预测架构**：
   - TimesNet, ETSformer, Autoformer, Informer, Reformer

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）长期预测（Long-term Forecasting）
| 模型 | 参数量(M) | 内存(MiB) | Avg MSE | Eff.*MSE |
|------|----------|-----------|--------|----------|
| **One-for-All** | **0.55** | **2.2** | **0.33** | **5.50** |
| GPT4TS | 11.7 | 371.0 | 0.33 | 0.26 |
| TIME-LLM | 6.46 | 3907 | 0.31 | 0.50 |
| TimesNet | 0.63 | 2.9 | 0.36 | 4.41 |

> 💡 **结论**：One-for-All 在匹配SOTA精度的同时，参数减少 **6.8–21×**，内存缩小 **168–1,776×**，参数效率提升 **5.5×**。

#### （2）少样本预测（Few-shot, 10%数据）
| 模型 | 参数量(M) | Avg MSE | Eff.*MSE |
|------|----------|--------|----------|
| **One-for-All** | **0.55** | **0.416** | **4.37** |
| GPT4TS | 11.7 | 0.400 | 0.21 |
| TimesNet | 0.63 | 0.526 | 3.02 |

> 💡 即使在极端低数据下（5%），One-for-All仍优于TimesNet在10%下的表现。

#### （3）零样本预测（Zero-shot, M3→M4）
| 模型 | 平均sMAPE |
|------|----------|
| **One-for-All** | **13.27** |
| GPT4TS | 13.55 |
| TimesNet | 15.01 |

> 💡 在M3→M4迁移中取得最佳平均性能，并在年度预测上显著领先。

#### （4）异常检测
| 模型 | 平均F1 |
|------|-------|
| **One-for-All** | **84.42%** |
| GPT4TS | 86.72% |
| TimeNet | 85.24% |

> 💡 尽管略低于最强基线，但在工业关键数据集 **SWaT (92.20%)** 和 **PSM (97.10%)** 上表现优异。

#### （5）分类任务
- 在 **Japanese Vowels** 达到 **98%准确率**，与最优持平。
- 在 **SCP1** 上达到 **93%**，显著优于Transformer类模型（高出35–39%）。

---

### 消融实验结果（Ablation Study）

#### （1）不同Rank的影响（Table VII）
| Rank | 参数量(M) | 内存(MiB) | Long-term MSE | Few-shot MSE |
|------|----------|------------|----------------|----------------|
| 2 | 0.068 | 0.277 | 0.41 | 0.49 |
| 8 | 0.275 | 1.100 | 0.33 | 0.42 |
| **16** | **0.550** | **2.200** | **0.33** | **0.42** |
| 32 | 1.100 | 4.425 | 0.33 | 0.42 |
| 1024 | 35.20 | 140.8 | 0.33 | 0.44 |

> ✅ **Rank 16即达性能饱和**（95%最大精度），更高rank收益极小，验证了rsLoRA的高效性。

#### （2）资源增长趋势
- 参数量与rank呈近似线性增长，而内存占用极低。
- Rank 16 是 **效率-性能帕累托前沿**，适用于边缘部署。

---

## 4. 关键结论和发现

### 主要发现
1. **rsLoRA实现了理论可证的梯度稳定性**，首次解决了LoRA在非平稳时间序列中的不稳定问题。
2. **Rank 16即可实现SOTA性能**，远低于传统LoRA所需的Rank 256+，极大降低资源需求。
3. **One-for-All是一个真正“统一”的框架**：单个2.2MiB模型可通用于预测、分类、异常检测等多种任务。
4. **在几乎所有任务中实现效率-精度最优权衡**：
   - 参数减少 **6.8–21×**
   - 内存减少 **168–1,776×**
   - 参数效率提升 **5.5×**
5. **具备强鲁棒性**：在不同预测长度（96–720步）、多种数据集上MSE波动 <1%，适合实际部署。

### 方法的局限性
- **召回率略低**：在部分异常检测任务（如SMAP）中，Recall为53.6%，低于某些基线。
- **零样本反向迁移较弱**：M4→M3方向表现不如GPT4TS。
- **依赖预训练LLM语义先验**：若时间序列与文本语义无关，可能限制潜力。

### 未来工作方向
1. **优化自适应patching策略**，以更好处理不规则采样数据。
2. **改进频率感知的rsLoRA缩放机制**，增强多周期建模能力。
3. **扩展至多模态场景**（如结合文本描述的时间序列预测）。
4. 探索更广泛的下游应用，如医疗诊断、金融风控等实时决策系统。

---

> 🔗 **代码开源地址**：[https://github.com/Prasanjit-Dey/Onefor_All](https://github.com/Prasanjit-Dey/Onefor_All)

</details>

---

### 14. [\texttt{ReproMIA}: A Comprehensive Analysis of Model Reprogramming for Proactive Membership Inference Attacks](https://arxiv.org/abs/2603.28942)

**Authors**: Chihan Huang, Huaijin Wang, Shuai Wang  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.28942v1  

#### Abstract
The pervasive deployment of deep learning models across critical domains has concurrently intensified privacy concerns due to their inherent propensity for data memorization. While Membership Inference Attacks (MIAs) serve as the gold standard for auditing these privacy vulnerabilities, conventional...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《ReproMIA: A Comprehensive Analysis of Model Reprogramming for Proactive Membership Inference Attacks》总结

---

## 1. 主要贡献和创新点

### 解决的问题
传统的 **Membership Inference Attack (MIA)** 方法存在两大瓶颈：
1. **计算成本高昂**：基于 shadow model 的方法需要训练多个代理模型，对于参数量巨大的 LLMs 和 Diffusion Models 而言不切实际。
2. **信号衰减严重**：现代正则化技术（如 label smoothing）使得成员（member）和非成员（non-member）样本在输出分布上高度相似，导致传统基于后验概率或熵值的方法难以区分。

此外，现有 MIA 多为**被动观察**（passive），缺乏主动探测和放大隐私泄露信号的机制。

### 提出的新方法与思路
本文提出 **ReproMIA**，一种统一且高效的**主动式**（proactive）MIA 框架，其核心思想是将 **Model Reprogramming** 技术作为**主动隐私探针**（active privacy probe），用于放大模型内部对训练数据的记忆痕迹。

- **核心洞察**：通过在输入空间学习一个可优化的变换函数 $ \delta^* $，可以对模型进行“深度压力测试”（deep stress test），从而主动诱导并放大成员与非成员样本在模型行为上的细微差异。
- **实现方式**：冻结目标模型参数，在输入空间引入一个轻量级的 reprogramming layer（如软提示 soft prompt 或全局扰动），通过双层优化（bilevel optimization）学习最优的 $ \delta^* $，以最大化成员与非成员的响应差异。

### 相比现有方法的优势
- **高效性**：无需微调或训练 shadow model，显著降低计算开销，适用于大规模模型。
- **主动性**：从“被动观察”转变为“主动探测”，能有效放大被现代训练策略抑制的隐私信号。
- **通用性**：框架可统一应用于多种模型架构，包括 **LLMs**、**Diffusion Models**、**Classification Models** 和 **GNNs**。
- **高性能**：尤其在低 FPR（False Positive Rate）场景下表现卓越，这是安全审计中最关键的指标。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了机器学习的多个核心领域，共超过十个基准数据集：
- **NLP / LLMs**:
  - **WikiMIA**: 基于维基百科文本，按时间划分成员/非成员。
  - **MIMIR**: 基于 The Pile 数据集，通过 n-gram 过滤构造更难区分的成员/非成员样本。
- **Computer Vision / Diffusion Models**:
  - **CIFAR-10**, **CIFAR-100**, **TinyImageNet**, **LAION-5B**, **COCO**.
- **Classification Models**:
  - **CINIC-10**, **STL-10**, **ImageNet-100**.
- **GNNs**:
  - **Cora**, **PubMed**, **Citeseer**.

### 实验设置和评估指标
- **威胁模型**：黑盒攻击（black-box），仅能查询模型输出（如 logits、loss）。
- **评估指标**：
  - **AUC** (Area Under the ROC Curve)：衡量整体判别能力。
  - **TPR@Low FPR**：在极低误报率下的真阳性率，如 **TPR@1%FPR**，是安全审计中的关键指标。
  - **Balanced Accuracy**：平衡准确率。
- **防御鲁棒性测试**：在输出添加噪声（logits perturbation）、输入平滑（input smoothing）、差分隐私（DP-SGD）等防御机制下测试性能。

### 基线方法对比
与多种 SOTA 基线方法进行了全面比较：
- **通用 MIA**：Loss, Ref, LiRA.
- **LLM 专用**：Zlib, Neighbor, Min-K%, Min-K%++, ReCaLL.
- **Diffusion Model 专用**：NaiveAttacker, SecMI, PIA, PIAN.
- **分类模型专用**：Salem et al., Yeom et al., Watson et al., LDC-MIA.

---

## 3. 主要实验结果和性能指标

### 关键性能数据
ReproMIA 在所有任务和数据集上均显著优于现有方法，尤其是在低 FPR 场景下取得突破性进展：

#### 对于 LLMs (WikiMIA 基准)
- 平均相比第二名 ReCaLL：
  - **AUC 提升 5.25%**
  - **TPR@1%FPR 提升 10.68%**

#### 对于 Diffusion Models (Stable Diffusion)
- 平均相比第二名 SecMI：
  - **AUC 提升 3.70%**
  - **TPR@1%FPR 提升 12.40%**

#### 综合性能 (Table 1, 5, 6, 9)
| 模型类别 | 指标 | ReproMIA 性能 | 最佳基线性能 | 提升幅度 |
|---------|------|---------------|--------------|----------|
| LLMs (WikiMIA) | AUC | 92.05% | 85.94% (ReCaLL) | +6.11% |
| LLMs (WikiMIA) | TPR@1%FPR | 25.52% | 18.62% (ReCaLL) | +6.90% |
| Diffusion (Stable Diffusion) | TPR@1%FPR | 31.52% | 19.61% (SecMI) | +11.91% |

### 与基线方法的对比结果
- 在所有超过 10 个基准数据集上，ReproMIA 在 **AUC** 和 **TPR@Low FPR** 上均达到 SOTA。
- 在极具挑战性的 **MIMIR** 基准上，尽管多数基线接近随机猜测，ReproMIA 仍能取得显著提升（平均 AUC 提升 0.79%-1.53%）。
- ROC 曲线显示，ReproMIA 在低 FPR 区域的曲线下面积远超其他方法。

### 消融实验结果
- **Prompt 长度影响**（Table 14）：性能在 prompt 长度约为 80 时达到峰值，表明过长或过短都会轻微影响效果，但整体鲁棒。
- **Min-K% ratio 影响**（Table 15）：当 `K=0.2` 时性能最佳，验证了尾部概率选择的重要性。
- **Shadow 数据量影响**（Table 16）：即使只有约 50 个 shadow 样本，ReproMIA 也能达到良好性能，证明其在小数据场景下的可行性。
- **扰动幅度影响**（Table 17）：存在一个最优区间（如 4/255 到 64/255），过大或过小的扰动都会损害性能。

---

## 4. 关键结论和发现

### 主要发现
1. **Model Reprogramming 是有效的主动隐私探针**：通过主动注入可学习的输入变换，能够系统性地放大模型对训练数据的记忆效应，从而显著增强 MIA 信号。
2. **理论支持**：从三个角度解释了 ReproMIA 的有效性：
   - **Loss Landscape Curvature**：成员样本位于更平坦的损失区域，对扰动更敏感。
   - **Gradient Flow Analysis**：非成员梯度流主导优化过程，使扰动模式 $ \delta^* $ 更倾向于破坏非成员输出。
   - **Information Theory**：$ I(\text{membership}; M_\theta(x+\delta^*)) > I(\text{membership}; M_\theta(x)) $，即 reprogramming 显著提升了观测信号中的互信息。
3. **通用性强**：ReproMIA 成功实例化于 LLMs、Diffusion Models、图像分类和图神经网络，验证了其跨架构、跨模态的普适性。

### 方法的局限性
1. **依赖 Shadow Dataset**：需要已知成员标签的 shadow 数据来学习 $ \delta^* $。虽然不要求与目标数据同分布，但严重失配会影响迁移效果。
2. **依赖模型输出**：需要访问 logits 或 loss 值。若仅有硬标签（hard labels）或 top-k 输出，梯度信号会减弱，可能影响性能。
3. **防御适应性未知**：目前未探索防御方是否能针对 ReproMIA 设计自适应防御策略。

### 未来工作方向
- 将 ReproMIA 扩展到更受限的攻击场景，例如仅能获取硬标签或 API 限流的情况。
- 探索防御方如何检测和抵御此类主动式攻击。
- 研究 ReproMIA 在联邦学习、多模态模型等更复杂场景下的应用。
- 开发无需 shadow 数据的无监督或弱监督版本。

</details>

---

### 15. [AMShortcut: An Inference- and Training-Efficient Inverse Design Model for Amorphous Materials](https://arxiv.org/abs/2603.29812)

**Authors**: Yan Lin, Jonas A. Finkler, Tao Du, Jilin Hu, Morten M. Smedskjaer  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.29812v1  

#### Abstract
Amorphous materials are solids that lack long-range atomic order but possess complex short- and medium-range order. Unlike crystalline materials that can be described by unit cells containing few up to hundreds of atoms, amorphous materials require larger simulation cells with at least hundreds or o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AMShortcut: An Inference- and Training-Efficient Inverse Design Model for Amorphous Materials

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

传统用于**amorphous materials**（非晶材料）的**probabilistic generative models**（概率生成模型），尤其是基于**diffusion models**的方法，在**inference效率**和**training效率**上面临严重挑战：

- **Inference效率低**：需要数百步采样（sampling steps）才能生成结构准确的样本，导致推理时间长，难以实现高通量设计。
- **Training效率低**：不同属性组合需训练多个独立模型，维护成本高；而classifier guidance等技术受限于缺乏可微分的property classifier。

### 提出了什么新方法或新思路

本文提出 **AMShortcut**，一种兼具**inference高效**与**training高效**的生成模型，其核心创新包括：

#### ✅ **Learning Shortcuts in Diffusion Process**
- 基于**Material SDE**框架，引入“**shortcut learning**”机制，使模型能跨越大步长进行准确更新。
- 定义**ground truth shortcut**为连续时间区间内的平均drift，并通过**self-consistency loss**进行监督训练，使得模型在仅用**few steps**（甚至1步）即可生成高质量结构。

#### ✅ **Flexible Material Denoiser**
- 设计一个**统一训练、灵活推理**的denoiser网络，支持在训练时使用全部属性，而在推理时仅条件化部分目标属性。
- 对缺失属性采用**null property embedding**（从标准正态分布采样的向量），实现无需额外训练或分类器的“无条件生成”，显著提升训练灵活性。

### 相比现有方法的优势

| 维度 | AMShortcut优势 |
|------|----------------|
| **Inference Efficiency** | 推理速度提升高达 **99%**，仅需 **1–5 steps** 即可达到传统方法250步的结构精度 |
| **Training Efficiency** | 仅需**一次训练**即可支持任意属性子集的条件生成，避免重复训练 |
| **Performance** | 在结构保真度（RDF/ADF）和逆向设计准确性（MAE/RMSE）上均优于基线 |
| **Flexibility** | 支持属性外推（extrapolation）和多任务联合条件生成 |

---

## 2. 核心实验方法和设置

### 使用的数据集

实验在三个非晶材料数据集上进行，涵盖单元素、固定组成和多元素系统：

| 数据集 | 描述 |
|--------|------|
| **a-Si**（amorphous silicon） | 10,000个样本，每样本256个Si原子，用于评估**结构准确性** |
| **a-SiO₂**（amorphous silica） | 6,000个样本，原子数80–250，性质由结构和密度决定，用于评估**逆向设计能力** |
| **MEG**（multi-element glass） | 9,027个样本，含11种元素（如Si, P, Li, O等），约800原子/样本，性质由化学组成主导 |

所有数据集通过**classical molecular dynamics**（LAMMPS + ASE）模拟生成。

### 实验设置和评估指标

#### 评估维度

| 类型 | 指标 | 说明 |
|------|------|------|
| **Structural Accuracy** | **RDF RMSD**, **ADF RMSD** | 衡量生成结构与真实结构在原子距离和键角分布上的偏差 |
| **Inverse Design Performance** | **MAE**, **RMSE**, **MAPE** | 衡量生成样本的属性与目标属性之间的误差 |
| **Efficiency** | **Generation Time** | 每生成一定数量样本所需的时间（秒） |

#### 采样步数范围
- 测试了 `ns = 1, 2, 3, 4, 5, 10, 25, 50, 100, 250` 步下的性能

### 基线方法对比

| 方法 | 类型 | 是否针对非晶优化 |
|------|------|------------------|
| **Material ODE** | 本文提出的确定性扩散变体 | 是 |
| **Material SDE** | 本文提出的随机扩散变体 | 是 |
| **CDVAE** | 扩散VAE，用于晶体材料 | 否 |
| **MatterGen** | 扩散模型，支持周期表级生成 | 否 |
| **Graphite** | 针对非晶碳的光谱引导扩散模型 | 是，但应用有限 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 结构准确性（a-Si 数据集）

| 方法 | 最佳RDF RMSD | 达到方式 | 时间消耗 |
|------|--------------|----------|----------|
| **AMShortcut** | **0.02513** | 仅 **1步** | 2.2分钟 |
| Material SDE | 0.01905 | 需 **250步** | 3.6小时 |
| CDVAE | 0.07572 | 250步 | 3.6小时 |

> ✅ **AMShortcut用1步达到接近最优结构精度，耗时仅为基线的1.16%**

#### 📈 逆向设计性能（a-SiO₂ 和 MEG 数据集）

##### a-SiO₂ – Shear Modulus（目标外推至训练分布之外）

| 方法 | ns=10（MAE/RMSE/MAPE） | ns=250（MAE/RMSE/MAPE） |
|------|-------------------------|--------------------------|
| **AMShortcut (All)** | **2.67 / 3.41 / 13.0%** | 3.03 / 3.60 / 13.0% |
| Material SDE (Target) | 19.17 / 20.31 / 65.8% | 3.41 / 4.21 / 15.6% |

> ✅ **AMShortcut在10步内即超越其他方法250步的表现**

##### MEG – Young’s Modulus & Li Concentration

| 方法 | ns=10（Young’s MAE） | ns=250（Young’s MAE） |
|------|-----------------------|------------------------|
| **AMShortcut (All)** | **8.20 GPa** | 8.70 GPa |
| Material SDE (Target) | 35.28 GPa | 9.73 GPa |

> ✅ **AMShortcut在少量步骤下表现更优，且随步数增加性能稳定**

### 与基线方法的对比结果

- **结构保真度**：
  - AMShortcut在 **1–5步** 内即可匹配甚至超越基线模型在 **250步** 的RDF/ADF表现。
  - ADF RMSD在1步时已与Material SDE在250步时相当，仅耗时 **1.16%**。

- **逆向设计精度**：
  - 在 **外推区域**（目标属性超出训练分布），AMShortcut仍能生成符合目标的样本，显示强泛化能力。
  - 在仅使用 **subset of properties** 条件化时，AMShortcut (All) 性能接近专门训练的 (Target) 模型。

### 消融实验结果（隐含分析）

虽然未明确列出消融表，但从以下对比可得关键结论：

| 对比项 | 发现 |
|--------|------|
| **AMShortcut vs Material SDE/OED** | 显示“shortcuts”机制极大提升采样效率 |
| **(All) vs (Target) 模型** | 表明flexible denoiser无需重新训练即可保持高性能 |
| **ns=1 vs ns=250** | 验证了学习到的long-step更新是有效的，而非简单降噪 |

---

## 4. 关键结论和发现

### 主要发现

1. **Diffusion models for amorphous materials 可以极高效地运行**  
   → 通过引入**shortcut learning**，可在**1–5步**内完成高质量生成，突破传统多步迭代瓶颈。

2. **单一模型可支持任意属性组合的逆向设计**  
   → **flexible material denoiser + null property embedding** 实现了“train once, infer flexibly”，大幅提升实用性。

3. **AMShortcut 在结构准确性和属性控制上均优于现有方法**  
   → 不仅速度快，而且精度更高，尤其在**外推场景**下表现稳健。

4. **高通量生成成为可能**  
   → 推理时间减少 **99%**，意味着在相同时间内可探索更大设计空间，加速新材料发现。

### 方法的局限性

- **无法解决annealed structure生成的根本限制**  
  → 当前所有diffusion models在生成低温退火结构时存在固有偏差，需结合**physics-guided HMC refinement**，但这会牺牲速度。
  
- **Single-step性能依赖多步随机性积累**  
  → 在逆向设计任务中，**ns=1** 时性能略有下降，因flexible conditioning依赖多步中的噪声演化。

- **尚未扩展到动态性质或功能预测**  
  → 当前仅关注静态结构与基本力学/组成性质，未涉及电子、光学等功能特性。

### 未来工作方向

1. **结合物理先验改进结构能量质量**  
   → 将**Hamiltonian Monte Carlo** 或 **energy-based refinement** 与AMShortcut结合，在保持速度的同时提升结构合理性。

2. **扩展至功能导向的逆向设计**  
   → 引入电导率、热导率、离子迁移率等功能属性作为条件输入。

3. **构建大规模非晶材料生成平台**  
   → 利用AMShortcut的高效性，建立**high-throughput inverse design pipeline**，用于电池电解质、光学玻璃等应用场景。

4. **探索zero-shot属性控制**  
   → 进一步提升flexible denoiser的泛化能力，实现对未见属性组合的推理支持。

--- 

> **总结一句话**：  
> **AMShortcut 通过“学习跳跃”和“灵活去噪”，实现了非晶材料逆向设计的速度与精度双重突破，为高通量材料发现提供了全新工具。**

</details>

---

### 16. [ASI-Evolve: AI Accelerates AI](https://arxiv.org/abs/2603.29640)

**Authors**: Weixian Xu, Tiantian Mi, Yixiu Liu, Yang Nan, Zhimeng Zhou, Lyumanshan Ye, Lin Zhang, Yu Qiao, Pengfei Liu  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.29640v1  

#### Abstract
Can AI accelerate the development of AI itself? While recent agentic systems have shown strong performance on well-scoped tasks with rapid feedback, it remains unclear whether they can tackle the costly, long-horizon, and weakly supervised research loops that drive real AI progress. We present ASI-E...

---

### 17. [Optimizing Donor Outreach for Blood Collection Sessions: A Scalable Decision Support Framework](https://arxiv.org/abs/2603.29643)

**Authors**: Andr\'e Carneiro, Pedro T. Monteiro, Rui Henriques  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.29643v1  

#### Abstract
Blood donation centers face challenges in matching supply with demand while managing donor availability. Although targeted outreach is important, it can cause donor fatigue via over-solicitation. Effective recruitment requires targeting the right donors at the right time, balancing constraints with ...

---

### 18. [ShapE-GRPO: Shapley-Enhanced Reward Allocation for Multi-Candidate LLM Training](https://arxiv.org/abs/2603.29871)

**Authors**: Rui Ai, Yu Pan, David Simchi-Levi, Chonghuan Wang  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.29871v1  

#### Abstract
In user-agent interaction scenarios such as recommendation, brainstorming, and code suggestion, Large Language Models (LLMs) often generate sets of candidate recommendations where the objective is to maximize the collective utility of the entire set rather than individual candidates independently. H...

---

### 19. [Mathematical Foundations of Modeling ETL Process Chains](https://arxiv.org/abs/2603.29877)

**Authors**: Levin Maier, Lucas Schulze, Robert Lilow, Lukas Hahn, Nikola Krasowski, Arnulf Barth, Sebastian Gaebel, Ferdi G\"uran, Oliver Hanau, Giovanni Wagner, Falk Borgmann, Oleg Arenz, Jan Peters  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.29877v1  

#### Abstract
Extract-Transform-Load (ETL) processes are core components of modern data processing infrastructures. The throughput of processed data records can be adjusted by changing the amount of allocated resources, i.e.~the number of parallel processing threads for each of the three ETL phases, but also depe...

---

### 20. [Causality-inspired Federated Learning for Dynamic Spatio-Temporal Graphs](https://arxiv.org/abs/2603.29384)

**Authors**: Yuxuan Liu, Wenchao Xu, Haozhao Wang, Zhiming He, Zhaofeng Shi, Chongyang Xu, Peichao Wang, Boyuan Zhang  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.29384v1  

#### Abstract
Federated Graph Learning (FGL) has emerged as a powerful paradigm for decentralized training of graph neural networks while preserving data privacy. However, existing FGL methods are predominantly designed for static graphs and rely on parameter averaging or distribution alignment, which implicitly ...

---

### 21. [Big2Small: A Unifying Neural Network Framework for Model Compression](https://arxiv.org/abs/2603.29768)

**Authors**: Jing-Xiao Liao, Haoran Wang, Tao Li, Daoming Lyu, Yi Zhang, Chengjun Cai, Feng-Lei Fan  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.29768v1  

#### Abstract
With the development of foundational models, model compression has become a critical requirement. Various model compression approaches have been proposed such as low-rank decomposition, pruning, quantization, ergodic dynamic systems, and knowledge distillation, which are based on different heuristic...

---

### 22. [REFINE: Real-world Exploration of Interactive Feedback and Student Behaviour](https://arxiv.org/abs/2603.29142)

**Authors**: Fares Fawzi, Seyed Parsa Neshaei, Marta Knezevic, Tanya Nazaretsky, Tanja K\"aser  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.29142v1  

#### Abstract
Formative feedback is central to effective learning, yet providing timely, individualised feedback at scale remains a persistent challenge. While recent work has explored the use of large language models (LLMs) to automate feedback, most existing systems still conceptualise feedback as a static, one...

---

### 23. [Route-Induced Density and Stability (RIDE): Controlled Intervention and Mechanism Analysis of Routing-Style Meta Prompts on LLM Internal States](https://arxiv.org/abs/2603.29206)

**Authors**: Dianxing Zhang, Gang Li, Sheng Li  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.29206v1  

#### Abstract
Routing is widely used to scale large language models, from Mixture-of-Experts gating to multi-model/tool selection. A common belief is that routing to a task ``expert'' activates sparser internal computation and thus yields more certain and stable outputs (the Sparsity--Certainty Hypothesis). We te...

---

### 24. [AgentFixer: From Failure Detection to Fix Recommendations in LLM Agentic Systems](https://arxiv.org/abs/2603.29848)

**Authors**: Hadar Mulian, Sergey Zeltyn, Ido Levy, Liane Galanti, Avi Yaeli, Segev Shlomov  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.29848v1  

#### Abstract
We introduce a comprehensive validation framework for LLM-based agentic systems that provides systematic diagnosis and improvement of reliability failures. The framework includes fifteen failure-detection tools and two root-cause analysis modules that jointly uncover weaknesses across input handling...

---

### 25. [Concept Training for Human-Aligned Language Models](https://arxiv.org/abs/2603.29123)

**Authors**: Christine Zhang, Dan Jurafsky, Chen Shani  
**Category**: cs.CL  
**Published**: 2026-04-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.29123v1  

#### Abstract
The next-token prediction (NTP) objective trains language models to predict a single continuation token at each step. In natural language, however, a prefix can be continued in many valid ways, and even similar meanings may differ in surface form. For example, the sentence ``this website is safe to ...

---

### 26. [SNEAK: Evaluating Strategic Communication and Information Leakage in Large Language Models](https://arxiv.org/abs/2603.29846)

**Authors**: Adar Avsian, Larry Heck  
**Category**: cs.CL  
**Published**: 2026-04-01  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.29846v1  

#### Abstract
Large language models (LLMs) are increasingly deployed in multi-agent settings where communication must balance informativeness and secrecy. In such settings, an agent may need to signal information to collaborators while preventing an adversary from inferring sensitive details. However, existing LL...

---

### 27. [C-TRAIL: A Commonsense World Framework for Trajectory Planning in Autonomous Driving](https://arxiv.org/abs/2603.29908)

**Authors**: Zhihong Cui, Haoran Tang, Tianyi Li, Yushuai Li, Peiyuan Guan, Amir Taherkordi, Tor Skeie  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.29908v1  

#### Abstract
Trajectory planning for autonomous driving increasingly leverages large language models (LLMs) for commonsense reasoning, yet LLM outputs are inherently unreliable, posing risks in safety-critical applications. We propose C-TRAIL, a framework built on a Commonsense World that couples LLM-derived com...

---

### 28. [Structured Intent as a Protocol-Like Communication Layer: Cross-Model Robustness, Framework Comparison, and the Weak-Model Compensation Effect](https://arxiv.org/abs/2603.29953)

**Authors**: Peng Gang  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.29953v1  

#### Abstract
How reliably can structured intent representations preserve user goals across different AI models, languages, and prompting frameworks? Prior work showed that PPS (Prompt Protocol Specification), a 5W3H-based structured intent framework, improves goal alignment in Chinese and generalizes to English ...

---

### 29. [Dual Perspectives in Emotion Attribution: A Generator-Interpreter Framework for Cross-Cultural Analysis of Emotion in LLMs](https://arxiv.org/abs/2603.29077)

**Authors**: Aizirek Turdubaeva, Uichin Lee  
**Category**: cs.CL  
**Published**: 2026-04-01  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.29077v1  

#### Abstract
Large language models (LLMs) are increasingly used in cross-cultural systems to understand and adapt to human emotions, which are shaped by cultural norms of expression and interpretation. However, prior work on emotion attribution has focused mainly on interpretation, overlooking the cultural backg...

---

### 30. [ZEUS: An Efficient GPU Optimization Method Integrating PSO, BFGS, and Automatic Differentiation](https://arxiv.org/abs/2603.28770)

**Authors**: Dominik Soos (Old Dominion University), Marc Paterno (Old Dominion University), Desh Ranjan (Old Dominion University), Mohammad Zubair (Old Dominion University)  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.28770v1  

#### Abstract
We introduce a novel, efficient computational method, ZEUS, for numerical optimization, and provide an open-source implementation. It has four key ingredients: (1) particle swarm optimization (PSO), (2) the use of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) method, (3) automatic differentiation (AD)...

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
