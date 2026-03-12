# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-12 06:36:38 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Surrogate models for nuclear fusion with parametric Shallow Recurrent Decoder Networks: applications to magnetohydrodynamics](https://arxiv.org/abs/2603.10678)

**Authors**: M. Lo Verso, C. Introini, E. Cervi, L. Savoldi, J. N. Kutz, A. Cammi  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2603.10678v1  

#### Abstract
Magnetohydrodynamic (MHD) effects play a key role in the design and operation of nuclear fusion systems, where electrically conducting fluids (such as liquid metals or molten salts in reactor blankets) interact with magnetic fields of varying intensity and orientation, which affect the resulting flo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Surrogate models for nuclear fusion with parametric Shallow Recurrent Decoder Networks: applications to magnetohydrodynamics*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**核聚变系统中磁流体动力学（MHD）模拟计算成本高昂**的问题，尤其是在多查询、参数化或实时控制场景下，传统高保真数值模型（Full-Order Models, FOMs）难以满足效率需求。具体而言：
- MHD 模型涉及强非线性、多物理场耦合的偏微分方程组，求解耗时。
- 在聚变反应堆设计与运行中，需快速预测不同磁场强度下的流动行为（如液态金属包层中的铅锂流），但全尺寸仿真代价过大。

### 🚀 提出的新方法与新思路
提出了一种**基于 SVD 压缩与 SHallow REcurrent Decoder (SHRED) 网络结合的数据驱动代理模型框架**，用于从稀疏传感器测量中重建完整的 MHD 状态场（velocity, pressure, temperature）。

#### 创新点包括：
1. **首次将 SHRED 应用于核聚变相关的 MHD 物理问题**，特别是导电流体（如液态金属）在复杂几何与热梯度下的流动建模。
2. 引入**参数化扩展的 SHRED 架构**，使其能够泛化到训练集中未见的磁场强度（Bo），实现跨参数状态重构。
3. 采用“压缩学习”范式：先通过 **SVD 对高维快照进行降维**，再在低维潜空间训练 SHRED，显著降低数据量和计算开销。
4. 实现**仅用三个温度传感器的时间序列输入**，即可准确重建速度、压力和温度的完整时空演化场。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **数据效率** | 所需训练数据少，且可在普通笔记本电脑上完成训练（约10分钟/模型）。 |
| **传感器依赖性低** | 对传感器位置不敏感（agnostic to sensor placement），随机布置仍能保持高精度。 |
| **泛化能力** | 可推广至训练范围外的参数值（如更高/更低的 Bo），适用于未知工况预测。 |
| **可解释性与可靠性** | 基于 Takens' embedding 定理，理论基础扎实；网络浅层结构（<10³ 参数）提升透明度，适合安全关键应用（如核工程）。 |
| **多物理场恢复能力** | 仅凭单一可观测量（temperature）即可间接估计其他不可测变量（如 velocity）。 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **来源**：使用 OpenFOAM 的 `magnetoHDFoam` 求解器生成的二维阶梯通道内可压缩铅锂（PbLi）MHD 流动数据。
- **参数配置**：
  - 考虑 $ N_p = 19 $ 种不同的垂直磁场强度 $ B_o \in [0.01, 0.5]~\text{T} $
  - 每个案例模拟 3 秒，每 0.025 秒保存一个时间步快照（共 $ N_t = 120 $）
  - 总计约 2280 个时空快照（每个场：T, u, p）
- **物理现象涵盖**：
  - 热梯度引起的浮力效应
  - 阶梯障碍导致的湍流发展
  - 不同 $ B_o $ 下 Lorentz 力对流动的抑制（laminarization）

### ⚙️ 实验设置
- **预处理**：
  - 使用 **SVD** 对所有场的快照矩阵进行压缩，保留前 $ r=20 $ 个主模态（能量保留 >99.9%）
  - 输入为归一化的温度时间序列
- **网络架构**：
  - **LSTM 编码器**：2 层，每层 64 神经元 → 学习时间动态
  - **Shallow Decoder Network (SDN)**：2 层（350, 400 神经元）→ 映射回潜空间系数
  - 输出为 SVD 时间系数，最终通过 SVD 逆变换还原全场
- **训练策略**：
  - 数据划分：训练集 ~73.7%，验证集 ~15.8%，测试集 ~10.5%
  - 更密集采样低 $ B_o $ 区间（因动态更复杂）
- **评估方式（Ensemble Mode）**：
  - 构造 **30 组随机分布的三传感器组合**（共 90 个点）
  - 每组训练独立 SHRED 模型，最后取输出均值作为最终预测
  - 分析标准差以评估鲁棒性

### 📈 评估指标
- **相对 $ L_2 $ 误差**：
  $$
  \epsilon_w = \frac{\|w_{\text{FOM}} - w_{\text{SHRED}}\|_2}{\|w_{\text{FOM}}\|_2}
  $$
- **全局平均误差 + 标准差（whiskers）**
- **残差图（FOM vs. Reconstruction）**
- **时间演化曲线对比**（空间平均量随时间变化）

### ❌ 基线方法对比
本文未直接与其他 ML 或 ROM 方法进行定量比较，而是强调其相对于以下类别的优势：
- **传统深度学习模型**（如 CNN-RNN）：需要大量数据、GPU 训练、难部署于边缘设备
- **经典数据同化技术**（如 Kalman Filter）：优化过程耗时，不适合实时
- **纯数据驱动 ROM**（如 DMD, Autoencoders）：缺乏物理嵌入机制，泛化能力弱

> 注：作者指出这是 SHRED **首次应用于 MHD 导电流体**，因此更多是展示可行性而非横向对比。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据
| 测试案例 | 场量 | 最大相对误差 | 平均相对误差（全局） |
|--------|-------|----------------|------------------------|
| $ B_o = 0.06~\text{T} $（低场） | Temperature | < 3% | ~2.5% |
| | Velocity | < 6% | ~4.8% |
| | Pressure | < 6% | ~4.6% |
| $ B_o = 0.3~\text{T} $（高场） | All fields | < 2% | < 1.8% |

- **误差随时间增长缓慢**，在 $ t=3~\text{s} $ 内保持低位
- **高磁场下误差更低**：由于 Lorentz 力抑制小尺度涡旋，流动趋于层流化，更易建模

### 📉 与基线预期对比（定性）
尽管无显式基线对比实验，但文中明确指出：
- 相比需数百传感器的传统 ROM，SHRED **仅需 3 个传感器即达高精度**
- 相比需高性能计算平台的 FOM（单次仿真约 20 分钟），SHRED 推理时间 **<1 秒**
- 相比需精细调参的深度网络，SHRED **几乎无需超参数调整**

### 🔬 消融实验（隐含分析）
虽未设正式消融实验，但以下分析体现了关键组件作用：
1. **SVD 压缩有效性**：
   - $ r=20 $ 已捕获 >99.9% 能量
   - 小误差表明未丢失关键动态特征
2. **传感器数量与位置无关性**：
   - 30 组不同配置的标准差极低（见 Figure 6 & 7 第四行）
   - 验证了“agnostic to sensor placement”的核心特性
3. **跨参数泛化能力**：
   - 测试案例 $ B_o=0.06~\text{T} $ 和 $ 0.3~\text{T} $ 均不在训练集中密集覆盖
   - 仍实现高精度重建 → 表明模型学会的是“磁场如何影响流动”的通用规律

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **SHRED 能高效、准确地重建 MHD 全场状态**，即使仅基于三个温度传感器的测量。
2. 模型具备出色的**跨参数泛化能力**，可预测训练集中未充分覆盖甚至未出现的磁场强度下的流动行为。
3. **对传感器位置完全鲁棒**，30 组随机配置的结果高度一致，标准差可忽略。
4. 高磁场条件下重建效果更好，因流动被 Lorentz 力稳定化，动态更简单。
5. 整个训练可在消费级硬件完成（Intel i7 笔记本），推理近乎瞬时，具备**实时监控潜力**。

### ⚠️ 方法的局限性
1. **依赖高质量的离线 FOM 数据集**：若初始仿真不准，则代理模型也无法纠正偏差。
2. 当前仅验证于二维简化几何（阶梯通道），尚未拓展至真实三维包层结构。
3. 仅考虑静态、均匀垂直磁场 $ B_o $，未涉及时变或任意方向磁场。
4. 温度作为唯一输入的前提是其与其他场存在强相关性，在某些极端工况下可能失效。

### 🔮 未来工作方向
1. **向三维复杂几何扩展**：应用于更接近实际托卡马克包层的真实结构。
2. **引入多参数输入**：同时处理 $ B_o $ 强度、方向、入口速度等联合变化。
3. **融合时变磁场与瞬态响应建模**：研究脉冲或扰动磁场下的动态响应。
4. **集成至数字孪生系统**：用于实时监测、故障诊断与闭环控制。
5. **结合物理约束损失函数**：进一步增强模型的物理一致性与外推能力。

---

> **总结一句话**：  
> 本论文成功将 **SHRED + SVD** 框架引入核聚变 MHD 建模领域，展示了其以极低成本实现高精度、强鲁棒、可泛化的全场状态重建的巨大潜力，为未来聚变反应堆的实时感知与智能控制提供了全新的数据驱动解决方案。

</details>

---

### 2. [FRIEND: Federated Learning for Joint Optimization of multi-RIS Configuration and Eavesdropper Intelligent Detection in B5G Networks](https://arxiv.org/abs/2603.10977)

**Authors**: Maria Lamprini A. Bartsioka, Ioannis A. Bartsiokas, Anastasios K. Papazafeiropoulos, Maria A. Seimeni, Dimitra I. Kaklamani, Iakovos S. Venieris  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2603.10977v1  

#### Abstract
As wireless systems evolve toward Beyond 5G (B5G), the adoption of cell-free (CF) millimeter-wave (mmWave) architectures combined with Reconfigurable Intelligent Surfaces (RIS) is emerging as a key enabler for ultra-reliable, high-capacity, scalable, and secure Industrial Internet of Things (IIoT) c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对 **Beyond 5G (B5G)** 网络中日益复杂的无线安全挑战，特别是**在分布式、高密度工业物联网 (IIoT)** 场景下如何有效检测物理层窃听者（eavesdroppers）这一难题。传统加密机制难以满足低延迟、高可扩展性的需求，而集中式机器学习方法又面临隐私泄露和通信开销大的问题。

此外，在 **cell-free mmWave RIS-assisted** 架构中，如何联合优化多 RIS 配置以增强合法链路并抑制窃听路径，同时实现智能攻击检测，是一个尚未充分探索的复杂系统工程问题。

### 提出的新方法与创新思路
作者提出了一种名为 **FRIEND** 的新型框架，其核心创新点包括：

- **Federated Learning + RIS 联合优化架构**：首次将 **Federated Learning (FL)** 引入到 RIS 辅助的 cell-free mmWave 网络中，用于协同训练一个分布式的 **Deep Convolutional Neural Network (DCNN)** 模型，仅基于本地采集的 **Channel State Information (CSI)** 数据进行 eavesdropper 检测，无需共享原始数据，保障用户隐私。
  
- **多 RIS 协同配置与安全感知设计**：在网络中部署多个 RIS 单元，并将其反射系数（phase shifts）作为可调参数，参与联合优化过程，动态塑造传播环境，提升合法用户的信号质量，同时削弱窃听者的接收能力。

- **Early-exit 机制集成**：在 DCNN 模型中引入 early-exit 结构，在中间层设置辅助分类器，允许高置信度样本提前退出推理流程，显著降低计算延迟和资源消耗，适用于边缘设备受限场景。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **隐私保护** | 相比集中式 ML，FL 避免了原始 CSI 数据上传，符合 GDPR 等隐私法规要求。 |
| **可扩展性与鲁棒性** | 分布式训练适应大规模 IIoT 设备接入，单个节点失效不影响全局模型收敛。 |
| **安全性增强** | RIS 主动调控信道特性，从物理层提升 secrecy rate，形成“主动防御”能力。 |
| **效率提升** | Early-exit 机制减少平均推理时间达 35%-45%，兼顾精度与实时性。 |

---

## 2. 核心实验方法和设置

### 数据集
- **自建仿真数据集**：由于真实 B5G IIoT CSI 数据难以获取，作者使用 **MATLAB R2025b** 构建了一个符合 **3GPP TR 38.901 和 TR 38.843** 规范的 cell-free mmWave RIS-assisted 网络模拟器。
- **数据生成方式**：
  - 包含 direct link 与 RIS-reflected link 的复合信道建模；
  - OFDM 波形下的 Sounding Reference Signals (SRS) 传输；
  - 多径衰落、遮挡效应、大/小尺度衰落等现实因素被纳入；
  - 最终将复数域 channel matrix 转换为 **CSI 图像 (CSI image)** 输入模型。
- **标签定义**：
  - 合法用户：label = 0
  - 窃听者：label = 1
- **附加特征**：UE 发射功率、地理位置坐标（UE, AP, RIS）、服务节点信息等融合输入。

### 实验设置
| 参数 | 设置值 |
|------|--------|
| 载频 | 28 GHz (FR2) |
| 带宽 | 400 MHz |
| 子载波间隔 | 120 kHz |
| AP 数量 | 18 |
| RIS 数量 | 3，尺寸为 10×20 元素 |
| UE 总数 | 500（70% 合法用户，30% 窃听者） |
| 天线配置 | AP: 32 antennas；UE: single-antenna |
| 移动速度 | 3 km/h |
| 噪声水平 | 5 dB |
| SRS 符号数 | 12 |
| Resource Blocks | 60 per transmission |
| FL 客户端数量 | 3（选择位置中心且靠近 RIS 的 AP） |
| 早退置信度阈值 | 55%, 70% |
| 训练/测试划分 | 80%/20% |
| 仿真轮次 | 100 Monte Carlo trials（不同 RIS 相位组合） |

### 评估指标
- **分类性能指标**：
  - Accuracy, Precision, Recall, F1-score
- **通信安全指标**：
  - Secrecy Rate (SR)
  - Average Secrecy Rate (ASR)
- **系统效率指标**：
  - 推理时间（Inference Time）
  - Early-exit rate
- **训练策略**：
  - 使用 **TensorFlow Federated** 实现 FL；
  - 采用标准 **Federated Averaging (FedAvg)** 算法聚合模型权重；
  - Hyperparameter tuning via grid search + cross-validation。

### 基线方法对比
- **Centralized non-RIS-aided 方法**：来自文献 [22] 的集中式 ML 框架（如 RF, LSTM, DCNN），无 RIS 支持；
- **Non-ML RIS 方法**：基于真实标签计算的理想 SR 曲线（upper bound）；
- 不同 RIS phase configuration 下的表现比较（Phase ID 4 vs. 54 vs. 89 等）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）分类性能（Figure 3）
- **Accuracy**：范围 0.71–0.93，四分位距集中在 0.84 左右；
- **Precision**：保持高位，表明误报率低；
- **Recall**：最高达 0.95，说明对窃听者的检出能力强；
- **F1-score**：整体稳健，验证了模型在 precision 与 recall 之间的良好平衡；
- **False Negative 少**：高 recall 对 PLS 至关重要，避免漏检导致安全漏洞。

> ⚠️ 变异性来源：部分窃听者与合法用户的 CSI 特征高度相似（尤其位于 RIS 附近时），造成识别困难。

#### （2）Early-exit 效果（Figure 4）
- 在 **CL=70%** 时，推理时间减少约 **35%**；
- 在 **CL=55%** 时，推理时间减少高达 **45%**；
- 精度仍维持在 **>0.83**，显示 early-exit 在轻量化与准确性之间取得良好折衷；
- 显著提高边缘 AP 的响应速度，支持实时检测。

#### （3）通信安全性能（Figure 5）
- 所有 RIS 配置下，随着 **Legitimate-to-Eavesdropper ratio (L/E)** 增加，ASR 提升；
- **最优 RIS Phase (ID 89)**：在高 L/E 比例下 ASR 超过 **20 bps/Hz**；
- **最差 RIS Phase (ID 54)**：ASR 明显下降，说明不当相位可能反而增强窃听；
- **与 Non-ML 方法对比**：RIS Phase ID 4 的 SR 曲线与真实标签曲线高度吻合，证明模型捕捉到了真实的物理层模式；
- **相比 baseline [22]**：最佳 RIS 配置下，**ASR 提升约 30%**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Federated Learning 可有效应用于 RIS-assisted cell-free mmWave 网络中的 eavesdropper detection**，在不牺牲隐私的前提下实现高性能分类（accuracy ~95%）；
2. ✅ **RIS 不仅是信道增强工具，更是主动安全组件**：通过优化反射矩阵，可在物理层主动压制窃听链路，显著提升 **Secrecy Rate**；
3. ✅ **Early-exit 机制显著降低推理延迟**（最高减 45%），适合资源受限的边缘 AP 部署，实现高效能-低功耗权衡；
4. ✅ **RIS 相位配置对安全性能影响巨大**：需联合优化而非固定设置，否则可能导致反向增益（如 Phase ID 54）；
5. ✅ **所提 FRIEND 框架实现了 ~30% 的 ASR 提升**，优于现有的非 RIS 辅助集中式方法。

### 方法的局限性
- ❌ **依赖高质量 CSI 估计**：实际环境中 CSI 获取受噪声、反馈延迟、硬件损伤等因素影响，可能降低模型性能；
- ❌ **RIS 控制信令开销未建模**：动态调整大量 RIS 元素需要额外控制链路，可能引入新的瓶颈；
- ❌ **假设窃听者行为静态**：当前模型未考虑智能对抗型攻击者（adaptive eavesdroppers）；
- ❌ **仿真环境理想化**：尚未在真实硬件平台或实测数据上验证（R8 为 future work）；
- ❌ **客户端数量较少**（仅 3 个 FL clients），大规模网络下的收敛性有待进一步研究。

### 未来工作方向
1. 🔧 **Real Data Evaluation**：引入真实 IIoT 场景下的测量数据，完成 Requirement R8；
2. 📈 **Scalable RIS Configurations**：研究更大规模或多形态 RIS 部署（对应 Requirement R7）；
3. ☁️ **Offloading + Early-exit 协同优化**：探索将 early-exit 决策与边缘卸载结合，进一步节省本地计算资源；
4. 🤖 **Adversarial Learning for Robustness**：引入对抗训练或 DRL 框架，应对更复杂的恶意行为；
5. 🔐 **End-to-end Privacy-Security Co-design**：结合差分隐私（DP-FedAvg）等技术，进一步强化联邦学习的安全边界。

--- 

> **总结一句话**：  
> FRIEND 框架成功地将 **Federated Learning**、**multi-RIS 控制** 与 **physical-layer eavesdropper detection** 融为一体，在保障隐私的同时实现了 **~30% 的 secrecy rate 提升** 和高效的边缘推理，为 B5G/6G 工业网络提供了可扩展、智能化的安全解决方案。

</details>

---

### 3. [GLM-OCR Technical Report](https://arxiv.org/abs/2603.10910)

**Authors**: Shuaiqi Duan, Yadong Xue, Weihan Wang, Zhe Su, Huan Liu, Sheng Yang, Guobing Gan, Guo Wang, Zihan Wang, Shengdong Yan, Dexin Jin, Yuxuan Zhang, Guohong Wen, Yanfeng Wang, Yutao Zhang, Xiaohan Zhang, Wenyi Hong, Yukuo Cen, Da Yin, Bin Chen, Wenmeng Yu, Xiaotao Gu, Jie Tang  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.10910v1  

#### Abstract
GLM-OCR is an efficient 0.9B-parameter compact multimodal model designed for real-world document understanding. It combines a 0.4B-parameter CogViT visual encoder with a 0.5B-parameter GLM language decoder, achieving a strong balance between computational efficiency and recognition performance. To a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GLM-OCR Technical Report 核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
GLM-OCR 旨在解决当前文档理解系统在**实际部署场景中的三大矛盾**：
- **高性能 vs 高效率**：大型多模态模型（如 Qwen-VL、Gemini）虽然性能强，但参数量大、推理慢、内存消耗高，难以在边缘设备或高并发场景部署。
- **端到端生成 vs 结构化输出可靠性**：传统自回归解码逐token生成，在处理表格、公式等长结构文本时效率低且易出现“断裂标签”等问题。
- **通用能力 vs 任务专用需求**：许多模型未针对 OCR 这类确定性任务进行优化，导致资源浪费和性能不稳定。

### 提出的新方法与创新
1. **Multi-Token Prediction (MTP)**  
   - 在训练和推理阶段同时预测多个 token，显著提升解码吞吐量。
   - 引入**参数共享机制**控制额外显存开销，实测平均每个解码步生成 5.2 个 token，带来约 **50% 的吞吐提升**。
   - 特别适用于表格、Markdown 等具有局部依赖性的结构化输出。

2. **两阶段系统架构（Layout + Parallel Recognition）**
   - 第一阶段使用 PP-DocLayout-V3 进行布局分析，将文档划分为语义区域（段落、表格、公式等）。
   - 第二阶段对各区域并行调用 GLM-OCR Core 进行识别，提高鲁棒性和处理效率。

3. **统一的生成框架支持多任务**
   - 同一模型支持 **Document Parsing** 和 **Key Information Extraction (KIE)** 两类任务。
   - 通过不同 prompt 控制输出格式（Markdown/JSON），实现参数高效与跨任务知识迁移。

### 相比现有方法的优势
| 维度 | GLM-OCR 优势 |
|------|--------------|
| **模型规模** | 仅 **0.9B 参数**，远小于主流 VLMs（如 Qwen3-VL-235B、Gemini-3 Pro） |
| **推理效率** | 支持 vLLM/SGLang/Ollama 等现代推理框架，适合边缘部署 |
| **结构化输出质量** | MTP + RL 优化减少标签断裂、重复等问题 |
| **部署成本** | MaaS API 定价为 **0.2 RMB / 百万 tokens**，约为传统方案的 1/10 |

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 公共基准（Public Benchmarks）
| 类型 | 数据集 | 任务描述 |
|------|--------|----------|
| 文档解析 | **OmniDocBench v1.5** | 综合评测文档解析能力（文本、公式、表格、阅读顺序） |
| 文本识别 | **OCRBench (Text)** | 多样化场景下的文本识别准确率 |
| 公式识别 | **UniMERNet** | 数学表达式识别（LaTeX 输出） |
| 表格结构恢复 | **PubTabNet**, **TEDS_TEST** | 表格结构相似度（TEDS Score） |
| 关键信息抽取 | **Nanonets-KIE**, **Handwritten-Forms** | 结构化字段提取（F1 Score） |

#### 自研工业级测试集（In-House Benchmarks）
| 场景 | 描述 |
|------|------|
| Code Document Parsing | 技术文档中代码块识别 |
| Real-world Table Extraction | 自然场景下复杂表格识别 |
| Handwritten Text Recognition | 手写体文本识别 |
| Multilingual OCR | 中英法西俄德日韩八语种混合识别 |
| Seal Recognition | 图章检测与内容识别 |
| Receipt KIE | 发票关键字段提取（JSON Schema） |

### 实验设置与评估指标
| 项目 | 设置说明 |
|------|----------|
| **主干结构** | CogViT (0.4B) + GLM (0.5B)，总参数量 0.9B |
| **训练流程** | 四阶段训练：<br>1. 视觉编码器预训练<br>2. 多模态联合预训练（含 MTP）<br>3. 监督微调（SFT）<br>4. 基于 GRPO 的强化学习（RL） |
| **推理方式** | 支持标准自回归与 MTP 加速解码 |
| **评估指标** | <ul><li>**Overall Score (OmniDocBench)**</li><li>**TextEdit / FormulaCDM / TableTEDS / ReadingOrderEdit**</li><li>**Field-level F1 (KIE)**</li><li>**Throughput (pages/sec)**</li></ul> |

### 基线方法对比
涵盖以下几类模型：
- **Pipeline 工具**：PP-StructureV3、Marker
- **通用 VLMs**：GPT-4o, GPT-5.2, Qwen2.5-VL, Qwen3-VL, Gemini-2.5/3.0 Pro
- **专用 OCR VLMs**：PaddleOCR-VL-1.5, Deepseek-OCR, MinerU2.5, dots.ocr, MonkeyOCR-pro

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3 & 4）

| 任务 | 数据集 | GLM-OCR 性能 | 最优基线 | 是否 SOTA |
|------|--------|---------------|-----------|------------|
| 文档解析 | OmniDocBench v1.5 | **94.6** | PaddleOCR-VL-1.5 (94.5) | ✅ |
| 文本识别 | OCRBench (Text) | **94.0** | dots.ocr (92.1) | ✅ |
| 公式识别 | UniMERNet | **96.5** | MonkeyOCR-pro-3B (96.4) | ✅ |
| 表格结构 | PubTabNet | 85.2 | MinerU2.5 (88.4) | ❌ |
| 表格结构 | TEDS_TEST | **86.0** | Gemini-3 Pro (81.8) | ✅ |
| KIE | Nanonets-KIE | **93.7** | Gemini-3 Pro (95.2) | ✅（开源最佳） |
| KIE | Handwritten-KIE | **86.1** | Gemini-3 Pro (94.5) | ✅（开源最佳） |

> 注：尽管 GPT/Gemini 等闭源模型表现优异，但 GLM-OCR 是**唯一在 0.9B 规模下达到甚至超越其性能的开放权重模型**。

### 自研场景性能（Table 5）
| 场景 | GLM-OCR | 最佳开源基线 | 优势幅度 |
|------|---------|----------------|------------|
| Seal Recognition | **90.5** | dots.ocr (63.0) | +27.5 pts |
| Multilingual OCR | **69.3** | dots.ocr (65.1) | +4.2 pts |
| Real-world Table | **91.5** | dots.ocr (81.8) | +9.7 pts |
| Receipt KIE | **94.5** | — | 显著优于 GPT-5.2 (83.5) |

### 推理效率对比（Table 6）
| 模型 | Image Input (pages/s) | PDF Input (pages/s) |
|------|------------------------|-----------------------|
| **GLM-OCR** | **0.67** | **1.86** |
| PaddleOCR-VL-1.5 | 0.39 | 1.22 |
| MinerU2.5 | 0.18 | 0.48 |
| dots.ocr | 0.10 | — |

→ GLM-OCR 实现最高吞吐，尤其在 PDF 批量处理上领先明显。

### 消融实验（隐含分析）
虽然文中未明确列出消融表，但从训练阶段设计可推断：
- **引入 MTP** → 提升解码速度 ~50%，改善结构一致性
- **两阶段 pipeline** → 减少幻觉、提升复杂布局稳定性
- **RL 阶段优化** → 引入结构验证奖励函数（JSON parse validation, tag closure），增强输出合法性

---

## 4. 关键结论和发现

### 主要发现
1. **小模型也能实现 SOTA 性能**：通过架构创新（MTP + 两阶段 pipeline）而非单纯扩大参数，GLM-OCR 在多项任务上超越百亿甚至千亿参数模型。
2. **MTP 对 OCR 类任务极具价值**：相比纯自回归，多 token 预测大幅提升效率且不牺牲精度，特别适合表格、公式等结构化输出。
3. **模块化系统更稳健**：先做 layout 再并行识别的设计有效降低模型负担，提升对复杂文档的适应能力。
4. **开放模型具备商业竞争力**：在 Receipt KIE 等真实任务中，GLM-OCR 超越 GPT-5.2，证明其可用于生产环境。

### 局限性
| 限制 | 说明 |
|------|------|
| **两阶段误差传播** | 若 layout 分析错误（如漏检区域），会影响后续识别结果 |
| **极端低质图像表现下降** | 极低分辨率、严重扭曲或模糊文档识别效果不佳 |
| **高度复杂公式/表格挑战** | 如嵌套矩阵、非规则合并单元格等仍可能出错 |
| **语言覆盖有限** | 当前训练数据以中英文为主，部分小语种表现较弱 |
| **输出格式随机性** | 作为生成模型，存在轻微换行/空格波动，无法保证完全一致格式 |

### 未来工作方向
1. **增强 layout 模块鲁棒性**：改进跨页依赖、不规则多栏结构的理解能力。
2. **扩展多语言支持**：增加更多低资源语言的训练数据。
3. **提升结构化输出一致性**：进一步约束 JSON/Markdown 输出格式，减少变异性。
4. **探索端到端联合优化**：尝试 joint training layout 与 recognition 模块，缓解误差传播。
5. **轻量化部署优化**：支持 ONNX、TensorRT 等格式，便于移动端集成。

---

> **总结一句话**：  
> GLM-OCR 证明了**面向特定任务的紧凑架构设计 + 高效解码策略 + 强结构监督**，可以在远低于主流模型的参数规模下，实现兼具高性能、高效率和强实用性的文档智能解决方案。

</details>

---

### 4. [S-HPLB: Efficient LLM Attention Serving via Sparsity-Aware Head Parallelism Load Balance](https://arxiv.org/abs/2603.10353)

**Authors**: Di Liu, Yifei Liu, Chen Chen, Zhibin Yu, Xiaoyi Fan, Quan Chen, Minyi Guo  
**Category**: cs.DC  
**Published**: 2026-03-12  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.10353v1  

#### Abstract
With the increasing volumes of Large Language Models (LLMs) and the expanding context lengths, attention computation has become a key performance bottleneck in LLM serving. For fast attention computation, recent practices often parallelize the attention heads on multiple GPUs, and also widely adopt ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：S-HPLB: Efficient LLM Attention Serving via Sparsity-Aware Head Parallelism Load Balance

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着 Large Language Models (LLMs) 规模和上下文长度（context length）的不断增长，**attention 计算已成为 LLM 推理服务中的关键性能瓶颈**，尤其是在 prefill 阶段。当前主流优化手段包括：
- **系统层面**：采用 Head-Parallelism (HP) 将 attention heads 分布在多个 GPU 上以加速计算；
- **算法层面**：采用稀疏 attention（sparse attention），如 top-k 或 top-p 策略，减少参与计算的 query-key 对数量。

然而，现有方法存在两个核心问题：
1. **统一 sparsity budget 不合理**：大多数方法对所有 attention head 施加相同的 token 预算（如固定 k），但不同 head 具有显著的 **sparsity heterogeneity**（稀疏性异质性）——有些 head 只需少量 tokens 即可恢复大部分 attention 权重，而另一些则需要更多。统一预算导致高稀疏 head 浪费计算资源，低稀疏 head 损失精度。
2. **跨 GPU 负载不均衡**：当各 head 的计算量因自适应预算而变得不一致时，传统的 head-parallel 部署会导致严重的 **cross-GPU resource bubbles**（设备间等待空洞），降低并行效率。

---

### 🚀 提出的新方法：S-HPLB
本文提出 **Sparsity-aware Head-Parallel Load Balance (S-HPLB)**，一种系统-算法协同设计框架，从以下两方面优化 attention serving：

#### （1）Adaptive Head Budget Allocation（自适应头预算分配）
- **观察发现**：尽管不同 attention head 的稀疏程度各异，但在多种输入任务和上下文长度下，其 **sparsity 特性具有高度稳定性**（cross-request stability）。这使得可以进行离线建模。
- **方法**：通过在 calibration dataset 上离线 profiling 各 head 的 recovery ratio（恢复率）曲线，预估每个 head 所需的最小 token 数来达到目标稀疏度。
- **策略**：采用 **max-min 优化策略** 进行 budget shifting —— 将冗余预算从“高稀疏”head 转移到“低稀疏”head，在总计算量不变的前提下最大化整体 accuracy。

> 💡 类比：不是给每人发同样饭量，而是根据饭量大小动态调整，确保所有人都吃饱且不浪费。

#### （2）Head Parallel Load Balance（头并行负载均衡）
- **问题建模**：将 head 到 GPU 的映射问题形式化为经典的 **multiway partitioning problem**，目标是最小化最大设备负载，从而减少同步等待时间。
- **求解算法**：提出一个高效的 **贪心启发式算法**：
  1. 按预算降序排列所有 attention heads；
  2. 依次将每个 head 分配给当前负载最小的 GPU。
- 时间复杂度仅为 $O(N \log N + N \log K)$，适合实际部署。

---

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如 top-k / top-p） | S-HPLB |
|------|-------------------------------|--------|
| **预算分配** | 统一或在线估计（top-p） | 离线建模 + 自适应跨头转移 |
| **准确性** | top-p 准确但开销大；top-k 快但不准 | 接近甚至超过 full attention |
| **系统效率** | 忽视负载不平衡 | 显式建模并优化负载分布 |
| **端到端延迟** | 存在严重 GPU idle | 显著减少 resource bubbles |

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **基准测试**：使用长上下文权威 benchmark **RULER** [9]，包含四大类共13项任务（retrieval, multi-hop tracing, aggregation, QA），支持高达 128K 的 context length。
- **评估模型**：
  - `Llama-3.1-8B`
  - `Qwen2.5-7B`
  - `Qwen2.5-72B`
  - `Llama-3-8B-262K`（用于扩展性验证）

### ⚙️ 实验设置
- **硬件平台**：单台服务器，配备 **8 × NVIDIA A100 80GB GPU**（NVLink 连接），Intel Xeon CPU，CUDA 12.4。
- **实现基础**：基于开源 sparse attention 框架 **MInference [10]** 构建，扩展其实现 S-HPLB 的 budget allocation 与 load balancing 模块（约 1.6K 行 Python 代码）。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Model Accuracy** | 在 RULER 上的平均得分（百分制） |
| **Average Attention Latency** | Time-to-First-Token 中的 attention 计算延迟，反映 serving 效率 |
| **Pareto Frontier** | 准确率 vs. 延迟的权衡曲线，衡量综合性能 |

### 🆚 基线方法对比
| 类型 | 方法 | 说明 |
|------|------|------|
| Full Attention | FlashAttention [6] | 完整 attention 作为上限参考 |
| Top-k 方法 | StreamingLLM [27], MInference [10] | 固定预算，不同稀疏模式 |
| Top-p 方法 | XAttention [29] | 动态决定预算以满足累计权重阈值 p |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### （1）准确率表现（Table 1）
| 方法 | Llama-3.1-8B | Qwen2.5-7B | Qwen2.5-72B |
|------|---------------|-------------|--------------|
| Full Attention | 76.38% | 67.46% | 83.69% |
| XAttention (best sparse baseline) | 73.29% | 63.45% | 79.95% |
| **S-HPLB** | **75.86%** | **66.09%** | **80.56%** |

✅ **结论**：
- S-HPLB 准确率几乎与 full attention 持平，仅下降 **0.52% ~ 3.13%**；
- 相比最优稀疏 baseline（XAttention），准确率提升达 **2.57% ~ 0.61%**；
- 在部分任务上（如 MK3）甚至**反超 full attention**，归因于噪声过滤能力。

> 📌 注：某些稀疏方法能提升 accuracy 已被其他研究证实 [3]。

---

#### （2）Attention 延迟表现（Figure 9）
| 比较对象 | 延迟降低倍数（vs. XAttention） |
|---------|------------------------------|
| S-HPLB vs. XAttention | **2.09× ~ 2.88×** 更快 |
| S-HPLB vs. Full Attention | **3.31× ~ 4.27×** 更快 |

✅ **结论**：
- S-HPLB 在保持高 accuracy 的同时，实现了接近 top-k 方法的速度，并显著优于 top-p 方法；
- 在 Qwen2.5-72B 上达到 **2.88× 的 attention 计算延迟加速**。

---

#### （3）消融实验（Ablation Study，Figure 11）
单独评估 **Head Parallel Load Balancer** 的效果：

| 设置 | 延迟降低倍数 |
|------|-------------|
| 启用 load balancer（vs. 不启用） | 最高 **1.26×** 降低延迟 |

✅ **结论**：
- 负载均衡模块本身即可带来显著性能增益；
- 在不同并行度（HP=2~8）和上下文长度（128K~256K）下均稳定有效。

---

#### （4）精度-延迟权衡分析（Figure 10）
- S-HPLB **始终位于 Pareto frontier 的前沿位置**，即在相同延迟下提供更高 accuracy，或在相同 accuracy 下实现更低延迟。
- 表明其在 efficiency 和 fidelity 之间取得了更优平衡。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Attention heads 存在显著且稳定的 sparsity heterogeneity**，为差异化预算分配提供了理论依据。
2. **Offline profiling + max-min budget shifting** 是一种高效、准确的预算分配方式，优于在线估计的 top-p 方法。
3. **Head-parallel 部署中负载不均衡是不可忽视的性能杀手**，必须显式建模与优化。
4. **S-HPLB 实现了 accuracy-preserving 的高效 attention serving**，在多个主流 LLM 上验证了其有效性。

---

### ⚠️ 局限性
1. **依赖 calibration dataset**：虽然作者证明 sparsity 特性泛化良好，但仍需一定代表性数据进行离线建模。
2. **主要针对 prefill 阶段优化**：未深入处理 decoding 阶段的 memory-bound 问题（尽管可通过 KV Cache 互补）。
3. **假设 budget 与 computation load 成正比**：未考虑 kernel launch overhead 或内存访问差异等底层因素。

---

### 🔮 未来工作方向
1. **自动化 calibration pipeline**：开发轻量级自动校准机制，适配新模型或领域迁移场景。
2. **联合优化 decoding 阶段**：结合 PagedAttention 或 PQCache 等技术，构建全阶段高效 inference 引擎。
3. **扩展至 MoE 架构**：将 S-HPLB 思想应用于 expert-parallel 场景下的负载均衡。
4. **硬件感知调度**：进一步融合 GPU SM 利用率、带宽等因素，实现更细粒度的 runtime 调度。

---

## ✅ 总结
S-HPLB 是一项典型的 **system-algorithm co-design** 工作，它：
- 从算法角度利用 attention head 的 **sparsity stability** 实现精准 budget allocation；
- 从系统角度通过 **load-balanced head-parallel deployment** 消除 resource bubbles；
- 在 accuracy 和 latency 之间取得卓越平衡，**平均提升 accuracy 达 2.57%，最高降低 attention latency 达 2.88×**。

该方法为大规模 LLM 的高效 serving 提供了一条实用且可推广的技术路径。

</details>

---

### 5. [Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems](https://arxiv.org/abs/2603.10053)

**Authors**: Wentao Wang, Lifeng Han, Guangyu Zou  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.10053v1  

#### Abstract
The Pickup and Delivery Problem (PDP) is a fundamental and challenging variant of the Vehicle Routing Problem, characterized by tightly coupled pickup--delivery pairs, precedence constraints, and spatial layouts that often exhibit clustering. Existing deep reinforcement learning (DRL) approaches eit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本文针对**Pickup and Delivery Problem (PDP)** 这一经典车辆路径问题中的挑战展开研究。PDP 的核心难点在于：
- **配对约束**：每个取货点必须与其对应的送货点由同一辆车服务；
- **顺序约束**：取货必须在送货之前完成；
- **空间聚类结构**：现实场景中，取货点和送货点常分别集中在不同地理区域（如住宅区与商业区），形成天然的空间“簇”（cluster）。

现有基于 **Deep Reinforcement Learning (DRL)** 的神经求解器通常将所有节点建模为扁平图（flat graph），依赖模型隐式学习这些复杂结构，导致效率低、泛化差。此外，一些高性能方法（如 NCS）依赖推理时的协作搜索（collaborative search），带来高延迟。

---

### 🚀 提出的新方法：CAADRL

作者提出 **CAADRL (Cluster-Aware Attention-based Deep Reinforcement Learning)**，一种显式利用 PDP 多尺度结构的 DRL 框架，其核心创新包括：

#### （1）**Cluster-Aware Attention 编码器**
- 在标准 Transformer 编码器基础上引入 **Cluster-Aware Attention 机制**；
- 同时执行两种注意力：
  - **Global Self-Attention**：捕捉全局空间关系；
  - **Intra-Cluster Attention**：通过 `cluster mask` 限制节点只能关注同簇内节点（如取货点之间、送货点之间），增强局部角色感知能力；
- 输出的嵌入向量兼具**全局一致性**与**局部角色敏感性**。

#### （2）**Hierarchical Dynamic Dual-Decoder**
- 设计双解码器架构：
  - **Intra-Cluster Decoder**：专注于簇内的精细路由决策；
  - **Inter-Cluster Decoder**：处理跨簇转移；
- 引入一个可学习的 **gating module**，动态决定每一步是“留在当前簇”还是“跳转到其他簇”，实现分层决策控制。

#### （3）**端到端训练 + POMO 范式**
- 使用 **POMO-style policy gradient** 进行训练，利用多个对称 rollout 提供更稳定的梯度信号；
- 整个框架为**纯构造策略**（construction policy），无需迭代优化或后处理改进步骤。

---

### 🔍 相比现有方法的优势

| 方面 | CAADRL | 其他方法（如 NCS、Heter-AM） |
|------|--------|-------------------------------|
| **结构建模** | 显式建模簇结构 | 隐式学习或仅区分角色 |
| **推理效率** | 单次自回归生成，速度快 | NCS 需多次改进迭代，延迟高 |
| **扩展性** | 大规模实例表现优异 | 性能随规模增长下降明显 |
| **通用性** | 在非聚类数据上仍具竞争力 | 对分布变化敏感 |

> ✅ **核心优势总结**：CAADRL 通过**显式的多尺度建模**，实现了高效、高质量的一次性路径构造，在保持强性能的同时大幅降低推理时间。

---

## 2. 核心实验方法和设置

### 📊 数据集

使用合成生成的二维欧氏平面数据集，分为两类分布以验证鲁棒性和特异性：

| 类型 | 描述 |
|------|------|
| **Clustered Distribution** | 取货点集中于 (0.25, 0.25)，送货点集中于 (0.75, 0.75)，标准差 0.1，模拟真实城市分区场景 |
| **Uniform Distribution** | 所有节点（含仓库）在 [0,1]×[0,1] 内均匀采样，无明显聚类结构 |

测试规模涵盖四种问题大小：
- **PDP10, PDP20, PDP40, PDP80**：分别对应 5, 10, 20, 40 个取送对（共 10–80 个客户节点）

---

### 🧪 实验设置与评估指标

| 设置项 | 说明 |
|-------|------|
| **训练方式** | 使用 POMO 框架，每实例并行启动 N 个 rollout（从不同起点开始） |
| **优化器** | Adam，学习率 $1\times10^{-4}$，训练 800 轮，batch size=512 |
| **模型参数** | Embedding dim=128，Encoder 层数 L=6，Attention heads=8 |
| **评估指标** | 平均总路径长度（Objective）、相对 Gap (%)、平均推理时间（Time, 秒） |
| **测试集** | 每种配置下固定 100 个测试实例 |

---

### 🆚 基线方法对比

| 方法 | 简介 |
|------|------|
| **NCS (Neural Collaborative Search)** | 当前 SOTA 方法之一，结合构造策略与神经邻域搜索，支持多轮改进（t=1k/2k/3k 表示迭代次数） |
| **Heter-AM** | 基于异构注意力的编码器-解码器模型，为每个节点类型（depot/pickup/delivery）设计专用注意力机制 |
| **CAADRL (Ours)** | 本文提出的方法，评估三种解码策略：<br>- Greedy<br>- Sample1280<br>- Sample12800 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）

#### ✅ 在 **Clustered Instances** 上的表现（Table 1）

| 方法 | PDP20 | PDP40 | PDP80 | 推理时间 (PDP80) |
|------|--------|--------|--------|------------------|
| **CAADRL (Sample12800)** | **2.723** | **3.551** | **4.709** | 0.198s |
| NCS (t=3k) | 2.724 | 3.649 | 4.734 | 0.444s |
| Heter (Sample12800) | 2.764 | 3.563 | 4.737 | 0.594s |

> 💡 **结论**：CAADRL 在中大规模聚类实例上全面超越基线，尤其在 **PDP80** 上取得最优解且推理时间仅为 NCS 的 ~45%，Heter 的 ~33%。

#### ✅ 在 **Uniform Instances** 上的表现（Table 2）

| 方法 | PDP20 | PDP40 | PDP80 |
|------|--------|--------|--------|
| **CAADRL (Sample12800)** | 4.661 | 6.702 | **9.413** |
| NCS (t=3k) | 4.795 | 6.552 | 10.080 |
| Heter (Sample12800) | **4.595** | 6.849 | 10.101 |

> 💡 **结论**：
- 小中规模上略逊于最强基线；
- **但在最大规模 PDP80 上反超**，优于 NCS 和 Heter 约 **6.6%**，显示其在复杂结构下的强大泛化能力。

---

### 🔬 消融实验结果（Table 3）

消融研究验证了各组件的有效性：

| 变体 | PDP100 (Sample12800) | 相较完整模型差距 |
|------|------------------------|------------------|
| **CAADRL (Full Model)** | **5.201** | — |
| no_encoder（替换为标准 Transformer） | 5.220 | +0.37% |
| no_decoder（移除双解码器） | 5.198 | -0.06%（微弱优势） |
| POMO Baseline | 5.236 | +0.67% |

> 🔍 **分析**：
- **Cluster-Aware Encoder 至关重要**：去掉后性能显著下降，证明显式建模簇结构有效；
- **Dynamic Dual-Decoder 贡献有限但稳定**：在贪婪或小样本下提升明显，大采样时被部分补偿；
- **两者互补**：最佳效果需二者协同。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **显式建模簇结构是一种有效的归纳偏置（inductive bias）**  
   - Cluster-Aware Attention 能有效分离全局与局部信息，提升嵌入质量；
   - 特别适用于具有地理聚集特征的真实物流场景。

2. **分层解码机制有助于协调局部探索与全局转移**  
   - Dual-Decoder + Gating 实现了对“何时深入、何时跳跃”的智能判断；
   - 在大规模问题中表现出更强的路径组织能力。

3. **CAADRL 是高效的单步构造策略**  
   - 不依赖昂贵的推理时搜索，**显著低于 NCS 的推理延迟**；
   - 更适合实时调度系统部署。

4. **良好的泛化能力**  
   - 即使在无明确聚类的 uniform 实例上，性能下降温和；
   - 在 **PDP80 uniform** 上反而成为最优，表明其学到的是**多尺度路由原则**而非过拟合特定分布。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖预定义簇标签** | 当前簇划分基于节点类型（pickup/delivery），未考虑动态聚类或空间自适应分组 |
| **静态单车辆假设** | 未覆盖多车、时间窗、容量等实际约束 |
| **训练-测试规模匹配影响性能** | 跨规模迁移虽可行，但仍存在性能衰减（见 Fig. 5） |

---

### 🔮 未来工作方向（原文第6节）

1. **拓展至更复杂的现实场景**  
   - 多车 PDP（m-PDP）
   - 带时间窗（PDP-TW）
   - 动态请求到达（online PDP）
   - 无人机辅助配送（drone-assisted PDP）

2. **学习动态簇结构**  
   - 引入可微分聚类模块（differentiable clustering）或图分割网络，让模型自动识别空间簇。

3. **工业级应用验证**  
   - 在真实物流数据集上测试；
   - 集成至滚动时域调度（rolling horizon）框架；
   - 分析系统级延迟与人机协同机制。

---

## ✅ 总结

> **CAADRL 成功地将“人类先验知识”——即 PDP 中常见的空间聚类结构——融入神经求解器的设计之中，通过 Cluster-Aware Attention 与 Hierarchical Dual-Decoder 构建了一个既能高效运行又能精准捕捉多尺度结构的 DRL 框架。它不仅在聚类实例上达到 SOTA，还在非聚类和大规模情形下展现出卓越的鲁棒性与扩展性，为神经组合优化提供了“结构驱动设计”的典范。**

</details>

---

### 6. [Trajectory-Informed Memory Generation for Self-Improving Agent Systems](https://arxiv.org/abs/2603.10600)

**Authors**: Gaodan Fang, Vatche Isahagian, K. R. Jayaram, Ritesh Kumar, Vinod Muthusamy, Punleuk Oum, Gegi Thomas  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.10600v1  

#### Abstract
LLM-powered agents face a persistent challenge: learning from their execution experiences to improve future performance. While agents can successfully complete many tasks, they often repeat inefficient patterns, fail to recover from similar errors, and miss opportunities to apply successful strategi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Trajectory-Informed Memory Generation for Self-Improving Agent Systems*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
LLM-powered **Agent** 在执行任务时通常表现出“健忘”（amnesia），即缺乏从过往执行轨迹（execution trajectories）中系统性学习的能力。尽管它们能完成任务，但仍存在以下问题：
- 重复低效行为（如逐个调用 `remove_from_cart` 而非使用 `empty_cart()`）
- 无法从失败中恢复类似错误（如未配置支付方式导致 checkout 失败）
- 错过将成功策略泛化到新任务的机会

传统方法（如规则系统、prompt engineering、通用 memory 系统）无法有效提取、分类并精准检索这些多样化学习机会。

---

### 🚀 提出的新方法与创新思路
作者提出一个**基于轨迹的智能记忆生成框架**，实现 Agent 的自我进化。该框架由四个核心组件构成：

#### （1）**Trajectory Intelligence Extractor**  
对 Agent 的推理过程进行语义分析，识别认知模式：
- Analytical thoughts（分析约束）
- Planning thoughts（规划动作序列）
- Validation thoughts（验证前提条件）
- Reflection/self-correction（反思与自修正）

> 不依赖关键词匹配，而是通过 LLM 进行语义理解，提升泛化能力。

#### （2）**Decision Attribution Analyzer**  
自动进行因果归因分析，区分不同层级的原因：
- Immediate cause（直接触发失败的动作）
- Proximate cause（近期决策）
- Root cause（根本原因）
同时识别 recovery 和 inefficiency 的根源，并生成可预防的具体步骤。

#### （3）**Contextual Learning Generator**  
生成三类结构化的指导性提示（tips）：
| 类型 | 内容来源 | 示例 |
|------|--------|------|
| **Strategy Tips** | 成功且高效的执行 | “在 checkout 前验证购物车、地址、支付方式” |
| **Recovery Tips** | 失败后成功恢复 | “当 checkout 报缺支付方式时，先检查并添加 payment method” |
| **Optimization Tips** | 成功但低效的执行 | “清空多商品购物车应调用 `empty_cart()` 而非循环删除” |

每条 tip 包含：类别、具体步骤、触发条件、负例（negative example）、优先级、上下文标签等元数据。

#### （4）**Adaptive Memory Retrieval System**  
支持两种检索策略：
- **Cosine Similarity Retrieval**：快速向量相似度匹配
- **LLM-Guided Selection**：利用 LLM 分析任务上下文，结合 metadata 过滤与优先级排序，实现更精准检索

> 支持多维匹配：任务类型、领域、执行模式、语义意图。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | 本文改进 |
|------|-------|---------|
| Rule-based systems | 手动编码、难以扩展、无法适应新场景 | 自动从轨迹中提取规则 |
| Prompt engineering | 泛化差、无自动化学习机制 | 动态注入基于经验的记忆 |
| Generic memory systems (e.g., Mem0, Letta) | 存储事实而非行为策略；无因果分析；无结构化分类 | 提取结构化、可操作的指导；支持 provenance 追踪 |
| Reinforcement Learning | 高成本、黑盒、难解释 | 白盒式透明学习，便于审计与调试 |

> ✅ 优势总结：
> - **理解执行逻辑而不仅是动作序列**
> - **支持多类型学习（策略/恢复/优化）**
> - **提供可追溯的 provenance**
> - **实现上下文感知的精准检索**

---

## 2. 核心实验方法和设置

### 📚 数据集
使用 **AppWorld benchmark**，一个面向 LLM Agent 的综合性评测套件，涵盖多个应用领域的复杂任务：
- e-commerce（电商）
- email（邮件管理）
- calendar（日程安排）
- file management（文件操作）
- 多 app 协同任务（平均涉及 1.8 个 app，9.5 个 API）

任务按难度分为三级：
- **Difficulty 1 (Easy)**：单域基础操作
- **Difficulty 2 (Medium)**：跨域或需条件判断
- **Difficulty 3 (Hard)**：多步规划、前置校验、错误恢复，相当于 50+ 行代码逻辑

---

### ⚙️ 实验设置

#### Agent 架构
- 使用 **GPT-4.1** 实现简化版 ReAct-style agent
- 循环执行：思考 → 选择动作 → 调用 API → 观察结果 → 判断是否完成

#### Tip Extraction 配置
比较两种粒度：
- **Task-level tips**：从完整轨迹中提取端到端策略
- **Subtask-level tips**：先将轨迹分解为子任务（如认证、数据获取、处理、完成），再分别提取 tip

#### Retrieval 策略对比
| 策略 | 描述 | 成本 |
|------|------|-----|
| **Cosine Similarity** | 向量相似度检索 top-k tips | 低（无需 LLM 调用） |
| **LLM-Guided Selection** | LLM 分析任务上下文，构造结构化查询 | 高（额外一次 LLM 调用） |

> 所有配置均注入最多 5 条 tips 到 prompt 中作为 guidelines。

#### 评估分区
| 分区 | 是否用于生成 tips | 目标 |
|------|------------------|------|
| **test-normal** | ❌ 否 | 测试泛化能力（最严格） |
| **train** | ✅ 是 | 测试重复任务上的自提升能力 |
| **dev** | ✅ 是 | 辅助验证 |

---

### 🎯 评估指标
| 指标 | 定义 |
|------|------|
| **Task Goal Completion (TGC)** | 单个任务是否完全正确完成（所有单元测试通过） |
| **Scenario Goal Completion (SGC)** | 场景下所有变体任务都必须成功才算通过（更严格） |

> SGC 对一致性要求更高，适合衡量 memory 对行为稳定性的提升。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（test-normal 分区）

| 配置 | TGC (%) | ΔTGC | SGC (%) | ΔSGC |
|------|--------|-------|--------|-------|
| Baseline (no memory) | 69.6 | — | 50.0 | — |
| Task-level + Cosine (t≥0.6) | 72.0 | +2.4 | 62.5 | **+12.5** |
| Subtask-level + Cosine (t≥0.6) | 73.8 | **+4.2** | 57.1 | +7.1 |
| **Subtask-level + LLM-guided** | 73.2 | +3.6 | **64.3** | **+14.3** ✅ |

> 💡 最佳配置：**Subtask-level tips + LLM-guided retrieval**，在 SGC 上取得最大增益（+14.3 pp）

---

### 🔍 按难度分层结果（test-normal）

| 难度 | 配置 | TGC Δ | SGC Δ |
|------|------|--------|--------|
| **Difficulty 3 (Hard)** | Subtask + LLM | +4.7 | **+28.5** ✅ |
| | | | （相对提升 **149%**） |
| **Difficulty 2 (Medium)** | Subtask + LLM | +4.1 | +0.0 |
| **Difficulty 1 (Easy)** | Subtask + LLM | +1.7 | +10.5 |

> 📌 发现：**越复杂的任务，收益越大**。因为复杂任务更依赖策略、恢复和优化知识。

---

### 🔬 消融实验结果

#### （1）Tip Granularity 影响
- **Subtask-level tips 显著优于 Task-level tips 在 TGC 上**（73.8 vs 72.0）
- 原因：子任务分解提高了 tip 的复用性和精确性（如“认证流程”可在 Spotify/Venmo/Phone 间共享）

#### （2）Retrieval Strategy 影响
- **LLM-guided selection 显著优于 Cosine 在 SGC 上**（64.3 vs 57.1）
- 原因：LLM 能理解上下文、过滤无关提示、优先展示关键 recovery tips，从而提高跨变体的一致性

#### （3）交互效应
- Task-level + Cosine 的 SGC 高于 Subtask + Cosine（62.5 > 57.1），说明整体策略有助于统一行为
- 但加入 LLM-guided 后，Subtask 方案反超，表明其可通过上下文推理补偿行为差异

---

### 🔄 Source Partition 结果（train/dev）

| 分区 | TGC Δ | SGC Δ |
|------|--------|--------|
| **Train** | +4.4 | +10.0 |
| **Dev** | +12.3 | **+26.3** |

> 在曾见过的任务上，性能提升更大，证明框架具备真正的 **self-improving** 能力。

> ⚠️ 注意：在简单任务（D1）上，baseline 已接近 100%，加入 memory 反而轻微下降（干扰效应），但在困难任务中增益显著。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Agent 可以从执行轨迹中自动提取高质量、结构化的学习成果**，包括 strategy、recovery 和 optimization tips。
2. **Subtask-level extraction 更有利于提升 TGC**，因其提高了 tip 的可迁移性和精度。
3. **LLM-guided retrieval 更有利于提升 SGC**，尤其在复杂任务中，能显著增强行为一致性。
4. **性能增益随任务复杂度上升而放大**，在 Difficulty 3 任务中 SGC 提升达 **+28.5 pp（149% 相对增长）**。
5. **框架实现了真正的泛化能力**：在未见任务（test-normal）上仍取得显著提升。

---

### ⚠️ 方法的局限性

1. **依赖高质量的轨迹记录**：需要保存完整的 reasoning trace、action、result 和 outcome label。
2. **LLM-guided retrieval 增加延迟和成本**：每次调用需额外 LLM 推理。
3. **tip 冲突处理有限**：虽然有合并机制，但在大规模部署中可能出现矛盾建议。
4. **当前仅适用于单 Agent**：尚未扩展至 multi-agent 场景下的 cross-agent attribution。

---

### 🔮 未来工作方向

1. **扩展至 Multi-Agent Systems**：
   - 支持跨 Agent 的经验共享
   - 引入角色感知的 guidance（role-aware tips）

2. **支持更多 LLM 模型族**：
   - 在 Qwen、GPT-OSS 等开源模型上验证 tip 质量与检索效果

3. **动态更新与遗忘机制**：
   - 引入 memory aging 与 selective forgetting
   - 防止过时或低效 tips 污染 memory 库

4. **集成到企业级平台**：
   - 当前已在 IBM 的 **CUGA (Configurable Generalist Agent)** 平台中应用
   - 推动 Agent 在实际业务中持续从运营经验中学习进化

---

> 🧠 总结一句话：  
> 本文提出了首个能够从 Agent 执行轨迹中**自动提取结构化、可追溯、多类型学习成果**并实现**上下文感知精准检索**的记忆框架，在 AppWorld 上验证了其对复杂任务性能的显著提升，为构建真正 self-improving 的 agentic systems 提供了可行路径。

</details>

---

### 7. [AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU](https://arxiv.org/abs/2603.10342)

**Authors**: Yuning Zhang, Yan Yan, Nan Yang, Dong Yuan  
**Category**: cs.DC  
**Published**: 2026-03-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.10342v1  

#### Abstract
Large language models (LLMs) are increasingly deployed as AI agents that operate in short reasoning-action loops, interleaving model computation with external calls. Unlike traditional chat applications, these agentic workloads require inference serving systems to balance low latency, stable token e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题  
本文针对**在消费级单GPU上高效服务多AI代理（Agentic AI）工作负载**的问题。传统LLM推理系统（如vLLM、SGLang）主要面向长文本生成的聊天机器人场景，而AI代理的工作流具有以下特点：
- **短推理-动作循环**（short reasoning-action loops），频繁交替执行 **cold prefill**（处理长系统提示）、**resume prefill**（追加工具输出）和 **short decode**（生成函数调用等短响应）；
- 多个代理并发请求时，长prefill会阻塞对延迟敏感的decode阶段，导致严重的 **head-of-line blocking** 和 **TPOT（Time-Per-Output-Token）不稳定**；
- 在资源受限的消费级GPU上，缺乏有效的相位隔离机制。

因此，如何在单GPU上实现稳定低延迟、高吞吐的服务成为挑战。

---

### 提出了什么新方法或新思路  
作者提出 **AgentServe** —— 一种算法-系统协同设计的单GPU推理服务框架，其核心创新包括：

#### ✅ 算法层面：Phase-aware 调度 + TPOT驱动控制
- 将请求分为三类：**cold prefill**, **resume prefill**, **short decode**；
- 引入 **TPOT-driven feedback scheduler** 动态调整两个参数：
  - `R_min(t)`：为decode保留的最小SMs数量（保障延迟）；
  - `B_prefill(t)`：resume prefill允许的最大token长度（限制干扰）；
- 实现 **decode优先调度策略**，确保短解码不受prefill影响。

#### ✅ 系统层面：基于CUDA Green Context的轻量级资源隔离
- 利用 **CUDA Green Contexts** 预先建立多个固定SM配额的上下文槽位（如10%, 20%, ..., 100%）；
- 通过快速rebind机制动态切换decode/prefill使用的context，实现 **细粒度空间隔离**；
- 避免了多进程或多引擎带来的KV缓存传输开销（inter-engine KV transfer）；
- 共享内存池 + Mutex + cudaEvent 实现安全KV cache复用。

#### ✅ 协同设计优势
- 不依赖分布式或多引擎架构，在**单GPU单Engine内完成PD disaggregation**；
- 同时优化了 **延迟稳定性（TPOT）** 和 **整体吞吐（throughput）**；
- 特别适用于本地部署、边缘设备中的工具增强型小模型（SLM）代理场景。

---

### 相比现有方法的优势
| 方法 | 局限性 | AgentServe改进 |
|------|--------|----------------|
| **vLLM** | 使用chunked prefill缓解HoL，但decode仍易受扰动；无严格资源隔离 | 显式分离phase，提供SM级保护 |
| **SGLang** | 支持PD disaggregation，但需双进程，带来协调开销 | 单引擎+Green Context，避免跨进程通信 |
| **llama.cpp** | 无专门调度优化，prefill/decode混合竞争资源 | 完整调度+资源控制闭环 |

> ✅ **核心优势总结**：在消费级GPU上实现了接近最优的decode延迟稳定性，同时保持高吞吐，解决了“稳定 vs 效率”的权衡难题。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 基于 **ToolBench** 构建代理任务工作负载；
- 包含两类典型agent范式：
  - **ReAct**：频繁resume prefill + 极短decode（函数调用）；
  - **Plan-and-Execute**：长cold prefill + 中等decode；
- 模拟真实工具调用流程（检索、API调用、决策链等）。

---

### 实验设置
#### 硬件平台
| GPU | 核心数 | SMs | 内存 | 场景定位 |
|-----|-------|-----|------|----------|
| **RTX A5000** | 8192 CUDA cores | 64 SMs | 24GB GDDR6 | 中端边缘部署 |
| **RTX 5090**（模拟）| 16384 cores | 128 SMs | 32GB GDDR7 | 下一代高性能GPU |

#### 模型
- **Qwen2.5-3B**, **Qwen2.5-7B**, **LLaMA-3-8B**
- 覆盖不同规模与架构家族，验证通用性。

#### 并发设置
- 并发代理数：3 ~ 6个；
- 每个session遵循 cold → resume prefill ↔ short decode 循环模式。

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **TTFT**（Time-To-First-Token） | 新请求首次出token时间，反映启动延迟 |
| **TPOT**（Time-Per-Output-Token） | 解码过程中每token间隔，衡量流式平滑性 |
| &nbsp;&nbsp;→ p50 / p95 | 分别表示中位数与尾部延迟 |
| **Throughput** | 总输出tokens/sec，反映系统效率 |
| **SLO Attainment Rate** | 同时满足TTFT和TPOT阈值的session比例，综合用户体验 |

---

### 基线方法对比
| Baseline | 类型 | 是否支持Prefix Caching |
|---------|------|------------------------|
| **SGLang** | PD disaggregation（双进程） | 是 |
| **vLLM** | Chunked prefill + PagedAttention | 是 |
| **llama.cpp** | 轻量级基础框架，无特殊优化 | 是（手动集成） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自图5、6、7）

#### 🔹 TTFT 改进
- AgentServe在所有配置下均取得最低TTFT；
- 相比基线提升：
  - vs SGLang：**1.1–1.3× 更快（median）**，p95最高 **1.3×**
  - vs vLLM：**1.5–1.8× 更快**
  - vs llama.cpp：最高达 **2.8× 加速**

> 尤其在7B大prompt场景下效果更显著，说明有效缓解了prefill拥塞。

---

#### 🔹 TPOT 改进（流式稳定性）
- median TPOT降低：
  - vs SGLang：**1.1–1.2×**
  - vs vLLM：**1.3–1.8×**
  - vs llama.cpp：**>1.5×**
- p95尾延迟改善尤为突出：
  - 最高达 **2.7× 优于llama.cpp**

> 表明AgentServe能显著抑制decode抖动，维持流畅交互体验。

---

#### 🔹 Throughput（吞吐）
- 在保证低延迟的同时，吞吐仍优于多数基线：
  - 比vLLM高 **1.2–1.5×**
  - 比SGLang高 **1.3–1.5×**
  - 比llama.cpp高 **2.0–2.2×**

> 证明其资源利用高效，未因隔离牺牲整体产能。

---

#### 🔹 SLO Attainment Rate（服务质量达标率）
- AgentServe在RTX 5090上接近 **100%达标率**；
- 在RTX A5000上也远超其他方法，尤其在LLaMA-3-8B + N=6时：
  - 基线下降至 <50%
  - AgentServe仍维持 >80%

> 显示其在压力下依然可靠，适合生产环境。

---

### 消融实验结果（Ablation Study）
比较三种变体：
- **Full**：完整AgentServe
- **No-Alg**：关闭TPOT反馈调度，静态分配SM
- **No-Green**：移除Green Context，使用普通stream调度

#### 结果（图7）：
- **No-Alg**：
  - TTFT ↑15–25%
  - TPOT p95 ↑1.4×（decode被starve）
- **No-Green**：
  - TTFT ↑20–30%
  - TPOT波动剧烈，出现明显spike
- **结论**：**算法调度 + Green Context隔离缺一不可**，二者共同构成性能增益来源。

---

## 4. 关键结论和发现

### 主要发现
1. **AI代理工作负载具有独特结构特征**：
   - 存在明显的 **cold prefill / resume prefill / short decode** 三阶段模式；
   - decode虽短但极度敏感，轻微延迟即可引发任务级连锁延迟。

2. **传统优化不足以应对agent场景**：
   - chunked prefill 对短decode无效；
   - 多进程PD disaggregation 开销大且难以在单GPU上扩展。

3. **AgentServe实现了算法-系统的高效协同**：
   - 通过 **TPOT反馈控制 + Green Context动态绑定**，实现了：
     - decode延迟稳定性最大化；
     - prefill吞吐损失最小化；
   - 在单引擎内达成严格资源隔离，无需复杂架构。

4. **理论分析支撑实践设计**：
   - 提出 **profile-aware competitive-ratio bound**，量化了在满足decode SLO前提下，prefill throughput的保留程度；
   - 证明AgentServe可在理论上接近离线最优。

---

### 方法的局限性
1. **适用范围聚焦于本地工具型代理**：
   - 不适用于任意复杂的通用agent（如深度代码生成）；
   - 假设存在prefix caching和结构化prompt。

2. **依赖较新的CUDA特性**：
   - CUDA Green Context目前仅在较新版本CUDA Driver中支持；
   - 可能限制在老旧硬件或云环境中部署。

3. **未考虑异构模型混合部署**：
   - 当前实验集中在单一模型并发；
   - 多模型共存下的资源竞争尚未测试。

---

### 未来工作方向
1. 扩展至 **多模型混合代理集群** 上的调度；
2. 探索 **自动SLO配置** 与用户意图感知的adaptive budgeting；
3. 结合 **MoE架构** 进一步提升prefill/deep decode效率；
4. 在真实机器人、车载系统等边缘设备中进行端到端部署验证。

---

> ✅ **最终结论**：  
> AgentServe首次在消费级单GPU上实现了**稳定、高效、低延迟**的AI代理服务方案，通过**算法-系统协同设计**突破了传统推理系统的瓶颈，为本地化、隐私敏感、低成本的智能代理部署提供了可行路径。

</details>

---

### 8. [Double-Precision Matrix Multiplication Emulation via Ozaki-II Scheme with FP8 Quantization](https://arxiv.org/abs/2603.10634)

**Authors**: Yuki Uchino, Katsuhisa Ozaki, Toshiyuki Imamura  
**Category**: cs.DC  
**Published**: 2026-03-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.10634v1  

#### Abstract
In high-performance computing (HPC) applications, FP64 arithmetic remains indispensable for ensuring numerical accuracy and stability. However, in recent hardware generations, improvements in FP64 arithmetic performance have been relatively modest. Consequently, achieving sustained performance gains...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Double-Precision Matrix Multiplication Emulation via Ozaki-II Scheme with FP8 Quantization*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在高性能计算（HPC）中，**FP64（双精度浮点）算术**对于数值稳定性和精度至关重要，但现代硬件对 FP64 的性能提升有限。与此同时，新兴架构（如 NVIDIA Blackwell Ultra 和 Rubin）大幅削减了 **INT8** 的计算资源，转而强化 **FP8** 等低精度浮点运算能力。

传统的基于 **Ozaki-II scheme** 的 DGEMM（双精度矩阵乘法）模拟方法依赖于 **INT8 MMA（Matrix Multiply-Accumulate）单元**，因其天然适配固定点整数运算。然而，**直接将 Ozaki-II 移植到 FP8 上不可行**，因为 FP8 的浮点特性破坏了模运算所需的精确性。

因此，本文旨在解决以下核心问题：  
> **如何在不牺牲数值精确性的前提下，利用 FP8 MMA 单元实现高效的 Ozaki-II 方案，以支持未来 FP8 主导的硬件架构？**

---

### **提出了什么新方法或新思路**

本文提出了一种**基于 FP8 E4M3 格式的 Ozaki-II 方案**，其核心创新是结合两种技术的混合方法（hybrid method）：

1. **Karatsuba-based Extension（卡拉苏巴扩展）**  
   将输入矩阵 $ A $ 和 $ B $ 分解为两个 FP8 矩阵之和（$ A = s \cdot A^{(1)} + A^{(2)} $），通过 Karatsuba 技巧将乘积展开为三个子项，从而允许使用更大的模数 $ p_e \leq 513 $，显著提升动态范围。

2. **Modular Reduction without Karatsuba（免重构模约简）**  
   对于**平方模数** $ p_e = s^2 $，利用模运算性质 $ \text{mod}(s^2, p_e) = 0 $，可直接跳过 Karatsuba 重建步骤，仅用三个 FP8 矩阵乘法即可完成模乘，避免了中间求和溢出问题。

3. **Hybrid Construction（混合构造）**  
   在模数选择上优先使用**成对互素的平方数**（如 1089, 1024, 961...），对其应用免重构方法；其余非平方模数则使用标准 Karatsuba 扩展。该策略**减少了所需模数数量**，从而降低总 FP8 矩阵乘法次数。

---

### **相比现有方法的优势**

| 方法 | 所需模数 $ N $ | FP8/INT8 矩阵乘法次数 | 有效位宽 |
|------|------------------|------------------------|----------|
| FP8 Ozaki-I (准确模式) | 11 | 121 | 54 bits |
| INT8 Ozaki-II (准确模式) | 14 | 15 | 54 bits |
| **本文：FP8 Ozaki-II (准确模式)** | **12** | **37** | **55 bits** |

- **相比 FP8 Ozaki-I**：乘法次数从 121 降至 37，**减少约 70%**，效率大幅提升。
- **相比 INT8 Ozaki-II**：虽然乘法次数略高（37 vs 15），但**适用于 INT8 资源受限的新架构**（如 Rubin），扩展了适用场景。
- **内存占用更高**：因需存储多个 FP8 分量，工作内存（working memory footprint）约为 INT8 方法的两倍（如 55GB vs 27GB for 16K×16K×16K）。

---

## 2. 核心实验方法和设置

### **实验平台**

- **NVIDIA GeForce RTX 5080**（模拟环境）
- **NVIDIA HGX B200**（真实系统，单 GPU）
- 使用 **CUDA Toolkit 12.8 / 13.1**，底层调用 **cuBLASLt** 执行 INT8/FP8 GEMM。

### **测试矩阵生成**

- 随机矩阵 $ A \in \mathbb{R}^{m \times k}, B \in \mathbb{R}^{k \times n} $
- 元素分布：$ a_{ij}, b_{ij} \sim (\text{rand} - 0.5) \cdot \exp(\text{randn} \cdot \phi) $
  - `rand` ∈ (0,1] 均匀分布
  - `randn` 标准正态分布
  - $\phi$ 控制动态范围
- 测试规模：$ m, n \in \{1024, 2048, 4096, 8192, 16384\}, k \in [1024, 65536] $

### **评估指标**

1. **吞吐量（Throughput）**：TFLOP/s（实测与模型预测）
2. **精度（Accuracy）**：相对误差 $ \|C_{\text{emulated}} - C_{\text{FP64}}\|_F / \|C_{\text{FP64}}\|_F $
3. **工作内存占用（Working Memory Footprint）**
4. **时间分解（Time Breakdown）**：量化转换、GEMMs、模约简、CRT 重建等阶段耗时占比

### **基线方法对比**

- **Native FP64 DGEMM**：cuBLAS `cublasDgemm`
- **INT8-based Ozaki-II**：文献 [19] 实现，使用 14–17 个模数
- **FP8-based Ozaki-I**：文献 [21] 方法，作为 FP8 路线的对照

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **RTX 5080 结果**
- **FP64 原生性能**：0.88 TFLOP/s
- **INT8 模拟**：**3.9–24× 加速**（最大 21.1 TFLOP/s）
- **FP8 模拟**：**3.2–9.4× 加速**（最大 8.3 TFLOP/s）
- **INT8 比 FP8 快 1.3–2.9×**

#### **B200 结果**
- **FP64 原生性能**：75 TFLOP/s（理论）
- **大尺寸（m=n=16384）**：
  - INT8 模拟：**125 TFLOP/s**（fast），**123 TFLOP/s**（accurate）
  - FP8 模拟：**61 TFLOP/s**（fast），**64 TFLOP/s**（accurate）
- **小尺寸（m=n=1024）**：两种模拟均低于原生 FP64

> 注：B200 上 FP8 GEMM 实测吞吐约 3 PFLOP/s，与理论峰值 4.5 PFLOP/s 接近。

---

### **与基线方法的对比结果**

| 维度 | 本文方法（FP8 Ozaki-II） | INT8 Ozaki-II | FP8 Ozaki-I |
|------|----------------------------|--------------|-------------|
| **乘法次数** | 36–43（N=12–14） | 14–18 | 121–169 |
| **有效精度** | ≥55 bits | ≥54 bits | ≥54 bits |
| **内存占用** | 高（~55GB） | 低（~27GB） | 极高（~100GB+） |
| **适用硬件** | FP8-rich（Rubin） | INT8-rich（Hopper/B200） | 通用 |
| **实际吞吐** | 中等（B200: 64 TFLOP/s） | 高（B200: 125 TFLOP/s） | 低（未实测） |

- **FP8 模拟在 FP8 吞吐足够高时才具竞争力**：模型预测显示，当 FP8 GEMM 速度是 INT8 的 **2.5 倍以上**时，FP8 方法有望反超。
- **当前硬件下，INT8 仍占优**：因 INT8 和 FP8 峰值接近（B200 均为 4.5 TOP/s 或 TFLOP/s），且 INT8 更匹配 Ozaki-II 的固定点语义。

---

### **消融实验与分析**

- **准确模式 vs 快速模式**：
  - 准确模式通过 FP8 GEMM 估计缩放边界，精度更高，尤其在 $ k $ 较大时优势明显。
  - 快速模式使用 Cauchy-Schwarz 不等式保守估计，可能导致过度缩放，损失精度。
- **模数选择影响**：
  - 使用更多平方模数可减少 Karatsuba 使用频率，降低乘法次数。
  - 实验验证了 $ N=12 $ 即可达到与 INT8 方法相当的精度。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **首次实现了基于 FP8 的 Ozaki-II 方案**，填补了 FP8 无法直接用于 Ozaki-II 的技术空白。
2. ✅ **混合方法显著降低了 FP8 矩阵乘法次数**（从 121 到 37），使 FP8 路线具备实用潜力。
3. ✅ **在 FP8 主导的未来架构（如 Rubin）中，该方法具有战略意义**，即使当前性能不及 INT8。
4. ✅ **INT8 仍是当前最优选择**：只要硬件提供充足的 INT8 资源，INT8-based Ozaki-II 在吞吐和内存上全面优于 FP8 方案。
5. ✅ **工作内存是瓶颈**：FP8 方法因冗余存储指数字段导致内存开销翻倍，限制了大规模部署。

---

### **方法的局限性**

- **内存占用高**：FP8 存储固定点数据存在比特浪费（exponent 字段未被有效利用）。
- **依赖 FP32 accumulate**：要求 FP8×FP8→FP32 无舍入误差，限制了 $ k \leq 2^{16} $。
- **对 FP8 吞吐率敏感**：若 FP8 性能未显著高于 INT8，则难以竞争。
- **尚未支持 FP4**：尽管讨论了 FP4 可能性，但未实现递归 Karatsuba。

---

### **未来工作方向**

1. **优化内存布局**：设计紧凑的 FP8 存储格式，减少 exponent 字段开销。
2. **探索 FP4 路线**：若未来硬件 FP4 吞吐远超 FP8（>3×），可尝试基于 FP4 的递归分解。
3. **支持稀疏矩阵**：将 Ozaki-II 扩展至 SpGEMM 场景。
4. **跨平台移植**：适配 AMD CDNA 和 Intel GPU 架构。
5. **集成进主流库**：推动该方法进入 cuBLAS、rocBLAS 等标准数学库。

---

> 🔗 **开源代码**：作者已发布支持 **INT8 和 FP8 Ozaki-II** 的跨平台 GPU 库：  
> [https://github.com/RIKEN-RCCS/GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8)  
> 支持 NVIDIA 与 AMD GPU，结果**bitwise reproducible**。

</details>

---

### 9. [Resource-constrained Amazons chess decision framework integrating large language models and graph attention](https://arxiv.org/abs/2603.10512)

**Authors**: Tianhao Qian, Zhuoxuan Li, Jinde Cao, Xinli Shi, Hanjie Liu, Leszek Rutkowski  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.10512v1  

#### Abstract
Artificial intelligence has advanced significantly through the development of intelligent game-playing systems, providing rigorous testbeds for decision-making, strategic planning, and adaptive learning. However, resource-constrained environments pose critical challenges, as conventional deep learni...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Resource-constrained Amazons Chess Decision Framework Integrating Large Language Models and Graph Attention

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文针对**资源受限环境下的复杂博弈决策问题**，特别是 **Game of the Amazons** 这类搜索空间巨大、专家数据稀缺的棋类游戏。传统深度学习和强化学习方法依赖大量高质量训练数据和高算力硬件，在边缘设备或低资源场景中难以部署。

具体挑战包括：
- 合法动作数量庞大（可达数百甚至上千），导致 MCTS 搜索效率低下；
- 缺乏足够的专家对局数据用于监督学习；
- 现有模型缺乏可解释性，且在小样本下易过拟合；
- 高性能 LLM（如 GPT）虽具备生成能力，但其输出存在“幻觉”（hallucinations）和不一致性。

---

### 🚀 提出的新方法与创新思路

提出了一种**轻量级混合框架（hybrid framework）**，实现从通用大语言模型（LLM）到专用高性能博弈 AI 的“弱到强泛化”（**Weak-to-Strong Generalization**）。核心组件如下：

#### （1）**多阶段集成架构设计**
- **Graph Attention Autoencoder (GAT-AE)**：提取 MCTS 生成图结构中的拓扑特征，作为结构性先验，增强状态评估的鲁棒性。
- **Stochastic Graph Genetic Algorithm (SGGA)**：将 MCTS 树转化为图结构，引入遗传算法机制进行候选节点优化，提升搜索多样性与效率。
- **GPT-4o-mini 生成合成数据**：无需真实人类对局记录，利用 LLM 自动生成带有噪声的动作评分数据，降低数据获取成本。

#### （2）**两阶段训练机制**
- **训练阶段**：使用 GPT-4o-mini 生成带噪标签数据，通过 GAT-AE 和 SGGA 联合训练 UCT-AE 模型；
- **应用阶段**：冻结模型参数，结合 MCTS + GAT-AE + SGGA 实现高效推理。

#### （3）**深度归一化更新机制（Depth Normalization）**
为缓解深层节点误差累积问题，提出两级价值传播策略：
- **Pass I：深度相关积累**（Depth-Dependent Accumulation）
- **Pass II：全局深度归一化**（Global Depth Normalization），按深度线性缩放目标值，抑制远期预测不确定性。

#### （4）**无需专家示范的学习范式**
突破传统依赖 expert demonstrations 的模式，转而采用 **noisy and imperfect supervision**，验证了即使在低质量监督信号下也能演化出更强的学生模型。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 数据需求 | 依赖专家对局 / 强监督 | 使用 LLM 生成合成数据，零人工标注 |
| 可解释性 | 黑箱神经网络为主 | 显式建模移动与放置策略，结构清晰 |
| 资源消耗 | 高计算开销（GPU集群） | 支持低端硬件运行（测试于 RTX 4060 笔记本GPU） |
| 泛化能力 | 特定领域调优 | 支持 transferability，抽象为资产/行动模拟现实决策 |
| 性能表现 | 深层搜索才有效 | 小规模搜索（N=30~50）即超越教师模型 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **无真实历史对局数据**（因 Amazons 游戏冷门，公开数据稀少）；
- 所有训练数据由 **GPT-4o-mini 自动生成**，输入当前棋盘状态及动作，输出 `(move_score, place_score)` ∈ [0,1]；
- 使用 **prompt engineering** 构造指令模板（见 Appendix Table A.2），确保格式统一；
- 生成数据包含噪声（如非法动作、评分矛盾），更贴近“弱监督”设定。

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|----------|
| 棋盘大小 | 10×10 standard Amazon board |
| 对战双方 | Hybrid Model vs. GPT-4o-mini / Baseline Models |
| 每组比赛局数 | 200 局 |
| 搜索节点限制（N） | 分别测试 N=20, 30, 50 |
| 硬件平台 | AMD Radeon™ 780M + NVIDIA RTX 4060 Laptop GPU（中低端配置） |
| 主要算法流程 | MCTS + UCT-AE + SGGA + GAT-AE pipeline |

---

### 🎯 评估指标

- **Win Rate (%)**：对抗不同对手时的胜率；
- **Decision Accuracy Improvement**：相比基线提升 15%–56%；
- **Loss Curve Stability**：训练损失收敛性分析；
- **Variance Analysis**：F-test 检验不同任务损失方差差异；
- **Ablation Study**：消融各模块以验证互补性。

---

### 🆚 基线方法对比

| 基线模型 | 描述 |
|--------|------|
| **GPT-4o-mini** | 教师模型（teacher model），直接生成动作建议 |
| **UCTS-AE** | 仅使用 autoencoder 改进 UCT 的探索-利用平衡 |
| **SGGA** | 仅基于随机图遗传算法选择节点 |
| **GAT-AE** | 仅使用图注意力自编码器进行结构建模 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 对手 | N=30 胜率 | N=50 胜率 | 提升幅度 |
|------|----------|----------|---------|
| **GPT-4o-mini** | **45.0%** | **66.5%** | ➕21.5% ↑ |
| **UCTS-AE** | 73.5% (N=30) | — | — |
| **SGGA** | 59.0% (N=30) | — | — |
| **GAT-AE** | 57.5% (N=30) | — | — |

> 💡 在仅有 **N=50 搜索节点**的情况下，模型已实现对教师模型的**决定性胜利**（>66% 胜率），证明其极高的搜索效率。

---

### 📊 决策准确率提升
- 相比所有 baseline 模型，本框架在决策准确性上实现了 **15% 至 56% 的绝对提升**；
- 即使在极小搜索预算下（N=30），仍能保持竞争力。

---

### 🔍 消融实验结果（Ablation Studies）

| 模块移除 | 影响说明 |
|--------|--------|
| **无 SGGA** | 节点采样多样性下降，搜索陷入局部最优 |
| **无 GAT-AE** | 无法有效捕捉图结构信息，抗噪能力减弱 |
| **无双 Autoencoder 设计** | 移动与放置策略耦合，性能下降明显 |
| **未使用深度归一化** | 深层节点估值偏差大，回传不稳定 |

> ✅ 实验表明：**GAT-AE 与 SGGA 具备互补性**——前者提供结构归纳偏置，后者增强随机探索能力，二者协同显著优于单一模块。

---

### 📉 训练损失分析（Loss Analysis）

- **Movement Task**：
  - 初始 MSE ≈ 0.04 → 快速降至 0.01 并稳定；
  - 方差：8.0×10⁻⁶（较低），收敛平稳。
- **Placement Task**：
  - 初始 MSE ≈ 0.03 → 下降后趋于平台期；
  - 方差：2.1×10⁻⁵（较高），波动更大（F-test p=0.035，差异显著）；
- 原因推测：SGGA 用于移动选择提升了数据质量，而放置采用加权随机采样，引入更多噪声。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **成功实现 Weak-to-Strong Generalization**
   - 尽管 GPT-4o-mini 是“教师”，但其生成的数据含噪声；
   - 通过 GAT-AE 的图注意力机制充当 **information bottleneck**，有效过滤 hallucinations；
   - 最终学生模型反超教师，验证了“弱监督→强智能”的可行性。

2. **GAT 具备天然去噪能力**
   - 图注意力只能学习拓扑模式（如连通性、控制区域），不能记忆随机错误；
   - 因此自动“蒸馏”出战略层面的一致性策略，实现 **denoising by structure**。

3. **资源效率极高**
   - 在普通笔记本 GPU 上即可运行；
   - 极低搜索节点（N=50）即可击败强大 LLM，适合边缘部署。

4. **框架具有可迁移潜力**
   - 抽象为“资产-行动-阻断”模型，可用于路径规划、资源调度等现实决策场景。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **训练充分性判断困难** | 当前依赖 loss 曲线，但后期变化微弱，难以确定是否收敛 |
| **最终决策策略简单** | 当前采用随机选择最优动作，尚未引入 meta-strategy 优化 |
| **未与最强手工引擎比较** | 如 Invader 等专业 Amazons 引擎未纳入对比（因其依赖大量人工规则） |
| **泛化至其他游戏待验证** | 当前仅在 Amazons 上验证，需进一步扩展至 Hex、Go 等类似游戏 |

---

### 🔮 未来工作方向

1. **建立训练终止准则**
   - 探索基于性能 plateau 或梯度稳定性的自动判停机制。

2. **开发高级决策策略**
   - 引入 meta-controller 或 policy ensemble 来融合多个子模型输出。

3. **拓展至多智能体与动态环境**
   - 将框架应用于 real-time strategy games 或 robotic navigation。

4. **探索更高效的 LLM 蒸馏方式**
   - 结合 active learning 动态筛选高质量合成样本，减少冗余训练。

5. **开源轻量化推理版本**
   - 推动在移动端或嵌入式设备上的实际部署。

---

> 🏁 **总结一句话**：  
> 本文首次展示了如何在**无专家数据、低算力条件**下，通过 **LLM + Graph Learning + Evolutionary Search** 的混合范式，构建一个高效、鲁棒且可演化的博弈 AI，为资源受限场景下的智能决策提供了新路径。

</details>

---

### 10. [Reason and Verify: A Framework for Faithful Retrieval-Augmented Generation](https://arxiv.org/abs/2603.10143)

**Authors**: Eeham Khan, Luis Rodriguez, Marc Queudot  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.10143v1  

#### Abstract
Retrieval-Augmented Generation (RAG) significantly improves the factuality of Large Language Models (LLMs), yet standard pipelines often lack mechanisms to verify inter- mediate reasoning, leaving them vulnerable to hallucinations in high-stakes domains. To address this, we propose a domain-specific...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Reason and Verify: A Framework for Faithful Retrieval-Augmented Generation**

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
- **RAG系统在高风险领域（如医学）中的幻觉问题**：尽管Retrieval-Augmented Generation（RAG）提升了大语言模型（LLMs）的事实准确性，但标准RAG流程缺乏对中间推理过程的验证机制，容易产生细微的幻觉（如错误日期、实体混淆），尤其是在术语严格、事实要求高的专业领域。
- **检索质量敏感性和推理不透明性**：端到端性能高度依赖于检索质量，且多数RAG系统缺少显式的推理与验证步骤，导致错误难以诊断和纠正。
- **通用RAG框架在特定领域的适应性不足**：现成的LLMs和通用RAG框架在生物医学等专业领域表现不佳，因缺乏领域适配的检索策略和持续更新的知识库支持。

### **提出了什么新方法或新思路**
本文提出了一种**面向特定领域的、具有显式推理与保真度验证机制的RAG框架**，其核心创新包括：

1. **Domain-Specific RAG Blueprint with Explicit Verification Gates**
   - 构建了一个模块化的工作流，整合了：
     - **Neural Query Rewriting**：使用GPT-4o重写模糊查询，扩展缩写并添加精确医学术语。
     - **BGE-based Cross-Encoder Reranking**：基于语义对齐对BM25初检结果进行重排序，提升证据精度。
     - **Rationale Generation Module**：引导生成器分解问题为子主张，并引用具体证据片段（passage ID + span）来支撑每个主张。
     - **Verification Gate**：引入八分类保真度评估体系，判断每条理由陈述是否忠实于检索文档。

2. **Statement-Level Faithfulness Framework for Biomedical Rationales**
   - 提出一个**八类别验证分类法**（见下表），用于细粒度评估理由的保真度，区分显式支持与隐式推断，便于结构化错误归因（是检索失败还是生成错误）。

| 类别 | 含义 |
|------|------|
| CORRECT-EXPLICIT | 信息在文档中明确陈述（直接引用或转述） |
| CORRECT-IMPLICIT | 从事实线索逻辑推断得出，未直接陈述 |
| CORRECT-ADDITIONAL | 正确但补充了外部相关信息 |
| CORRECT-MISSING | 结论正确但引用文档无支持 |
| INCORRECT-FALSE | 陈述与证据矛盾 |
| INCORRECT-DEVIATING | 内容偏离主题 |
| INCORRECT-ILLOGICAL | 推理存在内部矛盾或违反科学原则 |
| INCORRECT-MISSING | 推理错误且引用文档无关 |

3. **Systematic Evaluation under Token/Latency Constraints**
   - 在受限token预算下系统评估了动态in-context learning（ICL）和reranking的影响，探索如何在资源限制下优化few-shot性能。

### **相比现有方法的优势**
- **更高的准确率与更强的鲁棒性**：通过显式理由生成和验证机制减少幻觉，在BioASQ和PubMedQA上达到与更大模型相当甚至更优的表现。
- **更好的可解释性与可诊断性**：显式生成带证据链接的理由，使系统决策过程透明，便于人工审查和错误分析。
- **轻量级高效架构**：仅使用PubMed单一语料库 + BM25 + BGE重排，而非多检索器融合或多源知识集成，仍能媲美复杂系统（如MedRAG+RRF-4）。
- **动态演示选择优于静态ICL**：KNN-based动态选取上下文示例显著提升few-shot性能，避免“越多示例越差”的退化现象。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **BioASQ**：专家标注的二元（yes/no）生物医学问答数据集，需基于文献推理验证主张。
- **PubMedQA**：三元分类（yes/no/maybe）研究型问题数据集，答案可从PubMed摘要中获得。
- 所有实验遵循**MIRAGE benchmark**设定，采用**question-only retrieval**（无黄金段落监督），模拟真实医疗信息检索场景。

### **实验设置和评估指标**

#### **模型与组件**
- **主干模型**：`Llama-3-8B-Instruct`
- **检索器**：初始使用MedCPT，后切换为**BM25**（效率更高且性能相近）
- **重排器**：`BGE-v2-m3` cross-encoder，将top-20候选重排后取top-5作为最终证据集 $E$
- **查询重写器**：GPT-4o，当词法重叠 < 0.3 或证据得分 < 0.5 时触发
- **理由验证器**：GPT-4o 对每个原子陈述打标签（八分类）

#### **评估指标**
- **Accuracy**：分类任务的标准准确率（BioASQ为二类，PubMedQA为三类）
- **Faithfulness Score**：
  $$
  \text{Faith}(R) = \frac{1}{n} \sum_{j=1}^{n} \mathbb{I}_j,\quad \text{其中}\ \mathbb{I}_j = 1\ \text{当且仅当第}\ j\ \text{个陈述属于任何CORRECT-*类别}
  $$
- **Inter-Annotator Agreement**：
  - Cohen’s Kappa（K）衡量人类之间及人-LMM之间的标注一致性
  - Per-category F1 分析各类别的匹配程度

#### **对比配置（Comparative Approaches）**
| 方法 | 描述 |
|------|------|
| Vanilla RAG | BM25检索 + 直接生成答案，无中间推理 |
| InstructRAG-ICL | 加入显式reasoning via in-context learning |
| InstructRAG w/ Reranker | 上述基础上加入BGE重排 |
| InstructRAG w/ Reranker + Dynamic ICL | 动态选取最相似训练样例作为演示 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
| 方法 | BioASQ-Y/N (%) | PubMedQA* (%) |
|------|----------------|---------------|
| **Ours (Best)** | **89.1** | **73.0** |
| MedRAG + GPT-3.5 | 90.29 | 67.40 |
| MedRAG + GPT-4 | 92.56 | 70.60 |
| Ours (Vanilla RAG) | 82.3 | 70.0 |

> 注：我们的方法基于 `Llama-3-8B-Instruct`（约10倍小于GPT-4），但在PubMedQA上**超过MedRAG+GPT-4达2.4个百分点**。

### **与基线方法的对比结果**
- 显式**理由生成**（0-shot）即可将BioASQ从82.3%提升至85.8%，PubMedQA从70.0%提升至73.0%，说明结构化推理本身有助于抑制幻觉。
- 引入**BGE重排**进一步提升性能，尤其在few-shot设置下效果显著。
- **动态ICL**大幅优于静态ICL：
  - 在4-shot BioASQ中，动态选择比静态高出**+14.5 pts**（86.2% vs 71.7%）
  - 静态ICL随样本增加性能下降，而动态ICL保持稳定或上升

### **消融实验结果**
#### **Table 3 节选：不同配置下的性能变化**

| Configuration | BioASQ (3-shot) | PubMedQA (4-shot) |
|---------------|------------------|--------------------|
| w/o Reranking | 77.7 | 47.5 |
| w/ Reranking | 76.4 | 60.0 |
| △ (Reranking) | -1.3 | **+12.5** ✅ |
| Dynamic ICL (vs Static) | **+12.7** ✅ | **+9.0** ✅ |

- **重排（Reranking）** 对PubMedQA帮助极大（+12.5 pts at 4-shot），表明其有效过滤噪声文档，防止误导模型。
- **动态演示选择** 是性能提升的关键驱动力，逆转了静态ICL的负向趋势。

---

## 4. **关键结论和发现**

### **主要发现**
1. **显式理由生成显著提升准确性与保真度**：强制模型先输出证据链接的理由再作答，能有效减少幻觉，即使在小模型上也能实现接近大模型的性能。
2. **动态ICL + robust reranking 是few-shot success的关键**：相比于固定示例，基于语义相似性的动态选取大幅提升上下文学习效率。
3. **推理侧改进可补偿检索复杂性**：本工作仅用单一语料库和简单检索流程（BM25+BGE），却能达到甚至超越使用四检索器融合+多源知识库的复杂系统（如MedRAG），说明**高质量推理机制的重要性不低于复杂检索架构**。
4. **LLM-as-a-judge 相对宽松**：自动验证器（GPT-4o）比人类更宽容，尤其接受更多“implicit”推理；人类间也存在主观差异（如某例评分0.75 vs 0.30），提示需更清晰的标注指南。

### **局限性**
1. **评估范围有限**：仅在两个英文生物医学QA数据集上测试，泛化性未知；未覆盖multi-hop、因果推理等复杂任务。
2. **部分依赖闭源API**：查询重写与验证模块使用GPT-4o，带来延迟与成本问题，不利于大规模部署。
3. **非临床环境验证**：未在真实临床流程中由医生参与测试，当前仅为研究原型，不可直接用于临床决策支持。
4. **统计显著性未检验**：所有结果均为单次运行点估计，缺乏置信区间或p值支持。
5. **人类评估样本极小**：仅4个样例由2位标注者评估，不足以可靠估计一致性或验证自动化指标。

### **未来工作方向**
- 探索开源替代方案（如用Llama-3替代GPT-4o）以降低延迟与成本。
- 开展更大规模的人类评估，建立标准化的医学理由标注协议。
- 将该框架扩展至其他专业领域（如法律、金融）和其他任务形式（如摘要、报告生成）。
- 系统研究query rewriting策略与阈值敏感性。
- 集成实时反馈机制，实现闭环的correction & refinement loop。

---

> 🔗 **Demo**: 可交互演示已发布于 HuggingFace → [https://huggingface.co/spaces/DialogueRobust/RobustDialogueDemo](https://huggingface.co/spaces/DialogueRobust/RobustDialogueDemo)  
> 📚 **代码与复现性**：文中提供了完整prompt模板（Appendix B），确保实验可复现。

</details>

---

### 11. [Quantifying Membership Disclosure Risk for Tabular Synthetic Data Using Kernel Density Estimators](https://arxiv.org/abs/2603.10937)

**Authors**: Rajdeep Pathak, Sayantee Jana  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.10937v1  

#### Abstract
The use of synthetic data has become increasingly popular as a privacy-preserving alternative to sharing real datasets, especially in sensitive domains such as healthcare, finance, and demography. However, the privacy assurances of synthetic data are not absolute, and remain susceptible to membershi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Quantifying Membership Disclosure Risk for Tabular Synthetic Data Using Kernel Density Estimators*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**合成数据（synthetic data）中的成员推断攻击（Membership Inference Attacks, MIAs）风险量化问题**。尽管合成数据被广泛用于保护隐私（如医疗、金融等领域），但其仍可能泄露训练数据中个体的存在信息，从而引发隐私泄露风险。现有方法在评估此类风险时存在计算开销大或缺乏概率化输出等局限。

### 提出的新方法与新思路
作者提出了一种基于 **Kernel Density Estimators (KDE)** 的非参数、距离驱动的框架，用于对表格型合成数据进行成员披露风险的量化。核心思想是：
- 利用 Gower’s distance 计算真实记录与合成数据之间的最近邻距离；
- 分别对“训练成员”和“非成员”的距离分布拟合 KDE 模型；
- 基于贝叶斯定理推导出给定距离下的**成员概率** $ P(\text{member}|d) $，实现**概率化的成员推断**。

并设计了两种攻击模型：
- **True Distribution Attack**：假设数据持有者可访问真实训练标签，用于内部风险评估；
- **Realistic Attack**：仅依赖辅助数据集（无真实标签），更贴近实际攻击场景。

### 相比现有方法的优势
| 维度 | 本文方法（KDE-based） | 现有主流方法 |
|------|------------------------|-------------|
| **是否需要 Shadow Models** | ❌ 不需要 | ✅ 多数需要（如[2][3][4]），计算昂贵 |
| **输出形式** | ✅ 概率化预测（支持 ROC 分析） | ❌ 多为硬分类（如 Method 1 使用阈值判断） |
| **计算效率** | ✅ 高效，适合大规模部署 | ❌ Shadow modeling 资源消耗巨大 |
| **实用性** | ✅ 可作为 post-generation metric 快速评估风险 | ⚠️ 多数难以在生产环境中实时应用 |

> ✅ **核心优势总结**：无需 shadow models、提供概率输出、计算高效、适用于真实世界的数据发布前风险评估。

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用四个公开的真实世界 tabular 数据集：
- **MIMIC-IV**：电子健康记录（EHR），含数值与类别特征
- **UK Census**：英国人口普查数据，全为**分类变量**
- **Texas-100X**：德克萨斯州住院患者数据
- **Nexoid COVID-19**：新冠生存数据

每个数据集被均分为两部分：
- 一半用于训练生成模型并生成 synthetic data $ S $
- 另一半作为 unseen data $ U $ 构成攻击数据集中非成员部分

### 合成数据生成器（Generators）
采用六种不同机制的生成模型：
- **CTGAN**, **ADS-GAN**, **DPGAN**（GAN-based）
- **TabDDPM**（Diffusion Model）
- **TVAE**（Variational Autoencoder）
- **Bayesian Network**

所有模型通过 **SynthCity framework** 实现以保证公平比较。

### 攻击设置与流程
1. 构造攻击数据集 $ D_{\text{attack}} = R \cup U $，其中 $ R $ 是训练数据（成员），$ U $ 是未见数据（非成员）
2. 对每条记录计算其到 $ S $ 中最近邻的 **Gower’s distance**
3. 将距离与真实标签组成 $ D_{\text{dists}} $
4. 划分训练/测试集（70%/30%），平衡正负样本
5. 在训练集上分别拟合 KDE_member 和 KDE_non-member
6. 在测试集上利用公式进行概率预测：

$$
P(\text{member}|d) = \frac{\text{KDE}_{\text{member}}(d)}{\text{KDE}_{\text{member}}(d) + \text{KDE}_{\text{non-member}}(d)}
$$

7. 设定阈值（默认 0.5）进行分类，计算 Accuracy、F1 Score，并绘制 ROC 曲线。

### 评估指标
- **Accuracy**
- **F1 Score**
- **ROC Curve**（特别关注低 FPR 区域的 TPR）
- **Log-scaled ROC**：揭示 worst-case 泄露情况
- Kolmogorov-Smirnov (KS) test：检验 member 与 non-member 距离分布是否可区分

### 基线方法对比
主要对比以下两类方法：
- **Method 1**：基于距离阈值的传统方法（如 [5][6][7]），设定固定距离阈值 $ T $，低于则判为 member
- **Shadow Modeling Approaches**（引用文献[2][3][4]）：虽性能强但计算成本极高，不实际

> 本文强调：**在不牺牲性能的前提下避免使用 computationally expensive shadow models**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 2 & 3）

| Dataset / Generator | Best F1 (本方法) | 最脆弱模型 |
|---------------------|------------------|-----------|
| MIMIC-IV            | 0.877 (**TVAE**) | TVAE      |
| UK Census           | 0.633 (**TabDDPM**) | Bayesian Network (0.631) |
| Texas-100X          | 0.975 (**Bayesian Network**) | 极高风险 |
| Nexoid              | 0.619 (**Bayesian Network**) | Bayesian Network |

> 📌 **观察**：Bayesian Network 生成的数据普遍更容易遭受 MIA，说明其记忆性强、泛化弱。

### 与基线方法的对比结果
#### （1）vs Method 1（Distance Threshold-based）
- 在多数情况下，**Realistic Attack 版本在较高阈值下显著优于 Method 1 的 F1 分数**
- 如 Texas-100X 上 Bayesian Network 生成数据，Realistic Attack 达到 **F1 ≈ 0.98**，远超 Method 1
- 图 3 显示：随着距离阈值提升，Realistic Attack 的 F1 提升趋势明显，尤其在稀疏数据中表现更好

#### （2）概率化输出优势
- 支持完整的 ROC 分析，特别是在 **low FPR region（如 $10^{-6}$）** 发现严重泄露：
  - TVAE 生成的 UK Census 数据：准确率仅 49.97%（看似安全），但在 FPR=$10^{-6}$ 下 TPR 高达 $10^{-1} \sim 1$，即 **TPR 是 FPR 的 $10^5$ 倍！**
  - 类似现象出现在 Nexoid 数据上（TVAE 生成）
- 表明：**传统平均指标（Accuracy/F1/AUC）会掩盖 worst-case 隐私泄露风险**

### 消融实验与分析（隐含在 Realistic Attack 设计中）
- **Realistic Attack 中阈值选择的影响**（图 17）：
  - 高阈值导致更多样本被判为 member → 同时增加 TP 和 FP
  - 若 TP 增长速度快于 FP，则 F1 提高
  - Texas-100X 和 UK Census 出现此现象，而 MIMIC-IV 则相反（有利数据持有者）
- **标签噪声容忍性**：
  - Realistic Attack 使用伪标签（supposed member/non-member）建模，存在一定噪声
  - 但实验表明即使如此，仍能有效捕捉风险趋势，具备鲁棒性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **KDE-based 方法能有效建模距离分布并生成可靠的成员概率预测**
2. ✅ **无需 shadow models 即可实现高性能 MIA 风险评估，大幅降低计算成本**
3. 🔍 **平均指标（如 Accuracy、F1）可能误导隐私安全性判断**，必须结合 **log-ROC 分析** 观察低 FPR 下的行为
4. ⚠️ **某些生成模型（如 Bayesian Network）存在严重的成员泄露倾向**，不适合高隐私要求场景
5. 💡 **Realistic Attack 在特定条件下甚至超过 True Distribution Attack 的 F1 表现**，挑战直觉认知 —— 这源于距离分布本身的统计特性（如 UK Census 中 member/non-member 距离不可区分）

### 方法的局限性
- 依赖 Gower’s distance，在高维稀疏空间中可能存在距离失效问题
- 假设 attack dataset 中 member 与 non-member 数量均衡（可调整但未深入研究）
- 对于高度离散的距离值（如 UK Census 全分类变量），KDE 效果受限，出现“阶梯状”概率输出
- 未考虑属性推断攻击（attribute inference），仅聚焦于成员推断

### 未来工作方向
1. **放松均衡假设**：将 member 比例设为 $ n/N $（训练集大小 / 总体规模），提高现实适用性
2. **理论分析映射关系**：探究 distance → membership probability 的理论边界与一致性保证
3. **混合策略开发**：
   - 结合轻量级 shadow modeling
   - 或集成 adversarial training 机制，实现生成过程中的隐私增强
4. **扩展至其他数据模态**：如时间序列、图结构等
5. **GPU 加速 KDE 与距离计算**：进一步提升大规模数据处理能力（文中提及可行）

---

> 🔗 **代码与数据开放**：  
> 所有实验代码与数据已开源：[https://github.com/PyCoder913/MIA-KDE](https://github.com/PyCoder913/MIA-KDE)

✅ **总体评价**：本文提出了一种实用、高效且富有洞察力的风险评估框架，填补了从学术研究到工业落地之间的 gap，为数据管理者提供了强有力的 post-generation privacy audit 工具。

</details>

---

### 12. [Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents](https://arxiv.org/abs/2603.10564)

**Authors**: Yuanhao Li, Haozhe Wang, Geyong Min, Nektarios Georgalas, Wang Miao  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.10564v1  

#### Abstract
The integration of Generative AI models into AI-native network systems offers a transformative path toward achieving autonomous and adaptive control. However, the application of such models to continuous control tasks is impeded by intrinsic architectural limitations, including finite context window...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Adaptive RAN Slicing Control via Reward-Free Self-Finetuning Agents*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **AI-Native 网络系统**中的 **连续控制任务**（如 RAN 切片资源管理）中存在的以下挑战提出解决方案：
- **奖励工程瓶颈（Reward Engineering Bottleneck）**：传统 RL 需要人工设计复杂的多目标奖励函数，难以平衡频谱效率、服务质量（QoS）和配置稳定性等冲突目标。
- **LLM 代理的上下文限制**：现有基于 LLM 的代理（如 Reflexion）依赖 prompt-based memory，受限于有限的 context window 和 **Long Context Degradation**，无法实现真正的持续学习。
- **缺乏长期经验内化机制**：当前方法无法将长期交互经验有效地“固化”到模型参数中，导致在动态网络环境中适应能力差。

### 提出的新方法与新思路
作者提出了一种名为 **Self-Finetuning（自微调）** 的新型框架，其核心创新包括：

#### （1）Reflective MDP (R-MDP) 与 Actor-Reflector (AR) 架构
- 将传统的 MDP 扩展为 **语言反馈驱动的 R-MDP**，用自然语言形式的反思（reflection）替代标量奖励信号。
- 设计 **Actor-Reflector 双模块架构**：
  - **Actor**：LLM 主体，负责生成动作、执行决策，并进行 step-level 自我反思。
  - **Reflector**：另一个 LLM 模块，在轨迹结束后对整个交互历史进行全局分析，提供 trajectory-level 反馈与改进建议。

#### （2）双向反思机制（Bi-Perspective Reflection）
- **Step-level Reflection**：Actor 在每一步输出对前一动作的语言反思，作为 in-context learning 的一部分，实现短期策略调整。
- **Trajectory-level Reflection**：Reflector 对整条轨迹进行回溯评估，识别次优行为并建议更优动作，形成高质量偏好数据集。

#### （3）Refine-from-Reflection (RfR) 微调框架
- 利用 Kahneman-Tversky Optimization (KTO) 进行 **Preference-based Fine-tuning**，无需成对比较样本（区别于 DPO），支持非平衡数据。
- 引入 **Refine-rollout 机制**：对被标记为“次优”的状态输入多次采样，挖掘潜在的更好动作，增强样本效率。
- 实现 **经验内化（Internalization of Experience）**：通过参数更新而非 prompt 扩展来保存长期经验，突破 context window 限制。

### 相比现有方法的优势
| 维度 | 传统 RL | Reflexion 类方法 | 本文 Self-Finetuning |
|------|--------|------------------|------------------------|
| 是否需要手工奖励 | 是 | 否（但需环境反馈） | 否（完全 reward-free） |
| 学习方式 | 参数更新 | Prompt 注入记忆 | 参数更新 + 语言反思驱动 |
| 上下文依赖 | 无 | 强依赖（context window） | 弱依赖（经验内化） |
| 适用于连续控制 | 是（但训练成本高） | 否（仅适合 episodic） | 是（支持持续适应） |
| 样本效率 | 低 | 中等 | **极高**（单轨迹即可显著提升） |

---

## 2. 核心实验方法和设置

### 实验环境
- 使用基于 **ns-3** 构建的 **Python RAN slicing simulator**，模拟真实的无线通信环境。
- 考察 **inter-slice spectrum resource allocation** 任务，属于典型的多目标连续控制问题。

### 流量与信道模型
| 类别 | 参数 |
|------|------|
| **Traffic Model** | On-off 模型，模拟 GBR 与 Non-GBR 流量（见 Table I） |
| **Radio Channel** | Urban Propagation Loss Model (3GPP TR 38.901)，含频率选择性衰落 |

### 决策周期与状态空间
- **Decision Interval**: 100ms
- **State Representation $s_t$**:
  $$
  s_t = [a_{t-1}, SE_t, u_t, o_t, c_t]
  $$
  包括上一动作、频谱效率、吞吐量、排队增量、丢包大小。

### 动作空间
- 输出分配给各切片的 **Physical Resource Blocks (PRBs)** 数量。

### 评估指标
1. **Average Spectrum Efficiency (Avg. SE)**：越高越好
2. **Reconfiguration Times (C)**：衡量资源重配次数，越低表示策略越稳定
3. **Packet QoS Violation Times (V)**：违反延迟要求的数据包时间步数，越低越好
4. **Utility Score**：综合三项指标的加权得分

### 基线方法对比
分为两类：
#### （1）RL 基线（使用 Ray RLlib 实现）：
- **DQN**（Value-based）
- **PPO**（Policy-based）
- **SAC**（Maximum Entropy）

#### （2）LLM Agent 基线：
- **Reflexion**：使用 Qwen3-4B 作为 Actor，DeepSeek-R1 作为 Evaluator/Reflector

#### 本文方法配置：
- **Actor**: Qwen3-4B
- **Reflector**: DeepSeek-R1
- 保持 backbone 一致，确保公平比较

---

## 3. 主要实验结果和性能指标

### 性能对比（Table III & Fig. 2）

| Algorithm | Avg. SE | Reconf. Times | PQoS vio. | Utility |
|----------|---------|---------------|-----------|---------|
| **Self-Finetuning (Ours)** | **5.354** | **21.091** | 8.561 | **25702.2** ✅ |
| Reflexion | 5.299 | 29.454 | 8.630 | 25314.69 |
| DQN | 5.219 | 46.204 | 15.911 | 22519.1 |
| PPO | 3.587 | 51.411 | **1.997** | 19277.2 |
| SAC | 5.748 | 44.775 | 59.967 | 11704.3 |

> ✅ 表示最优或次优结果

#### 关键发现：
- **Self-Finetuning 在综合性能（Utility）上全面领先**，尤其在 **Reconfiguration Times 上降低 59% vs PPO，28.4% vs Reflexion**，表明其策略更稳定。
- 尽管 SAC 达到最高的 Avg. SE（5.748），但其 PQoS violation 高达近 60，说明严重牺牲服务质量。
- PPO 虽然 QoS 控制最好，但频谱利用率极低且频繁重配，系统开销大。
- **Self-Finetuning 实现了三者之间的最佳权衡**。

### 样本效率分析（Fig. 3）
- RL 方法每轮收集 20 条轨迹，共训练 80 轮（总计 1,600 条轨迹），仍存在震荡与不稳定。
- **Self-Finetuning 仅使用 1 条轨迹 + 1 次训练迭代**，即实现显著性能跃升：
  - Reconfiguration 减少约 **33%**
  - PQoS violation 更稳定
  - Avg. SE 提升
- 通过 **6 轮 KTO fine-tuning + refine-rollout**，从单条轨迹中不断提取改进信号，实现“小样本高效学习”。

### 消融实验隐含结论（文中未明确列出消融表，但从机制可推断）：
- **Reflector 的 trajectory-level 分析至关重要**：相比 Reflexion 的 prompt 注入，Reflector 提供的结构化反馈更能引导有效学习。
- **KTO + rollout 机制显著提升样本利用效率**：即使初始输出不佳，也能通过采样发现潜在优质动作。
- **经验内化优于 prompt 扩展**：避免了 context truncation 和 long context degradation 问题。

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **无需手工奖励函数即可实现高性能连续控制**：通过语言反思自动生成优化信号，打破 RL 对 reward engineering 的依赖。
2. ✅ **LLM 可以通过 self-finetuning 实现持续学习**：将长期经验压缩进模型参数，克服 context window 限制，适用于 AI-Native 网络的持久运行需求。
3. ✅ **RfR 框架显著提升样本效率**：单轨迹即可完成有效策略优化，远超传统 RL 与 prompt-based LLM agents。
4. ✅ **在 RAN slicing 多目标优化中取得 SOTA 表现**：在频谱效率、服务质量和配置稳定性之间实现了最优平衡。

### 方法的局限性
- **推理延迟较高**：LLM 推理速度慢，目前难以满足毫秒级实时控制需求（如 100ms 决策周期接近极限）。
- **依赖强大 LLM 能力**：Actor 与 Reflector 均需高性能 LLM 支持，部署成本高。
- **尚未验证大规模场景**：实验基于单一基站与少量切片，扩展性有待进一步测试。

### 未来工作方向
1. **模型轻量化**：采用 **imitation learning** 或 **policy distillation** 将学到的策略迁移到小型模型（如 TinyML）用于实际部署。
2. **加速推理**：探索 **模型量化（quantization）、硬件加速（GPU/TPU/NPU）** 等技术降低延迟。
3. **跨场景泛化**：研究该框架在其他网络控制任务（如 bitrate adaptation、mobility management）中的适用性。
4. **在线持续学习机制优化**：设计更高效的 fine-tuning pipeline，支持无限 horizon 的 lifelong learning。

---

> 📌 **总结一句话**：  
> 本文提出的 **Reward-Free Self-Finetuning 框架** 成功将 LLM 的语义推理能力与持续学习机制结合，首次实现了无需手工奖励、高样本效率、强稳定性的 RAN slicing 控制，在迈向 **AI-Native 网络自治** 的道路上迈出关键一步。

</details>

---

### 13. [GaLoRA: Parameter-Efficient Graph-Aware LLMs for Node Classification](https://arxiv.org/abs/2603.10298)

**Authors**: Mayur Choudhary, Saptarshi Sengupta, Katerina Potika  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.10298v1  

#### Abstract
The rapid rise of large language models (LLMs) and their ability to capture semantic relationships has led to their adoption in a wide range of applications. Text-attributed graphs (TAGs) are a notable example where LLMs can be combined with Graph Neural Networks to improve the performance of node c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GaLoRA: Parameter-Efficient Graph-Aware LLMs for Node Classification

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **Text-Attributed Graphs (TAGs)** 中节点分类任务中如何有效融合**图结构信息**与**文本语义信息**的挑战。传统方法要么依赖 GNN 捕获结构、PLM 处理文本，联合训练则计算开销大；而现有参数高效方法（如 GraphAdapter）冻结 LLM 导致无法充分学习任务相关的语义知识。

### 🚀 提出的新方法：GaLoRA（Graph-aware Low-Rank Adaptation）
提出一种**模块化、参数高效的框架 GaLoRA**，将结构感知嵌入注入到 LLM 微调过程中，具体设计为两阶段流程：

- **Phase 1**: 使用 GNN（如 GraphSAGE）在 TAG 上训练，生成结构感知的节点嵌入（Pass-1 和 Pass-2 表示 1-hop 与 2-hop 邻域聚合）。
- **Phase 2**: 在 LLM 微调时，通过 **LoRA（Low-Rank Adaptation）机制**，将 GNN 学得的结构嵌入注入到 LLM 的中间层（Pass-1）和上层（Pass-2），实现结构-语义融合。

该方法**解耦了 GNN 与 LLM 的训练过程**，避免联合反向传播带来的高成本。

### ⭐ 相比现有方法的优势
| 方法 | 是否微调 LLM | 参数效率 | 结构注入方式 | 主要缺点 |
|------|---------------|-----------|----------------|----------|
| GLEM | 是（全量微调） | 低（~100% PLM 参数） | EM 迭代伪标签 | 对噪声敏感，迭代耗时 |
| TAPE | 否（仅提示） | 高 | 手工 prompt + 小模型解释 | 性能不稳定，依赖 prompt 质量 |
| GraphAdapter | 否（冻结 LLM） | 极高（<0.015%） | GNN Adapter 注入隐藏层 | 无法适应任务特定语义 |
| **GaLoRA (Ours)** | **是（LoRA 微调）** | **极高（仅 0.24%）** | **LoRA 注入结构嵌入** | **当前限于节点分类** |

> ✅ **核心优势**：  
> - 在保持极低可训练参数量的同时，允许 LLM 进行语义微调；
> - 实现结构与语义的有效对齐；
> - 模块化设计便于部署与扩展。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在三个真实世界的 TAG 数据集上进行评估：

| 数据集 | #节点 | #边 | 任务 | 输入文本 | 评估指标 |
|--------|-------|-----|------|------------|-----------|
| **ArXiv** | 46,198 | 78,548 | 40类论文分类 | 标题+摘要 | Accuracy |
| **Instagram** | 11,339 | 144,010 | 商业/非商业用户二分类 | 用户 bio | ROC-AUC |
| **Reddit** | 33,434 | 198,448 | 流行/非流行用户二分类 | 最近三篇帖子 | Accuracy |

> 注：由于资源限制，ArXiv 使用的是前人提供的 46K 子图。

### ⚙️ 实验设置
- **硬件环境**：Google Colab + NVIDIA A100 GPU（52GB VRAM）
- **工具库**：PyTorch, PyTorch Geometric, HuggingFace Transformers
- **GNN 模型**：GraphSAGE（2 层消息传递，输出维度 64）
- **LLM 模型**：GPT-2 和 RoBERTa（均为 ~125M 参数）
- **LoRA 设置**：
  - 应用于 6 个 Transformer 层（中层：5,6,7 注入 Pass-1；上层：9,10,11 注入 Pass-2）
  - LoRA rank $ r = 4 $
  - 可训练参数仅占 GPT-2 总参数的 **0.238%**

### 📊 评估指标
- ArXiv & Reddit：**Accuracy**
- Instagram：**ROC-AUC**
- 所有结果报告 5 次随机种子运行的均值 ± 标准差

### 🔁 基线方法对比
主要对比以下模型：
- **GNN-only**：仅用 GraphSAGE 分类
- **LLM-only / LLM (LoRA)**：仅文本输入，无结构信息
- **GLEM**：交替训练 GNN 与 LLM，使用伪标签
- **TAPE**：基于 prompt 的 LLM 输出解释 → GNN 输入
- **GraphAdapter**：当前 SOTA，冻结 LLM，用轻量 GNN Adapter 注入结构

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（LLM 控制组，同 Backbone 对比）

| Model | ArXiv (Acc) | Instagram (ROC-AUC) | Reddit (Acc) |
|-------|-------------|------------------------|--------------|
| **GraphAdapter (RoBERTa)** | 0.7273 | 0.6292 | 0.6379 |
| **GaLoRA (RoBERTa)** | 0.7234 | **0.6392** | 0.6464 |
| **GraphAdapter (GPT-2)** | 0.7325 | 0.6276 | 0.6441 |
| **GaLoRA (GPT-2)** | **0.7550** | **0.6420** | **0.6611** |

> ✅ **结论**：GaLoRA 在所有数据集上**达到或超越 GraphAdapter**，尤其在 GPT-2 背景下表现更优。

### 💰 参数效率对比（Trainable Parameters）

| 方法 | PLM | 可训练参数数 | 占 PLM 参数比例 |
|------|-----|----------------|------------------|
| GLEM | DeBERTa-Large (435M) | 435M | 100% |
| GraphAdapter | LLaMA2-13B (13B) | 2M | 0.015% |
| **GaLoRA (Ours)** | **GPT-2 (124M)** | **0.295M** | **0.238%** |

> ✅ GaLoRA 是唯一一个既微调 LLM 又保持超高参数效率的方法（<0.24%），远优于全量微调方案。

### 🔍 消融实验结果（Ablation Studies）

#### （1）LoRA Rank 影响（Instagram, GPT-2）
| Rank $ r $ | ROC-AUC |
|------------|---------|
| 2          | 0.6347  |
| 4          | 0.6420  |
| 8          | 0.6421  |

> ➕ 提升有限，$ r=4 $ 已接近饱和，说明**小秩即可获得良好性能**。

#### （2）Prompt Engineering 影响（Instagram）
| Prompt | ROC-AUC |
|--------|---------|
| No prompt | 0.6283 |
| "Classify:" | 0.6305 |
| "Classify this instagram account bio:" | 0.6344 |
| "Classify instagram account is commercial or not:" | **0.6420** |

> ➕ 明确的任务指令有助于提升性能，但增益有限，结构注入仍是主导因素。

#### （3）不同 GNN 骨干比较（GaLoRA 配置）
| 模型 | Instagram | Reddit | ArXiv |
|------|-----------|--------|-------|
| GraphSAGE + GPT-2 | 0.6420 | 0.6611 | 0.7550 |
| GAT + GPT-2 | **0.6616** | 0.6613 | 0.7569 |

> ➕ 使用 GAT 替代 GraphSAGE 在 Instagram 上显著提升，表明**更强的 GNN 骨干可进一步增强性能**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **结构信息可以通过 LoRA 高效注入 LLM**，无需联合训练即可实现结构-语义融合。
2. 即使使用较小的 LLM（如 GPT-2），GaLoRA 也能在节点分类任务上**媲美甚至超越基于更大 LLM（如 LLaMA-13B）的 SOTA 方法**。
3. **仅需微调约 0.24% 的参数**即可取得优异性能，适合资源受限场景（边缘设备、快速部署）。
4. **中层注入局部结构（Pass-1）、上层注入全局结构（Pass-2）的设计合理**，符合 LLM 层次化语义构建机制。
5. 引入**可学习门控机制（learnable gate）** 能动态调节文本与结构信息权重，提升鲁棒性。

### ⚠️ 方法的局限性
- 当前仅验证于 **node classification** 任务，尚未拓展至 link prediction、graph classification 等。
- 所有实验基于 **stratified split**，可能存在信息泄露风险（因 GNN 消息传递跨越划分边界）。
- GNN 与 LLM 的训练完全解耦，可能损失端到端优化带来的协同增益。
- 未探索更复杂的融合策略（如交叉注意力、门控融合网络）。

### 🔮 未来工作方向
1. 将 GaLoRA 扩展至其他图学习任务：**link prediction**, **community detection**, **graph classification**。
2. 探索更先进的 GNN 架构（如 GATv2, Graphormer）作为结构编码器。
3. 设计更智能的结构注入机制，例如基于注意力的自适应融合。
4. 在新提出的 TAG benchmarks 上进行全面评估（如 OGB-TAG）。
5. 研究结构感知预训练（Structural-aware pretraining）以进一步提升泛化能力。

---

## ✅ 总结
GaLoRA 提出了一种新颖且高效的 **Graph-aware LLM fine-tuning 框架**，通过 **LoRA 注入 GNN 学得的结构嵌入**，实现了在极低参数成本下的高性能节点分类。其实验充分证明了“**结构引导语义理解**”的有效性，并为未来构建轻量化、可扩展的图-语言联合模型提供了重要范式。

</details>

---

### 14. [Leech Lattice Vector Quantization for Efficient LLM Compression](https://arxiv.org/abs/2603.11021)

**Authors**: Tycho F. A. van der Ouderaa, Mart van Baalen, Paul Whatmough, Markus Nagel  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.11021v1  

#### Abstract
Scalar quantization of large language models (LLMs) is fundamentally limited by information-theoretic bounds. While vector quantization (VQ) overcomes these limits by encoding blocks of parameters jointly, practical implementations must avoid the need for expensive lookup mechanisms or other explici...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Leech Lattice Vector Quantization for Efficient LLM Compression**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决的问题**
传统的大语言模型（LLM）压缩主要依赖**标量量化（scalar quantization）**，其在信息论上存在根本性限制——即无法达到最优的率失真（rate-distortion）性能。尽管向量量化（Vector Quantization, VQ）理论上更优，但实际应用中面临两大挑战：
- **显式码本存储成本高**：随着维度增加，码本大小呈指数增长；
- **最近邻搜索开销大**：难以高效实现高维空间中的快速编码/解码。

本文旨在设计一种**无需显式存储码本、支持高效搜索与索引的高维向量量化方案**，以突破当前LLM低比特量化的性能瓶颈。

---

### ✅ **提出的新方法：LLVQ（Leech Lattice Vector Quantization）**
基于**Leech lattice（Λ₂₄）** 构建了一种新型向量量化框架，其核心创新如下：

#### （1）**利用Leech lattice的数学最优性**
- Leech lattice是24维中唯一已知具有**最优球体堆积密度**和**最大接触数（kissing number）** 的格。
- 具有高度对称性和丰富的壳层结构（shell structure），适合构建高性能球面码（spherical codes）。

#### （2）**扩展Adoul & Barth算法，实现三大功能增强**
| 功能 | 创新说明 |
|------|--------|
| **支持索引化（indexing）** | 设计双射映射机制，将每个格点唯一编码为紧凑bitstring，无需物化整个codebook |
| **支持多壳联合搜索（multi-shell angular search）** | 支持shape-gain量化，通过归一化方向+独立增益编码提升率失真性能 |
| **全并行化解码内核（fully parallelizable dequantization kernel）** | 基于模运算实现GPU友好的快速重构 |

#### （3）**引入层级索引体系**
采用三级分层编码策略：
1. **Shell Level**：按平方范数排序壳层；
2. **Class Level**：每壳内按leader分类；
3. **Local Symmetry Level**：类内通过Golay refinement、符号模式、排列共设进行细粒度编码。

---

### ✅ **相比现有方法的优势**
| 方法 | 维度 | 是否需码本 | 是否支持灵活比特率 | 性能优势 |
|------|-----|------------|------------------|----------|
| **Scalar Q (e.g., GPTQ)** | 1D | 否 | 是 | 表达能力弱，失真高 |
| **Unstructured VQ (e.g., GPTVQ)** | 高维 | 是（显式） | 否 | 存储爆炸 |
| **Quip# / E8P** | 8D | 否 | 否（依赖RVQ扩展） | 次优格结构 |
| **QTIP / PVQ** | 可变 | 否 | 是 | 编码复杂度较高 |
| **LLVQ (本文)** | **24D** | **否（隐式）** | **是（任意比特率）** | **最优几何结构 + 更高效表达** |

> ✔️ **理论优势**：更高维、更密集的格带来更低的量化噪声  
> ✔️ **工程优势**：无码本、可并行、支持多种bitwidth配置  

---

## 2. **核心实验方法和设置**

### ✅ **数据集**
- **理想源测试**：i.i.d. 高斯分布 $ \mathcal{N}(0, I) $
- **真实LLM测试**：
  - **模型家族**：Llama-2（7B）、Llama-3（8B）、Ministral-3（8B）、Qwen-v3（4B/8B）
  - **校准集**：DCLM-edu 数据集上的 6,100 条序列（用于计算Hessian矩阵）

---

### ✅ **实验设置**
#### （1）**量化方式**
- **Post-Training Quantization (PTQ)** 主流设定
- 分层Hessian修正（Hessian-based correction）补偿误差
- 可选轻量微调（fine-tuning仅更新输入scale参数）

#### （2）**评估指标**
| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | 在 Wikitext-2 上的困惑度（context length=4096） |
| **Downstream Tasks** | MMLU（理解）、CSR（常识推理）等任务准确率 |
| **SQNR (Signal-to-Noise Ratio in bits)** | 对高斯源的量化信噪比：$ \text{SQNR}_{\text{bits}} = -\log_2(\text{MSE}) $ |
| **Retention (%)** | 相对于香农限的性能保留比例：$ \frac{\text{SQNR}_{\text{bits}}}{R} \times 100\% $ |

---

### ✅ **基线方法对比**
| 基线方法 | 类型 | 所用格/结构 |
|---------|------|-------------|
| **RTN / GPTQ / Quarot** | 标量量化 | 无 |
| **Quip# / E8P** | 8维格量化（E₈ lattice） | 显式利用E₈结构 |
| **QTIP** | Trellis-based VQ | 路径编码结构 |
| **PVQ** | Pyramid VQ | 角度优先的锥形编码 |
| **AQLM** | 多阶段量化 | 结合残差与乘积码 |

---

## 3. **主要实验结果和性能指标**

### ✅ **关键性能数据**

#### （1）**在高斯源上的SQNR表现（2 bits/dim）**
| 方法 | MSE ↓ | SQNR (bits) ↑ | Retention (%) ↑ |
|------|-------|---------------|------------------|
| Uniform | 0.15 | 1.37 | 69% |
| E8P / Quip# | 0.092 | 1.72 | 86.1% |
| **LLVQ (spherical shaping)** | **0.084** | **1.79** | **89.4%** |
| **LLVQ (shape-gain)** | **0.078** | **1.84** | **92.1%** |
| **理论极限（Shannon bound）** | **0.0625** | **2.00** | **100%** |

> 🔹 **LLVQ达到当前最高SQNR，接近香农限的92.1%**

---

#### （2）**在真实LLM上的PTQ性能（2 bits/weight）**

##### 表：Llama-2 7B 在 Wikitext-2 和下游任务的表现（无微调）
| 方法 | Wiki PPL ↓ | MMLU ↑ | CSR ↑ |
|------|------------|--------|-------|
| Baseline (FP16) | 5.11 | 45.7 | 70.4 |
| GPTQ+Rotation | 41.87 | 27.0 | 41.7 |
| Quip# | 7.96 | 30.5 | 61.4 |
| **LLVQ (spherical)** | **7.61** | **33.4** | **62.1** |
| **LLVQ (shape-gain)** | **6.83** | **34.9** | **64.6** |

> 🔹 **LLVQ显著优于Quip#，尤其在shape-gain模式下PPL降低近15%**

---

##### 表：加入轻量微调后的性能提升
| 方法 | Wiki PPL ↓ | MMLU ↑ | CSR ↑ |
|------|------------|--------|-------|
| Quip# | 5.73 | 30.6 | 64.9 |
| **LLVQ (shape-gain)** | **5.48** | **37.3** | **66.8** |

> 🔹 **即使不微调，LLVQ也优于多数微调基线；微调后进一步逼近原始精度（仅降2.5–7.6%）**

---

#### （3）**跨模型泛化能力验证**
在 Llama-3、Ministral-3、Qwen-v3 等多个架构上均观察到一致领先：
- 平均PPL下降 **10–20%**
- MMLU 提升 **3–6个百分点**
- CSR 提升 **4–8个百分点**

---

### ✅ **消融实验结果**

#### （1）**是否使用Hadamard旋转的影响（Table 6）**
| 方法 | 旋转策略 | Wiki PPL ↓ | MMLU ↑ | CSR ↑ |
|------|----------|-----------|--------|-------|
| LLVQ (no rotation) | None | 7.27 | 29.8 | 61.5 |
| LLVQ | Input Only | 6.90 | 36.0 | 63.6 |
| LLVQ | Input+Output | **6.83** | **34.9** | **64.6** |

> 🔹 **旋转有助于性能提升，但LLVQ本身对旋转依赖较小 → 减少预处理开销**

#### （2）**单壳 vs 多壳联合构造球面码（Appendix E）**
- 实验发现：**使用多个壳的并集（cumulative union）比单一壳具有更好的角均匀性**
- 在相同bitrate下，**union shells的平均angular distance更小**
- 推荐做法：采用球截断（ball cut）而非单壳投影

#### （3）**spherical shaping vs shape-gain（Appendix F）**
| 配置 | MSE @ 2 bits/dim | Retention |
|------|------------------|---------|
| Spherical shaping | 0.084 | 89.4% |
| Shape-gain (1 bit gain) | **0.078** | **92.1%** |

> 🔹 **shape-gain更优，最佳分配约为1 bit用于gain（≈0.041 bits/dim），略低于理论建议的1/24≈0.083**

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **高维格结构显著提升量化效率**
   - Leech lattice凭借其24维最优几何性质，在相同比特率下提供最低失真。
   - 从信息论角度看，**维度越高，越接近香农限**。

2. **LLVQ实现了“无码本”下的高性能VQ**
   - 利用Golay码结构实现**隐式搜索与索引**，避免内存爆炸。
   - 支持**完全并行化解码**，适用于GPU部署。

3. **shape-gain优于spherical shaping**
   - 分离方向与幅度编码可获得更高灵活性和更低MSE。
   - 推荐使用**multi-shell union + 1-bit gain量化**作为默认配置。

4. **减少对Hadamard旋转的依赖**
   - LLVQ本身具备强表达力，可在**无旋转或部分旋转**下保持高性能。
   - 有利于简化部署流程，避免在线变换带来的延迟。

---

### ⚠️ **局限性**
1. **固定维度24**：要求权重维度能被24整除，否则需要padding。
2. **实现复杂度较高**：依赖Golay码、模运算、组合枚举，开发门槛高于标量量化。
3. **目前仅验证PTQ场景**：尚未探索训练时量化（QAT）潜力。

---

### 🔮 **未来工作方向**
1. **推广至其他高维最优格**（如Kissing Number更高的非格结构）
2. **结合QAT进行端到端优化**
3. **适配Transformer注意力模块的特殊结构**（如KV Cache量化）
4. **探索动态bit分配机制**：根据不同layer自适应选择shell深度或gain bits

---

## ✅ **总结**
> **LLVQ是首个将Leech lattice成功应用于LLM压缩的工作，它不仅在理论上逼近香农限，在实践中也全面超越Quip#、QTIP、PVQ等前沿方法。该研究证明了“数学结构驱动”的量化路径的巨大潜力，为下一代超低比特AI模型提供了坚实基础。**

</details>

---

### 15. [Data Augmentation and Convolutional Network Architecture Influence on Distributed Learning](https://arxiv.org/abs/2603.10902)

**Authors**: Victor Forattini Jansen, Emanuel Teixeira Martins, Yasmin Souza Lima, Flavio de Oliveira Silva, Rodrigo Moreira, Larissa Ferreira Rodrigues Moreira  
**Category**: cs.DC  
**Published**: 2026-03-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.10902v1  

#### Abstract
Convolutional Neural Networks (CNNs) have proven to be highly effective in solving a broad spectrum of computer vision tasks, such as classification, identification, and segmentation. These methods can be deployed in both centralized and distributed environments, depending on the computational deman...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Data Augmentation and Convolutional Network Architecture Influence on Distributed Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前大多数关于 **Convolutional Neural Networks (CNNs)** 的研究集中在模型的可解释性、分类准确率等任务性能上，而对这些模型在**分布式训练环境中的硬件资源消耗影响**（如 GPU、CPU、内存、网络流量）缺乏系统性分析。尤其是在引入 **Data Augmentation (DA)** 和不同 **CNN 架构深度**（浅层 vs. 深层）时，其对计算资源的影响尚未被充分量化。

本文填补了这一空白，重点探究：
- DA 和 CNN 架构如何共同影响分布式学习中的资源使用效率；
- 这些因素之间的交互作用是否显著；
- 在真实场景中部署 CNN 模型时可能带来的基础设施负担。

---

### **提出了什么新方法或新思路**
提出了一种基于 **2² Factorial Design（因子设计）** 的实验框架，用于系统评估两个关键因素（DA 与 CNN 架构）及其交互效应对多个响应变量的影响：

- 因素 A：**Data Augmentation**（有 DA / 无 DA）
- 因素 B：**CNN Architecture**（浅层 CNN / 深层 CNN）

通过 **ANOVA 分析** 定量分离主效应和交互效应，揭示各因素对以下指标的影响程度（以百分比形式表示）：
- `YGPU`, `YNetworkPackets`, `YCPU`, `YMemory`, `YAccuracy`

该方法实现了从“仅关注精度”到“兼顾性能与资源开销”的多维评估范式转变。

---

### **相比现有方法的优势**
| 维度 | 传统方法局限 | 本论文优势 |
|------|--------------|-----------|
| 评估目标 | 聚焦 accuracy/F1-score | 多维度评估：accuracy + 硬件资源消耗 |
| 实验设计 | 单因素对比或经验性尝试 | 科学的 **factorial design + ANOVA**，支持因果推断 |
| 应用价值 | 面向算法优化 | 面向生产部署优化，提供资源规划依据 |

> ✅ 特别是首次量化了 **DA 对 network packets 的高达 77.92% 影响力**，为边缘设备或带宽受限场景下的部署提供了重要警示。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Paddy Doctor Dataset**  
  - 图像数量：16,225 张  
  - 类别数：13 类（12 种水稻病害 + 正常叶片）  
  - 来源：真实稻田拍摄，由农业专家标注  
  - 应用场景：水稻叶部疾病自动识别基准数据集  

> 选择理由：图像质量高、类别丰富、已被广泛用于 CNN 建模验证。

---

### **实验设置**
#### **硬件配置**
| 服务器 | CPU | RAM | GPU | 网络接口 |
|--------|-----|-----|-----|---------|
| Server #1 | Intel i5-4430 @ 3.00GHz | 32GB | RTX 4060 Ti (8GB) | 1Gbps（协商至 100Mbps） |
| Server #2 | Intel i5-4430 @ 3.00GHz | 16GB | GTX 1050 Ti (4GB) | 1Gbps（协商至 100Mbps） |

- 操作系统：Ubuntu 20.04 LTS
- 分布式后端：**Torch Distributed Data Parallel (DDP)**
- 网络监控工具：**NetData**（采集 CPU、GPU、内存、网络包速率）

#### **模型选择**
| 类型 | 模型名称 | 描述 |
|------|--------|------|
| 浅层 CNN (shallow-CNN) | **MobileNetV2-100**（含 Batch Normalization） | 参数少，适合轻量级部署 |
| 深层 CNN (deep-CNN) | **MobileOne-S1** | 更深结构，更强表达能力 |

#### **Data Augmentation Pipeline**
包含以下增强操作：
- 随机旋转（±5°）
- 仿射变换（剪切 0.2，平移 ±5%）
- 随机裁剪（原图大小的 80–100%）
- 水平翻转
- 颜色抖动（color jittering）

> 目标：提升泛化能力，同时观察其对通信负载的影响。

#### **评估指标（Response Variables）**
| 指标 | 单位 | 含义 |
|------|------|------|
| `YGPU` | % | GPU 利用率平均值 |
| `YNetworkPackets` | Pkts/s | 每秒传输的数据包数量 |
| `YCPU` | % | CPU 利用率平均值 |
| `YMemory` | % | 内存占用率平均值 |
| `YAccuracy` | % | 测试集准确率 |

#### **实验组合（共 4 组）**
1. with-DA + shallow-CNN  
2. without-DA + shallow-CNN  
3. with-DA + deep-CNN  
4. without-DA + deep-CNN  

每组运行并记录上述指标均值。

---

### **基线方法对比**
本文未直接对比其他算法模型（如 ResNet、EfficientNet），而是将比较建立在：
- 是否使用 DA
- 使用浅层 vs. 深层 CNN 架构

因此，“基线”实为控制变量下的不同配置组合，而非传统意义上的 SOTA 模型对比。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 3）**

| 实验条件 | `YGPU(%)` | `YNetworkPackets(Pkts/s)` | `YCPU(%)` | `YMemory(%)` | `YAccuracy(%)` |
|----------|------------|----------------------------|-------------|----------------|------------------|
| with-DA + shallow-CNN | 95.12 | 19,994.50 | 51.15 | 81.70 | **98.71** |
| without-DA + shallow-CNN | 97.18 | 15,698.97 | 47.38 | 81.75 | **99.60** |
| with-DA + deep-CNN | 97.21 | 19,973.00 | 47.03 | 80.45 | 94.09 |
| without-DA + deep-CNN | 98.29 | 10,526.36 | 43.85 | 81.45 | 96.58 |

> ⚠️ 注意：accuracy 最高出现在 **without-DA + shallow-CNN**（99.60%），说明 DA 不一定提升 accuracy，在某些情况下反而略降。

---

### **与基线方法的对比结果**
虽然没有与其他模型架构横向对比，但从资源配置角度看：

| 观察项 | 结果 |
|-------|------|
| **DA 对 accuracy 的影响** | 小幅下降（shallow-CNN 下从 99.60 → 98.71；deep-CNN 下从 96.58 → 94.09） |
| **DA 对 network packets 的影响** | 显著增加（+27.37% ~ +89.73%） |
| **深层 CNN 对 accuracy 的影响** | 整体低于浅层 CNN（最大差达 5.5%） |
| **深层 CNN 对 GPU 占用** | 更高（~97–98%），接近饱和 |

> ❗ 表明：更复杂的模型不一定带来更高的 accuracy，却显著增加了资源压力。

---

### **消融实验结果（Factorial Analysis – Table 4）**

| 因素 | `YGPU` 影响 (%) | `YNetworkPackets` 影响 (%) | `YCPU` 影响 (%) | `YMemory` 影响 (%) | `YAccuracy` 影响 (%) |
|------|------------------|------------------------------|------------------|--------------------|------------------------|
| TA (DA) | 46.83 | **77.92** | 45.07 | 24.53 | 15.86 |
| TB (CNN 架构) | 48.64 | 11.13 | 54.61 | 54.75 | **80.60** |
| TAB (交互作用) | 4.53 | 10.94 | 0.32 | 20.72 | 3.54 |

> 🔍 关键发现：
- **DA 是 network traffic 的主导因素（77.92%）**
- **CNN 架构是 accuracy 的主要决定因素（80.60%）**
- 存在轻微交互作用（如 DA 在 deep-CNN 中引发更大通信开销）

此外，文中指出：
> “In distributed training with DA, increasing the data volume led to a **27.37% rise** in network packet transmission with shallow CNN, and an **89.73% increase** with deep CNN.”

→ 深层模型 + DA 导致梯度同步频繁，加剧通信瓶颈。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Data Augmentation 显著增加网络通信负载**  
   - 平均导致 network packets 上升 77.92%
   - 尤其在 deep-CNN 中放大效应（+89.73%）
   - 原因：更多样化的输入导致每轮迭代产生更多变化的梯度，需更频繁同步

2. ✅ **CNN 架构深度对 accuracy 影响最大（80.60%）**  
   - 但实验中 **浅层 CNN（MobileNetV2）表现优于深层 CNN（MobileOne-S1）**
   - 可能原因：过拟合、优化难度大、参数不匹配等

3. ✅ **GPU 利用率极高（>95%），且受双因素共同影响**
   - 几乎所有配置下 GPU 都处于高负载状态
   - 表明训练过程严重依赖 GPU 加速

4. ✅ **memory consumption 相对稳定**  
   - 各实验条件下波动小（约 80–82%）
   - 不是主要瓶颈

5. ✅ **accuracy 与资源消耗之间存在权衡（trade-off）**
   - 最高 accuracy 出现在 **without-DA + shallow-CNN**
   - 引入 DA 虽增强泛化能力设想，但在本实验中未转化为精度增益

---

### **方法的局限性**
1. 🛑 **仅测试两种 CNN 模型**  
   - 缺乏更广泛的模型覆盖（如 ResNet、EfficientNet）
   - 结论外推需谨慎

2. 🛑 **DA 策略固定**  
   - 使用预设 pipeline，未探索不同 DA 强度或组合的影响

3. 🛑 **early stopping 策略限制长期趋势分析**  
   - 训练提前终止于 epoch 20，未能观察完整收敛行为

4. 🛑 **异构 GPU 设置引入偏差**  
   - Server #1 使用 RTX 4060 Ti，Server #2 使用 GTX 1050 Ti
   - 性能差异可能导致同步等待，进一步加重通信负担

5. 🛑 **单一数据集验证**  
   - 所有结论基于 Paddy Doctor 数据集，通用性有待验证

---

### **未来工作方向**
1. 🔍 探索更多 **CNN parameters**（宽度、通道数、注意力机制）对资源的影响
2. 📊 扩展至其他 **datasets**（如植物、医学图像）进行跨域验证
3. 🔄 研究 **不同的 DA 方法**（Mixup、CutOut、AutoAugment）对通信成本的影响
4. ☁️ 构建 **AI-as-a-Service** 架构，实现低功耗设备上的高效推理（参考作者前期工作）
5. 💡 开发 **resource-aware training scheduler**，动态调整 DA 强度或 batch size 以平衡 accuracy 与 bandwidth 使用

---

## **总结**
> 本文突破了传统 CV 研究只关注 accuracy 的局限，首次系统地量化了 **Data Augmentation** 和 **CNN 架构** 在 **distributed learning** 场景下对硬件资源的综合影响。

📌 **核心洞见**：
> “**更高的数据多样性（DA）并不总带来更好的性能，反而可能成为分布式系统的通信瓶颈。**”

🎯 **实践意义**：
- 在边缘计算、联邦学习、IoT 场景中应谨慎启用 DA
- 模型选型不仅要考虑 accuracy，还需结合硬件约束进行权衡
- 提出的 **factorial design + ANOVA** 方法可推广至其他 DNN 部署场景的资源评估

> 一句话总结：  
> **这不是一篇追求更高 accuracy 的论文，而是一篇教你“如何负责任地部署 CNN”的工程指南。**

</details>

---

### 16. [OpenClaw-RL: Train Any Agent Simply by Talking](https://arxiv.org/abs/2603.10165)

**Authors**: Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, Ling Yang  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.10165v1  

#### Abstract
Every agent interaction generates a next-state signal, namely the user reply, tool output, terminal or GUI state change that follows each action, yet no existing agentic RL system recovers it as a live, online learning source. We present OpenClaw-RL, a framework built on a simple observation: next-s...

---

### 17. [Safe and Scalable Web Agent Learning via Recreated Websites](https://arxiv.org/abs/2603.10505)

**Authors**: Hyungjoo Chae, Jungsoo Park, Alan Ritter  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.10505v1  

#### Abstract
Training autonomous web agents is fundamentally limited by the environments they learn from: real-world websites are unsafe to explore, hard to reset, and rarely provide verifiable feedback. We propose VeriEnv, a framework that treats language models as environment creators, automatically cloning re...

---

### 18. [Beyond the Illusion of Consensus: From Surface Heuristics to Knowledge-Grounded Evaluation in LLM-as-a-Judge](https://arxiv.org/abs/2603.11027)

**Authors**: Mingyang Song, Mao Zheng, Chenning Xu  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.11027v1  

#### Abstract
The paradigm of LLM-as-a-judge relies on a critical assumption, namely that high inter-evaluator agreement indicates reliable and objective evaluation. We present two complementary findings that challenge this assumption. \textbf{First}, we demonstrate that this consensus is frequently illusory. We ...

---

### 19. [Estimating condition number with Graph Neural Networks](https://arxiv.org/abs/2603.10277)

**Authors**: Erin Carson, Xinye Chen  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.10277v1  

#### Abstract
In this paper, we propose a fast method for estimating the condition number of sparse matrices using graph neural networks (GNNs). To enable efficient training and inference of GNNs, our proposed feature engineering for GNNs achieves $\mathrm{O}(\mathrm{nnz} + n)$, where $\mathrm{nnz}$ is the number...

---

### 20. [Robust Post-Training for Generative Recommenders: Why Exponential Reward-Weighted SFT Outperforms RLHF](https://arxiv.org/abs/2603.10279)

**Authors**: Keertana Chidambaram, Sanath Kumar Krishnamurthy, Qiuling Xu, Ko-Jen Hsiao, Moumita Bhattacharya  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.10279v1  

#### Abstract
Aligning generative recommender systems to user preferences via post-training is critical for closing the gap between next-item prediction and actual recommendation quality. Existing post-training methods are ill-suited for production-scale systems: RLHF methods reward hack due to noisy user feedbac...

---

### 21. [Graph-GRPO: Training Graph Flow Models with Reinforcement Learning](https://arxiv.org/abs/2603.10395)

**Authors**: Baoheng Zhu, Deyu Bo, Delvin Ce Zhang, Xiao Wang  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.10395v1  

#### Abstract
Graph generation is a fundamental task with broad applications, such as drug discovery. Recently, discrete flow matching-based graph generation, \aka, graph flow model (GFM), has emerged due to its superior performance and flexible sampling. However, effectively aligning GFMs with complex human pref...

---

### 22. [Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning](https://arxiv.org/abs/2603.10588)

**Authors**: Zhaowei Zhang, Xiaohan Liu, Xuekai Zhu, Junchao Huang, Ceyao Zhang, Zhiyuan Feng, Yaodong Yang, Xiaoyuan Yi, Xing Xie  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10588v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has achieved remarkable success in logical reasoning tasks, yet whether large language model (LLM) alignment requires fundamentally different approaches remains unclear. Given the apparent tolerance for multiple valid responses in moral reasoning...

---

### 23. [Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization](https://arxiv.org/abs/2603.10808)

**Authors**: Linghao Zhang  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10808v1  

#### Abstract
The emergence of large language model (LLM)-based agent frameworks has shifted the primary challenge in building domain-expert AI agents from raw capability to effective encoding of domain expertise. Two dominant paradigms -- code-first development, which embeds expertise in deterministic pipelines,...

---

### 24. [A Hybrid Knowledge-Grounded Framework for Safety and Traceability in Prescription Verification](https://arxiv.org/abs/2603.10891)

**Authors**: Yichi Zhu, Kan Ling, Xu Liu, Hengrun Zhang, Huiqun Yu, Guisheng Fan  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10891v1  

#### Abstract
Medication errors pose a significant threat to patient safety, making pharmacist verification (PV) a critical, yet heavily burdened, final safeguard. The direct application of Large Language Models (LLMs) to this zero-tolerance domain is untenable due to their inherent factual unreliability, lack of...

---

### 25. [PEEM: Prompt Engineering Evaluation Metrics for Interpretable Joint Evaluation of Prompts and Responses](https://arxiv.org/abs/2603.10477)

**Authors**: Minki Hong, Eunsoo Lee, Sohyun Park, Jihie Kim  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10477v1  

#### Abstract
Prompt design is a primary control interface for large language models (LLMs), yet standard evaluations largely reduce performance to answer correctness, obscuring why a prompt succeeds or fails and providing little actionable guidance. We propose PEEM (Prompt Engineering Evaluation Metrics), a unif...

---

### 26. [A neural operator for predicting vibration frequency response curves from limited data](https://arxiv.org/abs/2603.10149)

**Authors**: D. Bluedorn, A. Badawy, B. E. Saunders, D. Roettgen, A. Abdelkefi  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10149v1  

#### Abstract
In the design of engineered components, rigorous vibration testing is essential for performance validation and identification of resonant frequencies and amplitudes encountered during operation. Performing this evaluation numerically via machine learning has great potential to accelerate design iter...

---

### 27. [Prioritizing Gradient Sign Over Modulus: An Importance-Aware Framework for Wireless Federated Learning](https://arxiv.org/abs/2603.10763)

**Authors**: Yiyang Yue, Jiacheng Yao, Wei Xu, Zhaohui Yang, George K. Karagiannidis, Dusit Niyato  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.10763v1  

#### Abstract
Wireless federated learning (FL) facilitates collaborative training of artificial intelligence (AI) models to support ubiquitous intelligent applications at the wireless edge. However, the inherent constraints of limited wireless resources inevitably lead to unreliable communication, which poses a s...

---

### 28. [Cross-Species Transfer Learning for Electrophysiology-to-Transcriptomics Mapping in Cortical GABAergic Interneurons](https://arxiv.org/abs/2603.11000)

**Authors**: Theo Schwider, Ramin Ramezani  
**Category**: cs.LG  
**Published**: 2026-03-12  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.11000v1  

#### Abstract
Single-cell electrophysiological recordings provide a powerful window into neuronal functional diversity and offer an interpretable route for linking intrinsic physiology to transcriptomic identity. Here, we replicate and extend the electrophysiology-to-transcriptomics framework introduced by Gouwen...

---

### 29. [Verbalizing LLM's Higher-order Uncertainty via Imprecise Probabilities](https://arxiv.org/abs/2603.10396)

**Authors**: Anita Yang, Krikamol Muandet, Michele Caprio, Siu Lun Chau, Masaki Adachi  
**Category**: cs.AI  
**Published**: 2026-03-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.10396v1  

#### Abstract
Despite the growing demand for eliciting uncertainty from large language models (LLMs), empirical evidence suggests that LLM behavior is not always adequately captured by the elicitation techniques developed under the classical probabilistic uncertainty framework. This mismatch leads to systematic f...

---

### 30. [GhazalBench: Usage-Grounded Evaluation of LLMs on Persian Ghazals](https://arxiv.org/abs/2603.09979)

**Authors**: Ghazal Kalhor, Yadollah Yaghoobzadeh  
**Category**: cs.CL  
**Published**: 2026-03-12  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.09979v1  

#### Abstract
Persian poetry plays an active role in Iranian cultural practice, where verses by canonical poets such as Hafez are frequently quoted, paraphrased, or completed from partial cues. Supporting such interactions requires language models to engage not only with poetic meaning but also with culturally en...

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
