# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-26 06:54:02 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control](https://arxiv.org/abs/2603.24366)

**Authors**: Yifeng Zhang, Harsh Goel, Peizhuo Li, Mehul Damani, Sandeep Chinchali, Guillaume Sartoretti  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.24366v1  

#### Abstract
Adaptive traffic signal control (ATSC) is crucial in alleviating congestion, maximizing throughput and promoting sustainable mobility in ever-expanding cities. Multi-Agent Reinforcement Learning (MARL) has recently shown significant potential in addressing complex traffic dynamics, but the intricaci...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**自适应交通信号控制（ATSC）**中的两大核心挑战：
- **部分可观测性（Partial Observability）**：在去中心化环境中，单个路口（agent）只能获取局部交通状态，难以全面感知全局交通动态。
- **多智能体协调困难（Agent Coordination）**：缺乏有效的机制来建模相邻路口之间的动态依赖关系，导致决策短视、协作效率低。

这些问题限制了现有 MARL 方法在大规模城市路网中的可扩展性和性能稳定性。

---

### **提出的新方法与新思路**

#### **(1) Queue Dynamic State Encoding (QDSE)**
- **创新点**：提出一种基于车辆排队动力学模型的新型状态表示方法。
- **核心思想**：将传统仅关注“当前队列长度”的静态特征，扩展为包含**动态流入/流出预测能力**的复合特征向量。
- **具体特征包括**：
  - 当前停止车辆数 `Q(t)`
  - 进入车辆数 `Nin(t)` 和离开车辆数 `Nout(t)`
  - 移动车辆数 `Nr(t)` 及其与队尾的距离 `Dfr(t)`
  - 跟随首车的移动车辆数 `Nfr(t)`
- **优势**：使 RL agent 能够**预测未来拥堵趋势**，实现从“被动响应”到“主动调控”的转变。

#### **(2) Neighbor-aware Policy Optimization (NAPO)**
- **创新点**：设计了一种完全去中心化的 MARL 算法，通过注意力机制显式建模邻居间的**状态-动作依赖关系**。
- **关键技术组件**：
  - **注意力机制（Attention Mechanism）**：动态计算不同邻居对当前 agent 决策的影响权重。
  - **增强型 Actor-Critic 架构**：
    - **Actor Network**：基于时空聚合单元（Spatio-Temporal Network, STN），融合空间邻域信息与时间历史依赖。
    - **Critic Network**：引入**特权本地批评器（Privileged Local Critic）**，结合邻居的状态-动作序列进行价值估计。
  - **改进的优势函数（Advantage Function）**：将邻居影响加权整合进策略梯度更新中，提升信用分配准确性。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | CoordLight |
|------|--------|-----------|
| **状态表示** | 静态特征（如 vehicle count, pressure） | 动态预测性特征（QDSE） |
| **协调机制** | 固定通信或共享奖励 | 自适应注意力识别关键邻居 |
| **训练稳定性** | 易受环境非平稳性影响 | 通过 NAPO 改进优势估计，提升收敛性 |
| **可扩展性** | 多依赖集中式训练（CTDE） | 完全去中心化执行，适合大规模部署 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
采用三个真实世界的城市路网数据集，均来自开源仿真平台 **CityFlow**：
| 数据集 | 规模 | 描述 |
|-------|-----|------|
| **Jinan (China)** | 3×4 = 12 个路口 | 中小规模，三种不同流量需求（DJN1–3） |
| **Hangzhou (China)** | 4×4 = 16 个路口 | 中等规模，两种流量需求（DHZ1–2） |
| **New York (USA)** | 7×28 = 196 个路口 | 大规模复杂网络，最具挑战性 |

> 所有路口均为标准四向交叉口，每方向三条车道。

---

### **实验设置**
- **仿真时长**：每个 episode 3600 秒
- **决策周期**：每 5 秒做出一次相位选择
- **黄灯处理**：相位切换前插入 2 秒黄灯，新相位持续 3 秒
- **训练方式**：同质策略（homogeneous policy），所有路口共享参数
- **硬件配置**：Ubuntu + RTX 3060 GPU

---

### **评估指标**
主指标为 **Average Travel Time (平均行程时间)**（越小越好），定义如下：
$$
\text{Travel Time} = \frac{1}{N_v} \sum_{i=1}^{N_v} (\min(t_{\text{end}}, 3600) - t_{\text{start}})
$$
其中 $N_v$ 是总车辆数，若未完成行程则以 3600 秒计。

---

### **基线方法对比**
分为两类：

#### **传统方法**
- **FixedTime**：固定周期配时
- **MaxPressure (MP)**：基于上下游压力差的贪心控制
- **Advanced-MP**：考虑有效范围内移动车辆的增强版 MP

#### **MARL 方法**
| 方法 | 类型 | 特点 |
|------|-----|------|
| **CoLight / Advanced-CoLight** | 图注意力网络（GAT） |
| **MPLight / Advanced-MPLight** | 基于压力的 FRAP 架构 |
| **DenseLight*** | 非局部特征提取 + 密集反馈机制 |
| **SocialLight** | 分布式合作学习，衡量个体贡献 |

> *注：部分 DenseLight 结果引用原论文*

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table II）**

| 方法 | Jinan DJN(1) | Hangzhou DHz(2) | NYC DNY(2) |
|------|--------------|------------------|------------|
| FixedTime | 346.36 s | 359.44 s | 1660.29 s |
| MaxPressure | 273.96 s | 348.98 s | 1535.77 s |
| Advanced-MP | 253.61 s | 318.67 s | ~1060* s |
| SocialLight | 217.92 s | 288.55 s | 1106.69 s |
| **CoordLight (Ours)** | **199.24 s** | **250.87 s** | **1039.15 s** |

> ✅ 在所有场景下均取得最优表现

---

### **与基线方法的对比结果**
- **在 Jinan 上**：
  - 相比最佳基线 SocialLight，分别提升 **6.39% (DJN1)**、**8.57% (DJN2)**、**9.23% (DJN3)**
- **在 Hangzhou 上**：
  - 在高负载 DHz(2) 下，优于 DenseLight **7.87%**
- **在 New York（196 路口）上**：
  - 显著超越图注意力类方法（CoLight 等）
  - 比 SocialLight 减少约 **6.1%** 平均旅行时间
- **统计显著性检验（Unpaired t-test）**：
  - 所有 p-values < 1.1e-8，经 Bonferroni 校正后仍远低于阈值（1.57e-4），表明性能提升具有高度统计显著性

---

### **消融实验结果**

#### **(1) QDSE 状态表示的有效性（E节）**
- 在 DHz(2) 上比较不同 state encoding：
  - **QDSE 表现优于 VC、GP、EP、ATS、DTSE**
  - 尤其在 **travel time** 和 **queue std** 上表现最好
  - 即使比图像级表示 DTSE 更简洁，性能相当甚至略优 → **实现了精度与复杂度的良好平衡**

#### **(2) 组件消融研究（G节）**
五种变体对比（使用 Hangzhou DHz2）：
| 变体 | Avg Travel Time |
|------|-----------------|
| CoordLight (完整) | 250.87 s |
| w/o QDSE | ↑ 明显恶化 |
| w/o STN（无时空网络） | 快速收敛至次优解 |
| w/o AD（无状态-动作解码器） | 训练不稳定，性能下降 |
| Base（仅 FC + IPPO） | 最差 |

> 🔍 发现：**QDSE + STN + AD 三者协同作用显著**，缺一不可

#### **(3) 传感器噪声鲁棒性测试（F节）**
- 对 QDSE 中最关键的 `Dfr`（首车距离）添加高斯噪声（σ=10m, 20m, 30m）
- 结果显示：
  - 即使在 σ=30m 噪声下，平均旅行时间增加不超过 **2.34%**
  - 性能退化平缓 → **具备较强现实部署潜力**

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **QDSE 能有效捕捉交通流动态变化**，使 agent 具备预测未来拥堵的能力，从而做出更前瞻性的决策。
2. ✅ **NAPO 通过注意力机制实现“关键邻居识别”**，避免无效交互，提高协调效率。
3. ✅ **去中心化架构 + 局部观察 + 邻居感知机制** 可在不牺牲可扩展性的前提下，实现接近全局优化的效果。
4. ✅ CoordLight 在从小规模（12 路口）到超大规模（196 路口）的多种网络中均表现出色，验证了其**强泛化能力和可扩展性**。

---

### **方法的局限性**
- **假设理想检测设备**：实验基于完美摄像头输入，未考虑遮挡、误检等问题（尽管做了噪声测试）。
- **同质化网络假设**：目前仅适用于规则网格状路网，尚未验证在异构路口（T型、环岛等）上的表现。
- **相位切换约束**：当前 action space 限定为固定时长相位切换，未支持动态调整绿灯时长。
- **缺乏优先通行支持**：未考虑紧急车辆、公交优先等现实调度需求。

---

### **未来工作方向**
1. **拓展至异构网络**：适配不同类型路口布局与车道配置。
2. **引入相位时长优化**：允许 agent 学习动态调整绿灯持续时间。
3. **处理非同步控制**：支持各路口以不同频率决策的异步执行模式。
4. **增强现实鲁棒性**：
   - 融合多源传感数据（雷达、GPS 浮动车）
   - 引入容错机制应对传感器失效
5. **支持特殊事件管理**：
   - 应急车辆优先通行
   - 交通事故响应
   - 施工绕行引导

---

> 📌 **总体评价**：  
> CoordLight 提出了一套**兼具理论深度与工程实用性的去中心化 ATSC 框架**，通过 QDSE 与 NAPO 的协同设计，在保持去中心化优势的同时，显著提升了多路口协调能力。其实验充分、分析严谨，是当前 MARL for ATSC 领域的一项重要进展。代码已开源（GitHub: [marmotlab/CoordLight](https://github.com/marmotlab/CoordLight)），便于后续研究复现与扩展。

</details>

---

### 2. [LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load](https://arxiv.org/abs/2603.23640)

**Authors**: Pranay Tummalapalli, Sahil Arayakandy, Ritam Pal, Kautuk Kundan  
**Category**: cs.DC  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.23640v1  

#### Abstract
Deploying large language models on-device for always-on personal agents demands sustained inference from hardware tightly constrained in power, thermal envelope, and memory. We benchmark Qwen 2.5 1.5B (4-bit quantised) across four platforms: a Raspberry Pi 5 with Hailo-10H NPU, a Samsung Galaxy S24 ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load  
**——核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前关于 **Large Language Model (LLM)** 在边缘设备上的推理研究大多关注：
- 单次推理延迟（single-inference latency）
- 云端吞吐量（cloud-hosted throughput）
- 或仅作为附带观察提及热效应（thermal effects）

然而，在“始终在线”的个人智能代理（always-on personal agents）场景中，设备需要在**持续负载下长时间运行**，此时：
- **热管理（thermal management）**
- **功耗限制（power budget）**
- **内存带宽瓶颈（memory bandwidth）**

成为实际部署的关键瓶颈。这些因素导致**峰值硬件性能与真实世界可持续性能之间存在巨大差距**。

本文系统地填补了这一空白，首次对多种边缘平台在**持续负载下的LLM推理表现**进行了跨平台、长周期的实证分析。

---

### ✅ 提出的新方法与新思路

#### （1）提出“warm-condition sustained inference”基准测试框架
- 设计了一个**20轮连续推理**的实验协议，每轮间隔仅1秒，模拟真实交互式助手的高频查询场景。
- 强调从“冷启动”到“热稳定状态”的完整性能退化曲线，而非仅报告最佳性能。

#### （2）引入专用边缘NPU（Hailo-10H）进行独立评测
- 首次发布**非厂商主导的、针对Hailo-10H NPU模块的公开LLM推理基准**。
- 揭示了专用NPU在能效和稳定性方面的独特优势。

#### （3）统一模型与提示工程控制变量
- 所有平台使用相同的 **Qwen 2.5 1.5B (4-bit quantized)** 模型，并通过SHA-256校验权重一致性。
- 使用固定长度（258 tokens）且触发长输出的prompt，确保所有平台都经历**持久的decode阶段压力**，从而暴露内存与热瓶颈。

---

### ✅ 相比现有方法的优势

| 方面 | 本文优势 |
|------|--------|
| **评估维度更全面** | 不仅测throughput，还追踪power、temperature、energy/token、thermal state等，揭示性能背后的根本约束 |
| **更贴近现实用例** | 聚焦“持续负载”，而非瞬时峰值，反映真实agent使用模式 |
| **平台覆盖广** | 包含GPU（RTX 4050）、旗舰手机SoC（A18 Pro, Snapdragon 8 Gen 3）、专用NPU（Hailo-10H）三类主流边缘方案 |
| **揭示不同失败模式** | 发现iOS与Android在热节流机制上的本质差异 |

---

## 2. 核心实验方法和设置

### 🧪 使用的模型
- **Model**: `Qwen 2.5 1.5B`（4-bit quantized）
- **选择理由**：
  - 参数小于2B，适合端侧部署
  - 支持多平台原生推理栈
  - 内存占用 <1GB，满足所有设备限制
- **量化格式**：
  - RPi + Hailo-10H: GGUF Q4_0
  - S24 Ultra: TVM binary q4f16_2
  - iPhone 16 Pro: MLX safetensors Q4_0
  - RTX 4050: GPTQ Int4

> ⚠️ 注：尽管均为4-bit，但具体压缩方式略有差异，作者承认这是潜在混淆变量。

---

### 🖥️ 实验平台（4个）

| 平台 | 类型 | 加速器 |
|------|-----|-------|
| Raspberry Pi 5 + Hailo-10H | 边缘单板机 + 专用NPU | Hailo-10H (40 TOPS, <5W) |
| Samsung Galaxy S24 Ultra | 安卓旗舰手机 | Adreno 750 GPU + Hexagon NPU |
| iPhone 16 Pro | iOS旗舰手机 | A18 Pro GPU（未启用Neural Engine） |
| Laptop with RTX 4050 | 笔记本GPU | NVIDIA RTX 4050 (2560 CUDA cores) |

> 所有测试均在**电池供电**下进行，以反映真实移动场景。

---

### 📊 实验设置与评估指标

#### 实验流程（见 Figure 1）
1. 设备静置10分钟至室温平衡（22°C ± 2°C）
2. 加载模型进内存（warm-up run，结果丢弃）
3. 连续执行20轮推理，每轮间隔1秒
4. 每轮记录各项指标
5. 验证token数量一致性

#### 主要评估指标（Table 4）

| 指标 | 单位 | 定义 |
|------|------|------|
| Throughput (TPS) | tok/s | decode阶段生成token速率 |
| Decode time | ms | 解码总耗时 |
| Prefill time | ms | 首token延迟（Time to First Token） |
| Avg/Peak Power | W | 系统平均/峰值功耗 |
| Energy per token | mJ | 每个token消耗的能量 |
| GPU/CPU Temp | °C | 最高温度 |
| Thermal State | — | iOS特有：Normal/Warm/Hot |
| GPU Frequency | MHz | Android可读取实时频率 |

> ⚠️ 注意：由于平台API限制，部分指标不可得（如iOS无组件级功耗，Android Battery Manager不可靠）。

---

### 🔍 基线对比方式
本文不直接对比“算法”基线，而是将四类**硬件+软件栈组合**视为不同的“部署方案”进行横向比较：
- **RTX 4050 + vLLM (CUDA)** → 高性能边缘代表
- **iPhone 16 Pro + MLX (Metal)** → iOS生态代表
- **S24 Ultra + MLC-LLM (OpenCL)** → 安卓生态代表
- **RPi5 + Hailo-10H + hailo-ollama** → 专用NPU代表

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（Table 9）

| Platform | TPS (mean) | CV (%) | Avg Power (W) | Energy/token (mJ) |
|---------|------------|--------|----------------|--------------------|
| RTX 4050 | **131.70** | 2.2% | 34.12 | 297.3 |
| iPhone 16 Pro (Hot) | 22.56 | 5.1% | * | * |
| S24 Ultra | 9.93 | 8.0% | * | * |
| RPi + Hailo-10H | 6.914 | **0.04%** | 1.870 | **270.5** |

> *注：iPhone 和 S24 Ultra 缺少精确功耗测量*

---

### 🔍 详细平台表现

#### ✅ RTX 4050（高性能但高功耗）
- **Throughput**: 131.7 tok/s（最高）
- **Stability**: CV = 2.2%，非常稳定
- **Power**: 平均34.1 W（受电池限制约束于~35W）
- **Thermal**: 温度从55°C升至70°C，**无throttling**
- **Energy proportionality**: 0.259 W per tok/s

> 💡 表现接近理论极限，但依赖电源，不适合长期离电运行。

---

#### ⚠️ iPhone 16 Pro（初期快，后期严重降频）
- 初始两轮达 **40.35 tok/s**
- 第3轮进入“Warm”状态，性能下降37%
- 第8轮起进入“Hot”状态，**稳定在22.56 tok/s（-44% from peak）**
- **Throughput CV = 20.8%**（主要由热态切换引起）
- 电池20轮下降10%

> ❗ 显示iOS系统会主动降频保护，导致性能骤降，影响用户体验连续性。

---

#### ❌ Samsung Galaxy S24 Ultra（硬性中断）
- 前5轮平均 **9.93 tok/s**
- 第6轮：OS强制将GPU频率锁死在 **231 MHz**（原为629–680 MHz）
- GPU温度达 **78.3°C**，CPU达73.8°C，**推理终止**
- Benchmark无法完成全部20轮

> ⚠️ 展现出一种**破坏性更强的失败模式**：不是缓慢降速，而是直接不可用。

---

#### ✅ Raspberry Pi 5 + Hailo-10H（极致稳定低功耗）
- **Throughput**: 6.914 tok/s（最低）
- **但CV仅为0.04%！** 几乎零波动
- **功耗仅1.87 W**，峰值不到2W
- **Energy per token = 270.5 mJ**，全平台最低
- 温度稳定（NPU ~58.5°C），**全程无throttling**

> ✅ 尽管绝对速度慢，但在**能效比和稳定性上媲美RTX 4050**：
> - RTX 4050: 0.259 W/tok/s
> - Hailo-10H: 0.271 W/tok/s → **几乎相同能效比例**

---

### 🔬 消融实验与归因分析（隐含在讨论中）

虽然没有显式的ablation study，但作者通过对瓶颈的归因实现了类似分析：

| 平台 | 性能瓶颈归因 |
|------|-------------|
| RTX 4050 | 受限于**电池功率上限**（~35W），非散热 |
| iPhone 16 Pro | **Dynamic Voltage and Frequency Scaling (DVFS)** 导致持续降频 |
| S24 Ultra | **Android thermal governor** 强制GPU频率floor |
| Hailo-10H | **on-module memory bandwidth**（LPDDR4）是主要瓶颈，非NPU算力本身 |

> 说明：Hailo-10H虽标称40 TOPS，但autoregressive decode是memory-bound任务，无法发挥并行计算优势。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Thermal management 是移动端LLM推理的首要瓶颈**
   - 峰值算力 ≠ 可持续性能
   - iPhone在2轮内即开始降频，S24 Ultra在第6轮直接中断
   - “被动散热 + 小体积”决定了手机难以支撑长时间高负载推理

2. **专用边缘NPU具有独特价值**
   - Hailo-10H虽吞吐低（6.9 tok/s），但实现了：
     - 极致稳定的输出（CV=0.04%）
     - 超低功耗（<2W）
     - 优异的能效比例（≈ RTX 4050）
   - 特别适用于**异步、后台、低延迟容忍的任务**（如摘要生成、定时提醒）

3. **不同操作系统热策略差异显著**
   - iOS：渐进式DVFS降频（graceful degradation）
   - Android（此机型）：硬性频率锁定（hard floor），更具破坏性

4. **笔记本GPU性能强但依赖电源**
   - RTX 4050提供最强性能（131.7 tok/s）
   - 但在电池模式下受限于~35W，续航约2–3小时，**不适合作为“始终在线”设备**

5. **能量效率可在不同层级实现**
   - 高通/苹果宣称的TOPS数字不能反映真实decode性能
   - 实际能效取决于**软硬协同设计**，Hailo-10H证明低功耗平台也能达到与高端GPU相当的energy proportionality

---

### ⚠️ 方法的局限性

| 限制 | 说明 |
|------|------|
| 单一模型 & 单一prompt | 结果可能不泛化到其他模型或短回复对话场景 |
| 多种推理框架混杂 | 性能差异可能是软件栈导致，非纯硬件能力 |
| 功耗测量不一致 | RTX为GPU级，Hailo为整系统，不可直接对比 |
| 无多设备重复 | 未考虑device-to-device variability |
| Android功耗数据缺失 | Battery Manager API在GPU负载下不可靠 |
| 仅20轮迭代 | 更长期趋势仍需验证 |

---

### 🔮 未来工作方向

1. **延长测试至100+轮**，观察超长期热行为
2. **替换Android功耗采集方式**，采用硬件电流传感器
3. 探索**duty-cycling**（间歇运行）与**主动冷却**策略缓解热问题
4. 统一各平台的**quantization format**以减少偏差
5. 测试更多模型（如TinyLlama、Phi-3）和prompt类型
6. 探索**batched decoding**或**speculative decoding**提升NPU利用率

---

## ✅ 总结一句话

> 在持续负载下，**热管理取代峰值算力成为边缘LLM推理的核心瓶颈**；相比之下，专用NPU（如Hailo-10H）虽吞吐较低，却能在极低功耗下实现前所未有的稳定性与能效比例，为“始终在线”的个人代理提供了新的可行路径。

</details>

---

### 3. [Symbolic--KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning](https://arxiv.org/abs/2603.23854)

**Authors**: Salah A Faroughi, Farinaz Mostajeran, Amirhossein Arzani, Shirko Faroughi  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.23854v1  

#### Abstract
Symbolic discovery of governing equations is a long-standing goal in scientific machine learning, yet a fundamental trade-off persists between interpretability and scalable learning. Classical symbolic regression methods yield explicit analytic expressions but rely on combinatorial search, whereas n...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Symbolic-KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在科学机器学习（Scientific Machine Learning, SciML）中，存在一个长期存在的**可解释性与可扩展性之间的根本权衡**：

- **传统符号回归方法**（如遗传算法、SINDy）能生成显式的解析表达式，具有高度可解释性，但依赖组合搜索，计算成本高，难以扩展到高维问题。
- **深度神经网络**（如MLP、PINN）能高效处理大规模、高维数据，但其内部表示是“黑箱”，缺乏可解释性，无法直接提供人类可读的物理规律。

本文旨在弥合这一鸿沟，实现**既可扩展又可解释的模型**，以从数据中自动发现控制方程（governing equations）。

### **提出了什么新方法或新思路**

作者提出了一种新型神经架构——**Symbolic Kolmogorov-Arnold Networks (Symbolic-KAN)**，其核心思想是：

- 将**离散的符号结构**（discrete symbolic structure）嵌入到可训练的深度网络中。
- 基于**Kolmogorov-Arnold 表示定理**（KART），将多元函数分解为一元函数的叠加。
- 引入**分层门控机制**（hierarchical gating）和**符号正则化**（symbolic regularization），在训练过程中逐步将连续的混合选择“锐化”（sharpen）为离散的单选（one-hot selection）。
- 最终输出**紧凑的闭式表达式**（compact closed-form expressions），无需后处理的符号拟合。

### **相比现有方法的优势**

| 对比维度 | Symbolic-KAN | 传统KAN | SINDy / 符号回归 | PINN |
|---------|-------------|--------|------------------|------|
| **可解释性** | ⭐⭐⭐⭐⭐（直接输出符号表达式） | ⭐⭐⭐（结构较清晰，但表达式复杂） | ⭐⭐⭐⭐⭐（显式公式） | ⭐（黑箱） |
| **可扩展性** | ⭐⭐⭐⭐（基于梯度优化） | ⭐⭐⭐⭐ | ⭐⭐（组合搜索慢） | ⭐⭐⭐⭐⭐ |
| **发现能力** | ⭐⭐⭐⭐（可发现新原语组合） | ⭐⭐⭐ | ⭐⭐（依赖预设库） | ❌ |
| **端到端训练** | ✅ | ✅ | ❌ | ✅ |

**核心优势**：
- **无需后处理**：直接从训练中获得符号表达式，避免了传统方法中“先拟合再符号回归”的两阶段流程。
- **可扩展的原语发现**：可作为“原语发现机制”，识别出对任务最相关的解析组件，用于构建SINDy等稀疏方法的候选库。
- **物理一致性**：在PDE求解中，能同时满足物理约束并生成反映真实解析结构的符号表示。

---

## 2. 核心实验方法和设置

### **使用的数据集与问题类型**

实验覆盖三大类科学学习任务：

1. **数据驱动回归**（Data-driven regression）
   - 目标函数：`F(x) = x²`, `F(x) = sin(3x)/(1+x²) + 0.4cos(5x)`
   - 输入域：`x ∈ [0,5]`，采样 `N_tr = 250~650` 个点

2. **动力系统识别**（Inverse dynamical systems）
   - **Van der Pol 振荡器**：
     ```
     dx/dt = ay
     dy/dt = μ(1 - x^2.15)y - cx
     ```
   - 参数未知，需从时间序列数据中识别 `a`, `μ`, `c`
   - 时间范围：`t ∈ [0,20]` 和 `[0,50]`，`Δt = 0.01`

3. **偏微分方程求解**（Physics-informed PDE learning）
   - **反应扩散方程**（Reaction-Diffusion）：
     ```
     D∇²u + κ tanh(u) = f,  u(x) = sin⁶(6x)
     ```
     - 逆问题：估计参数 `κ`
     - 域：`[-2,2]` 和 `[-4,4]`
   - **拉普拉斯方程**（Laplace Equation）：
     ```
     ∇²u = 0,  u(x,y) = sin(πx) sinh(πy)
     ```
     - 正问题：求解 `u(x,y)` 在 `[0,1]×[0,1]`

### **实验设置和评估指标**

| 设置项 | 描述 |
|------|------|
| **网络结构** | 多层Symbolic-KAN，每层 `K_e` 个单元，每个单元 `E` 条边 |
| **原语库**（Primitive Library） | `{0, 1, x, x², x³, sin x, cos x, tanh x, exp x, log(1+|x|), ...}` |
| **训练阶段** | 两阶段：<br>1. **软训练**：使用Gumbel-Softmax松弛门控<br>2. **硬化**（Hardening）：转为one-hot选择，用L-BFGS微调 |
| **评估指标** | - **相对误差** `E(F) = \|F_pred - F_true\| / \|F_true\|`<br>- **参数识别误差**（如 `κ`, `μ`）<br>- **符号结构一致性**（是否选出正确原语） |

### **基线方法对比**

- **cPIKAN**：基于Chebyshev多项式的Physics-Informed KAN
- **PINN**：标准Physics-Informed Neural Network（tanh激活）
- **SINDy**：稀疏识别非线性动力学（文中未直接对比，但作为下游应用提及）

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **1. 数据驱动回归**
- **`F(x) = x²`**：相对误差 `1.04×10⁻⁵`，成功识别出 `x` 和 `x²` 原语，其他冗余项被剪枝。
- **`F(x) = sin(3x)/(1+x²) + 0.4cos(5x)`**：相对误差 `7.75×10⁻³`，选出 `sin`, `cos`, `lorentz (1/(1+x))` 等关键原语。

#### **2. Van der Pol 方程**
| 时间范围 | 参数 `a` | `μ` | `c` | 轨迹相对误差 `E(u)` |
|--------|--------|-----|-----|------------------|
| `T=20` | 1.0000 | 0.0099 | 0.9998 | `6.02×10⁻⁴` |
| `T=50` | 0.9999 | 0.0093 | 0.9992 | `5.87×10⁻³` |

- 所有参数识别误差 < 1%，表明对非整数幂和非线性阻尼具有强鲁棒性。
- 选出的原语主要为 `sin`, `cos`, `x`，与系统振荡行为一致。

#### **3. 反应扩散方程**
| 方法 | 域 | 验证误差 `E(u)` | 识别参数 `κ` | 相对误差 |
|-----|----|---------------|------------|----------|
| **Symbolic-KAN** | `[-2,2]` | `5.93×10⁻⁴` | 0.6994 | <0.1% |
| **cPIKAN** | `[-2,2]` | `8.25×10⁻⁴` | 0.6998 | <0.1% |
| **Symbolic-KAN** | `[-4,4]` | `9.37×10⁻³` | 0.6985 | ~0.2% |
| **cPIKAN** | `[-4,4]` | `2.07×10⁻¹` | 0.6611 | ~5.6% |

- 在大域上，Symbolic-KAN误差比cPIKAN低 **95%**，参数精度高 **5倍以上**。
- 比标准PINN误差低 **99%以上**。

#### **4. 拉普拉斯方程**
| 方法 | 验证误差 `E(u)` | 最大绝对误差 |
|-----|---------------|-------------|
| **Symbolic-KAN** | `1.11×10⁻³` | 极小 |
| **cPIKAN** | `8.76×10⁻³` | 大 |
| **PINN** | `2.71×10⁻³` | 中等 |

- Symbolic-KAN误差比cPIKAN低 **87%**，比PINN低 **59%**。
- 选出的原语为 `sin`, `cos`, `sinh`, `cosh`，与真解 `sin(πx) sinh(πy)` 完全一致。

### **消融实验（隐含分析）**

虽然未明确列出消融表，但以下设计体现了关键组件的作用：

- **门控机制**：若无门控，所有原语混合，无法得到简洁符号表达。
- **温度退火**（T → 0）：控制从软选择到硬选择的过渡，确保稳定收敛。
- **熵正则化**（Entropy Regularization）：促使 `α_kep` 收敛到one-hot，提升稀疏性。
- **非极大抑制**（NMS）：防止同一单元内多条边选择相同原语，增强多样性。

---

## 4. 关键结论和发现

### **主要发现**

1. **Symbolic-KAN 能可靠恢复正确的原语项和控制结构**：
   - 在回归、动力系统、PDE等多种任务中，均能识别出与真实解一致的关键数学原语（如 `sin`, `x²`, `tanh`, `sinh`）。

2. **生成的符号表示具有真实解析结构**：
   - 不仅拟合数据，还能“理解”背后的物理机制，例如在拉普拉斯方程中自动选择双曲函数。

3. **兼具高精度与强泛化能力**：
   - 在插值和外推区域均表现良好（如 `F(x)=x²` 外推至 `x>5` 仍保持二次趋势）。
   - 在长时域（`T=50`）和大空间域（`[-4,4]`）下仍保持稳定。

4. **可作为“前库选择器”**（Pre-library selector）：
   - 发现的原语可用于构建SINDy等方法的候选库，解决其“库必须预先包含正确项”的局限。

### **方法的局限性**

1. **依赖原语库的设计**：
   - 若真实解涉及库中不存在的函数（如特殊函数 `erf(x)`），可能无法准确发现。

2. **未完全解决外推问题**：
   - 虽然外推表现优于传统NN，但神经网络固有的外推挑战仍未彻底解决。

3. **训练过程较复杂**：
   - 两阶段训练（软训练 + 硬化）增加了实现难度，且门控变量需保守更新以保证稳定性。

### **未来工作方向**

1. **动态扩展原语库**：
   - 结合符号回归或程序合成，自动发现新的数学表达式加入库中。

2. **应用于更复杂的PDE和算子学习**：
   - 如Navier-Stokes方程、非线性薛定谔方程等。

3. **与贝叶斯方法结合**：
   - 为符号选择引入不确定性量化，提升可靠性。

4. **硬件部署与实时推理**：
   - 利用其紧凑符号形式，实现在边缘设备上的轻量级科学模型部署。

---

> **总结**：  
> Symbolic-KAN 是迈向**可扩展、可解释、机理化学习**的重要一步。它不仅是一个高性能的神经求解器，更是一个**自动化的科学发现工具**，能够在无需人工干预的情况下，从数据中提炼出简洁、可读、符合物理规律的数学表达式。

</details>

---

### 4. [Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation](https://arxiv.org/abs/2603.23838)

**Authors**: Han Zheng, Yining Ma, Brandon Araki, Jingkai Chen, Cathy Wu  
**Category**: cs.AI  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23838v1  

#### Abstract
Lifelong Multi-Agent Path Finding (MAPF) is critical for modern warehouse automation, which requires multiple robots to continuously navigate conflict-free paths to optimize the overall system throughput. However, the complexity of warehouse environments and the long-term dynamics of lifelong MAPF o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**Lifelong Multi-Agent Path Finding (Lifelong MAPF)** 在现代仓库自动化中的应用挑战展开研究。传统的一次性 MAPF 方法（如 CBS、PP）难以应对现实场景中机器人持续执行任务、动态生成目标、交通模式随时间演变等长期动态特性。这些问题导致：
- 静态优先级策略失效；
- 短视决策引发级联拥堵或死锁；
- 高密度环境下搜索空间爆炸。

现有学习方法在 lifelong 设置下表现不稳定，未能一致超越基于搜索的方法。

---

### 提出的新方法：RL-RH-PP
作者提出 **Reinforcement Learning-guided Rolling Horizon Prioritized Planning (RL-RH-PP)**，是首个将强化学习（RL）与经典搜索式规划器结合用于 lifelong MAPF 的混合框架。

#### 核心思想
- 将 **Prioritized Planning (PP)** 作为高效、可扩展的规划主干；
- 引入 **Rolling Horizon Prioritized Planning (RH-PP)** 框架以支持持续重规划；
- 利用 **深度强化学习** 动态生成高质量的全局优先级顺序（total priority order），替代随机或启发式排序。

#### 技术创新
1. **POMDP 建模**：将动态优先级分配建模为一个 Partially Observable Markov Decision Process (POMDP)，使 RL 能捕捉长期因果依赖。
2. **Transformer-style 神经网络架构**：
   - 编码器采用交替的 **temporal attention** 和 **spatial attention** 层，分别捕获单个 agent 的路径时序依赖和多 agent 间的空间交互。
   - 解码器自回归地解码出总优先级顺序。
3. **Top-K Sampling + 冲突修复机制**：从 RL 策略中采样多个候选优先级顺序，通过 RH-PP 执行并选择最优者，提升鲁棒性。

---

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|--------|
| **效率与可扩展性** | PP 的线性复杂度优于 CBS/PBS 的指数增长，适合大规模仓库；RL 仅指导“顺序”而非完整路径，计算开销可控。 |
| **长期协调能力** | RL 显式优化长期奖励（避免拥堵、提高吞吐量），克服了 PIBT 等局部贪婪方法的短视缺陷。 |
| **泛化能力** | 模型具备强大的 zero-shot 泛化能力，能适应不同 agent 密度、规划窗口长度和未见过的地图布局。 |
| **性能上限突破** | 实验证明 RL-RH-PP 可逆转 RH-PP 在高密度下的性能下降趋势，表明学习组件可部分弥补基础求解器的固有局限。 |

---

## 2. 核心实验方法和设置

### 数据集与仿真环境
构建了两个真实世界启发的 warehouse 地图：
1. **Amazon fulfillment center dense map**  
   - 障碍物密度：15.3%
   - 多条平行通道，较宽走廊
2. **Symbotic warehouse map**（首次引入 MAPF 研究）
   - 障碍物密度高达 **56.6%**
   - 存在瓶颈区域（如窄交叉道）、进出库分区，更易发生拥堵

任务由系统动态分配，agent 完成当前目标后立即获得新任务，模拟真实 fulfillment 中心运作。

---

### 实验设置
- **仿真时长**：T = 800 时间步
- **规划周期**：每 h=5 步重新规划一次
- **规划视野**：w ∈ {5, 10, ..., 30}
- **agent 数量**：N ∈ {80, 100, 120}（训练）；测试时进行 zero-shot 迁移至其他数量
- **硬件资源限制**：每个 planning step 最大 CPU 时间预算为 1 秒

---

### 评估指标
| 指标 | 含义 |
|------|------|
| **Throughput per agent (TPA)** | 单个 agent 在整个仿真期间完成的任务数（平均值） |
| **Total throughput** | 所有 agent 完成的总任务数 = TPA × N |
| **Solve time / Runtime per planning step** | 每次规划所消耗的 CPU/GPU 时间（RL-RH-PP 包含 GPU 推理时间） |

---

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **RH-CBS** | 搜索-based | Rolling Horizon 下的 Conflict-Based Search，保证最优但扩展性差 |
| **RH-PBS** | 搜索-based | Priority-Based Search，使用部分优先级树，平衡质量与效率 |
| **PIBT** | 分布式贪心 | Priority Inheritance with Backtracking，实时性强但易陷入局部次优 |
| **WPPL** | 混合方法 | 2023 League of Robot Runner 冠军方案，结合 PIBT 与 windowed MAPF-LNS 进行精炼 |
| **RH-PP (Random)** | 非学习基线 | 使用随机优先级顺序的滚动地平线 PP，作为 RL-RH-PP 的直接对照 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 表格汇总（来自 Table 3，N=120, w=20）

| Method | Amazon TPA ↑ | Symbotic TPA ↑ |
|--------|----------------|------------------|
| RH-CBS | 2.84 ± 0.29 | 1.50 ± 0.45 |
| RH-PBS | 3.37 ± 0.26 | 1.76 ± 1.10 |
| PIBT | 16.09 ± 0.48 | 2.67 ± 0.49 |
| WPPL | 23.59 ± 0.26 | 10.05 ± 1.33 |
| **RL-RH-PP (Ours)** | **25.56 ± 0.55** | **11.31 ± 2.21** |

> ✅ **结论**：RL-RH-PP 在两种地图上均达到最高吞吐量，尤其在高障碍密度的 Symbotic 地图上显著领先。

---

### 与基线方法的对比结果
- 在 Amazon 地图上，相比 WPPL 提升约 **8.4%** 吞吐量；
- 在 Symbotic 地图上，相比 WPPL 提升约 **12.5%**；
- 相比纯随机顺序的 RH-PP(K=5)，平均提升达 **25%** 总吞吐量；
- 在推理时间方面，RL-RH-PP 与 WPPL 相当，远快于 RH-CBS/RH-PBS。

> 🔍 **特别发现**：随着 agent 密度增加，RH-PBS 性能急剧下降，而 RL-RH-PP 表现稳健，显示出更强的抗拥堵能力。

---

### 消融实验结果

#### （1）奖励函数权重影响（Fig. 15）
- 设计奖励包含三项：距离项 $d_{i,t}$、拥堵惩罚 $c_{i,t}$、不可行惩罚 $s_{i,t}$
- 实验发现：
  - 当 $\kappa$（拥堵权重）和 $\sigma$（不可行权重）设为 **1000** 时效果最佳；
  - 权重过小收敛慢，过大无额外收益；
  - 不可行惩罚对训练稳定性更为敏感。

#### （2）编码器设计消融（Fig. 16）
| 变体 | 结果 |
|------|------|
| Full Model (ours) | 最高最终吞吐量，学习稳定 |
| w/o Temporal Attention | 在 Symbotic 上性能大幅下降 → 证明需建模长程时序依赖 |
| w/o Spatial Attention | 性能略有下降 → 空间交互仍重要 |
| 替换为 Yan & Wu (2024) 的 CNN 架构 | 完全无法收敛 → 3D Conv 局部感受野限制其处理全局冲突的能力 |

#### （3）与其他启发式的比较（Table 7）
- 对比 **Distance-Query Heuristic**（按最短路径长度排序）
- 结果显示 RL-RH-PP 在所有配置下全面超越该启发式，说明固定规则难以应对复杂动态场景。

#### （4）上下文 Bandit 消融（Fig. 17）
- 若将 RL 替换为单步决策（contextual bandit），虽初期收敛更快，但最终性能更低。
- 证明 **multi-step RL 规划能力** 是实现长期去拥堵的关键。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **学习可以有效增强传统启发式方法**：RL 并非取代 PP，而是通过智能生成优先级来“引导”它，在保持高效的同时大幅提升性能。
2. ✅ **RL-RH-PP 能主动管理拥堵**：
   - 优先级热力图显示，RL 策略会自动给拥堵区内的 agent 更高优先级；
   - 能战略性地让边界 agent “回退”，为内部 agent 开路，缓解死锁。
3. ✅ **强零样本泛化能力**：
   - 可泛化到不同 agent 数量（N）、规划窗口（w）和未见地图变体（如 aisle 长度变化、in/out 位置交换）；
   - 归功于 learnable position embedding + attention 架构的空间感知能力。
4. ✅ **实际部署潜力大**：
   - 推理延迟可控（~1s 内）；
   - 支持 anytime behavior（通过 Top-K sampling 提升质量）；
   - 开源代码已发布：[GitHub - MikeZheng777/RL-RH-PP](https://github.com/MikeZheng777/RL-RH-PP)

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **绝对位置嵌入限制跨尺寸迁移** | 当前 position embedding 基于固定地图大小索引，无法 zero-shot 迁移到更大/更小的地图。 |
| **Top-K evaluation 序列执行耗时** | 当前 Top-K evaluation 在 CPU 上串行运行，成为 K 增大时的瓶颈。 |
| **未联合优化 task assignment** | 当前仅优化路径规划，task 分配仍由外部系统决定，未来可联合建模。 |

---

### 未来工作方向
1. **并行化 Top-K evaluation**：利用多线程或 GPU 加速多个 priority order 的评估过程，提升 anytime 性能。
2. **实现 fully map-agnostic encoder**：设计不依赖绝对坐标的表示（如 relative coordinates 或 graph-based pooling），支持任意规模地图的 zero-shot transfer。
3. **扩展至 joint task and path planning**：让 decoder 输出 `(agent, task)` 对序列，统一解决任务分配与路径协调问题。
4. **增强鲁棒性**：纳入对延迟、传感器噪声、通信丢包等不确定性的建模，提升现实部署可靠性。
5. **探索通用 learning-guided optimization 框架**：将本范式推广至其他 long-horizon combinatorial optimization 问题。

</details>

---

### 5. [Lightweight Fairness for LLM-Based Recommendations via Kernelized Projection and Gated Adapters](https://arxiv.org/abs/2603.23780)

**Authors**: Nan Cui, Wendy Hui Wang, Yue Ning  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23780v1  

#### Abstract
Large Language Models (LLMs) have introduced new capabilities to recommender systems, enabling dynamic, context-aware, and conversational recommendations. However, LLM-based recommender systems inherit and may amplify social biases embedded in their pre-training data, especially when demographic cue...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Lightweight Fairness for LLM-Based Recommendations via Kernelized Projection and Gated Adapters**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **LLM-based Recommender Systems (RecLLMs)** 在推荐任务中表现出色，但由于其预训练数据中嵌入的社会偏见（如性别、年龄、职业等），容易在生成推荐时放大对特定人群的不公平性。
- 现有公平性方法存在以下问题：
  - 需要额外可训练参数（如 UP5 使用对抗训练）；
  - 优化不稳定，对超参数敏感；
  - 多属性去偏时易破坏任务有用特征，导致准确率显著下降。

### **提出的新方法**
本文提出一种**轻量级、无需额外训练周期**的去偏框架，结合两种核心技术：
1. **Kernelized Iterative Null-space Projection (RFF-INLP)**  
   - 将原始 INLP 扩展至非线性空间：通过 **Random Fourier Features (RFF)** 映射将表示提升到高维核空间，在该空间执行闭式解的正交投影，去除敏感属性信号。
   - 引入 **isotropic Gaussian perturbation** 提升投影鲁棒性。
   - 整个过程为**闭式计算（closed-form）**，不涉及梯度更新，无额外可训练参数。

2. **Two-level Gated Mixture-of-Experts (MoE) Adapter**
   - 第一级门控（outer gate）：基于上下文动态加权多个属性对应的投影矩阵，实现**自适应强度的去偏**。
   - 第二级门控（inner gate）：引入低秩 LoRA 专家模块，选择性恢复因投影丢失的任务相关特征，形成“**erase-then-repair**”机制。
   - MoE 结构仅引入 $ O(k) $ 新参数（$ k $ 为敏感属性数），保持轻量化。

### **相比现有方法的优势**
| 维度 | 本方法 | 现有方法（如 UP5） |
|------|--------|------------------|
| 可训练参数 | 仅 MoE adapter 部分需训练，主投影固定 | 对抗网络引入大量可训练参数 |
| 优化稳定性 | 投影部分无梯度，训练稳定 | 对抗训练易发散，依赖调参 |
| 多属性处理能力 | 支持联合去偏且保留任务性能 | 多属性下性能下降明显 |
| 推理延迟 | 几乎无增加（投影为矩阵乘法） | 因复杂模块带来额外开销 |
| 轻量化程度 | 极轻量，适合部署 | 参数多，资源消耗大 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **MovieLens-1M**  
   - 用户电影评分记录，带时间戳。
   - 敏感属性：Gender（2类）、Age（7类）、Occupation（21类）。
   - 分割方式：按时间排序，最后两个交互用于验证和测试。

2. **Insurance Dataset**（非洲保险公司客户数据）
   - 客户保险产品交互数据。
   - 敏感属性：Marital Status（8类）、Age（5类）、Occupation（6类）。
   - 数据无可靠时间戳，因此仅用于 **Direct Recommendation** 实验。

### **实验设置**
- **LLM Backbone**: Instruct Llama-3.22 1B（冻结主干），基于 **LLaRA** 框架构建 RecLLM。
- **Adapter 类型**:
  - Task-LoRA：插入注意力块中的 Q/K/V/O 投影层。
  - Attribute-LoRA：作为 MoE 专家，作用于输出层。
- **投影配置**:
  - RFF 维度 $ D = 4096 $
  - 噪声尺度 $ \eta = 0.05 $
  - 投影矩阵每轮检测 CLG > 阈值 $ T $ 时更新一次。

### **评估指标**
#### **Utility（推荐质量）**
- **Hit@{1, 3, 10}**：目标物品是否出现在 Top-k 推荐列表中（百分比 ↑）

#### **Fairness（公平性）**
- **Counterfactual Leakage Gap (CLG, $ \Delta_{cL} $)**：
  $$
  \Delta_{cL} = \frac{1}{m} \sum_{i=1}^{m} |AUC_i - 0.5|
  $$
  - $ AUC_i $：用 MLP 探针预测第 $ i $ 个敏感属性类别的 AUC；
  - 值越接近 0 表示越公平（无法从表示中推断出敏感信息）。

### **基线方法对比**
1. **LLaRA**：基础模型，无任何去偏机制。
2. **P5**：Prompt-based 方法，统一文本到文本范式。
3. **UP5**：当前最先进的公平性增强版本 P5，采用对抗学习进行去偏。

所有方法在相同候选采样协议和评估标准下比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### **Table 2: Sequential Recommendation on MovieLens**
| Attribute | Hit@1 (Ours) | Hit@1 (UP5) | $ \Delta_{cL} $ (Ours) | $ \Delta_{cL} $ (UP5) |
|----------|--------------|-------------|----------------------------|-------------------------|
| Gender   | **39.69**    | 26.82       | **0.90**                   | 4.19                    |
| Age      | **38.16**    | 31.23       | **0.60**                   | 2.91                    |
| Occupation | **38.58**  | 31.66       | **1.70**                   | 0.00                    |
| G+A+O (All) | **56.08** | —           | **0.17**                   | —                       |

> ✅ **观察**：在单属性和多属性场景下，本方法均显著优于 UP5，Hit@1 提升 8–13 个百分点，同时将平均 $ \Delta_{cL} $ 降至 **0.17%**，接近理想公平。

#### **Table 3: Direct Recommendation Results**
| Dataset     | Attribute | Hit@1 (Ours) | Hit@1 (UP5) | $ \Delta_{cL} $ (Ours) | $ \Delta_{cL} $ (UP5) |
|------------|-----------|--------------|-------------|----------------------------|-------------------------|
| MovieLens  | G         | **36.92**    | 16.38       | **0.60**                   | 4.19                    |
|            | A         | **37.14**    | 21.22       | **0.00**                   | 2.91                    |
|            | O         | **34.60**    | 21.00       | **0.60**                   | 0.00                    |
|            | G+A+O     | **43.20**    | 20.18       | **0.78**                   | 3.21                    |
| Insurance  | M+A+O     | **57.37**    | 81.63       | **0.13**                   | 0.74                    |

> ✅ **观察**：
> - 在 MovieLens 上，Hit@1 平均翻倍以上，公平性大幅提升（$ \Delta_{cL} $ 下降约 75%）。
> - 在 Insurance 数据上，尽管 UP5 在 Hit@1 上略高，但其 $ \Delta_{cL} $ 是本方法的 **5.7 倍**，说明牺牲了公平换取精度；而本方法实现了更优的 **accuracy-fairness trade-off**。

### **消融实验（隐含分析）**
虽然未明确列出消融表，但从设计逻辑可推知：
- 若仅使用线性 INLP（无 RFF），则无法捕捉非线性偏见信号，导致 $ \Delta_{cL} $ 下降有限；
- 若去掉 MoE adapter，则投影后信息损失严重，Hit@k 显著降低；
- 两级门控协同工作：“软投影 + 条件修复” 是维持高性能的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Kernelized INLP 能有效去除非线性偏见信号**  
   - 利用 RFF 将表示映射到核空间，使原本隐藏在非线性关系中的偏见变得线性可分，从而能被 INLP 成功消除。
   - 投影为闭式解，无需反向传播，真正实现“zero optimization cost”。

2. **Gated MoE Adapter 实现精准的信息修复**  
   - 第一级门控实现**细粒度控制去偏强度**，避免过度清洗；
   - 第二级门控通过低秩 LoRA 选择性注入有益信号，防止性能退化。

3. **“Erase-then-Repair” Pipeline 高效且实用**  
   - 先清除敏感信息，再局部修复任务性能，分离关注点，提升可控性和效果。

4. **在多个真实世界数据集上达到 SOTA 表现**  
   - 同时提升推荐准确率与公平性，打破传统“公平-效用权衡”困境。

### **方法的局限性**
- **依赖用户交互历史**：序列推荐表现优异，但在缺乏行为序列的冷启动场景中可能受限。
- **仅针对已知敏感属性**：需要标注数据训练探针来识别偏见方向，难以应对未知或隐式偏见。
- **投影操作仍有一定信息损失风险**：虽由 MoE 补偿，但在极端情况下可能导致语义漂移。

### **未来工作方向**
1. 探索不依赖用户历史的去偏方法（如利用先验知识或合成数据）；
2. 扩展至更多模态（如图文混合推荐）；
3. 研究自动发现潜在敏感维度的技术（unsupervised bias detection）；
4. 将该框架推广至其他 LLM 下游任务（如对话系统、搜索排序）中的公平性保障。

--- 

> 📌 **一句话总结**：  
> 本文提出了一种**无需额外训练、高效稳定的轻量级去偏方法**，通过 **kernelized INLP + gated MoE adapter** 的组合，在几乎不增加计算成本的前提下，显著提升了 RecLLMs 的公平性并保持甚至增强了推荐性能，为工业级部署提供了可行路径。

</details>

---

### 6. [On Gossip Algorithms for Machine Learning with Pairwise Objectives](https://arxiv.org/abs/2603.24128)

**Authors**: Igor Colin (LTCI, S2A, IP Paris), Aur\'elien Bellet (PREMEDICAL), Stephan Cl\'emen\c{c}on (LTCI, IDS, S2A, IP Paris), Joseph Salmon (IROKO, UM)  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.24128v1  

#### Abstract
In the IoT era, information is more and more frequently picked up by connected smart sensors with increasing, though limited, storage, communication and computation abilities. Whether due to privacy constraints or to the structure of the distributed system, the development of statistical learning me...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：On Gossip Algorithms for Machine Learning with Pairwise Objectives

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于**分布式机器学习中的成对目标函数（pairwise objectives）优化问题**，这类问题在许多统计学习任务中至关重要，例如：
- **相似性学习（metric learning）**
- **排序（ranking）**
- **聚类（clustering）**
- **图重建（graph reconstruction）**

这些问题的目标函数通常表现为**U-statistic of degree two**，即所有样本对上的平均值，形式为：

$$
U_n(h) = \frac{2}{n(n-1)}\sum_{i<j} h(x_i, x_j)
$$

然而，在传统的**去中心化（decentralized）系统**中，每个节点仅持有单个本地数据 $x_i$，无法直接计算涉及其他节点数据的成对交互项 $h(x_i, x_j)$。因此，标准的 gossip averaging 方法不再适用。

### 提出了什么新方法或新思路
作者提出并深入分析了一种名为 **GoSTA（Gossip-based Stochastic Averaging）** 的算法框架，用于解决以下两类问题：

#### （1）去中心化估计（Decentralized Estimation）
- 针对 U-statistic 的分布式估计问题，提出了 **GoSTA-SYNC** 算法。
- 算法结合了两个阶段：
  - **辅助观测传播（auxiliary observation propagation）**：通过 gossip 协议交换“辅助变量” $y_k(t)$，使各节点逐步获取全局数据视图。
  - **局部估计与平均（local estimation and averaging）**：利用当前本地数据 $(x_k, y_k(t))$ 构造成对估计，并通过 gossip 平均机制收敛至全局 U-statistic。

#### （2）去中心化优化（Decentralized Optimization）
- 针对形如 $F(\theta) = \frac{2}{n(n-1)}\sum_{i,j} f(\theta; x_i, x_j)$ 的目标函数最小化问题。
- 提出基于 **Dual Averaging** 的 gossip 优化算法（Algorithm 2），其核心是：
  - 在每次迭代中，随机边上的两个节点交换辅助观测 $y_i, y_j$。
  - 所有节点使用其当前的 $(x_k, y_k)$ 构造一个**有偏梯度估计** $\nabla f(\theta_k; x_k, y_k)$。
  - 将该梯度累加到 dual variable 中，并更新参数。

### 相比现有方法的优势
| 方面 | 本文贡献 |
|------|--------|
| **理论完整性** | 首次提供了 GoSTA 算法的**完整非渐近分析**，不仅给出期望偏差界（bias），还给出了方差界和总误差界（Theorem 1），填补了 Colin et al. (2015) 的空白。 |
| **收敛保证** | 对优化算法，首次严格证明了**梯度偏差（gradient bias）会随时间指数衰减**（Theorem 2），从而确保算法最终收敛，解决了 Colin et al. (2016) 中遗留的“偏差是否消失”的疑问。 |
| **下界分析** | 推导了针对成对目标的**去中心化优化下界**（Theorem 3），揭示了网络拓扑中“平均两跳距离” $\Delta$ 是影响收敛速度的关键因素，而非传统的直径（diameter）。 |
| **通用性** | 分析覆盖了同步（synchronous）和异步（asynchronous）两种设置，并讨论了多数据点每节点、差分隐私等扩展场景。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Breast Cancer Wisconsin (Original)** 数据集
  - 样本数 $n = 699$
  - 特征维度 $d = 11$
  - 每个节点存储一个样本（single point per node）

### 实验设置和评估指标
- **任务**：最大化 **AUC（Area Under the ROC Curve）**，这是一个典型的成对排序目标。
- **模型**：线性打分函数 $x \mapsto x^\top\theta$。
- **损失函数**：使用 logistic pairwise loss 作为 AUC 的凸代理：
  $$
  R_n(\theta) = \frac{1}{n^2}\sum_{i,j} \mathbf{1}_{\{l_i > l_j\}} \log(1 + \exp((x_j - x_i)^\top\theta))
  $$
- **网络拓扑**：比较三种不同连通性的图结构：
  1. **完全图（Complete graph）**：理想情况，信息混合最快。
  2. **2D 网格（2D grid）**：低连通性，大直径，通信瓶颈明显。
  3. **Watts-Strogatz 图**：小世界网络，介于前两者之间。
- **步长策略**：$\gamma(t) = 10^{-3}/\sqrt{t}$
- **评估指标**：
  - 目标函数值（loss）随迭代次数的变化。
  - 各节点局部估计的**一致性损失（consensus loss）**。
  - **偏差项（bias term）** $\|\epsilon(t)\|$ 的演化。

### 基线方法对比
- 本文方法（Algorithm 2）在不同拓扑下的表现自比较。
- 无显式正则化，纯去中心化 gossip 优化。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **图 1a（Loss evolution）**：
  - **完全图** 和 **Watts-Strogatz 图** 收敛速度显著快于 **网格图**。
  - 局部估计的分散程度（标准差）也随着网络连通性增强而降低。
- **图 1b（Bias term）**：
  - 在所有网络拓扑中，**偏差项 $\|\epsilon(t)\|$ 迅速趋近于零**。
  - 偏差在整个优化过程中始终比目标函数值小几个数量级，验证了理论分析中“偏差快速衰减”的结论。

### 消融实验结果
- 本文未进行传统意义上的消融实验（如移除某个模块），但通过**不同网络拓扑的对比**实现了类似效果：
  - **高连通性（高 spectral gap）** → 快速收敛 + 低偏差。
  - **低连通性（低 spectral gap）** → 收敛慢 + 初始偏差更大。
- 这验证了理论结论：**网络的谱间隙（spectral gap）直接影响算法效率**。

---

## 4. 关键结论和发现

### 主要发现
1. **GoSTA 算法有效且可靠**：能够在去中心化网络中准确估计和优化成对目标函数。
2. **偏差是暂时的**：由于辅助观测传播不充分导致的初始梯度偏差，会随着 gossip 信息混合而**指数级衰减**，不会影响最终收敛。
3. **网络拓扑至关重要**：**谱间隙（spectral gap）** 越大（即网络越连通），收敛速度越快。这与经典 gossip averaging 结论一致，但在成对场景下得到了新的理论支撑。
4. **下界揭示本质瓶颈**：成对优化的收敛下界依赖于“平均两跳距离” $\Delta$，表明信息需要经过中间节点传递，这是成对问题固有的通信复杂性。

### 方法的局限性
- **通信开销**：虽然避免了中心化，但 gossip 协议仍需大量通信轮次，尤其在稀疏图上。
- **内存限制**：每个节点只维护一个辅助观测，可能不足以捕捉复杂的数据分布，尤其是在高维或非均匀数据下。
- **理论假设**：分析基于固定图结构和理想通信，未考虑动态网络或链路故障。

### 未来工作方向
1. **多点每节点（Multiple points per node）**：推广到每个节点持有多个样本的场景（Section 6.1），可建模为“虚拟节点”图，理论可迁移。
2. **差分隐私（Differential Privacy）**：将本地差分隐私（Local DP）集成到 gossip 交换中，保护原始数据隐私（Section 6.2）。
3. **鲁棒性和公平性**：将鲁棒学习和公平性约束（本身常为 U-statistic 形式）纳入该框架，实现负责任的去中心化学习。
4. **更高效的采样策略**：探索非均匀 gossip 或重要性采样，加速成对估计的收敛。

--- 

> **总结**：本文为去中心化环境下的成对学习任务建立了坚实的理论基础，提出了实用的 gossip 算法，并通过严谨的理论分析和实验验证了其有效性，为 IoT、边缘计算等场景下的分布式机器学习提供了重要工具。

</details>

---

### 7. [Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping](https://arxiv.org/abs/2603.23998)

**Authors**: Yao Chen, Yilong Chen, Yinqi Yang, Junyuan Shang, Zhenyu Zhang, Zefeng Zhang, Shuaiyi Nie, Shuohuan Wang, Yu Sun, Hua Wu, HaiFeng Wang, Tingwen Liu  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.23998v1  

#### Abstract
Existing approaches to increasing the effective depth of Transformers predominantly rely on parameter reuse, extending computation through recursive execution. Under this paradigm, the network structure remains static along the training timeline, and additional computational depth is uniformly assig...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前的 **Transformer** 模型通过堆叠更多层来增加有效深度，从而提升推理能力，但这会显著增加参数量、内存占用和训练开销。为缓解此问题，已有研究引入 **recurrence**（递归）机制，在不增加参数的情况下扩展计算深度。

然而，现有方法（如 Block Loop）采用**静态、块级重用**策略，即在训练开始时就固定对整个 Transformer block 进行重复执行。这种粗粒度的深度分配导致：
- **计算冗余**：所有参数被同等对待，缺乏细粒度区分；
- **训练效率低**：额外计算开销高达 16–20%，且未考虑模型在训练过程中的动态演化特性。

---

### **提出了什么新方法或新思路**

本文提出 **Sparse Growing Transformer (SGT)**，一种**训练时稀疏深度分配框架**，其核心思想是：

> **“深度应随训练进程逐步生长，而非预先静态设定。”**

具体实现基于两个关键观察：

#### ✅ 观察 1：高熵注意力头（high-entropy attention heads）是语义整合的关键枢纽  
- 高熵头倾向于关注跨上下文分布的重要 token，承担全局推理功能；
- 低熵头则聚焦局部语法结构，作用更偏向局部建模。

#### ✅ 观察 2：层间存在“深到浅”的成熟轨迹（deep-to-shallow maturation）  
- 深层网络在训练早期快速分化并稳定；
- 浅层网络演化较慢，后期才逐渐专业化。

基于以上洞察，SGT 引入 **Training-Time Structural Sparsity** 范式，实现：
- **渐进式增长（Progressive Growth）**：从深层向浅层逐步激活递归；
- **熵引导循环（Entropy-Guided Attention Looping）**：仅对每层中熵最高的少数 attention head 进行递归。

---

### **相比现有方法的优势**

| 维度 | SGT | Block Loop |
|------|-----|-----------|
| **深度分配粒度** | 头级别（fine-grained） | 块级别（coarse-grained） |
| **结构动态性** | 动态生长（progressive） | 静态预设（static） |
| **计算效率** | 极低 FLOPs 开销（+1–3%） | 高开销（+16–20%） |
| **性能增益** | 更优推理表现 | 改进有限甚至退化 |

> ✅ **优势总结**：SGT 在极小额外成本下实现了更高的训练效率和更强的推理能力，尤其在长上下文外推任务上表现突出。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **预训练数据**：`C4`（20B tokens 子集）
- **评估任务**（涵盖多类推理与知识任务）：
  - **Reasoning**：ARC-Easy, OpenBookQA, SIQA, WinoGrande, HellaSwag
  - **Knowledge**：MMLU（STEM, Social Sciences, Humanities, Others）
  - **Math & Code**：Basic Arithmetic, MATH-500, GSM8K, HumanEval
  - **Long-context**：NarrativeQA, TriviaQA, RULER_NIAH_MK/MV, InfBenchQA

共 11 项任务，覆盖不同输入长度（78–3760 tokens），见 Table 5。

---

### **实验设置和评估指标**

- **模型架构**：基于 LLaMA-style，从零预训练（scratch）
- **模型规模**：275M / 573M / 1.2B 参数
- **序列长度**：4096
- **Batch Size**：1024
- **优化器**：AdamW，学习率 6e-4
- **硬件**：8×NVIDIA A100（40GB）
- **评估指标**：
  - 主要：**Perplexity (PPL)** 和 **Reasoning & Knowledge 平均准确率**
  - 分析：FLOPs 开销、收敛速度、消融分析、注意力可视化

---

### **基线方法对比**

| 方法 | 描述 |
|------|------|
| **Vanilla** | 标准 Transformer，无递归 |
| **Block Loop** | 固定选择若干层进行整块递归（以高熵层为起点） |
| **SGT (Ours)** | 渐进式、熵引导的头级稀疏递归，$K_{\text{max}} \in \{1,2,3\}$ |

> 所有方法在相同训练配置下公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 1）**

| Model | FLOPs ($\times 10^{18}$) | Avg Score (%) |
|-------|--------------------------|---------------|
| Vanilla (573M) | 87.41 | 35.39 |
| Block Loop | 101.76 (**+16.42%**) | 35.82 |
| **SGT (K=3)** | **88.96 (+1.77%)** | **36.41** ✅ |

> 🔺 SGT 在仅增加 **1.77% FLOPs** 的情况下，平均得分超过 Block Loop **0.59 pts**。

---

### **与基线方法的对比结果**

#### ✅ 推理任务显著提升（573M 规模）
| Task | Block Loop | SGT (K=3) | Δ |
|------|------------|-----------|----|
| ARC-E | 48.07 | **50.00** | +1.93 |
| WG | 50.19 | **53.43** | +3.24 |
| CSQA | 30.30 | **29.89** | -0.41（略降） |
| **Average** | 35.82 | **36.41** | **+0.59** ✅ |

> 💡 尽管 CSQA 微降，但总体大幅提升，说明 SGT 更擅长复杂关系推理。

#### ✅ 训练效率更高（图 4）
- 对齐相同 FLOPs 时，SGT 的 **训练 PPL 显著更低**；
- 在 $65\times10^{18}$ FLOPs 时，SGT 比 Vanilla 低 **0.48 PPL**；
- Block Loop 反而因冗余计算导致收敛更慢。

---

### **消融实验结果**

#### 🔹 表 2：不同循环组件的影响（573M）

| 方法 | FLOPs | PPL ↓ |
|------|--------|--------|
| Vanilla | 87.41 | 23.973 |
| Block Loop | 92.20 (+5.48%) | 23.560 |
| **High-Entropy Head Loop** | **87.74 (+0.38%)** | **23.452** ✅ |
| Low-Entropy Head Loop | 87.74 | 23.797 ❌ |

> ✅ **高熵头循环效果最好**，且 FLOPs 增加最少；  
> ❌ 低熵头循环几乎无效，验证了“高熵=关键推理单元”。

#### 🔹 表 3：生长方向对比（D2S vs S2D）

| 方法 | CSQA | SIQA | SCIQ | Avg |
|------|------|------|------|-----|
| S2D (浅→深) | 30.39 | 41.40 | 71.80 | 42.75 |
| **D2S (深→浅, ours)** | **31.29** | **42.22** | **72.00** | **43.49** ✅ |

> ✅ 深层先启动更稳定、性能更好；
> 图 5 显示 S2D 出现频繁熵震荡 → 训练不稳定。

#### 🔹 表 4：长上下文外推能力（PPL）

| 序列长度 | Vanilla | Block Loop | **SGT (High-Ent Loop)** |
|---------|--------|-------------|------------------------|
| 2048 (2×) | 24.21 | 24.18 | **24.16** |
| 3072 (3×) | 57.38 | 58.80 ↑ | **56.62** ↓ |
| 4096 (4×) | 116.94 | 119.48 ↑ | **111.66** ↓ |

> ✅ SGT 显著改善长文本泛化能力；  
> ❌ Block Loop 在中等外推时已出现性能下降。

---

## 4. 关键结论和发现

### **主要发现**

1. **Attention Entropy 是有效的信号代理**：
   - 高熵头是语义集成的核心，不应被视为噪声；
   - 可作为指导结构生长的可靠指标。

2. **深度分配应是动态、渐进的过程**：
   - 层间存在“深到浅”成熟路径；
   - 顺应该轨迹可提高训练稳定性与效率。

3. **细粒度稀疏递归优于粗粒度重复**：
   - 仅对关键 heads 增加深层计算即可获得更大收益；
   - 实现 **training-time structural sparsity**，大幅降低冗余。

4. **理论支持**（Appendix E）：
   - 高熵 attention 矩阵具有更好的 token-mixing 性质；
   - 在递归动力学中能更快收敛至 quasi-stationary state。

---

### **方法的局限性**

1. **实验规模受限**：
   - 最大验证模型为 1.2B，尚未扩展至十亿级以上大规模 LLM；
   - 是否适用于 MoE 或更大上下文仍需验证。

2. **超参数探索不足**：
   - 如 $h=2$, $L=3$, $K_{\text{max}}=3$ 设定较为经验性；
   - 缺乏自动调节机制。

3. **推理吞吐略有下降**（见 Table 10）：
   - 虽优于 Block Loop，但仍低于 Vanilla；
   - 需进一步优化 kernel 实现以减少延迟。

---

### **未来工作方向**

1. **扩展至更大模型**（>10B）验证可扩展性；
2. **结合 input-dependent routing**（如 MoE、SwitchHead），构建混合稀疏范式；
3. **设计自适应生长策略**，根据熵变化动态调整层数与头数；
4. **应用于 test-time reasoning**，探索 SGT 在推理阶段的潜力。

---

> 📌 **一句话总结**：  
> SGT 提出了一种**训练时自我生长的稀疏深度架构**，通过**熵引导的渐进式注意力循环**，在仅增加 **1–3% FLOPs** 的代价下，显著提升了模型的推理能力、训练效率和长文本泛化性能，为高效深度扩展提供了新范式。

</details>

---

### 8. [Optimizing Multilingual LLMs via Federated Learning: A Study of Client Language Composition](https://arxiv.org/abs/2603.24242)

**Authors**: Aleix Sant, Jordi Luque, Carlos Escolano  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.24242v1  

#### Abstract
Federated Learning (FL) of Large Language Models (LLMs) in multilingual environments presents significant challenges stemming from heterogeneous language distributions across clients and disparities in language resource availability. To address these challenges, we extended the FederatedScope-LLM fr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimizing Multilingual LLMs via Federated Learning: A Study of Client Language Composition

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文聚焦于**多语言环境下的 Federated Learning（FL）训练 Large Language Models（LLMs）所面临的两大挑战**：
- **客户端间语言分布异构性（non-IID）**：不同客户端可能只拥有单一语言的数据，导致严重的 **client drift**，影响全局模型收敛和泛化能力。
- **低资源语言在联邦训练中表现不佳**：由于数据量少、更新偏差，低资源语言容易被高资源语言主导。

传统 FL 方法（如 FedAvg）在处理这种语言层面的非独立同分布（non-IID）时效果受限，而集中式多语言微调虽性能强但牺牲了数据隐私。

---

### 🚀 提出的新方法与创新思路

#### （1）构建支持多语言 FL 的实验框架
- 在 **FederatedScope-LLM** 框架基础上进行扩展，增加了对多语言指令微调（instruction-tuning）的支持。
- 支持灵活的 prompt 集成、语言感知的数据处理流程和多语言 FL 数据管道。

#### （2）提出 **Local Dynamic Early Stopping for FL (LDES-FL)**
- 一种**基于客户端本地验证损失的动态早停机制**。
- 允许每个客户端根据自身 `validation loss` 自主决定是否暂停或重新加入训练。
- 支持“停止 → 下载新全局模型 → 若性能提升则恢复训练”的闭环机制。

> 🔍 **优势**：
> - 减少不必要的计算开销，提升训练效率与可持续性。
> - 更精细地捕捉多语言 FL 中各语言的收敛差异，避免“一刀切”式的全局早停。

#### （3）系统研究“客户端语言组成”这一设计变量的影响
- 设计了一系列从 **完全单语（100% mono）到高度多语（15% mono）** 的客户端配置。
- 探索了 within-client multilinguality 如何影响模型性能、公平性和优化成本。

> 💡 这是首次将“客户端内部的语言多样性”作为可控变量进行系统分析的研究之一。

---

### ⚖️ 相比现有方法的优势

| 方面 | 本文方法优势 |
|------|---------------|
| **隐私保护 vs 性能权衡** | 在保持数据去中心化的前提下，通过增加客户端多语性逼近集中式训练性能 |
| **训练效率** | LDES-FL 比标准联邦早停减少约 22%-32% 的优化步数，同时不损失性能 |
| **公平性提升** | 多语客户端显著缩小高低资源语言之间的性能差距，尤其利好低资源语言 |
| **实用性增强** | 提供了一个可复现、模块化的多语言 FL 实验平台（开源仓库） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **ALPACA CLEANED 多语言版本**，覆盖 8 种语言（ISO 639-1 编码）：
  - English (`en`)、Spanish (`es`)、German (`de`)、Catalan (`ca`)、Danish (`da`)、Serbian (`sr`)、Croatian (`hr`)、Basque (`eu`)
- 每种语言含 **52,002 条样本**，经 GPT-4 清洗并大规模英译扩增。
- 所有语言数据被划分为训练、验证和测试集，用于联邦与集中式实验。

> ✅ 客户端每轮获得 48,960 训练样本 + 1,020 验证样本；服务器持有固定的多语言测试集（共 4,008 样本）。

---

### ⚙️ 实验设置

| 组件 | 设置说明 |
|------|----------|
| **基础模型** | `salamandra-2b-instruct`（预训练涵盖 35 种欧洲语言） |
| **微调方式** | 使用 **LoRA**（Low-Rank Adaptation），rank=16, alpha=32 |
| **优化器** | OneBitAdam（lr=0.001, grad_clip=1.0） |
| **批大小** | micro-batch=2, gradient_accumulation_steps=16 → effective batch=32 |
| **每轮更新次数** | 每通信轮执行 10 次 optimizer step |
| **通信轮数上限** | 最多 160 轮（实际由 LDES-FL 控制终止） |
| **评估频率** | 每 5 轮验证一次 |

---

### 🧪 客户端语言组成设计（关键变量）
通过控制每个客户端中“单语 vs 多语”数据比例，构造不同异构程度的场景：

| 客户端组成 | 单语占比 | 多语占比 | 异构性等级 |
|-----------|---------|--------|------------|
| 100% mono | 100%     | 0%      | 高（●●●●●●） |
| 85% mono  | 85%      | 15%     | 较高        |
| 70% mono  | 70%      | 30%     | 中等        |
| 50% mono  | 50%      | 50%     | 中低        |
| 30% mono  | 30%      | 70%     | 低          |
| 15% mono  | 15%      | 85%     | 极低（接近 IID）|

> 注：所有客户端总样本数保持一致，仅调整语言构成。

---

### 📊 评估指标

| 指标 | 用途 |
|------|------|
| **ROUGE-L** | 衡量生成文本与参考答案的最长公共子序列匹配度 |
| **FBERT** | 基于 BERT 的语义相似性评分，反映生成质量 |
| **Mean Score** | 所有语言上的平均得分（衡量整体性能） |
| **Standard Deviation (σ)** | 各语言得分的标准差（衡量 multilingual fairness） |

---

### 🔁 基线方法对比

| 基线类型 | 描述 |
|--------|------|
| **Base Model** | 未经微调的原始 `salamandra-2b-instruct` 模型 |
| **Local FT (lang)** | 每种语言单独集中微调（无共享），代表语言特化上限 |
| **Local FT (multilingual)** | 所有语言数据合并后集中微调，视为性能上界（upper bound） |
| **FedAvg (w/ standard early stopping)** | 标准联邦平均算法，用于与 LDES-FL 对比 |
| **FedAvg (不同 mono/multi 配置)** | 主要比较对象，研究语言组成影响 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 3）

| 方法 | Mean ROUGE-L | σ_ROUGE-L ↓ | Mean FBERT | σ_FBERT ↓ |
|------|--------------|-------------|------------|-----------|
| Base Model | 0.167 | 7.70e-2 | 0.867 | 1.57e-2 |
| Local FT (multilingual) | **0.237** | 5.90e-2 | **0.884** | **1.12e-2** |
| FedAvg (100% mono) | 0.203 | 6.47e-2 | 0.877 | 1.26e-2 |
| FedAvg (15% mono) | **0.221** | 6.09e-2 | **0.880** | 1.23e-2 |

> ✅ 所有联邦模型均显著优于 Base Model（p<0.05）

---

### 🔍 与基线方法对比结果

#### （1）vs 集中式多语言微调（Local FT multilingual）
- **仍存在差距**：FedAvg 最佳设置（15% mono）比集中式训练低约 6.8% ROUGE-L。
- 但这是在**保护数据隐私的前提下实现的**，体现了 FL 的实用价值。

#### （2）vs 单语言本地微调（Local FT lang）
- **语言特化能力弱于 Local FT**：每个语言在其专属模型上表现最好。
- 但 **联邦模型提供更均衡的整体性能**，且避免为每种语言维护独立模型的成本。

#### （3）LDES-FL vs 标准联邦早停（见 Table 2）
| 设置 | 方法 | 优化步数 | ROUGE-L |
|------|------|--------|--------|
| 100% mono | Standard | 1.44e+04 | 0.202 |
| 100% mono | **LDES-FL** | **1.12e+04 (-22%)** | **0.203** |
| 15% mono | Standard | 1.11e+05 | 0.224 |
| 15% mono | **LDES-FL** | **7.57e+04 (-32%)** | **0.221** |

> ✅ LDES-FL 显著降低训练成本，同时维持甚至略优性能。

---

### 🔬 消融实验结果（ablation on client composition）

随着客户端内多语性增强（→ 更接近 IID 分布）：

| 观察项 | 变化趋势 |
|-------|--------|
| **平均性能（Mean ROUGE-L/FBERT）** | ↑ 持续上升（100% mono → 15% mono） |
| **跨语言方差（σ）** | ↓ 逐渐减小，表示公平性提高 |
| **训练所需优化步数** | ↑ 显著增加（见 Table 4） |
| **低资源语言增益** | ↑↑ 远高于高资源语言（见 Table 5） |

#### 表 5：按资源分组的性能提升（△ from 100% mono → 15% mono）

| 资源等级 | ROUGE-L 提升 | FBERT 提升 |
|---------|---------------|-------------|
| High (H) | +0.012 | +0.002 |
| Medium (M) | +0.016 | +0.004 |
| Low (L) | **+0.024** | **+0.004** |

> ✅ 结论：**增加客户端多语性对低资源语言帮助最大**，有效缓解“马太效应”。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **客户端语言组成是多语言 FL 的关键设计变量**  
   - 提高客户端内的多语性可显著改善全局模型的性能与公平性。
   - 是缓解 **client drift** 和语言间冲突的有效手段。

2. **联邦训练更适合学习统一的多语言模型，而非语言特化**  
   - 尽管不如单语言本地微调专精，但联邦训练能产出一个**综合性能更强、更公平的单一多语言模型**。
   - 特别适合希望以较低成本部署通用多语言服务的场景。

3. **集中式多语言微调仍是性能上界**  
   - 当允许数据集中时，联合优化带来正向跨语言迁移（cross-lingual transfer），优于分离训练。

4. **LDES-FL 提升训练效率与灵活性**  
   - 动态启停机制适应多语言异步收敛特性，节省高达 32% 的计算资源。

5. **低资源语言受益最大**  
   - 在更平衡的语言分布下，低资源语言获得不成比例的性能跃升，有助于缩小数字鸿沟。

---

### ⚠️ 局限性

1. **数据并非严格平行**  
   - 不同语言间的样本未对齐，可能存在内容重叠或模板共享，导致“伪跨语言迁移”现象。

2. **假设客户端数据量相等**  
   - 实际应用中客户端数据规模差异巨大，本文设定为理想化控制实验。

3. **仅使用 LoRA 微调策略**  
   - 未探索其他 PEFT 方法（如 prefix-tuning, adapter）在多语言 FL 中的表现差异。

4. **语言种类有限（仅 8 种）**  
   - 是否推广至更大规模多语言场景有待验证。

---

### 🔮 未来工作方向

1. **构建严格对齐的多语言数据划分**  
   - 使用翻译对齐句对，更好隔离真正的跨语言迁移效应。

2. **引入个性化 FL 方法（Personalized FL）**  
   - 允许客户端在共享知识的同时保留语言特性，兼顾 specialization 与 generalization。

3. **结合语言家族聚类或路由机制**  
   - 如 Guo et al. (2024) 提出的 language-family clustering，进一步优化参数共享模式。

4. **探索通信压缩与异步更新机制**  
   - 应对真实环境中客户端连接不稳定、延迟高等问题。

5. **扩展至更多任务类型**  
   - 如多语言摘要、对话系统、NER 等下游任务中的 FL 表现。

---

> 📦 开源地址：[https://github.com/Telefonica-Scientific-Research/FedEloquence](https://github.com/Telefonica-Scientific-Research/FedEloquence)  
> 该仓库提供了完整的多语言 FL 实验代码与配置，便于复现与拓展研究。

</details>

---

### 9. [MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination](https://arxiv.org/abs/2603.24579)

**Authors**: Zhuo Li, Yupeng Zhang, Pengyu Cheng, Jiajun Song, Mengyu Zhou, Hao Li, Shujie Hu, Yu Qin, Erchao Zhao, Xiaoxi Jiang, Guanjun Jiang  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.24579v1  

#### Abstract
Hallucination remains a critical bottleneck for large language models (LLMs), undermining their reliability in real-world applications, especially in Retrieval-Augmented Generation (RAG) systems. While existing hallucination detection methods employ LLM-as-a-judge to verify LLM outputs against retri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在检索增强生成（**Retrieval-Augmented Generation, RAG**）系统中仍普遍存在**幻觉（hallucination）**问题，即生成的内容虽然语言流畅，但与提供的证据文档相矛盾。现有的基于“LLM-as-a-judge”的验证方法存在**确认偏见（confirmation bias）**——验证器会无意识地复现原始生成中的错误，导致无法有效检测幻觉。

### 提出的新方法：MARCH
本文提出 **MARCH (Multi-Agent Reinforced self-Check for Hallucination)**，一种通过**多智能体强化学习（MARL）**实现的自检框架，其核心是引入**信息不对称（information asymmetry）**机制来打破确认偏见。

#### 三智能体协作架构：
- **Solver**：基于查询和检索文档生成初始响应。
- **Proposer**：将 Solver 的输出分解为可验证的原子命题（claim-level QA 对）。
- **Checker**：**仅基于检索文档**独立回答这些 QA 对，**完全不接触 Solver 的原始输出**，以避免认知污染。

该设计强制 Checker 成为一个“盲审员”（Blinded Auditor），确保验证过程是真正独立的交叉检验。

### 相比现有方法的优势
| 方面 | 现有方法 | MARCH |
|------|--------|-------|
| **验证机制** | 验证器同时看到输入、文档和生成结果 → 易产生确认偏见 | Checker 被剥夺对生成结果的访问 → 消除信息泄露 |
| **训练目标** | 依赖外部标注或粗粒度奖励（如 RLHF） | 自包含（self-contained）的验证闭环，无需人工标注 |
| **优化方式** | 行为克隆（SFT）易放大幻觉；标准 RL 缺乏细粒度监督 | 引入**零容忍奖励（Zero-Tolerance Reward, ZTR）**，任何单个 claim 错误即惩罚整个轨迹 |
| **可扩展性** | 依赖专家标注或外部工具 | 可在无监督/弱监督下进行自我进化，适用于多种模型家族 |

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 训练数据集（无 ground-truth labels）：
- **STEM Setting**: `BioASQ`（生物医学问答）
- **General Setting**: `2WikiMultiHopQA`, `MuSiQue`（多跳问答）
- 所有训练仅使用 query 和 retrieved documents，**不使用真实答案**

#### 评估基准（含幻觉检测与下游任务）：
| 类型 | 数据集 | 描述 |
|------|-------|------|
| **幻觉检测** | `RAGTruth`, `FaithBench`, `ContextualJudgeBench`, `FACTS Grounding` | 覆盖 QA、摘要、数据转文本等任务，评估事实一致性 |
| **多跳问答** | `HotpotQA`, `2WikiMultiHopQA`, `MuSiQue` | 测试复杂推理能力 |

### 实验设置与评估指标
- **基础模型**：Meta-Llama3.1-8B-Instruct
- **训练框架**：VerL + FSDP，采用 **PPO（Proximal Policy Optimization）**
- **奖励机制**：**Zero-Tolerance Reward (ZTR)**，即所有 claims 必须全部正确才得正奖励
- **评估协议**：
  - 使用 Qwen3-235B-A22B 作为 judge model 进行 response-level factuality eval
  - 每个测试 query 采样 8 次，通过多数投票决定最终结果

### 基线方法对比
- **通用 LLMs**：GPT-4o, Gemini, Claude 等闭源模型
- **开源小模型**：Phi-4-mini, Qwen2.5-14B, Ministral-8B 等
- **专用判别模型**：Prometheus-2, Skywork-8B, SFRJudge 系列
- **RAG 方法**：CoT, IRCoT, Iter-RetGen, RAFT, GenGround 等

---

## 3. 主要实验结果和性能指标

### 幻觉检测性能（越高越好）

#### 在 RAGTruth & FaithBench 上的表现：
| Model | Average Accuracy (%) |
|-------|------------------------|
| Meta-Llama3.1-8B-Instruct (Base) | 55.20 |
| **MARCH-STEM** | **74.93** (+19.73) |
| **MARCH-General** | **75.23** (+20.03) |

> ✅ **结论**：MARCH 显著提升 base model 的幻觉识别能力，甚至优于多个更大规模的闭源模型。

#### 在 FACTS Grounding 上的表现：
| Model | Factuality Score (%) |
|-------|------------------------|
| Llama3.1-8B | 57.09 |
| **MARCH-STEM** | **85.23** |
| **MARCH-General** | **80.12** |

> 🏆 **亮点**：MARCH-STEM 性能接近 Gemini 2.5 Flash（86.60），超越 GPT-4o（79.20）及多个大模型。

#### 在 ContextualJudgeBench 上的表现：
| Model | Average Accuracy (%) |
|-------|------------------------|
| Llama3.1-8B | 29.7 |
| **MARCH-General** | **51.6** (+21.9) |
| **MARCH-STEM** | **52.3** (+22.6) |

> 🔍 特别是在 **faithfulness** 和 **completeness** 维度提升显著，说明 MARCH 能更准确判断回答是否忠实于上下文。

---

### 多跳问答性能（HotpotQA / MuSiQue / 2WikiMQA）

| Method | Backbone | HotpotQA | MuSiQue | 2WikiMQA | Avg |
|--------|----------|----------|---------|----------|-----|
| IRCoT | GPT-4o | 66.4 | 44.2 | 78.0 | – |
| RAFT | Llama3.1-8B | 51.2 | 22.0 | 44.6 | – |
| **MARCH (10-Shots)** | **Llama3.1-8B** | **73.6** | **40.8** | **69.4** | – |
| **MARCH (CoT)** | **Llama3.1-8B** | **70.6** | **36.2** | **70.6** | – |

> 🚀 **突破性表现**：MARCH 在 HotpotQA 上达到 **73.6%**，**超过 GPT-4o + IRCoT（66.4%）**，表明其能显著增强小模型的复杂推理能力。

---

### 消融实验结果（Ablation Study）

#### 不同训练策略对比（Table 5）：
- **Solver-only 更新**：带来一定提升
- **Solver + Checker 联合更新**：效果最佳，在 Few-Shot 设置下绝对增益达 **+11.6%**
> ✅ 验证了 Checker 的审计信号对抑制事实漂移至关重要。

#### 奖励函数对比（Table 6）：
| Reward Type | Accuracy (%) |
|------------|-------------|
| Error Rate Reward (ERR) | 55.46 |
| **Zero-Tolerance Reward (ZTR)** | **61.25** |
> ✅ **ZTR 更优**：严格的“全对才成功”机制更适合复杂 RAG 任务。

#### 奖励标量影响：
- **Penalty-based (-1/0)**：59.06%
- Incentive-based (0/1)：50.42%
> ✅ 负向惩罚提供更强纠正梯度，更利于早期训练稳定。

#### 泛化性测试（Qwen3-8B）：
| Setting | Vanilla | MARCH |
|--------|--------|--------|
| General | 56.84 | **67.90** |
| STEM | 56.84 | **68.11** |
> ✅ MARCH 在不同模型家族上均有效，具备良好泛化性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **信息不对称是打破确认偏见的关键**：通过让 Checker “盲审”，实现了真正独立的事实核查。
2. ✅ **零容忍奖励机制有效驱动事实一致性**：相比比例惩罚，严格的一票否决更能促使模型追求完全正确。
3. ✅ **MARCH 可将小型 LLM（如 8B）提升至媲美甚至超越大型闭源模型的事实性水平**。
4. ✅ **该方法不仅减少幻觉，还增强了多跳推理能力**，说明事实性是复杂推理的基础。
5. ✅ **MARCH 是正交且可组合的**：可与 CoT、Few-Shot、RLHF 等方法无缝集成，带来叠加收益。

### 方法的局限性
1. ⚠️ **可能倾向于“少说少错”**：为避免被 Checker 抓住错误，Proposer 可能减少提问数量（reward hacking）。但可通过提示工程约束最小提问数缓解。
2. ⚠️ 当前聚焦于**数值类事实验证**，对非结构化语义一致性（如事件因果关系）覆盖有限。
3. ⚠️ 训练成本较高，需同步 rollout Solver 和 Checker 轨迹。

### 未来工作方向
- 扩展到更多类型的 claim（如时间顺序、逻辑蕴含）
- 探索动态调整 Proposer 提问密度的机制
- 将 MARCH 应用于开放域对话、代码生成等高风险场景
- 构建端到端的自进化 LLM agent 系统

> 💡 **总体评价**：MARCH 提供了一条**可扩展、无需人工标注、基于自我进化的路径**，推动 LLM 向**内在可信（inherently trustworthy）** 发展，是迈向“可验证智能体”的重要一步。

🔗 **代码开源地址**：[https://github.com/Qwen-Applications/MARCH](https://github.com/Qwen-Applications/MARCH)

</details>

---

### 10. [From AI Assistant to AI Scientist: Autonomous Discovery of LLM-RL Algorithms with LLM Agents](https://arxiv.org/abs/2603.23951)

**Authors**: Sirui Xia, Yikai Zhang, Aili Chen, Siye Wu, Siyu Yuan, Yanghua Xiao  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23951v1  

#### Abstract
Discovering improved policy optimization algorithms for language models remains a costly manual process requiring repeated mechanism-level modification and validation. Unlike simple combinatorial code search, this problem requires searching over algorithmic mechanisms tightly coupled with training d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *From AI Assistant to AI Scientist: Autonomous Discovery of LLM-RL Algorithms with LLM Agents*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
传统语言模型（LLM）的策略优化算法（如 PPO、GRPO）设计依赖于研究人员手动迭代机制级修改（如损失函数、优势估计、正则化等），过程耗时且资源密集。该论文旨在解决 **如何自动化地发现更优的 LLM-RL 策略优化算法**，而非仅进行超参数调优。

### 🚀 提出了什么新方法或新思路
提出 **POISE**（Policy Optimization through Iterative Search and Evidence），一个闭环框架，用于自动发现面向语言模型的策略优化算法。其核心思想是将算法搜索建模为“认识论进化搜索”（Epistemic Evolutionary Search），并引入以下关键机制：

- **结构化基因档案（Structured Genealogically Linked Archive）**：  
  每个算法提案与其可执行代码、标准化评估结果及自然语言反思（natural-language reflections）关联，支持跨代证据复用。
  
- **反射增强的进化求解器（Reflection-Augmented Evolutionary Solver）**：  
  利用 LLM 进行提案生成、分析训练动态，并通过反思提炼设计原则，指导后续搜索方向。

- **三阶段闭环流程**：
  1. **提案生成与选择**（Proposal Generation & Selection）
  2. **实现、验证与评估**（Implementation, Verification, Evaluation）
  3. **反思分析与档案更新**（Reflective Analysis & Archive Update）

### 🔍 相比现有方法的优势
| 维度 | 现有方法 | POISE |
|------|--------|-------|
| 搜索方式 | 手动设计 / 小范围组合尝试 | 自动化、系统性机制空间探索 |
| 反馈利用 | 孤立试错，缺乏知识积累 | 结构化归档 + 跨代证据复用 |
| 设计解释性 | 黑箱改进 | 提炼出可解释的设计原则（如 signal decoupling） |
| 探索效率 | 易陷入局部最优 | 支持从低分父节点演化出高性能子代 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在数学推理任务上进行实验，涵盖六个标准基准：
- **AIME24**, **AIME25**（American Invitational Mathematics Examination）
- **AMC**（American Mathematics Contest）
- **MATH-500**（Hendrycks et al., 2021）
- **Minerva**（Lewkowycz et al., 2022）
- **OlympiadBench**（He et al., 2024）

训练数据来自 MATH 数据集中 Level 3–5 的 5000 个样本。

### ⚙️ 实验设置
- **基础模型**：Qwen2.5-Math-1.5B（用于快速迭代）
- **训练框架**：VERL（Sheng et al., 2024）
- **优化器后端**：分布式 Policy Optimization Backend
- **初始基线**：GRPO（Group Relative Policy Optimization）
- **LLM 引擎**：Gemini-3-pro-preview（作为统一推理引擎驱动所有代理模块）
- **硬件配置**：8×80GB A100 GPUs
- **固定超参**：学习率 $10^{-6}$，全局 batch size 256，group size 8，训练 8 轮

### 📊 评估指标
| 指标 | 应用场景 |
|------|--------|
| `pass@32` | AIME24, AIME25, AMC（采样 32 次取首次正确） |
| `acc@1` | MATH, Minerva, OlympiadBench（确定性生成） |
| **Weighted Overall** | 综合得分（AIME 系列权重 0.2，其余 0.15） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（vs. GRPO 基线）

| 算法 | Weighted Overall | Δ vs. GRPO | AIME25 pass@32 | Δ vs. GRPO |
|------|------------------|------------|----------------|------------|
| **GRPO (baseline)** | 47.8 | — | 26.7% | — |
| **VM-AV-GRPO (best)** | **52.5** | **+4.6** | **43.3%** | **+16.7pp** |
| **AV-GRPO** | 50.9 | +3.1 | 36.7% | +10.0pp |
| **MSA-GRPO** | 50.7 | +2.8 | 43.3% | +16.7pp |
| **SVE-LNA-GRPO** | 48.7 | +0.9 | 30.0% | +3.3pp |

> ✅ **总体提升显著**：最佳变体 **VM-AV-GRPO** 在综合性能上提升 **+4.6**，尤其在最难的 AIME25 上从 26.7% 提升至 **43.3%**，接近人类专家水平。

### 🔬 发现的关键机制
POISE 成功发现了多个有效的新机制：
- **Analytic-Variance Scaling (AV)**：用解析方差替代经验方差进行优势归一化，使稀疏成功信号获得更强梯度。
- **Validity Masking (VM)**：仅基于格式有效的样本计算统计量，避免“格式对但逻辑错”的样本被错误奖励。
- **Conditional Normalization**：按组状态（全错/混合/全对）分别处理，减少信号干扰。
- **Correctness-First Efficiency Shaping**：先确保正确性，再压缩输出长度。

### 🔍 消融实验与控制变量研究
#### （1）长度压缩约束下的表现
引入自然语言指令引导搜索向“高准确率 + 短输出”方向演化：

| 算法 | Overall | Δ | 平均长度 | 长度比 |
|------|--------|----|----------|--------|
| **DACE-GRPO** | 51.7 | +3.9 | 335.7 | 0.709 |
| **CAG-GRPO** | 49.5 | +1.6 | 404.4 | 0.854 |
| **GRPO** | 47.8 | — | 473.6 | 1.000 |

- **DACE-GRPO** 实现最佳权衡：准确率大幅提升的同时，平均响应长度下降 **29.1%**。
- 成功机制：基于组通过率门控效率项（gated on pass rate），实现“难题保精度，简单题压长度”。

#### （2）失败案例分析
- **SA-GRPO**：引入双锚点与固定缩放导致性能严重退化（Overall 34.5），说明不当的机制组合会破坏训练稳定性。
- **DCBE-GRPO**：相对长度增益（RLI）机制引发过压缩，导致 AIME25 下降至 6.7%，验证了无条件压缩的风险。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **自动化策略优化算法发现是可行的**：  
   POISE 在无需人工干预的情况下，自主发现了优于人工设计的 LLM-RL 算法（如 VM-AV-GRPO），证明了“AI 科学家”范式的潜力。

2. **可解释的设计原则浮现**：  
   通过对进化谱系的分析，提炼出三大通用设计原则：
   - **Signal Decoupling**：分离格式、正确性、效率等信号流，避免相互干扰。
   - **Conditional Normalization**：根据不同组态（solved/failed/mixed）动态调整归一化策略。
   - **Correctness-First Efficiency Shaping**：优先保障正确性，再引入压缩偏好。

3. **负证据具有指导价值**：  
   失败的算法（如 FA-GRPO）提供了重要反馈（如“固定惩罚会导致方向性缺失”），后续版本（DFR-GRPO）据此改进为基于 token entropy 的零中心信号，提升了鲁棒性。

4. **低分父节点可能孕育高分后代**：  
   分析显示，在 21 轮父节点选择中，**超过一半时间最强后代并非来自当前最高分父节点**。例如：
   - BN-GRPO（Overall 44.2）→ VM-AV-GRPO（52.5, +8.3）
   - RA-GRPO（44.4）→ MSA-GRPO（50.7, +6.3）  
   表明应保留多样性而非仅贪图即时收益。

---

### ⚠️ 方法的局限性
1. **计算成本高昂**：  
   每次完整训练需大量 GPU 时间，限制了搜索广度与重复实验次数。

2. **泛化能力待验证**：  
   当前成果集中在数学推理领域，尚未验证在开放对话、代码生成、工具使用等任务上的迁移性。

3. **因果解释仍为假设性**：  
   尽管有反思机制，但性能增益的具体归因仍是基于相关性的“证据支撑假说”，非严格因果识别。

4. **安全风险存在**：  
   自动搜索可能产生不稳定或对齐不良的优化策略，需依赖人工监督与验证回路。

---

### 🔮 未来工作方向
1. **提升搜索效率**：  
   引入 surrogate modeling 或 early stopping 降低单次评估开销。

2. **跨任务与跨规模验证**：  
   将 POISE 应用于更大模型（如 70B+）及其他任务（如推理链压缩、多模态对齐）。

3. **构建更强的因果评估协议**：  
   开发机制级消融工具，精确识别每个组件的影响。

4. **推动协作型 AI 科学平台**：  
   扩展为多智能体协同科研系统（multi-agent lab），支持文献阅读、实验设计、论文撰写全流程自动化。

---

> 💡 **总结一句话**：  
> POISE 展示了从“AI 助手”迈向“AI 科学家”的关键一步——不仅能执行任务，还能**自主提出科学假设、验证机制、积累知识并提炼普适规律**，为算法设计开辟了一条可解释、可持续进化的自动化路径。

</details>

---

### 11. [Implicit Turn-Wise Policy Optimization for Proactive User-LLM Interaction](https://arxiv.org/abs/2603.23550)

**Authors**: Haoyu Wang, Yuxin Chen, Liang Luo, Buyun Zhang, Ellie Dingqiao Wen, Pan Li  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23550v1  

#### Abstract
Multi-turn human-AI collaboration is fundamental to deploying interactive services such as adaptive tutoring, conversational recommendation, and professional consultation. However, optimizing these interactions via reinforcement learning is hindered by the sparsity of verifiable intermediate rewards...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Implicit Turn-Wise Policy Optimization for Proactive User-LLM Interaction**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在多轮人机协作场景（如自适应辅导、对话推荐、专业咨询）中，通过强化学习（RL）优化大语言模型（LLM）面临两大挑战：
- **奖励稀疏性（Reward Sparsity）**：最终结果奖励（outcome reward）通常只在对话结束时获得，缺乏对中间步骤的细粒度反馈，导致训练效率低下。
- **用户响应的高随机性（High Stochasticity）**：用户行为多样且不可预测，使得基于价值函数（value model）的估计不稳定。

现有方法如显式过程奖励模型（Process Reward Models, PRMs）依赖大量人工标注或蒙特卡洛采样，成本高昂且难以扩展到在线多轮RL；而隐式PRM（implicit PRM）虽能从结果奖励中推导出token-level的密集奖励，但存在**高方差、过拟合和语义不一致**等问题。

---

### **提出的新方法：ITPO**
本文提出了 **Implicit Turn-wise Policy Optimization (ITPO)**，一种用于主动式用户-LLM交互的策略优化框架，其核心思想是：
- **将奖励建模从token-level提升到turn-level**：每个对话回合（turn）被视为一个语义规划的基本单元，聚合该回合内所有token的隐式奖励，形成更稳定、更具解释性的turn-wise过程奖励。
- **引入归一化机制（Normalization Mechanism）**：提出 **Norm-ITPO**，通过Softmax将turn-level的隐式奖励重新分配全局结果奖励，确保奖励尺度的一致性，增强训练稳定性。

---

### **相比现有方法的优势**
| 特性 | ITPO/Norm-ITPO | Token-level Implicit PRM | 显式PRM / LLM-as-a-Judge |
|------|----------------|--------------------------|----------------------------|
| **无需人工标注** | ✅ | ✅ | ❌（依赖标注或API调用） |
| **可扩展性** | ✅（在线更新） | ✅ | ❌（延迟高，成本大） |
| **训练稳定性** | ✅✅（turn-level + 归一化） | ❌（高方差） | ✅ |
| **语义可解释性** | ✅✅（turn为单位） | ❌（token级无意义） | ✅ |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在三个代表性的多轮协作任务上进行评估：
1. **Math Tutoring**  
   - 数据集：500道来自MATH的数据集  
   - 任务：学生提供不完整问题描述，LLM需主动提问以获取信息并解答  
   - 评估指标：Accuracy (ACC)，由LLM judge判断答案是否正确  

2. **Document Writing**  
   - 数据集：500篇Medium文章摘要  
   - 任务：LLM与用户模拟器迭代生成目标文档  
   - 评估指标：BLEU分数（衡量生成文本与目标文档的重叠度）  

3. **Medical Recommendation**  
   - 数据集：550个样本来自MTMedDialog基准  
   - 任务：医生LLM通过症状询问进行诊断并给出建议  
   - 评估指标：专家LLM judge打分（0–10分），其中8–10分为准确诊断+合理建议  

---

### **实验设置**
- **用户模拟器（User Simulator）**：使用Qwen2.5-14B-Instruct提示生成，模拟真实用户的模糊表达和多样化风格。
- **策略模型（Policy Model）**：主实验使用Qwen2.5-3B-Instruct，验证泛化性时也测试了Qwen3-4B和Qwen2.5-7B。
- **奖励设计**：最大化任务得分 $ R \in [0,1] $ 减去长度惩罚项 $ \lambda \times N $（$ N $为输出token数，$\lambda = 5e^{-6}$），防止冗长。
- **优势估计器（Advantage Estimators）**：结合PPO、GRPO、RLOO等主流RL算法进行策略更新。

---

### **基线方法对比**
| 基线方法 | 类型 | 描述 |
|---------|------|------|
| **Share along Trajectory** | 轨迹级 | 将最终结果奖励均匀广播到所有token |
| **Uniform Decomposition** | 回合级 | 使用Dirichlet分布随机分配回合奖励 |
| **Value Model (VM)** | token级 | 学习值函数，使用TD误差作为奖励 |
| **LLM-as-a-Judge** | 回合级 | 外部LLM（Qwen2.5-14B）分析每回合质量并赋权 |
| **Implicit PRM (PRIME)** | token级 | 原始隐式PRM方法，token级log-likelihood ratio |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**
在多种优势估计器下，**Norm-ITPO** 均显著优于所有基线：

| 任务 | 指标 | 最优方法（Norm-ITPO） | 第二名 | 提升幅度 |
|------|------|------------------------|--------|----------|
| **Math Tutoring** (RLOO) | Accuracy | **32.50%** | RLOO + Uniform (30.00%) | **+8.3%** |
| **Document Writing** (RLOO) | BLEU | **44.83** | RLOO + LLM-as-Judge (42.34) | **+5.9%** |
| **Medical Recommendation** (RLOO) | Score | **69.24** | RLOO + LLM-as-Judge (66.77) | **+3.7%** |

> 注：与最常用的“轨迹共享”基线相比，Norm-ITPO平均提升达 **12–34%**。

---

### **与基线方法的对比结果**
- **优于所有reward shaping方法**：在三个任务上均超越LLM-as-a-Judge、Uniform Decompose、PRIME等。
- **尤其在PPO设置下优势明显**：例如在Medical Recommendation任务中，Norm-ITPO比ITPO高出 **8.3%**，说明归一化机制对依赖值函数的学习至关重要。
- **优于训练额外value model的方法**：即使加入value model，ITPO仍带来额外 **10–16%** 性能增益。

---

### **消融实验结果**
- **Turn-level vs Token-level**：
  - 图3显示，turn-level偏好在约50步内稳定，而token-level需超过200步，收敛慢且波动大。
  - 图2显示，相同token的token-level奖励在不同训练阶段波动剧烈（红框），而turn-level奖励稳定上升。
- **归一化机制的作用**：
  - 图7显示，未归一化的隐式奖励与结果奖励之间的映射斜率波动大，导致非平稳目标。
  - Norm-ITPO通过强制尺度一致性，显著提升训练稳定性，尤其在PPO中效果显著。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Turn-level奖励优于token-level**：回合作为语义单元，能有效降低噪声、提高鲁棒性和可解释性。
2. **隐式奖励可自动学习有意义的偏好**：ITPO推断出的turn-wise偏好与人类判断高度一致（语义可解释性实验中达到 **75%** 匹配率，接近Gemini-3.0-Pro的58/64）。
3. **归一化机制显著提升稳定性**：Norm-ITPO通过将隐式奖励与结果奖励对齐，解决了尺度漂移问题，特别有利于PPO等依赖值函数的方法。
4. **通用性强**：ITPO可无缝集成PPO、GRPO、RLOO等多种优势估计器，并在不同规模模型（3B–7B）上表现一致。

---

### **方法的局限性**
- **依赖高质量的结果奖励信号**：若最终结果评估本身有偏或噪声大，隐式奖励学习也会受影响。
- **无法处理完全错误的早期决策**：虽然能识别关键回合，但若早期回复严重偏离目标，后续修正可能受限。
- **归一化超参数敏感**：温度参数 $ \eta $ 控制奖励集中程度，需适当调节。

---

### **未来工作方向**
- **扩展至多智能体协作**：将ITPO应用于多个LLM代理间的协作任务。
- **动态调整归一化策略**：根据对话进展自适应地调整 $ \eta $。
- **结合外部工具使用**：在Tool-use类任务中，探索turn-level信用分配与工具调用的联合优化。
- **真实用户部署验证**：当前实验基于LLM用户模拟器，未来可在真实用户环境中测试。

---

> **代码已开源**：https://github.com/Graph-COM/ITPO

</details>

---

### 12. [Residual Attention Physics-Informed Neural Networks for Robust Multiphysics Simulation of Steady-State Electrothermal Energy Systems](https://arxiv.org/abs/2603.23578)

**Authors**: Yuqing Zhou, Ze Tao, Fujun Liu  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23578v1  

#### Abstract
Efficient thermal management and precise field prediction are critical for the design of advanced energy systems, including electrohydrodynamic transport, microfluidic energy harvesters, and electrically driven thermal regulators. However, the steady-state simulation of these electrothermal coupled ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Residual Attention Physics-Informed Neural Networks for Robust Multiphysics Simulation of Steady-State Electrothermal Energy Systems

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**稳态电热耦合多物理场系统**（steady-state electrothermal coupled multiphysics systems）的数值模拟难题展开研究。这类系统广泛存在于微能量器件、电驱动热调控器和电液动力学传输等应用中，其挑战在于：
- 强非线性场耦合（如速度、压力、电势、温度相互反馈）
- 温度依赖的变系数（temperature-dependent coefficients）
- 复杂界面动力学（oblique interfaces with jump conditions）
- 传统 PINN 在此类问题上易出现优化失衡、梯度尺度不一致、局部结构捕捉能力弱等问题。

### 提出的新方法与创新思路
作者提出了一种名为 **Residual Attention Physics-Informed Neural Network (RA-PINN)** 的新型框架，其核心创新包括：

- ✅ **统一五场算子建模（Unified Five-Field Operator Formulation）**  
  将速度 $u$、压力 $p$、电势 $\phi$、温度 $T$ 和连续性约束整合为一个统一的 PDE 向量算子 $ \mathcal{N}(U) = 0 $，实现对多物理场的联合求解。

- ✅ **残差注意力机制（Residual-Attention Mechanism）**  
  结合 **残差连接（residual-connected feature propagation）** 与 **通道调制注意力（attention-guided channel modulation）**，增强网络对局部强梯度区域（如边界层、界面附近）的敏感性，提升特征表达能力。

- ✅ **自适应残差点采样（Adaptive Residual-Based Collocation Sampling）**  
  动态调整训练点分布，将更多采样点集中在残差较大的区域（高误差区），从而提高求解效率与精度。

- ✅ **接口一致性处理机制**  
  针对接口问题设计专门的传输残差项（transmission residual），确保跨斜界面的场连续性和通量跳跃条件满足。

### 相比现有方法的优势
| 维度 | RA-PINN 优势 |
|------|-------------|
| **表示能力** | 残差注意力机制能有效捕捉局部耦合结构和陡峭梯度，优于纯MLP或LSTM结构 |
| **鲁棒性** | 在变系数、间接约束（pressure gauge）、复杂几何界面下仍保持高保真度 |
| **泛化性** | 统一框架适用于多种典型电热耦合场景，无需针对每类问题重新设计架构 |
| **精度稳定性** | 在所有测试案例中均取得最低误差，尤其在强非线性条件下优势显著 |

---

## 2. 核心实验方法和设置

### 数据集与基准任务
论文并未使用真实世界数据集，而是构建了四个具有解析解或高分辨率仿真参考解的**合成 benchmark 问题**，覆盖典型电热耦合场景：

| Case | 物理特性 |
|------|----------|
| **Case 1** | 常系数耦合系统（Constant-coefficient coupling） |
| **Case 2** | 压力规范约束（Indirect pressure-gauge constraint） |
| **Case 3** | 温度依赖输运系数（Temperature-dependent transport） |
| **Case 4** | 斜向材料界面（Oblique-interface with jump conditions） |

所有案例定义在单位正方形域 $\Omega = [0,1] \times [0,1]$ 上，并提供参考场图用于定量比较。

### 实验设置与评估指标
- **输入输出**：输入为空间坐标 $(x, y)$，输出为五维场向量 $U = [u, v, p, \phi, T]$
- **训练方式**：基于自动微分计算 PDE 残差，端到端训练
- **损失函数组成**：
  $$
  \mathcal{L}_{\text{total}} = \lambda_{\text{res}} \|\mathcal{R}\|^2 + \lambda_b \|\mathcal{R}_b\|^2 + \lambda_{\text{gauge}} |\mathcal{R}_g|^2 + \lambda_r \|\mathcal{R}_t\|^2 + \lambda_{\text{data}} \|\mathcal{L}_{\text{data}}\|^2
  $$
- **评估指标**：
  - MSE（Mean Squared Error）
  - RMSE（Root Mean Squared Error）
  - MAE（Mean Absolute Error）
  - Relative $L^2$ Error（相对 $L^2$ 范数误差）

### 基线方法对比
与以下主流 PINN 变体进行对比：
- **Pure-MLP**：标准前馈神经网络作为基础 backbone
- **LSTM-PINN**：引入时序记忆结构以增强长期依赖建模
- **pLSTM-PINN**：并行 LSTM 架构，提升并行性与收敛速度

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 平均 Relative $L^2$ Error 对比（越低越好）

| Method / Case | Case 1 | Case 2 | Case 3 | Case 4 | Overall Avg |
|---------------|--------|--------|--------|--------|------------|
| **RA-PINN**   | 3.235×10⁻³ | 7.660×10⁻³ | 5.065×10⁻³ | 1.377×10⁻³ | **4.334×10⁻³** |
| LSTM-PINN     | 5.695×10⁻³ | 1.038×10⁻² | 7.155×10⁻³ | 1.449×10⁻³ | 6.170×10⁻³ |
| pLSTM-PINN    | 1.205×10⁻² | 3.608×10⁻² | 8.456×10⁻¹ | 3.895×10⁻² | 2.335×10⁻¹ |
| Pure-MLP      | 5.105×10⁻² | 4.868×10⁻² | 3.031×10⁻² | 1.061×10⁻² | 3.516×10⁻² |

> ✅ **RA-PINN 在所有 case 中均取得最优平均误差**

#### 最佳表现亮点
- 在 **Case 3（温度依赖输运）** 中，RA-PINN 的相对 $L^2$ 误差仅为 `5.065e-3`，而 pLSTM-PINN 高达 `8.456e-1`（相差两个数量级）
- 在 **Case 4（斜界面）** 中，尽管 LSTM-PINN 表现接近，RA-PINN 仍以更小的 MAE 和相对 $L^2$ 错误胜出
- 所有字段（u, v, p, φ, T）中，RA-PINN 在绝大多数指标上达到最小值（见 Table 1 加粗项）

### 与基线方法的对比结果
| 方面 | 对比结论 |
|------|---------|
| **精度全面领先** | RA-PINN 在 MSE、RMSE、MAE、Relative $L^2$ 四项指标上均优于其他模型，在所有四个 benchmark 中均排名第一 |
| **面对复杂物理更具鲁棒性** | 当引入变系数或界面跳跃时，Pure-MLP 和 pLSTM-PINN 性能急剧下降；而 RA-PINN 仍能维持高精度 |
| **结构保真度更高** | 可视化结果显示 RA-PINN 更好地保留了压力过渡带、电势尖峰和温度梯度结构，伪影最少 |

### 消融实验分析（隐含于不同 case 设计中）
虽然未明确列出消融实验表格，但通过四类 benchmark 的递进设计实现了功能验证：
- **Case 1 → Case 2**：验证了 RA-PINN 对“间接约束”（gauge condition）的良好适应能力
- **Case 1 → Case 3**：证明了其在**变量系数强非线性反馈**下的稳定性
- **Case 1 → Case 4**：展示了其在**非轴对齐界面**上的精确建模能力，体现了几何感知优势

此外，注意力机制与自适应采样的协同作用也被强调为关键增益来源。

---

## 4. 关键结论和发现

### 主要发现
1. 🔹 **RA-PINN 显著提升了稳态多物理场仿真的精度与鲁棒性**  
   在四种典型电热耦合场景下，RA-PINN 均优于当前主流 PINN 架构，特别是在强非线性、变系数和界面主导问题中表现突出。

2. 🔹 **残差注意力机制是解决优化不平衡的关键**  
   通过注意力门控放大携带陡峭梯度和局部耦合信息的通道，使网络能够动态聚焦难拟合区域，避免某些场被压制。

3. 🔹 **自适应采样增强了对关键区域的关注**  
   动态重加权 collocation points 分布，使得训练资源集中于高残差区域（如界面、边界层），显著提升收敛效率与最终精度。

4. 🔹 **统一框架具备良好可扩展性**  
   所提出的五场统一算子形式允许灵活激活不同物理约束（如 gauge 或 interface），便于推广至其他多物理场系统。

### 方法的局限性
- ⚠️ **训练成本较高**  
  如 Table 2 所示，RA-PINN 的训练时间最长（最高达 **39.81 小时**），远高于 pLSTM-PINN（约 4 小时）和 Pure-MLP（约 4–15 小时）。这限制了其在实时优化或大规模参数扫描中的应用。
- ⚠️ **尚未验证于三维或瞬态问题**  
  当前工作仅限于二维稳态问题，是否能拓展到三维空间或时间演化系统有待进一步研究。
- ⚠️ **超参数敏感性未充分讨论**  
  注意力模块权重、采样更新频率、损失系数 $\lambda$ 等可能影响性能，但文中未开展系统调参分析。

### 未来工作方向
- 🔄 开发轻量化版本 RA-PINN，结合知识蒸馏或稀疏化策略降低计算开销
- 🌐 推广至三维非规则几何与瞬态多物理场耦合问题
- 🤖 探索与数字孪生（digital twin）系统的集成，支持在线监测与反演诊断
- 📊 引入不确定性量化机制（如 Bayesian RA-PINN）以提升预测可信度
- 💡 结合物理先验发现机制，实现自动识别未知本构关系或源项

---

> ✅ **总结一句话**：  
> 本文提出的 **RA-PINN** 框架通过融合 **残差注意力机制** 与 **自适应残差采样**，在稳态电热耦合多物理场模拟中实现了前所未有的精度与鲁棒性，为下一代能源器件的高保真建模提供了强有力的计算工具。

</details>

---

### 13. [MetaKube: An Experience-Aware LLM Framework for Kubernetes Failure Diagnosis](https://arxiv.org/abs/2603.23580)

**Authors**: Wei Sun, Ting Wang, Xinran Tian, Wanshun Lan, Xuhan Feng, Haoyue Li, Fangxin Wang  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23580v1  

#### Abstract
Existing LLM-based Kubernetes diagnostic systems cannot learn from operational experience, operating on static knowledge bases without improving from past resolutions. We present MetaKube, an experience-aware LLM framework through three synergistic innovations: (1) an Episodic Pattern Memory Network...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MetaKube: An Experience-Aware LLM Framework for Kubernetes Failure Diagnosis

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的基于 **LLM** 的 Kubernetes 故障诊断系统存在三大根本缺陷：
- **无法从操作经验中学习**：现有 RAG 系统依赖静态知识库，无法利用历史诊断结果进行持续优化。
- **高质量诊断数据稀缺**：Kubernetes 故障知识分散在文档、论坛和私有 runbooks 中，缺乏结构化、高质量的训练与检索数据。
- **企业级数据隐私要求高**：生产环境中禁止将敏感日志和配置发送至外部 LLM API，而本地部署的大模型（如 70B+）计算开销过大，小模型（<10B）又能力不足。

### 提出的新方法与创新
为解决上述问题，作者提出 **MetaKube** —— 一种具备“经验感知”能力的 LLM 框架，其核心创新在于三个协同组件的设计：

#### （1）Episodic Pattern Memory Network (EPMN)
- **功能**：抽象并存储历史故障解决中的模式（patterns），支持基于相似度、时效性和成功历史的置信度校准检索。
- **优势**：实现快速模式匹配（直觉路径）和引导因果探索（分析路径），使系统能“记住”过去的经验并复用。

#### （2）Meta-Cognitive Controller（元认知控制器）
- **功能**：动态路由查询到不同处理路径：
  - 当问题熟悉度高（记忆相似性 > 阈值）时，走**直觉路径**（Pint），快速响应；
  - 否则触发**分析路径**（Pana），结合 KubeGraph 进行深度因果推理。
- **优势**：在速度与准确性之间自适应权衡，提升资源利用率。

#### （3）KubeLLM：领域专用 LLM
- 基于 **Qwen3-8B** 构建，通过领域特定的 **Supervised Fine-Tuning (SFT)** 在自建的 **Kubernetes Fault Resolution Dataset (KFRD)** 上增强诊断能力。
- 支持本地部署，保障数据隐私，同时性能接近 GPT-4。

### 相比现有方法的优势
| 维度 | MetaKube | 传统方法 |
|------|---------|----------|
| 学习机制 | ✅ 持续从操作经验中学习（experience-aware） | ❌ 静态知识库，无反馈学习 |
| 推理策略 | ✅ 双路径动态切换（直觉 vs 分析） | ❌ 单一推理模式 |
| 数据效率 | ✅ 自建高质量结构化数据集 KFRD 和 KubeGraph | ❌ 依赖碎片化公开数据 |
| 部署可行性 | ✅ 本地部署 8B 模型，兼顾性能与隐私 | ❌ 大模型难部署，小模型不准 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### （1）KubeFault（主测试集）
- 包含 **1,873 个真实世界 Kubernetes 故障场景**，覆盖六类错误：
  - Resource, Network, Scheduling, Image, Configuration, System Errors
- 每条样本包含：症状描述、环境上下文、日志、根因、解决方案步骤。
- 由电信公司运维工程师审核修正，确保权威性。

#### （2）Kubernetes Fault Resolution Dataset (KFRD)
- 自建数据集，共 **7,000 条高质量样本**。
- 构建流程四阶段：
  1. **收集**：Stack Overflow、GitHub issues
  2. **重构**：加入用户失败尝试 → 形成 `(Problem, Attempt, Solution)` 结构
  3. **增强**：使用 LLM 生成 Chain-of-Thought 推理链
  4. **去重与扩充**：语义去重 + 合成数据
- 划分：5,000 用于 SFT 训练，2,000 用于评估。

#### （3）KubeGraph
- 基于 GraphRAG 构建的知识图谱，包含 **44,022 实体** 和 **111,832 关系**。
- 覆盖六大故障类别，支持高效因果链检索。

---

### 实验设置与评估指标

#### 评估方式采用双轨制：
| 类型 | 描述 |
|------|------|
| **GPT-5 自动评分** | 对可扩展性友好，标准化打分 |
| **人类专家盲评** | 三位具有 5+ 年经验的运维工程师独立评分，隐藏模型身份 |

#### 四维评估指标（每项满分 10 分）：
| 指标 | 定义 |
|------|------|
| **Effectiveness (Eff.)** | 根因识别（40%）、解决方案质量（30%）、预防建议（30%） |
| **Equivalence (Equ.)** | 与参考方案一致性、方法论连贯性 |
| **Completeness (Com.)** | 步骤完整性（30%）、命令正确性（30%）、边缘情况处理（40%） |
| **Safety/Accuracy (S/A)** | 技术正确性、符合 K8s 最佳实践 |

最终得分取四项平均。

---

### 基线方法对比
| 基线模型 | 类型 |
|--------|------|
| GPT-4.1 / GPT-4.1-mini / Qwen3-8B | Zero-shot（无检索） |
| GPT-4.1 / GPT-4.1-mini / Qwen3-8B | GraphRAG 增强版本（使用通用知识图谱） |
| **MetaKube (ours)** | 提出的方法（融合 EPMN + KubeGraph + KubeLLM + 元控制器） |

所有实验均在相同硬件环境下运行（A100/A6000 GPU），保证公平性。

---

## 3. 主要实验结果和性能指标

### 总体性能对比（KubeFault 数据集，100 分制）

| 方法 | Eff. | Equ. | Com. | S/A | **Avg.** |
|------|------|------|------|-----|----------|
| GPT-4.1 (Zero-shot) | 72.1 | 74.3 | 69.8 | 78.9 | **73.8** |
| Qwen3-8B (Zero-shot) | 48.7 | 51.2 | 46.1 | 57.4 | **50.9** |
| GPT-4.1 (GraphRAG) | 89.3 | 92.6 | 91.4 | 94.1 | **91.9** |
| Qwen3-8B (GraphRAG) | 66.7 | 69.1 | 64.8 | 73.3 | **68.5** |
| **MetaKube (Ours)** | **91.2** | **90.8** | **87.3** | **92.5** | **90.5** |

> 📌 **关键发现**：
> - MetaKube 将 Qwen3-8B 的性能从 **50.9 提升至 90.5**，提升幅度达 **+39.6 分**。
> - 性能逼近 GPT-4.1 + GraphRAG（仅差 1.4 分），但完全本地部署，保障数据隐私。
> - 在 **Human Expert Evaluation** 中也取得最高分（75.2），尤其在 **Effectiveness** 和 **Safety/Accuracy** 表现优异。

---

### 消融实验结果（Ablation Studies）

#### （1）EPMN 消融实验（移除 EPMN 模块）
![](https://via.placeholder.com/400x200?text=EPMN+Ablation+Study)

| 指标 | 有 EPMN | 无 EPMN | 提升 |
|------|--------|--------|------|
| Effectiveness | 91.2 | 78.7 | **+12.5 (+15.9%)** |
| Equivalence | 90.8 | 78.5 | +12.3 |
| Safety/Accuracy | 92.5 | 75.8 | **+16.6%** |
| **Overall** | 90.5 | 78.5 | **+15.3%** |

> ✅ **结论**：EPMN 显著提升所有维度表现，尤其在安全性和有效性方面，说明“经验记忆”对复杂系统诊断至关重要。

#### （2）KubeLLM 消融实验（是否进行 SFT）
| 模型 | Effectiveness | Equivalence | Completeness | Safety | Avg. |
|------|-------------|------------|--------------|--------|-------|
| Qwen3-8B (原始) | 64.1 | 54.0 | 54.0 | 33.7 | ~51.5 |
| MetaKube w/o SFT | 79.3 | 78.5 | 75.8 | 70.1 | ~75.9 |
| **MetaKube (with SFT)** | **91.2** | **90.8** | **87.3** | **92.5** | **90.5** |

> ✅ **结论**：SFT 带来约 **+14.6 分** 提升，证明领域微调对小模型能力跃迁的关键作用。

#### （3）KubeGraph 消融实验
| 数据集 | w/KubeGraph | w/o KubeGraph | 提升 |
|--------|-------------|---------------|------|
| KubeFault（域内） | 75.2 | 34.6 | **+117.3%** |
| Telecom Dataset（域外） | 57.6 | 22.4 | **+157.1%** |

> ✅ **结论**：KubeGraph 不仅提升域内性能，更显著增强泛化能力，表明其捕捉到了跨环境的通用故障模式。

---

## 4. 关键结论和发现

### 主要发现
1. **经验积累是提升诊断准确性的关键**：  
   EPMN 成功实现了“从历史中学习”，使得系统能够不断提炼诊断模式，并在新问题中复用，形成正向循环。

2. **双路径架构优于单一推理模式**：  
   Meta-Cognitive Controller 能智能选择“快而准”或“慢而深”的路径，在简单问题上节省资源，在复杂问题上深入分析。

3. **小模型 + 领域精调 ≈ 大模型效果**：  
   通过 SFT + LoRA 微调 Qwen3-8B，可在不牺牲性能的前提下实现本地化部署，打破“必须用大模型”的迷思。

4. **结构化知识图谱极大提升泛化能力**：  
   KubeGraph 不仅提供事实知识，还支持多跳因果推理，尤其在未知或罕见故障中表现出色。

---

### 方法的局限性
- **冷启动问题**：初期缺乏历史诊断记录时，EPMN 效果受限。
- **依赖高质量初始数据**：KFRD 和 KubeGraph 的构建成本较高，需大量人工审核。
- **动态集群状态建模有限**：当前主要基于日志和配置，未完全整合实时监控流数据（如 Prometheus metrics）。

---

### 未来工作方向
1. **引入在线学习机制**：让 EPMN 支持增量更新，实现实时经验沉淀。
2. **融合多模态输入**：集成 metrics、traces、logs 构建统一可观测性接口。
3. **自动化知识图谱演化**：让 KubeGraph 能自动发现新关系并验证其有效性。
4. **跨集群迁移学习**：将在一个集群学到的模式迁移到其他组织的集群中，推动社区共建共享。

---

> 🔗 **开源信息**：  
> 项目代码、KFRD 数据集、KubeGraph 与 KubeLLM 模型均已开源：  
> [https://github.com/MetaKube-LLM-for-Kubernetes-Diagnosis/MetaKube](https://github.com/MetaKube-LLM-for-Kubernetes-Diagnosis/MetaKube)

</details>

---

### 14. [Kronecker-Structured Nonparametric Spatiotemporal Point Processes](https://arxiv.org/abs/2603.23746)

**Authors**: Zhitong Xu, Qiwei Yuan, Yinghao Chen, Yan Sun, Bin Shen, Shandian Zhe  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23746v1  

#### Abstract
Events in spatiotemporal domains arise in numerous real-world applications, where uncovering event relationships and enabling accurate prediction are central challenges. Classical Poisson and Hawkes processes rely on restrictive parametric assumptions that limit their ability to capture complex inte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Kronecker-Structured Nonparametric Spatiotemporal Point Processes

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **spatiotemporal point process** 模型在建模时空事件时面临以下挑战：
- **经典模型（如 Poisson 和 Hawkes 过程）** 依赖于参数化假设（如指数衰减核），难以捕捉复杂的交互模式（如抑制、中性影响、时变效应）。
- **神经点过程模型（Neural Point Processes）** 虽然提升了表达能力，但通过黑箱方式编码事件历史，缺乏对事件间关系的**可解释性**。

因此，如何在保持高建模灵活性的同时实现**透明且可解释的事件关系发现**，是本文要解决的核心问题。

---

### 🚀 提出的新方法：KSTPP
作者提出了 **Kronecker-Structured Nonparametric Spatiotemporal Point Process (KSTPP)**，其核心思想如下：

#### （1）非参数化建模
- 使用 **Gaussian Process (GP)** 对背景强度 $g(x, y)$ 和影响核 $f(\Delta t, \Delta x, \Delta y)$ 进行先验建模：
  - $g$: 空间 GP，建模自发事件；
  - $f$: 时空 GP，建模过去事件对未来的影响。
- 支持任意形式的影响（excitation, inhibition, neutrality），突破传统 Hawkes 只能建模激发的限制。

#### （2）Kronecker 结构提升可扩展性
- 采用**可分离乘积核（separable product kernels）**：
  $$
  K = K_0(\Delta t) \otimes K_1(\Delta x) \otimes K_2(\Delta y)
  $$
- 在结构化网格上定义诱导点（inducing points），使得协方差矩阵具有 **Kronecker 结构**。
- 利用 Kronecker 代数（Kronecker algebra）将高维运算分解为各维度独立计算，显著降低时间和内存复杂度。

#### （3）高效积分方案
- 针对似然函数中的不可解积分，提出 **tensor-product Gauss-Legendre quadrature** 方法，在结构化网格上进行高阶数值积分，兼顾精度与效率。

---

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **建模能力** | 支持非参数化、灵活的影响核，可捕获 excitation/inhibition/time-varying effects |
| **可解释性** | 显式输出 influence kernel $f$，便于分析事件间的时空影响模式 |
| **可扩展性** | Kronecker 结构使训练和推理可扩展至大规模事件集合（避免 $O(N^3)$ 复杂度） |
| **预测性能** | 在多个真实世界数据集上优于主流神经点过程模型 |

---

## 2. 核心实验方法和设置

### 📚 数据集
实验涵盖 **合成数据** 与 **三个真实世界基准数据集**：

| 数据集 | 描述 |
|-------|------|
| **SYN1 & SYN2** | 合成 spatiotemporal Hawkes 过程，分别引入时间依赖和空间依赖的抑制机制，用于验证模型恢复真实 intensity 和 kernel 的能力 |
| **Earthquake** | 日本 1990–2020 年地震记录，每月一个序列，研究地震触发与抑制模式 |
| **Covid-19** | 新泽西州每日新冠病例报告，每周一个序列 |
| **Citibike** | 纽约市 2019 年共享单车出行记录，每天一个序列 |

---

### 📊 实验设置与评估指标

#### 评估任务
- **下一事件预测**：
  - 时间预测：RMSE（Root Mean Square Error）
  - 位置预测：Euclidean Distance

#### 基线方法对比
| 类别 | 方法 |
|------|------|
| 参数化模型 | STHP（Spatiotemporal Hawkes Process）、Homogeneous Poisson Process |
| 神经点过程 | NSTPP、NHP、THP、DeepSTPP、NJSDE |
| 扩散生成模型 | DSTPP（Diffusion Spatiotemporal Point Process） |

#### 实现细节
- 使用 **PyTorch** 实现，Adam 优化器（学习率 $10^{-3}$）
- SoftPlus 作为正则链接函数（$\beta=1$）
- 每维度使用 8–16 个 Gauss-Legendre quadrature 节点
- 使用 SE 或 Matérn($\nu=5/2$) 作为 GP 核函数

---

## 3. 主要实验结果和性能指标

### 📈 关键性能表现（来自 Table 3）

| Model | Earthquake (Time ↓) | Earthquake (Space ↓) | Covid-19 (Time ↓) | Citibike (Time ↓) |
|-------|------------------------|-------------------------|--------------------|-------------------|
| Poisson | 0.412 | 9.45 | 0.113 | 0.239 |
| STHP | 0.424 | 8.35 | 0.100 | 0.633 |
| NSTPP | 0.547 | 8.11 | 0.145 | 0.355 |
| DeepSTPP | 0.341 | 9.20 | 0.197 | 0.234 |
| DSTPP (diffusion) | **0.375** | **6.77** | 0.093 | 0.200 |
| **KSTPP (Ours)** | **0.372** | **6.72** | **0.100** | **0.206** |

> ✅ **KSTPP 在所有数据集的空间预测中取得最优或次优结果，在时间预测上也达到顶尖水平**

---

### 🔬 强调亮点结果

#### （1）强度函数恢复（Intensity Recovery）
- 在 SYN1 和 SYN2 上，KSTPP 的 **相对 L2 误差最小**（见 Table 1 & 2）：
  - SYN1：相对 L2 error = `4.44e-2` vs 第二名 NSTPP (`5.57e-2`)
  - SYN2：`2.00e-2` vs `2.34e-2`（NHP）
- 可视化显示 KSTPP 准确还原了 excitation/inhibition 的时空动态。

#### （2）影响核估计（Influence Kernel Estimation）
- 图 3 和图 7 显示，KSTPP 成功识别出：
  - 小时间滞后下的局部激发；
  - 较大空间距离处的负值区域（即 inhibition）；
- 而 STHP 因强制非负核，无法识别抑制效应，导致严重偏差。

#### （3）真实数据中的模式发现（Earthquake）
- 图 4 展示 learned kernel 随 $\Delta t$ 变化的趋势：
  - $\Delta t \approx 0$: 强烈近场激发（符合 aftershock 特征）
  - $\Delta t > 0.5$: 影响迅速衰减至零（符合 Omori-Utsu 定律）
  - 同时存在局部负值 → 表明“stress shadow”现象（地震后某些区域活动被抑制）

> 💡 这些发现与地质学理论一致，证明模型具备**科学可解释性**

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **KSTPP 实现了灵活性与可解释性的统一**：
   - 通过 GP 非参数建模，支持复杂交互模式；
   - 显式输出 influence kernel，可用于科学分析。

2. **Kronecker 结构极大提升效率**：
   - 将高维 GP 推断转化为低维操作，实现对大规模事件序列的有效训练。

3. **预测性能媲美甚至超越神经模型与扩散模型**：
   - 在多个真实数据集中达到 SOTA 或接近 SOTA 水平，尤其在空间预测上领先。

4. **成功揭示现实世界的复杂事件机制**：
   - 如地震中的激发-抑制共存结构，为风险建模提供更精细视角。

---

### ⚠️ 局限性
- **假设空间域为矩形**：当前框架基于规则网格，难以直接应用于不规则地理区域（如城市边界）。
- **计算仍受限于网格分辨率**：虽然比全协方差快，但高分辨率网格会增加内存开销。
- **未考虑 mark（事件类型）**：仅建模事件发生的时间与位置，未扩展到多类型事件（marked point process）。

---

### 🔮 未来工作方向
1. 扩展至 **non-rectangular domains**（如图神经网络 + GP 结合）；
2. 引入 **event marks**，构建 marked KSTPP；
3. 探索 **online inference** 机制以支持实时预测；
4. 应用于更多领域：如犯罪预测、金融交易、社交媒体传播等需要可解释建模的场景。

---

> ✅ 总结：**KSTPP 是首个将 Kronecker-structured GP 成功应用于 spatiotemporal point process 的工作，在保持高度可解释性的同时实现了卓越的预测性能，为科学驱动的事件建模提供了新范式。**

</details>

---

### 15. [AVO: Agentic Variation Operators for Autonomous Evolutionary Search](https://arxiv.org/abs/2603.24517)

**Authors**: Terry Chen, Zhifan Ye, Bing Xu, Zihao Ye, Timmy Liu, Ali Hassani, Tianqi Chen, Andrew Kerr, Haicheng Wu, Yang Xu, Yu-Jung Chen, Hanfeng Chen, Aditya Kane, Ronny Krashinsky, Ming-Yu Liu, Vinod Grover, Luis Ceze, Roger Bringmann, John Tran, Wei Liu, Fung Xie, Michael Lightstone, Humphrey Shi  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.24517v1  

#### Abstract
Agentic Variation Operators (AVO) are a new family of evolutionary variation operators that replace the fixed mutation, crossover, and hand-designed heuristics of classical evolutionary search with autonomous coding agents. Rather than confining a language model to candidate generation within a pres...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AVO: Agentic Variation Operators for Autonomous Evolutionary Search

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于LLM的进化搜索框架（如FunSearch、AlphaEvolve）将大语言模型（LLM）局限于**单轮候选生成**角色，受限于固定流程（pipeline），无法主动查阅资料、调试代码、分析反馈或迭代优化策略。这种模式在面对高度手工调优的系统（如GPU kernel）时，难以实现深层次、多步骤的工程优化。

### 提出的新方法：Agentic Variation Operators (AVO)
论文提出 **Agentic Variation Operators (AVO)** ——一种全新的进化变异算子范式，其核心思想是：
- 将传统的 `Sample → Generate → Evaluate` 固定流水线，替换为一个**自主编码智能体（autonomous coding agent）** 的闭环循环。
- 该agent作为完整的 **variation operator**，统一承担采样、生成、测试、诊断、修复和验证等全部职责。
- AVO agent具备以下能力：
  - 访问历史解决方案（lineage）
  - 查询领域知识库（KC，含CUDA指南、PTX文档、硬件架构说明等）
  - 调用编译器、profiler、测试工具
  - 持久记忆（persistent memory）积累经验
  - 自主规划、实施、调试和验证修改

### 相比现有方法的优势
| 维度 | 传统LLM-in-the-loop方法 | AVO |
|------|------------------------|-----|
| 角色定位 | LLM仅作为Generate模块 | Agent作为完整Vary操作符 |
| 决策权 | 框架控制流程 | Agent全权主导探索路径 |
| 反馈机制 | 单次输出即提交 | 多轮内部edit-evaluate-diagnose循环 |
| 探索深度 | 表层代码变换为主 | 支持微架构级推理（register allocation, pipeline scheduling等） |
| 泛化能力 | 针对特定任务定制 | 通用coding agent + 领域知识注入 |

> ✅ **本质突破**：从“LLM辅助生成”升级为“Agent驱动演化”，实现了真正意义上的**自主持续优化**。

---

## 2. 核心实验方法和设置

### 实验目标
在NVIDIA Blackwell B200 GPU上，对multi-head attention (MHA) 和 grouped-query attention (GQA) kernel进行全自动性能优化，挑战当前最先进的人工优化实现。

### 数据集 / 测试配置
- **任务**：前向传播prefill阶段的attention计算
- **精度**：BF16
- **头维度**：128
- **序列长度**：{4K, 8K, 16K, 32K}
- **总token数固定为32K**，通过调整batch size实现（例如：seq=4K → bs=8；seq=32K → bs=1）
- **MHA配置**：16 query heads
- **GQA配置**：
  - Group=8: 32 query heads, 4 KV heads
  - Group=4: 32 query heads, 8 KV heads

### 评估指标
- **Throughput (TFLOPS)**：主要性能指标
- **Numerical Correctness**：与参考实现对比数值一致性，失败则得分为0
- **Geomean TFLOPS**：跨配置平均性能

### 基线方法对比
| 基线 | 类型 | 版本 |
|------|------|------|
| **cuDNN** | NVIDIA闭源优化kernel | v9.19.1（Blackwell专用优化） |
| **FlashAttention-4 (FA4)** | 开源SOTA attention kernel | 官方commit `71bf77c` |

### 实验设置细节
- **Agent**：基于前沿LLM构建的通用coding agent，支持自主编辑代码、执行shell命令、文件导航、文档检索。
- **Knowledge Base K** 包括：
  - CUDA编程指南
  - PTX ISA手册
  - Blackwell架构规格
  - FA4源码
- **评分函数 f(x)** = (correctness, throughput)，错误实现直接淘汰
- **运行时间**：连续7天无人干预自主演化
- **版本管理**：每次成功改进以git commit形式持久化

---

## 3. 主要实验结果和性能指标

### 关键性能数据（MHA）

#### 在因果掩码（causal=True）下：
| 方法 | 最高提升幅度 |
|------|-------------|
| vs cuDNN | **+3.5%** |
| vs FA4 | **+10.5%** |

#### 在非因果掩码（causal=False）下：
| 方法 | 最高提升幅度 |
|------|-------------|
| vs cuDNN | +2.4% （长序列）|
| vs FA4 | +3.9% （相对FA4论文报告值）|

> 🔥 达到最高 **1668 TFLOPS**（BF16, causal=False）

### GQA迁移性能（仅需30分钟自主适配）
- AVO agent将MHA kernel自动迁移到GQA，无需人工指导
- 结果显示优化具有强泛化性：

| 场景 | vs cuDNN | vs FA4 |
|------|--------|-------|
| Causal GQA | **+7.0%** | **+9.3%** |
| Non-causal GQA | +6.0% | +4.5% |

### 消融实验与关键优化分析（见Table 1）

| 优化项 | 版本跨度 | 非因果增益 | 因果增益 | 技术要点 |
|--------|----------|------------|-----------|---------|
| **Branchless accumulator rescaling** | v19→v20 | **+8.1%** | +1.6% | 消除条件分支，改用predicated select + 更轻量memory fence |
| **Correction/MMA pipeline overlap** | v29→v30 | +1.1% | +0.4% | 允许correction warp与第二stage PV GEMM重叠执行 |
| **Register rebalancing across warp groups** | v32→v33 | +2.1% | ~0% | 从softmax组转移寄存器给correction组，减少local memory spill |

> 📊 总共产生 **40个提交版本**，探索超过 **500个优化方向**，远超人类工程师同期可完成的工作量。

---

## 4. 关键结论和发现

### 主要发现
1. **AVO能实现专家级微架构优化**  
   agent发现的优化涉及register allocation、instruction pipeline scheduling、workload distribution等多个硬件层级，表明其具备真实的**硬件级推理能力**，而非表面代码改写。

2. **优化具有良好的可迁移性**  
   在MHA上发现的技术可快速迁移到GQA，仅需约30分钟自主适应，说明AVO学到的是**通用优化原则**而非常规pattern匹配。

3. **进化过程呈现典型工程演进特征**  
   - 初期：大幅跳跃式改进（closing coarse-grained gaps）
   - 后期：细粒度微调（fine-grained squeezing）
   - 存在明显plateau期，依赖supervisor机制跳出局部最优

4. **超越已有LLM增强进化框架**  
   相比FunSearch/AlphaEvolve等，AVO不再受限于预设workflow，展现出更强的**自主探索与纠错能力**。

### 方法的局限性
- 当前研究采用single-lineage模式，未探索population-level branching（如island模型），可能限制多样性。
- 严重依赖高质量的知识库（KC）和精确的evaluation feedback，若profiling不准可能导致误导向。
- 对极端corner case的处理仍存在风险（如numerical stability），需进一步强化verification机制。
- 成本较高：长时间运行需要稳定算力支持和资源调度保障。

### 未来工作方向
1. 扩展至其他高性能kernel（e.g., convolution, MLP, routing algorithms）
2. 构建多agent协作的evolutionary ecosystem（多个specialized agents竞争/合作）
3. 引入learned reward modeling或test-time training进一步提升agent策略学习能力
4. 探索AVO在不同硬件平台（AMD, Intel, ASIC）上的可移植性
5. 结合formal verification提升correctness保证强度

---

> ✅ **总体评价**：  
> AVO标志着**AI-driven systems optimization**进入新阶段——从“辅助编程”迈向“自主工程”。它不仅在attention kernel上刷新SOTA，更重要的是提供了一个通用框架，有望应用于各类需长期迭代优化的关键软件系统。

</details>

---

### 16. [DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving](https://arxiv.org/abs/2603.24587)

**Authors**: Pengxuan Yang, Yupeng Zheng, Deheng Qian, Zebin Xing, Qichao Zhang, Linbo Wang, Yichen Zhang, Shaoyu Guo, Zhongpu Xia, Qiang Chen, Junyu Han, Lingyun Xu, Yifeng Pan, Dongbin Zhao  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.24587v1  

#### Abstract
We introduce DreamerAD, the first latent world model framework that enables efficient reinforcement learning for autonomous driving by compressing diffusion sampling from 100 steps to 1 - achieving 80x speedup while maintaining visual interpretability. Training RL policies on real-world driving data...

---

### 17. [Semantic Centroids and Hierarchical Density-Based Clustering for Cross-Document Software Coreference Resolution](https://arxiv.org/abs/2603.24246)

**Authors**: Julia Matela, Frank Kr\"uger  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.24246v1  

#### Abstract
This paper describes the system submitted to the SOMD 2026 Shared Task for Cross-Document Coreference Resolution (CDCR) of software mentions. Our approach addresses the challenge of identifying and clustering inconsistent software mentions across scientific corpora. We propose a hybrid framework tha...

---

### 18. [Unveiling Hidden Convexity in Deep Learning: a Sparse Signal Processing Perspective](https://arxiv.org/abs/2603.23831)

**Authors**: Emi Zeger, Mert Pilanci  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.23831v1  

#### Abstract
Deep neural networks (DNNs), particularly those using Rectified Linear Unit (ReLU) activation functions, have achieved remarkable success across diverse machine learning tasks, including image recognition, audio processing, and language modeling. Despite this success, the non-convex nature of DNN lo...

---

### 19. [Mixed-signal implementation of feedback-control optimizer for single-layer Spiking Neural Networks](https://arxiv.org/abs/2603.24113)

**Authors**: Jonathan Haag, Christian Metzner, Dmitrii Zendrikov, Giacomo Indiveri, Benjamin Grewe, Chiara De Luca, Matteo Saponati  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.24113v1  

#### Abstract
On-chip learning is key to scalable and adaptive neuromorphic systems, yet existing training methods are either difficult to implement in hardware or overly restrictive. However, recent studies show that feedback-control optimizers can enable expressive, on-chip training of neuromorphic devices. In ...

---

### 20. [The DeepXube Software Package for Solving Pathfinding Problems with Learned Heuristic Functions and Search](https://arxiv.org/abs/2603.23873)

**Authors**: Forest Agostinelli  
**Category**: cs.AI  
**Published**: 2026-03-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.23873v1  

#### Abstract
DeepXube is a free and open-source Python package and command-line tool that seeks to automate the solution of pathfinding problems by using machine learning to learn heuristic functions that guide heuristic search algorithms tailored to deep neural networks (DNNs). DeepXube is comprised of the late...

---

### 21. [The Compression Paradox in LLM Inference: Provider-Dependent Energy Effects of Prompt Compression](https://arxiv.org/abs/2603.23528)

**Authors**: Warren Johnson  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.23528v1  

#### Abstract
The rapid proliferation of Large Language Models has created an environmental paradox: the very technology that could help solve climate challenges is itself becoming a significant contributor to global carbon emissions. We test whether prompt compression improves inference energy efficiency in 28,4...

---

### 22. [Stochastic Dimension-Free Zeroth-Order Estimator for High-Dimensional and High-Order PINNs](https://arxiv.org/abs/2603.24002)

**Authors**: Zhangyong Liang, Ji Zhang  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.24002v1  

#### Abstract
Physics-Informed Neural Networks (PINNs) for high-dimensional and high-order partial differential equations (PDEs) are primarily constrained by the $\mathcal{O}(d^k)$ spatial derivative complexity and the $\mathcal{O}(P)$ memory overhead of backpropagation (BP). While randomized spatial estimators s...

---

### 23. [Linear-Nonlinear Fusion Neural Operator for Partial Differential Equations](https://arxiv.org/abs/2603.24143)

**Authors**: Heng Wu, Junjie Wang, Benzhuo Lu  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.24143v1  

#### Abstract
Neural operator learning directly constructs the mapping relationship from the equation parameter space to the solution space, enabling efficient direct inference in practical applications without the need for repeated solution of partial differential equations (PDEs) - an advantage that is difficul...

---

### 24. [TsetlinWiSARD: On-Chip Training of Weightless Neural Networks using Tsetlin Automata on FPGAs](https://arxiv.org/abs/2603.24186)

**Authors**: Shengyu Duan, Marcos L. L. Sartori, Rishad Shafik, Alex Yakovlev  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.24186v1  

#### Abstract
Increasing demands for adaptability, privacy, and security at the edge have persistently pushed the frontiers for a new generation of machine learning (ML) algorithms with training and inference capabilities on-chip. Weightless Neural Network (WNN) is such an algorithm that is principled on lookup t...

---

### 25. [Efficient Benchmarking of AI Agents](https://arxiv.org/abs/2603.23749)

**Authors**: Franck Ndzomga  
**Category**: cs.AI  
**Published**: 2026-03-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.23749v1  

#### Abstract
Evaluating AI agents on comprehensive benchmarks is expensive because each evaluation requires interactive rollouts with tool use and multi-step reasoning. We study whether small task subsets can preserve agent rankings at substantially lower cost. Unlike static language model benchmarks, agent eval...

---

### 26. [Fast and Faithful: Real-Time Verification for Long-Document Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2603.23508)

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Haichen Zhang, Huamin Chen  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.23508v1  

#### Abstract
Retrieval-augmented generation (RAG) is increasingly deployed in enterprise search and document-centric assistants, where responses must be grounded in long and complex source materials. In practice, verifying that generated answers faithfully reflect retrieved documents is difficult: large language...

---

### 27. [Grounding Arabic LLMs in the Doha Historical Dictionary: Retrieval-Augmented Understanding of Quran and Hadith](https://arxiv.org/abs/2603.23972)

**Authors**: Somaya Eltanbouly, Samer Rashwani  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.23972v1  

#### Abstract
Large language models (LLMs) have achieved remarkable progress in many language tasks, yet they continue to struggle with complex historical and religious Arabic texts such as the Quran and Hadith. To address this limitation, we develop a retrieval-augmented generation (RAG) framework grounded in di...

---

### 28. [FinToolSyn: A forward synthesis Framework for Financial Tool-Use Dialogue Data with Dynamic Tool Retrieval](https://arxiv.org/abs/2603.24051)

**Authors**: Caishuang Huang, Yang Qiao, Rongyu Zhang, Junjie Ye, Pu Lu, Wenxi Wu, Meng Zhou, Xiku Du, Tao Gui, Qi Zhang, Xuanjing Huang  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.24051v1  

#### Abstract
Tool-use capabilities are vital for Large Language Models (LLMs) in finance, a domain characterized by massive investment targets and data-intensive inquiries. However, existing data synthesis methods typically rely on a reverse synthesis paradigm, generating user queries from pre-sampled tools. Thi...

---

### 29. [Alignment Reduces Expressed but Not Encoded Gender Bias: A Unified Framework and Study](https://arxiv.org/abs/2603.24125)

**Authors**: Nour Bouchouchi, Thiabult Laugel, Xavier Renard, Christophe Marsala, Marie-Jeanne Lesot, Marcin Detyniecki  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.24125v1  

#### Abstract
During training, Large Language Models (LLMs) learn social regularities that can lead to gender bias in downstream applications. Most mitigation efforts focus on reducing bias in generated outputs, typically evaluated on structured benchmarks, which raises two concerns: output-level evaluation does ...

---

### 30. [Causal Reconstruction of Sentiment Signals from Sparse News Data](https://arxiv.org/abs/2603.23568)

**Authors**: Stefania Stan, Marzio Lunghi, Vito Vargetto, Claudio Ricci, Rolands Repetto, Brayden Leo, Shao-Hong Gan  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.23568v1  

#### Abstract
Sentiment signals derived from sparse news are commonly used in financial analysis and technology monitoring, yet transforming raw article-level observations into reliable temporal series remains a largely unsolved engineering problem. Rather than treating this as a classification challenge, we prop...

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
