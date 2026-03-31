# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-31 07:00:03 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [OptINC: Optical In-Network-Computing for Scalable Distributed Learning](https://arxiv.org/abs/2603.28290)

**Authors**: Sijie Fei, Grace Li Zhang, Bing Li, Ulf Schlichtmann  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.28290v1  

#### Abstract
Distributed learning is widely used for training large models on large datasets by distributing parts of the model or dataset across multiple devices and aggregating the computed results for subsequent computations or parameter updates. Existing communication algorithms for distributed learning such...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：OptINC: Optical In-Network-Computing for Scalable Distributed Learning**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
在大规模分布式深度学习中，**通信开销**是训练效率的主要瓶颈。传统的 **ring all-reduce** 算法虽然广泛使用，但由于需要多轮通信（Reduce-Scatter 和 All-Gather），导致通信量接近理论最优值的两倍（相对开销接近 100%）。此外，现有 In-Network Computing（INC）方案依赖电学交换机，存在 **O-E-O 转换带来的延迟和能耗问题**。

### 🔧 提出的新方法与创新思路
本文提出 **OptINC（Optical In-Network-Computing）架构**，将梯度聚合计算从服务器卸载到 **光互连网络** 中，实现全光学域内的梯度平均与量化操作。其核心创新包括：

- **基于 ONN（Optical Neural Network）的光内计算**  
  利用由 **MZI（Mach-Zehnder Interferometer）阵列** 构成的 ONN，在光域直接执行梯度平均与非线性量化，避免传统多轮通信。
  
- **预处理与信号分割机制**  
  引入 **Preprocessing Unit (P)** 和 **Splitting Unit (T)**，降低 ONN 输入维度，缓解因服务器数量增加导致的数据复杂度指数增长问题。

- **硬件成本优化设计**  
  对 ONN 中的权重矩阵采用 **unitary-diagonal 近似分解（$W_s \approx \Sigma_a U_a$）**，减少所需 MZI 数量，硬件面积降低近 50%。

- **硬件感知训练算法（Hardware-Aware Training）**  
  在训练过程中周期性应用矩阵近似，并通过两阶段损失函数优化重建精度，确保即使在硬件约束下仍能保持高准确率。

### 🆚 相比现有方法的优势
| 维度 | 传统方法（如 ring all-reduce） | OptINC |
|------|-------------------------------|--------|
| 通信轮数 | $2(N-1)$ 轮 | **仅需 1 轮（信号穿透即完成）** |
| 通信开销 | 高达 ~87.5% 冗余传输 | **完全消除冗余通信** |
| 能耗与延迟 | 存在 O-E-O 转换开销 | **纯光路径，无 O-E-O 转换** |
| 可扩展性 | 多轮同步限制扩展性 | 支持级联拓扑，可扩展至更多服务器 |

---

## 2. **核心实验方法和设置**

### 📚 数据集与模型任务
实验验证了 OptINC 在真实分布式训练任务中的有效性：
- **CIFAR-100 + ResNet50**：图像分类任务，训练 300 轮。
- **Wikipedia-1B + LLaMA-based network**：语言建模任务，8 层 Transformer，隐藏维度 384，8 注意力头，训练 50,000 步。

梯度使用 **block quantization** 映射为固定点格式（如 8-bit 或 16-bit），并通过 **PAM4 编码**进行光信号传输。

### ⚙️ 实验设置
- **硬件假设**：
  - 使用 **NVIDIA H100 GPU**（算力 60 TFLOPs，效率 0.6）
  - 每台服务器配备 **8 个全双工光收发器**，带宽 **800 Gb/s**
  - 光学器件基于 **MZI 阵列** 实现线性变换，非线性激活参考 [31] 在光域实现
- **ONN 结构**：
  - 使用 **MLP** 架构（ReLU 激活）
  - 输入为编码后的 PAM4 信号组合，输出为全局平均梯度的 PAM4 表示
- **评估指标**：
  - **训练准确率 / Loss**
  - **端到端训练延迟（归一化）**
  - **通信数据量（归一化）**
  - **ONN 硬件面积（以 MZI 数量衡量）**
  - **误差引入概率与幅度（消融分析）**

### 🔄 基线方法对比
- **Ring All-Reduce**：标准数据并行通信协议，作为主要性能基线。
- **理想无误差 OptINC** vs **含误差注入的 OptINC**：用于评估硬件近似对最终模型性能的影响。

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据

#### （1）通信开销对比（图6）
| 服务器数 | Ring All-Reduce 通信开销（相对于理论最小） | OptINC 开销 |
|---------|------------------------------------------|------------|
| 4       | ~50%                                     | **0%**     |
| 8       | ~75%                                     | **0%**     |
| 16      | ~87.5%                                   | **0%**     |

👉 **结论**：OptINC 完全消除了 ring all-reduce 所需的额外通信轮次。

#### （2）硬件成本降低（表 I & II）
| 场景（bit/server） | 是否启用矩阵近似 | Area Ratio（相对原始 ONN） | ONN Accuracy |
|------------------|------------------|----------------------------|--------------|
| 8-bit / 4-server | 否               | 100%                       | 100%         |
|                  | 是               | **39.3%**                  | 100%         |
| 16-bit / 4-server| 是（Layer 4–6）  | **49.3%**                  | 100%         |

👉 **最大面积节省达 60.7%**，且通过硬件感知训练可维持 **100% 功能等效性**。

#### （3）模型训练表现（图7a）
| 模型 | 条件 | 准确率 / Loss 变化 |
|------|------|--------------------|
| ResNet50 | 无误差注入 | ↓0.03%（vs baseline） |
|           | 有误差注入（来自 Table II） | ↓0.55% |
| LLaMA-based NN | 无误差注入 | ↑0.018 loss |
|                  | 有误差注入 | ↑0.038 loss |

👉 尽管存在低概率误差（如 ±1 占 90%，±4 占 79.5%），但整体模型性能仍在可接受范围内。

#### （4）延迟改善（图7b）
| 模型 | Ring All-Reduce 延迟占比 | OptINC 总体延迟下降 |
|------|--------------------------|---------------------|
| ResNet50（CNN） | 通信主导 | **>25% 加速** |
| LLaMA-based（Transformer） | 计算与通信均衡 | **~17% 加速** |

👉 在通信密集型场景中加速更显著。

#### （5）可扩展性验证（级联拓扑）
- 使用 **两级级联 OptINC**（5 个单元），支持最多 **16 台服务器**。
- 修改 ONN 插入两个 64×64 近似层，适应更高分辨率输入。
- **硬件开销增加约 10.5%**，但可在修改后数据集上达到 **100% ONN 准确率**。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **OptINC 成功将梯度平均与量化操作迁移至光网络内部**，实现了真正的 **zero-overhead communication**。
2. 利用 **MZI-based ONN** 可高效实现非线性映射，结合硬件近似与感知训练，能在大幅降低硬件成本的同时保持功能一致性。
3. **通信不再是训练瓶颈**，尤其在 CNN 类模型中，OptINC 可带来超过 25% 的端到端加速。
4. 通过 **级联架构**，OptINC 可灵活扩展至更大规模系统，具备良好的工程部署潜力。

### ⚠️ 方法的局限性
- **当前未考虑物理层非理想因素**：如 MZI 相位漂移、温度波动、制造偏差等，可能影响实际部署稳定性。
- **ONN 训练依赖高质量标定数据集**：随着 $N$ 增大，即使有预处理，训练数据生成仍具挑战。
- **仅适用于固定通信模式的任务**：动态拓扑或频繁变化的通信需求难以适配（受限于 OCS 重配置延迟）。

### 🔮 未来工作方向
- 探索 **抗干扰鲁棒训练方法**，应对光学器件的工艺与热变异。
- 引入 **Neural Architecture Search (NAS)** 自动优化 ONN 结构。
- 研究其他 **光网络拓扑与协议**（如 fat-tree、all-to-all）下的通用性。
- 集成 **光电混合控制机制**，提升系统的灵活性与容错能力。

---

> 💡 **总结一句话**：  
> OptINC 通过构建一个基于 MZI 的光神经网络，在无需额外通信轮次的前提下完成了分布式训练中的梯度聚合，**首次实现了“通信零开销”的 scalable distributed learning 架构**，为下一代 AI 集群提供了全新的光子级解决方案。

</details>

---

### 2. [Bitboard version of Tetris AI](https://arxiv.org/abs/2603.26765)

**Authors**: Xingguo Chen, Pingshou Xiong, Zhenyu Luo, Mengfei Hu, Xinwen Li, Yongzhou L\"u, Guang Yang, Chao Li, Shangdong Yang  
**Category**: cs.AI  
**Published**: 2026-03-31  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.26765v1  

#### Abstract
The efficiency of game engines and policy optimization algorithms is crucial for training reinforcement learning (RL) agents in complex sequential decision-making tasks, such as Tetris. Existing Tetris implementations suffer from low simulation speeds, suboptimal state evaluation, and inefficient tr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Bitboard version of Tetris AI》论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有 Tetris 游戏实现（如 OpenAI Gym-Tetris）存在以下瓶颈：
- **模拟速度慢**：传统基于网格（grid-based）的实现无法高效执行碰撞检测、消行等操作，限制了大规模强化学习（RL）训练。
- **策略优化效率低**：主流方法依赖复杂的手工特征（如 Bertsekas features）或轨迹式训练范式，样本利用率低，训练耗时长。
- **缺乏高性能与易用性的统一**：高效实现多在底层语言中完成，难以与现代 Python RL 框架（如 PyTorch/TensorFlow）集成。

### 提出的新方法与创新思路
本文提出一个高性能的 Tetris AI 框架，结合 **bitboard 表示法** 与 **改进的 RL 算法设计**，核心贡献如下：

#### （1）Bitboard-Based Tetris 实现
- 将游戏板每列用一个 32 位整数表示（bitboard），利用 **bitwise operations** 加速核心操作：
  - 碰撞检测（`AND` 操作）
  - 消行判断（`AND` 所有列）
  - Afterstate 构建与 DT Features 提取
- 在 Java 中实现以最大化位运算性能，并通过 **Jpype** 提供符合 OpenAI Gym 接口的 Python 封装。

> ✅ **优势**：相比 OpenAI Gym-Tetris，**运行速度快 53 倍**（Java 实现下仅需 0.24 秒处理 10,000 步）。

#### （2）Afterstate-Evaluating Actor 网络
- 利用 Tetris 的 **afterstate 属性**（动作后、新块生成前的状态）构建状态价值函数 $ V(\text{as}) $，而非传统的 action-value 函数 $ Q(s,a) $。
- Actor 网络直接评估所有可行 afterstate 的 DT features，输出动作概率分布。

> ✅ **优势**：
> - 更少参数（输入维度从 48 降至 9）
> - 更高样本效率
> - 更稳定的学习过程（解耦决策确定性与环境随机性）

#### （3）Buffer-Optimized PPO 算法
- 改进标准的 trajectory-based PPO：
  - 不等待完整 episode 结束，而是当 replay buffer 积累到 `batchSize` 样本即开始更新。
  - 引入线性学习率衰减，提升收敛稳定性。
- 实现采样与更新时间的平衡，显著提高训练吞吐量。

> ✅ **优势**：
> - 训练步数减少至 **61,440 步（约 3 分钟）**
> - 更新占比从 3.97% 提升至 32.53%，资源利用更均衡

#### （4）OpenAI Gym 兼容接口
- 提供 Python-Java 接口（基于 Jpype），支持无缝接入主流 DRL 框架。
- 支持并行测试（`parallel_episode()`）、特征提取（`get_9feature()`）等功能。

---

## 2. 核心实验方法和设置

### 数据集与环境
- **游戏设置**：
  - 主要实验使用 **10×10 mini-board**（当前研究常用设定，兼顾挑战性与训练效率）
  - 对比实验也验证了在标准 **10×20 board** 上的表现
- **块生成器**：
  - 默认使用 **random generator**（i.i.d. 抽取 7 类方块）
  - 泛化性测试涵盖 **7-Bag** 和 **adversarial Z/S sequence**

### 实验设置
- **硬件平台**：AMD R7-7735H CPU, 16GB RAM, Windows 11
- **评估指标**：
  - 平均得分（Average Score / Removed Lines）
  - 训练时间与总交互步数（Total Steps）
  - 收敛稳定性（Mean ± SD over multiple runs）
- **训练配置**：
  - 使用 **DT features** 作为状态表示
  - 所有模型均采用线性层避免网络结构偏差
  - Discount factor $ \gamma = 0.99 $, GAE 参数 $ \lambda = 0.99 $

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **OpenAI Gym-Tetris** | 环境基线 | 基于 NES 模拟器，性能低下 |
| **CBMPI** | RL 方法 | 使用 DT features + RBF heights，需 8M 样本 |
| **dSiLU-TD(λ)** | RL 方法 | 使用神经网络 + Bertsekas features，最高分 4,900 |
| **STEW** | RL 方法 | 使用 7-Bag generator，泛化性受限 |
| **Trajectory-based PPO** | 算法基线 | 完整 episode 后更新，效率低 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 本文方法（Buffer PPO） | Trajectory PPO | 最优方法（dSiLU-TD(λ)） |
|------|------------------------|----------------|----------------------------|
| **平均得分（10×10）** | **3,829.04** | 3,840.30 | 4,900 |
| **训练步数** | **61,440** | 69,046,726 | 200,000 |
| **训练时间** | **~3 分钟** | ~3 小时 | 数小时以上 |
| **样本效率** | 极高（提升约 1124 倍） | 低 | 中等 |
| **环境速度（10k steps）** | 0.24s (Java), 0.59s (Python) | 12.92s | — |

> ⚠️ 注：虽然绝对分数略低于 SOTA 方法，但在极低资源消耗下达到接近水平，工程意义重大。

### 与其他方法对比结果

#### （1）环境效率对比（Table 4）
| 实现方式 | 处理 10,000 步耗时 |
|---------|------------------|
| OpenAI Gym-Tetris | 12.92 秒 |
| Bitboard Tetris (Java) | **0.24 秒（快 53×）** |
| Bitboard Tetris (Python via Jpype) | **0.59 秒（快 22×）** |

#### （2）算法性能对比（Table 8）
| 方法 | 总步数 | 平均得分 |
|------|-------|----------|
| Trajectory PPO | 69M | 3,840.30 |
| **Buffer PPO（本文）** | **61,440** | **3,829.04** |

> ➤ 节省超过 **1100 倍训练步数**，性能几乎持平。

#### （3）泛化能力测试（Table 11）
| 块生成规则 | CBMPI | Trajectory PPO | Buffer PPO |
|----------|--------|----------------|------------|
| Random | 4,300 | 4,965 | **3,591** |
| 7-Bag | 251,501 | 173,489 | **198,206** |
| Adversarial Z/S | 75 | 34 | **53** |

> ➤ 所有方法在对抗序列下表现差，说明鲁棒性仍是开放问题；但本文方法在 7-Bag 下表现最佳。

### 消融实验结果
#### （1）Afterstate vs. Action-Value Actor（Figure 12）
- Afterstate Actor 使用 **更少参数**（9D 输入 vs. 48D）
- 学习曲线更平滑，最终性能更高
- 验证了 afterstate 设计能有效降低估计方差，提升策略稳定性

#### （2）Buffer 机制对效率的影响（Table 7）
| 方法 | 采样时间 | 更新时间 | 总时间 | 更新占比 |
|------|--------|--------|------|--------|
| Trajectory PPO | 10,536s | 436s | 10,972s | 3.97% |
| Buffer PPO | 112s | 54s | **166s** | **32.53%** |

> ➤ 训练总时间缩短 **66 倍**，更新效率大幅提升。

---

## 4. 关键结论和发现

### 主要发现
1. **Bitboard 显著加速 Tetris 模拟**：
   - 利用 bitwise operations 可将核心逻辑提速数十倍，为大规模 RL 提供高效环境基础。
2. **Afterstate 是更优的价值评估路径**：
   - 解耦动作后果与环境随机性，使 Actor 能专注于 board configuration quality，提升学习效率。
3. **Buffer-based 更新优于 Trajectory-based**：
   - 在长 episode 游戏中，及时利用高质量 mid-game 样本能极大提升样本利用率。
4. **低资源下可实现竞争性性能**：
   - 仅用 **6 万步、3 分钟训练** 即可在 10×10 上达到近 3,830 分，适合快速原型开发与算法验证。

### 方法的局限性
- **未追求绝对最高分**：目标是“高效训练”，非打破 SOTA 分数记录。
- **跨尺寸泛化有限**：在 10×20 板上性能下降明显（Table 10），需针对性调优。
- **对 adversarial 序列敏感**：所有方法在连续 Z/S 块下迅速失败，鲁棒性不足。
- **依赖手工特征 DT features**：尚未完全发挥深度表示潜力。

### 未来工作方向
1. **融合深度特征**：
   - 探索将 DT features 与 CNN/Transformer 提取的 latent features 融合，提升状态表征能力。
2. **优化网络结构**：
   - 引入 Attention 或 Graph Networks 建模局部空间关系。
3. **增强鲁棒性训练**：
   - 在 adversarial block sequences 下进行对抗训练或 curriculum learning。
4. **扩展至其他游戏**：
   - 将 bitboard + afterstate + buffer PPO 范式推广至其他经典游戏（如 Columns、Puyo Puyo）。

---

> 🔗 **代码开源地址**：[https://github.com/GameAI-NJUPT/BitboardTetris](https://github.com/GameAI-NJUPT/BitboardTetris)（MIT License）

</details>

---

### 3. [GeoBlock: Inferring Block Granularity from Dependency Geometry in Diffusion Language Models](https://arxiv.org/abs/2603.26675)

**Authors**: Lipeng Wan, Junjie Ma, Jianhui Gu, Zeyang Liu, Xuyang Lu, Xuguang Lan  
**Category**: cs.CL  
**Published**: 2026-03-31  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.26675v1  

#### Abstract
Block diffusion enables efficient parallel refinement in diffusion language models, but its decoding behavior depends critically on block size. Existing block-sizing strategies rely on fixed rules or heuristic signals and do not account for the dependency geometry that determines which tokens can be...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GeoBlock: Inferring Block Granularity from Dependency Geometry in Diffusion Language Models**

---

## **1. 主要贡献和创新点**

### **解决的问题**
在 **diffusion language models (DLMs)** 中，**block diffusion** 允许并行化地对多个 token 进行联合去噪更新，从而提升解码效率。然而，其性能高度依赖于 **block size** 的选择：
- **小 block** 限制了并行性，收敛慢；
- **大 block** 可能同时更新不稳定的 token，导致过早或不一致的 refinement。

现有方法通常采用**固定规则**或基于 **token-level confidence、entropy、volatility** 等启发式信号来动态调整 block 大小，但这些信号仅反映局部不确定性，并未考虑 token 之间的**依赖结构（dependency structure）**。

因此，核心问题是：  
> 如何根据当前生成状态下的**依赖几何结构（dependency geometry）**，自适应地确定稳定且高效的 block 边界？

---

### **提出的新方法与新思路**
作者提出了 **GeoBlock**，一种**无需训练**、**基于注意力机制推断依赖结构**的 block 粒度推理框架。

#### **核心思想**
- 将 block 选择视为一个**结构性推理问题**，而非简单的超参数调节。
- 利用 **self-attention 矩阵**作为模型内部依赖关系的可观测代理（proxy），分析 token 间的依赖模式。
- 定义“**几何上自洽的 refinement 区域**”：内部耦合强、历史锚定强、对未来依赖弱。

#### **关键技术**
- **Closure Score**：量化候选 block 的依赖闭合程度：
  $$
  \text{Score}(x) = \frac{S_{C\to C} + \alpha S_{C\to H}}{S_{C\to C} + \alpha S_{C\to H} + S_{C\to F}}
  $$
  - $S_{C\to C}$：块内耦合强度（internal coupling）
  - $S_{C\to H}$：对已生成前缀的历史锚定（past anchoring）
  - $S_{C\to F}$：对未生成未来的依赖泄漏（future leakage）
  - $\alpha$：平衡系数

- **Right-Shift Rule**：在所有得分接近最大值的候选边界中，选择最靠右的一个，以最大化并行性同时保持稳定性。

- **多层注意力融合**：聚合多个 transformer 层中的注意力头输出，形成统一的依赖估计。

---

### **相比现有方法的优势**
| 维度 | 现有方法（如 AdaBlock） | GeoBlock |
|------|------------------------|---------|
| 决策依据 | Token-level 不确定性（confidence, entropy） | **Token 间依赖结构**（structural geometry） |
| 是否需要训练 | 否（多数为 heuristic） | ❌ 完全无需训练 |
| 动态适应性 | 基于局部置信度 | 基于全局依赖格局演化 |
| 并行效率 vs. 稳定性 | 难以兼顾 | 显式权衡，自动适应 |
| 通用性 | 特定模型设计 | 可无缝集成到任意 block diffusion 架构 |

> ✅ **优势总结**：GeoBlock 将 block 选择从“启发式调度”提升为“结构感知推理”，实现了更可靠、更高效的并行去噪。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在多个标准 benchmark 上进行评估，涵盖不同任务类型：
- **数学推理**：GSM8K, MATH
- **指令遵循**：IFEval
- **代码生成**：HumanEval, MBPP

---

### **实验设置与评估指标**

#### **Backbones**
- **Dream-7B** 和 **LLaDA-8B**：代表性的 diffusion language models，支持从头训练或由 autoregressive 模型适配而来。
- 所有方法复用相同预训练 backbone，**无重训练或微调**。

#### **Decoding Framework**
- 所有方法运行在相同的 **block-diffusion 解码流程**中。
- GeoBlock 替换原有的 block scheduler，**不修改模型参数或训练过程**。
- 使用 **Fast-dLLM** 的默认配置确保公平比较。
- 统一使用 **KV cache** 策略控制计算开销。

#### **评估指标**
| 指标 | 说明 |
|------|------|
| **Accuracy (Acc)** | 数学与指令任务的答案准确率 |
| **pass@1** | 代码生成任务的成功率 |
| **Number of Function Evaluations (NFE)** | 模型前向传播次数，衡量解码效率 |
| **Wall-clock throughput** | 实际运行时间（补充） |

---

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Vanilla Block** | 固定 block size（如 B16/B32/B64） |
| **Dynamic Decoding** | 基于 token confidence 动态调整 block |
| **AdaBlock** | 使用语义置信度信号进行 adaptive multi-token commitment |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| Model | Task | Best Baseline Acc | GeoBlock Acc | Δ Acc | NFE (GeoBlock) |
|-------|------|------------------|--------------|--------|----------------|
| Dream-7B | GSM8K | 75.44 (AdaBlock-B32) | **77.86** | **+2.42** | 168.92 |
| Dream-7B | IFEval | 45.84 (Dynamic-B16) | **46.88** | **+1.04** | 339.52 |
| LLaDA-8B | GSM8K | 81.35 (Dynamic-B64) | **81.88** | **+0.53** | 100.05 |
| LLaDA-8B | HumanEval | 37.80 (GeoBlock & AdaBlock) | **37.80** | ≈ | 149.48 |
| LLaDA-8B | MBPP | 37.80 (AdaBlock-B16) | **40.00** | **+2.20** | 84.61 |
| LLaDA-8B | IFEval | 66.67 (AdaBlock-B16) | **66.67** | ≈ | 250.91 |

> 📌 **总体趋势**：GeoBlock 在多数任务上达到**最佳或相当精度**，尤其在中等 block 设置下表现突出。

---

### **与基线方法的对比结果**
- **优于 Vanilla 和 Dynamic 方法**：在 GSM8K 和 IFEval 上显著提升准确率。
- **优于 AdaBlock**：尽管 AdaBlock 使用复杂 confidence 机制，GeoBlock 仍能在多个任务上超越它（如 MBPP +2.2%）。
- **更高的 Pareto 效率**：在 Accuracy-NFE 曲线上，GeoBlock 多数情况下占据主导地位（见 Figure 3），即在相似甚至更低 NFE 下取得更高精度。

#### **额外开销**
- **平均增加 NFE 比例**：约 **7–15%**（推理阶段的注意力分析开销）
- **原因**：GeoBlock 引入轻量级依赖分析，但无需额外 forward pass，仅利用已有 attention map。
- **实际影响**：**总 NFE 仍远低于 vanilla block（如 169 vs 256）**，效率依然显著优于传统方法。

---

### **消融实验结果（Ablation Studies）**

#### **(1) 锚定系数 α 的影响（Table 3）**
| α | Dream-7B Acc | LLaDA-8B Acc |
|----|-------------|-------------|
| 0.0 | 76.65 | 81.19 |
| 0.25 | 76.88 | 81.35 |
| 0.5 | **76.88** | **81.88** |
| 1.0 | 76.73 | 80.82 |

> ✅ **结论**：适度引入历史锚定（$\alpha=0.5$）效果最好；完全忽略历史（$\alpha=0$）或过度强调（$\alpha=1$）均不利。

#### **(2) 右移容忍度 δ 的影响（Table 4）**
| δ | Acc | NFE | Avg Block Len |
|-----|------|------|----------------|
| 0.0 | 80.74 | 115.03 | 6.59 |
| 0.1 | **81.88** | **100.05** | **13.42** |
| 0.2 | 80.21 | 98.64 | 19.57 |

> ✅ **结论**：适度右移（$\delta=0.1$）可在保持高精度的同时有效扩大 block，提升效率；过大则导致不稳定。

#### **(3) 注意力层选择的影响（Table 5）**
- **最佳组合**：mid-to-high layers（如 16-21-26）
- **权重分配**：均匀加权（0.333, 0.333, 0.334）表现稳健
- **结论**：后期层更能捕捉稳定的语义依赖结构，GeoBlock 对具体配置不敏感，具备良好鲁棒性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **依赖几何是决定 block 粒度的关键因素**：相比 token-level 不确定性，**token 间的结构性依赖关系**更能指导稳定且高效的并行更新。
2. ✅ **GeoBlock 能自适应识别几何一致的 refinement 区域**：通过 attention-derived closure score，自动区分“需顺序处理”的因果链与“可并行优化”的语义团块。
3. ✅ **无需训练即可实现 superior 性能**：完全基于推理时 attention 分析，**零训练成本**，易于部署。
4. ✅ **在 Accuracy-NFE 权衡上全面占优**：在多个任务上实现更高精度或更少 NFE，尤其适合复杂推理任务（如 GSM8K, IFEval）。

---

### **方法的局限性**
1. **依赖 attention 的质量**：若 attention map 噪声大或无法准确反映真实依赖，则 closure score 可能失真。
2. **假设连续 block 结构**：目前仅支持 contiguous blocks，尚未扩展至 non-contiguous 或 hierarchical block grouping。
3. **对短任务增益有限**：在 MBPP 等短程序合成任务上，激进的 multi-token commitment 反而可能降低性能（见正文分析）。
4. **计算开销随窗口增大线性增长**：虽然当前开销可控，但在极长序列上可能成为瓶颈。

---

### **未来工作方向**
1. **扩展至 variable-length 与 insertion-based diffusion**：将 GeoBlock 思想应用于动态长度生成场景。
2. **结合 lookahead 或 planning 机制**：预测未来依赖演化趋势，进一步优化 block 边界。
3. **探索非连续 block grouping**：允许跳跃式或图结构的 token 分组，突破 contiguous 假设。
4. **与其他 inference acceleration 技术结合**：如与 **KV cache pruning**, **early exiting**, **sparse attention** 联合优化。

---

> 🔚 **总结一句话**：  
> **GeoBlock 通过将 block 选择从“经验驱动”转变为“结构感知”，为 diffusion language models 提供了一种高效、可靠、无需训练的自适应并行解码新范式。**

</details>

---

### 4. [An Energy-Efficient Spiking Neural Network Architecture for Predictive Insulin Delivery](https://arxiv.org/abs/2603.27589)

**Authors**: Sahil Shrivastava  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.27589v1  

#### Abstract
Diabetes mellitus affects over 537 million adults worldwide. Insulin-dependent patients require continuous glucose monitoring and precise dose calculation while operating under strict power budgets on wearable devices. This paper presents PDDS - an in-silico, software-complete research prototype of ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An Energy-Efficient Spiking Neural Network Architecture for Predictive Insulin Delivery

---

## 1. 论文的主要贡献和创新点

### 解决的问题
糖尿病患者（尤其是T1D）需要持续监测血糖并精确计算胰岛素剂量。现有的**人工胰腺系统**（如Medtronic MiniMed、Tandem Control-IQ）依赖于**连续轮询机制**（continuous polling），即每5分钟无论血糖是否变化都触发完整计算流程（包括通信和推理），导致在低功耗可穿戴设备上能耗过高，难以长期运行。

此外，传统基于规则的方法（如ADA阈值）缺乏对**时间动态模式**的建模能力，无法识别危险的“非显性”低血糖事件（如反弹性高血糖前的快速下降）。

---

### 提出的新方法与创新思路

PDDS（Predictive Drug Delivery System）提出了一种**事件驱动的Spiking Neural Network**（SNN）架构，用于预测性胰岛素给药决策。其核心创新包括：

- ✅ **事件驱动架构**（Event-Driven Pipeline）  
  推理路径仅在血糖跨越预设阈值时激活，相比连续轮询减少约 **88% 的管道激活次数**，显著降低能耗。

- ✅ **三层LIF Spiking Neural Network**（PDDSSpikingNet）  
  使用Leaky Integrate-and-Fire神经元构建SNN，输入为Poisson编码的CGM特征窗口，输出为LOW/MEDIUM/HIGH三级严重程度分类。

- ✅ **CGM滞后补偿的紧急检测器**（EmergencyDetector）  
  在每次读数后立即运行，利用最小二乘法估计斜率，并向前投影15分钟以补偿**组织间液延迟**（interstitial lag），提前预警真实血液中的低血糖风险。

- ✅ **基于严重性的Sigmoidal剂量计算器**  
  受Chou等人开发的葡萄糖响应型PBA-insulin启发，设计了一个将SNN输出直接映射到胰岛素剂量的公式，其中HIGH类别的中点左移，实现更早干预。

- ✅ **面向边缘部署的研究级训练改进**  
  引入多项SNN专用优化技术：
  - **RMaxProp优化器**：解决梯度稀疏导致的学习不稳定问题；
  - **电压基础的Eligibility Traces**：增强第一层学习信号；
  - **Synaptic Balancing正则化**：防止权重失衡；
  - **校准的Poisson噪声与轴突延迟**：提升生物合理性与鲁棒性。

- ✅ **双操作模式共享基础设施**  
  支持`DIABETIC`（闭环注射）和`PREDIABETIC`（仅通知）两种模式，共用全部算法栈，便于未来升级。

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **能效** | SNN单次推理仅需 **1,551 fJ**，比bidirectional LSTM低 **79,267倍**，适合电池供电的长期可穿戴设备 |
| **硬件兼容性** | 天然适配**neuromorphic硬件**（如Intel Loihi、SynSense Xylo），支持异步事件处理 |
| **时间泛化能力** | 能捕捉ADA规则无法识别的时间模式（如post-hypoglycemia rebound） |
| **个性化潜力** | 支持通过联邦学习进行患者特异性微调 |

---

## 2. 核心实验方法和设置

### 数据集

| 数据源 | 样本数 | 角色 |
|--------|-------|------|
| **OhioT1DM** | 85,105个窗口（12名T1D患者，8周） | 主要训练数据，含临床医生标注的`hypo_event`标签 |
| **simglucose / UVa-Padova模拟器** | 42,920个窗口（30名虚拟患者） | FDA认可的生理模拟器，补充罕见极端情况 |
| **总训练样本** | **128,025个50分钟滑动窗口**（66.5%真实 + 33.5%模拟） | Gold层特征向量 |

> ⚠️ 测试集以simglucose为主，标签由ADA规则生成，存在**循环评估偏差**

---

### 特征工程（Gold Layer）
从每个50分钟CGM窗口提取10个归一化特征：

| 特征 | 含义 |
|------|------|
| `last_glucose_norm`, `mean_glucose_norm` | 当前值与均值 |
| `min/max_glucose_norm` | 极值捕捉峰值与谷底 |
| `abs_slope_norm`, `signed_slope_norm` | 变化速率大小与方向 |
| `glucose_std_norm`, `glucose_range_norm` | 波动性与峰谷差 |
| `time_below_70_pct`, `time_above_180_pct` | 时间占比指标 |

标签采用**ADA 2023标准**，优先级如下：
1. 若有`hypo_event`标注 → 强制标记为**HIGH**
2. 血糖 <54 或 >250 mg/dL，或变化率 >3 mg/dL/min → **HIGH**
3. 边界区间（54–70 或 180–250）或中等变化率 → **MEDIUM**
4. 否则 → **LOW**

最终分布：LOW 42.63%，MEDIUM 38.98%，HIGH 18.39%

---

### 实验设置与评估指标

#### 模型对比基线
- **ADA Rule-Based Classifier**：基于固定阈值的if/else逻辑
- **Bidirectional LSTM**：序列模型代表，参数量大
- **MLP**：全连接前馈网络，作为非时序对照

所有模型输入相同10维特征（SNN使用Poisson编码，T=50步长）

#### 评估维度
| 类别 | 指标 |
|------|------|
| 分类性能 | Accuracy, Precision, Recall, F1-score（尤其关注**HIGH类召回率**） |
| 安全性 | EmergencyDetector能否有效阻断错误注射 |
| 功能正确性 | 15个集成测试场景是否通过 |
| 能效分析 | 理论能量消耗（fJ/inference）基于SynOps vs MAC估算 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 数值 |
|------|------|
| **验证准确率**（Validation Accuracy） | **85.90%**（第44轮） |
| **测试准确率**（Test Accuracy） | **85.43%** |
| **HIGH类召回率**（Primary Safety Metric） | **90.72%** |
| **模型参数量** | **9,859**（与MLP相当） |
| **训练时间**（CPU-only） | ~2.1小时（7,589秒） |

> ✅ 所有15个功能测试场景均通过（PASS）

---

### 与基线方法的对比结果

#### （1）标准测试集对比（Table XII）

| 模型 | Accuracy | HIGH Recall | 参数量 | 推理能耗 |
|------|----------|-------------|--------|-----------|
| **SNN** | 85.24% | 88.84% | 9,859 | **1,551 fJ** |
| **Bi-LSTM** | 99.06% | 99.78% | 138,627 | 122.9 nJ |
| **MLP** | 99.00% | 99.49% | 9,859 | 8.7 nJ |

> 🔍 **解读**：
> - LSTM/MLP精度接近完美是因测试集标签由ADA规则生成 → 存在**循环评估偏见**
> - SNN因**Poisson随机编码引入噪声**，牺牲部分准确性换取能效优势
> - SNN能耗仅为LSTM的 **1/79,267**，是其核心竞争力

#### （2）与ADA规则透明比较（Table I）

尽管ADA规则在自身标签上达到**100% HIGH召回率**，但这属于预期中的循环效应。实际临床上：

- **SNN能识别ADA忽略的复杂模式**（如表II所示）：
  - 患者B血糖190 mg/dL但刚经历严重低血糖反弹 → SNN判为**HIGH**，ADA误判为MEDIUM
  - SNN具备记忆与趋势理解能力，而ADA仅看当前值

---

### 消融实验与训练改进效果（Table IX）

| 实验 | 特征数 | 架构 | 优化器 | Val Acc | HIGH Recall |
|------|--------|--------|--------|---------|------------|
| Exp.1（Baseline） | 2 | 2层, 16h | Adam | 57.9% | N/A |
| **Exp.2（PDDS）** | **10** | **3层 LIF (128-64-3)** | **RMaxProp** | **85.90%** | **90.72%** |

> ✅ 四项研究驱动改进共同促成+28个百分点的提升

---

### 时间基准测试：非显性低血糖窗口（Temporal Benchmark）

选取426个“非显性”低血糖窗口（当前血糖>70 mg/dL但有`hypo_event`标注），检验模型对**时间模式**的理解能力：

| 模型 | HIGH Recall | Interpretation |
|------|--------------|----------------|
| **ADA Rule** | 16.7% | 仅靠当前值判断，表现有限 |
| **SNN** | **9.2%** | 表现更差，暴露主要短板 |

> ❗ **关键发现**：两者均未达标，说明当前模型尚未掌握这些危险边缘案例。

原因分析：
- `hypo_event`样本仅占训练集 **0.8%**，即使加权也难以形成强表示
- 当前特征工程可能丢失原始时间序列细节

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **SNN在能效上具有压倒性优势**  
   单次推理能耗低至 **1,551 fJ**，比LSTM低近 **8万倍**，使其成为**可穿戴边缘设备的理想候选架构**。

2. ✅ **事件驱动设计大幅降低激活频率**  
   阈值边沿触发机制使管道激活减少88%，进一步节省功耗。

3. ⚠️ **分类准确率低于传统模型是预期代价**  
   SNN的85.24%准确率低于LSTM的99.06%，但这是由于**Poisson编码带来的固有噪声**所致，而非架构失败。

4. ❗ **最严重的局限在于时间模式识别不足**  
   在“非显性”低血糖窗口上，SNN召回率仅9.2%，甚至低于简单规则系统，揭示了当前方法的最大瓶颈。

5. 🔁 **标准测试集存在循环评估偏差**  
   因标签来自ADA规则，导致规则系统天然占优；真正挑战在于泛化到临床注释的真实风险。

---

### 方法的局限性

| 局限 | 描述 |
|------|------|
| **硬件边界未闭合** | 输入来自文件而非真实CGM传感器，输出未连接胰岛素泵 |
| **训练数据不平衡** | `hypo_event`事件稀少（<1%），影响模型对紧急模式的学习 |
| **特征工程损失信息** | 使用手工提取的10维特征，可能削弱SNN对原始时间序列的建模能力 |
| **剂量公式简化** | 使用Sigmoid近似，未整合完整的PK/PD动力学模型 |
| **无实时安全验证** | 所有测试均为软件仿真，非临床或硬件在环测试 |

---

### 未来工作方向

1. **增强对非显性低血糖的检测能力**
   - 对`hypo_event`样本进行**专门的数据增强**
   - 设计独立的**pre-hypoglycemia descent子分类器**
   - 改用**raw CGM time series作为输入**，结合RNN-like SNN结构

2. **推进临床验证路径**
   - 执行五阶段硬件集成路线图：
     1. 已完成：软件栈验证
     2. Q2–Q3 2026：接入物理CGM（BLE/USB）
     3. Q4 2026：生理模型台架测试
     4. 2027：IRB批准下的近人体测试（prediabetic通知）
     5. 2027–2028：临床试验、联邦学习、TinyML移植至neuromorphic芯片

3. **提升剂量建模精度**
   - 引入个体化PK/PD参数拟合
   - 结合饮食、运动等多模态输入

4. **开放科学实践**
   - 开源代码、训练流水线与评估脚本，支持复现

---

> 📌 **总结一句话**：  
> PDDS不是追求最高精度的模型，而是为**超低功耗可穿戴场景量身定制的事件驱动SNN架构**——它用适度的准确率换来了**数量级的能量节约**，并在真实临床风险识别上展现出超越规则系统的潜力，但仍需突破时间模式学习的瓶颈才能迈向临床应用。

</details>

---

### 5. [K-Means Based TinyML Anomaly Detection and Distributed Model Reuse via the Distributed Internet of Learning (DIoL)](https://arxiv.org/abs/2603.27393)

**Authors**: Abdulrahman Albaiz, Fathi Amsaad  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.27393v1  

#### Abstract
This paper presents a lightweight K-Means anomaly detection model and a distributed model-sharing workflow designed for resource-constrained microcontrollers (MCUs). Using real power measurements from a mini-fridge appliance, the system performs on-device feature extraction, clustering, and threshol...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题：** *K-Means Based TinyML Anomaly Detection and Distributed Model Reuse via the Distributed Internet of Learning (DIoL)*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在资源受限的微控制器（MCU）上部署基于数据驱动的异常检测模型面临三大挑战：
- **计算开销大**：在设备端训练模型消耗大量能量和时间；
- **缺乏标准化模型复用机制**：每个设备需重复训练，即使面对相同类型的设备和相似运行环境；
- **现有分布式学习方案不适用**：如 Federated Learning 要求网络连接、服务器协调和较大内存，不适合低功耗、离线运行的 TinyML 场景。

### 🚀 提出的新方法与创新思路
本论文提出了一套完整的端到端解决方案，核心贡献包括以下三点：

#### （1）On-MCU K-Means Anomaly Detection
- 在 STM32F446RE MCU 上实现轻量级 K-Means 聚类算法，直接进行**on-device training**。
- 使用五个电力特征（RMS、rolling mean、standard deviation、RMS slope、compressor ON duration）构建多变量输入。
- 仅使用约 20% 的初始正常数据进行训练，降低训练成本并保证代表性。

#### （2）Portable MODEL.TXT 格式支持跨设备模型复用
- 将训练后的模型参数（centroids、normalization statistics、anomaly threshold）序列化为一个紧凑、可读性强的文本文件 `MODEL.TXT`。
- 支持从 microSD 卡加载，无需重新训练即可执行推理，实现“**Train Once, Share Everywhere (TOSE)**”。

#### （3）Distributed Internet of Learning (DIoL) 工作流原型
- 首次提出并实现了面向 TinyML 的 **DIoL 架构**，允许设备间通过本地存储介质共享模型。
- 展示了一个两设备原型系统：Device A 完成训练并导出模型；Device B 直接加载模型进行推理，验证了跨设备模型迁移的可行性。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| 训练方式 | 多数依赖外部训练或云端生成模型 | 全流程 on-MCU 自主训练 |
| 模型复用 | 各设备独立训练，无法共享 | 一次训练，多设备复用（TOSE） |
| 分布式学习 | Federated Learning 需要通信基础设施 | 无须联网，支持离线模型交换 |
| 内存与能耗 | 复杂模型（如 Isolation Forest, LOF）占用高 | K-Means + 固定迭代 + 静态分配，适合 MCU |

> ✅ **核心优势总结**：**低成本、低能耗、可扩展性强**，特别适用于家庭、药房、小型仓库等部署大量同类型电器的场景。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **真实世界数据来源**：一台迷你冰箱（mini-fridge）连续 **14 天**的功率测量数据。
- **总样本数**：共处理 **43,285 条 feature records**。
- **训练集大小**：约 **8,600 个样本**（占总量 ~20%），取自设备稳定运行初期。
- **注入异常类型**：
  - 压缩机长时间运行（extended runtime）
  - 短周期启停（short cycling）
  - 长时间断电（prolonged power-off）

### ⚙️ 实验设置
- **硬件平台**：
  - MCU: STM32F446RE
  - 电流传感器: ACS712 Hall-effect sensor
  - 存储: microSD card
- **软件流程**：
  1. ADC采样 → RMS计算 → 特征提取
  2. 数据清洗 → 状态标注 → K-Means聚类训练
  3. 异常阈值设定（基于训练集中距离分布的 95th percentile）
  4. 模型导出为 `MODEL.TXT`
  5. Device B 加载模型并执行推理

### 📈 评估指标
| 指标类别 | 具体指标 |
|--------|--------|
| **检测性能** | Recall（检出率）、False Positive Rate（误报率）、异常定位准确性 |
| **运行效率** | 推理延迟（inference runtime）、训练时间 |
| **资源消耗** | 内存占用（SRAM/Flash）、模型文件大小、解析开销 |
| **模型复用效果** | DIoL 推理一致性、是否引入额外开销 |

### 🔤 基线方法对比
- **Z-Score-based detector**（单变量统计方法）作为对比基准：
  - 使用均值和标准差判断偏离程度
  - 也在同一平台上实现，用于比较检测行为和性能

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 指标 | 结果 |
|------|------|
| **异常检测准确率** | ✅ **100% Recall**：所有注入异常均被成功识别 |
| **误报情况** | ❗ 在正常操作期间保持极低 false positive |
| **推理吞吐量** | K-Means: **~738 records/sec**（43,285 条记录耗时 58,518 ms） |
| **训练耗时** | K-Means training: **59,328 ms**（仅执行一次） |
| **模型文件大小** | 极小，仅包含 centroids（k=3）、mean/std 和 threshold |
| **解析开销** | negligible（可忽略），不影响启动速度 |
| **DIoL 推理耗时** | **58,514 ms**，与原生推理几乎一致 |

### 🔁 与基线方法对比结果
| 维度 | Z-Score | K-Means (本文) |
|------|--------|----------------|
| 检测能力 | 能识别相同异常区间 | ✅ 更优：决策边界更平滑，抗噪声更强 |
| 输入维度 | 单变量 | ✅ 多变量融合，捕捉更复杂模式 |
| 运行效率 | 58,953 ms（略慢于 K-Means） | ⚡ 更快且更稳定 |
| 可解释性 | 高 | 中等（依赖聚类中心） |
| 是否支持 TOSE/DIoL | 否 | ✅ 是（已验证） |

> 💡 **关键发现**：尽管 Z-Score 表现尚可，但 K-Means 凭借多变量建模能力提供了更鲁棒的检测性能，同时完全兼容 DIoL 框架。

### 🔍 消融实验（隐含分析）
虽然未明确列出消融实验表格，但文中进行了多个关键设计选择的合理性验证：
- **k=3 的选择依据**：对应压缩机三种典型状态（关闭、稳态运行、瞬态/高负载），尝试其他 k 值未见明显提升；
- **20% 训练数据足够性**：实验证明该子集足以稳定估计 centroids 和 threshold；
- **固定迭代次数（3次 Lloyd's iteration）**：确保内存使用确定，避免动态分配；
- **DIoL 复用有效性**：Device B 的检测结果与 Device A **完全一致**，证明模型可移植性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **K-Means 可高效部署于 MCU 并完成 on-device training**，无需依赖外部工具或云服务。
2. **“Train Once, Share Everywhere” (TOSE) 在实践中可行**：通过 `MODEL.TXT` 实现跨设备零训练推理，显著节省能源和时间。
3. **DIoL 是首个面向 TinyML 的分布式模型共享范式**，支持离线、去中心化的模型传播，具有高度实用性。
4. **模型复用不带来任何推理性能损失**：DIoL 推理 runtime 与原生运行基本一致，解析开销可忽略。
5. **多变量 K-Means 比单变量 Z-Score 更具鲁棒性和适应性**，尤其适合家电侧信道分析任务。

### ⚠️ 方法的局限性
- **依赖初始正常数据质量**：若训练阶段包含异常数据，会导致 centroids 偏移，影响检测可靠性；
- **静态模型更新机制缺失**：当前模型不可在线更新，长期漂移或设备老化可能影响性能；
- **安全性未考虑**：`MODEL.TXT` 文件无加密或完整性校验，存在被篡改风险；
- **泛化能力有待验证**：目前仅在一个 mini-fridge 上测试，尚未扩展至多种电器或不同环境；
- **k 值需人工设定**：未集成自动确定最优簇数的方法（如肘部法则）。

### 🔮 未来工作方向
1. **扩展 DIoL 至网络化环境**：支持通过 Wi-Fi/LoRa 等无线方式远程分发模型；
2. **加入安全机制**：对 `MODEL.TXT` 添加数字签名、哈希校验或轻量级加密；
3. **支持模型增量更新**：探索轻量级 online learning 或差分模型推送；
4. **跨设备类型迁移学习**：研究如何将冰箱模型迁移到空调、洗衣机等类似设备；
5. **更大规模部署验证**：在多台设备组成的 fleet 中评估稳定性与可维护性。

---

## 总结一句话
> 本文提出了首个可在 MCU 上实现“训练一次、处处复用”的 **DIoL 框架**，结合轻量级 K-Means 实现了高效、节能、可扩展的 TinyML 异常检测系统，为大规模嵌入式设备智能运维提供了实用路径。

</details>

---

### 6. [Kernel-Smith: A Unified Recipe for Evolutionary Kernel Optimization](https://arxiv.org/abs/2603.28342)

**Authors**: He Du, Qiming Ge, Jiakai Hu, Aijun Yang, Zheng Cai, Zixian Huang, Sheng Yuan, Qinxiu Cheng, Xinchen Xie, Yicheng Chen, Yining Li, Jiaxing Xie, Huanan Dong, Yaguang Wu, Xiangjun Huang, Jian Yang, Hui Wang, Bowen Zhou, Bowen Li, Qipeng Guo, Kai Chen  
**Category**: cs.CL  
**Published**: 2026-03-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.28342v1  

#### Abstract
We present Kernel-Smith, a framework for high-performance GPU kernel and operator generation that combines a stable evaluation-driven evolutionary agent with an evolution-oriented post-training recipe. On the agent side, Kernel-Smith maintains a population of executable candidates and iteratively im...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Kernel-Smith: A Unified Recipe for Evolutionary Kernel Optimization》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 LLM 的高性能 GPU kernel 生成仍面临两大挑战：
- **搜索效率低**：多数系统依赖单次生成（one-shot generation）或多轮对话式调试，容易陷入局部最优，探索多样性不足。
- **优化能力弱**：模型缺乏持续迭代优化的能力，难以在测试时有效利用额外计算资源来逐步提升 kernel 性能。

### 提出了什么新方法或新思路
提出 **Kernel-Smith**，一个统一的框架，结合：
- **稳定评估驱动的进化智能体（evaluation-driven evolutionary agent）**
- **面向进化的后训练策略（evolution-oriented post-training recipe）**

#### 核心设计思想：
1. **进化式智能体架构（Evolutionary Agent Framework）**
   - 维护一个候选程序种群（population），通过多代演化持续优化。
   - 每轮从历史中选取高性能且多样化的候选作为上下文输入，引导模型进行代码变异（mutation）、重组（recombination）。
   - 引入结构化执行反馈（structured execution feedback），包括编译状态、正确性、加速比、硬件指标等，增强学习信号。

2. **面向进化的训练范式（Evolution-Oriented Post-Training）**
   - 不再仅优化“一次性生成”能力，而是将模型训练为“局部改进器（local improver）”。
   - 将长周期的进化轨迹分解为高增益的单步改进样本，用于监督微调（SFT）和强化学习（RL）。
   - 在 RL 阶段，只保留带来显著性能提升的“最佳步骤（best steps）”，避免学习无效或冗余操作。

### 相比现有方法的优势
| 方面 | Kernel-Smith 的优势 |
|------|---------------------|
| **搜索稳定性** | 后端专用评估服务（Triton on NVIDIA / Maca on MetaX）结合重复测量、异常值剔除、CUDAGraph 技术，将执行时间波动控制在 1% 内，确保可靠搜索动力学。 |
| **优化深度** | 支持长达 40 轮的迭代优化，充分利用 test-time compute，实现收益累积（gain compounding）。 |
| **泛化能力** | 统一协议可无缝迁移到不同硬件平台（如 MetaX MACA），无需修改 agent 目标。 |
| **实际部署价值** | 成功向上游项目（SGLang、LMDeploy、DLBlas）提交并合并 PR，证明其产出具备工程可用性。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **自建高质量 PyTorch 数据集**：从 GitHub 爬取开源仓库，提取 `torch.nn.Module` 子类，经静态分析、依赖解析、去重、LLM 辅助测试生成与执行过滤，最终获得 **59k 高质量模块**，覆盖 20 类功能。
- **Cluster-Seeded Expert Data**：对上述数据聚类（HDBSCAN），人工标注代表性中心样本，作为高质量种子用于合成更优进化轨迹。

### 实验设置和评估指标

#### 评估协议（Unified Evolutionary-Agent Protocol）
- 所有模型运行相同的 **Kernel-Smith agent 框架**，进行 **40 轮迭代演化**。
- 解码参数：temperature=0.6, top_p=0.95。
- 上下文长度限制：每轮输入输出均不超过 32K tokens。
- 每个模块独立测试 100 次取平均，保证稳定性。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Correctness (corr)** | 生成 kernel 数值精度满足阈值的比例（含 anti-hacking 检测）。 |
| **Fast Proportion (fast₁)** | 相对于 PyTorch eager 模式的加速比例（speedup > 1）。 |
| **Average Speedup Ratio (avg AMSR)** | 所有成功加速 kernel 的平均加速比；若未加速则记为 0，是核心性能指标。 |

### 基线方法对比
#### 开源模型：
- Qwen3-235B-A22B-think, Qwen3.5-397B-think
- DeepSeek-v3.2-Speciale
- Kimi-K2.5, MiniMax-M2.5

#### 商业闭源模型：
- **Gemini-3.0-pro**
- **Claude-4.6-opus**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（NVIDIA Backend, KernelBench）

| 模型 | Corr (%) | Fast₁ | **Avg AMSR** |
|------|----------|--------|--------------|
| **Kernel-Smith-235B-RL (ours)** | **96.33** | **0.70** | **3.70** ✅ |
| Claude-4.6-opus | 99.33 | 0.77 | 3.33 |
| Gemini-3.0-pro | 94.33 | 0.74 | 2.83 |
| DeepSeek-v3.2-Speciale | 94.67 | 0.61 | 3.44 |

#### 分难度层级表现（Avg AMSR）：
| Level | Easy (1) | Medium (2) | Hard (3) |
|-------|----------|------------|----------|
| **Ours** | 2.30 | **7.77** ✅ | 1.02 |
| Claude-4.6-opus | 2.14 | 5.83 | 2.02 |
| Gemini-3.0-pro | 2.46 | 4.78 | 1.26 |

> 💡 **关键发现**：Kernel-Smith 在中等难度任务上取得最大突破（**7.77× vs 5.83×**），说明其在复杂融合与调度优化方面具有更强推理能力。

---

### MetaX MACA Backend 结果

| 模型 | Corr (%) | Fast₁ | **Avg AMSR** |
|------|----------|--------|--------------|
| **Kernel-Smith-MACA-235B (ours)** | 100 | 0.84 | **14.26** ✅ |
| Qwen3-235B-2507-think | 100 | 0.80 | 12.30 |
| DeepSeek-v3.2-think | 97.8 | 0.73 | 8.01 |
| Kimi-K2.5 | 100 | 0.82 | 11.60 |

> ✅ 在异构平台（MACA）上依然超越大规模商业模型，验证了框架的**跨平台适应性**。

---

### 消融实验结果（隐含于训练设计中）
虽然文中未明确列出消融表，但从训练策略可推断以下关键设计的有效性：
- **仅使用“最佳改进步骤”进行 RL 训练** 显著优于使用全部步骤或仅初始步骤。
  - 全部步骤易导致模型记忆优质示例而非学习通用优化策略。
  - 初始步骤目标简单（仅迁移功能），不适合 RL。
- **Cluster-Seeded Expert Data** 提升了训练数据质量上限，使模型能学习更高阶的优化模式。
- **结构化反馈机制**（错误日志、性能曲线、硬件指标）帮助模型理解失败原因，提升修复效率。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **稳定评估是可靠进化的前提**  
   通过固定计算图、多次测量、异常检测和 CUDAGraph 技术，显著降低 profiling 噪声，防止因误判导致的搜索崩溃。

2. **模型应被训练为“局部改进器”而非“一次生成器”**  
   将进化轨迹压缩为高增益单步样本，使模型专注于学习有效的原子级优化动作（如 tile size 调整、fusion 决策），从而支持长期收益累积。

3. **统一进化协议具有强大泛化性和实用性**  
   在 NVIDIA Triton 和 MetaX MACA 上均达到 SOTA，且生成 kernel 已成功集成至 **SGLang、LMDeploy、DLBlas** 等生产系统。

4. **LLM 驱动的 kernel 优化已具备实际工程价值**  
   不只是 benchmark 刷榜，而是能产生真正可部署、可维护、经过充分测试的上游贡献。

---

### 方法的局限性
- **高度依赖自动化评估基础设施**：需要完善的编译、执行、profiling、anti-hacking 检测链路，部署成本较高。
- **对极端边缘 case 处理能力有限**：尽管有 100 次重复测试，但仍可能遗漏某些 rare race condition 或数值溢出问题。
- **当前聚焦于前向 pass kernel**：尚未系统处理反向传播、梯度检查、分布式通信等完整训练场景。

---

### 未来工作方向
1. **扩展更多 backend 支持**：如华为 NPU、Apple Metal、AMD ROCm 等，进一步验证框架通用性。
2. **自动化 Pull Request 流程**：将测试、文档、CI 验证等环节也纳入 agent 工作流，实现端到端自动提交。
3. **引入更丰富的工具与记忆机制**：如支持调用 external profiler API、缓存历史优化经验、构建 kernel knowledge graph。
4. **发展自适应搜索策略**：根据任务特征动态调整 mutation 强度、population 规模、feedback 密度等超参。

--- 

> 🔗 **项目主页**: [https://chat.intern-ai.org.cn/kernel-smith](https://chat.intern-ai.org.cn/kernel-smith)  
> 📄 **论文链接**: [arXiv:2603.28342](https://arxiv.org/abs/2603.28342)

</details>

---

### 7. [Liquid Networks with Mixture Density Heads for Efficient Imitation Learning](https://arxiv.org/abs/2603.27058)

**Authors**: Nikolaus Correll  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.27058v1  

#### Abstract
We compare liquid neural networks with mixture density heads against diffusion policies on Push-T, RoboMimic Can, and PointMaze under a shared-backbone comparison protocol that isolates policy-head effects under matched inputs, training budgets, and evaluation settings. Across tasks, liquid policies...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Liquid Networks with Mixture Density Heads for Efficient Imitation Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **迭代去噪策略的效率瓶颈**：当前主流的 **Diffusion Policy** 虽在模仿学习中表现优异，但其依赖多次迭代去噪（iterative denoising）进行动作生成，导致推理延迟高、部署复杂，尤其不适合实时机器人控制。
- **离散RNN的时间建模脆弱性**：传统离散时间循环网络（如LSTM、GRU）在长期依赖和连续动态建模上存在不稳定性。
- **多模态动作分布的模式坍塌**：确定性策略（如MSE训练）在面对多个可行动作序列时倾向于输出“平均动作”，破坏任务可行性。

### **提出的新方法或新思路**
- **Liquid Neural Networks + Mixture Density Network (MDN) 头部架构**：
  - 采用 **CfC (Closed-form Continuous-time)** 作为核心的 **Liquid Network** 架构，提供连续时间动态建模能力，避免数值积分开销。
  - 在策略头（policy head）引入 **Mixture Density Head**，显式建模多模态动作分布，避免模式平均。
  - 整体为单次前向传播（single forward pass），无需迭代采样。

- **公平比较协议（Fair Shared-Backbone Protocol）**：
  - 设计统一的共享编码器（shared transformer backbone），确保 Liquid 和 Diffusion 两种策略头接收完全相同的上下文输入，从而**隔离策略头本身的性能差异**，实现公平对比。

### **相比现有方法的优势**
- **参数更少**：Liquid + MDN 模型仅需约 **4.3M 参数**，约为 Diffusion（8.6M）的一半。
- **推理更快**：推理速度提升 **1.8–2.0×**（200ms vs 400ms 级别）。
- **预测精度更高**：在 best-of-K MSE 上领先 **2.4–2.5×**，尤其在高维任务（RoboMimic Can, PointMaze）中优势显著。
- **样本效率更强**：在低数据（1%–10%）和中等数据场景下，Liquid 表现更鲁棒，收敛更快。
- **部署更简单**：无需 distillation 或去噪流水线，适合边缘设备部署。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 任务类型 | 状态维度 | 动作维度 | 特点 |
|-------|--------|--------|--------|------|
| **Push-T** | 接触丰富操作（Contact-Rich Manipulation） | 5 | 3 | 无视觉输入，多模态短视界动作 |
| **RoboMimic Can** | 高维操作（High-Dimensional Manipulation） | 57 | 7 | 模拟机械臂操作，挑战高维控制 |
| **PointMaze** | 导航（Navigation） | 8 | 2 | 迷宫环境，存在左右绕行的双模选择 |

### **实验设置**
- **窗口化处理**：历史观测 $ H_o = 2 $，预测动作 $ H_p = 16 $。
- **归一化**：所有观测和动作使用 min-max 归一化至 [-1, 1]。
- **共享骨干网络**：
  - 视觉任务使用冻结的 vision encoder + shared transformer。
  - 非视觉任务使用 identity projection + shared transformer。
  - 输出统一 latent 表示供两个 policy head 使用。
- **模型配置**：
  - **Liquid + MDN**：5层 CfC 编码器（0.5× 规模） + GRU 解码器 + 5-component Gaussian Mixture 输出。
  - **Diffusion**：全规模 DDPM（1.0× 参数），固定 **50 步去噪**。
- **训练设置**：
  - 120 epochs，AdamW + warmup + cosine decay。
  - Gradient clipping at norm 1.0。
  - 固定随机种子（42）保证可复现性。

### **评估指标**
| 指标 | 类型 | 说明 |
|-----|-----|------|
| **NLL (Negative Log-Likelihood)** | 分布质量 | Liquid 报告精确 MDN NLL，Diffusion 报告 proxy NLL |
| **MSE** | 预测误差 | 包括 deterministic MSE、sample-mean MSE、best-of-K MSE（K=1,2,5,10） |
| **Success Rate** | 闭环性能 | 在 PyMunk (Push-T) 和 Gymnasium (PointMaze) 中测试任务成功率 |
| **Distance-Success** | 更严格的成功标准 | 最小目标距离 ≤ 0.20 才算成功 |
| **Latency (ms)** | 推理延迟 | 单步推理耗时（wall-clock time） |
| **Parameter Count** | 模型大小 | 总参数量（单位：百万） |
| **Diversity & Smoothness (jerk)** | 动作质量 | 动作多样性与平滑性 |

### **基线方法对比**
- **主对比对象**：**Diffusion Policy**（当前主流方法）
- 公平性保障：
  - 相同输入、相同 backbone、相同训练预算、相同评估协议。
  - Liquid 模型参数仅为 Diffusion 的一半（0.5× vs 1.0×），强调**参数效率**而非容量优势。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| Dataset | Model | Params (M) | NLL ↓ | MSE ↓ | Latency (ms) ↓ |
|--------|-------|----------|--------|--------|----------------|
| **Push-T** | Liquid + MDN | 4.34 | **-6.999** | 0.000158 | **195** |
| | Diffusion | 8.60 | -3.768 | 0.000155 | 381 |
| **RoboMimic Can** | Liquid + MDN | 4.36 | **-20.830** | **0.007** | **205** |
| | Diffusion | 8.84 | -15.732 | 0.124 | 380 |
| **PointMaze** | Liquid + MDN | 4.34 | **-8.615** | **0.045** | **252** |
| | Diffusion | 8.60 | -3.578 | 0.450 | 448 |

> ✅ **MSE 对比**：Liquid 在 RoboMimic Can 上比 Diffusion **低 18×**，在 PointMaze 上 **低 10×**，Push-T 基本持平。
>
> ✅ **NLL 对比**：Liquid 在所有任务上均取得更低（更好）的负对数似然，表明其密度估计更优。
>
> ✅ **延迟对比**：Liquid 推理速度快 **1.8–2.0×**。

### **与基线方法的对比结果**
- **Open-Loop（离线预测）**：
  - Liquid 在 **best-of-K MSE** 上平均优于 Diffusion **2.4–2.5×**（见 Figure 3）。
  - 在整个 16 步预测范围内，Liquid 每一步误差更低（Figure 4）。
  - 在低数据 regime（1%–10%），Liquid 收敛更快、更稳定（Figure 2）。
- **Closed-Loop（闭环部署）**：
  - **Push-T**：Liquid 成功率 **91%** vs Diffusion **88%**。
  - **PointMaze**：Liquid 成功率 **20.0%** vs Diffusion **9.5%**；Distance-Success **9.7%** vs **3.7%**。
  - 尽管离线指标优势明显，但闭环增益有限且任务相关，说明**离线指标不能完全反映部署性能**。

### **消融实验与分析（Appendix）**
- **Free-running validation loss** 用于模型选择，更能反映真实部署表现。
- **MDN 组件数 K=5** 是最佳平衡点，K=3 不足，K≥7 带来边际收益递减。
- **CfC 层数 5 层** 最优，少则表达不足，多则计算冗余。
- **Gradient clipping、warm-up、cosine decay** 对训练稳定性至关重要。
- **BatchNorm 不用于 Recurrent Layers**，LayerNorm 仅用于 Transformer Backbone。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Liquid + MDN 是一种高效且强大的模仿学习范式**：
   - 在参数减少 50% 的前提下，实现更高的预测精度和更快的推理速度。
   - 显式多模态建模（MDN）有效避免模式坍塌，特别适合多解任务（如 Push-T、PointMaze）。

2. **连续时间建模优于迭代去噪**：
   - Liquid 学习的是一个**可重用的轨迹生成器**，而 Diffusion 学习的是每一步的修正函数。
   - 理论上，迭代生成器存在 $ T = \Omega(1/\epsilon) $ 的复杂度下限，而 Neural ODE 类模型可通过高阶积分器突破此限制。

3. **样本效率优势显著**：
   - 在低数据和中等数据 regime，Liquid 表现远超 Diffusion，更适合现实世界中数据稀缺的场景。

4. **离线指标 ≠ 闭环性能**：
   - 尽管 Liquid 在 NLL 和 MSE 上大幅领先，但闭环成功率提升有限。
   - 说明**轨迹一致性、错误恢复能力、环境随机性**等因素在部署中起重要作用，需结合闭环验证。

5. **轻量化部署潜力巨大**：
   - 所有实验在 **Apple PowerBook M5（无GPU）** 上完成，证明 Liquid 模型可在资源受限设备上运行。

### **局限性**
- **控制频率下的延迟仍需优化**：尽管比 Diffusion 快，但在极高频率控制（如 >100Hz）下仍可能成为瓶颈。
- **backbone 表征质量主导视觉任务**：在高维视觉任务中，编码器的质量可能掩盖 policy head 的改进。
- **任务依赖性强**：在 Push-T 上主要提升延迟，在 PointMaze 上才显著提升成功率。
- **未涵盖所有加速 Diffusion 方法**：如 Consistency Distillation、One-Step Diffusion 等最新技术未直接对比。

### **未来工作方向**
- 探索 **Liquid + Diffusion hybrid 架构**：利用 Liquid 做粗略轨迹生成，Diffusion 微调。
- 将 Liquid 扩展到 **端到端视觉输入** 场景，验证其在像素级输入下的表现。
- 结合 **test-time adaptation** 或 **replanning** 机制，提升闭环鲁棒性。
- 进一步压缩模型以支持 **on-device robotics** 应用（如手机、嵌入式控制器）。

---

> **一句话总结**：  
> 本文提出了一种基于 **Liquid Neural Networks + Mixture Density Heads** 的新型模仿学习框架，在参数减半、推理提速近两倍的同时，实现了 **2.4× 更低的预测误差** 和更强的样本效率，是当前 **Diffusion Policy** 的有力替代方案，尤其适用于资源受限、数据稀疏的真实机器人系统。

</details>

---

### 8. [ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation for LLM Inference](https://arxiv.org/abs/2603.27138)

**Authors**: Qiuyang Zhang, Kai Zhou, Ding Tang, Kai Lu, Cheng Li, Zhenyu Yang, Peng Xu, Jiguang Wan  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.27138v1  

#### Abstract
Large language models encounter critical GPU memory capacity constraints during long-context inference, where KV cache memory consumption severely limits decode batch sizes. While existing research has explored offloading KV cache to DRAM, these approaches either demand frequent GPU-CPU data transfe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation for LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLM）在长上下文推理过程中面临严重的 **GPU 内存容量瓶颈**，尤其是 **KV Cache** 的内存消耗极大，限制了解码阶段的 batch size 和吞吐量。现有的 KV Cache 卸载方法存在以下两类问题：

- **基于召回的方法（如 InfiniGen）**：依赖频繁的 GPU-CPU 数据传输，导致 PCIe I/O 成为瓶颈，GPU 大量空闲等待。
- **协同注意力方法（如 HGCA）**：将部分 attention 计算放在 CPU 上并行执行，但由于 CPU 计算能力远弱于 GPU（约慢 20x），造成 **CPU 计算瓶颈**。

### 🚀 提出的新方法：ScoutAttention
ScoutAttention 是一种高效的 **GPU-CPU 协同稀疏注意力机制**，通过以下三个核心设计实现高性能 KV Cache 卸载：

1. **GPU-CPU 协同块级稀疏注意力（GPU-CPU Collaborative Block-wise Sparse Attention）**
   - 将 KV Cache 分成固定大小的 block，并保留每个 block 的摘要（block digest）和少量重要 block 在 GPU 上。
   - 利用 **temporal locality** 特性（相邻 token 所关注的重要 block 高度重叠），仅对未驻留在 GPU 上的 top-k block 在 CPU 上计算，大幅降低 CPU 负载。

2. **层前向 CPU 预计算（Layer-ahead CPU Pre-computation）**
   - 在 GPU 执行第 $i$ 层时，提前预测第 $i+1$ 层的 Query 向量（$Q^{i+1}_{\text{pred}}$），并启动其对应的 CPU 注意力计算。
   - 该策略利用整个 Transformer 层的时间窗口（含 attention、FFN、KV 投影等）进行 CPU 计算，使 CPU 拥有 **3 倍于传统并行方案的处理时间**，有效隐藏延迟。

3. **异步周期性 KV Cache 回调（Asynchronous Periodic KV Cache Recall）**
   - 定期从 DRAM 异步回调重要的 KV blocks 到 GPU，以纠正因生成过程中的“重要性漂移”带来的 CPU 负载上升。
   - 回调操作不在关键路径上，避免阻塞 GPU，且频率经过离线调优（默认阈值为 CPU 计算比例 <12%）。

### 🔍 相比现有方法的优势
| 方法 | 主要瓶颈 | ScoutAttention 改进 |
|------|----------|---------------------|
| InfiniGen（召回式） | PCIe I/O 瓶颈 → GPU 等待 61% 时间 | 消除频繁召回，采用协同计算 |
| HGCA（协同注意力） | CPU 计算瓶颈 → GPU 等待 57% 时间 | 引入预计算 + 稀疏化，显著减轻 CPU 负担 |

> ✅ ScoutAttention 成功规避了 I/O 和 compute 双重瓶颈，在保持高精度的同时大幅提升推理吞吐。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **LongBench**：广泛使用的长文本理解基准，涵盖多任务场景：
  - 单文档问答：Qasper, NarrativeQA
  - 多文档问答：2WikiMQA, DuReader
  - 摘要任务：GovReport, QMSum, SAMSum
  - 检索任务：PassageRetrieval
- 最大上下文长度达 **64k tokens**

### ⚙️ 实验设置
- **模型**：
  - 准确性评估：Qwen 3 8B
  - 性能评估：Qwen 3 14B
- **硬件平台**：
  - GPU：80GB HBM
  - CPU：36-core，通过 PCIe 4x16 互联
- **框架实现**：基于 **SGLang** 构建，使用 FlashInfer 实现高效 top-k 选择，IPEX 优化 CPU attention worker。
- **Block Size**：默认 32 tokens
- **Sparsification Budget**：2048 tokens（即最多参与 attention 的 block 数量）

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Decode Throughput**（tokens/sec） | 解码阶段每秒输出 token 数量 |
| **End-to-end Latency** | 包括 GPU 计算、CPU 计算、I/O 等端到端延迟 |
| **GPU Idle Time** | GPU 因等待 I/O 或 CPU 而空闲的比例 |
| **Accuracy Drop** | 相比 Full Attention 的平均准确率下降百分比 |
| **Cosine Similarity** | 验证预测 Query 与真实 Query 的相似性 |

### 🆚 基线方法对比
| 基线 | 类型 | 特点 |
|------|------|------|
| **FullKV** | 全量缓存 | 不卸载 KV Cache，作为性能下限参考 |
| **InfiniGen** | 基于召回的卸载 | 动态预测并提前召回重要 token |
| **HGCA** | 协同注意力 | GPU 与 CPU 并行计算 attention，CPU 使用滑动窗口稀疏策略 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**准确性表现**
- 在 LongBench 上，ScoutAttention 的平均准确率相比 Full Attention 仅下降：
  - **2.5% @ 1024 budget**
  - **2.1% @ 2048 budget**
- 精度损失极小，且优于 InfiniGen（得益于更合理的稀疏策略和高 query 相似性）。

#### （2）**解码吞吐量（Throughput）**
- 在输入长度为 **64k** 时：
  - ScoutAttention 达到 **5.1x speedup** vs FullKV
  - 达到 **2.1x speedup** vs InfiniGen / HGCA
- 随着输入增长，优势持续扩大（FullKV 受内存限制严重）。

#### （3）**GPU 利用率与空闲时间**
| 方法 | GPU Idle Time |
|------|---------------|
| HGCA | 57% |
| InfiniGen | 61% |
| **ScoutAttention** | **仅 6%** |

> 极低的 idle time 表明系统资源被高效利用，无明显 I/O 或 compute 瓶颈。

#### （4）**批处理扩展性（Batch Size Scaling）**
- 当 batch size 从 16 增加到 64：
  - HGCA / InfiniGen 扩展性差（<1.3x 提升）
  - ScoutAttention 实现 **1.78x（16→32）** 和 **1.48x（32→64）** 提升
- 显示出良好的可扩展性和稳定性。

#### （5）**消融实验（Ablation Study）**
| 配置 | Speedup vs Baseline | 说明 |
|------|---------------------|------|
| 无 Pre-computation & 无 Recall（w/o PC&PR） | 1.0x | 性能最差 |
| 加入 Pre-computation（PC） | **1.39x** | 显著减少 CPU 等待 |
| 再加入 Periodic Recall（PR） | **额外提升 1.20x** | 控制 CPU 负载漂移 |
| 完整 ScoutAttention | **总提速 2.1x** | 两者协同增效 |

> 结果验证了两个核心机制的有效性和互补性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **CPU-side attention 可以高效用于 KV Cache 卸载**，前提是解决其计算延迟问题。
2. **Query 的强 temporal locality** 支持使用预测 Query 进行 layer-ahead 预计算（cosine similarity > 0.93）。
3. **Temporal locality of important blocks** 使得只需在 CPU 上处理少量非本地 top-k blocks，显著降低开销。
4. **异步周期性回调机制** 能有效控制重要性漂移，维持低 CPU 负载。
5. ScoutAttention 实现了 **I/O 与 compute 瓶颈的双重突破**，是首个同时避免召回 I/O 和 CPU compute 瓶颈的 KV Cache 卸载框架。

### ⚠️ 方法的局限性
- 依赖于 **block-wise sparse attention** 的有效性，若模型 attention 分布高度分散，可能影响效果。
- 对 **PCIe 带宽仍有一定依赖**，虽然不频繁，但 recall 操作仍需一定带宽保障。
- 当前设计针对 **disaggregated prefill-decode 架构**，在统一架构中收益可能略有不同。

### 🔮 未来工作方向
- 探索 **动态调整 sparsification budget 和 recall interval** 的在线自适应机制。
- 扩展至 **multi-GPU + multi-CPU** 场景下的分布式协同推理。
- 结合 **quantization 或 low-rank approximation** 进一步压缩 CPU 计算负载。
- 应用于 **real-time streaming LLM inference** 场景，探索更低延迟的可能性。

---

> **总结一句话**：  
> ScoutAttention 通过 **layer-ahead CPU 预计算 + 协同块稀疏注意力 + 异步周期回调**，实现了 **高效、低延迟、高吞吐的 KV Cache 卸载方案**，在仅牺牲 2.1% 准确率的情况下，达到 **5.1x 吞吐提升**，显著优于现有方法。

</details>

---

### 9. [FlowRL: A Taxonomy and Modular Framework for Reinforcement Learning with Diffusion Policies](https://arxiv.org/abs/2603.27450)

**Authors**: Chenxiao Gao, Edward Chen, Tianyi Chen, Bo Dai  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.27450v1  

#### Abstract
Thanks to their remarkable flexibility, diffusion models and flow models have emerged as promising candidates for policy representation. However, efficient reinforcement learning (RL) upon these policies remains a challenge due to the lack of explicit log-probabilities for vanilla policy gradient es...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# FlowRL: A Taxonomy and Modular Framework for Reinforcement Learning with Diffusion Policies —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Diffusion Models (DMs)** 和 **Flow Models (FMs)** 的强化学习（DPRL）算法虽然展现出强大的策略表达能力，但由于缺乏显式的 `log-probability`，难以直接应用传统的 **policy gradient** 或 **reparameterization trick** 进行训练。此外，现有方法分散在不同设定（offline/online/offline-to-online）、采用不同的网络架构、噪声调度和评估协议，导致难以进行公平比较和系统分析。

因此，该领域面临两大挑战：
- 缺乏统一的理论框架来理解各种 DPRL 方法之间的关系；
- 缺少高效、模块化的开源工具支持可复现研究。

---

### 🚀 提出的新方法与新思路

本文提出三大核心贡献：

#### （1）**统一的 DPRL 分类体系（Taxonomy）**
首次从两个正交维度对现有 DPRL 算法进行系统分类：
- **Guidance Mechanism（引导机制）**：如何利用价值函数 $Q(s,a)$ 引导扩散过程；
- **Reference Policy（参考策略）**：正则化项中的先验分布选择（如 $\pi_{\text{ref}} = \text{Unif}(A), \pi_{k-1}, \pi_D$）。

由此构建了涵盖五类主流方法的完整分类表（见原文 Table 1），揭示了不同算法间的数学等价性和设计权衡。

#### （2）**模块化、高性能的开源框架 FLoWRL**
- 基于 **JAX** 构建，利用 JIT 编译实现高吞吐训练与推理；
- 支持多种 backend（TensorBoard, W&B, CSV）、checkpointing 和 Hydra 配置管理；
- 四层架构设计（Network → Model → Algorithm → Workflow），高度可组合，便于快速原型开发。

#### （3）**标准化的大规模基准测试（Benchmarking）**
在三个代表性连续控制套件上进行了全面评估：
- **Gym-Locomotion**
- **DeepMind Control Suite (DMC)**
- **IsaacLab**

提供了严格的公平比较环境，统一超参数、网络结构和训练流程，填补了此前文献中因实现差异带来的性能偏差问题。

---

### 🔍 相比现有方法的优势

| 维度 | FlowRL 的优势 |
|------|----------------|
| **理论层面** | 提供首个统一视角解释 Q-value guidance、weighted matching、reparameterization 等机制的关系 |
| **工程层面** | 开源代码库支持 JAX 加速 + 模块化设计，显著降低研究门槛 |
| **实验层面** | 大规模、跨平台 benchmark 提供可靠性能参考，推动领域标准化 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与任务

| Benchmark | 物理引擎 | 代表任务 | 数量 |
|---------|--------|--------|-----|
| **Gym-Locomotion** | MuJoCo | Ant, HalfCheetah, Hopper, Humanoid 等 | 6 个 |
| **DeepMind Control (DMC)** | MuJoCo | Cartpole, Cheetah-Run, Quadruped, Humanoid-Walk 等 | 8 个 |
| **IsaacLab** | GPU-accelerated sim | Lift-Cube-Franka, Velocity-Flat-Anymal-D 等 | 44 个（选 4 个用于 on-policy） |

> 注：Gym 和 DMC 主要用于 off-policy / offline 设置；IsaacLab 用于 on-policy 测试。

---

### ⚙️ 实验设置与评估指标

#### 训练设置概览（详见 Appendix A.2）

| 设置项 | Gym-Locomotion | DMC | IsaacLab |
|-------|----------------|-----|----------|
| 总帧数 | 1M | 1M | 100M |
| 批大小 | 256 | 512 | 6144（1024 并行 envs） |
| 评估频率 | 每 10K 帧 | 每 10K 帧 | 每 5M 帧 |
| 评估回合数 | 10 | 10 | 10 |
| 观测/奖励归一化 | 否 | 否 | 是（仅观测） |

#### 评估指标
- **主指标**：未折扣的 episodic return（均值 ± 标准差）
- **辅助分析**：
  - 归一化性能曲线（Normalized Return）
  - Performance Profile（达到某一阈值的比例）
  - 消融实验（ablation on architecture, diffusion steps, noise schedule）

---

### 🆚 基线方法对比

#### Offline RL 对比方法：
- **IQL**, **IDQL**（behavior cloning + BoN sampling）
- **Diffusion-QL**, **FQL**, **DAC**, **BDPO**

#### Online Off-policy 对比方法：
- **SAC**（Gaussian baseline）
- **QSM**, **DACER**, **SDAC**, **QVPO**, **DPMD**

#### Online On-policy 对比方法：
- **PPO**（baseline）
- **DPPO**, **FPO**, **GenPO**

所有方法尽可能保持一致的网络结构（MLP/SimBa）、优化器、学习率等，除非原论文将其作为核心创新。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）Offline RL 结果（Table 2）

| 方法 | Average Normalized Return | 最佳表现任务 |
|------|----------------------------|-------------|
| **BDPO** | **105.8** | 在 `halfcheetah-m-e`, `walker2d-m-e` 上领先 |
| **DAC** | 103.9 | 多数 medium-expert 数据集表现优异 |
| **IQL / IDQL** | ~85–95 | 明显落后于端到端训练的 diffusion policy |

> ✅ 发现：端到端训练的 diffusion policy 显著优于仅用 diffusion 作行为克隆 + BoN 的方法（如 IDQL），说明 RL 微调至关重要。

---

#### （2）Online Off-policy（Gym-Locomotion）

- **SDAC**, **DACER**, **DPMD** 表现最强，超越 SAC；
- **QSM** 和 **QVPO** 存在训练不稳定现象；
- **Performance Profile 图（Figure 3）** 显示 SDAC 在高分段覆盖率最高。

#### （3）Online On-policy（IsaacLab）

- **PPO** 整体最稳定且性能最佳；
- **GenPO** 性能次之但计算成本高（需 Jacobian）；
- **FPO** 出现训练崩溃（collapse），归因于负优势样本导致无界优化。

---

### 🔬 消融实验结果

#### （1）Action Dimensionality 影响（Figure 6）
- 随动作维度上升，**weighted matching 方法（如 SDAC）性能下降明显**；
- **Q-value guidance（QSM）和 reparameterization（DACER）更鲁棒**；
> 💡 原因：加权匹配依赖函数值采样，在高维空间效率低、方差大。

#### （2）Diffusion Steps 数量影响（Figure 7）
- **QSM & SDAC**：性能随步数增加单调提升；
- **DACER**：超过 5 步后性能下降；
> 💡 原因：reparameterization 类方法（如 BPTT）随链长增长梯度传播困难，优化变复杂。

#### （3）Backbone 架构影响（Figure 8）
- 将 MLP 替换为 **SimBa** 后，**QSM 和 DACER 在 DMC Hard 任务上均有显著提升**；
> ✅ 重要发现：**网络架构是强 confounding factor**，必须控制变量才能评估算法本身改进。

#### （4）Noise Schedule 影响（Figure 9）
- 将 cosine schedule 替换为 linear schedule，**性能几乎无变化**；
> ❌ 结论：在 RL 场景下，噪声调度的影响远小于在图像生成中的作用。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **DPRL 算法可通过 guidance + reference policy 两轴统一建模**，有助于理解不同方法的设计哲学；
2. **端到端训练的 diffusion policy 显著优于 behavior cloning + BoN**，尤其是在 offline RL 中；
3. **不同 guidance 范式各有优劣**：
   - **Q-value guidance**：适合高维动作空间；
   - **Weighted matching**：理论保证强，但在高维下效率受限；
   - **Reparameterization**：训练快但易受优化路径影响；
4. **网络架构（如 SimBa）对性能影响巨大**，不应忽视其作为变量的作用；
5. **噪声调度（cosine vs linear）在 RL 中影响微弱**，可简化设计。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **计算开销大** | 尤其是 reparameterization 和 on-policy 方法（如 GenPO）需要大量 Jacobian 计算 |
| **训练稳定性问题** | 如 FPO 在某些任务上会崩溃，需进一步正则化设计 |
| **未覆盖稀疏奖励与长程任务** | 当前 benchmark 多为 dense-reward 连续控制，尚未验证在复杂任务上的泛化性 |
| **硬件依赖性强** | IsaacLab 实验依赖大规模并行 GPU 模拟，普通研究者难复现 |

---

### 🔮 未来工作方向

1. **设计更鲁棒、低方差的高维动作空间引导方法**；
2. **探索 diffusion policy 在 long-horizon 和 sparse-reward 任务中的潜力**；
3. **结合 consistency models 或 flow matching 加速推理**；
4. **将 FLoWRL 扩展至 vision-language-action（VLA）模型训练**；
5. **发展无需显式 score estimation 的 policy gradient 近似方法**。

---

## 🔗 开源资源

- **代码仓库**：[https://github.com/typoverflow/flow-rl](https://github.com/typoverflow/flow-rl)
- 支持 JAX、多 backend 日志、Hydra 配置、即插即用组件
- 包含完整训练脚本与复现配置

> ✅ 一句话总结：**FlowRL 不仅是一个算法库，更是推动 diffusion-based RL 标准化研究的基础设施。**

</details>

---

### 10. [Heddle: A Distributed Orchestration System for Agentic RL Rollout](https://arxiv.org/abs/2603.28101)

**Authors**: Zili Zhang, Yinmin Zhong, Chengxu Yang, Chao Jin, Bingyang Wu, Xinming Wei, Yuliang Liu, Xin Jin  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.28101v1  

#### Abstract
Agentic Reinforcement Learning (RL) enables LLMs to solve complex tasks by alternating between a data-collection rollout phase and a policy training phase. During rollout, the agent generates trajectories, i.e., multi-step interactions between LLMs and external tools. Yet, frequent tool calls induce...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《HEDDLE: A Distributed Orchestration System for Agentic RL Rollout》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本论文针对 **Agentic Reinforcement Learning (RL)** 中的 **rollout 阶段效率瓶颈** 展开研究。在 Agentic RL 中，LLM 通过多步与外部工具交互生成轨迹（agentic trajectories），但由于这些轨迹呈现 **长尾分布（long-tailed distribution）** ——即少数复杂、多步的“拖沓者”（stragglers）显著延长整体执行时间——导致严重的资源闲置和训练吞吐量下降。

现有系统采用 **step-centric 设计**，将每一步视为独立请求处理，忽略了轨迹上下文，从而引发三大系统级问题：
- **队列延迟（queueing delay）**：长尾轨迹反复排队，累积严重延迟。
- **干扰开销（interference overhead）**：负载不均衡或频繁重调度加剧内存与计算争用。
- **单 token 时间膨胀（inflated per-token time）**：固定资源配置无法兼顾短轨迹高吞吐与长轨迹低延迟需求。

### 提出了什么新方法或新思路
作者提出 **HEDDLE**，一个**以轨迹为中心（trajectory-centric）的分布式编排系统**，从“何时（when）、何地（where）、如何（how）”三个维度优化 rollout 执行：

#### ① Trajectory-level Scheduling（何时调度）
- 引入 **Progressive Priority Scheduling (PPS)**，基于运行时预测动态提升长尾轨迹优先级。
- 使用可训练的 **runtime predictor** 融合初始 prompt 分析与运行中上下文，逐步精化轨迹长度估计。
- 支持抢占式执行（preemptive execution），确保高优先级请求能中断低优先级任务。

#### ② Trajectory-aware Placement（何处放置）
- 初始阶段使用 **Presorted Dynamic Programming** 将预测较长的轨迹隔离到不同 worker，最小化干扰系数 α。
- 运行时引入 **Opportunistic Migration**，利用工具调用间隙异步迁移 KV cache 和轨迹状态，纠正预测偏差而不阻塞主路径。

#### ③ Trajectory-adaptive Resource Manager（如何分配资源）
- 打破同构配置限制，为不同轨迹动态分配不同的 **model parallelism (MP)** 策略：
  - 长尾轨迹 → 高 MP → 减少 per-token time（低延迟）
  - 短轨迹 → 低 MP → 提升吞吐
- 使用 **sort-initialized simulated annealing** 快速搜索近似最优的 GPU 资源划分方案。

### 相比现有方法的优势
| 维度 | 现有方法（如 Slime, Verl） | HEDDLE |
|------|--------------------------|--------|
| 调度粒度 | Step-centric（轮询） | **Trajectory-centric + 动态优先级** |
| 放置策略 | Cache-affinity 或 Least-load | **预排序 DP + 运行时迁移** |
| 资源管理 | 固定同构配置 | **自适应异构并行策略** |
| 总体效果 | 无法缓解长尾瓶颈 | **系统性消除 straggler 影响** |

---

## 2. 核心实验方法和设置

### 使用的数据集
在三种典型的 Agentic RL 场景下进行测试：
- **Coding Agent**: `CodeForces` 数据集，配备沙箱工具用于代码执行与测试验证。
- **Search Agent**: `HotpotQA` 数据集，集成 Web Search 工具实现多跳推理。
- **Math Agent**: `DAPO-Math` 数据集，结合计算器与求解器工具解决数学问题。

所有任务均使用 GRPO 算法生成 16 个样本/提示，最大输出长度设为 40K tokens。

### 实验设置
- **硬件平台**：8 台服务器，共 64 张 NVIDIA Hopper GPU，支持 NVLink 和 InfiniBand（400Gb/s）及 GPUDirect RDMA。
- **模型规模**：Qwen3 系列（8B, 14B, 32B）instruction-tuned 版本。
- **框架基础**：基于 Verl、SGLang 和 Ray 构建，约 15K 行 Rust/Python/C++ 代码。
- **控制平面通信**：Ray + ZMQ。

### 评估指标
- 主要指标：**end-to-end rollout throughput（tokens/s）**
- 辅助分析指标：
  - 最长轨迹的 queueing delay
  - 干扰因子 α
  - per-token time
  - 预测精度（Pearson 相关系数、tail recall）

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Slime** | 基于 SGLang 的开源 RL 框架，按步骤路由至最轻载 worker（least-load） |
| **Verl** | 使用静态 cache-aware 放置，每个轨迹绑定单一 worker |
| **Verl*** | 混合策略：当负载偏斜超过阈值时切换为 least-load，否则保持 cache-aware |

> 所有 baseline 均使用 round-robin 调度和同构资源配置。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
如图 12 所示，在多种 workload 和 model 规模下，HEDDLE 显著优于所有 baseline：

| 模型 | 场景 | 相对提速（vs Verl） | 相对提速（vs Verl*） | 相对提速（vs Slime） |
|------|------|--------------------|-----------------------|------------------------|
| Qwen3-8B ~ 32B | Coding / Search / Math | **1.4× – 2.3×** | **1.1× – 2.4×** | **1.2× – 2.5×** |

> **最高达 2.5× 的端到端 rollout 吞吐提升**

且随着模型增大（如从 8B 到 32B），性能增益更加明显，说明 HEDDLE 在高争用场景下更具优势。

### 与基线方法的对比结果
- **Slime** 在 coding/math 上表现较好（因缓解了负载不均），但在 search 上不如 Verl（cache 效率低）。
- **Verl** 在 search 上占优（短序列高频交互，cache locality 更重要）。
- **Verl*** 表现居中，体现 trade-off。
- **HEDDLE 全面超越**，因其同时实现了：
  - 高 cache hit rate（通过 placement 与 migration）
  - 低干扰（通过隔离长尾轨迹）
  - 快速响应（通过抢占调度与资源适配）

### 消融实验结果（Ablation Studies）

#### ✅ 调度有效性（§7.2）
- HEDDLE 的 progressive trajectory predictor 在第二步后即可达到：
  - **Tail 20% Recall > 0.8**
  - **Pearson Correlation > 0.7**
- 相比 model-based / history-based 方法更准确。
- 使用 PPS 后，最长轨迹的 queueing delay 下降 **~40–60%**，整体 rollout time 缩短 **1.1×–1.26×**。

#### ✅ 放置策略有效性（§7.3）
- 相比 Least-load 和 Cache-aware，HEDDLE 实现 **1.2×–1.5× 吞吐提升**。
- 成功避免了 long-tail 轨迹被集中调度到同一 worker 导致的“雪崩效应”。

#### ✅ 资源管理有效性（§7.4）
- 固定配置：
  - Fix-1（低 MP）：初期吞吐高，但长尾拖慢整体进度。
  - Fix-8（高 MP）：长尾快，但整体吞吐受限。
- HEDDLE 动态调配，在整个 rollout 过程中维持高效，最终实现 **1.1×–1.3× 超越两者**。

#### ✅ 系统开销分析（§7.5）
| 操作 | 开销（平均） | 是否掩盖 |
|------|-------------|---------|
| Prediction | ~0.1–0.3s | ✅ 并行于 tool execution |
| KV Cache Migration | ~0.1–0.35s | ✅ 利用 tool call 间隙异步传输 |
| Placement Algorithm | ~37ms | ✅ 远小于 rollout 时间（数百秒） |
| Resource Manager | ~5s | ✅ 周期性执行，摊销成本 |

> 所有控制平面开销均可忽略不计。

---

## 4. 关键结论和发现

### 主要发现
1. **Agentic RL 的 rollout 瓶颈本质是“长尾轨迹”引发的系统级问题**，不能仅靠算法或底层推理优化解决。
2. **trajectory-centric 设计是突破瓶颈的关键**：必须将轨迹作为调度、放置和资源配置的基本单位。
3. **HEDDLE 通过协同设计三大机制，系统性解决了 queueing、interference 和 per-token time 三重挑战**，实现高达 **2.5× 的吞吐提升**。
4. **动态预测 + 渐进式调度 + 异步迁移** 是实现高性能且低开销的核心技术组合。

### 方法的局限性
- **依赖高质量的 runtime predictor**：若初始预测严重错误，可能影响早期调度质量。
- **KV cache 迁移仍有一定带宽消耗**：在极端密集迁移场景下可能成为瓶颈（尽管实验中未观察到）。
- **当前假设工具执行可弹性扩展（serverless）**：若工具本身也成为瓶颈，则需额外协调。

### 未来工作方向
- 结合 **asynchronous RL**，进一步容忍部分 straggler，提升训练稳定性。
- 与 **speculative decoding** 或 **PD disaggregation** 正交集成，探索更细粒度的加速潜力。
- 探索 **multi-agent 协同 rollout** 场景下的扩展编排能力。
- 将 HEDDLE 的思想推广至其他具有长尾行为的 AI 工作流系统（如 planning, retrieval-augmented generation）。

---

> **总结一句话**：  
> HEDDLE 通过**以轨迹为中心的协同编排架构**，首次系统性解决了 Agentic RL 中由长尾轨迹引起的 rollout 效率瓶颈，实现了高达 **2.5× 的端到端吞吐提升**，为大规模自主智能体训练提供了高效的基础设施支持。

</details>

---

### 11. [Taming the Instability: A Robust Second-Order Optimizer for Federated Learning over Non-IID Data](https://arxiv.org/abs/2603.28316)

**Authors**: Yuanqiao Zhang, Tiantian He, Yuan Gao, Yixin Wang, Yew-Soon Ong, Maoguo Gong, A. K. Qin, Hui Li  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.28316v1  

#### Abstract
In this paper, we present Federated Robust Curvature Optimization (FedRCO), a novel second-order optimization framework designed to improve convergence speed and reduce communication cost in Federated Learning systems under statistical heterogeneity. Existing second-order optimization methods are of...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Taming the Instability: A Robust Second-Order Optimizer for Federated Learning over Non-IID Data

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**Federated Learning (FL)** 在统计异构（statistical heterogeneity）场景下存在的两大挑战：
- **收敛速度慢**：主流基于一阶优化的方法（如 FedAvg）忽略损失函数的曲率信息（curvature information），导致收敛缓慢，通信轮次多。
- **数值不稳定性**：直接将二阶优化方法（如 K-FAC）应用于 FL 时，在非独立同分布（non-IID）数据下会出现严重的数值不稳定问题，表现为梯度爆炸（gradient explosion）和客户端漂移（client drift），甚至导致发散。

### 提出的新方法：FedRCO
作者提出了一种名为 **Federated Robust Curvature Optimization (FedRCO)** 的新型二阶优化框架，其核心创新在于通过三个关键组件实现稳定且高效的联邦学习：

1. **Gradient Anomaly Monitor（梯度异常监测器）**  
   实时监控预处理后的梯度范数，计算“异常分数”（anomaly score）。当分数超过阈值时，识别出两种失败模式：
   - **Accumulated Divergence**（累积漂移）：渐进式漂移，由局部曲率与全局景观不匹配引起。
   - **Sudden Explosion**（突发爆炸）：因 Fisher 信息矩阵（FIM）接近奇异而导致梯度被过度放大。

2. **Fail-Safe Resilience Protocol（容错恢复协议）**  
   当检测到异常时触发恢复机制：
   - 对于累积漂移：采用软回滚（soft rollback），限制更新步长。
   - 对于突发爆炸或多轮持续异常：执行硬重置（hard reset），丢弃当前曲率统计量，重置本地模型为最新全局模型，并重新初始化优化器参数。

3. **Curvature-Preserving Adaptive Aggregation（保曲率自适应聚合策略）**  
   避免标准平均聚合破坏局部曲率几何结构。采用加权插值方式更新本地模型：
   $$
   \theta_{\text{local}}^{\text{new}} = \gamma \cdot \theta_{\text{global}} + (1-\gamma) \cdot \theta_{\text{old}}, \quad \gamma = \frac{\text{Acc}_{\text{local}}}{\text{Acc}_{\text{local}} + \text{Acc}_{\text{global}}}
   $$
   该策略在本地表现优异时保留更多局部信息，防止全局漂移抹除精确的局部二阶梯度方向。

此外，还采用了 **Lazy Inverse Update** 策略以降低计算开销，仅周期性地更新逆矩阵。

### 相比现有方法的优势
- **稳定性强**：首次系统分析并有效解决了二阶优化在 FL 中的不稳定性问题。
- **收敛快**：利用曲率信息显著加速收敛，减少通信轮次。
- **效率高**：通信复杂度与 FedAvg 相同（$O(d)$），计算复杂度可控（$O(d_{\text{in}}d_{\text{out}}(d_{\text{in}}+d_{\text{out}})/T_{\text{inv}})$）。
- **鲁棒性好**：在极端 non-IID 场景下仍能保持高性能。

---

## 2. 核心实验方法和设置

### 数据集
- **CIFAR-10**：用于图像分类任务。
- **EMNIST**：扩展的 MNIST 数据集，包含手写字符和数字，类别更多（62类），更具挑战性。

### 数据划分策略（模拟 non-IID）
1. **Dirichlet 分布划分**：控制浓度参数 $\alpha \in \{0.1, 0.5, 1.0\}$，$\alpha=0.1$ 表示极不平衡的标签分布。
2. **Pathological 划分**：每个客户端仅分配固定数量的类别（CIFAR-10: 2 或 5 类；EMNIST: 10 或 30 类）。
3. **IID 划分**：作为对照组。

### 实验设置
- 客户端数量：$\{10, 50, 100\}$
- 参与比例（party ratio）：$\{0.1, 0.5, 0.8, 1.0\}$
- 通信轮次：1600
- 每轮本地训练轮数（local epochs）：20
- 批大小（batch size）：32
- 学习率：0.00625
- 模型架构：轻量级 CNN（适用于边缘设备）

### 评估指标
- **测试准确率（Test Accuracy）**
- **训练损失（Training Loss）**
- **收敛速度（Wall-clock time to target accuracy）**
- **通信轮次（Communication Rounds）**

### 基线方法对比
- **一阶方法**：
  - FedAvg
  - FedAvgM
  - FedProx
  - FedAdam
- **二阶方法**：
  - LocalNewton
  - FedPM
- **消融版本**：
  - FedRCO-ori：使用简单平均而非提出的聚合策略

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | CIFAR-10 ($\alpha=0.1$) | EMNIST ($\alpha=0.1$) |
|------|--------------------------|------------------------|
| FedAvg | 56.3% | 81.8% |
| FedProx | 55.3% | 81.7% |
| FedAdam | 55.7% | 84.5% |
| LocalNewton | 50.9% | 80.4% |
| FedPM | 54.7% | 80.5% |
| **FedRCO-ori** | 69.5% | 85.4% |
| **FedRCO** | **78.8%** | **90.2%** |

> ✅ 在最极端的 non-IID 设置下（Dir($\alpha=0.1$)），FedRCO 显著优于所有基线。

### 与基线方法的对比结果
- **显著提升准确性**：
  - 在 CIFAR-10 上比 FedAvg 提升 **+22.5%**。
  - 在 EMNIST 上比 FedAvg 提升 **+8.4%**。
- **更快收敛**：
  - 如 Fig. 2(b) 所示，FedRCO 在 **1000 秒内达到 70% 准确率**，而 FedAvg 和 FedProx 需要超过 **10,000 秒**。
- **更少通信轮次**：由于每轮进展更大，达到相同性能所需通信轮次大幅减少。
- **更强鲁棒性**：在 Pathological 和低参与率（0.1）等严苛条件下依然保持领先。

### 消融实验结果
1. **聚合策略有效性**：
   - FedRCO vs FedRCO-ori：在 CIFAR-10 上从 69.5% 提升至 78.8%，证明所提聚合策略对保留局部几何结构至关重要。
2. **反向更新频率 $T_{\text{inv}}$ 影响**：
   - 如 Fig. 3 所示，$T_{\text{inv}}=200$ 效果最佳，过频（如 20）或过懒（如 500）均会降低性能。
   - 说明 **Lazy Inverse Update 不仅是计算优化，更是性能稳定器**。
3. **梯度监控必要性**：
   - 如 Fig. 4 所示，无监控时梯度剧烈波动甚至爆炸；加入监控后梯度被有效约束在稳定范围内。
4. **可扩展性验证**：
   - 在不同客户端数量（10–100）和参与率（0.1–1.0）下，FedRCO 均保持领先，展现出良好可扩展性。

---

## 4. 关键结论和发现

### 主要发现
- **二阶优化在 FL 中潜力巨大但需谨慎使用**：K-FAC 能有效改善病态条件数（condition number），理论上可极大加速收敛（Proposition 3.3–3.4）。
- **不稳定性根源明确**：Rank Deficiency（小批量导致 FIM 秩亏）和 Curvature Mismatch（non-IID 导致局部/全局曲率失配）是两大主因（Proposition 3.1–3.2）。
- **FedRCO 成功平衡了效率与稳定**：通过 Gradient Monitor + Resilience Protocol + Adaptive Aggregation 三位一体机制，实现了稳定、快速、高效的联邦训练。
- **理论与实践一致**：理论分析（Theorems 5.1–5.4）证明 FedRCO 可控 client drift 并保证收敛至最优解邻域，实验结果充分验证了这一点。

### 方法的局限性
- **依赖于 K-FAC 架构假设**：目前实现基于全连接层和卷积层的 K-FAC 近似，对其他网络结构（如 Transformer）的支持需进一步研究。
- **超参数敏感性**：虽然整体鲁棒，但异常检测阈值（$T_{\text{low}}, T_{\text{high}}$）、阻尼系数（$\epsilon$）等仍需合理设置。
- **计算资源要求高于纯一阶方法**：尽管已优化，但仍比 FedAvg 多出约 6.4% 的计算时间（Fig. 2a），在极低端设备上可能受限。

### 未来工作方向
- 将 FedRCO 扩展至 **个性化 Federated Learning** 场景。
- 探索 **动态调整 $T_{\text{inv}}$ 和 damping 参数** 的自适应机制。
- 结合 **差分隐私（DP）或安全聚合（Secure Aggregation）**，构建更全面的隐私保护 FL 框架。
- 应用于更大规模模型（如 Vision Transformers）和真实世界边缘设备部署。

> 📌 **总体评价**：FedRCO 是首个系统解决二阶优化在 FL 中不稳定性的框架，兼具理论深度与工程实用性，为高效、鲁棒的分布式机器学习提供了重要新路径。

</details>

---

### 12. [SARL: Label-Free Reinforcement Learning by Rewarding Reasoning Topology](https://arxiv.org/abs/2603.27977)

**Authors**: Yifan Wang, Bolian Li, David Cho, Ruqi Zhang, Fanping Sui, Ananth Grama  
**Category**: cs.AI  
**Published**: 2026-03-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.27977v1  

#### Abstract
Reinforcement learning has become central to improving large reasoning models, but its success still relies heavily on verifiable rewards or labeled supervision. This limits its applicability to open ended domains where correctness is ambiguous and cannot be verified. Moreover, reasoning trajectorie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SARL: Label-Free Reinforcement Learning by Rewarding Reasoning Topology

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Reinforcement Learning from Verifiable Rewards (RLVR)** 范式严重依赖于可验证的任务（如数学题、代码生成），需要明确的 `ground truth` 或自动验证器来提供奖励信号。这导致其在**开放域任务**（open-ended tasks）中失效，例如创意写作、战略规划、哲学推理等，因为这些任务缺乏客观正确答案。

此外，传统 RL 主要优化最终输出（outcome-based），忽视了对中间推理过程（reasoning trajectory）的控制，容易导致模型过早利用（early exploitation）而非泛化。

### 提出了什么新方法或新思路
本文提出 **Structure Aware Reinforcement Learning (SARL)** ——一种无需标签的强化学习框架，其核心思想是：

> **不奖励“答案是否正确”，而是奖励“思考方式是否合理”**。

具体来说：
- 从每个生成的 `<think>` 块中提取中间推理步骤（reasoning steps）。
- 构建一个 **Reasoning Map**：将语义相似的推理步骤聚类为节点，连续不同类型的步骤之间建立边，形成图结构。
- 定义 **Structure Reward (SR)**，衡量该图的 **small-world topology** 特性：
  - **局部连通性**（Local Clustering Coefficient）：反映功能模块内的紧密协作。
  - **全局效率**（Average Shortest Path Length）：反映跨模块的信息传递效率。
- 最终奖励定义为：  
  $$
  SR(G) = \frac{2 - C(G)}{1 + L(G)}
  $$  
  鼓励既**局部一致**又**全局高效**的推理路径。

### 相比现有方法的优势
| 方法 | 是否需要 GT | 是否适用于开放域 | 是否关注推理结构 | 算法通用性 |
|------|-------------|------------------|--------------------|------------|
| RLVR (e.g., PPO w/ GT) | ✅ 是 | ❌ 否 | ❌ 黑箱优化结果 | ⭕ 可用多种算法 |
| EMPO [Entropy Min.] | ❌ 否 | ✅ 是 | ❌ 仅降低不确定性 | ❌ 通常限于 GRPO |
| TTRL [Majority Vote] | ❌ 否 | ❌ 否（需共识答案） | ❌ 结果导向 | ❌ 限于 GRPO |
| **SARL (Ours)** | ❌ 否 | ✅ 是 | ✅ 显式建模并优化结构 | ✅ 兼容 PPO 和 GRPO |

- **首次实现完全无监督的结构化 RL**，摆脱对 `ground truth` 的依赖。
- **奖励信号来自内部推理拓扑**，而非外部反馈，更具生物学合理性（受人脑小世界网络启发）。
- **算法无关性**（algorithm-agnostic）：可在 PPO 和 GRPO 上稳定训练，适用性更广。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 设置 | 数据集 | 描述 |
|------|--------|------|
| **Verifiable Reasoning** | **AIME 1983–2024** | 数学竞赛题，共约 1,000 道，要求多步符号推理，有确定解，用于闭集任务评估。 |
| **Non-verifiable Reasoning** | **OpenRubrics-v2** | 包含创意写作、计划制定、编程建议、信息咨询等多样化开放任务，仅有偏好标注，无标准答案。 |

### 实验设置和评估指标
- **模型**：基于 **Qwen3-4B** 进行 post-training。
- **训练框架**：
  - RL 实验使用 `veRL` 框架；
  - DPO 使用 `LlamaFactory`。
- **超参数**：
  - Rollouts per prompt: 8
  - Batch size: 256
  - 使用 FSDP2 并行，在单节点 8×A100 上完成。

#### 评估指标
| 任务类型 | 评估基准 | 主要指标 |
|---------|----------|----------|
| 数学任务 | MATH500, AIME25, AMC23, Minerva Math | **avg@8**（多次采样下的期望准确率） |
| 开放任务 | **WildBench Leaderboard** | **WB Score (Elo rating)** + 分类宏平均得分 |

### 基线方法对比
| 类型 | 基线方法 |
|------|--------|
| **Label-free RL** | EMPO（熵最小化）、TTRL（多数投票） |
| **Preference-based** | DPO（使用偏好数据） |
| **Oracle（上限）** | Ground-Truth RL（PPO/GRPO with answer verification） |

---

## 3. 主要实验结果和性能指标

### 数学任务表现（Verifiable Domain）
> 在 AIME 等挑战性数学任务上，SARL 不仅媲美甚至超越了基于真实标签的 RL！

| 方法 | AIME25 ↑ | AMC23 ↑ | MATH500 ↑ | Minerva ↑ | **Avg Δ** |
|------|--------|--------|----------|----------|-----------|
| Base | 31.67 | 82.81 | 90.10 | 53.45 | 64.51 |
| PPO w/ GT | 41.67 | 86.56 | 92.75 | 59.74 | **+5.67** |
| **PPO w/ SARL** | **42.92** | 85.00 | 92.53 | 61.08 | **+5.87** ✅ |
| GRPO w/ GT | 46.67 | 84.38 | 93.15 | 62.45 | **+7.15** |
| **GRPO w/ SARL** | **45.83** | **87.50** | **93.30** | **61.99** | **+7.65** ✅ |

- **关键发现**：
  - SARL 在 **AIME25** 上提升最大（+14.16 pts），说明复杂长链推理更能受益于结构约束。
  - 即使没有 ground truth，SARL 在 GRPO 下仍达到 **最佳平均增益 +7.65%**，超过 oracle GT-RL。
  - 是唯一同时兼容 **PPO 和 GRPO** 的 label-free 方法。

### 开放任务表现（Open-Ended Domain）
> 在无标准答案的任务中，SARL 显著优于所有基线，证明其可扩展至真实世界复杂场景。

| 方法 | Creative | Planning | Math | Info | Code | **WB Score Δ** |
|------|--------|--------|-----|-----|-----|----------------|
| Base | 51.01 | 36.23 | 16.35 | 48.71 | 14.72 | 29.91 |
| DPO | 51.16 | 37.82 | 17.06 | 48.37 | 14.34 | **+0.43** |
| EMPO | 51.01 | 36.11 | 17.62 | 45.69 | 12.92 | **-0.71** |
| **PPO w/ SARL** | **57.05** | **45.87** | **27.70** | **53.47** | **30.05** | **+10.35** ✅ |
| **GRPO w/ SARL** | 55.34 | 43.56 | 26.83 | 51.98 | 29.91 | **+9.10** ✅ |

- **关键发现**：
  - 在 **Creative Writing (+6.04)** 和 **Planning (+9.64)** 上取得巨大进步，表明结构先验有助于组织非形式化思维。
  - DPO 效果微弱（+0.43），说明偏好数据不足以驱动显著改进。
  - EMPO 表现下降，显示单纯减少熵可能导致退化。

### 消融与分析实验（Ablation & Analysis）

#### 📊 训练稳定性分析（Figure 2）
- **KL Divergence 更低**：SARL 引起的策略漂移最小，接近零，训练更稳定。
- **Policy Entropy 更高**：保持探索能力，避免早熟收敛（premature exploitation）。

> → SARL 实现了“**低扰动 + 高探索**”的理想训练动态。

#### 📏 输出长度分析（Table 4 & 8）
| 方法 | 平均 Token 数 | 相比 Base 变化 |
|------|---------------|----------------|
| Base (Math) | 5736.6 | 0.0 |
| SARL (PPO) | 5075.6 | **-661.0** |
| SARL (GRPO) | 4587.8 | **-1148.8** |
| Base (Open) | 2677.91 | 0.0 |
| SARL (PPO) | 2303.17 | **-374.74** |

- **SARL 生成更短但更高效的推理链**，排除“靠堆长度提分”的质疑。
- 性能提升源于**质量而非数量**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **推理结构本身可以作为有效的自监督信号**：无需外部标签，仅通过优化 reasoning map 的 small-world topology 就能显著提升模型能力。
2. ✅ **SARL 在闭集和开集任务上均有效**：
   - 数学任务：媲美甚至超越 ground-truth RL（最高 +7.65% avg gain）。
   - 开放任务：WildBench 上提升高达 **+10.35 WB Score**，远超 DPO 和 EMPO。
3. ✅ **带来更稳定、更具探索性的训练过程**：
   - 更低 KL divergence → 更少策略漂移。
   - 更高 policy entropy → 更强泛化潜力。
4. ✅ **生成更简洁高效的推理路径**：性能提升不是靠增加长度，而是提高组织效率。

### 方法的局限性
- 对**极简单任务**（few-step reasoning）效果有限，因结构组织空间小。
- 当前仅采用统一的 **small-world prior**，可能并非所有领域最优。
- 聚类方法（KMeans/HDBSCAN）引入额外计算开销，虽可通过独立 embedding server 缓解。

### 未来工作方向
- 探索更多样的 **domain-specific structural priors**（如树状、层次化、循环结构）。
- 动态调整结构目标，适应不同类型任务。
- 将 SARL 应用于更大规模模型（如 Qwen3-72B）或多模态推理。
- 结合过程奖励模型（PRM）进行混合监督，进一步提升性能。

---

## 🔚 总结
**SARL 开启了一条全新的强化学习路径：不再问“答得对不对”，而是问“想得清不清楚”。**

它通过构建每条响应的 **Reasoning Map** 并奖励其 **small-world topology**，实现了真正意义上的 **label-free, verifier-free, open-domain capable** 的 reasoning optimization。实验证明，这种结构感知的训练不仅能匹配甚至超越依赖 ground truth 的传统 RL，还能在开放任务中释放巨大潜力，推动 LLM 向更通用、更鲁棒的推理智能体迈进。

</details>

---

### 13. [Differentiable Power-Flow Optimization](https://arxiv.org/abs/2603.28203)

**Authors**: Muhammed \"Oz, Jasmin H\"orter, Kaleb Phipps, Charlotte Debus, Achim Streit, Markus G\"otz  
**Category**: cs.AI  
**Published**: 2026-03-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.28203v1  

#### Abstract
With the rise of renewable energy sources and their high variability in generation, the management of power grids becomes increasingly complex and computationally demanding. Conventional AC-power-flow simulations, which use the Newton-Raphson (NR) method, suffer from poor scalability, making them im...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Differentiable Power-Flow Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统 AC power-flow 仿真（如 Newton-Raphson, NR）在面对大规模、高复杂度电网（如含分布式能源的联合输配电网）时面临**可扩展性差**的问题：
- NR 方法依赖雅可比矩阵（Jacobian）的显式构造与求逆，计算和内存开销随系统规模呈超线性增长；
- 数据驱动的 surrogate 模型虽快，但可能违反物理约束，缺乏泛化能力。

随着可再生能源渗透率提高，对**时间序列分析、N-1 潮流扫描、动态仿真**等高频次、多场景仿真的需求激增，传统方法难以满足效率要求。

---

### **提出了什么新方法或新思路**
提出 **Differentiable Power-Flow (DPF)** —— 一种将 AC power-flow 问题重构为**可微分模拟（differentiable simulation）** 的新范式。

#### **核心思想**：
- 将 power-balance 方程 $ S_{\text{bus}} = V(Y_{\text{bus}}V)^* $ 构建为一个**端到端可微的计算图**；
- 使用 **gradient-based optimization**（如 Adam、SGD）最小化功率残差损失函数 $ \mathcal{L}(V) = \|S_{\text{bus}} - S_{\text{calc}}\|^2 $；
- 利用现代 ML 框架（如 PyTorch）的 **automatic differentiation (autodiff)**、**GPU 加速**、**batching** 和 **sparse tensor 支持**实现高效并行计算。

---

### **相比现有方法的优势**
| 特性 | DPF | NR | DC Approximation | Data-driven Models |
|------|-----|----|------------------|--------------------|
| **物理一致性** | ✅ 完全保留物理方程 | ✅ 高精度 | ❌ 忽略无功、电阻等 | ⚠️ 可能违反约束 |
| **可扩展性** | ✅ 线性时间/内存增长 | ❌ 超线性增长 | ✅ 快但不准确 | ✅ 推理快 |
| **硬件加速** | ✅ 支持 GPU + batching | ⚠️ 有限支持 | ✅ 易并行 | ✅ 易部署 |
| **多任务适用性** | ✅ 时间序列、N-1 分析天然适配 | ⚠️ 单次求解优化 | ✅ 用于筛查 | ✅ 适合稳态预测 |

> ✅ **DPF 在保持物理一致性的前提下，实现了更好的可扩展性和工程集成性**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **标准测试系统**：
  - `IEEE-118`（118 buses）
  - `case9241pegase`（9,241 buses，欧洲电网模型）
- **合成扩展数据**：
  - 通过复制 `case9241pegase` 并添加随机连接，生成更大规模网络（达数十万节点），用于验证缩放行为。
- **数据来源**：
  - LIPS benchmark suite（Learning Industrial Physical Simulation）
  - LightSim2Grid 提供的标准 grid 数据

---

### **实验设置和评估指标**

#### **硬件环境**
- GPU：NVIDIA A100-40GB × 4
- CPU：Intel Xeon Platinum 8368
- 框架：PyTorch + CUDA 12.4，使用 **sparse CSR 格式存储 $ Y_{\text{bus}} $**

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **Runtime per power-flow** | 单次潮流计算耗时（ms） |
| **Convergence iterations** | 达到收敛所需的迭代次数 |
| **Solution quality** | 与 NR 解之间的电压幅值/相角误差 |
| **Scaling behavior** | 运行时间随节点数/边数的增长趋势（线性 vs 二次） |
| **Memory usage** | 额外内存占用（尤其关注 Jacobian 存储） |

#### **训练策略**
- 使用 **Adam + ReduceLROnPlateau** 调度器
- 初始学习率通过 Optuna 超参搜索获得
- 支持 warm-start（复用前一时步解作为初值）

---

### **基线方法对比**
| 方法 | 类型 | 说明 |
|------|------|------|
| **Newton-Raphson (NR)** | 数值迭代法（二阶） | 工业标准，高精度，但扩展性差 |
| **DC Power-Flow** | 线性近似 | 快速筛查工具，忽略非线性项，误差大（~20%） |
| **FDLF / GS / SA** | 其他数值方法 | 文中提及但未重点比较 |
| **GNN-based surrogates (e.g., GNS, KCLNet)** | 数据驱动 | 快但可能输出物理不可行解 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **单次潮流求解（Small Grids）**
| 方法 | IEEE-118 (CPU) | case9241pegase (CPU) |
|------|---------------|------------------------|
| NR | **0.12 ms** | **12.37 ms** |
| DPF (base) | ~800 ms (1000 iters) | ~5 s |
| DC | <1 ms | <1 ms |

> 🔴 **小网格上 DPF 慢于 NR**，因其需数百至上千次梯度更新。

#### **时间序列场景（Warm-start + Batching）**
- **优势显现**：利用前一时步解作为初始化，迭代数从 1000 → **100**
- **Batching 加速显著**：
  - 在 `case9241pegase` 上，batch size=64 时：
    - 单次潮流时间从 **2 ms → 0.45 ms**
    - 吞吐量提升约 **4.4 倍**

#### **缩放行为（Scaling Behavior）**
| 方法 | 时间复杂度 | 内存复杂度 |
|------|-----------|------------|
| NR | $ O(\text{nnz}(J)^{1.5} \sim N^2) $ | $ O(N^2) $（因 fill-in） |
| DPF | $ O(\text{nnz}(Y_{\text{bus}})) \sim O(N) $ | $ O(N) $ |

- 图 8 显示：当网络规模扩大至百万级节点时，
  - **NR 运行时间急剧上升（接近二次）**
  - **DPF 几乎保持线性增长**

> ✅ **DPF 在超大规模系统中展现出明显优势**

#### **精度表现**
- DPF 解的质量介于 **NR 与 DC 之间**：
  - 电压幅值误差 < 1%
  - 相角误差较小
- 不需要完全收敛即可用于**快速筛查**（early stopping）

---

### **消融实验结果**
- **Warm-start 复用**：使迭代数减少 **90%**
- **Batching**：显著提升 GPU 利用率，降低单位成本
- **Sparse tensor 表示**：节省内存，避免稠密化带来的爆炸
- **不同 optimizer 对比**：Adam、SGD、RMSprop 性能相近，最终选择 Adam 因其鲁棒性更好

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **DPF 是一种物理保真、可扩展的潮流求解新范式**：
   - 无需牺牲物理一致性即可实现高效计算；
   - 特别适用于**大规模、多场景、高频次**的应用。

2. ✅ **在特定应用场景下超越传统方法**：
   - **时间序列仿真**：warm-start + batching 使其效率大幅提升；
   - **N-1 安全分析**：天然支持批量处理多个故障场景；
   - **动态操作模拟**：中间解可用于快速筛查。

3. ✅ **理论优势在超大规模系统中得以体现**：
   - 当节点数达到 **数十万至百万级**（如联合输配电网），DPF 的线性缩放特性将使其反超 NR。

4. ✅ **易于实现与集成**：
   - 基于 PyTorch 实现，代码开源；
   - 可无缝接入 ML pipeline，支持自动微分用于逆问题求解（如参数识别）。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **小网格性能劣势** | 在 <10k 节点系统中仍慢于优化后的 NR（如 LightSim2Grid） |
| **收敛速度较慢** | 需要更多迭代（线性收敛 vs NR 的二次收敛） |
| **依赖良好初始化** | 若初值远离真实解，收敛更慢 |
| **尚未支持所有约束** | 如暂未处理电压越限、线路热限等 inequality constraints |

---

### **未来工作方向**
1. **进一步优化实现**：
   - 更高效的 optimizer 设计（如 preconditioning）
   - 自适应学习率与 early stopping 策略
2. **分布式与并行化扩展**：
   - 在超大规模系统上采用分布式训练
3. **结合 hybrid 方法**：
   - 用 DPF 初始化 NR，提升整体稳定性与速度
4. **拓展应用领域**：
   - 实时调度、OPF（Optimal Power Flow）、故障定位
5. **集成到工业平台**：
   - 与 Grid2Op、Pandapower 等框架深度整合

---

> 🎯 **总结一句话**：  
> **DPF 并非要取代 NR，而是为“超大规模 + 多场景 + 快速响应”的新兴电力系统分析需求提供了一个可扩展、物理一致且软硬协同的新工具链**。

</details>

---

### 14. [jaxsgp4: GPU-accelerated mega-constellation propagation with batch parallelism](https://arxiv.org/abs/2603.27830)

**Authors**: Charlotte Priestley, Will Handley  
**Category**: cs.DC  
**Published**: 2026-03-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.27830v1  

#### Abstract
As the population of anthropogenic space objects transitions from sparse clusters to mega-constellations exceeding 100,000 satellites, traditional orbital propagation techniques face a critical bottleneck. Standard CPU-bound implementations of the Simplified General Perturbations 4 (SGP4) algorithm ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：`jaxsgp4: GPU-accelerated mega-constellation propagation with batch parallelism`

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着低地球轨道（LEO）中卫星数量迅速增长，尤其是由数万乃至百万级卫星组成的**mega-constellations**（如Starlink），传统的轨道传播方法面临严重的计算瓶颈。当前主流的 **SGP4** 模型多基于 CPU 实现（如C++版本），采用串行或有限多线程执行方式，在处理大规模星座的碰撞预警（conjunction assessment）和空间态势感知（Space Situational Awareness, SSA）任务时效率极低。

### 🚀 提出的新方法与创新思路
本文提出 **jaxsgp4** —— 一个基于 **JAX** 框架的高性能、纯函数式重构的 SGP4 实现，其核心创新在于：
- 将传统过程式、状态依赖的 SGP4 算法重写为**纯函数范式**（pure functional paradigm），以兼容 JAX 的自动变换系统；
- 利用 JAX 提供的关键能力实现**批量并行化**（batch parallelism）：
  - `jax.jit`：Just-In-Time 编译优化
  - `jax.vmap`：自动向量化（支持对卫星和时间两个维度同时批处理）
  - 支持在 GPU/TPU/CPU 上无缝运行，无需修改代码
- 首次系统性地探索了 **32-bit 浮点精度**（FP32）在 SGP4 中的应用可行性，提出“精度换吞吐”的合理权衡。

### 🔍 相比现有方法的优势
| 维度 | jaxsgp4 | 传统 C++ SGP4 / OSGP4 |
|------|--------|------------------------|
| 并行能力 | ✅ 支持大规模 batch parallelism（卫星 × 时间） | ❌ 主要串行或轻量多线程 |
| 硬件加速 | ✅ 原生支持 GPU/TPU 加速（通过 XLA） | ❌ 仅限 CPU |
| 内存效率 | ✅ O(N + M) 内存扩展（N: 卫星数, M: 时间步） | ⚠️ OSGP4 为 O(NM)，易内存溢出 |
| 自动微分 | ✅ 完全可微分（支持 `jax.grad`, `jax.jacobian`） | ⚠️ OSGP4 虽可微但未充分优化并行梯度 |
| 性能表现 | ⏱️ 单次传播延迟极低（<4ms 全星座） | ⏱️ 数百秒至小时级 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用来自 [Celestrak](https://celestrak.com) 的真实 **Starlink TLE 数据集**（共 9,341 颗卫星），历元时间为 2026年1月13日。
- 为测试极限规模，构建人工目录：将 Starlink TLE 数据平铺（tiling）生成最多达 **183万颗卫星**的人工星座，用于压力测试。

### ⚙️ 实验设置
| 项目 | 设置说明 |
|------|----------|
| **硬件平台** | - CPU 基线：Intel Xeon @ 2.20GHz<br>- GPU 对比：<br> • NVIDIA Tesla T4（16GB GDDR6）<br> • NVIDIA A100-SXM4（40GB HBM2） |
| **软件环境** | Google Colab 云端环境，使用 JAX + XLA 后端 |
| **精度配置** | - jaxsgp4：默认使用 **FP32**（32位浮点）<br>- C++ 基线：使用 **FP64**（64位双精度） |
| **评估指标** | - 单次完整传播耗时（wall-clock time）<br>- 相对于 C++ 基线的 **speedup 倍数**<br>- 位置与速度误差（vs. FP64 基线）<br>- 内存占用与扩展性分析 |

### 🔁 基线方法对比
- **主基线**：官方 C++ 版本 SGP4（源自 [Vallado et al., 2006]，即 SpaceTrack Report #3），通过 `sgp4` Python 包调用（v2.25）
- **次要对比**：PyTorch 实现的 **OSGP4**（Acciarini et al., 2025），也支持 GPU 和自动微分，但侧重于可微性而非并行优化

> 注：所有 timing 测量均排除数据传输开销（预加载到 GPU）和首次编译开销（warm-up），反映稳态性能。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**极致加速效果**
- 在单块 **NVIDIA A100 GPU** 上：
  - 可在 **3.8 ms 内完成整个 Starlink 星座（9,341 颗卫星）向未来 1,000 个时间点的传播**
  - 相比传统 C++ 实现，达到 **超过 1500× 的加速比**（最高达 1592×）
- 在较便宜的 **T4 GPU** 上仍可达 **~250× 加速**

#### （2）**并行扩展特性**
- 展现出典型的 **flat scaling regime（平坦缩放区）**：
  - 当批量大小低于硬件饱和阈值时，增加任务量几乎不增加运行时间
  - 例如：在 A100 上，从 1 颗到 100,000 颗卫星的传播耗时基本不变
- “Break-even” 点出现在约 **300–500 次独立传播任务**，之后 GPU 开始显著优于 CPU

#### （3）**内存效率优势**
- jaxsgp4 内存复杂度为 **O(N + M)**，而 OSGP4 为 **O(N × M)**
- 这使得 jaxsgp4 能够处理更大规模的任务，避免因中间张量过大导致 GPU 内存溢出

#### （4）**32-bit 精度误差分析**
| 指标 | 结果 |
|------|------|
| 初始时刻位置偏差 | ~1 米 |
| 两周后位置误差中位数 | < 1 km |
| 95% 分位最坏情况 | ~2 km/周增长 |
| 速度误差 | 最大几 m/s（相对误差极小） |

> ✅ 重要发现：这些数值误差远小于 SGP4 模型本身固有的物理误差（通常为 **1 km@epoch，每日恶化 1–3 km**）。因此，使用 FP32 是合理的工程取舍。

#### （5）与其他可微实现对比
- 与 OSGP4 相比，**jaxsgp4 在相同任务下快 >10×**
- 支持更高效的 **batched gradients**（结合 `jax.grad` + `jax.vmap`），适用于大规模轨道优化

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **GPU 批量并行是应对 mega-constellation 计算挑战的根本出路**  
   传统 CPU 串行模式无法满足未来百万级对象的实时 SSA 需求；而利用现代 GPU 的数千核心进行 batch parallelism 可带来 **三个数量级的速度提升**。

2. **32-bit 精度完全可行且性价比极高**  
   在 SGP4 的背景下，FP32 引入的量化误差（米级）远小于模型本身的物理不确定性（公里级），却能大幅提升 GPU 吞吐量（更高 FLOPS 和带宽利用率）。

3. **功能范式重构解锁多重能力**  
   将 SGP4 改造为纯函数形式不仅便于 JIT 编译和向量化，还天然支持 **automatic differentiation**，为后续集成机器学习、梯度优化等高级应用打开通道。

4. **成本效益逆转：GPU 更贵但更便宜**  
   尽管 A100 实例每小时价格约为 CPU 的 10 倍，但由于 **1500× 加速**，完成同等任务的总成本反而降低约 **150 倍**。对于大规模作业，GPU 成为更经济的选择。

---

### ⚠️ 方法的局限性
1. **目前仅支持近地轨道（LEO）卫星**（周期 < 225 分钟）  
   尚未包含 SDP4（Deep Space Perturbations）模块，暂不适用于 GEO 或高椭圆轨道卫星。

2. **FP32 下 TLE 历元编码存在系统性误差**  
   当前将历元存储为“年积日 + 小数天”的单一 FP32 值，整数部分占用过多有效位，可能导致最大约 **1 秒的时间偏移**（对应 LEO 中 ~7 km 位置误差）。  
   ➤ 建议解决方案：拆分为整数日 + 小数日字段，或直接输入“自历元起经过分钟数”。

3. **依赖高质量 TLE 输入**  
   本工作未改进 SGP4 模型本身的精度缺陷，仍受限于 TLE 的平均根数表示及其建模简化（如大气阻力粗略估计）。

---

### 🔮 未来工作方向
1. **集成 SDP4 深空摄动模型**，扩展至 MEO/GEO 轨道
2. **应用于高保真半解析模型**（如 SGP4-XP）或数值积分器，结合 JAX 实现高速高精度传播
3. **科学应用拓展**：
   - Kessler Syndrome 的蒙特卡洛模拟
   - 天文观测污染预测（需密集时空网格传播）
   - 基于梯度的星座构型优化（最小化碰撞风险、最大化覆盖）
4. **部署到实际 SSA 系统**，如 Space Fence 或商业追踪平台

---

> **开源信息**：  
> - 项目地址：[https://github.com/cmpriestley/jax_sgp4](https://github.com/cmpriestley/jax_sgp4)  
> - 实验复现资源（脚本、数据、绘图代码）已归档于 Zenodo：[DOI:10.5281/zenodo.19322209](https://doi.org/10.5281/zenodo.19322209)

</details>

---

### 15. [Match or Replay: Self Imitating Proximal Policy Optimization](https://arxiv.org/abs/2603.27515)

**Authors**: Gaurav Chaudhary, Laxmidhar Behera, Washim Uddin Mondal  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.27515v1  

#### Abstract
Reinforcement Learning (RL) agents often struggle with inefficient exploration, particularly in environments with sparse rewards. Traditional exploration strategies can lead to slow learning and suboptimal performance because agents fail to systematically build on previously successful experiences, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Match or Replay: Self Imitating Proximal Policy Optimization*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
强化学习（Reinforcement Learning, RL）在稀疏奖励（sparse rewards）和部分可观测环境（partially observable environments）中面临严重的**探索效率低**和**样本效率差**问题。传统探索策略往往缺乏对过去成功经验的有效利用，导致学习缓慢甚至陷入局部最优。

此外，现有的自模仿学习（Self-Imitation Learning, SIL）方法多基于**off-policy算法**，难以直接应用于主流的on-policy算法如PPO，且通常依赖外部专家演示或修改奖励函数，限制了其通用性和稳定性。

---

### 提出的新方法与创新思路
本文提出了一种全新的**on-policy自模仿框架**：**Self-Imitating Proximal Policy Optimization (SIPP)**，并设计了两种策略分别应对不同类型的奖励环境：

- **MATCH策略（用于密集奖励环境）**  
  利用**最优传输距离（Optimal Transport Distance）** 和**Sinkhorn算法**，衡量当前策略的状态分布与历史最高回报轨迹之间的相似性，并优先采样与高奖励轨迹匹配度高的状态-动作对进行训练，从而引导探索向高价值区域集中。

- **REPLAY策略（用于稀疏/二值奖励环境）**  
  维护一个**模仿缓冲区（imitation buffer）** 存储成功的完整轨迹，在训练时以一定概率从该缓冲区中重放这些高回报轨迹，实现对长程依赖行为的强化学习。

> ✅ **核心创新**：首次将自模仿机制无缝集成到纯on-policy框架（PPO）中，无需引入off-policy修正、目标网络或额外模型。

---

### 相比现有方法的优势
| 对比维度 | SIPP | 其他方法（如SIL, SVPG等） |
|--------|------|--------------------------|
| **算法范式兼容性** | 完全on-policy，与PPO原生兼容 | 多为off-policy或混合模式 |
| **是否修改奖励函数** | ❌ 不修改真实奖励 | 部分方法通过加权优势项改变奖励 |
| **是否需要外部专家** | ❌ 仅使用自身历史成功经验 | 有些需专家示范 |
| **理论一致性** | 保持PPO的稳定性和理论保证 | SIL与on-policy缺乏强理论连接 |
| **适用场景广度** | 支持dense/sparse rewards + low/high-dim states + fully/partially observable MDPs | 多限于低维、完全可观测环境 |

---

## 2. 核心实验方法和设置

### 使用的数据集与任务环境
实验覆盖多种复杂且具有挑战性的RL基准环境：

| 类型 | 环境名称 | 描述 |
|------|--------|------|
| **密集奖励连续控制** | MuJoCo (10 tasks) | 包括`Ant-v4`, `HalfCheetah-v4`, `Humanoid-v4`等，提供平滑密集反馈 |
| **稀疏奖励导航任务** | PointMaze (multi-goal) | 二维迷宫中目标随机变化，仅到达终点时获得稀疏奖励 |
| **部分可观测3D环境** | Animal-AI Olympics (5变体) | 第一人称视角3D竞技场，含障碍物、隧道、可移动箱子等，仅达成目标有二值奖励 |

---

### 实验设置与评估指标

#### 模型架构
- **MuJoCo / PointMaze**: MLP（两层64单元，tanh激活）
- **Animal-AI**: CNN（三层卷积，输入为4帧堆叠的84×84 RGB图像）

#### 关键超参数
- **IET系数（Imitation-Exploration Trade-off, ε）**：控制探索与模仿的比例
  - MATCH中决定采样方式（均匀 vs OT加权）
  - REPLAY中决定是否从imitation buffer中采样轨迹
- **IMITATION BUFFER SIZE**
  - MATCH: 1（只保留最佳轨迹）
  - REPLAY: 10（存储多个成功轨迹）

#### 评估指标
- **Episodic Return / Reward**: 每轮episode的累计奖励
- **Success Rate**（Animal-AI）: 成功抵达目标的比例
- 所有结果报告**7种子（seeds）均值 ± 标准差**

---

### 基线方法对比
| 基线 | 简要说明 |
|------|---------|
| **PPO** | Vanilla Proximal Policy Optimization，无任何模仿机制 |
| **SIL-PPO** | Off-policy自模仿方法扩展至PPO，依据return > value选择模仿样本 |
| **SVPG-PPO** | On-policy自模仿方法，通过Stein variational gradient最小化分布差异 |
| **PER-PPO** | Prioritized Experience Replay，基于TD-error优先回放缓冲区样本 |

> ⚠️ 注意：由于官方实现限制，SIL/SVPG未支持Animal-AI这类部分可观测环境，故在此任务上仅与PPO比较。

---

## 3. 主要实验结果和性能指标

### 关键性能表现（见Figures 1–4, 7–9）

#### 在MuJoCo密集奖励任务上的表现（Figure 1 & 7）
- SIPP-MATCH在所有10个任务上**显著优于所有基线**
- 尤其在`Hopper-v4`, `Walker2d-v4`, `Humanoid-v4`等高难度任务上提升明显
- 平均收敛速度加快约**30%-50%**
- 示例：
  - `Ant-v4`: SIPP最终reward ≈ 5500，PPO ≈ 4800
  - `HalfCheetah-v4`: SIPP ≈ 11000，PPO ≈ 9500

#### 在PointMaze稀疏奖励任务上的表现（Figure 3 & 8）
- SIPP-REPLAY在四个变体中均**大幅领先基线**
- 成功率提升显著，尤其在大型迷宫（Large_Diverse）中体现更强泛化能力
- 示例：
  - `PointMaze_Large_Diverse_G-v3`: SIPP ≈ 280 episodic reward，PPO ≈ 150

#### 在Animal-AI Olympics上的表现（Figure 4 & 9）
- 所有五个子任务中，SIPP-REPLAY**远超PPO**
- 成功率从<20%（PPO）提升至>60%（SIPP），部分任务接近80%
- 特别是在“Goal-occluded tunnel”（需推箱子）和“Goal-on wall”（需爬坡）等复杂任务中表现出卓越的长期规划能力
- 是首个成功应用于此类**部分可观测POMDP环境**的自模仿方法

---

### 消融实验结果（Ablation Study）

#### IET系数（ε）的影响（Figure 5, 6, 10, 11）
- **密集奖励环境（MuJoCo）**：较小的ε（0.1–0.2）效果更好  
  → 因环境本身已有良好反馈，过度模仿反而抑制探索
- **稀疏奖励环境（PointMaze）**：较高的ε（0.3–0.5）更优  
  → 需要更多依赖成功轨迹来克服探索困难
- 结论：**ε应根据任务特性调整**，验证了IET机制的有效性

#### 缓冲区大小影响（未图示，文中提及）
- REPLAY策略中，增大imitation buffer size有助于保留多样性成功路径，进一步提升性能

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **自模仿能有效增强on-policy RL的探索能力**，尤其是在稀疏奖励和部分可观测环境中。
2. ✅ **MATCH策略通过最优传输实现“软匹配”**，避免硬阈值判断带来的噪声，适用于密集反馈场景。
3. ✅ **REPLAY策略通过轨迹级重放解决了长程依赖问题**，特别适合稀疏奖励下的行为固化。
4. ✅ SIPP框架**不修改奖励、不引入额外模型、保持PPO稳定性**，具备良好的实用性和可部署性。
5. ✅ SIPP是**首个在Animal-AI Olympics等复杂POMDP环境中成功应用的自模仿方法**，展示了强大的泛化能力。

---

### 方法的局限性
1. **对IET系数敏感**：虽然提出了调参指南（Appendix B），但仍需手动调节ε以适应不同任务。
2. **缓冲区容量有限**：仅保存固定数量的成功轨迹，可能丢失早期有用经验。
3. **未处理负经验**：仅模仿成功轨迹，未考虑如何规避失败行为。
4. **计算开销增加**：MATCH中的OT距离计算带来额外O(T²)复杂度（虽用Sinkhorn加速）。

---

### 未来工作方向
1. **动态调节IET系数**：根据学习阶段或环境反馈自动调整探索-模仿平衡。
2. **结合内在激励（intrinsic motivation）**：如与RND（Random Network Distillation）结合（Appendix A已初步尝试），同时促进新颖性探索与成功行为巩固。
3. **扩展至离线RL（Offline RL）**：利用静态数据集进行自模仿训练，提升数据利用率。
4. **引入记忆机制**：使用RNN或Transformer建模长期依赖，更好地复现复杂序列行为。
5. **多智能体扩展**：将SIPP推广至MARL场景，支持协作式自模仿学习。

---

> 🔚 **总结一句话**：  
> SIPP通过MATCH与REPLAY双策略，首次实现了高效、稳定、通用的on-policy自模仿学习，在密集与稀疏奖励环境下均取得SOTA性能，为解决RL中的探索难题提供了新范式。

</details>

---

### 16. [Greedy Is a Strong Default: Agents as Iterative Optimizers](https://arxiv.org/abs/2603.27415)

**Authors**: Yitao Li  
**Category**: cs.AI  
**Published**: 2026-03-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.27415v1  

#### Abstract
Classical optimization algorithms--hill climbing, simulated annealing, population-based methods--generate candidate solutions via random perturbations. We replace the random proposal generator with an LLM agent that reasons about evaluation diagnostics to propose informed candidates, and ask: does t...

---

### 17. [The Last Fingerprint: How Markdown Training Shapes LLM Prose](https://arxiv.org/abs/2603.27006)

**Authors**: E. M. Freeburg  
**Category**: cs.CL  
**Published**: 2026-03-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.27006v1  

#### Abstract
Large language models produce em dashes at varying rates, and the observation that some models "overuse" them has become one of the most widely discussed markers of AI-generated text. Yet no mechanistic account of this pattern exists, and the parallel observation that LLMs default to markdown-format...

---

### 18. [Preconditioned Attention: Enhancing Efficiency in Transformers](https://arxiv.org/abs/2603.27153)

**Authors**: Hemanth Saratchandran  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.27153v1  

#### Abstract
Central to the success of Transformers is the attention block, which effectively models global dependencies among input tokens associated to a dataset. However, we theoretically demonstrate that standard attention mechanisms in transformers often produce ill-conditioned matrices with large condition...

---

### 19. [From Independent to Correlated Diffusion: Generalized Generative Modeling with Probabilistic Computers](https://arxiv.org/abs/2603.27996)

**Authors**: Nihal Sanjay Singh, Mazdak Mohseni-Rajaee, Shaila Niazi, Kerem Y. Camsari  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.27996v1  

#### Abstract
Diffusion models have emerged as a powerful framework for generative tasks in deep learning. They decompose generative modeling into two computational primitives: deterministic neural-network evaluation and stochastic sampling. Current implementations usually place most computation in the neural net...

---

### 20. [Efficient Counting and Simulation in Content-Oblivious Rings](https://arxiv.org/abs/2603.28260)

**Authors**: J\'er\'emie Chalopin, Yi-Jun Chang, Giuseppe Antonio Di Luna, Haoran Zhou  
**Category**: cs.DC  
**Published**: 2026-03-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.28260v1  

#### Abstract
In the content-oblivious (CO) model (proposed by Censor-Hillel et al.), processes inhabit an asynchronous network and communicate only by exchanging pulses. A series of works has clarified the computational power of this model. In particular, it was shown that, when a leader is present and the netwo...

---

### 21. [Conformalized Signal Temporal Logic Inference under Covariate Shift](https://arxiv.org/abs/2603.27062)

**Authors**: Yixuan Wang, Danyang Li, Matthew Cleaveland, Roberto Tron, Mingyu Cai  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.27062v1  

#### Abstract
Signal Temporal Logic (STL) inference learns interpretable logical rules for temporal behaviors in dynamical systems. To ensure the correctness of learned STL formulas, recent approaches have incorporated conformal prediction as a statistical tool for uncertainty quantification. However, most existi...

---

### 22. [Rethinking Language Model Scaling under Transferable Hypersphere Optimization](https://arxiv.org/abs/2603.28743)

**Authors**: Liliang Ren, Yang Liu, Yelong Shen, Weizhu Chen  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.28743v1  

#### Abstract
Scaling laws for large language models depend critically on the optimizer and parameterization. Existing hyperparameter transfer laws are mainly developed for first-order optimizers, and they do not structurally prevent training instability at scale. Recent hypersphere optimization methods constrain...

---

### 23. [Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation](https://arxiv.org/abs/2603.28744)

**Authors**: Vit\'oria Barin Pacela, Shruti Joshi, Isabela Camacho, Simon Lacoste-Julien, David Klindt  
**Category**: cs.LG  
**Published**: 2026-03-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.28744v1  

#### Abstract
The linear representation hypothesis states that neural network activations encode high-level concepts as linear mixtures. However, under superposition, this encoding is a projection from a higher-dimensional concept space into a lower-dimensional activation space, and a linear decision boundary in ...

---

### 24. [daVinci-LLM:Towards the Science of Pretraining](https://arxiv.org/abs/2603.27164)

**Authors**: Yiwei Qin, Yixiu Liu, Tiantian Mi, Muhang Xie, Zhen Huang, Weiye Si, Pengrui Lu, Siyuan Feng, Xia Wu, Liming Liu, Ye Luo, Jinlong Hou, Qipeng Guo, Yu Qiao, Pengfei Liu  
**Category**: cs.AI  
**Published**: 2026-03-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.27164v1  

#### Abstract
The foundational pretraining phase determines a model's capability ceiling, as post-training struggles to overcome capability foundations established during pretraining, yet it remains critically under-explored. This stems from a structural paradox: organizations with computational resources operate...

---

### 25. [Aligning LLMs with Graph Neural Solvers for Combinatorial Optimization](https://arxiv.org/abs/2603.27169)

**Authors**: Shaodi Feng, Zhuoyi Lin, Yaoxin Wu, Haiyan Yin, Yan Jin, Senthilnath Jayavelu, Xun Xu  
**Category**: cs.AI  
**Published**: 2026-03-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.27169v1  

#### Abstract
Recent research has demonstrated the effectiveness of large language models (LLMs) in solving combinatorial optimization problems (COPs) by representing tasks and instances in natural language. However, purely language-based approaches struggle to accurately capture complex relational structures inh...

---

### 26. [CARGO: Carbon-Aware Gossip Orchestration in Smart Shipping](https://arxiv.org/abs/2603.27857)

**Authors**: Alexandros S. Kalafatelis, Nikolaos Nomikos, Vasileios Nikolakakis, Nikolaos Tsoulakos, Panagiotis Trakadas  
**Category**: cs.AI  
**Published**: 2026-03-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.27857v1  

#### Abstract
Smart shipping operations increasingly depend on collaborative AI, yet the underlying data are generated across vessels with uneven connectivity, limited backhaul, and clear commercial sensitivity. In such settings, server-coordinated FL remains a weak systems assumption, depending on a reachable ag...

---

### 27. [SCOPE: Tree-based Self-Correcting Online Log Parsing via Syntactic-Semantic Collaboration](https://arxiv.org/abs/2603.27247)

**Authors**: Dongyi Fan, Suqiong Zhang, Lili He, Ming Liu, Yifan Huo  
**Category**: cs.CL  
**Published**: 2026-03-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.27247v1  

#### Abstract
Log parsing is a critical step for automated log analysis in complex systems. Traditional heuristic-based methods offer high efficiency but are limited in accuracy due to overlooking semantic context. In contrast, recent LLM-based parsers improve accuracy via se mantic understanding but incur high l...

---

### 28. [Mitigating Hallucination on Hallucination in RAG via Ensemble Voting](https://arxiv.org/abs/2603.27253)

**Authors**: Zequn Xie, Zhengyang Sun  
**Category**: cs.CL  
**Published**: 2026-03-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.27253v1  

#### Abstract
Retrieval-Augmented Generation (RAG) aims to reduce hallucinations in Large Language Models (LLMs) by integrating external knowledge. However, RAG introduces a critical challenge: hallucination on hallucination," where flawed retrieval results mislead the generation model, leading to compounded hall...

---

### 29. [Top-down string-to-dependency Neural Machine Translation](https://arxiv.org/abs/2603.27938)

**Authors**: Shuhei Kondo, Katsuhito Sudoh, Yuji Matsumoto  
**Category**: cs.CL  
**Published**: 2026-03-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.27938v1  

#### Abstract
Most of modern neural machine translation (NMT) models are based on an encoder-decoder framework with an attention mechanism. While they perform well on standard datasets, they can have trouble in translation of long inputs that are rare or unseen during training. Incorporating target syntax is one ...

---

### 30. [DongYuan: An LLM-Based Framework for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis](https://arxiv.org/abs/2603.28191)

**Authors**: Hua Li, Yingying Li, Xiaobin Feng, Xinyi Fu, Lifeng Dong, Qingfeng Yang, Yanzhe Chen, Xiaoju Feng, Zhidong Cao, Jianbin Guo, Yanru Du  
**Category**: cs.CL  
**Published**: 2026-03-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.28191v1  

#### Abstract
The clinical burden of spleen-stomach disorders is substantial. While large language models (LLMs) offer new potential for medical applications, they face three major challenges in the context of integrative Chinese and Western medicine (ICWM): a lack of high-quality data, the absence of models capa...

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
