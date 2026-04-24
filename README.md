# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-24 07:24:56 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Distributed Generative Inference of LLM at Internet Scales with Multi-Dimensional Communication Optimization](https://arxiv.org/abs/2604.21072)

**Authors**: Jiu Chen, Shuangyan Yang, Xu Xiong, Hexiao Duan, Xinran Zhang, Jie Ren, Dong Li  
**Category**: cs.DC  
**Published**: 2026-04-24  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2604.21072v1  

#### Abstract
Decentralized LLM inference distributes computation among heterogeneous nodes across the internet, offering a performant and cost-efficient solution, alternative to traditional centralized inference. However, the low cross-node network bandwidth makes communication the primary bottleneck. In this pa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Distributed Generative Inference of LLM at Internet Scales with Multi-Dimensional Communication Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**去中心化大语言模型（LLM）推理在互联网规模下的通信瓶颈问题**。由于分布式节点间网络带宽低（通常为 20–500 Mbps），跨节点传输激活张量（activations）成为性能的主要限制因素，远超本地计算时间。

传统集中式推理依赖高性能数据中心内网（如 InfiniBand），而**去中心化环境无法使用 GPUDirect RDMA 和低延迟协议**，导致通信开销极高。因此，如何在异构、低带宽的互联网环境中优化分布式 LLM 推理吞吐量，是本文的核心挑战。

---

### 提出的新方法与创新思路

作者提出了名为 **BloomBee** 的去中心化 LLM 推理框架，其核心思想是“**以通信为中心的多维协同优化**”，整合以下关键技术：

#### （1）**多维度通信优化的联合建模**
首次将 **layer assignment（层分配）、micro-batching（微批处理）、tensor offloading（KV 缓存卸载）** 三个技术统一建模为一个优化问题，并通过 **dynamic programming（动态规划）** 求解最优配置，在 GPU 内存约束下最大化吞吐量。

#### （2）**面向低带宽的轻量无损压缩（Lossless Compression）**
提出一种基于 **byte-split 的新型数据布局转换方法**，将 FP16 浮点数按高低字节分离后再进行 ZSTD 压缩，显著提升压缩率。相比 ZSTD 和 ZipNN，压缩后体积分别减少 **33% 和 35%**。

#### （3）**适应弱网环境的推测解码（Speculative Decoding, SD）改进**
传统 SD 在去中心化场景中因需传输大量候选 token 而加重通信负担。BloomBee 引入：
- **MLP 分类器对 draft tree 进行剪枝（pruning）**，提前丢弃低概率候选 token；
- **padding-free 批量打包传输机制**，避免填充浪费；
- **异步 KV cache compact**，避免阻塞主流程。

从而实现“**带宽感知的 SD 启用策略**”：仅当实测带宽高于 break-even bandwidth $ S^* $ 时才启用 SD。

#### （4）**系统级协同设计**
- 支持异构 GPU 和网络环境下的自动调度；
- 开源实现（GitHub 已发布），支持多种主流 LLM 架构（LLaMA、Mixtral 等）；
- 引入 **specification-driven code generation** 自动生成模型适配代码，降低部署成本。

---

### 相比现有方法的优势

| 特性 | BloomBee | Petals | Helix |
|------|---------|--------|-------|
| 层分配优化 | ✅ 动态规划求解 | ❌ 贪心算法 | ✅ MILP 求解（耗时高） |
| 微批流水 | ✅ 支持 | ❌ 不支持 | ❌ 不支持 |
| KV 卸载 | ✅ 支持 CPU offload | ❌ 不支持 | ⚠️ 部分支持 |
| 无损压缩 | ✅ 自研 byte-split + ZSTD | ⚠️ 量化（有损） | ❌ 无 |
| 推测解码 | ✅ 带剪枝的 SD | ❌ 不支持 | ⚠️ 支持但未优化通信 |
| 实际部署效率 | ✅ 高（毫秒级决策） | 中等 | ❌ 低（小时级求解） |

> BloomBee 在保持精度的同时，全面优化通信路径，且运行时开销极低。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **AlpacaEval [22]**：用于生成测试 prompts，评估推理质量与延迟分布。
- **Alpaca [36]**：用于训练 draft tree 剪枝分类器（二分类标签：是否被目标模型接受）。
- 实际推理输入长度为 **128 tokens**，输出长度为 **128 tokens**，batch size 默认为 **32**。

---

### 实验设置

#### 网络环境（E1–E6）
| 环境 | 描述 |
|------|------|
| **E1** | 数据中心内部，45 Gbps 以太网（理想情况） |
| **E2–E5** | 模拟互联网带宽：500 Mbps、250 Mbps、125 Mbps、20 Mbps |
| **E6** | 真实地理分布集群：<br>• California: A100<br>• New Jersey: RTX 4090<br>• Canada: RTX 4090 |

#### 模型
- 主要模型：**LLaMA-13B, 30B, 65B**
- 补充模型：**Falcon-7B, Falcon-40B, Mixtral-8×7B**

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput (tok/s)** | 每秒生成 token 数量（主要指标） |
| **Average Latency** | 平均每请求完成时间 |
| **Per-sample Completion Time** | 单个样本完成时间分布（反映用户体验） |
| **Cost Efficiency (tok/s/$/h)** | 单位 GPU 成本下的吞吐量 |
| **Communication Volume Reduction** | 激活张量压缩前后大小对比 |

---

### 基线方法对比
- **Petals [3]**：开源去中心化推理框架，采用贪心层分配和量化压缩。
- **Helix [24]**：基于 Max-Flow 的异构调度系统，支持灵活 pipeline 构建。

> 两者均未集成 micro-batching、lossless compression 或通信友好的 SD。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 结果 |
|------|------|
| **最高吞吐提升** | 达到 **1.76×** 于 Petals，**1.46×** 于 Helix（LLaMA-30B, E5） |
| **平均延迟降低** | 最高达 **43.20%** |
| **压缩效率（vs ZSTD）** | 激活张量压缩至原大小 **46%**（ZSTD 为 69%，ZipNN 为 71%） |
| **KV 缓存卸载节省 GPU 数量** | 可从 3 GPU 减少到 **2 GPU**，节省成本 33% |
| **draft tree 剪枝效果** | 减少传输量 **60%**，保留 **96% 接受率** |

---

### 与基线方法的对比结果（Figure 8）

| 场景 | BloomBee vs Petals | BloomBee vs Helix |
|------|---------------------|--------------------|
| LLaMA-30B, E5 (20 Mbps) | **67 tok/s vs 38 tok/s → +76%** | **67 vs 45.8 → +46%** |
| LLaMA-30B, E3 (250 Mbps) | ~100 vs ~70 → +43% | ~100 vs ~60 → +67% |
| LLaMA-65B, E1 (45 Gbps) | 优势缩小（计算主导） | 仍高出约 20–30% |

> 在低带宽环境下优势显著；随着带宽增加，SD 成为主要增益来源。

---

### 消融实验结果

#### （1）**Offloading 对成本效率的影响（Figure 9）**
- 在 E5 下，使用 offloading 后：
  - GPU 数量从 3→2，成本从 \$1.68/h → \$1.12/h
  - **吞吐每美元提升 1.82×（41.8 vs 22.9 tok/s/\$/h）**
- 即使在高带宽 E1，也有 **1.02–1.05× 提升**

#### （2）**Micro-batching 效果（Figure 10）**
- E1/E2（高带宽）：几乎无收益（通信不占主导）
- E3–E5（低带宽）：增益递增
  - E3: +5.3%
  - E4: +17.9%
  - E5: **+39.5%**

#### （3）**Compression 效果（Figure 11）**
- E1/E2：无明显收益（原始传输仅 ~10ms）
- E5：**单独压缩提升 18.4%**，结合 micro-batching 达 **76.3% 提升**
- 组合增益 > 加和，说明存在协同效应（micro-batch 改变数据分布利于压缩）

#### （4）**Speculative Decoding（Figure 12）**
- 在 E1：**中位完成时间快 43%（22.8s vs 40.2s）**
- 在 E2：快 24%
- 在 E3：仍有 4% 提升
- 未剪枝 SD 性能更差，验证剪枝必要性

> SD 显著改善单样本响应时间，适合实时服务场景。

---

## 4. 关键结论和发现

### 主要发现
1. **通信是互联网规模 LLM 推理的绝对瓶颈**  
   在典型家庭上行链路（20–500 Mbps）下，NIC-NIC 传输时间可达 GPU 计算时间的 **1.4–5 倍以上**。

2. **多维优化必须协同设计**  
   layer assignment、micro-batching、offloading 存在强耦合关系（如内存约束影响可部署层数），独立优化次优。

3. **无损压缩在低带宽下极具价值**  
   利用 FP16 数据结构特性（高低字节熵不同）可大幅提升压缩率，且压缩时间可被通信掩盖。

4. **推测解码需“带宽感知”才能生效**  
   传统 SD 在弱网下反而拖慢系统；引入剪枝机制后可在高带宽环境获得高达 **+12.5% 吞吐提升**。

5. **KV 缓存卸载有效降低硬件门槛**  
   将 KV cache 卸载至 CPU 内存，可在不影响性能前提下减少 GPU 使用数量，显著提升 **cost efficiency**。

---

### 方法的局限性
1. **依赖稳定的端到端 TCP/IP 连接**  
   当前未考虑极端网络抖动或频繁断连场景。

2. **对 very deep pipeline 更敏感**  
   随着 pipeline stage 增多，fill/drain 开销上升，micro-batching 收益下降。

3. **draft model 需与 target model 对齐**  
   若 draft model 质量差，acceptance rate 下降，SD 效益减弱。

4. **目前仅支持 pipeline parallelism**  
   未整合 tensor parallelism 或专家路由（MoE）等高级并行策略。

---

### 未来工作方向
1. **支持动态网络变化下的自适应重配置**  
   如带宽波动时动态切换 compression/micro-batching 策略。

2. **扩展至 MoE 模型与专家分散部署**  
   结合 expert routing 与通信优化。

3. **探索联邦学习风格的安全聚合机制**  
   在隐私保护前提下实现去中心化训练+推理一体化。

4. **进一步降低压缩延迟**  
   设计专用硬件加速模块或 INT8-friendly 压缩方案。

5. **构建全球志愿者节点激励生态**  
   结合区块链或 Token 激励机制推动更大规模去中心化推理网络。

---

> 💡 **总结一句话**：  
> **BloomBee 证明了在低带宽互联网环境下，通过“通信优先”的多维协同优化，可以高效运行大规模 LLM 推理，为去中心化 AI 提供了一条可行的技术路径。**

</details>

---

### 2. [Super Apriel: One Checkpoint, Many Speeds](https://arxiv.org/abs/2604.19877)

**Authors**: SLAM Labs,  :, Oleksiy Ostapenko, Raymond Li, Torsten Scholak, Alireza Mousavi-Hosseini, Aman Tiwari, Denis Kocetkov, Joel Lamy Poirier, Kelechi Ogueji, Nanda H Krishna, Rafael Pardinas, Sathwik Tejaswi Madhusudhan, Shruthan Radhakrishna, Srinivas Sunkara, Valerie Becaert  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.19877v1  

#### Abstract
We release Super Apriel, a 15B-parameter supernet in which every decoder layer provides four trained mixer choices -- Full Attention (FA), Sliding Window Attention (SWA), Kimi Delta Attention (KDA), and Gated DeltaNet (GDN). A placement selects one mixer per layer; placements can be switched between...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Super Apriel: One Checkpoint, Many Speeds 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大型语言模型在长上下文推理场景下面临**推理成本瓶颈**，尤其是 Full Attention (FA) 在自回归解码时的 KV Cache 内存开销随序列长度呈线性增长，导致吞吐量受限。传统混合架构（hybrid architectures）虽然通过部分替换为高效 token mixer（如 SWA、GDN）来提升速度，但其 **placement（每层使用哪种 mixer）是固定的**，只能提供单一的速度-质量权衡点。

这带来了以下限制：
- 难以适应异构工作负载（如短提示高并发 vs 长上下文生成）
- 无法实现负载自适应服务（高峰时段切换到高效模式）
- 不同任务对长程依赖敏感度不同，固定 placement 无法灵活调整

### 提出的新方法与创新
Super Apriel 提出了一种 **token-mixer supernet** 架构，实现了“一个检查点，多种速度”的部署范式。

#### 核心创新点：
1. **Supernet 架构设计**
   - 每个 decoder 层内置 **四种训练好的 token mixer**：Full Attention (FA)、Sliding Window Attention (SWA)、Kimi Delta Attention (KDA) 和 Gated DeltaNet (GDN)。
   - 所有 mixer 共享大部分参数（FFN、Embedding、Normalization），仅 mixer 模块权重独立。
   - 一个 **placement** 是指为每一层选择一个 mixer 类型的配置方案。

2. **运行时灵活切换（Runtime Flexibility）**
   - 在服务时可动态切换 placement，无需重新加载模型权重。
   - 支持 per-request 级别的 placement 路由，实现多速度预设（presets）共存于同一实例。

3. **免草稿模型的投机解码（Speculative Decoding）**
   - 利用共享检查点，将高效的混合 placement 作为 **draft model**，全注意力 placement 作为 **verifier**。
   - 无需额外训练和部署独立的草稿模型，简化系统复杂度。

4. **基于代理模型的放置优化（Placement Optimization）**
   - 使用 **cluster expansion surrogate model** 对 placement 质量进行建模，支持精确的成本约束优化。
   - 可快速扫描庞大的组合空间（$4^{48}$），找到帕累托前沿上的最优 trade-off。

5. **开放发布**
   - 开源了 supernet 权重、Fast-LLM 训练代码、vLLM 推理扩展及 placement 优化工具包。

### 相比现有方法的优势
| 维度 | 传统方法 | Super Apriel |
|------|----------|-------------|
| **灵活性** | 固定 placement，单一点 | 单一 checkpoint 支持多个运行时速度预设 |
| **部署效率** | 多个模型需分别训练、部署 | 一次训练，多种用途（主干 + 草稿模型） |
| **搜索效率** | 依赖启发式搜索或逐个验证 | 代理模型支持精确、高效的全局优化 |
| **适用场景** | 单一目标优化 | 支持任务感知、负载自适应的服务策略 |

---

## 2. 核心实验方法和设置

### 数据集
#### 训练数据
- **Distillation Dataset**（266B tokens）：
  - 来源于 Apriel 预训练语料 + SFT 数据
  - 强调高质量推理轨迹（multi-step proofs, structured problem-solving, code with logical dependencies）
  - 包含文本、图像（multimodal）输入
- **SFT Dataset**（用于监督微调）：
  - 以数学与 STEM（38.7%）、代码（36.1%）为主
  - 包含聊天、工具调用、安全等内容

#### 评估数据集（Evaluation Suite）
分为 **开发基准（dev benchmarks）** 和 **未见基准（unseen benchmarks）**

| 类别 | 基准名称 | 描述 |
|------|--------|------|
| **开发基准** | MMLU | 多学科多项选择题 |
| | GSM8K, MATH500, AIME24/25 | 数学推理（生成式） |
| | FDA, SWDE | 零样本信息抽取 |
| | NIAH, RULER | 合成长上下文检索与推理 |
| **未见基准** | MMLU-Pro | 更难版本 MMLU |
| | GPQA | 博士级科学问题 |
| | HLE | “人类最后考试”，极限挑战题 |
| | LCB | 无污染代码生成 |
| | T2-Bench | 工具增强对话代理 |
| | IFEval | 指令遵循能力 |
| | AIME(NV) | NVIDIA 提取协议下的 AIME |

> 注：placement 优化目标默认为 MATH500、AIME24、AIME25 上教师轨迹的平均 log-likelihood。

### 实验设置
- **模型规模**：15B 参数（48 层），基于 Apriel 1.6 构建
- **Mixer 成本系数**（Regression-based cost model）：
  - FA: 1.00
  - SWA: 0.48
  - KDA: 0.21
  - GDN: 0.14
- **训练流程**：
  1. **S1: Distillation**：从冻结的 Apriel 1.6 教师模型进行随机 placement 抽样蒸馏
     - 损失函数：Activation Matching + Reverse KL + Forward KL
     - 仅更新 mixer 权重，其余参数冻结
  2. **S2: SFT**：在特定预设 placement 上进行指令微调（targeted training）

- **推理设置**：
  - 使用 vLLM 实现 CUDA Graph 加速
  - 支持 per-request placement 切换（基于 LoRA-like 路由机制）
  - 吞吐量测量在 H100 GPU 上完成

### 评估指标
| 指标 | 定义 |
|------|------|
| **Decode Throughput Speedup** | 相对于 all-FA 预设的解码吞吐量倍数 |
| **Quality Retention (%)** | 平均准确率相对于教师模型的比例 |
| **Pareto Frontier** | 速度 vs 质量的最优权衡曲线 |
| **Speculative Decoding Speedup** | 使用混合 draft + all-FA verifier 的端到端加速比 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4 & Table 13）

| Preset | FA/SWA/KDA/GDN | Avg Acc | Retention | Speedup @32k |
|--------|----------------|---------|-----------|------------|
| **all-FA** | 48/0/0/0 | 74.2 | 100% | 1.0× |
| **RegLklhd-26** | 12/26/6/4 | 71.1 | 96% | 2.9× |
| **RegLklhd-18** | 3/25/4/16 | 69.7 | 94% | 4.8× |
| **RegLklhd-13** | 0/16/13/19 | 60.2 | 81% | 6.9× |
| **RegLklhd-10** | 0/10/5/33 | 57.2 | 77% | **10.7×** |

> ✅ 最快预设达到 **10.7× 解码吞吐量**，保留 **77% 质量**

### 与基线方法对比（Figure 9 & Table 13）

| 模型 | 参数量 | Speedup | Math Avg | All-tasks Avg |
|------|--------|--------|----------|---------------|
| **Super Apriel (RegLklhd-26)** | 15B | 2.9× | 88.3 | 71.1 |
| Apriel-H1 | 15B | 2.0× | 80.4 | 58.4 |
| Qwen-3.5 | 27B | 0.5× | 92.6 | 80.7 |
| Nemotron-3-Nano | 30B | 4.1× | 89.0 | 72.6 |
| Falcon-H1R | 7B | 4.6× | 78.6 | 64.9 |

> 🔍 Super Apriel 在相近速度下显著优于 Apriel-H1（+12.7 分），且优于更小模型（如 OLMo-Hybrid-Think 7B）

### 长上下文扩展优势（Figure 10）
- Super Apriel 的高效预设在长上下文下增益更大：
  - **RegLklhd-10**: 从 16K 的 4.2× 提升至 32K 的 **10.7×**（+155%）
  - 外部混合模型仅提升 5–46%
- 原因：SWA/KDA/GDN 的状态大小固定，不受上下文长度影响

### 投机解码性能（Figure 11）
- 使用 all-GDN 作为 draft model 可获得最高加速比
- 接受率在整个成本范围内保持高位，表明 supernet 内部分布一致性良好
- 实现了无需额外草稿模型的高效 speculative decoding

### 消融实验结果（Section 5）

#### （1）Placement Ranking 稳定性
- 在 0.5B 模型上，排名早期即稳定（Spearman ρ > 0.98）
- 但在 15B 模型上，**帕累托前沿的 placement 表现出更高不稳定性**（Kendall’s W 下降）
- 结论：**不能从小模型外推大模型的最佳 placement**

#### （2）训练策略对比（0.5B 模型）
- **Stochastic Training**（均匀采样所有 placement）能最好地提升整体性能
- **Targeted/Hybrid Training** 导致非目标 placement 学习缓慢
- 但在 15B 上仍采用 targeted SFT，因其更适合最终部署目标

#### （3）SFT 效果显著
- 所有预设在 SFT 后均有大幅提升，尤其数学任务：
  - 如 `Idealized|All-6` 在 AIME'24 上提升 **+20.0 分**
- 表明 SFT 能有效恢复因蒸馏丢失的复杂推理能力

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **单一 supernet 可支持多种运行时速度预设**，实现“一次训练，多种用途”。
2. ✅ **推荐预设在 2.9× 至 10.7× 吞吐量之间，质量保留 96% 至 77%**，覆盖广泛应用场景。
3. ✅ **长上下文下优势放大**：高效 placement 在 32K 上相对加速可达 80–155%，远超外部基线。
4. ✅ **共享检查点可用于投机解码**，避免单独训练 draft model。
5. ❗ **最佳 placement 在小模型上稳定，在大模型上前沿不稳定**，警示不可简单外推。
6. ✅ **cluster expansion surrogate 可高效探索巨大组合空间**，并识别帕累托最优。

### 方法局限性
- **长距离任务退化**：大量使用 GDN/KDA 会导致 RULER/NIAH 等长程检索任务严重下降。
- **代理模型假设限制**：cluster expansion 截断于短程交互，可能忽略长程层间依赖。
- **线性成本模型偏差**：对极端稀疏配置拟合不佳，需排除 singleton placements。
- **训练策略依赖**：目前缺乏大规模 controlled comparison 来确定最优训练策略（stochastic vs targeted）。
- **推理引擎依赖**：性能受 vLLM 等后端实现影响，存在版本漂移风险。

### 未来工作方向
1. **强化学习微调（RL）**：计划使用 Group Relative Policy Optimization 进行推理与智能体任务优化。
2. **扩大 mixer 词汇表**：引入 Mamba-2、Lightning Attention 等新型 mixer。
3. **生产级训练策略对比**：在 15B 规模上公平比较 stochastic 与 targeted placement sampling。
4. **部署灵活性增强**：
   - 支持 per-request 动态路由
   - 内存优化（model thinning, 共享 mixer weights）
5. **跨教师泛化性研究**：验证该方法是否适用于其他教师模型。
6. **探索更大规模下的 landscape dynamics**：确认 15B 上观察到的现象是否延续到更大模型。

---

> 📌 **总结一句话**：  
> Super Apriel 通过构建一个支持运行时灵活切换的 **mixer supernet**，首次实现了“**一个检查点，多种速度**”的部署愿景，在保持高质量的同时提供高达 **10.7× 的解码吞吐量提升**，并通过代理模型优化和投机解码进一步增强了实用性与效率。

</details>

---

### 3. [Scalable AI Inference: Performance Analysis and Optimization of AI Model Serving](https://arxiv.org/abs/2604.20420)

**Authors**: Hung Cuong Pham, Fatih Gedikli  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.20420v1  

#### Abstract
AI research often emphasizes model design and algorithmic performance, while deployment and inference remain comparatively underexplored despite being critical for real-world use. This study addresses that gap by investigating the performance and optimization of a BentoML-based AI inference system f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Scalable AI Inference: Performance Analysis and Optimization of AI Model Serving*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**AI模型部署与推理阶段的性能瓶颈**，弥补了当前研究中“重训练、轻部署”的差距。尽管大量研究关注模型架构和训练精度，但在真实生产环境中，**推理延迟（latency）、吞吐量（throughput）和服务弹性（resilience）** 同样至关重要。

具体而言，论文解决了以下问题：
- 如何在现实流量模式下（如突发流量）高效地进行 **AI Model Serving**；
- 如何系统性优化从模型到部署栈的全链路性能；
- 在资源受限或简化部署环境下（如单节点集群），如何提升服务的自愈能力。

### 🚀 提出的新方法与思路
论文提出了一套**多层级优化框架**，应用于基于 **BentoML** 构建的AI推理服务，并结合 **K3s** 轻量级Kubernetes集群实现弹性部署。其核心创新点包括：

1. **端到端优化策略**：
   - 在 **Model Level**：将原始 FP32 PyTorch 模型转换为 ONNX 格式，并应用图优化（graph optimization）和半精度（FP16）量化。
   - 在 **Runtime Level**：禁用Hugging Face Transformers中的梯度追踪与dropout等非必要计算。
   - 在 **Service Level**：启用 BentoML 的 **adaptive batching** 动态批处理机制，以应对波动负载。
   - 在 **Deployment Level**：部署至 **single-node K3s cluster**，利用Kubernetes的自动恢复机制增强容错性。

2. **真实感流量建模**：
   - 使用 **Exponential 分布模拟稳定流量**（Poisson过程）
   - 使用 **Gamma 分布模拟不同程度的bursty traffic**（a=1.2 和 a=0.8），更贴近实际用户行为。

3. **综合评估体系**：
   - 结合 **Load Testing**（压力测试）与 **Resilience Testing**（弹性测试），全面评估系统在高负载和故障场景下的表现。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **性能** | 优化后系统吞吐量提升数百倍，延迟降低两个数量级 |
| **灵活性** | 支持多种优化组合（如 ONNX + FP16），可按需配置 |
| **实用性** | 验证了即使在单节点、单副本的轻量部署下也能实现自动恢复，适合边缘或中小规模部署 |
| **可复现性** | 公开实验设置、流量模型与评估流程，便于复现与比较 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **模型**：采用由 graphworks.ai 提供的预训练 **RoBERTa Base** 模型，用于 **Sentiment Analysis**（情感分析）任务。
- **测试数据集**：`Sp1786/Multiclass-Sentiment-Analysis-Dataset`，包含 positive、negative、neutral 三类标签。
- **评估子集**：固定随机种子抽取 **1,000 条样本**，确保各实验间公平对比。

### ⚙️ 实验设置
- **硬件环境**：
  - CPU: 2×Intel Xeon Gold 6230 @ 2.10GHz
  - GPU: 1×NVIDIA Quadro RTX 8000 (48GB VRAM)
  - 内存: 693 GB RAM
  - 软件栈: CUDA 13.0, PyTorch 2.9.1+cu128, ONNX Runtime 1.20.1, BentoML 1.4.35, Locust 2.43.3

- **部署架构**：
  - 基线：Docker 单容器部署（无编排）
  - 优化版：部署于 **single-node K3s cluster**，仅一个 replica

- **流量模式设计**（均保持平均到达率 λ=0.5 req/s，即平均间隔 2 秒）：
  1. **Steady Traffic**：指数分布（Exponential），标准 Poisson 流
  2. **Moderate Burstiness**：Gamma 分布，形状参数 α=1.2（方差较小）
  3. **Extreme Burstiness**：Gamma 分布，形状参数 α=0.8（高度聚集）

- **评估工具**：使用 **Locust** 进行负载生成，采集 RPS、延迟分位数、响应时间等指标。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Latency (p50/p95/p99)** | 推理延迟的中位数与尾部延迟 |
| **Throughput (samples/s)** | 每秒处理样本数 |
| **Response Time (min/avg/max)** | 请求往返时间 |
| **Failure Rate** | 错误请求占比 |
| **Total Test Duration** | 完成全部请求所需时间 |
| **Model Size** | 存储占用空间 |
| **Accuracy & F1 Score** | 分类准确性验证（确认优化未影响预测质量） |

### 🔁 基线方法对比
| 方法 | 描述 |
|------|------|
| **Baseline** | `simpletransformers` + FP32 PyTorch，直接封装模型，无额外优化 |
| **Optimized Variants** | 多种组合方式：<br>- Hugging Face Transformers + FP32/FP16 PyTorch<br>- ONNX + FP32/FP16 + graph optimization |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ▶️ 模型优化效果（Batch Size=32）

| 方法 | Latency (ms/sample) | Throughput (samples/s) | Model Size |
|------|---------------------|------------------------|-----------|
| Baseline (FP32 PyTorch) | 135.72 | 7.37 | ~498.7 MB |
| Opt. (FP16 PyTorch) | 0.56 | **1,774.72** | ~498.7 MB |
| Opt. (FP16 ONNX) | **0.53** | **1,901.56** | **~249.4 MB** |

> ✅ **FP16 ONNX 实现最佳性能：延迟降至亚毫秒级，吞吐达近 2,000 samples/s，存储减半**

#### ▶️ 准确性保留情况
所有变体在外部测试集上表现一致：
- **Accuracy**: 0.42
- **Macro F1**: 0.32
- **Weighted F1**: 0.34  
👉 表明 **ONNX 导出、图优化、FP16 量化均未损害模型预测能力**

> ❗ 注：低准确率归因于 **domain shift**（训练域为能源新闻，测试域为短文本），而非模型退化。

---

### 🔍 与基线方法的对比结果（Load Testing）

#### 表：**Latency 对比（单位：ms）**

| 方法 | Steady (p50) | Moderate (p50) | Extreme (p50) |
|------|--------------|----------------|---------------|
| Baseline | 2700 | 3000 | 3100 |
| FP16 ONNX | **27** | **26** | **26** |

> ⏱️ **延迟降低超过 100 倍！**

#### 表：**Throughput 对比**
- 基线：约 **0.2 RPS**
- 所有优化版本：接近理论上限 **0.5 RPS**（匹配输入流量速率）

> ✅ 优化后系统能完全吸收输入负载，而基线严重滞后

#### 总耗时对比
- 基线完成测试需 **>80分钟**
- 优化版本仅需 **~30分钟**

---

### 🔧 消融实验结果（Ablation Study）

| 优化维度 | 效果 |
|---------|------|
| **替换 simpletransformers → Hugging Face Transformers** | 显著减少运行时开销，提升可控性 |
| **启用 adaptive batching** | 提升吞吐量，尤其在中高负载下效果明显 |
| **ONNX + graph optimization** | 进一步压缩延迟，提高执行效率 |
| **FP16 量化** | 几乎无损精度前提下，显著加速计算并减小内存占用 |
| **K3s 部署 + 自动恢复机制** | 实现 Pod 终止后的自动重启，无需人工干预 |

> ✅ 多项优化叠加产生协同效应，整体性能远超单一改进

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **BentoML 是构建可扩展 AI 推理服务的有效平台**，尤其配合合理的优化策略时表现出色。
2. **模型格式与精度选择对性能影响巨大**：
   - 将 PyTorch 模型导出为 **ONNX + FP16** 可带来显著性能增益（延迟↓、吞吐↑、体积↓）
   - 在本案例中，**FP16 不损失 Accuracy/F1**，是理想的部署配置
3. **adaptive batching 是应对动态负载的关键技术**，能有效平滑突发请求的影响。
4. **即使是 single-node K3s 集群也能提供可观的 resilience 提升**：
   - Kubernetes 可自动检测并重建失败的 Pod
   - 相比纯 Docker 部署，具备自我修复能力，显著降低运维负担
5. **真实流量建模（Gamma/Bursty）对于评估系统鲁棒性至关重要**，固定速率测试无法反映真实挑战。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **单节点部署** | 无法实现高可用（HA），仅验证了“自愈”而非“不间断服务” |
| **单一模型与任务** | 结论可能不适用于其他模型（如 LLMs）或任务类型 |
| **Domain Shift 影响评估** | 外部测试集性能偏低，可能掩盖部分优化收益 |
| **未使用专用 AI 加速器** | 实验基于 GPU，未涉及 TPU、NPU 或推理芯片优化 |
| **未探索更低精度（INT8/FP8/BF16）** | 当前仅验证 FP16，未来可进一步量化 |

---

### 🔮 未来工作方向

1. **扩展至 Multi-node Cluster**：验证在多节点、多副本下的 fault tolerance 与 autoscaling 能力。
2. **引入更高级的批处理调度算法**：如动态调整 batch window、优先级队列等。
3. **支持更低精度量化（INT8, FP8）与量化感知训练（QAT）**
4. **集成监控与自动化调优系统**：实现基于负载反馈的自动参数调节（如 batch size、worker 数量）
5. **跨硬件平台评估**：在边缘设备（Jetson）、云实例（AWS Inferentia）等上验证通用性
6. **探索 LLM 特定优化技术**：如 KV Cache 复用、continuous batching（如 vLLM）

---

## ✅ 总结

本论文系统性地评估并优化了一个基于 **BentoML** 的 AI 推理服务，在 **model、runtime、service、deployment** 四个层面实施优化，最终实现了：
- **延迟降低 >100 倍**
- **吞吐提升 >250 倍**
- **模型大小减半**
- **部署弹性显著增强**

研究成果为构建**高效、可扩展、具弹性的生产级 AI 服务**提供了实用指南，特别适用于希望在有限资源下实现高性能推理的企业与开发者。

</details>

---

### 4. [Fast Bayesian equipment condition monitoring via simulation based inference: applications to heat exchanger health](https://arxiv.org/abs/2604.20735)

**Authors**: Peter Collett, Alexander Johannes Stasik, Simone Casolo, Signe Riemer-S{\o}rensen  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.20735v1  

#### Abstract
Accurate condition monitoring of industrial equipment requires inferring latent degradation parameters from indirect sensor measurements under uncertainty. While traditional Bayesian methods like Markov Chain Monte Carlo (MCMC) provide rigorous uncertainty quantification, their heavy computational b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast Bayesian equipment condition monitoring via simulation based inference: applications to heat exchanger health

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
工业设备（如**heat exchanger**）的健康状态监测通常依赖于对不可观测的**latent degradation parameters**（如结垢系数、泄漏率等）进行推断。传统**Bayesian inference**方法（如**MCMC**）虽能提供严格的不确定性量化，但由于其需要反复调用物理仿真模型，计算开销巨大，难以满足**real-time process control**的需求。

### 提出了什么新方法或新思路
本文提出了一种基于**Simulation-Based Inference (SBI)** 的AI驱动框架，利用**amortized neural posterior estimation**实现快速贝叶斯设备状态监测。具体而言：
- 采用**Sequential Neural Posterior Estimation (SNPE)**，训练一个神经密度估计器（Neural Density Estimator），学习从热流体观测数据到退化参数后验分布的直接映射。
- 该方法是**likelihood-free**的，无需显式构建似然函数，仅依赖前向仿真生成训练数据。
- 利用**amortization**机制，在离线训练后，推理阶段可实现近实时诊断。

### 相比现有方法的优势
| 维度 | 传统 MCMC | 本文 SBI 方法 |
|------|-----------|----------------|
| 推理速度 | 慢（每次推理需数千次仿真） | 快（推理时间加速 **82×**） |
| 可扩展性 | 难以部署于多资产、高频监控场景 | 支持大规模、实时部署 |
| 不确定性量化 | 提供严格后验分布 | 保持与 MCMC 相当的不确定性质量 |
| 模型兼容性 | 要求可微或解析似然 | 可用于“black-box”仿真器 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **合成数据集**：基于一个**stochastic heat exchanger model**生成，模拟了五类故障场景及一个无故障基准，共 **6 种 scenario**，每种生成 500 条带噪声的时间序列数据，总计 **3,000 条样本**。
- 故障机制建模为随机过程：
  - **Tube Fouling**：通过 Compound Poisson Process 模拟突发性结垢事件。
  - **Internal Leakage**：通过指数增长过程模拟流体流失。

### 实验设置和评估指标
#### 实验设置
- **SBI 训练**：使用 **50,000 次前向仿真** 进行离线训练，输入为 25 维**summary statistics**（包括温差、流量变化、趋势特征等）。
- **神经网络架构**：采用 **Neural Spline Flow (NSF)** 作为密度估计器，具备高表达能力以捕捉复杂后验。
- **MCMC 基线**：使用 **NumPyro** 实现，采用 **NUTS sampler**，每任务运行 4 条链 × (2,000 预热 + 3,000 采样) = 20,000 次仿真。

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Wasserstein Distance** | 衡量 SBI 与 MCMC 后验分布之间的相似性（越小越好） |
| **CRPS (Continuous Ranked Probability Score)** | 评估概率预测的准确性与锐度（越低越好） |
| **Credible Interval Coverage** | 检查真实参数是否落在置信区间内 |
| **Failure-mode Identification Accuracy** | 分类准确率，判断是否正确识别故障模式 |
| **Posterior Predictive Checks** | 可视化生成数据与观测的一致性 |

### 基线方法对比
- **Baseline**: **MCMC (NUTS)** —— 视为“黄金标准”后验参考。
- **Proposed**: **SBI with SNPE (NSF)** —— 所提方法。
- 对比维度包括：诊断精度、不确定性质量、推理效率。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **推理加速比** | **82× faster** than MCMC |
| **平均单次推理时间** | SBI: **0.029s/call**；MCMC: **2.4s/call** |
| **训练成本** | 一次性 **5,000 次仿真**（约 19s） |
| **break-even point** | 约 **6 次推理后** 成本低于 MCMC |

### 与基线方法的对比结果
#### （1）故障模式识别准确率（Table II）
| Scenario | MCMC 准确率 | SBI 准确率 |
|--------|-------------|-----------|
| Weak Fouling | 100% | 100% |
| Batch SD | 100% | 100% |
| Boiler FW | 100% | 100% |
| Mild Leak | 99.8% | 100% |
| Severe Leak | 99.6% | 100% |
| No Failure | 98.2% | 98.6% |

✅ **结论**：SBI 在所有场景下均达到与 MCMC 相当甚至更优的分类性能。

#### （2）参数估计质量
- **Figure 5 & 6** 显示 SBI 与 MCMC 的后验中位数高度一致，尤其在 `T`（故障起始时间）、`βf`、`β` 等关键参数上。
- **Figure 7**：1D Wasserstein distance 大部分集中在低值区域，表明后验形状接近。
- **Figure 8**：CRPS 分布几乎重叠，说明 SBI 保留了 MCMC 的预测锐度和校准性。

#### （3）挑战性场景分析（Scenario 2: Batch Process Shutdown）
- 特征：**低发生率（λ=0.5）+ 大跳跃幅度（βf=0.03）**
- 由于事件稀疏，观测信息极少，导致 `λ` 参数存在**结构性不可识别性**（structural identifiability limit）
- MCMC 和 SBI 均倾向于将 `λ` 后验收缩至先验中心（log-normal center ≈ 2.0），高于真实值（0.5）
- ✅ 尽管参数估计有偏，但**故障模式仍被正确识别**，且**预测轨迹物理合理**

### 消融实验结果（隐含分析）
虽然未明确列出消融实验，但文中进行了以下关键变量分析：
- **summary statistics 设计**：25维手工设计特征有效压缩信息，避免端到端时序建模的高维挑战。
- **prior choice 影响**：验证了在稀疏事件下 prior 主导 posterior 的现象，揭示了方法边界。
- **amortization effect**：发现 SBI 对 `λ` 的 posterior 更窄（正则化效应），而 MCMC 探索更广，反映 amortization 的平滑作用。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **SBI 可以在不牺牲诊断精度的前提下，实现高达 82× 的推理加速**，使其适用于 real-time condition monitoring。
2. ✅ **amortized neural posterior estimation 能够很好地逼近 MCMC 的后验分布**，在 failure-mode classification 和参数估计方面表现一致。
3. ✅ 即使在**低概率、稀疏事件**（low-probability, sparse-event）故障场景下，SBI 仍能稳健识别故障模式并生成合理的退化预测。
4. ✅ 该方法具有**model-agnostic**特性，适用于无法获取显式 likelihood 的“black-box”仿真系统，特别适合 legacy industrial systems。

### 方法的局限性
1. **结构性不可识别性问题**：在观测信息极弱的情况下（如稀疏结垢事件），参数估计受限于 prior 分布，难以完全恢复真实值。
2. **依赖仿真模型保真度**：若生成数据的 stochastic model 与实际物理不符，可能导致迁移失败。
3. **summary statistics 可能丢失信息**：手工设计的统计量可能未能充分利用原始时间序列中的全部动态信息。
4. **训练成本前置**：虽然推理快，但需要大量仿真数据进行离线训练，对复杂系统可能耗时。

### 未来工作方向
1. **Real-world validation**：在真实工业数据上测试方法鲁棒性，处理 sensor drift、unmodeled disturbances 等现实挑战。
2. **Adaptive / Online SBI**：开发支持在线更新的 SBI 框架，应对分布漂移（distributional shift）。
3. **Joint latent trajectory inference**：尝试同时推断参数与潜在退化路径 `z(t)`，提升诊断分辨率。
4. **Integration with Digital Twin**：将 SBI 集成至数字孪生平台，实现全厂级、风险感知的 predictive maintenance。
5. **Hybrid modeling**：结合 PINNs 或 differentiable simulators 进一步提升仿真效率与可解释性。

---

> 🔗 **代码开源地址**：  
> https://github.com/petercollett-cognite/sbi_mcmc_heat_exchanger.git

</details>

---

### 5. [F\textsuperscript{2}LP-AP: Fast \& Flexible Label Propagation with Adaptive Propagation Kernel](https://arxiv.org/abs/2604.20736)

**Authors**: Yutong Shen, Ruizhe Xia, Jingyi Liu, Yinqi Liu  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.20736v1  

#### Abstract
Semi-supervised node classification is a foundational task in graph machine learning, yet state-of-the-art Graph Neural Networks (GNNs) are hindered by significant computational overhead and reliance on strong homophily assumptions. Traditional GNNs require expensive iterative training and multi-lay...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《F²LP-AP: Fast & Flexible Label Propagation with Adaptive Propagation Kernel》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Graph Neural Networks (GNNs)** 在半监督节点分类任务中面临两大瓶颈：
- **计算开销大**：依赖梯度迭代训练和多层消息传递，难以在大规模图上高效部署。
- **强同质性假设（Homophily Assumption）**：假设相连节点标签相似，导致在**异质图（Heterophilous Graphs）** 上性能严重下降。

同时，现有的无训练方法（如 **Label Propagation, LP**）虽然高效，但采用固定传播规则，缺乏对局部拓扑结构的适应能力，易出现过平滑或噪声放大问题。

---

### ✅ 提出的新方法与思路
本文提出 **F²LP-AP**（Fast & Flexible Label Propagation with Adaptive Propagation Kernel），一种**完全无需训练**、计算高效的节点分类框架，其核心创新包括：

#### （1）**结构自适应传播机制（Structure-Adaptive Propagation）**
- 引入 **Local Clustering Coefficient (LCC)** 作为局部拓扑感知指标，动态调整每个节点的传播参数：
  - 高 LCC → 密集同质社区 → 减少传播步数 $K$，降低 teleport 概率 $\alpha$，促进特征平滑。
  - 低 LCC → 稀疏异质边界 → 增加 $K$，提高 $\alpha$，保留原始特征锚点。
- 映射函数 $f_\alpha(\text{LCC})$, $g_K(\text{LCC})$ 为预定义启发式规则，**无需学习**。

#### （2）**鲁棒原型构建（Robust Prototype Construction）**
- 使用 **Geometric Median** 而非算术均值构建类别原型（class prototype），显著增强对异常值和标签噪声的鲁棒性。
- 利用 **Weiszfeld算法** 迭代求解，仅需3–5次即可收敛，复杂度远低于梯度优化。

#### （3）**解析式分类（Analytical Classification）**
- 最终预测通过计算节点表示与类原型之间的 **cosine similarity** 完成，避免 Softmax 参数学习。
- 整个流程由确定性算法构成，**无任何可训练参数**，实现真正意义上的 training-free 推理。

---

### ✅ 相比现有方法的优势
| 维度 | F²LP-AP | 传统 GNNs | 经典 LP / APPNP |
|------|--------|----------|----------------|
| 是否需要训练 | ❌ 否 | ✅ 是 | ❌ 否 |
| 计算效率 | ⭐ 极高（秒级推理） | ⚠️ 高（GPU训练耗时） | ⭐ 高 |
| 对异质图适应性 | ✅ 强（基于LCC自适应） | ❌ 弱（依赖同质假设） | ❌ 弱（固定参数） |
| 抗噪能力 | ✅ 强（几何中位数） | ⚠️ 中等 | ❌ 弱（均值敏感） |
| 可扩展性 | ✅ 极佳 | ⚠️ 受限于内存和训练成本 | ✅ 佳 |

> 🎯 **核心优势**：在保持**零训练开销**的同时，实现了媲美甚至超越有监督 GNN 的精度，并具备跨同质/异质图的泛化能力。

---

## 2. 核心实验方法和设置

### ✅ 数据集
共使用 **8个基准图数据集**，按同质比（Homophily Ratio, H）分为两类：
- **强同质图（H > 0.8）**：
  - Cora, CiteSeer, PubMed（引文网络）
- **异质图（H < 0.4）**：
  - Texas, Wisconsin, Cornell（WebKB 页面）
  - Chameleon, Squirrel（Wikipedia 页面）

覆盖不同领域、规模（数百至数万节点）、同质程度，全面测试模型鲁棒性。

---

### ✅ 实验设置与评估指标
- **划分方式**：标准 60/20/20 训练/验证/测试划分。
- **评估指标**：
  - **Classification Accuracy**
  - **Macro-F1 Score**
  - **Execution Time (seconds)** —— 衡量推理效率
- **硬件环境统一**，确保公平比较。
- **随机种子固定为0**，保证可复现性。
- F²LP-AP 参数范围：
  - $K \in [2, 15]$
  - $\alpha \in [0.05, 0.2]$

---

### ✅ 基线方法对比
涵盖三类代表性方法：

#### （1）**经典与SOTA模型**
- **GCN***：有监督 GNN 基线（带*表示需训练）
- **Label Propagation (LP)**：传统无训练方法
- **kNN@5**：基于特征最近邻分类
- **CoHOp**：最新无训练 SOTA 方法

#### （2）**原型学习变体（Ablation Baselines）**
- **PrototypeOnly-Mean**：仅用均值原型 + 无传播
- **PrototypeOnly-GeoMed**：仅用几何中位数原型
- **FixedAPPNP-Proto**：固定参数 APPNP + 原型匹配（非自适应）

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table 1）

| Dataset | F²LP-AP (Acc.) | 最优基线 (Acc.) | GCN* (Acc.) |
|--------|----------------|------------------|-------------|
| **Cora (H=0.85)** | **0.835** | 0.821 (GCN*) | 0.821 |
| **CiteSeer (H=0.81)** | **0.708** | 0.719 (GCN*) | 0.719 |
| **PubMed (H=0.84)** | **0.782** | 0.798 (GCN*) | 0.798 |
| **Texas (H=0.31)** | **0.842** | 0.553 (GCN*) | 0.553 |
| **Wisconsin (H=0.37)** | **0.825** | 0.608 (GCN*) | 0.608 |
| **Cornell (H=0.34)** | **0.763** | 0.500 (GCN*) | 0.500 |
| **Chameleon (H=0.26)** | 0.395 | 0.561 (GCN*) | 0.561 |
| **Squirrel (H=0.23)** | 0.288 | 0.322 (GCN*) | 0.322 |

> 🔥 **亮点发现**：
> - 在多个**异质图**（Texas, Wisconsin, Cornell）上，F²LP-AP 不仅大幅超越 GCN（最高提升达 **+28.9%**），还达到 **SOTA 性能**。
> - 在同质图上也表现优异，**在 Cora 上超越所有方法**，包括有监督 GCN。
> - 所有结果均为 **training-free 方法中的最佳**（加粗标出）。

---

### ✅ 推理效率对比
- 在 **PubMed** 上，F²LP-AP 推理时间仅为 **0.044 秒**，而 GCN 需 **1.485 秒** → **提速约 33 倍**。
- 平均执行时间在 **0.01–0.09 秒之间**，远低于 CoHOp 和 GCN。
- 即使相比简单 LP（~0.013s），也仅略有增加，但带来了巨大性能增益。

---

### ✅ 消融实验结果（Ablation Study）

#### （1）**自适应传播的有效性**
- 相比 **FixedAPPNP**（固定 $K=5, \alpha=0.1$）：
  - Cora 上提升 **26.9%**（0.658 → 0.835）
  - Wisconsin 上提升 **13.8%**
- 表明：**动态参数调整对性能至关重要**，尤其在复杂拓扑中。

#### （2）**几何中位数 vs 算术均值（Figure 3）**
- **GeoMedian** 在所有数据集上均优于或等于 Mean。
- 在低同质图（如 Chameleon, Squirrel）上增益更明显：
  - Chameleon: +12.8%
  - Squirrel: +14.3%
- 验证了其对噪声和离群点的更强鲁棒性。

#### （3）**传播模块的重要性**
- **PrototypeOnly-GeoMed**（无传播）性能显著低于完整模型。
- 说明：**自适应传播是提取高阶结构信息的关键环节**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **F²LP-AP 实现了 training-free 方法的性能突破**：
   - 在多种图结构下达到甚至超越有监督 GNN 的精度。
   - 尤其在**异质图上表现卓越**，解决了传统 GNN 的“异质性瓶颈”。

2. **局部拓扑感知（LCC）可有效指导自适应传播**：
   - 无需训练即可实现节点级别的个性化传播策略。
   - 动态调节 $K$ 和 $\alpha$ 显著提升了模型灵活性与准确性。

3. **几何中位数是更鲁棒的原型估计方式**：
   - 对标签噪声和结构异常具有天然抵抗力。
   - 特别适用于真实世界中存在噪声的小样本场景。

4. **高效且稳定**：
   - 推理速度快（<0.1s），适合资源受限或实时应用。
   - 多次运行标准差小（如 Cora: ±0.004），表明高度稳定。

---

### ⚠️ 方法的局限性
1. **依赖单一拓扑指标 LCC**：
   - 在极端稀疏或高度噪声的图中可能失效。
   - 缺乏对多维结构特征的建模能力。

2. **启发式映射函数非数据驱动**：
   - $f_\alpha$, $g_K$ 为人工设计，未从数据中自动学习最优关系。
   - 泛化上限受限于函数形式和超参设定。

3. **性能仍受原始特征质量制约**：
   - 若输入特征本身判别性弱（如 Squirrel），即使优化传播也难以大幅提升准确率。

---

### 🔮 未来工作方向
1. **引入多维度结构描述符**（multi-dimensional structural descriptors）替代单一 LCC，提升拓扑感知能力。
2. 设计轻量级学习机制（lightweight learning）来优化映射函数，兼顾效率与表达力。
3. 结合自监督预训练特征，进一步释放无训练框架的潜力。
4. 探索在动态图、大规模工业图谱中的实际部署方案。

---

## ✅ 总结
F²LP-AP 是一项在 **training-free 图学习**方向上的重要进展。它通过 **LCC 驱动的自适应传播核** 与 **几何中位数原型构造**，成功实现了**高精度、高效率、强鲁棒性**的节点分类，在同质与异质图上均表现出色。该方法为构建无需训练、即插即用的图分析工具提供了新的范式，具有广阔的应用前景。

</details>

---

### 6. [Optimizing High-Throughput Distributed Data Pipelines for Reproducible Deep Learning at Scale](https://arxiv.org/abs/2604.21275)

**Authors**: Kashish Mittal, Di Yu, Roozbeh Ketabi, Arushi Arora, Brendon Lapp, Peng Zhang  
**Category**: cs.DC  
**Published**: 2026-04-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.21275v1  

#### Abstract
Training massive-scale deep learning models on datasets spanning tens of terabytes presents critical challenges in hardware utilization and training reproducibility. In this paper, we identify and resolve profound data-loading bottlenecks within distributed GPU training pipelines using the Petastorm...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *Optimizing High-Throughput Distributed Data Pipelines for Reproducible Deep Learning at Scale*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对大规模深度学习训练中两个关键瓶颈展开研究：
- **数据加载吞吐量低**：在使用分布式 GPU 训练时，由于网络 I/O 和 CPU 密集型数据转换（如 PyArrow 到 NumPy）导致 GPU 利用率极低（仅 10–15%），造成严重的 GPU 饥饿（GPU starvation）。
- **训练不可重现性**：即使固定随机种子，在多 worker 分布式环境下仍存在显著的运行间方差（run-to-run variance），影响模型评估与 A/B 测试的可靠性。

### 提出的新方法与创新思路
作者提出了一套优化的 **Petastorm 数据加载架构**，结合以下三项核心技术：

#### ✅ **Push-down Worker-Level Transformations**
- 将原本由主线程执行的 CPU 密集型数据格式转换（如 PyArrow → NumPy）下推到各个 Petastorm worker 线程中并行处理。
- 优势：释放主线程压力，避免其成为瓶颈；实现 CPU 负载均衡，提升整体 pipeline 吞吐量。

#### ✅ **Quota-Managed Local Disk Caching via FanoutCache**
- 引入 `diskcache.FanoutCache` 在本地磁盘缓存已转换的 NumPy 数据块。
- 设计配额管理机制（quota management），防止缓存无限增长，适用于超出单节点存储容量的大规模数据集。
- 优势：跨 epoch 避免重复从 HDFS 加载相同 row groups，消除冗余网络 I/O 与 CPU 开销。

#### ✅ **Deterministic Worker Scheduling with Round-Robin Queues**
- 替换原有的共享 ventilator 和 result queue 架构，为每个 worker 创建独立的输入/输出队列。
- 采用严格的 **round-robin** 调度策略分配 row groups，并按确定顺序合并结果。
- 升级 RNG 实现：弃用旧版 `np.random.RandomState`，改用更稳定的 `np.random.default_rng`。
- 优势：彻底消除多线程竞争条件（race conditions），确保完全可重现的数据加载序列。

### 相比现有方法的优势
| 方法 | 局限性 | 本工作的改进 |
|------|--------|--------------|
| PyTorch DataLoader / tf.data | 不支持高效读取分布式 Parquet 数据 | 基于 Petastorm 实现原生 Parquet 支持 |
| NVIDIA DALI | 主要面向图像任务（JPEG 解码等） | 针对表格型数据的序列化/反序列化优化 |
| In-memory caching (Alluxio/Redis) | 成本高，无法承载数十 TB 数据 | 使用本地磁盘 + 配额控制，性价比更高 |
| 默认 Petastorm | 存在非确定性调度与主线程瓶颈 | 全链路确定性 + 并行化转换 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **存储格式**：Apache Parquet™ 文件
- **存储系统**：Hadoop Distributed File System (HDFS)
- **数据规模**：
  - 数十 TB 大小
  - 数百亿行记录（tens of billions of rows）
  - 数百个特征字段（hundreds of features）
- 应用于工业级推荐系统模型训练

### 实验设置
- **计算平台**：多 GPU 实例组成的集群
- **集群管理器**：Ray™
- **分布式训练框架**：Horovod（基于 ring-allreduce 进行梯度同步）
- **数据加载器**：Petastorm（原始 vs 优化版本对比）
- **训练模式**：多 epoch 训练，典型场景为大规模 deep learning recommendation models

### 评估指标
| 指标类别 | 具体指标 |
|---------|----------|
| **性能指标** | end-to-end training time, GPU utilization (%) |
| **成本效益** | compute cost reduction |
| **可重现性** | run-to-run variance in training loss, Mean Average Precision (MAP) shift |
| **稳定性** | consistency across independent runs |

### 基线方法对比
- **Baseline**：标准 Petastorm 架构
  - 主线程负责数据转换
  - 所有 worker 共享 ventilator 和 result queue
  - 使用默认随机数生成器
- **Optimized**：本文提出的三重优化组合（push-down + caching + deterministic scheduling）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | Baseline | Optimized | 提升幅度 |
|------|----------|-----------|----------|
| **端到端训练时间** | ~22 小时/epoch | **~3 小时/epoch** | ⬇️ 减少约 86%（**6× speedup**） |
| **平均 GPU 利用率** | ~12% | **>60%** | ⬆️ 提升 5 倍以上 |
| **计算成本** | 基准水平 | **降低近 80%** | 显著节省云资源开销 |
| **MAP shift (run-to-run)** | ~0.5% | **~0.13%** | ⬇️ 下降超 70%，接近可忽略水平 |

### 与基线方法的对比结果
- **图示证据支持**：
  - 图5显示 baseline 下 GPU 大部分时间处于 idle 状态；
  - 图6展示优化后 GPU 利用率持续高于 60%，训练完成时间大幅缩短。
- **损失曲线一致性增强**：
  - 图7（baseline）显示不同运行间的 loss 曲线差异明显；
  - 图8（optimized）中各次运行几乎完全重合，验证了强 reproducibility。

### 消融实验分析（隐含于文中）
虽然未明确列出消融表，但从论述逻辑可推断各组件贡献：
- **仅解决网络 I/O（内存缓存）** → GPU 利用率短暂上升至 >60%，但本地磁盘缓存无效说明存在 CPU 瓶颈。
- **引入 push-down transformation** → 解除主线程阻塞，显著提升吞吐。
- **加入 FanoutCache** → 实现跨 epoch 缓存命中，进一步减少延迟。
- **重构调度队列为 round-robin + 独立队列** → 彻底消除运行间差异，MAP shift 显著下降。

---

## 4. 关键结论和发现

### 主要发现
1. **数据管道是大规模训练的实际瓶颈**：即便拥有高性能 GPU 和先进模型结构，若数据加载不匹配，硬件利用率将严重受限。
2. **全栈协同优化至关重要**：单纯加速某一部分（如网络或 CPU）不足以解决问题，必须同时优化 I/O、CPU 转换与并发控制。
3. **可重现性依赖底层系统设计**：看似“确定”的配置（固定 seed）仍可能因线程调度不确定性而失效，需从架构层面保障 determinism。
4. **工程优化可带来堪比算法进步的收益**：本文通过系统级改进实现了 **6× 训练加速** 和 **80% 成本节约**，效果远超多数模型结构调整。

### 方法的局限性
- **内存与磁盘权衡**：push-down transformation 牺牲一定内存效率以换取吞吐提升，对内存敏感场景需谨慎使用。
- **适用范围聚焦于表格数据**：当前优化主要针对 Parquet 格式的 tabular data，对图像、文本等其他模态适配需额外设计。
- **依赖特定生态系统**：方案深度集成 Ray + Horovod + Petastorm + HDFS，迁移到其他平台需较大改造。

### 未来工作方向
1. **平台级推广**：推动该优化架构成为公司内部所有 DL 模型的标准数据加载方式。
2. **开源贡献**：计划将这些改进回馈给开源社区，集成进公共 Petastorm 项目，助力更多组织应对类似挑战。
3. **自动化缓存策略**：探索动态调整缓存配额与预取策略，进一步提升缓存命中率。
4. **扩展至流式训练场景**：研究如何在持续更新的数据上维持 determinism 与高效 pipeline。

--- 

> **总结一句话**：  
> 本文证明，在超大规模深度学习中，**优化数据管道比升级模型更能显著提升训练效率与可复现性** —— 一次系统工程的胜利。

</details>

---

### 7. [A Task Decomposition and Planning Framework for Efficient LLM Inference in AI-Enabled WiFi-Offload Networks](https://arxiv.org/abs/2604.21399)

**Authors**: Mingqi Han, Xinghua Sun  
**Category**: cs.DC  
**Published**: 2026-04-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.21399v1  

#### Abstract
AI WiFi offload is emerging as a promising approach for providing large language model (LLM) services to resource-constrained wireless devices. However, unlike conventional edge computing, LLM inference over WiFi must jointly address heterogeneous model capabilities, wireless contention, uncertain t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Task Decomposition and Planning Framework for Efficient LLM Inference in AI-Enabled WiFi-Offload Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对 **AI-enabled WiFi-offload networks** 中的 **LLM inference offloading** 问题，解决了以下挑战：
- **任务复杂性不确定**：LLM 推理任务（如问答、推理、多模态理解）的难度、输出长度和质量难以预先估计。
- **异构资源环境**：多个 UE 和多个边缘 AP 具有不同的计算能力（Fe）、内存带宽（Abm）和模型规模（如 7B/14B/32B），需协同调度。
- **无线竞争与通信开销**：WiFi 网络中存在 CSMA/CA 冲突、RTS/CTS 握手、backoff 延迟等实际协议行为，影响端到端延迟。
- **传统 offloading 范式局限**：传统的 binary 或 partial offloading 无法有效利用 chain-of-thought 类任务的可分解性，且忽视语义依赖关系。

### 🚀 提出的新方法与创新思路
作者提出了一种 **基于 LLM 的任务分解与规划框架（LLM-based task decomposition and planning framework）**，其核心创新包括：

1. **任务分解 + 协同执行机制**
   - 支持三种模式：local-only、offload-edge（整任务卸载）、decomposition-and-plan（任务拆解后本地-边缘协同执行）。
   - 引入 **Planner LLM** 对输入任务进行智能拆解，生成具有角色、提示、依赖关系的 subtasks 集合 $ S(T) = \{S_1, ..., S_K\} $。

2. **子任务属性预测**
   - Planner 不仅负责拆分任务，还预测每个 subtask 的：
     - 预期输出 token 数量（output token length）
     - 执行难度（difficulty）
     - 在不同节点上的预期正确率（correctness）
   - 这些信息用于更精确地建模 **execution latency** 和 **response accuracy**。

3. **分解感知的调度策略（Decomposition-aware scheduling）**
   - 设计加权评分函数 $ J_{k,i} = w_a \alpha_{k,i} - w_d T_{\text{tot},k,i} $，综合考虑准确性与延迟。
   - 基于此选择最优执行节点（UE 或 edge AP）和聚合节点（aggregator）。

4. **轻量化 Planner 的蒸馏训练**
   - 使用强大的 teacher model（DeepSeek-v3.2）生成高质量规划标签。
   - 对 student model（Qwen2.5-7B-Instruct）进行监督微调（supervised fine-tuning），实现高性能且适合边缘部署的轻量级 planner。

### 🔍 相比现有方法的优势
| 方面 | 本文方法优势 |
|------|--------------|
| **灵活性** | 支持动态任务拆解 vs. 传统 binary/partial offloading 的固定粒度 |
| **精度提升** | 利用异构模型特长匹配子任务（如复杂推理交给大模型） |
| **延迟优化** | 显式建模 WiFi contention、queuing、communication overhead，做出更真实决策 |
| **实用性** | 蒸馏后的 lightweight planner 可部署在边缘设备上 |

---

## 2. 核心实验方法和设置

### 📚 数据集
任务采样自三个公开 benchmark，覆盖多样化场景：
- **AIME-2024 (Math)**：数学推理类任务
- **LiveBench-Reasoning (Daily)**：日常逻辑推理
- **GPQA (Science)**：科学领域高难度问题

> 注：这些数据集用于生成 prompt 并评估 response accuracy。

### ⚙️ 实验设置
| 参数 | 设置 |
|------|------|
| 场景大小 | 50m × 50m 室内环境 |
| Edge AP 数量 | 2 或 4 个 |
| UE 数量 | 5 或 7 个 |
| WiFi 标准 | 受 WiFi 7 启发，支持 RTS/CTS、binary exponential backoff、TF 触发帧 |
| 信道带宽 | 40 MHz |
| 中心频率 | 5 GHz |
| 路径损耗模型 | TGax indoor model（含墙体衰减） |
| 仿真时长 | 每 episode 600 秒，共 10 次独立运行取平均 |

#### 边缘与终端配置
| 类型 | 模型规模 | 计算能力（FP16） | 内存带宽 |
|------|----------|------------------|---------|
| Edge AP | 7B / 14B / 32B | [120, 312] TFLOPS | [0.6–2]×10¹² Bytes/s |
| UE | 1.5B | [20, 48] TFLOPS | [0.2–0.4]×10¹² Bytes/s |

#### 评估指标
- **Average Latency**：任务从提交到返回最终答案的时间
- **Response Accuracy**：生成答案与参考答案的一致性（人工或自动评分）
- **Overall Reward**：综合延迟与准确性的目标函数  
  $$
  \text{Reward} = w_a \cdot \text{accuracy} - w_d \cdot \text{latency}
  $$  
  （文中设 $ w_a = 1.0, w_d = 0.01 $）

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Local-Only** | 任务完全在 UE 上执行 |
| **Nearest-Edge** | 整个任务卸载至最近的 edge AP，不分解 |
| **Proposed Planner (DS-v3.2)** | 使用大型 teacher model 进行规划（非实际部署） |
| **Proposed Fine-tuned Planner (Qwen-7B)** | 经蒸馏训练后的轻量级 planner |
| **Original Qwen-7B Planner** | 未经 fine-tuning 的原始小模型作为对照 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Fig. 2）

| 方法 | Overall Reward | Average Latency (s) | Accuracy (avg.) |
|------|----------------|------------------------|----------------|
| **DS-v3.2 (Teacher)** | **0.318** | 13.773 | ~0.54 |
| **Fine-tuned Qwen-7B (Proposed)** | **0.222** | **12.049** | ~0.51 |
| **Nearest-Edge** | 0.122 | 14.542 | ~0.30 |
| **Local-Only** | -0.181 | 19.367 | ~0.15 |
| **Original Qwen-7B** | 0.087 | 16.300 | — |

> ✅ **关键结论**：
- 提出的方法相比 **Nearest-Edge**：
  - **降低平均延迟 20%**（14.542 → 12.049 s）
  - **提升总体 reward 80%**（0.122 → 0.222）
- 蒸馏后的轻量 planner 性能接近大模型，甚至 **延迟更低**（因推理更快）

### 📈 分数据集准确率表现（Fig. 2a）
| 方法 | AIME-2024 | LiveBench | GPQA |
|------|-----------|------------|-------|
| DS-v3.2 | 0.500 | 0.500 | 0.609 |
| Fine-tuned Qwen-7B | 0.429 | 0.429 | 0.600 |
| Nearest-Edge | 0.189 | 0.200 | 0.250 |
| Local-Only | 0.000 | 0.100 | 0.100 |

> 💡 发现：对于高难度任务（如 AIME 数学题），**任务分解带来的增益尤为显著**；而 Local-Only 几乎无法完成。

### 🔍 消融分析（隐含于结果比较中）
虽然未明确列出消融实验表格，但从以下对比可推断关键组件作用：

| 对比项 | 结果差异 | 推论 |
|--------|----------|------|
| DS-v3.2 vs. Original Qwen-7B | Reward: 0.318 vs. 0.087 | 表明 **planner 能力对性能至关重要** |
| Fine-tuned Qwen-7B vs. Original Qwen-7B | Reward: 0.222 vs. 0.087 | 验证 **teacher-guided fine-tuning 的有效性** |
| Proposed vs. Nearest-Edge | Latency ↓20%, Reward ↑80% | 证明 **task decomposition + collaborative execution 的优越性** |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **任务分解是提升 LLM 推理效率的关键**  
   将复杂任务拆分为语义相关的 subtasks，并分配给最适合的节点执行，可以显著提高响应质量和降低延迟。

2. **Planner 的估计能力直接影响系统性能**  
   能够准确预测 subtask 的 difficulty 和 output length 的 planner，才能做出高质量的调度决策。

3. **轻量化 Planner 经蒸馏后可达近似大模型性能**  
   Qwen-7B 经 DeepSeek-v3.2 指导训练后，在 reward 上达到 teacher 的 70%，同时延迟更低，具备实用价值。

4. **传统 offloading 策略在 AI-WiFi 场景下表现不佳**  
   - Local-Only：受限于 UE 算力，延迟高、准确率低；
   - Nearest-Edge：无法利用异构资源，缺乏灵活性。

5. **必须联合建模通信、计算与排队延迟**  
   忽视 WiFi contention、inter-node coordination 开销会导致调度失真。

### ⚠️ 局限性
- **仿真假设较强**：尚未在真实硬件平台上验证，尤其是 WiFi 7 的 TF 机制和 multi-AP coordination。
- **Planner 开销未完全计入**：planning 本身也有计算成本，虽小但仍可能影响实时性。
- **任务依赖建模较简单**：目前仅用 dependency set 表示先后关系，未深入建图结构（如 DAG）。
- **安全性与隐私未讨论**：任务拆解可能导致敏感信息泄露。

### 🔮 未来工作方向
1. **引入 adaptive decomposition threshold**：自动判断何时应拆解任务，避免过度拆分带来额外开销。
2. **更精细的 MAC 层建模**：结合 WiFi 7 的 multi-link operation (MLO) 和 scheduled access。
3. **动态负载感知调度**：实时监控各 AP 的 queue length 以优化负载均衡。
4. **跨层联合优化**：将 communication resource allocation（如功率控制、频段选择）与 computation scheduling 联合设计。
5. **支持更多 LLM 架构与推理范式**：如 MoE models、speculative decoding 等。

---

> ✅ **总结一句话**：  
> 本文提出了一个面向 AI-enabled WiFi-offload 网络的 **LLM 推理任务分解与规划框架**，通过 **Planner LLM 预测子任务特性 + 分解感知调度算法**，实现了比 local-only 和 nearest-edge 更优的 **latency-accuracy tradeoff**，并通过 **knowledge distillation** 使轻量级 planner 具备实用部署潜力。

</details>

---

### 8. [Decoupled DiLoCo for Resilient Distributed Pre-training](https://arxiv.org/abs/2604.21428)

**Authors**: Arthur Douillard, Keith Rush, Yani Donchev, Zachary Charles, Nova Fallen, Ayush Dubey, Ionel Gog, Josef Dean, Blake Woodworth, Zachary Garrett, Nate Keating, Jenny Bishop, Henry Prior, Edouard Yvinec, Arthur Szlam, Marc'Aurelio Ranzato, Jeff Dean  
**Category**: cs.CL  
**Published**: 2026-04-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.21428v1  

#### Abstract
Modern large-scale language model pre-training relies heavily on the single program multiple data (SPMD) paradigm, which requires tight coupling across accelerators. Due to this coupling, transient slowdowns, hardware failures, and synchronization overhead stall the entire computation, wasting signi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Decoupled DiLoCo for Resilient Distributed Pre-training》论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现代大规模语言模型（LLM）预训练严重依赖**单程序多数据**（SPMD）范式，该范式要求所有加速器在每一步进行全局同步。这种紧耦合架构存在以下关键瓶颈：

- **可靠性差**：单个硬件故障或延迟（straggler）会导致整个训练停滞。
- **资源浪费**：频繁的同步开销和故障恢复过程造成大量计算时间浪费。
- **扩展性受限**：随着芯片数量增加，硬件故障率呈线性上升，系统可用性急剧下降。

尽管已有如 **DiLoCo** 和 **Streaming DiLoCo** 等方法降低了通信带宽，但它们仍保持同步机制，无法从根本上解决因硬件故障导致的训练中断问题。

---

### 提出了什么新方法或新思路

本文提出 **Decoupled DiLoCo**，一种去耦合、异步的分布式训练框架，旨在打破锁步同步（lock-step synchronization）壁垒，实现高可用、高容错的预训练。

其核心思想是将传统的“一致性优先”（Consistency-first）转向“可用性优先”（Availability-first），借鉴 CAP 定理视角，牺牲强一致性以换取更高的 **Availability** 和 **Partition Tolerance**。

#### 主要创新点包括：

- **解耦学习者（Learners）**：将全局集群划分为多个独立、异步运行的 `Learner`，每个 Learner 在自己的数据分片上执行本地优化（inner optimization），无需等待其他 Learner。
- **中心化同步器（Syncer）**：引入一个轻量级的中央 `Syncer`，负责异步聚合 Learner 发送的参数片段（fragments），并执行外层优化（outer optimization）。
- **最小法定数聚合（Minimum Quorum Aggregation）**：Syncer 不等待所有 Learner，只需收到至少 `K` 个 Learner 的更新即可进行聚合，避免因个别 Learner 故障而阻塞。
- **自适应宽限期（Adaptive Grace Window）**：利用计算与通信之间的空闲时间，动态延长等待窗口，尽可能纳入更多 Learner 的更新，提升样本效率而不影响吞吐。
- **基于 token 的动态加权合并（Token-weighted Merging）**：根据 Learner 处理的 token 数量和步数动态赋予权重，缓解速度差异带来的偏差。
- **径向方向平均（Radial-Directional Averaging, RDA）**：提出一种新的梯度合并策略，分别对梯度的方向和模长进行平均，提高外层优化器的超参稳定性。

---

### 相比现有方法的优势

| 特性 | Standard Data-Parallel | Elastic Data-Parallel | Decoupled DiLoCo |
|------|------------------------|------------------------|------------------|
| 同步模式 | 全局同步（Synchronous） | 部分弹性同步 | 完全异步（Asynchronous） |
| 故障容忍 | 差（需重启） | 中等（可缩容） | 极强（局部隔离） |
| 通信带宽 | 高 | 中等 | 极低（碎片化 + 异步） |
| 可用性（Uptime） | 易中断 | 有停机时间 | 接近 100% |
| 扩展性 | 受限于同步开销 | 受限于重配置开销 | 极佳 |
| 支持异构硬件 | 困难 | 困难 | 天然支持 |

---

## 2. 核心实验方法和设置

### 使用的数据集

实验采用混合文本与视觉数据进行预训练，下游任务涵盖多个标准基准：

#### 文本任务（Text Benchmarks）
- **ARC-Challenge/Easy**：科学常识问答
- **BoolQ**：阅读理解是非题
- **HellaSwag**：常识推理续写
- **PIQA**：物理常识推理
- **SIQA**：社交常识推理
- **Winogrande**：代词消解与常识推理

#### 视觉任务（Vision Benchmarks）
- **ChartQA**：图表逻辑推理
- **COCO-Captions**：图像描述生成
- **DocVQA**：文档图像问答
- **InfographicVQA**：信息图问答
- **MMMU**：多学科专家级多模态理解
- **TextVQA**：图像内文本识别与推理

---

### 实验设置和评估指标

#### 模型与规模
- 使用 **Gemma** 系列模型（2B, 5B, 9B 参数）进行密集模型实验。
- 使用激活参数为 2.8B 和 3.8B 的 **Mixture-of-Experts (MoE)** 架构。
- 模型被划分为 `P=24` 个非重叠的参数片段（fragments），每 `H=24` 步同步一次。

#### 硬件模拟
- 模拟多达 **240万** 芯片的训练环境，假设单芯片平均无故障时间（MTBIchip）为 1 年。
- 通过泊松分布模拟随机故障，并使用指数韦布尔分布模拟恢复延迟。
- 引入“混沌工程”（Chaos Engineering）测试极端故障场景下的鲁棒性。

#### 评估指标
- **Goodput**：集群中实际用于有效计算的时间占比，衡量系统效率。
- **System Uptime**：系统持续推进训练步骤的比例，反映可用性。
- **下游任务性能**：各基准任务上的准确率或得分，评估模型质量。
- **MFU**（Model Flops Utilization）：虽未直接报告，但文中指出无回归。

---

### 基线方法对比

- **Standard Data-Parallel (DP)**：传统同步数据并行。
- **Elastic Data-Parallel**：支持动态扩缩容的弹性版本。
- **Decoupled DiLoCo (M=8)**：本文方法，使用 8 个 Learner。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在极端硬件故障下（MTBIchip=1年, Nchip=1.2M）的 Goodput 对比：

| 方法 | Goodput |
|------|---------|
| Standard DP (M=1) | 58% |
| Elastic DP (M=1) | 58% |
| **Decoupled DiLoCo (M=8)** | **88%** |

> ✅ **Decoupled DiLoCo 将 Goodput 提升了约 30个百分点**。

#### 系统可用性（Uptime）表现：

| Learners M | Nchip=2.4M 下的 Uptime |
|------------|------------------------|
| 1 (DP) | 18% |
| 8 | 80% |
| 16 | 99% |

> ✅ 当 `M≥8` 时，系统可达 **100% 运行时间**，真正实现零全局停机。

---

### 与基线方法的对比结果

#### 在 5B 密集模型上的下游性能（Table 2）：

| 方法 | Text (Avg) | Vision (Avg) | Goodput |
|------|------------|--------------|---------|
| Data-Parallel (理想环境) | 70.1 | 58.7 | 100% |
| Elastic DP (1.2M chips) | 69.7 | 59.4 | 58% |
| **Decoupled DiLoCo (M=8)** | **69.8** | **58.6** | **88%** |

> ✅ **Decoupled DiLoCo 在显著更高 Goodput 下，实现了与标准 DP 相当甚至略优的模型性能**。

#### MoE 模型结果（Table 3）：

| 方法 | Text (Avg) | Vision (Avg) |
|------|------------|--------------|
| Data-Parallel | 68.4 | 54.9 |
| **Decoupled DiLoCo (M=8)** | **68.4** | **55.1** |

> ✅ 在 MoE 架构下同样保持竞争力。

---

### 消融实验结果

#### （1）不同 Learner 数量的影响（Table 11）

| M | Goodput (1.2M chips) | Text (Avg) |
|----|------------------------|-------------|
| 2 | 70% | 54.4 |
| 4 | 82% | 55.0 |
| 8 | 88% | 55.4 |
| 16 | 93% | 53.4 |

> 🔍 更多 Learner 提升 Goodput，但 `M=16` 时略有性能下降，可通过更大模型/更多 token 补偿。

#### （2）合并策略对比（Table 8–9）

- 在 `M=8` 时，`RDA`（非嵌入部分）+ `Avg`（嵌入部分）表现最佳。
- 在 `M=16` 时，`RDA` 显著优于直接平均（Avg），尤其在非嵌入层。

#### （3）外梯度压缩（Table 10）

| 压缩精度 | Text (Avg) | Vision (Avg) |
|----------|------------|--------------|
| bf16 (16-bit) | 55.4 | 36.6 |
| **int4 (4-bit)** | **55.3** | **34.4** |
| int2 | 42.6 | 5.3 |

> ✅ **int4 压缩几乎无损**，大幅降低通信开销。

#### （4）碎片化策略（Table 7）

| 策略 | Text (Avg) | Vision (Avg) |
|-------|------------|--------------|
| Layer Fragmentation | 55.3 | 35.5 |
| Tensor Fragmentation | 55.4 | 35.8 |
| **Balanced Tensor Fragmentation** | **55.4** | **35.8** |

> ✅ **平衡张量碎片化** 在系统层面最优（更均衡带宽），且不影响模型性能。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Decoupled DiLoCo 成功打破了 SPMD 的同步瓶颈**，通过 Learner-Syncer 架构实现了真正的异步训练。
2. ✅ **系统可用性接近 100%**，即使在百万级芯片规模下也能维持极高 Goodput（88% vs 基线 58%）。
3. ✅ **模型性能不受影响**：在文本和视觉任务上，Decoupled DiLoCo 与标准数据并行训练的模型质量相当，甚至在某些任务上更优。
4. ✅ **天然支持异构与机会性算力**：可无缝整合不同代际的 TPU（如 v5e 与 v5p），并支持“算力捡漏”（scavenging），显著缩短训练时间。
5. ✅ **后训练能力完整保留**：经过相同 post-training 流程后，Decoupled DiLoCo 预训练的模型在数学、代码、多语言等任务上表现一致甚至更好（Table 6）。

---

### 方法的局限性

- **需要额外的 Syncer 组件**：虽然 Syncer 资源消耗小（仅 CPU），但仍引入了新的系统复杂性。
- **对 Learner 数量敏感**：过少 Learner 会降低容错性；过多可能轻微影响收敛（可通过调优缓解）。
- **依赖 Pathways 调度系统**：目前实现高度依赖 Google 内部的 Pathways 架构，通用性有待验证。
- **极端压缩风险**：外梯度压缩至 2-bit 或 1-bit 会导致性能崩溃，需谨慎选择压缩策略。

---

### 未来工作方向

1. **进一步优化 Syncer 架构**：探索完全去中心化的 Syncer 或 P2P 同步机制。
2. **结合更先进的压缩技术**：如 MuLoCo、TIES-Merging 等，进一步降低通信成本。
3. **跨地域分布式训练**：利用 Decoupled DiLoCo 对网络延迟不敏感的特性，推动全球范围内的协作训练。
4. **应用于更大规模模型**：扩展至万亿参数级别，验证其在极限规模下的有效性。
5. **开放系统实现**：推动该框架在开源社区的应用与演进。

---

## 总结

**Decoupled DiLoCo** 是一项面向未来的大规模预训练基础设施革新。它不再追求“完美同步”，而是拥抱现实世界的混乱与不确定性，通过**解耦、异步、最小法定数、动态加权**等设计，在不牺牲模型性能的前提下，极大提升了系统的**韧性、可用性和资源利用率**。该工作标志着从“一致性优先”到“可用性优先”的范式转变，为构建真正鲁棒、可持续、可扩展的 AI 基础设施指明了方向。

</details>

---

### 9. [Accelerating PayPal's Commerce Agent with Speculative Decoding: An Empirical Study on EAGLE3 with Fine-Tuned Nemotron Models](https://arxiv.org/abs/2604.19767)

**Authors**: Ally Qin, Jian Wan, Sarat Mudunuri, Srinivasan Manoharan  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.19767v1  

#### Abstract
We evaluate speculative decoding with EAGLE3 as an inference-time optimization for PayPal's Commerce Agent, powered by a fine-tuned llama3.1-nemotron-nano-8B-v1 model. Building on prior work (NEMO-4-PAYPAL) that reduced latency and cost through domain-specific fine-tuning, we benchmark EAGLE3 via vL...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accelerating PayPal's Commerce Agent with Speculative Decoding: An Empirical Study on EAGLE3 with Fine-Tuned Nemotron Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 PayPal 的 **Commerce Agent** 生产环境中，尽管已通过领域特定微调（domain-specific fine-tuning）显著降低了延迟和成本，但在满足严格 SLA（如响应时间 <2 秒）的同时进一步提升吞吐量并降低硬件开销仍具挑战。  
本文聚焦于推理阶段的效率瓶颈，探索如何在不牺牲输出质量的前提下加速 **autoregressive generation**。

### 🚀 提出的新方法/新思路
提出将 **speculative decoding** 技术应用于 PayPal 商业场景中的微调模型，并采用 **EAGLE3** 这一无需训练的 draft model 架构进行推理优化：
- 在 **vLLM** 上部署 fine-tuned `llama3.1-nemotron-nano-8B-v1` 模型，结合 EAGLE3 实现 speculative decoding。
- 与生产环境使用的 **NVIDIA NIM** 基线（基于标准自回归解码）在相同硬件上对比，验证其有效性。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **无需额外训练** | EAGLE3 是 training-free 方法，避免维护独立 draft model 的运维复杂性 |
| **零额外硬件成本** | 在相同的 2×H100 GPU 上运行，实现更高性能 |
| **可预测性强** | 接受率（acceptance rate）在不同并发和温度下高度稳定，便于容量规划 |
| **性价比高** | 单张 H100 上的 speculative decoding 可媲美甚至超越双卡 NIM 性能，潜在节省 **50% GPU 成本** |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- 并未使用公开数据集，而是基于 **PayPal Commerce Agent 的真实生产工作负载**。
- 聚焦于 **query formulation 组件**：将自然语言购物查询转化为结构化搜索参数（JSON 格式），用于下游商品检索与推荐。

### ⚙️ 实验设置
| 项目 | 配置 |
|------|------|
| **目标模型** | fine-tuned `llama3.1-nemotron-nano-8B-v1`（via LoRA） |
| **部署方式** | vLLM + EAGLE3（实验组） vs. NVIDIA NIM（基线） |
| **硬件平台** | 2×NVIDIA H100 GPU, 50 CPU cores, 400 GiB RAM |
| **speculative token 数量** | γ = 3（Spec-3）、γ = 5（Spec-5） |
| **并发级别（concurrency）** | 1, 4, 8, 16, 32 |
| **采样温度（temperature）** | 0（greedy decoding）、0.5（stochastic sampling） |
| **总配置数** | 5 × 2 × 2 = **40 种实验配置** |
| **每轮测试请求量** | 50 个请求（前 3 个为预热） |

### 📈 评估指标
| 类别 | 指标 |
|------|------|
| **性能指标** | Throughput (req/s), Mean/P50 Latency (ms), GPU Utilization (%) |
| **接受率** | Acceptance Rate = Accepted Draft Tokens / Total Draft Tokens |
| **输出质量** | LLM-as-Judge：<br>• Individual scoring（5 分制，5 项标准）<br>• Pairwise comparison（A/B 测试 + 位置随机化防偏） |

### 🆚 基线方法对比
- **Baseline**: NVIDIA NIM 部署的原生 autoregressive 推理（当前生产方案）
- **Method**: vLLM + EAGLE3 speculative decoding（same model checkpoint）
- 控制变量：相同模型权重、相同硬件资源、相同 guided JSON 输出格式

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ Throughput 提升（vs. NIM 基线）
| 并发等级 | Spec-3 提升 | Spec-5 提升 |
|--------|-------------|-------------|
| 1      | **+49%**     | +34%        |
| 4      | +48%         | +22%        |
| 8      | +29%         | +4%         |
| 16     | +22%         | +4%         |
| 32     | +25%         | +14%        |

> ➤ **Spec-3 实现 22–49% 吞吐提升**

#### ✅ Latency 降低
| 并发等级 | Spec-3 降幅 | Spec-5 降幅 |
|--------|-------------|-------------|
| 1      | **-33%**     | -25%        |
| 4      | -32%         | -18%        |
| 8      | -22%         | -3%         |
| 16     | -18%         | -4%         |
| 32     | -20%         | -12%        |

> ➤ **Spec-3 实现 18–33% 延迟下降**

#### ✅ Acceptance Rate 特性分析
| 配置 | Acceptance Rate |
|------|------------------|
| γ=3（所有条件） | **~35.5%**（极其稳定） |
| γ=5（所有条件） | **~25%**（较低且随并发上升略降） |
| 温度影响（t=0 vs t=0.5） | 几乎无差异（<1% 变化） |

> ➤ 接受率不受并发或温度影响 → 易于生产部署

#### ✅ GPU 利用率表现
- 尽管吞吐更高，**Spec-3 的 GPU 利用率通常低于或等于基线**，尤其在高并发时更低（如 c=32 时降至 49.9% vs NIM 的 67.8%）
- 表明 speculative decoding 更高效地利用计算资源，而非“耗更多算力”

#### ✅ 单 GPU vs 双 GPU 对比（关键发现）
| 指标 | Spec (1×H100) | NIM (2×H100) | 对比 |
|------|---------------|--------------|------|
| Throughput | 3.19 req/s | 3.14 req/s | **+6.0%** |
| Mean Latency | 2,292ms | 2,529ms | **-4.4%** |
| P50 Latency | 2,279ms | 2,485ms | **-4.5%** |
| GPU Cost | 1×H100 | 2×H100 | **-50%** |

> ➤ **单卡 speculative 解码 ≈ 或优于双卡 NIM，节省一半 GPU 成本**

#### ✅ 输出质量保持
- **LLM-as-Judge 评估显示：**
  - 所有配置下 individual score 均为 **5.00/5.00**
  - Pairwise comparison 中 win/tie 分布均衡，无系统性偏好
  - Temperature 0 时几乎完全一致（tie rate >90%），符合理论保证

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Speculative decoding with EAGLE3 显著加速推理**：
   - 在真实电商场景中实现 **22–49% 吞吐提升** 和 **18–33% 延迟下降**
   - 不改变模型或输出分布，**完全保留生成质量**

2. **γ=3 是最优配置**：
   - γ=5 因接受率低（~25%）导致收益递减，且浪费更多计算
   - γ=3 接受率稳定在 ~35.5%，跨并发和温度不变，适合生产部署

3. **硬件效率大幅提升**：
   - **单张 H100 上的 speculative decoding 可替代双卡 NIM 部署**
   - 实现 **50% GPU 成本节约**，同时满足 sub-2-second SLA（P50 ≤ 1.6s, P99 ≤ 2.6s）

4. **温度对接受率无显著影响**：
   - 与先前研究不同，temperature 0 与 0.5 接受率几乎一致
   - 归因于 **guided JSON 输出限制了 token 分布熵**，削弱温度调节作用
   - 对生产友好：可自由调整多样性而不影响推理效率

5. **完整优化路径形成**：
   - 结合前期工作 [1] 的 **LoRA 微调** 与本文的 **speculative decoding**
   - 实现端到端累计 **>60% 延迟降低**

---

### ⚠️ 局限性
1. **任务单一性**：仅评估 query formulation 组件，其他 agent 子任务可能表现不同
2. **利用率采样粒度粗**：Prometheus 每 60 秒采集一次，无法反映细粒度波动
3. **draft model 未微调**：EAGLE3 使用通用架构，若能在 commerce 数据上 fine-tune draft model，接受率有望提升
4. **评价偏倚风险**：使用同一类 Nemotron 模型作为 judge，可能存在评估偏差

---

### 🔮 未来工作方向
1. **Fine-tune EAGLE3 draft model** on commerce-specific data to improve acceptance rate
2. Extend evaluation to other components of the **multi-agent commerce system**
3. Explore alternative speculative decoding frameworks (e.g., **Medusa**, **lookahead decoding**)
4. Investigate dynamic adjustment of γ based on input type or context length
5. Integrate with other optimizations (quantization, prefix caching) for compounded gains

--- 

> 💡 **总结一句话**：  
> 本文证明，在 PayPal 商务 Agent 场景中，**EAGLE3 + speculative decoding 是一种低成本、高质量、易部署的推理加速方案**，可在不增加硬件的情况下显著提升性能，并支持 **50% GPU 成本削减**，为大规模 LLM 应用提供了实用的工程范式。

</details>

---

### 10. [Bridging the Training-Deployment Gap: Gated Encoding and Multi-Scale Refinement for Efficient Quantization-Aware Image Enhancement](https://arxiv.org/abs/2604.21743)

**Authors**: Dat To-Thanh, Nghia Nguyen-Trong, Hoang Vo, Hieu Bui-Minh, Tinh-Anh Nguyen-Nhu  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.21743v1  

#### Abstract
Image enhancement models for mobile devices often struggle to balance high output quality with the fast processing speeds required by mobile hardware. While recent deep learning models can enhance low-quality mobile photos into high-quality images, their performance is often degraded when converted ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Bridging the Training-Deployment Gap: Gated Encoding and Multi-Scale Refinement for Efficient Quantization-Aware Image Enhancement

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**移动设备上的图像增强模型在训练与部署之间存在显著性能差距**的问题。具体表现为：
- 模型在训练时通常采用 FP32/FP16 高精度浮点格式，但在实际部署到手机等边缘设备时需转换为 INT8 低精度量化格式。
- 这种转换会导致严重的**视觉退化**，如颜色偏移（color shift）、条带效应（banding）和纹理失真，尤其是在对像素级敏感的任务（如图像增强、超分辨率）中更为明显。
- 现有 Post-Training Quantization (PTQ) 方法无法有效应对 Deep ISP 模型中非高斯、长尾、输入依赖性强的激活分布。

### 提出了什么新方法或新思路
作者提出了一套**面向移动端部署优化的高效图像增强框架**，其核心创新包括：

- **Quantization-Aware Training (QAT) 框架集成**  
  将 QAT 显式引入图像增强任务的训练流程中，通过 FakeQuant 操作模拟 INT8 推理过程中的舍入与裁剪噪声，使网络在训练阶段就学习适应低精度表示。

- **Gated Encoder Block**  
  设计一种双分支门控编码器结构，利用 `tanh` 激活生成两个特征流，并通过 Hadamard 积进行元素级门控调制。同时保留原始分支输出 `(xa, xb)` 作为多通道 skip connection 输入解码器，提升细节重建能力。

- **Multi-Scale Refinement 策略**  
  在编码器和解码器路径中多个尺度（S/2, S/4, S/8）插入 UNet-style 的残差 refinement block，联合优化全局光照结构与局部纹理细节。

- **Multi-branch Feature Fusion**  
  解码器融合上采样特征、精炼后的编码器特征以及原始门控分支的 skip 特征，实现多层次语义聚合。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **部署一致性** | 采用 QAT 实现训练-部署一致性，避免 PTQ 导致的颜色失真和结构崩塌 |
| **感知质量保持** | 在 INT8 下仍能维持接近 FP32 的 PSNR 和 SSIM 性能 |
| **计算效率** | 模型轻量且支持 TFLite 和 QNN HTP 加速，在骁龙 8 Gen 2 上实现 41.8ms 超低延迟 |
| **可扩展性** | 支持不同 channel width 配置，适用于从资源受限到高性能场景 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **DPED dataset**：包含 iPhone、Canon 70D 和 Sony Alpha 相机同步拍摄的图像对。
- **训练集**：超过 160k 对 100×100 的 iPhone 与 Canon 图像 patch。
- **验证集**：9310 对来自 iPhone、Sony、BlackBerry 的测试 patch。

### 实验设置
- **实现平台**：PyTorch + PyTorch Lightning
- **优化器**：Adam，cosine annealing 学习率调度（warmup 5 epoch）
- **初始学习率**：0.00001 → 最大 0.0001
- **Batch Size**：有效 batch size 128（gradient accumulation ×2）
- **训练精度**：bfloat16
- **梯度裁剪**：[-1.0, 1.0]
- **Loss 权重**：α=2.0, β=1.0, γ=1.0（对应 PSNR、Cosine Similarity、Outlier-Aware Loss）

### 评估指标
| 指标 | 描述 |
|------|------|
| **PSNR** | 峰值信噪比，衡量像素级保真度 |
| **SSIM** | 结构相似性，反映结构一致性 |
| **MOS** | 主观评分（Mean Opinion Score） |
| **Latency** | 推理延迟（单位：ms），在 Snapdragon 8 Gen 2 手机上使用 AI Benchmark 测试 |
| **Final Score** | 综合得分（基于 PSNR、MOS、延迟等加权） |

### 基线方法对比
- **Baseline [11]**：DPED 原始 CNN 方法
- **PPCN [10]**：Perception-Preserving Convolutional Networks
- **PTQ Models**：直接对 FP32 模型进行后训练量化
- **QAT Models**：本文提出的量化感知训练版本

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Mobile AI 2026 Challenge 成绩）
| 方法 | PSNR (dB) | SSIM | MOS | Adreno GPU (ms) | Arm GPU (ms) | Final Score |
|------|-----------|-------|-----|------------------|---------------|--------------|
| DaHua-IIG | 22.20 | 0.7881 | 4.1 | 23.8 | 60.4 | 163.0 |
| **Capybara (Ours)** | **21.82** | **0.7653** | **3.2** | **291.0** | **266.0** | **3.8** |
| DH-XHDL-Team | 20.55 | 0.7601 | 1.2 | 30.8 | 52.4 | 0.28 |

> 注：尽管未获第一，但在真实设备推理条件下表现稳健。

### 与基线方法的对比结果（Qualitative Analysis）
- **FP32 模型**：能恢复清晰文字（如车牌）和精细纹理，视觉质量接近 baseline 和 PPCN。
- **PTQ (8-bit)**：出现严重颜色偏差（如天空变粉红）、噪声伪影和模糊。
- **QAT (8-bit)**：成功抑制颜色偏移，恢复大部分结构细节，视觉效果接近 FP32 模型。

### 消融实验结果

#### ✅ 架构组件消融（Table 2）
| 变体 | PSNR (dB) | SSIM | Latency (ms) |
|------|------------|--------|----------------|
| Full Model | **22.194** | **0.796** | 469 |
| w/o Residual | 22.029 | 0.793 | 464 |
| w/o Fusion Refiner | 21.940 | 0.793 | 396 |
| w/o Res Refiner | 20.398 | 0.789 | **237** |

> 发现：refinement 模块是保证高质量的关键，去除任一 refinement 均导致 PSNR 显著下降；global residual 路径虽仅增加 5ms 开销，但带来重要质量增益。

#### ✅ 通道宽度消融（Table 3）
| Channel Width (c) | PSNR (dB) | SSIM | Params (M) | Latency (ms) |
|--------------------|------------|--------|-------------|----------------|
| 16 | 21.875 ❌ | 0.781 | 0.23 | **180** |
| 24 | 22.035 | 0.785 | 0.516 | 249 |
| **32** | **22.194** | **0.796** | **0.915** | 469 |
| 64 | 22.359 | 0.806 | 3.651 | 1432 |

> 结论：c=32 是最优平衡点——相比 c=64 减少 **74.9% 参数量** 和 **67.24% 延迟**，仅牺牲轻微精度。

#### ✅ 损失函数消融（Table 4）
| Loss Function | Train Time (hrs) | PSNR (dB) | SSIM |
|---------------|------------------|------------|--------|
| **Our loss** (PSNR + Cosine + Outlier) | **1.67** | **22.194** | 0.796 |
| Loss variant 1 (MSSSIM 替换 Cosine) | 8.58 | 19.79 | **0.8763** |

> 发现：虽然 MSSSIM 提升结构相似性，但训练耗时长、PSNR 更低，实用性差；本文损失更高效且更适合快速收敛。

#### ✅ 量化策略对比（Table 5）
| 模型类型 | PSNR (dB) | SSIM | TFLite GPU (ms) | QNN HTP (ms) |
|----------|------------|--------|------------------|----------------|
| FP32 QAT | 22.194 | 0.796 | 469 | 151 |
| INT8 PTQ | 20.576 | 0.6139 | 319 | 41.4 |
| **INT8 QAT (Ours)** | **21.050** | **0.725** | **319** | **41.8** |

> 关键发现：
> - QAT 在 INT8 下比 PTQ 提升 **+0.474 dB PSNR** 和 **+0.111 SSIM**
> - 利用 QNN HTP 加速器，INT8 QAT 实现 **~3.6x 速度加速**（151ms → 41.8ms）

---

## 4. 关键结论和发现

### 主要发现
1. **训练-部署不一致是移动端图像增强的主要瓶颈**，尤其在 INT8 推理下易引发严重视觉伪影。
2. **QAT 是缓解量化退化的有效手段**，能够显著提升 INT8 模型的感知质量和数值稳定性。
3. **Gated Encoder + Multi-Scale Refinement 架构有助于保留细粒度特征**，特别适合处理颜色敏感和动态范围大的图像增强任务。
4. **合理的损失设计（PSNR + Cosine + Outlier-Aware）可在保证训练效率的同时获得高质量输出**。
5. **结合 QNN HTP 等专用 NPU 可实现极致低延迟推理**，证明该方案具备强工业落地潜力。

### 方法的局限性
- 当前模型在极端低光或高噪声输入下的鲁棒性尚未充分验证。
- 多尺度 refinement 增加了计算负担，对于极低端设备可能仍显沉重。
- 所有实验基于 DPED 数据集，泛化能力有待在更多真实场景中检验。

### 未来工作方向
- 探索 **data-free QAT** 或 **zero-shot calibration** 技术以降低标注成本。
- 引入 **neural architecture search (NAS)** 自动寻找最优 channel width 与 depth 配置。
- 扩展至视频增强任务，研究时序一致性约束下的 QAT 优化策略。
- 探索 **混合精度量化**（如部分层保留 FP16）以进一步平衡性能与效率。

---

> 🔗 **代码地址**：https://github.com/GenAI4E/QATIE.git

</details>

---

### 11. [Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems](https://arxiv.org/abs/2604.21794)

**Authors**: Ye Yu, Heming Liu, Haibo Jin, Xiaopeng Yuan, Peng Kuang, Haohan Wang  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.21794v1  

#### Abstract
Multi-agent systems built on large language models have shown strong performance on complex reasoning tasks, yet most work focuses on agent roles and orchestration while treating inter-agent communication as a fixed interface. Latent communication through internal representations such as key-value c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前基于 **Large Language Models (LLMs)** 的 **Multi-Agent Systems (MAS)** 在复杂推理任务中表现出色，但其性能提升主要集中在 **agent 角色设计** 和 **工作流编排** 上。而 **inter-agent communication** 通常被视为一个固定的、不可学习的接口，尤其是依赖于 **text-based communication**（即通过自然语言传递中间结果）。

这种设计存在两个根本缺陷：

1. **信息损失**：内部连续的隐状态（如 hidden states 或 KV caches）必须被 **离散化为文本 token** 才能传递，导致信息压缩和保真度下降。
2. **优化边界**：离散通信阻断了梯度传播，使得通信过程无法与推理过程进行端到端联合优化。

因此，现有方法未能将“如何有效沟通”本身作为一个可学习的组件来优化。

---

### 🚀 提出了什么新方法或新思路

作者提出 **DiffMAS**（Differentiable Multi-Agent System），一种将 **latent communication** 作为可训练组件的监督学习框架。

#### 核心思想：
- 将 **Key-Value (KV) caches** 作为 **continuous latent communication medium**，替代传统的文本通信。
- 构建一个 **共享的 KV trace**，由前序 agent 逐步追加 KV 块，最终由最后一个 agent 解码生成答案。
- 采用 **Supervised Fine-Tuning (SFT)** 对整个多 agent 推理轨迹进行端到端训练，仅更新 **LoRA 参数**，保持主干模型冻结。

#### 两阶段流程：
1. **Stage I: 构建 KV Trace**  
   Agent 1 到 K-1 依次执行，prefill 已有 KV cache 并追加新生成的 KV 段，不进行梯度更新。
2. **Stage II: 最终解码与训练**  
   最后一个 agent 基于完整的 KV trace 进行自回归解码，计算 **Cross-Entropy Loss**，并反向传播梯度以更新 LoRA 参数。

---

### 🔍 相比现有方法的优势

| 方法 | 缺陷 | DiffMAS 如何改进 |
|------|------|----------------|
| **Single-agent** | 缺乏分工协作能力 | 引入多 agent 协作机制 |
| **TextMAS** | 文本通信导致信息损失、无法端到端优化 | 使用连续 KV 通信，保留更多信息，支持梯度流动 |
| **LatentMAS (training-free)** | 虽然使用 KV cache，但无训练，表示未对齐，易导致不稳定解码 | 显式学习编码与解码策略，提升稳定性 |
| **C2C / StitchMAS** | 使用预训练融合模块或独立生成 KV | DiffMAS 共享上下文，构建连续 trace，实现全局协调 |

> ✅ **核心优势**：DiffMAS 实现了 **communication-as-a-learnable-component**，使系统能够学习“如何表达”和“如何理解”中间推理状态。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

涵盖三大类任务，评估泛化性和鲁棒性：

| 类别 | 数据集 | 描述 |
|------|--------|------|
| **数学与科学推理** | AIME2024, AIME2025, GPQA-Diamond | 多步符号推理、精确数值/分类答案 |
| **常识推理** | OpenBookQA | 基于小学科学知识的结构化推理 |
| **代码生成** | HumanEval+, MBPP+ | Python 程序功能正确性与泛化能力 |

---

### ⚙️ 实验设置和评估指标

#### 模型规模
- 主要使用 **Qwen3 系列**（4B, 8B, 14B）
- 同时测试 **Mistral3-8B**, **DeepSeek-R1-Distill-Qwen-32B**

#### 训练设置
- 使用 **LoRA** 进行参数高效微调（rank=8, α=16）
- 学习率：5e-5，AdamW + Cosine LR Schedule
- 冻结主干模型，仅训练 LoRA 适配器
- 小样本训练（如 210 样本用于数学，50 用于代码），验证低资源下的有效性

#### 评估指标
- **Accuracy (ACC)**：主要性能指标
- **Perplexity (PPL)**：衡量解码稳定性
- **Self-Consistency**：采样多次看结果一致性，反映推理稳定性
- **Token-level Entropy**：分析决策不确定性动态

---

### 🆚 基线方法对比

| 基线 | 描述 |
|------|------|
| **Single** | 单模型直接生成 |
| **TextMAS** | 多 agent 通过自然语言通信 |
| **LatentMAS** | 使用 KV cache 但无训练（training-free） |
| **C2C** | Cache-to-Cache，使用学习融合模块交换 KV |
| **StitchMAS** | 各 agent 独立生成 KV 后拼接（消融对照） |

所有方法使用相同 agent 角色（Planner → Critic → Refiner → Solver）和提示词，仅通信方式不同。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

| 模型 | 任务 | Single | TextMAS | LatentMAS | **DiffMAS** |
|------|------|--------|---------|-----------|-------------|
| Qwen3-4B | AIME24 | 43.3% | 46.7% (+3.4%) | 50.0% (+6.7%) | **63.3% (+20.0%)** |
| Qwen3-8B | AIME24 | 50.0% | 50.0% (+0.0%) | 56.7% (+6.7%) | **76.7% (+26.7%)** |
| Qwen3-8B | GPQA-Diamond | 39.9% | 43.4% (+3.5%) | 45.5% (+5.6%) | **60.1% (+20.2%)** |
| Qwen3-14B | HumanEval+ | 77.2% | 81.5% (+4.3%) | 86.8% (+9.6%) | **87.7% (+10.5%)** |
| DeepSeek-32B | HumanEval+ | 80.7% | 82.4% (+1.7%) | 83.3% (+2.6%) | **88.5% (+7.8%)** |

> 💡 **最高提升达 +26.7%**（Qwen3-8B on AIME24），且在多个尺度上均取得 SOTA。

---

### 🔍 与基线方法的对比结果

- **显著优于 Single 和 TextMAS**：尤其在数学任务上，DiffMAS 表现出更强的多步推理能力。
- **优于 LatentMAS**：尽管 LatentMAS 也使用 KV，但由于缺乏训练，常出现 **decoding instability** 和 **reasoning drift**。
- **远超 C2C**：C2C 在 OpenHermes 上训练，偏向指令跟随而非长程推理，导致在 AIME 等任务上表现差（甚至低于单模型）。
- **优于 StitchMAS**：说明简单拼接 KV 不够，需要 **连续共享 trace** 和联合优化。

---

### 🔧 消融实验结果

#### （1）DiffMAS vs. TextMAS + SFT（Table 4）
| 方法 | AIME24 | AIME25 | GPQA-Diamond |
|------|-------|--------|--------------|
| TextMAS + SFT | 76.7% | 50.0% | 53.5% |
| **DiffMAS** | **76.7%** | **56.7%** | **60.1%** |

> 结论：当训练/测试分布相近时（AIME24），两者相当；但在更难或分布偏移任务上（AIME25, GPQA），**DiffMAS 明显胜出**，说明其增益不仅来自任务学习，更来自**通信本身的优化**。

#### （2）通信步数影响（Table 3）
| 步数 | 0 | 10 | 40 | 100 | 200 |
|------|----|-----|-----|------|------|
| 准确率 | 50.0% | **76.7%** | 63.3% | 66.7% | 63.3% |

> 结论：少量通信即可带来巨大提升，但过长 trace 会引入噪声，说明 DiffMAS 学到了**紧凑高效的通信协议**。

#### （3）StitchMAS vs. DiffMAS（Table 5）
| 方法 | AIME24 | GPQA-Diamond |
|------|--------|---------------|
| StitchMAS | 60.0% | 48.4% |
| **DiffMAS** | **76.7%** | **60.1%** |

> 结论：即使都使用 SFT，**连续 trace 构建方式** 比独立生成再拼接更优，强调了**共享计算图的重要性**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Communication 可以且应该被学习**  
   将 inter-agent communication 视为可学习组件，是提升 MAS 性能的关键路径。

2. **Latent Communication + SFT > Training-Free Latent**  
   单纯使用 KV cache 不足以保证稳定，必须通过监督训练对齐编码与解码行为。

3. **DiffMAS 提升稳定性与准确性**  
   - 更低的 **Perplexity**（平均 1.24 vs. 1.31）
   - 更高的 **Self-Consistency**（更多问题获得 3~4 次一致正确）
   - 更平滑的 **Token-level Entropy 曲线**

4. **非覆盖式通信避免梯度衰减**  
   Proposition 3.1 证明：concatenative KV trace 避免了 overwriting communication 中的深度相关梯度衰减，使早期 agent 的贡献也能有效回传。

---

### ⚠️ 方法的局限性

1. **依赖固定 agent 流程**  
   当前框架假设 agent 顺序固定（如 Planner → Critic → Refiner → Solver），尚未支持动态拓扑或自组织架构。

2. **KV trace 随深度增长**  
   非覆盖通信导致 KV trace 线性增长，可能引入冗余或干扰，限制最大 agent 数量。

3. **仍需人工设计角色与提示**  
   agent 功能划分和 prompt 设计仍是手工完成，未实现完全自动化。

4. **仅适用于 decoder-only 架构**  
   依赖于 transformer 的 KV cache 机制，难以直接迁移到 encoder-decoder 或其他架构。

---

### 🔮 未来工作方向

1. **Dynamic DiffMAS**：支持动态 agent 调度与拓扑演化。
2. **Compressed Latent Communication**：引入 latent pruning 或 summarization 机制，控制 trace 长度。
3. **End-to-End Agent Design**：联合优化 agent 角色、prompt 与 communication 协议。
4. **跨模型 latent communication**：探索异构 LLM 间的可学习 latent 接口。
5. **强化学习扩展**：结合 RL 实现 communication policy 的自主进化。

---

## 总结

> **DiffMAS** 是首个将 **latent communication** 显式建模为可学习操作符，并通过 **SFT 实现端到端优化** 的多智能体语言系统框架。它突破了传统 text-based communication 的瓶颈，在数学、代码、常识等任务上实现了高达 **+26.7%** 的准确率提升，同时显著增强了推理的 **稳定性** 与 **一致性**。该工作为构建真正可优化的 **end-to-end multi-agent reasoning systems** 开辟了新方向。

</details>

---

### 12. [Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts](https://arxiv.org/abs/2604.19835)

**Authors**: Chaitanya Dwivedi, Binxuan Huang, Himanshu Gupta, Pratik Jayarao, Neeraj Varshney, Bing Yin  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.19835v1  

#### Abstract
Mixture-of-Experts (MoE) has become the dominant architecture for scaling large language models: frontier models routinely decouple total parameters from per-token computation through sparse expert routing. Scaling laws show that under fixed active computation, model quality scales predictably with ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **训练大规模 MoE 模型成本高昂**：尽管 Mixture-of-Experts (MoE) 架构通过稀疏路由实现了高容量模型的高效推理（decoupling total parameters from per-token computation），但其训练成本依然很高。原因在于：
  - 所有专家参数、梯度和优化器状态都需驻留在显存中，内存开销随总参数量增长。
  - 专家分布在多个设备上时，需要大量 `all-to-all` 通信，占训练时间的 45–50%。
- 因此，从零开始训练一个具有大量专家的大规模 MoE 模型在计算资源上非常昂贵。

### **提出了什么新方法或新思路**
提出 **Expert Upcycling（专家再利用）** ——一种在持续预训练（Continued Pre-training, CPT）过程中逐步扩展 MoE 容量的方法，具体步骤如下：

1. **阶段一**：先在一个较小的 E-expert MoE 模型上进行预训练。
2. **阶段二（Upcycling 操作）**：
   - **Expert Replication**：将已有专家复制多次（可非均匀复制），构建 mE-expert 模型；
   - **Router Extension**：扩展路由器权重，并对复制专家的路由偏置添加微小噪声以打破对称性。
3. **阶段三**：在扩展后的模型上继续预训练（CPT），让复制出的专家通过梯度多样性实现专业化（specialization）。

该方法的关键是 **保持 Top-K 路由不变**，因此每 token 的计算量（FLOPs）和激活参数数量不变，仅增加总参数量。

### **相比现有方法的优势**
| 对比维度 | Expert Upcycling | 其他方法 |
|--------|------------------|---------|
| **训练效率** | 显著降低 GPU 小时消耗（节省 ~32%） | 固定大小训练需全程运行大模型 |
| **初始化质量** | 复制已有专家提供“暖启动”（warm initialization），起始损失远低于随机初始化 | 随机初始化需重新学习表示 |
| **适用场景** | 可用于已发布的 MoE 检查点的增量升级（sunk-cost 场景下节省 ~67% 成本） | 不支持复用已有 MoE 检查点 |
| **架构兼容性** | 支持 Interleaved MoE 和 Full MoE 架构 | 如 SPARKLING [56] 会改变激活参数 |

此外，本文还引入了：
- **Utility-based Expert Selection**：基于梯度重要性的非均匀复制策略（如 `||g||²`），在 CPT 预算有限时，**gap closure 效果提升超过三倍**。
- **理论框架**：将质量差距分解为 **capacity gap** 和 **initialization gain** 两项，指导实践设计。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **小规模消融实验**：使用 DCLM 数据集。
- **主实验（7B→13B）**：使用自定义高质量数据混合体，强调指令遵循、逻辑推理和数学能力，避免与预训练数据泄露。

> 注：所有 CPT 阶段均使用独立于初始预训练的数据分割，确保无数据泄漏。

### **实验设置**
- **模型架构**：
  - 主要采用 **Interleaved MoE**（交替密集层与 MoE 层），减少 `all-to-all` 通信频率。
  - 同时验证了 **Full MoE** 架构（所有层均为 MoE）下的有效性。
- **参数规模**：
  - 主实验：从 ~7B 总参数（32 专家）扩展到 ~13B（64 专家），激活参数约 1B。
  - 消融实验：在 ~1B 总参数级别进行系统测试。
- **路由机制**：Top-K 路由（K=2 或 K=8），无共享专家。
- **关键技术组件**：
  - 使用 **Loss-free Load Balancing** [52] 确保每个复制专家都能获得梯度信号，防止“representation collapse”。

### **评估指标**
- **主任务性能**：
  - **Validation Loss**（越低越好）
  - **下游基准平均准确率**（11 项任务）：
    - MMLU, BBH CoT, GSM8K, IFEval, HellaSwag, ARC-Challenge/Easy, PIQA, OpenBookQA, SciQ, Social IQA
- **效率指标**：
  - **GPU Hours**（衡量训练成本）
  - **Gap Closure Efficiency**（归一化质量差距闭合程度）：
    $$
    \text{Efficiency} = \frac{L(\text{Fixed-E}) - L(\text{Upcycled})}{L(\text{Fixed-E}) - L(\text{Fixed-mE})}
    $$

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Fixed-32** | 32-expert 模型全程训练（下界） |
| **Fixed-64** | 64-expert 模型从头训练（上界/质量天花板） |
| **Upcycled (Ours)** | 32→64 专家扩展后继续训练（本文方法） |
| **Sparse Upcycling [25]** | Dense → MoE 的转换方法（作为跨范式对比） |
| **Uniform vs. Utility-based Duplication** | 均匀复制 vs. 基于梯度重要性选择性复制 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（7B→13B 实验）**

#### ✅ **Table 1: 50% 与 100% CPT 下的性能对比**

| Metric | Fixed-32 | Upcycled (Ours) | Fixed-64 |
|-------|----------|------------------|-----------|
| **Val. Loss ↓** | 1.301 | **1.263** | 1.267 |
| **Avg Acc ↑** | 52.9 | **56.4** | 56.7 |
| **GPU Hours** | — | **27,888** | **41,328** |

> 🔥 在 **100% CPT** 条件下，Upcycled 模型几乎完全追平 Fixed-64 的性能，同时节省 **~32% GPU 小时**。

#### 📈 **图示结果（Figure 2）**
- **左图**：Upcycled (32→64) 仅需 27,888 GPU 小时，比 Fixed-64（41,328）节省 32%，比 Fixed-32（21,168）多 32%。
- **中图**：验证损失上，Upcycled（1.305）显著优于 Fixed-32（1.339），接近 Fixed-64（1.308）。
- **右图**：在 HellaSwag、PIQA、OpenBookQA、Social IQA 上甚至超越 Fixed-64。

---

### **与其他方法的对比结果**

#### 🔁 **与 Sparse Upcycling [25] 的比较（Table 5）**
| Target K/E | Fixed-E | Fixed-mE | Ours (MoE→MoE) | Sparse Upc. (Dense→MoE) |
|------------|--------|----------|------------------|----------------------------|
| 3.13%      | 2.894  | 2.788    | **2.808**        | 3.049                      |

> ⚠️ 当目标激活比极低（如 3.13%）时，**Sparse Upcycling 完全失效**，而 Expert Upcycling 仍能有效逼近目标性能。

#### 🎯 **Utility-based vs. Uniform Duplication（Table 4）**
在 25% CPT 预算下：
- **Uniform**：Gap Closure Efficiency = 8.2%
- **||g||²-based**：Gap Closure Efficiency = **26.5%** ➕ **超三倍提升**

> 表明：**在 CPT 时间受限时，基于梯度的重要性采样至关重要**。

---

### **消融实验结果**

#### 🕰️ **训练预算分配（Table 3）**
- **CPT 预算 ≥ 50%** 是实现强 gap closure 的必要条件。
- **过渡时机（transition timing）**：
  - 过早（T/T < 0.1）会导致源模型未充分专业化，影响 warm initialization 质量；
  - 最佳范围：**T/T ∈ [0.12, 0.38]** 可实现近似完全 gap closure。

#### 🧪 **不同 MoE 架构与规模的泛化性（Table 2）**
| 模型规模 | Fixed-256 | Upcycled-512 | Fixed-512 | Efficiency (%) |
|---------|-----------|---------------|------------|----------------|
| 154M    | 3.564     | 3.519         | 3.516      | 93.8           |
| 1B      | 2.819     | 2.767         | 2.763      | 92.9           |

> 表明 Expert Upcycling 在 **154M 到 1B 参数范围**内均有效，适用于多种 MoE 架构。

#### 🛠️ **多样性初始化策略无效（Appendix D）**
尝试了 20 种启发式方法（噪声注入、正交化、SVD 扰动等），结果表明：
- 所有扰动策略均无法显著优于简单复制（copy-paste）；
- 更大的初始化损失会导致更差的最终性能（Spearman ρ ≈ 0.8）；
- 结论：**保持低初始损失比人为引入多样性更重要**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Expert Upcycling 是一种高效且有效的 MoE 容量扩展方式**：
   - 在不增加推理成本的前提下，实现更大 MoE 模型的质量收益。
   - 通过 warm initialization 显著缩短收敛路径。

2. ✅ **训练成本大幅降低**：
   - 从头训练相比，节省 **~32% GPU 小时**；
   - 若已有 MoE 检查点，则仅需增量训练，节省高达 **~67%**。

3. ✅ **Utility-based selection 在低预算下极具优势**：
   - 基于 `||g||²` 的专家选择策略最优，能显著加速 gap closure。

4. ✅ **理论框架具有指导意义**：
   - 质量差距可分解为：
     - **Capacity Gap**（受 CPT 长度控制）
     - **Initialization Gain**（受复制策略和源模型质量控制）
   - 为实际部署提供了清晰的设计原则。

5. ✅ **优于跨范式方法（如 Sparse Upcycling）**：
   - MoE→MoE 的容量跳跃远小于 Dense→MoE，更容易被 CPT 弥补。

---

### **方法的局限性**
- **依赖高质量源模型**：若原始 MoE 未充分训练，warm initialization 优势减弱。
- **仅支持整数倍扩展（m ≥ 2）**：目前未探索非整数倍或动态调整。
- **尚未应用于千亿级以上前沿模型**：在极端规模下可能出现 router collapse 或 load imbalance 加剧等问题。
- **暂未结合 MoE 压缩技术**（如 DeRS [21]）：扩展后模型参数量翻倍，可能带来部署挑战。

---

### **未来工作方向**
1. **迭代式 Expert Upcycling**：
   - 通过多次小步扩展（如 32→64→128→256），始终保持较小 capacity gap，进一步提升效率。

2. **组合其他增长轴**：
   - 与 **layer stacking** [9] 或 **width expansion** 结合，实现复合增益。

3. **探索更优的初始化与路由机制**：
   - 设计更精细的 router bias 初始化；
   - 引入 adaptive router（类似 Nexus [14]）支持渐进式集成。

4. **结合 MoE 压缩技术**：
   - 扩展后再应用 **expert pruning** 或 **DeRS-style delta tuning**，实现“先扩后压”的灵活调控。

5. **面向低资源场景的轻量化版本**：
   - 探索在边缘设备上的局部 upcycling 机制。

---

> 💡 **一句话总结**：  
> **Expert Upcycling 提供了一种“可持续升级”的 MoE 训练范式——不是抛弃旧模型重训，而是将其“再利用”为更大模型的种子，既省算力又保质量，有望成为未来 MoE 模型训练的标准流程。**

</details>

---

### 13. [On the Quantization Robustness of Diffusion Language Models in Coding Benchmarks](https://arxiv.org/abs/2604.20079)

**Authors**: Aarav Gupta, Gururaj Deshpande, Chandreyi Chakraborty  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.20079v1  

#### Abstract
Auto-regressive Large Language Models (LLMs) achieve strong performance on coding tasks, but incur high memory and inference costs. Diffusion-based language models (d-LLMs) offer bounded inference cost via iterative denoising, but their behavior under post-training quantization (PTQ) has been sparse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：On the Quantization Robustness of Diffusion Language Models in Coding Benchmarks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **auto-regressive Large Language Models (AR LLMs)** 在代码生成等任务上表现优异，但其自回归式的逐 token 生成方式导致**高内存占用和推理延迟**，尤其在长序列场景下更为明显。虽然 **post-training quantization (PTQ)** 被广泛用于压缩 AR LLMs 以降低部署成本，但在新兴的 **diffusion-based language models (d-LLMs)** 上的应用尚属空白。

本文首次系统地研究了 **d-LLMs 在 PTQ 下的鲁棒性表现**，特别是在低比特量化（如 2–4 bit）下的性能保持能力，并与同规模的 AR 模型进行公平比较。

### 提出的新方法/新思路
- **标准化对比框架**：构建了一个统一的评估流程，在相同架构、相似参数量（均为 1.7B）、相同量化工具链下，对 d-LLM（CoDA）与 AR LLM（Qwen3-1.7B）进行 PTQ 鲁棒性比较。
- **适配 HAWQ 到 d-LLMs**：将原本为 CNN 设计的 **Hessian Aware Quantization (HAWQ)** 方法成功迁移并优化至 d-LLM 场景，提出了一种适用于大模型的近似实现方案（基于有限差分 + 稀疏采样），实现了混合精度量化。
- **揭示范式差异带来的量化鲁棒性优势**：发现 d-LLMs 在低比特量化下表现出更强的稳定性，推测这可能源于其训练目标（去噪）和双向注意力机制带来的更强内部表示鲁棒性。

### 相比现有方法的优势
- **更优的低比特鲁棒性**：CoDA 在 2–4 bit 下性能下降显著小于 Qwen3，尤其在 3-bit 出现“灾难性崩溃”时仍能维持基本功能。
- **更好的 Pareto 权衡**：通过 HAWQ 实现的 mixed-precision 配置可在准确率、延迟、内存之间提供平滑且可控的权衡路径。
- **无需微调即可稳定量化**：仅使用 PTQ 即可获得良好效果，避免了 QAT 所需的额外训练开销。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 类型 | 数据集 |
|------|--------|
| **校准数据集（Calibration Dataset）** | WikiText [Merity et al., 2016] —— 用于 GPTQ 和 HAWQ 的敏感度分析 |
| **评估数据集（Evaluation Benchmarks）** | - **HumanEval** [Chen et al., 2021] <br> - **MBPP (Mostly Basic Programming Problems)** [Austin et al., 2021] <br> 包括 `Instruct` 和 `Plus` 版本 |

### 实验设置
- **模型选择**：
  - **d-LLM**: CoDA 1.7B（基于 Qwen 架构改造的 diffusion 编码模型）
  - **AR LLM**: Qwen3-1.7B（作为对照组）
- **量化方法**：
  - **GPTQ**：weight-only PTQ，测试 bitwidth = {2, 3, 4, 8, 16}
  - **HAWQ**：mixed-precision PTQ，按模块敏感度分配 16/8/4-bit
- **硬件平台**：NVIDIA L40S / H100 GPU
- **量化配置**：
  - Group size = 128
  - 使用 Marlin kernel 加速 4/8-bit 推理
  - HAWQ 中采用稀疏采样（p=0.1）和 5 次幂迭代加速计算

### 评估指标
| 指标类别 | 具体指标 |
|---------|----------|
| **准确性** | Pass@1（HumanEval / MBPP） |
| **效率** | 平均单步延迟（ms）：<br> - CoDA：单个 denoising step<br> - Qwen3：单个 token 生成 |
| **资源消耗** | 内存占用（隐含于 bitwidth 控制） |
| **综合权衡** | Accuracy-Latency-Memory 的 Pareto frontier 表现 |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Full Precision Baseline (16-bit)** | 原始未量化模型 |
| **GPTQ @ 2/3/4/8-bit** | 统一精度量化，衡量极端压缩下的鲁棒性 |
| **HAWQ Mixed-Precision** | 分层动态分配 bitwidth，探索细粒度控制能力 |
| **Round-to-Nearest (RTN)** | 作为最简单的量化基线（文中虽未直接列出 RTN 结果，但 GPTQ 明确优于它） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Figure 2）

#### ✅ GPTQ 量化结果（Pass@1）

| Bitwidth | HumanEval+ (CoDA/Qwen3) | MBPP+ (CoDA/Qwen3) |
|----------|--------------------------|--------------------|
| 16-bit (baseline) | 0.439 / 0.628 | 0.619 / 0.664 |
| 8-bit | 0.421 / 0.616 | 0.632 / 0.656 |
| 4-bit | **0.421 / 0.409** | **0.574 / 0.503** |
| 3-bit | **0.292 / 0.000** | **0.479 / 0.000** |
| 2-bit | 0.000 / 0.000 | 0.000 / 0.000 |

> 🔍 观察：  
> - CoDA 在 **4-bit 仅损失 ~1.8–5 pts**，而 Qwen3 损失高达 **12–23.2 pts**
> - Qwen3 在 **3-bit 完全崩溃（0.000）**，而 CoDA 仍有可观测输出（~0.3）

#### ⏱️ 推理延迟（Figure 1）
| Bitwidth | CoDA Latency (ms) | Qwen3 Latency (ms) |
|----------|--------------------|---------------------|
| 16-bit | 28.329 ± 0.305 | **26.843 ± 0.305** |
| 8-bit | ~28.3 | ~32.0 |
| 4-bit | ~28.3 | ~32.0 |
| 3-bit | >30 | >35 |
| 2-bit | >30 | >35 |

> 🔍 观察：
> - 基线状态下 Qwen3 更快（得益于高度优化的 causal attention）
> - 一旦启用 GPTQ（尤其是 4/8-bit 使用 Marlin kernel），**CoDA 反超 Qwen3 达 25–40%**

### 与基线方法的对比结果
| 对比维度 | 结果 |
|--------|------|
| **量化鲁棒性** | CoDA 在 2–4 bit 下显著优于 Qwen3，尤其在 3-bit 不发生完全崩溃 |
| **精度损失比例** | 从 16-bit → 4-bit，CoDA 平均性能下降约 **8%**，Qwen3 下降达 **~40%** |
| **延迟竞争力** | 尽管 baseline 略慢，但量化后 CoDA 因更少的框架级开销（如 position handling、output projection）反而更快 |
| **混合精度灵活性** | HAWQ 成功构建了平滑的 Pareto frontier，例如 50/50 的 16/8-bit 或 8/4-bit 配置几乎无损性能 |

### 消融实验结果（Implicit Ablation）
- **不同 bitwidth 影响**：验证了低比特（<4）对 AR 模型破坏性强，而 d-LLM 更具容忍性。
- **HAWQ 敏感度排序有效性**：高敏感层集中在早期 decoder 层和 attention 输出投影，支持了分层量化策略合理性。
- **多精度组合挑战**：尝试三精度混合（16/8/4）时出现模型崩溃，表明当前 PTQ 缺乏恢复机制，需后续引入微调。

---

## 4. 关键结论和发现

### 主要发现
1. **d-LLMs 具有更强的量化鲁棒性**：  
   在相同 PTQ 设置下，CoDA 在 2–4 bit 下的性能退化远小于 Qwen3，说明 **diffusion 范式本身可能具备更高的参数扰动容忍度**。

2. **系统级开销成为瓶颈转移的关键因素**：  
   传统观点认为 AR 模型因单 token 高效而占优，但在量化后，**CoDA 的完整序列处理减少了每步的框架开销累积**，使其在低算力环境下更具优势。

3. **mixed-precision 是可行方向**：  
   HAWQ 可有效指导 d-LLM 的混合精度设计，在保持高准确率的同时灵活调节内存与速度，形成良好的 trade-off 曲线。

4. **训练数据差异可能是干扰因素**：  
   Qwen3 使用了 36T tokens 进行预训练，远超 CoDA 的 180B，可能导致其“过训练”，从而在 PTQ 下更易退化（参考 Kumar et al., 2024 的 scaling law 发现）。

### 方法的局限性
- **非完全公平比较**：两模型训练数据、超参不一致，无法完全归因于“generation paradigm”差异。
- **HAWQ 实现为近似版本**：由于 PyTorch 与 FlashAttention 兼容问题，采用有限差分和稀疏采样，牺牲了一定精度。
- **缺乏 QAT 支持**：目前仅验证 PTQ，未探索 quantization-aware training 是否可进一步提升性能。
- **三精度及以上混合失败**：当前方法难以稳定支持复杂混合策略，提示需要 post-quantization fine-tuning。

### 未来工作方向
1. **从零训练对称模型**：在同一数据、架构、tokenizer 下分别训练 diffusion 与 AR 模型，彻底隔离 generation paradigm 的影响。
2. **领域专用校准数据**：改用 OpenCoder 等编程导向数据集进行 calibration，提升量化在代码任务上的针对性。
3. **深入探索 HAWQ 多精度配置**：自动化搜索最优 bit 分配策略，结合 sensitivity 与 layer type 进行规则引导。
4. **引入 QAT 或轻量微调**：尝试在量化后加入少量监督微调（SFT）或 loss-driven recovery，缓解低比特崩溃问题。
5. **扩展到更多 d-LLM 架构**：验证结论是否适用于其他 diffusion LM 如 Seed-Diffusion、Mercury 等。

---

> 📌 **总结一句话**：  
> 该论文首次证明，**diffusion-based LLMs 在 post-training quantization 下展现出比 auto-regressive 模型更强的鲁棒性**，尤其是在 2–4 bit 极端压缩场景中仍能维持可用性能，结合 mixed-precision 策略可实现更高效的部署方案，为未来边缘端代码生成提供了新思路。

</details>

---

### 14. [Trust but Verify: Introducing DAVinCI -- A Framework for Dual Attribution and Verification in Claim Inference for Language Models](https://arxiv.org/abs/2604.21193)

**Authors**: Vipula Rawte, Ryan Rossi, Franck Dernoncourt, Nedim Lipka  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.21193v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable fluency and versatility across a wide range of NLP tasks, yet they remain prone to factual inaccuracies and hallucinations. This limitation poses significant risks in high-stakes domains such as healthcare, law, and scientific communication, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Trust but Verify: Introducing DAVinCI —— 核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）虽然在自然语言生成任务中表现出色，但普遍存在**事实错误**（factual inaccuracies）和**幻觉**（hallucinations）问题。这在医疗、法律、科学传播等高风险领域尤为危险。现有的验证系统通常将**归因**（attribution）与**验证**（verification）视为独立模块，缺乏协同机制。

### 🚀 提出的新方法：DAVinCI 框架
本文提出 **DAVinCI**（Dual Attribution and Verification in Claim Inference），一个集成化的双阶段框架，旨在提升 LLM 输出的事实可靠性与可解释性。

#### 双阶段流程：
1. **Attribution（归因）**  
   - 将生成的声明（claim）链接到内部模型组件或外部证据源。
   - 支持两种模式：
     - **Full Evidence Attribution**：使用完整证据文本。
     - **Span-Based Attribution**：通过 QA 模型提取关键证据片段。

2. **Verification（验证）**  
   - 使用基于 **entailment-based classifier** 对 claim-evidence 对进行分类，输出 `Supported` / `Refuted` / `Not Enough Information (NEI)`。
   - 引入**置信度重校准**（confidence recalibration）：当置信分数低于阈值 $ T $（默认 0.6）时，强制降级为 NEI，以减少过度自信的误判。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | DAVinCI |
|------|--------|--------|
| 架构 | 分离式（retrieval + generation + optional verification） | 统一管道，**归因驱动验证** |
| 归因粒度 | 多为粗粒度检索 | 支持 full-passage 和 fine-grained span 归因 |
| 可靠性控制 | 缺乏显式信任管理 | 显式引入 confidence recalibration 机制 |
| 可审计性 | 输出不可追溯 | 提供透明、可溯源的推理链 |

> ✅ **核心创新**：首次将 internal/external attribution 与 entailment-based verification 在模块化架构中统一，并支持灵活的信任校准。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **FEVER** | 来自维基百科的事实核查数据集，包含 claim 与标注证据句子，标签为 `Entailment`, `Contradiction`, `Neutral` |
| **CLIMATE-FEVER** | 聚焦气候变化领域的事实核查数据集，更具领域挑战性，标签为 `Supports`, `Refutes`, `Not Enough Info` |

> ⚠️ 两个数据集均提供黄金标准证据（gold-standard evidence），便于评估 attribution 质量。

### 🧪 实验设置
- **硬件环境**：Apple MacBook M4 芯片，32GB RAM
- **模型库**：Hugging Face Transformers
- **License 合规性**：所有模型与数据集采用 MIT 许可证，确保可复现性

#### 评估指标
- Accuracy
- Precision（macro & weighted）
- Recall（macro & weighted）
- F1-Score（macro & weighted）

#### 基线方法对比
| 方法 | 描述 |
|-----|------|
| **Baseline (Verifier-only)** | 仅使用 full evidence 输入给 NLI 模型，无 attribution 控制或 recalibration |
| **DAVinCI-Recalibrated** | 完整 DAVinCI 流程：span-based attribution + verification + recalibration ($T=0.6$) |

#### 对比的 NLI 模型
1. `microsoft/deberta-large-mnli`
2. `FacebookAI/roberta-large-mnli`
3. `facebook/bart-large-mnli`
4. `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`

---

## 3. 主要实验结果和性能指标

### 📈 总体性能提升（vs. Baseline）
DAVinCI 在多个模型上显著优于 baseline，在 **accuracy、precision、recall、F1-score 上平均提升 5–20%**。

#### 表格摘要（来自 Table 2 & 3）：

| Model | Dataset | Acc (Base) → Acc (DAVinCI) | Δ Acc | Macro F1 (Base → DA) | Δ F1 |
|-------|--------|-----------------------------|-------|------------------------|------|
| deberta-large-mnli | FEVER | 0.42 → 0.48 | +6% | 0.36 → 0.41 | +5% |
| roberta-large-mnli | FEVER | 0.36 → 0.44 | +8% | 0.30 → 0.38 | +8% |
| bart-large-mnli | FEVER | 0.42 → 0.43 | +1% | 0.36 → 0.37 | +1% |
| roberta-large-snli | FEVER | 0.38 → 0.42 | +4% | 0.34 → 0.40 | +6% |
| **Best performer** | **CLIMATE-FEVER** | **0.65 → 0.66** | **+1%** | **~0.57 → ~0.63** | **+6%** |

> 💡 即使 baseline 表现较差的模型（如 roberta-large-mnli），也通过 DAVinCI 得到明显增强，说明该框架具有**普适性增益能力**。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）Full Evidence vs. Span-Based Attribution（Tables 4 & 5）
| 设置 | 性能表现 |
|------|---------|
| **Full Evidence** | 显著优于 span-based 方法（+9% ~ +18%） |
| **Span-Based** | 容易丢失上下文信息，导致 recall 和 F1 下降明显 |

> ✅ 发现：**完整的证据上下文对验证至关重要**，尤其在复杂推理场景下。

#### （2）不同置信度阈值的影响（Threshold Tuning, Tables 6 & 7）
测试了 $ T = 0.7, 0.8, 0.9 $

| 阈值 $T$ | 特点 |
|----------|------|
| **0.7** | 最佳平衡点：accuracy 最高，precision 保持良好，recall 下降有限 |
| **0.8 / 0.9** | 更保守策略，precision 提升但 recall 明显下降，F1 整体降低 |

> ✅ 结论：**T=0.7 是最优选择**，可在减少 false positive 的同时维持较高覆盖率。

#### （3）关键组件影响分析
- **Evidence Quality**：full evidence > span-based（context 完整性决定成败）
- **Recalibration Threshold**：直接影响 precision-recall trade-off
- **Retrieval Quality**：高质量 retrieval 是前提条件

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **归因质量直接影响验证效果**：full evidence attribution 显著优于 span-based 方法，表明**语义完整性是准确验证的基础**。
2. **DAVinCI 显著提升各项指标**：相比纯 verification baseline，带来 **5–20% 的综合性能提升**，尤其改善 precision 与 F1。
3. **置信度重校准有效抑制过自信预测**：通过简单阈值机制即可实现更可靠的决策边界，符合人类 fact-checker 的审慎行为。
4. **模块化设计支持灵活扩展**：可集成不同 retriever、verifier 和 calibration 策略，适用于多种下游任务。

### ⚠️ 局限性
1. **依赖高质量外部证据**：在开放域（open-domain）中若 retrieval 不准确，整体性能会下降。
2. **静态验证器限制**：当前 verifier 基于单跳推理（single-hop），难以处理多跳（multi-hop）或复杂逻辑推理。
3. **未建模 internal attribution**：尚未追踪 claim 到训练数据或 prompt 的内部来源路径。
4. **语言局限**：目前仅在英文数据集（FEVER, CLIMATE-FEVER）上验证，未覆盖多语言或低资源场景。
5. **手动调参**：confidence threshold 需人工设定，泛化能力受限。

### 🔮 未来工作方向
1. **整合 dense retriever（如 DPR, E5）** 提升 open-domain retrieval 能力。
2. **引入 multi-hop reasoning 模块** 应对复杂推理任务。
3. **探索 internal attribution 技术** 如 prompt tracing 或 activation clustering，增强模型自我解释能力。
4. **应用于生成任务**：让 LLM 在生成过程中同步输出 source-aware 文本。
5. **扩展至 multilingual 和 low-resource setting**。
6. **开发 adaptive calibration 策略**：用学习方式替代固定阈值，提高跨任务鲁棒性。
7. **开展 human-in-the-loop 评测**：衡量真实用户对 DAVinCI 输出的信任程度与可用性。

---

## 📌 总结
DAVinCI 是一项推动 **可信 AI（trustworthy AI）** 发展的重要工作。它不仅提升了 LLM 输出的事实准确性，还提供了**可审计、可解释、可校准**的推理链条。其“**先归因、再验证、后校准**”的三段式思想，为构建下一代负责任的语言系统提供了清晰的技术路径。

> 🔗 开源地址：[https://github.com/vr25/davinci](https://github.com/vr25/davinci)

</details>

---

### 15. [Temporally Extended Mixture-of-Experts Models](https://arxiv.org/abs/2604.20156)

**Authors**: Zeyu Shen, Peter Henderson  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.20156v1  

#### Abstract
Mixture-of-Experts models, now popular for scaling capacity at fixed inference speed, switch experts at nearly every token. Once a model outgrows available GPU memory, this churn can render optimizations like offloading and pre-fetching ineffective. We make the case that the options framework in rei...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Temporally Extended Mixture-of-Experts Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 **Mixture-of-Experts (MoE)** 模型在推理时通常对每个 token 都重新选择激活的专家（expert），导致“**频繁切换**”（frequent switching）。当模型规模超出 GPU 内存容量时，这种高频率的专家切换会严重破坏 **offloading、prefetching** 等内存优化策略的有效性，造成显著的延迟和吞吐量下降。

尽管已有研究通过 **expert pruning** 或 **caching/prefetching** 来缓解内存压力，但这些方法大多忽略了一个关键点：**动态加载专家本质上是一个具有时间成本的决策过程**。

### 提出了什么新方法或新思路
本文提出 **“时序扩展的 MoE”**（Temporally Extended Mixture-of-Experts, TEMoE），其核心思想是将 MoE 中的专家选择建模为 **强化学习中的“选项”**（options）。

- **引入“控制器”**（controller）：在每一层 MoE 上添加一个轻量级控制器，决定是否**保持当前专家集合**或**切换到新的集合**。
- **基于 Option-Critic 框架**：采用带有 **deliberation cost**（决策成本）的 Option-Critic 架构进行训练，显式地将“切换专家”的代价纳入优化目标。
- **专家掩码作为选项**（expert mask as option）：将允许被路由的专家子集视为一个持续多个 token 的“选项”，只有当预期收益超过切换成本时才触发切换。

### 相比现有方法的优势
| 维度 | 传统 MoE / 基线方法 | 本文方法（TEMoE） |
|------|---------------------|------------------|
| **切换频率** | 几乎每步都换（>50%） | 可降至 <5%，甚至 <1% |
| **内存效率** | 必须预载所有专家或依赖预测缓存 | 只需保留少量活跃专家，支持高效 offloading |
| **灵活性** | 固定专家池或静态剪枝 | 支持动态扩展专家池，便于持续学习（continual learning） |
| **训练方式** | 多为后处理剪枝或启发式缓存 | 通过端到端学习实现时间连续性 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练数据**：`Nemotron Post-Training Dataset v2`，包含 10 类任务（chat, code, math, STEM, 多语言等），共 128 prompts × 2048 tokens。
- **评估数据集**：
  - **MATH**：数学推理题（200 道）
  - **MMLU**：大规模多任务语言理解（200 道）
  - **MMMLU**：多语言 MMLU（200 道）

### 实验设置和评估指标
- **基础模型**：`gpt-oss-20b`（24 层，每层 32 专家，top-4 路由）
- **硬件平台**：4× NVIDIA H200（140GB 显存）
- **训练配置**：
  - 使用 LoRA（rank=16）微调专家和注意力参数
  - 控制器独立训练，学习率 `1e-4`
  - 自蒸馏奖励（self-distillation reward）：基于 teacher model（原始模型）与 student model 输出的 reverse KL
  - 引入 teacher mixing（比例 τ=0.2）防止退化
- **关键超参**：deliberation cost `η ∈ {0.02, 0.03, 0.04}`，专家预算 `k ∈ {8, 16}`

### 评估指标
- **Switch Rate (%)**：token 级别上专家掩码发生变化的比例
- **Accuracy (%)**：在 MATH/MMLU/MMMLU 上的任务准确率
- **Perplexity & Repetition Rate**：用于监控生成质量稳定性

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Frequency-based** | 保留校准集上使用最频繁的 k 个专家 |
| **Reconstruction Loss Minimization** | 选择能最好重构原 MoE 输出的 k 个专家子集 |
| **Random Selection** | 每步随机选 k 个专家 |
| **Wanda (Structured)** | 结构化权重剪枝方法 |
| **Base Model** | 原始未修改的 gpt-oss-20b |

> ⚠️ 注：不与 MoE-infinity 等缓存系统比较，因其优化的是运行时调度而非路由策略本身。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

#### 当 `k=16` 时（Table 2）：
| 方法 | MATH Acc (%) | MMLU Acc (%) | MMMLU Acc (%) | Switch Rate (%) |
|------|---------------|---------------|----------------|------------------|
| Base Model | 71.5 | 79.5 | 67.5 | >50 (implicit) |
| Frequency | 53.5 | 55.5 | 42.0 | — |
| Reconstruction | 51.5 | 35.0 | 48.0 | — |
| Ours (`η=0.02`) | **64.0** | **72.5** | **59.5** | **4.2** |
| Ours (`η=0.04`) | 55.0 | 63.0 | 49.5 | **1.2** |

#### 当 `k=8` 时（Table 3）：
| 方法 | MATH Acc (%) | MMLU Acc (%) | MMMLU Acc (%) | Switch Rate (%) |
|------|---------------|---------------|----------------|------------------|
| Base Model | 71.5 | 79.5 | 67.5 | >50 |
| Frequency | 11.5 | 12.5 | 8.5 | — |
| Reconstruction | 7.5 | 2.5 | 1.0 | — |
| Ours (`η=0.02`) | **27.5** | **48.5** | **39.0** | **7.4** |
| Ours (`η=0.04`) | 15.5 | 38.0 | 22.5 | **5.0** |

> ✅ **关键发现**：
> - 在 `k=16` 下，本方法可将 switch rate 从 >50% 降至 **<5%**，同时保留高达 **90% 的原始模型准确性**。
> - 准确率与 `η` 和 `k` 呈现明显 trade-off：更高的 `η` 或更小的 `k` 导致更低的 switch rate，但也带来一定性能损失。
> - 所有基线在低 `k` 设置下性能急剧下降，而本文方法仍能维持相对合理的输出质量。

### 消融实验与分析（隐含于训练曲线与案例）
- **训练动态显示稳定收敛**：reward 上升、switch rate 下降、perplexity 降低（见 Fig. 5）
- **避免重复崩溃**（catastrophic repetition）：repetition rate 始终低于 0.2（见 Fig. A3）
- **案例展示**：在 MATH 任务中，本文方法能维持连贯推理，而基线方法（如 reconstruction）迅速退化为无意义文本或无限重复（见 Appendix A5 示例）

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **MoE 路由天然适合建模为 s-MDP 与 options 框架**：将专家掩码视为 temporally extended action，能有效捕捉时间连续性。
2. ✅ **极低切换率可行**：通过引入 deliberation cost，可在仅激活 8–16 个专家的情况下将 switch rate 降至 **1–5%**，远优于传统 MoE 的几乎每步切换。
3. ✅ **轻量训练即可转换现有模型**：无需从头预训练，仅用少量 LoRA 参数和自蒸馏即可将已有 MoE 模型转化为 TEMoE。
4. ✅ **开启多项系统优化机会**：
   - **内存高效服务**（memory-efficient serving）：只需驻留少量专家，大幅减少 VRAM 占用（如节省 37%-55%）。
   - **时序分块训练**（temporal chunking for training）：支持按 chunk 加载专家，降低峰值内存。
   - **可扩展的持续学习**（expandable continual learning）：新增专家不影响单 token 计算开销。

### 方法的局限性
| 局限性 | 说明 |
|-------|------|
| **Per-layer 独立控制** | 各层控制器独立决策，可能导致跨层切换不同步，不利于统一 offloading；理想应为 joint option across layers。 |
| **Deliberation Cost 是超参** | `η` 尚未与真实硬件延迟对齐，未来需根据实际设备 calibrate。 |
| **评估范围有限** | 仅测试了 MATH/MMLU/MMMLU，缺少代码生成、长对话等任务。 |
| **未完全解耦 temporal extension 与 self-distillation 影响** | 性能提升可能部分来自参数微调而非路由机制本身。 |

### 未来工作方向
1. **将 temporal extension 融入预训练**：设计原生支持时间连续性的 MoE 架构，而非依赖后训练调整。
2. **构建跨层联合控制器**：定义全局 option，使所有层同步切换专家，简化内存管理。
3. **硬件感知的 deliberation cost 建模**：将 `η` 与实际 PCIe 传输延迟、CPU-GPU 带宽绑定，实现真正的 cost-aware routing。
4. **探索更复杂 option 结构**：结合自然语言中的主题、论证链等高层语义结构指导专家调度。
5. **集成至完整推理系统**：与 MoE-infinity 等 prefetching 系统结合，验证端到端延迟与吞吐增益。

---

> 📌 **总结一句话**：  
> 本文首次将 **options 框架** 引入 MoE 路由，提出 **时序扩展 MoE**（TEMoE），通过一个带 **deliberation cost** 的控制器，实现了专家切换率从 >50% 到 <5% 的突破性压缩，同时保留了大部分原始性能，为大模型的 **内存高效部署** 与 **持续学习** 提供了一条原则性强、可落地的新路径。

</details>

---

### 16. [Beyond N-gram: Data-Aware X-GRAM Extraction for Efficient Embedding Parameter Scaling](https://arxiv.org/abs/2604.21724)

**Authors**: Yilong Chen, Yanxi Xie, Zitian Gao, He Xin, Yihao Xiao, Renbiao Liu, Haoming Luo, Yifan Luo, Zhengmao Ye, Tingwen Liu, Xin Zhao, Ran Tao, Bryan Dai  
**Category**: cs.CL  
**Published**: 2026-04-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.21724v1  

#### Abstract
Large token-indexed lookup tables provide a compute-decoupled scaling path, but their practical gains are often limited by poor parameter efficiency and rapid memory growth. We attribute these limitations to Zipfian under-training of the long tail, heterogeneous demand across layers, and "slot colla...

---

### 17. [Continuous Semantic Caching for Low-Cost LLM Serving](https://arxiv.org/abs/2604.20021)

**Authors**: Baran Atalar, Xutong Liu, Jinhang Zuo, Siwei Wang, Wei Chen, Carlee Joe-Wong  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20021v1  

#### Abstract
As Large Language Models (LLMs) become increasingly popular, caching responses so that they can be reused by users with semantically similar queries has become a vital strategy for reducing inference costs and latency. Existing caching frameworks have proposed to decide which query responses to cach...

---

### 18. [Robustness of Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable Distribution Grids](https://arxiv.org/abs/2604.20403)

**Authors**: Burak Karabulut, Carlo Manna, Chris Develder  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20403v1  

#### Abstract
Fault location in distribution grids is critical for reliability and minimizing outage durations. Yet, it remains challenging due to partial observability, given sparse measurement infrastructure. Recent works show promising results by combining Recurrent Neural Networks (RNNs) and Graph Neural Netw...

---

### 19. [Efficient Test-Time Inference via Deterministic Exploration of Truncated Decoding Trees](https://arxiv.org/abs/2604.20500)

**Authors**: Xueyan Li, Johannes Zenn, Ekaterina Fadeeva, Guinan Su, Mrinmaya Sachan, Jonas Geiping  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20500v1  

#### Abstract
Self-consistency boosts inference-time performance by sampling multiple reasoning traces in parallel and voting. However, in constrained domains like math and code, this strategy is compute-inefficient because it samples with replacement, repeatedly revisiting the same high-probability prefixes and ...

---

### 20. [Stream-CQSA: Avoiding Out-of-Memory in Attention Computation via Flexible Workload Scheduling](https://arxiv.org/abs/2604.20819)

**Authors**: Yiming Bian, Joshua M. Akey  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20819v1  

#### Abstract
The scalability of long-context large language models is fundamentally limited by the quadratic memory cost of exact self-attention, which often leads to out-of-memory (OOM) failures on modern hardware. Existing methods improve memory efficiency to near-linear complexity, while assuming that the ful...

---

### 21. [Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows](https://arxiv.org/abs/2604.21816)

**Authors**: Anuj Sadani, Deepak Kumar  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.21816v1  

#### Abstract
The Model Context Protocol (MCP) has become a common interface for connecting large language model (LLM) agents to external tools, but its reliance on stateless, eager schema injection imposes a hidden per-turn overhead the MCP Tax or Tools Tax that practitioner reports place between roughly 10k and...

---

### 22. [TRACES: Tagging Reasoning Steps for Adaptive Cost-Efficient Early-Stopping](https://arxiv.org/abs/2604.21057)

**Authors**: Yannis Belkhiter, Seshu Tirupathi, Giulio Zizzo, John D. Kelleher  
**Category**: cs.CL  
**Published**: 2026-04-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.21057v1  

#### Abstract
The field of Language Reasoning Models (LRMs) has been very active over the past few years with advances in training and inference techniques enabling LRMs to reason longer, and more accurately. However, a growing body of studies show that LRMs are still inefficient, over-generating verification and...

---

### 23. [Are LLM Uncertainty and Correctness Encoded by the Same Features? A Functional Dissociation via Sparse Autoencoders](https://arxiv.org/abs/2604.19974)

**Authors**: Het Patel, Tiejin Chen, Hua Wei, Evangelos E. Papalexakis, Jia Chen  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.19974v1  

#### Abstract
Large language models can be uncertain yet correct, or confident yet wrong, raising the question of whether their output-level uncertainty and their actual correctness are driven by the same internal mechanisms or by distinct feature populations. We introduce a 2x2 framework that partitions model pr...

---

### 24. [Learning to Solve the Quadratic Assignment Problem with Warm-Started MCMC Finetuning](https://arxiv.org/abs/2604.20109)

**Authors**: Yicheng Pan, Ruisong Zhou, Haijun Zou, Tianyou Li, Zaiwen Wen  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.20109v1  

#### Abstract
The quadratic assignment problem (QAP) is a fundamental NP-hard task that poses significant challenges for both traditional heuristics and modern learning-based solvers. Existing QAP solvers still struggle to achieve consistently competitive performance across structurally diverse real-world instanc...

---

### 25. [A Delta-Aware Orchestration Framework for Scalable Multi-Agent Edge Computing](https://arxiv.org/abs/2604.20129)

**Authors**: Samaresh Kumar Singh, Joyjit Roy  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.20129v1  

#### Abstract
The Synergistic Collapse occurs when scaling beyond 100 agents causes superlinear performance degradation that individual optimizations cannot prevent. We observe this collapse with 150 cameras in Smart City deployment using MADDPG, where Deadline Satisfaction drops from 78% to 34%, producing approx...

---

### 26. [Explicit Dropout: Deterministic Regularization for Transformer Architectures](https://arxiv.org/abs/2604.20505)

**Authors**: Vidhi Agrawal, Illia Oleksiienko, Alexandros Iosifidis  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.20505v1  

#### Abstract
Dropout is a widely used regularization technique in deep learning, but its effects are typically realized through stochastic masking rather than explicit optimization objectives. We propose a deterministic formulation that expresses dropout as an additive regularizer directly incorporated into the ...

---

### 27. [Language as a Latent Variable for Reasoning Optimization](https://arxiv.org/abs/2604.21593)

**Authors**: Linjuan Wu, Haoran Wei, Jialong Tang, Shuang Luo, Baosong Yang, Yongliang Shen, Weiming Lu  
**Category**: cs.CL  
**Published**: 2026-04-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.21593v1  

#### Abstract
As LLMs reduce English-centric bias, a surprising trend emerges: non-English responses sometimes outperform English on reasoning tasks. We hypothesize that language functions as a latent variable that structurally modulates the model's internal inference pathways, rather than merely serving as an ou...

---

### 28. [Transparent Screening for LLM Inference and Training Impacts](https://arxiv.org/abs/2604.19757)

**Authors**: Arnault Pachot, Thierry Petit  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19757v1  

#### Abstract
This paper presents a transparent screening framework for estimating inference and training impacts of current large language models under limited observability. The framework converts natural-language application descriptions into bounded environmental estimates and supports a comparative online ob...

---

### 29. [Fast Amortized Fitting of Scientific Signals Across Time and Ensembles via Transferable Neural Fields](https://arxiv.org/abs/2604.19979)

**Authors**: Sophia Zorek, Kushal Vyas, Yuhao Liu, David Lenz, Tom Peterka, Guha Balakrishnan  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19979v1  

#### Abstract
Neural fields, also known as implicit neural representations (INRs), offer a powerful framework for modeling continuous geometry, but their effectiveness in high-dimensional scientific settings is limited by slow convergence and scaling challenges. In this study, we extend INR models to handle spati...

---

### 30. [Meta Additive Model: Interpretable Sparse Learning With Auto Weighting](https://arxiv.org/abs/2604.20111)

**Authors**: Xuelin Zhang, Xinyue Liu, Lingjuan Wu, Hong Chen  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.20111v1  

#### Abstract
Sparse additive models have attracted much attention in high-dimensional data analysis due to their flexible representation and strong interpretability. However, most existing models are limited to single-level learning under the mean-squared error criterion, whose empirical performance can degrade ...

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
