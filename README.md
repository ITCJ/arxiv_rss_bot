# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-24 07:24:39 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Distributed Generative Inference of LLM at Internet Scales with Multi-Dimensional Communication Optimization](https://arxiv.org/abs/2604.21072)

**Authors**: Jiu Chen, Shuangyan Yang, Xu Xiong, Hexiao Duan, Xinran Zhang, Jie Ren, Dong Li  
**Category**: cs.DC  
**Published**: 2026-04-24  
**Score**: 16.5  
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
本文针对**去中心化大语言模型（LLM）推理在互联网规模下的通信瓶颈问题**。由于分布式节点间网络带宽低（通常为 20–500 Mbps），跨节点传输激活张量（activations）成为性能的主要瓶颈，远超本地计算时间。

传统集中式推理依赖高性能数据中心内网（如 InfiniBand），而**去中心化环境无法使用 GPUDirect RDMA 和低延迟协议**，导致通信开销极高。因此，如何在异构、低带宽的互联网环境中优化 LLM 推理吞吐量，是一个关键挑战。

---

### 提出的新方法与创新思路
作者提出 **BloomBee** ——一个面向互联网规模的分布式 LLM 推理框架，其核心是**多维度通信优化协同设计**，主要包括以下技术组合：

#### （1）**Layer Assignment + Tensor Offloading + Micro-batching 联合优化**
- **Layer Assignment**：将 Transformer 层分配到不同地理分布的 GPU 上。
- **Tensor Offloading**：将 KV Cache 部分卸载至 CPU 内存，释放 GPU 显存以容纳更多层，从而减少 pipeline 中的 hop 数。
- **Micro-batching**：将输入 batch 拆分为 micro-batches，实现 computation 与 communication 的重叠。
- **联合建模为动态规划问题**：通过 dynamic programming 在 GPU 显存约束下求解最优的 `{Li, αi, B, M}` 组合，最大化 throughput。

#### （2）**定制化的无损压缩（Lossless Compression）**
- 发现 FP16 激活值中高低字节具有不同熵特性。
- 提出 **byte-split transformation**：先按高/低字节分离再压缩，相比 ZSTD 和 ZipNN 可提升压缩率（压缩后体积减少 33%-35%）。
- 使用 ZSTD 的 entropy coding 结合该结构化布局，兼顾效率与压缩比。

#### （3）**适用于低带宽环境的推测解码（Speculative Decoding, SD）**
- 标准 SD 在跨互联网场景会因传输大量 draft tokens 导致通信开销剧增。
- 引入 **learned classifier 进行候选树剪枝（pruning）**：
  - 在第一阶段 worker 节点输出处预测 token 是否可能被接受。
  - 仅保留高置信度 candidates，平均减少 60% 传输量，同时保持 96% 的 acceptance rate。
- 动态启用机制：基于带宽估算“盈亏平衡点” `S*`，决定是否开启 SD。

#### （4）**轻量级调度器与自动化决策流程**
- 支持自动选择最佳优化策略组合（如仅压缩、微批处理+压缩、SD+剪枝等），适应不同网络条件。

---

### 相比现有方法的优势
| 方面 | BloomBee | Petals / Helix |
|------|----------|----------------|
| **通信优化粒度** | 多维协同优化（layer、batch、offload、compression、SD） | 单一或部分优化（如量化） |
| **KV Cache 管理** | 支持 CPU offloading，降低 GPU 需求 | 必须全驻留 GPU |
| **压缩有效性** | 利用浮点结构设计专用 layout，压缩率更高 | 通用压缩（ZSTD/zlib）或权重专用（ZipNN） |
| **SD 实用性** | 加入剪枝机制使其在低带宽下仍有效 | 未考虑通信代价，易失效 |
| **调度效率** | 动态规划求解快（毫秒级） | Helix 使用 MILP，耗时长达数小时 |

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用 **AlpacaEval** 中的 prompts 进行推理测试。
- 模型激活分析来自多个主流 LLM：LLaMA-13B/30B/65B、Mixtral-8×7B、Falcon-40B。

### 实验设置
#### 模型配置
- 主要模型：**LLaMA-30B**（默认 batch size=32, seq len=128）
- 其他验证模型：LLaMA-13B/65B、Falcon-7B/40B、Mixtral-8×7B

#### 集群环境（E1–E6）
| 环境 | 描述 |
|------|------|
| **E1** | 单数据中心，45 Gbps 带宽（对照组） |
| **E2–E5** | 模拟互联网环境，带宽分别为 500/250/125/20 Mbps |
| **E6** | 真实跨区域部署：<br>• California (A100)<br>• New Jersey (RTX 4090)<br>• Canada (RTX 4090)，实测带宽见 Table 3 |

#### 基线方法
- **Petals**：开源去中心化 LLM 推理框架，支持 pipeline parallelism 和量化。
- **Helix**：基于 Max-Flow 的异构资源调度系统，支持灵活路由。

> ⚠️ 两者均未集成 BloomBee 所提出的 multi-dimensional communication optimizations。

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput (tok/s)** | 每秒生成 token 数量（主要指标） |
| **Latency** | 平均每请求延迟（尤其是 per-sample completion time） |
| **Cost Efficiency (tok/s/$/h)** | 吞吐量 / GPU 成本，衡量性价比 |
| **Communication Volume** | 每跳传输的数据量（KB） |
| **Break-even Bandwidth (`S*`)** | SD 开启所需的最小带宽阈值 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总
| 指标 | BloomBee 表现 | 对比基线 |
|------|--------------|-----------|
| **最大吞吐提升** | **1.76×** 超过 Petals | @ E5, LLaMA-30B |
| **最高加速比（vs Helix）** | **1.64×** | @ E6, LLaMA-30B |
| **平均延迟降低** | 最高达 **43.20%** | vs state-of-the-art |
| **压缩效率** | 激活大小降至 **46%**（原 76.5MiB → 35.2MiB） | 比 ZSTD 小 33%，比 ZipNN 小 35% |
| **KV Cache Offloading 效果** | 减少 1 块 GPU 使用（3→2） | 成本下降 33% |
| **Speculative Decoding 剪枝效果** | 传输量 ↓60%，acceptance rate 保持 96% | 使 SD 在低带宽可用 |

---

### 与基线方法对比（Figure 8）
| 环境 | BloomBee (tok/s) | Petals | Helix |
|------|------------------|--------|-------|
| **E5 (20 Mbps)** | **67.0** | 38.1 (+1.76×) | 45.9 (+1.46×) |
| **E3 (250 Mbps)** | ~95 | ~85 | ~65 |
| **E6 (真实异构)** | **108.0** | 89.0 (+21.3%) | 66.0 (+63.6%) |

> ✅ 在所有低带宽环境下，BloomBee 显著优于基线。

---

### 消融实验结果

#### （1）Offloading 成本效益（Figure 9）
- 使用 offloading 后，GPU 数从 3 减至 2，成本从 \$1.68/h → \$1.12/h。
- **吞吐量/\$ 提升达 1.82×（E5 下 41.8 vs 22.9 tok/s/\$/h）**
- 即便在高带宽 E1，也有 1.02–1.05× 提升。

#### （2）Micro-batching 效果（Figure 10）
| 环境 | 提升幅度 |
|------|---------|
| E3 | +5.3% |
| E4 | +17.9% |
| E5 | **+39.5%** |
- 更大的 micro-batch size（16 > 8）表现更好，提供更多 overlap 机会。

#### （3）Compression 效果（Figure 11）
- 在 E5 下：
  - 仅 compression：+18.4%
  - 仅 micro-batching：+39.5%
  - 二者结合：**+76.3%**
- 增益非线性叠加，说明存在协同效应（micro-batch reshape 数据利于压缩）。

#### （4）Speculative Decoding（Figure 12）
| 环境 | SD+Prune vs Auto |
|------|------------------|
| E1 | +12.5% 吞吐，**中位完成时间快 43%**（22.8s vs 40.2s） |
| E2 | +1.0%，快 24% |
| E3 | 不启用（低于 break-even bandwidth） |
- **SD 显著改善 per-sample latency**，适合实时响应场景。

#### （5）不同模型下的最优策略选择（Table 4）
| 模型 | 高带宽（E1） | 低带宽（E4/E5） |
|------|-------------|---------------|
| 13B | SD | Compression + Micro-batching |
| 30B | SD → Autoregressive → C+M |
| 65B | SD | C+M（当带宽 <125 Mbps） |

> BloomBee 能根据环境自动选择最优策略组合。

---

## 4. 关键结论和发现

### 主要发现
1. 🔹 **通信是互联网规模 LLM 推理的首要瓶颈**  
   - 在典型家庭上行链路（20–500 Mbps）下，NIC-NIC 传输时间可达 GPU 计算时间的 **1.4–5×**。
   - 本地 CPU-GPU 数据搬运开销可忽略（<3ms），优化重点应放在跨站点通信。

2. 🔹 **多维度通信优化必须协同设计**  
   - Layer assignment、micro-batching、offloading 相互耦合（受显存限制）。
   - 分别调优次优，联合建模为 DP 问题可高效求得全局最优。

3. 🔹 **无损压缩可通过结构感知显著提效**  
   - 激活张量中高/低字节熵差异明显，**byte-split layout 比 exponent/mantissa 分离更优**。
   - 可实现接近 lossy quantization 的压缩率而不损失精度。

4. 🔹 **Speculative Decoding 在低带宽下需剪枝才可行**  
   - 未经剪枝的 SD 因传输膨胀反而降低性能。
   - 引入轻量级 classifier 实现智能剪枝，可在不牺牲 accuracy 的前提下启用 SD。

5. 🔹 **自动化策略选择至关重要**  
   - 不同模型大小、网络带宽下最优配置不同。
   - BloomBee 可动态判断何时启用 compression、micro-batching 或 SD。

---

### 方法的局限性
1. ❗ **依赖准确 profiling**：compute 和 communication 时间需预先测量，对动态变化网络适应性有限。
2. ❗ **当前仅支持 linear pipeline**：不支持 Parallax 类的动态多路径拓扑。
3. ❗ **SD classifier 需额外训练开销**：虽单次开销小，但需适配每个新模型。
4. ❗ **未解决冷启动问题**：首次请求仍需完整 autoregressive 解码。

---

### 未来工作方向
1. ✅ **扩展至动态网络环境**：在线调整策略应对波动带宽。
2. ✅ **支持更复杂并行模式**：如 tensor parallelism over internet（需新型通信协议）。
3. ✅ **端边云协同推理架构**：整合 mobile、IoT 设备参与 inference。
4. ✅ **隐私保护增强**：结合 homomorphic encryption 或 federated learning 思想。
5. ✅ **绿色 AI 视角优化能耗**：在吞吐之外引入 energy-to-token 指标。

---

## 总结
> **BloomBee 是首个将通信作为头等优化目标的互联网尺度 LLM 推理框架**。它通过 layer-offload-microbatch 联合调度、结构感知无损压缩、剪枝型 speculative decoding 等多维创新，在真实低带宽环境下实现了高达 **1.76× 吞吐提升** 和 **43.2% 延迟下降**，显著推动了去中心化 AI 的实用化进程。

📌 **代码已开源**：[https://github.com/ai-decentralized/BloomBee](https://github.com/ai-decentralized/BloomBee)

</details>

---

### 2. [Scalable AI Inference: Performance Analysis and Optimization of AI Model Serving](https://arxiv.org/abs/2604.20420)

**Authors**: Hung Cuong Pham, Fatih Gedikli  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2604.20420v1  

#### Abstract
AI research often emphasizes model design and algorithmic performance, while deployment and inference remain comparatively underexplored despite being critical for real-world use. This study addresses that gap by investigating the performance and optimization of a BentoML-based AI inference system f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scalable AI Inference: Performance Analysis and Optimization of AI Model Serving

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本研究聚焦于**AI模型部署与推理阶段的性能瓶颈**，填补了当前AI研究中“重训练、轻部署”的空白。尽管大量工作集中在模型架构设计和算法精度上，但在真实生产环境中，**低延迟、高吞吐、可扩展性和容错能力**才是决定系统可用性的关键因素。

该论文系统地分析并优化了一个基于 **BentoML** 的AI推理服务，在**现实流量模式**和**容器化部署环境**下的表现，解决了以下实际挑战：
- 如何在资源受限环境下实现高效推理？
- 如何应对突发流量（bursty traffic）对服务稳定性的影响？
- 如何提升单节点部署下的服务弹性（resilience）？

---

### 🚀 提出的新方法与思路
论文提出了一套**多层次优化策略（multi-level optimization）**，涵盖从模型到部署栈的完整链条：

| 层级 | 优化措施 |
|------|--------|
| **Model Level** | 将 FP32 PyTorch 模型转换为 ONNX 格式，并进行图结构优化（graph optimization），再量化至 FP16 半精度格式 |
| **Runtime Level** | 在 Hugging Face Transformers 中禁用梯度追踪、Dropout 和 BatchNorm 的训练行为 |
| **Service Level** | 启用 BentoML 的 adaptive batching 动态批处理机制，根据实时流量自动调整 batch size 和等待窗口 |
| **Deployment Level** | 部署于轻量级 Kubernetes 发行版 **K3s** 上，利用其自愈机制增强容错能力 |

> 🔍 创新之处在于：首次将 BentoML + ONNX + K3s 组合应用于端到端推理优化，并在**随机化负载建模**（gamma/exponential 分布）下进行全面评估。

---

### ⚖️ 相比现有方法的优势
| 对比维度 | 本文方案优势 |
|--------|-------------|
| **性能 vs. simpletransformers** | 显著降低延迟（百倍提升）、提高吞吐量（达 1900 samples/s） |
| **灵活性 vs. 黑盒框架** | 放弃 high-level wrapper（如 simpletransformers），直接使用 Hugging Face Transformers + ONNX Runtime，获得细粒度控制 |
| **部署效率 vs. Docker-only** | 引入 K3s 实现自动化恢复，无需人工干预即可应对 Pod/Control Plane 失败 |
| **测试真实性 vs. 固定速率压测** | 使用 Poisson 与 Gamma 分布模拟真实世界中的稳态与突发流量，更具现实意义 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **模型**：由 graphworks.ai 提供的预训练 RoBERTa-base 模型，用于 sentiment analysis（情感分类）
- **评估数据集**：`Sp1786/Multiclass-Sentiment-Analysis-Dataset`（Hugging Face 数据集）
  - 包含 positive、negative、neutral 三类标签
  - 固定子集：1,000 条样本，固定随机种子以确保可复现性

---

### ⚙️ 实验设置
#### 硬件配置
- CPU: 2 × Intel Xeon Gold 6230 @ 2.10GHz  
- GPU: 1 × NVIDIA Quadro RTX 8000 (48GB VRAM)  
- 内存：693 GB RAM  
- 软件栈：
  - CUDA v13.0
  - BentoML v1.4.35
  - PyTorch v2.9.1+cu128
  - ONNX Runtime v1.20.1
  - Locust v2.43.3（用于负载测试）

#### 流量建模（Load Testing Scenarios）
所有场景均保持平均请求到达率 λ = 0.5 req/s（即平均间隔 2 秒），仅改变方差（burstiness）：

| 场景 | 分布类型 | 参数 | 特征 |
|------|----------|-------|------|
| Scenario 1: Steady Traffic | Exponential | λ = 0.5 | 泊松过程，稳定到达 |
| Scenario 2: Moderate Burstiness | Gamma | α = 1.2, β ≈ 1.67 | 变异性略低，更规律 |
| Scenario 3: Extreme Burstiness | Gamma | α = 0.8, β = 2.5 | 极高方差，成簇到达 |

> ✅ 使用 Kernel Density Estimation (KDE) 验证生成流量符合理论分布。

---

### 🎯 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| **性能** | Latency（p50/p95/p99）、Throughput（samples/s）、Response Time（min/avg/max） |
| **可靠性** | Failure Rate（错误率）、Resilience（中断后恢复能力） |
| **资源效率** | Model Size（存储占用）、Memory Footprint |
| **准确性保留** | Accuracy、Macro F1、Weighted F1 |

---

### 🔁 基线方法对比
| 基线 | 描述 |
|-----|------|
| **Baseline**: `simpletransformers + FP32 PyTorch` | 使用 high-level wrapper，未启用任何优化，作为原始参考 |
| **Optimized Variants** | 多种组合方式：
  - FP32 PyTorch (direct Hugging Face)
  - FP16 PyTorch
  - FP32 ONNX
  - Optimized ONNX (graph opt.)
  - FP16 ONNX |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### 表 1：模型推理延迟（Latency per sample, ms）——越低越好
| 方法 \ Batch Size | 1 | 32 |
|------------------|----|-----|
| Baseline (FP32 PT + simpletransformers) | 2177.49 | 135.72 |
| Opt. (FP32 PyTorch) | 6.38 | 2.09 |
| Opt. (FP16 PyTorch) | 6.22 | 0.56 |
| Opt. (FP32 ONNX) | 4.09 | 2.43 |
| **Opt. (FP16 ONNX)** | **1.95** | **0.53** |

✅ **FP16 ONNX 实现最低延迟**，相比基线降低超过 **4000%**

---

#### 表 2：吞吐量（Throughput, samples/s）——越高越好
| 方法 \ Batch Size | 1 | 32 |
|------------------|-----|--------|
| Baseline | 0.46 | 7.37 |
| Opt. (FP32 PyTorch) | 156.85 | 478.01 |
| Opt. (FP16 PyTorch) | 160.71 | 1774.72 |
| Opt. (FP32 ONNX) | 244.51 | 412.14 |
| **Opt. (FP16 ONNX)** | **512.94** | **1901.56** |

✅ **FP16 ONNX 达到近 1900 samples/s**，是基线的 **258 倍以上**

---

#### 表 3：不同流量场景下的延迟表现（Latency p50, ms）
| 方法 \ 场景 | Steady | Moderate Burst | Extreme Burst |
|-----------|--------|----------------|----------------|
| Baseline | 2700 | 3000 | 3100 |
| **Opt. (FP16 ONNX)** | **27** | **26** | **26** |

✅ 优化后延迟下降两个数量级，且对 burstiness 不敏感

---

#### 表 4：响应时间（Response Time Avg, ms）
| 方法 \ 场景 | Steady | Moderate | Extreme |
|-----------|--------|----------|---------|
| Baseline | ~2845 | ~2931 | ~3015 |
| **Opt. (FP16 ONNX)** | **28.47** | **27.48** | **27.00** |

✅ 平均响应时间从 **~3秒 → ~27毫秒**

---

#### 🔍 准确性保留情况
所有变体在 Accuracy、F1 Score 上完全一致：
- **Accuracy**: 0.42
- **Macro F1**: 0.32
- **Weighted F1**: 0.34

> ❗ 注意：准确率较低归因于 domain shift（训练域为能源新闻，测试域为非正式短文本），而非优化导致退化。

---

#### 💾 存储节省
- FP32 模型大小：~498.7 MB
- FP16 模型大小：**~249.4 MB**（精确减半）
> ✅ 无损压缩 + 性能提升双重收益

---

#### 🔄 容错性测试结果（Resilience Testing）
在 K3s 单节点集群中模拟多种故障：
- 控制平面重启
- containerd 终止
- Deployment rollout
- ReplicaSet/Pod 删除

✅ 结果显示：
- 所有故障均被 Kubernetes 自动检测并重建 Pod
- RPS 短暂波动后迅速恢复
- 错误数最终归零
- 相比纯 Docker 部署需手动重启，**显著提升 resilience**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **FP16 + ONNX 是高性能推理的最佳实践**
   - 结合格式转换、图优化与半精度计算，可在不损失准确性的前提下大幅提升性能。
   - 推荐用于生产环境中的 latency-sensitive 应用。

2. **adaptive batching 显著改善吞吐与尾延迟**
   - 动态批处理机制有效平滑流量波动，尤其适用于 bursty 场景。

3. **K3s 提供轻量但有效的弹性保障**
   - 即使是单节点单副本部署，也能通过自愈机制实现自动恢复，优于传统 Docker-only 方案。

4. **流量建模必须贴近真实场景**
   - 固定速率压测易误导结论；使用 gamma/exponential 分布更能反映现实压力。

5. **high-level wrapper（如 simpletransformers）不适合生产部署**
   - 抽象层引入额外开销，限制优化空间，应优先采用底层库（如 Hugging Face Transformers）。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **单节点 K3s 无法提供高可用（HA）** | 缺乏多节点冗余，不能容忍节点级故障 |
| **仅验证 RoBERTa 模型** | 结论是否泛化至其他模型（如 LLMs）尚待验证 |
| **未使用专用加速器** | 实验仅用 GPU，未涉及 TPU、NPU 或 Inferentia 等专用 AI 芯片 |
| **外部数据存在 domain shift** | 导致准确率偏低，影响部分指标解释性 |

---

### 🔮 未来工作方向
1. **探索更低精度量化**
   - 尝试 FP8、INT8、BF16 等格式，进一步压缩模型与加速推理（需硬件支持）

2. **扩展至多节点 K3s 集群**
   - 实现 horizontal scaling 与 fault tolerance，逼近生产级 HA 架构

3. **集成更多硬件加速器**
   - 测试 ONNX Runtime 在 Intel OpenVINO、NVIDIA TensorRT、AWS Inferentia 上的表现

4. **构建 workload-aware 自适应系统**
   - 根据实时流量特征动态切换 batching 策略或精度模式

5. **跨框架横向评测**
   - 对比 BentoML vs. TorchServe vs. TensorFlow Serving vs. KServe 的综合表现

---

## ✅ 总结
本论文系统评估并优化了基于 **BentoML** 的 AI 推理服务，证明了通过 **model-level（ONNX + FP16）、runtime-level（disable grads）、service-level（adaptive batching）、deployment-level（K3s）** 的协同优化，可以实现：
- **超百倍延迟下降**
- **近千倍吞吐提升**
- **全自动容错恢复**
- **无损模型精度**

研究成果为构建**高效、可扩展、鲁棒的生产级 AI inference pipeline** 提供了实用指南，具有较强的工程指导价值。

</details>

---

### 3. [A Task Decomposition and Planning Framework for Efficient LLM Inference in AI-Enabled WiFi-Offload Networks](https://arxiv.org/abs/2604.21399)

**Authors**: Mingqi Han, Xinghua Sun  
**Category**: cs.DC  
**Published**: 2026-04-24  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2604.21399v1  

#### Abstract
AI WiFi offload is emerging as a promising approach for providing large language model (LLM) services to resource-constrained wireless devices. However, unlike conventional edge computing, LLM inference over WiFi must jointly address heterogeneous model capabilities, wireless contention, uncertain t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Task Decomposition and Planning Framework for Efficient LLM Inference in AI-Enabled WiFi-Offload Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **AI-enabled WiFi-offload networks** 中的 **LLM inference offloading** 问题，解决以下挑战：
- **任务复杂性不确定**：LLM 推理任务（如问答、推理、多模态理解）的难度、输出长度和质量难以预先量化。
- **异构模型能力差异**：不同边缘 AP 部署的 LLM 模型在规模、计算能力和内存带宽上存在显著差异。
- **无线信道竞争与通信开销**：WiFi 网络中存在 CSMA/CA 机制下的信道争用、RTS/CTS 握手、ACK 开销等现实通信行为。
- **传统 offloading 范式局限**：传统的 binary 或 partial offloading 无法有效利用 LLM 任务的语义依赖性和可分解性（如 chain-of-thought）。

### 提出的新方法与思路
提出了一种 **基于 LLM 的任务分解与规划框架（LLM-based task decomposition and planning framework）**，其核心创新包括：

- **Planner LLM 的引入**：
  - 使用一个 **Planner LLM** 对输入任务进行分析，决定是否需要分解，并生成可执行的子任务集合 $ S(T) = \{S_1, ..., S_K\} $。
  - 在分解过程中，Planner 同时预测每个子任务的关键属性：
    - 预期输出 token 数量（output token length）
    - 子任务难度（difficulty）
    - 在不同节点上的预期正确率（expected correctness）

- **分解感知的调度策略（Decomposition-aware scheduling）**：
  - 基于 Planner 提供的信息，设计了一个加权评分函数 $ J_{k,i} = w_a \alpha_{k,i} - w_d T_{\text{tot},k,i} $，综合考虑准确性与延迟。
  - 实现子任务到本地 UE 或边缘 AP 的最优分配、聚合节点选择及执行顺序优化。

- **轻量化 Planner 的蒸馏训练**：
  - 为适应边缘部署资源限制，采用 **knowledge distillation** 方法，将大型教师模型（DeepSeek-v3.2）的规划能力迁移到轻量级学生模型（Qwen2.5-7B-Instruct）上。

### 相比现有方法的优势
| 方面 | 本方法优势 |
|------|-----------|
| **灵活性** | 支持动态任务分解 vs. 传统 binary/partial offloading 的固定模式 |
| **精度提升** | 利用异构模型特长匹配子任务，提升最终答案质量 |
| **延迟优化** | 协同执行避免单点瓶颈，减少排队延迟 |
| **实用性** | 蒸馏后的轻量 Planner 可部署于边缘设备，兼顾性能与可行性 |

---

## 2. 核心实验方法和设置

### 数据集
任务采样自三个公开基准，覆盖多样化推理场景：
- **AIME-2024 (Math)**：数学推理类任务
- **LiveBench-Reasoning (Daily)**：日常逻辑与常识推理
- **GPQA (Science)**：科学领域高难度问答

### 实验设置
- **仿真环境**：50m × 50m 室内 WiFi 场景，支持多 UE 和多 edge AP 共存
- **网络参数**：
  - WiFi 标准参考 **WiFi 7**，包含 RTS/CTS、binary exponential backoff、TF 触发帧等机制
  - 信道带宽：40 MHz，中心频率 5 GHz
  - AP 数量：2 或 4 个；UE 数量：5 或 7 个
- **硬件配置**：
  - 边缘 AP 模型大小：7B / 14B / 32B
  - UE 本地模型：1.5B
  - GPU 计算能力（Fe）：[120, 312] TFLOPS (fp16)
  - 内存带宽（Abm）：[0.6, 2] × 10¹² Bytes/s
- **任务到达**：泊松过程，到达率 λ = 0.1 tasks/s
- **每轮模拟时间**：600 秒，共运行 10 个独立 episode 取平均

### 评估指标
- **Average Latency**：端到端任务完成时间
- **Overall Reward**：综合延迟与准确性的加权目标函数值
- **Task Accuracy**：按数据集分类的任务正确率
- **Communication Overhead**：包含上行传输、子任务分发、结果回传等各阶段延迟建模

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Local-Only** | 所有任务仅由 UE 本地小模型处理 |
| **Nearest-Edge** | 整体任务卸载至最近的边缘 AP，不分解 |
| **Proposed Planner (DS-v3.2)** | 使用大模型作为 Planner（理想情况） |
| **Proposed Fine-tuned Planner (Qwen-7B)** | 蒸馏后轻量级 Planner，实际可部署版本 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Fig. 2）

| 方法 | 平均延迟 (s) | 总体 Reward | 准确率 (GPQA) | 准确率 (AIME-2024) |
|------|---------------|--------------|----------------|--------------------|
| **Local-Only** | 19.367 | -0.181 | ~0.1 | 0.0 |
| **Nearest-Edge** | 14.542 | 0.122 | 0.313 | 0.189 |
| **DS-v3.2 Planner** | 13.773 | **0.318** | **0.609** | 0.500 |
| **Fine-tuned Qwen-7B Planner** | **12.049** | 0.222 | 0.600 | 0.429 |
| **原始 Qwen-7B（未微调）** | 16.300 | 0.087 | — | — |

> 注：延迟单位为秒（s），Reward 为归一化得分。

### 与基线方法对比结果
- 相比 **Nearest-Edge**：
  - 平均延迟降低 **20%**（从 14.542s → 12.049s）
  - 总体 Reward 提升 **80%**（0.122 → 0.222）
- 相比 **Local-Only**：
  - 显著改善延迟与奖励，尤其在复杂任务（如 AIME）上实现从“几乎无法求解”到部分成功
- **轻量 Planner 表现接近大模型**：
  - 尽管使用更小模型，fine-tuned Qwen-7B 在准确率上逼近 DS-v3.2（如 GPQA 上 0.600 vs. 0.609）
  - 且平均延迟更低，表明其调度效率更高

### 消融实验与发现（隐含分析）
虽然文中未明确列出消融实验表格，但从结果可推断：
- **任务分解的有效性**：Local-Only 与 Nearest-Edge 均无分解能力，性能远低于 proposed 方法 → 分解是关键增益来源
- **Planner 质量的重要性**：未经微调的 Qwen-7B 表现差（Reward 仅 0.087），说明直接使用通用小模型无法胜任规划任务
- **知识蒸馏的有效性**：通过 teacher guidance 微调后，轻量模型性能大幅提升 → 验证了 distillation 策略的成功

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **任务分解能显著改善 latency-accuracy tradeoff**  
   动态将复杂任务拆分为子任务并协同执行，比整体卸载或本地运行更具优势。

2. ✅ **Planner LLM 是系统性能的关键**  
   能够准确估计子任务难度、输出长度和预期正确性，是实现高效调度的前提。

3. ✅ **异构资源需智能编排**  
   不同边缘节点的模型能力、负载状态和通信条件差异大，必须联合优化才能发挥最大效益。

4. ✅ **轻量化 Planner 具备实用价值**  
   经过 teacher-guided fine-tuning 的 Qwen-7B 模型可在保持高性能的同时满足边缘部署需求，具备工程落地潜力。

### 方法的局限性
- 🚫 **依赖高质量 Planner 输出**：若 Planner 自身推理错误（如误判子任务依赖关系），可能导致调度失败或性能下降。
- 🚫 **扩展性尚未验证**：当前仿真规模较小（最多 4 AP + 7 UE），大规模密集网络中的可扩展性有待研究。
- 🚫 **MAC 层建模仍简化**：虽引入 WiFi 7 特性，但仍假设理想同步与调度协调，未完全模拟真实干扰与拥塞。
- 🚫 **未考虑动态环境变化**：如移动性、突发流量、模型更新等动态因素未纳入考量。

### 未来工作方向
- 🔁 **自适应 decomposition policy**：根据实时网络状态动态调整是否分解及分解粒度。
- 📡 **更精细的 MAC/PHY 层联合建模**：结合 OFDMA、multi-link operation 等 WiFi 7 高级特性优化通信效率。
- 🤝 **分布式 Planner 架构**：探索多个 AP 协同决策的去中心化规划机制。
- ⚙️ **Joint communication-computation orchestration**：进一步整合无线资源分配（如功率控制、信道选择）与计算调度。
- 🧠 **强化学习增强 Planner**：引入 RL 进行在线策略优化，提升长期累积 reward。

---

> **总结一句话**：  
> 本文提出了一种 **Planner-guided collaborative inference** 新范式，在 AI-enabled WiFi-offload 网络中实现了 **更优的 LLM 推理延迟-准确性平衡**，并通过 **knowledge distillation** 使高性能规划能力得以在边缘轻量部署，为未来智能无线边缘系统提供了重要技术路径。

</details>

---

### 4. [Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems](https://arxiv.org/abs/2604.21794)

**Authors**: Ye Yu, Heming Liu, Haibo Jin, Xiaopeng Yuan, Peng Kuang, Haohan Wang  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.21794v1  

#### Abstract
Multi-agent systems built on large language models have shown strong performance on complex reasoning tasks, yet most work focuses on agent roles and orchestration while treating inter-agent communication as a fixed interface. Latent communication through internal representations such as key-value c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **Multi-Agent Systems (MAS)** 虽然在复杂推理任务中表现出色，但其**通信机制通常被视为固定的接口**，例如通过自然语言文本传递中间结果。这种设计存在以下问题：
- **信息损失**：内部连续的隐状态（如 Key-Value Caches）被强制解码为离散 token，导致信息压缩和失真。
- **优化边界**：离散通信阻断了梯度传播，使得通信过程无法与推理行为联合优化。
- **表达能力受限**：文本协议难以承载细粒度的推理信号。

因此，尽管 agent 角色和流程编排已被广泛研究，**通信本身尚未成为可学习、可优化的组件**。

### 提出了什么新方法或新思路
本文提出 **DiffMAS**（Differentiable Multi-Agent System），一个将**潜通信（latent communication）作为可训练接口**的框架，实现多智能体系统的端到端优化。

#### 核心思想：
- 将 **Key-Value (KV) Caches** 作为连续的潜通信媒介，在 agent 之间直接传递。
- 构建一个**共享的 KV Trace**，由前序 agents 逐步填充，最终 agent 在此基础上进行自回归生成。
- 采用 **Supervised Fine-Tuning (SFT)** 对整个多 agent 推理轨迹进行训练，仅更新最终 agent 的 **LoRA 参数**，保持主干模型冻结。

#### 两阶段流程：
1. **Stage I: 构建 KV Trace**  
   Agent 1 到 K-1 依次执行推理，将其生成的 KV 状态追加到共享缓存中，不进行梯度更新。
2. **Stage II: 最终生成与反向传播**  
   最终 agent 基于完整的 KV Trace 进行解码，计算与标准答案的交叉熵损失，并将梯度反向传播至其 LoRA 模块，从而联合优化通信与推理。

### 相比现有方法的优势
| 方法 | 通信方式 | 是否可学习 | 主要缺点 |
|------|----------|------------|----------|
| **Single-agent** | 无 | — | 缺乏任务分解能力 |
| **TextMAS** | 自然语言文本 | 否 | 信息丢失，梯度断裂 |
| **LatentMAS (training-free)** | 原始 KV Caches | 否 | 分布不匹配，解码不稳定 |
| **C2C** | 学习融合模块 | 是 | 需额外架构，训练数据分布不一致 |
| ✅ **DiffMAS (本文)** | **可学习的 KV Trace** | ✅ | **端到端优化，稳定且高效** |

**优势总结**：
- **端到端可微**：首次实现对多 agent 潜通信路径的联合优化。
- **参数高效**：仅需 LoRA 微调，无需修改主干模型。
- **稳定性高**：相比非训练型潜通信，显著提升解码一致性与低困惑度。
- **通用性强**：适用于数学、科学问答、代码生成、常识推理等多种任务。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖多个高难度基准，涵盖不同推理类型：

| 类别 | 数据集 | 描述 |
|------|--------|------|
| **数学与科学推理** | AIME24, AIME25, GPQA-Diamond | 多步符号推理，要求精确数值或分类答案 |
| **常识推理** | OpenBookQA | 基于小学科学知识的结构化推理 |
| **代码生成** | HumanEval+, MBPP+ | Python 程序合成，评估功能正确性与泛化能力 |

### 实验设置和评估指标
- **模型规模**：Qwen3-4B, 8B, 14B；Mistral3-8B；DeepSeek-R1-Distill-Qwen-32B
- **训练配置**：
  - 使用 **LoRA**（rank=8, α=16）进行参数高效微调
  - 学习率：5e-5，AdamW + Cosine Schedule
  - 梯度累积：64 micro-batches
- **推理配置**：
  - 温度 = 0.6，top-p = 0.95
  - 输出长度上限根据不同任务设定（最高达 32,768 tokens）
- **评估指标**：
  - **Accuracy (ACC)**：主要性能指标
  - **Perplexity (PPL)**：衡量解码稳定性
  - **Self-Consistency**：采样多次看输出一致性，反映推理鲁棒性

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Single** | 单模型直接生成 |
| **TextMAS** | 多 agent 文本通信（如 Planner → Critic → Refiner → Solver） |
| **LatentMAS** | 不训练的 KV 共享，即 training-free latent exchange |
| **C2C** | Cache-to-Cache，使用学习模块融合 KV 表示 |

所有方法使用相同的 agent 角色、顺序和 prompt，仅改变通信方式。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

| 模型 | 任务 | Single | TextMAS | LatentMAS | ✅ **DiffMAS** |
|------|------|--------|---------|-----------|----------------|
| Qwen3-4B | AIME24 | 43.3% | 46.7% (+3.4%) | 50.0% (+6.7%) | **63.3% (+20.0%)** |
| Qwen3-8B | AIME24 | 50.0% | 50.0% (+0.0%) | 56.7% (+6.7%) | **76.7% (+26.7%)** |
| Qwen3-8B | GPQA-Diamond | 39.9% | 43.4% (+3.5%) | 45.5% (+5.6%) | **60.1% (+20.2%)** |
| Qwen3-14B | HumanEval+ | 77.2% | 81.5% (+4.3%) | 86.8% (+9.6%) | **87.7% (+10.5%)** |
| Qwen3-14B | MBPP+ | 68.5% | 72.8% (+4.3%) | 75.7% (+7.2%) | **77.2% (+8.7%)** |
| DeepSeek-32B | HumanEval+ | 80.7% | 82.4% (+1.7%) | 83.3% (+2.6%) | **88.5% (+7.8%)** |

> ✅ **最高提升达 +26.7%（AIME24）**，且在所有任务和模型尺度上均取得最佳性能。

### 与基线方法的对比结果
- **vs. Single-agent**：全面超越，尤其在小模型上增益显著（如 Qwen3-4B 在 AIME24 上从 43.3% → 63.3%）。
- **vs. TextMAS**：大幅领先，说明潜通信优于文本通信。
- **vs. LatentMAS**：虽然后者已有一定提升，但 DiffMAS 通过训练进一步拉大差距（如 Qwen3-8B 在 GPQA-Diamond 上 45.5% → 60.1%）。
- **vs. C2C**：C2C 因训练数据（OpenHermes-2.5）偏向指令跟随而非长程推理，表现较差，尤其在数学任务上崩溃（如 AIME24 上为 0.0%）。

### 消融实验结果

#### （1）学习通信 vs. 学习任务（Table 4）
| 方法 | AIME24 | AIME25 | GPQA-Diamond |
|------|--------|--------|--------------|
| TextMAS + SFT | 76.7% | 50.0% | 53.5% |
| ✅ DiffMAS | **76.7%** | **56.7%** | **60.1%** |

> 在分布偏移任务（AIME25, GPQA）上，DiffMAS 显著优于仅做任务微调的方法，证明“**学会沟通**”带来了泛化优势。

#### （2）通信步数的影响（Table 3）
- 引入少量通信步骤（10步）即可带来巨大提升（50.0% → 76.7%）。
- 步数过多（>40）反而性能下降，表明**过长的 KV Trace 会引入噪声和冗余**。
> 结论：DiffMAS 学会了一种**紧凑高效的通信协议**。

#### （3）连续 vs. 拼接式潜通信（Table 5）
| 方法 | AIME24 | GPQA-Diamond |
|------|--------|--------------|
| StitchMAS（独立生成后拼接） | 60.0% | 48.4% |
| ✅ DiffMAS（连续构建 Trace） | **76.7%** | **60.1%** |

> 证明 **共享的、连续构建的 KV Trace** 比独立生成再拼接更有效，支持了端到端训练的重要性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **潜通信可以且应该被学习**：将 KV Caches 作为通信媒介并进行端到端训练，能显著提升多 agent 系统的推理能力和稳定性。
2. **通信是优化的关键目标**：相比单纯改进 agent 能力或流程设计，**优化通信机制本身**是提升系统性能的新维度。
3. **DiffMAS 实现了稳定性与表达性的平衡**：
   - 相比文本通信：保留更多推理细节；
   - 相比非训练型潜通信：避免了解码混乱和注意力漂移。
4. **非覆盖式通信具有梯度优势**：理论分析表明，**concatenative KV Trace** 避免了 overwriting communication 中的深度依赖梯度衰减，使各阶段 agent 的贡献更均衡。

### 方法的局限性
- **依赖预设的 agent 流程**：当前框架假设 agent 的角色和顺序是固定的，未探索动态调度或拓扑学习。
- **KV Trace 可能冗余**：随着 agent 数量增加，Trace 长度线性增长，可能引入干扰或计算开销。
- **仅限于序列式架构**：目前基于串行 agent 设计，是否适用于图状或多分支结构尚待验证。
- **训练数据量小但有效**：虽然在小数据集上成功，但在更大规模、更复杂任务上的扩展性仍需验证。

### 未来工作方向
- **动态 agent 调度**：结合强化学习或搜索策略，自动决定 agent 的调用顺序与数量。
- **压缩与去噪机制**：引入轻量级模块对 KV Trace 进行过滤或压缩，提升效率。
- **跨模型潜通信**：探索异构 LLM 之间的可学习潜通信协议。
- **全可微 MAS**：将 agent 决策、角色分配等也纳入可学习范畴，迈向真正的 end-to-end 多 agent 系统优化。

---

> **总结一句话**：  
> DiffMAS 首次将 **multi-agent communication** 视为可学习的计算图一部分，通过 **端到端微调 KV-mediated latent trace**，实现了推理准确性与解码稳定性的双重突破，为构建更强大、更智能的协作式语言系统开辟了新路径。

</details>

---

### 5. [Super Apriel: One Checkpoint, Many Speeds](https://arxiv.org/abs/2604.19877)

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

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大型语言模型（LLM）在长上下文生成任务中面临 **Full Attention (FA)** 推理成本高昂的问题。FA 层的 KV Cache 随序列长度呈线性增长，导致内存占用高、解码速度慢。虽然已有多种高效注意力机制（如 SWA、GDN、KDA），但传统混合架构通常在训练前或训练后**固定**某一层的注意力配置（placement），缺乏部署时的灵活性。

这带来了以下限制：
- 单一模型只能服务一个固定的“速度-质量”权衡点；
- 不同负载（短提示高并发 vs 长上下文重推理）需部署多个独立模型；
- 无法实现运行时动态切换以适应流量高峰或任务需求。

### 提出的新方法与创新
Super Apriel 提出了一种 **Token-Mixer Supernet** 架构，其核心创新在于：

#### ✅ **One Checkpoint, Many Speeds**
- 在每个 decoder layer 中集成 **四种预训练的 mixer 类型**：  
  `FA`（Full Attention）、`SWA`（Sliding Window Attention）、`KDA`（Kimi Delta Attention）、`GDN`（Gated DeltaNet）。
- 所有 mixer 共享 FFN、Embedding、LayerNorm 等参数，仅 mixer 权重不同。
- **Placement** 定义为每层选择一个 mixer 的组合（共 $4^{48}$ 种可能）。
- 在 serving 时可**无需重新加载权重**地在不同 placement 间切换，从而从**单个 checkpoint 支持多个推理速度档位**。

#### ✅ **Placement Optimization Toolkit**
- 引入基于 **Cluster Expansion** 的代理模型（surrogate model），将复杂的组合优化问题转化为可精确求解的动态规划（DP）问题。
- 可快速扫描整个 $4^{48}$ 空间，找出每个成本预算下的最优 placement，构建完整的 Pareto 前沿。

#### ✅ **Speculative Decoding without Draft Model**
- 利用共享 checkpoint 的特性，直接使用轻量级 placement（如全 GDN）作为 draft model，全 FA 作为 target verifier，省去额外训练 draft 模型的成本。

#### ✅ **Stochastic Distillation + Targeted SFT**
- **Distillation 阶段**：对所有 mixer 并行进行随机采样训练（stochastic placement sampling），确保所有配置都被充分学习。
- **SFT 阶段**：针对选定的 Pareto 最优 presets 进行定向微调，进一步提升特定配置的质量。

### 相比现有方法的优势
| 维度 | 传统方法 | Super Apriel |
|------|--------|-------------|
| 架构灵活性 | 固定 placement，不可变 | 运行时灵活切换 placement |
| 部署效率 | 多个模型对应多个速度点 | 单一 checkpoint 支持多速度 |
| 搜索效率 | 网格搜索 / NAS / Beam Search | 精确 DP 求解 Pareto 前沿 |
| 资源利用率 | 需维护多个模型 | 减少存储与运维开销 |
| Speculative Decoding | 需单独训练 draft model | 内建 draft model，零额外成本 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### **Distillation 数据集（266B tokens）**
| 类别 | 比例 |
|------|-----|
| Reasoning traces & SFT | 29.3% |
| Code | 10.0% |
| Math & STEM | 5.0% |
| Web / encyclopedic text | 6.7% |
| Image (multimodal) | 49.0% |

> 注：强调高质量推理轨迹（multi-step proofs, structured problem-solving），避免纯预训练数据导致推理能力崩溃。

#### **SFT 数据集（137B tokens）**
| 类别 | 比例 |
|------|-----|
| SFT: Code | 36.1% |
| SFT: Math & STEM | 38.7% |
| SFT: Chat, generic reasoning & IF | 11.3% |
| SFT: Tool use | 12.0% |
| SFT: Safety, robustness & content moderation | 2.0% |

---

### 实验设置

| 参数 | 设置 |
|------|------|
| 模型大小 | 15B parameters, 48 layers |
| Base Model | Apriel 1.6 (derived from Pixtral-12B) |
| Mixer Types | FA, SWA ($w=4096$), KDA, GDN |
| 训练框架 | Fast-LLM |
| Serving 框架 | vLLM |
| 硬件 | H100 GPUs |
| 序列长度 | Distillation: 16k; SFT: 32k |

---

### 评估指标

#### **质量指标（Quality Score）**
- **Dev Benchmarks**（用于 placement 优化目标）：
  - MMLU（57 subjects）
  - GSM8K（数学应用题）
  - MATH500（竞赛数学）
  - AIME 2024 / 2025
  - FDA、SWDE（信息抽取）
  - NIAH、RULER（长程检索）
- **Unseen Benchmarks**（最终评估，不参与训练决策）：
  - MMLU-Pro、GPQA、HLE、LCB、T2-Bench、IFEval、AIME(NV)

> 默认优化目标：MATH500、AIME-24、AIME-25 上的 trace log-likelihood 平均值。

#### **效率指标（Throughput）**
- 解码吞吐量（tokens/s/GPU）
- 相对于 all-FA 的 speedup（@16k 和 @32k）
- 成本模型：
  - **Idealized**: 假设各 layer 成本独立
  - **Regression-based**: 基于实测数据拟合线性模型（更准确）

#### **基线对比模型**
- **内部基线**：Apriel-H1 (15B hybrid)
- **外部基线**：
  - Qwen-3.5 27B (MoE + GDN)
  - Nemotron-3-Nano 30B (Mamba + MoE)
  - Falcon-H1R 7B (Mamba)
  - Nemotron-Nano 12B v2
  - OLMo-Hybrid-Think 7B

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4 & 13）

| Preset | FA/SWA/KDA/GDN | Avg Acc (%) | Retention (%) | Speedup @32k |
|--------|----------------|------------|--------------|-------------|
| **all-FA** | 48/0/0/0 | 74.2 | 100% | 1.0× |
| **RegLklhd-26** | 12/26/6/4 | 71.1 | 96% | **2.9×** |
| **RegLklhd-18** | 3/25/4/16 | 69.7 | 94% | **4.8×** |
| **RegLklhd-13** | 0/16/13/19 | 60.2 | 81% | **6.9×** |
| **RegLklhd-10** | 0/10/5/33 | 57.2 | 77% | **10.7×** |

> 所有配置均来自**同一个 checkpoint**。

---

### 与基线方法的对比结果

#### ✅ **优于同类 15B 模型**
| Model | Size | Speedup | Math Avg | All-tasks Avg |
|-------|------|---------|----------|---------------|
| **Super Apriel RegLklhd-26** | 15B | **2.9×** | **88.3** | **71.1** |
| Apriel-H1 | 15B | 2.0× | 80.4 | 58.4 |
| → 提升：+7.9 数学分，+12.7 总体分

#### ✅ **相比更大模型仍具竞争力**
| Model | Size | Speedup | Math Avg | All-tasks Avg |
|-------|------|---------|----------|---------------|
| **Super Apriel RegLklhd-10** | 15B | **10.7×** | 81.2 | 57.2 |
| Nemotron-3-Nano 30B | 30B | 4.1× | 89.0 | 83.0 |
| Qwen-3.5 27B | 27B | 0.55× | 92.6 | 80.7 |

> Super Apriel 在 **一半参数量下实现近 11× 加速**，适合高吞吐场景。

#### ✅ **长上下文优势显著**
- Super Apriel 高效 placements 在 **从 16k 到 32k 上下文时获得 80–155% 的相对加速增益**。
- 对比外部混合模型（仅 5–46% 增益），说明其高效 mixer（SWA/KDA/GDN）具有固定内存开销，在长文本中优势放大。

---

### 消融实验结果

#### 🔍 **Placement Ranking Stability**
- 在 **0.5B 开发模型上**，placement 排名在训练早期即稳定（Spearman ρ > 0.98）。
- 但在 **15B 主模型上**，Pareto 前沿（尤其是中等成本配置）表现出更高波动性。
- **结论**：小规模实验不能完全外推至大模型，必须在目标尺度上验证 placement 优化策略。

#### 🔍 **Training Strategy Ablation**
- **Distillation 阶段**：采用 **stochastic placement sampling** 更优，能避免 false optima，并保证所有配置均衡发展。
- **SFT 阶段**：采用 **targeted training** 更高效，集中优化已知优质 presets。

#### 🔍 **SFT 显著提升性能**
- 所有 placement 在 SFT 后均有明显提升，尤其在数学推理任务上：
  - 例如 `Idealized|All-6` 在 AIME'24 上从 63.3 → 83.3（+20.0 分）
  - 在未见任务（如 Tau2、LCB）上也表现强劲，证明泛化能力。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **单一 checkpoint 可支持多样化推理模式**  
   Super Apriel 验证了“一次训练，多种速度”的可行性，通过运行时 placement 切换满足异构负载需求。

2. ✅ **高效 mixer 在长上下文中优势巨大**  
   SWA/KDA/GDN 的固定状态设计使其在 32k+ 上下文下获得远超传统混合模型的加速比。

3. ✅ **Cluster Expansion + DP 实现高效 placement 搜索**  
   将指数级搜索空间压缩为可精确求解的优化问题，是构建 Pareto 前沿的关键。

4. ✅ **Stochastic Distillation 是稳健起点**  
   在 distillation 阶段使用随机 placement sampling 可防止过早收敛到局部最优，保障探索完整性。

5. ✅ **Speculative Decoding 可内建实现**  
   利用自身轻量 placement 作 draft model，无需额外模型即可实现高速推理。

---

### 局限性（Limitations）

| 问题 | 描述 |
|------|------|
| **长程任务退化** | 使用 GDN/KDA 的 placement 在 NIAH/RULER 等长程检索任务上严重退化。 |
| **代理模型假设限制** | Cluster Expansion 假设短程交互为主，若存在跨层依赖则可能欠拟合。 |
| **Log-likelihood 代理偏差** | 使用 trace likelihood 作为生成准确性的代理，可能存在与 exact-match 分数的差距。 |
| **线性成本模型误差** | 当前成本模型对“少数派 mixer”估计不准，需排除 singleton placements。 |
| **推理引擎依赖性强** | 性能受 vLLM 等引擎版本影响，CUDA Graph 支持尚不完善。 |

---

### 未来工作方向（Outlook）

1. **Post-Training RL**  
   计划使用 Group Relative Policy Optimization（GRPO）进行强化学习微调，探索 FA 是否可作为 KL 正则项来稳定训练。

2. **Scaling & Generalization**  
   - 将该范式扩展到其他 teacher 模型；
   - 引入更多 mixer 类型（如 Mamba-2、Lightning Attention）；
   - 在生产规模上比较 stochastic vs targeted sampling。

3. **Deployment Roadmap**
   - 实现 per-request placement routing；
   - 支持 model thinning 以减少 GPU 内存占用；
   - 推动 inference engine 对混合架构的原生支持。

4. **Understanding Landscape Drift**
   - 验证 15B 上观察到的 frontier volatility 是否在更大模型中持续存在；
   - 探索如何在训练过程中动态识别最优 placement。

---

## 附录：开源资源

✅ **已全部开源！**

| 资源 | 地址 |
|------|------|
| 模型权重 | `SuperApriel-15B-Base`, `SuperApriel-15B-Instruct` |
| 训练代码 | [Fast-LLM](https://github.com/ServiceNow/Fast-LLM) |
| 推理服务 | vLLM 扩展支持 supernet serving |
| 工具包 | `place-layers`（即将发布） |
| 训练日志 | SuperApriel-15B / 0.5B |

> GitHub: https://github.com/ServiceNow/Fast-LLM  
> Hugging Face: https://huggingface.co/ServiceNow  

--- 

> **一句话总结**：  
> Super Apriel 通过构建一个支持四种 mixer 的 supernet，实现了“一个 checkpoint，多种推理速度”，并结合 surrogate-guided search 和 targeted SFT，在保持高质量的同时达到最高 **10.7× 解码加速**，为灵活高效的 LLM 部署提供了新范式。

</details>

---

### 6. [Fast Bayesian equipment condition monitoring via simulation based inference: applications to heat exchanger health](https://arxiv.org/abs/2604.20735)

**Authors**: Peter Collett, Alexander Johannes Stasik, Simone Casolo, Signe Riemer-S{\o}rensen  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.20735v1  

#### Abstract
Accurate condition monitoring of industrial equipment requires inferring latent degradation parameters from indirect sensor measurements under uncertainty. While traditional Bayesian methods like Markov Chain Monte Carlo (MCMC) provide rigorous uncertainty quantification, their heavy computational b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast Bayesian equipment condition monitoring via simulation based inference: applications to heat exchanger health

---

## 1. 论文的主要贡献和创新点

### 解决的问题
工业设备（如换热器）的**健康状态监测**面临以下挑战：
- 关键退化参数（如结垢、泄漏）无法直接测量，只能通过**间接传感器数据**推断。
- 传统贝叶斯方法（如 MCMC）虽能提供严格的不确定性量化，但计算开销巨大，难以满足**实时诊断**需求。

### 提出的新方法
提出一种基于 **Simulation-Based Inference (SBI)** 的 AI 驱动框架，结合 **amortized neural posterior estimation (NPE)** 实现快速贝叶斯推理。

#### 核心思路：
- 利用物理仿真模型生成大量“参数输入 → 观测输出”的配对数据。
- 使用神经密度估计器（Neural Spline Flow, NSF）学习从观测摘要统计量到**完整后验分布**的映射。
- 推理阶段无需重复调用仿真器，实现**近即时诊断**。

### 相比现有方法的优势
| 维度 | 传统 MCMC | 本文 SBI 方法 |
|------|-----------|----------------|
| **推理速度** | 每次诊断需数千次仿真调用，耗时数秒至分钟 | 训练后单次推理仅需 **0.029 秒**，加速 **82×** |
| **可扩展性** | 不适用于高频、多资产场景 | 支持**大规模、实时部署** |
| **不确定性量化** | 提供准确后验分布 | 后验质量与 MCMC 相当，保持可靠置信区间 |
| **适用性** | 要求显式 likelihood 函数 | **likelihood-free**，适用于黑箱模拟器或复杂系统 |

> ✅ **创新点总结**：首次将 amortized SBI 成功应用于工业设备 PHM（Prognostics and Health Management），实现了高保真贝叶斯推理与实时性的统一。

---

## 2. 核心实验方法和设置

### 数据集
- **合成数据集**：基于一个**随机退化模型**生成，包含六种典型工况（见下表）。
- 总样本量：每种场景生成 **500 组带噪声的时间序列数据**，共 3,000 条记录。
- 模拟器基础：采用 **effectiveness-NTU 模型** 进行热流体仿真。

#### 故障场景设计（Table I）
| 场景 | 故障类型 | 参数设置 |
|------|--------|---------|
| 1. Weak Fouling | 结垢 | 高频低幅跳跃 (`λ=5.0`, `βf=0.005`) |
| 2. Batch SD | 批处理停机导致结垢 | 低频高幅跳跃 (`λ=0.5`, `βf=0.03`) |
| 3. Boiler FW | 锅炉给水系统严重结垢 | 中频高幅 (`λ=3.0`, `βf=0.05`) |
| 4. Mild Leak | 轻微泄漏 | 缓慢增长 (`β=0.0005`) |
| 5. Severe Leak | 严重泄漏 | 快速增长 (`β=0.0010`) |
| 6. No Failure | 正常运行 | 无故障 |

### 实验设置
- **训练阶段（SBI）**：
  - 使用 **50,000 次前向仿真** 构建训练集。
  - 输入为 **25维摘要统计量**（summary statistics），包括温度差、流量变化等的均值、标准差、趋势等。
  - 模型架构：**Neural Spline Flow (NSF)** + MLP conditioner（2层×50单元）。
  - 工具包：`sbi` (PyTorch-based)。
- **推理阶段**：
  - 对每个测试样本进行后验采样。
  - MCMC 使用 **NumPyro** 实现 NUTS sampler，共 20,000 次采样（4 chains × 5,000 samples）。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Failure-mode accuracy** | 分类正确率（六类：无故障、结垢、泄漏、两者） |
| **Wasserstein distance** | 衡量 SBI 与 MCMC 后验分布之间的相似性（越小越好） |
| **CRPS (Continuous Ranked Probability Score)** | 评价概率预测的锐度与准确性（越小越好） |
| **Credible interval coverage** | 真实参数是否落在预测置信区间内 |
| **Inference time** | 单次诊断耗时（含预处理） |

### 基线方法对比
- **MCMC (No-U-Turn Sampler)**：作为黄金标准基准，提供精确但缓慢的后验估计。
- **SBI (Sequential Neural Posterior Estimation, SNPE)**：本文提出的方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### (1) 故障模式识别准确率（Table II）
| 场景 | MCMC 准确率 | SBI 准确率 |
|------|-------------|-----------|
| Weak Fouling | 100% | 100% |
| Batch SD | 100% | 100% |
| Boiler FW | 100% | 100% |
| Mild Leak | 99.8% | 100% |
| Severe Leak | 99.6% | 100% |
| No Failure | 98.2% | 98.6% |

✅ **结论**：SBI 在所有场景下均达到与 MCMC 相当甚至更优的分类精度。

#### (2) 参数估计一致性
- **图5（Scatterplot of posterior medians）** 显示 SBI 与 MCMC 的后验中位数高度一致，集中在对角线上。
- **图6（Marginalized posteriors）** 显示两种方法在 `T`（起始时间）、`βf`、`β`、`λ` 上的估计分布基本重合。

#### (3) 后验分布质量
- **Wasserstein Distance（图7）**：大多数情况下距离接近零，表明 SBI 后验与 MCMC 高度相似；仅在稀疏事件（如 Batch SD）中略有偏差。
- **CRPS（图8）**：SBI 与 MCMC 的 CRPS 分布几乎完全重叠，说明其概率预测质量相当。

#### (4) 计算效率（Table III & 图10）
| 指标 | MCMC | SBI |
|------|------|-----|
| 单次推理时间 | **2.4 秒** | **0.029 秒** |
| 加速倍数 | — | **82×** |
| 成本平衡点 | — | 约 **6 次诊断后** SBI 更便宜 |

> 💡 注：虽然当前模拟器较快（每次 ~0.03s），但在真实工厂中若单次仿真耗时达分钟级，该加速比将带来**数十分钟到数小时的实际节省**。

### 消融实验（隐含分析）
尽管未明确列出消融实验，但文中进行了多个敏感性分析：
- **不同训练规模（5k vs 50k 仿真）**：验证了训练充分性。
- **不同 summary statistics 设计**：确认所选 25 维特征足以捕捉关键动态。
- **稀疏事件下的 prior 影响**：揭示了 `λ` 在低频场景中易受先验主导（structural identifiability limit），非算法缺陷。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **SBI 可以逼近 MCMC 的诊断精度**：在故障识别和参数估计方面表现一致，且提供可靠的不确定性量化。
2. ⚡️ **推理速度提升两个数量级**：实现 **82× 加速**，支持实时、高频、多资产监控。
3. 🔁 **amortization 特性显著提升可扩展性**：一次性训练即可服务无限次推理，适合工业部署。
4. 🧠 **即使个别参数不可辨识，整体诊断仍稳健**：例如在稀疏结垢事件中，`λ` 和 `βf` 存在 trade-off，但故障模式仍能被准确识别。
5. 🔄 **方法具有模型无关性（model-agnostic）**：不依赖显式 likelihood，适用于黑箱模拟器或老旧系统。

### 局限性
1. **结构性可辨识性限制**：
   - 在稀疏事件（如 Batch SD）中，由于观测窗口内事件极少，`λ` 和 `βf` 存在强耦合，导致估计困难。
   - 此为模型本身限制，非 SBI 方法缺陷。
2. **依赖合成数据训练**：
   - 当前基于理想化随机过程建模，可能无法完全反映真实工业系统的复杂性（如传感器漂移、未建模扰动）。
3. **手工设计 summary statistics**：
   - 虽然高效，但可能丢失原始时间序列中的部分信息；未来可探索端到端学习表示。

### 未来工作方向
1. **真实数据验证**：在实际工业现场数据上测试泛化能力。
2. **在线自适应训练**：当系统行为发生偏移时，动态更新神经网络。
3. **联合轨迹推断**：尝试同时估计潜在退化路径 `z(t)` 而不仅是参数 `θ`。
4. **集成至数字孪生平台**：将 SBI 模块嵌入工业数字孪生系统，实现全流程风险感知决策。
5. **扩展至其他设备**：应用于泵、压缩机、反应器等多参数工业系统。

---

> 🔗 **代码开源地址**：[https://github.com/petercollett-cognite/sbi_mcmc_heat_exchanger.git](https://github.com/petercollett-cognite/sbi_mcmc_heat_exchanger.git)

</details>

---

### 7. [Decoupled DiLoCo for Resilient Distributed Pre-training](https://arxiv.org/abs/2604.21428)

**Authors**: Arthur Douillard, Keith Rush, Yani Donchev, Zachary Charles, Nova Fallen, Ayush Dubey, Ionel Gog, Josef Dean, Blake Woodworth, Zachary Garrett, Nate Keating, Jenny Bishop, Henry Prior, Edouard Yvinec, Arthur Szlam, Marc'Aurelio Ranzato, Jeff Dean  
**Category**: cs.CL  
**Published**: 2026-04-24  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.21428v1  

#### Abstract
Modern large-scale language model pre-training relies heavily on the single program multiple data (SPMD) paradigm, which requires tight coupling across accelerators. Due to this coupling, transient slowdowns, hardware failures, and synchronization overhead stall the entire computation, wasting signi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Decoupled DiLoCo for Resilient Distributed Pre-training*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大规模语言模型（LLM）预训练严重依赖**单程序多数据**（SPMD）范式，该范式要求所有加速器在每一步进行全局同步。这种紧耦合架构存在以下关键瓶颈：
- **可靠性差**：单个硬件故障或延迟（straggler）会导致整个训练停滞。
- **资源浪费**：故障检测、集群重配置等过程引入显著停机时间（downtime），造成大量计算资源浪费。
- **扩展性受限**：随着芯片数量增加，系统平均无故障时间（MTBF）急剧下降，使训练效率难以维持。

### 提出的新方法与新思路
作者提出 **Decoupled DiLoCo**，一种去耦合的分布式训练框架，旨在打破SPMD的锁步同步壁垒，实现高可用、高容错的预训练。

其核心思想是将“一致性”（Consistency）置于“可用性”（Availability）和“分区容忍性”（Partition Tolerance）之后，类比于分布式系统的CAP定理。

#### 主要创新点包括：
- **异步学习者架构**（Asynchronous Learners）：
  - 将全局集群分解为多个独立运行的“学习者”（Learners），每个学习者在本地数据分片上执行内层优化（如AdamW），无需等待其他节点。
- **中心化同步器**（Central Synchronizer）：
  - 引入一个轻量级的中央同步器（Syncer），负责异步接收来自各学习者的参数片段，并执行外层优化（Outer Optimization）。
- **最小法定数聚合**（Minimum Quorum Aggregation）：
  - 同步器不等待全部学习者，只需收到至少 `K` 个学习者的更新即可聚合，避免因个别节点故障而阻塞。
- **自适应宽限期窗口**（Adaptive Grace Window）：
  - 利用计算与通信之间的空闲时间（slack），动态延长等待时间以纳入更多学习者，提升样本效率而不牺牲吞吐。
- **基于令牌的加权合并**（Token-weighted Merging）：
  - 根据学习者处理的数据量（token count）和步数动态赋予权重，确保快速学习者的贡献更大。
- **径向方向平均法**（Radial-Directional Averaging, RDA）：
  - 改进的参数合并策略，分别对梯度的方向和模长取平均，增强超参稳定性并提升大M下的性能。

### 相比现有方法的优势
| 特性 | SPMD / Data-Parallel | Elastic DP | Decoupled DiLoCo |
|------|------------------------|------------|-------------------|
| 同步模式 | 全局同步 | 锁步同步（可弹性缩放） | 完全异步 |
| 故障容忍 | 差（需重启） | 中等（有停机） | 极强（零全局停机） |
| 通信带宽 | 高 | 降低峰值 | 显著降低（总量+峰值） |
| 可用性 | 低 | 中等 | 接近100% |
| 资源利用率 | 易受straggler影响 | 缩放时有开销 | 高且稳定 |

---

## 2. 核心实验方法和设置

### 数据集
- **文本任务**：
  - ARC-Challenge/Easy, BoolQ, HellaSwag, PIQA, SIQA, Winogrande
- **视觉任务**（多模态）：
  - COCO-Captions, ChartQA, DocVQA, InfographicVQA, MMMU, TextVQA
- **后训练评估**：
  - MMLU-Pro, GMMLU-lite, GSM8K, HumanEval

### 模型与训练设置
- **基础模型**：Gemma 4 系列（2B, 5B, 9B 参数）
- **架构类型**：Dense 和 Mixture-of-Experts (MoE)
- **训练规模**：
  - 最高达 **1T tokens**，最大模拟芯片数达 **2.4 million**
- **碎片划分**：`P=24` 个非重叠参数片段，每 `H=24` 步同步一次
- **通信重叠**：`T=2` 步
- **最小法定数**：默认 `K=1`
- **硬件模拟**：通过混沌工程（Chaos Engineering）注入硬件故障，模拟真实场景中的中断

### 评估指标
- **Goodput**：集群实际用于有效计算的时间比例（核心指标）
- **System Uptime**：系统处于“正在推进训练步”的时间占比
- **下游任务性能**：多项文本与视觉基准测试的准确率/得分
- **验证损失变化**（Validation Loss Delta）：衡量训练效率提升

### 基线方法对比
- **标准 Data-Parallel (DP)**：传统同步训练
- **Elastic Data-Parallel**：支持动态扩缩容的SPMD变体
- **原始 DiLoCo / Streaming DiLoCo**：前序工作，仍为同步框架

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在极端硬件故障下（MTBI_chip = 1年，N_chip = 1.2M）：
| 方法 | Goodput | 文本平均得分 | 视觉平均得分 |
|------|---------|--------------|--------------|
| Elastic DP | 58% | ~69.7 | ~59.4 |
| **Decoupled DiLoCo (M=8)** | **88%** | **69.8** | **58.6** |

> ✅ **Decoupled DiLoCo 在更恶劣条件下实现了高出约30个百分点的Goodput，同时保持了相当甚至略优的模型性能。**

#### 系统可用性（Uptime）表现：
当学习者数量 `M≥4` 时，在高达240万芯片的模拟中，系统**uptime可达100%**，即**从未发生全局停机**。

#### 不同学习者数量的影响（Goodput规则）：
> **一个 M 学习者系统 ≈ 单学习者系统使用 M 倍少的芯片所能达到的 Goodput。**
这表明 Decoupled DiLoCo 可有效“稀释”硬件故障密度带来的负面影响。

---

### 与基线方法的对比结果

#### 图像：跨芯片数的 Goodput 对比（见 Table 1）
| #Learners M | N_chip=2.4M 下 Goodput |
|-------------|------------------------|
| 1 (no elasticity) | 18% |
| 1 (elastic) | 40% |
| **8** | **80%** |
| **16** | **86%** |

✅ **Decoupled DiLoCo 将 Goodput 提升超过两倍以上。**

#### 图像：训练速度与带宽需求（见 Figure 13 & Table 13）
- 在相同计算利用率下，Decoupled DiLoCo 所需跨数据中心带宽比 Data-Parallel **低两个数量级**。
- 使用 `int4` 压缩通信后，进一步减少至 **1/1000**。

#### 多模态与MoE模型上的表现（见 Figure 7b）
- 在 MoE 架构（激活参数 2.8B/3.8B）上，Decoupled DiLoCo 性能与标准 DP **完全持平**。
- 尽管局部负载均衡无法全局优化，但在足够大的尺度下仍能收敛到高质量模型。

---

### 消融实验结果

#### （1）合并策略比较（RDA vs Direct Averaging）
| 合并方式 | 文本平均得分（M=8） | 文本平均得分（M=16） |
|----------|--------------------|---------------------|
| Direct Avg | 54.1 | 52.7 |
| **RDA（非嵌入层） + Avg（嵌入层）** | **55.4** | **54.5** |

✅ **RDA 显著提升了大M下的性能，尤其在 M=16 时优势明显。**

#### （2）外梯度压缩（bf16 → int4）
| 精度 | 相对验证损失增量 | 文本平均得分 |
|------|------------------|---------------|
| bf16 | +0.09% | 55.4 |
| **int4** | **+0.09%** | **55.3** |
| 2bit | +32% | 42.6 |
| 1bit | +83% | 37.3 |

✅ **int4 压缩几乎无损，而更低精度则导致严重退化。**

#### （3）碎片划分策略（见 Table 7）
| 策略 | 文本平均得分 | 峰值带宽 |
|------|--------------|-----------|
| Layer Fragmentation | 55.3 | 高（bursty） |
| Tensor Fragmentation | 55.4 | 中 |
| **Balanced Tensor Fragmentation** | **55.4** | **最低且最平稳** |

✅ **平衡张量碎片化**在保持性能的同时显著优化了通信效率。

#### （4）学习者数量 M 的影响（见 Table 11）
- `M=2` → Goodput 仅 70%
- `M=8` → Goodput 达 88%
- `M=16` → Goodput 达 93%，但模型性能略有下降（可通过增大token预算补偿）

✅ **增加 M 可大幅提升 Goodput，是提高容错性的关键杠杆。**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Decoupled DiLoCo 成功打破了SPMD的同步瓶颈**，实现了真正意义上的异步、高可用预训练。
2. ✅ **在极端硬件故障环境下（百万级芯片、频繁中断），系统可维持接近100% uptime 和 >88% goodput**，远优于现有弹性方案。
3. ✅ **模型质量不受损害**：在文本、视觉、MoE等多种架构和任务上，最终性能与标准 Data-Parallel **完全相当甚至更优**。
4. ✅ **天然支持资源 scavenging**：可无缝整合临时可用的异构计算资源（如不同代次TPU），无需停机。
5. ✅ **通信效率极高**：相比传统方法，所需带宽降低两个数量级以上，适用于地理分布训练。

### 方法的局限性
- **算法复杂度上升**：需要设计新的同步机制、恢复协议和一致性快照算法（如Chandy-Lamport）。
- **调试难度增加**：由于异步性和非确定性，复现特定训练轨迹需依赖事件日志（event tape）回放。
- **小规模收益有限**：在小集群或稳定环境中，其优势不如在超大规模、高故障率场景中显著。
- **对超参调优敏感**：如 `K`, `H`, `grace window` 等需根据硬件环境调整。

### 未来工作方向
- **扩展至联邦学习与边缘训练**：利用其高容错特性，在不稳定网络中训练LLM。
- **结合更先进的压缩技术**：如 MuLoCo、TIES-Merging 等，进一步降低通信成本。
- **自动化弹性调度**：构建智能控制器，根据实时故障率动态调整 `M` 和 `K`。
- **探索完全去中心化版本**：移除中心同步器，实现完全对等（peer-to-peer）的训练架构。

---

## 总结
**Decoupled DiLoCo 是迈向下一代分布式训练的重要一步**。它不再追求“完美一致”，而是拥抱“混沌现实”，通过**解耦学习者、异步通信、最小法定数聚合与智能合并策略**，在保证模型性能的前提下，极大提升了训练系统的**韧性、效率与灵活性**。该工作不仅具有工程实践价值，也为未来在**地理分散、异构、不可靠硬件上训练超大规模模型**提供了理论和技术基础。

</details>

---

### 8. [F\textsuperscript{2}LP-AP: Fast \& Flexible Label Propagation with Adaptive Propagation Kernel](https://arxiv.org/abs/2604.20736)

**Authors**: Yutong Shen, Ruizhe Xia, Jingyi Liu, Yinqi Liu  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.20736v1  

#### Abstract
Semi-supervised node classification is a foundational task in graph machine learning, yet state-of-the-art Graph Neural Networks (GNNs) are hindered by significant computational overhead and reliance on strong homophily assumptions. Traditional GNNs require expensive iterative training and multi-lay...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# F²LP-AP: Fast & Flexible Label Propagation with Adaptive Propagation Kernel 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Graph Neural Networks (GNNs)** 在半监督节点分类任务中面临两大瓶颈：
- **计算开销大**：依赖梯度反向传播进行迭代训练，对大规模图耗时且占用大量 GPU 资源。
- **强同质性假设（Homophily Assumption）**：传统 GNN 假设相连节点标签相似，在异质图（Heterophilous Graphs）上性能显著下降。

同时，现有的 **训练免费方法（Training-free Methods）** 如 Label Propagation (LP) 和 APPNP 虽然避免了训练过程，但通常采用固定的传播参数，缺乏对局部拓扑结构的适应能力，导致在复杂图中表现不佳。

---

### 🚀 提出的新方法与核心思想
本文提出 **F²LP-AP**（Fast & Flexible Label Propagation with Adaptive Propagation Kernel），一种无需训练、高效灵活的标签传播框架，其核心创新如下：

#### （1）**基于 Local Clustering Coefficient (LCC) 的自适应传播机制**
- 利用每个节点的 **LCC**（局部聚类系数）作为拓扑感知指标，动态调整两个关键传播参数：
  - **传播深度 $K_u$**：高 LCC（密集社区）→ 浅层传播；低 LCC（稀疏边界）→ 更深传播以获取上下文。
  - **跳跃概率 $\alpha_u$**：高同质区域降低 $\alpha$ 以增强平滑；高异质风险区提高 $\alpha$ 保留原始特征锚点。
- 映射函数为预定义启发式规则，无需参数学习，实现 **node-wise 自适应**。

> 公式形式：  
> $$
> \alpha_u = f_\alpha(\text{LCC}_u), \quad K_u = \text{round}(g_K(\text{LCC}_u))
> $$

#### （2）**基于 Geometric Median 的鲁棒原型构建**
- 不再使用易受噪声影响的 **arithmetic mean** 构建类别原型，而是采用 **Geometric Median**（几何中位数），具有高达 50% 的崩溃点（breakdown point），能有效抵抗异常值干扰。
- 使用 Weiszfeld 算法求解，仅需 3–5 次迭代即可收敛，计算成本极低。

#### （3）**完全解析式的分类流程（Analytical Inference）**
- 最终预测通过计算节点表示与各类别原型之间的 **cosine similarity** 得到：
  $$
  y_u = \arg\max_c \frac{\mathbf{x}_u^\top \mathbf{P}_c}{\|\mathbf{x}_u\| \cdot \|\mathbf{P}_c\|}
  $$
- 整个推理流程无任何可学习参数，不涉及 Softmax 或梯度更新，真正实现 **training-free**。

---

### 🔍 相比现有方法的优势
| 维度 | F²LP-AP | 传统 GNNs | 经典 LP / APPNP |
|------|--------|----------|----------------|
| 是否需要训练 | ❌ No | ✅ Yes | ❌ No |
| 参数是否全局固定 | ❌ Node-wise 自适应 | ✅ 全局统一 | ✅ 固定参数 |
| 对异质图适应性 | ✅ 强（LCC 驱动） | ❌ 差（依赖同质假设） | ❌ 弱 |
| 抗噪能力 | ✅ 强（Geometric Median） | ⚠️ 中等 | ❌ 弱（均值敏感） |
| 推理效率 | ✅ 极高（毫秒级） | ❌ 高延迟（含训练） | ✅ 快但精度低 |

---

## 2. 核心实验方法和设置

### 📚 数据集
共选用 **8 个基准图数据集**，按同质性分为两类：
- **强同质图（Homophilous）**：
  - Cora ($H=0.85$), CiteSeer ($H=0.81$), PubMed ($H=0.84$)
- **异质图（Heterophilous）**：
  - Texas ($H=0.31$), Wisconsin ($H=0.37$), Cornell ($H=0.34$), Chameleon ($H=0.26$), Squirrel ($H=0.23$)

涵盖学术引用网络与网页链接图，规模从数百到上万节点。

---

### 🧪 实验设置与评估指标
- **任务**：半监督节点分类（Transductive Setting）
- **划分比例**：标准 60/20/20（Train/Val/Test）
- **评估指标**：
  - **Accuracy (Acc.)**
  - **Macro-F1 Score**
  - **Execution Time (s)** —— 衡量推理效率
- **硬件环境统一**，确保公平比较
- **随机种子固定为 0**，保证可复现性
- **F²LP-AP 参数范围**：
  - $K \in [2, 15]$, $\alpha \in [0.05, 0.2]$，由 LCC 动态决定

---

### 🔁 基线方法对比
分为三类进行比较：

#### （1）**代表性模型**
- **GCN***：经典 GNN，有监督训练
- **LabelProp**：经典标签传播
- **kNN@5**：基于特征最近邻分类
- **CoHOp**：SOTA 的训练免费方法

#### （2）**消融变体（Ablation Baselines）**
- **PrototypeOnly-Mean**：仅用均值原型 + 无传播
- **PrototypeOnly-GeoMed**：仅用几何中位数原型 + 无传播
- **FixedAPPNP-Proto**：固定参数 APPNP + 几何中位数原型

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Dataset | Method | Acc. | F1 | Time (s) |
|--------|--------|------|-----|---------|
| **Cora** | GCN* | 0.821 | 0.809 | 1.350 |
| | **F²LP-AP** | **0.835** | **0.821** | **0.056** |
| **CiteSeer** | GCN* | 0.719 | 0.691 | 1.210 |
| | **F²LP-AP** | **0.708** | **0.685** | **0.092** |
| **PubMed** | GCN* | 0.798 | 0.794 | 1.485 |
| | **F²LP-AP** | **0.782** | **0.779** | **0.044** |
| **Texas** | GCN* | 0.553 | 0.365 | 1.015 |
| | **F²LP-AP** | **0.842** | **0.787** | **0.016** |
| **Wisconsin** | GCN* | 0.608 | 0.269 | 1.053 |
| | **F²LP-AP** | **0.825** | **0.589** | **0.024** |
| **Cornell** | GCN* | 0.500 | 0.203 | 1.014 |
| | **F²LP-AP** | **0.763** | **0.519** | **0.018** |

> ✅ **F²LP-AP 在所有异质图上达到 SOTA 性能**，远超 GCN  
> ✅ 在 Cora 上超越 GCN，成为整体最优  
> ✅ 推理时间仅为 GCN 的 ~3%~10%，速度提升一个数量级以上

---

### 🔍 消融实验结果（Ablation Study）

#### （1）**自适应传播的有效性**
- 在 Cora 上，F²LP-AP 比 FixedAPPNP 提升 **26.9%**（0.835 vs 0.658）
- 在 Wisconsin 上比 FixedAPPNP 提升 **13.8%**
- 表明 **动态参数调整显著提升性能**

#### （2）**Geometric Median 的优势**
- 在全部 8 个数据集上，**GeoMedian 均优于或等于 Mean**
- 平均提升 **5.8% Accuracy**
- 在低同质图如 Chameleon (+12.8%) 和 Squirrel (+14.3%) 提升更明显
- 验证其在噪声和异质环境下的鲁棒性

#### （3）**效率分析**
- 尽管引入动态参数，F²LP-AP 执行时间仍控制在 **0.016–0.089 秒之间**
- 仅比 FixedAPPNP 略慢，远快于 GCN
- 实现了“高性能 + 高效率”的理想平衡

---

### 🖼️ 可视化证据
- **t-SNE 图**（Figure 4）显示：经 F²LP-AP 处理后的节点表示在 Cora 和 Texas 上均呈现更清晰的簇结构，类内更紧凑，类间分离更好。
- **混淆矩阵**（Figure 2）显示：主对角线明亮集中，说明分类准确率高，尤其在 Texas 上表现稳定。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **F²LP-AP 是首个将 LCC 用于驱动自适应传播参数的训练免费方法**，实现了对同质与异质图的统一建模。
2. **无需训练即可达到甚至超越有监督 GNN 的性能**，特别是在异质图上大幅领先。
3. **Geometric Median 显著提升了原型鲁棒性**，是应对标签噪声的关键设计。
4. **推理速度快、资源消耗低**，适用于实时或边缘部署场景。
5. **结构感知的 adaptive propagation 比全局固定策略更具表达力和泛化能力**。

---

### ⚠️ 方法的局限性
1. **依赖单一拓扑指标 LCC**：在极端稀疏或高度噪声的图中可能失效，无法全面刻画复杂结构。
2. **映射函数为启发式设计**：非数据驱动，可能存在次优配置。
3. **性能上限受限于原始特征质量**：若输入特征本身判别性弱，则难以弥补。
4. **未考虑边属性或多关系结构**：适用于简单同构图。

---

### 🔮 未来工作方向
- 探索 **多维结构描述符**（multi-dimensional structural descriptors）替代单一 LCC。
- 引入轻量级学习机制（如 meta-learning）优化映射函数 $f_\alpha, g_K$。
- 扩展至 **inductive setting** 和 **动态图** 场景。
- 结合自监督预训练特征进一步提升性能边界。

---

## ✅ 总结
**F²LP-AP** 是一项在 **训练免费图学习** 方向上的重要进展。它通过 **LCC 驱动的自适应传播核** 与 **几何中位数原型构建**，实现了在无需任何参数学习的前提下，兼具 **高精度、强鲁棒性和极致效率** 的节点分类能力。实验充分验证了其在多样图结构下的优越表现，为未来高效、通用的图学习系统提供了新范式。

</details>

---

### 9. [Trust but Verify: Introducing DAVinCI -- A Framework for Dual Attribution and Verification in Claim Inference for Language Models](https://arxiv.org/abs/2604.21193)

**Authors**: Vipula Rawte, Ryan Rossi, Franck Dernoncourt, Nedim Lipka  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.21193v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable fluency and versatility across a wide range of NLP tasks, yet they remain prone to factual inaccuracies and hallucinations. This limitation poses significant risks in high-stakes domains such as healthcare, law, and scientific communication, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Trust but Verify: Introducing DAVinCI – A Framework for Dual Attribution and Verification in Claim Inference for Language Models*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）虽然在自然语言生成任务中表现出色，但其输出常包含**事实错误**（factual inaccuracies）和**幻觉**（hallucinations），这在医疗、法律、科学等高风险领域尤为危险。当前大多数系统将**归因**（attribution）与**验证**（verification）视为独立模块，缺乏协同机制，导致可解释性和可信度不足。

### 提出的新方法：DAVinCI 框架
作者提出 **DAVinCI**（Dual Attribution and Verification in Claim Inference），一个集成化的双阶段框架，旨在提升 LLM 输出的事实可靠性与可审计性。

#### 核心设计：
1. **Attribution（归因）阶段**  
   - 将生成的声明（claim）链接到**内部模型组件**或**外部证据源**。
   - 支持两种模式：
     - **Full Evidence Attribution**：使用完整证据文本。
     - **Span-Based Attribution**：通过 QA 模型提取最相关的片段（如 `roberta-base-squad2`）。

2. **Verification（验证）阶段**  
   - 利用基于 **entailment 的分类器**（如 DeBERTa、RoBERTa）判断声明与证据之间的关系：`Supported` / `Refuted` / `Not Enough Information (NEI)`。
   - 引入**置信度重校准**（confidence recalibration）机制：若预测置信度低于阈值 $T$（默认 0.6），则强制降级为 `NEI`，以避免过度自信的误判。

### 相比现有方法的优势
| 维度 | DAVinCI 的优势 |
|------|----------------|
| **架构整合性** | 首次将 internal/external attribution 与 entailment verification 在统一 pipeline 中结合，形成闭环可解释流程。 |
| **可信决策** | 通过 confidence recalibration 实现保守推理，在模糊情况下主动承认“信息不足”，更贴近人类 fact-checker 行为。 |
| **模块化设计** | 可灵活接入不同 retriever、verifier 和 calibrator，便于集成至现有 LLM 系统。 |
| **性能提升显著** | 在多个数据集上相比 baseline 提升 **5–20%** 的 accuracy、F1、precision 和 recall。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **FEVER** | 来自维基百科的声明验证数据集，标签为 `Entailment`, `Contradiction`, `Neutral`。用于通用事实核查任务。 |
| **CLIMATE-FEVER** | 聚焦气候变化领域的声明验证数据集，更具领域挑战性，强调 domain-specific reasoning。 |

> ✅ 两个数据集均提供黄金标准证据（gold-standard evidence），支持 full-span 与 span-based 归因评估。

### 实验设置
- **硬件环境**：Apple MacBook M4 芯片，32GB RAM。
- **工具库**：Hugging Face Transformers。
- **许可选择**：优先采用 MIT 许可的模型与数据集，确保可复现性。
- **输入格式**：`[Claim][SEP][Evidence]`
- **输出**：三分类标签 + confidence score。

### 评估指标
- **Accuracy**
- **Precision**（macro / weighted）
- **Recall**（macro / weighted）
- **F1-Score**（macro / weighted）
- 多证据聚合策略：majority voting 或 weighted averaging。

### 基线方法对比
| 基线模型 | Hugging Face ID |
|---------|------------------|
| Baseline (Verifier-only) | 使用 full evidence 输入的标准 NLI 分类器 |
| microsoft/deberta-large-mnli | 当前最强 NLI 模型之一 |
| FacebookAI/roberta-large-mnli | RoBERTa 架构代表 |
| facebook/bart-large-mnli | 序列到序列风格的 NLI 模型 |
| ynie/roberta-large-snlimnli_fever_anli_R1_R2_R3-nli | 多任务训练增强版 |

> ⚖️ 对比设置：
> - **Baseline**: Verifier-only + full evidence
> - **DAVinCI-Recalibrated**: QA span + verification + recalibration ($T=0.6$)

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 2–7）

#### ✅ 在 FEVER 数据集上的表现（Table 2）
| Model | Baseline Acc | DAVinCI Acc | Δ Acc | Macro F1 (Baseline → DAVinCI) |
|--------|---------------|--------------|--------|-------------------------------|
| deberta-large-mnli | 0.42 | **0.48** | +6% | 0.36 → **0.41** |
| roberta-large-mnli | 0.36 | **0.44** | +8% | 0.30 → **0.38** |
| bart-large-mnli | 0.42 | **0.43** | +1% | 0.36 → **0.37** |
| roberta-large-snli | 0.38 | **0.42** | +4% | 0.34 → **0.40** |

> 🔺 所有模型均受益于 DAVinCI 框架，平均提升约 **5–20%** 的各项指标。

#### ✅ 在 CLIMATE-FEVER 上的表现（Table 3）
| Model | Baseline Acc | DAVinCI Acc | Δ Acc | Weighted F1 提升 |
|--------|---------------|--------------|--------|------------------|
| deberta-large-mnli | 0.60 | **0.63** | +3% | 0.51 → **0.55** |
| roberta-large-snli | 0.65 | **0.66** | +1% | 0.57 → **0.56**（略有下降但更稳健） |

> 📈 表明 DAVinCI 在专业领域同样有效，尤其在高准确率基础上仍能维持稳定输出。

### 与基线方法的对比结果
- DAVinCI 在所有四个主流 NLI 模型上均优于纯验证 baseline。
- 最佳表现由 `deberta-large-mnli` 实现：
  - FEVER 上达到 **0.62 weighted precision** 和 **0.48 accuracy**。
- 即使 baseline 性能较低的模型（如 `roberta-large-mnli`），也能通过 DAVinCI 显著提升鲁棒性。

### 消融实验结果（Ablation Study）

#### （1）Full Evidence vs. Span-Based Attribution（Tables 4 & 5）
| 设置 | FEVER Acc | CLIMATE-FEVER Acc | 结论 |
|------|------------|--------------------|------|
| Full Evidence | 最高达 **0.48** | 最高达 **0.65** | ✅ 明显优于 span-based |
| Span-Based | 最高仅 **0.39** | 最高 **0.64**（但 F1 下降明显） | ❌ 丢失上下文导致验证质量下降 |

> 🔍 发现：`roberta-large-snli` 在 full evidence 下 F1 从 0.19 提升至 0.48（+152%！），说明**完整语境对验证至关重要**。

#### （2）Threshold Tuning 效果（Tables 6 & 7）
| Threshold $T$ | 准确率趋势 | Precision | Recall | 推荐场景 |
|----------------|-------------|-----------|--------|----------|
| 0.7 | ✅ 最高（FEVER: 0.47, CF: 0.64） | 高 | 较高 | ✅ 平衡精度与覆盖率，推荐默认值 |
| 0.8 | 略降 | 更高 | 下降 | 适用于高 precision 需求 |
| 0.9 | 进一步下降 | 极高 | 显著下降 | ❌ 过于保守，牺牲 recall |

> 🎯 结论：**$T=0.7$ 是最优折衷点**，可在控制 false positive 同时保持良好 recall。

---

## 4. 关键结论和发现

### 主要发现
1. **归因质量决定验证上限**：full evidence attribution 显著优于 span-based 方法（+9–18%），证明**保留完整上下文是关键**。
2. **DAVinCI 显著提升性能**：相比 baseline，accuracy、F1、precision、recall 全面提升 **5–20%**。
3. **置信度校准有效抑制过拟合误判**：threshold 机制可在不确定时主动退让至 `NEI`，提高系统可信度。
4. **模块化设计支持灵活配置**：可适配多种 retriever、verifier 和 threshold 策略，适用于不同风险偏好场景。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| （i）依赖高质量证据 | 开放域检索中可能无法获取可靠来源，影响归因效果。 |
| （ii）静态验证模型限制 | 当前 verifier 难以处理 multi-hop 或复杂逻辑推理。 |
| （iii）未实现 internal attribution | 尚未追踪声明是否源自训练数据或 prompt，缺乏深层溯源能力。 |
| （iv）仅限英文数据集 | 在 multilingual 或低资源语言中尚未验证。 |
| （v）手动设定阈值 | recalibration threshold 需人工调参，泛化能力受限。 |

### 未来工作方向
1. **扩展至开放域检索**（open-domain retrieval）与多跳推理（multi-hop reasoning）。
2. **引入 internal attribution 技术**：如 prompt tracing、activation clustering，增强模型内部可解释性。
3. **应用于生成任务**：让 LLM 在生成文本时同步输出 source-aware 和 verifiable 内容。
4. **支持多语言与低资源环境**：推动全球化可信 NLP 发展。
5. **开发 adaptive calibration 策略**：用学习方式替代固定 threshold，提升跨任务适应性。
6. **人机协作评估**：开展 human-in-the-loop 实验，衡量真实应用场景下的 trustworthiness。

---

> 💡 **总体评价**：  
> DAVinCI 不仅是一个技术框架，更是迈向**可审计、可信赖 AI 系统**的重要一步。它倡导“**Trust but Verify**”理念——允许信任，但必须验证。代码已开源：[GitHub - vr25/davinci](https://github.com/vr25/davinci)，为后续研究提供了坚实基础。

</details>

---

### 10. [Optimizing High-Throughput Distributed Data Pipelines for Reproducible Deep Learning at Scale](https://arxiv.org/abs/2604.21275)

**Authors**: Kashish Mittal, Di Yu, Roozbeh Ketabi, Arushi Arora, Brendon Lapp, Peng Zhang  
**Category**: cs.DC  
**Published**: 2026-04-24  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.21275v1  

#### Abstract
Training massive-scale deep learning models on datasets spanning tens of terabytes presents critical challenges in hardware utilization and training reproducibility. In this paper, we identify and resolve profound data-loading bottlenecks within distributed GPU training pipelines using the Petastorm...

---

### 11. [Continuous Semantic Caching for Low-Cost LLM Serving](https://arxiv.org/abs/2604.20021)

**Authors**: Baran Atalar, Xutong Liu, Jinhang Zuo, Siwei Wang, Wei Chen, Carlee Joe-Wong  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.20021v1  

#### Abstract
As Large Language Models (LLMs) become increasingly popular, caching responses so that they can be reused by users with semantically similar queries has become a vital strategy for reducing inference costs and latency. Existing caching frameworks have proposed to decide which query responses to cach...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Continuous Semantic Caching for Low-Cost LLM Serving

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Semantic Caching** 方法大多假设查询空间是离散且有限的，即所有可能的查询可以被枚举为一个固定集合。然而，在现实世界中，LLM 用户的查询是无限且连续分布在高维语义嵌入空间中的（如通过 Sentence-BERT 得到的 embedding）。这种连续性和不确定性使得传统基于离散优化的方法难以扩展。

本文首次提出了在**连续语义查询空间**下进行语义缓存的理论框架，解决了以下挑战：
- 如何在无限查询空间中估计未知的查询到达概率和服务成本；
- 如何设计可扩展的缓存策略以应对连续输入；
- 如何在在线学习过程中平衡探索（query LLM 获取反馈）与利用（serve from cache），同时控制缓存切换开销（switching cost）。

---

### 提出的新方法与思路

#### （1）连续语义缓存模型（Continuous Semantic Caching Model）
- 引入 **ε-net discretization** 将连续查询空间划分为有限个 Voronoi 区域，每个区域用一个代表点表示。
- 在此基础上将原问题转化为有限候选集上的组合优化问题，但仍允许动态构建 ε-net。

#### （2）结合 Kernel Ridge Regression (KRR) 的学习机制
- 利用 **KRR** 对未见过的查询的服务成本 $ c(x) $ 进行泛化估计，实现跨相似查询的成本迁移。
- 引入**悲观估计（pessimistic cost estimation）** 和 **乐观探索（optimistic exploration）** 机制来处理不确定性。

#### （3）三种场景下的算法设计
| 场景 | 算法名称 | 特点 |
|------|--------|------|
| Oracle（已知参数） | Reverse Greedy | 给出近似最优解的基准 |
| Offline（有历史数据） | **CUCB-SC-Cont** | 基于 KRR 学习成本，使用 pessimistic 成本运行 Reverse Greedy |
| Online（流式到来） | **CLCB-SC-LS-Cont** | 动态构建 ε-net，阶段式低频更新缓存，使用 LCB 探索 |

#### （4）理论保障
- 首次证明了在连续空间下语义缓存问题的 **sublinear regret bound**，即：
  $$
  \text{Reg}(T) = O(\sqrt{T})
  $$
- 分析了误差来源：arrival estimation error、cost estimation error、discretization error，并给出了联合界。
- 证明该框架可退化为已有离散方法的结果，说明其为严格推广。

---

### 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **建模能力更强** | 支持无限、连续的查询空间，更贴近真实 LLM 使用场景 |
| **泛化能力强** | 通过 KRR 实现对未见查询的成本预测，避免“零样本”失效 |
| **理论保证完整** | 提供 suboptimality gap 与 regret bound，支持收敛性分析 |
| **系统实用性高** | 显式考虑 switching cost，减少频繁缓存更新带来的开销 |
| **灵活性好** | 支持动态 ε-net 构建，适应新出现的语义簇 |

---

## 2. 核心实验方法和设置

### 数据集
使用两类数据集进行验证：

1. **Synthetic Dataset**
   - 基于 50 个原始自然语言 prompt 构造。
   - 每次采样后添加 Gaussian noise 到其 embedding 上，模拟连续扰动。
   - 查询总数：约 2000–3000 条。
   - Embedding 维度：384（使用 `sentence-transformers` 模型生成）。

2. **Real-world Combined Dataset: NQ + TriviaQA**
   - 各取 2500 条来自 [Natural Questions](https://ai.google.com/research/NaturalQuestions) 和 [TriviaQA](https://nlp.cs.washington.edu/triviaqa/)。
   - 查询直接来自真实用户提问，更具多样性。
   - 流模式采用 bursty 非均匀分布：交替从两个数据集中抽样，且服从 log-normal 流行度分布（长尾特性）。

---

### 实验设置

| 设置项 | 描述 |
|-------|------|
| **Embedding 模型** | Sentence-BERT (`all-MiniLM-L6-v2`)，输出 384 维向量 |
| **距离度量** | 归一化后的 Euclidean distance |
| **服务成本 $ c(x) $** | 使用 GPT-2 tokenizer 的 token 长度并归一化至 [0,1] |
| **不匹配代价函数 $ \phi(d) $** | $ \phi(d) = \min(sd, 1) $，其中 $ s $ 是缩放因子 |
| **缓存大小 $ k $** | 默认为 5，在部分 ablation 中变化 |
| **评估周期 $ T $** | 最多 300 轮（online），offline 使用 1000 条记录训练 |

---

### 评估指标

| 指标 | 定义 | 用途 |
|-----|------|------|
| **Suboptimality Gap** | $ L(M) - \alpha \cdot L(M^*) $，衡量离 α-近似最优解的距离 | Offline 性能 |
| **Average Regret** | 累积损失相对于最优缓存的差距，归一化后平均 | Online 性能 |
| **Runtime / Latency** | 单次决策或整轮运行时间 | 效率评估 |
| **Switching Cost** | 缓存更新次数 × 更新代价 | 衡量系统稳定性 |

---

### 基线方法对比

| 基线方法 | 简介 |
|--------|------|
| **CUCB-SC** [19] | 离散语义缓存经典方法，基于经验均值估计成本 |
| **Greedy** | 选择历史频率×成本最高的查询缓存 |
| **Epsilon-Greedy** | 以一定概率探索 LLM 查询 |
| **LFU** | Least Frequently Used，仅依据访问频率排序 |
| **CLCB-SC-LS** [19] | 离散版本的在线自适应算法 |
| **CLCB-Frozen-Cont**（文中补充） | 固定候选池的变体，用于对比动态构建效果 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 场景 | 方法 | 性能提升（vs 最佳基线） | 具体数值 |
|------|------|--------------------------|---------|
| **Offline** | CUCB-SC-Cont | ↓ **73.32%** suboptimality gap | 当 stream length=100 时 |
| **Online (synthetic)** | CLCB-SC-LS-Cont | ↓ **72.41%** regret | 最终 avg regret = 0.0083 |
| **Online (real-world)** | CLCB-SC-LS-Cont | ↓ **43.14%** regret | regret = 0.0061，runtime ≈ 250s |

> 注：以上均为图中读取的最佳结果。

---

### 与基线方法的对比结果

#### （1）Offline Setting 结果（Fig 2a-b）
- **CUCB-SC-Cont** 在不同 ε 下表现稳定，在中等 ε（≈0.4）达到最佳。
- 当数据流较短时（如 100 条），相比 CUCB-SC 提升达 **73.32%**，显示其强泛化能力。
- 随着 stream length 增加，其他方法缓慢改善，而 CUCB-SC-Cont 始终领先。

#### （2）Online Setting 结果（Fig 2c-f）
- **CLCB-SC-LS-Cont** 在 synthetic 和 real-world 数据上均取得最低 regret。
- 在 synthetic 数据中，比 CLCB-SC-LS（离散版）降低 **72.41%** regret。
- 在 bursty real-world stream 中仍保持 **43.14%** 的 regret 降低。
- 虽然 runtime 略高（↑47.8%），但在实际部署中仍具竞争力。

#### （3）效率权衡（Efficiency Trade-off）
- 图 (d)(f) 显示：尽管 CLCB-SC-LS-Cont 运行稍慢，但每单位 runtime 带来的 regret 下降显著更高。
- 表明其“计算换质量”的策略有效。

---

### 消融实验结果（Ablation Study）

#### （1）ε-net 半径 $ \epsilon $ 的影响（Fig 2b）
- 太小的 $ \epsilon $ 导致 discretization 过细，样本稀疏，估计不准；
- 太大的 $ \epsilon $ 导致语义差异过大，引入高 mismatch cost；
- 存在一个**最优 $ \epsilon^* $**，论文进一步推导出其与维度、样本数的关系：
  $$
  \epsilon^* = 
  \begin{cases}
  O(n^{-1/(de + 2)}) & \text{if } p \geq de - 2 \\
  O(n^{-2/(2de + p)}) & \text{otherwise}
  \end{cases}
  $$

#### （2）缓存大小 $ k $ 的影响（Fig 5-6）
- 所有方法随 $ k $ 增大性能提升，但边际收益递减。
- CUCB-SC-Cont 的 suboptimality gap 增长最慢，表明其能更高效地利用缓存空间。

#### （3）成本预测误差演化（Fig 12）
- RMSE、MAE 快速下降并在约 1000 轮后趋于平稳。
- 表明 KRR 能快速捕捉成本函数结构，适合在线学习。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **连续语义缓存是可行且必要的**  
   真实 LLM 查询天然聚类成多个语义簇（Fig 3-4 UMAP/PCA 可视化证实），支持 ε-net 划分的有效性。

2. ✅ **KRR + ε-net 是处理连续空间的有效组合**  
   能够在未知成本的情况下泛化到新查询，显著优于独立估计（如 empirical mean）。

3. ✅ **动态 ε-net 构建优于静态池**  
   CLCB-SC-LS-Cont 能自动适应新语义区域，而冻结池方法（如 CLCB-Frozen-Cont）在复杂流中表现受限。

4. ✅ **理论指导实践：存在最优 discretization 粒度 $ \epsilon^* $**  
   并非越精细越好，需平衡估计误差与离散化误差。

5. ✅ **低频缓存更新是实用的关键**  
   设计 stage-based switching 机制可在不牺牲性能的前提下大幅减少 switching cost（$ O(\log \log T) $）。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖聚类假设（Assumption 2）** | 若查询完全均匀分布于高维空间，则 ε-net 规模爆炸，性能下降。但现实中查询高度集中，此假设合理。 |
| **KRR 计算复杂度较高** | 在线阶段需维护 Gram 矩阵，时间复杂度 $ O(t^2) $，可通过 Nyström 或随机特征近似缓解。 |
| **当前仅支持单跳缓存** | 不支持 multi-turn context 或 chain-of-thought 类型的响应复用。 |
| **未建模用户个性化偏好** | 假设所有用户共享同一缓存，未区分 user-specific 内容。 |

---

### 未来工作方向

1. 🔮 **建立语义缓存问题的信息论下界（lower bound）**
   - 当前只有上界（regret bound），尚无 minimax 下界，无法判断是否已达最优。

2. 🔄 **研究更高效的 kernel approximation 方法**
   - 如使用 Random Fourier Features 或 inducing points 加速 KRR。

3. 🌐 **扩展至 hierarchical / tiered caching 架构**
   - 结合边缘节点本地缓存与中心化语义缓存。

4. 🧠 **融合 context-aware 与 multi-turn 语义匹配**
   - 支持对话历史感知的缓存查找。

5. 📊 **更真实的 query arrival modeling**
   - 建立基于 real-world trace 的 generative model 来模拟动态查询流。

--- 

> **总结一句话**：  
> 本文首次建立了**连续语义空间下 LLM 响应缓存的理论与算法体系**，通过 **ε-net + KRR + 动态低频更新机制**，实现了在无限查询流中的高效、低成本服务，在真实与合成数据上均显著优于现有方法。

</details>

---

### 12. [Agentic AI for Personalized Physiotherapy: A Multi-Agent Framework for Generative Video Training and Real-Time Pose Correction](https://arxiv.org/abs/2604.21154)

**Authors**: Abhishek Dharmaratnakar, Srivaths Ranganathan, Anushree Sinha, Debanshu Das  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.21154v1  

#### Abstract
At-home physiotherapy compliance remains critically low due to a lack of personalized supervision and dynamic feedback. Existing digital health solutions rely on static, pre-recorded video libraries or generic 3D avatars that fail to account for a patient's specific injury limitations or home enviro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Agentic AI for Personalized Physiotherapy: A Multi-Agent Framework for Generative Video Training and Real-Time Pose Correction*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统居家物理治疗（at-home physiotherapy）存在**依从性低**和**动作执行错误率高**的问题。现有数字健康平台多依赖静态视频教程或通用3D虚拟形象，缺乏个性化反馈机制，无法适应患者的特定伤情限制和家庭环境。

### 🚀 提出的新方法与创新思路
本文提出了一种基于 **Multi-Agent System (MAS)** 的新型架构，构建了一个闭环的智能远程康复系统。该框架由四个专业化微代理（micro-agents）协同工作：

1. **Clinical Extraction Agent**  
   - 从非结构化临床笔记中提取运动学约束（如最大关节角度、速度限制），转化为标准化JSON格式。
2. **Video Synthesis Agent**  
   - 利用生成式AI模型（如diffusion models）生成符合患者个体限制的“physio-twin”训练视频（即定制化虚拟治疗师演示）。
3. **Vision Processing Agent**  
   - 基于MediaPipe等轻量级姿态估计算法，在边缘设备上实现实时人体关键点检测与关节角计算（≥30 FPS）。
4. **Diagnostic Feedback Agent**  
   - 结合LLM与确定性规则，提供安全、可解释的实时纠正指令，并链接回原始医嘱。

整个系统形成一个**从医嘱解析 → 视频生成 → 动作执行 → 实时纠错**的端到端闭环。

### 🔍 相比现有方法的优势
| 维度 | 传统方案 | 本论文方法 |
|------|--------|-----------|
| 个性化程度 | 静态/通用视频 | 患者专属“physio-twin”，按伤情定制动作范围 |
| 反馈时效性 | 异步/无反馈 | 实时视觉追踪 + 即时语音/文本提示 |
| 安全性保障 | 缺乏硬性限制 | 显式kinematic constraints防止过度拉伸 |
| 可解释性 | 黑箱决策 | XAI集成：所有反馈均可追溯至医生原始处方 |
| 部署隐私性 | 云端处理风险 | Vision Agent本地运行，保护敏感影像数据 |

> ✅ **核心创新**：首次将**Agentic AI范式**引入个性化物理治疗领域，实现语义理解、内容生成、感知控制与诊断推理的动态协作。

---

## 2. 核心实验方法和设置

### 📦 数据集
- **未使用公开数据集**，而是基于模拟临床场景进行原型验证。
- 输入为人工构造的**非结构化临床笔记**（如：“肩袖撕裂术后，肩关节外展不超过90°”）。
- 视频合成与姿态估计部分采用**真实摄像头输入**（RGB帧流）进行实时测试。

### ⚙️ 实验设置
- 开发语言：Python
- 架构层级：
  - **云层**：Clinical Extraction Agent 和 Video Synthesis Agent 运行在云端。
  - **边缘层**：Vision Processing Agent 和 Diagnostic Feedback Agent 在终端设备（如平板/手机）本地运行。
- 共享状态对象 `PatientState` 实现各Agent间状态同步。

### 🎯 评估指标
| 模块 | 评估指标 |
|------|---------|
| Clinical Extraction Agent | 文本解析准确率（Clinical Text Parsing Accuracy） |
| Video Synthesis Agent | 视频生成时间（Generation Time） |
| Vision Processing Agent | 姿态估计延迟（Pose Estimation Latency）、关节角度误差（Joint Angle Error Margin） |
| 整体系统 | 是否满足实时交互需求（<50ms延迟）、安全性与可解释性 |

### 🔁 基线方法对比
尽管尚未开展大规模临床试验，文中隐含对比了以下基线：
- **传统Tele-rehab平台**：仅提供预录视频，无实时反馈。
- **基于规则引擎的虚拟教练**：灵活性差，难以扩展至多种病症。
- **纯LLM驱动系统**：存在hallucination风险，可能违反生理安全边界。
- **依赖专用硬件的传感系统**（如IMUs、Kinect）：成本高、部署复杂。

> 本系统优势在于**无需额外硬件**，仅通过普通摄像头即可完成闭环反馈。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自Table III）
| 组件 | 指标 | 估计值 | 目标阈值 | 达成情况 |
|------|-----|-------|----------|--------|
| Vision Processing Agent | Pose Estimation Latency | **28 ms** | < 50 ms | ✅ 超出预期 |
| Vision Processing Agent | Joint Angle Error Margin | **±3.2°** | < 5° | ✅ 满足精度要求 |
| Clinical Extraction Agent | Clinical Text Parsing Accuracy | **96.5%** | > 95% | ✅ 达标 |
| Video Synthesis Agent | Video Generation Time | **45 秒** | < 60 秒 | ✅ 满足可用性 |

> 所有核心模块均达到或超过设计目标，支持实时人机交互。

### 🔀 与基线方法的对比结果
- 相比传统静态视频教学：
  - 提供**动态个性化示范**，避免患者模仿超出自身能力的动作。
  - 实现**主动干预式指导**，而非被动观看。
- 相比通用虚拟教练系统：
  - 支持对**任意伤情组合**进行快速适配（通过自然语言输入）。
  - 输出反馈具有**临床可追溯性**（via XAI），增强医患信任。

### ❌ 消融实验（Ablation Study）
- 论文中**未报告消融实验**。
- 但强调了**确定性规则在Diagnostic Feedback中的关键作用**：即使LLM参与生成反馈，最终输出必须受控于硬编码的安全边界（如最大角度±容忍度），以防止幻觉导致危险建议。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Agentic AI可用于构建安全、个性化的远程康复闭环系统**。
2. 多代理架构能有效解耦复杂任务（语义解析、视频生成、视觉感知、诊断反馈），提升系统模块化与可维护性。
3. **生成式AI + 实时CV + LLM + XAI 的融合是可行且高效的路径**，尤其适用于需要高安全性和可解释性的医疗场景。
4. 所有关键技术组件（MediaPipe、LLM、diffusion video models）已具备实际部署潜力，延迟与精度满足临床可用标准。

### ⚠️ 局限性
1. **视频合成质量仍需优化**：当前生成的“physio-twin”在时空一致性（temporal consistency）和解剖准确性（anatomical fidelity）方面仍有改进空间。
2. **尚未进行真实患者临床试验**：目前仅为原型验证，缺乏对真实康复效果的影响评估。
3. **依赖高质量基础模型**：视频生成效果受限于当前foundation video models的发展水平。
4. **未考虑复杂动作序列或多关节协同运动**：当前focus在单个动作的关键姿态控制。

### 🔮 未来工作方向
1. **开展全规模临床试验**：与医院合作，对比本系统与传统疗法在康复进度、依从性、再受伤率等方面的差异。
2. **集成可穿戴传感器（如IMUs）进行交叉验证**：用于校准和验证视觉系统的测量精度。
3. **增强视频生成的真实感与时序连贯性**：探索更先进的video diffusion architectures。
4. **支持更多病种与复合动作**：扩展至中风后康复、脊柱矫正等复杂场景。
5. **长期用户行为建模**：引入记忆机制，跟踪患者进步并动态调整训练难度。

---

## 总结

> 本论文提出了一种开创性的 **Agentic AI-driven MAS 框架**，成功整合了**生成式AI、计算机视觉、大语言模型与可解释AI**，实现了从医生处方到个性化训练再到实时姿势纠偏的完整闭环。虽然尚处原型阶段，但其技术路线展现了极强的临床转化潜力，为未来**智能化、去中心化、高可及性的数字健康服务**提供了重要范式参考。

</details>

---

### 13. [Temporally Extended Mixture-of-Experts Models](https://arxiv.org/abs/2604.20156)

**Authors**: Zeyu Shen, Peter Henderson  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.20156v1  

#### Abstract
Mixture-of-Experts models, now popular for scaling capacity at fixed inference speed, switch experts at nearly every token. Once a model outgrows available GPU memory, this churn can render optimizations like offloading and pre-fetching ineffective. We make the case that the options framework in rei...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Temporally Extended Mixture-of-Experts Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代 **Mixture-of-Experts (MoE)** 模型在推理时虽然能保持固定的计算量，但其专家切换频率极高——几乎每个 token 都会更换激活的专家集合。当模型总参数超出 GPU 内存容量时，这种频繁切换会导致以下问题：
- **内存效率低下**：无法有效利用 offloading、prefetching 等优化技术；
- **延迟增加**：每次切换都需要从主机内存加载新专家，造成显著延迟；
- **训练与持续学习受限**：缺乏对专家选择的时间连续性建模。

当前 MoE 架构普遍忽略“切换成本”，假设所有专家始终驻留在 GPU 中，这在现实部署中不可行。

---

### **提出的新方法与新思路**
本文提出 **Temporally Extended MoE（时间扩展 MoE）**，将 MoE 路由问题重新形式化为一个 **semi-Markov Decision Process (s-MDP)**，并引入 **Options Framework** 来控制专家掩码（expert mask）的持久性。

#### **核心创新：**
- 将每个 MoE 层中的 **expert mask** 视为一个 **option**，该 option 可以持续多个 token 步骤；
- 引入轻量级 **controller**（策略网络），决定何时保留当前 expert mask，以及何时切换到新的 mask；
- 控制器通过 **Option-Critic 架构** 进行训练，并显式加入 **deliberation cost**（决策代价）作为正则项，鼓励控制器仅在收益大于切换成本时才进行切换。

这种方法实现了 **temporal abstraction**：专家集合不再逐 token 切换，而是形成“时间块”（temporal chunks），从而支持更高效的内存管理和训练策略。

---

### **相比现有方法的优势**
| 方面 | 传统 MoE / 现有方法 | 本文方法（Temporally Extended MoE） |
|------|---------------------|------------------------------------|
| **专家切换频率** | 几乎每步都变（>50% switch rate） | 显著降低至 <5%，甚至 <1% |
| **内存效率** | 必须缓存大量专家或频繁 offload | 只需维持少量活跃专家，大幅减少 VRAM 占用 |
| **训练方式** | 固定路由或静态剪枝 | 动态、可学习的时间扩展路由 |
| **持续学习潜力** | 扩展专家池困难 | 支持动态添加新专家而不影响推理开销 |
| **系统兼容性** | 不利于 prefetching/cache 设计 | 天然适合 temporal chunking 和高效 serving |

此外，本方法可在已有预训练 MoE 模型上通过 **轻量微调** 实现转换，无需从头训练。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：`Nemotron Post-Training Dataset v2`，包含 10 类任务（chat、code、math、STEM、多语言等），共 128–16 prompts per batch。
- **评估基准**：
  - `MATH`：数学推理题（200 道）
  - `MMLU`：大规模多任务语言理解
  - `MMMLU`：多语言版本 MMLU

所有评估使用温度采样（temperature=0.5）、Top-p=0.95，最大响应长度为 2048 tokens。

---

### **实验设置与评估指标**

#### **模型基础**
- 使用 `gpt-oss-20b` 模型：
  - 24 层 Transformer
  - 每层 32 个专家（N=32）
  - Top-4 路由（k=4 激活）
- 控制器设计：
  - 每层独立 controller
  - 输入：LLM 隐藏状态 $ h_t^{(l)} $ 和当前 expert mask
  - 输出：终止概率 $ \beta $ 和新 mask 选择策略
  - 使用 **Low-Rank Adaptation (LoRA)** 微调主模型参数（rank=16）

#### **训练细节**
- 使用 **self-distillation reward**：
  $$
  r_t = \log p_{\text{teacher}}(a_t|x, a_{<t}) - \log p_{\text{student}}(a_t|x, a_{<t})
  $$
  教师模型为原始 gpt-oss-20b，学生为带 controller 的变体。
- 采用混合采样：$ p_{\text{mix}} = (1-\tau)p_{\text{student}} + \tau p_{\text{teacher}} $，$\tau=0.2$
- 优化目标包含：
  - Critic loss（TD error）
  - Intra-option policy gradient
  - Termination gradient with deliberation cost $\eta$
  - Option selection gradient

#### **评估指标**
- **准确率（Accuracy）**：在 MATH/MMLU/MMMLU 上的正确率（%）
- **switch rate (%)**：expert mask 发生变化的比例
- **95% 置信区间** 报告统计显著性

---

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Frequency-based pruning** | 保留校准集中最常被激活的 k 个专家 |
| **Reconstruction loss minimization** | 选择能最好重构原输出的专家子集 |
| **Random selection** | 随机选取 k 个专家 |
| **Wanda (structured pruning)** | 结构化权重剪枝方法 |
| **Base Model** | 原始未修改的 gpt-oss-20b |

> 注：不与 MoE-infinity 等 prefetching 方法比较，因其目标不同（后者不改变路由行为）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 2 & 3）**

#### **k=16（允许最多 16 个专家参与路由）**

| Method | MATH Acc (%) | MMLU Acc (%) | MMMLU Acc (%) | Switch Rate (%) |
|--------|---------------|---------------|----------------|------------------|
| Base Model | 71.5 ± 5.9 | 79.5 ± 5.7 | 67.5 ± 6.5 | >50% |
| Frequency | 53.5 | 55.5 | 42.0 | — |
| Reconstruction | 51.5 | 35.0 | 48.0 | — |
| Random | 15.0 | 33.5 | 24.0 | — |
| Wanda | 3.5 | 9.0 | 7.0 | — |
| **Ours ($\eta=0.02$)** | **64.0 ± 6.7** | **72.5 ± 6.3** | **59.5 ± 6.9** | **4.2 ± 0.02** |
| **Ours ($\eta=0.04$)** | 55.0 | 63.0 | 49.5 | **1.2** |

> ✅ 在 switch rate 降至 **4.2% → 1.2%** 的同时，仍保留高达 **90% 的原始模型准确性**。

#### **k=8（更严格限制）**

| Method | MATH Acc (%) | MMLU Acc (%) | MMMLU Acc (%) | Switch Rate (%) |
|--------|---------------|---------------|----------------|------------------|
| Base Model | 71.5 ± 5.9 | 79.5 ± 5.7 | 67.5 ± 6.5 | >50% |
| **Ours ($\eta=0.02$)** | 27.5 ± 6.1 | 48.5 ± 6.9 | 39.0 ± 6.5 | 7.4 |
| **Ours ($\eta=0.04$)** | 15.5 | 38.0 | 22.5 | **5.0** |

> ⚠️ 性能下降明显，但仍优于大多数剪枝基线，且 switch rate 仍远低于原始模型。

---

### **与基线方法的对比结果**
- 所有 pruning 基线在 accuracy 上严重退化（如 Wanda 几乎崩溃）；
- 本文方法在 **相同 k 设置下显著优于所有静态剪枝方法**；
- 即使只允许 16 个专家参与路由（而非全部 32），也能接近 base model 表现；
- **deliberation cost $\eta$ 可调节 trade-off**：越大则 switch rate 越低，但 accuracy 略降。

---

### **消融实验与分析（Ablation Insights）**
- **Temporal Continuity Visualization**（Fig. 6–7）显示：
  - 原始模型 expert 激活高度碎片化；
  - 加入 controller 后，expert mask 持续数十至上百个 token，表现出强时间连续性。
- **Training Dynamics**（Fig. 5）表明：
  - 随着训练进行，switch rate 下降并稳定；
  - 更高的 $\eta$ 导致更低的最终 switch rate；
  - Reward 和 Perplexity 持续改善，说明 student 逐步逼近 teacher。
- **Repetition Analysis**（Fig. A3）验证：
  - 无灾难性重复现象（catastrophic repetition），生成质量可控。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **MoE 模型可以被转化为 temporally extended 形式**，即使是在 post-training 阶段；
2. ✅ 通过引入 **option-critic + deliberation cost**，可有效学习何时切换专家，实现 **switch rate 从 >50% 降至 <5%**；
3. ✅ 在极低切换率下，仍能保留 **高达 90% 的 base model accuracy**；
4. ✅ 该框架支持 **memory-efficient serving、chunked training、continual learning with expandable experts**；
5. ✅ 使用 **LoRA + self-distillation** 可实现轻量化适配，适用于已有 MoE 模型改造。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **Per-layer options 分离** | 当前每层独立决策，导致各层 switch 时间点不一致；理想情况应跨层同步以简化内存管理 |
| **Deliberation cost 是超参而非实测延迟** | $\eta$ 当前是人工设定，尚未与真实硬件加载延迟绑定 |
| **未覆盖全部能力维度** | 缺少代码生成、长对话等任务评估 |
| **依赖 teacher model 自蒸馏** | 虽然合理，但可能掩盖部分路由改进的真实来源 |
| **组合空间爆炸风险** | 若推广至 joint multi-layer options，option 空间呈指数增长，难以训练 |

---

### **未来工作方向**
1. **将 temporal extension 嵌入 pre-training 目标**：让模型天生具备时间抽象能力；
2. **构建 end-to-end 内存优化系统**：结合 MoE-Infinity 等 prefetching 方法，实现真正的低延迟 serving；
3. **跨层联合 options 学习**：定义全局 option，在所有层同步切换 expert mask；
4. **硬件感知的 deliberation cost 建模**：基于实际 GPU/CPU 传输延迟自动调整 $\eta$；
5. **探索自然语言中的 temporal structure 与 expert routing 的关联**：例如主题一致性、推理链延续性是否对应特定 expert 组合；
6. **用于持续学习场景**：动态添加新专家模块，由 controller 自动学会何时启用。

---

> 📌 **总结一句话**：  
> 本文首次将 **Options Framework** 引入 MoE 路由机制，提出了 **Temporally Extended MoE** 新范式，证明了通过轻量控制器即可大幅降低专家切换频率，在几乎不失效的前提下解锁内存高效推理、分块训练与持续扩展能力，为下一代大规模 MoE 模型提供了原则性的优化路径。

</details>

---

### 14. [Probabilistic Verification of Neural Networks via Efficient Probabilistic Hull Generation](https://arxiv.org/abs/2604.21556)

**Authors**: Jingyang Li, Xin Chen, Hongfei Fu, Guoqiang Li  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.21556v1  

#### Abstract
The problem of probabilistic verification of a neural network investigates the probability of satisfying the safe constraints in the output space when the input is given by a probability distribution. It is significant to answer this problem when the input is affected by disturbances often modeled b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Probabilistic Verification of Neural Networks via Efficient Probabilistic Hull Generation**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对**神经网络在随机输入扰动下的安全性验证问题**，即**Probabilistic Verification**。具体而言，当神经网络的输入受到高斯噪声等概率分布影响时，传统形式化验证方法难以处理无界输入空间，且无法量化满足安全属性的概率。

现有方法如 **ProbStar** 虽能进行概率验证，但受限于仅支持 ReLU 激活函数，且在高维空间中效率低下；而基于均匀划分的 **branch-and-bound (BaB)** 方法则面临“维度灾难”，计算开销巨大。

### **提出了什么新方法或新思路**
作者提出了一种**基于回归树引导的概率性 hull 生成框架**，用于高效估计神经网络的安全概率上下界。其核心思想是通过构建“**probabilistic hulls**”——输入空间中与高斯分布对齐的有界区域——来保守地估算安全概率。

#### **三大创新点**：
1. **回归树引导的状态空间划分策略（Regression Tree-Guided State Space Splitting）**  
   利用回归树自适应地将输入空间划分为 box 区域，优先在决策边界附近细分，避免在纯安全或不安全区域进行不必要的划分，从而高效生成大体积的 safe/unsafe probabilistic hulls。

2. **边界感知采样方法（Boundary-Aware Sampling）**  
   通过采样识别输出接近安全边界的输入点，并利用这些样本训练回归树，使划分更贴近真实的安全边界，提升 hull 的质量和收敛速度。

3. **基于概率优先级的迭代精炼机制（Iterative Refinement with Probabilistic Prioritization）**  
   在每次迭代中选择未确定区域中概率质量最高的 hull 进行细分，确保每一步都最大化对最终概率区间的影响，显著提高效率。

### **相比现有方法的优势**
- **通用性强**：可处理任意激活函数（如 tanh），而 ProbStar 仅限 ReLU。
- **效率更高**：相比 BaB 和 ProbStar，在多个基准上实现 **最高达 10× 的加速**。
- **精度更高**：生成的上下界更紧（tighter bounds），尤其在未知概率 $ U_s - L_s $ 上表现优异。
- **可扩展性好**：在高维输入（如 Rocket Lander 的 9 维状态）下仍保持良好性能。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **ACAS Xu**  
   - 45 个 DNN 控制器，用于飞机防撞系统。
   - 输入维度：5，输出维度：5，每层 50 个神经元。
   - 验证性质：Property P2，要求网络不输出“COC”作为最优动作。

2. **Rocket Lander Controller**  
   - 模拟 SpaceX Falcon 9 火箭垂直着陆控制。
   - 输入维度：9（位置、速度、角度等），输出：主推力与侧向氮气推进器控制。
   - 安全性质：防止左右倾斜失控（P1 & P2）。

3. **Distilled DNNs from ACAS Xu**  
   - 使用知识蒸馏得到的 tanh 激活函数版本，用于测试方法对非 ReLU 网络的支持能力。

### **实验设置和评估指标**
- **输入分布**：高斯分布 $ \mathcal{N}(\mu, \Sigma) $，协方差矩阵为对角阵。
- **终止条件**：$ \max_{R \in R_{\text{unknown}}} P(R) \leq \epsilon $，其中 $ \epsilon = 10^{-5} $ 或 $ 10^{-3} $。
- **评估指标**：
  - **安全概率下界 $ L_s $** 和上界 $ U_s $**
  - **未知概率宽度 $ U_s - L_s $**（越小越好）
  - **运行时间（Time）**：包括串行与并行版本
  - **是否超时（T.O. > 2 小时）**

### **基线方法对比**
- **ProbStar [14]**：基于 star set 的概率可达性分析，仅支持 ReLU。
- **Basic Branch-and-Bound (BaB) [2]**：基于最长维度二分划分的传统 BaB 方法。
- 所有方法均使用 **CROWN** 进行 hull 的安全性验证。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **ACAS Xu 结果（Table 1）**
| Network | Method | $ L_s $ | $ U_s $ | $ U_s - L_s $ | Time (parallel) |
|--------|--------|---------|---------|---------------|------------------|
| 1-6    | Ours   | 0.9847  | 1.0000  | **0.0153**     | **46.58s**       |
|        | ProbStar | 0.9337 | 0.9972 | 0.0635        | 315.75s          |
|        | BaB    | —       | —       | —             | T.O.             |

- 在所有网络中，**我们的方法均取得最紧的 $ U_s - L_s $**，平均比 ProbStar 收缩超过 **50%**。
- 平均运行时间仅为 ProbStar 的 **约 1/7**，且远快于 BaB（后者全部超时）。
- 并行版本（GPU）相比串行平均提速 **4.8×**。

#### **Rocket Lander 结果（Table 2 & 3）**
| Net & Prop | Method | $ U_s - L_s $ | Time (parallel) |
|-----------|--------|----------------|------------------|
| 0&1       | Ours   | **0.6516**      | **33.77s**       |
|           | ProbStar | 0.7150        | 141.05s          |
| 1&2       | Ours   | **0.0791**      | **12.95s**       |
|           | ProbStar | 0.3569        | 117.49s          |

- 在所有配置下，**$ U_s - L_s $ 显著更小**，部分情况缩小近 **4 倍以上**。
- 运行时间大幅降低，**最快达到 12.95 秒 vs. ProbStar 的 117.49 秒**。

#### **效率对比实验（Table 4）**
在相同时间内（1800s），比较找到的 hull 数量与边界紧度：
- 我们的方法在 **远短于截止时间** 内完成（如 68–303s），而 BaB 全部超时。
- 得到的 $ U_s - L_s $ 更小，说明在有限时间内能获得更精确的结果。

---

### **消融实验结果（Ablation Study, Table 5 & 6）**

#### **组件贡献分析（Table 5）**
| 配置 | Sampling | $ U_s - L_s $ (Rocket) | Time (s) |
|------|----------|------------------------|----------|
| Baseline | Uniform-only | 0.9194 | 30.26 |
| +Impurity Scheduler | Mixed (0.75,0.25) | **0.2371** | **21.25** |

- **Impurity Scheduler 是最关键组件**：启用后 $ U_s - L_s $ 下降超过 **70%**。
- 单纯调整采样比例效果有限，但结合调度器后，混合采样（0.75 uniform + 0.25 distributional）表现最佳。

#### **调度阈值 $ \beta $ 影响（Table 6）**
| $ \beta $ | Splitting Mode | $ U_s - L_s $ (ACAS) | Time (s) |
|-----------|----------------|-----------------------|----------|
| 1.0       | Longest-dim only | 0.0650              | 279.46   |
| 0.75      | Late activation | **0.0341**           | 211.61   |
| 0.0       | Immediate soft | 0.0610               | 349.11   |

- 对 ACAS Xu，**延迟启用 impurity measure（$ \beta = 0.75 $）效果最好**，说明初始全局划分有助于后续精细搜索。
- 对 Rocket Lander，则 **立即启用（$ \beta = 0 $）最优**，表明高维下需尽早引入边界感知分裂。

---

## **4. 关键结论和发现**

### **主要发现**
1. **边界感知的回归树划分显著优于均匀或最长维度划分**，能在更少迭代中生成高质量 hull。
2. **混合采样 + impurity scheduler 构成高效闭环**：采样聚焦边界 → 回归树沿边界分裂 → 高概率 hull 快速确认 → 加速收敛。
3. **方法具有良好的可扩展性**，在 9 维输入空间中仍保持高效，远超 BaB 与 ProbStar。
4. **支持非 ReLU 激活函数**，展示了更强的通用性和实用性。

### **方法的局限性**
1. **依赖 CROWN 进行 hull 验证**，其本身可能成为性能瓶颈，尤其是在深层网络中。
2. **仍受维度灾难影响**，尽管通过边界感知缓解，但在极端高维场景下仍可能退化。
3. **某些靠近边界的 hull 需要大量细分才能判定**，未来可引入收缩策略（shrinking methods）优化。

### **未来工作方向**
1. **开发专门针对 probabilistic hull 的高效 output range 分析技术**，减少对 CROWN 的依赖。
2. **引入符号化表示**（如 Taylor models）替代 box 表示，以更好逼近复杂边界。
3. **设计 hull 收缩算法**，快速从模糊区域中提取纯 safe/unsafe 子集。
4. **探索更多激活函数与网络结构的泛化能力**，推动工业级应用。

---

> ✅ **总结一句话**：本文提出了一种**高效、通用、精准的神经网络概率验证框架**，通过**回归树引导的边界感知划分**，在多个挑战性基准上实现了**精度与速度的双重突破**，为安全关键系统的可信 AI 部署提供了有力工具。

</details>

---

### 15. [Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts](https://arxiv.org/abs/2604.19835)

**Authors**: Chaitanya Dwivedi, Binxuan Huang, Himanshu Gupta, Pratik Jayarao, Neeraj Varshney, Bing Yin  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.19835v1  

#### Abstract
Mixture-of-Experts (MoE) has become the dominant architecture for scaling large language models: frontier models routinely decouple total parameters from per-token computation through sparse expert routing. Scaling laws show that under fixed active computation, model quality scales predictably with ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
- **训练大规模 MoE 模型成本高昂**：尽管 Mixture-of-Experts (MoE) 架构通过稀疏路由实现了总参数量与每 token 计算量的解耦，但其训练成本依然极高。原因在于：
  - 所有专家的权重、梯度和优化器状态都必须驻留在设备内存中，导致显存需求随总参数量线性增长。
  - 专家分布在多个设备上时，需要频繁的 all-to-all 通信，占用了大量训练时间（可达 45–50%）。
- 现有方法如从头训练固定规模的 MoE 或从 dense 模型转换为 MoE（sparse upcycling），在效率和质量之间难以兼顾。

### **提出了什么新方法或新思路**
提出 **Expert Upcycling** —— 一种在持续预训练（Continued Pre-training, CPT）过程中逐步扩展 MoE 容量的方法，具体流程如下：

1. **阶段一**：先在一个较小的 E-expert MoE 模型上进行预训练。
2. **阶段二（Upcycling）**：在某个过渡步 $T$，应用 **upcycling operator $U_m$** 将模型扩展为 $mE$ 个专家：
   - **Expert Replication**：将每个已有专家复制 $r_e \geq 1$ 次，满足 $\sum r_e = mE$。
   - **Router Extension**：复制原始路由器权重，并对副本添加微小偏置噪声以打破对称性。
3. **阶段三（CPT）**：在扩展后的 $mE$-expert 模型上继续训练，利用随机梯度多样性驱动专家专业化。

该方法保持 **Top-K 路由不变**，因此推理时的激活参数和每 token FLOPs 不变。

### **相比现有方法的优势**
| 方法 | 类型 | 是否改变推理成本 | 是否复用已有表示 | 优势 |
|------|------|------------------|------------------|------|
| 从头训练 Fixed-size MoE | Dense → MoE | 否 | 否 | 高质量但高成本 |
| Sparse Upcycling [25] | Dense → MoE | 是（引入稀疏） | 是 | 复用 dense checkpoint |
| **Expert Upcycling（本文）** | **MoE → Larger MoE** | **否** | **是** | **高效 + 高质量 + 可迭代升级** |

- **计算更高效**：前 $T$ 步在小模型上运行，节省约 **32% GPU 小时**；若已有 checkpoint，则可节省高达 **67%**。
- **初始化更优**：复制已训练专家提供“暖启动”（warm initialization），初始损失远低于随机初始化。
- **支持渐进式扩容**：可在不中断服务的前提下，持续提升模型容量。

此外，本文还提出：
- **Utility-based Expert Selection**：基于梯度重要性分数（如 $||g_e||^2$）非均匀复制高价值专家，在 CPT 预算有限时，**gap closure 效果提升超过三倍**。
- **理论框架分解质量差距**：将最终性能差距分解为：
  - **Capacity Gap（容量差距）**：因早期在小模型中训练造成；
  - **Initialization Gain（初始化增益）**：因更好初始化带来的优势。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **主实验（7B→13B）**：使用精心策划的数据混合物，强调指令遵循、逻辑推理和数学能力。
- **小规模消融实验（~1B 参数）**：使用 **DCLM** 数据集，用于快速验证不同策略。

所有实验采用 **独立划分的预训练与 CPT 数据集**，避免数据泄露。

### **实验设置**
- **模型架构**：
  - 主要使用 **Interleaved MoE**（交替堆叠 dense 和 MoE 层），减少 all-to-all 通信开销。
  - 也验证了 **Full MoE** 架构（全部层均为 MoE）的有效性。
- **参数规模**：
  - 主实验：从 **7B 总参数 → 13B 总参数**（活动参数 ~1B）。
  - 消融实验：在 ~154M 到 ~1B 总参数范围内测试。
- **路由配置**：Top-K=2 或 Top-K=8，无共享专家。
- **训练方式**：使用数据并行 + 张量并行，优化器使用 Warmup-Stable-Decay (WSD) 调度。

### **评估指标**
- **Validation Loss（↓）**：衡量语言建模能力。
- **Downstream Benchmark Accuracy（↑）**：在 11 个任务上评估，包括：
  - MMLU, BBH CoT, GSM8K, IFEval, HellaSwag, ARC-Challenge/Easy, PIQA, OpenBookQA, SciQ, Social IQA
- **Upcycling Efficiency**（归一化差距闭合率）：
  $$
  \text{Efficiency} = \frac{L(\text{Fixed-E}) - L(\text{Upcycled})}{L(\text{Fixed-E}) - L(\text{Fixed-mE})}
  $$
  数值越接近 1 表示 gap closure 越完全。

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Fixed-32** | 仅使用 32-expert 模型全程训练（下界） |
| **Fixed-64** | 64-expert 模型从头训练（上界/质量天花板） |
| **Upcycled (32→64)** | 本文方法：先训 32-expert，再 upcycle 至 64-expert 并继续训练 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（Table 1）**

| 方法 | Val. Loss ↓ | Avg Acc ↑ | GPU Hours |
|------|-------------|-----------|----------|
| Fixed-32 | 1.301 | 52.9 | 21,168 |
| **Upcycled (32→64)** | **1.263** | **56.4** | **27,888** |
| Fixed-64 | 1.267 | 56.7 | 41,328 |

> ✅ 在 **100% CPT** 下，upcycled 模型在验证损失和平均准确率上均 **匹配 Fixed-64**，同时节省 **~32% GPU 小时**。

#### **50% CPT 结果**
- Upcycled 已经显著优于 Fixed-32，且在多数任务（如 HellaSwag, PIQA, Social IQA）上 **持平甚至超越 Fixed-64**。
- 知识密集型任务（MMLU, GSM8K）仍有差距，但可通过延长 CPT 收敛。

### **与基线方法的对比结果**
- **vs. Fixed-size Training**：
  - 质量相当，但训练成本降低 32%。
- **vs. Sparse Upcycling (dense → MoE)**：
  - 在低激活比（activation ratio）下，expert upcycling 显著优于 sparse upcycling（Table 5）。
  - 当目标激活比降至 3.13% 时，sparse upcycling 几乎无法闭合差距（loss: 3.049 vs. 2.808），而 expert upcycling 仍能有效逼近目标。

### **消融实验结果**

#### **(1) 训练预算分配（Table 3）**
- **CPT 预算越多，gap closure 越好**：
  - CPT 占比从 10% → 100%，效率从 34.7% → 98.0%。
  - 至少需要 **50% CPT** 才能实现强 gap closure。
- **过渡时机建议**：
  - 若从零开始训练，建议在总步数的 **12–38%** 进行 upcycling。
  - 过早（<5%）可能导致源模型未充分专业化，影响 warm initialization。

#### **(2) 专家选择策略（Table 4）**
| 策略 | 25% CPT 效率 | 100% CPT 效率 |
|------|--------------|---------------|
| Uniform Duplication | 8.2% | 76.8% |
| **Utility-based ($||g||^2$)** | **26.5%** | **97.8%** |

> 🔥 在 CPT 有限时，utility-based 方法 **gap closure 效果提升超三倍**。

#### **(3) 激活比影响（Table 5）**
| 目标 K/E | Upcycled Loss | Sparse Upcycling Loss |
|---------|--------------|------------------------|
| 25%     | 3.061        | 3.087                  |
| 6.25%   | 2.992        | 3.069                  |
| **3.13%** | **2.808**    | **3.049**              |

> 随着目标激活比下降，expert upcycling 优势愈发明显。

#### **(4) 初始化多样性扰动（Appendix D）**
尝试多种扰动策略（noise injection, drop-upcycling, orthogonalization 等）均未能超越简单复制（copy-paste），部分反而损害性能。

> 💡 **低初始损失比人为引入多样性更重要**：perturbation 会破坏已有表示，迫使 CPT 先恢复而非专注 specialization。

---

## 4. **关键结论和发现**

### **主要发现**
1. **Expert Upcycling 是一种高效且高质量的 MoE 扩容范式**：
   - 在保持推理成本不变的前提下，成功复现大 MoE 模型的质量。
   - 训练成本降低 **32%**，若已有 checkpoint 则可降 **67%**。

2. **Warm Initialization 至关重要**：
   - 复制已训练专家使初始损失接近原模型（~1.38 vs. 原始 1.32），远优于随机初始化（>10.5）。
   - “冷启动”会导致严重性能退化。

3. **Utility-based Selection 显著提升初期收敛速度**：
   - 基于梯度范数 $||g_e||^2$ 的非均匀复制是最优策略。
   - 特别适用于 CPT 预算受限场景。

4. **Symmetry Breaking 自然发生**：
   - 路由器偏置噪声 + loss-free load balancing + 随机梯度多样性，足以驱动专家分化。
   - 无需复杂初始化机制。

5. **适用于多种 MoE 架构和规模**：
   - 在 Interleaved MoE 和 Full MoE 上均有效。
   - 从 154M 到 7B 参数规模均可适用。

### **方法的局限性**
- **依赖足够长的 CPT**：若 CPT 时间太短，capacity gap 难以闭合。
- **当前仅支持整数倍扩展（m=2）**：更大或非整数倍扩展需进一步研究。
- **假设数据分布平稳**：若 CPT 数据分布剧烈变化，可能影响迁移效果。
- **尚未在千亿级以上模型验证**：实际部署中可能存在通信瓶颈或 router collapse 风险。

### **未来工作方向**
- **多阶段连续 upcycling**：通过多次翻倍专家数量，实现平滑扩容（iterative doubling）。
- **结合 pruning 形成闭环**：upcycling 增加容量 → 训练 → pruning 回收效率。
- **探索其他 selection criteria**：如基于 Fisher 信息、KL 散度等。
- **动态调整 m 和 K**：根据任务需求灵活控制容量与延迟权衡。
- **应用于 fine-tuning 或 domain adaptation**：在特定领域增量添加专家。

---

> 📌 **一句话总结**：  
> **Expert Upcycling 提供了一种“可持续升级”的 MoE 训练范式——不是从零重建巨人，而是让已有巨人不断生长。它不仅节省了近三分之一的训练资源，还能达到与从头训练相当甚至更好的性能，是迈向高效稀疏模型训练的重要一步。**

</details>

---

### 16. [A Delta-Aware Orchestration Framework for Scalable Multi-Agent Edge Computing](https://arxiv.org/abs/2604.20129)

**Authors**: Samaresh Kumar Singh, Joyjit Roy  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.20129v1  

#### Abstract
The Synergistic Collapse occurs when scaling beyond 100 agents causes superlinear performance degradation that individual optimizations cannot prevent. We observe this collapse with 150 cameras in Smart City deployment using MADDPG, where Deadline Satisfaction drops from 78% to 34%, producing approx...

---

### 17. [Robustness of Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable Distribution Grids](https://arxiv.org/abs/2604.20403)

**Authors**: Burak Karabulut, Carlo Manna, Chris Develder  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.20403v1  

#### Abstract
Fault location in distribution grids is critical for reliability and minimizing outage durations. Yet, it remains challenging due to partial observability, given sparse measurement infrastructure. Recent works show promising results by combining Recurrent Neural Networks (RNNs) and Graph Neural Netw...

---

### 18. [Efficient Test-Time Inference via Deterministic Exploration of Truncated Decoding Trees](https://arxiv.org/abs/2604.20500)

**Authors**: Xueyan Li, Johannes Zenn, Ekaterina Fadeeva, Guinan Su, Mrinmaya Sachan, Jonas Geiping  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.20500v1  

#### Abstract
Self-consistency boosts inference-time performance by sampling multiple reasoning traces in parallel and voting. However, in constrained domains like math and code, this strategy is compute-inefficient because it samples with replacement, repeatedly revisiting the same high-probability prefixes and ...

---

### 19. [Stream-CQSA: Avoiding Out-of-Memory in Attention Computation via Flexible Workload Scheduling](https://arxiv.org/abs/2604.20819)

**Authors**: Yiming Bian, Joshua M. Akey  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.20819v1  

#### Abstract
The scalability of long-context large language models is fundamentally limited by the quadratic memory cost of exact self-attention, which often leads to out-of-memory (OOM) failures on modern hardware. Existing methods improve memory efficiency to near-linear complexity, while assuming that the ful...

---

### 20. [TRACES: Tagging Reasoning Steps for Adaptive Cost-Efficient Early-Stopping](https://arxiv.org/abs/2604.21057)

**Authors**: Yannis Belkhiter, Seshu Tirupathi, Giulio Zizzo, John D. Kelleher  
**Category**: cs.CL  
**Published**: 2026-04-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.21057v1  

#### Abstract
The field of Language Reasoning Models (LRMs) has been very active over the past few years with advances in training and inference techniques enabling LRMs to reason longer, and more accurately. However, a growing body of studies show that LRMs are still inefficient, over-generating verification and...

---

### 21. [Accelerating PayPal's Commerce Agent with Speculative Decoding: An Empirical Study on EAGLE3 with Fine-Tuned Nemotron Models](https://arxiv.org/abs/2604.19767)

**Authors**: Ally Qin, Jian Wan, Sarat Mudunuri, Srinivasan Manoharan  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.19767v1  

#### Abstract
We evaluate speculative decoding with EAGLE3 as an inference-time optimization for PayPal's Commerce Agent, powered by a fine-tuned llama3.1-nemotron-nano-8B-v1 model. Building on prior work (NEMO-4-PAYPAL) that reduced latency and cost through domain-specific fine-tuning, we benchmark EAGLE3 via vL...

---

### 22. [Bridging the Training-Deployment Gap: Gated Encoding and Multi-Scale Refinement for Efficient Quantization-Aware Image Enhancement](https://arxiv.org/abs/2604.21743)

**Authors**: Dat To-Thanh, Nghia Nguyen-Trong, Hoang Vo, Hieu Bui-Minh, Tinh-Anh Nguyen-Nhu  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.21743v1  

#### Abstract
Image enhancement models for mobile devices often struggle to balance high output quality with the fast processing speeds required by mobile hardware. While recent deep learning models can enhance low-quality mobile photos into high-quality images, their performance is often degraded when converted ...

---

### 23. [AgenticQwen: Training Small Agentic Language Models with Dual Data Flywheels for Industrial-Scale Tool Use](https://arxiv.org/abs/2604.21590)

**Authors**: Yuanjie Lyu, Chengyu Wang, Haonan Zheng, Yuanhao Yue, Junbing Yan, Ming Wang, Jun Huang  
**Category**: cs.CL  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.21590v1  

#### Abstract
Modern industrial applications increasingly demand language models that act as agents, capable of multi-step reasoning and tool use in real-world settings. These tasks are typically performed under strict cost and latency constraints, making small agentic models highly desirable. In this paper, we i...

---

### 24. [Beyond N-gram: Data-Aware X-GRAM Extraction for Efficient Embedding Parameter Scaling](https://arxiv.org/abs/2604.21724)

**Authors**: Yilong Chen, Yanxi Xie, Zitian Gao, He Xin, Yihao Xiao, Renbiao Liu, Haoming Luo, Yifan Luo, Zhengmao Ye, Tingwen Liu, Xin Zhao, Ran Tao, Bryan Dai  
**Category**: cs.CL  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.21724v1  

#### Abstract
Large token-indexed lookup tables provide a compute-decoupled scaling path, but their practical gains are often limited by poor parameter efficiency and rapid memory growth. We attribute these limitations to Zipfian under-training of the long tail, heterogeneous demand across layers, and "slot colla...

---

### 25. [Fast Amortized Fitting of Scientific Signals Across Time and Ensembles via Transferable Neural Fields](https://arxiv.org/abs/2604.19979)

**Authors**: Sophia Zorek, Kushal Vyas, Yuhao Liu, David Lenz, Tom Peterka, Guha Balakrishnan  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.19979v1  

#### Abstract
Neural fields, also known as implicit neural representations (INRs), offer a powerful framework for modeling continuous geometry, but their effectiveness in high-dimensional scientific settings is limited by slow convergence and scaling challenges. In this study, we extend INR models to handle spati...

---

### 26. [On the Quantization Robustness of Diffusion Language Models in Coding Benchmarks](https://arxiv.org/abs/2604.20079)

**Authors**: Aarav Gupta, Gururaj Deshpande, Chandreyi Chakraborty  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.20079v1  

#### Abstract
Auto-regressive Large Language Models (LLMs) achieve strong performance on coding tasks, but incur high memory and inference costs. Diffusion-based language models (d-LLMs) offer bounded inference cost via iterative denoising, but their behavior under post-training quantization (PTQ) has been sparse...

---

### 27. [Synthetic Flight Data Generation Using Generative Models](https://arxiv.org/abs/2604.20293)

**Authors**: Karim Aly, Alexei Sharpanskykh  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.20293v1  

#### Abstract
The increasing adoption of synthetic data in aviation research offers a promising solution to data scarcity and confidentiality challenges. This study investigates the potential of generative models to produce realistic synthetic flight data and evaluates their quality through a comprehensive four-s...

---

### 28. [Lifecycle-Aware Federated Continual Learning in Mobile Autonomous Systems](https://arxiv.org/abs/2604.20745)

**Authors**: Beining Wu, Jun Huang  
**Category**: cs.LG  
**Published**: 2026-04-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.20745v1  

#### Abstract
Federated continual learning (FCL) allows distributed autonomous fleets to adapt collaboratively to evolving terrain types across extended mission lifecycles. However, current approaches face several key challenges: 1) they use uniform protection strategies that do not account for the varying sensit...

---

### 29. [Who Defines Fairness? Target-Based Prompting for Demographic Representation in Generative Models](https://arxiv.org/abs/2604.21036)

**Authors**: Marzia Binta Nizam, James Davis  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.21036v1  

#### Abstract
Text-to-image(T2I) models like Stable Diffusion and DALL-E have made generative AI widely accessible, yet recent studies reveal that these systems often replicate societal biases, particularly in how they depict demographic groups across professions. Prompts such as 'doctor' or 'CEO' frequently yiel...

---

### 30. [Propensity Inference: Environmental Contributors to LLM Behaviour](https://arxiv.org/abs/2604.21098)

**Authors**: Olli J\"arviniemi, Oliver Makins, Jacob Merizian, Robert Kirk, Ben Millwood  
**Category**: cs.AI  
**Published**: 2026-04-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.21098v1  

#### Abstract
Motivated by loss of control risks from misaligned AI systems, we develop and apply methods for measuring language models' propensity for unsanctioned behaviour. We contribute three methodological improvements: analysing effects of changes to environmental factors on behaviour, quantifying effect si...

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
