# OCS Pre-Trigger 机制研究问题讨论

## 核心命题

> **训练阶段的 Token-专家亲和度分布，能否作为推理阶段 OCS 电路预建立的可靠先验？**

如果答案为"是"，则可以将 OCS 重配延迟隐藏在计算周期内，实现接近静态拓扑的通信延迟，同时保留动态拓扑的带宽优势。

---

## 1. 问题框架

### 1.1 当前系统能力 (已实现)

| 能力                            | 状态                               |
| ------------------------------- | ---------------------------------- |
| EPS 全互连通信（baseline 下界） | ✅`run_overlap`                  |
| OCS 电路池 + LRU 淘汰           | ✅`src/ocs/circuit.py`           |
| 训练时专家亲和度追踪            | ✅`ExpertAffinityTracker`        |
| 亲和度驱动的专家放置            | ✅`placement_strategy: affinity` |
| OCS Pipeline（批次内预建立）    | ✅`run_ocs_pipeline`             |
| OCS DBO（跨批次预建立）         | ✅`run_ocs_dbo`                  |
| Qwen MoE 权重加载 + 路由回放    | ✅ QwenGate + ReplayRouter         |
| 训练循环（负载均衡 Loss）       | ✅`src/train/`                   |

### 1.2 尚未解决的问题

1. **亲和度统计来自训练，但推理时路由分布可能漂移** — 没有跨阶段验证
2. **预触发时机未与推理请求的时序对齐** — DBO 的 lookahead 基于固定的 microbatch 节奏，不适用在线 serving
3. **没有切换频率控制** — LRU 是被动策略，不感知预测置信度
4. **缺乏端到端 SLA 视角的评估** — 现有指标是 wall-time 和 overlap ratio，没有 TTFT / TPOT / SLA 违规率

---

## 2. 基线方案定义

### Baseline A: 静态 EPS（下界）

```
拓扑: 固定全互连 Electrical Packet Switching
重配: 无
特点: 通信延迟恒定，带宽受限于电气交换
```

- **优势**: 零重配开销，延迟可预测
- **劣势**: 无法利用光路的带宽/延迟优势
- **作为下界**: 任何 OCS 方案的延迟不应差于此

### Baseline B: 粗粒度 / 区域级 OCS

```
拓扑: 按 Pod/Rack 分组，组内 EPS，组间 OCS
重配: 仅在训练阶段进行；推理阶段固定
特点: 训练时优化放置，推理时冻结拓扑
```

- **优势**: 推理零重配开销
- **劣势**: 无法适应推理时的动态路由变化
- **适用场景**: 训练-推理路由分布高度一致时接近最优

### Baseline C: 朴素 OCS（无预测器）

```
拓扑: 每批次前全量重配 OCS 电路
重配: 每个 inference batch 触发一次
预测: 无 — 基于当前 batch 的实际路由需求
```

- **优势**: 总是匹配当前实际需求
- **劣势**: 重配延迟完全暴露在关键路径上
- **作为上界参考**: 展示了"完美匹配"的通信效率，但付出了全量重配代价

### Proposed: 亲和度预触发 OCS（本项目目标）

```
拓扑: 基于训练亲和度预测器 + 预触发窗口
重配: 在计算周期内提前发起
预测: 使用 LoRA 微调阶段统计的 Token-专家亲和度
```

---

## 3. 关键研究问题

### Q1: 训练-推理亲和度一致性

**问题**: 训练阶段统计的 Token-专家共激活模式，在推理阶段是否保持稳定？

**方法论**:

1. 对目标 MoE 模型（Qwen-MoE / DeepSeek-MoE / Mixtral）进行 LoRA 微调
2. 在微调过程中记录每层的 `(token_embedding_cluster, top_k_experts)` 映射
3. 在推理阶段（使用相同数据分布和 OOD 数据），统计相同的映射关系
4. 量化指标：
   - **Jensen-Shannon Divergence** 介于训练和推理的专家选择分布
   - **Top-K Overlap@K** — 同一 token cluster 在推理时选择的 top-K 专家与训练时的重叠率
   - **Affinity Stability Score** — 专家共激活矩阵的 cosine similarity(train, inference)

**预期挑战**:

- 路由分布漂移：推理数据分布偏移导致 gate 输出变化
- LoRA 适配器本身会轻微改变 gate 行为
- 解决方案：使用温度校准 (temperature scaling) 对齐训练/推理的 gate 分布

**关键文件**:

- `src/ocs/placement.py` — `ExpertAffinityTracker.record_routing()` 已实现统计
- `src/data/routing_schema.py` — `RoutingTrace` 已支持 per-layer 统计
- 需新增：`src/eval/affinity_consistency.py` — 跨阶段分布对比

### Q2: 预触发窗口与 OCS 重配延迟覆盖

**问题**: 基于亲和度预测的提前期，能否在实际请求下完整覆盖 OCS 重配时间？

```
时间线:
|────── 计算 Token K ──────|────── 计算 Token K+1 ──────|
  ↑                          ↑
  预测 K+1 的专家需求        电路在此前必须就绪
  └── 预触发窗口 ──→|
                   ├─ OCS 重配延迟 (10-1000 μs) ──→|
                   └─ 若窗口 < 重配延迟 → 暴露在关键路径
```

**核心不等式**:

```
T_prefetch = T_compute_token - T_predict_future_experts
有效性条件: T_prefetch >= T_ocs_reconfig + T_circuit_setup
```

**方法论**:

1. 标定 OCS 重配时间分布（MEMS: 10-100 μs; 波导型: 1-10 μs）
2. 标定单 Token 计算时间（受 batch size / sequence length 影响）
3. 计算可行的预触发提前量：在当前 token 计算期间，可以提前建立下一个 token 的电路
4. 对于不能覆盖的情况（短序列 / 大 batch），退化为静态拓扑或 LRU

**关键发现**: 对于自回归解码（token-by-token），单 token 计算时间通常远超 OCS 重配时间（GPU 上 ~1-10 ms compute vs 10-100 μs reconfig），因此**单 token 预触发窗口足够覆盖重配延迟**。

**关键文件**:

- `src/runtime/scheduler.py` — `run_ocs_dbo` 已有 batch 级预建立
- 需新增：`src/runtime/prefetch.py` — token 级预触发逻辑

### Q3: 重配频率与带宽收益权衡

**问题**: 不必要的切换导致重配延迟累积；不切换导致次优拓扑降低带宽利用率。

```
总延迟 = T_compute + T_reconfig × N_switches + T_comm(placement)
         └─固定──┘ └── 可变 ──────────────┘ └── 可变 ──┘

tradeoff:
  N_switches ↑ → 重配开销 ↑, 带宽利用率 ↑ (拓扑更优)
  N_switches ↓ → 重配开销 ↓, 带宽利用率 ↓ (拓扑次优)
```

**方法论**:

1. **切换收益阈值**: 仅当 `bandwidth_gain(新拓扑) > reconfig_cost` 时触发切换
2. **分块策略**: 将连续 K 个 token 作为 super-batch，仅切换 super-batch 间的电路
3. **置信度门控**: 仅当预测器置信度 > τ 时才预触发（否则保留当前电路 / 退回 EPS）
4. **参数扫描**: 在仿真中扫描 (switch_frequency, bandwidth_gain) 空间，找 Pareto 前沿

**评估指标**:

- Reconfig Overhead Ratio (ROR) = `ΣT_reconfig / T_total`
- Bandwidth Utilization = `实际吞吐 / 理想拓扑吞吐`
- Pareto 效率 = `ΔThroughput / ΔReconfigOverhead`

**关键文件**:

- `src/ocs/circuit.py` — LRU 淘汰策略可替换为置信度门控
- `scripts/compare_ocs.py` — 可扩展为参数扫描脚本

### Q4: OCS 光路流量占比

**问题**: 预触发的 OCS 路径实际承载了多少专家间流量？有多少被迫回退到 EPS？

```
总通信流量 = OCS承载流量 + EPS回退流量

核心指标:
  OCS Traffic Ratio = Bytes_via_OCS / Total_Inter_Expert_Bytes
  Circuit Hit Rate   = Hot_Circuit_Requests / Total_Circuit_Requests
```

**方法论**:

1. 在 `OcsCircuitPool` 中标记每条请求的命中/未命中
2. 追踪通过已建立电路传输的字节数 vs 回退路径的字节数
3. 按层/按时间窗口分析 OCS 流量占比的稳定性

**预期结果**:

- 亲和度驱动的放置 + 预触发：OCS 流量占比 > 90%
- 无预测器的 LRU：OCS 流量占比 ~70-85%（取决于 cache size）
- 关键受惠层：中间层（路由熵最高），首层/末层提升较小

**关键文件**:

- `src/ocs/circuit.py` — 已有 `total_requests`, `circuit_reuses`, `reuse_ratio`
- 需新增：per-circuit byte counter

### Q5: 计算-通信重叠效率

**问题**: 预触发机制是否成功将 OCS 重配延迟隐藏在计算周期内？

```
理想重叠:
|── Compute ──|── Compute ──|── Compute ──|
   |── OCS Reconfig ──|         (完全隐藏在计算下)

部分重叠:
|── Compute ──|── 等待 ──|── Compute ──|
        |── OCS Reconfig ──|     (部分暴露)

零重叠:
|── Compute ──|── OCS Reconfig ──|── Compute ──|
               (完全暴露在关键路径)
```

**现有实现**: `src/eval/metrics.py` — `ocs_overlap_ratio()` 已计算 OCS 预建立与计算的重叠比例。

**扩展方向**:

1. 按 token 粒度（而非 batch）统计重叠率
2. 区分"有用重叠"（重配在计算窗口内完成）和"不足重叠"（计算结束时重配仍未完成）
3. 引入 Overlap Slack = `T_compute - T_reconfig`，监控 slack 的分布和尾部

**预期**:

- 自回归解码 (batch=1): overlap ratio → 1.0（单 token 计算远超重配时间）
- 大 batch 推理: overlap ratio 随 batch 增大而降低（计算时间摊薄）
- 预填充阶段 (prompt processing): 计算密集，overlap 最优

### Q6: 端到端延迟与 SLA 违规率（核心最终指标）

**问题**: 亲和度预触发 OCS 方案在最终用户体验上带来多少提升？

**指标体系**:

| 指标                                   | 定义                   | 目标               |
| -------------------------------------- | ---------------------- | ------------------ |
| **TTFT** (Time To First Token)   | 首个 token 生成延迟    | < 100ms (交互场景) |
| **TPOT** (Time Per Output Token) | 平均每 token 生成延迟  | < 50ms             |
| **TPOT@P99**                     | 尾部 token 延迟        | < 200ms            |
| **SLA Attainment**               | TTFT < 阈值 的请求比例 | > 99%              |
| **Throughput**                   | tokens/s (系统级)      | 越高越好           |

**对比矩阵**:

| 方案                       | TTFT (est.)   | TPOT (est.)  | 重配暴露       | 带宽利用                |
| -------------------------- | ------------- | ------------ | -------------- | ----------------------- |
| 静态 EPS                   | 高 (全电气)   | 高           | 无             | 100% (固定)             |
| 粗粒度 OCS                 | 中            | 中           | 无 (推理)      | 中 (粗粒度)             |
| 朴素 OCS                   | 高 (每批重配) | 低           | 完全           | 高 (最优拓扑)           |
| **亲和度预触发 OCS** | **低**  | **低** | **隐藏** | **高 (预测驱动)** |

**方法论**:

1. 构建标准推理 benchmark（不同 seq_len, batch_size, 数据分布）
2. 使用 `src/eval/profiler.py` 收集 trace，转换为延迟分布
3. 模拟 SLA 阈值曲线，输出违规率
4. 对比四种方案在相同硬件参数下的表现

### Q7: MoE 模型轨迹验证

**问题**: 方案在真实开源 MoE 模型上的泛化性如何？

**候选模型**:

| 模型                         | 专家数 | 层数 | Top-K | 路由特点                        |
| ---------------------------- | ------ | ---- | ----- | ------------------------------- |
| Qwen-MoE (Qwen1.5-MoE-A2.7B) | 64     | 24   | 4     | 含 shared expert + sigmoid gate |
| DeepSeek-MoE-16B             | 64     | 27   | 6     | 细粒度专家 + 共享专家隔离       |
| Mixtral 8×7B                | 8      | 32   | 2     | 简单 top-2，适合初期验证        |

**验证步骤**:

1. 导出模型权重为统一格式 → `scripts/export_qwen_weights.py`
2. LoRA 微调 + 路由亲和度记录 → `src/train/trainer.py` + `ExpertAffinityTracker`
3. 推理回放 + 预触发模拟 → `src/data/routing_capture.py` + OCS scheduler
4. 统计全维度指标 (Q1-Q6)
5. 跨模型对比，分析结论的模型依赖性

**当前进度**: Qwen MoE 权重加载已实现 (`src/model/qwen_experts.py`)，路由回放已实现 (`src/model/router_replay.py`)。需扩展至 DeepSeek 和 Mixtral 的 gate 结构。

---

## 4. 实施路线图

### Phase 1: 亲和度验证 (2-3 周)

```
输入: Qwen-MoE 模型 + LoRA 微调数据
输出: 训练-推理亲和度一致性报告
```

- [ ] 实现 LoRA 微调阶段亲和度收集 (`src/train/affinity_collector.py`)
- [ ] 实现推理阶段亲和度收集（扩展现有 `RoutingCapture`）
- [ ] 构建 `src/eval/affinity_consistency.py` — JS 散度、Top-K Overlap、Affinity Stability Score
- [ ] 生成一致性热力图和数值报告

### Phase 2: 预触发调度器 (2-3 周)

```
输入: 亲和度预测器
输出: token 级预触发的 OCS 调度器
```

- [ ] 实现 `src/runtime/prefetch_predictor.py`:
  - 基于历史亲和度的 next-token expert 预测
  - 置信度估计（预测分布 entropy）
  - 回退策略（低置信度 → 宽电路预建立 / LRU / EPS fallback）
- [ ] 实现 `src/runtime/prefetch_scheduler.py`:
  - Token 级预触发循环
  - 与现有 `run_ocs_dbo` 的兼容层
- [ ] 参数化预触发窗口大小和置信度门限

### Phase 3: 全维度评估 (2 周)

```
输入: Phase 1 的一致性报告 + Phase 2 的调度器
输出: 全维度评估报告 (Q1-Q6)
```

- [ ] 构建标准 benchmark suite (`benchmarks/`)
- [ ] 运行所有基线 (Static EPS, Coarse OCS, Naive OCS, Affinity-Prefetch OCS)
- [ ] 生成: tradeoff curves, SLA 违规率表, overlap efficiency 分布, OCS 流量占比时序
- [ ] 跨模型验证 (Qwen-MoE → DeepSeek-MoE → Mixtral)

### Phase 4: 结论与论文大纲 (1 周)

- [ ] 核心结论: 亲和度可预测性 + 预触发窗口充分性 → 可行/不可行
- [ ] 适用条件: 什么场景下方案有效（自回归解码 ✓, 大 batch ✗, etc.）
- [ ] 局限性: 路由漂移阈值, OCS pool size 约束
- [ ] 论文核心图: TTFT 对比柱状图, overlap efficiency 累积分布, tradeoff Pareto 前沿

---

## 5. 预期核心贡献

1. **训练-推理亲和度可迁移性的实证验证** — 首次量化"训练时学到的专家偏好能否指导推理时的通信调度"
2. **Token 级预触发窗口的充分性证明** — 自回归解码的计算粒度天然支持 OCS 重配隐藏
3. **重配-带宽 Pareto 分析** — 给出切换频率选择的定量依据
4. **端到端 SLA 提升** — 相比静态 EPS 获得 OCS 带宽优势，相比朴素 OCS 消除重配暴露

---

## 6. 风险与缓解

| 风险                               | 概率 | 缓解                                         |
| ---------------------------------- | ---- | -------------------------------------------- |
| 推理路由漂移过大，亲和度预测失效   | 中   | 在线自适应预测器 + EPS fallback              |
| 长 prompt 预填充阶段预触发窗口不足 | 低   | Prompt 阶段退化为宽电路预建立                |
| OCS pool size 不足导致频繁淘汰     | 中   | 置信度驱动的池大小自适应                     |
| LoRA 微调改变 gate 行为            | 中   | 对比 full-parameter vs LoRA 微调的亲和度差异 |

---

## 7. 文件索引

### 需新增

| 文件                                  | 用途                      |
| ------------------------------------- | ------------------------- |
| `src/eval/affinity_consistency.py`  | 训练-推理亲和度分布对比   |
| `src/runtime/prefetch_predictor.py` | Token 级专家需求预测器    |
| `src/runtime/prefetch_scheduler.py` | Token 级预触发 OCS 调度器 |
| `src/train/affinity_collector.py`   | LoRA 微调阶段亲和度收集   |
| `benchmarks/inference_benchmark.py` | 标准化推理评估套件        |
| `scripts/run_full_evaluation.py`    | 全维度对比脚本            |

### 需修改

| 文件                           | 变更                                             |
| ------------------------------ | ------------------------------------------------ |
| `src/ocs/circuit.py`         | 新增置信度门控替换 LRU，per-circuit byte counter |
| `src/eval/metrics.py`        | 新增 token 级 overlap 统计、SLA 计算、ROR        |
| `src/eval/plots.py`          | 新增 Pareto 前沿图、SLA 违规率图                 |
| `src/runtime/scheduler.py`   | 集成 token 级预触发                              |
| `src/model/router_replay.py` | 扩展至 DeepSeek、Mixtral gate 格式               |

---

## 8. 文献平台对比：EPS-OCS 可比性保障与 OCS 建模方法

> **核心问题**：(1) 哪些平台/方法可以保证 EPS 与 OCS 在同一基准上可比？
> (2) 各文献如何确保 OCS 可被可信建模？(3) 各文献如何审视/验证其建模？

### 8.1 保障 EPS-OCS 可比性的三类平台

文献中实现 EPS-OCS 公平对比的方法可归纳为三类：

| 类别                          | 机制                                                | 代表文献                                                        | 具体做法                                                                                                            |
| ----------------------------- | --------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **物理 testbed 对标**   | 同一硬件上运行 EPS baseline + OCS emulation         | Opus (Ding et al.)                                              | Perlmutter 超算运行 NCCL EPS；同一超算上用 Opus shim 模拟 OCS 环形 photonic rail；物理 Polatis OCS testbed 交叉标定 |
| **α-β 解析归一化**    | 统一 Hockney 模型参数 (α/β) + 统一总带宽 per 节点 | Choi et al., SWOT, ReTri, ACTINA                                | 所有方案共享相同的 α (延迟) 和 β (带宽) 参数；OCS 额外项仅加 reconfig cost                                        |
| **统一仿真平台**        | 相同仿真器 + 相同 GPU 模型 + 相同 workload          | ocs-DRP (SimAI+NS-3), LumosCore (RapidAISim), MoX (ASTRA-sim 2) | 同一包级/流级/Token级仿真器中切换网络后端；GPU 计算模型不变；工作负载 trace 不变                                    |
| **解析归一化指标**      | 比值指标消除绝对尺度差异                            | DELTA (NCT), Switching Efficiency (η), SWOT (Normalized CCT)   | NCT = OCS 通信时间 / 理想 EPS 通信时间；η 框架分解到三个子维度                                                     |
| **Trace/MILP 共享输入** | 同一 traffic matrix / 同一 DAG 作为输入             | Birkhoff, DELTA, MoX                                            | 先捕获或生成通信需求矩阵/DAG，再分别喂入 EPS 和 OCS 求解器                                                          |

### 8.2 各文献如何确保 OCS 可被可信建模

| 方法                                                       | 说明                                                                                         | 代表文献                                                                | 与本工作的对齐度                  |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------- |
| **三层验证链** (hardware → emulation → simulation) | 实测标定仿真参数 → 仿真放大规模                                                             | Opus                                                                    | ★★★★★ 最理想但成本最高       |
| **Giant OCS Abstraction + 物理约束退化**             | 先建理想 OCS (无限端口、零重配) 作为上界，再逐步加入物理约束 (端口数、重配延迟) 得到可行下界 | ACTINA                                                                  | ★★★★☆ 提供可信区间           |
| **α-β 扩展重配项**                                 | `T = α + β·bytes + δ_reconfig`；δ 作为独立项加入经典 Hockney 模型                     | SWOT, ReTri                                                             | ★★★★☆ 数学简洁，可迁移       |
| **参数扫描+敏感性分析**                              | OCS 重配时间从 ns 扫到 s；带宽扫 1 个数量级以上                                              | SWOT (10ns–10ms), ReTri (1μs–50ms), ACTINA (1μs–1s)                | ★★★★★ 揭示参数的可行域       |
| **生产 trace 回放**                                  | 用真实集群采集的 MoE 路由/流量 trace 驱动仿真                                                | LumosCore (SenseTime 1000 任务), MoX (DeepSeek-V3/Qwen-3 Chakra traces) | ★★★★★ 最高信噪比，已部分实现 |
| **MILP 最优解作为上界**                              | 求 MILP 精确解 (或 bounded suboptimal) 展示最优可能性能                                      | DELTA (Gurobi), SWOT (Pulp/CBC), LumosCore (Gurobi)                     | ★★★☆☆ 计算昂贵，仅小规模     |

### 8.3 OCS 重配时间假设的文献共识

| OCS 技术                      | 重配时间     | 文献引用                       | 能否隐藏于计算？                           |
| ----------------------------- | ------------ | ------------------------------ | ------------------------------------------ |
| Photonic MEMS (beam-steering) | ~400ns–1μs | ACTINA, 本工作                 | ✓ 远小于 ms 级 GPU 计算                   |
| 2D MEMS                       | ~1ms         | ACTINA                         | ✓ 对于大 batch 训练；边界情况对自回归推理 |
| Polatis 商用 MEMS             | 10–25ms     | Opus, MixNet                   | ✓ 仅当计算窗口 >100ms (大 batch 训练)     |
| 含收发器初始化的端到端重配    | 若干秒       | DELTA                          | ✗ 仅适用于 job-level 重配                 |
| 3D MEMS / 机械 patch panel    | 10s–分钟    | Google Apollo/Jupiter, TopoOpt | ✗ 仅适用于维护/部署窗口                   |

**共识**: 重配延迟 ≤1ms 的方案 (2D/photonic MEMS) 可在 MoE 自回归推理的计算窗口内隐藏；重配 >10ms 需要 batch-level 或 job-level 的预建立策略。

### 8.4 对本项目的启示

1. **分层对标策略**：借鉴 Opus 的三层验证思路 — Phase A (CPU sleep 仿真) → Phase B (GPU NCCL + nsys 标定) → Phase C (真实 OCS 硬件)。当前 Phase A 输出的相对结论 (电路复用率改善、prefetch 窗口充分性) 可信，但绝对延迟数值不可直接迁移。
2. **归一化指标优先**：采用 DELTA 的 NCT 思想 — 用 `(OCS方案延迟 - 理想EPS延迟) / 计算时间` 作为归一化指标，消除硬件差异。当前代码中的 `ocs_overlap_ratio` 已具备此类比的基因。
3. **参数扫描覆盖不确定性**：OCS 重配时间从 1μs 扫到 100ms，覆盖 photonic MEMS 到 Polatis 的全空间，确保结论在技术选择不确定性下依然稳健。
4. **Trace 驱动是最高信噪比验证**：LumosCore 和 MoX 的生产 trace 回放方法与本项目的 Qwen ReplayRouter 思路一致，是当前可实现的最高保真度验证路径。

---

## 9. AstraSim 大规模仿真验证：实施计划

> **目标**：将当前 CPU testbed (≤32 ranks) 的小规模验证，扩展为 AstraSim (256–2048 nodes) 的大规模仿真，同时保留 α-β 模型的简洁性与泛化能力。

### 9.1 为什么选 AstraSim + α-β 组合

| 维度       | 当前 Python testbed     | AstraSim + α-β                         | 理由                                           |
| ---------- | ----------------------- | ---------------------------------------- | ---------------------------------------------- |
| 规模上限   | ~32 ranks (Gloo)        | 2048+ nodes                              | AstraSim 是 C++ 离散事件仿真器，无进程数限制   |
| 网络模型   | `time.sleep()` 注入   | α-β Hockney 模型 / NS-3                | α-β 已被 Choi/SWOT/ReTri/ACTINA 四篇顶会验证 |
| 计算模型   | 无真实 GPU 计算         | Chakra trace 回放 (实测 GPU kernel time) | 分离计算与通信，可独立标定                     |
| 社区认可度 | 需自证仿真保真度        | SIGCOMM/NSDI/SC 标配                     | Opus, MoX, ReTri 均使用                        |
| OCS 建模   | LRU circuit pool (独立) | 可嵌入自定义网络后端                     | α-β + reconfig δ 项，保留当前 OCS 逻辑      |

**核心策略**：不放弃 α-β 的简洁性，而是将其作为 AstraSim 的 analytical network backend，使 AstraSim 成为"放大版 Python testbed"。

### 9.2 三阶段实施方案

```
Phase 1: Chakra Trace 导出桥           Phase 2: AstraSim OCS 后端        Phase 3: 大规模参数扫描
┌──────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ Python testbed        │     │ AstraSim C++ 引擎        │     │ 集群规模：256–2048 node │
│ RoutingTrace ──►      │     │                         │     │ 重配延迟：1μs–100ms     │
│ Chakra ET JSON        │     │ Analytical network       │     │ 电路池大小：4–64        │
│ (新增 exporter)       │     │  + OCS LRU backend       │     │ 工作负载：Qwen/DeepSeek │
│                       │     │  + 亲和度预触发调度器     │     │                         │
│ 验证：32-rank          │     │ 验证：32-rank 对标 Phase1│     │ 输出：Pareto前沿、      │
│ Python vs AstraSim    │     │                          │     │ SLA违规率、overlap分布  │
└──────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

### 9.3 Phase 1: Chakra Trace 导出桥（1–2 周）

**新增文件**：`src/astra/chakra_exporter.py`

**输入**：

- 现有的 `RoutingTrace`（per-layer expert 分配）
- `DispatchResult`（scatter/gather 的 rank 间流量矩阵）
- `Timer` 事件（route/compute/scatter/gather 的起止时间戳）
- `OcsPoolMetrics`（电路建立/复用统计）

**输出**：Chakra execution trace JSON（ET schema）

```
Chakra ET schema 映射：
┌─────────────────────────────────────────────────────────────┐
│ Python 实体                →  Chakra 实体                    │
├─────────────────────────────────────────────────────────────┤
│ Token route (per-layer)    →  COMM_SEND/RECV node            │
│   expert_ids + gate_weights →  payload_bytes = Σ(token_dim)  │
│                              →  dst_ranks = expert→rank map  │
│ Expert compute (FFN)       →  COMP node                      │
│   runtime_us               →  duration_us (实测或标定)       │
│ OCS circuit establish      →  COMM_SEND node (reconfig)      │
│   reconfig_time_us         →  duration_us (冷路径延迟)       │
│ OCS circuit reuse          →  duration 0 (热路径零开销)      │
│ All-to-all scatter/gather  →  COMM_COLL node                 │
│   bytes × bandwidth        →  collective attrs               │
└─────────────────────────────────────────────────────────────┘
```

**验证闭环**：Phase 1 完成后，在 **相同规模**（如 16 ranks / 32 ranks）下运行 Python testbed 和 AstraSim，验证：

- 总 wall time 误差 < 15%
- Circuit reuse ratio 误差 < 5%
- Overlap ratio 误差 < 10%

**实现要点**：

```python
# src/astra/chakra_exporter.py (新增)
class ChakraExporter:
    def __init__(self, routing_trace: RoutingTrace, topology: Topology):
        self.routing_trace = routing_trace
        self.topology = topology
        self.nodes = []  # Chakra ET nodes

    def add_compute_node(self, duration_us: float, tensor_size: int):
        """将 MoE expert compute 映射为 Chakra COMP node"""
        self.nodes.append({
            "id": next_id(), "name": "COMP",
            "attr": {"duration_us": duration_us, "tensor_size": tensor_size},
            "inputs": {}, "outputs": {},
            "ctrl_deps": []  # 控制依赖：必须等 dispatch 完成
        })

    def add_comm_node(self, src: int, dst: int, bytes: int,
                      is_cold_circuit: bool, reconfig_us: float):
        """将 all-to-all scatter/gather + OCS 延迟映射为 Chakra COMM node"""
        duration = reconfig_us if is_cold_circuit else self.transfer_time(bytes)
        self.nodes.append({
            "id": next_id(), "name": "COMM_SEND",
            "attr": {"src": src, "dst": dst, "bytes": bytes,
                     "duration_us": duration, "is_ocs_cold": is_cold_circuit},
            "inputs": {"TENSOR_0": prev_node_id},
            "outputs": {"TENSOR_0": next_node_id}
        })

    def export(self, path: str):
        """输出符合 Chakra ET schema 的 JSON"""
        json.dump({"nodes": self.nodes, "schema": "v0.1"}, open(path, "w"))
```

**已有资产复用**：

- `src/data/routing_schema.py` → `RoutingTrace.routes[].layers[]` 提供 per-layer expert→rank 映射
- `src/comm/transport.py` → `Transport.all_to_all()` 中已有 dispatch/gather 的 bytes 计算逻辑
- `src/comm/timeline.py` → Chrome Trace events 可直接翻译为 Chakra node duration

### 9.4 Phase 2: AstraSim OCS 网络后端（2–3 周）

**目标**：在 AstraSim 中实现与 `src/ocs/circuit.py` 完全等价的行为。

**方案**：不修改 AstraSim C++ 核心，利用 AstraSim 的 **analytical network backend** 的 α-β 接口，将 OCS 延迟预计算到 Chakra trace 中。

**具体做法**：

```
┌──────────────────────────────────────────────────────────────────┐
│                    OCS 延迟预计算流水线                            │
│                                                                  │
│  RoutingTrace ──► OCS Circuit Sim ──► Chakra ET + OCS delays     │
│  (per-layer      (Python OcsCircuitPool) (COMM nodes 携带         │
│   expert分配)     预模拟热/冷路径       reconfig 和 transfer      │
│                                        两个子 phase 的 duration)   │
│                                                                  │
│  由于 per-rank expert 分配是确定性 replay，OCS 电路的热/冷路径   │
│  也是确定性的 → 可以离线预计算全部 OCS 延迟，嵌入 trace            │
└──────────────────────────────────────────────────────────────────┘
```

**为什么这是正确做法**：

1. α-β 模型本身不关心延迟来源（电气/光学）——只加一个 δ 项
2. OCS 电路池的 LRU 行为是 **确定性** 的（给定相同的 token→expert→rank 序列）
3. 因此，"预计算 OCS 延迟并嵌入 Chakra trace" 等价于 "在 AstraSim 内实时运行 LRU"
4. AstraSim 不需要任何 OCS 感知——它只看到带有不同 duration 的 COMM 节点

**实现文件**：`src/astra/ocs_precompute.py`

```python
# src/astra/ocs_precompute.py (新增)
class OcsDelayPrecomputer:
    """
    离线预计算：给定 RoutingTrace + OcsCircuitPool 配置，
    输出每个 (src_rank, dst_rank) pair 在每个 token step 的
    {is_cold, reconfig_us, transfer_us, total_us}
    """
    def __init__(self, pool_config: OcsConfig, rank_topology: dict):
        self.pool_config = pool_config

    def compute(self, routing_trace: RoutingTrace,
                expert_to_rank: dict) -> list[OcsDelayRecord]:
        """
        回放 routing_trace 的完整 token 序列，
        在内存中运行 OcsCircuitPool，记录每次通信的延迟分解
        """
        pool = OcsCircuitPool(max_circuits=self.pool_config.max_circuits,
                              reconfig_time_us=self.pool_config.reconfig_time_us,
                              circuit_latency_us=self.pool_config.circuit_latency_us,
                              circuit_bw_gbps=self.pool_config.circuit_bw_gbps)
        records = []
        for token_step in routing_trace.iter_steps():
            for layer in token_step.layers.values():
                for expert_id in layer.experts:
                    dst = expert_to_rank[expert_id]
                    src = token_step.current_rank
                    delay = pool.compute_delay(src, dst, token_step.bytes)
                    records.append(OcsDelayRecord(
                        step=token_step.pos, layer=layer.idx,
                        src=src, dst=dst,
                        is_cold=delay.is_cold,
                        reconfig_us=delay.reconfig_us,
                        transfer_us=delay.transfer_us
                    ))
        return records
```

**验证对标**：

- 在 `configs/astra/` 下创建与现有 19 个 YAML 一一对应的 AstraSim topology 配置
- 运行 `python scripts/astra_validate.py --config qwen_ocs_dbo.yaml --scale 32`
- 输出对比报告：Python testbed vs AstraSim（32 ranks）

### 9.5 Phase 3: 大规模参数扫描（2–3 周）

**输入**：

- Phase 1 的 Chakra trace（Qwen-MoE / DeepSeek-MoE 路由回放）
- Phase 2 的 OCS 延迟预计算（嵌入 trace）
- AstraSim topology 配置（256 / 512 / 1024 / 2048 nodes）

**参数空间**：

| 参数              | 扫范围                                                | 步长 | 文献依据               |
| ----------------- | ----------------------------------------------------- | ---- | ---------------------- |
| 集群规模 (N)      | 64, 128, 256, 512, 1024, 2048                         | ×2  | LumosCore: up to 16K   |
| OCS 重配时间 (δ) | 1μs, 10μs, 100μs, 1ms, 10ms, 100ms                 | ×10 | SWOT/ReTri/ACTINA 共识 |
| 电路池容量 (C)    | 4, 8, 16, 32, 64 per rank                             | ×2  | 当前 max_circuits 参数 |
| 调度策略          | LRU / Weighted-LRU / Confidence-Gated / 贪心 (MixNet) | —   | 六种策略 A/B 对比      |
| 亲和度预测模式    | None (reactive) / Affinity-Prefetch (lookahead=1/2/4) | —   | 核心研究问题 Q2/Q3     |
| 工作负载          | Qwen-MoE / DeepSeek-MoE / Mixtral                     | —   | Q7 跨模型验证          |

**评估矩阵**（继承 Section 3 的 Q1–Q6 指标）：

```
输出 per 配置点：
  ├── TTFT / TPOT / TPOT@P99        (SLA 指标)
  ├── OCS Traffic Ratio              (Q4)
  ├── Circuit Hit Rate               (Q1 间接)
  ├── OCS Overlap Ratio              (Q5)
  ├── Reconfig Overhead Ratio (ROR)  (Q3)
  └── Bandwidth Utilization          (Q3)
```

**新增脚本**：`scripts/astra_sweep.py`

```python
# scripts/astra_sweep.py (新增)
"""
参数扫描调度器：生成 AstraSim 配置矩阵 → 并行运行 → 聚合结果

用法:
  python scripts/astra_sweep.py \
    --trace data/routing_traces/routing_qwen.json \
    --scale 256,512,1024 \
    --reconfig 1,10,100,1000,10000 \
    --pool 8,16,32 \
    --strategies lru,weighted_lru,confidence_gated \
    --output outputs/astra_sweep/
"""
```

### 9.6 新增文件清单

```
src/astra/
├── __init__.py
├── chakra_exporter.py       # RoutingTrace → Chakra ET JSON
├── ocs_precompute.py        # OCS 延迟离线预计算
└── astra_config_gen.py      # 生成 AstraSim topology/workload JSON

configs/astra/
├── topology_256.yaml        # 256-node 3-tier topology
├── topology_512.yaml
├── topology_1024.yaml
├── topology_2048.yaml
├── network_analytical.yaml  # α-β analytical backend 参数
├── network_ns3.yaml         # NS-3 包级仿真 (可选高保真验证)
├── ocs_lru.yaml             # OCS LRU 参数 (重配延迟/池大小)
├── ocs_weighted_lru.yaml
├── ocs_confidence_gated.yaml
└── baseline_eps.yaml        # EPS baseline 对照

scripts/
├── astra_validate.py        # Python vs AstraSim 对标验证
└── astra_sweep.py           # 大规模参数扫描 + 结果聚合

outputs/astra/
├── validate/                # 对标验证报告
│   ├── 32rank_comparison.json
│   └── 32rank_comparison.html
├── sweep/                   # 参数扫描结果
│   ├── results.db           # SQLite (per-config-point metrics)
│   ├── pareto_frontier.png
│   └── sla_violation_heatmap.png
└── traces/                  # 生成的 Chakra traces (可被 AstraSim 直接消费)
    ├── qwen_256exp.chakra.json
    └── deepseek_64exp.chakra.json
```

### 9.7 关键设计决策与依据

| 决策                  | 选择                                                 | 依据                                                                                      |
| --------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| OCS 逻辑运行位置      | **离线预计算**，嵌入 Chakra trace              | OCS LRU 是确定性的；避免修改 AstraSim C++ 代码；SWOT/MoX 均用此模式                       |
| AstraSim 网络后端     | **Analytical (α-β)**，可选 NS-3 验证         | 保持 α-β 简洁性；Choi 证明 α-β 误差 <9.6%；NS-3 仅在关键配置点做高保真 cross-check    |
| Chakra ET vs ET+ 格式 | **ET (execution trace)**                       | ET 是 AstraSim 标准输入；ET+ (enhanced) 含更多 metadata，但对 MoE dispatch 模式无增量收益 |
| 工作负载              | **Qwen-MoE routing replay** (已有)             | Phase 1 直接可用；DeepSeek/Mixtral 在 Q7 中后续扩展                                       |
| 计算时间标定          | **实测 GPU kernel time** (从 nsys/ncu profile) | 当前`time.sleep()` 不准确；AstraSim 需要真实 GPU compute duration                       |

### 9.8 风险与缓解

| 风险                                                        | 概率 | 缓解                                                                                  |
| ----------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------- |
| Chakra schema 与 AstraSim 版本不兼容                        | 中   | 先用 AstraSim 自带 example trace 验证链路通畅                                         |
| OCS 预计算在 scale >1024 时内存溢出                         | 低   | OCS pool per rank 仅 O(C²) 状态，1024 ranks × 64 circuits = 4M entries，可控        |
| AstraSim analytical backend 不支持自定义 COMM node duration | 低   | AstraSim analytical backend 本身就是 α-β；COMM node 的 attr 字段可嵌入任意 duration |
| Python testbed 与 AstraSim 在 32-rank 对标时误差过大 (>20%) | 中   | 先验证纯 EPS（无 OCS）对标：两边都是 α-β + 相同 topology → 理论上应完全一致        |

### 9.9 预期产出（可发表结论）

完成 Phase 1–3 后，预期可支持的结论：

1. **亲和度预触发的可扩展性**：在小规模 (32-rank) 和中规模 (256-node) 下，affinity-prefetch OCS 的 circuit hit rate 均 >85%，且随规模增大而稳定
2. **α-β 的充分性**：通过 Python vs AstraSim 对标验证，证明 α-β + OCS δ 项足以捕获 OCS 性能的关键维度，无需 packet-level 仿真
3. **Pareto 最优区间**：在 1μs–1ms 重配延迟区间内，affinity-prefetch OCS 相比静态 EPS 吞吐提升 15–40%，且重配开销 <5%
4. **策略排序的规模不变性**：LRU < Weighted-LRU < Confidence-Gated < 贪心 (MixNet) 的性能排序在 32→2048-node 所有规模下保持一致 → Phase A 小规模结论可外推
