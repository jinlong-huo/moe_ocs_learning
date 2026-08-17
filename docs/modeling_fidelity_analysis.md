# EPS/OCS 建模保真度分析与文献对齐

## 核心问题

> (1) 当前 EPS/OCS 建模与学术界/工业界文献的异同？
> (2) 建模可信度如何？能否作为真实 MoE 权重 + 物理集群部署的前置验证？

---

## 1. 当前建模架构总览

### 1.1 整体仿真管线

```
  ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────────┐
  │  Router  │ →  │  Transport   │ →  │  dist.all_to │ →  │  Expert       │
  │ (gate)   │    │ (delay inj.) │    │  _all (Gloo) │    │  Compute      │
  └──────────┘    └──────┬───────┘    └──────────────┘    └───────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Flat Delay    Topology-Aware   OCS Circuit Pool
    (us + jitter) (3-tier hier.)   (LRU + reconfig)
```

三种延迟注入模式互斥（优先级：OCS > Topology > Flat）。延迟在 `dist.all_to_all_single` 调用前通过 `time.sleep()` 注入。

### 1.2 EPS 建模细节

**Flat Delay 模式** (`src/comm/transport.py:99-105`):

```python
jitter = random.uniform(-comm_delay_jitter_us, comm_delay_jitter_us)
total = max(0.0, comm_delay_us + jitter)
time.sleep(total / 1_000_000.0)
```

- 所有 rank pair 等延迟
- Jitter 从均匀分布采样（非高斯噪声）

**Topology-Aware 模式** (`src/comm/topology.py:141-188`):

```
delay = (tier_latency + bytes / (tier_bw * 1000)) * delay_multiplier
```

- 3 层拓扑：Intra-node (NVLink, 1μs/900GBps) → Intra-pod (IB, 3μs/200GBps) → Cross-pod (10μs/100GBps)
- 计算 worst-tier 延迟（取 all-to-all 参与 rank 中最慢的 tier）
- 字节量按 `tensor.numel() * element_size()` 计入带宽分量

### 1.3 OCS 建模细节

**电路池** (`src/ocs/circuit.py`):

```
OcsCircuitPool:
  max_circuits: int          # 每 rank 最大同时电路数
  reconfig_time_us: float    # 冷路径建立延迟 (1-1000)
  circuit_latency_us: float  # 光路基础延迟 (1-5)
  circuit_bw_gbps: float     # 光路带宽 (200)
  _circuits: OrderedDict     # LRU 缓存 (key=(src,dst))
```

**核心操作**:

```
establish(src, dst):
  if key in pool:
    move_to_end(key)          # 提升至 MRU
    return 0.0                # 热路径，零开销
  else:
    if pool_full:
      popitem(last=False)     # 淘汰 LRU
      evictions += 1
    insert(key, OcsCircuit)   # 冷路径
    return reconfig_time_us

compute_delay(src, dst, bytes):
  reconfig = establish(src, dst)
  transfer = circuit_latency_us + bytes / (bw_gbps * 1000)
  return reconfig + transfer
```

**预建立模式** (`src/runtime/scheduler.py`):

- `ocs_pipeline`: post-route → `pre_establish_circuits(target_ranks)` → scatter
- `ocs_dbo`: batch K 的计算期间 pre-establish batch K+1 的电路（一步前瞻）

---

## 2. 与文献的关键对比

### 2.1 对标文献矩阵

| 维度               | 本工作                       | TopoOpt (NSDI'23)       | MixNet/mFabric (SIGCOMM'25) | ACTINA (SC'25)   | Google Apollo/Jupiter |
| ------------------ | ---------------------------- | ----------------------- | --------------------------- | ---------------- | --------------------- |
| **OCS 粒度** | per-rank 电路池              | 跨作业拓扑优化          | 区域级 (per-EP-block)       | 理论建模         | Spine-Leaf 替换       |
| **重配时机** | 每 micro-batch / 前瞻        | 仅作业启动前            | 每次迭代 (训练中)           | 分析框架         | ~10s 级 (维护窗口)    |
| **重配延迟** | 1–1000 μs (可配置)         | 秒级 (机械 patch panel) | <25ms (Polatis MEMS)        | 理论分析         | ~10s (3D MEMS)        |
| **调度策略** | **LRU**                | 单次全局优化 (ILP)      | 贪心瓶颈感知                | NP-Complete 证明 | 二部图匹配 (Orion)    |
| **预测机制** | 亲和度追踪 (co-activation)   | 无 (静态流量模式)       | MixNet-Copilot (条件概率)   | N/A              | Rail-Aligned 路由     |
| **部署形态** | CPU 仿真 (Gloo + time.sleep) | Telescent 硬件 + RDMA   | 32×A100 硬件原型           | 仿真             | 147K TPU 生产集群     |
| **MoE 支持** | ✓ (top-K routing)           | ✗ (Dense DNN)          | ✓ (核心场景)               | N/A              | ✗ (Dense Gemini)     |

### 2.2 关键差异深度分析

#### 差异 1: LRU vs 贪心/ILP 电路调度

**我们的做法**: 每个 rank 独立维护 `OrderedDict` LRU 缓存。冷电路淘汰最近最少使用的连接。

**MixNet 的做法** (SIGCOMM 2025): 全局贪心算法 — 识别 bottleneck server pair，优先分配 OCS 电路给通信量最大的 rank pair。不是 LRU，而是 **demand-driven per-iteration allocation**。

**TopoOpt 的做法** (NSDI 2023): ILP 求解器全局优化拓扑（单次），不考虑 per-iteration 重配。

**本质差异**:

- LRU 是 **reactive** 的 — 假设过去的使用模式会延续
- 贪心/ILP 是 **proactive** 的 — 每次基于当前需求显式求解
- LRU 的优势: O(1) 决策，零规划开销
- 贪心/ILP 的优势: 理论上最优（对于当前 batch），但每次需重新计算

**适用性判据**:

- 当 token-专家亲和度在连续 batch 间 **高度稳定** → LRU 近似最优（热路径命中率高）
- 当路由分布 **快速变化** → LRU 可能持有错误电路，贪心重分配优于 LRU

#### 差异 2: Per-Rank 独立池 vs 全局 Fabric 约束

**我们的做法**: 每个 rank 有独立的 `max_circuits` 池，rank A→B 的建立不影响 rank C→D。

**现实 OCS 硬件**: OCS 是一个 **共享交换矩阵**（如 136×136 MEMS mirror array）。所有 rank 的电路共享同一个物理 fabric：

- 一个输入端口只能连接一个输出端口（单播约束）
- 重新配置一条路径可能影响其他路径（cross-talk in MEMS mirror settling）
- 端口耗尽时需全局协调

**MixNet 的处理**: 使用 Polatis 576×576 OCS，分区给不同的 server group，组内独立调度。

**影响**: 我们的模型在小规模下（≤32 ranks, ≤32 circuits/rank）近似合理，因为 OCS port count 通常远超活跃电路数。但在大规模下（数百 ranks），per-rank 独立假设可能高估电路可用性。

#### 差异 3: time.sleep() 延迟注入 vs 真实网络动力学

**我们的做法**: 在 `dist.all_to_all_single` 前同步 sleep。

**问题**:

1. **无排队/拥塞建模**: 真实 EPS 中，多个 all-to-all 流同时竞争同一交换机端口 → 排队延迟
2. **无 cut-through 效应**: 真实交换机在收到整个 packet 前就开始转发（cut-through switching），而 sleep 模拟的是 store-and-forward
3. **Gloo 本身有真实延迟**: CPU 上的 Gloo all-to-all 有实际 TCP 延迟，与 sleep 叠加但不可控
4. **无流量突发**: 多个 micro-batch 的 scatter 可能同时到达同一交换机

**文献做法**:

- TopoOpt: 使用 FlexNet 和 FlexNetPacket（packet-level 仿真器），建模交换机缓冲区、排队、流控
- MixNet: 使用真实 Polatis OCS + 100G Ethernet 硬件测量，不依赖仿真

#### 差异 4: 亲和度预测器 vs MixNet-Copilot

**我们的做法**: `ExpertAffinityTracker` 记录 co-activation 计数，贪心聚类放置专家。预测是 **spatial** 的（哪些专家经常一起被选中），用于 placement。

**MixNet-Copilot**: 基于相邻层专家负载的条件概率预测。预测是 **temporal** 的（下一层/下一次迭代可能需要哪些专家 pair），用于 per-iteration circuit allocation。

**互补性**: 我们的 spatial 亲和度可用于专家放置（减少跨 rank 通信），MixNet-Copilot 的 temporal 预测可用于电路预建立。两者可以结合。

#### 差异 5: OCS 重配时间量级假设

| 来源          | 重配时间                         | 技术                         | 在训练中隐藏？              |
| ------------- | -------------------------------- | ---------------------------- | --------------------------- |
| 我们          | 1–1000 μs                      | MEMS (beam-steering) 假设    | 是 (μs 级远小于 ms 级计算) |
| MixNet        | 10–25 ms                        | Polatis MEMS                 | 是 (>100ms 计算窗口)        |
| TopoOpt       | ~1s                              | Telescent 机械 patch panel   | 否 (仅在作业前)             |
| Google Apollo | ~10s                             | 3D MEMS (136×136→300×300) | 否 (仅在维护/部署时)        |
| ACTINA 总结   | 1ms (2D MEMS) – 200ms (3D MEMS) | 多种技术                     | 取决于技术选择              |

**关键**: 我们的 1–1000 μs 假设对应 **2D MEMS 或 photonic MEMS**（ACTINA 数据: 2D MEMS ~1ms, Photonic MEMS ~400ns）。这是合理的 OCS 技术选择。但如果我们目标对标 MixNet 所用的商用 Polatis 设备（~10–25ms），则需大幅上调 `reconfig_time_us`。

---

## 3. 可信度评估

### 3.1 分层可信度矩阵

| 组件                                                              | 可信度     | 依据                                                              |
| ----------------------------------------------------------------- | ---------- | ----------------------------------------------------------------- |
| **MoE 计算流程** (route→scatter→compute→gather→combine) | ★★★★★ | 与 5 个生产框架结构完全对齐 ([docs/alignment.md](docs/alignment.md) 已验证)                 |
| **Router 架构** (Linear gate + top-K)                       | ★★★★☆ | 缺 softmax in top-2, 缺 capacity factor, 但核心正确               |
| **Expert 架构** (FFN GELU / SwitchGLU)                      | ★★★★★ | 与 Megatron/Tutel/Megablocks 一致，Qwen 权重可直接加载            |
| **All-to-All 通信模式**                                     | ★★★★☆ | 使用`all_to_all_single`，但缺 variable split sizes (非均匀分布) |
| **EPS 延迟建模** (3-tier topology)                          | ★★★☆☆ | 层级结构正确，但缺排队/拥塞/流控建模                              |
| **OCS 电路池 LRU**                                          | ★★★☆☆ | LRU 策略未在文献中验证，但作为一阶近似合理                        |
| **OCS 重配延迟** (1–1000 μs)                              | ★★★★☆ | 与 2D MEMS / photonic MEMS 物理参数一致                           |
| **预建立/DBO 管线**                                         | ★★★★☆ | 与 MixNet 的 "hidden behind compute" 逻辑一致                     |
| **时间度量** (per-event ns timer)                           | ★★★★★ | 真实时间测量，非离散事件仿真                                      |

### 3.2 不可转移的假设

以下结论如果在当前 testbed 上得出，**不能直接搬移到 GPU 集群**:

1. **绝对延迟数值**: `time.sleep()` + Gloo 的绝对延迟与 NCCL + NVLink/IB 完全不同
2. **Overlap ratio 的绝对值**: CPU 多进程的并发特性 ≠ GPU CUDA stream 并发
3. **OCS 电路复用率**: Per-rank 独立池假设在大规模下可能过于乐观

### 3.3 可以转移的结论

以下结论即使在 CPU testbed 上得出，**对 GPU 集群也有参考价值**:

1. **OCS 热/冷路径的相对收益**: 电路复用率从 70%→95% 的相对改善量级可迁移
2. **预触发窗口的充分性**: 只要 `T_compute >> T_reconfig` 成立（自回归解码天然满足），预触发的可行性判断可迁移
3. **亲和度一致性的统计特性**: Token-专家共激活的 JS 散度、Top-K overlap 等统计量仅依赖路由分布，不依赖硬件
4. **重配频率 vs 带宽的 Pareto 前沿形状**: 权衡的定性关系不依赖底层传输介质
5. **调度策略的比较排序**: 如果 A 方案在仿真中优于 B 方案，在实际系统中方向相同（但量级可能不同）

---

## 4. 与文献的对齐建议

### 4.1 LRU 策略的调整方向

**当前问题**: 纯 LRU 没有考虑到电路的"价值"差异。一个被频繁使用的关键电路和一个偶尔使用的电路，在 LRU 中仅仅通过访问时间区分。

**调整方案 1: Weighted-LRU（加权 LRU）**

```
eviction_score(circuit) = last_used_time + λ / traffic_volume
```

- 高流量电路即使最近未使用也不易被淘汰
- λ 控制流量权重 vs 时间权重

**调整方案 2: Confidence-Gated Establishment（置信度门控）**

```
if affinity_confidence(expert_a, expert_b) < τ:
    fallback_to_EPS(src, dst)           # 不浪费电路给低置信度预测
else:
    pre_establish_circuit(src, dst)     # 高置信度才建立
```

- 与 MixNet-Copilot 的条件概率预测逻辑一致
- 门限 τ 变为可调超参数

**调整方案 3: Hybrid — 分池管理**

```
pool = {
    pinned_circuits:   固定电路 (top-3 常用 rank pair, 永不被淘汰)
    dynamic_circuits:  动态电路 (LRU / 贪心分配)
}
```

- Google Orion 使用类似逻辑 — 基础拓扑固定 + 动态增量调整
- Pinned circuits 基于训练阶段统计的最常用 rank pair

### 4.2 OCS 延迟模型的扩充

**当前**: 仅建模 per-circuit 延迟。`max_delay_us` 取所有 target_rank 中的最大值。

**建议扩充**:

1. **Fabric-level port constraint**: `global_port_limit` — 当所有 rank 的活跃电路总数超过 OCS 端口数时，触发排队
2. **Reconfig grouping**: 多个电路同时 reconfig 可以并行（不同 mirror），但共享一个 reconfig budget
3. **Bidirectional-aware**: 当前为单向电路 `(src, dst)`，实际 OCS 通常按双向链路配置

### 4.3 EPS 模型的补充验证

**建议**: 增加一个 packet-level 仿真基线（如基于 ns-3 或 OMNeT++ 的简化模型）来标定我们简化延迟模型的误差范围。或者，直接引用文献数据来约束 flat delay 参数的合理取值。

---

## 5. 真实集群部署路径

### 5.1 三阶段迁移路线

```
Phase A (当前): CPU testbed, Gloo, time.sleep()
  │  已验证: MoE 计算正确性, 路由逻辑, 亲和度统计
  │  未验证: 网络动力学, GPU 并发, OCS 控制面
  │
  ▼
Phase B: GPU 仿真 testbed
  │  替换: Gloo → NCCL, multiprocessing → CUDA stream
  │  保留: Transport delay injection (但 sleep 在 GPU stream 上)
  │  新增: NCCL profiler (nsys/ncu) 真实延迟标定
  │  新增: OCS 模拟器 → gRPC 控制面 (模拟 OCS controller)
  │
  ▼
Phase C: 真实 OCS 硬件 + GPU 集群
  │  硬件: NVIDIA GPU + Polatis/DiCon MEMS OCS 或 Telescent patch panel
  │  控制面: OCS controller API (NETCONF/REST) 替代仿真 pre_establish
  │  验证: 端到端 MoE 训练/推理 + 真实 OCS 重配
```

### 5.2 关键接口抽象

当前代码已经为真实部署预留了干净的接口边界：

```python
# Transport.pre_establish_circuits() — 这是未来的 OCS controller API
def pre_establish_circuits(self, target_ranks: list) -> float:
    if self.ocs_circuit_pool is None:
        return 0.0
    # 当前: 调用仿真电路池
    for dst in target_ranks:
        self.ocs_circuit_pool.establish(self.rank, dst, current_ns)

    # 未来: 替换为 gRPC/REST 调用真实 OCS controller
    # ocs_controller.establish_circuits(src=self.rank, dsts=target_ranks)
```

```python
# OcsCircuitPool.establish() — 仿真重配延迟
# 未来: 真实的延迟由 OCS 硬件返回 (或异步回调)
```

这个抽象层次意味着：

- **Scheduler 逻辑完全不需要修改** — `run_ocs_pipeline` / `run_ocs_dbo` 仅调用 `transport.pre_establish_circuits()`
- **Router / MoE / Expert 完全不需要修改** — 它们不感知传输层
- **仅需替换 Transport 和 OcsCircuitPool 的实现**

### 5.3 真实 OCS 硬件的选择

| 硬件                       | 重配时间  | 端口数 | 商用状态            | 适合阶段  |
| -------------------------- | --------- | ------ | ------------------- | --------- |
| Polatis 576×576 (MEMS)    | 10–25 ms | 576    | 商用 (MixNet 使用)  | Phase C   |
| DiCon MEMS 192×192        | ~20 ms    | 192    | 商用                | Phase C   |
| Telescent 机械 patch panel | ~1s       | 1000+  | 商用 (TopoOpt 使用) | Phase B/C |
| 仿真 photonic MEMS         | ~400 ns   | 240    | 研究阶段            | Phase A/B |

**推荐**: Phase C 优先使用 Polatis 或 DiCon MEMS OCS，因为 MixNet 已经验证了这条技术路线适用于 MoE 训练。

### 5.4 "真实 MoE 权重"对接状态

| 能力                                                  | 状态      | 文件                            |
| ----------------------------------------------------- | --------- | ------------------------------- |
| Qwen-MoE 权重加载 (SwitchGLU)                         | ✅ 已实现 | `src/model/qwen_experts.py`   |
| Qwen gate 权重加载 (Linear + softmax + topk + shared) | ✅ 已实现 | `src/model/qwen_experts.py`   |
| 路由回放 (expert ID remapping, top-K resize)          | ✅ 已实现 | `src/model/router_replay.py`  |
| HuggingFace 路由捕获                                  | ✅ 已实现 | `src/data/routing_capture.py` |
| 训练时专家亲和度收集                                  | ✅ 已实现 | `src/ocs/placement.py`        |
| DeepSeek-MoE gate 格式                                | ❌ 待适配 | 需要不同的 gate 结构            |
| Mixtral gate 格式                                     | ❌ 待适配 | 需要不同的 gate 结构            |

**当前可直接进行的实验**: 加载 Qwen-MoE 权重 → 使用 ReplayRouter 回放真实路由 → 在 OCS testbed 上测量电路复用/预触发效果。这是完全端到端的真实路由验证，只是通信层仍在仿真。

---

## 6. 总结

### 6.1 可信度判断

| 问题                             | 答案                                                                   |
| -------------------------------- | ---------------------------------------------------------------------- |
| MoE 计算逻辑是否与生产框架对齐？ | **是** — 5 框架审计确认                                         |
| 通信模式是否匹配？               | **是** — all-to-all dispatch/gather 模式与所有框架一致          |
| EPS 延迟建模是否精确？           | **部分** — 层级结构正确，但缺排队/拥塞/流控                     |
| OCS LRU 策略是否文献验证？       | **否** — LRU 是本工作的原创简化，MixNet/TopoOpt 使用不同策略    |
| 重配延迟参数是否合理？           | **是** — 1–1000 μs 覆盖了 2D MEMS 和 photonic MEMS 的物理范围 |
| 结论对真实集群有参考价值吗？     | **是，但需区分可迁移结论 vs 不可迁移的绝对值**                   |

### 6.2 核心建议

1. **LRU → Weighted-LRU + Confidence-Gated**: 将电路淘汰策略从纯时间驱动改为流量感知 + 置信度门控，与 MixNet 的贪心 bottleneck-aware 逻辑对齐
2. **增加 EPS 拥塞仿真**: 至少加入 per-tier 的简单队列模型（M/M/1 近似），标定我们的 flat delay 假设的乐观偏差
3. **重配延迟参数设为 1ms (2D MEMS)**: 当前默认 50μs 过于乐观。ACTINA 和 MixNet 数据建议至少 1ms
4. **保持当前接口抽象**: Transport / OcsCircuitPool 的接口设计已经支持无缝迁移到真实硬件
5. **优先做 Qwen weight + ReplayRouter + OCS 实验**: 这是最高信噪比的下一步——用真实路由验证仿真结论

### 6.3 文献引用速查

| 文献                                   | 会议/年份           | 与本工作的关系                                                   |
| -------------------------------------- | ------------------- | ---------------------------------------------------------------- |
| **MixNet/mFabric** (Liao et al.) | SIGCOMM 2025        | OCS for MoE training，最直接对标。贪心电路分配，非 LRU           |
| **TopoOpt** (Wang et al.)        | NSDI 2023           | OCS for Dense DNN，job-level 拓扑优化，无 in-training 重配       |
| **ACTINA** (Cao et al.)          | SC 2025             | OCS 重配调度的理论分析，NP-Complete 证明                         |
| **LumosCore** (et al.)           | arXiv 2024          | 光电混合架构，多项式时间重配算法                                 |
| **Google Apollo/Jupiter**        | SIGCOMM 2022 / 生产 | 3D MEMS OCS for 数据中心 Spine-Leaf，~10s 重配，非 per-iteration |
| **Megatron-LM** (NVIDIA)         | 持续演进            | MoE 训练框架参考（router + dispatch + expert）                   |
| **Tutel** (Microsoft)            | OSDI 2022           | MoE 训练框架参考（adaptive parallelism + overlap）               |
| **Megablocks** (Databricks)      | MLSys 2023          | MoE 训练框架参考（variable-split all-to-all）                    |
| **DeepEP** (DeepSeek)            | GitHub 2024         | Expert-parallel comm 参考（elastic dispatch buffers）            |

### 6.4 文献平台-建模-可比性全景对比

> 核心问题：哪些平台/方法可以保证 EPS 与 OCS 在同一基准上可比？各文献如何确保 OCS 建模可信？

| 文献 | 平台/方法论 | EPS 建模 | OCS 建模 | 可比性机制 | 验证方式 |
|------|-----------|---------|---------|-----------|---------|
| **Choi et al.** (2026) | α-β 解析模型 (Hockney)，NCCL 实测拟合 | 4 种拓扑：scale-up fat-tree (450 GB/s)、scale-out (50 GB/s)、3D torus、full-mesh | 无动态重配；OCS 仅用于 torus wrap-around 静态链路 | 统一 per-XPU 总带宽归一化；相同 GPU 计算模型、相同 TPOT SLO | DGX H100 实测验证 (<9.6% 误差)；SGLang trace 验证 (<7.5%) |
| **Opus** (Ding et al., 2026) | **三层验证**：物理 testbed → 超算模拟 → AstraSim 仿真 | Perlmutter 原生 NCCL+Slingshot-11 EPS；仿真中静态全激活链路作为 EPS 上界 | Polatis Series 6000 (200ms 重配)；SOTA MEMS <25ms；环形 photonic rail | 同一硬件（Perlmutter）上运行同一工作负载；仿真中 EPS 持有 Opus 所有可能链路 | 硬件测量 OCS 重配时间线；仿真 vs 实测交叉验证 |
| **ocs-DRP** (Guo & Ye, 2026) | SimAI + NS-3 包级仿真 | ROFT/HPN/Astral/LEAN：DCQCN 协议 + DLB 动态负载均衡 | OCS 层连接 leaf-spine，控制 inter-rail 连通比 R；功耗仅 0.28% | 统一 H100 GPU 模型、统一工作负载 (Mixtral/DeepSeek)、统一 SimAI 平台 | 纯仿真，无硬件验证 |
| **DELTA** (Ye et al., 2021) | MILP (Gurobi) + 解析归一化 | "理想无阻塞电气网络" 作为基准 (NCT=1) | OCS inter-pod 点对点电路；重配需若干秒（含收发器初始化），仅静态 | **Normalized Communication Time (NCT)**：OCS 通信时间 / 理想 EPS 通信时间 | Gurobi MILP 求解质量 vs 启发式；无硬件验证 |
| **LumosCore** (Han et al., 2021) | RapidAISim 流级仿真 + MIP | 3-tier Clos (12.8 Tbps BCM56980 交换芯片)，ECMP 负载均衡 | leaf-spine-OCS 三层架构；per-task 粒度重配；256 端口 MEMS-OCS | 统一 GPU server 模型、统一 SenseTime 生产 trace、统一调度约束 | 生产工作负载 trace 回放；与 Clos 对比 |
| **SWOT** (Wu et al., 2026) | α-β 解析模型 + MILP (Pulp/CBC) | "Ideal" = 无约束通信 (NIC 总带宽上限) | k 并行 OCS 平面，200μs 默认重配 (扫 10ns–10ms)；800/k Gbps per link | 统一 800 Gbps 总带宽/节点；同一 collective 算法集；best-of-breed 基线 | 参数扫描数值评估；MILP 120s 求解 |
| **ReTri** (Juerss et al., 2026) | AstraSim + ns-3 | 静态 ring 最短路径 All-to-All | ORN (2n OCS 端口)；重配扫 1μs–50ms；双向光路 | 统一 400 Gbps 链路、统一 α-β 成本模型 (含重配项) | 仿真参数扫描；无硬件验证 |
| **Birkhoff** (Amponsah et al., 2026) | MoE trace 驱动仿真 | "理想无拥塞 All-to-All" 上界 | BvN/贪心分解为 matchings 序列；重配开销取决于 matching 数量 | 同一 MoE traffic matrix + 同一 expert compute model | MoE trace 回放；无硬件验证 |
| **MoX** (Silberstein, 2026) | ASTRA-sim 2 + Chakra EP traces | "理想全连接包交换机" 上界 | 静态 direct-connect expander (800 Gbps 分 8×100)；无动态重配 | 统一 800 Gbps/端点；同一 DeepSeek-V3/Qwen-3 traces | Token 级 trace (真实 GPU 采集)；max-link-load vs slowdown 相关性验证 |
| **ACTINA** (Cao et al., 2025) | Calculon 解析模型 + α-β | OCS-based Fat-Tree、TPUPod 3D-Torus (混合光电) | Giant OCS Abstraction (上界) → OCSBCube (物理)；重配扫 1μs–1s | 统一 H100 计算模型 + 统一注入带宽 + 统一 GPT-3/Megatron 工作负载 | 解析模型；DP 最优重配 vs 实时启发式 |
| **In-Package Optical** (Patel et al., 2026) | AMPED 解析模型 | NVLink Gen4 + InfiniBand (电气 baseline) | In-package OIO (DWDM 静态光链路，204.8 Tbps/GPU)；无 OCS | 统一 GPT-3 175B + 统一 node 配置；对比电气 vs 光学物理链路 | AMPED 模型 min-GPT 实测验证；工艺 projection (Imec) |
| **RailS** (Xu et al., 2025) | 大规模可编程 DC 仿真器 | Rail 架构 + spine-leaf EPS；ECMP/动态负载均衡 | 无 OCS | 仅 EPS 内部 (ECMP vs RailS)；非 EPS-OCS 对比 | 生产 MoE traces + LPT 解析界 |
| **Switching Efficiency** (Ye et al., 2021) | 自定义通信调度仿真 | 3D-Torus (6-port switch/GPU) vs Rail-Optimized (9:1 tiered bw) | OCS 仅作为 Torus 维度映射使能器 (无重配延迟建模) | 统一 Switching Efficiency (η) 框架分解 | 纯解析框架；诊断工具而非预测模型 |
| **MiGOCS** (Tang et al., 2026) | 自定义仿真 + 贪心 TPE | Gemini (全空间交换 OCS fabric) — 非 EPS 对比 | MiGOCS: space+WSS 混合粒度；无重配延时显式建模 | 统一 OCS port count 预算；统一 traffic matrix | 合成 traffic 仿真；无硬件验证 |

**核心技术路线分类**：

| 类别 | 代表文献 | EPS-OCS 可比性保障机制 |
|------|---------|----------------------|
| **物理 testbed 对标** | Opus (Ding) | 同一台超算上运行 EPS baseline 和 OCS emulation |
| **α-β 模型归一化** | Choi, SWOT, ReTri, ACTINA | 统一 Hockney 参数 (α/β) + 统一总带宽 per 节点 |
| **同一仿真平台** | ocs-DRP, LumosCore, MoX | NS-3 / AstraSim / RapidAISim + 统一 GPU 模型 |
| **解析归一化指标** | DELTA (NCT), Switching Efficiency (η) | 比值指标消除绝对尺度差异 |
| **Trace/MILP 共享** | Birkhoff, DELTA, MoX | 同一 traffic matrix / 同一 trace 输入 |

---

*本文档随 codebase 演进而更新。最后更新：2026-07-23。*
