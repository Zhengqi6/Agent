# MemorySpace Project Log

记录项目里程碑、实验结果与下一步计划，便于团队与审稿人了解进展。

---

## 2025-10-22 Summary

- **核心系统完成**：MemorySpace + 分配器、TTL/Mark-Sweep/Generational GC、策略插件、读写锁、WAL/Checkpoint、Profiler（含 GC `pause_duration`）。
- **检索与分层**：SimilarityRetriever（稀疏/稠密），TieredMemoryManager（热/冷、LRU/Clock/LFU、write-back/through、批量升温），ColdStorage + PageTable。
- **实验基线**：模拟环境、UFO 长上下文、Shakespeare 容量、GC 消融、Tiered 实验、文本缓存观测。
- **复现脚本**：`scripts/run_all.sh` 集成单测、text8 容量扫描与 HotpotQA 基准，一键生成 `results/text8_capacity_sparse.csv` 与 `results/hotpot_summary.csv`。
- **策略学习**：新增 `BanditGCPolicy`（epsilon-greedy）在多策略间自适应选择 GC 触发逻辑。
- **大规模文本测试**：
  - `text8` (95MB, 80k chunks, 512B) → `results/text8_capacity_sparse.csv`。MemoryManager 相比 Sliding Window 提升 1.1–2.1pp（hit@1），对 Reservoir 提升 ~32–40pp。
  - 之前 `enwik8` (95MB, 50k chunks) → `results/enwik8_capacity_sparse.csv`。MemoryManager 相比 Reservoir 提升 ~37pp。
- **任务级基准**：HotpotQA (train[:200]) → `experiments.hotpot_benchmark.py`；MemoryManager hit@5=0.735 vs Sliding Window 0.68（supporting fact recall）。
- **任务级基准**：HotpotQA (train[:200] 与 train[:5000]) → `experiments.hotpot_benchmark.py`；在 5k 样本上 MemoryManager hit@5=0.798 vs Sliding Window 0.601（supporting fact recall）。
- **代码 & 文档**：README、`docs/paper.tex`、`docs/code_overview.md` 持续更新；实验数据写入 `results/`。

---

## 最新里程碑

### 2025-10-16 — Prototype Ready
- 完成 MemoryManager + GC + Profiler 基础，实现 Tiered 原型、分页统计；运行小规模长上下文/容量/GC 实验。

### 2025-10-22 — Large Text Benchmarks
- 下载 `enwik8` 和 `text8`，在 50k/80k 文本块上运行 `capacity_sweep`（命令见 README/日志）。观察 MemoryManager 在高负载下对 Sliding/Reservoir 的提升。

### 2025-10-22 — HotpotQA Task Benchmark
- `experiments.hotpot_benchmark` (依赖 `datasets`, `numpy`, `pyarrow`) 比较 MemoryManager 与 Sliding Window 的 supporting-fact 召回；初步结果写入 `results/hotpot_summary.csv`。

---

## Reviewer-driven Action Plan

1. **更大规模数据**：
   - 引入百万级样本（对话日志、长时序 QA、BookCorpus）；观察 GC/paging 命中率随负载变化。
2. **现实 baseline 对比**：
   - 集成 LangChain vector store、MemGPT Memory Planner、MemLab 等方案，对比命中率、延迟、资源占用。
3. **任务级指标**：
   - 在长文本 QA/对话/规划任务上记录 F1、EM、Task success、人工评分，验证系统对真实任务的收益。
4. **分页与写策略实验**：
   - 构造读/写密集、热点迁移等 workload，比较 LRU/Clock/LFU 与 write-back/through 的性能曲线、GC 停顿分布、WAL 增长。
5. **持久化/恢复实验**：
   - 模拟定时 checkpoint + 随机崩溃，测量恢复时间、数据丢失窗口、命中率变化，记录 checkpoint/WAL 恢复成本。
6. **并发压力与 GC 模式**：
   - 模拟多 Agent 并发读写，衡量吞吐/延迟，并探索增量或并发 GC 原型。
7. **可视化/错误分析**：
   - 利用 Profiler/WAL 绘制 GC pause 分布、缺页时间序列、热/冷命中热图；整理典型错误案例。
8. **复现性与公开细节**：
   - 完善 `scripts/run_all.sh`，补充配置文件示例；整理实验日志、检查点、模型输出示例（可考虑附录或公开仓库）；明确硬件配置、依赖版本。

---

## 分阶段计划

### Phase 1 — 架构与策略
1. **多层存储**
   - ✅ Tiered Memory + 分页统计。
   - 🔜 在 ≥10^5 条记录的数据集上比较 write-back/write-through + 批量升温组合，输出热/冷命中时序与写放大分析。
2. **分页置换**
   - ✅ PageTable (LRU/Clock/LFU) + `page_faults/evictions/writes` 统计。
   - 🔜 设计读写密集、热点迁移 workload，生成性能曲线与热图。
3. **GC 策略**
   - ✅ GCPolicy（三种策略）+ `BanditGCPolicy` 自适应选择。
   - 🔜 在长文本 QA/对话场景下比较策略组合的任务指标与 GC 停顿，输出可视化与错误分析。

### Phase 2 — 并发与容错
1. **并发控制**
   - ✅ 读写锁实现多读单写。
   - 🔜 模拟多 Agent 并发访问，测量吞吐/延迟，探索增量/并发 GC。
2. **持久化保证**
   - ✅ WAL + Checkpoint 原型。
   - 🔜 长时间运行与随机崩溃实验，记录恢复时间、未刷写窗口、命中率变化。

### Phase 3 — 指标与可视化
1. **MemoryPerf 扩展**
   - ✅ 记录 GC 停顿、分页统计。
   - 🔜 增加分配延迟、热/冷命中率时间序列，统一指标导出接口。
2. **Notebook + 图表**
   - 基于 profiler/WAL 生成 GC pause 分布、缺页率、热图等，纳入论文系统分析章节。

---

## Backlog
- 高阶学习型策略（RL/元策略）进一步优化 GC 与升降级决策。
- Embedding Adapter 自动切换稀疏/稠密检索或混合策略。
- 百万级数据扩展与离线 offload/流水线支持。

> 本日志需在每次重大实验后更新，以保持对外可追踪性。
