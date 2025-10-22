# MemorySpace Code Overview

本文件总结项目当前的代码结构，逐文件说明职责与主要函数/类的作用，便于后续维护和论文撰写。

## 顶层入口

- `src/agent_memory/__init__.py`
  - 对外导出核心组件：`MemoryObject`, `MemoryManager`, 分配器、垃圾回收器、`SimilarityRetriever`，方便使用 `from agent_memory import ...`。

## 内存核心层

- `src/agent_memory/memory_object.py`
  - 定义单条记忆 `MemoryObject` 及其元数据（payload、重要度、TTL、引用计数等）。
  - 方法 `touch()` 更新访问时间；`expired()` 判断 TTL；`importance_score()` 结合衰减与引用次数动态评估重要值。

- `src/agent_memory/memory_space.py`
  - 模拟连续堆空间，管理空闲/已分配段。
  - 提供 `allocate()`、`deallocate()`、`resize()`、`fragmentation()` 等操作，并带有空闲表重排、合并等辅助方法。

- `src/agent_memory/allocators.py`
  - 抽象类 `Allocator` 规范分配接口。
  - `FirstFitAllocator`：直接使用 `MemorySpace.allocate`。
  - `BestFitAllocator`：在空闲段中选最小适配块，降低外部碎片。

- `src/agent_memory/garbage_collectors.py`
  - `GarbageCollector` 基类；实现 TTL、Mark-and-Sweep、Generational 三类 GC。
  - `MarkAndSweepCollector.collect()` 扫描可达集合后释放无主对象。
  - `TimeToLiveCollector.collect()` 根据 TTL 释放对象。
  - `GenerationalCollector.collect()` 根据引用次数提升代数，并按重要度/时间淘汰年轻代。

- `src/agent_memory/gc_policy.py`
  - GC 策略插件：`PeriodicGCPolicy`、`AdaptiveFragmentationPolicy`、`ObjectGrowthPolicy`。
  - `should_trigger(manager, reason)` 决定是否触发 GC，`notify_gc()` 在 GC 结束后更新策略状态。

- `src/agent_memory/locks.py`
  - 读写锁 `ReadWriteLock`，提供 `read_lock()`/`write_lock()` 上下文管理器，支持多读单写，供线程安全模式下使用。

- `src/agent_memory/persistence.py`
  - `WriteAheadLog`：追加 JSON 行记录 store/free 操作，支持重放。
  - `CheckpointManager`：通过 pickle 序列化/反序列化当前对象集合，支持崩溃恢复。

- `src/agent_memory/pager.py`
  - `PageTable`：支持 LRU、Clock、LFU 页面置换，记录 `page_faults`、`evictions`、`loads`、`writes`；提供 `mark_dirty/clear_dirty` 与 `switch_clock_victim` 等辅助函数。

- `src/agent_memory/cold_storage.py`
  - 使用内存字典模拟冷存储（pickle 序列化），统计写入/读取次数。

- `src/agent_memory/memory_manager.py`
  - 系统中枢，整合分配器、检索、GC、策略、WAL/Checkpoint 与线程安全。
  - 主要方法：
    - `store()` / `store_existing()`：创建对象、触发策略、写 WAL。
    - `load()` / `retrieve()`：访问对象并更新最近使用信息。
    - `add_root` / `pin` 等：维护保持集。
    - `free_object()`：释放对象并写 WAL。
    - `run_gc()`：执行 GC 记录 `pause_duration`。
    - `stats()` / `debug_snapshot()`：输出堆状态。
    - `_maybe_trigger_policies()`、`tick()`：策略驱动 GC。
    - `checkpoint()`：调用 CheckpointManager 保存快照。

- `src/agent_memory/tiered_manager.py`
  - 构建热层 MemorySpace 与冷层 ColdStorage 的分层记忆管理。
  - 支持 LRU/Clock/LFU 分页、write-back/write-through 策略、批量升温等选项。
  - `stats()` 输出热/冷命中与分页统计（`hot_hits`、`cold_hits`、`page_faults` 等）。

- `src/agent_memory/retrieval.py`
  - 支持稀疏 bag-of-words 与可选稠密嵌入（SentenceTransformer）的检索组件，`query()` 返回按相似度排序的对象。

## 数据加载器

- `src/datasets/ufo_loader.py`：读取 UFO CSV，兼容字段差异。
- `src/datasets/text_chunker.py`：按窗口/重叠划分文本，生成 `TextChunk`。
- `src/datasets/samsum_loader.py`：调用 HuggingFace datasets 获取 SAMSum 对话。

## 实验脚本

- `src/experiments/environment.py`：`SimulatedEnvironment` 生成 `Experience`。
- `src/experiments/run_simulation.py`：运行小规模模拟，观察记忆管理行为。
- `src/experiments/benchmark_memory.py`：比较不同 allocator/GC 组合的命中率、碎片率、GC 次数。
- `src/experiments/ufo_experiment.py`：在真实 UFO 数据上评估策略。
- `src/experiments/baselines.py`：Sliding Window、Reservoir Memory 基线。
- `src/experiments/long_context_study.py`：多 session 长上下文检索实验。
- `src/experiments/capacity_sweep.py`：不同堆容量、GC、基线组合的扩展测试。
- `src/experiments/gc_ablation.py`：GC 组合对比。
- `src/experiments/tiered_benchmark.py`：评估热/冷 + 分页/写策略对命中率和分页统计的影响。
- `src/experiments/instrumentation.py`：`MemoryProfiler` 输出 JSONL/CSV 事件日志。

## 测试用例

- `tests/` 目录覆盖关键组件：分页、持久化、GC 策略、Tiered 管理器、基线等，确保功能正确性。

## 文档

- `readme.md`：总体架构、实验指引、系统特性。
- `docs/paper.tex`：论文格式描述。
- `docs/project_log.md`：阶段目标与 Roadmap。
- `docs/code_overview.md`：即本文件。
