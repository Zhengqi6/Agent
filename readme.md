# MemorySpace: Systems-Inspired Memory Manager for LLM Agents

MemorySpace 将经典计算机系统（操作系统内存管理、缓存分层、数据库日志等）的思想引入 Agent 外部记忆，使得 LLM Agent 在处理长上下文与连续任务时，拥有可控、可观测、可持久化的记忆栈。

核心目标：
- **系统化管理**：提供真实的堆分配、垃圾回收、热/冷分层、分页置换与写回策略，而非简单的“向量库 + 时间衰减”。
- **可观测 & 可恢复**：通过 MemoryProfiler、WAL、Checkpoint，让每一次存储/回收都有迹可循，可以复现与回滚。
- **任务级验证**：在长文本阈值（text8/enwik8）与 HotpotQA supporting-fact 召回等任务中展示系统效益，并保留脚本一键复现。

---

## 功能速览

| 模块 | 亮点 |
| --- | --- |
| **MemoryManager** | First/Best-Fit 分配；TTL / Mark-Sweep / Generational GC；策略插件（周期/碎片率/对象增长）；BanditGCPolicy epsilon-greedy 自适应。 |
| **TieredMemoryManager** | 热层 MemorySpace + 冷层 ColdStorage；LRU/Clock/LFU 分页；write-back/write-through；批量升温；热/冷命中率与页面统计。 |
| **Persistence** | WriteAheadLog 记录 store/free；CheckpointManager 快照；可选读写锁支持多读单写。 |
| **Retrieval & Baseline** | SimilarityRetriever（稀疏 + Sentence-Transformer 稠密）；SlidingWindow、Reservoir 作为对照。 |
| **Instrumentation** | MemoryProfiler 输出 JSONL/CSV（分配、检索、GC 事件含 `pause_duration`）；`scripts/run_all.sh` 集成单测、text8、HotpotQA 实验。 |

---

## 仓库结构

```
├── data/                # 示例数据（text8/enwik8/tinyshakespeare/ufo 等）
├── docs/                # 论文草稿、项目日志、代码总览
├── results/             # 实验输出 CSV
├── scripts/             # 运行脚本（run_all.sh）
├── src/
│   ├── agent_memory/    # MemorySpace 核心实现
│   ├── datasets/        # 数据加载工具
│   └── experiments/     # 实验脚本（capacity_sweep、hotpot_benchmark 等）
└── tests/               # 单元测试
```

> ⚠️ `data/enwik8` 与 `data/text8` 约 95 MB，已包含在仓库中。若仓库体积敏感，可删除后使用脚本重新下载。

---

## 安装依赖

建议使用 Python 3.11，并创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy==1.26.4 datasets==2.14.6 sentence-transformers pyarrow==16.0.0
```

部分脚本（例如 `hotpot_benchmark`）会访问 HuggingFace Hub，如遇网络限制请提前配置令牌或使用镜像。

---

## 快速开始

### 1. 运行单测与核心实验

```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

该脚本会执行：
1. `PYTHONPATH=src python -m unittest discover -s tests`
2. text8 容量扫描（输出 `results/text8_capacity_sparse.csv`）
3. HotpotQA train[:200] supporting-fact 召回（`results/hotpot_summary.csv`）

### 2. 关键实验命令

- **长文本容量（text8/enwik8）**
  ```bash
  PYTHONPATH=src python -m experiments.capacity_sweep \
    --data-path data/text8 \
    --chunk-size 512 \
    --limit 80000 \
    --capacities 4096 8192 \
    --query-interval 10 \
    --baselines sliding reservoir \
    --collector-variants ttl_mark ttl_mark_gen \
    --seeds 1 \
    --output results/text8_capacity_sparse.csv
  ```

- **HotpotQA 任务级基准（5k 样本）**
  ```bash
  PYTHONPATH=src python -m experiments.hotpot_benchmark \
    --dataset-split train[:5000] \
    --max-samples 5000 \
    --hot-capacity 32768 \
    --sliding-window 4096 \
    --output results/hotpot_summary_train5000.csv
  ```

- **分层记忆 & 分页策略**
  ```bash
  PYTHONPATH=src python -m experiments.tiered_benchmark \
    --data-path data/ufo.csv \
    --limit 2000 \
    --query-interval 15 \
    --fragmentation-threshold 0.4 \
    --root-capacity 10 \
    --paging-strategy lfu \
    --write-policy write_back \
    --promotion-batch 2 \
    --output results/tiered_summary.csv
  ```

更多实验（GC ablation、长上下文模拟等）请参考 `src/experiments/` 内脚本。

---

## 核心模块一览

- `src/agent_memory/memory_manager.py`
  - 负责对象分配、引用管理、GC 调度、WAL/Checkpoint、线程安全。
- `src/agent_memory/gc_policy.py`
  - 定义 `GCPolicy` 接口及 BanditGCPolicy；可插拔策略。
- `src/agent_memory/tiered_manager.py`
  - 实现热/冷两级存储、分页策略、批量升温与指标导出。
- `src/agent_memory/pager.py`
  - LRU/Clock/LFU 页面置换，实现 `page_faults/evictions/writes` 统计。
- `src/experiments/*`
  - 各类实验脚本，统一输出 CSV，方便绘图分析。

单元测试位于 `tests/`，覆盖 MemoryManager、TieredManager、Pager、WAL、GC 策略等关键组件。

---

## 当前实验结果概览

| 实验 | 主要设置 | 指标亮点 |
| --- | --- | --- |
| text8 容量扫描 | 80k × 512B，GC=TTL+Mark | MemoryManager hit@1≈0.97 vs Sliding 0.95；对 Reservoir 提升 0.32–0.40 |
| enwik8 容量扫描 | 50k × 512B | MemoryManager hit@1≈0.96 vs Sliding 0.94；对 Reservoir 提升 ~0.37 |
| HotpotQA Supporting-Fact | train[:5000]，top-5 | MemoryManager hit@5=0.798 vs Sliding 0.601 |
| CNN/DailyMail 容量扫描 | train 150k × 512B，window=120 | MemoryManager hit@1=0.831 vs Sliding 0.822；对 Reservoir 提升 0.17–0.27 |
| UltraChat 多轮对话 | train\_sft 150k × 512B，window=150 | MemoryManager hit@1=0.814 vs Sliding 0.801；对 Reservoir 提升 0.26–0.29 |

所有结果均写入 `results/` 目录，可直接用于画图或统计。

---

## 下一步计划

详见 `docs/project_log.md`，重点包括：
- **Benchmark parity**：对齐 LongMem、OMEGA-Memory 等最新论文的设定，补充 LongBench、InfiniteBench、NarrativeQA、Qasper、GovReport 等基准，并引入 LangChain VectorStore、MemGPT Planner、MemLab、MemRAG 作为行业 baseline，统一比较命中率、延迟、吞吐；
- **Data scaling**：扩展至百万级对话/跨任务日志（如 ShareGPT、BookCorpus、LongDialog），并将 MultiDoc2Dial、LongFormSum 等真实任务接入 `experiments/`，同时输出 hit@k + F1/EM；
- **System stress**：在冷存储与 WAL 上记录写放大、落盘延迟、命中率时序，验证高负载下的可控性；
- **Policy learning**：基于运行日志构建离线 MDP，用 policy-gradient/DQN 探索 GC 与分页组合策略，保留启发式策略做安全回退；
- **Documentation refresh**：更新论文草稿的相关工作、实验表格与附录，使 MemorySpace 的系统创新点和性能对比一目了然。

---

## 参考
- LongMem: Directly Optimizing Language Models for Long-Text Tasks (arXiv:2507.02259)
- OMEGA-Memory: Independent-Context Multi-Conversation Generation for Large Language Model Memory (arXiv:2509.24704)

如有问题或建议，欢迎提交 issue 或 PR。EOF
