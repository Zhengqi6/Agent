from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from agent_memory import (
    BestFitAllocator,
    FirstFitAllocator,
    GenerationalCollector,
    MarkAndSweepCollector,
    MemoryManager,
    SimilarityRetriever,
    TimeToLiveCollector,
)
from agent_memory.allocators import Allocator
from agent_memory.garbage_collectors import GarbageCollector
from experiments.environment import SimulatedEnvironment

AllocatorFactory = Callable[[], Allocator]
CollectorFactory = Callable[[], GarbageCollector]


@dataclass
class ExperimentConfig:
    label: str
    capacity: int
    allocator_factory: AllocatorFactory
    collector_factories: List[CollectorFactory]
    steps: int = 200
    query_interval: int = 5
    fragmentation_threshold: float = 0.45
    root_capacity: int = 5


def run_single(
    config: ExperimentConfig,
    seed: int,
    *,
    trajectory_dir: Optional[str] = None,
) -> Dict[str, float]:
    env = SimulatedEnvironment(seed=seed)
    manager = MemoryManager(
        config.capacity,
        allocator=config.allocator_factory(),
        collectors=[factory() for factory in config.collector_factories],
        retriever=SimilarityRetriever(),
    )

    root_queue: List[str] = []
    high_importance_inserted = 0

    queries = 0
    hits_at_1 = 0
    hits_at_3 = 0
    mrr_sum = 0.0

    heap_used_sum = 0.0
    fragmentation_sum = 0.0
    object_sum = 0.0
    observations = 0

    trajectory_rows: List[Dict[str, float]] = []

    for step in range(1, config.steps + 1):
        gc_before = len(manager.gc_events)
        experience = env.next_experience()
        payload = {"text": experience.text, "metadata": experience.metadata}
        obj = manager.store(
            payload,
            tags=experience.tags,
            importance=experience.importance,
            ttl=experience.ttl,
            pin=experience.importance > 0.9,
        )

        if experience.importance > 0.85:
            high_importance_inserted += 1

        if experience.importance >= 0.6:
            manager.add_root(obj.id)
            root_queue.append(obj.id)
            if len(root_queue) > config.root_capacity:
                removed = root_queue.pop(0)
                manager.remove_root(removed)
        else:
            # Link recent objects together to create short-lived references.
            for target_id in root_queue[-2:]:
                manager.add_reference(obj.id, target_id)

        stats = manager.stats()
        heap_used_sum += stats["heap_used"]
        fragmentation_sum += stats["fragmentation"]
        object_sum += stats["objects"]
        observations += 1

        query_topic: Optional[str] = None
        hit1 = 0
        hit3 = 0
        mrr_increment = 0.0

        if step % config.query_interval == 0:
            query_topic = experience.metadata["topic"]
            query = env.make_query(query_topic)
            retrieved = manager.retrieve(query, top_k=3)
            queries += 1

            rank = None
            for idx, candidate in enumerate(retrieved, start=1):
                topic = candidate.payload.get("metadata", {}).get("topic")
                if topic == query_topic:
                    rank = idx
                    break
            if rank is not None:
                if rank == 1:
                    hit1 = 1
                    hit3 = 1
                elif rank <= 3:
                    hit3 = 1
                hits_at_1 += hit1
                hits_at_3 += hit3
                mrr_increment = 1.0 / rank
                mrr_sum += mrr_increment

        gc_delta = len(manager.gc_events) - gc_before
        freed_this_step = 0
        if gc_delta > 0:
            for event in manager.gc_events[-gc_delta:]:
                freed_this_step += len(event.get("freed_ids", []))

        trajectory_rows.append(
            {
                "step": step,
                "objects": stats["objects"],
                "heap_used": stats["heap_used"],
                "fragmentation": stats["fragmentation"],
                "roots": stats["roots"],
                "pinned": stats["pinned"],
                "query_topic": query_topic or "",
                "hit_at_1": hit1,
                "hit_at_3": hit3,
                "mrr_increment": mrr_increment,
                "gc_cycles_total": len(manager.gc_events),
                "gc_cycles_delta": gc_delta,
                "gc_freed": freed_this_step,
            }
        )

        if stats["fragmentation"] > config.fragmentation_threshold:
            manager.run_gc(trigger="high_fragmentation")

    if trajectory_dir and trajectory_rows:
        os.makedirs(trajectory_dir, exist_ok=True)
        trajectory_path = os.path.join(trajectory_dir, f"{config.label}_seed{seed}.csv")
        with open(trajectory_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trajectory_rows)

    final_stats = manager.stats()
    gc_cycles = len(manager.gc_events)
    collector_counts: Counter[str] = Counter()
    collector_freed: Counter[str] = Counter()
    for event in manager.gc_events:
        for detail in event.get("collectors", []):
            name = detail.get("collector")
            collector_counts[name] += 1
            collector_freed[name] += detail.get("freed", 0)

    final_high_importance = sum(1 for obj in manager.objects.values() if obj.importance > 0.85)
    retention_ratio = (
        final_high_importance / high_importance_inserted if high_importance_inserted else 1.0
    )

    summary: Dict[str, float] = {
        "config": config.label,
        "seed": seed,
        "steps": config.steps,
        "queries": queries,
        "hit_at_1": hits_at_1 / queries if queries else 0.0,
        "hit_at_3": hits_at_3 / queries if queries else 0.0,
        "mrr": mrr_sum / queries if queries else 0.0,
        "avg_heap_used": heap_used_sum / observations if observations else 0.0,
        "avg_fragmentation": fragmentation_sum / observations if observations else 0.0,
        "avg_object_count": object_sum / observations if observations else 0.0,
        "gc_cycles": gc_cycles,
        "ttl_cycles": float(collector_counts.get("ttl", 0)),
        "ttl_freed": float(collector_freed.get("ttl", 0)),
        "mark_and_sweep_cycles": float(collector_counts.get("mark_and_sweep", 0)),
        "mark_and_sweep_freed": float(collector_freed.get("mark_and_sweep", 0)),
        "generational_cycles": float(collector_counts.get("generational", 0)),
        "generational_freed": float(collector_freed.get("generational", 0)),
        "high_importance_inserted": float(high_importance_inserted),
        "high_importance_retained": float(final_high_importance),
        "high_importance_retention": retention_ratio,
        "final_object_count": float(final_stats["objects"]),
        "final_heap_used": float(final_stats["heap_used"]),
        "final_fragmentation": final_stats["fragmentation"],
    }
    return summary


def build_default_configs(args: argparse.Namespace) -> List[ExperimentConfig]:
    configs = [
        ExperimentConfig(
            label="first_fit_ttl_ms",
            capacity=4096,
            allocator_factory=FirstFitAllocator,
            collector_factories=[TimeToLiveCollector, MarkAndSweepCollector],
        ),
        ExperimentConfig(
            label="best_fit_gen",
            capacity=4096,
            allocator_factory=BestFitAllocator,
            collector_factories=[TimeToLiveCollector, GenerationalCollector, MarkAndSweepCollector],
        ),
        ExperimentConfig(
            label="first_fit_mark",
            capacity=4096,
            allocator_factory=FirstFitAllocator,
            collector_factories=[MarkAndSweepCollector],
        ),
    ]

    for config in configs:
        config.steps = args.steps
        config.query_interval = args.query_interval
        config.fragmentation_threshold = args.fragmentation_threshold
        config.root_capacity = args.root_capacity
        if args.capacity:
            config.capacity = args.capacity
    return configs


def write_summary(path: str, records: Iterable[Dict[str, float]]) -> None:
    records = list(records)
    if not records:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark memory strategies under synthetic workloads.")
    parser.add_argument("--steps", type=int, default=200, help="Number of experience steps per run.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds to evaluate.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset applied to generated seeds.")
    parser.add_argument("--output", type=str, default="results/summary.csv", help="Path to CSV summary output.")
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default=None,
        help="Optional directory to write per-run trajectory CSV files.",
    )
    parser.add_argument("--capacity", type=int, default=None, help="Override memory capacity for all configs.")
    parser.add_argument("--query-interval", type=int, default=5, help="Issue retrieval queries every N steps.")
    parser.add_argument(
        "--fragmentation-threshold",
        type=float,
        default=0.45,
        help="Trigger GC when fragmentation exceeds this level.",
    )
    parser.add_argument(
        "--root-capacity",
        type=int,
        default=5,
        help="Maximum number of active roots retained in the working set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_default_configs(args)
    seeds = [args.seed_offset + index for index in range(args.seeds)]

    summaries: List[Dict[str, float]] = []
    for config in configs:
        for seed in seeds:
            summary = run_single(config, seed, trajectory_dir=args.trajectory_dir)
            summaries.append(summary)

    write_summary(args.output, summaries)

    for summary in summaries:
        print(
            f"[{summary['config']} seed={summary['seed']}] "
            f"hit@1={summary['hit_at_1']:.2f} hit@3={summary['hit_at_3']:.2f} "
            f"mrr={summary['mrr']:.2f} gc_cycles={int(summary['gc_cycles'])} "
            f"avg_fragmentation={summary['avg_fragmentation']:.3f}"
        )


if __name__ == "__main__":
    main()
