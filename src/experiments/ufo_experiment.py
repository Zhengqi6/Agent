from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, deque
from typing import Dict, Iterable, List, Optional

from agent_memory import (
    BestFitAllocator,
    FirstFitAllocator,
    GenerationalCollector,
    MarkAndSweepCollector,
    MemoryManager,
    SimilarityRetriever,
    TimeToLiveCollector,
)
from datasets.ufo_loader import UfoSighting, load_ufo_dataset
from experiments.benchmark_memory import ExperimentConfig


def sighting_to_payload(sighting: UfoSighting) -> Dict[str, Dict[str, str] | str]:
    location_parts = [sighting.city.title() if sighting.city else "", sighting.state.upper(), sighting.country.upper()]
    location = ", ".join([part for part in location_parts if part])
    text = (
        f"{sighting.datetime or 'Unknown time'} - {location or 'Unknown location'}: "
        f"{sighting.comments or 'No description provided.'} "
        f"Shape observed: {sighting.shape or 'unknown'}. "
        f"Colors reported: {sighting.colors or 'unspecified'}. "
        f"Duration reported: {sighting.duration or 'unknown'}."
    )
    metadata = {
        "city": sighting.city,
        "state": sighting.state,
        "country": sighting.country or "unknown",
        "shape": sighting.shape or "unknown",
        "duration": sighting.duration,
        "datetime": sighting.datetime,
        "latitude": sighting.latitude,
        "longitude": sighting.longitude,
    }
    return {"text": text, "metadata": metadata}


def compute_importance(sighting: UfoSighting) -> float:
    narrative = sighting.comments or sighting.colors or ""
    base = min(len(narrative) / 400.0, 1.0)
    if sighting.shape.lower() in {"triangle", "diamond", "fireball"}:
        base = min(1.0, base + 0.25)
    if sighting.duration and any(char.isdigit() for char in sighting.duration):
        base = min(1.0, base + 0.1)
    return round(base, 3)


def compute_ttl(sighting: UfoSighting) -> Optional[float]:
    if sighting.country.lower() == "us":
        return 1800.0  # 30 minutes
    if sighting.shape.lower() in {"triangle", "circle"}:
        return None
    return 900.0


def importance_label(importance: float) -> str:
    if importance >= 0.85:
        return "critical"
    if importance >= 0.6:
        return "high"
    if importance >= 0.3:
        return "medium"
    return "low"


def build_default_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(
            label="ufo_first_fit_ttl_ms",
            capacity=8192,
            allocator_factory=FirstFitAllocator,
            collector_factories=[TimeToLiveCollector, MarkAndSweepCollector],
            steps=0,
            query_interval=0,
            fragmentation_threshold=0.0,
            root_capacity=0,
        ),
        ExperimentConfig(
            label="ufo_best_fit_gen",
            capacity=8192,
            allocator_factory=BestFitAllocator,
            collector_factories=[TimeToLiveCollector, GenerationalCollector, MarkAndSweepCollector],
            steps=0,
            query_interval=0,
            fragmentation_threshold=0.0,
            root_capacity=0,
        ),
    ]


def run_ufo_workload(
    config: ExperimentConfig,
    sightings: List[UfoSighting],
    *,
    query_interval: int,
    fragmentation_threshold: float,
    root_capacity: int,
    trajectory_path: Optional[str] = None,
) -> Dict[str, float]:
    manager = MemoryManager(
        config.capacity,
        allocator=config.allocator_factory(),
        collectors=[factory() for factory in config.collector_factories],
        retriever=SimilarityRetriever(),
    )

    root_queue: deque[str] = deque(maxlen=root_capacity)
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

    for step, sighting in enumerate(sightings, start=1):
        payload = sighting_to_payload(sighting)
        importance = compute_importance(sighting)
        ttl = compute_ttl(sighting)
        tags = [
            sighting.shape.lower() or "unknown",
            sighting.country.lower() or "unknown",
            importance_label(importance),
        ]

        gc_before = len(manager.gc_events)
        obj = manager.store(
            payload,
            tags=tags,
            importance=importance,
            ttl=ttl,
            pin=importance >= 0.9,
        )

        if importance >= 0.9:
            high_importance_inserted += 1

        if importance >= 0.6:
            manager.add_root(obj.id)
            root_queue.append(obj.id)
        elif root_queue:
            recent_targets = list(root_queue)[-2:]
            for target_id in recent_targets:
                manager.add_reference(obj.id, target_id)

        if len(root_queue) > root_capacity:
            removed = root_queue.popleft()
            manager.remove_root(removed)

        stats = manager.stats()
        heap_used_sum += stats["heap_used"]
        fragmentation_sum += stats["fragmentation"]
        object_sum += stats["objects"]
        observations += 1

        query_shape = (sighting.shape or "").strip().lower()
        hit1 = 0
        hit3 = 0
        mrr_increment = 0.0

        if query_shape and step % query_interval == 0:
            query = f"{query_shape} ufo sighting"
            retrieved = manager.retrieve(query, top_k=3)
            queries += 1
            rank = None
            for idx, candidate in enumerate(retrieved, start=1):
                metadata = candidate.payload.get("metadata", {})
                candidate_shape = str(metadata.get("shape", "")).strip().lower()
                if candidate_shape == query_shape:
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
                "query_shape": query_shape,
                "hit_at_1": hit1,
                "hit_at_3": hit3,
                "mrr_increment": mrr_increment,
                "gc_cycles_total": len(manager.gc_events),
                "gc_cycles_delta": gc_delta,
                "gc_freed": freed_this_step,
            }
        )

        if stats["fragmentation"] > fragmentation_threshold:
            manager.run_gc(trigger="high_fragmentation")

    if trajectory_path and trajectory_rows:
        fieldnames = list(trajectory_rows[0].keys())
        with open(trajectory_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
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

    final_high_importance = sum(1 for obj in manager.objects.values() if obj.importance >= 0.9)
    retention_ratio = (
        final_high_importance / high_importance_inserted if high_importance_inserted else 1.0
    )

    summary = {
        "config": config.label,
        "steps": float(len(sightings)),
        "queries": float(queries),
        "hit_at_1": hits_at_1 / queries if queries else 0.0,
        "hit_at_3": hits_at_3 / queries if queries else 0.0,
        "mrr": mrr_sum / queries if queries else 0.0,
        "avg_heap_used": heap_used_sum / observations if observations else 0.0,
        "avg_fragmentation": fragmentation_sum / observations if observations else 0.0,
        "avg_object_count": object_sum / observations if observations else 0.0,
        "gc_cycles": float(gc_cycles),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UFO dataset experiments with agent memory.")
    parser.add_argument("--data-path", type=str, default="data/ufo.csv", help="Path to downloaded UFO dataset CSV.")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum number of records to process.")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed for shuffling the dataset.")
    parser.add_argument("--query-interval", type=int, default=10, help="Issue retrieval queries every N records.")
    parser.add_argument(
        "--fragmentation-threshold",
        type=float,
        default=0.35,
        help="Trigger GC when fragmentation exceeds this threshold.",
    )
    parser.add_argument("--root-capacity", type=int, default=12, help="Maximum number of persistent root pointers.")
    parser.add_argument("--output", type=str, default="results/ufo_summary.csv", help="Path to write summary CSV.")
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default=None,
        help="Optional directory for per-config trajectory CSV dumps.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        default=["ufo_first_fit_ttl_ms", "ufo_best_fit_gen"],
        help="Which experiment configurations to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sightings = list(load_ufo_dataset(args.data_path, limit=args.limit))
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(sightings)

    configs = {config.label: config for config in build_default_configs()}
    chosen_configs = [configs[label] for label in args.configs if label in configs]

    summaries: List[Dict[str, float]] = []

    for config in chosen_configs:
        summary = run_ufo_workload(
            config,
            sightings,
            query_interval=args.query_interval,
            fragmentation_threshold=args.fragmentation_threshold,
            root_capacity=args.root_capacity,
            trajectory_path=(
                f"{args.trajectory_dir}/{config.label}.csv" if args.trajectory_dir else None
            ),
        )
        summaries.append(summary)
        print(
            f"[{config.label}] "
            f"hit@1={summary['hit_at_1']:.3f} hit@3={summary['hit_at_3']:.3f} mrr={summary['mrr']:.3f} "
            f"avg_fragmentation={summary['avg_fragmentation']:.3f} "
            f"gc_cycles={int(summary['gc_cycles'])}"
        )

    if summaries:
        fieldnames = list(summaries[0].keys())
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
