from __future__ import annotations

import argparse
import csv
import random
from typing import Dict, Iterable, List, Sequence, Tuple

from agent_memory import (
    BestFitAllocator,
    GenerationalCollector,
    MarkAndSweepCollector,
    MemoryManager,
    SimilarityRetriever,
    TimeToLiveCollector,
)
from datasets.ufo_loader import UfoSighting, load_ufo_dataset
from experiments.instrumentation import MemoryProfiler


CollectorConfig = Tuple[str, List]


COLLECTOR_VARIANTS: Dict[str, CollectorConfig] = {
    "ttl_only": ("TTL", [TimeToLiveCollector]),
    "ttl_mark": ("TTL+MarkSweep", [TimeToLiveCollector, MarkAndSweepCollector]),
    "ttl_mark_gen": ("TTL+MarkSweep+Generational", [TimeToLiveCollector, MarkAndSweepCollector, GenerationalCollector]),
    "mark_only": ("MarkSweep", [MarkAndSweepCollector]),
}


def sighting_payload(sighting: UfoSighting) -> Dict[str, object]:
    return {
        "text": f"{sighting.datetime}: {sighting.comments or ''} ({sighting.shape}) in {sighting.city}, {sighting.state}",
        "metadata": {
            "city": sighting.city,
            "state": sighting.state,
            "country": sighting.country,
            "shape": sighting.shape,
            "datetime": sighting.datetime,
        },
    }


def run_gc_trial(
    sightings: Sequence[UfoSighting],
    collector_factories: List,
    *,
    capacity: int,
    query_interval: int,
    rng: random.Random,
) -> Dict[str, float]:
    profiler = MemoryProfiler(run_id=f"gc_{len(collector_factories)}")
    collectors = [factory() for factory in collector_factories]
    manager = MemoryManager(
        capacity,
        allocator=BestFitAllocator(),
        collectors=collectors,
        retriever=SimilarityRetriever(),
        profiler=profiler,
    )

    hits = 0
    queries = 0

    root_queue: List[str] = []
    root_budget = 15

    for step, sighting in enumerate(sightings, start=1):
        payload = sighting_payload(sighting)
        importance = 1.0 if (sighting.shape or "").lower() in {"triangle", "fireball"} else 0.5
        ttl = 900.0 if importance > 0.8 else 450.0
        size_hint = max(48, min(160, len(str(payload["text"])) // 4))
        try:
            obj = manager.store(payload, size=size_hint, importance=importance, ttl=ttl)
        except MemoryError:
            manager.run_gc(trigger="memory_error")
            continue
        if importance >= 0.7:
            while len(root_queue) >= root_budget:
                removed = root_queue.pop(0)
                manager.remove_root(removed)
            manager.add_root(obj.id)
            root_queue.append(obj.id)

        if step % query_interval == 0 and sighting.shape:
            queries += 1
            query = f"{sighting.shape.lower()} sighting"
            retrieved = manager.retrieve(query, top_k=1)
            if retrieved and retrieved[0].payload["metadata"].get("shape", "").lower() == sighting.shape.lower():
                hits += 1

        if step % 40 == 0:
            manager.run_gc(trigger="periodic")

    profiler.flush()
    stats = manager.stats()
    gc_counts = {collector["collector"]: collector["freed"] for event in manager.gc_events for collector in event["collectors"]}
    total_gc_cycles = len(manager.gc_events)

    return {
        "capacity": float(capacity),
        "collector_combo": "+".join(factory.__name__ for factory in collector_factories),
        "queries": float(queries),
        "hit_at_1": hits / queries if queries else 0.0,
        "final_heap_used": float(stats["heap_used"]),
        "final_fragmentation": stats["fragmentation"],
        "gc_cycles": float(total_gc_cycles),
        "gc_ttl_freed": float(gc_counts.get("ttl", 0)),
        "gc_mark_freed": float(gc_counts.get("mark_and_sweep", 0)),
        "gc_gen_freed": float(gc_counts.get("generational", 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Garbage collector ablation using UFO dataset.")
    parser.add_argument("--data-path", type=str, default="data/ufo.csv")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--capacity", type=int, default=16384)
    parser.add_argument("--query-interval", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/gc_ablation.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sightings = list(load_ufo_dataset(args.data_path, limit=args.limit))
    if not sightings:
        raise RuntimeError("Dataset is empty.")

    rows: List[Dict[str, float]] = []
    for variant, (_, factories) in COLLECTOR_VARIANTS.items():
        for seed_idx in range(args.seeds):
            seed = args.seed_offset + seed_idx
            rng = random.Random(seed)
            shuffled = sightings[:]
            rng.shuffle(shuffled)
            summary = run_gc_trial(
                shuffled,
                factories,
                capacity=args.capacity,
                query_interval=args.query_interval,
                rng=rng,
            )
            summary["variant"] = variant
            summary["seed"] = float(seed)
            rows.append(summary)
            print(
                f"[{variant} seed={seed}] hit@1={summary['hit_at_1']:.3f} "
                f"gc_cycles={summary['gc_cycles']:.0f} heap_used={summary['final_heap_used']:.1f}"
            )

    write_results(args.output, rows)


def write_results(path: str, rows: Sequence[Dict[str, float]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
