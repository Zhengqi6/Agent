from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from agent_memory import (
    BestFitAllocator,
    GenerationalCollector,
    MarkAndSweepCollector,
    MemoryManager,
    SimilarityRetriever,
    TimeToLiveCollector,
)
from datasets.text_chunker import TextChunk, load_text_chunks
from experiments.baselines import ReservoirMemory, SlidingWindowMemory
from experiments.instrumentation import MemoryProfiler


def tokenize(text: str) -> List[str]:
    tokens = []
    for token in text.split():
        token = "".join(ch for ch in token if ch.isalnum())
        if token:
            tokens.append(token.lower())
    return tokens


def chunk_to_payload(chunk: TextChunk) -> Dict[str, object]:
    return {
        "text": chunk.text,
        "metadata": {
            "document_id": chunk.document_id,
            "start": chunk.start,
            "end": chunk.end,
        },
    }


CollectFactoryList = List


COLLECTOR_VARIANTS: Dict[str, CollectFactoryList] = {
    "ttl_mark": [TimeToLiveCollector, MarkAndSweepCollector],
    "ttl_mark_gen": [TimeToLiveCollector, MarkAndSweepCollector, GenerationalCollector],
}

BASELINE_BUILDERS = {
    "sliding": lambda window: SlidingWindowMemory(max_items=window),
    "reservoir": lambda window: ReservoirMemory(max_items=window),
}


def run_capacity_trial(
    chunks: Sequence[TextChunk],
    capacity: int,
    *,
    window_size: int,
    query_interval: int,
    rng: random.Random,
    collector_name: str,
    collector_factories: CollectFactoryList,
    baseline_name: str,
) -> Dict[str, float]:
    profiler = MemoryProfiler(run_id=f"capacity_{capacity}_{collector_name}_{baseline_name}")
    manager = MemoryManager(
        capacity,
        allocator=BestFitAllocator(),
        collectors=[factory() for factory in collector_factories],
        retriever=SimilarityRetriever(),
        profiler=profiler,
    )
    baseline = BASELINE_BUILDERS[baseline_name](window_size)

    manager_hits = 0
    baseline_hits = 0
    queries = 0

    root_queue: List[str] = []
    root_budget = max(5, window_size // 2)

    for step, chunk in enumerate(chunks, start=1):
        payload = chunk_to_payload(chunk)
        tokens = tokenize(chunk.text)
        importance = min(len(tokens) / 120.0, 1.0)
        ttl = 900.0 if len(tokens) < 80 else 1500.0
        size_hint = max(64, min(192, len(chunk.text) // 4))

        obj = manager.store(payload, size=size_hint, importance=importance, ttl=ttl)
        manager.add_root(obj.id)
        root_queue.append(obj.id)
        if len(root_queue) > root_budget:
            removed = root_queue.pop(0)
            manager.remove_root(removed)
        baseline.store(payload, size=size_hint, importance=importance)

        if step % (window_size) == 0:
            manager.run_gc(trigger="periodic_sweep")

        if tokens and step % query_interval == 0:
            queries += 1
            keyword = rng.choice(tokens)
            query = f"{keyword} scene"

            mgr_objs = manager.retrieve(query, top_k=1)
            base_objs = baseline.retrieve(query, top_k=1)

            if mgr_objs and tokens_match(mgr_objs[0].payload["text"], keyword):
                manager_hits += 1
            if base_objs and tokens_match(base_objs[0].payload["text"], keyword):
                baseline_hits += 1

    profiler.flush()

    return {
        "capacity": float(capacity),
        "queries": float(queries),
        "manager_hit1": manager_hits / queries if queries else 0.0,
        "baseline_hit1": baseline_hits / queries if queries else 0.0,
        "delta_hit1": (manager_hits - baseline_hits) / queries if queries else 0.0,
        "allocator": "BestFitAllocator",
        "window_size": float(window_size),
        "manager_variant": collector_name,
        "baseline": baseline_name,
    }


def tokens_match(text: str, keyword: str) -> bool:
    return keyword.lower() in tokenize(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capacity ablation for MemoryManager vs sliding window baseline.")
    parser.add_argument("--data-path", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--capacities", type=int, nargs="+", default=[4096, 8192, 12288, 16384])
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--query-interval", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--collector-variants",
        type=str,
        nargs="+",
        default=["ttl_mark", "ttl_mark_gen"],
        choices=list(COLLECTOR_VARIANTS.keys()),
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["sliding", "reservoir"],
        choices=list(BASELINE_BUILDERS.keys()),
    )
    parser.add_argument("--output", type=str, default="results/capacity_sweep.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = list(
        load_text_chunks(
            args.data_path,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            limit=args.limit,
        )
    )
    if not chunks:
        raise RuntimeError("No chunks generated from dataset.")

    rows: List[Dict[str, float]] = []
    for capacity in args.capacities:
        for seed_idx in range(args.seeds):
            seed = args.seed_offset + seed_idx
            shuffled = chunks[:]
            random.Random(seed).shuffle(shuffled)
            for collector_name in args.collector_variants:
                factories = COLLECTOR_VARIANTS[collector_name]
                for baseline_name in args.baselines:
                    trial_rng = random.Random(seed)
                    summary = run_capacity_trial(
                        shuffled,
                        capacity,
                        window_size=args.window_size,
                        query_interval=args.query_interval,
                        rng=trial_rng,
                        collector_name=collector_name,
                        collector_factories=factories,
                        baseline_name=baseline_name,
                    )
                    summary["seed"] = float(seed)
                    rows.append(summary)
                    print(
                        f"[capacity={capacity} seed={seed} manager={collector_name} baseline={baseline_name}] "
                        f"hit@1(manager)={summary['manager_hit1']:.3f} "
                        f"hit@1({baseline_name})={summary['baseline_hit1']:.3f}"
                    )

    write_summary(args.output, rows)


def write_summary(path: str, rows: Sequence[Dict[str, float]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[Tuple[int, str, str], List[float]] = {}
    for row in rows:
        key = (int(row["capacity"]), row["manager_variant"], row["baseline"])
        grouped.setdefault(key, []).append(row["delta_hit1"])

    for (cap, manager_variant, baseline_name), values in grouped.items():
        avg = statistics.mean(values)
        print(
            f"[capacity={cap} manager={manager_variant} baseline={baseline_name}] "
            f"Δhit@1 mean={avg:.3f}"
        )


if __name__ == "__main__":
    main()
