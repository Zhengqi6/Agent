from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from agent_memory import (
    BestFitAllocator,
    MemoryManager,
    SimilarityRetriever,
    TimeToLiveCollector,
    MarkAndSweepCollector,
)
from datasets.ufo_loader import load_ufo_dataset, UfoSighting
from experiments.baselines import SlidingWindowMemory


@dataclass
class QueryOutcome:
    correct_top1: bool
    correct_top3: bool
    correct_top5: bool
    mrr: float
    position: Optional[int]


def build_payload(sighting: UfoSighting) -> Dict[str, object]:
    text = (
        f"{sighting.datetime or 'Unknown time'} | {sighting.city or 'Unknown city'} "
        f"{sighting.state or ''} {sighting.country or ''}: "
        f"{sighting.comments or 'No comments provided.'} "
        f"Shape={sighting.shape or 'unknown'}, Colors={sighting.colors or 'unspecified'}, "
        f"Duration={sighting.duration or 'unknown'}."
    )
    metadata = {
        "city": sighting.city,
        "state": sighting.state,
        "country": sighting.country,
        "shape": sighting.shape,
        "datetime": sighting.datetime,
        "duration": sighting.duration,
        "colors": sighting.colors,
    }
    return {"text": text, "metadata": metadata}


def make_query(sighting: UfoSighting) -> Tuple[str, Dict[str, str]]:
    metadata = {
        "shape": (sighting.shape or "unknown").lower(),
        "city": (sighting.city or "unknown").lower(),
    }
    description = (
        f"report of {metadata['shape']} craft spotted in {metadata['city']} "
        f"around {sighting.datetime or 'an unspecified time'}"
    )
    return description, metadata


def evaluate_retrieval(
    retrieved: Sequence[Dict[str, object]],
    target_metadata: Dict[str, str],
) -> QueryOutcome:
    target_shape = target_metadata["shape"]
    target_city = target_metadata["city"]
    mrr = 0.0
    position: Optional[int] = None

    for idx, entry in enumerate(retrieved, start=1):
        metadata = entry["metadata"]
        candidate_shape = str(metadata.get("shape", "unknown")).lower()
        candidate_city = str(metadata.get("city", "unknown")).lower()
        if candidate_shape == target_shape and candidate_city == target_city:
            position = idx
            mrr = 1.0 / idx
            break

    return QueryOutcome(
        correct_top1=position == 1,
        correct_top3=position is not None and position <= 3,
        correct_top5=position is not None and position <= 5,
        mrr=mrr,
        position=position,
    )


def run_trial(
    sightings: Sequence[UfoSighting],
    *,
    session_length: int,
    queries_per_session: int,
    manager_capacity: int,
    window_size: int,
    rng: random.Random,
    dense_model: Optional[str] = None,
) -> Dict[str, float]:
    retriever_kwargs = {"dense_model": dense_model} if dense_model else {}
    manager = MemoryManager(
        manager_capacity,
        allocator=BestFitAllocator(),
        collectors=[TimeToLiveCollector(), MarkAndSweepCollector()],
        retriever=SimilarityRetriever(**retriever_kwargs),
    )
    baseline = SlidingWindowMemory(max_items=window_size)

    manager_results: List[QueryOutcome] = []
    baseline_results: List[QueryOutcome] = []
    root_queue: deque[str] = deque()

    for session_start in range(0, len(sightings), session_length):
        session = sightings[session_start : session_start + session_length]
        if not session:
            break

        for sighting in session:
            payload = build_payload(sighting)
            importance = min(len(str(payload["text"])) / 600.0, 1.0)
            ttl = None if importance > 0.85 else 600.0
            text_length = len(str(payload["text"]))
            size_hint = max(32, min(128, text_length // 8 if text_length // 8 else 32))
            obj = manager.store(
                payload,
                size=size_hint,
                tags=[sighting.shape or "unknown"],
                importance=importance,
                ttl=ttl,
            )
            manager.add_root(obj.id)
            root_queue.append(obj.id)
            while len(root_queue) > window_size:
                removed = root_queue.popleft()
                manager.remove_root(removed)
            baseline.store(payload, size=size_hint, tags=[sighting.shape or "unknown"], importance=importance)

        sampled = rng.sample(session, k=min(len(session), queries_per_session))
        for sighting in sampled:
            query, target_meta = make_query(sighting)
            mgr_objs = manager.retrieve(query, top_k=5)
            base_objs = baseline.retrieve(query, top_k=5)
            mgr_payloads = [obj.payload for obj in mgr_objs]
            base_payloads = [obj.payload for obj in base_objs]
            manager_results.append(evaluate_retrieval(mgr_payloads, target_meta))
            baseline_results.append(evaluate_retrieval(base_payloads, target_meta))

        manager.run_gc(trigger="session_end")

    return summarise_results(manager_results, baseline_results)


def summarise_results(
    manager_results: Iterable[QueryOutcome],
    baseline_results: Iterable[QueryOutcome],
) -> Dict[str, float]:
    mgr = list(manager_results)
    base = list(baseline_results)

    def aggregate(outcomes: List[QueryOutcome]) -> Dict[str, float]:
        if not outcomes:
            return {"hit1": 0.0, "hit3": 0.0, "hit5": 0.0, "mrr": 0.0}
        return {
            "hit1": sum(o.correct_top1 for o in outcomes) / len(outcomes),
            "hit3": sum(o.correct_top3 for o in outcomes) / len(outcomes),
            "hit5": sum(o.correct_top5 for o in outcomes) / len(outcomes),
            "mrr": sum(o.mrr for o in outcomes) / len(outcomes),
        }

    mgr_stats = aggregate(mgr)
    base_stats = aggregate(base)

    improvement = {
        "improve_hit1": mgr_stats["hit1"] - base_stats["hit1"],
        "improve_hit3": mgr_stats["hit3"] - base_stats["hit3"],
        "improve_hit5": mgr_stats["hit5"] - base_stats["hit5"],
        "improve_mrr": mgr_stats["mrr"] - base_stats["mrr"],
    }

    return {
        **{f"manager_{key}": value for key, value in mgr_stats.items()},
        **{f"baseline_{key}": value for key, value in base_stats.items()},
        **improvement,
        "num_queries": float(len(mgr)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-context retrieval study comparing memory systems.")
    parser.add_argument("--data-path", type=str, default="data/ufo.csv")
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--session-length", type=int, default=60)
    parser.add_argument("--queries-per-session", type=int, default=8)
    parser.add_argument("--manager-capacity", type=int, default=8192)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/long_context_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sightings = list(load_ufo_dataset(args.data_path, limit=args.limit))
    if not sightings:
        raise RuntimeError("Dataset is empty; download data/ufo.csv first.")

    configurations = [
        ("sparse", None),
        ("dense", "all-MiniLM-L6-v2"),
    ]

    results: List[Dict[str, float]] = []
    for variant_name, dense_model in configurations:
        for offset in range(args.seeds):
            seed = args.seed_offset + offset
            rng = random.Random(seed)
            shuffled = sightings[:]
            rng.shuffle(shuffled)
            summary = run_trial(
                shuffled,
                session_length=args.session_length,
                queries_per_session=args.queries_per_session,
                manager_capacity=args.manager_capacity,
                window_size=args.window_size,
                rng=rng,
                dense_model=dense_model,
            )
            summary["seed"] = float(seed)
            summary["retriever"] = variant_name
            results.append(summary)
            print(
                f"[variant={variant_name} seed={seed}] manager_hit@1={summary['manager_hit1']:.3f} "
                f"baseline_hit@1={summary['baseline_hit1']:.3f} "
                f"Δhit@1={summary['improve_hit1']:.3f} "
                f"ΔMRR={summary['improve_mrr']:.3f}"
            )

    if results:
        aggregates = {}
        for key in results[0].keys():
            values = [row[key] for row in results]
            if all(isinstance(value, float) for value in values):
                aggregates[f"mean_{key}"] = statistics.mean(values)
                if len(values) > 1:
                    aggregates[f"stdev_{key}"] = statistics.pstdev(values)
        print("Aggregate:", {k: round(v, 3) for k, v in aggregates.items() if isinstance(v, float)})
        write_csv(args.output, results, aggregates)


def write_csv(path: str, rows: Sequence[Dict[str, float]], aggregates: Dict[str, float]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        aggregate_row = {key: aggregates.get(f"mean_{key}") for key in fieldnames}
        aggregate_row.update({"seed": "mean"})
        writer.writerow(aggregate_row)


if __name__ == "__main__":
    main()
