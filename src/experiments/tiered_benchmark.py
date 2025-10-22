from __future__ import annotations

import argparse
import csv
import random
from typing import Dict, Iterable, List, Optional

from agent_memory import (
    BestFitAllocator,
    MarkAndSweepCollector,
    MemoryManager,
    SimilarityRetriever,
    TimeToLiveCollector,
)
from agent_memory.tiered_manager import TieredMemoryManager
from datasets.ufo_loader import UfoSighting, load_ufo_dataset


class ManagerAdapter:
    def __init__(self, manager: MemoryManager) -> None:
        self.manager = manager

    def store(self, payload: Dict[str, object], importance: float, ttl: Optional[float]) -> None:
        self.manager.store(payload, importance=importance, ttl=ttl)

    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, object]]:
        return [obj.payload for obj in self.manager.retrieve(query, top_k=top_k)]

    def stats(self) -> Dict[str, float]:
        return self.manager.stats()


class TieredAdapter:
    def __init__(self, manager: TieredMemoryManager) -> None:
        self.manager = manager

    def store(self, payload: Dict[str, object], importance: float, ttl: Optional[float]) -> None:
        self.manager.store(payload, importance=importance, ttl=ttl)

    def retrieve(self, query: str, top_k: int = 1) -> List[Dict[str, object]]:
        return [obj.payload for obj in self.manager.retrieve(query, top_k=top_k)]

    def stats(self) -> Dict[str, float]:
        return self.manager.stats()


def payload_from_sighting(sighting: UfoSighting) -> Dict[str, object]:
    text = (
        f"{sighting.datetime or 'Unknown time'} - {sighting.city or 'Unknown city'}, "
        f"{sighting.state or ''} {sighting.country or ''}: "
        f"{sighting.comments or 'No description provided.'} "
        f"Shape={sighting.shape or 'unknown'}."
    )
    metadata = {
        "city": sighting.city or "",
        "state": sighting.state or "",
        "country": sighting.country or "",
        "shape": sighting.shape or "",
    }
    return {"text": text, "metadata": metadata}


def compute_importance(sighting: UfoSighting) -> float:
    base = min(len(sighting.comments or "") / 400.0, 1.0)
    if (sighting.shape or "").lower() in {"triangle", "fireball", "disk"}:
        base = min(1.0, base + 0.2)
    return base


def compute_ttl(sighting: UfoSighting) -> Optional[float]:
    if not sighting.shape:
        return 600.0
    if (sighting.shape or "").lower() in {"triangle", "fireball"}:
        return None
    return 900.0


def evaluate_manager(
    adapter: ManagerAdapter | TieredAdapter,
    sightings: Iterable[UfoSighting],
    *,
    query_interval: int,
) -> Dict[str, float]:
    hits = 0
    queries = 0
    steps = 0
    for sighting in sightings:
        steps += 1
        payload = payload_from_sighting(sighting)
        importance = compute_importance(sighting)
        ttl = compute_ttl(sighting)
        adapter.store(payload, importance=importance, ttl=ttl)

        if sighting.shape and steps % query_interval == 0:
            query = f"{sighting.shape.lower()} sighting"
            result_payloads = adapter.retrieve(query, top_k=1)
            queries += 1
            if result_payloads:
                metadata = result_payloads[0].get("metadata", {})
                if (metadata.get("shape", "") or "").lower() == (sighting.shape or "").lower():
                    hits += 1

    stats = adapter.stats()
    stats.update(
        {
            "steps": float(steps),
            "queries": float(queries),
            "hit_at_1": hits / queries if queries else 0.0,
        }
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark tiered vs baseline memory managers.")
    parser.add_argument("--data-path", type=str, default="data/ufo.csv")
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--query-interval", type=int, default=12)
    parser.add_argument("--hot-capacity", type=int, default=8192)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--paging-strategy", type=str, default="lru", choices=["lru", "clock", "lfu"])
    parser.add_argument("--write-policy", type=str, default="write_back", choices=["write_back", "write_through"])
    parser.add_argument("--promotion-batch", type=int, default=1)
    parser.add_argument("--output", type=str, default="results/tiered_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_results: List[Dict[str, float]] = []

    for index in range(args.seeds):
        seed = args.seed_offset + index
        rng = random.Random(seed)
        sightings = list(load_ufo_dataset(args.data_path, limit=args.limit))
        rng.shuffle(sightings)

        baseline_manager = MemoryManager(
            args.hot_capacity,
            allocator=BestFitAllocator(),
            collectors=[TimeToLiveCollector(), MarkAndSweepCollector()],
            retriever=SimilarityRetriever(),
        )
        tiered_manager = TieredMemoryManager(
            hot_capacity=args.hot_capacity // 2,
            cold_retriever=SimilarityRetriever(),
            enable_paging=True,
            page_size=256,
            max_pages=256,
            paging_strategy=args.paging_strategy,
            write_policy=args.write_policy,
            promotion_batch_size=args.promotion_batch,
        )

        baseline_stats = evaluate_manager(
            ManagerAdapter(baseline_manager),
            sightings,
            query_interval=args.query_interval,
        )
        baseline_stats.update({"seed": float(seed), "variant": "baseline"})

        tiered_stats = evaluate_manager(
            TieredAdapter(tiered_manager),
            sightings,
            query_interval=args.query_interval,
        )
        tiered_stats.update({"seed": float(seed), "variant": "tiered"})

        base_results.append(baseline_stats)
        base_results.append(tiered_stats)

        print(
            f"[seed={seed}] baseline hit@1={baseline_stats['hit_at_1']:.3f} | "
            f"tiered hit@1={tiered_stats['hit_at_1']:.3f} "
            f"(hot_hits={tiered_stats.get('hot_hits', 0):.0f}, "
            f"cold_hits={tiered_stats.get('cold_hits', 0):.0f})"
        )

    if base_results:
        fieldnames = sorted({key for row in base_results for key in row.keys()})
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(base_results)


if __name__ == "__main__":
    main()
