from __future__ import annotations

import argparse
import time

from agent_memory import (
    BestFitAllocator,
    GenerationalCollector,
    MarkAndSweepCollector,
    MemoryManager,
    SimilarityRetriever,
    TimeToLiveCollector,
)
from experiments.environment import SimulatedEnvironment


def run_simulation(iterations: int, capacity: int, query_interval: int) -> None:
    env = SimulatedEnvironment(seed=42)
    manager = MemoryManager(
        capacity,
        allocator=BestFitAllocator(),
        collectors=[
            TimeToLiveCollector(),
            GenerationalCollector(promotion_threshold=2, young_object_age=90.0),
            MarkAndSweepCollector(),
        ],
        retriever=SimilarityRetriever(),
    )

    for step in range(1, iterations + 1):
        experience = env.next_experience()
        payload = {
            "text": experience.text,
            "metadata": experience.metadata,
        }
        obj = manager.store(
            payload,
            tags=experience.tags,
            importance=experience.importance,
            ttl=experience.ttl,
            pin=experience.importance > 0.85,
        )
        manager.add_root(obj.id)

        if step % query_interval == 0:
            query = "memory policy execution"
            results = manager.retrieve(query, top_k=3)
            print(f"[step {step}] query '{query}' returned {[r.id for r in results]}")

        if manager.space.fragmentation() > 0.4:
            manager.run_gc(trigger="high_fragmentation")

    print("Final stats:", manager.stats())
    snapshot = manager.debug_snapshot()
    print("Heap map:", snapshot["space"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent memory simulation.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of experiences to generate.")
    parser.add_argument("--capacity", type=int, default=4096, help="Heap capacity in bytes.")
    parser.add_argument("--query-interval", type=int, default=5, help="Query memory every N iterations.")
    args = parser.parse_args()
    run_simulation(args.iterations, args.capacity, args.query_interval)
