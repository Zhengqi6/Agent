import time
import unittest

from agent_memory import (
    BestFitAllocator,
    MarkAndSweepCollector,
    MemoryManager,
    SimilarityRetriever,
    TimeToLiveCollector,
)


class MemoryManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        collectors = [TimeToLiveCollector(), MarkAndSweepCollector()]
        self.manager = MemoryManager(
            capacity=512,
            allocator=BestFitAllocator(),
            collectors=collectors,
            retriever=SimilarityRetriever(),
        )

    def test_store_and_stats(self) -> None:
        obj = self.manager.store({"text": "Plan experiment", "metadata": {"kind": "plan"}}, importance=0.9, pin=True)
        self.manager.add_root(obj.id)
        stats = self.manager.stats()
        self.assertEqual(stats["objects"], 1)
        self.assertGreater(stats["heap_used"], 0)

    def test_time_to_live_collection(self) -> None:
        collector_now = time.time()
        manager = MemoryManager(
            capacity=256,
            collectors=[TimeToLiveCollector(now=collector_now), MarkAndSweepCollector()],
        )
        obj = manager.store({"text": "Transient cache item"}, ttl=1.0)
        obj.created_at = collector_now - 5.0
        manager.run_gc()
        self.assertEqual(manager.stats()["objects"], 0)

    def test_retrieval_updates_reference_count(self) -> None:
        first = self.manager.store({"text": "Research memory allocation strategies"})
        second = self.manager.store({"text": "Evaluate garbage collector designs"})
        self.manager.add_root(first.id)
        self.manager.add_root(second.id)
        results = self.manager.retrieve("memory allocation")
        self.assertTrue(first in results or second in results)
        self.assertGreaterEqual(first.reference_count + second.reference_count, 1)


if __name__ == "__main__":
    unittest.main()
