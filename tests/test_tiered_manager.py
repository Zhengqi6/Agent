import unittest

from agent_memory.tiered_manager import TieredMemoryManager


class TieredManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = TieredMemoryManager(
            hot_capacity=256,
            max_hot_objects=2,
            enable_paging=True,
            page_size=64,
            max_pages=8,
            paging_strategy="clock",
        )

    def test_demotes_to_cold_when_hot_limit_exceeded(self) -> None:
        self.manager.store({"text": "alpha memory"}, importance=0.2)
        self.manager.store({"text": "beta memory"}, importance=0.2)
        self.manager.store({"text": "gamma memory"}, importance=0.2)

        self.assertLessEqual(len(self.manager.hot_manager.objects), 2)
        self.assertGreaterEqual(self.manager.metrics.demotions, 1)
        cold_ids = self.manager.all_cold_ids()
        self.assertTrue(len(cold_ids) >= 1)

        results = self.manager.retrieve("alpha", top_k=1)
        self.assertTrue(results)
        self.assertIn("alpha", results[0].payload["text"])
        self.assertGreaterEqual(self.manager.metrics.promotions, 1)
        self.assertIn(results[0].id, self.manager.hot_manager.objects)
        stats = self.manager.stats()
        self.assertGreaterEqual(stats["page_faults"], 1.0)

    def test_write_through_persists_immediately(self) -> None:
        manager = TieredMemoryManager(
            hot_capacity=256,
            max_hot_objects=2,
            enable_paging=False,
            write_policy="write_through",
        )
        obj = manager.store({"text": "delta"})
        self.assertTrue(manager.cold_store.contains(obj.id))


if __name__ == "__main__":
    unittest.main()
