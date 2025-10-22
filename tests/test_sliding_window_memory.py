import unittest

from experiments.baselines import ReservoirMemory, SlidingWindowMemory


class SlidingWindowMemoryTests(unittest.TestCase):
    def test_window_evicts_oldest(self) -> None:
        memory = SlidingWindowMemory(max_items=2)
        first = memory.store({"text": "first"})
        second = memory.store({"text": "second"})
        third = memory.store({"text": "third"})

        results = memory.retrieve("third", top_k=5)
        retrieved_payloads = [obj.payload["text"] for obj in results]

        self.assertIn("third", retrieved_payloads)
        first_results = memory.retrieve("first", top_k=5)
        first_payloads = [obj.payload["text"] for obj in first_results]
        self.assertNotIn("first", first_payloads)
        self.assertEqual(memory.stats()["objects"], 2.0)


class ReservoirMemoryTests(unittest.TestCase):
    def test_reservoir_limits_size(self) -> None:
        reservoir = ReservoirMemory(max_items=3)
        for idx in range(10):
            reservoir.store({"text": f"item-{idx}"})
        stats = reservoir.stats()
        self.assertEqual(stats["objects"], 3.0)
        self.assertEqual(stats["max_items"], 3.0)
        self.assertEqual(stats["seen"], 10.0)


if __name__ == "__main__":
    unittest.main()
