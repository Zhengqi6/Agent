import unittest

from experiments.benchmark_memory import ExperimentConfig, run_single
from agent_memory import FirstFitAllocator, MarkAndSweepCollector, TimeToLiveCollector


class BenchmarkHarnessTests(unittest.TestCase):
    def test_summary_contains_core_metrics(self) -> None:
        config = ExperimentConfig(
            label="test_config",
            capacity=1024,
            allocator_factory=FirstFitAllocator,
            collector_factories=[TimeToLiveCollector, MarkAndSweepCollector],
            steps=12,
            query_interval=3,
            fragmentation_threshold=0.5,
            root_capacity=3,
        )
        summary = run_single(config, seed=123)
        self.assertIn("hit_at_1", summary)
        self.assertIn("avg_fragmentation", summary)
        self.assertGreaterEqual(summary["queries"], 1)
        self.assertGreaterEqual(summary["gc_cycles"], 0)


if __name__ == "__main__":
    unittest.main()
