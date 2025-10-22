import unittest

from agent_memory import MemoryManager, TimeToLiveCollector
from agent_memory.gc_policy import PeriodicGCPolicy, AdaptiveFragmentationPolicy


class GCPPolicyTests(unittest.TestCase):
    def test_periodic_policy_triggers(self) -> None:
        manager = MemoryManager(256, collectors=[TimeToLiveCollector()], policies=[PeriodicGCPolicy(interval=2)])
        manager.store({"text": "a"})
        manager.store({"text": "b"})
        self.assertGreaterEqual(len(manager.gc_events), 1)

    def test_fragmentation_policy_respects_threshold(self) -> None:
        policy = AdaptiveFragmentationPolicy(threshold=0.0, min_interval_s=0.0)
        manager = MemoryManager(256, collectors=[TimeToLiveCollector()], policies=[policy])
        manager.store({"text": "c"})
        manager.tick()
        self.assertGreaterEqual(len(manager.gc_events), 1)


if __name__ == "__main__":
    unittest.main()
