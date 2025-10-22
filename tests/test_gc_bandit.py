import unittest

from agent_memory.gc_policy import BanditGCPolicy, GCPolicy
from agent_memory.memory_manager import MemoryManager
from agent_memory import TimeToLiveCollector


class DummyPolicy(GCPolicy):
    def __init__(self, should_trigger=True):
        self.should_trigger_flag = should_trigger
        self.trigger_calls = 0
        self.notified_events = []

    def should_trigger(self, manager: MemoryManager, reason: str) -> bool:
        self.trigger_calls += 1
        return self.should_trigger_flag

    def notify_gc(self, manager: MemoryManager, event: dict) -> None:
        self.notified_events.append(event)


class BanditPolicyTests(unittest.TestCase):
    def test_bandit_updates_values(self) -> None:
        true_policy = DummyPolicy(should_trigger=True)
        false_policy = DummyPolicy(should_trigger=False)
        bandit = BanditGCPolicy([true_policy, false_policy], epsilon=0.0, pause_penalty=0.0)

        manager = MemoryManager(256, collectors=[TimeToLiveCollector()], policies=[bandit])

        result = bandit.should_trigger(manager, "tick")
        self.assertTrue(result)
        self.assertEqual(true_policy.trigger_calls, 1)
        self.assertEqual(false_policy.trigger_calls, 0)

        event = {"freed_ids": ["a", "b"], "pause_duration": 0.5}
        bandit.notify_gc(manager, event)
        self.assertAlmostEqual(bandit.values[0], 2.0)
        self.assertEqual(len(true_policy.notified_events), 1)

    def test_bandit_handles_no_trigger(self) -> None:
        false_policy = DummyPolicy(should_trigger=False)
        bandit = BanditGCPolicy([false_policy], epsilon=0.0)
        manager = MemoryManager(256, collectors=[TimeToLiveCollector()], policies=[bandit])
        self.assertFalse(bandit.should_trigger(manager, "tick"))
        bandit.notify_gc(manager, {"freed_ids": [], "pause_duration": 1.0})
        self.assertEqual(bandit.values[0], 0.0)


if __name__ == "__main__":
    unittest.main()
