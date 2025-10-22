from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, List

if TYPE_CHECKING:
    from .memory_manager import MemoryManager


class GCPolicy(ABC):
    @abstractmethod
    def should_trigger(self, manager: "MemoryManager", reason: str) -> bool:
        """Return True if GC should run given the current reason."""

    def notify_gc(self, manager: "MemoryManager", event: dict) -> None:
        """Called after GC completes; `event` contains GC statistics."""


class PeriodicGCPolicy(GCPolicy):
    def __init__(self, interval: int) -> None:
        self.interval = interval
        self._counter = 0

    def should_trigger(self, manager: "MemoryManager", reason: str) -> bool:
        self._counter += 1
        if self._counter >= self.interval:
            self._counter = 0
            return True
        return False


class AdaptiveFragmentationPolicy(GCPolicy):
    def __init__(self, threshold: float = 0.4, min_interval_s: float = 5.0) -> None:
        self.threshold = threshold
        self.min_interval_s = min_interval_s
        self._last_triggered: float = 0.0

    def should_trigger(self, manager: "MemoryManager", reason: str) -> bool:
        if reason not in {"allocation", "tick", "external"}:
            return False
        now = time.time()
        if now - self._last_triggered < self.min_interval_s:
            return False
        if manager.space.fragmentation() >= self.threshold:
            self._last_triggered = now
            return True
        return False

    def notify_gc(self, manager: "MemoryManager", event: dict) -> None:
        self._last_triggered = time.time()


class ObjectGrowthPolicy(GCPolicy):
    def __init__(self, growth_ratio: float = 0.5, window: int = 100) -> None:
        self.growth_ratio = growth_ratio
        self.window = window
        self._baseline_objects: Optional[int] = None
        self._events = 0

    def should_trigger(self, manager: "MemoryManager", reason: str) -> bool:
        if reason not in {"store", "tick"}:
            return False
        self._events += 1
        if self._events < self.window:
            return False
        self._events = 0
        current = len(manager.objects)
        if self._baseline_objects is None:
            self._baseline_objects = max(current, 1)
            return False
        if current >= self._baseline_objects * (1.0 + self.growth_ratio):
            self._baseline_objects = current
            return True
        return False


class BanditGCPolicy(GCPolicy):
    """Epsilon-greedy selector over multiple GC policies."""

    def __init__(
        self,
        policies: List[GCPolicy],
        *,
        epsilon: float = 0.1,
        pause_penalty: float = 0.1,
    ) -> None:
        if not policies:
            raise ValueError("BanditGCPolicy requires at least one policy")
        self.policies = policies
        self.epsilon = epsilon
        self.pause_penalty = pause_penalty
        self.counts = [0.0] * len(policies)
        self.values = [0.0] * len(policies)
        self._last_idx: Optional[int] = None

    def should_trigger(self, manager: "MemoryManager", reason: str) -> bool:
        if len(self.policies) == 1:
            idx = 0
        else:
            if random.random() < self.epsilon:
                idx = random.randrange(len(self.policies))
            else:
                idx = max(range(len(self.policies)), key=lambda i: self.values[i])
        if self.policies[idx].should_trigger(manager, reason):
            self._last_idx = idx
            return True
        self._last_idx = None
        return False

    def notify_gc(self, manager: "MemoryManager", event: dict) -> None:
        idx = self._last_idx
        if idx is None:
            return
        reward = len(event.get("freed_ids", [])) - self.pause_penalty * event.get("pause_duration", 0.0)
        self.counts[idx] += 1
        n = self.counts[idx]
        self.values[idx] += (reward - self.values[idx]) / n
        self.policies[idx].notify_gc(manager, event)
        self._last_idx = None
