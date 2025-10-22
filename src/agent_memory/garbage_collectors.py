from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .memory_manager import MemoryManager


class GarbageCollector(ABC):
    """Common interface for all GC strategies."""

    name: str

    @abstractmethod
    def collect(self, manager: "MemoryManager") -> List[str]:
        """
        Execute a collection cycle.

        Returns:
            list of object ids that were freed during the cycle.
        """

    def should_trigger(self, manager: "MemoryManager") -> bool:
        """Allow collectors to short-circuit if no action is required."""
        return True


class MarkAndSweepCollector(GarbageCollector):
    def __init__(self) -> None:
        self.name = "mark_and_sweep"

    def collect(self, manager: "MemoryManager") -> List[str]:
        reachable = manager.reachable_ids()
        freed: List[str] = []
        for object_id in list(manager.objects.keys()):
            if object_id not in reachable and not manager.is_pinned(object_id):
                freed.append(object_id)
                manager.free_object(object_id)
        return freed


class TimeToLiveCollector(GarbageCollector):
    def __init__(self, now: Optional[float] = None) -> None:
        self.name = "ttl"
        self._now_override = now

    def collect(self, manager: "MemoryManager") -> List[str]:
        now = self._now_override or time.time()
        freed: List[str] = []
        for obj in list(manager.objects.values()):
            if obj.expired(now):
                freed.append(obj.id)
                manager.free_object(obj.id)
        return freed


class GenerationalCollector(GarbageCollector):
    def __init__(
        self,
        promotion_threshold: int = 3,
        retention_generations: int = 2,
        young_object_age: float = 300.0,
    ) -> None:
        """
        promotion_threshold: number of touches before an object is promoted.
        retention_generations: how many generations to keep before relying on other GC.
        young_object_age: age in seconds after which young objects can be collected.
        """

        self.name = "generational"
        self.promotion_threshold = promotion_threshold
        self.retention_generations = retention_generations
        self.young_object_age = young_object_age

    def collect(self, manager: "MemoryManager") -> List[str]:
        now = time.time()
        freed: List[str] = []

        # Promote hot objects first so they are shielded from the young collection pass.
        for obj in manager.objects.values():
            if obj.generation < self.retention_generations and obj.reference_count >= self.promotion_threshold:
                obj.generation += 1

        for obj in list(manager.objects.values()):
            if obj.generation == 0:
                if now - obj.last_accessed_at > self.young_object_age and obj.importance < 0.3:
                    freed.append(obj.id)
                    manager.free_object(obj.id)
            elif obj.generation >= self.retention_generations:
                # Delegate to mark and sweep once objects reach the old generation.
                continue

        return freed
