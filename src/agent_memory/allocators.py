from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .memory_object import MemoryObject
from .memory_space import MemorySegment, MemorySpace


class Allocator(ABC):
    """Abstract allocation strategy."""

    @abstractmethod
    def allocate(self, space: MemorySpace, obj: MemoryObject) -> Optional[MemorySegment]:
        ...


class FirstFitAllocator(Allocator):
    """
    Basic first-fit allocator: traverse the free list in address order and pick
    the first segment that fits. Works well on low-fragmentation heaps.
    """

    def allocate(self, space: MemorySpace, obj: MemoryObject) -> Optional[MemorySegment]:
        return space.allocate(obj.size, obj.id)


class BestFitAllocator(Allocator):
    """
    Best-fit allocator: choose the smallest free segment that can fit the object.
    Reduces wasted space at the cost of extra scanning overhead.
    """

    def allocate(self, space: MemorySpace, obj: MemoryObject) -> Optional[MemorySegment]:
        candidates: List[MemorySegment] = [
            segment for segment in space.free_segments() if segment.size >= obj.size
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda seg: seg.size)
        target = candidates[0]
        # Force space to allocate from a specific segment by temporarily reordering
        space.pop_free_segment(target)
        space.push_free_segment_front(target)
        result = space.allocate(obj.size, obj.id)
        # Restore default address ordering for the free list.
        space.reorder_free_list()
        return result
