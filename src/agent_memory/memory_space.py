from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(slots=True)
class MemorySegment:
    start: int
    size: int
    object_id: Optional[str] = None

    @property
    def end(self) -> int:
        return self.start + self.size

    def split(self, size: int) -> Tuple["MemorySegment", Optional["MemorySegment"]]:
        """Return allocated segment plus remainder, if any."""
        allocated = MemorySegment(start=self.start, size=size, object_id=self.object_id)
        remainder_size = self.size - size
        if remainder_size <= 0:
            return allocated, None
        remainder = MemorySegment(start=self.start + size, size=remainder_size)
        return allocated, remainder


class MemorySpace:
    """
    Simulated contiguous heap with manual allocation and coalescing.

    The space tracks both allocated segments (via object_id) and the free-list.
    Segments are kept sorted by address to simplify fragmentation analysis.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._free_segments: List[MemorySegment] = [MemorySegment(0, capacity)]
        self._allocated: Dict[str, MemorySegment] = {}

    def available(self) -> int:
        return sum(segment.size for segment in self._free_segments)

    def allocated(self) -> int:
        return sum(segment.size for segment in self._allocated.values())

    def fragmentation(self) -> float:
        if not self._free_segments or self.available() == 0:
            return 0.0
        largest = max(segment.size for segment in self._free_segments)
        return 1.0 - (largest / self.available())

    def allocate(self, size: int, object_id: str) -> Optional[MemorySegment]:
        """
        Allocate a segment using first-fit strategy by default.
        Returns the allocated segment or None if insufficient space exists.
        """
        for index, segment in enumerate(self._free_segments):
            if segment.size >= size:
                allocated, remainder = segment.split(size)
                allocated.object_id = object_id
                self._allocated[object_id] = allocated
                if remainder:
                    self._free_segments[index] = remainder
                else:
                    del self._free_segments[index]
                return allocated
        return None

    def free_segments(self) -> List[MemorySegment]:
        """Return a copy of the current free list for inspection."""
        return list(self._free_segments)

    def pop_free_segment(self, segment: MemorySegment) -> None:
        """Remove a segment from the free list if present."""
        try:
            self._free_segments.remove(segment)
        except ValueError:
            pass

    def push_free_segment_front(self, segment: MemorySegment) -> None:
        """Insert a segment at the front of the free list."""
        self._free_segments.insert(0, segment)

    def reorder_free_list(self) -> None:
        """Sort the free list by address to maintain deterministic traversal."""
        self._free_segments.sort(key=lambda seg: seg.start)

    def deallocate(self, object_id: str) -> None:
        segment = self._allocated.pop(object_id, None)
        if not segment:
            return
        segment.object_id = None
        self._insert_free_segment(segment)

    def resize(self, object_id: str, new_size: int) -> bool:
        """
        Attempt to resize an allocated segment in place.
        Shrinking always succeeds. Growing succeeds if adjacent space is free.
        """
        segment = self._allocated.get(object_id)
        if not segment:
            return False
        if new_size == segment.size:
            return True
        if new_size < segment.size:
            # Return excess to free list
            remainder = MemorySegment(segment.start + new_size, segment.size - new_size)
            segment.size = new_size
            self._insert_free_segment(remainder)
            return True

        growth = new_size - segment.size
        neighbor = self._find_right_neighbor(segment)
        if neighbor and neighbor.size >= growth:
            # consume portion of neighbor
            self._free_segments.remove(neighbor)
            if neighbor.size > growth:
                remainder = MemorySegment(neighbor.start + growth, neighbor.size - growth)
                self._insert_free_segment(remainder)
            segment.size = new_size
            return True
        return False

    def snapshot(self) -> Dict[str, List[Tuple[int, int]]]:
        """Expose current allocation map for diagnostics."""
        return {
            "allocated": [(seg.start, seg.size) for seg in self._allocated.values()],
            "free": [(seg.start, seg.size) for seg in self._free_segments],
        }

    def _insert_free_segment(self, segment: MemorySegment) -> None:
        self._free_segments.append(segment)
        self._free_segments.sort(key=lambda s: s.start)
        self._coalesce()

    def _coalesce(self) -> None:
        if not self._free_segments:
            return
        coalesced: List[MemorySegment] = [self._free_segments[0]]
        for segment in self._free_segments[1:]:
            prev = coalesced[-1]
            if prev.end == segment.start:
                prev.size += segment.size
            else:
                coalesced.append(segment)
        self._free_segments = coalesced

    def _find_right_neighbor(self, segment: MemorySegment) -> Optional[MemorySegment]:
        for candidate in self._free_segments:
            if candidate.start == segment.end:
                return candidate
        return None
