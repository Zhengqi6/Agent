from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from agent_memory import MemoryObject, SimilarityRetriever


class SlidingWindowMemory:
    """
    Simplified baseline memory inspired by replay buffers used in long-context LLM papers.

    Keeps only the most recent `max_items` memories without allocator/GC dynamics.
    """

    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max_items
        self._objects: Deque[MemoryObject] = deque()
        self._retriever = SimilarityRetriever()

    def store(
        self,
        payload: Dict[str, object],
        *,
        size: Optional[int] = None,
        tags: Optional[List[str]] = None,
        importance: float = 0.0,
    ) -> MemoryObject:
        computed_size = size if size is not None else len(str(payload))
        obj = MemoryObject(payload=payload, size=computed_size, tags=tags or [], importance=importance)
        self._objects.append(obj)
        self._retriever.index(obj)
        if len(self._objects) > self.max_items:
            oldest = self._objects.popleft()
            self._retriever.remove(oldest.id)
        return obj

    def retrieve(self, query: str, *, top_k: int = 5) -> List[MemoryObject]:
        results = self._retriever.query(query, top_k=top_k)
        ordered_objects: List[MemoryObject] = []
        for obj, _ in results:
            obj.touch()
            ordered_objects.append(obj)
        return ordered_objects

    def stats(self) -> Dict[str, float]:
        return {
            "objects": float(len(self._objects)),
            "max_items": float(self.max_items),
        }


class ReservoirMemory:
    """
    Reservoir-sampling baseline that retains a uniform sample of past memories.

    Useful for evaluating coverage vs. recency trade-offs.
    """

    def __init__(self, max_items: int = 256) -> None:
        self.max_items = max_items
        self._objects: List[MemoryObject] = []
        self._retriever = SimilarityRetriever()
        self._seen = 0

    def store(
        self,
        payload: Dict[str, object],
        *,
        size: Optional[int] = None,
        tags: Optional[List[str]] = None,
        importance: float = 0.0,
    ) -> MemoryObject:
        computed_size = size if size is not None else len(str(payload))
        obj = MemoryObject(payload=payload, size=computed_size, tags=tags or [], importance=importance)
        self._seen += 1
        if len(self._objects) < self.max_items:
            self._objects.append(obj)
            self._retriever.index(obj)
        else:
            idx = self._reservoir_index(self._seen)
            if idx is not None:
                replaced = self._objects[idx]
                self._retriever.remove(replaced.id)
                self._objects[idx] = obj
                self._retriever.index(obj)
        return obj

    def retrieve(self, query: str, *, top_k: int = 5) -> List[MemoryObject]:
        results = self._retriever.query(query, top_k=top_k)
        ordered_objects: List[MemoryObject] = []
        for obj, _ in results:
            obj.touch()
            ordered_objects.append(obj)
        return ordered_objects

    def stats(self) -> Dict[str, float]:
        return {
            "objects": float(len(self._objects)),
            "max_items": float(self.max_items),
            "seen": float(self._seen),
        }

    def _reservoir_index(self, seen: int) -> Optional[int]:
        # classic reservoir sampling probability
        from random import randrange

        j = randrange(seen)
        if j < self.max_items:
            return j
        return None
