from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from .allocators import Allocator, FirstFitAllocator
from .garbage_collectors import GarbageCollector, MarkAndSweepCollector, TimeToLiveCollector
from .memory_object import MemoryObject
from .persistence import CheckpointManager, WriteAheadLog
from .memory_space import MemorySegment, MemorySpace
from .gc_policy import GCPolicy
from .locks import ReadWriteLock
from .retrieval import SimilarityRetriever

if TYPE_CHECKING:
    from experiments.instrumentation import MemoryProfiler
class MemoryManager:
    """
    High-level facade that exposes allocation, access, and reclamation APIs.

    The manager stitches together the allocator, GC policies, and retrieval
    logic, making it simple to plug into agent loops.
    """

    def __init__(
        self,
        capacity: int,
        *,
        allocator: Optional[Allocator] = None,
        collectors: Optional[List[GarbageCollector]] = None,
        retriever: Optional[SimilarityRetriever] = None,
        profiler: Optional["MemoryProfiler"] = None,
        policies: Optional[List[GCPolicy]] = None,
        thread_safe: bool = False,
        wal: Optional[WriteAheadLog] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.space = MemorySpace(capacity)
        self.allocator = allocator or FirstFitAllocator()
        self.collectors = collectors or [TimeToLiveCollector(), MarkAndSweepCollector()]
        self.retriever = retriever or SimilarityRetriever()
        self.profiler = profiler
        self.policies = policies or []
        self._thread_safe = thread_safe
        self._lock = ReadWriteLock() if thread_safe else None
        self._wal = wal
        self._checkpoint_manager = checkpoint_manager

        self.objects: Dict[str, MemoryObject] = {}
        self._roots: Set[str] = set()
        self._pinned: Set[str] = set()
        self._edges: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_edges: Dict[str, Set[str]] = defaultdict(set)
        self.gc_events: List[Dict[str, Any]] = []
        self._allocation_events = 0

    # -- Allocation -----------------------------------------------------------------
    def store(
        self,
        payload: Dict[str, Any],
        *,
        size: Optional[int] = None,
        tags: Optional[List[str]] = None,
        importance: float = 0.0,
        ttl: Optional[float] = None,
        embedding: Optional[List[float]] = None,
        pin: bool = False,
        root: bool = False,
    ) -> MemoryObject:
        """Allocate space for the payload and register it with all subsystems."""
        computed_size = size if size is not None else len(str(payload))
        obj = MemoryObject(
            payload=payload,
            size=computed_size,
            tags=tags or [],
            importance=importance,
            ttl=ttl,
            embedding=embedding,
        )
        def write_op() -> MemoryObject:
            segment = self._allocate(obj)
            obj.address = segment.start if segment else None
            self.objects[obj.id] = obj
            if pin:
                self.pin(obj.id)
            if root:
                self.add_root(obj.id)
            self.retriever.index(obj)
            self._allocation_events += 1
            self._maybe_trigger_policies("store")
            if self._wal:
                self._wal.append({
                    "op": "store",
                    "object_id": obj.id,
                    "payload": obj.payload,
                    "size": obj.size,
                })
            return obj

        if self._lock:
            with self._lock.write_lock():
                return write_op()
        return write_op()

    def store_existing(
        self,
        obj: MemoryObject,
        *,
        pin: bool = False,
        root: bool = False,
    ) -> MemoryObject:
        """
        Reintroduce an existing MemoryObject (e.g., promoted from cold storage).
        """
        if obj.id in self.objects:
            raise ValueError(f"Object {obj.id} already present in memory manager.")
        segment = self._allocate(obj)
        obj.address = segment.start if segment else None
        self.objects[obj.id] = obj
        if pin:
            self.pin(obj.id)
        if root:
            self.add_root(obj.id)
        self.retriever.index(obj)
        return obj

    def _allocate(self, obj: MemoryObject) -> MemorySegment:
        segment = self.allocator.allocate(self.space, obj)
        if segment:
            if self.profiler:
                self.profiler.record_event(
                    "allocation",
                    {
                        "object_id": obj.id,
                        "size": obj.size,
                        "strategy": type(self.allocator).__name__,
                        "heap_used": self.space.allocated(),
                        "heap_free": self.space.available(),
                    },
                )
            return segment
        self._run_gc_cycle(trigger="allocation_failure")
        segment = self.allocator.allocate(self.space, obj)
        if not segment:
            raise MemoryError(f"Unable to allocate {obj.size} bytes for object {obj.id}")
        if self.profiler:
            self.profiler.record_event(
                "allocation_post_gc",
                {
                    "object_id": obj.id,
                    "size": obj.size,
                    "strategy": type(self.allocator).__name__,
                    "heap_used": self.space.allocated(),
                    "heap_free": self.space.available(),
                },
            )
        return segment

    # -- Access --------------------------------------------------------------------
    def load(self, object_id: str) -> Optional[MemoryObject]:
        def read_op() -> Optional[MemoryObject]:
            obj = self.objects.get(object_id)
            if obj:
                obj.touch()
                if self.profiler:
                    self.profiler.record_event("load", {"object_id": object_id})
                self._maybe_trigger_policies("load")
            return obj

        if self._lock:
            with self._lock.read_lock():
                return read_op()
        return read_op()

    def retrieve(self, query: str, *, top_k: int = 5) -> List[MemoryObject]:
        def read_op() -> List[MemoryObject]:
            return self.retriever.query(query, top_k=top_k)

        if self._lock:
            with self._lock.write_lock():
                results = read_op()
        else:
            results = read_op()
        ordered_objects = []
        for obj, score in results:
            obj.touch()
            obj.reference_count += 1
            if self.profiler:
                self.profiler.record_event(
                    "retrieve_hit",
                    {
                        "object_id": obj.id,
                        "score": score,
                        "query": query,
                        "ref_count": obj.reference_count,
                    },
                )
            ordered_objects.append(obj)
        if self.profiler:
            self.profiler.record_event(
                "retrieve_complete",
                {
                    "query": query,
                    "hits": len(ordered_objects),
                    "top_k": top_k,
                },
            )
        self._maybe_trigger_policies("retrieve")
        return ordered_objects

    # -- Reference Management ------------------------------------------------------
    def add_root(self, object_id: str) -> None:
        if object_id in self.objects:
            self._roots.add(object_id)

    def remove_root(self, object_id: str) -> None:
        self._roots.discard(object_id)

    def add_reference(self, source_id: str, target_id: str) -> None:
        if source_id not in self.objects or target_id not in self.objects:
            return
        if target_id not in self._edges[source_id]:
            self._edges[source_id].add(target_id)
            self._reverse_edges[target_id].add(source_id)
            self.objects[target_id].reference_count += 1

    def remove_reference(self, source_id: str, target_id: str) -> None:
        if target_id in self._edges.get(source_id, set()):
            self._edges[source_id].remove(target_id)
            self._reverse_edges[target_id].remove(source_id)
            self.objects[target_id].reference_count = max(
                0, self.objects[target_id].reference_count - 1
            )

    def reachable_ids(self) -> Set[str]:
        """Traverse the reference graph starting from roots and pinned objects."""
        visited: Set[str] = set()
        frontier: List[str] = list(self._roots | self._pinned)
        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)
            frontier.extend(self._edges.get(current, set()))
        return visited

    # -- Object lifetime -----------------------------------------------------------
    def pin(self, object_id: str) -> None:
        if object_id in self.objects:
            self._pinned.add(object_id)

    def unpin(self, object_id: str) -> None:
        self._pinned.discard(object_id)

    def is_pinned(self, object_id: str) -> bool:
        return object_id in self._pinned

    def free_object(self, object_id: str) -> None:
        obj = self.objects.pop(object_id, None)
        if not obj:
            return
        self.space.deallocate(object_id)
        self._cleanup_graph(object_id)
        self._roots.discard(object_id)
        self._pinned.discard(object_id)
        self.retriever.remove(object_id)
        if self.profiler:
            self.profiler.record_event(
                "free",
                {
                    "object_id": object_id,
                    "heap_used": self.space.allocated(),
                    "heap_free": self.space.available(),
                },
            )
        if self._wal:
            self._wal.append({"op": "free", "object_id": object_id})

    def _cleanup_graph(self, object_id: str) -> None:
        for source in list(self._reverse_edges.get(object_id, set())):
            self._edges[source].discard(object_id)
        self._reverse_edges.pop(object_id, None)
        self._edges.pop(object_id, None)

    # -- Garbage collection --------------------------------------------------------
    def run_gc(self, trigger: str = "manual") -> Dict[str, Any]:
        return self._run_gc_cycle(trigger=trigger)

    def _run_gc_cycle(self, trigger: str) -> Dict[str, Any]:
        freed_total: List[str] = []
        cycle_details: List[Dict[str, Any]] = []
        cycle_start = time.time()
        for collector in self.collectors:
            if not collector.should_trigger(self):
                continue
            freed = collector.collect(self)
            if freed:
                freed_total.extend(freed)
            cycle_details.append({"collector": collector.name, "freed": len(freed)})
        stats = self.stats()
        pause_duration = time.time() - cycle_start
        event = {
            "trigger": trigger,
            "freed_ids": freed_total,
            "heap_used": stats["heap_used"],
            "fragmentation": stats["fragmentation"],
            "collectors": cycle_details,
            "timestamp": time.time(),
            "pause_duration": pause_duration,
        }
        self.gc_events.append(event)
        if self.profiler:
            self.profiler.record_event(
                "gc_cycle",
                {
                    "trigger": trigger,
                    "freed_ids": len(freed_total),
                    "heap_used": stats["heap_used"],
                    "fragmentation": stats["fragmentation"],
                    "pause_duration": pause_duration,
                },
            )
        for policy in self.policies:
            policy.notify_gc(self, event)
        return event

    # -- Introspection -------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        stats = {
            "objects": len(self.objects),
            "pinned": len(self._pinned),
            "roots": len(self._roots),
            "heap_capacity": self.space.capacity,
            "heap_used": self.space.allocated(),
            "heap_free": self.space.available(),
            "fragmentation": self.space.fragmentation(),
        }
        if self._lock:
            stats["thread_safe"] = 1.0
        if self._wal:
            stats["wal_enabled"] = 1.0
        return stats

    def debug_snapshot(self) -> Dict[str, Any]:
        object_summaries = [
            {
                "id": obj.id,
                "size": obj.size,
                "tags": obj.tags,
                "importance": obj.importance,
                "ttl": obj.ttl,
                "generation": obj.generation,
                "ref_count": obj.reference_count,
                "address": obj.address,
            }
            for obj in self.objects.values()
        ]
        return {
            "objects": object_summaries,
            "space": self.space.snapshot(),
            "roots": list(self._roots),
            "pinned": list(self._pinned),
        }

    def _maybe_trigger_policies(self, reason: str) -> None:
        for policy in self.policies:
            try:
                if policy.should_trigger(self, reason):
                    self.run_gc(trigger=f"policy:{policy.__class__.__name__}")
            except Exception:
                continue

    def tick(self) -> None:
        self._maybe_trigger_policies("tick")

    def checkpoint(self) -> None:
        if not self._checkpoint_manager:
            return
        self._checkpoint_manager.write(self.objects.values())
