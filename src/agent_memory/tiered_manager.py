from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .cold_storage import ColdStorage
from .memory_manager import MemoryManager
from .memory_object import MemoryObject
from .pager import PageTable
from .retrieval import SimilarityRetriever


@dataclass
class TieredMetrics:
    hot_hits: int = 0
    cold_hits: int = 0
    promotions: int = 0
    demotions: int = 0
    promotion_failures: int = 0
    demotion_failures: int = 0


class TieredMemoryManager:
    """
    Two-tier memory manager (hot MemorySpace + cold storage) with LRU promotion/demotion.
    """

    def __init__(
        self,
        hot_capacity: int,
        *,
        hot_manager: Optional[MemoryManager] = None,
        cold_store: Optional[ColdStorage] = None,
        cold_retriever: Optional[SimilarityRetriever] = None,
        hot_reserve_ratio: float = 0.25,
        max_hot_objects: Optional[int] = None,
        enable_paging: bool = False,
        page_size: int = 256,
        max_pages: int = 128,
        paging_strategy: str = "lru",
        write_policy: str = "write_back",
        promotion_batch_size: int = 1,
    ) -> None:
        self.hot_manager = hot_manager or MemoryManager(hot_capacity)
        self.cold_store = cold_store or ColdStorage()
        self.cold_retriever = cold_retriever or SimilarityRetriever()
        self.metrics = TieredMetrics()
        self.hot_reserve_ratio = hot_reserve_ratio
        self.max_hot_objects = max_hot_objects
        self._lru: "OrderedDict[str, float]" = OrderedDict()
        self.pager: Optional[PageTable] = PageTable(page_size, strategy=paging_strategy) if enable_paging else None
        self.max_pages = max_pages
        self.write_policy = write_policy
        self.promotion_batch_size = max(1, promotion_batch_size)

    def store(
        self,
        payload: Dict[str, object],
        *,
        size: Optional[int] = None,
        tags: Optional[List[str]] = None,
        importance: float = 0.0,
        ttl: Optional[float] = None,
        pin: bool = False,
        root: bool = False,
    ) -> MemoryObject:
        obj = self.hot_manager.store(
            payload,
            size=size,
            tags=tags,
            importance=importance,
            ttl=ttl,
            pin=pin,
            root=root,
        )
        if self.pager:
            self.pager.ensure_resident(obj, self.max_pages)
            if self.write_policy == "write_back":
                self.pager.mark_dirty(obj.id)
        if self.write_policy == "write_through":
            self._persist_to_cold(obj, mark_dirty=False)
        self._mark_hot_access(obj.id)
        self._ensure_hot_capacity()
        return obj

    def load(self, object_id: str) -> Optional[MemoryObject]:
        obj = self.hot_manager.load(object_id)
        if obj:
            self.metrics.hot_hits += 1
            self._mark_hot_access(object_id)
            if self.pager:
                self.pager.ensure_resident(obj, self.max_pages)
            return obj

        cold_obj = self._promote_from_cold(object_id)
        if cold_obj:
            self.metrics.cold_hits += 1
        return cold_obj

    def retrieve(self, query: str, *, top_k: int = 5) -> List[MemoryObject]:
        results: List[MemoryObject] = []
        hot_results = self.hot_manager.retrieve(query, top_k=top_k)
        for obj in hot_results:
            self.metrics.hot_hits += 1
            self._mark_hot_access(obj.id)
            results.append(obj)
        if len(results) >= top_k:
            return results[:top_k]

        remaining = top_k - len(results)
        cold_candidates = self.cold_retriever.query(query, top_k=remaining)
        for chunk_start in range(0, len(cold_candidates), self.promotion_batch_size):
            batch = [candidate[0].id for candidate in cold_candidates[chunk_start : chunk_start + self.promotion_batch_size]]
            promoted_batch = self._promote_from_cold_batch(batch)
            results.extend(promoted_batch)
            if len(results) >= top_k:
                break
        return results[:top_k]

    def stats(self) -> Dict[str, float]:
        hot_stats = self.hot_manager.stats()
        return {
            **hot_stats,
            "cold_items": float(len(self.cold_store)),
            "hot_lru_size": float(len(self._lru)),
            "hot_hits": float(self.metrics.hot_hits),
            "cold_hits": float(self.metrics.cold_hits),
            "promotions": float(self.metrics.promotions),
            "demotions": float(self.metrics.demotions),
            "promotion_failures": float(self.metrics.promotion_failures),
            "demotion_failures": float(self.metrics.demotion_failures),
            "page_faults": float(self.pager.stats.page_faults if self.pager else 0.0),
            "page_evictions": float(self.pager.stats.evictions if self.pager else 0.0),
            "page_writes": float(self.pager.stats.writes if self.pager else 0.0),
            "write_policy": self.write_policy,
        }

    def _mark_hot_access(self, object_id: str) -> None:
        now = time.time()
        if object_id in self._lru:
            self._lru.move_to_end(object_id)
        self._lru[object_id] = now

    def _ensure_hot_capacity(self) -> None:
        reserve = int(self.hot_manager.space.capacity * self.hot_reserve_ratio)
        while (
            self.hot_manager.space.available() < reserve
            or (self.max_hot_objects and len(self.hot_manager.objects) > self.max_hot_objects)
        ):
            if not self._lru:
                break
            object_id, _ = self._lru.popitem(last=False)
            if object_id not in self.hot_manager.objects:
                continue
            obj = self.hot_manager.objects[object_id]
            if obj.reference_count > 0:
                # skip pinned or referenced objects
                self._lru[object_id] = obj.last_accessed_at
                self._lru.move_to_end(object_id)
                continue
            self._demote_to_cold(object_id, obj)

    def _demote_to_cold(self, object_id: str, obj: MemoryObject) -> None:
        try:
            self.hot_manager.free_object(object_id)
            self._persist_to_cold(obj, mark_dirty=False)
            self.metrics.demotions += 1
        except Exception:
            self.metrics.demotion_failures += 1

    def _promote_from_cold(self, object_id: str) -> Optional[MemoryObject]:
        if not self.cold_store.contains(object_id):
            return None
        obj = self.cold_store.load(object_id)
        if obj is None:
            self.metrics.promotion_failures += 1
            return None
        try:
            obj.touch()
            self.hot_manager.store_existing(obj)
            if self.pager:
                self.pager.ensure_resident(obj, self.max_pages)
            self._mark_hot_access(obj.id)
            self.cold_store.delete(obj.id)
            self.cold_retriever.remove(obj.id)
            self.metrics.promotions += 1
            self._ensure_hot_capacity()
            return obj
        except MemoryError:
            self.metrics.promotion_failures += 1
            return None
        except Exception:
            self.metrics.promotion_failures += 1
            return None

    def all_hot_objects(self) -> Iterable[MemoryObject]:
        return list(self.hot_manager.objects.values())

    def all_cold_ids(self) -> List[str]:
        return list(self.cold_store._store.keys())

    def _promote_from_cold_batch(self, object_ids: List[str]) -> List[MemoryObject]:
        promoted: List[MemoryObject] = []
        for object_id in object_ids:
            obj = self._promote_from_cold(object_id)
            if obj:
                promoted.append(obj)
        return promoted

    def _persist_to_cold(self, obj: MemoryObject, mark_dirty: bool = True) -> None:
        self.cold_store.save(obj)
        self.cold_retriever.index(obj)
        if self.pager:
            if mark_dirty:
                self.pager.mark_dirty(obj.id)
            else:
                self.pager.clear_dirty(obj.id)
