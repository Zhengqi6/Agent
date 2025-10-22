from __future__ import annotations

import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .memory_object import MemoryObject


@dataclass
class Page:
    page_id: Tuple[str, int]
    data: bytes
    dirty: bool = False


@dataclass
class PagerStats:
    page_faults: int = 0
    evictions: int = 0
    loads: int = 0
    writes: int = 0


class PageTable:
    def __init__(self, page_size: int, *, strategy: str = "lru") -> None:
        self.page_size = page_size
        self._table: Dict[Tuple[str, int], Page] = {}
        self.stats = PagerStats()
        self._lru: "OrderedDict[Tuple[str, int], None]" = OrderedDict()
        self.strategy = strategy
        self._clock_hand = 0
        self._clock_list: List[Tuple[str, int]] = []
        self._clock_reference: Dict[Tuple[str, int], bool] = {}
        self._lfu_counts: Dict[Tuple[str, int], int] = {}

    def _object_to_pages(self, obj: MemoryObject) -> List[Page]:
        payload = obj.payload
        raw = pickle.dumps(payload)
        pages: List[Page] = []
        for idx in range(0, len(raw), self.page_size):
            chunk = raw[idx : idx + self.page_size]
            pages.append(Page((obj.id, idx // self.page_size), chunk))
        return pages

    def ensure_resident(self, obj: MemoryObject, max_pages: int) -> None:
        pages = self._object_to_pages(obj)
        for page in pages:
            if page.page_id not in self._table:
                self._maybe_evict(max_pages)
                self._table[page.page_id] = page
                if self.strategy == "clock":
                    self._clock_list.append(page.page_id)
                    self._clock_reference[page.page_id] = True
                elif self.strategy == "lfu":
                    self._lfu_counts[page.page_id] = 1
                self.stats.page_faults += 1
            self._touch(page.page_id)

    def read(self, object_id: str) -> Optional[Dict[str, object]]:
        relevant = [page for key, page in self._table.items() if key[0] == object_id]
        if not relevant:
            return None
        relevant.sort(key=lambda page: page.page_id[1])
        combined = b"".join(page.data for page in relevant)
        self.stats.loads += 1
        self._touch_bulk(page.page_id for page in relevant)
        return pickle.loads(combined)

    def _maybe_evict(self, max_pages: int) -> None:
        while len(self._table) >= max_pages:
            if self.strategy == "clock":
                if not self._clock_list:
                    break
                victim = self._select_clock_victim()
                if victim is None:
                    break
                page = self._table.pop(victim, None)
                if page is None:
                    continue
                self._clock_reference.pop(victim, None)
                try:
                    self._clock_list.remove(victim)
                except ValueError:
                    pass
                if page.dirty:
                    self.stats.writes += 1
                self.stats.evictions += 1
            elif self.strategy == "lfu":
                if not self._lfu_counts:
                    break
                victim = min(self._lfu_counts.items(), key=lambda item: (item[1], item[0][1]))[0]
                self._lfu_counts.pop(victim, None)
                page = self._table.pop(victim, None)
                if page is None:
                    continue
                if page.dirty:
                    self.stats.writes += 1
                self.stats.evictions += 1
            else:
                if not self._lru:
                    break
                page_id, _ = self._lru.popitem(last=False)
                page = self._table.pop(page_id)
                if page.dirty:
                    self.stats.writes += 1
                self.stats.evictions += 1

    def _touch(self, page_id: Tuple[str, int]) -> None:
        if self.strategy == "clock":
            if page_id in self._clock_reference:
                self._clock_reference[page_id] = True
            return
        if self.strategy == "lfu":
            if page_id in self._lfu_counts:
                self._lfu_counts[page_id] += 1
            return
        if page_id in self._lru:
            self._lru.move_to_end(page_id)
        self._lru[page_id] = None

    def _touch_bulk(self, page_ids: Iterable[Tuple[str, int]]) -> None:
        for page_id in page_ids:
            self._touch(page_id)

    def mark_dirty(self, object_id: str) -> None:
        for key, page in self._table.items():
            if key[0] == object_id:
                page.dirty = True

    def clear_dirty(self, object_id: str) -> None:
        for key, page in self._table.items():
            if key[0] == object_id:
                page.dirty = False

    def _select_clock_victim(self) -> Optional[Tuple[str, int]]:
        if not self._clock_list:
            return None
        n = len(self._clock_list)
        for _ in range(n * 2):
            page_id = self._clock_list[self._clock_hand % n]
            self._clock_hand = (self._clock_hand + 1) % max(n, 1)
            if not self._clock_reference.get(page_id, False):
                return page_id
            self._clock_reference[page_id] = False
        return self._clock_list.pop(0) if self._clock_list else None
