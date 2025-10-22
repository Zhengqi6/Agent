from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, Optional

from .memory_object import MemoryObject


@dataclass
class ColdStorageStats:
    writes: int = 0
    reads: int = 0
    deletes: int = 0


class ColdStorage:
    """
    Simple in-memory cold storage that simulates disk persistence using pickling.
    """

    def __init__(self) -> None:
        self._store: Dict[str, bytes] = {}
        self.stats = ColdStorageStats()

    def save(self, obj: MemoryObject) -> None:
        self._store[obj.id] = pickle.dumps(obj)
        self.stats.writes += 1

    def load(self, object_id: str) -> Optional[MemoryObject]:
        blob = self._store.get(object_id)
        if blob is None:
            return None
        self.stats.reads += 1
        return pickle.loads(blob)

    def delete(self, object_id: str) -> None:
        if object_id in self._store:
            del self._store[object_id]
            self.stats.deletes += 1

    def contains(self, object_id: str) -> bool:
        return object_id in self._store

    def __len__(self) -> int:
        return len(self._store)
