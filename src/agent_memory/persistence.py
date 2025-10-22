from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from .memory_object import MemoryObject


class WriteAheadLog:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def append(self, entry: Dict[str, object]) -> None:
        json.dump(entry, self._handle)
        self._handle.write("\n")
        self._handle.flush()

    def replay(self) -> Iterator[Dict[str, object]]:
        if not self.path.exists():
            return iter(())  # pragma: no cover
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


class CheckpointManager:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, objects: Iterable[MemoryObject]) -> None:
        snapshot = {obj.id: obj for obj in objects}
        with self.path.open("wb") as handle:
            pickle.dump(snapshot, handle)

    def load(self) -> Optional[Dict[str, MemoryObject]]:
        if not self.path.exists():
            return None
        with self.path.open("rb") as handle:
            return pickle.load(handle)
