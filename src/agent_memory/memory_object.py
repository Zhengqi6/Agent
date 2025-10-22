from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class MemoryObject:
    """
    Encapsulates a unit of memory stored in the agent heap.

    The size attribute should approximate the number of bytes required to store
    the payload. In most experiments we use len(str(payload)) to derive it, but
    callers can supply domain-specific measurements.
    """

    payload: Dict[str, Any]
    size: int
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    importance: float = 0.0
    ttl: Optional[float] = None
    address: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    reference_count: int = 0
    generation: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def touch(self) -> None:
        """Update the last_accessed timestamp to now."""
        self.last_accessed_at = time.time()

    def expired(self, now: Optional[float] = None) -> bool:
        """Check if the memory object has exceeded its TTL."""
        if self.ttl is None:
            return False
        now = now or time.time()
        return self.created_at + self.ttl < now

    def importance_score(self, decay_halflife: float = 3600.0) -> float:
        """
        Combine base importance with a time-decayed recency score.

        decay_halflife controls how fast recency decays. Lower values make
        recent memories more dominant.
        """
        age_seconds = time.time() - self.last_accessed_at
        decay = 0.5 ** (age_seconds / decay_halflife)
        return self.importance * 0.7 + decay * 0.3 + self.reference_count * 0.1
