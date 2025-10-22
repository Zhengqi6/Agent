"""
Agent memory framework inspired by classical memory management.

Expose high-level classes for building experiments quickly.
"""

from .memory_object import MemoryObject
from .memory_manager import MemoryManager
from .allocators import FirstFitAllocator, BestFitAllocator
from .garbage_collectors import (
    MarkAndSweepCollector,
    TimeToLiveCollector,
    GenerationalCollector,
)
from .retrieval import SimilarityRetriever

__all__ = [
    "MemoryObject",
    "MemoryManager",
    "FirstFitAllocator",
    "BestFitAllocator",
    "MarkAndSweepCollector",
    "TimeToLiveCollector",
    "GenerationalCollector",
    "SimilarityRetriever",
]
