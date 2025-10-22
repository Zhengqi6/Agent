from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[misc]

from .memory_object import MemoryObject

Vector = Dict[str, float]
Embedder = Callable[[MemoryObject], Vector]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _default_embedder(obj: MemoryObject) -> Vector:
    """Produce a sparse bag-of-words vector from the payload text."""
    text = ""
    if "text" in obj.payload:
        text = str(obj.payload["text"])
    else:
        text = " ".join(str(value) for value in obj.payload.values())
    tokens = _tokenize(text)
    counts = Counter(tokens)
    if not counts:
        return {}
    norm = math.sqrt(sum(value ** 2 for value in counts.values()))
    if norm == 0:
        return dict(counts)
    return {token: value / norm for token, value in counts.items()}


class SimilarityRetriever:
    """
    Lightweight retrieval layer that mirrors embedding-based search.

    Consumers can supply a custom `embedder` to integrate real vector stores.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        *,
        dense_model: Optional[str] = None,
    ) -> None:
        if embedder is not None and dense_model is not None:
            raise ValueError("Provide either a custom embedder or dense_model, not both.")

        self._objects: Dict[str, MemoryObject] = {}
        self._dense_model_name = dense_model
        self._dense_model = None
        self._dense_vectors: Dict[str, List[float]] = {}

        if dense_model and SentenceTransformer is not None:
            self._dense_model = SentenceTransformer(dense_model)
            self.embedder = self._dense_embed
        else:
            self.embedder = embedder or _default_embedder
        self._vectors: Dict[str, Vector] = {}

    def index(self, obj: MemoryObject) -> None:
        self._objects[obj.id] = obj
        if self._dense_model:
            embedding = self.embedder(obj)
            self._dense_vectors[obj.id] = embedding  # type: ignore[assignment]
        else:
            self._vectors[obj.id] = self.embedder(obj)

    def remove(self, object_id: str) -> None:
        self._objects.pop(object_id, None)
        self._vectors.pop(object_id, None)
        self._dense_vectors.pop(object_id, None)

    def rebuild(self, objects: Iterable[MemoryObject]) -> None:
        self._objects = {}
        self._vectors = {}
        self._dense_vectors = {}
        for obj in objects:
            self.index(obj)

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter_fn: Optional[Callable[[MemoryObject], bool]] = None,
    ) -> List[Tuple[MemoryObject, float]]:
        query_vec = self._embed_query(query)
        scored: List[Tuple[MemoryObject, float]] = []
        if self._dense_model:
            results = self._dense_query(query_vec, top_k, filter_fn)
            scored.extend(results)
        else:
            for object_id, vector in self._vectors.items():
                obj = self._objects[object_id]
                if filter_fn and not filter_fn(obj):
                    continue
                score = self._cosine_similarity(query_vec, vector)
                if score > 0.0:
                    scored.append((obj, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def _embed_query(self, query: str) -> Vector:
        if self._dense_model:
            embedding = self._dense_model.encode([query], normalize_embeddings=True)[0]  # type: ignore[operator]
            return {"__dense__": embedding}  # sentinel, actual vector stored separately
        tokens = _tokenize(query)
        counts = Counter(tokens)
        if not counts:
            return {}
        norm = math.sqrt(sum(value ** 2 for value in counts.values()))
        return {token: value / norm for token, value in counts.items()}

    @staticmethod
    def _cosine_similarity(lhs: Vector, rhs: Vector) -> float:
        if not lhs or not rhs:
            return 0.0
        if "__dense__" in lhs and "__dense__" in rhs:
            dense_lhs = lhs["__dense__"]  # type: ignore[assignment]
            dense_rhs = rhs["__dense__"]  # type: ignore[assignment]
            return float(sum(x * y for x, y in zip(dense_lhs, dense_rhs)))
        common = set(lhs.keys()) & set(rhs.keys())
        return sum(lhs[token] * rhs[token] for token in common)

    def _dense_embed(self, obj: MemoryObject) -> List[float]:
        text = ""
        if "text" in obj.payload:
            text = str(obj.payload["text"])
        else:
            text = " ".join(str(value) for value in obj.payload.values())
        assert self._dense_model is not None
        return self._dense_model.encode([text], normalize_embeddings=True)[0]  # type: ignore[operator]

    def _dense_query(
        self,
        query_vec: Vector,
        top_k: int,
        filter_fn: Optional[Callable[[MemoryObject], bool]],
    ) -> List[Tuple[MemoryObject, float]]:
        if "__dense__" not in query_vec:
            return []
        dense_query = query_vec["__dense__"]  # type: ignore[assignment]
        scored: List[Tuple[MemoryObject, float]] = []
        for object_id, vector in self._dense_vectors.items():
            obj = self._objects[object_id]
            if filter_fn and not filter_fn(obj):
                continue
            score = float(sum(x * y for x, y in zip(dense_query, vector)))
            scored.append((obj, score))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:top_k]
