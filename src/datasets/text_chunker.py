from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class TextChunk:
    text: str
    start: int
    end: int
    document_id: str


def load_text_chunks(
    path: str,
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    limit: Optional[int] = None,
    document_id: Optional[str] = None,
) -> Iterator[TextChunk]:
    """
    Slide a window over a plaintext file to produce overlapping chunks.

    Args:
        path: path to the text file.
        chunk_size: number of characters per chunk.
        overlap: characters to overlap between consecutive chunks.
        limit: optional maximum number of chunks.
        document_id: identifier stored in each chunk (defaults to filename stem).
    """
    text_path = Path(path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found at {path}")

    content = text_path.read_text(encoding="utf-8")
    content = re.sub(r"\s+", " ", content).strip()
    doc_id = document_id or text_path.stem

    step = max(1, chunk_size - overlap)
    count = 0
    for start in range(0, max(len(content) - chunk_size + 1, 1), step):
        end = min(start + chunk_size, len(content))
        snippet = content[start:end]
        if snippet:
            yield TextChunk(text=snippet, start=start, end=end, document_id=doc_id)
            count += 1
        if limit is not None and count >= limit:
            break
