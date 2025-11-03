from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from datasets import load_dataset


@dataclass
class DialogueExample:
    dialogue: str
    summary: str
    id: str


def load_samsum(split: str = "train", limit: Optional[int] = None) -> Iterator[DialogueExample]:
    """
    Stream SAMSum dialogue summaries.

    Args:
        split: dataset split to load.
        limit: optional cap on number of examples.
    """
    dataset = load_dataset("samsum", split=split)
    count = 0
    for record in dataset:
        yield DialogueExample(dialogue=record["dialogue"], summary=record["summary"], id=str(record["id"]))
        count += 1
        if limit is not None and count >= limit:
            break
