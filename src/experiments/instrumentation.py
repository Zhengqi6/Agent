from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MemoryProfiler:
    """
    Lightweight event logger for MemoryManager.

    Records structured events and optionally flushes them to disk as CSV or JSONL.
    """

    run_id: str
    output_dir: Optional[str] = None
    write_immediately: bool = False
    events: List[Dict[str, object]] = field(default_factory=list)

    def record_event(self, event_type: str, payload: Dict[str, object]) -> None:
        record = {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "event": event_type,
            **payload,
        }
        self.events.append(record)
        if self.write_immediately and self.output_dir:
            self._append_jsonl(record)

    def flush(self) -> None:
        if not self.output_dir or not self.events:
            return
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_path / f"{self.run_id}.jsonl"
        csv_path = output_path / f"{self.run_id}.csv"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in self.events:
                handle.write(json.dumps(record) + "\n")
        fieldnames = sorted({key for event in self.events for key in event.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)

    def _append_jsonl(self, record: Dict[str, object]) -> None:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_path / f"{self.run_id}.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
