from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class UfoSighting:
    datetime: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    shape: str = ""
    duration: str = ""
    comments: str = ""
    latitude: str = ""
    longitude: str = ""
    colors: str = ""


def load_ufo_dataset(path: str, limit: Optional[int] = None) -> Iterator[UfoSighting]:
    """
    Stream the UFO sightings dataset row by row.

    Args:
        path: path to the CSV file (downloaded beforehand).
        limit: optional cap on records for quick experiments.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            sighting = UfoSighting(
                datetime=row.get("Date_time") or row.get("Time") or "",
                city=row.get("city") or row.get("City") or "",
                state=row.get("state") or row.get("State") or "",
                country=row.get("country") or row.get("Country") or "us",
                shape=row.get("shape") or row.get("Shape Reported") or "",
                duration=row.get("duration") or row.get("Duration") or "",
                comments=row.get("comments") or row.get("Comments") or "",
                latitude=row.get("latitude") or row.get("Latitude") or "",
                longitude=row.get("longitude") or row.get("Longitude") or "",
                colors=row.get("Colors Reported") or row.get("colors") or "",
            )
            yield sighting
            if limit is not None and index >= limit:
                break
