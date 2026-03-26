"""General helper functions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load records from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write records to a JSONL file."""
    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two Python lists."""
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length.")

    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
