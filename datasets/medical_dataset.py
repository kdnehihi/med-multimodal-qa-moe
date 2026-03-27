"""Dataset utilities for baseline multimodal medical QA training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from utils.helpers import load_jsonl


class MedicalDataset(Dataset):
    """Simple JSONL-backed dataset.

    Expected input schema:
    {
        "image": "path_or_null",
        "question": "...",
        "answer": "..."
    }
    """

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self.samples = load_jsonl(self.data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_value = sample.get("image")
        image_path = None if image_value in (None, "", "null") else str(image_value)
        return {
            "image": image_path,
            "question": str(sample.get("question", "")).strip(),
            "answer": str(sample.get("answer", "")).strip(),
        }
