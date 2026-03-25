"""Dataset skeleton for multimodal medical QA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from utils.helpers import load_jsonl


class MedicalDataset(Dataset):
    """Minimal dataset scaffold for symptom, text_image, and qa tasks.

    Expected JSONL format:
    {
        "image": "path_or_none",
        "question": "...",
        "answer": "...",
        "task_type": "symptom" | "text_image" | "qa"
    }
    """

    def __init__(self, data_path: str | Path) -> None:
        self.data_path = Path(data_path)
        self.samples = load_jsonl(self.data_path)

        # TODO: initialize tokenizer and image transforms here.
        # Expected components:
        # - question tokenizer
        # - answer tokenizer
        # - image preprocessing pipeline

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        # TODO: load image if the sample contains an image path.
        # Expected output:
        # - image tensor for multimodal tasks
        # - None or placeholder for text-only qa
        image = sample.get("image")

        # TODO: tokenize / preprocess question and answer.
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "task_type": sample.get("task_type", "qa"),
        }
