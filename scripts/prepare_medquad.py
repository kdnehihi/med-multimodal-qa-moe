"""Prepare a text-only medical QA subset into the project JSONL format."""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import ensure_dir, load_config, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare MedQuAD text-only QA data.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to project config.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Hugging Face dataset name override.")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path.")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to export.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Print dataset column names and one sample for debugging.",
    )
    return parser.parse_args()


def extract_qa_fields(sample: dict[str, Any]) -> tuple[str, str]:
    """Extract question and answer fields with a few common fallbacks."""
    raw_text = str(sample.get("text", "")).strip()
    if raw_text:
        pattern = re.compile(
            r"### Instruction:\s*(.*?)\s*### Response:\s*(.*)",
            flags=re.DOTALL,
        )
        match = pattern.search(raw_text)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            if question and answer:
                return question, answer

    question = str(
        sample.get("question")
        or sample.get("Question")
        or sample.get("query")
        or sample.get("q")
        or sample.get("focus")
        or sample.get("question_text")
        or ""
    ).strip()
    answer = str(
        sample.get("answer")
        or sample.get("Answer")
        or sample.get("response")
        or sample.get("best_answer")
        or sample.get("a")
        or sample.get("answer_text")
        or sample.get("context")
        or ""
    ).strip()
    return question, answer


def normalize_sample(sample: dict[str, Any], index: int) -> dict[str, Any] | None:
    """Normalize a text-only QA sample into project format."""
    question, answer = extract_qa_fields(sample)
    if not question or not answer:
        return None

    return {
        "id": f"medquad_{index:05d}",
        "image": None,
        "question": question,
        "answer": answer,
        "task_type": "qa",
        "source_dataset": "medquad",
        "source_split": "hf",
    }


def main() -> None:
    """Export a MedQuAD subset."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    medquad_cfg = config["data"]["medquad"]

    dataset_name = args.dataset_name or medquad_cfg["dataset_name"]
    output_path = PROJECT_ROOT / (args.output or medquad_cfg["output_path"])
    sample_size = args.sample_size or medquad_cfg["sample_size"]

    dataset = load_dataset(dataset_name, split="train")
    if args.show_schema:
        print("Column names:", dataset.column_names)
        print("First sample:", dataset[0])
        return

    total_available = len(dataset)
    target_size = min(sample_size, total_available)

    indices = list(range(total_available))
    random.Random(args.seed).shuffle(indices)

    normalized_records: list[dict[str, Any]] = []
    skipped = 0
    for dataset_index in indices:
        normalized = normalize_sample(dataset[int(dataset_index)], index=len(normalized_records) + skipped)
        if normalized is None:
            skipped += 1
            continue
        normalized_records.append(normalized)
        if len(normalized_records) >= target_size:
            break

    ensure_dir(output_path.parent)
    write_jsonl(normalized_records, output_path)

    print(f"Loaded {total_available} samples from {dataset_name}")
    print(f"Exported {len(normalized_records)} text-only QA samples to {output_path}")
    print(f"Skipped {skipped} malformed records")


if __name__ == "__main__":
    main()
