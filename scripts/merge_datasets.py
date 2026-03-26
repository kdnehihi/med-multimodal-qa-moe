"""Merge processed task datasets into one shuffled training file."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import ensure_dir, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse merge arguments."""
    parser = argparse.ArgumentParser(description="Merge multiple processed datasets into one shuffled JSONL file.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/processed/healmqa_1500.jsonl",
            "data/processed/vqa_rad_1500.jsonl",
            "data/processed/medquad_2000.jsonl",
        ],
        help="Input JSONL files to merge.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/train_merged.jsonl",
        help="Merged output JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    return parser.parse_args()


def reduce_record(record: dict) -> dict:
    """Keep only the three core fields required by the user."""
    return {
        "image": record.get("image"),
        "question": record.get("question"),
        "answer": record.get("answer"),
    }


def main() -> None:
    """Merge and shuffle processed datasets."""
    args = parse_args()

    merged_records = []
    for input_path_str in args.inputs:
        input_path = PROJECT_ROOT / input_path_str
        records = load_jsonl(input_path)
        merged_records.extend(reduce_record(record) for record in records)

    random.Random(args.seed).shuffle(merged_records)

    output_path = PROJECT_ROOT / args.output
    ensure_dir(output_path.parent)
    write_jsonl(merged_records, output_path)

    print(f"Merged {len(args.inputs)} datasets into {output_path}")
    print(f"Total records: {len(merged_records)}")


if __name__ == "__main__":
    main()
