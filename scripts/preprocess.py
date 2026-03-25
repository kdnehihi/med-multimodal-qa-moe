"""Preprocessing script scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import load_jsonl


def main() -> None:
    """Entry point for dataset preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess raw data into model-ready format.")
    parser.add_argument("--input", type=str, required=True, help="Raw JSONL input path.")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    samples = load_jsonl(input_path)

    # TODO: validate sample fields.
    # Expected checks:
    # - image can be None or a valid path
    # - question is a non-empty string
    # - answer is a non-empty string
    # - task_type is one of: symptom, text_image, qa

    # TODO: normalize text, build train/val splits, and save processed JSONL files.
    print(f"Loaded {len(samples)} samples from {input_path}")
    print("Preprocessing scaffold ready.")


if __name__ == "__main__":
    main()
