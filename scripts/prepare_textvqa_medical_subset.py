"""Prepare a VQA-RAD subset in the project JSONL format.

This file intentionally reuses the old script path so existing commands do not
need to change, but the data source is now VQA-RAD instead of TextVQA.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import ensure_dir, load_config, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare a medical VQA-RAD subset.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to project config.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Hugging Face dataset name override.")
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated splits to use, for example train or train,test.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path.")
    parser.add_argument("--image-output-dir", type=str, default=None, help="Directory to save exported images.")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to export.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def parse_splits(split_value: str | list[str]) -> list[str]:
    """Parse split configuration into a list of split names."""
    if isinstance(split_value, list):
        return [str(item).strip() for item in split_value if str(item).strip()]
    return [item.strip() for item in str(split_value).split(",") if item.strip()]


def load_vqa_rad_dataset(dataset_name: str, split_names: list[str]):
    """Load one or more VQA-RAD splits from Hugging Face."""
    split_datasets = []
    for split_name in split_names:
        split_dataset = load_dataset(dataset_name, split=split_name)
        split_dataset = split_dataset.add_column("source_split", [split_name] * len(split_dataset))
        split_datasets.append(split_dataset)
    if len(split_datasets) == 1:
        return split_datasets[0]
    return concatenate_datasets(split_datasets)


def save_image(image, image_output_dir: Path, index: int, split_name: str) -> str:
    """Save a dataset image locally and return its project-relative path."""
    ensure_dir(image_output_dir)
    filename = f"vqa_rad_{split_name}_{index:04d}.jpg"
    output_path = image_output_dir / filename
    image.convert("RGB").save(output_path, format="JPEG", quality=95)
    return str(output_path.relative_to(PROJECT_ROOT))


def normalize_sample(sample: dict[str, Any], image_relative_path: str) -> dict[str, Any]:
    """Convert a VQA-RAD sample to the project JSONL format."""
    return {
        "image": image_relative_path,
        "question": str(sample.get("question", "")).strip(),
        "answer": str(sample.get("answer", "")).strip(),
        "task_type": "xray",
        "source_dataset": "vqa-rad",
        "source_split": sample.get("source_split"),
    }


def main() -> None:
    """Export a subset from VQA-RAD."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    vqa_rad_cfg = config["data"]["vqa_rad"]

    dataset_name = args.dataset_name or vqa_rad_cfg["dataset_name"]
    split_names = parse_splits(args.splits or vqa_rad_cfg["splits"])
    output_path = PROJECT_ROOT / (args.output or vqa_rad_cfg["output_path"])
    image_output_dir = PROJECT_ROOT / (args.image_output_dir or vqa_rad_cfg["image_output_dir"])
    sample_size = args.sample_size or vqa_rad_cfg["sample_size"]

    dataset = load_vqa_rad_dataset(dataset_name=dataset_name, split_names=split_names)
    total_available = len(dataset)
    target_size = min(sample_size, total_available)

    indices = list(range(total_available))
    random.Random(args.seed).shuffle(indices)
    selected_indices = indices[:target_size]

    records: list[dict[str, Any]] = []
    for export_index, dataset_index in enumerate(selected_indices, start=1):
        sample = dataset[int(dataset_index)]
        image_relative_path = save_image(
            image=sample["image"],
            image_output_dir=image_output_dir,
            index=export_index,
            split_name=str(sample.get("source_split", "unknown")),
        )
        records.append(normalize_sample(sample, image_relative_path))

    ensure_dir(output_path.parent)
    write_jsonl(records, output_path)

    print(f"Loaded {total_available} samples from {dataset_name} [{', '.join(split_names)}]")
    print(f"Exported {len(records)} samples to {output_path}")
    print(f"Saved images to {image_output_dir}")


if __name__ == "__main__":
    main()
