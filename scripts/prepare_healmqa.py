"""Prepare the HealMQA task-1 dataset into the project JSONL format."""

from __future__ import annotations

import argparse
import shutil
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import ensure_dir, load_config, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare HealMQA data.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to project config.")
    parser.add_argument("--annotations", type=str, default=None, help="Override HealMQA JSON path.")
    parser.add_argument("--image-dir", type=str, default=None, help="Override HealMQA image directory.")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path.")
    parser.add_argument("--image-output-dir", type=str, default=None, help="Directory to save processed images.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    """Load HealMQA records from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Expected HealMQA_500.json to contain a list of records.")
    return payload


def export_image(image_path: Path, image_output_dir: Path) -> str:
    """Copy an image into the processed directory and return the project-relative path."""
    ensure_dir(image_output_dir)
    output_path = image_output_dir / image_path.name
    if not output_path.exists():
        shutil.copy2(image_path, output_path)
    return str(output_path.relative_to(PROJECT_ROOT))


def normalize_record(record: dict, image_dir: Path, image_output_dir: Path) -> dict | None:
    """Normalize a HealMQA record into project format."""
    image_name = str(record.get("image", "")).strip()
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    uri = record.get("uri")

    if not image_name or not question or not answer:
        return None

    image_path = image_dir / image_name
    if not image_path.exists():
        return None

    processed_image_path = export_image(image_path, image_output_dir=image_output_dir)

    return {
        "id": f"healmqa_{uri}" if uri is not None else f"healmqa_{image_name}",
        "image": processed_image_path,
        "question": question,
        "answer": answer,
        "task_type": "symptom",
        "source_dataset": "healmqa",
        "source_split": "custom",
        "uri": uri,
    }


def main() -> None:
    """Convert HealMQA into the shared project format."""
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    healmqa_cfg = config["data"]["healmqa"]

    annotations_path = PROJECT_ROOT / (args.annotations or healmqa_cfg["annotations_path"])
    image_dir = PROJECT_ROOT / (args.image_dir or healmqa_cfg["image_dir"])
    output_path = PROJECT_ROOT / (args.output or healmqa_cfg["output_path"])
    image_output_dir = PROJECT_ROOT / (args.image_output_dir or healmqa_cfg["image_output_dir"])

    records = load_records(annotations_path)
    normalized_records = []
    skipped = 0
    for record in records:
        normalized = normalize_record(
            record,
            image_dir=image_dir,
            image_output_dir=image_output_dir,
        )
        if normalized is None:
            skipped += 1
            continue
        normalized_records.append(normalized)

    ensure_dir(output_path.parent)
    write_jsonl(normalized_records, output_path)

    print(f"Loaded {len(records)} HealMQA records from {annotations_path}")
    print(f"Exported {len(normalized_records)} normalized records to {output_path}")
    print(f"Saved images to {image_output_dir}")
    print(f"Skipped {skipped} malformed or missing-image records")


if __name__ == "__main__":
    main()
