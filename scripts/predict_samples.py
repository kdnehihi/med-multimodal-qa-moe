"""Preview baseline predictions on a few validation samples."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.medical_dataset import MedicalDataset
from models.multimodal_model import MedicalMoEModel
from utils.helpers import load_config, write_jsonl
from utils.logging import setup_logger


def set_seed(seed: int) -> None:
    """Set seeds for reproducible validation sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    """Choose a safe runtime device."""
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_indices(dataset_size: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Reproduce the same train/validation split as training."""
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(dataset_size * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def main() -> None:
    """Load a checkpoint and preview a few validation predictions."""
    parser = argparse.ArgumentParser(description="Preview baseline predictions.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/query_router_biomedclip_dinov2_flan_t5_base/best_baseline.pt",
        help="Path to a saved checkpoint.",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Number of validation samples to preview.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Maximum generation length.")
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="outputs/predictions/baseline_preview_predictions.jsonl",
        help="Optional JSONL file for saving preview results.",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    logger = setup_logger("predict")
    seed = int(config["seed"])
    set_seed(seed)

    device = resolve_device(config["training"]["device"])
    logger.info("Using device=%s", device)

    dataset = MedicalDataset(PROJECT_ROOT / config["data"]["train_path"])
    _, val_indices = split_indices(
        dataset_size=len(dataset),
        val_ratio=float(config["data"]["val_ratio"]),
        seed=seed,
    )
    val_samples = [dataset[index] for index in val_indices]
    num_samples = min(args.num_samples, len(val_samples))
    preview_samples = val_samples[:num_samples]

    model = MedicalMoEModel(
        vision_model_name=config["model"]["vision_model_name"],
        second_vision_model_name=config["model"]["second_vision_model_name"],
        text_model_name=config["model"]["text_model_name"],
        max_question_length=int(config["training"]["max_question_length"]),
        max_answer_length=int(config["training"]["max_answer_length"]),
        freeze_vision=bool(config["model"]["freeze_vision"]),
        freeze_text_encoder=bool(config["model"]["freeze_text_encoder"]),
        gradient_checkpointing=False,
    )

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_paths = [sample["image"] for sample in preview_samples]
    questions = [sample["question"] for sample in preview_samples]
    expected_answers = [sample["answer"] for sample in preview_samples]

    predicted_answers = model.generate(
        image_paths=image_paths,
        questions=questions,
        max_new_tokens=args.max_new_tokens,
    )

    preview_records = []
    for index, (sample, prediction) in enumerate(zip(preview_samples, predicted_answers), start=1):
        record = {
            "index": index,
            "image": sample["image"],
            "question": sample["question"],
            "expected_answer": sample["answer"],
            "predicted_answer": prediction.strip(),
        }
        preview_records.append(record)

        print("=" * 100)
        print(f"Sample #{index}")
        print(f"Image: {sample['image']}")
        print(f"Question: {sample['question']}")
        print(f"Expected: {sample['answer']}")
        print(f"Predicted: {prediction.strip()}")

    output_path = PROJECT_ROOT / args.output_jsonl
    write_jsonl(preview_records, output_path)
    logger.info("Saved preview predictions to %s", output_path)


if __name__ == "__main__":
    main()
