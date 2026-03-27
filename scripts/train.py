"""Train the baseline BiomedCLIP + FLAN-T5 model."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.medical_dataset import MedicalDataset
from models.multimodal_model import MedicalMoEModel
from trainers.trainer import Trainer
from utils.helpers import load_config
from utils.logging import setup_logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_collate_fn():
    """Return a simple collator that preserves raw strings and image paths."""

    def collate_fn(batch):
        return {
            "image": [item["image"] for item in batch],
            "question": [item["question"] for item in batch],
            "answer": [item["answer"] for item in batch],
        }

    return collate_fn


def resolve_device(device_name: str) -> torch.device:
    """Choose a safe training device."""
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_dataset(dataset: MedicalDataset, val_ratio: float, seed: int) -> tuple[Subset, Subset]:
    """Create a reproducible train/validation split from one merged JSONL file."""
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def main() -> None:
    """Train the baseline model and log training/validation loss."""
    parser = argparse.ArgumentParser(description="Train the baseline medical QA model.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    logger = setup_logger("train")
    seed = int(config["seed"])
    set_seed(seed)

    dataset = MedicalDataset(PROJECT_ROOT / config["data"]["train_path"])
    train_dataset, val_dataset = split_dataset(
        dataset=dataset,
        val_ratio=float(config["data"]["val_ratio"]),
        seed=seed,
    )

    training_config = config["training"]
    collate_fn = build_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_config["eval_batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
        collate_fn=collate_fn,
    )

    device = resolve_device(training_config["device"])
    logger.info("Using device=%s", device)
    logger.info("Train samples=%s | Val samples=%s", len(train_dataset), len(val_dataset))

    model = MedicalMoEModel(
        vision_model_name=config["model"]["vision_model_name"],
        text_model_name=config["model"]["text_model_name"],
        max_question_length=int(config["training"]["max_question_length"]),
        max_answer_length=int(config["training"]["max_answer_length"]),
        freeze_vision=bool(config["model"]["freeze_vision"]),
        freeze_text_encoder=bool(config["model"]["freeze_text_encoder"]),
        gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        logger=logger,
    )
    trainer.train()


if __name__ == "__main__":
    main()
