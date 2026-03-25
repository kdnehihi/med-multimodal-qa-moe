"""Training entry script scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.medical_dataset import MedicalDataset
from models.multimodal_model import MedicalMoEModel
from trainers.trainer import Trainer
from utils.helpers import load_config
from utils.logging import setup_logger


def main() -> None:
    """Parse config and prepare the training scaffold."""
    parser = argparse.ArgumentParser(description="Train medical-moe-assistant.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    logger = setup_logger("train")
    logger.info("Loaded config from %s", args.config)

    train_dataset = MedicalDataset(PROJECT_ROOT / config["data"]["train_path"])
    val_dataset = MedicalDataset(PROJECT_ROOT / config["data"]["val_path"])

    # TODO: create DataLoader objects from train_dataset and val_dataset.
    # TODO: instantiate optimizer, scheduler, and loss function.
    model = MedicalMoEModel(
        hidden_dim=config["model"]["hidden_dim"],
        instruction_dim=config["model"]["instruction_dim"],
    )
    trainer = Trainer(model=model, train_loader=train_dataset, val_loader=val_dataset, config=config)

    logger.info("Trainer scaffold created.")
    logger.info("Implement Trainer.train() and call it here when ready.")
    _ = trainer


if __name__ == "__main__":
    main()
