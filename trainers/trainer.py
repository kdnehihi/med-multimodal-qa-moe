"""Minimal trainer for the baseline multimodal QA model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.helpers import ensure_dir


class Trainer:
    """Small trainer focused on stable baseline runs and loss tracking."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict[str, Any],
        logger,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logger = logger

        training_config = config["training"]
        self.epochs = int(training_config["epochs"])
        self.gradient_accumulation_steps = int(training_config["gradient_accumulation_steps"])
        self.max_train_steps = int(training_config["max_train_steps"])
        self.eval_steps = int(training_config["eval_steps"])
        self.log_steps = int(training_config["log_steps"])
        self.output_dir = Path(training_config["output_dir"])

        ensure_dir(self.output_dir)
        self.model.to(self.device)

    def train(self) -> None:
        """Run baseline training and report train/validation loss."""
        self.logger.info("Starting baseline training on device=%s", self.device)
        global_step = 0
        best_val_loss = math.inf

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_batches = 0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)
            self.optimizer.zero_grad(set_to_none=True)

            for batch_index, batch in enumerate(progress, start=1):
                outputs = self.model(
                    image_paths=batch["image"],
                    questions=batch["question"],
                    answers=batch["answer"],
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps
                loss.backward()

                running_loss += float(loss.detach().item()) * self.gradient_accumulation_steps
                running_batches += 1

                if batch_index % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % self.log_steps == 0:
                        average_loss = running_loss / max(running_batches, 1)
                        self.logger.info("step=%s train_loss=%.4f", global_step, average_loss)
                        running_loss = 0.0
                        running_batches = 0

                    if global_step % self.eval_steps == 0:
                        val_loss = self.evaluate()
                        self.logger.info("step=%s val_loss=%.4f", global_step, val_loss)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_checkpoint("best_baseline.pt", step=global_step, val_loss=val_loss)

                    if global_step >= self.max_train_steps:
                        self.logger.info("Reached max_train_steps=%s. Stopping early.", self.max_train_steps)
                        self.save_checkpoint("last_baseline.pt", step=global_step, val_loss=best_val_loss)
                        return

            if batch_index % self.gradient_accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

        self.save_checkpoint("last_baseline.pt", step=global_step, val_loss=best_val_loss)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute mean validation loss."""
        self.model.eval()
        losses: list[float] = []
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            outputs = self.model(
                image_paths=batch["image"],
                questions=batch["question"],
                answers=batch["answer"],
            )
            losses.append(float(outputs["loss"].detach().item()))
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    def save_checkpoint(self, filename: str, step: int, val_loss: float) -> None:
        """Persist a small checkpoint for later inspection."""
        checkpoint_path = self.output_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": step,
                "val_loss": val_loss,
                "config": self.config,
            },
            checkpoint_path,
        )
        self.logger.info("Saved checkpoint to %s", checkpoint_path)
