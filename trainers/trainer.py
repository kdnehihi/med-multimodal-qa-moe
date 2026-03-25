"""Trainer skeleton for medical-moe-assistant."""

from __future__ import annotations

from typing import Any


class Trainer:
    """Minimal training scaffold.

    This class only defines where training and evaluation logic should go.
    """

    def __init__(self, model, train_loader=None, val_loader=None, config: dict[str, Any] | None = None) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}

    def train(self) -> None:
        """Training loop skeleton."""
        # TODO: implement:
        # 1. iterate over train_loader
        # 2. run model forward pass
        # 3. compute generation loss
        # 4. backpropagate
        # 5. optimizer step
        raise NotImplementedError("Training logic is intentionally left as a scaffold.")

    def evaluate(self) -> None:
        """Evaluation loop skeleton."""
        # TODO: implement:
        # - validation forward pass
        # - metric computation
        # - routing distribution summaries
        raise NotImplementedError("Evaluation logic is intentionally left as a scaffold.")
