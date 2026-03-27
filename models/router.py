"""Router skeleton for expert selection."""

from __future__ import annotations

import torch
from torch import nn


class Router(nn.Module):
    """Instruction-aware router placeholder.

    Input:
    - instruction embedding

    Output:
    - vision weights for 2 virtual vision experts
    - adapter weights for 3 language adapters
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.vision_head = nn.Linear(hidden_dim, 2)
        self.adapter_head = nn.Linear(hidden_dim, 3)

    def forward(self, instruction_embedding):
        """Predict routing outputs from an instruction embedding."""
        hidden = self.backbone(instruction_embedding)
        vision_logits = self.vision_head(hidden)
        adapter_logits = self.adapter_head(hidden)
        return {
            "vision_logits": vision_logits,
            "vision_weights": torch.softmax(vision_logits, dim=-1),
            "adapter_logits": adapter_logits,
            "adapter_weights": torch.softmax(adapter_logits, dim=-1),
        }
