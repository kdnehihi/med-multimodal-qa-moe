"""Adapter skeletons for language-side experts."""

from __future__ import annotations

from torch import nn


class Adapter(nn.Module):
    """Minimal bottleneck adapter scaffold.

    Structure:
    down -> activation -> up
    """

    def __init__(self, hidden_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)

    def forward(self, hidden_states):
        """Transform hidden states through the adapter block."""
        # TODO: plug this into the language model hidden states.
        # Expected input:
        # - hidden states from the base language model
        # Expected output:
        # - adapted hidden states to mix back into the model
        return self.up_proj(self.activation(self.down_proj(hidden_states)))
