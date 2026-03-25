"""Vision MoE scaffold with virtual experts."""

from __future__ import annotations

from torch import nn


class VisionMoE(nn.Module):
    """Two virtual vision experts over a shared visual feature."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.natural_image_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_image_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features, routing_weights):
        """Route a shared visual feature through two virtual experts.

        Args:
            features: Shared visual feature from the vision encoder.
            routing_weights: Router output for the two virtual experts.
        """
        natural_feature = self.natural_image_head(features)
        text_feature = self.text_image_head(features)

        # TODO: implement weighted combination using routing_weights.
        # Expected output:
        # - one combined visual feature for downstream multimodal fusion
        _ = routing_weights
        return {
            "natural_feature": natural_feature,
            "text_feature": text_feature,
            "combined_feature": None,
        }
