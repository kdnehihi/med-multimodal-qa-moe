"""Vision MoE utilities for combining two real visual experts."""

from __future__ import annotations

from torch import Tensor, nn


class VisionMoE(nn.Module):
    """Softly combine two projected visual embeddings."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature_a: Tensor, feature_b: Tensor, routing_weights: Tensor) -> dict[str, Tensor]:
        """Mix two vision expert outputs with soft routing weights."""
        if routing_weights.dim() == 1:
            routing_weights = routing_weights.unsqueeze(0)

        combined_feature = (
            routing_weights[:, 0:1] * feature_a
            + routing_weights[:, 1:2] * feature_b
        )
        return {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "combined_feature": combined_feature,
        }
