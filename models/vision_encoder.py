"""Vision encoder scaffold."""

from __future__ import annotations

from torch import nn


class VisionEncoder(nn.Module):
    """Shared image encoder placeholder.

    Expected responsibility:
    - receive image tensor input
    - return a shared visual representation for downstream expert heads
    """

    def __init__(self, encoder_name: str = "clip-vit-base-patch32") -> None:
        super().__init__()
        self.encoder_name = encoder_name

        # TODO: load a pretrained CLIP / ViT encoder here.
        # Expected input:
        # - image tensor of shape [batch, channels, height, width]
        # Expected output:
        # - visual feature tensor of shape [batch, hidden_dim]
        self.backbone = nn.Identity()

    def forward(self, image):
        """Encode image into a shared feature representation."""
        # TODO: replace placeholder logic with real image encoding.
        return self.backbone(image)
