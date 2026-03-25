"""Top-level multimodal model scaffold."""

from __future__ import annotations

from torch import nn

from models.language_moe import LanguageMoE
from models.router import Router
from models.vision_encoder import VisionEncoder
from models.vision_moe import VisionMoE


class MedicalMoEModel(nn.Module):
    """Research scaffold for a simplified multimodal medical MoE model."""

    def __init__(self, hidden_dim: int = 256, instruction_dim: int = 384) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.vision_moe = VisionMoE(hidden_dim=hidden_dim)
        self.router = Router(input_dim=instruction_dim, hidden_dim=hidden_dim)
        self.language_moe = LanguageMoE(hidden_dim=hidden_dim)

    def forward(self, image, question, task_type):
        """Forward method skeleton for multimodal medical QA.

        Args:
            image: Raw image tensor or image placeholder.
            question: Question text or tokenized question input.
            task_type: One of symptom, text_image, qa.
        """
        _ = image
        _ = question
        _ = task_type

        # TODO: build instruction embedding from question and task type.
        # Expected output:
        # - instruction embedding tensor for router input

        # TODO: encode image into a shared visual feature.
        # Expected output:
        # - shared visual feature tensor from VisionEncoder

        # TODO: route instruction embedding into vision and adapter weights.
        # Expected output:
        # - vision routing weights or logits
        # - adapter routing weights or logits

        # TODO: apply vision expert routing.
        # Expected output:
        # - weighted visual feature from VisionMoE

        # TODO: fuse question representation with visual representation.
        # Expected output:
        # - multimodal representation for generation

        # TODO: call LanguageMoE for answer generation.
        # Expected output:
        # - generation logits or decoded tokens

        return {
            "routing_outputs": None,
            "visual_outputs": None,
            "language_outputs": None,
        }
