"""Vision encoder utilities for the baseline model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoProcessor


class VisionEncoder(nn.Module):
    """Wrapper around a pretrained vision-language image tower.

    The baseline uses BiomedCLIP as a frozen image encoder and only consumes
    its pooled image embedding for downstream projection into the T5 hidden
    space.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.processor = None
        self.image_transform = None
        self.uses_open_clip = "biomedclip" in model_name.lower()

        if self.uses_open_clip:
            try:
                import open_clip
            except ImportError as error:
                raise ImportError(
                    "BiomedCLIP requires the `open_clip_torch` package. "
                    "Install it with `python -m pip install open_clip_torch timm`."
                ) from error

            self.backbone, _, self.image_transform = open_clip.create_model_and_transforms(
                model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            )
            self.output_dim = self._infer_open_clip_output_dim()
        else:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            if hasattr(self.backbone.config, "projection_dim"):
                self.output_dim = int(self.backbone.config.projection_dim)
            elif hasattr(self.backbone.config, "hidden_size"):
                self.output_dim = int(self.backbone.config.hidden_size)
            else:
                raise AttributeError("Could not infer vision output dimension from the encoder config.")

    def _infer_open_clip_output_dim(self) -> int:
        """Infer embedding size for OpenCLIP-based backbones.

        BiomedCLIP commonly returns a 512-dimensional normalized image embedding.
        This helper checks a few common fields first, then falls back to 512.
        """
        if hasattr(self.backbone, "visual") and hasattr(self.backbone.visual, "output_dim"):
            value = getattr(self.backbone.visual, "output_dim")
            if value is not None:
                return int(value)

        if hasattr(self.backbone, "visual") and hasattr(self.backbone.visual, "proj"):
            proj = getattr(self.backbone.visual, "proj")
            if hasattr(proj, "shape") and len(proj.shape) == 2:
                return int(proj.shape[1])

        if hasattr(self.backbone, "text_projection"):
            text_projection = getattr(self.backbone, "text_projection")
            if hasattr(text_projection, "shape") and len(text_projection.shape) == 2:
                return int(text_projection.shape[1])

        # BiomedCLIP ViT-B/16 embeddings are typically 512-dim.
        return 512

    def preprocess(self, images: list[Any], device: torch.device) -> dict[str, Tensor]:
        """Convert PIL images into model-ready tensors."""
        if self.uses_open_clip:
            if self.image_transform is None:
                raise RuntimeError("OpenCLIP image transform was not initialized.")
            pixel_values = torch.stack([self.image_transform(image) for image in images]).to(device)
            return {"pixel_values": pixel_values}

        processed = self.processor(images=images, return_tensors="pt")
        return {key: value.to(device) for key, value in processed.items() if isinstance(value, Tensor)}

    def forward(self, **vision_inputs: Tensor) -> Tensor:
        """Encode images into pooled feature vectors."""
        if self.uses_open_clip:
            return self.backbone.encode_image(vision_inputs["pixel_values"])

        outputs = self.backbone(**vision_inputs)

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0]

        raise ValueError("Unsupported vision output structure for the selected encoder.")
