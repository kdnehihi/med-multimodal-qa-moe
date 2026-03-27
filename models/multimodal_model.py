"""Baseline multimodal model built with BiomedCLIP and FLAN-T5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

from models.vision_encoder import VisionEncoder


class MedicalMoEModel(nn.Module):
    """Shared baseline model without MoE routing.

    Design:
    - frozen BiomedCLIP image encoder
    - FLAN-T5-base text generator
    - a trainable visual projector that prepends one visual prefix token
    """

    def __init__(
        self,
        vision_model_name: str,
        text_model_name: str,
        max_question_length: int,
        max_answer_length: int,
        freeze_vision: bool = True,
        freeze_text_encoder: bool = True,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

        self.vision_encoder = VisionEncoder(model_name=vision_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = T5ForConditionalGeneration.from_pretrained(text_model_name)

        if gradient_checkpointing:
            self.text_model.gradient_checkpointing_enable()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        text_hidden_dim = int(self.text_model.config.d_model)
        self.visual_projector = nn.Sequential(
            nn.Linear(self.vision_encoder.output_dim, text_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(text_hidden_dim),
        )
        self.no_image_embedding = nn.Parameter(torch.zeros(text_hidden_dim))

        if freeze_vision:
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = False

        if freeze_text_encoder:
            for parameter in self.text_model.encoder.parameters():
                parameter.requires_grad = False

    def load_images(self, image_paths: list[str | None]) -> list[Image.Image | None]:
        """Load images from disk while handling text-only samples gracefully."""
        images: list[Image.Image | None] = []
        for path in image_paths:
            if not path:
                images.append(None)
                continue
            image = Image.open(Path(path)).convert("RGB")
            images.append(image)
        return images

    def build_visual_prefix(self, image_paths: list[str | None], device: torch.device) -> Tensor:
        """Build one projected visual token per sample.

        Text-only samples receive a learned no-image embedding.
        """
        batch_size = len(image_paths)
        prefix = self.no_image_embedding.unsqueeze(0).expand(batch_size, -1).to(device).clone()

        valid_indices = [index for index, path in enumerate(image_paths) if path]
        if not valid_indices:
            return prefix

        loaded_images = self.load_images([image_paths[index] for index in valid_indices])
        valid_images = [image for image in loaded_images if image is not None]
        if not valid_images:
            return prefix

        with torch.no_grad():
            vision_inputs = self.vision_encoder.preprocess(valid_images, device=device)
            image_features = self.vision_encoder(**vision_inputs)

        projected_features = self.visual_projector(image_features)
        for local_index, batch_index in enumerate(valid_indices):
            prefix[batch_index] = projected_features[local_index]

        return prefix

    def tokenize_batch(
        self,
        questions: list[str],
        answers: list[str] | None = None,
        device: torch.device | None = None,
    ) -> dict[str, Tensor]:
        """Tokenize questions and answers for T5."""
        question_texts = [f"medical qa: {question.strip()}" for question in questions]
        model_inputs = self.tokenizer(
            question_texts,
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )

        labels = None
        if answers is not None:
            target_tokens = self.tokenizer(
                answers,
                padding=True,
                truncation=True,
                max_length=self.max_answer_length,
                return_tensors="pt",
            )
            labels = target_tokens["input_ids"]
            labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        if device is None:
            return {"input_ids": model_inputs["input_ids"], "attention_mask": model_inputs["attention_mask"], "labels": labels}

        tokenized: dict[str, Tensor] = {
            "input_ids": model_inputs["input_ids"].to(device),
            "attention_mask": model_inputs["attention_mask"].to(device),
        }
        if labels is not None:
            tokenized["labels"] = labels.to(device)
        return tokenized

    def forward(self, image_paths: list[str | None], questions: list[str], answers: list[str] | None = None) -> dict[str, Any]:
        """Run a forward pass and return loss/logits."""
        device = next(self.parameters()).device
        tokenized = self.tokenize_batch(questions=questions, answers=answers, device=device)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = tokenized.get("labels")

        text_embeddings = self.text_model.encoder.embed_tokens(input_ids)
        visual_prefix = self.build_visual_prefix(image_paths=image_paths, device=device).unsqueeze(1)
        inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)

        prefix_attention = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=device)
        extended_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=labels,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(self, image_paths: list[str | None], questions: list[str], max_new_tokens: int = 48) -> list[str]:
        """Generate answers for qualitative validation."""
        device = next(self.parameters()).device
        tokenized = self.tokenize_batch(questions=questions, device=device)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        text_embeddings = self.text_model.encoder.embed_tokens(input_ids)
        visual_prefix = self.build_visual_prefix(image_paths=image_paths, device=device).unsqueeze(1)
        inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)

        prefix_attention = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=device)
        extended_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        generated = self.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
