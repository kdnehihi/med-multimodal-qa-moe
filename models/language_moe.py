"""Language MoE scaffold."""

from __future__ import annotations

from torch import nn

from models.adapters import Adapter


class LanguageMoE(nn.Module):
    """Placeholder language backbone with adapter expert slots."""

    def __init__(self, hidden_dim: int = 512, adapter_dim: int = 64) -> None:
        super().__init__()

        # TODO: replace with a real small language model such as T5-small.
        self.base_model = nn.Identity()

        self.diagnosis_adapter = Adapter(hidden_dim=hidden_dim, adapter_dim=adapter_dim)
        self.extraction_adapter = Adapter(hidden_dim=hidden_dim, adapter_dim=adapter_dim)
        self.general_qa_adapter = Adapter(hidden_dim=hidden_dim, adapter_dim=adapter_dim)

    def forward(self, input_ids, attention_mask=None, adapter_weights=None):
        """Run the language backbone and adapter experts."""
        _ = input_ids
        _ = attention_mask
        _ = adapter_weights

        # TODO: implement:
        # 1. base language model forward
        # 2. extract hidden states
        # 3. run adapter experts
        # 4. combine adapter outputs using router weights
        # 5. generate answer logits or tokens
        return {
            "base_output": None,
            "logits": None,
        }
