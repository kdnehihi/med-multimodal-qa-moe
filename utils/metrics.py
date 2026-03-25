"""Placeholder metric helpers."""

from __future__ import annotations


def compute_accuracy(predictions, targets):
    """Placeholder accuracy function.

    TODO:
    - define how predictions and targets are compared
    - support token-level or answer-level accuracy
    """
    _ = predictions
    _ = targets
    return None


def summarize_routing(vision_weights, adapter_weights):
    """Placeholder router summary helper.

    TODO:
    - aggregate routing weights over a batch or epoch
    - return readable statistics for logging
    """
    _ = vision_weights
    _ = adapter_weights
    return {}
