"""Semantic equivalence checks between two models."""

from __future__ import annotations

from typing import Any, Mapping

import torch


@torch.no_grad()
def predict_label(model, x: Any) -> torch.Tensor:
    """Run ``model`` on ``x`` and return the predicted class labels.

    Handles vision models (logits ``Tensor``) and HuggingFace models (outputs
    exposing ``.logits``), and both tensor inputs and dict/keyword batches
    (e.g. tokenizer output ``{"input_ids": ..., "attention_mask": ...}``).
    """
    if isinstance(x, Mapping):
        out = model(**x)
    else:
        out = model(x)
    logits = out.logits if hasattr(out, "logits") else out
    return logits.argmax(dim=-1)


@torch.no_grad()
def agreement_mask(model, cl_model, x: Any) -> torch.Tensor:
    """Per-input agreement between two models on batch ``x``.

    Args:
        model: the first model.
        cl_model: the second model to compare against.
        x: a batch of inputs (a ``Tensor`` of shape ``(B, ...)`` or a mapping of
            named input tensors each with leading batch dimension ``B``).

    Returns:
        A boolean ``Tensor`` of shape ``(B,)`` that is ``True`` for every input
        on which ``model`` and ``cl_model`` predict the same (argmax) label.
    """
    y = predict_label(model, x)
    cl_y = predict_label(cl_model, x)
    return y == cl_y


def consistent_num(model, cl_model, x: Any) -> int:
    """Number of inputs in batch ``x`` on which both models predict the same label.

    Args:
        model: the first model.
        cl_model: the second model to compare against.
        x: a batch of inputs (a ``Tensor`` of shape ``(B, ...)`` or a mapping of
            named input tensors each with leading batch dimension ``B``).

    Returns:
        The count of inputs where ``model`` and ``cl_model`` agree on the
        predicted (argmax) label.
    """
    return int(agreement_mask(model, cl_model, x).sum().item())
