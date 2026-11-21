"""BERT-base-uncased (Subject #7).

Datasets: SST-2, MRPC (GLUE) — sentence/sentence-pair classification.

Thin wrapper over HuggingFace ``transformers``. Both GLUE tasks here are binary
classification, so ``num_labels`` defaults to 2.
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn

__all__ = ["build_bert", "DEFAULT_BERT_NAME"]

DEFAULT_BERT_NAME = "bert-base-uncased"


def build_bert(
    pretrained_name: str = DEFAULT_BERT_NAME,
    num_labels: int = 2,
    return_tokenizer: bool = True,
    **kwargs,
):
    """Build a BERT sequence-classification model (and optionally its tokenizer).

    Args:
        pretrained_name: HuggingFace model id or local path.
        num_labels: number of output classes (2 for SST-2 / MRPC).
        return_tokenizer: if True returns ``(model, tokenizer)`` else ``model``.

    Returns:
        ``(model, tokenizer)`` or ``model``.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model: nn.Module = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name, num_labels=num_labels, **kwargs
    )
    if not return_tokenizer:
        return model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    return model, tokenizer
