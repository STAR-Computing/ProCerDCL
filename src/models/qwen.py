"""Qwen3.5-4B-Base (Subject #10).

Dataset: WikiText-2 — autoregressive language modeling / perplexity.

Thin wrapper over HuggingFace ``transformers`` for the 4B-parameter Qwen base
model. The exact repository id is configurable via ``pretrained_name`` so it can
be pointed at whichever Qwen base checkpoint is available locally / on the Hub.
``torch_dtype="auto"`` and ``device_map`` are exposed because a 4B model is large.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["build_qwen", "DEFAULT_QWEN_NAME"]

# Adjust to the precise repo id you intend to use (e.g. a local snapshot path).
DEFAULT_QWEN_NAME = "Qwen/Qwen3-4B-Base"


def build_qwen(
    pretrained_name: str = DEFAULT_QWEN_NAME,
    return_tokenizer: bool = True,
    torch_dtype="auto",
    device_map=None,
    **kwargs,
):
    """Build a Qwen causal-LM model (and optionally its tokenizer).

    Args:
        pretrained_name: HuggingFace model id or local path of the 4B base model.
        return_tokenizer: if True returns ``(model, tokenizer)`` else ``model``.
        torch_dtype: dtype for weights ("auto" picks the checkpoint's dtype).
        device_map: passed through to ``from_pretrained`` (e.g. "auto" to shard).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model: nn.Module = AutoModelForCausalLM.from_pretrained(
        pretrained_name, torch_dtype=torch_dtype, device_map=device_map, **kwargs
    )
    if not return_tokenizer:
        return model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    return model, tokenizer
