"""GPT-2 (small, 124M) (Subject #9).

Dataset: WikiText-2 — autoregressive language modeling / perplexity.

Thin wrapper over HuggingFace ``transformers`` for the 124M-parameter GPT-2.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["build_gpt2", "DEFAULT_GPT2_NAME"]

DEFAULT_GPT2_NAME = "gpt2"  # the 124M "small" checkpoint


def build_gpt2(
    pretrained_name: str = DEFAULT_GPT2_NAME,
    return_tokenizer: bool = True,
    **kwargs,
):
    """Build a GPT-2 causal-LM model (and optionally its tokenizer).

    Args:
        pretrained_name: HuggingFace model id or local path ("gpt2" == 124M).
        return_tokenizer: if True returns ``(model, tokenizer)`` else ``model``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model: nn.Module = AutoModelForCausalLM.from_pretrained(pretrained_name, **kwargs)
    if not return_tokenizer:
        return model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
