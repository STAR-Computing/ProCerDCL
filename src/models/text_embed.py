"""Embedding-space adapter for text-classification subjects (E1).

Why this exists
---------------
The verifier, the samplers and the attacks all assume a *continuous* input
tensor: they draw perturbations uniformly from an L2 epsilon-ball around ``x``
and call ``model(x + delta) -> logits``. Token ids are discrete, so an
epsilon-ball over them is meaningless.

We therefore define the epsilon-ball in the model's **frozen input-embedding
space**: for a tokenized input we take ``E(ids) -> (L, d)`` once, and treat that
matrix as the base input ``x``. :class:`EmbeddingClassifier` makes a
HuggingFace sequence-classification model look exactly like a vision model over
that space -- it accepts a plain ``(B, L, d)`` tensor and returns ``(B, C)``
logits -- so every downstream component works unchanged, including
``torch.compile`` and the TensorRT/ONNX export paths.

The attention mask is fixed per input (it depends only on the tokenization, not
on the perturbation), so it is baked into the wrapper as a buffer rather than
passed per call. That keeps the call signature a single tensor, which is what
the sampler and the compiled-model plumbing expect.

Note the semantics this pins down for the paper: the property being certified
is still "the compiled model disagrees with the original model's predicted
label", and the perturbation set is an L2 ball in embedding space. Token-level
(synonym-substitution) neighbourhoods are a different perturbation set and are
left as future work.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "EmbeddingClassifier",
    "input_embeddings",
    "tokenize_fixed",
    "is_text_task",
]


def is_text_task(task) -> bool:
    """True if ``task`` is one of the text-classification subjects (E1)."""
    from utils import NLP_MODEL_KEYS

    return task.model_key in NLP_MODEL_KEYS


def tokenize_fixed(tokenizer, text: str, max_length: int):
    """Tokenize one example to a fixed length.

    Every input is padded to ``max_length`` so the embedding tensor has a
    constant shape across the dataset. Constant shapes are what let the driver
    compile the model once and then reuse it for every input (and they keep the
    TensorRT/ONNX export paths on a single static shape).

    Returns ``(input_ids, attention_mask)``, both ``(max_length,)``.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"][0], enc["attention_mask"][0]


def _embedding_module(model: nn.Module) -> nn.Module:
    """Return the input-embedding lookup of a HF sequence-classification model."""
    # get_input_embeddings() is part of the PreTrainedModel API and covers
    # BERT, DistilBERT, RoBERTa, ... without per-architecture branching.
    get = getattr(model, "get_input_embeddings", None)
    if get is None:
        raise TypeError(
            f"{type(model).__name__} has no get_input_embeddings(); cannot "
            f"build an embedding-space view of it"
        )
    return get()


@torch.no_grad()
def input_embeddings(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Map token ids to the frozen input embeddings used as the base input.

    Args:
        model: HF sequence-classification model.
        input_ids: ``(B, L)`` or ``(L,)`` token ids.

    Returns:
        ``(B, L, d)`` (or ``(L, d)`` if the input had no batch dimension).
    """
    emb = _embedding_module(model)
    squeeze = input_ids.dim() == 1
    ids = input_ids.unsqueeze(0) if squeeze else input_ids
    out = emb(ids.to(next(emb.parameters()).device))
    return out.squeeze(0) if squeeze else out


class EmbeddingClassifier(nn.Module):
    """Wrap a HF text classifier as ``(B, L, d) embeddings -> (B, C) logits``.

    Args:
        model: a HuggingFace sequence-classification model.
        attention_mask: ``(L,)`` mask for the tokenized input this wrapper is
            specialised to. Broadcast over the batch at call time. If ``None``,
            all positions are treated as real tokens.

    The wrapper holds a reference to ``model`` (it does not copy weights), so
    wrapping the original and the compiled model keeps them sharing semantics
    exactly as the vision subjects do.
    """

    def __init__(self, model: nn.Module, attention_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model
        if attention_mask is not None:
            mask = attention_mask.detach().to(torch.long).flatten()
            self.register_buffer("attention_mask", mask, persistent=False)
        else:
            self.attention_mask = None

    @torch.no_grad()
    def set_mask(self, attention_mask: torch.Tensor) -> None:
        """Point the wrapper at a new input's mask, in place.

        Compilation is expensive, so a driver compiles the wrapper once and
        then walks many inputs through it. Every input is padded to the same
        ``max_length``, so the buffer's *shape* never changes and only its
        values do -- which means ``torch.compile`` does not retrace. Mutating
        the buffer is therefore the cheap way to switch inputs; rebuilding the
        wrapper per image would force a recompile each time.
        """
        if self.attention_mask is None:
            raise RuntimeError(
                "wrapper was built without a mask; construct it with one to "
                "enable set_mask()"
            )
        new = attention_mask.detach().to(
            device=self.attention_mask.device, dtype=self.attention_mask.dtype
        ).flatten()
        if new.shape != self.attention_mask.shape:
            raise ValueError(
                f"mask shape {tuple(new.shape)} != {tuple(self.attention_mask.shape)}; "
                f"pad every input to the task's fixed max_length"
            )
        self.attention_mask.copy_(new)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if inputs_embeds.dim() == 2:  # (L, d) -> (1, L, d)
            inputs_embeds = inputs_embeds.unsqueeze(0)
        b = inputs_embeds.shape[0]
        kwargs = {"inputs_embeds": inputs_embeds}
        if self.attention_mask is not None:
            kwargs["attention_mask"] = self.attention_mask.unsqueeze(0).expand(b, -1)
        out = self.model(**kwargs)
        # HF returns a ModelOutput; .logits is the (B, C) tensor we need.
        return out.logits if hasattr(out, "logits") else out[0]
