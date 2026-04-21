"""Helpers shared across attacks."""
from __future__ import annotations

from typing import Sequence

from ..data.registry import Behavior
from ..models.chat_template import build_input_ids
from ..models.load import LoadedModel


def tokenize_behavior(
    bundle: LoadedModel, behavior: Behavior, suffix_str: str
):
    """Build (slot, target_ids) for a single behavior."""
    import torch

    slot = build_input_ids(
        bundle.tokenizer, bundle.spec, behavior.prompt, suffix_str
    )
    target_ids = bundle.tokenizer(
        behavior.target_prefix, add_special_tokens=False
    ).input_ids
    return slot, torch.tensor([target_ids], dtype=torch.long)


def init_suffix_ids(bundle: LoadedModel, init_str: str, length: int) -> list[int]:
    """Tokenize init_str; pad / truncate to ``length``."""
    ids = bundle.tokenizer(init_str, add_special_tokens=False).input_ids
    if len(ids) < length:
        # Repeat the bang token to pad.
        bang = bundle.tokenizer("!", add_special_tokens=False).input_ids
        if not bang:
            raise ValueError("Tokenizer has no '!' token — pick a different init.")
        while len(ids) < length:
            ids = ids + bang
    return ids[:length]
