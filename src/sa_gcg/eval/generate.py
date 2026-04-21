"""Generation primitives.

Two flavors:
  - ``generate_with_suffix``: append (or in PAIR's case, replace) the suffix
    onto the user message and decode greedy / temperature-0 / 256 new tokens.
  - ``generate_replacement``: PAIR-style "the suffix IS the prompt" path.

Hooks (e.g. directional ablation for B6 Ortho) can be passed in.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional, Sequence

from ..models.chat_template import ChatSpec, build_input_ids
from ..models.load import LoadedModel
from ..utils.logging import get_logger

LOG = get_logger(__name__)


@contextmanager
def _maybe_hooks(handles):
    try:
        yield
    finally:
        if handles:
            for h in handles:
                h.remove()


def generate_with_suffix(
    bundle: LoadedModel,
    user_messages: Sequence[str],
    suffix_str: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    system: Optional[str] = None,
    hook_handles: Optional[Iterable] = None,
    batch_size: int = 4,
) -> list[str]:
    """Append ``suffix_str`` to each user message, decode, return responses."""
    import torch

    model, tok, spec = bundle.model, bundle.tokenizer, bundle.spec
    device = next(model.parameters()).device

    out_texts: list[str] = []
    handles = list(hook_handles) if hook_handles is not None else None
    with _maybe_hooks(handles):
        for i in range(0, len(user_messages), batch_size):
            batch = list(user_messages[i : i + batch_size])
            ids_list = []
            for u in batch:
                slot = build_input_ids(tok, spec, u, suffix_str, system=system)
                ids_list.append(slot.input_ids)
            # Right-pad
            max_len = max(len(x) for x in ids_list)
            pad_id = tok.pad_token_id
            padded = [
                # left-pad so generation starts at the rightmost real token
                [pad_id] * (max_len - len(x)) + list(x)
                for x in ids_list
            ]
            attn = [
                [0] * (max_len - len(x)) + [1] * len(x) for x in ids_list
            ]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
            if temperature == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(temperature)

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            new_tok = out[:, input_ids.shape[1] :]
            for row in new_tok:
                out_texts.append(tok.decode(row, skip_special_tokens=True).strip())
    return out_texts


def generate_replacement(
    bundle: LoadedModel,
    full_user_prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    system: Optional[str] = None,
    batch_size: int = 4,
) -> list[str]:
    """For PAIR-style attacks where the 'suffix' is actually the full
    rewritten user prompt. Sends the prompt as the entire user turn.
    """
    import torch

    model, tok, spec = bundle.model, bundle.tokenizer, bundle.spec
    device = next(model.parameters()).device
    out_texts: list[str] = []

    for i in range(0, len(full_user_prompts), batch_size):
        batch = list(full_user_prompts[i : i + batch_size])
        ids_list = []
        for u in batch:
            sys_text = system if system is not None else (spec.default_system or "")
            text = (
                spec.prefix_tmpl.format(system=sys_text)
                + u
                + spec.suffix_postfix
            )
            ids_list.append(tok(text, add_special_tokens=False).input_ids)
        max_len = max(len(x) for x in ids_list)
        pad_id = tok.pad_token_id
        padded = [[pad_id] * (max_len - len(x)) + list(x) for x in ids_list]
        attn = [[0] * (max_len - len(x)) + [1] * len(x) for x in ids_list]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                do_sample=(temperature > 0),
                temperature=max(float(temperature), 1.0),
                use_cache=True,
            )
        new_tok = out[:, input_ids.shape[1] :]
        for row in new_tok:
            out_texts.append(tok.decode(row, skip_special_tokens=True).strip())
    return out_texts
