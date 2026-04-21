"""§12.3 Adaptive attack: SA-GCG with an extra perplexity penalty.

Uses the existing ``perplexity_penalty`` field on ``AttackConfig``, which
SA-GCG already wires into ``_suffix_perplexity_proxy`` (negative entropy
of the soft suffix). The penalty weight 0.1 is the recommended default;
override for sensitivity sweeps.

For the most rigorous version we also expose a "real" two-model proxy:
forward the soft suffix through a base LM and read its CE on the suffix
itself. This lives in ``adaptive_perplexity_loss`` below; SA-GCG can be
extended to call it via a custom hook.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from ..attack.base import AttackConfig
from ..models.load import LoadedModel


def adaptive_sa_gcg_config(
    base_cfg: AttackConfig,
    *,
    perplexity_penalty: float = 0.1,
) -> AttackConfig:
    """Return a copy of ``base_cfg`` with the perplexity penalty enabled."""
    return dataclasses.replace(base_cfg, perplexity_penalty=perplexity_penalty)


def adaptive_perplexity_loss(
    z,  # (L, |V|) Gumbel-softmax over suffix positions
    base_bundle: LoadedModel,
):
    """Two-model PPL proxy: forward the soft suffix through a base LM and
    read its mean per-token CE.

    The trick: instead of one-hotting and embedding-lookup, we matmul ``z``
    with the base LM's embedding matrix to get a *soft* embedding sequence.
    Then forward through the base LM and compute CE on the *most-likely
    next token under z's argmax* (a self-CE proxy). This stays
    differentiable through ``z``.

    Heavy: requires a second 7B forward pass per step. Use only for §12.3.
    """
    import torch
    import torch.nn.functional as F

    base_model = base_bundle.model
    base_emb = base_model.get_input_embeddings()
    device = next(base_model.parameters()).device

    z_local = z.to(device).to(base_emb.weight.dtype)
    emb_seq = z_local @ base_emb.weight  # (L, d)
    if base_bundle.tokenizer.bos_token_id is not None:
        bos_emb = base_emb(
            torch.tensor([base_bundle.tokenizer.bos_token_id], device=device)
        )
        emb_seq = torch.cat([bos_emb, emb_seq], dim=0)
    out = base_model(inputs_embeds=emb_seq.unsqueeze(0), use_cache=False)
    logits = out.logits[0, :-1]  # (L, V)
    target = z_local.argmax(dim=-1).detach()  # (L,)
    return F.cross_entropy(logits, target)
