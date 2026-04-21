"""Cross-entropy on the target prefix.

For each prompt we score the log-probability of the ``target_ids`` sequence
appearing immediately after the templated-prompt tokens. This is the
classical GCG loss; we reimplement it on top of ``inputs_embeds`` so the
same path handles both discrete (GCG) and continuous (Soft-GCG, SA-GCG)
suffix sources.
"""
from __future__ import annotations

from typing import Optional


def target_ce_loss(
    model,
    *,
    inputs_embeds,
    target_ids,
    prompt_len: int,
    attention_mask=None,
    reduction: str = "mean",
):
    """Compute CE over target positions only.

    Parameters
    ----------
    model : HF causal LM whose forward accepts ``inputs_embeds``.
    inputs_embeds : (batch, seq, d_model) float tensor. Assumed to *already
        include* the target tokens' embeddings at positions
        ``prompt_len .. prompt_len + target_len - 1``.
    target_ids : (batch, target_len) int tensor of the desired next-token ids.
    prompt_len : position of the first target token in the sequence.
    """
    import torch
    import torch.nn.functional as F

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (batch, seq, vocab)
    # Predict token t from position t-1.
    target_len = target_ids.shape[1]
    pred_slice = logits[:, prompt_len - 1 : prompt_len - 1 + target_len, :]
    # flatten for F.cross_entropy
    loss = F.cross_entropy(
        pred_slice.reshape(-1, pred_slice.size(-1)),
        target_ids.reshape(-1),
        reduction=reduction,
    )
    return loss


def first_token_weighted_ce(
    model,
    *,
    inputs_embeds,
    target_ids,
    prompt_len: int,
    first_token_weight: float = 5.0,
    attention_mask: Optional[object] = None,
):
    """Carlini-Wagner style first-token-heavy CE.

    Motivation: in jailbreak attacks, getting the model to produce the very
    first affirmative token (``Sure``) is disproportionately hard and
    predictive of success. Upweighting it is a common trick from the GCG
    follow-up literature and from the soft-GCG codebase.
    """
    import torch
    import torch.nn.functional as F

    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    target_len = target_ids.shape[1]
    pred_slice = logits[:, prompt_len - 1 : prompt_len - 1 + target_len, :]
    per_pos = F.cross_entropy(
        pred_slice.reshape(-1, pred_slice.size(-1)),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(target_ids.shape)
    weights = torch.ones_like(per_pos)
    weights[:, 0] = first_token_weight
    return (per_pos * weights).sum() / weights.sum()
