"""Perplexity defense (Jain et al. 2023) and the threshold-calibrated filter.

Approach:
  - Compute mean per-token negative log-likelihood of the suffix under a
    *base* (not chat) Llama-2-7B. This gives PPL-like score.
  - Threshold = 99th percentile of mean-token-NLL on Alpaca *assistant
    outputs* (the cleanest "natural language" reference distribution).
  - Filter rejects suffixes with NLL above the threshold.

We separate the *score* (per-suffix NLL) from the *threshold* so the same
calibration set powers both the GCG / SA-GCG / AutoDAN bypass measurements
and the adaptive defense.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..models.load import LoadedModel
from ..utils.logging import get_logger

LOG = get_logger(__name__)


def suffix_perplexity(
    base_bundle: LoadedModel,
    suffix_str: str,
    *,
    add_bos: bool = True,
) -> float:
    """Mean per-token cross-entropy of ``suffix_str`` under ``base_bundle``."""
    import torch

    tok = base_bundle.tokenizer
    model = base_bundle.model
    device = next(model.parameters()).device
    ids = tok(suffix_str, add_special_tokens=False).input_ids
    if not ids:
        return float("inf")
    if add_bos and tok.bos_token_id is not None:
        ids = [tok.bos_token_id] + ids
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[:, :-1]
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean().item()
    return float(nll)


@dataclass
class PerplexityFilter:
    threshold_nll: float
    base_bundle: LoadedModel

    def passes(self, suffix_str: str) -> bool:
        return suffix_perplexity(self.base_bundle, suffix_str) <= self.threshold_nll


def calibrate_threshold(
    base_bundle: LoadedModel,
    reference_texts: Iterable[str],
    *,
    quantile: float = 0.99,
) -> float:
    """Compute ``quantile`` of per-token mean NLL over ``reference_texts``."""
    nlls = [suffix_perplexity(base_bundle, t) for t in reference_texts]
    nlls = [n for n in nlls if math.isfinite(n)]
    if not nlls:
        raise RuntimeError("PerplexityFilter: empty reference set after filtering.")
    nlls.sort()
    idx = max(0, min(len(nlls) - 1, int(quantile * (len(nlls) - 1))))
    return nlls[idx]


def perplexity_filter(
    base_bundle: LoadedModel,
    suffixes: Iterable[str],
    threshold_nll: float,
) -> list[bool]:
    return [suffix_perplexity(base_bundle, s) <= threshold_nll for s in suffixes]
