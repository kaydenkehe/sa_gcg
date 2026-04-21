"""B2: SGCG-CE — Soft-GCG with cross-entropy only (no activation loss).

The plain continuous-relaxation baseline. SA-GCG with ``composite_lambda=0``
should reproduce this attack to within 1 pp (smoke test S4).
"""
from __future__ import annotations

from .base import AttackBase
from .registry import register
from .sa_gcg import SAGCG


@register("soft_gcg")
class SoftGCG(SAGCG):
    """Same code path as SA-GCG with the activation term zeroed out."""

    def __init__(self, model_bundle, config):
        # Force composite_lambda=0; preserve other config.
        config = _replace(config, composite_lambda=0.0, polish_steps=config.polish_steps)
        super().__init__(model_bundle, config)


def _replace(cfg, **updates):
    import dataclasses

    return dataclasses.replace(cfg, **updates)
