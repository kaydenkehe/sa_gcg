"""Single-pass GradNorm surrogates (Du et al. 2018, simplified).

Motivation (EVAL_PLAN.md §14.3): CE loss and the activation-projection loss
live on different natural scales; directly summing them means the larger one
dominates regardless of ``lambda``. We record the gradient norms of each
term w.r.t. ``phi`` at step 1 and use them as fixed divisors for the rest of
training. ``lambda = 1`` then becomes a *genuine* equal weighting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GradNormSurrogates:
    g_ce: float = 1.0
    g_act: float = 1.0
    frozen: bool = False

    def capture_if_first(
        self,
        *,
        phi,
        loss_ce,
        loss_act,
        eps: float = 1e-8,
    ) -> None:
        """On the first call, compute ||grad_phi L|| for each term separately."""
        import torch

        if self.frozen:
            return
        with torch.no_grad():
            g_ce = _grad_norm(loss_ce, phi)
            g_act = _grad_norm(loss_act, phi)
        self.g_ce = max(float(g_ce), eps)
        self.g_act = max(float(g_act), eps)
        self.frozen = True


def _grad_norm(loss, phi) -> float:
    import torch

    grads = torch.autograd.grad(
        loss, phi, retain_graph=True, create_graph=False, allow_unused=True
    )
    g = grads[0]
    if g is None:
        return 0.0
    return float(g.detach().norm().item())
