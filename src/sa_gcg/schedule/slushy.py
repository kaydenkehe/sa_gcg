"""Gumbel-softmax temperature schedules for SGCG/SA-GCG.

"Slushy" refers to the 3-phase schedule introduced in Soft-GCG: start hot
(``tau_warm``) for exploration, cool linearly to ``tau_anneal`` for most
steps, then snap to ``tau_final`` (near-zero) at the tail so the argmax of
phi agrees with the soft-forward argmax.

The linear schedule is retained for ablation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LinearSchedule:
    tau_0: float = 2.0
    tau_1: float = 0.1
    n_steps: int = 500

    def __call__(self, step: int) -> float:
        t = step / max(1, self.n_steps - 1)
        return float(self.tau_0 + (self.tau_1 - self.tau_0) * min(1.0, max(0.0, t)))


@dataclass
class SlushySchedule:
    tau_warm: float = 2.0
    tau_anneal: float = 0.3
    tau_final: float = 0.03
    warm_frac: float = 0.1
    anneal_frac: float = 0.75
    n_steps: int = 500

    def __call__(self, step: int) -> float:
        n = max(1, self.n_steps)
        t = step / n
        if t < self.warm_frac:
            return float(self.tau_warm)
        if t < self.warm_frac + self.anneal_frac:
            local = (t - self.warm_frac) / self.anneal_frac
            return float(self.tau_warm + (self.tau_anneal - self.tau_warm) * local)
        # tail: tau_anneal -> tau_final (final 15% by default)
        tail_frac = max(1e-6, 1.0 - self.warm_frac - self.anneal_frac)
        local = (t - self.warm_frac - self.anneal_frac) / tail_frac
        local = min(1.0, max(0.0, local))
        return float(self.tau_anneal + (self.tau_final - self.tau_anneal) * local)


def make_schedule(name: str, n_steps: int, **kwargs):
    if name == "slushy":
        return SlushySchedule(n_steps=n_steps, **kwargs)
    if name == "linear":
        return LinearSchedule(n_steps=n_steps, **kwargs)
    raise ValueError(f"Unknown schedule {name!r}. Options: slushy, linear.")
