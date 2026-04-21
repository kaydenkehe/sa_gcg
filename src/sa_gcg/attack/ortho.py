"""B6: Ortho — weight orthogonalization (Arditi et al. 2024).

Not a prompt-level attack — it permanently edits the model's weights so the
refusal direction cannot be written by any block. We provide a hook-based
inference-time variant (directional ablation) so it can be used as a
baseline in our same eval pipeline without persisting to disk.

The actual edit:
  W_out' = W_out - r r^T W_out  for every weight matrix that writes to the
  residual stream (attention W_O and MLP down-proj). Equivalent to running
  the model with a forward hook that subtracts ``r r^T x`` from the residual
  stream after every block.
"""
from __future__ import annotations

import time
from typing import Sequence

from ..data.registry import Behavior
from ..direction.extractor import load_direction
from ..utils.artifact import AttackResult
from .base import AttackBase
from .registry import register


@register("ortho")
class Ortho(AttackBase):
    """Returns no suffix; the eval-side caller is expected to register the
    directional-ablation hooks listed in ``meta`` before generation.
    """

    def run(self, behaviors: Sequence[Behavior]) -> AttackResult:
        if self.config.direction_path is None:
            raise ValueError("Ortho requires --direction-path")
        rec = load_direction(self.config.direction_path)
        return AttackResult(
            suffix_ids=[],
            suffix_str="",
            wall_clock_s=0.0,
            n_steps=0,
            meta={
                "attack": self.name,
                "applies_at": "generation_time",
                "intervention": "directional_ablation",
                "ell_star": rec.ell_star,
                "p_star": rec.p_star,
                "direction_path": self.config.direction_path,
            },
        )


def directional_ablation_hooks(model, direction):
    """Attach hooks that subtract the projection onto ``direction`` from the
    residual stream after every decoder block. Returns a list of removable
    hook handles.

    ``direction`` should be a (d_model,) unit vector on the model's device.
    """
    import torch

    layers = model.model.layers if hasattr(model, "model") else model.layers
    handles = []
    d = direction.to(next(model.parameters()).device).to(next(model.parameters()).dtype)
    rrT = lambda x: x - (x @ d).unsqueeze(-1) * d

    def hook(module, args, output):
        if isinstance(output, tuple):
            hs = output[0]
            return (rrT(hs),) + output[1:]
        return rrT(output)

    for layer in layers:
        handles.append(layer.register_forward_hook(hook))
    return handles
