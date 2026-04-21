"""B0: No-suffix baseline."""
from __future__ import annotations

import time

from ..utils.artifact import AttackResult
from .base import AttackBase
from .registry import register


@register("no_suffix")
class NoSuffix(AttackBase):
    """Returns an empty suffix; the eval scripts treat this as a vanilla forward."""

    def run(self, behaviors):
        t0 = time.time()
        return AttackResult(
            suffix_ids=[],
            suffix_str="",
            wall_clock_s=time.time() - t0,
            n_steps=0,
            meta={"attack": self.name},
        )
