"""Common attack interface.

Every attack: implements ``run`` returning an ``AttackResult``. Sub-attacks
(e.g. universal vs. individual, multi-model joint) are configured via
``AttackConfig``, not by separate classes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from ..data.registry import Behavior
from ..utils.artifact import AttackResult


@dataclass
class AttackConfig:
    name: str = "noop"
    suffix_length: int = 20
    n_steps: int = 500
    batch_size: int = 256
    top_k: int = 256
    init_suffix: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"  # 20 bangs
    target_loss_threshold: float = 0.05
    seed: int = 0
    wall_clock_budget_s: float | None = None  # None = no budget enforcement

    # Universal vs individual
    universal: bool = False  # True: optimize one suffix over all behaviors

    # Multi-model joint (§7.3)
    extra_models: Sequence[str] = field(default_factory=tuple)

    # Activation-loss configuration (used by SA-GCG, AGCG)
    activation_scope: str = "layer"
    composite_lambda: float = 1.0
    direction_path: str | None = None  # path to a saved DirectionRecord

    # SA-GCG-specific
    schedule: str = "slushy"
    polish_steps: int = 50
    use_first_token_weight: bool = True
    first_token_weight: float = 5.0
    random_direction: bool = False  # for §10.2 random-direction control
    perplexity_penalty: float = 0.0  # for §12 adaptive defense

    # Misc
    verbose: bool = False


class AttackBase(ABC):
    name: str = "base"

    def __init__(self, model_bundle, config: AttackConfig):
        self.model_bundle = model_bundle
        self.config = config

    @abstractmethod
    def run(self, behaviors: Sequence[Behavior]) -> AttackResult:
        """Optimize a single suffix.

        For ``config.universal=True``, ``behaviors`` is the training pool (≥1).
        For per-behavior attacks, the caller invokes ``run`` once per behavior.
        """
        raise NotImplementedError
