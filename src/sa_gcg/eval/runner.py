"""Evaluation runner: take a (suffix, behaviors) pair and produce all
configured ASR metrics. Caches generations so each metric reuses them.

The runner is metric-set-agnostic: the caller passes a list of scorers
(``substring`` is a function, the others are dataclasses with ``.score()``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..data.registry import Behavior
from ..models.load import LoadedModel
from ..utils.artifact import EvalResult
from ..utils.logging import get_logger
from .generate import generate_replacement, generate_with_suffix
from .substring import score_substring

LOG = get_logger(__name__)


# A scorer is a callable returning EvalResult given (prompts, generations).
# StrongRejectScorer / HarmBenchScorer / LlamaGuardScorer instances all
# satisfy this via their ``.score`` bound method.
ScoreFn = Callable[[Sequence[str], Sequence[str]], EvalResult]


@dataclass
class EvalRunner:
    bundle: LoadedModel
    scorers: dict[str, ScoreFn] = field(default_factory=dict)
    max_new_tokens: int = 256
    temperature: float = 0.0
    system: Optional[str] = None
    batch_size: int = 4
    is_full_prompt: bool = False  # True for PAIR (suffix replaces user msg)
    hook_handles: Optional[list] = None

    def run(
        self, behaviors: Sequence[Behavior], suffix_str: str
    ) -> dict[str, EvalResult]:
        """Generate once, score with every metric. Returns one EvalResult per
        scorer name; substring is always added as a baseline.
        """
        prompts = [b.prompt for b in behaviors]
        if self.is_full_prompt:
            gens = generate_replacement(
                self.bundle,
                full_user_prompts=[suffix_str or p for p in prompts] if suffix_str else prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                system=self.system,
                batch_size=self.batch_size,
            )
        else:
            gens = generate_with_suffix(
                self.bundle,
                user_messages=prompts,
                suffix_str=suffix_str,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                system=self.system,
                hook_handles=self.hook_handles,
                batch_size=self.batch_size,
            )

        results: dict[str, EvalResult] = {}
        # Substring is free; always include.
        results["substring"] = score_substring(gens)
        results["substring"].meta["generations"] = gens

        # Optional metrics
        for name, fn in self.scorers.items():
            try:
                # HarmBench wants (behaviors, generations); substring/SR want
                # (prompts, generations). For HarmBench we use the behavior
                # text as the "behavior"; for the others we pass prompts.
                if name == "harmbench_cls":
                    res = fn(prompts, gens)
                else:
                    res = fn(prompts, gens)
                results[name] = res
            except Exception as e:
                LOG.error("Scorer %s failed: %s", name, e)
                results[name] = EvalResult(score=float("nan"), meta={"error": str(e)})
        return results
