"""Eval subpackage.

Each metric exposes a ``score(prompts, generations) -> EvalResult`` callable
so that the runner can apply any combination of metrics to one set of
generations. The HarmBench classifier is the primary metric; everything
else is reported in tables for cross-paper comparability.
"""
from .substring import score_substring  # noqa: F401
from .harmbench_cls import HarmBenchScorer  # noqa: F401
from .strongreject import StrongRejectScorer  # noqa: F401
from .llamaguard import LlamaGuardScorer  # noqa: F401
from .alpaca_judge import AlpacaPairwiseJudge  # noqa: F401
from .generate import generate_with_suffix, generate_replacement  # noqa: F401
from .runner import EvalRunner  # noqa: F401
