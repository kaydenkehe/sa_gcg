"""Open transfer: append the suffix to each prompt, query a held-out HF
model, score with the same eval pipeline.

Per EVAL_PLAN §9.1, this is the path used for the four open targets:
Vicuna-7B-v1.5, Llama-2-13B, Llama-3-8B, Mistral-7B-Instruct-v0.2.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..data.registry import Behavior
from ..eval.runner import EvalRunner
from ..eval.harmbench_cls import HarmBenchScorer
from ..eval.strongreject import StrongRejectScorer
from ..models.load import load_chat_model
from ..utils.artifact import EvalResult
from ..utils.logging import get_logger

LOG = get_logger(__name__)


def run_open_transfer(
    target_name_or_path: str,
    behaviors: Sequence[Behavior],
    suffix_str: str,
    *,
    metric_set: Iterable[str] = ("substring", "harmbench_cls", "strongreject"),
    max_new_tokens: int = 256,
    system: Optional[str] = None,
    dtype: str = "fp16",
    device: str = "cuda",
    harmbench_scorer: Optional[HarmBenchScorer] = None,
    strongreject_scorer: Optional[StrongRejectScorer] = None,
    is_full_prompt: bool = False,
) -> dict[str, EvalResult]:
    """Load the target, run generation+scoring, free the target."""
    LOG.info("Open transfer to %s (n=%d behaviors)", target_name_or_path, len(behaviors))
    target = load_chat_model(target_name_or_path, dtype=dtype, device=device)

    metric_set = set(metric_set)
    scorers: dict = {}
    if "harmbench_cls" in metric_set:
        scorers["harmbench_cls"] = (harmbench_scorer or HarmBenchScorer()).score
    if "strongreject" in metric_set:
        scorers["strongreject"] = (strongreject_scorer or StrongRejectScorer()).score

    runner = EvalRunner(
        bundle=target,
        scorers=scorers,
        max_new_tokens=max_new_tokens,
        system=system,
        is_full_prompt=is_full_prompt,
    )
    results = runner.run(behaviors, suffix_str)

    # Free target before next call (caller may iterate over many targets).
    try:
        import torch

        del target
        torch.cuda.empty_cache()
    except Exception:
        pass
    return results
