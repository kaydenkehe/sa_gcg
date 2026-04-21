"""Disjointness checks between training, evaluation, and direction-extraction sets.

Used to fail-fast in CI per EVAL_PLAN.md §15 (Implementation Notes).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

from .registry import Behavior


class VerifySplitError(AssertionError):
    """Raised when two prompt sets are not disjoint."""


def verify_disjoint(
    a: Sequence[Behavior] | Iterable[str],
    b: Sequence[Behavior] | Iterable[str],
    *,
    a_name: str = "A",
    b_name: str = "B",
) -> None:
    """Assert no prompt appears in both ``a`` and ``b`` (case-insensitive,
    whitespace-normalized).
    """
    set_a = _norm_set(a)
    set_b = _norm_set(b)
    overlap = set_a & set_b
    if overlap:
        sample = list(overlap)[:5]
        raise VerifySplitError(
            f"{a_name} and {b_name} overlap on {len(overlap)} prompts. "
            f"First 5: {sample}"
        )


def _norm_set(seq: Sequence[Behavior] | Iterable[str]) -> set[str]:
    return {_norm(_prompt(x)) for x in seq}


def _prompt(x: Behavior | str) -> str:
    return x.prompt if isinstance(x, Behavior) else x


def _norm(s: str) -> str:
    return " ".join(s.lower().split())
