"""SmoothLLM character perturbation correctness (no model load)."""
from __future__ import annotations

import random

from sa_gcg.defenses.smoothllm import _perturb


def test_swap_changes_at_most_k_chars():
    rng = random.Random(0)
    s = "abcdefghijklmnop"  # 16 chars
    out = _perturb(s, "swap", p=0.25, rng=rng)
    diffs = sum(1 for a, b in zip(s, out) if a != b)
    # Up to ceil(16 * 0.25) = 4 swap operations; some may collide on the
    # same position, so ``diffs`` <= 4.
    assert len(out) == len(s)
    assert diffs <= 4


def test_insert_grows_string():
    rng = random.Random(0)
    s = "abcdefghij"
    out = _perturb(s, "insert", p=0.3, rng=rng)
    assert len(out) > len(s)


def test_patch_replaces_contiguous_chunk():
    rng = random.Random(0)
    s = "abcdefghijklmnop"
    out = _perturb(s, "patch", p=0.25, rng=rng)
    assert len(out) == len(s)
