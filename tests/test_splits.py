"""Split disjointness checks."""
from __future__ import annotations

import pytest

from sa_gcg.data.registry import Behavior
from sa_gcg.data.splits import VerifySplitError, verify_disjoint


def _b(p):
    return Behavior(id="x", prompt=p, target_prefix="t", source="advbench")


def test_disjoint_passes():
    a = [_b("alpha"), _b("beta")]
    b = [_b("gamma")]
    verify_disjoint(a, b)


def test_overlap_raises():
    a = [_b("hello world"), _b("alpha")]
    b = [_b("Hello   world"), _b("zeta")]  # case + whitespace insensitive
    with pytest.raises(VerifySplitError):
        verify_disjoint(a, b)


def test_strings_or_behaviors():
    verify_disjoint(["a", "b"], ["c"])
    with pytest.raises(VerifySplitError):
        verify_disjoint(["A", "b"], ["a"])
