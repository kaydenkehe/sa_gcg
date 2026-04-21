"""Sanity tests for the stats subpackage."""
from __future__ import annotations

import math

from sa_gcg.stats.tests import (
    benjamini_hochberg,
    clustered_bootstrap_ci,
    paired_mcnemar,
    wilson_ci,
)


def test_mcnemar_no_difference():
    a = [True, False, True, False]
    b = [True, False, True, False]
    res = paired_mcnemar(a, b)
    assert res.b == res.c == 0
    assert res.p_value == 1.0


def test_mcnemar_one_sided_evidence():
    # A wins in 5 of 5 discordant trials.
    a = [True] * 5 + [True, True, True, True, True]
    b = [False] * 5 + [True, True, True, True, True]
    res = paired_mcnemar(a, b)
    assert res.b == 5
    assert res.c == 0
    # Two-sided p: 2 * (1/32) = 0.0625
    assert abs(res.p_value - 2 * (1 / 32)) < 1e-12


def test_bh_classic():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205]
    rej = benjamini_hochberg(p, alpha=0.05)
    # Largest i with p_(i) <= i/8 * 0.05:
    # i=1: 0.0062  | 0.001 <= 0.0062  ok
    # i=2: 0.0125  | 0.008 <= 0.0125  ok
    # i=3: 0.01875 | 0.039 > -> no
    # i=4: 0.025   | 0.041 > -> no
    # i=5: 0.03125 | 0.042 > -> no
    # so reject only the smallest two.
    assert rej == [True, True, False, False, False, False, False, False]


def test_wilson_ci_bounds():
    p, lo, hi = wilson_ci(50, 100, alpha=0.05)
    assert abs(p - 0.5) < 1e-9
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_clustered_bootstrap_ci_smoke():
    outcomes = [True, False, True, True, False, False, True, False]
    clusters = [0, 0, 0, 1, 1, 1, 2, 2]
    pt, lo, hi = clustered_bootstrap_ci(outcomes, clusters, n_resamples=500, seed=0)
    assert 0 <= lo <= pt <= hi <= 1
    assert math.isfinite(lo) and math.isfinite(hi)
