"""Statistical tests for RQ1/RQ2 comparisons.

No SciPy dependency — we compute the exact McNemar p-value via the
binomial CDF (scipy is absent in the lean requirement). The rest uses
numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class McNemarResult:
    b: int   # outcomes where A=1, B=0 (A better)
    c: int   # outcomes where A=0, B=1 (B better)
    p_value: float
    effect: float  # (b - c) / (b + c), the direction of improvement


def paired_mcnemar(a: Sequence[bool], b: Sequence[bool]) -> McNemarResult:
    """Exact paired McNemar test on two aligned 0/1 outcome vectors.

    Two-sided. Null = Pr(A > B) = Pr(B > A). Uses the binomial exact test
    on ``b`` successes out of ``b + c`` discordant trials.
    """
    if len(a) != len(b):
        raise ValueError("paired_mcnemar: length mismatch")
    b_cnt = sum(1 for x, y in zip(a, b) if x and not y)
    c_cnt = sum(1 for x, y in zip(a, b) if y and not x)
    n = b_cnt + c_cnt
    if n == 0:
        return McNemarResult(b=0, c=0, p_value=1.0, effect=0.0)
    k = min(b_cnt, c_cnt)
    # Two-sided exact binomial: 2 * min(1, sum Bin(n,0.5) <= k).
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    p_two = min(1.0, 2.0 * p)
    return McNemarResult(
        b=b_cnt, c=c_cnt, p_value=p_two, effect=(b_cnt - c_cnt) / max(1, n)
    )


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Return ``reject[i]`` for BH FDR control at level ``alpha``.

    Classical BH: sort ascending, find largest i such that p_{(i)} <= i/m * alpha,
    reject all tests up to and including that rank.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    ranked = [p_values[i] for i in order]
    threshold_rank = -1
    for i, p in enumerate(ranked, start=1):
        if p <= (i / m) * alpha:
            threshold_rank = i
    reject_sorted = [i <= threshold_rank for i in range(1, m + 1)]
    out = [False] * m
    for i, o in enumerate(order):
        out[o] = reject_sorted[i]
    return out


def clustered_bootstrap_ci(
    outcomes: Sequence[bool],
    cluster_ids: Sequence[int],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Cluster bootstrap CI for a proportion.

    Cluster resampling re-samples clusters (behaviors) with replacement;
    within each resampled cluster all outcomes are kept. Returns
    (point_estimate, lo, hi) at 1-alpha coverage.
    """
    import random
    from statistics import mean

    if len(outcomes) != len(cluster_ids):
        raise ValueError("clustered_bootstrap_ci: length mismatch")
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = sum(outcomes) / n

    by_cluster: dict[int, list[bool]] = {}
    for o, c in zip(outcomes, cluster_ids):
        by_cluster.setdefault(c, []).append(bool(o))
    cluster_list = list(by_cluster.values())
    K = len(cluster_list)

    rng = random.Random(seed)
    sims: list[float] = []
    for _ in range(n_resamples):
        total = 0
        n_total = 0
        for _k in range(K):
            pick = cluster_list[rng.randrange(0, K)]
            total += sum(pick)
            n_total += len(pick)
        sims.append(total / max(1, n_total))
    sims.sort()
    lo = sims[int(alpha / 2 * n_resamples)]
    hi = sims[int((1 - alpha / 2) * n_resamples) - 1]
    return point, lo, hi


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score CI for a proportion. Only valid when outcomes are
    approximately independent — use clustered bootstrap for the main
    comparisons.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    z = _z_from_alpha(alpha)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _z_from_alpha(alpha: float) -> float:
    # Hardcoded common levels (scipy-free).
    TABLE = {
        0.10: 1.6449,
        0.05: 1.9600,
        0.01: 2.5758,
    }
    key = min(TABLE, key=lambda k: abs(k - alpha))
    return TABLE[key]
