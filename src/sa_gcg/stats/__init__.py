"""Statistical analysis primitives (EVAL_PLAN §13).

  - paired_mcnemar : exact McNemar on paired binary outcomes
  - benjamini_hochberg : BH FDR correction on a set of p-values
  - clustered_bootstrap_ci : cluster-resampling CI for behavior-correlated data
  - wilson_ci : fallback CI for independent Bernoulli outcomes
"""
from .tests import (  # noqa: F401
    benjamini_hochberg,
    clustered_bootstrap_ci,
    paired_mcnemar,
    wilson_ci,
)
