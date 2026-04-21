"""Mechanism subpackage: per-layer cosine curves, random-direction control,
extraction-layer sensitivity sweep. Directly supports EVAL_PLAN §10.
"""
from .cosine_curves import cosine_curves_for_suffix  # noqa: F401
from .layer_sweep import layer_sweep  # noqa: F401
