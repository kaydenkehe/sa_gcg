"""SA-GCG: Soft Activation-Guided GCG.

Composite-loss adversarial suffix attack combining the continuous-relaxation
optimizer of Soft-GCG with the activation-projection loss of Activation-Guided
GCG, both reduced to a discrete suffix via Gumbel-softmax annealing.

See ``EVAL_PLAN.md`` in the repository root for the experimental design.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sa_gcg")
except PackageNotFoundError:
    __version__ = "0.1.0+unknown"

__all__ = ["__version__"]
