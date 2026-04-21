"""Activation-projection loss (the core of Activation-Guided GCG / SA-GCG).

Three scopes, as defined in EVAL_PLAN.md Appendix F:

  - single:  <h[ell*, p*], r>^2
  - layer:   mean over p of <h[ell*, p], r>^2
  - global:  mean over (ell, p) of <h[ell, p], r>^2

The global scope requires intercepting every block, not just one; we handle
it by registering a reader on every decoder layer.
"""
from __future__ import annotations

from enum import Enum
from typing import Sequence


class ActivationScope(str, Enum):
    SINGLE = "single"
    LAYER = "layer"
    GLOBAL = "global"


def activation_projection_loss(
    *,
    residual_by_layer: dict[int, object],
    refusal_direction: object,
    scope: ActivationScope | str,
    ell_star: int,
    p_star: int = -1,
):
    """Compute the activation-projection loss for a single forward pass.

    Parameters
    ----------
    residual_by_layer : maps layer index -> (batch, seq, d_model) tensor
        captured by ``ResidualReader``. At minimum the entry for ``ell_star``
        is required; ``global`` scope consumes all layers.
    refusal_direction : (d_model,) unit tensor on same device/dtype as
        residuals.
    scope : which positions/layers to penalize.
    ell_star, p_star : extraction layer / position.
    """
    import torch

    if isinstance(scope, str):
        scope = ActivationScope(scope)

    r = refusal_direction

    if scope is ActivationScope.SINGLE:
        h = residual_by_layer[ell_star][:, p_star, :]  # (B, d)
        proj = (h @ r) ** 2  # (B,)
        return proj.mean()

    if scope is ActivationScope.LAYER:
        h = residual_by_layer[ell_star]  # (B, T, d)
        proj = (h @ r) ** 2  # (B, T)
        return proj.mean()

    if scope is ActivationScope.GLOBAL:
        if not residual_by_layer:
            raise ValueError("global scope requires residuals from every layer")
        projs = []
        for ell, h in residual_by_layer.items():
            projs.append(((h @ r) ** 2).mean())
        return torch.stack(projs).mean()

    raise ValueError(f"Unknown scope {scope!r}")


def all_layer_cosines(
    residual_by_layer: dict[int, object],
    direction: object,
    *,
    position: int = -1,
):
    """Per-layer cosine similarity at a fixed token position.

    Used for RQ3 mechanistic plot (Arditi Figure 5 analog). Returns a dict
    mapping layer -> float. Expects residuals and direction on CPU or same
    device; detaches before computing cosine.
    """
    import torch

    out = {}
    d_norm = direction / direction.norm().clamp_min(1e-8)
    for ell, h in residual_by_layer.items():
        h_pos = h[:, position, :].detach().to(d_norm.dtype)
        h_norm = h_pos / h_pos.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out[int(ell)] = float((h_norm @ d_norm.to(h_norm.device)).mean().item())
    return out


def register_all_layer_readers(layers_module) -> dict[int, object]:
    """Return a dict of ``ResidualReader`` objects, one per decoder layer.

    Caller is responsible for ``__enter__`` / ``__exit__`` on each; or use
    ``AllLayerReaders`` below as a context manager.
    """
    from ..models.hooks import ResidualReader

    return {i: ResidualReader(layer) for i, layer in enumerate(layers_module)}


class AllLayerReaders:
    """Context manager that captures residuals at every decoder block.

    Use with the ``global`` activation scope or with RQ3 cosine curves.
    """

    def __init__(self, layers_module, *, layers: Sequence[int] | None = None):
        from ..models.hooks import ResidualReader

        indices = list(layers) if layers is not None else list(range(len(layers_module)))
        self._readers = {i: ResidualReader(layers_module[i]) for i in indices}

    def __enter__(self) -> dict[int, object]:
        for r in self._readers.values():
            r.__enter__()
        return {i: r for i, r in self._readers.items()}

    def __exit__(self, *exc) -> None:
        for r in self._readers.values():
            r.__exit__(*exc)

    @property
    def tensors(self) -> dict[int, object]:
        return {i: r.tensor for i, r in self._readers.items()}
