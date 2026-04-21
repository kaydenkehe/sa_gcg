"""Forward-pre-hook utility for capturing residual-stream activations.

EVAL_PLAN.md §14.2: we use a forward *pre*-hook on ``model.model.layers[ell]``
because we want the input to the block (= residual stream after block ell-1
+ pre-RMSNorm), which is the canonical place to extract the refusal direction
per Arditi et al. 2024.
"""
from __future__ import annotations

from typing import Any


class ResidualReader:
    """Context manager that exposes ``self.tensor`` after a forward pass.

    Usage::

        reader = ResidualReader(model.model.layers[14])
        with reader:
            model(input_ids=...)
        h = reader.tensor   # (batch, seq, d_model)
    """

    def __init__(self, layer):
        self.layer = layer
        self._handle = None
        self.tensor = None

    def __enter__(self) -> "ResidualReader":
        self._handle = self.layer.register_forward_pre_hook(self._hook, with_kwargs=False)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, args) -> Any:
        # The first positional arg to a Llama decoder block is ``hidden_states``.
        self.tensor = args[0]
        return None  # don't modify
