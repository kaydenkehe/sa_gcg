"""EVAL_PLAN §10.1: per-layer cosine of the residual stream to the refusal
direction, averaged over the held-out eval set. Produces Figure 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..data.registry import Behavior
from ..losses.activation_loss import AllLayerReaders, all_layer_cosines
from ..models.chat_template import build_input_ids
from ..models.load import LoadedModel
from ..utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass
class CosineCurveRecord:
    per_layer_mean: dict[int, float]
    per_layer_sem: dict[int, float]
    n_behaviors: int
    suffix_str: str
    model_name: str


def cosine_curves_for_suffix(
    bundle: LoadedModel,
    behaviors: Sequence[Behavior],
    suffix_str: str,
    direction,  # torch.Tensor (d_model,)
    *,
    position: int = -1,
    system: Optional[str] = None,
) -> CosineCurveRecord:
    """Average per-layer cosine over ``behaviors`` with the given suffix.

    The measurement is at the final prompt token (before generation), which
    is what the activation-projection loss was trained to minimize.
    """
    import math

    import torch

    model, tok, spec = bundle.model, bundle.tokenizer, bundle.spec
    layers_module = model.model.layers if hasattr(model, "model") else model.layers
    num_layers = len(layers_module)

    d = direction.to(next(model.parameters()).device).to(torch.float32)

    # Running sums for mean + SEM across behaviors, per layer.
    layer_ids = list(range(num_layers))
    sums = {l: 0.0 for l in layer_ids}
    sqs = {l: 0.0 for l in layer_ids}
    n = 0

    for beh in behaviors:
        slot = build_input_ids(tok, spec, beh.prompt, suffix_str, system=system)
        input_ids = torch.tensor([slot.input_ids], device=model.device)
        with AllLayerReaders(layers_module) as readers, torch.no_grad():
            model(input_ids=input_ids)
            layer_tensors = {i: r.tensor for i, r in readers.items()}
        per_layer = all_layer_cosines(layer_tensors, d, position=position)
        for l, c in per_layer.items():
            sums[l] += c
            sqs[l] += c * c
        n += 1

    mean = {l: sums[l] / max(1, n) for l in layer_ids}
    var = {l: max(0.0, sqs[l] / max(1, n) - mean[l] ** 2) for l in layer_ids}
    sem = {l: math.sqrt(var[l] / max(1, n)) for l in layer_ids}

    return CosineCurveRecord(
        per_layer_mean=mean,
        per_layer_sem=sem,
        n_behaviors=n,
        suffix_str=suffix_str,
        model_name=bundle.name_or_path,
    )
