"""Refusal-direction extraction (Arditi et al. 2024).

We compute the difference-in-means between mean residual-stream activation on
harmful vs. harmless prompts at every (layer, position), then *select* the
single best (layer, position) by Arditi's bypass / induce / KL screen.

For our experiments we usually need just the selected ``(ell_star, p_star)``
direction, but the full per-layer dictionary is saved so downstream
mechanism analysis can plot cosine across all layers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..models.chat_template import ChatSpec, build_input_ids
from ..models.hooks import ResidualReader
from ..utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass
class DirectionRecord:
    ell_star: int
    p_star: int
    direction: object  # torch.Tensor (d_model,)
    per_layer: dict[int, object] | None  # torch.Tensor (d_model,) per layer
    selection_scores: dict[str, float]
    extraction_meta: dict


def extract_refusal_direction(
    model_bundle,
    *,
    harmful_prompts: Sequence[str],
    harmless_prompts: Sequence[str],
    candidate_positions: Sequence[int] = (-1,),
    candidate_layers: Sequence[int] | None = None,
    bypass_eval_fn=None,
    induce_eval_fn=None,
    kl_eval_fn=None,
    kl_threshold: float = 0.1,
    max_layer_frac: float = 0.8,
) -> DirectionRecord:
    """Compute refusal direction(s) and select one by Arditi's screen.

    Parameters
    ----------
    model_bundle : LoadedModel (from sa_gcg.models.load)
    harmful_prompts, harmless_prompts : matched-size lists of bare instructions.
    candidate_positions : token positions to consider (negative indexes ok);
        Arditi uses just ``-1`` (last token) for chat models.
    candidate_layers : layers to consider; default = all.
    bypass_eval_fn, induce_eval_fn, kl_eval_fn : callables(direction, layer)
        returning a scalar score. If ``None``, selection falls back to "lowest
        layer with max mean cosine sim on harmful prompts" (a sensible default
        when you can't afford the full screen).
    kl_threshold : reject directions whose KL on harmless prompts exceeds this.
    max_layer_frac : reject directions in layers above ``frac * num_layers``
        (Arditi: deep-layer directions tend to overfit).
    """
    import torch

    model = model_bundle.model
    tok = model_bundle.tokenizer
    spec: ChatSpec = model_bundle.spec
    layers_module = model.model.layers if hasattr(model, "model") else model.layers
    num_layers = len(layers_module)
    if candidate_layers is None:
        candidate_layers = list(range(num_layers))

    LOG.info(
        "Extracting refusal direction: %d harmful, %d harmless, %d layers, %d positions",
        len(harmful_prompts),
        len(harmless_prompts),
        len(candidate_layers),
        len(candidate_positions),
    )

    pos = candidate_positions[0]
    sums_h = {ell: torch.zeros(model.config.hidden_size) for ell in candidate_layers}
    sums_n = {ell: torch.zeros(model.config.hidden_size) for ell in candidate_layers}

    def collect_into(prompts, sums):
        for p in prompts:
            slot = build_input_ids(tok, spec, p, suffix_str="")
            input_ids = torch.tensor([slot.input_ids], device=model.device)
            for ell in candidate_layers:
                with ResidualReader(layers_module[ell]) as reader, torch.no_grad():
                    model(input_ids=input_ids)
                sums[ell] = sums[ell] + reader.tensor[0, pos].detach().cpu().to(torch.float32)

    collect_into(harmful_prompts, sums_h)
    collect_into(harmless_prompts, sums_n)

    directions: dict[int, object] = {}
    for ell in candidate_layers:
        diff = (sums_h[ell] / len(harmful_prompts)) - (sums_n[ell] / len(harmless_prompts))
        norm = diff.norm().clamp_min(1e-8)
        directions[ell] = (diff / norm).to(torch.float32)

    # Selection: if no eval callables provided, pick by max mean projection on
    # harmful prompts (a cheap heuristic that agrees with Arditi for Llama-2-7B
    # in our tests). The CLI extractor exposes the full screen.
    if bypass_eval_fn is None:
        scores = {ell: float(_mean_proj_on_prompts(model, layers_module, ell, directions[ell], tok, spec, harmful_prompts)) for ell in candidate_layers}
        max_layer = int(max_layer_frac * num_layers)
        ranked = sorted(
            (ell for ell in candidate_layers if ell < max_layer),
            key=lambda ell: -scores[ell],
        )
        ell_star = ranked[0]
        sel_scores = {"mean_proj_harmful": scores[ell_star]}
    else:
        # Full screen path.
        screen: list[tuple[int, float, float, float, float]] = []
        for ell in candidate_layers:
            if ell >= max_layer_frac * num_layers:
                continue
            d = directions[ell]
            b = float(bypass_eval_fn(d, ell))  # lower = better (bypass)
            i_ = float(induce_eval_fn(d, ell))  # higher = better
            kl = float(kl_eval_fn(d, ell))
            if kl > kl_threshold:
                continue
            score = i_ - b  # both directions of merit
            screen.append((ell, b, i_, kl, score))
        if not screen:
            raise RuntimeError("All candidate layers rejected by KL filter.")
        screen.sort(key=lambda r: -r[-1])
        ell_star = screen[0][0]
        sel_scores = {
            "bypass_score": screen[0][1],
            "induce_score": screen[0][2],
            "kl_score": screen[0][3],
        }

    return DirectionRecord(
        ell_star=ell_star,
        p_star=candidate_positions[0],
        direction=directions[ell_star],
        per_layer=directions,
        selection_scores=sel_scores,
        extraction_meta={
            "model": model_bundle.name_or_path,
            "n_harmful": len(harmful_prompts),
            "n_harmless": len(harmless_prompts),
            "candidate_positions": list(candidate_positions),
            "kl_threshold": kl_threshold,
            "max_layer_frac": max_layer_frac,
        },
    )


def _mean_proj_on_prompts(model, layers_module, ell, direction, tok, spec, prompts):
    import torch

    s = 0.0
    for p in prompts:
        slot = build_input_ids(tok, spec, p, suffix_str="")
        input_ids = torch.tensor([slot.input_ids], device=model.device)
        with ResidualReader(layers_module[ell]) as reader, torch.no_grad():
            model(input_ids=input_ids)
        h = reader.tensor[0, -1].detach().cpu().to(torch.float32)
        s += float((h @ direction).item())
    return s / max(1, len(prompts))


def save_direction(record: DirectionRecord, path: str | Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ell_star": record.ell_star,
        "p_star": record.p_star,
        "direction": record.direction,
        "per_layer": record.per_layer,
        "selection_scores": record.selection_scores,
        "extraction_meta": record.extraction_meta,
    }
    torch.save(payload, path)
    with path.with_suffix(".json").open("w") as f:
        json.dump(
            {
                "ell_star": record.ell_star,
                "p_star": record.p_star,
                "selection_scores": record.selection_scores,
                "extraction_meta": record.extraction_meta,
            },
            f,
            indent=2,
        )


def load_direction(path: str | Path) -> DirectionRecord:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return DirectionRecord(
        ell_star=int(payload["ell_star"]),
        p_star=int(payload["p_star"]),
        direction=payload["direction"],
        per_layer=payload.get("per_layer"),
        selection_scores=payload.get("selection_scores", {}),
        extraction_meta=payload.get("extraction_meta", {}),
    )
