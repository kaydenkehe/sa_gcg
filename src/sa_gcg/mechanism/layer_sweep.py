"""EVAL_PLAN §10.3: sensitivity of SA-GCG to the choice of ``ell_star``.

For each ``ell_star`` in a grid, rerun SA-GCG (headline cell), measure
ASR on held-out behaviors, and return the (layer, ASR) table.

This is thin glue over the attack builder + eval runner; the actual
compute is in the attack and the classifier.
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from ..attack.base import AttackConfig
from ..attack.registry import build_attack
from ..data.registry import Behavior
from ..direction.extractor import DirectionRecord, save_direction
from ..eval.runner import EvalRunner
from ..models.load import LoadedModel
from ..utils.logging import get_logger

LOG = get_logger(__name__)


def layer_sweep(
    bundle: LoadedModel,
    behaviors_train: Sequence[Behavior],
    behaviors_eval: Sequence[Behavior],
    direction_record: DirectionRecord,
    candidate_layers: Iterable[int],
    base_cfg: AttackConfig,
    *,
    scorers: dict | None = None,
    attack_name: str = "sa_gcg",
) -> list[dict]:
    """Run an attack once per layer; return [{layer, asr, suffix}, ...]."""
    scorers = scorers or {}
    rows: list[dict] = []
    runner = EvalRunner(bundle=bundle, scorers=scorers)

    for ell in candidate_layers:
        LOG.info("Layer sweep: ell_star = %d", ell)
        swept = copy.deepcopy(direction_record)
        swept.ell_star = int(ell)
        swept.direction = direction_record.per_layer[int(ell)] if direction_record.per_layer else direction_record.direction

        with tempfile.TemporaryDirectory() as tmp:
            dir_path = Path(tmp) / "dir.pt"
            save_direction(swept, dir_path)
            cfg = copy.deepcopy(base_cfg)
            cfg.direction_path = str(dir_path)

            attack = build_attack(attack_name, bundle, cfg)
            result = attack.run(behaviors_train)

        eval_results = runner.run(behaviors_eval, result.suffix_str)
        rows.append({
            "ell_star": int(ell),
            "suffix_str": result.suffix_str,
            "metrics": {k: v.score for k, v in eval_results.items()},
        })
    return rows
