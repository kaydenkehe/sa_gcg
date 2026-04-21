"""``sa-gcg-mechanism`` — produce Figure 2 + the random-direction control table.

Inputs: a saved direction (``--direction-path``) and a suffix
(``--suffix-from``). Computes per-layer cosine curves on a held-out
behavior set, writes:

  ``cosine_curve.json``  (per-layer mean + sem)

For the random-direction control the user runs ``sa-gcg-attack
--random-direction`` separately and points this script at the resulting
suffix; the cosine curves themselves don't change because the *direction*
is what changed in the attack, not the eval direction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.registry import load_behaviors
from ..direction.extractor import load_direction
from ..mechanism.cosine_curves import cosine_curves_for_suffix
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from ._common import add_dataset_args, add_model_args, ensure_dir, load_model_from_args

LOG = get_logger("sa_gcg.cli.run_mechanism")


def _load_suffix(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        p = p / "suffix.json"
    return json.loads(p.read_text()).get("str", "")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sa-gcg-mechanism")
    add_model_args(p)
    add_dataset_args(p)
    p.add_argument("--direction-path", required=True, type=str)
    p.add_argument("--suffix-from", required=True, type=str)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = _build_parser().parse_args()
    set_seed(args.seed)
    bundle = load_model_from_args(args)
    behaviors = load_behaviors(
        args.dataset, data_root=args.data_root, limit=args.limit, offset=args.offset
    )
    suffix_str = _load_suffix(args.suffix_from)
    record = load_direction(args.direction_path)
    LOG.info(
        "Computing per-layer cosine curves: %d behaviors, ell_star=%d",
        len(behaviors), record.ell_star,
    )
    rec = cosine_curves_for_suffix(
        bundle, behaviors, suffix_str, record.direction, position=args.position,
    )
    out = ensure_dir(args.out_dir)
    with (out / "cosine_curve.json").open("w") as f:
        json.dump(
            {
                "per_layer_mean": rec.per_layer_mean,
                "per_layer_sem": rec.per_layer_sem,
                "n_behaviors": rec.n_behaviors,
                "model": rec.model_name,
                "ell_star": record.ell_star,
                "p_star": record.p_star,
                "suffix_str": rec.suffix_str,
            },
            f, indent=2,
        )
    LOG.info("Wrote %s", out / "cosine_curve.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
