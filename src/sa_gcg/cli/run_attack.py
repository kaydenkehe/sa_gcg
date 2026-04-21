"""``sa-gcg-attack`` — run any registered attack and dump artifacts.

The single entry point for SA-GCG and all baselines (B0–B6). Selects the
attack via ``--attack`` and routes config through ``AttackConfig``.

Examples
--------

Universal SA-GCG (headline) on Llama-2-7B-Chat::

    sa-gcg-attack --attack sa_gcg --universal \
        --direction-path runs/direction/llama2.pt \
        --activation-scope layer --composite-lambda 1.0 \
        --polish-steps 50 --n-steps 500 \
        --dataset advbench --limit 25 \
        --out-dir runs/sa_gcg/llama2/seed_0 --seed 0

Individual GCG on one behavior::

    sa-gcg-attack --attack gcg --dataset advbench --limit 1 \
        --n-steps 500 --out-dir runs/gcg/llama2/beh_0 --seed 0
"""
from __future__ import annotations

import argparse

from ..attack.base import AttackConfig
from ..attack.registry import ATTACKS, build_attack
from ..data.registry import load_behaviors
from ..utils.artifact import ArtifactWriter
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from ._common import (
    add_dataset_args,
    add_model_args,
    add_run_args,
    ensure_dir,
    load_model_from_args,
)

LOG = get_logger("sa_gcg.cli.run_attack")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sa-gcg-attack", description="Run a single attack and write artifacts."
    )
    add_model_args(p)
    add_dataset_args(p)
    add_run_args(p)
    p.add_argument("--attack", required=True, choices=sorted(ATTACKS),
                   help="Attack name. SA-GCG is 'sa_gcg'.")
    p.add_argument("--universal", action="store_true",
                   help="One suffix over the entire (limited) behavior set.")
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--suffix-length", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--top-k", type=int, default=256)
    p.add_argument("--init-suffix", default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
    p.add_argument("--target-loss-threshold", type=float, default=0.05)
    p.add_argument("--wall-clock-budget-s", type=float, default=None)

    # Activation-loss / direction
    p.add_argument("--activation-scope", default="layer",
                   choices=["single", "layer", "global"])
    p.add_argument("--composite-lambda", type=float, default=1.0)
    p.add_argument("--direction-path", default=None,
                   help="Path to a saved DirectionRecord (sa-gcg-extract-direction).")

    # SA-GCG knobs
    p.add_argument("--schedule", default="slushy", choices=["slushy", "linear"])
    p.add_argument("--polish-steps", type=int, default=50)
    p.add_argument("--first-token-weight", type=float, default=5.0)
    p.add_argument("--no-first-token-weight", action="store_true")
    p.add_argument("--random-direction", action="store_true",
                   help="EVAL_PLAN §10.2 random-direction control.")
    p.add_argument("--perplexity-penalty", type=float, default=0.0,
                   help="EVAL_PLAN §12 adaptive defense.")
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    set_seed(args.seed)

    behaviors = load_behaviors(
        args.dataset, data_root=args.data_root, limit=args.limit, offset=args.offset
    )
    if not behaviors:
        raise RuntimeError("No behaviors loaded. Check --dataset / --limit / --offset.")
    LOG.info("%d behaviors loaded from %s", len(behaviors), args.dataset)

    bundle = load_model_from_args(args)

    cfg = AttackConfig(
        name=args.attack,
        suffix_length=args.suffix_length,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        top_k=args.top_k,
        init_suffix=args.init_suffix,
        target_loss_threshold=args.target_loss_threshold,
        seed=args.seed,
        wall_clock_budget_s=args.wall_clock_budget_s,
        universal=args.universal,
        activation_scope=args.activation_scope,
        composite_lambda=args.composite_lambda,
        direction_path=args.direction_path,
        schedule=args.schedule,
        polish_steps=args.polish_steps,
        use_first_token_weight=not args.no_first_token_weight,
        first_token_weight=args.first_token_weight,
        random_direction=args.random_direction,
        perplexity_penalty=args.perplexity_penalty,
        verbose=args.verbose,
    )

    attack = build_attack(args.attack, bundle, cfg)
    out = ensure_dir(args.out_dir)
    LOG.info("Running attack '%s' -> %s", args.attack, out)
    result = attack.run(behaviors)

    writer = ArtifactWriter(out)
    writer.write_attack_result(result)
    LOG.info(
        "Done: suffix=%r wall_clock=%.1fs n_steps=%d",
        result.suffix_str[:80], result.wall_clock_s, result.n_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
