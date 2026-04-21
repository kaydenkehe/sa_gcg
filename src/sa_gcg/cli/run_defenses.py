"""``sa-gcg-defenses`` — perplexity filter + SmoothLLM evaluation.

Adaptive SA-GCG (re-running the attack with a perplexity penalty) is run
through ``sa-gcg-attack --perplexity-penalty 0.1`` and then re-evaluated
through this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.registry import load_behaviors
from ..defenses.perplexity import (
    calibrate_threshold,
    perplexity_filter,
    suffix_perplexity,
)
from ..defenses.smoothllm import SmoothLLM
from ..eval.harmbench_cls import HarmBenchScorer
from ..models.load import load_chat_model
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from ._common import add_dataset_args, add_model_args, ensure_dir, load_model_from_args

LOG = get_logger("sa_gcg.cli.run_defenses")


def _load_suffix(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        p = p / "suffix.json"
    return json.loads(p.read_text()).get("str", "")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sa-gcg-defenses")
    add_model_args(p)
    add_dataset_args(p)
    p.add_argument("--suffix-from", required=True, type=str,
                   help="Suffix to evaluate (artifact dir or path to suffix.json).")
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--seed", type=int, default=0)

    # Perplexity filter
    p.add_argument("--base-model", default="meta-llama/Llama-2-7b-hf",
                   help="Base (non-chat) LM for the perplexity filter.")
    p.add_argument("--reference-corpus", type=str, default=None,
                   help="Newline-separated text for threshold calibration. "
                        "Defaults to a small Alpaca sample bundled with the package.")
    p.add_argument("--ppl-quantile", type=float, default=0.99)

    # SmoothLLM
    p.add_argument("--smoothllm-op", default="swap", choices=["swap", "insert", "patch"])
    p.add_argument("--smoothllm-p", type=float, default=0.10)
    p.add_argument("--smoothllm-copies", type=int, default=10)
    p.add_argument("--no-smoothllm", action="store_true")
    return p


def _bundled_alpaca_sample() -> list[str]:
    """A tiny built-in calibration set so the CLI is runnable without
    extra downloads. Real runs should pass --reference-corpus.
    """
    return [
        "The capital of France is Paris.",
        "To boil an egg, place it in a pot of water and bring to a rolling boil for 9 minutes.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
        "A simple sorting algorithm is bubble sort, which repeatedly swaps adjacent elements.",
        "The mitochondria are the powerhouses of the cell, generating ATP through respiration.",
        "Newton's second law states that force equals mass times acceleration: F = m a.",
        "To make a peanut butter sandwich, spread peanut butter on bread and add jelly.",
        "The Pythagorean theorem says a squared plus b squared equals c squared in a right triangle.",
        "Mount Everest is the tallest mountain on Earth at 8,848 meters above sea level.",
        "DNA is a double-helix molecule that encodes the genetic instructions of all known life.",
    ]


def main():
    args = _build_parser().parse_args()
    set_seed(args.seed)
    suffix_str = _load_suffix(args.suffix_from)
    out = ensure_dir(args.out_dir)

    # ---------- Perplexity filter ----------
    base_bundle = load_chat_model(args.base_model, dtype=args.dtype, device=args.device)
    if args.reference_corpus:
        ref = [
            line.strip()
            for line in Path(args.reference_corpus).read_text().splitlines()
            if line.strip()
        ]
    else:
        LOG.warning("Using bundled tiny Alpaca calibration set; pass --reference-corpus for real runs.")
        ref = _bundled_alpaca_sample()
    threshold = calibrate_threshold(base_bundle, ref, quantile=args.ppl_quantile)
    nll = suffix_perplexity(base_bundle, suffix_str)
    passes_filter = nll <= threshold
    ppl_payload = {
        "threshold_nll": threshold,
        "suffix_nll": nll,
        "passes_filter": passes_filter,
        "ppl_quantile": args.ppl_quantile,
        "n_reference": len(ref),
    }
    with (out / "perplexity.json").open("w") as f:
        json.dump(ppl_payload, f, indent=2)
    LOG.info("Perplexity: nll=%.3f thresh=%.3f passes=%s", nll, threshold, passes_filter)

    # Free base model before loading target.
    try:
        import torch

        del base_bundle
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ---------- SmoothLLM ----------
    if not args.no_smoothllm:
        target = load_model_from_args(args)
        behaviors = load_behaviors(
            args.dataset, data_root=args.data_root, limit=args.limit, offset=args.offset
        )
        smoother = SmoothLLM(
            bundle=target,
            op=args.smoothllm_op,
            p_perturb=args.smoothllm_p,
            n_copies=args.smoothllm_copies,
            harmbench_scorer=HarmBenchScorer(),
        )
        result = smoother.score(behaviors, suffix_str)
        with (out / "smoothllm.json").open("w") as f:
            json.dump(
                {
                    "asr_under_smoothllm": result.score,
                    "per_sample": list(result.per_sample),
                    "meta": result.meta,
                },
                f, indent=2,
            )
        LOG.info("SmoothLLM ASR: %.3f", result.score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
