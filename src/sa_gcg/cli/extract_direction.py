"""``sa-gcg-extract-direction`` — re-extract the refusal direction.

Reads a harmful + harmless prompt set, computes diff-of-means at every
candidate (layer, position), selects via the Arditi screen (or the cheap
mean-projection fallback when no eval callables are wired), and writes a
``DirectionRecord`` to ``--out``.

The output is a ``.pt`` (with the direction tensor) plus a sibling
``.json`` summary.

Example::

    sa-gcg-extract-direction \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --harmful prompts/harmful128.txt \\
        --harmless prompts/harmless128.txt \\
        --out runs/direction/llama2.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..direction.extractor import extract_refusal_direction, save_direction
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from ._common import add_model_args, ensure_dir, load_model_from_args

LOG = get_logger("sa_gcg.cli.extract_direction")


def _read_lines(path: str | Path) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sa-gcg-extract-direction")
    add_model_args(p)
    p.add_argument("--harmful", required=True, type=str,
                   help="Newline-separated harmful instructions.")
    p.add_argument("--harmless", required=True, type=str,
                   help="Newline-separated harmless instructions.")
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--position", type=int, default=-1, help="Token position (default last).")
    p.add_argument("--candidate-layers", type=int, nargs="*", default=None,
                   help="Layers to consider; default = all.")
    p.add_argument("--max-layer-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = _build_parser().parse_args()
    set_seed(args.seed)
    bundle = load_model_from_args(args)
    harmful = _read_lines(args.harmful)
    harmless = _read_lines(args.harmless)
    LOG.info("Extracting from %d harmful / %d harmless prompts", len(harmful), len(harmless))

    record = extract_refusal_direction(
        bundle,
        harmful_prompts=harmful,
        harmless_prompts=harmless,
        candidate_positions=(args.position,),
        candidate_layers=args.candidate_layers,
        max_layer_frac=args.max_layer_frac,
    )
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    save_direction(record, out_path)
    LOG.info(
        "Saved direction to %s (ell_star=%d, p_star=%d)",
        out_path, record.ell_star, record.p_star,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
