"""``sa-gcg-transfer`` — open + closed transfer eval for one (suffix, dataset).

Reads a suffix from a ``runs/.../suffix.json`` file (the artifact written
by ``sa-gcg-attack``), then runs the EVAL_PLAN §9 transfer pipeline:

  - ``--targets open:vicuna-7b llama-2-13b ...`` for HF open transfer
  - ``--targets api:openai api:anthropic api:google`` for closed

Each target writes ``transfer_<target>.json`` under ``--out-dir``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.registry import load_behaviors
from ..eval.harmbench_cls import HarmBenchScorer
from ..eval.strongreject import StrongRejectScorer
from ..transfer.closed_api import (
    AnthropicBackend,
    ClosedTransferConfig,
    GeminiBackend,
    OpenAIBackend,
    run_closed_transfer,
)
from ..transfer.open_transfer import run_open_transfer
from ..utils.artifact import EvalResult
from ..utils.logging import get_logger
from ._common import add_dataset_args, ensure_dir

LOG = get_logger("sa_gcg.cli.run_transfer")


_OPEN_TARGET_ALIASES = {
    "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
}

_CLOSED_BACKENDS = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "google": GeminiBackend,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sa-gcg-transfer")
    add_dataset_args(p)
    p.add_argument("--suffix-from", required=True, type=str,
                   help="Path to a suffix.json (artifact dir) or attack run dir.")
    p.add_argument("--targets", nargs="+", required=True,
                   help="e.g. 'open:llama-2-13b api:openai'")
    p.add_argument("--metrics", nargs="+",
                   default=["substring", "harmbench_cls", "strongreject"],
                   choices=["substring", "harmbench_cls", "strongreject"])
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--n-queries", type=int, default=1, help="Closed-API only.")
    p.add_argument("--is-full-prompt", action="store_true",
                   help="Suffix replaces the user message (PAIR-style).")
    p.add_argument("--openai-model", default=None,
                   help="Override OpenAIBackend.model (e.g. gpt-4o-2024-08-06).")
    p.add_argument("--anthropic-model", default=None,
                   help="Override AnthropicBackend.model (e.g. claude-sonnet-4-5).")
    p.add_argument("--google-model", default=None,
                   help="Override GeminiBackend.model (e.g. gemini-1.5-pro).")
    return p


def _load_suffix(path: str) -> str:
    p = Path(path)
    if p.is_dir():
        p = p / "suffix.json"
    payload = json.loads(p.read_text())
    return payload.get("str", "")


def _run_open(target_alias: str, args, behaviors, suffix_str, hb, sr) -> dict[str, EvalResult]:
    name_or_path = _OPEN_TARGET_ALIASES.get(target_alias, target_alias)
    return run_open_transfer(
        name_or_path,
        behaviors,
        suffix_str,
        metric_set=tuple(args.metrics),
        is_full_prompt=args.is_full_prompt,
        harmbench_scorer=hb,
        strongreject_scorer=sr,
    )


def _run_closed(provider: str, args, behaviors, suffix_str, hb, sr) -> dict[str, EvalResult]:
    backend_cls = _CLOSED_BACKENDS[provider]
    override = {
        "openai": args.openai_model,
        "anthropic": args.anthropic_model,
        "google": args.google_model,
    }[provider]
    backend = backend_cls(model=override) if override else backend_cls()
    cfg = ClosedTransferConfig(
        suffix_str=suffix_str,
        is_full_prompt=args.is_full_prompt,
        n_queries=args.n_queries,
    )
    return run_closed_transfer(
        backend, behaviors, cfg, harmbench_scorer=hb, strongreject_scorer=sr,
    )


def main():
    args = _build_parser().parse_args()
    behaviors = load_behaviors(
        args.dataset, data_root=args.data_root, limit=args.limit, offset=args.offset
    )
    suffix_str = _load_suffix(args.suffix_from)
    LOG.info("Suffix loaded (%d chars). Targets: %s", len(suffix_str), args.targets)
    out_dir = ensure_dir(args.out_dir)

    # Single shared HarmBench classifier across targets (it's heavy).
    hb = HarmBenchScorer() if "harmbench_cls" in args.metrics else None
    sr = StrongRejectScorer() if "strongreject" in args.metrics else None

    for target in args.targets:
        if ":" not in target:
            raise ValueError(f"Target must be 'open:NAME' or 'api:provider', got {target!r}")
        kind, name = target.split(":", 1)
        LOG.info("---- %s : %s", kind, name)
        if kind == "open":
            results = _run_open(name, args, behaviors, suffix_str, hb, sr)
        elif kind == "api":
            results = _run_closed(name, args, behaviors, suffix_str, hb, sr)
        else:
            raise ValueError(f"Unknown target kind {kind!r}")

        out_path = out_dir / f"transfer_{kind}_{name.replace('/', '_')}.json"
        with out_path.open("w") as f:
            json.dump(
                {
                    k: {"score": v.score, "per_sample": list(v.per_sample), "meta": v.meta}
                    for k, v in results.items()
                },
                f,
                indent=2,
                default=str,
            )
        LOG.info(
            "%s/%s scores: %s",
            kind, name, {k: v.score for k, v in results.items()},
        )

    if hb is not None:
        hb.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
