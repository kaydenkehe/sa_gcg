"""Shared CLI helpers: arg parsing pieces that show up in multiple
entry points (model loading, dataset selection, output dir).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..models.load import load_chat_model


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf",
                   help="HF model id or local path (default: Llama-2-7B-Chat).")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None,
                   help="If set (e.g. 'auto'), use HF device_map and ignore --device.")


def add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="advbench",
                   choices=["advbench", "harmbench", "jailbreakbench", "strongreject"])
    p.add_argument("--data-root", default=None,
                   help="Directory holding the raw CSVs. Defaults to bundled.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)


def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", default="runs/default", type=str)
    p.add_argument("--seed", type=int, default=0)


def load_model_from_args(args) -> "LoadedModel":  # type: ignore[name-defined]
    return load_chat_model(
        args.model,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
