"""Dataset registry.

Each registered dataset returns a list of ``Behavior`` records. We keep the
schema deliberately small (id, prompt, target_prefix, source) so a single
loader works across AdvBench, HarmBench, JailbreakBench, StrongREJECT.

For attack training we use ``target_prefix`` as the affirmative response
target ("Sure, here is..."). For evaluation we don't need it; only the prompt.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable

from ..utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass(frozen=True)
class Behavior:
    id: str
    prompt: str
    target_prefix: str
    source: str  # "advbench" | "harmbench" | "jailbreakbench" | "strongreject"
    category: str | None = None


def _data_dir() -> Path:
    """Path to the bundled raw datasets."""
    return Path(__file__).parent / "raw"


# ---------- Loaders ----------

def _load_advbench(path: Path) -> list[Behavior]:
    rows: list[Behavior] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompt = (row.get("goal") or row.get("behavior") or row.get("prompt") or "").strip()
            target = (row.get("target") or _affirmative(prompt)).strip()
            if not prompt:
                continue
            rows.append(
                Behavior(
                    id=f"advbench_{i:03d}",
                    prompt=prompt,
                    target_prefix=target,
                    source="advbench",
                )
            )
    return rows


def _load_harmbench(path: Path) -> list[Behavior]:
    rows: list[Behavior] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompt = (row.get("Behavior") or row.get("behavior") or row.get("prompt") or "").strip()
            if not prompt:
                continue
            cat = row.get("FunctionalCategory") or row.get("SemanticCategory")
            rows.append(
                Behavior(
                    id=f"harmbench_{i:03d}",
                    prompt=prompt,
                    target_prefix=_affirmative(prompt),
                    source="harmbench",
                    category=cat,
                )
            )
    return rows


def _load_jailbreakbench(path: Path) -> list[Behavior]:
    rows: list[Behavior] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompt = (row.get("Goal") or row.get("goal") or row.get("Behavior") or "").strip()
            if not prompt:
                continue
            target = (row.get("Target") or row.get("target") or _affirmative(prompt)).strip()
            cat = row.get("Category") or row.get("category")
            rows.append(
                Behavior(
                    id=f"jailbreakbench_{i:03d}",
                    prompt=prompt,
                    target_prefix=target,
                    source="jailbreakbench",
                    category=cat,
                )
            )
    return rows


def _load_strongreject(path: Path) -> list[Behavior]:
    rows: list[Behavior] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompt = (row.get("forbidden_prompt") or row.get("prompt") or "").strip()
            if not prompt:
                continue
            cat = row.get("category")
            rows.append(
                Behavior(
                    id=f"strongreject_{i:03d}",
                    prompt=prompt,
                    target_prefix=_affirmative(prompt),
                    source="strongreject",
                    category=cat,
                )
            )
    return rows


def _affirmative(prompt: str) -> str:
    """Default GCG-style affirmative target for a behavior."""
    return f"Sure, here is {_first_clause(prompt)}"


def _first_clause(prompt: str) -> str:
    p = prompt.strip().rstrip(".?!").lower()
    for prefix in ("write ", "explain ", "describe ", "give ", "create ", "provide ", "develop "):
        if p.startswith(prefix):
            return prefix + p[len(prefix):]
    return "how to " + p


# ---------- Registry ----------

DATASETS: dict[str, Callable[[Path], list[Behavior]]] = {
    "advbench": _load_advbench,
    "harmbench": _load_harmbench,
    "jailbreakbench": _load_jailbreakbench,
    "strongreject": _load_strongreject,
}

DATASET_FILENAMES = {
    "advbench": "advbench.csv",
    "harmbench": "harmbench_test.csv",
    "jailbreakbench": "jailbreakbench.csv",
    "strongreject": "strongreject.csv",
}


def load_behaviors(
    name: str,
    *,
    data_root: str | Path | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[Behavior]:
    """Load a dataset by name.

    ``data_root`` defaults to the package's bundled ``data/raw``. Pass an
    explicit path to point at the third_party/refusal_direction/dataset/raw
    location used by the existing AGCG/refusal pipeline.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list(DATASETS)}")
    root = Path(data_root) if data_root else _data_dir()
    fname = DATASET_FILENAMES[name]
    path = root / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            f"See README.md §3 (Datasets) for fetching instructions."
        )
    behaviors = DATASETS[name](path)
    end = offset + limit if limit is not None else None
    behaviors = behaviors[offset:end]
    LOG.info("Loaded %d behaviors from %s", len(behaviors), name)
    return behaviors
