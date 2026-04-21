"""Artifact (run-output) writing.

Every attack run dumps a directory of:
  - ``suffix.json``     final suffix (ids + detokenized)
  - ``meta.json``       run config + timing
  - ``loss_curve.csv``  per-step CE / activation / total loss
  - ``cosine.csv``      per-step refusal-direction cosine (if computed)
  - ``stdout.log``      training stdout

Standardising this layout lets the eval scripts pick up runs by glob without
inventing a new schema per attack.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class AttackResult:
    """Standardized return value of every attack."""

    suffix_ids: list[int]
    suffix_str: str
    wall_clock_s: float
    n_steps: int
    loss_curve: list[float] = field(default_factory=list)
    cosine_curve: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Standardized return value of every metric.

    ``score`` is the headline number (a proportion in [0, 1] for ASR-like
    metrics, or a float for things like wall-clock). ``per_sample`` keeps the
    behavior-level outcomes so we can do paired tests downstream.
    """

    score: float
    per_sample: list[float | bool] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class ArtifactWriter:
    """Atomic-ish writer for the standard run directory layout."""

    def __init__(self, run_dir: str | os.PathLike[str]):
        self.path = Path(run_dir)
        self.path.mkdir(parents=True, exist_ok=True)

    def write_suffix(self, ids: Sequence[int], detokenized: str) -> None:
        with (self.path / "suffix.json").open("w") as f:
            json.dump({"ids": list(ids), "str": detokenized}, f, indent=2)

    def write_meta(self, meta: dict[str, Any]) -> None:
        with (self.path / "meta.json").open("w") as f:
            json.dump(_to_jsonable(meta), f, indent=2)

    def write_curve(self, name: str, values: Iterable[float]) -> None:
        path = self.path / f"{name}.csv"
        with path.open("w") as f:
            f.write("step,value\n")
            for i, v in enumerate(values):
                f.write(f"{i},{float(v):.6e}\n")

    def write_eval(self, name: str, result: EvalResult) -> None:
        with (self.path / f"eval_{name}.json").open("w") as f:
            json.dump(
                {
                    "score": result.score,
                    "per_sample": list(result.per_sample),
                    "meta": _to_jsonable(result.meta),
                },
                f,
                indent=2,
            )

    def write_attack_result(self, result: AttackResult) -> None:
        self.write_suffix(result.suffix_ids, result.suffix_str)
        if result.loss_curve:
            self.write_curve("loss", result.loss_curve)
        if result.cosine_curve:
            self.write_curve("cosine", result.cosine_curve)
        self.write_meta(
            {
                "wall_clock_s": result.wall_clock_s,
                "n_steps": result.n_steps,
                **result.meta,
            }
        )


def _to_jsonable(obj: Any) -> Any:
    """Best-effort recursion to make dataclasses / tensors JSON-friendly."""
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    if hasattr(obj, "__float__"):
        try:
            return float(obj)
        except Exception:
            pass
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return repr(obj)
