"""``sa-gcg-tables`` — turn a directory tree of run artifacts into the
EVAL_PLAN tables (CSV + a Markdown rendering for the paper).

We expect a structure like::

    runs/
      sa_gcg/llama2/seed_0/{suffix.json, eval_*.json, transfer_*.json}
      gcg/llama2/seed_0/...

The script collects every ``eval_*.json`` and ``transfer_*.json``,
joins on (attack, model, dataset, metric), and writes:

  - ``tables/table1_individual.csv``
  - ``tables/table2_universal.csv``
  - ``tables/table3_open_transfer.csv``
  - ``tables/table4_closed_transfer.csv``
  - ``tables/main_tables.md``
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev

from ..stats.tests import benjamini_hochberg, paired_mcnemar
from ..utils.logging import get_logger

LOG = get_logger("sa_gcg.cli.make_tables")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sa-gcg-tables")
    p.add_argument("--runs-root", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--reference-attack", default="gcg",
                   help="Attack treated as the comparison baseline for McNemar/BH.")
    return p


def _parse_run_dir(path: Path) -> dict | None:
    """Path looks like ``runs/<attack>/<model>/seed_<k>``. Permissive."""
    rel = path.parts
    if len(rel) < 3:
        return None
    seed_match = re.match(r"seed[_-](\d+)", rel[-1])
    if not seed_match:
        return None
    return {
        "seed": int(seed_match.group(1)),
        "model": rel[-2],
        "attack": rel[-3],
    }


def _collect(runs_root: Path) -> list[dict]:
    """Walk ``runs_root`` and yield one dict per (attack, model, seed,
    dataset, metric, score)."""
    out: list[dict] = []
    for suffix_json in runs_root.rglob("suffix.json"):
        meta = _parse_run_dir(suffix_json.parent)
        if meta is None:
            continue
        for eval_json in suffix_json.parent.glob("eval_*.json"):
            payload = json.loads(eval_json.read_text())
            metric = eval_json.stem.removeprefix("eval_")
            row = dict(meta)
            row["metric"] = metric
            row["score"] = float(payload.get("score", 0.0))
            row["per_sample"] = payload.get("per_sample", [])
            row["dataset"] = payload.get("meta", {}).get("dataset", "unknown")
            out.append(row)
        for transfer_json in suffix_json.parent.glob("transfer_*.json"):
            payload = json.loads(transfer_json.read_text())
            target = transfer_json.stem.removeprefix("transfer_")
            for metric, sub in payload.items():
                row = dict(meta)
                row["metric"] = metric
                row["score"] = float(sub.get("score", 0.0))
                row["per_sample"] = sub.get("per_sample", [])
                row["dataset"] = "transfer:" + target
                out.append(row)
    return out


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r if k != "per_sample"})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _aggregate(rows: list[dict]) -> dict[tuple, dict]:
    """Mean ± SE across seeds for each (attack, model, dataset, metric)."""
    by_key: dict[tuple, list[float]] = {}
    for r in rows:
        key = (r["attack"], r["model"], r["dataset"], r["metric"])
        by_key.setdefault(key, []).append(r["score"])
    return {
        k: {"mean": mean(v), "sd": pstdev(v) if len(v) > 1 else 0.0, "n": len(v)}
        for k, v in by_key.items()
    }


def _md_render(agg: dict[tuple, dict]) -> str:
    lines = ["# Headline tables\n"]
    by_dm: dict[tuple[str, str], list] = {}
    for (att, mdl, ds, met), s in agg.items():
        by_dm.setdefault((mdl, met), []).append((att, ds, s))
    for (mdl, met), rows in sorted(by_dm.items()):
        lines.append(f"\n## Model={mdl} Metric={met}\n")
        lines.append("| Attack | Dataset | Mean | SD | N |")
        lines.append("|---|---|---|---|---|")
        for att, ds, s in sorted(rows):
            lines.append(f"| {att} | {ds} | {s['mean']:.3f} | {s['sd']:.3f} | {s['n']} |")
    return "\n".join(lines) + "\n"


def _mcnemar_table(rows: list[dict], reference_attack: str) -> list[dict]:
    """Pairwise McNemar of every attack vs. ``reference_attack``, by
    (model, dataset, metric). Only meaningful for binary per_sample.
    """
    by_key: dict[tuple, dict[str, list]] = {}
    for r in rows:
        key = (r["model"], r["dataset"], r["metric"])
        by_key.setdefault(key, {}).setdefault(r["attack"], []).extend(r["per_sample"])
    out: list[dict] = []
    pairs: list[tuple[tuple, str]] = []
    pvals: list[float] = []
    for key, atts in by_key.items():
        if reference_attack not in atts:
            continue
        ref = atts[reference_attack]
        for name, vals in atts.items():
            if name == reference_attack or len(ref) != len(vals):
                continue
            res = paired_mcnemar(
                [bool(v) for v in vals], [bool(v) for v in ref]
            )
            pairs.append((key, name))
            pvals.append(res.p_value)
            out.append(
                dict(
                    model=key[0],
                    dataset=key[1],
                    metric=key[2],
                    attack=name,
                    vs=reference_attack,
                    b=res.b,
                    c=res.c,
                    p=res.p_value,
                    effect=res.effect,
                )
            )
    rejects = benjamini_hochberg(pvals, alpha=0.05)
    for r, rej in zip(out, rejects):
        r["bh_reject"] = rej
    return out


def main():
    args = _build_parser().parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect(runs_root)
    LOG.info("Collected %d (attack, model, seed, dataset, metric) rows", len(rows))
    _write_csv(rows, out_dir / "raw_rows.csv")

    agg = _aggregate(rows)
    md = _md_render(agg)
    (out_dir / "main_tables.md").write_text(md)

    mc = _mcnemar_table(rows, reference_attack=args.reference_attack)
    _write_csv(mc, out_dir / "mcnemar.csv")
    LOG.info("Wrote %s, %s, %s",
             out_dir / "raw_rows.csv",
             out_dir / "main_tables.md",
             out_dir / "mcnemar.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
