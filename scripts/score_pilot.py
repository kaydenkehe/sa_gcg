#!/usr/bin/env python3
"""EVAL_PLAN §8 — score the 12 SA-GCG pilot cells, identify the headline.

Loads Llama-2-7B-Chat once to generate completions for every cell on its
5 training behaviors, then frees it and loads HarmBench-cls once to score
all 60 generations in one batch. Avoids OOM on 24 GB GPUs.

Outputs:
  - <cell_dir>/pilot_eval.json     per-cell HB-cls score + generations
  - runs/pilot/sa_gcg/pilot_summary.json    headline + Pareto analysis
  - stdout: sorted table + Pareto verdict + headline
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PILOT_ROOT = Path(os.environ.get("PILOT_ROOT", "runs/pilot/sa_gcg"))
N_BEHAVIORS = int(os.environ.get("N_BEHAVIORS", "5"))
DATASET = os.environ.get("DATASET", "advbench")
MODEL = os.environ.get("MODEL", "meta-llama/Llama-2-7b-chat-hf")


def load_pilot_cells():
    cells = []
    for cell_dir in sorted(PILOT_ROOT.glob("scope_*")):
        suffix_path = cell_dir / "llama2/seed_0/suffix.json"
        meta_path = cell_dir / "llama2/seed_0/meta.json"
        if not suffix_path.exists() or not meta_path.exists():
            print(f"  WARN: missing artifact in {cell_dir}, skipping")
            continue
        suffix = json.loads(suffix_path.read_text())["str"]
        meta = json.loads(meta_path.read_text())
        cells.append(
            {
                "name": cell_dir.name,
                "suffix": suffix,
                "wall_clock_s": meta.get("wall_clock_s", 0.0),
                "out_dir": cell_dir / "llama2/seed_0",
            }
        )
    return cells


def main():
    from sa_gcg.data.registry import load_behaviors
    from sa_gcg.eval.generate import generate_with_suffix
    from sa_gcg.eval.harmbench_cls import HarmBenchScorer
    from sa_gcg.models.load import load_chat_model

    cells = load_pilot_cells()
    print(f"Loaded {len(cells)} pilot cells")
    if not cells:
        raise SystemExit("No cells found.")

    behaviors = load_behaviors(DATASET, limit=N_BEHAVIORS)
    prompts = [b.prompt for b in behaviors]
    print(f"Scoring on {len(prompts)} training prompts from {DATASET}")

    # ---- Phase 1: generate with target model ----------------------------
    print(f"\n[1/3] Loading target {MODEL} ...")
    target = load_chat_model(MODEL)

    cell_gens: dict[str, list[str]] = {}
    for i, cell in enumerate(cells, 1):
        print(f"  [{i}/{len(cells)}] generating for {cell['name']}")
        gens = generate_with_suffix(
            target,
            user_messages=prompts,
            suffix_str=cell["suffix"],
            max_new_tokens=256,
            temperature=0.0,
            batch_size=4,
        )
        cell_gens[cell["name"]] = gens

    # Free target.
    del target
    import torch

    torch.cuda.empty_cache()

    # ---- Phase 2: load HB-cls once and score all ------------------------
    print("\n[2/3] Loading HarmBench-cls and scoring all generations ...")
    hb = HarmBenchScorer()

    results = []
    for cell in cells:
        gens = cell_gens[cell["name"]]
        res = hb.score(prompts, gens)
        results.append(
            {
                "name": cell["name"],
                "wall_clock_s": cell["wall_clock_s"],
                "harmbench_cls": res.score,
                "per_sample": res.per_sample,
                "generations": gens,
            }
        )
        out_path = cell["out_dir"] / "pilot_eval.json"
        with out_path.open("w") as f:
            json.dump(
                {
                    "harmbench_cls": {
                        "score": res.score,
                        "per_sample": res.per_sample,
                        "meta": res.meta,
                    },
                    "generations": gens,
                },
                f,
                indent=2,
                default=str,
            )

    hb.close()

    # ---- Phase 3: print + Pareto analysis -------------------------------
    print("\n[3/3] Results")
    print("=" * 72)
    print(f"{'cell':52s} {'wall_s':>8s} {'asr':>6s}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: (-x["harmbench_cls"], x["wall_clock_s"])):
        print(f"{r['name']:52s} {r['wall_clock_s']:8.0f} {r['harmbench_cls']:6.2f}")
    print("=" * 72)

    # Pareto-dominated: lower ASR AND higher wall-clock than some other cell.
    dominated = []
    for r in results:
        for s in results:
            if s is r:
                continue
            if s["harmbench_cls"] > r["harmbench_cls"] and s["wall_clock_s"] < r["wall_clock_s"]:
                dominated.append({"cell": r["name"], "dominated_by": s["name"]})
                break

    print("\nPareto analysis (a cell is dominated if some other cell has BOTH higher ASR AND lower wall-clock):")
    if not dominated:
        print("  (no cells dominated; all on the Pareto frontier)")
    else:
        for d in dominated:
            print(f"  {d['cell']}  dominated by  {d['dominated_by']}")

    retained = [r for r in results if r["name"] not in {d["cell"] for d in dominated}]
    print(f"\nRetained cells for main grid: {len(retained)} (plan caps at 6)")
    for r in retained:
        print(f"  {r['name']}  asr={r['harmbench_cls']:.2f}  wall_s={r['wall_clock_s']:.0f}")

    # Headline: highest ASR, ties broken by lowest wall-clock.
    headline = max(results, key=lambda r: (r["harmbench_cls"], -r["wall_clock_s"]))
    print(f"\n*** PRE-REGISTERED HEADLINE: {headline['name']} ***")
    print(f"    asr={headline['harmbench_cls']:.2f}  wall_s={headline['wall_clock_s']:.0f}")

    summary = {
        "headline_cell": headline["name"],
        "headline_asr": headline["harmbench_cls"],
        "headline_wall_s": headline["wall_clock_s"],
        "pareto_dominated": dominated,
        "retained_cells": [r["name"] for r in retained],
        "all_cells": [
            {k: v for k, v in r.items() if k != "generations"} for r in results
        ],
    }
    out_path = PILOT_ROOT / "pilot_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
