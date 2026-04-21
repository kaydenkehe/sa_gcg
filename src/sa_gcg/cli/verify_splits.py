"""``sa-gcg-verify-splits`` — Appendix A disjointness check.

Asserts the six pairwise splits documented in EVAL_PLAN §Appendix A. Exit
code 0 = all clean; 1 = any overlap. Used in CI.

Pass ``--harmful``/``--harmless`` for the extraction sets (path to .txt).
The four eval datasets (advbench/harmbench/jailbreakbench/strongreject)
are auto-loaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..data.registry import load_behaviors
from ..data.splits import VerifySplitError, verify_disjoint
from ..utils.logging import get_logger

LOG = get_logger("sa_gcg.cli.verify_splits")


def _read_lines(path: str | Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.strip().startswith("#")]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sa-gcg-verify-splits")
    p.add_argument("--harmful", type=str, required=True)
    p.add_argument("--harmless", type=str, required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--advbench-train-n", type=int, default=25)
    p.add_argument("--advbench-eval-n", type=int, default=100)
    return p


def main():
    args = _build_parser().parse_args()
    harmful = _read_lines(args.harmful)
    harmless = _read_lines(args.harmless)

    advbench_all = load_behaviors("advbench", data_root=args.data_root)
    train = advbench_all[: args.advbench_train_n]
    eval_set = advbench_all[
        args.advbench_train_n : args.advbench_train_n + args.advbench_eval_n
    ]
    harmbench = load_behaviors("harmbench", data_root=args.data_root)
    jbb = load_behaviors("jailbreakbench", data_root=args.data_root)
    sr = load_behaviors("strongreject", data_root=args.data_root)

    cases = [
        (harmful, train, "harmful_extraction", "advbench_train_25"),
        (harmful, eval_set, "harmful_extraction", "advbench_eval_100"),
        (harmful, harmbench, "harmful_extraction", "harmbench_159"),
        (harmful, jbb, "harmful_extraction", "jailbreakbench_100"),
        (harmful, sr, "harmful_extraction", "strongreject_60"),
        (train, eval_set, "advbench_train_25", "advbench_eval_100"),
    ]
    n_fail = 0
    for a, b, an, bn in cases:
        try:
            verify_disjoint(a, b, a_name=an, b_name=bn)
            LOG.info("OK: %s ∩ %s = ∅", an, bn)
        except VerifySplitError as e:
            n_fail += 1
            LOG.error("FAIL: %s", e)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
