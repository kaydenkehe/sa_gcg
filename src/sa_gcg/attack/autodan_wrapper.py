"""B4: AutoDAN (Liu et al. ICLR 2024) — wrapper around the upstream repo.

AutoDAN is a hierarchical-genetic attack over human-written jailbreak
templates. We do not reimplement it; instead we shell out to the upstream
``SheltonLiu-N/AutoDAN`` runner and harvest its output suffix.

To enable:
    git clone https://github.com/SheltonLiu-N/AutoDAN third_party/AutoDAN
    cd third_party/AutoDAN && pip install -r requirements.txt
    export AUTODAN_REPO=/abs/path/to/third_party/AutoDAN

The wrapper assumes the upstream's ``autodan_hga_eval.py`` interface; if the
upstream API changes, edit ``_run_subprocess`` below.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ..utils.artifact import AttackResult
from ..utils.logging import get_logger
from .base import AttackBase
from .registry import register

LOG = get_logger(__name__)


@register("autodan")
class AutoDAN(AttackBase):
    def run(self, behaviors):
        repo = os.environ.get("AUTODAN_REPO")
        if repo is None or not Path(repo).is_dir():
            raise RuntimeError(
                "AutoDAN wrapper: set AUTODAN_REPO to a clone of "
                "https://github.com/SheltonLiu-N/AutoDAN."
            )

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "behaviors.csv"
            out_path = Path(tmp) / "suffix.txt"
            with in_path.open("w") as f:
                f.write("goal,target\n")
                for beh in behaviors:
                    f.write(f"{_csv_escape(beh.prompt)},{_csv_escape(beh.target_prefix)}\n")
            t0 = time.time()
            self._run_subprocess(repo, in_path, out_path)
            elapsed = time.time() - t0
            suffix_str = out_path.read_text().strip() if out_path.exists() else ""

        ids = self.model_bundle.tokenizer(suffix_str, add_special_tokens=False).input_ids
        return AttackResult(
            suffix_ids=ids,
            suffix_str=suffix_str,
            wall_clock_s=elapsed,
            n_steps=-1,
            meta={"attack": self.name, "extern_repo": repo},
        )

    def _run_subprocess(self, repo: str, in_path: Path, out_path: Path) -> None:
        """Invoke the upstream entry point.

        The upstream API moves; we keep this in one place so a one-line edit
        adapts to a new release. The current expected interface is a script
        that reads ``--behaviors_path`` (CSV) and writes the final suffix to
        ``--out_path``. If your fork uses different flags, edit here.
        """
        cmd = [
            "python",
            "autodan_hga_eval.py",
            "--model_path",
            self.model_bundle.name_or_path,
            "--behaviors_path",
            str(in_path),
            "--out_path",
            str(out_path),
            "--num_steps",
            str(self.config.n_steps),
            "--seed",
            str(self.config.seed),
        ]
        LOG.info("AutoDAN: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=repo, check=True)


def _csv_escape(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'
