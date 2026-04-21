"""B5: PAIR (Chao et al. 2023) — wrapper.

PAIR is a black-box attacker LLM that iteratively refines the prompt against
a target LLM. The upstream lives at https://github.com/patrickrchao/JailbreakingLLMs.
Like AutoDAN we shell out, except PAIR also needs an attacker-LLM endpoint
(by default Vicuna-13B; can be swapped to GPT-4 via env vars).

To enable:
    git clone https://github.com/patrickrchao/JailbreakingLLMs third_party/PAIR
    cd third_party/PAIR && pip install -r requirements.txt
    export PAIR_REPO=/abs/path/to/third_party/PAIR
    # optional: export PAIR_ATTACKER_MODEL=gpt-4o-mini  (uses OPENAI_API_KEY)
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


@register("pair")
class PAIR(AttackBase):
    def run(self, behaviors):
        repo = os.environ.get("PAIR_REPO")
        if repo is None or not Path(repo).is_dir():
            raise RuntimeError(
                "PAIR wrapper: set PAIR_REPO to a clone of "
                "https://github.com/patrickrchao/JailbreakingLLMs."
            )

        suffix_str = ""
        elapsed = 0.0
        for beh in behaviors:
            with tempfile.TemporaryDirectory() as tmp:
                t0 = time.time()
                cmd = [
                    "python",
                    "main.py",
                    "--target-model",
                    self.model_bundle.name_or_path,
                    "--attack-model",
                    os.environ.get("PAIR_ATTACKER_MODEL", "vicuna-13b-v1.5"),
                    "--judge-model",
                    os.environ.get("PAIR_JUDGE_MODEL", "gpt-4o-mini"),
                    "--goal",
                    beh.prompt,
                    "--target-str",
                    beh.target_prefix,
                    "--n-iterations",
                    "20",
                    "--n-streams",
                    "5",
                    "--out-path",
                    str(Path(tmp) / "result.json"),
                ]
                LOG.info("PAIR: %s", " ".join(cmd))
                subprocess.run(cmd, cwd=repo, check=True)
                elapsed += time.time() - t0
                result_path = Path(tmp) / "result.json"
                if result_path.exists():
                    payload = json.loads(result_path.read_text())
                    # PAIR's output is a full prompt, not a suffix; we record
                    # the entire text and let the eval treat it as a "prompt
                    # replacement" rather than a suffix append.
                    suffix_str = payload.get("best_prompt", "")
                if not self.config.universal:
                    break  # one suffix per behavior in non-universal mode
        ids = self.model_bundle.tokenizer(suffix_str, add_special_tokens=False).input_ids
        return AttackResult(
            suffix_ids=ids,
            suffix_str=suffix_str,
            wall_clock_s=elapsed,
            n_steps=-1,
            meta={"attack": self.name, "extern_repo": repo, "is_full_prompt": True},
        )
