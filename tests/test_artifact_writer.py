"""ArtifactWriter integration — write/read round-trip."""
from __future__ import annotations

import json

from sa_gcg.utils.artifact import ArtifactWriter, AttackResult, EvalResult


def test_attack_result_roundtrip(tmp_path):
    res = AttackResult(
        suffix_ids=[1, 2, 3],
        suffix_str="hi",
        wall_clock_s=3.5,
        n_steps=42,
        loss_curve=[0.5, 0.4, 0.3],
        cosine_curve=[0.9, 0.6, 0.1],
        meta={"attack": "test", "lambda": 0.5},
    )
    out = tmp_path / "run"
    w = ArtifactWriter(out)
    w.write_attack_result(res)
    payload = json.loads((out / "suffix.json").read_text())
    assert payload["str"] == "hi"
    assert payload["ids"] == [1, 2, 3]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["wall_clock_s"] == 3.5
    assert meta["n_steps"] == 42
    assert meta["attack"] == "test"
    assert (out / "loss.csv").exists()
    assert (out / "cosine.csv").exists()


def test_eval_result_roundtrip(tmp_path):
    er = EvalResult(score=0.75, per_sample=[True, True, False, True], meta={"metric": "x"})
    w = ArtifactWriter(tmp_path)
    w.write_eval("x", er)
    payload = json.loads((tmp_path / "eval_x.json").read_text())
    assert payload["score"] == 0.75
    assert payload["per_sample"] == [True, True, False, True]
