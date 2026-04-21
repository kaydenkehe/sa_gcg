"""Data registry — load the bundled tiny AdvBench."""
from __future__ import annotations

from sa_gcg.data.registry import _load_advbench, load_behaviors
from pathlib import Path


def test_advbench_tiny_loads():
    path = Path(__file__).resolve().parents[1] / "src" / "sa_gcg" / "data" / "raw" / "advbench_tiny.csv"
    rows = _load_advbench(path)
    assert len(rows) >= 5
    assert all(r.source == "advbench" for r in rows)
    assert all(r.prompt for r in rows)


def test_load_behaviors_limit_offset(tmp_path):
    src = Path(__file__).resolve().parents[1] / "src" / "sa_gcg" / "data" / "raw" / "advbench_tiny.csv"
    dst = tmp_path / "advbench.csv"
    dst.write_text(src.read_text())
    rows = load_behaviors("advbench", data_root=tmp_path, limit=2, offset=1)
    assert len(rows) == 2
