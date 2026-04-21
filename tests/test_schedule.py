"""Slushy + linear schedule sanity."""
from __future__ import annotations

from sa_gcg.schedule.slushy import LinearSchedule, SlushySchedule, make_schedule


def test_linear_endpoints():
    s = LinearSchedule(tau_0=2.0, tau_1=0.1, n_steps=11)
    assert abs(s(0) - 2.0) < 1e-12
    assert abs(s(10) - 0.1) < 1e-12


def test_slushy_three_phases():
    s = SlushySchedule(
        tau_warm=2.0, tau_anneal=0.3, tau_final=0.03,
        warm_frac=0.1, anneal_frac=0.7, n_steps=100,
    )
    # Phase 1 (warm): step 0..9 = 2.0
    assert s(0) == 2.0
    assert s(5) == 2.0
    # Phase 2 (anneal): step 10 starts the linear ramp
    assert 0.3 < s(20) < 2.0
    # Phase 3 (tail): final 20 steps fall to tau_final
    assert s(99) < 0.3


def test_make_dispatcher():
    assert isinstance(make_schedule("slushy", 100), SlushySchedule)
    assert isinstance(make_schedule("linear", 100), LinearSchedule)
    try:
        make_schedule("nope", 100)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown schedule")
