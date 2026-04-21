"""Perplexity defense unit tests (no real model)."""
from __future__ import annotations

from sa_gcg.defenses.perplexity import calibrate_threshold, perplexity_filter


class _FakeBundle:
    """A bundle whose ``suffix_perplexity`` proxy returns a hash-based
    deterministic NLL. Used through monkeypatching below.
    """


def test_perplexity_filter_picks_threshold(monkeypatch):
    import sa_gcg.defenses.perplexity as P

    fake_nlls = {"safe": 1.0, "weird": 4.0, "extreme": 7.0}

    def fake_ppl(bundle, s, add_bos=True):
        return fake_nlls.get(s, 5.0)

    monkeypatch.setattr(P, "suffix_perplexity", fake_ppl)
    bundle = _FakeBundle()
    # Calibrate on natural-text-shape NLLs (~ 2-3 nats); threshold ~ 3.0.
    threshold = calibrate_threshold(
        bundle, ["x"] * 100, quantile=0.99,
    )  # all "x" -> 5.0; quantile = 5.0
    assert threshold == 5.0
    flags = perplexity_filter(bundle, ["safe", "weird", "extreme"], threshold)
    assert flags == [True, True, False]
