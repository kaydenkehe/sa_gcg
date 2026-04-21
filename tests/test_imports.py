"""Smoke: every public submodule imports without side effects."""
from __future__ import annotations


def test_top_level():
    import sa_gcg  # noqa: F401

    assert hasattr(sa_gcg, "__version__")


def test_subpackages():
    import importlib

    for name in [
        "sa_gcg.utils",
        "sa_gcg.utils.artifact",
        "sa_gcg.utils.logging",
        "sa_gcg.utils.seed",
        "sa_gcg.data.registry",
        "sa_gcg.data.splits",
        "sa_gcg.models.chat_template",
        "sa_gcg.models.hooks",
        "sa_gcg.losses.activation_loss",
        "sa_gcg.losses.ce_loss",
        "sa_gcg.losses.gradnorm",
        "sa_gcg.schedule.slushy",
        "sa_gcg.attack.base",
        "sa_gcg.attack.registry",
        "sa_gcg.eval.substring",
        "sa_gcg.stats.tests",
        "sa_gcg.defenses.perplexity",
        "sa_gcg.defenses.smoothllm",
    ]:
        importlib.import_module(name)


def test_attack_registry_complete():
    """All 8 attacks (B0-B6 + SA-GCG) register on import."""
    from sa_gcg.attack import ATTACKS  # triggers side-effect imports

    expected = {
        "no_suffix",
        "gcg",
        "soft_gcg",
        "agcg",
        "sa_gcg",
        "ortho",
        "autodan",
        "pair",
    }
    assert expected.issubset(set(ATTACKS)), set(ATTACKS)
