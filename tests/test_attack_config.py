"""Sanity-check the attack-config dataclass and registry indirection."""
from __future__ import annotations

from sa_gcg.attack.base import AttackConfig
from sa_gcg.attack.registry import ATTACKS, build_attack


def test_default_config():
    c = AttackConfig()
    assert c.suffix_length == 20
    assert c.batch_size == 256
    assert c.composite_lambda == 1.0
    assert c.activation_scope in {"single", "layer", "global"}


def test_build_attack_unknown_raises():
    try:
        build_attack("does_not_exist", None, AttackConfig())
    except KeyError:
        return
    raise AssertionError("Expected KeyError for unknown attack name")


def test_attacks_registered_have_run_method():
    for name, cls in ATTACKS.items():
        assert hasattr(cls, "run"), name
