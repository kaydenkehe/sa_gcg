"""Attack registry — single dispatch point for the runner CLI."""
from __future__ import annotations

from .base import AttackBase, AttackConfig

ATTACKS: dict[str, type[AttackBase]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        ATTACKS[name] = cls
        return cls

    return deco


def build_attack(name: str, model_bundle, config: AttackConfig) -> AttackBase:
    if name not in ATTACKS:
        raise KeyError(f"Unknown attack {name!r}. Registered: {sorted(ATTACKS)}")
    return ATTACKS[name](model_bundle, config)


# Side-effect registrations:
from . import no_suffix as _no_suffix  # noqa: E402,F401
from . import gcg as _gcg  # noqa: E402,F401
from . import soft_gcg as _soft_gcg  # noqa: E402,F401
from . import activation_gcg as _act_gcg  # noqa: E402,F401
from . import sa_gcg as _sa_gcg  # noqa: E402,F401
from . import ortho as _ortho  # noqa: E402,F401
from . import autodan_wrapper as _autodan  # noqa: E402,F401
from . import pair_wrapper as _pair  # noqa: E402,F401
