"""Logging configuration."""
from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "sa_gcg") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level_name = os.environ.get("SA_GCG_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
