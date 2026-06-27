"""Shared filesystem paths for RetriEval (golden-set runs, spend ledger)."""
from __future__ import annotations
import os
from pathlib import Path


def home() -> Path:
    p = Path(
        os.environ.get("RETRIEVAL_HOME")
        or os.environ.get("TOUCHSTONE_HOME")
        or (Path.home() / ".retrieval")
    )
    p.mkdir(parents=True, exist_ok=True)
    return p
