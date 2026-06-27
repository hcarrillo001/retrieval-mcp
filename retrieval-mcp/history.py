"""
Run history for RetriEval — now backed by a pluggable store (local file or
Supabase). Records include the generator + judge model so the dashboard can
compare runs across LLMs.
"""
from __future__ import annotations
import time
import uuid

from store import get_store


def save_run(golden_set: str, threshold: float, aggregate: dict, label: str = "",
             per_case: list | None = None, generator_model: str = "",
             judge_model: str = "") -> str:
    run_id = "r-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "golden_set": golden_set,
        "threshold": threshold,
        "generator_model": generator_model,
        "judge_model": judge_model,
        "aggregate": aggregate,
        "per_case": per_case or [],
    }
    get_store().append(record)
    return run_id


def load_runs(golden_set: str | None = None, last_n: int | None = None) -> list:
    return get_store().list(golden_set, last_n)


def get_run(run_id: str) -> dict | None:
    if run_id == "latest":
        runs = get_store().list()
        return runs[-1] if runs else None
    return get_store().get(run_id)
