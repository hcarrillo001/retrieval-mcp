"""
Golden-set loading + normalization for Touchstone.

Accepts jsonl / json / csv files (or inline records) and normalizes whatever
field names the source uses into Touchstone's canonical case schema:

  input, actual_output, expected_output, context (list), retrieval_context (list)

So HotpotQA's "question"/"answer", a RAG export's "contexts", or your own
"prompt"/"response" all map to the same shape.
"""
from __future__ import annotations
import json
import csv
from pathlib import Path

# common aliases -> canonical key
_ALIASES = {
    "input": "input", "question": "input", "query": "input", "prompt": "input",
    "actual_output": "actual_output", "answer": "actual_output",
    "response": "actual_output", "output": "actual_output", "prediction": "actual_output",
    "expected_output": "expected_output", "expected": "expected_output",
    "ground_truth": "expected_output", "reference": "expected_output",
    "gold": "expected_output", "gold_answer": "expected_output",
    "context": "context", "contexts": "context",
    "retrieval_context": "retrieval_context", "retrieved": "retrieval_context",
    "retrieved_contexts": "retrieval_context", "passages": "retrieval_context",
}

_LIST_FIELDS = {"context", "retrieval_context"}


def _normalize(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        key = _ALIASES.get(k.strip().lower())
        if not key:
            continue
        if key in _LIST_FIELDS and isinstance(v, str):
            v = [v]
        out[key] = v
    return out


def load_records(path_or_inline, fmt: str = "auto") -> list:
    """Load and normalize golden cases from: a file path (incl. uploaded files),
    an http(s) URL, an inline JSON array string, or JSONL text."""
    # inline list of dicts
    if isinstance(path_or_inline, list):
        return [_normalize(r) for r in path_or_inline]

    text = str(path_or_inline)

    # remote URL
    if text.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(text) as resp:
            body = resp.read().decode("utf-8")
        if fmt == "auto":
            fmt = "jsonl" if text.rstrip().endswith((".jsonl", ".ndjson")) else (
                "csv" if text.rstrip().endswith(".csv") else "json")
        if fmt in ("jsonl", "ndjson"):
            return [_normalize(json.loads(l)) for l in body.splitlines() if l.strip()]
        if fmt in ("csv", "tsv"):
            delim = "\t" if fmt == "tsv" else ","
            return [_normalize(r) for r in csv.DictReader(body.splitlines(), delimiter=delim)]
        data = json.loads(body)
        return [_normalize(r) for r in (data if isinstance(data, list) else [data])]

    # Inline JSON must be handled BEFORE any filesystem probe. Path.exists() on a
    # long string does not return False, it raises OSError [Errno 36] "File name
    # too long", so inline golden sets used to fail once they grew past the OS
    # path limit (~255 bytes per component).
    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        try:
            data = json.loads(text)
            data = data if isinstance(data, list) else [data]
            return [_normalize(r) for r in data]
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            return [_normalize(r) for r in rows]

    p = Path(text)
    try:
        exists = p.exists()
    except OSError:
        exists = False  # too long / invalid to be a path: fall through to inline

    if not exists:
        # treat as inline JSON (object-per-line or a JSON array)
        try:
            data = json.loads(text)
            data = data if isinstance(data, list) else [data]
            return [_normalize(r) for r in data]
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            return [_normalize(r) for r in rows]

    if fmt == "auto":
        fmt = p.suffix.lstrip(".").lower()

    if fmt in ("jsonl", "ndjson"):
        with p.open() as f:
            return [_normalize(json.loads(line)) for line in f if line.strip()]
    if fmt == "json":
        with p.open() as f:
            data = json.load(f)
        data = data if isinstance(data, list) else [data]
        return [_normalize(r) for r in data]
    if fmt in ("csv", "tsv"):
        delim = "\t" if fmt == "tsv" else ","
        with p.open() as f:
            return [_normalize(row) for row in csv.DictReader(f, delimiter=delim)]
    raise ValueError(f"Unsupported format: {fmt}")


def load_hf(dataset: str, split: str = "validation", limit: int = 50,
            mapping: dict | None = None) -> list:
    """Optional: pull a public benchmark from HuggingFace (e.g. 'hotpot_qa').
    Requires `datasets`. mapping lets you remap source columns to canonical keys."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=f"{split}[:{limit}]")
    out = []
    for row in ds:
        if mapping:
            row = {mapping.get(k, k): v for k, v in row.items()}
        out.append(_normalize(row))
    return out
