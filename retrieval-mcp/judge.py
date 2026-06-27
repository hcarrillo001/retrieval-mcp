"""
Judge backend for RetriEval, with a hard spend cap.

Backends (swap via env):
    RETRIEVAL_JUDGE_BACKEND   "anthropic" (default) | "ollama"
    RETRIEVAL_JUDGE_MODEL     e.g. "claude-sonnet-4-6" or "deepseek-r1:70b"
    OLLAMA_URL                default "http://localhost:11434/api/chat"

Cost control:
    RETRIEVAL_BUDGET_USD      hard cap; once cumulative judge spend reaches it,
                              further Anthropic calls raise BudgetExceeded.
                              0 / unset = unlimited.
    RETRIEVAL_PRICE_IN/OUT    override $/1M tokens if the defaults drift.

Spend is metered from real token usage and persisted to $RETRIEVAL_HOME/spend.json,
so the cap holds across restarts and (when deployed) across requests.
Anthropic backend reads ANTHROPIC_API_KEY from the environment.
"""
from __future__ import annotations
import os
import re
import json

from paths import home

# Approximate USD per 1M tokens (input, output). Editable via env; verify against
# current Anthropic pricing for your chosen model.
PRICES = {
    "opus": (15.0, 75.0),
    "sonnet": (3.0, 15.0),
    "haiku": (0.80, 4.0),
}


class BudgetExceeded(RuntimeError):
    pass


def _price_for(model: str):
    pin, pout = os.environ.get("RETRIEVAL_PRICE_IN"), os.environ.get("RETRIEVAL_PRICE_OUT")
    if pin and pout:
        return float(pin), float(pout)
    for key, val in PRICES.items():
        if key in model:
            return val
    return (3.0, 15.0)


# ---- spend ledger -----------------------------------------------------------
def _spend_file():
    return home() / "spend.json"


def get_spend() -> float:
    f = _spend_file()
    if f.exists():
        try:
            return float(json.loads(f.read_text()).get("usd", 0.0))
        except Exception:
            return 0.0
    return 0.0


def _add_spend(usd: float) -> None:
    _spend_file().write_text(json.dumps({"usd": round(get_spend() + usd, 6)}))


def reset_spend() -> None:
    _spend_file().write_text(json.dumps({"usd": 0.0}))


def budget() -> float:
    try:
        return float(os.environ.get("RETRIEVAL_BUDGET_USD", "0") or 0)
    except ValueError:
        return 0.0


def budget_status() -> dict:
    b, s = budget(), get_spend()
    return {"spent_usd": round(s, 4), "budget_usd": b or None,
            "remaining_usd": (round(b - s, 4) if b else None)}


def _check_budget() -> None:
    b = budget()
    if b and get_spend() >= b:
        raise BudgetExceeded(
            f"Spend cap ${b:.2f} reached (used ${get_spend():.4f}). "
            f"Raise RETRIEVAL_BUDGET_USD or call reset_budget to continue."
        )


# ---- JSON parsing -----------------------------------------------------------
def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


# ---- backends ---------------------------------------------------------------
def _anthropic(system: str, user: str, max_tokens: int = 1024) -> str:
    from anthropic import Anthropic

    _check_budget()  # refuse before spending if cap already hit
    client = Anthropic()
    model = (os.environ.get("RETRIEVAL_JUDGE_MODEL")
             or os.environ.get("TOUCHSTONE_JUDGE_MODEL", "claude-sonnet-4-6"))
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    u = resp.usage
    pin, pout = _price_for(model)
    _add_spend((u.input_tokens / 1e6) * pin + (u.output_tokens / 1e6) * pout)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _ollama(system: str, user: str, max_tokens: int = 1024) -> str:
    import urllib.request

    model = (os.environ.get("RETRIEVAL_JUDGE_MODEL")
             or os.environ.get("TOUCHSTONE_JUDGE_MODEL", "deepseek-r1:70b"))
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
    payload = {"model": model, "stream": False, "format": "json",
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "options": {"num_predict": max_tokens}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["message"]["content"]


def _openai(system: str, user: str, max_tokens: int = 1024) -> str:
    """OpenAI-compatible Chat Completions. Works with OpenAI and any compatible
    endpoint (OpenRouter, Together, a local vLLM, etc.) via OPENAI_BASE_URL —
    so users can bring whatever LLM they want with their own key."""
    import urllib.request

    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("RETRIEVAL_JUDGE_MODEL", "gpt-4o")
    payload = {"model": model, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def get_judge():
    backend = (os.environ.get("RETRIEVAL_JUDGE_BACKEND")
               or os.environ.get("TOUCHSTONE_JUDGE_BACKEND", "anthropic")).lower()
    return {"ollama": _ollama, "openai": _openai}.get(backend, _anthropic)


def judge_json(system: str, user: str, max_tokens: int = 1024) -> dict:
    return _extract_json(get_judge()(system, user, max_tokens))
