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
import contextvars

from paths import home

# ---- Sandbox judge presets --------------------------------------------------
# Swappable FREE models for the public sandbox. Keys live ONLY on the server
# (set the *_key_env var on Railway); the browser only ever sends the short id.
# All are OpenAI-compatible endpoints, so they run through the _openai backend.
SANDBOX_PRESETS = {
    "groq-llama": {
        "label": "Llama 3.3 70B · via Groq (free)",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "gemini-flash": {
        "label": "Gemini 2.5 Flash · Google (free)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
    "qwen-72b": {
        "label": "Qwen 2.5 72B · via OpenRouter (free)",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "qwen/qwen-2.5-72b-instruct:free",
    },
    "deepseek": {
        # DeepSeek (Chinese) via OpenRouter's free slot — same OPENROUTER_API_KEY as
        # Qwen. VERIFY the current free slug on openrouter.ai (free model ids churn,
        # e.g. deepseek/deepseek-r1:free or deepseek/deepseek-chat-v3-0324:free).
        # DeepSeek also has a cheap (not free) direct API at https://api.deepseek.com
        # with model "deepseek-chat" if you'd rather have reliability over a free tier.
        "label": "DeepSeek V3 · via OpenRouter (free)",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "deepseek/deepseek-chat-v3-0324:free",
    },
    "ollama-cloud": {
        # Ollama Cloud's OpenAI-compatible endpoint. VERIFY the exact base URL and
        # model id at docs.ollama.com/cloud (host may be ollama.com/v1 vs
        # api.ollama.com/v1; lighter models like gpt-oss:20b stretch the free quota).
        "label": "gpt-oss 20B · Ollama Cloud (free)",
        "base_url": "https://ollama.com/v1",
        "key_env": "OLLAMA_API_KEY",
        "model": "gpt-oss:20b",
    },
}
DEFAULT_SANDBOX_MODEL = os.environ.get("SANDBOX_DEFAULT_MODEL", "groq-llama")

# per-call override: {"base_url","key","model"} set for the duration of one eval
_override: "contextvars.ContextVar[dict|None]" = contextvars.ContextVar(
    "judge_override", default=None)


def sandbox_models() -> list[dict]:
    """Public list of selectable models (id + label only — no keys/urls)."""
    out = []
    for mid, p in SANDBOX_PRESETS.items():
        out.append({"id": mid, "label": p["label"],
                    "available": bool(os.environ.get(p["key_env"], ""))})
    return out


class judge_as:
    """Context manager: route judge calls through a sandbox preset for this call.
        with judge_as("groq-llama"):
            run the metrics...
    """
    def __init__(self, model_id: str):
        self.model_id = model_id or DEFAULT_SANDBOX_MODEL
        self._token = None

    def __enter__(self):
        p = SANDBOX_PRESETS.get(self.model_id)
        if not p:
            raise ValueError(f"unknown sandbox model '{self.model_id}'")
        key = os.environ.get(p["key_env"], "")
        if not key:
            raise RuntimeError(f"model '{self.model_id}' not configured "
                               f"(missing {p['key_env']})")
        self._token = _override.set(
            {"base_url": p["base_url"], "key": key, "model": p["model"]})
        return self

    def __exit__(self, *exc):
        if self._token is not None:
            _override.reset(self._token)
        return False

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
    so users can bring whatever LLM they want with their own key.
    If a per-call sandbox override is active, use its endpoint/key/model."""
    import urllib.request

    ov = _override.get()
    if ov:
        base, key, model = ov["base_url"].rstrip("/"), ov["key"], ov["model"]
    else:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("RETRIEVAL_JUDGE_MODEL", "gpt-4o")
    payload = {"model": model, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def get_judge():
    # an active sandbox override always routes through the OpenAI-compatible path
    if _override.get():
        return _openai
    backend = (os.environ.get("RETRIEVAL_JUDGE_BACKEND")
               or os.environ.get("TOUCHSTONE_JUDGE_BACKEND", "anthropic")).lower()
    return {"ollama": _ollama, "openai": _openai}.get(backend, _anthropic)


# Ceiling for a judge's JSON reply. Long outputs yield long replies (one entry
# per extracted claim), so the budget scales with the prompt up to this cap.
JUDGE_MAX_TOKENS_CAP = int(os.environ.get("RETRIEVAL_JUDGE_MAX_TOKENS", "8192"))


def judge_json(system: str, user: str, max_tokens: int = 1024) -> dict:
    """Ask the judge for JSON.

    A fixed 1024-token reply budget silently truncates the JSON on long inputs,
    which then fails to parse. So: scale the budget with the prompt size, and if
    the reply still comes back cut off mid-JSON, retry once with a bigger budget
    before giving up.
    """
    est = len(system) + len(user)
    budget = max(max_tokens, min(JUDGE_MAX_TOKENS_CAP, est // 3))
    fn = get_judge()
    raw = fn(system, user, budget)
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        retry = min(JUDGE_MAX_TOKENS_CAP, max(budget * 3, 4096))
        if retry <= budget:
            raise  # already at the cap; a bigger budget won't help
        return _extract_json(fn(system, user, retry))
