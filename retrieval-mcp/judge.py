"""
Judge backend for RetriEval, with a hard spend cap.

Backends (swap via env):
    RETRIEVAL_JUDGE_BACKEND   "anthropic" (default) | "ollama" | "openai" | "jev"
    OPENROUTER_API_KEY        enables the "nemotron" sandbox preset (NVIDIA, free)
    NEMOTRON_MODEL            default "nvidia/nemotron-3.5-super-vl:free"
    RETRIEVAL_ALLOW_LOCAL_IMAGES  "1" (default) lets vision metrics read image files
                              from disk; set "0" on a shared hosted server
    RETRIEVAL_MAX_IMAGE_BYTES / RETRIEVAL_MAX_IMAGES   per-image size and per-call count caps
                              (jev: see the Jev section below)
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
import time
import contextvars

from paths import home

# ---- Sandbox judge presets --------------------------------------------------
# Swappable FREE models for the public sandbox. Keys live ONLY on the server
# (set the *_key_env var on Railway); the browser only ever sends the short id.
# All are OpenAI-compatible endpoints, so they run through the _openai backend.
def _pretty(model_id: str) -> str:
    """'openai/gpt-oss-120b' -> 'gpt-oss-120b'. Vendor prefixes and ':free'
    suffixes are routing detail, not something a visitor needs to read."""
    name = model_id.split("/")[-1]
    # ':free' is an OpenRouter routing marker; ':20b' is an Ollama size tag —
    # drop the first, keep the second
    if name.endswith(":free"):
        name = name[:-len(":free")]
    return name.replace(":", " ") or model_id


def _label(env_var: str, provider: str, model_id: str) -> str:
    """Dropdown label. Set <env_var> to override; otherwise it is generated from
    whatever model is actually configured, so swapping a retired model updates
    the UI at the same time instead of leaving a stale name on screen."""
    return os.environ.get(env_var) or f"{_pretty(model_id)} \u00b7 {provider}"


def _model(env_var: str, default: str) -> str:
    """Provider model ids get retired without notice (Groq dropped
    llama-3.3-70b-versatile). Reading them from env means a dead model is a
    Railway env change plus a restart, not a code push and redeploy."""
    return os.environ.get(env_var, default)


SANDBOX_PRESETS = {
    "groq-llama": {
        "label": _label("GROQ_LABEL", "via Groq (free)", _model("GROQ_MODEL", "openai/gpt-oss-120b")),
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": _model("GROQ_MODEL", "openai/gpt-oss-120b"),
    },
    # NVIDIA Nemotron, an open-weight model, via OpenRouter's free endpoint.
    # Free tier is rate-limited (about 20 requests/minute, 50/day until the
    # account has bought credit), so it is hidden until OPENROUTER_API_KEY is
    # set. Swap the model with NEMOTRON_MODEL; the label follows.
    "nemotron": {
        "label": _label("NEMOTRON_LABEL", "NVIDIA via OpenRouter (free)",
                        _model("NEMOTRON_MODEL", "nvidia/nemotron-3.5-super-vl:free")),
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": _model("NEMOTRON_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
        "optional": True,
    },
    # Jev scores claims; the LLM preset named in "decomposer" splits the answer
    # into claims. Listed in the picker only once TYPESAFE_API_KEY is set.
    "jev": {
        "label": os.environ.get("JEV_LABEL") or "Jev \u00b7 TypeSafe (experimental)",
        "kind": "jev",
        "key_env": "TYPESAFE_API_KEY",
        "decomposer": "groq-llama",
        "model": os.environ.get("JEV_MODEL", "jev-latest"),
    },
    # Other providers (Gemini, OpenRouter/Qwen, OpenRouter/DeepSeek, Ollama Cloud)
    # were removed while their free model ids are unverified — a dead option in a
    # public picker is worse than a short list. To restore one, add a block here
    # with its base_url/key_env and a *_MODEL env var; the label generates itself.
}
DEFAULT_SANDBOX_MODEL = os.environ.get("SANDBOX_DEFAULT_MODEL", "groq-llama")

# per-call override: {"base_url","key","model"[,"jev"]} set for one sandbox eval
_override: "contextvars.ContextVar[dict|None]" = contextvars.ContextVar(
    "judge_override", default=None)
# per-call Jev switch from the MCP tools' `judge` argument: None = server default
_jev_mode: "contextvars.ContextVar[bool|None]" = contextvars.ContextVar(
    "jev_mode", default=None)

# metrics Jev can score (mirrors metrics.JEV; judge.py never imports metrics.py)
JEV_METRICS = ("answer_relevancy", "faithfulness", "hallucination")


def sandbox_models() -> list[dict]:
    """Public list of selectable models (id + label only — no keys/urls)."""
    out = []
    for mid, p in SANDBOX_PRESETS.items():
        ok = bool(os.environ.get(p["key_env"], ""))
        if p.get("kind") == "jev":
            if not ok:
                continue  # the picker shows every entry; never list a dead one
            d = SANDBOX_PRESETS[p["decomposer"]]
            ok = bool(os.environ.get(d["key_env"], ""))
            out.append({"id": mid, "label": p["label"], "available": ok,
                        "metrics": list(JEV_METRICS)})
            continue
        if not ok and p.get("optional"):
            continue
        out.append({"id": mid, "label": p["label"], "available": ok})
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
        if p.get("kind") == "jev":
            # Jev reads TYPESAFE_API_KEY itself; the override carries the LLM
            # that splits answers into claims, plus the flag that turns Jev on
            d = SANDBOX_PRESETS[p["decomposer"]]
            dkey = os.environ.get(d["key_env"], "")
            if not dkey:
                raise RuntimeError(f"model '{self.model_id}' needs {d['key_env']} "
                                   f"for splitting answers into claims")
            self._token = _override.set({"base_url": d["base_url"], "key": dkey,
                                         "model": d["model"], "jev": True,
                                         "split": jev_split_mode(sandbox=True)})
            return self
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
    # reasoning models (Nemotron, DeepSeek-R1, Qwen3) may wrap their reply in
    # <think>...</think>, and braces inside the thinking would confuse the
    # first-{ / last-} slice below. Drop closed blocks, then an unclosed one.
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S)
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


# ---- images -----------------------------------------------------------------
# Vision metrics hand the judge a list of image references. They are resolved
# here, not in metrics.py, so the metrics stay I/O-free and testable with stubs.
# A reference is a path, an http(s) URL, a data: URI, or a dict {"src": ..., "t": ...}
# where t is an optional timestamp in seconds (frames from a video or a flight).
MAX_IMAGE_BYTES = int(os.environ.get("RETRIEVAL_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
MAX_IMAGES = int(os.environ.get("RETRIEVAL_MAX_IMAGES", "20"))

_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]


def _sniff(data: bytes) -> str:
    for magic, mt in _MAGIC:
        if data.startswith(magic):
            return mt
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _public_host(url: str) -> bool:
    """False for loopback, private, link-local and reserved addresses, so an
    image URL can't be used to reach the server's own network."""
    import ipaddress
    import socket
    from urllib.parse import urlparse
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def _read_url(url: str) -> bytes:
    import urllib.request
    if not _public_host(url):
        raise ValueError(f"image URL is not on a public host: {url}")
    req = urllib.request.Request(url, headers={
        "User-Agent": "retriEVAL/1.0 (+https://retrieval-mcp.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read(MAX_IMAGE_BYTES + 1)
    return data


def _local_images_allowed() -> bool:
    # off on a shared hosted server: a caller there should not be able to make
    # the judge read files off the server's disk
    return os.environ.get("RETRIEVAL_ALLOW_LOCAL_IMAGES", "1").lower() not in ("0", "false", "no", "off")


def load_image(ref) -> dict:
    """Resolve one image reference to {"media_type", "data" (base64), "t", "label"}."""
    import base64
    from pathlib import Path

    t = None
    if isinstance(ref, dict):
        t = ref.get("t")
        ref = ref.get("src") or ref.get("url") or ref.get("path") or ""
    ref = str(ref or "").strip()
    if not ref:
        raise ValueError("empty image reference")
    if ref.startswith("data:"):
        head, _, b64 = ref.partition(",")
        raw = base64.b64decode(b64)
        label = "inline image"
    elif ref.startswith(("http://", "https://")):
        raw = _read_url(ref)
        label = ref
    else:
        if not _local_images_allowed():
            raise ValueError("local image paths are disabled on this server; pass an image URL instead")
        path = Path(ref).expanduser()
        if not path.is_file():
            raise ValueError(f"image not found: {ref}")
        raw = path.read_bytes()
        label = path.name
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image is over {MAX_IMAGE_BYTES // (1024 * 1024)} MB: {label}")
    mt = _sniff(raw)
    if not mt:
        raise ValueError(f"not a PNG, JPEG, GIF or WebP image: {label}")
    return {"media_type": mt, "data": base64.b64encode(raw).decode("ascii"),
            "t": t, "label": label}


def load_images(refs) -> list:
    refs = refs if isinstance(refs, list) else [refs]
    if len(refs) > MAX_IMAGES:
        raise ValueError(f"{len(refs)} images is over the limit of {MAX_IMAGES} "
                         f"(set RETRIEVAL_MAX_IMAGES); sample fewer frames")
    return [load_image(r) for r in refs]


# ---- backends ---------------------------------------------------------------
def _anthropic_content(user: str, images):
    if not images:
        return user
    blocks = [{"type": "image", "source": {"type": "base64", "media_type": im["media_type"],
                                            "data": im["data"]}} for im in images]
    return blocks + [{"type": "text", "text": user}]


def _anthropic(system: str, user: str, max_tokens: int = 1024, images=None) -> str:
    from anthropic import Anthropic

    _check_budget()  # refuse before spending if cap already hit
    client = Anthropic()
    model = (os.environ.get("RETRIEVAL_JUDGE_MODEL")
             or os.environ.get("TOUCHSTONE_JUDGE_MODEL", "claude-sonnet-4-6"))
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": _anthropic_content(user, images)}],
    )
    u = resp.usage
    pin, pout = _price_for(model)
    _add_spend((u.input_tokens / 1e6) * pin + (u.output_tokens / 1e6) * pout)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _ollama(system: str, user: str, max_tokens: int = 1024, images=None) -> str:
    import urllib.request

    model = (os.environ.get("RETRIEVAL_JUDGE_MODEL")
             or os.environ.get("TOUCHSTONE_JUDGE_MODEL", "deepseek-r1:70b"))
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
    payload = {"model": model, "stream": False, "format": "json",
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "options": {"num_predict": max_tokens}}
    if images:
        # Ollama takes base64 images on the message; the model must be a vision one
        payload["messages"][1]["images"] = [im["data"] for im in images]
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["message"]["content"]


# Free-tier providers cap tokens per minute; on a 429 we wait and retry this many
# times, as long as the provider says the limit clears within the max wait.
_RATE_LIMIT_RETRIES = int(os.environ.get("JUDGE_RATE_LIMIT_RETRIES", "2"))
_RATE_LIMIT_MAX_WAIT = float(os.environ.get("JUDGE_RATE_LIMIT_MAX_WAIT", "30"))


def _retry_after(headers, detail: str = "") -> float:
    """Seconds until a 429 clears: Retry-After, then the text of the error
    ("Please try again in 7.66s" / "in 1m2.5s"), else a short default."""
    try:
        v = (headers or {}).get("retry-after")
        if v:
            return max(0.0, float(v))
    except (TypeError, ValueError):
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", detail or "")
    if m:
        secs = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
        return secs + 60 * int(m.group(1) or 0)
    return 5.0


def _openai(system: str, user: str, max_tokens: int = 1024, images=None) -> str:
    """OpenAI-compatible Chat Completions. Works with OpenAI and any compatible
    endpoint (OpenRouter, Together, Groq, a local vLLM, etc.) via OPENAI_BASE_URL —
    so users can bring whatever LLM they want with their own key.
    If a per-call sandbox override is active, use its endpoint/key/model."""
    import urllib.request
    import urllib.error

    ov = _override.get()
    if ov:
        base, key, model = ov["base_url"].rstrip("/"), ov["key"], ov["model"]
    else:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("RETRIEVAL_JUDGE_MODEL", "gpt-4o")
    key = (key or "").strip()          # stray whitespace/newline from a pasted env var
    if key.lower().startswith("bearer "):
        key = key[7:].strip()          # tolerate a key pasted WITH the prefix
    content = user
    if images:
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url",
             "image_url": {"url": f"data:{im['media_type']};base64,{im['data']}"}}
            for im in images]
    payload = {"model": model, "max_tokens": max_tokens,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": content}]}
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 # Default urllib UA ("Python-urllib/3.x") is blocked with a bare
                 # 403 by Cloudflare-fronted APIs. Identify ourselves properly.
                 "User-Agent": "retriEVAL/1.0 (+https://retrieval-mcp.com)",
                 "Accept": "application/json"})
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                msg = json.loads(r.read())["choices"][0]["message"]
                # reasoning models can spend the whole budget thinking and return
                # content=None; hand back "" so judge_json retries with more room
                return msg.get("content") or ""
        except urllib.error.HTTPError as e:
            # urllib's str() drops the response body, which is where providers put
            # the real reason. Surface it so failures are diagnosable.
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                detail = ""
            code = e.code
            if code == 429 and attempt < _RATE_LIMIT_RETRIES and "per day" not in detail.lower():
                # free tiers cap tokens per MINUTE (Groq: 8K for gpt-oss-120b), so a
                # second run right after a long one gets a 429 that clears in
                # seconds. Wait it out instead of failing the run.
                wait = _retry_after(e.headers, detail)
                if wait <= _RATE_LIMIT_MAX_WAIT:
                    time.sleep(wait + 0.5)
                    continue
            break
    # a retired/renamed model id is the most common failure here, and the raw
    # provider blob reads as "this product is broken" to anyone trying it
    if code == 404 or "model_not_found" in detail:
        raise RuntimeError(
            f"This judge model is unavailable right now (the provider no longer "
            f"serves '{model}'). Pick another model from the list \u2014 the others "
            f"are unaffected."
        ) from None
    if code == 429:
        if "per day" in detail.lower():
            raise RuntimeError(
                "The free judge has used up today's allowance with its provider. "
                "Try Jev, or come back tomorrow."
            ) from None
        raise RuntimeError(
            "The free judge is busy right now (provider rate limit). Wait about a "
            "minute and try again, or run it with Jev."
        ) from None
    if code in (401, 403):
        raise RuntimeError(
            "This judge model isn't configured on the server (missing or rejected "
            "API key). Pick another model from the list."
        ) from None
    raise RuntimeError(f"judge HTTP {code} from {base}: {detail or code}") from None


# ---- Jev (TypeSafe AI) decision model ---------------------------------------
# Jev is not a text judge: it answers typed questions (Noul / Choice / Score)
# with numbers and no reasons. So it cannot sit behind the (system, user) -> str
# shape the rubrics use. Instead it gets its own entry point, jev_ask(), and the
# jev-aware metrics in metrics.py receive it as a callable next to jj. An LLM is
# still needed for the steps Jev cannot do (splitting an answer into claims, and
# every metric without a jev path); that LLM is RETRIEVAL_DECOMPOSER_BACKEND.
#
#   RETRIEVAL_JUDGE_BACKEND=jev
#   TYPESAFE_API_KEY              required
#   JEV_MODEL                     default "jev-latest"
#   TYPESAFE_URL                  default https://api.typesafe.ai/v1/systemone
#   RETRIEVAL_DECOMPOSER_BACKEND  anthropic (default) | ollama | openai
#   JEV_PRICE_IN                  $/1M input tokens, default 0.042 (output free)
#   JEV_MAX_WORKERS               parallel Jev calls per metric, default 8

JEV_DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"


def _backend_name() -> str:
    return (os.environ.get("RETRIEVAL_JUDGE_BACKEND")
            or os.environ.get("TOUCHSTONE_JUDGE_BACKEND", "anthropic")).lower()


def jev_enabled() -> bool:
    """True when Jev should score the metrics that have a jev path.
    Precedence: sandbox preset, then the tool call's `judge` argument, then
    RETRIEVAL_JUDGE_BACKEND."""
    ov = _override.get()
    if ov:
        return bool(ov.get("jev"))
    mode = _jev_mode.get()
    if mode is not None:
        return mode
    return _backend_name() == "jev"


# friendlier names a caller can use for a model preset
_JUDGE_ALIASES = {"nvidia": "nemotron"}


def model_presets() -> list:
    """Presets the `judge` argument accepts by name (LLM judges, not Jev)."""
    return sorted(k for k, p in SANDBOX_PRESETS.items() if p.get("kind") != "jev")


class use_judge:
    """Context manager for the MCP tools' `judge` argument:
        "" / "default"  whatever RETRIEVAL_JUDGE_BACKEND says
        "jev"           Jev for faithfulness + answer_relevancy, LLM for the rest
        "llm"           the LLM judge only, even when the server default is jev
        "nemotron"      a named model preset for this call only (alias "nvidia"):
                        NVIDIA Nemotron via OpenRouter, free; "groq-llama" etc.
                        work the same way. Needs that preset's key on the server."""
    def __init__(self, judge: str = ""):
        j = (judge or "default").strip().lower()
        j = _JUDGE_ALIASES.get(j, j)
        self._preset = None
        if j in ("default", "jev", "llm"):
            if j == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
                raise RuntimeError("judge='jev' needs TYPESAFE_API_KEY on the server.")
            self.mode = {"default": None, "jev": True, "llm": False}[j]
        elif j in model_presets():
            self._preset = judge_as(j)      # checks the key when entered
            self.mode = False               # a named LLM preset never routes to Jev
        else:
            raise ValueError(f"judge must be 'jev', 'llm', '' or one of "
                             f"{', '.join(model_presets())} (got '{judge}')")
        self._token = None

    def __enter__(self):
        self._token = _jev_mode.set(self.mode)
        if self._preset is not None:
            try:
                self._preset.__enter__()
            except Exception:
                _jev_mode.reset(self._token)
                raise
        return self

    def __exit__(self, *exc):
        if self._preset is not None:
            self._preset.__exit__(*exc)
        _jev_mode.reset(self._token)
        return False


def jev_ask(state, questions: dict) -> dict:
    """One System One call. `state` is a string, object or list; `questions`
    is the raw API shape, e.g. {"supported": {"type": "noul", "instructions": "..."}}.
    Returns the `answers` object keyed by question name. Spend is metered on
    input tokens (output is free) and counts against RETRIEVAL_BUDGET_USD."""
    import urllib.request
    import urllib.error

    _check_budget()
    key = (os.environ.get("TYPESAFE_API_KEY") or "").strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    if not key:
        raise RuntimeError("RETRIEVAL_JUDGE_BACKEND=jev needs TYPESAFE_API_KEY.")
    url = os.environ.get("TYPESAFE_URL", JEV_DEFAULT_URL)
    payload = {"state": state, "model": os.environ.get("JEV_MODEL", "jev-latest"),
               "questions": questions}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "User-Agent": "retriEVAL/1.0 (+https://retrieval-mcp.com)",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            detail = ""
        if e.code in (401, 403):
            raise RuntimeError("Jev rejected the API key (check TYPESAFE_API_KEY).") from None
        if e.code == 429:
            raise RuntimeError("Jev rate limit hit; retry shortly or lower "
                               "JEV_MAX_WORKERS.") from None
        raise RuntimeError(f"Jev HTTP {e.code}: {detail or e.reason}") from None
    usage = body.get("usage") or {}
    price_in = float(os.environ.get("JEV_PRICE_IN", "0.042"))
    _add_spend((usage.get("input_tokens", 0) or 0) / 1e6 * price_in)
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError(f"Jev returned no answers: {str(body)[:300]}")
    return answers


def jev_map(fn, items: list) -> list:
    """Run fn over items in parallel (a Jev call is 70-500 ms, so ten claims
    done serially take seconds; in parallel it is about one round trip).
    Order is preserved and the first exception propagates."""
    from concurrent.futures import ThreadPoolExecutor

    if not items:
        return []
    workers = max(1, min(len(items), int(os.environ.get("JEV_MAX_WORKERS", "8"))))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def jev_split_mode(sandbox: bool = False) -> str:
    """How Jev metrics find claims.
        "sentences"  split in code, Jev also decides what is a claim: ~1 s, no LLM
        "llm"        an LLM rewrites the answer into standalone claims: finer
                     claims, but the LLM call dominates the run time
    The sandbox preset defaults to sentences (JEV_SANDBOX_SPLIT); MCP runs to
    llm (RETRIEVAL_JEV_SPLIT)."""
    ov = _override.get()
    if ov and ov.get("split"):
        return ov["split"]
    env = "JEV_SANDBOX_SPLIT" if sandbox else "RETRIEVAL_JEV_SPLIT"
    v = (os.environ.get(env) or ("sentences" if sandbox else "llm")).strip().lower()
    return v if v in ("sentences", "llm") else "llm"


def judge_label() -> str:
    """Name of whatever is actually scoring, recorded with each run."""
    ov = _override.get()
    llm = (ov or {}).get("model") or os.environ.get("RETRIEVAL_JUDGE_MODEL", "claude-sonnet-4-6")
    if jev_enabled():
        return f"{os.environ.get('JEV_MODEL', 'jev-latest')} + {llm}"
    return llm


def get_judge():
    # an active sandbox override always routes through the OpenAI-compatible path
    if _override.get():
        return _openai
    backend = _backend_name()
    if backend == "jev":
        # the text LLM that splits answers into claims and runs every metric
        # that has no jev path
        backend = os.environ.get("RETRIEVAL_DECOMPOSER_BACKEND", "anthropic").lower()
    return {"ollama": _ollama, "openai": _openai}.get(backend, _anthropic)


# Ceiling for a judge's JSON reply. Long outputs yield long replies (one entry
# per extracted claim), so the budget scales with the prompt up to this cap.
JUDGE_MAX_TOKENS_CAP = int(os.environ.get("RETRIEVAL_JUDGE_MAX_TOKENS", "8192"))


def judge_json(system: str, user: str, max_tokens: int = 1024, images=None) -> dict:
    """Ask the judge for JSON.

    A fixed 1024-token reply budget silently truncates the JSON on long inputs,
    which then fails to parse. So: scale the budget with the prompt size, and if
    the reply still comes back cut off mid-JSON, retry once with a bigger budget
    before giving up.
    """
    est = len(system) + len(user)
    budget = max(max_tokens, min(JUDGE_MAX_TOKENS_CAP, est // 3))
    fn = get_judge()
    # images are loaded once and only passed when present, so backends (and
    # test stubs) that never see images keep their three-argument signature
    extra = {"images": load_images(images)} if images else {}
    raw = fn(system, user, budget, **extra)
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        retry = min(JUDGE_MAX_TOKENS_CAP, max(budget * 3, 4096))
        if retry <= budget:
            raise  # already at the cap; a bigger budget won't help
        return _extract_json(fn(system, user, retry, **extra))
