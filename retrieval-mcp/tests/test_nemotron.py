"""NVIDIA Nemotron sandbox preset, and the reasoning-model fixes it needs."""
import importlib
import io
import json

import pytest


@pytest.fixture
def judge(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "NEMOTRON_MODEL", "NEMOTRON_LABEL"):
        monkeypatch.delenv(var, raising=False)
    import judge as j
    return importlib.reload(j)


def test_nemotron_hidden_without_key(judge):
    assert "nemotron" not in [m["id"] for m in judge.sandbox_models()]


def test_nemotron_listed_with_key(monkeypatch, judge):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    m = [m for m in judge.sandbox_models() if m["id"] == "nemotron"][0]
    assert m["available"] is True
    assert "nemotron" in m["label"].lower()
    assert "NVIDIA" in m["label"]


def test_model_and_label_follow_env(monkeypatch):
    monkeypatch.setenv("NEMOTRON_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    import judge as j
    j = importlib.reload(j)
    p = j.SANDBOX_PRESETS["nemotron"]
    assert p["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert p["label"].startswith("nemotron-3-ultra-550b-a55b")


def test_judge_as_routes_to_openrouter(monkeypatch, judge):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        seen["model"] = json.loads(req.data)["model"]
        return Resp(json.dumps({"choices": [{"message": {"content": "{\"ok\": 1}"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with judge.judge_as("nemotron"):
        out = judge._openai("sys", "user")
    assert out == '{"ok": 1}'
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-or-test"
    assert seen["model"].startswith("nvidia/")


def test_none_content_becomes_empty_string(monkeypatch, judge):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: Resp(
        json.dumps({"choices": [{"message": {"content": None, "reasoning": "..."}}]}).encode()))
    with judge.judge_as("nemotron"):
        assert judge._openai("sys", "user") == ""


@pytest.mark.parametrize("raw", [
    '<think>maybe {"score": 0} no wait</think>{"score": 1}',
    'reasoning {"x": 2} still thinking</think>\n{"score": 1}',
    '```json\n{"score": 1}\n```',
    '{"score": 1}',
])
def test_extract_json_ignores_thinking(judge, raw):
    assert judge._extract_json(raw) == {"score": 1}


# ---- choosing NVIDIA per call from the MCP tools ---------------------------------

def test_use_judge_routes_one_call_to_nemotron(monkeypatch, judge):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert judge._override.get() is None
    with judge.use_judge("nemotron"):
        assert judge._override.get()["base_url"] == "https://openrouter.ai/api/v1"
        assert judge.get_judge() is judge._openai
        assert judge.jev_enabled() is False
        assert judge.judge_label().startswith("nvidia/")
    assert judge._override.get() is None          # back to the server default


def test_nvidia_alias(monkeypatch, judge):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    with judge.use_judge("NVIDIA"):
        assert judge._override.get()["model"].startswith("nvidia/")


def test_nemotron_without_key_says_what_is_missing(judge):
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        with judge.use_judge("nemotron"):
            pass
    assert judge._override.get() is None
    assert judge._jev_mode.get() is None


def test_unknown_judge_lists_options(judge):
    with pytest.raises(ValueError, match="nemotron"):
        judge.use_judge("gpt-9")
