"""visual_faithfulness and image support in the judge layer.

No network and no API key: images are tiny files with the right magic bytes,
the judge is a stub, and HTTP calls go to a fake urlopen.

    cd retrieval-mcp && python -m pytest tests/test_visual.py -q
"""
import base64
import importlib
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 24
JPEG = b"\xff\xd8\xff\xe0" + b"\0" * 24


@pytest.fixture
def judge(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_ALLOW_LOCAL_IMAGES", raising=False)
    import judge as j
    return importlib.reload(j)


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "living_room.png"
    p.write_bytes(PNG)
    return str(p)


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---- loading ----------------------------------------------------------------

def test_load_path_detects_type(judge, png, tmp_path):
    im = judge.load_image(png)
    assert im["media_type"] == "image/png"
    assert base64.b64decode(im["data"]) == PNG
    jpg = tmp_path / "x.jpg"
    jpg.write_bytes(JPEG)
    assert judge.load_image(str(jpg))["media_type"] == "image/jpeg"


def test_load_keeps_timestamp(judge, png):
    assert judge.load_image({"src": png, "t": 12.5})["t"] == 12.5


def test_load_data_uri(judge):
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert judge.load_image(uri)["media_type"] == "image/png"


def test_rejects_non_image(judge, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("not an image")
    with pytest.raises(ValueError, match="not a PNG"):
        judge.load_image(str(f))


def test_rejects_missing_file(judge):
    with pytest.raises(ValueError, match="not found"):
        judge.load_image("/nope/missing.png")


def test_rejects_oversize(judge, png, monkeypatch):
    monkeypatch.setattr(judge, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(ValueError, match="over"):
        judge.load_image(png)


def test_local_paths_can_be_disabled(judge, png, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_ALLOW_LOCAL_IMAGES", "0")
    with pytest.raises(ValueError, match="disabled"):
        judge.load_image(png)


def test_private_urls_refused(judge):
    for url in ("http://127.0.0.1/x.png", "http://10.0.0.5/x.png", "http://localhost/x.png"):
        with pytest.raises(ValueError, match="public host"):
            judge.load_image(url)


def test_public_url_fetched(judge, monkeypatch):
    monkeypatch.setattr(judge, "_public_host", lambda url: True)
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: Resp(PNG))
    assert judge.load_image("https://example.com/a.png")["media_type"] == "image/png"


def test_too_many_images(judge, png, monkeypatch):
    monkeypatch.setattr(judge, "MAX_IMAGES", 2)
    with pytest.raises(ValueError, match="limit"):
        judge.load_images([png, png, png])


# ---- backends ---------------------------------------------------------------

def test_judge_json_passes_images_only_when_given(judge, png, monkeypatch):
    calls = []

    def stub(system, user, max_tokens, **kw):
        calls.append(kw)
        return '{"score": 1}'

    monkeypatch.setattr(judge, "get_judge", lambda: stub)
    judge.judge_json("s", "u")
    judge.judge_json("s", "u", images=[png])
    assert calls[0] == {}
    assert calls[1]["images"][0]["media_type"] == "image/png"


def test_three_argument_backends_still_work(judge, monkeypatch):
    monkeypatch.setattr(judge, "get_judge", lambda: (lambda s, u, m: '{"ok": true}'))
    assert judge.judge_json("s", "u") == {"ok": True}


def test_openai_sends_image_url_parts(judge, png, monkeypatch):
    sent = {}

    def fake(req, timeout=None):
        sent.update(json.loads(req.data))
        return Resp(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    judge._openai("sys", "describe", 100, images=judge.load_images([png]))
    content = sent["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_text_only_unchanged(judge, monkeypatch):
    sent = {}

    def fake(req, timeout=None):
        sent.update(json.loads(req.data))
        return Resp(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    judge._openai("sys", "plain", 100)
    assert sent["messages"][1]["content"] == "plain"


def test_anthropic_content_blocks(judge, png):
    blocks = judge._anthropic_content("describe", judge.load_images([png]))
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["media_type"] == "image/png"
    assert blocks[-1] == {"type": "text", "text": "describe"}
    assert judge._anthropic_content("plain", None) == "plain"


def test_ollama_attaches_images(judge, png, monkeypatch):
    sent = {}

    def fake(req, timeout=None):
        sent.update(json.loads(req.data))
        return Resp(json.dumps({"message": {"content": "{}"}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    judge._ollama("sys", "describe", 100, images=judge.load_images([png]))
    assert sent["messages"][1]["images"] == [base64.b64encode(PNG).decode()]


# ---- the metric -------------------------------------------------------------

def _stub_judge(claims):
    seen = {}

    def jj(system, user, max_tokens=1024, images=None):
        seen["images"], seen["user"] = images, user
        return {"reasoning": "the pillow is not visible", "claims": claims}
    return jj, seen


CLAIMS = [
    {"claim": "There is a blue backpack.", "quote": "a blue backpack",
     "supported": True, "image": 1},
    {"claim": "The backpack is on the couch.", "quote": "on the couch",
     "supported": True, "image": 1},
    {"claim": "It is next to a red pillow.", "quote": "next to a red pillow",
     "supported": False, "image": None, "why": "No pillow is visible."},
]
DESC = "There is a blue backpack on the couch next to a red pillow."


def test_metric_scores_claims_against_images():
    import metrics as M
    jj, seen = _stub_judge(CLAIMS)
    case = {"actual_output": DESC, "images": ["photo.jpg"]}
    r = M.visual_faithfulness(case, jj, threshold=0.7)
    assert r["metric"] == "visual_faithfulness"
    assert r["score"] == pytest.approx(2 / 3, abs=1e-3)
    assert r["success"] is False
    assert r["details"]["kind"] == "visual"
    assert r["details"]["unsupported_claims"] == ["It is next to a red pillow."]
    assert r["details"]["claims"][0]["image"] == 1
    assert seen["images"] == ["photo.jpg"]          # references, resolved by the judge
    assert "Image 1" in seen["user"]


def test_metric_lists_frame_timestamps():
    import metrics as M
    jj, seen = _stub_judge(CLAIMS[:1])
    case = {"actual_output": "A blue backpack.",
            "images": [{"src": "f12.jpg", "t": 12}, {"src": "f13.jpg", "t": 13.5}]}
    M.visual_faithfulness(case, jj)
    assert "Image 1 (t=12s)" in seen["user"] and "Image 2 (t=13.5s)" in seen["user"]


def test_metric_requires_images():
    import metrics as M
    assert M.missing_inputs("visual_faithfulness", {"actual_output": "x"}) == ["images"]
    assert M.missing_inputs("visual_faithfulness", {"actual_output": "x", "images": ["a"]}) == []
    assert "visual_faithfulness" in M.BUILTIN


def test_explain_talks_about_images():
    import metrics as M
    jj, _ = _stub_judge(CLAIMS)
    r = M.visual_faithfulness({"actual_output": DESC, "images": ["p.jpg"]}, jj)
    v = M.explain("visual_faithfulness", [{"index": 0, "input": "", "scores": {"visual_faithfulness": r}}], 0.7)
    assert v["status"] == "fail"
    assert "not visible in the images" in v["drivers"][0]


# ---- golden sets and saved runs ----------------------------------------------

def test_goldenset_image_aliases():
    from goldensets import load_records
    recs = load_records(json.dumps([
        {"answer": "a backpack", "image": "a.jpg|b.jpg"},
        {"answer": "a backpack", "frames": [{"src": "f1.jpg", "t": 1}]},
    ]))
    assert recs[0]["images"] == ["a.jpg", "b.jpg"]
    assert recs[1]["images"] == [{"src": "f1.jpg", "t": 1}]


def test_saved_runs_never_store_image_bytes():
    import server
    refs = server._image_refs(["a.jpg", {"src": "data:image/png;base64,AAAA", "t": 3},
                               {"src": "https://x/y.png"}])
    assert refs == [{"src": "a.jpg"}, {"src": "(inline image)", "t": 3},
                    {"src": "https://x/y.png"}]


def test_evaluate_case_end_to_end(monkeypatch):
    import server
    jj, seen = _stub_judge(CLAIMS)
    monkeypatch.setattr(server, "judge_json", jj)
    out = server._evaluate_case("Where is my backpack?", DESC, ["visual_faithfulness"],
                                "", None, None, 0.7, False, ["photo.jpg"])
    assert out["scores"]["visual_faithfulness"]["score"] == pytest.approx(2 / 3, abs=1e-3)
    assert seen["images"] == ["photo.jpg"]
