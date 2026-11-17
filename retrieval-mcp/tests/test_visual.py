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
        # the first call is the claim check; later ones are per-image rechecks
        seen.setdefault("images", images)
        seen.setdefault("user", user)
        if "You are shown ONE image" in user:
            return {"shown": []}
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


def test_evaluate_case_tool_accepts_timestamped_frames(monkeypatch):
    """Through the MCP tool layer, not just the Python function: the tool's
    schema must let {"src", "t"} frames through (it used to require strings)."""
    import asyncio
    import server
    jj, seen = _stub_judge(CLAIMS)
    monkeypatch.setattr(server, "judge_json", jj)
    frames = [{"src": "frame_12.jpg", "t": 12.0}, "frame_13.jpg"]
    out = asyncio.run(server.mcp.call_tool("evaluate_case", {
        "input": "Where is my backpack?", "actual_output": DESC,
        "metrics": ["visual_faithfulness"], "images": frames, "save": False}))
    text = out[0][0].text if isinstance(out, tuple) else out[0].text
    assert '"visual_faithfulness"' in text
    assert seen["images"] == frames


# ---- recorded facts and the per-image recheck --------------------------------

def _scripted_judge(*replies):
    """A judge that answers each call in turn and records what it was sent."""
    calls = []

    def jj(system, user, max_tokens=1024, images=None):
        calls.append({"user": user, "images": images})
        return replies[min(len(calls), len(replies)) - 1]
    return jj, calls


REPORT = "Found the blue backpack. The drone flew 262 cm and landed about 1 m from it."
FACTS = ["The drone flew forward 262 cm in total.",
         "It landed 100 cm from the blue backpack."]


def test_recorded_facts_back_claims_a_picture_cannot_show():
    import metrics as M
    first = {"claims": [
        {"claim": "The drone found the blue backpack.", "quote": "Found the blue backpack",
         "supported": True, "image": 2, "source": "image"},
        {"claim": "The drone flew 262 cm.", "quote": "flew 262 cm",
         "supported": True, "image": None, "source": "facts"},
        {"claim": "It landed about 1 m from the backpack.", "quote": "landed about 1 m from it",
         "supported": True, "image": None, "source": "facts"}]}
    jj, calls = _scripted_judge(first)
    r = M.visual_faithfulness({"actual_output": REPORT, "images": ["a.png", "b.png"],
                               "context": FACTS}, jj)
    assert r["score"] == 1.0 and r["details"]["facts"] == 2
    assert [c["source"] for c in r["details"]["claims"]] == ["image", "facts", "facts"]
    assert "RECORDED FACTS" in calls[0]["user"] and "262 cm in total" in calls[0]["user"]
    assert len(calls) == 1                      # nothing rejected, so no recheck


def test_no_facts_means_no_facts_section():
    import metrics as M
    jj, calls = _scripted_judge({"claims": CLAIMS[:1]})
    M.visual_faithfulness({"actual_output": "A blue backpack.", "images": ["a.png"]}, jj)
    assert "RECORDED FACTS" not in calls[0]["user"]


def test_recheck_finds_claim_shown_in_only_one_image():
    """The flight bug: 'found the blue backpack' rejected because image 1 is the couch."""
    import metrics as M
    first = {"claims": [
        {"claim": "There is a couch.", "quote": "couch", "supported": True, "image": 1},
        {"claim": "The drone found the blue backpack.", "quote": "blue backpack",
         "supported": False, "image": None, "why": "Image 1 shows only a couch."}]}
    jj, calls = _scripted_judge(first, {"shown": []}, {"shown": [1]})
    r = M.visual_faithfulness({"actual_output": "A couch, then the blue backpack.",
                               "images": ["couch.png", "blue.png", "black.png"]}, jj)
    assert r["score"] == 1.0 and r["details"]["rechecked"] == 1
    row = r["details"]["claims"][1]
    assert row["supported"] and row["image"] == 2 and row["rechecked"]
    assert calls[1]["images"] == ["couch.png"] and calls[2]["images"] == ["blue.png"]
    assert len(calls) == 3                      # stopped once nothing was left


def test_recheck_cannot_rescue_a_wrong_claim():
    import metrics as M
    first = {"claims": [{"claim": "The backpack is red.", "quote": "red backpack",
                         "supported": False, "image": None, "why": "It is blue."}]}
    jj, calls = _scripted_judge(first, {"shown": []})
    r = M.visual_faithfulness({"actual_output": "A red backpack.",
                               "images": ["a.png", "b.png"]}, jj)
    assert r["score"] == 0.0 and "rechecked" not in r["details"]
    assert len(calls) == 3                      # first pass + one per image


def test_recheck_skipped_for_a_single_image():
    import metrics as M
    first = {"claims": [{"claim": "The backpack is red.", "quote": "red backpack",
                         "supported": False, "image": None}]}
    jj, calls = _scripted_judge(first)
    M.visual_faithfulness({"actual_output": "A red backpack.", "images": ["a.png"]}, jj)
    assert len(calls) == 1


def test_recheck_is_capped(monkeypatch):
    import metrics as M
    monkeypatch.setattr(M, "VISUAL_RECHECK_MAX", 2)
    first = {"claims": [{"claim": "A red backpack.", "quote": "red backpack",
                         "supported": False, "image": None}]}
    jj, calls = _scripted_judge(first, {"shown": []})
    M.visual_faithfulness({"actual_output": "A red backpack.",
                           "images": [f"{i}.png" for i in range(8)]}, jj)
    assert len(calls) == 3


def test_explain_mentions_recorded_facts():
    import metrics as M
    first = {"claims": [{"claim": "Flew 500 cm.", "quote": "flew 262 cm",
                         "supported": False, "image": None, "why": "The log says 262 cm."}]}
    jj, _ = _scripted_judge(first, {"shown": []})
    r = M.visual_faithfulness({"actual_output": REPORT, "images": ["a.png", "b.png"],
                               "context": FACTS}, jj)
    v = M.explain("visual_faithfulness", [{"index": 0, "input": "", "scores": {"visual_faithfulness": r}}], 0.7)
    assert "recorded facts" in v["drivers"][0]
