"""Headless check of the scores widget against fake hosts.

Each host puts the widget in a sandboxed iframe (allow-scripts only, opaque
origin) under the MCP Apps default CSP, then delivers a real run_eval
CallToolResult the way that host does.

    python tests/widget_harness.py [path/to/server.py]   # default: ./server.py
"""
import importlib.util
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent.parent

RESULT = {
    "content": [{"type": "text", "text": "{...}"}],
    "structuredContent": {
        "threshold": 0.7, "total_cases": 5,
        "aggregate": {"answer_relevancy": {"mean_score": 0.98},
                      "faithfulness": {"mean_score": 0.70}},
        "report_url": "https://www.retrieval-mcp.com/report?run=r-1"},
}

CSP = "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

# host-side scripts. `F` is the iframe element, `R` the CallToolResult.
HOSTS = {
    # SEP-1865: host waits for ui/initialize, replies, waits for initialized,
    # and only then sends ui/notifications/tool-result.
    "mcp-apps (spec handshake)": """
      window.addEventListener('message', e => {
        const d = e.data || {}; log.push(d.method || ('resp#' + d.id));
        if (d.method === 'ui/initialize')
          F.contentWindow.postMessage({jsonrpc:'2.0', id:d.id, result:{
            protocolVersion:'2026-01-26', hostInfo:{name:'fake',version:'1'},
            hostCapabilities:{}, hostContext:{theme:'light'}}}, '*');
        if (d.method === 'ui/notifications/initialized')
          F.contentWindow.postMessage({jsonrpc:'2.0', method:'ui/notifications/tool-result', params:R}, '*');
      });""",
    # ChatGPT Apps SDK: window.openai.toolOutput = structuredContent, set late.
    "chatgpt (window.openai)": "INJECT_OPENAI",
    # A host that never delivers, and whose window.openai throws when inspected.
    "silent host, hostile window.openai": "INJECT_HOSTILE",
}

INJECT = {
    "INJECT_OPENAI": """<script>setTimeout(function(){window.openai={toolOutput:%s};
        window.dispatchEvent(new Event('openai:set_globals'));},600)</script>""",
    "INJECT_HOSTILE": """<script>window.openai=new Proxy({}, {ownKeys(){throw new Error('no')},
        get(t,k){return k==='toolOutput'?null:undefined}});</script>""",
}


def js(v) -> str:
    """JSON for embedding inside a <script>: the widget itself contains
    </script>, which would otherwise end the host's script tag early."""
    return json.dumps(v).replace("</", "<\\/")


def widget_html(server_path: Path) -> str:
    spec = importlib.util.spec_from_file_location("srv_under_test", server_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE))
    spec.loader.exec_module(mod)
    return mod._SCORES_HTML


def run(server_path: Path) -> dict:
    html = widget_html(server_path)
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, host_js in HOSTS.items():
            doc = html
            if host_js in INJECT:
                pre = INJECT[host_js]
                if "%s" in pre:
                    pre = pre % json.dumps(RESULT["structuredContent"])
                doc = pre + html
                host_js = ""
            page = b.new_page()
            page.set_content(f"""<!doctype html><body><script>
              const log = []; window.hostLog = log; const R = {js(RESULT)};
              </script><iframe id=f sandbox="allow-scripts" width=500 height=260></iframe>
              <script>const F = document.getElementById('f');
              {host_js}
              F.srcdoc = {js(f'<meta http-equiv="Content-Security-Policy" content="{CSP}">' + doc)};
              </script>""")
            page.wait_for_timeout(300)
            frame = page.frames[1]
            state = {"painted": False, "text": ""}
            for _ in range(28):  # up to ~14 s: the widget's own give-up point is 10 s
                state = frame.evaluate("""() => { const r = document.getElementById('root');
                    return {painted: !!document.querySelector('.rr-card'),
                            text: r ? r.innerText : ''}; }""")
                if state["painted"] or "No tool result" in state["text"]:
                    break
                page.wait_for_timeout(500)
            out[name] = {"painted": state["painted"],
                         "text": " ".join(state["text"].split())[:160],
                         "host_saw": page.evaluate("window.hostLog")}
            page.close()
        b.close()
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "server.py"
    for k, v in run(target).items():
        print(f"{'PAINTED' if v['painted'] else 'NO DATA':8} {k}\n         widget: {v['text']}"
              f"\n         host got: {v['host_saw']}")
