"""
Storage backends for run history.

Selected automatically:
  * SupabaseStore  — when SUPABASE_URL and SUPABASE_SERVICE_KEY are set. Run
                     history lives in Postgres, so the local CLI, the deployed
                     HTTP MCP, and the website dashboard all share one history.
  * FileStore      — otherwise. JSONL in $RETRIEVAL_HOME (the local default).

Both expose the same tiny interface: append(record), list(golden_set, last_n),
get(run_id). Supabase access is via PostgREST over urllib (no SDK dependency).
"""
from __future__ import annotations
import os
import json
import urllib.parse
import urllib.request

from paths import home


class FileStore:
    def __init__(self):
        self.path = home() / "runs.jsonl"

    def append(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def list(self, golden_set: str | None = None, last_n: int | None = None) -> list:
        if not self.path.exists():
            return []
        runs = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        if golden_set:
            runs = [r for r in runs if r.get("golden_set") == golden_set]
        runs.sort(key=lambda r: r["timestamp"])
        return runs[-last_n:] if last_n else runs

    def get(self, run_id: str) -> dict | None:
        for r in self.list():
            if r["run_id"] == run_id:
                return r
        return None


class SupabaseStore:
    def __init__(self):
        self.url = os.environ["SUPABASE_URL"].rstrip("/")
        self.key = os.environ["SUPABASE_SERVICE_KEY"]
        self.table = os.environ.get("RETRIEVAL_RUNS_TABLE", "runs")

    def _headers(self, extra=None):
        h = {"apikey": self.key, "Authorization": f"Bearer {self.key}",
             "Content-Type": "application/json"}
        if extra:
            h.update(extra)
        return h

    def _req(self, method, path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers=self._headers(headers))
        with urllib.request.urlopen(req) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else None

    def append(self, record: dict) -> None:
        self._req("POST", f"/rest/v1/{self.table}", body=record,
                  headers={"Prefer": "return=minimal"})

    def list(self, golden_set: str | None = None, last_n: int | None = None) -> list:
        q = {"select": "*", "order": "timestamp.asc"}
        if golden_set:
            q["golden_set"] = f"eq.{golden_set}"
        if last_n:
            q["order"] = "timestamp.desc"
            q["limit"] = str(last_n)
        rows = self._req("GET", f"/rest/v1/{self.table}?" + urllib.parse.urlencode(q)) or []
        return list(reversed(rows)) if last_n else rows

    def get(self, run_id: str) -> dict | None:
        path = f"/rest/v1/{self.table}?run_id=eq.{urllib.parse.quote(run_id)}&select=*"
        rows = self._req("GET", path) or []
        return rows[0] if rows else None


def get_store():
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
        return SupabaseStore()
    return FileStore()
