-- RetriEval run history table. Run this in the Supabase SQL editor.
create table if not exists runs (
  run_id          text primary key,
  timestamp       text not null,
  label           text default '',
  golden_set      text not null,
  threshold       real default 0.7,
  generator_model text default '',
  judge_model     text default '',
  aggregate       jsonb not null,     -- { metric: { mean_score, pass_rate, n } }
  per_case        jsonb default '[]', -- compact per-case scores
  inserted_at     timestamptz default now()
);

create index if not exists runs_golden_set_idx on runs (golden_set);
create index if not exists runs_timestamp_idx  on runs (timestamp desc);

-- RLS on. The MCP server and the dashboard API both use the SERVICE key, which
-- bypasses RLS. Anon/public clients get nothing unless you add a read policy.
alter table runs enable row level security;

-- (Optional) allow public READ of aggregates only, e.g. for a public dashboard:
-- create policy "public read" on runs for select to anon using (true);
