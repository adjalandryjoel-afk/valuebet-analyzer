-- ═══════════════════════════════════════════════════
--  VALUE BET ANALYZER — tables du miroir cloud
--  À coller dans Supabase : SQL Editor → Run
-- ═══════════════════════════════════════════════════

create table if not exists matches_cloud (
  match_key  text primary key,
  updated_at timestamptz not null default now(),
  payload    jsonb not null
);

create table if not exists bets_cloud (
  bet_key      text primary key,
  match_key    text not null,
  result       text,
  closing_odds double precision,
  clv_pct      double precision,
  updated_at   timestamptz not null default now(),
  payload      jsonb not null
);

create index if not exists idx_bets_cloud_match
  on bets_cloud (match_key);

create index if not exists idx_bets_cloud_pending
  on bets_cloud (match_key)
  where result is null;
