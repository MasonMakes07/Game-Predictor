-- Stores every fetch of Vegas spreads as its own row (append-only history),
-- so line movement over time can be tracked. Run once in the Supabase SQL
-- Editor to create the table.

create table if not exists vegas_odds_snapshots (
    id              bigint generated always as identity primary key,
    game_id         text        not null,
    commence_time   timestamptz not null,
    home_team       text        not null,
    away_team       text        not null,
    spreads         jsonb       not null,   -- {"draftkings": -4.5, "fanduel": -4.0}
    home_spread_avg numeric     not null,
    num_books       integer     not null,
    fetched_at      timestamptz not null default now()
);

-- Speeds up "give me the latest snapshot per game" and "all snapshots for
-- this game" queries, which is how the app will read this table back.
create index if not exists idx_vegas_odds_snapshots_game_id
    on vegas_odds_snapshots (game_id, fetched_at desc);
