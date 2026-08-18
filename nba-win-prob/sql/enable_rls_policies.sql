-- vegas_odds_snapshots holds public NBA odds data (not user data), so a
-- permissive policy for the anon/publishable key is fine here. Run once in
-- the Supabase SQL Editor, after create_vegas_odds_snapshots.sql.

alter table vegas_odds_snapshots enable row level security;

create policy "Allow anon read" on vegas_odds_snapshots
    for select
    to anon
    using (true);

create policy "Allow anon insert" on vegas_odds_snapshots
    for insert
    to anon
    with check (true);
