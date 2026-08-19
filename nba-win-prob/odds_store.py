import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ── Settings ──────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TABLE_NAME = "vegas_odds_snapshots"

_client = None


# ── Lazily create and cache the Supabase client ────────────────────────────────
def get_client():
    """Returns a cached Supabase client, creating it on first call."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL / SUPABASE_KEY are not set. Add them to a .env "
                "file (see .env.example) — get them from your Supabase "
                "project's Settings -> API page."
            )
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── Insert one snapshot row per game fetched from The Odds API ────────────────
def save_odds_snapshot(games):
    """
    Inserts one row per game into vegas_odds_snapshots, using the same dicts
    returned by vegas_odds.fetch_vegas_spreads(). Each call is a new
    snapshot (append-only) so line movement over time can be queried later.
    Returns the number of rows inserted.
    """
    if not games:
        return 0

    rows = [
        {
            "game_id":         game["game_id"],
            "commence_time":   game["commence_time"],
            "home_team":       game["home_team"],
            "away_team":       game["away_team"],
            "spreads":         game["spreads"],
            "home_spread_avg": game["home_spread_avg"],
            "num_books":       game["num_books"],
        }
        for game in games
    ]

    get_client().table(TABLE_NAME).insert(rows).execute()
    return len(rows)


# ── Read the newest stored snapshot for each upcoming game ────────────────────
def fetch_latest_games():
    """
    Returns the most recent snapshot row per game_id, ordered by tip-off time,
    so each upcoming game appears once with its freshest spread.
    """
    response = (
        get_client()
        .table(TABLE_NAME)
        .select("*")
        .order("fetched_at", desc=True)
        .execute()
    )

    latest_by_game = {}
    for row in response.data:
        # Rows arrive newest-first, so the first sighting of a game_id wins.
        latest_by_game.setdefault(row["game_id"], row)

    return sorted(
        latest_by_game.values(), key=lambda game: game["commence_time"]
    )


# ── Standalone test run ────────────────────────────────────────────────────────
def main():
    """Fetches live spreads and saves them as a new snapshot, for a quick test."""
    from vegas_odds import fetch_vegas_spreads

    games = fetch_vegas_spreads()
    inserted = save_odds_snapshot(games)
    print(f"[ODDS_STORE] Inserted {inserted} rows into '{TABLE_NAME}'.")


if __name__ == "__main__":
    main()
