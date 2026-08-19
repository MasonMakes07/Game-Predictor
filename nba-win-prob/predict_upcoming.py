import sys

import requests

from odds_store import TABLE_NAME, get_client

# ── Settings ──────────────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:5000"
REQUEST_TIMEOUT = 10

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Reduce a full team name to the nickname the server can match ──────────────
def team_lookup_key(full_name):
    """
    Returns a team's nickname (e.g. 'Clippers'), which is unique across all 30
    NBA teams. The odds feed and team_stats_latest disagree on city prefixes
    ('Los Angeles Clippers' vs 'LA Clippers'), and the server's find_team()
    does a substring match that fails when the input is the longer of the two.
    Matching on the nickname sidesteps every such variant.
    """
    return full_name.strip().split()[-1]


# ── Read the newest stored snapshot for each upcoming game ────────────────────
def fetch_latest_games():
    """
    Pulls every odds snapshot from Supabase and keeps only the most recent row
    per game_id, so each upcoming game appears once with its freshest spread.
    Returns a list of game dicts ordered by tip-off time.
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

    return sorted(latest_by_game.values(), key=lambda game: game["commence_time"])


# ── Ask the running server to predict one matchup ─────────────────────────────
def predict_game(home_team, away_team):
    """
    Posts a single matchup to the prediction server and returns its JSON, or
    None if the server could not resolve one of the teams.
    """
    response = requests.post(
        f"{SERVER_URL}/predict",
        json={
            "home_team": team_lookup_key(home_team),
            "away_team": team_lookup_key(away_team),
        },
        timeout=REQUEST_TIMEOUT,
    )
    result = response.json()
    return None if "error" in result else result


# ── Print one row of the model-vs-Vegas comparison table ──────────────────────
def format_row(game, prediction):
    """Returns a formatted comparison line for a single game."""
    matchup = f"{game['away_team']} @ {game['home_team']}"

    if prediction is None:
        return f"  {matchup:<48}  {'TEAM NOT MATCHED':>28}"

    home_prob = prediction["home_win_prob"]
    vegas = game["home_spread_avg"]
    model_spread = prediction.get("home_margin")

    if model_spread is None:
        # calibrate_spread.py has not been run, so there is no model spread
        # to compare against the Vegas line.
        return f"  {matchup:<48}  home {home_prob:>6.1%}   vegas {vegas:>+6.1f}"

    edge = model_spread - vegas
    return (
        f"  {matchup:<48}  home {home_prob:>6.1%}"
        f"   model {model_spread:>+6.1f}   vegas {vegas:>+6.1f}"
        f"   edge {edge:>+6.1f}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Predicts every upcoming game stored in Supabase against its Vegas line."""
    try:
        requests.get(f"{SERVER_URL}/health", timeout=5).raise_for_status()
    except Exception:
        print(
            f"[ERROR] Cannot reach the prediction server at {SERVER_URL}.\n"
            "Start it first:  python nba-win-prob/server/app.py"
        )
        return

    games = fetch_latest_games()
    if not games:
        print(
            f"[ERROR] No rows in '{TABLE_NAME}'. Populate it first:\n"
            "  python nba-win-prob/update_odds.py"
        )
        return

    print(f"\n  {len(games)} upcoming games from Supabase\n")

    unmatched = 0
    for game in games:
        prediction = predict_game(game["home_team"], game["away_team"])
        if prediction is None:
            unmatched += 1
        print(format_row(game, prediction))

    print()
    if unmatched:
        print(f"  [WARN] {unmatched} game(s) had a team the server could not match.")


if __name__ == "__main__":
    main()
