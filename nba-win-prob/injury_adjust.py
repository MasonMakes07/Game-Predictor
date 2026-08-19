import time
import unicodedata
from datetime import date

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats

PPG_FLOOR = 70.0   # minimum adjusted ppg to prevent nonsensical values

# A missing player's minutes go to a replacement who still scores, so the team
# loses far less than his raw scoring average. This is the share of his points
# the replacement is assumed to absorb — the team's net loss is the remainder.
# Tunable: raise it to soften injury impact, lower it to sharpen it.
REPLACEMENT_SCORING_SHARE = 0.7


# ── Determine the current NBA season string from today's date ─────────────────
def current_season():
    """Returns the current NBA season string (e.g. '2025-26') based on today."""
    today = date.today()
    # New seasons start in October; before that we're still in last year's
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


CURRENT_SEASON = current_season()


# ── Strip case and accents so "Doncic" matches the stored "Dončić" ────────────
def normalize_name(name):
    """Lowercases a name and removes accents for accent-insensitive matching."""
    decomposed = unicodedata.normalize("NFKD", str(name).strip().lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


# ── Fetch season-wide player stats from nba_api ───────────────────────────────
def fetch_player_stats(season=CURRENT_SEASON):
    """Fetches per-player season totals and returns a cleaned DataFrame."""
    time.sleep(0.6)   # rate limit
    raw = LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="Totals",
        league_id_nullable="00",
    ).get_data_frames()[0]

    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS",
            "PLUS_MINUS"]
    df = raw[keep].copy()
    df = df[df["GP"] > 0].reset_index(drop=True)

    # Normalized once here so lookups never re-normalize 500+ names per player.
    df["name_normalized"] = df["PLAYER_NAME"].map(normalize_name)
    return df


# ── Accent- and case-insensitive substring match on player name ───────────────
def fuzzy_find_player(name, player_stats_df):
    """Returns the first player row whose name contains the query, or None."""
    target = normalize_name(name)
    if not target:
        return None

    # The NBA API stores some names accented ("Luka Dončić") and others plain
    # ("Alperen Sengun"), so both sides are normalized before comparing.
    if "name_normalized" in player_stats_df.columns:
        normalized_names = player_stats_df["name_normalized"]
    else:
        normalized_names = player_stats_df["PLAYER_NAME"].map(normalize_name)

    matches = player_stats_df[
        normalized_names.str.contains(target, regex=False, na=False)
    ]
    return matches.iloc[0] if len(matches) else None


# ── Adjust a team stats row for one or more missing players ───────────────────
def apply_injury_adjustments(team_row, missing_names, player_stats_df):
    """
    Lowers a team's scoring for each missing player, holds points allowed
    fixed, and re-derives the dependent stats. Returns (adjusted_row, log).
    """
    row      = team_row.copy()
    orig_ppg = float(row["ppg"])
    # Points allowed is held fixed: an absent scorer does not improve his
    # team's defense. Captured before the loop so it can never drift.
    opp_ppg  = float(row["opp_ppg"])
    log      = []    # list of human-readable adjustment strings

    for name in missing_names:
        player = fuzzy_find_player(name, player_stats_df)

        if player is None:
            log.append(f"  [INJURY] Player not found: '{name}' — skipped")
            continue

        raw_ppg_loss = float(player["PTS"]) / float(player["GP"])
        net_ppg_loss = raw_ppg_loss * (1.0 - REPLACEMENT_SCORING_SHARE)

        row["ppg"] = max(float(row["ppg"]) - net_ppg_loss, PPG_FLOOR)

        log.append(
            f"  [INJURY] {player['PLAYER_NAME']} ({player['TEAM_ABBREVIATION']})"
            f"  ppg -{net_ppg_loss:.1f} (raw -{raw_ppg_loss:.1f})"
        )

    # build_features.py treats plus_minus_avg as measured and derives
    # opp_ppg from it. Here the causality is reversed — scoring changed and
    # defense did not — so the margin is re-derived instead. Either way the
    # identity plus_minus_avg == ppg - opp_ppg holds, which is what the model
    # was trained on.
    row["opp_ppg"]        = opp_ppg
    row["plus_minus_avg"] = float(row["ppg"]) - opp_ppg

    # Scale off_efficiency proportionally to the PPG drop
    if orig_ppg > 0:
        row["off_efficiency"] = float(row["off_efficiency"]) * (
            float(row["ppg"]) / orig_ppg
        )

    return row, log
