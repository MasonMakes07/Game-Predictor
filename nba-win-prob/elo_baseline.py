import os

import numpy as np
import pandas as pd

from evaluation import (
    TEST_SEASON,
    always_home_probs,
    evaluate_model,
    print_comparison,
    season_start_year,
)

# ── Settings ──────────────────────────────────────────────────────────────────
_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "matchup_features.parquet")

START_RATING     = 1500.0
K_FACTOR         = 20.0    # how far a rating moves per game
HOME_ADVANTAGE   = 100.0   # Elo points added to the home side, ~0.64 win prob
SEASON_REGRESSION = 0.25   # fraction pulled back toward the mean each offseason


# ── Convert an Elo rating gap into a win probability ──────────────────────────
def elo_expected(rating_diff):
    """Returns P(win) for a team rated rating_diff points above its opponent."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


# ── Walk every game in date order, predicting before updating ─────────────────
def run_elo(frame):
    """
    Rates every team causally: each game is predicted from ratings that reflect
    only prior games, then those ratings are updated with the result. Returns
    the frame with an elo_prob column plus the final ratings dict.
    """
    games = frame.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["season"]    = season_start_year(games["GAME_DATE"])
    # GAME_ID breaks ties so the ordering is deterministic across runs.
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    ratings        = {}
    current_season = None
    probabilities  = np.zeros(len(games), dtype=float)

    for position, game in enumerate(games.itertuples(index=False)):
        # Between seasons, pull every rating partway back to the mean: rosters
        # turn over and last year's gaps overstate this year's.
        if current_season is not None and game.season != current_season:
            for team in ratings:
                ratings[team] += SEASON_REGRESSION * (START_RATING - ratings[team])
        current_season = game.season

        home_rating = ratings.setdefault(game.home_team, START_RATING)
        away_rating = ratings.setdefault(game.away_team, START_RATING)

        home_prob = elo_expected(home_rating + HOME_ADVANTAGE - away_rating)
        probabilities[position] = home_prob

        # Update only after predicting, so no game informs its own forecast.
        adjustment = K_FACTOR * (game.home_win - home_prob)
        ratings[game.home_team] = home_rating + adjustment
        ratings[game.away_team] = away_rating - adjustment

    games["elo_prob"] = probabilities
    return games, ratings


# ── Standalone run: score Elo on the held-out season ──────────────────────────
def main():
    """Rates all seasons, then scores Elo against the trivial baseline."""
    frame = pd.read_parquet(DATA_PATH)
    rated, final_ratings = run_elo(frame)

    held_out = rated[rated["season"] == TEST_SEASON]
    y_true   = held_out["home_win"].values

    results = [
        evaluate_model("Always home", y_true, always_home_probs(y_true)),
        evaluate_model("Elo", y_true, held_out["elo_prob"].values),
    ]
    print_comparison(results)

    # Sanity check a human can judge instantly: are these plausibly the best
    # teams? If not, the implementation is wrong regardless of the metrics.
    print("  Final Elo ratings — top 10")
    ranked = sorted(final_ratings.items(), key=lambda pair: pair[1], reverse=True)
    for rank, (team, rating) in enumerate(ranked[:10], start=1):
        print(f"    {rank:>2}. {team:<26} {rating:7.1f}")

    print("\n  Bottom 5")
    for rank, (team, rating) in enumerate(ranked[-5:], start=len(ranked) - 4):
        print(f"    {rank:>2}. {team:<26} {rating:7.1f}")
    print()


if __name__ == "__main__":
    main()
