import os
import sys

import lightgbm as lgb
import matplotlib
# Non-interactive backend so the script runs unattended and just writes PNGs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import (
    TEST_SEASON,
    VAL_SEASON,
    score_predictions,
    season_start_year,
    split_by_season,
)
from train_model import FEATURE_COLS, LABEL_COL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Settings ──────────────────────────────────────────────────────────────────
_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(_ROOT, "data",   "matchup_features.parquet")
MODEL_DIR  = os.path.join(_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gbdt_model.txt")
IMPORTANCE_FIG = os.path.join(MODEL_DIR, "gbdt_importance.png")

RANDOM_SEED    = 42
EARLY_STOPPING = 100
MAX_ROUNDS     = 3000
DENSE_WINDOW   = "4D"   # window for the third-game-in-four-nights flag

# Situational columns added on top of the 19 engineered features. Every one is
# known before tip-off, so none of them leak.
SITUATIONAL_COLS = [
    "home_rest_days", "away_rest_days", "rest_advantage",
    "home_back_to_back", "away_back_to_back",
    "home_third_in_four", "away_third_in_four",
]

LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "learning_rate":    0.02,
    "num_leaves":       15,     # small: ~6.6k training rows overfit fast
    "min_data_in_leaf": 60,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.80,
    "bagging_freq":     1,
    "lambda_l2":        5.0,
    "verbosity":        -1,
    "seed":             RANDOM_SEED,
}


# ── Derive schedule context for every team-game ───────────────────────────────
def build_team_schedule(frame):
    """
    Explodes each matchup into two team-game rows and computes rest days, the
    back-to-back flag, and the third-game-in-four-nights flag per team. Rest is
    measured within a season, so an opener has no rest value rather than a
    meaningless gap spanning the offseason.
    """
    home = frame[["GAME_ID", "GAME_DATE", "season", "home_team"]].rename(
        columns={"home_team": "team"}
    )
    away = frame[["GAME_ID", "GAME_DATE", "season", "away_team"]].rename(
        columns={"away_team": "team"}
    )
    schedule = pd.concat([home, away], ignore_index=True)
    schedule["GAME_DATE"] = pd.to_datetime(schedule["GAME_DATE"])
    schedule = schedule.sort_values(["team", "season", "GAME_DATE"])

    grouped = schedule.groupby(["team", "season"], sort=False)
    schedule["rest_days"]    = grouped["GAME_DATE"].diff().dt.days
    schedule["back_to_back"] = (schedule["rest_days"] == 1).astype(float)

    # Games inside a trailing 4-day window, current game included: a value of 3
    # means this is the third game in four nights.
    density = (
        schedule.set_index("GAME_DATE")
        .groupby(["team", "season"], sort=False)["GAME_ID"]
        .rolling(DENSE_WINDOW)
        .count()
        .reset_index(name="games_in_window")
    )
    schedule = schedule.merge(
        density, on=["team", "season", "GAME_DATE"], how="left"
    )
    schedule["third_in_four"] = (schedule["games_in_window"] >= 3).astype(float)

    return schedule[
        ["GAME_ID", "team", "rest_days", "back_to_back", "third_in_four"]
    ]


# ── Attach the schedule context back onto each matchup row ────────────────────
def add_situational_features(frame):
    """
    Returns the matchup frame with home/away rest, back-to-back, and dense-week
    flags joined on, plus the home-minus-away rest advantage.
    """
    frame = frame.copy()
    frame["season"] = season_start_year(frame["GAME_DATE"])
    schedule = build_team_schedule(frame)

    for side in ("home", "away"):
        side_schedule = schedule.rename(
            columns={
                "team":          f"{side}_team",
                "rest_days":     f"{side}_rest_days",
                "back_to_back":  f"{side}_back_to_back",
                "third_in_four": f"{side}_third_in_four",
            }
        )
        frame = frame.merge(
            side_schedule, on=["GAME_ID", f"{side}_team"], how="left"
        )

    frame["rest_advantage"] = frame["home_rest_days"] - frame["away_rest_days"]
    return frame


# ── Train the booster with early stopping on the validation season ────────────
def train_booster(train_df, val_df, feature_cols):
    """
    Fits LightGBM, stopping once the validation season's log loss stops
    improving. Returns the trained booster.
    """
    train_set = lgb.Dataset(
        train_df[feature_cols], label=train_df[LABEL_COL].values
    )
    val_set = lgb.Dataset(
        val_df[feature_cols], label=val_df[LABEL_COL].values,
        reference=train_set,
    )

    booster = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    best_val = booster.best_score["val"]["binary_logloss"]
    print(f"\n[GBDT] Best iteration: {booster.best_iteration} "
          f"(val log loss {best_val:.4f})")
    return booster


# ── Feature importance chart ──────────────────────────────────────────────────
def plot_importance(booster, feature_cols, top_n=20):
    """Saves a horizontal bar chart of the most-used features by total gain."""
    gains = booster.feature_importance(importance_type="gain")
    order = np.argsort(gains)[-top_n:]
    names = [feature_cols[i] for i in order]

    figure, axis = plt.subplots(figsize=(9, 7.5))
    figure.patch.set_facecolor("#fcfcfb")
    axis.set_facecolor("#fcfcfb")
    # One series, one colour: shading bars by length would double-encode the
    # value the bar already shows.
    axis.barh(range(len(order)), gains[order], color="#2a78d6", height=0.62)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(names, fontsize=9, color="#52514e")
    axis.set_xlabel("total gain", fontsize=9, color="#52514e", labelpad=6)
    axis.set_title(f"GBDT feature importance — top {len(order)}",
                   fontsize=12, fontweight="bold", color="#0b0b0b",
                   loc="left", pad=12)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#c3c2b7")
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors="#898781", labelsize=9, length=3, width=0.8)
    axis.xaxis.grid(True, color="#e1e0d9", linewidth=0.8)
    axis.set_axisbelow(True)

    figure.tight_layout()
    figure.savefig(IMPORTANCE_FIG, dpi=150, facecolor="#fcfcfb")
    plt.close(figure)
    print(f"[GBDT] Importance chart saved to {IMPORTANCE_FIG}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Trains the booster on the temporal split and scores the held-out season."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    frame = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS + [LABEL_COL])
    frame = add_situational_features(frame)

    feature_cols = FEATURE_COLS + SITUATIONAL_COLS
    train_df, val_df, test_df = split_by_season(frame)

    print(f"[GBDT] {len(feature_cols)} features ({len(FEATURE_COLS)} engineered "
          f"+ {len(SITUATIONAL_COLS)} situational)")
    print(f"  Train : {len(train_df):,}  (seasons before {VAL_SEASON})")
    print(f"  Val   : {len(val_df):,}  (season {VAL_SEASON})")
    print(f"  Test  : {len(test_df):,}  (season {TEST_SEASON}, held out)\n")

    booster = train_booster(train_df, val_df, feature_cols)
    booster.save_model(MODEL_PATH, num_iteration=booster.best_iteration)
    print(f"[GBDT] Model saved to {MODEL_PATH}")

    probs  = booster.predict(test_df[feature_cols],
                             num_iteration=booster.best_iteration)
    scores = score_predictions(test_df[LABEL_COL].values, probs)

    print(f"\n[SUMMARY] Held-out season {TEST_SEASON}-"
          f"{str(TEST_SEASON + 1)[-2:]}  ({len(test_df):,} games)")
    print(f"  Accuracy    : {scores['accuracy']:.1%}")
    print(f"  Log-loss    : {scores['log_loss']:.4f}")
    print(f"  Brier score : {scores['brier']:.4f}")

    plot_importance(booster, feature_cols)
    print("\n  Compare against Elo and the MLP:")
    print("    python nba-win-prob/compare_models.py\n")


if __name__ == "__main__":
    main()
