import os
import sys

import pandas as pd

from evaluation import (
    TEST_SEASON,
    bootstrap_difference,
    evaluate_model,
    print_comparison,
    split_by_season,
)
from train_gbdt import (
    DATA_PATH,
    SITUATIONAL_COLS,
    add_situational_features,
    train_booster,
)
from train_model import FEATURE_COLS, LABEL_COL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Train one booster on a given feature set and score the held-out season ────
def score_feature_set(label, feature_cols, train_df, val_df, test_df):
    """
    Fits a booster on one feature set and returns its held-out probabilities
    alongside the scored result. Nothing is written to disk, so the ablation
    never overwrites the model compare_models.py serves.
    """
    print(f"\n[ABLATION] {label} — {len(feature_cols)} features")
    booster = train_booster(train_df, val_df, feature_cols)

    probs = booster.predict(
        test_df[feature_cols], num_iteration=booster.best_iteration
    )
    return probs, evaluate_model(label, test_df[LABEL_COL].values, probs)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """
    Isolates whether the GBDT's edge comes from the model class or the added
    schedule features, by training the same booster on each feature set.
    """
    frame = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS + [LABEL_COL])
    frame = add_situational_features(frame)
    train_df, val_df, test_df = split_by_season(frame)

    engineered_probs, engineered = score_feature_set(
        "GBDT — 19 engineered only", FEATURE_COLS,
        train_df, val_df, test_df,
    )
    full_probs, full = score_feature_set(
        "GBDT — 19 + 7 situational", FEATURE_COLS + SITUATIONAL_COLS,
        train_df, val_df, test_df,
    )

    print_comparison([engineered, full])

    y_true = test_df[LABEL_COL].values
    mean_diff, low, high, wins = bootstrap_difference(
        y_true, full_probs, engineered_probs, metric="log_loss"
    )
    verdict = "significant" if (low > 0 or high < 0) else "NOT significant"

    print(f"  Paired difference — situational features added vs removed")
    print(f"    log loss change : {mean_diff:+.4f}")
    print(f"    95% interval    : [{low:+.4f}, {high:+.4f}]  → {verdict}")
    print(f"    helps in        : {wins:.0%} of bootstrap resamples\n")

    if not (low > 0 or high < 0):
        print("  Read: the schedule features do not measurably help. The GBDT's\n"
              "  edge over the MLP and Elo comes from the model class, not from\n"
              "  rest days or back-to-backs.\n")


if __name__ == "__main__":
    main()
