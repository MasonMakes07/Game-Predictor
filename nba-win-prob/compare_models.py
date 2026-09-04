import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from elo_baseline import run_elo
from evaluation import (
    TEST_SEASON,
    always_home_probs,
    evaluate_model,
    metrics_payload,
    print_comparison,
    print_differences,
    save_json,
    split_by_season,
)
# Imported rather than redefined so there is never a second copy of the
# architecture to drift out of sync (Lessons.md #1).
from train_model import FEATURE_COLS, LABEL_COL, WinProbModel
from train_gbdt import SITUATIONAL_COLS, add_situational_features

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Settings ──────────────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(_ROOT, "data",   "matchup_features.parquet")
MODEL_PATH  = os.path.join(_ROOT, "models", "best_model.pt")
SCALER_PATH = os.path.join(_ROOT, "models", "scaler.npy")
GBDT_PATH   = os.path.join(_ROOT, "models", "gbdt_model.txt")
METRICS_PATH    = os.path.join(_ROOT, "models", "metrics.json")
COMPARISON_PATH = os.path.join(_ROOT, "models", "comparison.json")


# ── Score the saved MLP on a set of games ─────────────────────────────────────
def mlp_probs(frame):
    """
    Loads best_model.pt with the saved scaler and returns its predicted home
    win probability per row, or None if the model has not been trained yet.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        return None

    scaler_mean, scaler_scale = np.load(SCALER_PATH)
    features = frame[FEATURE_COLS].values.astype(np.float32)
    scaled   = (features - scaler_mean) / scaler_scale

    model = WinProbModel()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        return model(
            torch.tensor(scaled, dtype=torch.float32)
        ).squeeze(1).numpy()


# ── Score the saved LightGBM booster on a set of games ────────────────────────
def gbdt_probs(frame):
    """
    Loads gbdt_model.txt and returns its predicted home win probability per
    row, or None if the booster has not been trained yet.
    """
    if not os.path.exists(GBDT_PATH):
        return None

    booster = lgb.Booster(model_file=GBDT_PATH)
    return booster.predict(frame[FEATURE_COLS + SITUATIONAL_COLS])


# ── Assemble every model's predictions for the held-out season ────────────────
def build_predictions():
    """
    Returns a frame of held-out games carrying the true label and one column of
    predicted probabilities per model, aligned on GAME_ID.
    """
    frame = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS + [LABEL_COL])
    # Schedule context is derived across every season before splitting, so a
    # team's rest is measured against its own prior game rather than the first
    # game that happens to fall inside the held-out slice.
    frame = add_situational_features(frame)
    _, _, test_df = split_by_season(frame)

    # Elo must see every season to build its ratings, then we keep only the
    # held-out rows it predicted before ever seeing their results.
    rated, _ = run_elo(frame)
    elo_test = rated[rated["season"] == TEST_SEASON][["GAME_ID", "elo_prob"]]

    predictions = test_df[
        ["GAME_ID", "GAME_DATE", "home_team", "away_team", LABEL_COL]
    ].merge(elo_test, on="GAME_ID", how="left")

    predictions["always_home"] = always_home_probs(predictions[LABEL_COL].values)

    mlp = mlp_probs(test_df)
    if mlp is not None:
        predictions["mlp"] = mlp

    gbdt = gbdt_probs(test_df)
    if gbdt is not None:
        predictions["gbdt"] = gbdt

    return predictions


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Scores every available model on the held-out season and prints a table."""
    predictions = build_predictions()
    y_true      = predictions[LABEL_COL].values

    models = [("Always home", "always_home"), ("Elo", "elo_prob")]
    if "mlp" in predictions.columns:
        models.append(("MLP (temporal)", "mlp"))
    else:
        print("\n  [WARN] models/best_model.pt not found — MLP row skipped.")

    if "gbdt" in predictions.columns:
        models.append(("GBDT (LightGBM)", "gbdt"))
    else:
        print("  [WARN] models/gbdt_model.txt not found — GBDT row skipped.")

    results = [
        evaluate_model(label, y_true, predictions[column].values)
        for label, column in models
    ]
    print_comparison(results)

    best = min(results, key=lambda row: row["log_loss"])
    print(f"  Lowest log loss: {best['name']} ({best['log_loss']:.4f})\n")

    # Marginal intervals above are conservative because every model predicts
    # the same games. The paired test below is the one that decides the gate.
    print_differences(y_true, predictions, models, ("Elo", "elo_prob"))

    # Full table with intervals, for the writeup and the comparison section.
    save_json(
        COMPARISON_PATH,
        {
            "test_season": TEST_SEASON,
            "n_games":     int(len(y_true)),
            "models":      results,
        },
    )
    print(f"\n  Comparison saved to {COMPARISON_PATH}")

    # Keep metrics.json in step with whichever model the server serves, so the
    # dashboard never shows a number no script produced.
    if "mlp" in predictions.columns:
        save_json(
            METRICS_PATH,
            metrics_payload(
                "mlp_pytorch_3layer", y_true, predictions["mlp"].values
            ),
        )
        print(f"  Served-model metrics saved to {METRICS_PATH}\n")


if __name__ == "__main__":
    main()
