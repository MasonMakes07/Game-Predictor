import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

# ── Settings (must mirror train_model.py so the split is reproduced) ──────────
_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(_ROOT, "data",   "matchup_features.parquet")
MODEL_DIR    = os.path.join(_ROOT, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.pt")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.npy")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

FEATURE_COLS = [
    "win_pct_home",        "ppg_home",
    "opp_ppg_home",        "plus_minus_avg_home",   "last10_win_pct_home",
    "win_pct_away",        "ppg_away",
    "opp_ppg_away",        "plus_minus_avg_away",   "last10_win_pct_away",
    "home_split_win_pct",  "away_split_win_pct",    "net_rtg_diff",
    "home_team_away_ppg",  "away_team_home_ppg",
    "off_efficiency_home", "off_efficiency_away",
    "tov_rate_home",       "tov_rate_away",
]
LABEL_COL   = "home_win"
N_FEATURES  = len(FEATURE_COLS)
VAL_SPLIT   = 0.2
RANDOM_SEED = 42


# ── Model: must match train_model.py exactly ──────────────────────────────────
class WinProbModel(nn.Module):
    """Three hidden layers with BatchNorm, ReLU, and Dropout; outputs a win prob."""

    def __init__(self, input_dim=N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Runs the forward pass and returns P(home win)."""
        return self.net(x)


# ── Rebuild the exact validation split train_model.py used ────────────────────
def load_validation_split():
    """
    Reproduces train_model.py's split (same seed, same ratio, same dropna) and
    scales it with the saved scaler. Returns (X_val_scaled, y_val).
    """
    frame = pd.read_parquet(DATA_PATH).dropna(
        subset=FEATURE_COLS + [LABEL_COL]
    )
    features = frame[FEATURE_COLS].values
    labels   = frame[LABEL_COL].values

    _, x_val, _, y_val = train_test_split(
        features, labels, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    # scaler.npy holds [mean_, scale_] saved by train_model.py, so the same
    # transform is applied here without refitting on validation data.
    scaler_mean, scaler_scale = np.load(SCALER_PATH)
    return (x_val - scaler_mean) / scaler_scale, y_val


# ── Score the trained model on the validation split ───────────────────────────
def compute_metrics():
    """
    Loads best_model.pt and scores it on the held-out split, returning a dict
    of accuracy, log-loss, Brier score, and the sample count behind them.
    """
    x_val, y_val = load_validation_split()

    model = WinProbModel()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        probs = model(
            torch.tensor(x_val, dtype=torch.float32)
        ).squeeze(1).numpy()

    return {
        "accuracy":     round(float(accuracy_score(y_val, (probs >= 0.5))), 4),
        "log_loss":     round(float(log_loss(y_val, probs)), 4),
        "brier_score":  round(float(brier_score_loss(y_val, probs)), 4),
        "val_samples":  int(len(y_val)),
    }


# ── Standalone run ────────────────────────────────────────────────────────────
def main():
    """Scores the saved model and writes models/metrics.json for the server."""
    metrics = compute_metrics()

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"[EVAL] Accuracy    : {metrics['accuracy']:.1%}")
    print(f"[EVAL] Log-loss    : {metrics['log_loss']:.4f}")
    print(f"[EVAL] Brier score : {metrics['brier_score']:.4f}")
    print(f"[EVAL] Val samples : {metrics['val_samples']}")
    print(f"[EVAL] Saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
