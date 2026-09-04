import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# ── Settings ──────────────────────────────────────────────────────────────────
# Seasons are labelled by the calendar year they start in, so 2025 means the
# 2025-26 season. The most recent season is held out entirely; the one before
# it validates (early stopping, calibration fitting).
TEST_SEASON = 2025
VAL_SEASON  = 2024
BOOTSTRAP_N = 1000
RANDOM_SEED = 42
PROB_EPS    = 1e-6   # keeps log loss finite if a model emits exactly 0 or 1


# ── Label each game with the season it belongs to ─────────────────────────────
def season_start_year(dates):
    """
    Returns each date's season starting year. NBA seasons run October to June,
    so anything before October belongs to the previous year's season
    (2026-04-12 is part of the 2025-26 season).
    """
    dates = pd.to_datetime(dates)
    return dates.dt.year.where(dates.dt.month >= 10, dates.dt.year - 1)


# ── Split a matchup frame into train / validation / test by season ────────────
def split_by_season(frame, val_season=VAL_SEASON, test_season=TEST_SEASON):
    """
    Splits chronologically: seasons before val_season train, val_season
    validates, test_season is held out. Raises if the result is not strictly
    ordered in time. Returns (train, val, test).
    """
    frame = frame.copy()
    frame["season"] = season_start_year(frame["GAME_DATE"])

    train = frame[frame["season"] < val_season]
    val   = frame[frame["season"] == val_season]
    test  = frame[frame["season"] == test_season]

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Empty split — train={len(train)} val={len(val)} test={len(test)}. "
            f"Check that seasons {val_season} and {test_season} exist."
        )

    # Guards the exact bug this module exists to fix: no training game may fall
    # on or after the first held-out game.
    latest_train = pd.to_datetime(train["GAME_DATE"]).max()
    first_test   = pd.to_datetime(test["GAME_DATE"]).min()
    if latest_train >= first_test:
        raise ValueError(
            f"Temporal split violated: latest training game {latest_train} is "
            f"not before first test game {first_test}."
        )

    return train, val, test


# ── Core metrics for a set of predicted win probabilities ─────────────────────
def score_predictions(y_true, probs):
    """
    Returns accuracy, log loss, and Brier score for predicted probabilities of
    a home win. Probabilities are clipped so log loss stays finite.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs  = np.clip(np.asarray(probs, dtype=float), PROB_EPS, 1 - PROB_EPS)

    return {
        "accuracy": float(accuracy_score(y_true, (probs >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "brier":    float(brier_score_loss(y_true, probs)),
    }


# ── Bootstrap confidence intervals for each metric ────────────────────────────
def bootstrap_metrics(y_true, probs, n_iterations=BOOTSTRAP_N, seed=RANDOM_SEED):
    """
    Resamples the evaluation set with replacement to produce 95% intervals.
    Differences smaller than these intervals are noise, not findings.
    Returns {metric: (low, high)}.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs  = np.asarray(probs, dtype=float)

    rng     = np.random.default_rng(seed)
    n_games = len(y_true)
    samples = {"accuracy": [], "log_loss": [], "brier": []}

    for _ in range(n_iterations):
        idx = rng.integers(0, n_games, size=n_games)
        # A resample containing only one outcome class cannot be scored. With
        # ~55% home wins over 1,000+ games this effectively never fires.
        if len(np.unique(y_true[idx])) < 2:
            continue
        for name, value in score_predictions(y_true[idx], probs[idx]).items():
            samples[name].append(value)

    return {
        name: (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
        for name, values in samples.items()
    }


# ── Paired bootstrap on the difference between two models ─────────────────────
def bootstrap_difference(y_true, probs_a, probs_b, metric="log_loss",
                         n_iterations=BOOTSTRAP_N, seed=RANDOM_SEED):
    """
    Resamples games and scores BOTH models on the same resample, returning the
    distribution of (a - b) for one metric. Because the two models predict the
    same games, their errors are correlated — this paired test is far more
    powerful than checking whether marginal intervals overlap.

    Returns (mean_difference, low, high, share_of_resamples_where_a_beats_b).
    """
    y_true  = np.asarray(y_true, dtype=int)
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)

    rng         = np.random.default_rng(seed)
    n_games     = len(y_true)
    differences = []

    for _ in range(n_iterations):
        idx = rng.integers(0, n_games, size=n_games)
        if len(np.unique(y_true[idx])) < 2:
            continue
        score_a = score_predictions(y_true[idx], probs_a[idx])[metric]
        score_b = score_predictions(y_true[idx], probs_b[idx])[metric]
        differences.append(score_a - score_b)

    differences = np.array(differences)
    # For log loss and Brier, lower is better, so a wins when the difference is
    # negative. For accuracy the direction flips.
    a_wins = (
        float(np.mean(differences > 0)) if metric == "accuracy"
        else float(np.mean(differences < 0))
    )

    return (
        float(np.mean(differences)),
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
        a_wins,
    )


# ── Render pairwise comparisons against one reference model ───────────────────
def print_differences(y_true, predictions, models, reference, metric="log_loss"):
    """
    Prints each model's paired difference versus a reference model. An interval
    that excludes zero is a real difference; one that spans zero is not.
    """
    print(f"  Paired difference in {metric} vs {reference[0]} "
          f"(negative = better than {reference[0]})\n")
    print(f"  {'Model':<24}{'Difference':<16}{'95% interval':<26}{'Beats ref':<10}")
    print(f"  {'-' * 24}{'-' * 16}{'-' * 26}{'-' * 10}")

    for label, column in models:
        if column == reference[1]:
            continue
        mean_diff, low, high, wins = bootstrap_difference(
            y_true, predictions[column].values,
            predictions[reference[1]].values, metric=metric,
        )
        verdict = "yes" if (low > 0 or high < 0) else "not sig."
        print(f"  {label:<24}{mean_diff:<+16.4f}"
              f"{f'[{low:+.4f}, {high:+.4f}]':<26}{f'{wins:.0%} {verdict}':<10}")
    print()


# ── Score one model and package point estimates with intervals ────────────────
def evaluate_model(name, y_true, probs, n_iterations=BOOTSTRAP_N):
    """Returns point estimates plus bootstrap intervals for a single model."""
    return {
        "name": name,
        "n":    len(y_true),
        **score_predictions(y_true, probs),
        "ci":   bootstrap_metrics(y_true, probs, n_iterations),
    }


# ── Render the comparison table ───────────────────────────────────────────────
def print_comparison(results):
    """
    Prints one row per model with 95% intervals, so overlapping intervals make
    it obvious when two models are not actually distinguishable.
    """
    print(f"\n  Held-out season — n = {results[0]['n']:,} games\n")
    print(f"  {'Model':<28}{'Accuracy':<26}{'Log loss':<26}{'Brier':<26}")
    print(f"  {'-' * 28}{'-' * 26}{'-' * 26}{'-' * 26}")

    for row in results:
        cells = []
        for metric, fmt in (
            ("accuracy", ".1%"), ("log_loss", ".4f"), ("brier", ".4f")
        ):
            low, high = row["ci"][metric]
            cells.append(
                f"{format(row[metric], fmt)} "
                f"[{format(low, fmt)}, {format(high, fmt)}]"
            )
        print(f"  {row['name']:<28}{cells[0]:<26}{cells[1]:<26}{cells[2]:<26}")
    print()


# ── Persist metrics to disk ───────────────────────────────────────────────────
def save_json(path, payload):
    """Writes a metrics payload as indented JSON, creating the folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


# ── Build the payload the server reads for /health ────────────────────────────
def metrics_payload(model_name, y_true, probs, test_season=TEST_SEASON):
    """
    Returns the served-model metrics dict written to models/metrics.json. Key
    names are kept stable because server/app.py and the dashboard sidebar read
    them directly.
    """
    scores = score_predictions(y_true, probs)
    return {
        "accuracy":    round(scores["accuracy"], 4),
        "log_loss":    round(scores["log_loss"], 4),
        "brier_score": round(scores["brier"], 4),
        "val_samples": int(len(y_true)),
        "model":       model_name,
        "eval_split":  f"held-out season {test_season}-{str(test_season + 1)[-2:]}",
    }


# ── Baseline that predicts the home team every time ───────────────────────────
def always_home_probs(y_true, home_win_rate=None):
    """
    Returns a constant probability for every game — the trivial baseline any
    real model must beat. Defaults to the observed home win rate, which is the
    best possible constant predictor by log loss.
    """
    if home_win_rate is None:
        home_win_rate = float(np.mean(y_true))
    return np.full(len(y_true), home_win_rate, dtype=float)
