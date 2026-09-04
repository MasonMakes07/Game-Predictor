import os
import sys

import matplotlib
# Non-interactive backend so the script runs unattended and just writes PNGs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from compare_models import gbdt_probs
from evaluation import (
    TEST_SEASON,
    VAL_SEASON,
    bootstrap_difference,
    evaluate_model,
    print_comparison,
    save_json,
    split_by_season,
)
from train_gbdt import DATA_PATH, add_situational_features
from train_model import FEATURE_COLS, LABEL_COL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Output ────────────────────────────────────────────────────────────────────
_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
RELIABILITY_FIG   = os.path.join(MODEL_DIR, "reliability.png")
CALIBRATION_JSON  = os.path.join(MODEL_DIR, "calibration.json")

# ── Palette (validated categorical slots 1-3) ─────────────────────────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_SOFT  = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"
METHOD_COLORS = {
    "Raw GBDT": "#2a78d6",
    "Platt":    "#eb6834",
    "Isotonic": "#1baf7a",
}

N_BINS        = 10
MIN_BIN_GAMES = 20    # thinner bins are noise, not calibration signal
PROB_EPS      = 1e-6

plt.rcParams.update({
    "font.family":      ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor":   SURFACE,
    "text.color":       INK,
})


# ── Fit Platt scaling on the validation season ────────────────────────────────
def fit_platt(val_probs, val_labels):
    """
    Fits a one-parameter logistic correction on the model's log-odds. Platt
    scaling can only stretch or shift the curve, so it cannot invent structure
    the model did not already have.
    """
    clipped = np.clip(val_probs, PROB_EPS, 1 - PROB_EPS)
    logits  = np.log(clipped / (1 - clipped)).reshape(-1, 1)

    model = LogisticRegression(C=1e10, solver="lbfgs")
    model.fit(logits, val_labels)

    def transform(probs):
        """Applies the fitted logistic correction to new probabilities."""
        safe = np.clip(probs, PROB_EPS, 1 - PROB_EPS)
        return model.predict_proba(
            np.log(safe / (1 - safe)).reshape(-1, 1)
        )[:, 1]

    return transform


# ── Fit isotonic regression on the validation season ──────────────────────────
def fit_isotonic(val_probs, val_labels):
    """
    Fits a free-form monotonic correction. More flexible than Platt, and more
    prone to overfitting on a single season, which is why both are reported.
    """
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(val_probs, val_labels)
    return model.predict


# ── Binned reliability plus a single summary number ───────────────────────────
def reliability_bins(probs, labels, n_bins=N_BINS):
    """
    Returns (bin_centers, observed_rates, bin_counts) for bins holding at least
    MIN_BIN_GAMES games, so sparse tails do not masquerade as miscalibration.
    """
    edges   = np.linspace(0, 1, n_bins + 1)
    centers, observed, counts = [], [], []

    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (probs >= low) & (probs < high)
        if in_bin.sum() < MIN_BIN_GAMES:
            continue
        centers.append(float(probs[in_bin].mean()))
        observed.append(float(labels[in_bin].mean()))
        counts.append(int(in_bin.sum()))

    return np.array(centers), np.array(observed), np.array(counts)


# ── Expected and maximum calibration error ────────────────────────────────────
def calibration_error(probs, labels, n_bins=N_BINS):
    """
    Returns (expected_calibration_error, max_deviation). ECE weights each bin's
    gap by how many games fall in it; max deviation is the worst single decile.
    """
    centers, observed, counts = reliability_bins(probs, labels, n_bins)
    if len(centers) == 0:
        return float("nan"), float("nan")

    gaps = np.abs(centers - observed)
    return (
        float(np.sum(gaps * counts) / counts.sum()),
        float(gaps.max()),
    )


# ── Reliability diagram: raw against both corrections ─────────────────────────
def plot_reliability(curves, labels, n_games):
    """Draws each method's reliability curve against the perfect diagonal."""
    figure, axis = plt.subplots(figsize=(7.4, 6.6))
    axis.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)
    axis.annotate("perfectly calibrated", xy=(0.14, 0.165), rotation=38,
                  color=INK_MUTED, fontsize=8.5, ha="center")

    for name, probs in curves.items():
        centers, observed, _ = reliability_bins(probs, labels)
        axis.plot(centers, observed, "-o", color=METHOD_COLORS[name],
                  linewidth=2, markersize=8, markeredgecolor=SURFACE,
                  markeredgewidth=2, label=name, zorder=3)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    axis.set_xlabel("predicted win probability", color=INK_SOFT, fontsize=9,
                    labelpad=6)
    axis.set_ylabel("observed win rate", color=INK_SOFT, fontsize=9, labelpad=6)
    axis.set_title(
        f"Reliability — held-out {TEST_SEASON}-{str(TEST_SEASON + 1)[-2:]} "
        f"({n_games:,} games)",
        color=INK, fontsize=12, fontweight="bold", loc="left", pad=12,
    )

    legend = axis.legend(frameon=False, fontsize=9.5, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    figure.tight_layout()
    figure.savefig(RELIABILITY_FIG, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"[CALIBRATION] Reliability diagram saved to {RELIABILITY_FIG}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """
    Fits Platt and isotonic corrections on the validation season, applies them
    to the held-out season, and reports whether either actually helps.
    """
    frame = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLS + [LABEL_COL])
    frame = add_situational_features(frame)
    _, val_df, test_df = split_by_season(frame)

    val_probs  = gbdt_probs(val_df)
    test_probs = gbdt_probs(test_df)
    if val_probs is None or test_probs is None:
        print("[CALIBRATION] models/gbdt_model.txt not found — run train_gbdt.py")
        return

    val_labels  = val_df[LABEL_COL].values
    test_labels = test_df[LABEL_COL].values

    print(f"[CALIBRATION] Fitting on season {VAL_SEASON} "
          f"({len(val_df):,} games), applying to {TEST_SEASON} "
          f"({len(test_df):,} games)\n")

    # Both corrections are fit on validation only — fitting on the held-out
    # season would make the reported calibration meaningless.
    curves = {
        "Raw GBDT": test_probs,
        "Platt":    fit_platt(val_probs, val_labels)(test_probs),
        "Isotonic": fit_isotonic(val_probs, val_labels)(test_probs),
    }

    results = [
        evaluate_model(name, test_labels, probs) for name, probs in curves.items()
    ]
    print_comparison(results)

    print(f"  {'Method':<14}{'ECE':>10}{'Max decile gap':>18}")
    print(f"  {'-' * 42}")
    summary = {}
    for name, probs in curves.items():
        ece, max_gap = calibration_error(probs, test_labels)
        summary[name] = {"ece": round(ece, 4), "max_decile_gap": round(max_gap, 4)}
        print(f"  {name:<14}{ece:>10.4f}{max_gap:>18.4f}")
    print()

    # Did either correction actually change anything on held-out data?
    for name in ("Platt", "Isotonic"):
        mean_diff, low, high, wins = bootstrap_difference(
            test_labels, curves[name], curves["Raw GBDT"], metric="log_loss"
        )
        verdict = "significant" if (low > 0 or high < 0) else "not significant"
        print(f"  {name} vs raw: {mean_diff:+.4f} log loss "
              f"[{low:+.4f}, {high:+.4f}] — {verdict} "
              f"(helps in {wins:.0%} of resamples)")

    plot_reliability(curves, test_labels, len(test_labels))

    save_json(CALIBRATION_JSON, {
        "test_season":  TEST_SEASON,
        "fit_season":   VAL_SEASON,
        "n_games":      int(len(test_labels)),
        "calibration":  summary,
        "models":       results,
    })
    print(f"[CALIBRATION] Summary saved to {CALIBRATION_JSON}\n")


if __name__ == "__main__":
    main()
