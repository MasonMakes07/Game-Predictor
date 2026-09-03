import os
import sys

import matplotlib
# Non-interactive backend so the script runs unattended and just writes PNGs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from compare_models import build_predictions
from evaluation import TEST_SEASON, bootstrap_difference, evaluate_model
from train_model import LABEL_COL

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Output ────────────────────────────────────────────────────────────────────
_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(_ROOT, "models")
METRICS_FIG     = os.path.join(MODEL_DIR, "evaluation_metrics.png")
CALIBRATION_FIG = os.path.join(MODEL_DIR, "evaluation_calibration.png")

# ── Palette ───────────────────────────────────────────────────────────────────
# Categorical slots 1-3 from the validated reference palette. Colour follows the
# model, never its rank, so a model keeps its hue across every panel.
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_SOFT  = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"

MODEL_COLORS = {
    "Always home":    "#2a78d6",   # slot 1, blue
    "Elo":            "#eb6834",   # slot 2, orange
    "MLP (temporal)": "#1baf7a",   # slot 3, aqua
}
# Columns in the predictions frame, in display order.
MODEL_COLUMNS = [
    ("Always home",    "always_home"),
    ("Elo",            "elo_prob"),
    ("MLP (temporal)", "mlp"),
]
REFERENCE = ("Elo", "elo_prob")

MIN_BIN_GAMES = 20   # reliability bins thinner than this are noise, not signal

plt.rcParams.update({
    "font.family":     ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor":   SURFACE,
    "text.color":       INK,
})


# ── Legend proxies so one key can cover every panel in a figure ───────────────
def model_legend_handles(available):
    """
    Builds dot-and-line proxies matching the marks the panels draw, so a single
    figure-level key explains the colour of every panel at once.
    """
    return [
        Line2D([0], [0], color=MODEL_COLORS[label], marker="o", linewidth=2,
               markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
               label=label)
        for label, _ in available
    ]


# ── Shared axis chrome: recessive hairlines, no top/right spines ──────────────
def style_axis(axis, xlabel=None, ylabel=None, title=None):
    """Applies the recessive grid and axis treatment used by every panel."""
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
        axis.spines[side].set_linewidth(0.8)

    axis.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    if title:
        axis.set_title(title, color=INK, fontsize=11, fontweight="bold",
                       pad=10, loc="left")
    if xlabel:
        axis.set_xlabel(xlabel, color=INK_SOFT, fontsize=9, labelpad=6)
    if ylabel:
        axis.set_ylabel(ylabel, color=INK_SOFT, fontsize=9, labelpad=6)


# ── One metric panel: point estimate with its 95% interval per model ──────────
def plot_metric(axis, results, metric, title, value_format, lower_is_better):
    """
    Draws a dot-and-interval panel for one metric. Bars are wrong here: log loss
    has no meaningful zero, so a dot plot keeps the axis on the range that
    actually differs between models.
    """
    names  = [row["name"] for row in results]
    y_positions = np.arange(len(results))[::-1]

    for y_position, row in zip(y_positions, results):
        low, high = row["ci"][metric]
        color     = MODEL_COLORS[row["name"]]

        axis.plot([low, high], [y_position, y_position], color=color,
                  linewidth=2, solid_capstyle="round", zorder=2)
        # 2px surface ring keeps the marker legible where it meets its interval.
        axis.plot(row[metric], y_position, "o", color=color, markersize=9,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        # Value labels wear text ink, never the series colour. They also supply
        # the "relief" the palette validator requires for the low-contrast slot.
        axis.annotate(format(row[metric], value_format),
                      (row[metric], y_position), textcoords="offset points",
                      xytext=(0, 13), ha="center", color=INK, fontsize=9,
                      fontweight="bold")

    axis.set_yticks(y_positions)
    axis.set_yticklabels(names, color=INK_SOFT, fontsize=9.5)
    axis.set_ylim(-0.7, len(results) - 0.3)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)

    # Pad the axis so neither the interval ends nor the value labels above them
    # collide with the panel edge.
    lows  = [row["ci"][metric][0] for row in results]
    highs = [row["ci"][metric][1] for row in results]
    span  = max(highs) - min(lows)
    axis.set_xlim(min(lows) - span * 0.20, max(highs) + span * 0.20)

    direction = "lower is better" if lower_is_better else "higher is better"
    style_axis(axis, xlabel=direction, title=title)


# ── Forest panel: paired difference against the reference model ───────────────
def plot_differences(axis, y_true, predictions, metric="log_loss"):
    """
    Draws each model's paired difference versus the reference, with the zero
    line. An interval crossing zero means the two models are indistinguishable
    on this sample — which is the whole point of the panel.
    """
    rows = []
    for label, column in MODEL_COLUMNS:
        if column == REFERENCE[1] or column not in predictions:
            continue
        mean_diff, low, high, _ = bootstrap_difference(
            y_true, predictions[column].values,
            predictions[REFERENCE[1]].values, metric=metric,
        )
        rows.append((label, mean_diff, low, high))

    y_positions = np.arange(len(rows))[::-1]
    axis.axvline(0, color=AXIS, linewidth=1.2, zorder=1)

    for y_position, (label, mean_diff, low, high) in zip(y_positions, rows):
        color    = MODEL_COLORS[label]
        crosses  = low <= 0 <= high
        axis.plot([low, high], [y_position, y_position], color=color,
                  linewidth=2, solid_capstyle="round", zorder=2)
        axis.plot(mean_diff, y_position, "o", color=color, markersize=9,
                  markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        axis.annotate(
            f"{mean_diff:+.4f}  {'not significant' if crosses else 'significant'}",
            (mean_diff, y_position), textcoords="offset points",
            xytext=(0, 13), ha="center", color=INK, fontsize=9,
            fontweight="bold",
        )

    axis.set_yticks(y_positions)
    axis.set_yticklabels([row[0] for row in rows], color=INK_SOFT, fontsize=9.5)
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)

    # Same padding rule as the metric panels, and the zero line must stay in
    # frame even when every interval sits to one side of it.
    lows  = [row[2] for row in rows] + [0.0]
    highs = [row[3] for row in rows] + [0.0]
    span  = max(highs) - min(lows)
    axis.set_xlim(min(lows) - span * 0.20, max(highs) + span * 0.20)

    style_axis(
        axis,
        xlabel=f"paired difference in {metric.replace('_', ' ')} vs "
               f"{REFERENCE[0]}  (left of zero = better)",
        title=f"Paired difference vs {REFERENCE[0]}",
    )


# ── Reliability curve: predicted probability against observed win rate ────────
def plot_reliability(axis, y_true, predictions, n_bins=10):
    """
    Plots observed win rate against predicted probability per decile. A model
    sitting on the diagonal is calibrated: when it says 70%, teams win 70%.
    Bins holding fewer than MIN_BIN_GAMES games are dropped as noise.
    """
    axis.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)
    # Sits on the low end of the diagonal, below where either curve starts, so
    # it never overlaps a series line.
    axis.annotate("perfectly calibrated", xy=(0.13, 0.155), rotation=38,
                  color=INK_MUTED, fontsize=8.5, ha="center")

    edges = np.linspace(0, 1, n_bins + 1)
    for label, column in MODEL_COLUMNS:
        # A constant predictor has no spread to bin, so it is reported in the
        # caption rather than drawn as a single meaningless point.
        if column not in predictions or column == "always_home":
            continue

        probs   = predictions[column].values
        centers, observed = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            in_bin = (probs >= low) & (probs < high)
            if in_bin.sum() < MIN_BIN_GAMES:
                continue
            centers.append(probs[in_bin].mean())
            observed.append(y_true[in_bin].mean())

        axis.plot(centers, observed, "-o", color=MODEL_COLORS[label],
                  linewidth=2, markersize=8, markeredgecolor=SURFACE,
                  markeredgewidth=2, label=label, zorder=3)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    style_axis(axis, xlabel="predicted win probability",
               ylabel="observed win rate", title="Calibration")
    legend = axis.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_SOFT)


# ── Distribution of predicted probabilities ───────────────────────────────────
def plot_distribution(axis, predictions):
    """
    Shows how much each model actually commits. A model hugging 0.5 is barely
    discriminating, however good its log loss looks.
    """
    bins = np.linspace(0, 1, 26)
    for label, column in MODEL_COLUMNS:
        if column not in predictions or column == "always_home":
            continue
        axis.hist(predictions[column].values, bins=bins, histtype="step",
                  linewidth=2, color=MODEL_COLORS[label], label=label)

    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    style_axis(axis, xlabel="predicted home win probability",
               ylabel="games", title="Prediction spread")
    legend = axis.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_SOFT)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Builds both evaluation figures from the held-out season predictions."""
    predictions = build_predictions()
    y_true      = predictions[LABEL_COL].values
    n_games     = len(y_true)

    available = [
        (label, column) for label, column in MODEL_COLUMNS
        if column in predictions.columns
    ]
    results = [
        evaluate_model(label, y_true, predictions[column].values)
        for label, column in available
    ]

    season_label = f"{TEST_SEASON}-{str(TEST_SEASON + 1)[-2:]}"

    # ── Figure 1: metrics and the paired difference ───────────────────────────
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    figure.suptitle(
        f"Model comparison — held-out {season_label} season ({n_games:,} games)",
        color=INK, fontsize=14, fontweight="bold", x=0.055, ha="left", y=0.975,
    )
    figure.text(
        0.055, 0.938,
        "Points are estimates; bars are 95% bootstrap intervals.",
        color=INK_SOFT, fontsize=10, ha="left",
    )

    plot_metric(axes[0][0], results, "accuracy", "Accuracy", ".1%", False)
    plot_metric(axes[0][1], results, "log_loss", "Log loss", ".4f", True)
    plot_metric(axes[1][0], results, "brier", "Brier score", ".4f", True)
    plot_differences(axes[1][1], y_true, predictions)

    # One shared key for all four panels — colour means the same thing in each,
    # so repeating a legend per panel would be noise.
    legend = figure.legend(
        handles=model_legend_handles(available), loc="upper left",
        bbox_to_anchor=(0.052, 0.917), frameon=False, ncol=len(available),
        fontsize=9.5, handletextpad=0.6, columnspacing=2.2,
    )
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    figure.tight_layout(rect=[0, 0, 1, 0.885])
    figure.savefig(METRICS_FIG, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"[PLOT] Saved {METRICS_FIG}")

    # ── Figure 2: calibration and prediction spread ───────────────────────────
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    figure.suptitle(
        f"Calibration — held-out {season_label} season ({n_games:,} games)",
        color=INK, fontsize=14, fontweight="bold", x=0.055, ha="left", y=0.975,
    )
    figure.text(
        0.055, 0.915,
        "Always-home is omitted: it predicts one constant value, so it has no "
        "spread to bin.",
        color=INK_SOFT, fontsize=10, ha="left",
    )

    plot_reliability(axes[0], y_true, predictions)
    plot_distribution(axes[1], predictions)

    figure.tight_layout(rect=[0, 0, 1, 0.88])
    figure.savefig(CALIBRATION_FIG, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"[PLOT] Saved {CALIBRATION_FIG}")

    # Table view: every plotted value is also readable as text (accessibility
    # twin for the figures, and what gets pasted into the writeup).
    print(f"\n  Table view — held-out {season_label}, n = {n_games:,}\n")
    print(f"  {'Model':<18}{'Accuracy':>10}{'Log loss':>12}{'Brier':>10}")
    print(f"  {'-' * 50}")
    for row in results:
        print(f"  {row['name']:<18}{row['accuracy']:>9.1%}"
              f"{row['log_loss']:>12.4f}{row['brier']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
