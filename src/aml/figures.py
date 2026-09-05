"""Figures for the typology description.

Identity is carried by axis position rather than colour, so only two hues are
needed, one per split. Ring counts per typology are small, between 12 and 54,
so the ring level distributions are drawn as jittered points with a median rule
rather than box plots, which would imply more precision than the data supports.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES, rng

SERIES_COLOURS = {"HI-Small": "#2a78d6", "LI-Small": "#eb6834"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d7d2"

PANELS = [
    ("ring_size", "Transactions per ring"),
    ("span_days", "Ring span, days"),
    ("distinct_accounts", "Distinct accounts"),
    ("distinct_currencies", "Distinct currencies"),
]


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def typology_profile_figure(rings: pd.DataFrame, path: Path | None = None) -> Path:
    """Four panel profile of the eight typologies across both splits."""
    order = (
        rings.loc[rings["split"] == "HI-Small"]
        .groupby("typology")["ring_size"]
        .median()
        .sort_values()
        .index.tolist()
    )
    positions = {name: i for i, name in enumerate(order)}
    offsets = {"HI-Small": 0.18, "LI-Small": -0.18}

    fig, axes = plt.subplots(1, len(PANELS), figsize=(13.5, 4.2), sharey=True)

    for ax, (column, label) in zip(axes, PANELS):
        for split, colour in SERIES_COLOURS.items():
            subset = rings.loc[rings["split"] == split]
            jitter = rng("figure", "typology_jitter", split).normal(0, 0.045, len(subset))
            y = subset["typology"].map(positions).to_numpy() + offsets[split] + jitter
            ax.scatter(
                subset[column],
                y,
                s=13,
                color=colour,
                alpha=0.55,
                linewidths=0.5,
                edgecolors="white",
                label=split if column == "ring_size" else None,
                zorder=3,
            )
            medians = subset.groupby("typology")[column].median()
            for name, value in medians.items():
                base = positions[name] + offsets[split]
                ax.plot(
                    [value, value],
                    [base - 0.13, base + 0.13],
                    color=colour,
                    linewidth=2.0,
                    solid_capstyle="butt",
                    zorder=4,
                )
        _style(ax)
        ax.set_xlabel(label, fontsize=9, color=TEXT_PRIMARY)
        ax.set_ylim(-0.7, len(order) - 0.3)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=9, color=TEXT_PRIMARY)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.005),
        frameon=False,
        fontsize=9,
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    fig.suptitle(
        "Planted laundering typologies differ in size, duration and currency mix",
        fontsize=12,
        color=TEXT_PRIMARY,
        x=0.006,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.006,
        0.930,
        "One point per ring, vertical rule is the median. Synthetic data, shapes are generator defined.",
        fontsize=8.5,
        color=TEXT_SECONDARY,
        ha="left",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.915))
    target = path or (FIGURES / "typology_profile.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, facecolor="white")
    plt.close(fig)
    return target


def detection_figure(evaluation: pd.DataFrame, budget: float,
                     path: Path | None = None) -> Path:
    """Ring recall by model variant, with the bootstrap interval on each bar.

    Two panels, one per split, sharing an axis so the HI and LI levels are
    directly comparable. The intervals are drawn because the whole point of the
    comparison is whether the gap between the two "without format" bars is
    larger than the noise, and a bar chart without them would invite the reader
    to assume it is.
    """
    order = ["txn_with_format", "txn_without_format",
             "graph_with_format", "graph_without_format"]
    labels = ["transaction\nwith format", "transaction\nwithout format",
              "graph\nwith format", "graph\nwithout format"]
    # The two honest variants are the comparison; the format ones are context.
    fill = {"txn_with_format": "#c9c8c3", "graph_with_format": "#c9c8c3",
            "txn_without_format": "#eb6834", "graph_without_format": "#2a78d6"}

    frame = evaluation[evaluation["alert_budget"] == budget]
    splits = sorted(frame["split"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(10, 4.2), sharey=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        rows = frame[frame["split"] == split].set_index("variant")
        values = [rows.loc[v, "ring_recall_at_least_1"] for v in order]
        lo = [rows.loc[v, "ring_recall_at_least_1_lo"] for v in order]
        hi = [rows.loc[v, "ring_recall_at_least_1_hi"] for v in order]
        errors = [[v - l for v, l in zip(values, lo)],
                  [h - v for v, h in zip(values, hi)]]
        positions = range(len(order))
        ax.bar(positions, values, color=[fill[v] for v in order], width=0.62)
        ax.errorbar(positions, values, yerr=errors, fmt="none",
                    ecolor=TEXT_SECONDARY, elinewidth=1.1, capsize=4)
        for x, value in zip(positions, values):
            ax.text(x, value + 0.02, f"{value:.3f}", ha="center",
                    fontsize=8, color=TEXT_PRIMARY)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(split, fontsize=10, color=TEXT_PRIMARY, pad=8)
        _style(ax)

    axes[0].set_ylabel("Ring recall, at least one transaction flagged",
                       fontsize=9, color=TEXT_SECONDARY)
    axes[0].set_ylim(0, 0.72)
    fig.suptitle(
        f"Structure recovers what the generator artefact was faking "
        f"(alert budget {budget:.1%})",
        fontsize=11, color=TEXT_PRIMARY, y=0.99,
    )
    fig.tight_layout()
    path = path or FIGURES / "detection_by_variant.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path
