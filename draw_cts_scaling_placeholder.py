#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "cts_scaling_placeholder.pdf"
OUT_PNG = ROOT / "cts_scaling_placeholder.png"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.85,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.9,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.6,
            "axes.unicode_minus": False,
        }
    )


def placeholder_data() -> tuple[np.ndarray, dict[str, dict[str, np.ndarray | str]]]:
    # TODO: Replace these layout placeholder values with measured results
    # before submission. The K=256 points for CTS and Attention Top-K are
    # aligned with the current main table; the remaining points and intervals
    # are illustrative only and are used to finalize the figure design.
    budgets = np.array([64, 128, 256, 512, 1024])
    return budgets, {
        "Random": {
            "color": "#7f8794",
            "marker": "o",
            "overall": np.array([55.8, 58.4, 60.1, 61.0, 61.8]),
            "action": np.array([50.0, 54.2, 56.8, 58.2, 59.0]),
        },
        "Uniform": {
            "color": "#009e73",
            "marker": "s",
            "overall": np.array([59.5, 63.0, 65.2, 66.1, 66.5]),
            "action": np.array([52.5, 58.1, 61.8, 63.4, 64.1]),
        },
        "Attention Top-K": {
            "color": "#d55e00",
            "marker": "^",
            "overall": np.array([61.8, 66.4, 70.9, 71.8, 72.0]),
            "action": np.array([56.0, 62.2, 67.1, 68.0, 68.4]),
        },
        "CTS": {
            "color": "#0072b2",
            "marker": "D",
            "overall": np.array([65.7, 70.8, 73.6, 74.2, 74.3]),
            "action": np.array([60.5, 65.9, 68.7, 69.5, 69.8]),
        },
    }


def confidence_interval(budgets: np.ndarray, base: float) -> np.ndarray:
    scale = np.sqrt(256.0 / budgets)
    return base * np.clip(scale, 0.72, 1.85)


def style_axis(ax: plt.Axes, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim(58, 1130)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#cfd5df", linewidth=0.58, linestyle=(0, (3, 3)))
    ax.grid(axis="x", color="#e4e8ef", linewidth=0.38, linestyle=(0, (2, 4)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(256, color="#5f6877", linewidth=0.88, linestyle=(0, (4, 3)), zorder=1)
    ax.tick_params(axis="both", length=3.0, width=0.76)


def draw_panel(
    ax: plt.Axes,
    budgets: np.ndarray,
    series: dict[str, dict[str, np.ndarray | str]],
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    style_axis(ax, ylabel, ylim)
    for name in ["Random", "Uniform", "Attention Top-K", "CTS"]:
        item = series[name]
        values = item[metric]
        ci = confidence_interval(budgets, 1.05 if metric == "overall" else 1.15)
        ax.fill_between(
            budgets,
            values - ci,
            values + ci,
            color=item["color"],
            alpha=0.11 if name in ["Random", "Uniform"] else 0.14,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            budgets,
            values,
            marker=item["marker"],
            linestyle="-",
            color=item["color"],
            markersize=4.65,
            linewidth=1.65,
            markerfacecolor=item["color"],
            markeredgecolor="white",
            markeredgewidth=0.62,
            label=name,
            zorder=4,
        )


def draw_gain_panel(
    ax: plt.Axes,
    budgets: np.ndarray,
    series: dict[str, dict[str, np.ndarray | str]],
) -> None:
    style_axis(ax, "CTS gain (pts)", (0.0, 5.2))
    ax.axhline(0.0, color="#8b95a5", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)

    gain_specs = [
        ("Overall", "overall", "#0072b2", "o", "-"),
        ("Action", "action", "#b23a48", "s", "--"),
    ]
    for label, metric, color, marker, linestyle in gain_specs:
        gain = series["CTS"][metric] - series["Attention Top-K"][metric]
        ci = confidence_interval(budgets, 0.36 if metric == "overall" else 0.42)
        ax.fill_between(
            budgets,
            gain - ci,
            gain + ci,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            budgets,
            gain,
            marker=marker,
            linestyle=linestyle,
            color=color,
            markersize=4.1,
            linewidth=1.45,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.58,
            label=label,
            zorder=4,
        )

    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.45,
        handletextpad=0.35,
        borderpad=0.2,
    )


def main() -> None:
    configure_matplotlib()
    budgets, series = placeholder_data()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 1.95),
        dpi=300,
        sharex=True,
        gridspec_kw={"wspace": 0.32},
    )

    draw_panel(axes[0], budgets, series, "overall", "Overall Acc. (%)", (53, 76))
    draw_panel(axes[1], budgets, series, "action", "Action Acc. (%)", (48, 71.5))
    draw_gain_panel(axes[2], budgets, series)
    axes[0].set_title("(a) Overall", pad=5)
    axes[1].set_title("(b) Action", pad=5)
    axes[2].set_title("(c) Gain over Top-$K$", pad=5)

    for ax in axes:
        ax.set_xlabel(r"Visual-token budget $K$")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(k) for k in budgets])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, 1.015),
        handlelength=1.45,
        handletextpad=0.38,
        columnspacing=0.95,
    )

    fig.subplots_adjust(left=0.068, right=0.988, bottom=0.235, top=0.755)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.015)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
