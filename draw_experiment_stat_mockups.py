#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "mockups"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 11.5,
        }
    )


def resolution_bin_data() -> tuple[np.ndarray, list[str], dict[str, dict[str, np.ndarray]]]:
    counts = np.array([1487, 1563, 240], dtype=float)
    labels = ["<1080p\n$n=1,487$", "1080p-4K\n$n=1,563$", ">4K\n$n=240$"]

    series = {
        "Qwen3-VL-4B": {
            "color": "#8897ad",
            "marker": "s",
            "acc": np.array([58.5, 49.5, 41.0]),
        },
        "w/o AREP": {
            "color": "#d9831f",
            "marker": "^",
            "acc": np.array([69.0, 67.0, 60.5]),
        },
        "PowerBrain": {
            "color": "#1e5d91",
            "marker": "o",
            "acc": np.array([72.0, 74.0, 77.5]),
        },
    }

    for item in series.values():
        p = item["acc"] / 100.0
        item["ci"] = 1.96 * np.sqrt(p * (1.0 - p) / counts) * 100.0

    return counts, labels, series


def forest_plot_data() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    comparisons = [
        "vs Qwen3-VL-4B",
        "vs w/o RAG",
        "vs w/o SFT",
        "vs w/o AREP",
        "vs w/o CTS",
    ]
    metrics = ["Overall", "Perception", "Reasoning", "Action"]
    deltas = np.array(
        [
            [23.9, 24.7, 23.5, 23.5],
            [9.4, 9.0, 9.7, 9.5],
            [6.8, 7.7, 7.9, 4.8],
            [3.8, 4.6, 3.7, 3.1],
            [2.7, 3.3, 3.2, 1.6],
        ]
    )
    ci = np.array(
        [
            [1.8, 1.7, 1.9, 2.1],
            [1.0, 1.1, 1.0, 1.2],
            [1.0, 1.1, 1.2, 1.1],
            [0.9, 1.0, 1.1, 1.2],
            [0.8, 0.9, 0.9, 1.0],
        ]
    )
    return comparisons, metrics, deltas, ci


def arep_bootstrap_data() -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    scales = ["Large\n$n=170$", "Medium\n$n=165$", "Micro\n$n=165$"]
    ns = np.array([170, 165, 165])
    probs = {
        "PowerBrain": np.array([0.885, 0.780, 0.674]),
        "w/o AREP": np.array([0.835, 0.685, 0.544]),
    }

    samples = {}
    for name, p_arr in probs.items():
        draws = []
        for n, p in zip(ns, p_arr):
            acc = rng.binomial(int(n), p, size=1200) / n * 100.0
            draws.append(acc)
        samples[name] = np.array(draws, dtype=object)
    return scales, ns, samples


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#cfd5df", linestyle=(0, (3, 3)), linewidth=0.85)
    ax.set_axisbelow(True)


def plot_resolution_binned(ax: plt.Axes) -> None:
    _, labels, series = resolution_bin_data()
    x = np.arange(len(labels))
    offsets = {"Qwen3-VL-4B": -0.03, "w/o AREP": 0.0, "PowerBrain": 0.03}

    style_axes(ax)
    for name in ["Qwen3-VL-4B", "w/o AREP", "PowerBrain"]:
        item = series[name]
        ax.errorbar(
            x + offsets[name],
            item["acc"],
            yerr=item["ci"],
            fmt=item["marker"] + "-",
            color=item["color"],
            ecolor=item["color"],
            elinewidth=1.4,
            capsize=3.5,
            markersize=7.5,
            linewidth=2.0,
            markerfacecolor=item["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=name,
            zorder=4,
        )

    gains = series["PowerBrain"]["acc"] - series["Qwen3-VL-4B"]["acc"]
    gain_text = " / ".join([f"+{v:.1f}" for v in gains])
    ax.text(
        0.98,
        0.05,
        f"$\\Delta$(PowerBrain - backbone): {gain_text} pts",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=11.4,
        color="#215b88",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#eef5fb", "edgecolor": "#c9dceb"},
    )

    ax.set_title("Resolution-Stratified Diagnostic Accuracy", pad=12)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Image resolution bin")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(35, 84)
    ax.legend(loc="upper left", frameon=False, ncol=1)


def plot_delta_forest(ax: plt.Axes) -> None:
    comparisons, metrics, deltas, ci = forest_plot_data()
    colors = ["#1e5d91", "#2296a8", "#d9831f", "#c84e44"]
    markers = ["o", "s", "D", "^"]
    y = np.arange(len(comparisons))[::-1]
    offsets = np.array([-0.24, -0.08, 0.08, 0.24])

    for idx, yi in enumerate(y):
        if idx % 2 == 0:
            ax.axhspan(yi - 0.44, yi + 0.44, color="#f6f8fb", zorder=0)

    ax.axvline(0.0, color="#7f8794", linewidth=1.0, linestyle="--", zorder=1)
    for m_idx, metric in enumerate(metrics):
        ax.errorbar(
            deltas[:, m_idx],
            y + offsets[m_idx],
            xerr=ci[:, m_idx],
            fmt=markers[m_idx],
            color=colors[m_idx],
            ecolor=colors[m_idx],
            elinewidth=1.35,
            capsize=3.0,
            markersize=6.5,
            linewidth=0.0,
            markeredgecolor="white",
            markeredgewidth=0.75,
            label=metric,
            zorder=3,
        )

    style_axes(ax)
    ax.grid(axis="x", color="#cfd5df", linestyle=(0, (3, 3)), linewidth=0.85)
    ax.grid(axis="y", visible=False)
    ax.set_title("Paired Gain With 95% Confidence Interval", pad=12)
    ax.set_xlabel("Improvement over baseline / ablation (percentage points)")
    ax.set_yticks(y)
    ax.set_yticklabels(comparisons)
    ax.set_xlim(-1.0, 28.5)
    ax.legend(loc="lower right", frameon=False, ncol=2)


def plot_arep_bootstrap(ax: plt.Axes) -> None:
    scales, _, samples = arep_bootstrap_data()
    power_color = "#1e5d91"
    wo_color = "#d9831f"
    style_axes(ax)

    positions = np.array([1.0, 1.8, 3.3, 4.1, 5.6, 6.4])
    power_data = [samples["PowerBrain"][0], samples["PowerBrain"][1], samples["PowerBrain"][2]]
    wo_data = [samples["w/o AREP"][0], samples["w/o AREP"][1], samples["w/o AREP"][2]]
    combined = [power_data[0], wo_data[0], power_data[1], wo_data[1], power_data[2], wo_data[2]]

    box = ax.boxplot(
        combined,
        positions=positions,
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1b1f24", "linewidth": 1.6},
        whiskerprops={"linewidth": 1.1, "color": "#5b6573"},
        capprops={"linewidth": 1.1, "color": "#5b6573"},
        boxprops={"linewidth": 1.1, "color": "#5b6573"},
    )

    fills = [power_color, wo_color] * 3
    for patch, color in zip(box["boxes"], fills):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)

    means = [np.mean(arr) for arr in combined]
    ax.scatter(positions, means, s=26, color="white", edgecolors="#1b1f24", zorder=4)

    centers = [1.4, 3.7, 6.0]
    deltas = [5.0, 9.5, 13.0]
    for center, delta, y_loc in zip(centers, deltas, [92.0, 86.0, 77.0]):
        ax.text(
            center,
            y_loc,
            f"$\\Delta$ {delta:.1f} pts",
            ha="center",
            va="bottom",
            fontsize=11.9,
            color="#b84439",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#fff4f1", "edgecolor": "#f1c8c1"},
        )

    ax.axvline(2.55, color="#d7dce5", linewidth=1.0)
    ax.axvline(4.85, color="#d7dce5", linewidth=1.0)
    ax.set_title("Bootstrap Accuracy Distribution by Object Scale", pad=12)
    ax.set_ylabel("Bootstrap subset accuracy (%)")
    ax.set_xlabel("Target scale group")
    ax.set_xticks(centers)
    ax.set_xticklabels(scales)
    ax.set_ylim(42, 95)
    ax.legend(
        handles=[
            Patch(facecolor=power_color, edgecolor="#5b6573", label="PowerBrain", alpha=0.78),
            Patch(facecolor=wo_color, edgecolor="#5b6573", label="w/o AREP", alpha=0.78),
        ],
        loc="upper right",
        frameon=False,
    )


def save_panel(
    stem: str,
    draw_fn,
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    draw_fn(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_overview() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), dpi=300)
    plot_resolution_binned(axes[0])
    plot_delta_forest(axes[1])
    plot_arep_bootstrap(axes[2])
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_DIR / "exp_stat_mockups_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "exp_stat_mockups_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(exist_ok=True)
    save_panel("resolution_binned_accuracy_mockup", plot_resolution_binned, (6.6, 4.8))
    save_panel("paired_delta_forest_mockup", plot_delta_forest, (7.1, 5.0))
    save_panel("arep_bootstrap_boxplot_mockup", plot_arep_bootstrap, (6.8, 4.8))
    save_overview()
    print("Generated mockup figures:")
    for path in sorted(OUT_DIR.iterdir()):
        if path.suffix in {".png", ".pdf"}:
            print(path)


if __name__ == "__main__":
    main()
