#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "resolution_dynamic_size_cloud_regen.pdf"
PNG_PATH = ROOT / "resolution_dynamic_size_cloud_regen.png"

BIN_LABELS = ("<=0.5", "0.5-1", "1-2.1", "2.1-4", "4-6", "6-8.3", "8.3-12", ">12")
BIN_SHARES = np.array([8.6, 14.8, 21.8, 18.2, 16.6, 12.7, 4.8, 2.5])
BIN_COLORS = (
    "#4c72b0",
    "#4c72b0",
    "#4c72b0",
    "#009e73",
    "#009e73",
    "#009e73",
    "#d55e00",
    "#d55e00",
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.82,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.unicode_minus": False,
        }
    )


def draw() -> None:
    configure_matplotlib()

    x = np.arange(len(BIN_LABELS))
    fig, ax = plt.subplots(figsize=(3.48, 1.86), dpi=320)
    fig.subplots_adjust(left=0.125, right=0.988, bottom=0.305, top=0.785)

    ax.axvline(2.5, color="#737d8c", lw=0.78, ls=(0, (3.0, 2.5)), zorder=1)
    ax.axvline(5.5, color="#737d8c", lw=0.78, ls=(0, (3.0, 2.5)), zorder=1)

    ax.bar(
        x,
        BIN_SHARES,
        width=0.66,
        color=BIN_COLORS,
        edgecolor="white",
        linewidth=0.85,
        alpha=0.93,
        zorder=3,
    )
    for xi, yi in zip(x, BIN_SHARES):
        ax.text(xi, yi + 0.55, f"{yi:.1f}", ha="center", va="bottom", fontsize=6.55, color="#313844")

    ax.set_xlim(-0.55, len(BIN_LABELS) - 0.45)
    ax.set_ylim(0, 25)
    ax.set_ylabel("Share (%)", labelpad=2.2)
    ax.set_xlabel("Image area bin (megapixels)", labelpad=2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks([0, 10, 20])
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.56, color="#cfd5df", alpha=0.95, zorder=0)
    ax.tick_params(axis="both", width=0.7, length=3, pad=1.5)

    for spine in ("right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.82)
    ax.spines["left"].set_linewidth(0.82)

    legend_handles = [
        Patch(facecolor="#4c72b0", edgecolor="none", label="<=1080p 45.2%"),
        Patch(facecolor="#009e73", edgecolor="none", label="1080p-4K 47.5%"),
        Patch(facecolor="#d55e00", edgecolor="none", label=">4K 7.3%"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.955),
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.85,
        borderpad=0.0,
    )

    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.012)
    fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)


if __name__ == "__main__":
    draw()
