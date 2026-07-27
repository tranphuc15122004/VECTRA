#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


OUT_DIR = Path(__file__).resolve().parent / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def make_solomon_benchmark() -> None:
    labels = ["100 customers", "200 customers", "400 customers"]
    series = {
        "COAST(g)": [60.39, 108.70, 208.20],
        "MARDAM(g)": [61.56, 149.27, 317.57],
        "AM(g)": [68.18, 124.88, 245.43],
        "PolyNet(g)": [65.91, 129.44, 245.33],
        "GPHH": [74.63, 187.16, 361.37],
    }
    colors = {
        "COAST(g)": "#1f77b4",
        "MARDAM(g)": "#ff7f0e",
        "AM(g)": "#2ca02c",
        "PolyNet(g)": "#d62728",
        "GPHH": "#9467bd",
    }

    x = np.arange(len(labels))
    width = 0.16

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    offsets = np.linspace(-2, 2, num=len(series)) * width
    for offset, (name, values) in zip(offsets, series.items()):
        ax.bar(x + offset, values, width=width, label=name, color=colors[name])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean normalized cost")
    ax.set_ylim(0, 390)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_csv_benchmark.pdf")


def make_pyth_scale() -> None:
    labels = ["n20m1", "n50m3", "n100m5", "n200m10", "n400m20"]
    x = np.arange(len(labels))
    series = {
        "COAST(g)": [24.78, 43.92, 94.85, 182.38, 355.99],
        "MARDAM(g)": [26.13, 46.48, 99.50, 203.24, 430.48],
        "AM(g)": [25.40, 49.58, 103.85, 196.53, 378.98],
        "PolyNet(g)": [25.58, 49.93, 105.13, 200.89, 389.72],
    }
    colors = {
        "COAST(g)": "#1f77b4",
        "MARDAM(g)": "#ff7f0e",
        "AM(g)": "#2ca02c",
        "PolyNet(g)": "#d62728",
    }
    markers = {
        "COAST(g)": "o",
        "MARDAM(g)": "s",
        "AM(g)": "^",
        "PolyNet(g)": "D",
    }

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for name, values in series.items():
        ax.plot(
            x,
            values,
            marker=markers[name],
            linewidth=2.0,
            markersize=6,
            label=name,
            color=colors[name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean normalized cost")
    ax.set_ylim(0, 450)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_pyth_scale.pdf")


def make_dynamic_heatmaps() -> None:
    dod_labels = ["0.10", "0.25", "0.50", "0.75"]
    tw_labels = ["0.25", "0.50", "0.75", "1.00"]
    matrices = {
        "COAST": np.array([
            [41.7, 43.2, 44.1, 45.5],
            [41.5, 42.7, 44.3, 45.5],
            [41.2, 43.1, 44.4, 45.8],
            [42.4, 43.6, 45.0, 46.9],
        ]),
        "MARDAM": np.array([
            [42.2, 43.4, 44.4, 46.2],
            [42.9, 43.6, 45.0, 45.9],
            [43.7, 44.2, 45.6, 46.4],
            [45.2, 46.1, 47.4, 48.0],
        ]),
        "PolyNet": np.array([
            [43.6, 45.9, 48.8, 52.9],
            [43.9, 46.0, 48.1, 51.0],
            [44.0, 45.8, 47.4, 49.7],
            [44.5, 46.0, 47.5, 49.4],
        ]),
        "AM": np.array([
            [44.7, 46.9, 48.5, 52.3],
            [44.8, 46.5, 48.5, 50.9],
            [44.7, 46.5, 48.8, 50.2],
            [45.1, 46.9, 47.6, 49.5],
        ]),
    }

    vmin = min(matrix.min() for matrix in matrices.values())
    vmax = max(matrix.max() for matrix in matrices.values())
    cmap = LinearSegmentedColormap.from_list(
        "coast_orange",
        ["#fff7ed", "#fed7aa", "#fb923c", "#f97316", "#c2410c"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), constrained_layout=True)
    axes = axes.flatten()
    image = None

    for ax, (name, matrix) in zip(axes, matrices.items()):
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(name, fontsize=11)
        ax.set_xticks(range(len(tw_labels)))
        ax.set_xticklabels(tw_labels)
        ax.set_yticks(range(len(dod_labels)))
        ax.set_yticklabels(dod_labels)
        ax.set_xlabel("TW ratio")
        ax.set_ylabel("DoD")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text_color = "white" if value > (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.95)
        cbar.set_label("Mean normalized cost")

    _save(fig, "fig_dynamic_sensitivity.pdf")


def main() -> None:
    make_solomon_benchmark()
    make_pyth_scale()
    make_dynamic_heatmaps()


if __name__ == "__main__":
    main()
