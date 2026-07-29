#!/usr/bin/env python3
"""Generate readable, evidence-focused figures for the AAAI experiment section."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "figures"
COLORS = {
    "COAST": "#0F4C81",
    "MARDAM": "#D97706",
    "AM": "#2A9D8F",
    "PolyNet": "#7B61A8",
    "GPHH": "#C2410C",
    "C+C": "#9A3412",
    "C+W": "#B45309",
    "WIQ+C": "#6B7280",
}


def save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / filename, bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_level(token: str) -> float:
    return float(token.split("_")[1].replace("p", "."))


def load_dynamic_algorithm(algorithm: str) -> dict[tuple[float, float], np.ndarray]:
    root = ROOT / "output" / "dynamic_benchmark" / algorithm
    records: dict[tuple[float, float], np.ndarray] = {}
    for path in sorted(root.glob("dod_*/tw_*/seed_42/summary.csv")):
        dod = parse_level(path.parents[2].name)
        tw = parse_level(path.parents[1].name)
        records[(dod, tw)] = pd.read_csv(path)["normalized_cost"].to_numpy(dtype=float)
    if len(records) != 16:
        raise RuntimeError(f"Expected 16 dynamic cells for {algorithm}, found {len(records)}")
    return records


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def make_rq1_benchmark_profile() -> None:
    scales = ["h100", "h200", "h400"]
    coast = np.array([60.39, 108.70, 208.20])
    neural = {
        "MARDAM": np.array([61.56, 149.27, 317.57]),
        "AM": np.array([68.18, 124.88, 245.43]),
        "PolyNet": np.array([65.91, 129.44, 245.33]),
    }
    classical = {
        "GPHH": np.array([74.63, 187.16, 361.37]),
        "C+C": np.array([142.54, 274.28, 626.07]),
        "C+W": np.array([135.48, 264.91, 617.85]),
        "WIQ+C": np.array([178.39, 353.05, 730.85]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.1), constrained_layout=True)
    x = np.arange(len(scales))
    for ax, series, title, ymax in [
        (axes[0], neural, "(a) Neural baselines", 40),
        (axes[1], classical, "(b) GPHH and dispatching heuristics", 80),
    ]:
        for name, costs in series.items():
            gain = (costs - coast) / costs * 100.0
            ax.plot(x, gain, marker="o", markersize=4.5, linewidth=1.9, label=name, color=COLORS[name])
            # The detailed tables report exact values.  Annotating four nearly
            # coincident classical curves makes the graphic harder, not easier,
            # to read, so numeric labels are reserved for the neural comparison.
            if title.startswith("(a)"):
                for xpos, value in zip(x, gain):
                    ax.annotate(f"{value:.1f}", (xpos, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x, scales)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("COAST cost reduction (%)")
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    save(fig, "fig_rq1_solomon_profile.pdf")


def dynamic_matrices() -> tuple[list[float], list[float], dict[str, np.ndarray]]:
    dods = [0.10, 0.25, 0.50, 0.75]
    tws = [0.25, 0.50, 0.75, 1.00]
    coast = load_dynamic_algorithm("vectra")
    mardam = load_dynamic_algorithm("mardam")
    matrices = {
        "COAST": np.array([[coast[(dod, tw)].mean() for tw in tws] for dod in dods]),
        "MARDAM": np.array([[mardam[(dod, tw)].mean() for tw in tws] for dod in dods]),
        "PolyNet": np.array(
            [[43.64, 45.85, 48.76, 52.86], [43.90, 45.96, 48.13, 50.99], [44.03, 45.77, 47.40, 49.73], [44.47, 45.99, 47.49, 49.44]]
        ),
        "AM": np.array(
            [[44.67, 46.90, 48.46, 52.30], [44.84, 46.55, 48.46, 50.87], [44.73, 46.53, 48.75, 50.21], [45.10, 46.93, 47.60, 49.53]]
        ),
    }
    return dods, tws, matrices


def make_rq2_heatmap() -> None:
    dods, tws, matrices = dynamic_matrices()
    gap = (matrices["MARDAM"] - matrices["COAST"]) / matrices["MARDAM"] * 100.0
    fig, ax = plt.subplots(figsize=(5.3, 4.05), constrained_layout=True)
    image = ax.imshow(gap, cmap="Blues", vmin=0, vmax=7, aspect="auto")
    ax.set_xticks(range(len(tws)), [f"{value:.2f}" for value in tws])
    ax.set_yticks(range(len(dods)), [f"{value:.2f}" for value in dods])
    ax.set_xlabel("Time-window ratio")
    ax.set_ylabel("Degree of dynamism (DoD)")
    ax.set_title("COAST advantage over MARDAM by operating condition", loc="left", fontweight="bold")
    for row in range(gap.shape[0]):
        for col in range(gap.shape[1]):
            value = gap[row, col]
            ax.text(col, row, f"{value:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white" if value >= 4 else "#102A43")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("COAST cost reduction (%)")
    save(fig, "fig_rq2_heatmap.pdf")


def make_rq2_marginals() -> None:
    dods, tws, matrices = dynamic_matrices()
    coast = load_dynamic_algorithm("vectra")
    mardam = load_dynamic_algorithm("mardam")
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), constrained_layout=True)
    rng = np.random.default_rng(20260727)
    for axis, factor, levels, other_levels, xlabel, title in [
        (axes[0], "DoD", dods, tws, "Degree of dynamism (DoD)", "(a) Dynamism sensitivity"),
        (axes[1], "TW", tws, dods, "Time-window ratio", "(b) Time-window sensitivity"),
    ]:
        for name, matrix in matrices.items():
            values = matrix.mean(axis=1) if factor == "DoD" else matrix.mean(axis=0)
            axis.plot(levels, values, marker="o", markersize=4.5, linewidth=2.0, label=name, color=COLORS[name])
        for name, records, color in [("COAST", coast, COLORS["COAST"]), ("MARDAM", mardam, COLORS["MARDAM"])]:
            lows, highs = [], []
            for level in levels:
                samples = np.concatenate([records[(level, other)] for other in other_levels]) if factor == "DoD" else np.concatenate([records[(other, level)] for other in other_levels])
                _, low, high = bootstrap_mean_ci(samples, rng)
                lows.append(low)
                highs.append(high)
            axis.fill_between(levels, lows, highs, color=color, alpha=0.14, linewidth=0)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Mean normalized cost")
        axis.grid(axis="y", color="#CBD5E1", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    save(fig, "fig_rq2_marginals.pdf")


def make_rq2_diagnostics() -> None:
    """A single RQ2 figure: interaction map plus the two marginal mechanisms."""
    dods, tws, matrices = dynamic_matrices()
    coast = matrices["COAST"]
    mardam = matrices["MARDAM"]
    relative = 100.0 * (mardam - coast) / mardam
    rng = np.random.default_rng(20260727)
    raw_coast = load_dynamic_algorithm("vectra")
    raw_mardam = load_dynamic_algorithm("mardam")

    fig = plt.figure(figsize=(7.25, 4.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0])
    heat_ax = fig.add_subplot(grid[:, 0])
    dod_ax = fig.add_subplot(grid[0, 1:])
    tw_ax = fig.add_subplot(grid[1, 1:])

    image = heat_ax.imshow(relative, cmap="Blues", vmin=0.0, vmax=7.0, aspect="auto")
    heat_ax.set_xticks(range(len(tws)), [f"{value:.2f}" for value in tws])
    heat_ax.set_yticks(range(len(dods)), [f"{value:.2f}" for value in dods])
    heat_ax.set_xlabel("TW ratio")
    heat_ax.set_ylabel("Degree of dynamism (DoD)")
    heat_ax.set_title("(a) Interaction", loc="left", fontweight="bold")
    for row in range(relative.shape[0]):
        for col in range(relative.shape[1]):
            value = relative[row, col]
            heat_ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white" if value > 4.3 else "#102A43")
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.046, pad=0.04)
    colorbar.set_label("COAST reduction (%)")

    def marginal(axis: plt.Axes, values: list[float], factor: str) -> None:
        for name, matrix in matrices.items():
            mean = matrix.mean(axis=1 if factor == "dod" else 0)
            axis.plot(values, mean, marker="o", markersize=4.2, linewidth=1.75, label=name, color=COLORS[name])
        if factor == "dod":
            coast_samples = [np.concatenate([raw_coast[(dod, tw)] for tw in tws]) for dod in dods]
            mardam_samples = [np.concatenate([raw_mardam[(dod, tw)] for tw in tws]) for dod in dods]
        else:
            coast_samples = [np.concatenate([raw_coast[(dod, tw)] for dod in dods]) for tw in tws]
            mardam_samples = [np.concatenate([raw_mardam[(dod, tw)] for dod in dods]) for tw in tws]
        for name, samples in [("COAST", coast_samples), ("MARDAM", mardam_samples)]:
            intervals = np.asarray([bootstrap_mean_ci(sample, rng)[1:] for sample in samples])
            axis.fill_between(values, intervals[:, 0], intervals[:, 1], color=COLORS[name], alpha=0.14, linewidth=0)
        axis.set_ylabel("Mean cost")
        axis.grid(axis="y", color="#CBD5E1", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    marginal(dod_ax, dods, "dod")
    dod_ax.set_title("(b) Dynamism marginal", loc="left", fontweight="bold")
    dod_ax.set_xticks(dods)
    dod_ax.set_xlabel("DoD")
    dod_ax.legend(frameon=False, ncol=4, fontsize=7, loc="upper left")
    marginal(tw_ax, tws, "tw")
    tw_ax.set_title("(c) Time-window marginal", loc="left", fontweight="bold")
    tw_ax.set_xticks(tws)
    tw_ax.set_xlabel("TW ratio")
    save(fig, "fig_rq2_diagnostics.pdf")


def make_rq3_scale_profile() -> None:
    scales = ["n50m3", "n100m5", "n200m10", "n400m20"]
    x = np.arange(len(scales))
    direct = {
        "$-$ Ownership": [2.9, 2.7, 2.1, 1.7],
        "$-$ Lookahead": [1.9, 1.7, 1.1, 0.3],
        "$-$ Edge features": [3.4, 2.5, 2.5, 2.7],
    }
    structural = {"B0": [3.1, 2.6, 2.3, 2.1], "B1": [3.4, 3.0, 2.1, 1.5], "B3": [2.3, 2.1, 1.3, 0.5]}
    b5 = [3.6, 7.2, 42.1, 44.9]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.0), constrained_layout=True)
    palette = ["#D97706", "#7B61A8", "#2A9D8F"]
    for (name, values), color in zip(direct.items(), palette):
        axes[0].plot(x, values, marker="o", linewidth=2.0, markersize=4.5, label=name, color=color)
    for (name, values), color in zip(structural.items(), ["#6B7280", "#A16207", "#4B5563"]):
        axes[0].plot(x, values, marker="o", linestyle="--", linewidth=1.4, markersize=3.5, label=name, color=color, alpha=0.9)
    axes[0].set_title("(a) Direct removals and controls", loc="left", fontweight="bold")
    axes[0].set_xticks(x, scales)
    axes[0].set_ylim(0, 4.1)
    axes[0].set_ylabel("Cost increase vs. COAST (%)")
    axes[0].grid(axis="y", color="#CBD5E1", linewidth=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left")

    bars = axes[1].bar(x, b5, color=["#F2C14E", "#F2C14E", "#C2410C", "#9F1239"], width=0.62)
    axes[1].set_title("(b) Fixed equal-weight fusion (B5)", loc="left", fontweight="bold")
    axes[1].set_xticks(x, scales)
    axes[1].set_ylim(0, 50)
    axes[1].set_ylabel("Cost increase vs. COAST (%)")
    axes[1].grid(axis="y", color="#CBD5E1", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, b5):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"+{value:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    save(fig, "fig_rq3_scale_profile.pdf")


def make_rq3_diagnostics() -> None:
    """Combine scale and targeted-regime ablations without shrinking either view."""
    scales = ["n50m3", "n100m5", "n200m10", "n400m20"]
    x = np.arange(len(scales))
    profiles = {
        "$-$ Ownership": np.array([2.9, 2.7, 2.1, 1.7]),
        "$-$ Lookahead": np.array([1.9, 1.7, 1.1, 0.3]),
        "$-$ Edge features": np.array([3.4, 2.5, 2.5, 2.7]),
        "B0": np.array([3.1, 2.6, 2.3, 2.1]),
        "B1": np.array([3.4, 3.0, 2.1, 1.5]),
        "B3": np.array([2.3, 2.1, 1.3, 0.5]),
    }
    styles = {
        "$-$ Ownership": (COLORS["MARDAM"], "-"),
        "$-$ Lookahead": (COLORS["PolyNet"], "-"),
        "$-$ Edge features": (COLORS["AM"], "-"),
        "B0": ("#7C8798", "--"),
        "B1": ("#A86914", "--"),
        "B3": ("#5C6777", "--"),
    }
    b5 = np.array([3.6, 7.2, 42.1, 44.9])

    summary = pd.read_csv(ROOT / "output" / "ood_eval" / "ood_summary.csv")
    regimes = [("id_n50m3", "ID"), ("ood_burst_dynamic", "Burst"), ("ood_n100m5", "Scale"), ("ood_n50m6", "Fleet"), ("ood_sparse_spatial", "Sparse"), ("ood_tight_tw", "Tight TW")]
    variants = [("no_ownership", "$-$ Ownership"), ("no_lookahead", "$-$ Lookahead"), ("edgeoff", "$-$ Edge features"), ("b5", "B5 fixed fusion")]
    deltas = np.zeros((len(variants), len(regimes)))
    pvalues: list[float] = []
    cells: list[tuple[int, int]] = []
    for col, (regime, _) in enumerate(regimes):
        coast_row = summary[(summary.model == "vectra") & (summary.dataset == regime)].iloc[0]
        coast_costs = read_costs(coast_row.result_json)
        for row, (variant, _) in enumerate(variants):
            variant_row = summary[(summary.model == variant) & (summary.dataset == regime)].iloc[0]
            variant_costs = read_costs(variant_row.result_json)
            deltas[row, col] = 100.0 * (variant_costs.mean() - coast_costs.mean()) / coast_costs.mean()
            pvalues.append(float(wilcoxon(variant_costs, coast_costs, alternative="two-sided").pvalue))
            cells.append((row, col))
    significant = np.zeros_like(deltas, dtype=bool)
    for (row, col), adjusted in zip(cells, adjusted_holm(pvalues)):
        significant[row, col] = adjusted < 0.05

    fig = plt.figure(figsize=(7.25, 6.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.13])
    direct_ax = fig.add_subplot(grid[0, 0])
    b5_ax = fig.add_subplot(grid[0, 1])
    stress_ax = fig.add_subplot(grid[1, :])
    for name, values in profiles.items():
        color, linestyle = styles[name]
        direct_ax.plot(x, values, marker="o", markersize=4.3, linewidth=1.75, linestyle=linestyle, color=color, label=name)
    direct_ax.set_title("(a) Direct removals and controls", loc="left", fontweight="bold")
    direct_ax.set_xticks(x, scales)
    direct_ax.set_ylim(0, 4.1)
    direct_ax.set_ylabel("Cost increase vs. COAST (%)")
    direct_ax.grid(axis="y", color="#CBD5E1", linewidth=0.7)
    direct_ax.spines[["top", "right"]].set_visible(False)
    direct_ax.legend(frameon=False, fontsize=6.7, ncol=2, loc="upper left")
    bars = b5_ax.bar(x, b5, color=["#F6C453", "#F6C453", "#C2410C", "#A90F43"], width=0.62)
    b5_ax.set_title("(b) Fixed equal-weight fusion (B5)", loc="left", fontweight="bold")
    b5_ax.set_xticks(x, scales)
    b5_ax.set_ylim(0, 50)
    b5_ax.set_ylabel("Cost increase vs. COAST (%)")
    b5_ax.grid(axis="y", color="#CBD5E1", linewidth=0.7)
    b5_ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, b5):
        b5_ax.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"+{value:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    image = stress_ax.imshow(deltas, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=6.0), aspect="auto")
    stress_ax.set_xticks(range(len(regimes)), [label for _, label in regimes])
    stress_ax.set_yticks(range(len(variants)), [label for _, label in variants])
    stress_ax.set_xlabel("Evaluation regime")
    stress_ax.set_title("(c) Targeted evaluation regimes", loc="left", fontweight="bold")
    for row in range(deltas.shape[0]):
        for col in range(deltas.shape[1]):
            value = deltas[row, col]
            marker = " $\\bullet$" if significant[row, col] else ""
            stress_ax.text(col, row, f"{value:+.1f}{marker}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white" if value > 3.0 or value < -1.5 else "#102A43")
    for value in np.arange(-0.5, len(regimes), 1):
        stress_ax.axvline(value, color="white", linewidth=0.7, alpha=0.55)
    for value in np.arange(-0.5, len(variants), 1):
        stress_ax.axhline(value, color="white", linewidth=0.7, alpha=0.55)
    colorbar = fig.colorbar(image, ax=stress_ax, fraction=0.022, pad=0.018)
    colorbar.set_label("Cost change vs. COAST (%)")
    stress_ax.text(1.0, -0.20, "$\\bullet$ Holm-corrected $p<0.05$", transform=stress_ax.transAxes, fontsize=8, ha="right")
    save(fig, "fig_rq3_diagnostics.pdf")


def adjusted_holm(pvalues: list[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def read_costs(path: str) -> np.ndarray:
    data = json.loads((ROOT / path).read_text())
    return np.asarray(data["normalized_costs"], dtype=float)


def make_rq3_stress_profile() -> None:
    summary = pd.read_csv(ROOT / "output" / "ood_eval" / "ood_summary.csv")
    regimes = [("id_n50m3", "ID"), ("ood_burst_dynamic", "Burst"), ("ood_n100m5", "Scale"), ("ood_n50m6", "Fleet"), ("ood_sparse_spatial", "Sparse"), ("ood_tight_tw", "Tight TW")]
    variants = [("no_ownership", "$-$ Ownership"), ("no_lookahead", "$-$ Lookahead"), ("edgeoff", "$-$ Edge features"), ("b5", "B5 fixed fusion")]
    deltas = np.zeros((len(variants), len(regimes)))
    pvalues: list[float] = []
    cells: list[tuple[int, int]] = []
    for col, (regime, _) in enumerate(regimes):
        coast_row = summary[(summary.model == "vectra") & (summary.dataset == regime)].iloc[0]
        coast_costs = read_costs(coast_row.result_json)
        for row, (variant, _) in enumerate(variants):
            variant_row = summary[(summary.model == variant) & (summary.dataset == regime)].iloc[0]
            variant_costs = read_costs(variant_row.result_json)
            deltas[row, col] = 100.0 * (variant_costs.mean() - coast_costs.mean()) / coast_costs.mean()
            pvalues.append(float(wilcoxon(variant_costs, coast_costs, alternative="two-sided").pvalue))
            cells.append((row, col))
    significant = np.zeros_like(deltas, dtype=bool)
    for (row, col), adjusted in zip(cells, adjusted_holm(pvalues)):
        significant[row, col] = adjusted < 0.05

    fig, ax = plt.subplots(figsize=(7.25, 3.25), constrained_layout=True)
    image = ax.imshow(deltas, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=6.0), aspect="auto")
    ax.set_xticks(range(len(regimes)), [label for _, label in regimes], fontsize=7, rotation=25, ha="right")
    ax.set_yticks(range(len(variants)), [label for _, label in variants])
    ax.set_xlabel("Evaluation regime")
    ax.set_title("Component effects across targeted evaluation regimes", loc="left", fontweight="bold")
    for row in range(deltas.shape[0]):
        for col in range(deltas.shape[1]):
            value = deltas[row, col]
            marker = " $\\bullet$" if significant[row, col] else ""
            ax.text(col, row, f"{value:+.1f}{marker}", ha="center", va="center", fontsize=9, fontweight="bold", color="white" if value > 3.0 or value < -1.5 else "#102A43")
    for x in np.arange(-0.5, len(regimes), 1):
        ax.axvline(x, color="white", linewidth=0.7, alpha=0.55)
    for y in np.arange(-0.5, len(variants), 1):
        ax.axhline(y, color="white", linewidth=0.7, alpha=0.55)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.018)
    colorbar.set_label("Cost change vs. COAST (%)")
    ax.text(1.0, -0.19, "$\\bullet$ Holm-corrected $p<0.05$", transform=ax.transAxes, fontsize=8, ha="right")
    save(fig, "fig_rq3_stress.pdf")


def make_rq2_interaction_column() -> None:
    """Single-column DoD--TW interaction map with legible native-size labels."""
    dods, tws, matrices = dynamic_matrices()
    reduction = 100.0 * (matrices["MARDAM"] - matrices["COAST"]) / matrices["MARDAM"]
    fig, ax = plt.subplots(figsize=(3.35, 3.10), constrained_layout=True)
    image = ax.imshow(reduction, cmap="Blues", vmin=0.0, vmax=7.0, aspect="auto")
    ax.set_xticks(range(len(tws)), [f"{value:.2f}" for value in tws])
    ax.set_yticks(range(len(dods)), [f"{value:.2f}" for value in dods])
    ax.set_xlabel("TW ratio")
    ax.set_ylabel("Dynamism (DoD)")
    ax.set_title("COAST advantage over MARDAM", loc="left", fontweight="bold")
    for row in range(reduction.shape[0]):
        for col in range(reduction.shape[1]):
            value = reduction[row, col]
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8.2, fontweight="bold", color="white" if value > 4.3 else "#102A43")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.035)
    colorbar.set_label("Reduction (%)", fontsize=8)
    save(fig, "fig_rq2_interaction_column.pdf")


def make_rq2_marginals_column() -> None:
    """Two vertically stacked one-column trend plots for RQ2."""
    dods, tws, matrices = dynamic_matrices()
    raw_coast = load_dynamic_algorithm("vectra")
    raw_mardam = load_dynamic_algorithm("mardam")
    rng = np.random.default_rng(20260727)
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.35), constrained_layout=True)

    for axis, levels, factor, title in [
        (axes[0], dods, "dod", "Dynamism sensitivity"),
        (axes[1], tws, "tw", "Time-window sensitivity"),
    ]:
        for name, matrix in matrices.items():
            mean = matrix.mean(axis=1 if factor == "dod" else 0)
            axis.plot(levels, mean, marker="o", markersize=3.8, linewidth=1.55, label=name, color=COLORS[name])
        if factor == "dod":
            coast_samples = [np.concatenate([raw_coast[(dod, tw)] for tw in tws]) for dod in dods]
            mardam_samples = [np.concatenate([raw_mardam[(dod, tw)] for tw in tws]) for dod in dods]
            xlabel = "DoD"
        else:
            coast_samples = [np.concatenate([raw_coast[(dod, tw)] for dod in dods]) for tw in tws]
            mardam_samples = [np.concatenate([raw_mardam[(dod, tw)] for dod in dods]) for tw in tws]
            xlabel = "TW ratio"
        for name, samples in [("COAST", coast_samples), ("MARDAM", mardam_samples)]:
            intervals = np.asarray([bootstrap_mean_ci(sample, rng)[1:] for sample in samples])
            axis.fill_between(levels, intervals[:, 0], intervals[:, 1], color=COLORS[name], alpha=0.14, linewidth=0)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(levels)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Mean cost")
        axis.grid(axis="y", color="#CBD5E1", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.4, ncol=2, loc="upper left")
    save(fig, "fig_rq2_marginals_column.pdf")


def make_rq3_scale_column() -> None:
    """Direct-removal trends, with endpoint labels instead of a legend."""
    scales = ["n50m3", "n100m5", "n200m10", "n400m20"]
    x = np.arange(len(scales))
    profiles = [
        ("Ownership", np.array([2.9, 2.7, 2.1, 1.7]), "#C5815B", -4),
        ("Lookahead", np.array([1.9, 1.7, 1.1, 0.3]), "#7D70AA", 0),
        ("Edge features", np.array([3.4, 2.5, 2.5, 2.7]), "#579C90", 2),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 2.85), constrained_layout=True)
    for name, values, color, offset in profiles:
        ax.plot(x, values, marker="o", markersize=4.6, linewidth=2.0, color=color)
        ax.annotate(
            f"{name}\n{values[-1]:.1f}%",
            xy=(x[-1], values[-1]),
            xytext=(8, offset),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.2,
            color=color,
            fontweight="bold",
        )
    ax.set_xticks(x, scales)
    ax.set_xlim(-0.12, 3.86)
    ax.set_ylim(0, 3.95)
    ax.set_ylabel("Mean-cost increase vs. COAST (%)")
    ax.grid(axis="y", color="#DCE5EC", linewidth=0.7)
    ax.set_facecolor("#FCFDFE")
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#94A3B8")
        ax.spines[spine].set_linewidth(0.7)
    save(fig, "fig_rq3_scale_column.pdf")


def make_rq3_stress_column() -> None:
    """Effect-size heatmap for direct removals and the fixed-fusion control."""
    summary = pd.read_csv(ROOT / "output" / "ood_eval" / "ood_summary.csv")
    regimes = [("id_n50m3", "ID"), ("ood_burst_dynamic", "Burst"), ("ood_n100m5", "Scale"), ("ood_n50m6", "Fleet"), ("ood_sparse_spatial", "Sparse"), ("ood_tight_tw", "Tight")]
    variants = [
        ("no_ownership", "No ownership"),
        ("no_lookahead", "No lookahead"),
        ("edgeoff", "No edge features"),
        ("b5", "Fixed fusion (B5)"),
    ]
    deltas = np.zeros((len(variants), len(regimes)))
    for col, (regime, _) in enumerate(regimes):
        coast_row = summary[(summary.model == "vectra") & (summary.dataset == regime)].iloc[0]
        coast_costs = read_costs(coast_row.result_json)
        for row, (variant, _) in enumerate(variants):
            variant_row = summary[(summary.model == variant) & (summary.dataset == regime)].iloc[0]
            variant_costs = read_costs(variant_row.result_json)
            deltas[row, col] = 100.0 * (variant_costs.mean() - coast_costs.mean()) / coast_costs.mean()

    fig, ax = plt.subplots(figsize=(3.35, 3.12), constrained_layout=True)
    soft_diverging = matplotlib.colors.LinearSegmentedColormap.from_list(
        "soft_blue_peach", ["#8FB8D0", "#DDECF1", "#FAF8F4", "#F1D6CE", "#D99482"]
    )
    image = ax.imshow(deltas, cmap=soft_diverging, norm=TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=6.0), aspect="auto")
    regime_labels = ["Base", "Burst", "Scale", "Fleet", "Sparse", "Tight"]
    ax.set_xticks(range(len(regimes)), regime_labels, fontsize=7.0, rotation=18, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(variants)), [label for _, label in variants], fontsize=7.7)
    ax.tick_params(axis="x", pad=2)
    for row in range(deltas.shape[0]):
        for col in range(deltas.shape[1]):
            value = deltas[row, col]
            ax.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=7.45, fontweight="bold", color="white" if value > 3.8 or value < -1.6 else "#213547")
    for value in np.arange(-0.5, len(regimes), 1):
        ax.axvline(value, color="#F8FAFC", linewidth=0.9, alpha=0.95)
    for value in np.arange(-0.5, len(variants), 1):
        ax.axhline(value, color="#F8FAFC", linewidth=0.9, alpha=0.95)
    ax.axhline(2.5, color="#B8C5CF", linewidth=0.8, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#AAB8C4")
        spine.set_linewidth(0.7)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.08, pad=0.18)
    colorbar.outline.set_edgecolor("#AAB8C4")
    colorbar.outline.set_linewidth(0.6)
    colorbar.set_label("Mean-cost change vs. COAST (%)", fontsize=7.0, labelpad=2)
    colorbar.ax.tick_params(labelsize=6.5, pad=1)
    save(fig, "fig_rq3_stress_column.pdf")


def make_rq2_four_heatmaps() -> None:
    """Four neural methods as 1x4 horizontal heatmaps with shared colour scale."""
    dods, tws, matrices = dynamic_matrices()
    methods = ["COAST", "MARDAM", "AM", "PolyNet"]
    all_values = np.concatenate([matrices[m].ravel() for m in methods])
    vmin, vmax = all_values.min(), all_values.max()

    fig, axes = plt.subplots(1, 4, figsize=(7.25, 3.35), constrained_layout=True)
    for ax, name in zip(axes, methods):
        matrix = matrices[name]
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(tws)), [f"{v:.2f}" for v in tws], fontsize=6.5)
        ax.set_yticks(range(len(dods)), [f"{v:.2f}" for v in dods], fontsize=6.5)
        ax.set_xlabel("TW ratio", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("DoD", fontsize=7)
        ax.set_title(name, fontweight="bold", fontsize=9, color=COLORS.get(name, "#333333"))
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=5.8,
                        fontweight="bold", color="white" if value > (vmin + vmax) / 2 else "#333333")

    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.025, shrink=0.42)
    cbar.set_label("Normalized cost", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    save(fig, "fig_rq2_four_heatmaps.pdf")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.facecolor": "white"})
    make_rq2_interaction_column()
    make_rq2_marginals_column()
    make_rq3_scale_column()
    make_rq3_stress_column()
    make_rq2_four_heatmaps()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
