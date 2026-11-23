"""Figures 1-4 of the paper, rebuilt from the transcribed aggregates.

Every figure states its source and dataset type in its footer. Figures built from synthetic
data are watermarked.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from . import normalization  # noqa: E402

BLUE, ORANGE, GREEN, GREY, VERMILION = "#0072B2", "#E69F00", "#009E73", "#6B6B6B", "#D55E00"
SOURCE = "Karim et al. (2024), IJCESEN 10(4):2578-2589"
DATASET_NOTE = "dataset type: published aggregate"
SVG_METADATA = {"Date": None, "Creator": "grameen-empowerment"}

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "svg.hashsalt": "grameen-empowerment", "svg.fonttype": "none",
})


def save_figure(fig: plt.Figure, stem: str, outdir: Path) -> list[Path]:
    """Write PNG and SVG; returns the written paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    png, svg = outdir / f"{stem}.png", outdir / f"{stem}.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight", metadata=SVG_METADATA)
    plt.close(fig)
    return [png, svg]


def _footer(fig: plt.Figure, text: str) -> None:
    fig.subplots_adjust(bottom=0.2)
    fig.text(0.01, 0.005, text, fontsize=7.5, color=GREY, ha="left", va="bottom")


def _label_bars(ax: plt.Axes, values: list[float]) -> None:
    for i, v in enumerate(values):
        ax.text(i, v + 0.6, f"{v:g}%", ha="center", va="bottom", fontsize=10)


def figure2_income_by_sector(df: pd.DataFrame, outdir: Path) -> list[Path]:
    """Figure 2 / Table 4. 'Overall' is plotted as its own reported value, hatched."""
    t = df[df["source_table"] == "Table 4"]
    labels, vals = list(t["group_label"]), [float(v) for v in t["value"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(labels, vals, color=[BLUE, ORANGE, GREY], width=0.6)
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor("white")
    _label_bars(ax, vals)
    ax.set_ylim(0, 38)
    ax.set_ylabel("Income increase (%)")
    ax.set_title("Income growth by sector (Table 4)")
    ax.text(0.99, 0.97, "'Overall' is reported separately by the paper;\ngroup weights are unknown.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=GREY)
    _footer(fig, f"Source: {SOURCE}, Table 4, p. 2585; {DATASET_NOTE}")
    return save_figure(fig, "fig2_income_growth_by_sector", outdir)


def figure3_autonomy_by_age(df: pd.DataFrame, outdir: Path) -> list[Path]:
    """Figure 3 / Table 5, using the printed boundaries (under 30, 30-45, over 45)."""
    t = df[df["source_table"] == "Table 5"]
    labels, vals = list(t["group_label"]), [float(v) for v in t["value"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(labels, vals, color=VERMILION, marker="o", markersize=8, linewidth=1.8)
    for x, v in zip(labels, vals, strict=True):
        ax.annotate(f"{v:g}%", (x, v), textcoords="offset points", xytext=(0, 9), ha="center")
    ax.set_ylim(15, 45)
    ax.set_ylabel("Autonomy increase (%)")
    ax.set_xlabel("Age group (age 45 belongs to 30-45)")
    ax.set_title("Decision-making autonomy by age group (Table 5)")
    ax.grid(axis="y", alpha=0.3)
    _footer(fig, f"Source: {SOURCE}, Table 5, p. 2585; {DATASET_NOTE}")
    return save_figure(fig, "fig3_autonomy_by_age_group", outdir)


def figure4_social_mobility(df: pd.DataFrame, outdir: Path) -> list[Path]:
    """Primary Figure 4 replacement: the three reported Table 6 values, education and location kept."""
    t = df[df["source_table"] == "Table 6"]
    rows = [f"{r.group_label}\n({r.secondary_label} location)" for r in t.itertuples()]
    vals = [float(v) for v in t["value"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ypos = list(range(len(vals)))[::-1]
    ax.hlines(ypos, 0, vals, color=GREY, alpha=0.5, linewidth=1.5)
    ax.scatter(vals, ypos, s=90, color=GREEN, zorder=3)
    for y, v in zip(ypos, vals, strict=True):
        ax.annotate(f"{v:g}%", (v, y), textcoords="offset points", xytext=(10, -3))
    ax.set_yticks(ypos, rows)
    ax.set_xlim(0, 62)
    ax.set_xlabel("Social mobility (%) as printed in Table 6")
    ax.set_title("Social mobility by education level and location (Table 6)")
    ax.text(0.99, 0.97, "Education and location are paired one-to-one;\nseparate effects cannot be identified.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=GREY)
    _footer(fig, f"Source: {SOURCE}, Table 6, p. 2585; {DATASET_NOTE}")
    return save_figure(fig, "fig4_social_mobility_table6", outdir)


def figure4_pie_comparison(labels: pd.DataFrame, outdir: Path) -> list[Path]:
    """Comparison only: the pie as printed, with its normalisation stated in the caption."""
    comp = normalization.compare_with_printed(labels)
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    ax.pie(comp["table6_value_percent"], labels=comp["education_level"], colors=[ORANGE, GREEN, BLUE],
           autopct=lambda p: f"{p:.1f}%", startangle=90, counterclock=False, wedgeprops={"edgecolor": "white"})
    ax.set_title("Figure 4 as printed: Table 6 values divided by their sum (110)")
    caption = "\n".join(textwrap.wrap(normalization.CAPTION, 100))
    fig.text(0.5, 0.115, caption, fontsize=8, ha="center", va="top", color="#222222")
    _footer(fig, f"Source: {SOURCE}, Figure 4, p. 2586, and Table 6, p. 2585; {DATASET_NOTE}; comparison only")
    fig.subplots_adjust(bottom=0.27)
    return save_figure(fig, "fig4_pie_normalization_comparison", outdir)


def _box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, w: float = 2.6, h: float = 0.8) -> None:
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", fc=color, ec="#333333", lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


def _arrow(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], label: str = "") -> None:
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2})
    if label:
        ax.text((a[0] + b[0]) / 2 + 0.1, (a[1] + b[1]) / 2, label, fontsize=8, style="italic", ha="left")


def figure1_framework(outdir: Path) -> list[Path]:
    """Redrawn study framework (conceptual; no data). Empowerment branch and challenges branch."""
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.set_xlim(-6.4, 6.0)
    ax.set_ylim(0.8, 8.6)
    ax.axis("off")
    _box(ax, (0, 8.0), "Microcredit programs", "#D9ECF7")
    _box(ax, (0, 6.6), "Grameen Bank model", "#FFF6D6")
    _arrow(ax, (0, 7.6), (0, 7.0), "modeled by")
    _box(ax, (-3.0, 5.0), "Women's empowerment\n(outcomes branch)", "#CDEBDD", 3.0, 0.9)
    _box(ax, (3.0, 5.0), "Challenges\n(dependency branch)", "#F8D9C4", 3.0, 0.9)
    _arrow(ax, (-0.6, 6.2), (-2.6, 5.5), "facilitates")
    _arrow(ax, (0.6, 6.2), (2.6, 5.5), "leads to")
    _box(ax, (-3.0, 3.4), "Socio-economic outcomes", "#EBC9C9", 3.0)
    _arrow(ax, (-3.0, 4.55), (-3.0, 3.8), "improves")
    outcomes = ("Household\ndecision-making", "Income\nimprovement", "Education\naccess", "Social\nmobility")
    for x, t in zip((-5.6, -3.85, -2.1, -0.35), outcomes, strict=True):
        _box(ax, (x, 1.7), t, "#E6E6E6", 1.65, 0.9)
        _arrow(ax, (-3.0, 3.0), (x, 2.15))
    challenges = ("Dependency", "Repayment\npressure", "Sustainability\nconcerns")
    for x, t in zip((1.5, 3.1, 4.7), challenges, strict=True):
        _box(ax, (x, 3.4), t, "#F6D9E6", 1.5, 0.9)
        _arrow(ax, (3.0, 4.55), (x, 3.85))
    ax.set_title("Study framework (conceptual diagram; no data)", fontsize=12)
    _footer(fig, f"Redrawn from Figure 1, p. 2584 of {SOURCE}; dataset type: none (conceptual)")
    return save_figure(fig, "fig1_study_framework", outdir)


def watermark_synthetic(fig: plt.Figure) -> None:
    fig.text(0.5, 0.5, "SYNTHETIC", fontsize=46, color="#CC0000", alpha=0.16, ha="center", va="center", rotation=25)
