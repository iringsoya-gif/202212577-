"""
Visualization Module — Data Preprocessing
==========================================
전문가급 matplotlib 다크테마 시각화
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Design tokens ──────────────────────────────────────────────────────────
DARK_BG   = "#0f0f1a"
CARD_BG   = "#1a1a2e"
GRID_COL  = "#2d2d44"
TEXT_COL  = "#e2e8f0"
PURPLE    = "#6c63ff"
PINK      = "#ff6584"
TEAL      = "#43b89c"
AMBER     = "#ffc966"
BLUE      = "#61dafb"

SALARY_COL = PURPLE
AGE_COL    = TEAL
NORM_SALARY = PINK
NORM_AGE    = AMBER


def _apply_dark(fig, axes):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        if ax is None:
            continue
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.title.set_color(TEXT_COL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)
        ax.grid(color=GRID_COL, linestyle="--", linewidth=0.5, alpha=0.6)


def _millions_fmt(x, _):
    if abs(x) >= 1e8:
        return f"{x/1e8:.1f}B"
    elif abs(x) >= 1e6:
        return f"{x/1e6:.0f}M"
    elif abs(x) >= 1e4:
        return f"{x/1e4:.0f}K"
    return f"{x:.0f}"


def plot_raw_distribution(X: np.ndarray, filename: str = "raw_distribution.png") -> str:
    """정규화 전 원본 데이터 분포 (2×2 subplot)
    Row 1: Salary histogram + scatter
    Row 2: Age histogram + box comparison
    """
    salary, age = X[:, 0], X[:, 1]

    fig = plt.figure(figsize=(14, 9))
    _apply_dark(fig, [])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── [0,0] Salary Histogram ─────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    _apply_dark(fig, [ax0])
    n, bins, patches = ax0.hist(salary, bins=30, color=SALARY_COL, alpha=0.75, edgecolor="none")
    # gradient color on bars
    norm_vals = (bins[:-1] - bins[:-1].min()) / (bins[:-1].max() - bins[:-1].min() + 1)
    for patch, nv in zip(patches, norm_vals):
        patch.set_facecolor(plt.cm.cool(0.3 + nv * 0.5))
    ax0.axvline(salary.mean(), color=PINK, linestyle="--", linewidth=1.5, label=f"Mean: {salary.mean()/1e6:.1f}M")
    ax0.set_xlabel("Salary (KRW)", fontsize=10)
    ax0.set_ylabel("Count", fontsize=10)
    ax0.set_title("Salary Distribution (Before)", fontsize=12, fontweight="bold")
    ax0.xaxis.set_major_formatter(FuncFormatter(_millions_fmt))
    ax0.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)
    # stats box
    ax0.text(0.97, 0.95,
             f"min: {salary.min()/1e6:.0f}M\nmax: {salary.max()/1e6:.0f}M\nstd: {salary.std()/1e6:.1f}M",
             transform=ax0.transAxes, ha="right", va="top", fontsize=8,
             color=TEXT_COL, fontfamily="monospace",
             bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL, alpha=0.8, boxstyle="round,pad=0.3"))

    # ── [0,1] Age Histogram ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    _apply_dark(fig, [ax1])
    n2, bins2, patches2 = ax1.hist(age, bins=25, color=AGE_COL, alpha=0.75, edgecolor="none")
    norm_vals2 = (bins2[:-1] - bins2[:-1].min()) / (bins2[:-1].max() - bins2[:-1].min() + 1)
    for patch, nv in zip(patches2, norm_vals2):
        patch.set_facecolor(plt.cm.YlOrRd(0.3 + nv * 0.5))
    ax1.axvline(age.mean(), color=PINK, linestyle="--", linewidth=1.5, label=f"Mean: {age.mean():.1f} yrs")
    ax1.set_xlabel("Age (years)", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.set_title("Age Distribution (Before)", fontsize=12, fontweight="bold")
    ax1.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)
    ax1.text(0.97, 0.95,
             f"min: {age.min():.0f} yrs\nmax: {age.max():.0f} yrs\nstd: {age.std():.1f} yrs",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8,
             color=TEXT_COL, fontfamily="monospace",
             bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL, alpha=0.8, boxstyle="round,pad=0.3"))

    # ── [1,0] Salary vs Age Scatter ────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _apply_dark(fig, [ax2])
    sc = ax2.scatter(age, salary, c=salary, cmap="cool", alpha=0.5, s=20, edgecolors="none")
    plt.colorbar(sc, ax=ax2, label="Salary").ax.yaxis.set_tick_params(color=TEXT_COL)
    ax2.set_xlabel("Age (years)", fontsize=10)
    ax2.set_ylabel("Salary (KRW)", fontsize=10)
    ax2.set_title("Salary vs Age — Raw Scale Problem", fontsize=12, fontweight="bold")
    ax2.yaxis.set_major_formatter(FuncFormatter(_millions_fmt))
    # annotation
    ax2.text(0.03, 0.95, "Salary range: ~125M KRW\nAge range: ~43 yrs\n→ Scale mismatch!",
             transform=ax2.transAxes, va="top", fontsize=8, color=PINK,
             bbox=dict(facecolor=DARK_BG, edgecolor=PINK, alpha=0.8, boxstyle="round,pad=0.3"))

    # ── [1,1] Range comparison bar ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _apply_dark(fig, [ax3])
    features = ["Salary\n(KRW)", "Age\n(years)"]
    ranges = [salary.max() - salary.min(), age.max() - age.min()]
    colors_bar = [SALARY_COL, AGE_COL]
    bars = ax3.bar(features, ranges, color=colors_bar, alpha=0.8, width=0.5)
    for bar, val in zip(bars, ranges):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f"{_millions_fmt(val, None)}", ha="center", va="bottom",
                 color=TEXT_COL, fontsize=10, fontweight="bold")
    ax3.set_title("Feature Range Comparison", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Value Range", fontsize=10)
    ax3.yaxis.set_major_formatter(FuncFormatter(_millions_fmt))
    ax3.text(0.5, 0.5, "~3,000x\nDifference!", transform=ax3.transAxes,
             ha="center", va="center", fontsize=14, fontweight="bold",
             color=PINK, alpha=0.7)

    fig.suptitle("Before Normalization — Raw Feature Distributions", fontsize=15,
                 fontweight="bold", color=TEXT_COL, y=0.98)

    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(OUTPUT_DIR / filename)


def plot_normalized_distribution(X_mm: np.ndarray, X_zs: np.ndarray,
                                  filename: str = "normalized_distribution.png") -> str:
    """정규화 후 분포 (Min-Max vs Z-Score 비교)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _apply_dark(fig, axes.ravel())

    sal_mm, age_mm = X_mm[:, 0], X_mm[:, 1]
    sal_zs, age_zs = X_zs[:, 0], X_zs[:, 1]

    # Min-Max salary
    axes[0, 0].hist(sal_mm, bins=30, color=NORM_SALARY, alpha=0.8, edgecolor="none")
    axes[0, 0].axvline(sal_mm.mean(), color=AMBER, linestyle="--", linewidth=1.5,
                        label=f"Mean: {sal_mm.mean():.3f}")
    axes[0, 0].set_xlim(-0.05, 1.05)
    axes[0, 0].set_title("Salary — Min-Max Scaled", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Normalized Value [0, 1]", fontsize=10)
    axes[0, 0].legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
                       labelcolor=TEXT_COL, fontsize=9)

    # Min-Max age
    axes[0, 1].hist(age_mm, bins=25, color=NORM_AGE, alpha=0.8, edgecolor="none")
    axes[0, 1].axvline(age_mm.mean(), color=PINK, linestyle="--", linewidth=1.5,
                        label=f"Mean: {age_mm.mean():.3f}")
    axes[0, 1].set_xlim(-0.05, 1.05)
    axes[0, 1].set_title("Age — Min-Max Scaled", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Normalized Value [0, 1]", fontsize=10)
    axes[0, 1].legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
                       labelcolor=TEXT_COL, fontsize=9)

    # Z-Score salary
    axes[1, 0].hist(sal_zs, bins=30, color=PURPLE, alpha=0.8, edgecolor="none")
    axes[1, 0].axvline(0, color=TEAL, linestyle="--", linewidth=1.5, label="Mean = 0")
    axes[1, 0].set_title("Salary — Z-Score Standardized", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Standard Deviations from Mean", fontsize=10)
    axes[1, 0].legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
                       labelcolor=TEXT_COL, fontsize=9)

    # Z-Score age
    axes[1, 1].hist(age_zs, bins=25, color=BLUE, alpha=0.8, edgecolor="none")
    axes[1, 1].axvline(0, color=TEAL, linestyle="--", linewidth=1.5, label="Mean = 0")
    axes[1, 1].set_title("Age — Z-Score Standardized", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Standard Deviations from Mean", fontsize=10)
    axes[1, 1].legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
                       labelcolor=TEXT_COL, fontsize=9)

    fig.suptitle("After Normalization — Min-Max [0,1] vs Z-Score", fontsize=15,
                 fontweight="bold", color=TEXT_COL, y=0.98)
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(OUTPUT_DIR / filename)


def plot_comparison(X_raw: np.ndarray, X_mm: np.ndarray,
                    filename: str = "comparison_plot.png") -> str:
    """전/후 side-by-side 산점도 비교"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark(fig, [ax1, ax2])

    sal_raw, age_raw = X_raw[:, 0], X_raw[:, 1]
    sal_mm, age_mm   = X_mm[:, 0], X_mm[:, 1]

    # Before
    sc1 = ax1.scatter(age_raw, sal_raw, c=sal_raw, cmap="cool", alpha=0.55, s=25, edgecolors="none")
    cb1 = plt.colorbar(sc1, ax=ax1)
    cb1.ax.yaxis.set_tick_params(color=TEXT_COL)
    cb1.ax.yaxis.set_major_formatter(FuncFormatter(_millions_fmt))
    ax1.set_xlabel("Age (years)", fontsize=11)
    ax1.set_ylabel("Salary (KRW)", fontsize=11)
    ax1.set_title("BEFORE Normalization", fontsize=13, fontweight="bold", color=PINK)
    ax1.yaxis.set_major_formatter(FuncFormatter(_millions_fmt))
    ax1.text(0.03, 0.97,
             f"Salary std: {sal_raw.std()/1e6:.1f}M KRW\nAge std: {age_raw.std():.1f} yrs\n"
             f"→ Extremely different scales",
             transform=ax1.transAxes, va="top", fontsize=9, color=PINK,
             bbox=dict(facecolor=DARK_BG, edgecolor=PINK, alpha=0.85, boxstyle="round,pad=0.4"))

    # After
    sc2 = ax2.scatter(age_mm, sal_mm, c=sal_mm, cmap="plasma", alpha=0.55, s=25, edgecolors="none")
    cb2 = plt.colorbar(sc2, ax=ax2, label="Salary (normalized)")
    cb2.ax.yaxis.set_tick_params(color=TEXT_COL)
    ax2.set_xlabel("Age (normalized)", fontsize=11)
    ax2.set_ylabel("Salary (normalized)", fontsize=11)
    ax2.set_title("AFTER Min-Max Normalization", fontsize=13, fontweight="bold", color=TEAL)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.text(0.03, 0.97,
             f"Salary std: {sal_mm.std():.3f}\nAge std: {age_mm.std():.3f}\n"
             f"→ Comparable scales [0,1]",
             transform=ax2.transAxes, va="top", fontsize=9, color=TEAL,
             bbox=dict(facecolor=DARK_BG, edgecolor=TEAL, alpha=0.85, boxstyle="round,pad=0.4"))

    # Arrow between subplots
    fig.text(0.495, 0.5, "→", fontsize=30, ha="center", va="center",
             color=AMBER, fontweight="bold")
    fig.text(0.495, 0.42, "normalize", fontsize=9, ha="center", va="center",
             color=AMBER, alpha=0.8)

    fig.suptitle("Before vs After Normalization — Scatter Plot Comparison",
                 fontsize=14, fontweight="bold", color=TEXT_COL, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(OUTPUT_DIR / filename)


def plot_variance_comparison(X_raw: np.ndarray, X_mm: np.ndarray, X_zs: np.ndarray,
                              filename: str = "variance_comparison.png") -> str:
    """분산 변화 비교 (원본 / Min-Max / Z-Score)"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    _apply_dark(fig, axes)

    features = ["Salary", "Age"]
    methods  = [("Raw", X_raw), ("Min-Max", X_mm), ("Z-Score", X_zs)]
    colors   = [SALARY_COL, AGE_COL]

    # Variance bar chart per method
    x_pos = np.arange(len(features))
    width = 0.35

    for ax_idx, (method, X) in enumerate(methods):
        ax = axes[ax_idx]
        variances = [float(X[:, i].var()) for i in range(2)]

        bars = ax.bar(x_pos, variances, width=0.55, color=colors, alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, variances):
            label = _millions_fmt(val, None) if ax_idx == 0 and val > 1e6 else f"{val:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    label, ha="center", va="bottom", fontsize=9,
                    color=TEXT_COL, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(features, fontsize=10)
        ax.set_title(f"{method}\nVariance", fontsize=11, fontweight="bold",
                     color=[PINK, TEAL, AMBER][ax_idx])
        ax.set_ylabel("Variance", fontsize=9)
        if ax_idx == 0:
            ax.yaxis.set_major_formatter(FuncFormatter(_millions_fmt))

    # Box plots side-by-side in each axis using twin axis for distribution shape
    # Add "shape" indicator
    for ax_idx, (method, X) in enumerate(methods):
        ax = axes[ax_idx]
        desc = {
            "Raw": "Salary: ~4×10¹⁴\nAge: ~100\nScale: ×10¹²!",
            "Min-Max": "Both in [0, 1]\nVariance < 1\nComparable!",
            "Z-Score": "Mean = 0\nStd = 1\nSymmetric",
        }
        ax.text(0.97, 0.95, desc[method], transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=TEXT_COL,
                fontfamily="monospace",
                bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL, alpha=0.85, boxstyle="round,pad=0.3"))

    fig.suptitle("Variance Change: Raw → Min-Max → Z-Score",
                 fontsize=14, fontweight="bold", color=TEXT_COL)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(OUTPUT_DIR / filename)


def plot_prediction(X_raw: np.ndarray, X_mm: np.ndarray,
                    new_raw: np.ndarray, new_mm: np.ndarray,
                    filename: str = "prediction_plot.png") -> str:
    """사용자 입력 정규화 결과 시각화 (전/후 위치 표시)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark(fig, [ax1, ax2])

    sal_raw, age_raw = X_raw[:, 0], X_raw[:, 1]
    sal_mm,  age_mm  = X_mm[:, 0],  X_mm[:, 1]
    n_sal, n_age     = float(new_raw[0]), float(new_raw[1])
    n_sal_mm, n_age_mm = float(new_mm[0, 0]), float(new_mm[0, 1])

    # Before: scatter + new point
    ax1.scatter(age_raw, sal_raw, c=GRID_COL, alpha=0.35, s=18, edgecolors="none")
    ax1.scatter([n_age], [n_sal], c=PINK, marker="*", s=500,
                edgecolors="white", linewidths=0.8, zorder=5, label="Your Input")
    ax1.axhline(n_sal, color=PINK, linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.axvline(n_age, color=PINK, linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Age (years)", fontsize=11)
    ax1.set_ylabel("Salary (KRW)", fontsize=11)
    ax1.set_title("Input Position — Raw Scale", fontsize=13, fontweight="bold")
    ax1.yaxis.set_major_formatter(FuncFormatter(_millions_fmt))
    ax1.text(0.03, 0.97,
             f"Salary: {_millions_fmt(n_sal, None)}\nAge: {n_age:.0f} yrs",
             transform=ax1.transAxes, va="top", fontsize=10, color=PINK, fontweight="bold",
             bbox=dict(facecolor=DARK_BG, edgecolor=PINK, alpha=0.85, boxstyle="round,pad=0.4"))
    ax1.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL, fontsize=10)

    # After: normalized scatter + new point
    ax2.scatter(age_mm, sal_mm, c=GRID_COL, alpha=0.35, s=18, edgecolors="none")
    ax2.scatter([n_age_mm], [n_sal_mm], c=TEAL, marker="*", s=500,
                edgecolors="white", linewidths=0.8, zorder=5, label="Your Input (normalized)")
    ax2.axhline(n_sal_mm, color=TEAL, linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.axvline(n_age_mm, color=TEAL, linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Age (normalized)", fontsize=11)
    ax2.set_ylabel("Salary (normalized)", fontsize=11)
    ax2.set_title("Input Position — Normalized [0, 1]", fontsize=13, fontweight="bold")
    ax2.text(0.03, 0.97,
             f"Salary: {n_sal_mm:.4f}\nAge:    {n_age_mm:.4f}",
             transform=ax2.transAxes, va="top", fontsize=10, color=TEAL,
             fontweight="bold", fontfamily="monospace",
             bbox=dict(facecolor=DARK_BG, edgecolor=TEAL, alpha=0.85, boxstyle="round,pad=0.4"))
    ax2.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL, fontsize=10)

    # Percentile info
    sal_pct = float((sal_raw < n_sal).mean() * 100)
    age_pct = float((age_raw < n_age).mean() * 100)
    fig.text(0.5, -0.02,
             f"Salary Percentile: {sal_pct:.1f}th  |  Age Percentile: {age_pct:.1f}th",
             ha="center", fontsize=11, color=AMBER, fontweight="bold")

    fig.suptitle("Prediction — Where Does Your Input Land?",
                 fontsize=14, fontweight="bold", color=TEXT_COL)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(OUTPUT_DIR / filename)
