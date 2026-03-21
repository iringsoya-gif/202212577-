"""
Visualizer — Gradient Descent
===============================
전문가급 matplotlib 다크테마 시각화
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from pathlib import Path

from gradient_descent import GDResult, compare_learning_rates

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Design tokens ──────────────────────────────────────────────────────────
DARK_BG  = "#0f0f1a"
CARD_BG  = "#1a1a2e"
GRID_COL = "#2d2d44"
TEXT_COL = "#e2e8f0"
PURPLE   = "#6c63ff"
PINK     = "#ff6584"
TEAL     = "#43b89c"
AMBER    = "#ffc966"
BLUE     = "#61dafb"
RED      = "#f87171"


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


def _gradient_segments(xs, ys, cmap_name="cool"):
    """Returns a LineCollection with gradient color along the path."""
    points  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    n       = len(segs)
    colors  = [plt.get_cmap(cmap_name)(i / max(n - 1, 1)) for i in range(n)]
    lc = LineCollection(segs, colors=colors, linewidth=2.2, zorder=3)
    return lc


def plot_gd_path(result: GDResult, filename: str = "gd_path.png") -> str:
    """경사 하강법 경로 — f(x)=x² 포물선 위에 공의 이동 시각화"""
    fig, ax = plt.subplots(figsize=(12, 7))
    _apply_dark(fig, ax)

    # ── Parabola curve ────────────────────────────────────
    x_range = max(abs(result.x_start) * 1.25, 2.0)
    xs_curve = np.linspace(-x_range, x_range, 600)
    ys_curve = xs_curve ** 2

    # Fill under curve
    ax.fill_between(xs_curve, ys_curve, alpha=0.06, color=PURPLE)

    # Main parabola
    ax.plot(xs_curve, ys_curve, color=PURPLE, linewidth=2.0, alpha=0.8,
            label="f(x) = x²", zorder=2)

    # ── Step points ───────────────────────────────────────
    steps_arr  = result.steps
    xs_steps   = np.array([s.x    for s in steps_arr])
    ys_steps   = np.array([s.loss for s in steps_arr])

    n_show = min(len(steps_arr), 80)   # cap dots for readability

    # Gradient-colored path line
    if len(xs_steps) > 1:
        lc = _gradient_segments(xs_steps[:n_show], ys_steps[:n_show], "cool")
        ax.add_collection(lc)

    # Step dots — sized by recency (early=small, recent=larger)
    if len(xs_steps) > 2:
        mid_x = xs_steps[1:n_show-1]
        mid_y = ys_steps[1:n_show-1]
        sizes = np.linspace(10, 35, len(mid_x))
        alphas = np.linspace(0.25, 0.65, len(mid_x))
        for xi, yi, sz, al in zip(mid_x, mid_y, sizes, alphas):
            ax.scatter(xi, yi, s=sz, c=TEAL, alpha=al, edgecolors="none", zorder=4)

    # ── Start & End markers ───────────────────────────────
    ax.scatter(xs_steps[0],  ys_steps[0], s=260, marker="*",
               c=PINK, edgecolors="white", linewidths=0.8, zorder=6,
               label=f"Start  x={xs_steps[0]:.2f}")
    ax.scatter(xs_steps[-1], ys_steps[-1], s=260, marker="*",
               c=TEAL, edgecolors="white", linewidths=0.8, zorder=6,
               label=f"End    x={xs_steps[-1]:.5f}")

    # Minimum annotation
    ax.scatter(0, 0, s=120, marker="D", c=AMBER,
               edgecolors="white", linewidths=0.8, zorder=7, label="Global Min (0, 0)")

    # ── Gradient arrow at start ───────────────────────────
    if len(steps_arr) > 1:
        x0, y0 = xs_steps[0], ys_steps[0]
        dx = xs_steps[1] - x0
        dy = ys_steps[1] - y0
        ax.annotate("", xy=(x0 + dx * 0.8, y0 + dy * 0.8), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
        ax.text(x0 + dx * 0.5, y0 + dy * 0.5 + max(ys_curve) * 0.03,
                f"  -α∇f  ", fontsize=9, color=AMBER, ha="center")

    # ── Stats box ─────────────────────────────────────────
    status = "Converged" if result.converged else ("DIVERGED" if result.diverged else "Max steps")
    color_s = TEAL if result.converged else (RED if result.diverged else AMBER)
    info_text = (
        f"α (lr):    {result.learning_rate}\n"
        f"Steps:     {result.n_steps}\n"
        f"Status:    {status}\n"
        f"Final x:   {result.final_x:.6f}\n"
        f"Final loss:{result.final_loss:.2e}\n"
        f"Reduction: {result.convergence_pct:.2f}%"
    )
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
            va="top", fontsize=8.5, color=TEXT_COL, fontfamily="monospace",
            bbox=dict(facecolor=DARK_BG, edgecolor=color_s, alpha=0.9,
                      boxstyle="round,pad=0.5"))

    # ── Divergence warning ────────────────────────────────
    if result.diverged:
        ax.text(0.5, 0.5, "DIVERGED!\nα too large", transform=ax.transAxes,
                ha="center", va="center", fontsize=20, fontweight="bold",
                color=RED, alpha=0.25)

    ax.set_xlabel("x", fontsize=12, labelpad=8)
    ax.set_ylabel("f(x) = x²  (Loss)", fontsize=12, labelpad=8)
    ax.set_title(f"Gradient Descent Path  |  α={result.learning_rate}  |  Start x={result.x_start}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
              labelcolor=TEXT_COL, fontsize=9, loc="upper right")

    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(out_path)


def plot_loss_curve(result: GDResult, filename: str = "loss_curve.png") -> str:
    """Step별 Loss 감소 곡선"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _apply_dark(fig, [ax1, ax2])

    steps_arr = result.steps
    step_nums = np.array([s.step    for s in steps_arr])
    losses    = np.array([s.loss    for s in steps_arr])
    grads     = np.array([s.gradient for s in steps_arr])

    # ── Left: Loss curve ──────────────────────────────────
    cmap  = LinearSegmentedColormap.from_list("lg", [PINK, TEAL])
    n     = len(step_nums)
    for i in range(n - 1):
        c = cmap(i / max(n - 1, 1))
        ax1.plot(step_nums[i:i+2], losses[i:i+2], color=c, linewidth=2.2)

    ax1.scatter(step_nums, losses, c=PURPLE, s=20, zorder=5, alpha=0.6)
    ax1.scatter(step_nums[-1], losses[-1], c=TEAL, s=120, zorder=6,
                edgecolors="white", linewidths=0.8,
                label=f"Final loss: {losses[-1]:.2e}")

    # Log scale if range is large
    if losses[0] > 0 and losses[-1] > 0:
        if losses[0] / (losses[-1] + 1e-30) > 100:
            ax1.set_yscale("log")
            ax1.set_ylabel("Loss (log scale)", fontsize=11)
        else:
            ax1.set_ylabel("Loss  f(x) = x²", fontsize=11)
    else:
        ax1.set_ylabel("Loss  f(x) = x²", fontsize=11)

    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_title("Loss vs Step", fontsize=13, fontweight="bold")
    ax1.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL, fontsize=9)

    # Reduction label
    if losses[0] > 0:
        red = (losses[0] - losses[-1]) / losses[0] * 100
        ax1.text(0.97, 0.95, f"Total reduction\n{red:.2f}%",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=9, color=TEAL, fontfamily="monospace",
                 bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL, alpha=0.85,
                           boxstyle="round,pad=0.4"))

    # ── Right: Gradient magnitude ─────────────────────────
    abs_grads = np.abs(grads)
    ax2.fill_between(step_nums, abs_grads, alpha=0.25, color=AMBER)
    ax2.plot(step_nums, abs_grads, color=AMBER, linewidth=2.0)
    ax2.scatter(step_nums[-1], abs_grads[-1], c=TEAL, s=120, zorder=5,
                edgecolors="white", linewidths=0.8,
                label=f"|grad| final: {abs_grads[-1]:.2e}")
    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("|Gradient|  |f'(x)| = |2x|", fontsize=11)
    ax2.set_title("Gradient Magnitude vs Step", fontsize=13, fontweight="bold")
    ax2.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
               labelcolor=TEXT_COL, fontsize=9)

    # Zero line
    ax2.axhline(0, color=GRID_COL, linewidth=0.8)

    fig.suptitle(f"Convergence Analysis  |  α={result.learning_rate}  |  {result.n_steps} steps",
                 fontsize=13, fontweight="bold", color=TEXT_COL)
    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(out_path)


def plot_comparison(x_start: float = 8.0, filename: str = "comparison.png") -> str:
    """여러 학습률의 수렴 속도 비교"""
    results = compare_learning_rates(x_start=x_start, max_steps=60)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _apply_dark(fig, axes)

    colors_map = {
        "Too Slow (α=0.01)":   BLUE,
        "Optimal (α=0.1)":     TEAL,
        "Fast (α=0.45)":       AMBER,
        "Diverging (α=1.05)":  RED,
    }

    # ── Left: Path on parabola ────────────────────────────
    x_range = abs(x_start) * 1.3
    xs_c    = np.linspace(-x_range, x_range, 500)
    axes[0].plot(xs_c, xs_c**2, color=PURPLE, linewidth=2.0, alpha=0.7, label="f(x)=x²", zorder=1)
    axes[0].fill_between(xs_c, xs_c**2, alpha=0.05, color=PURPLE)
    axes[0].scatter(0, 0, s=150, marker="D", c=AMBER, edgecolors="white",
                    linewidths=0.8, zorder=8, label="Minimum")

    for label, res in results.items():
        c = colors_map[label]
        xs_s = [s.x    for s in res.steps]
        ys_s = [s.loss for s in res.steps]
        n = min(len(xs_s), 30)
        axes[0].plot(xs_s[:n], ys_s[:n], color=c, linewidth=1.5, alpha=0.85, zorder=3)
        axes[0].scatter(xs_s[0], ys_s[0], s=80, c=c, marker="o",
                        edgecolors="white", linewidths=0.5, zorder=5)
        axes[0].scatter(xs_s[-1], ys_s[-1], s=80, c=c, marker="*",
                        edgecolors="white", linewidths=0.5, zorder=5, label=label)

    axes[0].set_xlabel("x", fontsize=11)
    axes[0].set_ylabel("f(x) = x²", fontsize=11)
    axes[0].set_title("Path Comparison — 4 Learning Rates", fontsize=12, fontweight="bold")
    axes[0].legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL, fontsize=8, loc="upper right")

    # ── Right: Loss curve comparison ──────────────────────
    for label, res in results.items():
        c = colors_map[label]
        steps_n = [s.step for s in res.steps]
        losses  = [min(s.loss, 1e5) for s in res.steps]  # cap for display
        axes[1].plot(steps_n, losses, color=c, linewidth=2.0, alpha=0.9,
                     label=f"{label} ({res.n_steps}s)")

    axes[1].set_xlabel("Step", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_title("Loss Convergence Comparison", fontsize=12, fontweight="bold")
    axes[1].set_yscale("symlog")
    axes[1].legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COL,
                   labelcolor=TEXT_COL, fontsize=8)

    # Guidance text
    guidance = (
        "α too small → slow convergence\n"
        "α optimal  → fast convergence\n"
        "α too large → oscillation/divergence"
    )
    axes[1].text(0.97, 0.05, guidance, transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=8, color=TEXT_COL,
                 fontfamily="monospace",
                 bbox=dict(facecolor=DARK_BG, edgecolor=GRID_COL, alpha=0.85,
                           boxstyle="round,pad=0.4"))

    fig.suptitle("Learning Rate Comparison  (f(x) = x²)",
                 fontsize=14, fontweight="bold", color=TEXT_COL)
    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(out_path)
