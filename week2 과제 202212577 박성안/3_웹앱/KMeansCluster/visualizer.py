"""
Visualization Module for K-Means Clustering
============================================
전문가급 시각화: matplotlib dark theme 기반
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없이 PNG 저장
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 팔레트 (Design Spec과 동일) ──────────────────────────────────────────────
DARK_BG = "#0f0f1a"
CARD_BG = "#1a1a2e"
CLUSTER_COLORS = ["#6c63ff", "#ff6584", "#43b89c", "#ffc966", "#61dafb", "#f97316"]
GRID_COLOR = "#2d2d44"
TEXT_COLOR = "#e2e8f0"
ACCENT = "#6c63ff"


def _apply_dark_style(fig, axes):
    """다크 테마 공통 스타일 적용"""
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=10)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)


def plot_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    k: int,
    highlight_point: np.ndarray = None,
    highlight_cluster: int = None,
    filename: str = "cluster_plot.png",
) -> str:
    """클러스터 산점도 생성 및 PNG 저장

    Parameters
    ----------
    highlight_point : ndarray (2,)  새 고객 좌표 (예측 시 표시)
    highlight_cluster : int         새 고객의 예측 클러스터
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_dark_style(fig, ax)

    legend_patches = []
    cluster_names = {
        0: "Regular (Group A)",
        1: "Premium (Group B)",
        2: "VIP (Group C)",
    }

    for j in range(k):
        mask = labels == j
        color = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
        name = cluster_names.get(j, f"Cluster {j}")

        ax.scatter(
            X[mask, 1], X[mask, 0],  # x=방문횟수, y=구매금액
            c=color, alpha=0.65, s=60,
            edgecolors="none", zorder=2,
        )
        # 중심점 별표
        ax.scatter(
            centroids[j, 1], centroids[j, 0],
            c=color, marker="*", s=400,
            edgecolors="white", linewidths=0.8, zorder=5,
        )
        legend_patches.append(mpatches.Patch(color=color, label=name))

    # 새 고객 예측 포인트
    if highlight_point is not None:
        hc = CLUSTER_COLORS[highlight_cluster % len(CLUSTER_COLORS)] if highlight_cluster is not None else "#ff0000"
        ax.scatter(
            highlight_point[1], highlight_point[0],
            c="#ff4444", marker="*", s=600,
            edgecolors="white", linewidths=1.0, zorder=6, label="New Customer",
        )
        ax.annotate(
            "  New Customer",
            xy=(highlight_point[1], highlight_point[0]),
            fontsize=11, color="#ff4444", fontweight="bold",
            xytext=(10, 10), textcoords="offset points",
        )
        legend_patches.append(mpatches.Patch(color="#ff4444", label="New Customer"))

    ax.set_xlabel("Visit Count (times/month)", fontsize=12, labelpad=10)
    ax.set_ylabel("Purchase Amount (10k KRW)", fontsize=12, labelpad=10)
    ax.set_title(f"K-Means Customer Segmentation  (K={k})", fontsize=16, fontweight="bold", pad=15)

    legend = ax.legend(
        handles=legend_patches, loc="upper left",
        framealpha=0.3, facecolor=CARD_BG,
        edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=10,
    )

    ax.text(
        0.99, 0.02,
        f"N={len(X)} samples  ·  K={k} clusters",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#aaaacc",
    )

    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(out_path)


def _detect_elbow(values: list) -> int:
    """Detect elbow point using maximum perpendicular distance method.

    Draws a line from first to last point; the point with max distance
    to that line is the elbow.
    Returns index (0-based) of the elbow point.
    """
    n = len(values)
    if n < 3:
        return 0
    arr = np.array(values, dtype=float)
    xs = np.arange(n, dtype=float)
    # Normalize
    xs_n = (xs - xs[0]) / (xs[-1] - xs[0] + 1e-12)
    ys_n = (arr - arr[-1]) / (arr[0] - arr[-1] + 1e-12)
    # Direction vector of the line (start → end)
    line_vec = np.array([xs_n[-1] - xs_n[0], ys_n[-1] - ys_n[0]])
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return 0
    unit = line_vec / line_len
    # Perpendicular distance of each point to the line
    dists = []
    for i in range(n):
        pt = np.array([xs_n[i] - xs_n[0], ys_n[i] - ys_n[0]])
        proj = np.dot(pt, unit) * unit
        perp = pt - proj
        dists.append(np.linalg.norm(perp))
    return int(np.argmax(dists))


def plot_wcss_iterations(
    wcss_history: list,
    filename: str = "wcss_plot.png",
) -> str:
    """WCSS vs Iteration chart with auto-detected elbow / convergence annotation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, ax)

    iters = list(range(1, len(wcss_history) + 1))

    # Gradient colored line
    cmap = LinearSegmentedColormap.from_list("grad", [CLUSTER_COLORS[0], CLUSTER_COLORS[2]])
    colors = [cmap(i / max(len(iters) - 1, 1)) for i in range(len(iters))]

    for i in range(len(iters) - 1):
        ax.plot(iters[i:i+2], wcss_history[i:i+2], color=colors[i], linewidth=2.5, solid_capstyle="round")

    # All data points
    ax.scatter(iters, wcss_history, c=CLUSTER_COLORS[0], s=40, zorder=5)

    # Auto-detect elbow point
    elbow_idx = _detect_elbow(wcss_history)
    elbow_iter = iters[elbow_idx]
    elbow_wcss = wcss_history[elbow_idx]
    ax.scatter([elbow_iter], [elbow_wcss], c=CLUSTER_COLORS[3], s=200, zorder=7,
               edgecolors="white", linewidths=1.2, marker="v",
               label=f"Elbow (iter={elbow_iter})")
    ax.annotate(
        f"  Elbow\n  WCSS={elbow_wcss:,.0f}",
        xy=(elbow_iter, elbow_wcss), xytext=(elbow_iter + 0.3, elbow_wcss * 1.05),
        fontsize=8, color=CLUSTER_COLORS[3],
        arrowprops=dict(arrowstyle="->", color=CLUSTER_COLORS[3], lw=0.8),
    )

    # Convergence point
    ax.scatter(iters[-1], wcss_history[-1], c=CLUSTER_COLORS[2], s=150, zorder=6,
               edgecolors="white", linewidths=1.0, label=f"Converged (iter={iters[-1]})")

    # Stats box
    reduction = (wcss_history[0] - wcss_history[-1]) / wcss_history[0] * 100
    stats_text = (
        f"Init WCSS : {wcss_history[0]:>10,.0f}\n"
        f"Final WCSS: {wcss_history[-1]:>10,.0f}\n"
        f"Reduction : {reduction:>9.1f}%"
    )
    ax.text(
        0.98, 0.95, stats_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color=TEXT_COLOR, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_BG, edgecolor=GRID_COLOR, alpha=0.85),
    )

    ax.set_xlabel("Iteration", fontsize=12, labelpad=10)
    ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)", fontsize=12, labelpad=10)
    ax.set_title("WCSS Convergence by Iteration  (K-Means++)", fontsize=16, fontweight="bold", pad=15)
    ax.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=10)

    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(out_path)


def plot_elbow(wcss_by_k: dict, optimal_k: int = None, filename: str = "k_selection_plot.png") -> str:
    """K 선택 Elbow Method 그래프"""
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, ax)

    ks = list(wcss_by_k.keys())
    wcss_vals = list(wcss_by_k.values())

    ax.plot(ks, wcss_vals, color=CLUSTER_COLORS[0], linewidth=2.5, marker="o",
            markersize=8, markerfacecolor=CLUSTER_COLORS[1], markeredgecolor="white",
            markeredgewidth=0.8)

    if optimal_k is not None and optimal_k in wcss_by_k:
        ax.scatter([optimal_k], [wcss_by_k[optimal_k]], c=CLUSTER_COLORS[2],
                   s=200, zorder=6, edgecolors="white", linewidths=1.5,
                   label=f"Optimal K = {optimal_k}")
        ax.axvline(x=optimal_k, color=CLUSTER_COLORS[2], linestyle="--", linewidth=1.2, alpha=0.5)
        ax.legend(framealpha=0.3, facecolor=CARD_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=10)

    ax.set_xlabel("Number of Clusters (K)", fontsize=12, labelpad=10)
    ax.set_ylabel("WCSS (Inertia)", fontsize=12, labelpad=10)
    ax.set_title("Elbow Method — Optimal K Selection", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(ks)

    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return str(out_path)


def plot_prediction(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    k: int,
    new_point: np.ndarray,
    predicted_cluster: int,
    filename: str = "prediction_plot.png",
) -> str:
    """예측 결과 시각화 (cluster_plot + 신규 고객 강조)"""
    return plot_clusters(
        X, labels, centroids, k,
        highlight_point=new_point,
        highlight_cluster=predicted_cluster,
        filename=filename,
    )
