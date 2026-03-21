"""
Gradient Descent — Pure NumPy Implementation
=============================================
f(x) = x²  손실함수에서 최솟값을 찾는 경사 하강법
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class GDStep:
    step:     int
    x:        float
    loss:     float
    gradient: float
    delta_x:  float   # x_new - x_old


@dataclass
class GDResult:
    steps:           List[GDStep]
    converged:       bool
    diverged:        bool
    final_x:         float
    final_loss:      float
    x_start:         float
    loss_start:      float
    learning_rate:   float
    n_steps:         int
    convergence_pct: float   # (loss_start - final_loss) / loss_start * 100


def loss_fn(x: float) -> float:
    """손실 함수: f(x) = x²"""
    return float(x ** 2)


def gradient_fn(x: float) -> float:
    """그래디언트: f'(x) = 2x"""
    return float(2.0 * x)


def run_gradient_descent(
    x_start:       float = 8.0,
    learning_rate: float = 0.1,
    max_steps:     int   = 100,
    tolerance:     float = 1e-8,
    diverge_thresh: float = 1e6,
) -> GDResult:
    """
    경사 하강법 실행

    업데이트 규칙: x_{t+1} = x_t - α * f'(x_t) = x_t - α * 2x_t

    Parameters
    ----------
    x_start       : 시작 위치
    learning_rate : 학습률 α
    max_steps     : 최대 반복 횟수
    tolerance     : 수렴 판정 기준 (|Δx| < tol)
    diverge_thresh: 발산 판정 기준 (|x| > thresh)

    Returns
    -------
    GDResult
    """
    x         = float(x_start)
    loss_start = loss_fn(x)
    steps: List[GDStep] = []

    # 시작점 기록 (step 0)
    steps.append(GDStep(
        step=0, x=x,
        loss=loss_fn(x),
        gradient=gradient_fn(x),
        delta_x=0.0,
    ))

    converged = False
    diverged  = False

    for i in range(1, max_steps + 1):
        grad   = gradient_fn(x)
        x_new  = x - learning_rate * grad
        loss_v = loss_fn(x_new)
        delta  = x_new - x

        steps.append(GDStep(
            step=i, x=float(x_new),
            loss=float(loss_v),
            gradient=float(gradient_fn(x_new)),
            delta_x=float(delta),
        ))

        # 발산 체크
        if abs(x_new) > diverge_thresh or np.isnan(x_new) or np.isinf(x_new):
            diverged = True
            x = x_new
            break

        # 수렴 체크
        if abs(delta) < tolerance:
            converged = True
            x = x_new
            break

        x = x_new

    final_loss = loss_fn(x)
    conv_pct   = max(0.0, (loss_start - final_loss) / (loss_start + 1e-30) * 100)

    return GDResult(
        steps=steps,
        converged=converged,
        diverged=diverged,
        final_x=float(x),
        final_loss=float(final_loss),
        x_start=float(x_start),
        loss_start=float(loss_start),
        learning_rate=float(learning_rate),
        n_steps=len(steps) - 1,
        convergence_pct=float(conv_pct),
    )


def compare_learning_rates(
    x_start: float = 8.0,
    max_steps: int = 50,
) -> dict:
    """여러 학습률로 GD를 실행하여 비교 데이터 반환"""
    lrs = {
        "Too Slow (α=0.01)":   0.01,
        "Optimal (α=0.1)":     0.1,
        "Fast (α=0.45)":       0.45,
        "Diverging (α=1.05)":  1.05,
    }
    results = {}
    for label, lr in lrs.items():
        res = run_gradient_descent(x_start=x_start, learning_rate=lr,
                                   max_steps=max_steps, tolerance=1e-10)
        results[label] = res
    return results


PRESETS = {
    "slow":    {"x_start": 8.0, "learning_rate": 0.01, "label": "Too Slow"},
    "optimal": {"x_start": 8.0, "learning_rate": 0.10, "label": "Optimal"},
    "fast":    {"x_start": 8.0, "learning_rate": 0.45, "label": "Fast"},
    "diverge": {"x_start": 8.0, "learning_rate": 1.05, "label": "Diverging"},
}
