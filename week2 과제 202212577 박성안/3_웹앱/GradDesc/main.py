"""
Gradient Descent Visualizer — FastAPI
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import time

from gradient_descent import run_gradient_descent, compare_learning_rates, PRESETS, GDStep
from visualizer import plot_gd_path, plot_loss_curve, plot_comparison

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Gradient Descent Visualizer", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR),          name="output")


# ── Schemas ──────────────────────────────────────────────────────────────────
class SimulateRequest(BaseModel):
    x_start:       float = Field(default=8.0,  ge=-20.0, le=20.0)
    learning_rate: float = Field(default=0.1,  gt=0.0,   le=2.0)
    max_steps:     int   = Field(default=100,  ge=5,     le=500)
    tolerance:     float = Field(default=1e-8)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/simulate")
async def simulate(req: SimulateRequest):
    """경사 하강법 시뮬레이션 + 3개 PNG 생성"""
    result = run_gradient_descent(
        x_start=req.x_start,
        learning_rate=req.learning_rate,
        max_steps=req.max_steps,
        tolerance=req.tolerance,
    )

    t = int(time.time())
    p1 = plot_gd_path(result)
    p2 = plot_loss_curve(result)
    p3 = plot_comparison(x_start=req.x_start)

    # Step data (max 200 rows for UI table)
    steps_data = [
        {
            "step":     s.step,
            "x":        round(s.x, 8),
            "loss":     round(s.loss, 8),
            "gradient": round(s.gradient, 8),
            "delta_x":  round(s.delta_x, 8),
        }
        for s in result.steps[:200]
    ]

    return {
        "status":          "success",
        "gd_path_plot":    f"/output/gd_path.png?t={t}",
        "loss_curve_plot": f"/output/loss_curve.png?t={t}",
        "comparison_plot": f"/output/comparison.png?t={t}",
        "steps":           steps_data,
        "n_steps":         result.n_steps,
        "converged":       result.converged,
        "diverged":        result.diverged,
        "final_x":         round(result.final_x, 10),
        "final_loss":      round(result.final_loss, 10),
        "x_start":         req.x_start,
        "loss_start":      round(result.loss_start, 4),
        "learning_rate":   req.learning_rate,
        "convergence_pct": round(result.convergence_pct, 4),
        "formula": {
            "loss":     "f(x) = x²",
            "gradient": "f'(x) = 2x",
            "update":   "x := x - α · f'(x)",
        },
    }


@app.get("/api/presets")
async def get_presets():
    return PRESETS


@app.get("/api/health")
async def health():
    return {"status": "ok"}
