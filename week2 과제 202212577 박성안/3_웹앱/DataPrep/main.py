"""
Data Preprocessing API — FastAPI
=================================
Min-Max Scaling & Z-Score Normalization 시각화 서비스
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import numpy as np
import time

from preprocessing import MinMaxScaler, StandardScaler, generate_employee_data, compute_stats
from visualizer import (
    plot_raw_distribution,
    plot_normalized_distribution,
    plot_comparison,
    plot_variance_comparison,
    plot_prediction,
)

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Data Preprocessing Visualizer", version="1.0.0")
app.mount("/static",  StaticFiles(directory=BASE_DIR / "static"),  name="static")
app.mount("/output",  StaticFiles(directory=OUTPUT_DIR),           name="output")

# ── 전역 캐시 ────────────────────────────────────────────────────────────────
_cache: dict = {
    "X_raw": None,
    "X_mm":  None,
    "X_zs":  None,
    "mm_scaler":  None,
    "zs_scaler":  None,
    "stats_raw":  None,
    "stats_mm":   None,
}


# ── Schemas ──────────────────────────────────────────────────────────────────
class NormalizeRequest(BaseModel):
    n_samples: int = Field(default=300, ge=50, le=1000)
    method: str = Field(default="both", description="'minmax' | 'zscore' | 'both'")


class PredictRequest(BaseModel):
    salary: float = Field(..., gt=0, description="연봉 (원)")
    age:    float = Field(..., gt=0, le=100, description="나이 (세)")


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/normalize")
async def normalize(req: NormalizeRequest):
    """
    데이터 생성 → Min-Max + Z-Score 정규화 → 5개 PNG 저장
    """
    # 1. 데이터 생성
    X_raw = generate_employee_data(n_samples=req.n_samples)

    # 2. Min-Max Scaling
    mm = MinMaxScaler()
    X_mm = mm.fit_transform(X_raw)

    # 3. Z-Score Standardization
    zs = StandardScaler()
    X_zs = zs.fit_transform(X_raw)

    # 캐시 저장
    _cache.update({
        "X_raw": X_raw, "X_mm": X_mm, "X_zs": X_zs,
        "mm_scaler": mm, "zs_scaler": zs,
        "stats_raw": compute_stats(X_raw, ["salary", "age"]),
        "stats_mm":  compute_stats(X_mm,  ["salary", "age"]),
    })

    # 4. 시각화 PNG 생성
    t = int(time.time())
    p1 = plot_raw_distribution(X_raw)
    p2 = plot_normalized_distribution(X_mm, X_zs)
    p3 = plot_comparison(X_raw, X_mm)
    p4 = plot_variance_comparison(X_raw, X_mm, X_zs)

    stats_raw = _cache["stats_raw"]
    stats_mm  = _cache["stats_mm"]

    return {
        "status": "success",
        "n_samples": req.n_samples,
        "method": req.method,
        "raw_distribution":       f"/output/raw_distribution.png?t={t}",
        "normalized_distribution": f"/output/normalized_distribution.png?t={t}",
        "comparison_plot":        f"/output/comparison_plot.png?t={t}",
        "variance_plot":          f"/output/variance_comparison.png?t={t}",
        "stats": {
            "salary": {
                "raw_min":   round(stats_raw["salary"]["min"]),
                "raw_max":   round(stats_raw["salary"]["max"]),
                "raw_mean":  round(stats_raw["salary"]["mean"]),
                "raw_std":   round(stats_raw["salary"]["std"]),
                "raw_var":   stats_raw["salary"]["var"],
                "mm_min":    round(stats_mm["salary"]["min"], 6),
                "mm_max":    round(stats_mm["salary"]["max"], 6),
                "mm_mean":   round(stats_mm["salary"]["mean"], 4),
                "mm_std":    round(stats_mm["salary"]["std"], 4),
            },
            "age": {
                "raw_min":  round(stats_raw["age"]["min"], 1),
                "raw_max":  round(stats_raw["age"]["max"], 1),
                "raw_mean": round(stats_raw["age"]["mean"], 2),
                "raw_std":  round(stats_raw["age"]["std"], 2),
                "raw_var":  stats_raw["age"]["var"],
                "mm_min":   round(stats_mm["age"]["min"], 6),
                "mm_max":   round(stats_mm["age"]["max"], 6),
                "mm_mean":  round(stats_mm["age"]["mean"], 4),
                "mm_std":   round(stats_mm["age"]["std"], 4),
            },
        },
        "formula": {
            "minmax": "x_scaled = (x - x_min) / (x_max - x_min)",
            "zscore": "x_std = (x - mean) / std",
        },
        "scaler_params": {
            "salary": {
                "min": round(float(mm.min_[0])),
                "max": round(float(mm.max_[0])),
                "range": round(float(mm.data_range_[0])),
            },
            "age": {
                "min": round(float(mm.min_[1]), 2),
                "max": round(float(mm.max_[1]), 2),
                "range": round(float(mm.data_range_[1]), 2),
            },
        },
    }


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """사용자 입력(연봉, 나이) → 정규화 값 예측 + 시각화"""
    if _cache["mm_scaler"] is None:
        raise HTTPException(status_code=400, detail="먼저 /api/normalize를 호출하세요.")

    mm: MinMaxScaler   = _cache["mm_scaler"]
    zs: StandardScaler = _cache["zs_scaler"]
    X_raw = _cache["X_raw"]
    X_mm  = _cache["X_mm"]

    # 입력 정규화
    new_raw = np.array([[req.salary, req.age]])
    new_mm  = mm.transform(new_raw)
    new_zs  = zs.transform(new_raw)

    sal_mm  = float(new_mm[0, 0])
    age_mm  = float(new_mm[0, 1])
    sal_zs  = float(new_zs[0, 0])
    age_zs  = float(new_zs[0, 1])

    # 퍼센타일
    sal_pct = float((X_raw[:, 0] < req.salary).mean() * 100)
    age_pct = float((X_raw[:, 1] < req.age).mean() * 100)

    # 예측 시각화
    t = int(time.time())
    pred_path = plot_prediction(X_raw, X_mm, new_raw[0], new_mm)

    # 해석 메시지
    interpretation = _interpret(sal_mm, age_mm, sal_pct, age_pct)

    return {
        "status": "success",
        "input": {"salary": req.salary, "age": req.age},
        "minmax": {
            "salary": round(sal_mm, 6),
            "age":    round(age_mm, 6),
        },
        "zscore": {
            "salary": round(sal_zs, 4),
            "age":    round(age_zs, 4),
        },
        "percentile": {
            "salary": round(sal_pct, 1),
            "age":    round(age_pct, 1),
        },
        "interpretation": interpretation,
        "prediction_plot": f"/output/prediction_plot.png?t={t}",
        "formula_applied": {
            "salary": f"({req.salary:,.0f} - {mm.min_[0]:,.0f}) / {mm.data_range_[0]:,.0f} = {sal_mm:.6f}",
            "age":    f"({req.age:.1f} - {mm.min_[1]:.1f}) / {mm.data_range_[1]:.1f} = {age_mm:.6f}",
        },
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_ready": _cache["mm_scaler"] is not None}


# ── Helper ────────────────────────────────────────────────────────────────────
def _interpret(sal_mm: float, age_mm: float, sal_pct: float, age_pct: float) -> str:
    sal_tier = "High earner" if sal_pct > 75 else "Above average" if sal_pct > 50 else "Below average" if sal_pct > 25 else "Entry level"
    age_tier = "Senior" if age_pct > 75 else "Mid-career" if age_pct > 40 else "Junior"
    return (
        f"{sal_tier} (top {100-sal_pct:.0f}% salary) · {age_tier} profile. "
        f"Normalized salary {sal_mm:.3f} and age {age_mm:.3f} are now directly comparable for ML models."
    )
