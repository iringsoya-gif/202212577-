"""
K-Means Clustering API
FastAPI + numpy 기반 K-Means 클러스터링 웹 서비스
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
import numpy as np

from kmeans import KMeans, generate_customer_data, find_optimal_k
from visualizer import (
    plot_clusters,
    plot_wcss_iterations,
    plot_elbow,
    plot_prediction,
)

# ── 앱 초기화 ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="K-Means Clustering Visualizer",
    description="순수 NumPy K-Means 클러스터링 시각화 API",
    version="1.0.0",
)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# ── 전역 상태 (학습 완료 모델 캐시) ───────────────────────────────────────────
_cache: dict = {
    "model": None,
    "X": None,
    "k": None,
}


# ── 스키마 ────────────────────────────────────────────────────────────────────
class ClusterRequest(BaseModel):
    k: int = Field(default=3, ge=1, le=10, description="클러스터 수")
    max_iter: int = Field(default=100, ge=10, le=500, description="최대 반복 횟수")
    n_samples: int = Field(default=200, ge=50, le=1000, description="샘플 수")


class PredictRequest(BaseModel):
    purchase_amount: float = Field(..., gt=0, description="구매금액 (만원)")
    visit_count: float = Field(..., gt=0, description="방문횟수 (회/월)")


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """메인 UI 서빙"""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/cluster")
async def run_clustering(req: ClusterRequest):
    """K-Means 클러스터링 실행 및 시각화 PNG 저장

    1. 고객 데이터 생성
    2. K-Means 학습 (순수 numpy)
    3. 클러스터 산점도 PNG 저장
    4. WCSS 수렴 그래프 PNG 저장
    5. K 선택 Elbow 그래프 PNG 저장
    """
    # 1. 데이터 생성
    X = generate_customer_data(n_samples=req.n_samples)

    # 2. K-Means 학습
    model = KMeans(k=req.k, max_iter=req.max_iter, random_state=42)
    model.fit(X)

    # 전역 캐시에 저장 (예측 API 에서 재사용)
    _cache["model"] = model
    _cache["X"] = X
    _cache["k"] = req.k

    # 3. 시각화 생성 및 PNG 저장
    cluster_path = plot_clusters(X, model.labels_, model.centroids_, req.k)
    wcss_path = plot_wcss_iterations(model.wcss_history_)
    elbow_path = plot_elbow(
        find_optimal_k(X),
        optimal_k=req.k,
    )

    # 클러스터별 사이즈
    cluster_sizes = [int((model.labels_ == j).sum()) for j in range(req.k)]

    # 클러스터 라벨 이름 (k=3 기본)
    cluster_names = _get_cluster_names(model.centroids_, req.k)

    return {
        "status": "success",
        "cluster_plot": f"/output/cluster_plot.png?t={_timestamp()}",
        "wcss_plot": f"/output/wcss_plot.png?t={_timestamp()}",
        "k_selection_plot": f"/output/k_selection_plot.png?t={_timestamp()}",
        "k": req.k,
        "n_samples": req.n_samples,
        "n_iter": model.n_iter_,
        "wcss_final": round(float(model.inertia_), 2),
        "wcss_history": [round(float(w), 2) for w in model.wcss_history_],
        "cluster_sizes": cluster_sizes,
        "cluster_names": cluster_names,
        "centroids": model.centroids_.tolist(),
    }


@app.post("/api/predict")
async def predict_cluster(req: PredictRequest):
    """새 고객 데이터의 군집 예측

    Request: 구매금액 + 방문횟수
    Response: 예측 군집 + 거리 + 설명 + 시각화 PNG
    """
    if _cache["model"] is None:
        raise HTTPException(
            status_code=400,
            detail="먼저 /api/cluster를 호출하여 모델을 학습시키세요.",
        )

    model: KMeans = _cache["model"]
    X = _cache["X"]
    k = _cache["k"]

    # 예측
    new_point = np.array([req.purchase_amount, req.visit_count])
    predicted_cluster, distance = model.predict_with_distance(new_point)

    # 예측 시각화 PNG 저장
    pred_path = plot_prediction(X, model.labels_, model.centroids_, k, new_point, predicted_cluster)

    cluster_names = _get_cluster_names(model.centroids_, k)
    cluster_desc = _get_cluster_description(predicted_cluster, new_point, cluster_names)

    return {
        "status": "success",
        "predicted_cluster": int(predicted_cluster),
        "cluster_name": cluster_names.get(predicted_cluster, f"Cluster {predicted_cluster}"),
        "cluster_description": cluster_desc,
        "distance_to_centroid": round(distance, 3),
        "input": {"purchase_amount": req.purchase_amount, "visit_count": req.visit_count},
        "centroid": model.centroids_[predicted_cluster].tolist(),
        "prediction_plot": f"/output/prediction_plot.png?t={_timestamp()}",
    }


@app.get("/api/wcss")
async def get_wcss_data():
    """학습된 모델의 WCSS 히스토리 반환"""
    if _cache["model"] is None:
        raise HTTPException(status_code=400, detail="모델이 학습되지 않았습니다.")
    model: KMeans = _cache["model"]
    return {
        "wcss_history": [round(float(w), 2) for w in model.wcss_history_],
        "n_iter": model.n_iter_,
        "wcss_final": round(float(model.inertia_), 2),
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model_trained": _cache["model"] is not None}


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def _get_cluster_names(centroids: np.ndarray, k: int) -> dict:
    """Auto-assign cluster names based on purchase amount (centroid index 0)"""
    if k == 3:
        order = np.argsort(centroids[:, 0])
        names_list = ["Regular", "Premium", "VIP"]
        return {int(order[i]): names_list[i] for i in range(3)}
    return {j: f"Cluster {j}" for j in range(k)}


def _get_cluster_description(cluster: int, point: np.ndarray, names: dict) -> str:
    name = names.get(cluster, f"Cluster {cluster}")
    amount, visits = point[0], point[1]
    desc_map = {
        "Regular": "Low-frequency, low-spend segment. Potential for upselling.",
        "Premium": "Mid-tier segment with growth potential. Target with loyalty programs.",
        "VIP": "High-frequency, high-spend loyalist. Retain with exclusive offers.",
    }
    base = desc_map.get(name, "Customer segment identified by K-Means.")
    return (
        f"Predicted: {name} Customer  |  "
        f"Purchase {amount:.0f}K KRW / {visits:.0f} visits per month.  "
        f"{base}"
    )


def _timestamp() -> int:
    import time
    return int(time.time())
