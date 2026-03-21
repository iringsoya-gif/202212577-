"""
FastAPI server for Hooke's Law TensorFlow Neural Network
Run: python main.py  →  http://localhost:8000
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Ensure output & static directories exist before mounting
os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)

from model import HookesLawModel  # noqa: E402 (import after os.makedirs)

app = FastAPI(
    title="Hooke's Law Neural Network",
    description="TensorFlow linear regression to learn F = kx",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")

_model = HookesLawModel()


# ── Request / response schemas ─────────────────────────────────────────────

class TrainParams(BaseModel):
    epochs: int = 500
    learning_rate: float = 0.01
    n_samples: int = 200


class PredictRequest(BaseModel):
    mass: float


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as fh:
        return fh.read()


@app.post("/train")
async def train(params: TrainParams):
    """Train the TensorFlow Hooke's Law model and return metrics + plot paths."""
    return _model.train(
        epochs=params.epochs,
        learning_rate=params.learning_rate,
        n_samples=params.n_samples,
    )


@app.post("/predict")
async def predict(req: PredictRequest):
    """Predict spring extension for a given mass (kg)."""
    return _model.predict(req.mass)


@app.get("/health")
async def health():
    return {"status": "ok", "model_trained": _model.trained}


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
