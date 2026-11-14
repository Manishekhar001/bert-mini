"""FastAPI service: send reports and receive one calibrated class prediction per report.

    uvicorn stage4.serve.app:app --port 8000
    MINIJEV_BACKEND=onnx_int8 MINIJEV_DIR=artifacts/final uvicorn stage4.serve.app:app --port 8000

    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"texts": ["On final approach we received a TCAS RA and executed a go-around ..."]}'
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from stage4.inference import Predictor

MODEL_DIR = os.environ.get("MINIJEV_DIR", "artifacts/final")
BACKEND = os.environ.get("MINIJEV_BACKEND", "onnx_int8")      # torch | onnx | onnx_int8
THREADS = int(os.environ.get("MINIJEV_THREADS", "4"))
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["predictor"] = Predictor(MODEL_DIR, BACKEND, num_threads=THREADS)
    yield
    _state.clear()


app = FastAPI(title="Mini Jev: ASRS report triage", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64, description="Report narratives")


class PredictResponse(BaseModel):
    results: list[dict]
    latency_ms: float
    backend: str


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND}


@app.get("/schema")
def schema():
    p = _state["predictor"]
    return {"classes": p.schema.classes}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    results = _state["predictor"].predict(req.texts)
    return PredictResponse(results=results, latency_ms=(time.perf_counter() - t0) * 1000.0, backend=BACKEND)
