"""
Optional HTTP service so your existing Node dashboard (or anything else)
can read the prediction over REST.

    uvicorn groww_predictor.serve:app --port 8000

Endpoints:
    GET /health
    GET /predict            -> last cached prediction (data/predictions.json)
    POST /predict/run       -> run a live prediction now and return it
    GET /backtest           -> last backtest summary (runs if dataset present)
"""
from __future__ import annotations
import json
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, time as dtime

from .config import CFG

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
except Exception as e:  # noqa
    raise SystemExit("Install fastapi+uvicorn to use the server: pip install fastapi uvicorn")


@asynccontextmanager
async def lifespan(app):
    from .scheduler import start_scheduler
    sched = start_scheduler()
    try:
        yield
    finally:
        if sched:
            sched.shutdown(wait=False)


app = FastAPI(title="Groww Most-Traded Predictor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_DASHBOARD = Path(__file__).parent / "dashboard.html"


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if _DASHBOARD.exists():
        return _DASHBOARD.read_text(encoding="utf-8")
    return "<h1>dashboard.html missing</h1>"


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    active = CFG.model_mode
    if CFG.model_mode == "auto":
        try:
            from .model import load_best_model
            active = f"auto→{(load_best_model() or 'analytic')}"
        except Exception:  # noqa
            active = "auto→analytic"
    return {"ok": True, "metric": CFG.rank_metric, "model": CFG.model_mode,
            "active_model": active,
            "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/predict")
def get_prediction():
    if CFG.predictions_path.exists():
        return json.loads(CFG.predictions_path.read_text())
    return {"error": "no prediction yet — run POST /predict/run after 09:25 IST"}


@app.post("/predict/run")
def run_prediction():
    from .predict_live import predict_live
    return predict_live()


@app.get("/backtest")
def get_backtest():
    if not CFG.dataset_path.exists():
        return {"error": "no dataset — run python -m groww_predictor.build_dataset first"}
    from .backtest import backtest
    return backtest().to_dict(orient="index")