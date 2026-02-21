"""
Infant Growth Digital Twin — FastAPI Backend
=============================================

Multi-Method ML + WHO Growth Standards Hybrid Prediction System

REST API endpoints:
    POST   /infants                     Create infant profile
    GET    /infants                     List all infants
    GET    /infants/{id}                Get infant profile + observations
    POST   /infants/{id}/observations   Record a new measurement
    GET    /infants/{id}/predict        Get growth predictions
    GET    /infants/{id}/alerts         Get anomaly alerts
    GET    /infants/{id}/zscore         Get current z-scores & percentiles
    GET    /who/percentile-lines        Get WHO reference percentile lines
    GET    /model/info                  Get model info and comparison table
    GET    /health                      Health check
    GET    /                            Clinical dashboard
"""
import sys
import json
import math
from pathlib import Path
from datetime import datetime, date
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from typing import Optional, List
import secrets

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS_DIR, DATA_DIR, PORT, HOST, MODEL3_FEATURES
from config.settings import AUTH_ENABLED, AUTH_USERNAME, AUTH_PASSWORD
from src.models.who_engine import WHOZScoreEngine
from src.models.growth_model import GrowthModel
from src.models.predictor import GrowthPredictor
from src.models.data_structures import InfantProfile

# ── Auth ─────────────────────────────────────────────────────────
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """HTTP Basic Auth — only enforced when AUTH_ENABLED=true."""
    if not AUTH_ENABLED:
        return True
    correct_user = secrets.compare_digest(credentials.username, AUTH_USERNAME)
    correct_pass = secrets.compare_digest(credentials.password, AUTH_PASSWORD)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

# ── Global State ──────────────────────────────────────────────

_predictor: GrowthPredictor = None
_who_engine: WHOZScoreEngine = None
_infants: dict[str, InfantProfile] = {}
_metadata: dict = {}


def _load_system():
    """Load trained model and initialize predictor."""
    global _predictor, _who_engine, _metadata

    _who_engine = WHOZScoreEngine()

    model_path = MODELS_DIR / 'best_model.pkl'
    if not model_path.exists():
        raise RuntimeError(
            f"No trained model at {model_path}. "
            "Run `python -m src.training.train` first."
        )

    best_model = GrowthModel.load(model_path)

    # Load population z-score stats
    pop_stats_path = MODELS_DIR / 'pop_z_stats.json'
    pop_z_stats = {}
    if pop_stats_path.exists():
        with open(pop_stats_path) as f:
            pop_z_stats = json.load(f)

    _predictor = GrowthPredictor(
        who_engine=_who_engine,
        weight_model=best_model,
    )
    _predictor.set_pop_z_stats(pop_z_stats)

    # Load metadata
    meta_path = MODELS_DIR / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            _metadata = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    print("Loading ML+WHO hybrid prediction system...")
    _load_system()
    method = _metadata.get('best_method', 'unknown')
    r2 = _metadata.get('best_r2', 0)
    print(f"✓ System ready — Best model: {method} (R²={r2:.4f})")
    yield
    print("Shutting down")


# ── App ──────────────────────────────────────────────────────

_deps = [Depends(verify_credentials)] if AUTH_ENABLED else []

app = FastAPI(
    title="Infant Growth Digital Twin API",
    description=(
        "Multi-Method ML + WHO Growth Standards hybrid prediction engine. "
        "Provides personalized growth trajectories, conformal prediction CIs, "
        "anomaly detection, and WHO percentile mapping for infants 0–36 months."
    ),
    version="3.0.0",
    lifespan=lifespan,
    dependencies=_deps,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────

class CreateInfantRequest(BaseModel):
    infant_id: str = Field(..., description="Unique identifier")
    sex: str = Field(..., pattern="^(male|female)$")
    name: Optional[str] = None
    birth_weight_kg: Optional[float] = Field(None, gt=0, le=10)
    birth_date: Optional[str] = None

class ObservationRequest(BaseModel):
    age_months: float = Field(..., ge=0, le=36)
    metric: str = Field(..., pattern="^(weight_for_age|length_for_age|head_circumference_for_age)$")
    value: float = Field(..., gt=0)

class ObservationResponse(BaseModel):
    age_months: float
    metric: str
    value: float
    z_score: Optional[float] = None
    percentile: Optional[float] = None

class PredictionResponse(BaseModel):
    age_months: float
    metric: str
    predicted_value: float
    predicted_zscore: float
    predicted_percentile: float
    ci_lower: float
    ci_upper: float
    method: str

class AlertResponse(BaseModel):
    metric: str
    anomaly_type: str
    severity: str
    message: str
    z_score: Optional[float] = None
    percentile_drop: Optional[float] = None

class WHOPercentilePoint(BaseModel):
    age_months: float
    value: float

class WHOPercentileLine(BaseModel):
    percentile: int
    points: List[WHOPercentilePoint]


# ── Helper ────────────────────────────────────────────────────

def _safe(val):
    if val is None:
        return None
    try:
        if math.isnan(val):
            return None
    except (TypeError, ValueError):
        return None
    return round(val, 3)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": _predictor is not None,
        "best_method": _metadata.get("best_method"),
        "best_r2": _metadata.get("best_r2"),
        "n_training_samples": _metadata.get("n_training_samples"),
        "metrics_available": _who_engine.available_metrics if _who_engine else [],
        "infants_tracked": len(_infants),
        "version": "3.0.0",
    }


# ── Infant CRUD ───────────────────────────────────────────────

@app.post("/infants", status_code=201)
async def create_infant(req: CreateInfantRequest):
    if req.infant_id in _infants:
        raise HTTPException(409, f"Infant '{req.infant_id}' already exists")

    profile = InfantProfile(
        name=req.name or req.infant_id,
        sex=req.sex,
        birth_weight_kg=req.birth_weight_kg,
        birth_date=req.birth_date,
        infant_id=req.infant_id,
    )
    _infants[req.infant_id] = profile
    return profile.to_dict()


@app.get("/infants")
async def list_infants():
    return {
        "count": len(_infants),
        "infants": [
            {"infant_id": p.infant_id, "sex": p.sex,
             "observation_count": len(p.observations)}
            for p in _infants.values()
        ],
    }


@app.get("/infants/{infant_id}")
async def get_infant(infant_id: str):
    if infant_id not in _infants:
        raise HTTPException(404, f"Infant '{infant_id}' not found")
    return _infants[infant_id].to_dict()


# ── Observations ──────────────────────────────────────────────

@app.post("/infants/{infant_id}/observations",
          response_model=ObservationResponse)
async def add_observation(infant_id: str, req: ObservationRequest):
    if infant_id not in _infants:
        raise HTTPException(404, f"Infant '{infant_id}' not found")

    profile = _infants[infant_id]
    obs = profile.add_observation(
        req.age_months, req.metric, req.value, _who_engine
    )

    return ObservationResponse(
        age_months=obs.age_months, metric=obs.metric, value=obs.value,
        z_score=_safe(obs.z_score), percentile=_safe(obs.percentile),
    )


# ── Predictions ───────────────────────────────────────────────

@app.get("/infants/{infant_id}/predict",
         response_model=List[PredictionResponse])
async def predict_growth(
    infant_id: str,
    metric: str = Query(
        "weight_for_age",
        pattern="^(weight_for_age|length_for_age|head_circumference_for_age)$"
    ),
    months: str = Query("6,9,12,18,24,36",
                        description="Comma-separated future ages"),
):
    if infant_id not in _infants:
        raise HTTPException(404, f"Infant '{infant_id}' not found")

    future_ages = [float(x.strip()) for x in months.split(",")]
    profile = _infants[infant_id]
    predictions = _predictor.predict(profile, metric, future_ages)

    return [
        PredictionResponse(
            age_months=p.age_months, metric=p.metric,
            predicted_value=round(p.predicted_value, 3),
            predicted_zscore=round(p.predicted_zscore, 3),
            predicted_percentile=round(p.predicted_percentile, 1),
            ci_lower=round(p.ci_lower, 3),
            ci_upper=round(p.ci_upper, 3),
            method=p.method,
        )
        for p in predictions
    ]


# ── Anomaly Detection ────────────────────────────────────────

@app.get("/infants/{infant_id}/alerts",
         response_model=List[AlertResponse])
async def get_alerts(infant_id: str):
    if infant_id not in _infants:
        raise HTTPException(404, f"Infant '{infant_id}' not found")

    profile = _infants[infant_id]
    anomalies = _predictor.detect_anomalies(profile)

    return [
        AlertResponse(
            metric=a.metric, anomaly_type=a.anomaly_type,
            severity=a.severity, message=a.message,
            z_score=_safe(a.z_score),
            percentile_drop=_safe(a.percentile_drop),
        )
        for a in anomalies
    ]


# ── Z-Scores ─────────────────────────────────────────────────

@app.get("/infants/{infant_id}/zscore")
async def get_zscores(infant_id: str):
    if infant_id not in _infants:
        raise HTTPException(404, f"Infant '{infant_id}' not found")

    profile = _infants[infant_id]
    return [
        {
            "metric": o.metric, "age_months": o.age_months,
            "value": round(o.value, 3),
            "z_score": _safe(o.z_score),
            "percentile": _safe(o.percentile),
        }
        for o in profile.observations
    ]


# ── WHO Reference Lines ──────────────────────────────────────

@app.get("/who/percentile-lines")
async def get_who_percentiles(
    metric: str = Query(
        "weight_for_age",
        pattern="^(weight_for_age|length_for_age|head_circumference_for_age)$"
    ),
    sex: str = Query("male", pattern="^(male|female)$"),
    percentiles: str = Query("3,15,50,85,97"),
):
    pct_list = [int(x.strip()) for x in percentiles.split(",")]
    ages = list(range(0, 37))
    lines = []

    for pct in pct_list:
        points = []
        for age in ages:
            val = _who_engine.get_percentile_value(metric, sex, float(age), pct)
            if val and not math.isnan(val):
                points.append(WHOPercentilePoint(
                    age_months=float(age), value=round(val, 2)
                ))
        lines.append(WHOPercentileLine(percentile=pct, points=points))

    return {"metric": metric, "sex": sex, "lines": lines}


# ── Model Info ────────────────────────────────────────────────

@app.get("/model/info")
async def model_info():
    comparison_path = MODELS_DIR / 'comparison.csv'
    comparison = None
    if comparison_path.exists():
        import pandas as pd
        df = pd.read_csv(comparison_path, index_col=0)
        comparison = df.to_dict(orient='records')

    return {
        "metadata": _metadata,
        "comparison_table": comparison,
        "best_model_summary": _predictor.weight_model.summary()
        if _predictor else None,
    }


# ── Dashboard ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    dashboard_path = PROJECT_ROOT / "dashboard" / "index.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text())
    return HTMLResponse(content="""
    <html><body>
    <h1>Infant Growth Digital Twin API v3.0</h1>
    <p>ML+WHO Hybrid Prediction System</p>
    <p>Visit <a href="/docs">/docs</a> for the interactive API documentation.</p>
    </body></html>
    """)


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host=HOST, port=PORT, reload=True)
