# Infant Growth Digital Twin v3.0

**Multi-Method ML + WHO Growth Standards Hybrid Prediction System**

A production-grade REST API for personalized infant growth monitoring (0–36 months), combining 8 machine learning methods with WHO Child Growth Standards and conformal prediction confidence intervals.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI REST API                      │
│         11 endpoints · CORS · Auto-docs (/docs)         │
├─────────────┬─────────────┬─────────────────────────────┤
│  Strategy 1 │  Strategy 2 │     Clinical Alerts         │
│  ML Model   │  WHO Z-Score│  Extreme z-score detection  │
│  (primary)  │  Persistence│  Percentile crossing alerts │
│  + Conformal│  (fallback) │                             │
│  CIs (90%)  │             │                             │
├─────────────┴─────────────┴─────────────────────────────┤
│              WHO LMS Z-Score Engine                      │
│   weight-for-age · length-for-age · head-circ-for-age   │
├─────────────────────────────────────────────────────────┤
│           NHANES Data Pipeline (10 cycles)               │
│     Download → Parse XPT → Transform → MICE Impute      │
└─────────────────────────────────────────────────────────┘
```

## ML Methods (Auto-Selected via 5-Fold CV)

| Method | Description |
|--------|------------|
| Elastic Net | L1+L2 regularized linear regression |
| Ridge | L2 regularized regression |
| Extra Trees | Extremely randomized trees ensemble |
| Gradient Boosting | Sequential boosted decision trees |
| Random Forest | Bagged decision tree ensemble |
| SVR | Support vector regression (RBF kernel) |
| KNN | K-nearest neighbors regression |
| XGBoost | Extreme gradient boosting |

The best method is automatically selected by R² on 5-fold cross-validation.

## Key Features

- **Conformal Prediction**: Distribution-free 90% confidence intervals
- **MICE Imputation**: Recovers incomplete NHANES cases via iterative imputation
- **Hybrid Strategy**: ML predictions (primary) with WHO z-score persistence (fallback)
- **Anomaly Detection**: Extreme z-scores (±2.0 SD warning, ±3.0 SD critical) and percentile crossing alerts (>25-point drop)
- **WHO Percentile Mapping**: P3, P15, P50, P85, P97 reference curves
- **Regulatory Alignment**: EMA PROCOVA, FDA Digital Twin Guidance, ICH E9(R1)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + model status |
| `POST` | `/infants` | Create infant profile |
| `GET` | `/infants` | List all tracked infants |
| `GET` | `/infants/{id}` | Get infant profile + history |
| `POST` | `/infants/{id}/observations` | Record a measurement |
| `GET` | `/infants/{id}/predict` | Growth predictions with CIs |
| `GET` | `/infants/{id}/alerts` | Anomaly detection alerts |
| `GET` | `/infants/{id}/zscore` | All z-scores & percentiles |
| `GET` | `/who/percentile-lines` | WHO reference curves |
| `GET` | `/model/info` | Model metadata & comparison |
| `GET` | `/` | Clinical dashboard |

## Quick Start (Local)

```bash
# Clone
git clone https://github.com/neerajpandey1974/infant-growth-ml-who-hybrid.git
cd infant-growth-ml-who-hybrid

# Install
pip install -r requirements.txt

# Train (downloads NHANES data + trains 8 ML methods)
python -m src.training.train

# Run API
python -m src.api.server
# → http://localhost:10000
# → http://localhost:10000/docs (interactive API docs)
```

## Deploy to Render

### Option A: Blueprint (Recommended)

1. Push to GitHub: `https://github.com/neerajpandey1974/infant-growth-ml-who-hybrid`
2. Go to [Render Dashboard](https://dashboard.render.com)
3. **New** → **Blueprint** → select the repo
4. Render reads `render.yaml` and deploys automatically

### Option B: Manual

1. **New** → **Web Service** → connect your GitHub repo
2. Set **Runtime** to **Docker**
3. Set **Health Check Path** to `/health`
4. Deploy

The Dockerfile handles everything: installs dependencies, downloads NHANES data, trains all 8 ML methods, and starts the API server.

## Example Usage

```bash
# Create an infant
curl -X POST https://your-app.onrender.com/infants \
  -H "Content-Type: application/json" \
  -d '{"infant_id": "baby-001", "sex": "female", "birth_weight_kg": 3.2}'

# Record a weight measurement at 3 months
curl -X POST https://your-app.onrender.com/infants/baby-001/observations \
  -H "Content-Type: application/json" \
  -d '{"age_months": 3.0, "metric": "weight_for_age", "value": 5.8}'

# Get growth predictions with 90% confidence intervals
curl "https://your-app.onrender.com/infants/baby-001/predict?metric=weight_for_age&months=6,9,12,18,24"

# Check for anomalies
curl "https://your-app.onrender.com/infants/baby-001/alerts"

# Get WHO percentile curves
curl "https://your-app.onrender.com/who/percentile-lines?metric=weight_for_age&sex=female"
```

## Project Structure

```
infant-growth-ml-who-hybrid/
├── config/
│   └── settings.py              # NHANES cycles, features, paths
├── src/
│   ├── api/
│   │   └── server.py            # FastAPI REST API (11 endpoints)
│   ├── models/
│   │   ├── who_engine.py        # WHO LMS z-score engine
│   │   ├── growth_model.py      # 8-method ML comparison framework
│   │   ├── predictor.py         # Hybrid ML+WHO predictor
│   │   └── data_structures.py   # InfantProfile, Observation, Prediction
│   ├── ingestion/
│   │   └── pipeline.py          # NHANES download + transform
│   └── training/
│       └── train.py             # Full training pipeline
├── dashboard/
│   └── index.html               # Clinical dashboard
├── tests/
│   └── test_api.py              # API tests
├── Dockerfile                   # Render deployment
├── render.yaml                  # Render blueprint
├── requirements.txt             # Python dependencies
└── README.md
```

## Data

- **Source**: NHANES (National Health and Nutrition Examination Survey)
- **Cycles**: 10 survey cycles (1999–2018)
- **Age Range**: 0–36 months
- **Features**: Demographics (sex, race/ethnicity, income, family size) + anthropometrics (weight, length, head circumference, arm circumference, arm length)
- **WHO Standards**: Lambda-Mu-Sigma (LMS) method for z-score computation

## License

MIT
