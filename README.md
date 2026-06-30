# Churn Prediction MLOps Pipeline

End-to-end MLOps project for predicting customer churn for a telecom company. The project covers the full lifecycle of an ML system — from the API to CI/CD, experiment tracking, and monitoring — built incrementally following the MLOps Maturity Model.

> **Note:** This project is currently paused due to other commitments. The remaining steps (Airflow, Evidently) are planned for a future phase.

---

## Why this project

The goal wasn't to build the best possible model, but to understand hands-on how an ML model goes from a notebook to a production-ready, monitored, and reproducible system.

## Architecture

```
Training (Jupyter)
    │
    ▼
MLflow + DagsHub  (Experiment Tracking & Model Registry)
    │
    ▼
FastAPI + PostgreSQL  (API + Persistence)
    │
    ▼
Docker  (Containerization)
    │
    ▼
GitHub Actions (CI) → Docker Hub → Railway (Deploy)
    │
    ▼
Prometheus + Grafana  (Monitoring)
```

---

## MLOps Levels

| Level | Description | Status |
|-------|-------------|--------|
| 0 | Model + FastAPI + Docker + PostgreSQL + Git | ✅ |
| 1 | CI with GitHub Actions + Deploy to Railway | ✅ |
| 2 | Airflow ETL pipeline + Evidently drift detection | 🔄 |
| 3 | MLflow Experiment Tracking with DagsHub | ✅ |
| 4 | Prometheus + Grafana Monitoring | ✅ |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | XGBoost, scikit-learn, pandas |
| API | FastAPI, Pydantic, SQLAlchemy |
| Database | PostgreSQL (Railway) / SQLite (local & CI) |
| Containerization | Docker |
| CI | GitHub Actions, Docker Hub |
| Hosting | Railway |
| Experiment Tracking | MLflow, DagsHub |
| Monitoring | Prometheus, Grafana |
| Testing | pytest |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|--------------|
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch predictions for multiple customers |
| GET | `/health` | Health check |
| GET | `/predictions` | All stored predictions |
| GET | `/metrics` | Prometheus metrics |

Full interactive documentation available at `/docs` (Swagger UI).

---

## Project Structure

```
Churn/
├── main.py                 # FastAPI application
├── database.py              # SQLAlchemy models & DB connection
├── Dockerfile                # Production image definition
├── docker-compose.yaml       # Local dev: API + PostgreSQL + Prometheus + Grafana
├── requirements.txt
├── test_main.py              # pytest integration tests
├── training.ipynb            # MLflow training notebook
├── generate_data.py          # Dataset chunking for future Airflow pipeline
├── dags/
│   └── churn_pipeline.py     # Airflow DAG (in progress)
├── models/
│   ├── churn_model.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
└── .github/
    └── workflows/
        └── ci.yml             # GitHub Actions CI pipeline
```

---

## Key Technical Decisions

**Class imbalance:** Instead of resampling techniques (SMOTE/oversampling), `scale_pos_weight` was used in XGBoost — it changes how the model is "penalized" in the loss function, without touching the data itself. This avoids by design any risk of data leakage that could come from incorrectly ordering resampling and train/test split.

**Recall as the primary metric:** In a churn problem, the cost of missing a customer who would actually leave is higher than the cost of an unnecessary retention offer to someone who wouldn't — so the model was optimized for recall on the churn class, with a conscious trade-off in precision.

**Loading the model from a Registry, not a static file:** The API loads the model directly from the MLflow Model Registry (`models:/model_scale_pos_5/1`) at startup. This means switching the production model is done from the DagsHub UI, with no code change or new deployment required.

**Docker Compose for local use only:** Railway runs only the image built from the Dockerfile; Docker Compose is used exclusively for local development, to coordinate the API, PostgreSQL, Prometheus, and Grafana together over a private network.

**Integration tests on SQLite:** CI tests run against SQLite (not PostgreSQL) for simplicity and speed. Known limitation: this doesn't give full parity with the production environment — a planned next step is adding a PostgreSQL service to the GitHub Actions workflow.

---

## Setup

### Locally with Docker Compose (API + DB + Monitoring)
```bash
docker-compose up --build
```

### Locally without Docker
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Swagger UI
```
http://localhost:8000/docs
```

### Monitoring (when running via Docker Compose)
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

---

## CI/CD Pipeline

```
git push → GitHub Actions
    │
    ▼
pytest (integration tests)
    │
    ▼
docker build (from the Dockerfile)
    │
    ▼
push → Docker Hub (:latest)
    │
    ▼
Railway auto-detects new image → redeploys
```

> Railway is configured to watch the `:latest` tag on Docker Hub and automatically redeploy when a new image is pushed. In practice, this detection relies on periodic polling rather than a push-based webhook, so the delay between a successful CI run and the actual redeploy can range from a few minutes to a few hours. A more deterministic approach — triggering the deploy directly via the Railway API/CLI as the final CI step — was considered and is a planned improvement for instant, predictable deployments.

---

## Known Limitations & Next Steps

- [ ] Airflow DAG for automatic retraining as new data arrives (chunks)
- [ ] Evidently for data/prediction drift detection
- [ ] PostgreSQL service in CI for full environment parity with production
- [ ] Versioned Docker image tags instead of `latest` (rollback capability)
- [ ] Faster, deterministic CD: trigger Railway deploy directly via API/CLI from CI, instead of relying on Railway's polling-based auto-update
- [ ] Semantic validation in the Pydantic schema (e.g. non-negative tenure values)
- [ ] Fallback mechanism if model loading from DagsHub fails
- [ ] Kubernetes deployment (once real scaling is needed)

---

## Live Demo

🚀 [Railway Deployment](#)

---

*Personal project, built to gain hands-on understanding of the full MLOps lifecycle.*
