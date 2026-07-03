# Churn Prediction MLOps Pipeline

End-to-end MLOps project for predicting customer churn for a telecom company. The project covers the full lifecycle of an ML system — from the API to CI/CD, experiment tracking, automated retraining, monitoring, and explainability — built step-by-step following a custom MLOps progression roadmap.

---

## Why this project

The goal wasn't to build the best possible model, but to understand hands-on how an ML model goes from a notebook to a monitored, reproducible system with an automated retraining pipeline.

---

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
    │
    ▼
Airflow DAG  (ETL + Automated Retraining)
    │
    ▼
Evidently (Drift Detection) + SHAP (Explainability)
```

---

## MLOps Levels

| Level | Description | Status |
|-------|-------------|--------|
| 0 | Model + FastAPI + Docker + PostgreSQL + Git | ✅ |
| 1 | CI with GitHub Actions + Deploy to Railway | ✅ |
| 2 | Airflow ETL pipeline + Evidently drift detection | ✅ |
| 3 | MLflow Experiment Tracking with DagsHub | ✅ |
| 4 | Prometheus + Grafana Monitoring | ✅ |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | XGBoost, scikit-learn, pandas |
| Explainability | SHAP |
| API | FastAPI, Pydantic, SQLAlchemy |
| Database | PostgreSQL (Railway) / SQLite (local & CI) |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions, Docker Hub, Railway |
| Experiment Tracking | MLflow, DagsHub |
| Monitoring | Prometheus, Grafana |
| Pipeline Orchestration | Apache Airflow |
| Drift Detection | Evidently |
| Testing | pytest |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single customer prediction + SHAP top 2 features |
| POST | `/predict/batch` | Batch predictions + SHAP top 2 features per customer |
| GET | `/health` | Health check |
| GET | `/predictions` | All stored predictions |
| GET | `/metrics` | Prometheus metrics |

### Example Response `/predict`

```json
{
  "Churn": 1,
  "probability": 0.878,
  "input_data": { ... },
  "top_features": {
    "Contract_Two year": 0.447,
    "InternetService_Fiber optic": 0.303
  }
}
```

Full interactive documentation available at `/docs` (Swagger UI).

---

## Airflow Pipeline (Level 2)

The `churn_pipeline` DAG runs every 3 minutes and processes 10 chunks of new data sequentially:

```
read_next_chunk → load_to_postgres → check_drift → retrain_and_predict_model → update_chunk
```

| Task | Description |
|------|-------------|
| `read_next_chunk` | Reads the next CSV chunk from `data/chunks/` using an Airflow Variable as counter |
| `load_to_postgres` | Stores new customer data in PostgreSQL |
| `check_drift` | Runs Evidently DataDrift report (train vs all new data) and saves HTML |
| `retrain_and_predict_model` | Retrains XGBoost on train + new data, evaluates on fixed test set, logs to DagsHub |
| `update_chunk` | Increments chunk counter; stops automatically when all chunks are processed |

### MLflow Metrics per Run
- `recall_churn`, `f1_churn`, `precision_churn`
- `train_logloss` and `test_logloss` per iteration (for overfitting detection)

---

## Project Structure

```
Churn/
├── main.py                 # FastAPI application + SHAP explainability
├── database.py             # SQLAlchemy models & DB connection
├── Dockerfile              # Production image definition
├── docker-compose.yaml     # Local dev: API + PostgreSQL + Prometheus + Grafana + Airflow
├── requirements.txt
├── test_main.py            # pytest integration tests
├── training.ipynb          # MLflow training notebook
├── generate_data.py        # Data splits + chunk generation for Airflow
├── dags/
│   ├── churn_pipeline.py   # Airflow DAG
│   └── database_airflow.py # SQLAlchemy models for Airflow (compatible with Python 3.8)
├── models/
│   ├── churn_model.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   └── chunks/             # 10 CSV chunks for Airflow
├── reports/                # Evidently HTML drift reports
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions CI pipeline
```

---

## Key Technical Decisions

**Class imbalance:** Instead of resampling techniques (SMOTE/oversampling), `scale_pos_weight` was used in XGBoost — it changes how the model is penalized in the loss function, without touching the data itself. This avoids any risk of data leakage that could come from incorrectly ordering resampling and train/test split.

**Recall as the primary metric:** In a churn problem, the cost of missing a customer who would actually leave is higher than the cost of an unnecessary retention offer — so the model was optimized for recall on the churn class, with a conscious trade-off in precision.

**Loading the model from a Registry, not a static file:** The API loads the model directly from the MLflow Model Registry (`models:/model_scale_pos_5/1`) at startup. Switching the production model is done from the DagsHub UI, with no code change or new deployment required.

**SHAP for local explainability:** Each prediction returns the top 2 features that most influenced the result, using SHAP TreeExplainer. This gives actionable insight — e.g. "this customer is predicted to churn mainly due to Month-to-month contract and high monthly charges".

**Fixed test set for retraining evaluation:** During Airflow retraining, the model is always evaluated on the same 15% test set. This allows fair comparison of metrics across runs as new data chunks are added.

**Overfitting detection via logloss curves:** Each Airflow run logs `train_logloss` and `test_logloss` per iteration to MLflow. If train loss drops while test loss rises, overfitting is occurring — visible directly in DagsHub.

**Docker Compose for local use only:** Railway runs only the image built from the Dockerfile; Docker Compose is used exclusively for local development, to coordinate the API, PostgreSQL, Prometheus, Grafana, and Airflow together.

**Integration tests on SQLite:** CI tests run against SQLite for simplicity and speed. Known limitation: this doesn't give full parity with the production environment.

---

## Setup

### Locally with Docker Compose (API + DB + Monitoring + Airflow)
```bash
docker-compose up --build
```

### Airflow User (first time only)
```bash
docker-compose exec airflow-webserver airflow users create --username airflow --password airflow --firstname Admin --lastname Admin --role Admin --email admin@example.com
```

### Locally without Docker
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### URLs
- Swagger UI: `http://localhost:8000/docs`
- Airflow UI: `http://127.0.0.1:8080`
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
docker build
    │
    ▼
push → Docker Hub (:latest)
    │
    ▼
Railway auto-detects new image → redeploys
```

> Railway is configured to watch the `:latest` tag on Docker Hub and automatically redeploy when a new image is pushed. The delay between a successful CI run and the actual redeploy can range from a few minutes to a few hours due to polling-based detection.

---

## Known Limitations & Next Steps

- [ ] Auto-promote retrained model to Railway after each Airflow run
- [ ] Ground truth delay simulation: chunks already have Churn labels — in real production, labels arrive weeks/months later
- [ ] Airflow writes directly to the database instead of going through the `/predict/batch` endpoint
- [ ] PostgreSQL service in CI for full environment parity with production
- [ ] Versioned Docker image tags instead of `latest` (for rollback capability)
- [ ] Alerting on pipeline task failure
- [ ] Frontend (Streamlit or React)

---

## Live Demo

🚀 Railway Deployment (currently paused — free tier limitations)
📊 [DagsHub Experiments](https://dagshub.com/Spengian/churn---prediction.mlflow)

---

*Personal project, built to gain hands-on understanding of the full MLOps lifecycle.*
