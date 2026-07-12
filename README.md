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
| PATCH | `/predictions/batch/churn` | Batch update ground truth labels |
| GET | `/health` | Health check |
| GET | `/predictions` | All stored predictions |
| GET | `/metrics` | Prometheus metrics |

### Example Response `/predict`

```json
{
  "id": 1,
  "churn_pred": 1,
  "churn_real": null,
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

The `churn_pipeline` DAG runs every 3 minutes and processes 10 chunks of new data sequentially (`max_active_runs=1` to prevent overlapping runs from racing on the chunk counter):

```
read_next_chunk → simulate_ground_truth → check_drift → retrain_and_predict_model → champion_vs_challenger → update_chunk
```

| Task | Description |
|------|-------------|
| `read_next_chunk` | Reads the next raw CSV chunk path from `data/chunks_raw/` using an Airflow Variable as counter |
| `simulate_ground_truth` | Sends chunk to `/predict/batch` API → stores predictions with `churn_real=null` → updates ground truth via `PATCH /predictions/batch/churn` |
| `check_drift` | Runs Evidently DataDrift report (raw train vs all new data with confirmed labels) and saves HTML |
| `retrain_and_predict_model` | Incrementally retrains XGBoost on train + confirmed data (`churn_real IS NOT NULL`), with a periodic full reset above a tree-count cap. Evaluates on a fixed test set, logs to DagsHub. Tracks the latest model URI in an Airflow Variable so retraining builds on the previous run instead of always restarting from the original model. |
| `champion_vs_challenger` | Statistically compares the new model against production using McNemar's test; promotes via MLflow registry alias if significantly better |
| `update_chunk` | Increments chunk counter; stops automatically when all chunks are processed |

### Ground Truth Simulation
In real production, ground truth labels (whether a customer actually churned) arrive weeks or months after the initial prediction. This pipeline simulates that delay: the API first stores predictions with `churn_real=null`, then the Airflow task updates the real label via `PATCH` — mimicking the delayed label arrival pattern.

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
├── Dockerfile.airflow      # Custom Airflow image with pre-installed packages
├── docker-compose.yaml     # Local dev: API + PostgreSQL + Prometheus + Grafana + Airflow
├── requirements.txt
├── test_main.py            # pytest integration tests
├── training.ipynb          # MLflow training notebook
├── generate_data.py        # Data splits + chunk generation for Airflow
├── dags/
│   ├── churn_pipeline.py   # Airflow DAG
│   ├── preprocessing.py    # Shared encoder/scaler preprocessing (used by retraining + evaluation)
│   └── database_airflow.py # SQLAlchemy models for Airflow (compatible with Python 3.8)
├── models/
│   ├── churn_model.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── train_data_raw.csv
│   ├── test_data_raw.csv
│   └── chunks_raw/         # 10 raw CSV chunks for Airflow simulate_ground_truth task
├── reports/                # Evidently HTML drift reports
└── .github/
    └── workflows/
        └── ci.yml          # GitHub Actions CI pipeline
```

---

## Key Technical Decisions

**Class imbalance:** Instead of resampling techniques (SMOTE/oversampling), `scale_pos_weight` was used in XGBoost — it changes how the model is penalized in the loss function, without touching the data itself. This avoids any risk of data leakage that could come from incorrectly ordering resampling and train/test split.

**Recall as the primary metric:** In a churn problem, the cost of missing a customer who would actually leave is higher than the cost of an unnecessary retention offer — so the model was optimized for recall on the churn class, with a conscious trade-off in precision.

**Automated model promotion, no manual switching:** The API loads the model via a `@champion` MLflow alias, refreshed live every few minutes by a background scheduler. Promotion is fully automated by the `champion_vs_challenger` Airflow task — no manual DagsHub UI interaction or redeploy is needed to switch production models.

**SHAP for local explainability:** Each prediction returns the top 2 features that most influenced the result, using SHAP TreeExplainer. This gives actionable insight — e.g. "this customer is predicted to churn mainly due to Month-to-month contract and high monthly charges".

**Ground truth via PATCH endpoints:** Predictions are stored with `churn_real=null` at inference time. Ground truth labels are added later via `PATCH /predictions/{id}/churn`, simulating the real-world delay between prediction and outcome. Retraining only uses records with confirmed labels (`churn_real IS NOT NULL`).

**Fixed test set for retraining evaluation:** During Airflow retraining, the model is always evaluated on the same 15% test set. This allows fair comparison of metrics across runs as new data chunks are added.

**Overfitting detection via logloss curves:** Each Airflow run logs `train_logloss` and `test_logloss` per iteration to MLflow. If train loss drops while test loss rises, overfitting is occurring — visible directly in DagsHub.

**Custom Airflow Docker image:** Instead of installing packages at runtime via `_PIP_ADDITIONAL_REQUIREMENTS` (which reinstalls on every container restart), a custom `Dockerfile.airflow` pre-installs all dependencies at build time.

**Docker Compose for local use only:** Railway runs only the image built from the Dockerfile; Docker Compose is used exclusively for local development, to coordinate the API, PostgreSQL, Prometheus, Grafana, and Airflow together.

**Integration tests on PostgreSQL:** CI runs a PostgreSQL service alongside the test suite (instead of SQLite), giving full parity with the production database engine and catching engine-specific issues (e.g. type handling, constraints) that SQLite would silently miss.

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
- [x] ~~No champion/challenger model evaluation~~ — implemented: retrained models are now statistically compared against production using McNemar's test before promotion. See [Champion/Challenger System](#championchallenger-system) below.
- [x] ~~Recall drops significantly after retraining (~0.83 → ~0.55)~~ — root cause identified and fixed. See [Debugging the Recall Drop](#debugging-the-recall-drop) below.
- [ ] Versioned Docker image tags instead of `latest` (for rollback capability)
- [x] ~~Alerting on pipeline task failure~~ — a basic `on_failure_callback` placeholder is now wired into every task via `default_args`. In a real production environment this would post to a Slack webhook instead of just logging.
- [ ] Prometheus metrics are exposed but no custom Grafana dashboards or alerting rules have been built yet.
- [ ] Νo unit tests, integration tests are limited.
- [ ] Frontend (Streamlit or React)

---

## Debugging the Recall Drop

Early versions of the pipeline showed a sharp recall drop after retraining (~0.86 in production vs ~0.55 in Airflow). Investigation uncovered three separate, compounding issues:

1. **The API never loaded the retrained model.** `retrain_and_predict_model` logged a new model to MLflow every run, but the API always loaded a hardcoded `models:/model_scale_pos_5/1`. The "0.86 vs 0.55" comparison was effectively comparing two unrelated models evaluated on different data — not a real before/after.
2. **Training was always from scratch.** Every run called `mlflow.xgboost.load_model(...)` and then `.fit()` **without** `xgb_model=`, silently discarding all prior tuning and starting with random tree initialization each time.
3. **`scale_pos_weight` was static.** The class-imbalance weighting was never recalculated as new data was added, so it drifted out of sync with the actual class ratio over time.

**Fix:** incremental training (`xgb_model=model.get_booster()`) with a small number of new trees per run, a periodic full reset once a tree-count cap is reached (retrained on all accumulated data with a freshly computed `scale_pos_weight`), and live model refresh in the API via a background scheduler (see below).

### A second, subtler bug: `best_iteration` masking predictions

After fixing the above, the challenger model still produced **identical** predictions to the champion on every single test sample, even though tree counts clearly differed. Root cause: `eval_set` triggers XGBoost's `best_iteration` tracking automatically, and `model.predict()` silently only uses trees up to that point — so newly added trees were being computed (non-zero gain, confirmed via `trees_to_dataframe()`) but never actually used at inference time. Fixed with explicit `early_stopping_rounds` + `iteration_range=(0, model.best_iteration + 1)` on prediction calls.

## Champion/Challenger System

Retraining alone doesn't guarantee an improvement, so every retrained model ("challenger") is statistically compared against the current production model ("champion") before being promoted:

1. Both models predict on the same fixed 15% test set.
2. Predictions are compared pairwise per sample, producing a 2x2 contingency table of agreements/disagreements.
3. **McNemar's test** (`statsmodels`) is used instead of a plain recall comparison, because both models are evaluated on the exact same paired samples — the disagreement cells are what matters statistically, not just the raw recall difference.
4. Promotion (`mlflow.register_model` + `@champion` alias) only happens if the difference is statistically significant (p < 0.05) **and** the challenger's recall is higher.

**Note:** on this project's demo dataset, no meaningful drift exists between the training data and incoming chunks — so the challenger correctly ends up statistically indistinguishable from the champion in most runs, and promotion (correctly) doesn't happen. This isn't a bug in the pipeline; it's the pipeline correctly recognizing there's nothing genuinely new to learn from the data available.

## Live Model Reload (no redeploy needed)

The API refreshes its model, scaler-dependent explainer, and encoder periodically (every 3 minutes, via `APScheduler`) by re-reading the `@champion` MLflow alias, instead of only loading the model once at startup. This decouples the ML model lifecycle from the application deployment lifecycle: a new champion can be promoted by Airflow and picked up by the live API without any Docker rebuild, image push, or Railway redeploy.

---

## Live Demo

🚀 Railway Deployment (currently paused — free tier limitations)
📊 [DagsHub Experiments](https://dagshub.com/Spengian/churn---prediction.mlflow)

---

*Personal project, built to gain hands-on understanding of the full MLOps lifecycle.*
