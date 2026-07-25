from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException
import joblib 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from database import Base, engine, Session, get_db, CustomerPred
from contextlib import asynccontextmanager
from sqlalchemy import select
import mlflow
import os 
from prometheus_fastapi_instrumentator import Instrumentator
import shap
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from sharedfunc import sharedfunc
import threading

lock = threading.Lock()

def reload_model():
    global model, explainer
    with lock:
        try:
            model = mlflow.xgboost.load_model("models:/model_scale_pos_5@champion")
        except Exception:
            model = mlflow.xgboost.load_model("models:/model_scale_pos_5/1")
        explainer = shap.TreeExplainer(model)

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Spengian/churn---prediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Spengian"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')
reload_model() 

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    scheduler = BackgroundScheduler()
    scheduler.add_job(reload_model, 'interval', minutes=3)
    scheduler.start()
    yield
    # εδώ γίνεται το shutdown
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

Instrumentator().instrument(app).expose(app)
class CustomerInput(BaseModel):
    gender           :  str
    SeniorCitizen    :  int
    Partner          :  str
    Dependents       :  str
    tenure           :  int
    PhoneService     :  str
    MultipleLines    :  str
    InternetService  :  str
    OnlineSecurity   :  str
    OnlineBackup     :  str
    DeviceProtection :  str
    TechSupport      :  str
    StreamingTV      :  str
    StreamingMovies  :  str
    Contract         :  str
    PaperlessBilling :  str
    PaymentMethod    :  str
    MonthlyCharges   :  float

class BatchInput(BaseModel):
    customers: list[CustomerInput]

class CustomerOutput(BaseModel):
    id: int
    churn_pred: int
    churn_real: Optional[int] = None
    probability : float
    input_data : dict
    top_features : dict

class BatchOutput(BaseModel):
    result: list[CustomerOutput]

class ChurnBatchUpdate(BaseModel):
    updates: list[dict]

@app.post("/predict", status_code=201, response_model= CustomerOutput)
def data_input(input : CustomerInput, db: Session = Depends(get_db)):
    df_input = pd.DataFrame([input.model_dump()])
    with lock:
        local_model = model
        local_explainer = explainer
    predictions, prediction_proba, shap_values, feature_names = sharedfunc(df_input, encoder, scaler, local_model, local_explainer)
    top_indices = np.argsort(np.abs(shap_values[0]))[-2:][::-1]
    top_features = feature_names[top_indices]
    top_shap_values = shap_values[0][top_indices]
    top_features_dict = dict(zip(top_features.tolist(), top_shap_values.tolist()))
    new_pred = CustomerPred(churn_pred = int(predictions[0]),
                            churn_real = None, 
                            probability = float(prediction_proba[0][1]),
                            input_data = input.model_dump())
    db.add(new_pred)
    db.flush()
    record_id = new_pred.id
    db.commit()
    db.refresh(new_pred)
    return {"id": record_id,
            "input_data": input.model_dump(), 
            "churn_pred": int(predictions[0]),
            "churn_real": None, 
            "probability": float(prediction_proba[0][1]), 
            "top_features": top_features_dict}

@app.post("/predict/batch", status_code=201, response_model= BatchOutput)
def batch_input(input : BatchInput, db: Session = Depends(get_db)):
    df_input = pd.DataFrame([c.model_dump() for c in input.customers])
    with lock:
            local_model = model
            local_explainer = explainer
    predictions, prediction_proba, shap_values, feature_names = sharedfunc(df_input, encoder, scaler, local_model, local_explainer)
    top_features_list = []
    ids_list = []
    for i, (pred, proba, inp) in enumerate(zip(predictions,prediction_proba[:,1], input.customers)):
        new_pred = CustomerPred(churn_pred = int(pred),
                                churn_real = None, 
                                probability = float(proba),
                                input_data = inp.model_dump())
        top_indices = np.argsort(np.abs(shap_values[i]))[-2:][::-1]
        top_feat = dict(zip(feature_names[top_indices].tolist(), shap_values[i][top_indices].tolist()))
        top_features_list.append(top_feat)
        db.add(new_pred)
        db.flush()  # για να πάρεις το id χωρίς commit
        ids_list.append(new_pred.id)
    db.commit()
    return {"result": [{"id": record_id, 
                        "input_data": inp.model_dump(), 
                        "churn_pred": cp, 
                        "churn_real": None,
                        "probability": pb, 
                        "top_features": tf} 
                   for cp, pb, inp, tf, record_id in zip(predictions, prediction_proba[:,1], input.customers, top_features_list, ids_list)]}

@app.patch("/predictions/batch/churn")
def batch_update_churn(body: ChurnBatchUpdate, db: Session = Depends(get_db)):
    for update in body.updates:
        pred = db.query(CustomerPred).filter(CustomerPred.id == update["id"]).first()
        if pred:
            pred.churn_real = update["churn_real"]
    db.commit()
    return {"updated": len(body.updates)}

@app.get("/health")
def get_status():
    return {"status": "ok"}

@app.get("/predictions")
def get_pred(db: Session = Depends(get_db)):
    query = select(CustomerPred)
    result = db.execute(query)
    return {"customers": result.scalars().all()}