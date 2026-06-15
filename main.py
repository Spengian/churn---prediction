from pydantic import BaseModel
from fastapi import FastAPI, Depends
import joblib 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from database import Base, engine, Session, get_db, CustomerPred
from contextlib import asynccontextmanager
from sqlalchemy import select
import mlflow
import dagshub
import os 
from prometheus_fastapi_instrumentator import Instrumentator


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Spengian/churn---prediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Spengian"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
model = mlflow.xgboost.load_model("models:/model_scale_pos_5/1")
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield
    # εδώ γίνεται το shutdown

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
    Churn : int
    probability : float
    input_data : dict

class BatchOutput(BaseModel):
    result: list[CustomerOutput]

encode_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

@app.post("/predict", status_code=201, response_model= CustomerOutput)
def data_input(input : CustomerInput, db: Session = Depends(get_db)):
    df_input = pd.DataFrame([input.model_dump()])
    cat_cols = df_input.select_dtypes(include='object').columns.tolist()
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']    
    df_categ = df_input[cat_cols]
    df_num = df_input[num_cols]
    encoded_df_categ = encoder.transform(df_categ)
    df_final = np.hstack([df_num.values, encoded_df_categ])
    df_final_scaled = scaler.transform(df_final)
    predictions = model.predict(df_final_scaled)
    prediction_proba = model.predict_proba(df_final_scaled) 
    new_pred = CustomerPred(churn = int(predictions[0]), 
                            probability = float(prediction_proba[0][1]),
                            input_data = input.model_dump())
    db.add(new_pred)
    db.commit()
    db.refresh(new_pred)
    return {"input_data": input.model_dump(), "Churn": int(predictions[0]), "probability": float(prediction_proba[0][1])}

@app.post("/predict/batch", status_code=201, response_model= BatchOutput)
def data_input(input : BatchInput, db: Session = Depends(get_db)):
    df_input = pd.DataFrame([c.dict() for c in input.customers])
    cat_cols = df_input.select_dtypes(include='object').columns.tolist()
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']    
    df_categ = df_input[cat_cols]
    df_num = df_input[num_cols]
    encoded_df_categ = encoder.transform(df_categ)
    df_final = np.hstack([df_num.values, encoded_df_categ])
    df_final_scaled = scaler.transform(df_final)
    predictions = model.predict(df_final_scaled)
    prediction_proba = model.predict_proba(df_final_scaled)
    for pred, proba, inp in zip(predictions,prediction_proba[:,1], input.customers):
        new_pred = CustomerPred(churn = int(pred), 
                                probability = float(proba),
                                input_data = inp.dict())
        db.add(new_pred)
    db.commit()
    return {"result": [{"input_data": id.dict(), "Churn": p, "probability":pb} for p,pb,id in zip(predictions,prediction_proba[:,1], input.customers)]}

@app.get("/health")
def get_status():
    return {"status": "ok"}

@app.get("/predictions")
def get_pred(db: Session = Depends(get_db)):
    query = select(CustomerPred)
    result = db.execute(query)
    return {"customers": result.scalars().all()}