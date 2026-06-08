from pydantic import BaseModel
from fastapi import FastAPI
import joblib 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')

app = FastAPI()

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

class CustomerOutput(BaseModel):
    Churn : int
    probability : float

encode_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

@app.post("/predict", status_code=201, response_model= CustomerOutput)
async def data_input(input : CustomerInput):
    df_input = pd.DataFrame([input.dict()])
    cat_cols = df_input.select_dtypes(include='object').columns.tolist()
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']    
    df_categ = df_input[cat_cols]
    df_num = df_input[num_cols]
    encoded_df_categ = encoder.transform(df_categ)
    df_final = np.hstack([df_num.values, encoded_df_categ])
    df_final_scaled = scaler.transform(df_final)
    predictions = model.predict(df_final_scaled)
    prediction_proba = model.predict_proba(df_final_scaled) 
    return {"Churn": int(predictions[0]), "probability": float(prediction_proba[0][1])}

@app.get("/health")
async def get_status():
    return {"status": "ok"}