import numpy as np
import pandas as pd

encode_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

def sharedfunc(df_input: pd.DataFrame, encoder, scaler, model, explainer):
    cat_cols = df_input.select_dtypes(include='object').columns.tolist()
    num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']    
    df_categ = df_input[cat_cols]
    df_num = df_input[num_cols]
    encoded_df_categ = encoder.transform(df_categ)
    df_final = np.hstack([df_num.values, encoded_df_categ])
    df_final_scaled = scaler.transform(df_final)
    shap_values = explainer.shap_values(df_final_scaled)
    feature_names = np.array(encode_cols)
    predictions = model.predict(df_final_scaled)
    prediction_proba = model.predict_proba(df_final_scaled) 
    return predictions, prediction_proba, shap_values, feature_names






