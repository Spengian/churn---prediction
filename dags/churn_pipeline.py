from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.models import Variable
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/opt/airflow/project')
from database_airflow import Session, CustomerPred
import mlflow
from airflow.decorators import task, dag
from airflow.exceptions import AirflowSkipException
from sklearn.metrics import recall_score, f1_score, precision_score
import os 
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.preprocessing import StandardScaler
import requests
from sklearn.preprocessing import OneHotEncoder

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@task
def read_next_chunk():
    current_chunk = int(Variable.get("current_chunk", default_var=0))
    file_path = f"/opt/airflow/project/data/chunks_raw/chunk_{current_chunk}.csv"
    return file_path

@task
def simulate_ground_truth(file_path): 
    df = pd.read_csv(file_path)
    churn_labels = df['Churn'].values
    df = df.drop("Churn", axis=1)
    input_data = df.to_dict(orient="records")
    response = requests.post(
        "http://api:8000/predict/batch",
        json = {"customers": input_data}
    )
    response.raise_for_status()
    data = response.json()
    updates = [{"id": r["id"], "churn_real": int(cl)} for r, cl in zip(data["result"], churn_labels)]
    update_churn = requests.patch(
        "http://api:8000/predictions/batch/churn",
        json = {"updates": updates}
    )

@task
def check_drift():
    db = Session()
    records = db.query(CustomerPred).filter(CustomerPred.churn_real != None).all()
    df_new = pd.DataFrame([r.input_data for r in records])
    df_train = pd.read_csv('/opt/airflow/project/data/train_data_raw.csv')
    df_train = df_train.drop('Churn', axis=1)
    os.makedirs('/opt/airflow/project/reports', exist_ok=True)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_train, current_data=df_new)
    report.save_html('/opt/airflow/project/reports/drift_report.html')
    db.close()

@task
def retrain_and_predict_model():
    db = Session()
    try:
        records = db.query(CustomerPred).filter(CustomerPred.churn_real != None).all()
        df_new = pd.DataFrame([r.input_data for r in records])
        churn_col = [r.churn_real for r in records]
        df_test = pd.read_csv('/opt/airflow/project/data/test_data_raw.csv')
        cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                    'PaperlessBilling', 'PaymentMethod']
        num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        df_train = pd.read_csv('/opt/airflow/project/data/train_data_raw.csv')
        encoder.fit(df_train[cat_cols])
        encoded_cols = num_cols + list(encoder.get_feature_names_out(cat_cols))
        df_train_cat = encoder.transform(df_train[cat_cols])
        df_train_num = df_train[num_cols].values
        churn_train = df_train['Churn'].values 
        df_train_encoded = np.hstack([df_train_num, df_train_cat])
        df_train = pd.DataFrame(df_train_encoded, columns=encoded_cols)
        df_train['Churn'] = churn_train
        df_new_cat = encoder.transform(df_new[cat_cols])
        df_new_num = df_new[num_cols].values
        df_new_encoded = np.hstack([df_new_num, df_new_cat])
        df_new = pd.DataFrame(df_new_encoded, columns=encoded_cols)
        df_new['Churn'] = churn_col
        df = pd.concat([df_train, df_new], ignore_index=True)
        churn_test = df_test['Churn'].values
        df_test_cat = encoder.transform(df_test[cat_cols])
        df_test_num = df_test[num_cols].values
        df_test_encoded = np.hstack([df_test_num, df_test_cat])
        df_test = pd.DataFrame(df_test_encoded, columns=encoded_cols)
        df_test['Churn'] = churn_test
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        X_test = df_test.drop('Churn',axis = 1)
        y_test = df_test['Churn']
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        X_test_sc = scaler.transform(X_test)
        os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
        os.environ["MLFLOW_TRACKING_USERNAME"] =  os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
        mlflow.set_experiment("churn-prediction")
        model = mlflow.xgboost.load_model("models:/model_scale_pos_5/1")
        with mlflow.start_run() as run:
            model.fit(X_sc, y, eval_set=[(X_sc, y), (X_test_sc, y_test)], verbose=False)
            y_pred = model.predict(X_test_sc)
            mlflow.log_metric("recall_churn", recall_score(y_test, y_pred))
            mlflow.log_metric("f1_churn", f1_score(y_test, y_pred))
            mlflow.log_metric("precision_churn", precision_score(y_test, y_pred))
            results = model.evals_result()
            train_logloss = results['validation_0']['logloss']  # train
            test_logloss = results['validation_1']['logloss']   # test  
            for i, (tr, te) in enumerate(zip(train_logloss, test_logloss)):
                if i % 25 == 0:
                    mlflow.log_metric("train_logloss", tr, step=i)
                    mlflow.log_metric("test_logloss", te, step=i)
            mlflow.log_metric("final_logloss", train_logloss[-1])
            current_chunk = int(Variable.get("current_chunk", default_var=0))
            mlflow.xgboost.log_model(model, f"model_chunk_{current_chunk}")
    finally:
        db.close()

@task
def update_chunk():
    current_chunk = int(Variable.get("current_chunk", default_var=0))
    if not os.path.exists(f"/opt/airflow/project/data/chunks_raw/chunk_{current_chunk}.csv"):
        raise AirflowSkipException("All chunks processed!")
    Variable.set("current_chunk", current_chunk + 1)

# Creating DAG Object
@dag(
    default_args=default_args,
    schedule_interval='*/3 * * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False
)

def churn_pipeline():
    t1 = read_next_chunk()
    t2 = simulate_ground_truth(t1)
    t3 = check_drift()
    t4 = retrain_and_predict_model()
    t5 = update_chunk()
    t1 >> t2 >> t3 >> t4 >> t5

churn_pipeline()



