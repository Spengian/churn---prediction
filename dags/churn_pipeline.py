from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.models import Variable
import pandas as pd
import sys
sys.path.insert(0, '/opt/airflow/project')
from database_airflow import Session, CustomerPred
import mlflow
import dagshub 
from airflow.decorators import task, dag
from airflow.exceptions import AirflowSkipException
from sklearn.metrics import recall_score, f1_score, precision_score
import os 
import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.preprocessing import StandardScaler

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@task
def read_next_chunk():
    current_chunk = int(Variable.get("current_chunk", default_var=0))
    file_path = f"/opt/airflow/project/data/chunks/chunk_{current_chunk}.csv"
    #Το Airflow δεν μπορεί να περάσει DataFrame μεταξύ tasks απευθείας — χρησιμοποιεί XCom για να περνάει δεδομένα. Αλλά το XCom δεν υποστηρίζει DataFrames.
    return file_path

@task
def load_to_postgres(file_path): 
    df = pd.read_csv(file_path)
    db = Session()
    for _, row in df.iterrows():
        pred = CustomerPred(input_data = row.drop("Churn").to_dict(), churn = int(row["Churn"]), probability = 0)
        db.add(pred)
    db.commit()
    db.close()

@task
def check_drift():
    db = Session()
    records = db.query(CustomerPred).all()
    df_new = pd.DataFrame([r.input_data for r in records])
    df_new['Churn'] = [r.churn for r in records]
    df_train = pd.read_csv('/opt/airflow/project/data/train_data.csv')
    os.makedirs('/opt/airflow/project/reports', exist_ok=True)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_train, current_data=df_new)
    report.save_html('/opt/airflow/project/reports/drift_report.html')
    db.close()

@task
def retrain_and_predict_model():
    db = Session()
    records = db.query(CustomerPred).all()
    df_new = pd.DataFrame([r.input_data for r in records])
    df_new['Churn'] = [r.churn for r in records]
    df_train = pd.read_csv('/opt/airflow/project/data/train_data.csv')
    df_test = pd.read_csv('/opt/airflow/project/data/test_data.csv')
    df = pd.concat([df_train, df_new], ignore_index=True)
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
        mlflow.log_metric("final_logloss", train_logloss[-1])
        current_chunk = int(Variable.get("current_chunk", default_var=0))
        mlflow.xgboost.log_model(model, f"model_chunk_{current_chunk}")
    db.close()

@task
def update_chunk():
    current_chunk = int(Variable.get("current_chunk", default_var=0))
    next_chunk = current_chunk + 1
    if not os.path.exists(f"/opt/airflow/project/data/chunks/chunk_{next_chunk}.csv"):
        raise AirflowSkipException("All chunks processed!")
    Variable.set("current_chunk", next_chunk)

# Creating DAG Object
@dag(
    default_args=default_args,
    schedule_interval='*/3 * * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False
)

def churn_pipeline():
    t1 = read_next_chunk()
    t2 = load_to_postgres(t1)
    t3 = check_drift()
    t4 = retrain_and_predict_model()
    t5 = update_chunk()
    t1 >> t2 >> t3 >> t4 >> t5

churn_pipeline()



