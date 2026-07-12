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
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier
from preprocessing import prepare_data
from mlflow import MlflowClient

def alert_on_failure(context):
    task_id = context['task_instance'].task_id
    dag_id = context['task_instance'].dag_id
    exception = context.get('exception')
    print(f"ALERT: Task {task_id} in DAG {dag_id} failed: {exception}")

default_args = {
    'owner': 'airflow',
    'retries': 0,
    'on_failure_callback': alert_on_failure,
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
        X_sc, y, X_test_sc, y_test = prepare_data(df_new, churn_col)
        ###################################################
        os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
        os.environ["MLFLOW_TRACKING_USERNAME"] =  os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
        mlflow.set_experiment("churn-prediction")
        model_uri = Variable.get("last_model_uri", default_var="models:/model_scale_pos_5/1")
        print(f"DEBUG: Loading model from URI: {model_uri}")
        model = mlflow.xgboost.load_model(model_uri)
        with mlflow.start_run() as run:
            MAX_TREES = 200
            current_trees = model.get_booster().num_boosted_rounds()

            if current_trees < MAX_TREES:
                # Incremental: συνέχισε πάνω στα υπάρχοντα trees
                print(f"BEFORE fit: {model.get_booster().num_boosted_rounds()} trees")
                model.n_estimators = 25  # λίγα νέα trees κάθε φορά
                model.fit(X_sc, y, xgb_model=model.get_booster(), 
                        eval_set=[(X_sc, y), (X_test_sc, y_test)], verbose=False)
                print(f"AFTER fit: {model.get_booster().num_boosted_rounds()} trees")
                trees_df = model.get_booster().trees_to_dataframe()
                last_25_trees = trees_df[trees_df['Tree'] >= 17]
                print(f"Νέα trees - Gain stats:\n{last_25_trees['Gain'].describe()}")
                print(f"best_iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'not set'}")
                print(f"best_ntree_limit: {model.best_ntree_limit if hasattr(model, 'best_ntree_limit') else 'not set'}")
            else:
                # Reset: fresh training από την αρχή, πάνω σε ΟΛΟ το accumulated data
                neg = y.value_counts()[0]
                pos = y.value_counts()[1]
                dynamic_scale_pos_weight = neg / pos

                model = XGBClassifier(scale_pos_weight=dynamic_scale_pos_weight, n_estimators=100, early_stopping_rounds = 5)
                model.fit(X_sc, y, eval_set=[(X_sc, y), (X_test_sc, y_test)], verbose=False)
            print(f"challenger proba (first 5): {model.predict_proba(X_test_sc)[:5, 1]}")
            best_iter = model.best_iteration
            challenger_preds = model.predict(X_test_sc, iteration_range=(0, best_iter + 1))
            mlflow.log_metric("recall_churn", recall_score(y_test, challenger_preds))
            mlflow.log_metric("f1_churn", f1_score(y_test, challenger_preds))
            mlflow.log_metric("precision_churn", precision_score(y_test, challenger_preds))
            results = model.evals_result()
            train_logloss = results['validation_0']['logloss']  # train
            test_logloss = results['validation_1']['logloss']   # test  
            for i, (tr, te) in enumerate(zip(train_logloss, test_logloss)):
                if i % 5 == 0:
                    mlflow.log_metric("train_logloss", tr, step=i)
                    mlflow.log_metric("test_logloss", te, step=i)
            mlflow.log_metric("final_logloss", train_logloss[-1])
            current_chunk = int(Variable.get("current_chunk", default_var=0))
            mlflow.xgboost.log_model(model, f"model_chunk_{current_chunk}")
            Variable.set("last_model_uri", f"runs:/{run.info.run_id}/model_chunk_{current_chunk}")
    finally:
        db.close()
    return y_test.tolist(), challenger_preds.tolist(), df_new.to_dict(orient="records"), churn_col, run.info.run_id, current_chunk # μικρά, ελαφριά arrays -> OK για XCom

@task
def champion_vs_challenger(data):
    y_test, challenger_preds, df_new_records, churn_col, run_id, current_chunk = data
    df_new = pd.DataFrame(df_new_records)
    _, _, X_test_sc, y_test = prepare_data(df_new, churn_col)
    ######################################
    # Προβλέψεις champion & challenger πάνω στο ΙΔΙΟ 15% test set
    champion_model = mlflow.xgboost.load_model("models:/model_scale_pos_5/1")
    champion_preds = champion_model.predict(X_test_sc)
    print(f"champion proba (first 5): {champion_model.predict_proba(X_test_sc)[:5, 1]}")
    champion_correct = (champion_preds == y_test)
    challenger_correct = (challenger_preds == y_test)
    a = d = b = c = 0
    for i in range(len(y_test)):
        if champion_correct[i] and challenger_correct[i]:
            a += 1
        elif not champion_correct[i] and not challenger_correct[i]:
            d += 1
        elif champion_correct[i] and not challenger_correct[i]:
            b += 1
        elif not champion_correct[i] and challenger_correct[i]:
            c += 1

    data = [[a, b], [c, d]]
    result = mcnemar(data, exact=False)
    challenger_recall = recall_score(y_test, challenger_preds)
    champion_recall = recall_score(y_test, champion_preds)
    print(f"a={a}, b={b}, c={c}, d={d}, champion_recall={champion_recall}, challenger_recall={challenger_recall}")
    with mlflow.start_run(run_id = run_id):
        mlflow.log_metric("p-value", result.pvalue)
    if result.pvalue < 0.05 and challenger_recall > champion_recall:  
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model_chunk_{current_chunk}",
            name="model_scale_pos_5"
        )
        client = MlflowClient()
        client.set_registered_model_alias("model_scale_pos_5", "champion", result.version)
    else:
        pass  # keep same champion

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
    catchup=False,
    max_active_runs=1
)

def churn_pipeline():
    t1 = read_next_chunk()
    t2 = simulate_ground_truth(t1)
    t3 = check_drift()
    t4 = retrain_and_predict_model()
    t5 = champion_vs_challenger(t4)
    t6 = update_chunk()
    t1 >> t2 >> t3 >> t4 >> t5 >> t6

churn_pipeline()



