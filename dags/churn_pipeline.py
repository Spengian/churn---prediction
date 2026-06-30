from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.models import Variable
import pandas as pd

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Creating DAG Object
dag = DAG(
    'churn_pipeline',
    default_args=default_args,
    schedule_interval='*/5 * * * *',  # κάθε 5 λεπτά για testing
    start_date=datetime(2024, 1, 1),
    catchup=False
)

def read_next_chunk():
    current_chunk = int(Variable.get("current_chunk", default_var=0))
    file_path = f"C:/Users/spegi/OneDrive/Documents/Churn/data/chunks/chunk_{current_chunk}.csv"
    #Το Airflow δεν μπορεί να περάσει DataFrame μεταξύ tasks απευθείας — χρησιμοποιεί XCom για να περνάει δεδομένα. Αλλά το XCom δεν υποστηρίζει DataFrames.
    return file_path