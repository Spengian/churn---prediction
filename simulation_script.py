import pandas as pd
import requests

current_chunk = 0  
file_path = f"C:\\Users\\spegi\\OneDrive\\Documents\\Churn\\data\\chunks_raw\\chunk_{current_chunk}.csv"

df = pd.read_csv(file_path)
churn_labels = df['Churn'].values
df = df.drop("Churn", axis=1)
input_data = df.to_dict(orient="records")
response = requests.post(
    "http://localhost:8000/predict/batch",
    json = {"customers": input_data}
)

data = response.json()
updates = [{"id": r["id"], "churn_real": int(cl)} for r, cl in zip(data["result"], churn_labels)]
update_churn = requests.patch(
    f"http://localhost:8000/predictions/batch/churn",
    json = {"updates": updates}
)