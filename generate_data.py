import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import joblib 

file_path = r"C:\Users\spegi\OneDrive\Documents\Churn\data\WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['Churn'] = df['Churn'].replace({"Yes": 1, "No": 0})

df_new = df.drop(columns=['customerID','TotalCharges'])
encoded_df = pd.get_dummies(df_new,
                            columns = df_new.select_dtypes(include='object').columns,
                            drop_first=True)
encoded_df = encoded_df.replace({True: 1, False: 0})
cat_cols = df_new.select_dtypes(include='object').columns.tolist()
num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']

X = df_new.drop('Churn', axis=1)
y = df_new['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_new, y_test, y_new = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_train_categorical = X_train[cat_cols]
X_train_numerical = X_train[num_cols]
X_temp_categorical = X_temp[cat_cols]
X_temp_numerical = X_temp[num_cols]
X_test_categorical = X_test[cat_cols]
X_test_numerical = X_test[num_cols]
X_new_categorical = X_new[cat_cols]
X_new_numerical = X_new[num_cols]

# fit τον encoder στο training data
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop = "first")
encoder = joblib.load('models/encoder.pkl')
X_train_encoded = encoder.fit_transform(X_train_categorical)
X_temp_encoded = encoder.transform(X_temp_categorical)
X_test_encoded = encoder.transform(X_test_categorical)
X_new_encoded = encoder.transform(X_new_categorical)

X_train_final = np.hstack([X_train[num_cols].values, X_train_encoded])
X_temp_final = np.hstack([X_temp[num_cols].values, X_temp_encoded])
X_test_final = np.hstack([X_test[num_cols].values, X_test_encoded])
X_new_final = np.hstack([X_new[num_cols].values, X_new_encoded])

print(f"Training: {X_train_final.shape}")
print(f"Test: {X_test_final.shape}")
print(f"New data: {X_new_final.shape}")

chunks_X = np.array_split(X_new_final, 10)
chunks_y = np.array_split(y_new.values, 10)

print(f"Chunk size: {len(chunks_X[0])}")

os.makedirs('data/chunks', exist_ok=True)

# παιρνεις τα column names μετα το encoding
encoded_cols = num_cols + list(encoder.get_feature_names_out(cat_cols))

for i, (chunk_x, chunk_y) in enumerate(zip(chunks_X, chunks_y)):
    chunk_df = pd.DataFrame(chunk_x, columns=encoded_cols)
    chunk_df['Churn'] = chunk_y
    chunk_df.to_csv(f'data/chunks/chunk_{i}.csv', index=False)

print("Chunks saved!")

train_data = pd.DataFrame(X_train_final, columns=encoded_cols)
train_data['Churn'] = y_train.values
train_data.to_csv(f'data/train_data.csv', index = False)

test_data = pd.DataFrame(X_test_final, columns=encoded_cols)
test_data['Churn'] = y_test.values
test_data.to_csv(f'data/test_data.csv', index = False)

os.makedirs('data/chunks_raw', exist_ok=True)

chunks_X_raw = np.array_split(X_new, 10)
chunks_y_raw = np.array_split(y_new.values, 10)

for i, (chunk_x, chunk_y) in enumerate(zip(chunks_X_raw, chunks_y_raw)):
    chunk_df = pd.DataFrame(chunk_x)
    chunk_df['Churn'] = chunk_y
    chunk_df.to_csv(f'data/chunks_raw/chunk_{i}.csv', index=False)

print("Raw chunks saved!")

train_data_raw = X_train.copy()
train_data_raw['Churn'] = y_train.values
train_data_raw.to_csv('data/train_data_raw.csv', index=False)

test_data_raw = X_test.copy()
test_data_raw['Churn'] = y_test.values
test_data_raw.to_csv('data/test_data_raw.csv', index=False)