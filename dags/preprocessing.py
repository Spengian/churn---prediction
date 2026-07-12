from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
import numpy as np

def prepare_data(df_new, churn_col):
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
    ###################################################
    df_test = pd.read_csv('/opt/airflow/project/data/test_data_raw.csv')
    churn_test = df_test['Churn'].values
    df_test_cat = encoder.transform(df_test[cat_cols])
    df_test_num = df_test[num_cols].values
    df_test_encoded = np.hstack([df_test_num, df_test_cat])
    df_test = pd.DataFrame(df_test_encoded, columns=encoded_cols)
    df_test['Churn'] = churn_test
    ###################################################
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_test = df_test.drop('Churn',axis = 1)
    y_test = df_test['Churn']
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_test_sc = scaler.transform(X_test)
    return X_sc, y, X_test_sc, y_test