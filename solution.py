import pandas as pd
import numpy as np
import joblib

def preprocess(df):
    data = df.copy()

    # --- STEP 1: Handling Missing Values ---
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    obj_cols = data.select_dtypes(include=['object']).columns
    data[obj_cols] = data[obj_cols].fillna('Unknown')

    # --- STEP 2: Enhanced Feature Engineering ---
    month_map = {name: i for i, name in enumerate(
        ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December'], 1)}
    
    data['Month_Num'] = data['Policy_Start_Month'].map(month_map).fillna(1).astype(np.int8)
    data['Month_Sin'] = np.sin(2 * np.pi * data['Month_Num'] / 12).astype(np.float32)
    data['Month_Cos'] = np.cos(2 * np.pi * data['Month_Num'] / 12).astype(np.float32)
    
    # Interaction Features (The F1 Boosters)
    data['Total_Deps'] = (data['Adult_Dependents'] + data['Child_Dependents'] + data['Infant_Dependents']).astype(np.int8)
    data['Risk_Score'] = (data['Previous_Claims_Filed'] * (data['Vehicles_on_Policy'] + 1)).astype(np.float32)
    data['Financial_Capacity'] = (data['Estimated_Annual_Income'] / (data['Total_Deps'] + 1)).astype(np.float32)
    data['Policy_Intensity'] = (data['Previous_Policy_Duration_Months'] / (data['Policy_Amendments_Count'] + 1)).astype(np.float32)

    # --- STEP 3: Categorical Encoding ---
    cat_to_encode = ['Employment_Status', 'Broker_Agency_Type', 'Acquisition_Channel',
                     'Region_Code', 'Deductible_Tier', 'Payment_Schedule']
    
    for col in cat_to_encode:
        data[col] = data[col].astype(str)
        unique_labels = sorted(data[col].unique())
        label_map = {val: i for i, val in enumerate(unique_labels)}
        data[col] = data[col].map(label_map).fillna(-1).astype(np.int32)

    return data

def load_model():
    return joblib.load('model.joblib')

def predict(df, model):
    exclude = ['User_ID', 'Purchased_Coverage_Bundle', 'Policy_Start_Month', 'Month_Num', 'Broker_ID', 'Employer_ID']
    features = [c for c in df.columns if c not in exclude]
    preds = model.predict(df[features])
    return pd.DataFrame({'User_ID': df['User_ID'], 'Purchased_Coverage_Bundle': preds.astype(int)})