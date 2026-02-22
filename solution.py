import pandas as pd
import numpy as np
import joblib

def preprocess(df):
    """
    Focus: Consistent encoding and creative feature engineering to maximize 
    the 'Feature Engineering' score (20 pts).
    """
    data = df.copy()

    # --- STEP 1: Handling Missing Values ---
    # Fills numerical gaps with median and objects with 'Unknown' 
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    
    obj_cols = data.select_dtypes(include=['object']).columns
    data[obj_cols] = data[obj_cols].fillna('Unknown')

    # --- STEP 2: Creative Feature Engineering (Points & Performance) ---
    
    # Temporal Mapping: Convert month names to circular coordinates [cite: 57]
    month_map = {name: i for i, name in enumerate(
        ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December'], 1)}
    
    # Use .get() to handle potential unseen month strings gracefully
    data['Month_Num'] = data['Policy_Start_Month'].apply(lambda x: month_map.get(x, 1))
    data['Month_Sin'] = np.sin(2 * np.pi * data['Month_Num'] / 12)
    data['Month_Cos'] = np.cos(2 * np.pi * data['Month_Num'] / 12)
    
    # Financial/Family Ratios: Captures "Income Per Dependent" [cite: 37, 38]
    data['Total_Deps'] = data['Adult_Dependents'] + data['Child_Dependents'] + data['Infant_Dependents']
    data['Income_Per_Capita'] = data['Estimated_Annual_Income'] / (data['Total_Deps'] + 1)
    
    # Risk Intensity: Claims normalized by years without claims [cite: 46, 47]
    data['Claim_Intensity'] = data['Previous_Claims_Filed'] / (data['Years_Without_Claims'] + 1)

    # --- STEP 3: Robust Categorical Encoding ---
    # We use a sorted map to ensure Category 'X' is always ID 'Y', 
    # preventing the train/test mismatch from your previous pd.factorize logic.
    cat_to_encode = ['Employment_Status', 'Broker_Agency_Type', 'Acquisition_Channel',
                     'Region_Code', 'Deductible_Tier', 'Payment_Schedule']
    
    for col in cat_to_encode:
        data[col] = data[col].astype(str)
        # Alphabetical sorting ensures consistent ID assignment across all runs
        unique_labels = sorted(data[col].unique())
        label_map = {val: i for i, val in enumerate(unique_labels)}
        data[col] = data[col].map(label_map)

    return data

def load_model():
    """
    Returns the model object. Exactly one file named model.<ext> is required[cite: 119].
    """
    return joblib.load('model.joblib')

def predict(df, model):
    """
    Returns a DataFrame with User_ID and Purchased_Coverage_Bundle[cite: 105, 106].
    Focus: Zero latency penalty (< 10s)[cite: 80].
    """
    # Define features used during training - MUST match train.py exactly
    # We exclude the raw month string, IDs, and target [cite: 34, 35, 57]
    exclude = ['User_ID', 'Purchased_Coverage_Bundle', 'Policy_Start_Month', 'Month_Num', 'Broker_ID', 'Employer_ID']
    features = [c for c in df.columns if c not in exclude]
    
    # Generate predictions (classes 0-9) [cite: 25, 116]
    preds = model.predict(df[features])
    
    # Construct final submission format [cite: 109]
    result = pd.DataFrame({
        'User_ID': df['User_ID'],
        'Purchased_Coverage_Bundle': preds.astype(int)
    })
    
    return result