import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure the runner can find the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.solution import load_model, predict, preprocess

@pytest.fixture
def sample_input():
    """Generates a sample dataframe matching the actual train.csv structure."""
    data = {
        'User_ID': ['TEST_001', 'TEST_002'],
        'Policy_Cancelled_Post_Purchase': [0, 1],
        'Policy_Start_Year': [2016, 2017],
        'Policy_Start_Week': [50, 10],
        'Policy_Start_Day': [8, 4],
        'Grace_Period_Extensions': [0, 1],
        'Previous_Policy_Duration_Months': [3, 12],
        'Adult_Dependents': [2, 1],
        'Child_Dependents': [0.0, 1.0],
        'Infant_Dependents': [0, 0],
        'Region_Code': ['AUT', 'PRT'],
        'Existing_Policyholder': [0, 1],
        'Previous_Claims_Filed': [0, 2],
        'Years_Without_Claims': [0, 5],
        'Policy_Amendments_Count': [0, 1],
        'Broker_ID': [9.0, 250.0],
        'Employer_ID': [np.nan, 101.0],
        'Underwriting_Processing_Days': [0, 2],
        'Vehicles_on_Policy': [0, 1],
        'Custom_Riders_Requested': [0, 2],
        'Broker_Agency_Type': ['Urban_Boutique', 'National_Corporate'],
        'Deductible_Tier': ['Tier_4_Zero_Ded', 'Tier_1_High_Ded'],
        'Acquisition_Channel': ['Aggregator_Site', 'Direct_Website'],
        'Payment_Schedule': ['Monthly_EFT', 'Annual'],
        'Employment_Status': ['Employed_FullTime', 'Self_Employed'],
        'Estimated_Annual_Income': [26267.93, 55000.00],
        'Days_Since_Quote': [28, 4],
        'Policy_Start_Month': ['April', 'July']
    }
    return pd.DataFrame(data)

def test_load_model():
    model = load_model()
    assert model is not None

def test_prediction_format(sample_input):
    model = load_model()
    processed_input = preprocess(sample_input)
    
    results = predict(processed_input, model)
    
    assert isinstance(results, pd.DataFrame)
    assert 'User_ID' in results.columns
    assert 'Purchased_Coverage_Bundle' in results.columns
    assert len(results) == len(sample_input)

def test_prediction_values(sample_input):
    model = load_model()
    processed_input = preprocess(sample_input)
    results = predict(processed_input, model)
    bundles = results['Purchased_Coverage_Bundle'].unique()
    
    for b in bundles:
        assert 0 <= int(b) <= 9

def test_latency_check(sample_input):
    import time
    model = load_model()
    processed_input = preprocess(sample_input)
    
    start_time = time.time()
    _ = predict(processed_input, model)
    duration = time.time() - start_time
    
    assert duration < 2.0