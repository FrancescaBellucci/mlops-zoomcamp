from unittest.mock import patch
import os

import requests
from deepdiff import DeepDiff

import mlflow
import pickle

ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH")

download_model("fddbb35346874fd89d03a1558b7a5881","isolation_forest") 
download_model("b234e2475dc1456385bb7ecbf05f83ef","xgboost") 

def test_prediction_app():
    url = 'http://localhost:9696/predict'

    bank_customer = {
        "CreditScore": 500,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 3,
        "Balance": 1000,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000,
        "Complain": 0,
        "SatisfactionScore": 3,
        "Card Type": "GOLD",
        "Point Earned": 450,
    }

    actual_response = requests.post(url, json=bank_customer, timeout=100).json()
    expected_response = {"churn_prediction": 0}

    diff = DeepDiff(actual_response, expected_response)

    assert 'type_changes' not in diff
    assert 'values_changed' not in diff
    
def download_model(run_id, model_name):
    
    logged_model = f"{ARTIFACT_PATH}/1/{run_id}/artifacts/model/"
    model = mlflow.pyfunc.load_model(logged_model)
    
    model_path = '../models/' + model_name + '.bin'
    
    with open(model_path, 'wb') as f_out:
        pickle.dump(model, f_out)


# For offline test
# @patch('requests.post')
# def test_predictions(mock_post):
# mock_post.return_value = {"prediction": 0}
# actual_response = requests.post(url, json=bank_customer, timeout=10)
# expected_response = {"prediction": 0}
