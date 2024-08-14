import requests
import DeepDiff

url = 'http://localhost:9696/predict'

bank_customer = {"CreditScore": 500,
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
            "Point Earned": 450
            }

actual_response = requests.post(url, json=bank_customer).json()
expected_response = {"churn_prediction": 0}

diff = DeepDiff(actual_response, expected_response)

assert 'type_changes' not in diff
assert 'values_changed' not in diff
