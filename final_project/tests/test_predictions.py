from unittest.mock import patch

import requests
from deepdiff import DeepDiff


@patch('requests.post')
def test_predictions(mock_post):
    mock_post.return_value = {"prediction": 0}
    url = "http://localhost:9696/predict"
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
    actual_response = requests.post(url, json=bank_customer, timeout=10)
    expected_response = {"prediction": 0}

    diff = DeepDiff(actual_response, expected_response)

    assert 'type_changes' not in diff
    assert 'values_changed' not in diff


# url = 'http://localhost:9696/predict'

# bank_customer = {
#     "CreditScore": 500,
#     "Geography": "Germany",
#     "Gender": "Female",
#     "Age": 42,
#     "Tenure": 3,
#     "Balance": 1000,
#     "NumOfProducts": 1,
#     "HasCrCard": 1,
#     "IsActiveMember": 1,
#     "EstimatedSalary": 50000,
#     "Complain": 0,
#     "SatisfactionScore": 3,
#     "Card Type": "GOLD",
#     "Point Earned": 450,
# }

# actual_response = requests.post(url, json=bank_customer, timeout=10).json()
# expected_response = {"churn_prediction": 0}

# diff = DeepDiff(actual_response, expected_response)

# assert 'type_changes' not in diff
# assert 'values_changed' not in diff
