import json

import requests

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

response = requests.post(url, json=bank_customer).json()

if response["churn_prediction"] == 1:
    print('The customer will quit.')
else:
    print('The customer will not quit.')
