import sys
import pickle
from pathlib import Path

import pandas as pd
import sklearn
import xgboost as xgb
from flask import Flask, jsonify, request

parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))

from model_training import TARGET, ISF_VARIABLES


def preprocess_data(data: pd.DataFrame):

    print("Preprocessing Data...")

    categorical_variables = ['Geography', 'Gender', 'Card Type']

    data[categorical_variables] = data[categorical_variables].astype('category')

    print("Data read and preprocessed.")

    return data


def load_model(model_name: str):

    if model_name not in ["isolation_forest", "xgboost"]:
        print(
            "Error: unknown model. model_name can only be \"isolation_forest\" or \"xgboost\"."
        )
        return None

    print("Loading model...")

    model_path = '../models/' + model_name + '.bin'

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print("Model loaded.")

    return model


def fit_model(model, train_data: pd.DataFrame):

    if not isinstance(
        model, (xgb.sklearn.XGBClassifier, sklearn.ensemble.IsolationForest)
    ):

        print(
            """Error: model unknown.
            The only acceptable types of models are isolation forest and xgboost."""
        )
        return None

    print("Fitting model...")

    if isinstance(model, xgb.sklearn.XGBClassifier):
        X_train = train_data.drop(TARGET, axis=1)
        y_train = train_data[TARGET]

        model.fit(X_train, y_train)

    else:
        model.fit(train_data[ISF_VARIABLES])

    print("Model fit.")

    return None


def compute_predictions(model, pred_data: pd.DataFrame):

    print("Computing predictions...")

    if not isinstance(
        model, (xgb.sklearn.XGBClassifier, sklearn.ensemble.IsolationForest)
    ):

        print(
            """Error: model unknown.
            The only acceptable types of models are isolation forest and xgboost."""
        )
        return None

    if isinstance(model, xgb.sklearn.XGBClassifier):
        preds = model.predict(pred_data)

    else:
        preds = model.predict(pred_data[ISF_VARIABLES])

    return preds


### Main Method ###


def make_predictions(pred_data):

    train_data_path = "../data/customer_churn_training_data.parquet"

    pred_data = preprocess_data(pred_data)
    train_data = pd.read_parquet(train_data_path)

    isf = load_model("isolation_forest")
    fit_model(isf, train_data)

    if compute_predictions(isf, pred_data) == 1:
        y_pred = 0
    else:
        xgb_classifier = load_model("xgboost")
        fit_model(xgb_classifier, train_data)
        y_pred = compute_predictions(xgb_classifier, pred_data)

    return y_pred


### Flask application ###

app = Flask('churn_prediction')


@app.route("/predict", methods=['POST'])
def get_request():

    request_body = request.get_json()

    pred_data = pd.DataFrame(data=request_body, index=[0])

    y_pred = make_predictions(pred_data)

    result = {"churn_prediction": y_pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
