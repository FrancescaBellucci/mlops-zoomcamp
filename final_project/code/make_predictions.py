import pandas as pd
import numpy as np

import sklearn
from sklearn.ensemble import  IsolationForest

import xgboost as xgb
from xgboost import XGBClassifier

import pickle

from prefect import task, flow
from prefect.task_runners import SequentialTaskRunner

from model_training import preprocess_data

### Prefect Tasks ###


@task
def load_model(model_name: str):
    
    if model_name not in ["isolation_forest","xgboost"]:
        print("Error: unknown model. model_name can only be \"isolation_forest\" or \"xgboost\".")
        return None
    
    model_path = '../models/'+ model_name + '.bin' 
    
    with open(model_path,'rb') as f:
        model = pickle.load(f)
        
    return model
        
@task
def fit_model(model, train_data: pd.DataFrame):   
    
    if type(model) not in [xgb.sklearn.XGBClassifier,sklearn.ensemble._iforest.IsolationForest]:
        print("Error: model unknown. The only acceptable types of models are isolation forest and xgboost.")  
        return None
           
    if type(model) == xgb.sklearn.XGBClassifier:
        X_train = train_data.drop(TARGET, axis=1)
        y_train = train_data[TARGET]

        model.fit(X_train, y_train)

        
    else:        
        model.fit(train_data[ISF_VARIABLES])
        
    return None

@task 
def compute_predictions(model, pred_data: pd.DataFrame):
    
    if type(model) not in [xgb.sklearn.XGBClassifier,sklearn.ensemble._iforest.IsolationForest]:
        print("Error: model unknown. The only acceptable types of models are isolation forest and xgboost.")
        return None
    
    if type(model) == xgb.sklearn.XGBClassifier:
        preds = model.predict(pred_data.drop(TARGET, axis=1))
        
    else:        
        preds = model.predict(pred_data[ISF_VARIABLES])
        
    return preds

### Prefect Flows ###

@flow(task_runner=SequentialTaskRunner())
def make_predictions(pred_data_path: str): 
    
    #pred_data = preprocess_data(pred_data_path)
    pred_data = pd.read_parquet(pred_data_path)
    train_data = pd.read_parquet(TRAIN_DATA_PATH)
    
    isf = load_model("isolation_forest")
    fit_model(isf, train_data)
    anomalies = compute_predictions(isf, pred_data)
    
    pred_data['Predictions'] = 1-anomalies
    idx_outliers = pred_data[pred_data['Predictions'] == 2].index
    
    xgb_classifier = load_model("xgboost")
    fit_model(xgb_classifier, train_data)
    X_pred = pred_data[pred_data['Predictions']==2].drop(['Predictions'], axis=1)
    pred_xgb = compute_predictions(xgb_classifier, X_pred)
    
    pred_data.loc[idx_outliers, "Predictions"] = pred_xgb
    y_pred = pred_data["Predictions"].values
    
    return y_pred


if __name__ == "__main__":

    TRAIN_DATA_PATH = "../data/customer_churn_training_data.parquet"
    ISF_VARIABLES = ['Exited', 'Age','CreditScore','NumOfProducts'] 
    TARGET = 'Exited'
    
    pred_data_path = "../data/customer_churn_validation_data.parquet"
    
    make_predictions(pred_data_path)