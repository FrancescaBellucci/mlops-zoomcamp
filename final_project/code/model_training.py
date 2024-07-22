import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, f1_score

import xgboost as xgb

import optuna

import mlflow
from mlflow.tracking import MlflowClient

import pickle

from prefect import task, flow
from prefect.task_runners import SequentialTaskRunner

### Prefect Tasks ###

@task
def preprocess_data(file_path: str):
    
    data = pd.read_parquet(file_path)

    columns_to_drop = ['RowNumber','CustomerId','Surname']
    data.drop(columns_to_drop, 
          axis = 1, inplace = True)
    
    categorical_variables = ['Geography', 'Gender', 'Card Type']
    
    data[categorical_variables] = data[categorical_variables].astype('category')
    
    return(data)

@task
def prepare_dataset(data: pd.DataFrame, save_data: bool):
    
    test_size = 0.2
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=RANDOM_STATE)
    
    if save_data == True:
        train_data.to_parquet('../data/customer_churn_training_data.parquet', engine="pyarrow")
        val_data.to_parquet('../data/customer_churn_validation_data.parquet', engine="pyarrow")
    
    return train_data, val_data

@task
def optimize_model(model_name: str, train_data: pd.DataFrame, val_data: pd.DataFrame):
       
    if model_name == "xgboost":
        X_train, y_train, X_val, y_val, train_matrix, val_matrix = preparing_xgboost_data(train_data,val_data)
    
    # Objective function for xgboost
    def objective_xgb(trial):
    
        with mlflow.start_run(run_name = "xgboost_classifier"):
            mlflow.set_tag("Project", "bank_churn_prediction")
            mlflow.set_tag("Developer", "Francesca")
            mlflow.set_tag("Model", "xgboost")
            
            params = {
                'objective': 'binary:logistic',
                'n_estimators': trial.suggest_int('n_estimators', 10, 1000, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1),
                'subsample': trial.suggest_discrete_uniform('subsample', 0.05, 1.0, 0.05),
                'max_depth': trial.suggest_int('max_depth', 2, 100),
                'seed': RANDOM_STATE
            }

            mlflow.log_params(params)
            
            classifier = xgb.train(
                    params = params,
                    dtrain = train_matrix,
                    num_boost_round = 1000,
                    evals=[(val_matrix, 'validation')],
                    early_stopping_rounds = 50
                )

            y_pred = np.rint(classifier.predict(val_matrix))

            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            mlflow.xgboost.log_model(classifier, artifact_path="mlflow_models")
        
        return recall 

    # Objective function for isolation forest
    def objective_isf(trial):
        
        
        with mlflow.start_run(run_name = "anomaly_detection"):
            mlflow.set_tag("Project", "bank_churn_prediction")
            mlflow.set_tag("Developer", "Francesca")
            mlflow.set_tag("Model", "isolation_forest")
        
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 500, 10),
                'contamination': trial.suggest_discrete_uniform('contamination', 0.05, 0.5, 0.05),
                'max_features': trial.suggest_int('max_features', 1, 4),
                'bootstrap': True,
                'warm_start': True,
                'random_state': RANDOM_STATE
            }
            
            mlflow.log_params(params)

            isf = IsolationForest(**params)
            isf.fit(train_data[ISF_VARIABLES])
            
            anomalies = isf.predict(val_data[ISF_VARIABLES])
            
            pct_returning, pct_inliers = compute_isf_metrics(val_data[ISF_VARIABLES], anomalies)
            score = (1 - pct_returning)*(1 - pct_inliers)
            
            mlflow.log_metric("pct_returning", pct_returning)
            mlflow.log_metric("pct_inliers", pct_inliers)
            mlflow.log_metric("score", score)
            
            mlflow.sklearn.log_model(isf, artifact_path="mlflow_models")
        
        return score
    
    study = optuna.create_study(direction='minimize' if model_name == "isolation_forest" else "maximize") 
    study.optimize(objective_isf if model_name=="isolation_forest" else objective_xgb, n_trials=N_TRIALS) 
    
    return study.best_trial

@task
def register_model(best_trial, model_name: str):
    
    if model_name == "isolation_forest":
        model = IsolationForest(**best_trial.params)
        
    else:
        model = xgb.XGBClassifier(**best_trial.params, enable_categorical=True)
    
    model_path = '../models/'+ model_name + '.bin'
    
    #client = MlflowClient("http://127.0.0.1:5000")
    client.search_registered_models()
    
    run_id = client.search_runs(experiment_ids=['1'])[N_TRIALS - best_trial.number].info.run_id
    
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/"+ model_name,
        name=model_name
    )
    
    with open(model_path, 'wb') as f_out:
        pickle.dump(model, f_out)
    
    mlflow.end_run()
    
    return None

### Prefect Flows ###

@flow(task_runner=SequentialTaskRunner())
def train_model(model_name: str, data_file: str, save_data: bool):
    
    if model_name not in ["isolation_forest","xgboost"]:
        print("Error: unknown model. model_name can only be \"isolation_forest\" or \"xgboost\".")
        return None
    
    data = preprocess_data(data_file)
    train_data, val_data = prepare_dataset(data, save_data)  
    optimized_model = optimize_model(model_name, train_data, val_data)
    register_model(optimized_model, model_name)    
    
    return None

### Auxiliary Functions ###

def compute_isf_metrics(val_data: pd.DataFrame, anomalies):
    
    val_data['Anomalies'] = anomalies
    
    pct_returning = len(val_data[(val_data['Anomalies']==1) & (val_data[TARGET]==0)])/len(val_data[val_data['Anomalies']==1])
    pct_inliers = len(val_data[val_data['Anomalies']==1])/len(anomalies)
    
    return pct_returning, pct_inliers  

def preparing_xgboost_data(train_data: pd.DataFrame, val_data: pd.DataFrame):
    
    X_train = train_data.drop(TARGET, axis=1)
    y_train = train_data[TARGET]

    X_val = val_data.drop(TARGET, axis=1)
    y_val = val_data[TARGET]
        
    train_matrix = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    val_matrix = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    
    return X_train, y_train, X_val, y_val, train_matrix, val_matrix
    

if __name__ == "__main__":
    
    RANDOM_STATE = 2024
    
    MLFLOW_TRACKING_URI = "sqlite:///../bank_churn.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("bank_churn_prediction")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    N_TRIALS = 5
    
    ISF_VARIABLES = ['Exited', 'Age','CreditScore','NumOfProducts'] 
    TARGET = 'Exited'
    
    model = "xgboost"
    data_file = "../data/customer_churn_records.parquet"
    save_data = False
     
    train_model(model, data_file, save_data)