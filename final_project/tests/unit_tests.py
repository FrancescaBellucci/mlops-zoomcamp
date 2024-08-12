from pathlib import Path
import sys

import pandas as pd

import xgboost as xgb

import pickle

from unittest.mock import patch
from pytest import approx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import code.model_training as model_training
import code.predict as predict
import code.monitoring as monitoring


RANDOM_STATE = model_training.RANDOM_STATE
ISF_VARIABLES = model_training.ISF_VARIABLES
TARGET = model_training.TARGET

with open('./models/xgboost.bin','rb') as f:
    XGB_CLASSIFIER = pickle.load(f)
    
with open('./models/isolation_forest.bin','rb') as f:
    ISF = pickle.load(f)

TEST_DATAFRAME = pd.DataFrame(data = [[619,'France','Female',42,2,0,1,1,1,101348.88,1,1,2,'DIAMOND',464],
                                           [608,'Spain','Male',41,1,0,1,0,0,10000,1,1,1,'SILVER',100],
                                           [500,'Germany','Female',42,3,1000,1,1,1,50000,0,0,3,'GOLD',450]],
                                   columns=['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts',
                                            'HasCrCard','IsActiveMember','EstimatedSalary','Exited','Complain',
                                            'Satisfaction Score','Card Type','Point Earned'])

CATEGORICAL_VARIABLES = ['Geography', 'Gender', 'Card Type']
TEST_DATAFRAME[CATEGORICAL_VARIABLES] = TEST_DATAFRAME[CATEGORICAL_VARIABLES].astype('category')

TEST_TRAIN = TEST_DATAFRAME.loc[[2, 0]]
TEST_VAL = TEST_DATAFRAME.loc[[1]]



def test_read_data():
    
    expected = TEST_DATAFRAME
    
    actual = model_training.read_data('./tests/test_data.parquet')
    
    assert actual.shape == expected.shape, "Shape mismatch"
    assert (actual.columns == expected.columns).all(), "Columns mismatch"
    assert (actual.values == expected.values).all(), "Values mismatch"

def test_preprocess_data():
    
    expected = TEST_DATAFRAME
    
    actual = model_training.preprocess_data(TEST_DATAFRAME)
    
    assert actual.shape == expected.shape, "Shape mismatch"
    assert (actual.columns == expected.columns).all(), "Columns mismatch"
    assert (actual.dtypes == expected.dtypes).all(), "Data types mismatch"
    assert (actual.values == expected.values).all(), "Values mismatch"
    

def test_prepare_dataset():
    
    expected_train = TEST_TRAIN
    expected_val = TEST_VAL
    
    actual_train, actual_val = model_training.prepare_dataset(TEST_DATAFRAME,False)
    
    assert actual_train.shape == expected_train.shape, "Training data shape mismatch"
    assert (actual_train.columns == expected_train.columns).all(), "Training data columns mismatch"
    assert (actual_train.dtypes == expected_train.dtypes).all(), "Training data types mismatch"
    assert (actual_train.values == expected_train.values).all(), "Training data values mismatch"
    
    assert actual_val.shape == expected_val.shape, "Validation data shape mismatch"
    assert (actual_val.columns == expected_val.columns).all(), "Validation data columns mismatch"
    assert (actual_val.dtypes == expected_val.dtypes).all(), "Validation data types mismatch"
    assert (actual_val.values == expected_val.values).all(), "Validation data values mismatch"
    
    
def test_compute_isf_metrics():
    
    anomalies = [0, 0, 1]
    
    expected_pct_returning = 1
    expected_pct_inliers = 0.333
    
    actual_pct_returning, actual_pct_inliers = model_training.compute_isf_metrics(TEST_DATAFRAME, anomalies)
    
    assert actual_pct_returning == approx(expected_pct_returning, rel=1e-2)
    assert actual_pct_inliers == approx(expected_pct_inliers, rel = 1e-2)
    

def test_preparing_xgboost_data():
    
    expected_X_train = TEST_TRAIN.drop('Exited',axis=1)
    expected_y_train = TEST_TRAIN['Exited']
    
    expected_X_val = TEST_VAL.drop('Exited',axis=1)
    expected_y_val = TEST_VAL['Exited']
    
    expected_train_matrix = xgb.DMatrix(expected_X_train, label=expected_y_train, enable_categorical=True)
    expected_val_matrix = xgb.DMatrix(expected_X_val, label=expected_y_val, enable_categorical=True)
    
    actual_X_train, actual_y_train, actual_X_val, actual_y_val, actual_train_matrix, actual_val_matrix = model_training.preparing_xgboost_data(TEST_TRAIN,TEST_VAL)
    
    assert actual_X_train.shape == expected_X_train.shape, "X_train Shape mismatch"
    assert (actual_X_train.columns == expected_X_train.columns).all(), "X_train Columns mismatch"
    assert (actual_X_train.dtypes == expected_X_train.dtypes).all(), "X_train Data types mismatch"
    assert (actual_X_train.values == expected_X_train.values).all(), "X_train Values mismatch"
    
    assert actual_y_train.shape == expected_y_train.shape, "y_train Shape mismatch"
    assert (actual_y_train.dtypes == expected_y_train.dtypes), "y_train Data types mismatch"
    assert (actual_y_train.values == expected_y_train.values).all(), "y_train Values mismatch"
    
    assert actual_X_val.shape == expected_X_val.shape, "X_val Shape mismatch"
    assert (actual_X_val.columns == expected_X_val.columns).all(), "X_val Columns mismatch"
    assert (actual_X_val.dtypes == expected_X_val.dtypes).all(), "X_val Data types mismatch"
    assert (actual_X_val.values == expected_X_val.values).all(), "X_val Values mismatch"
    
    assert actual_y_val.shape == expected_y_val.shape, "y_val Shape mismatch"
    assert (actual_y_val.dtypes == expected_y_val.dtypes), "y_val Data types mismatch"
    assert (actual_y_val.values == expected_y_val.values).all(), "y_val Values mismatch"
    
    assert actual_train_matrix.get_data().shape == expected_train_matrix.get_data().shape, "Dimensions of training DMatrices are not equal"
    assert (actual_train_matrix.get_data().toarray() == expected_train_matrix.get_data().toarray()).all(), "Data of training DMatrices are not equal"
    assert all(actual_train_matrix.get_label()[i] == expected_train_matrix.get_label()[i] for i in range(len(expected_train_matrix.get_label()))), "Labels of training DMatrices are not equal"

    assert actual_val_matrix.get_data().shape == expected_val_matrix.get_data().shape, "Dimensions of validation DMatrices are not equal"
    assert (actual_val_matrix.get_data().toarray() == expected_val_matrix.get_data().toarray()).all(), "Data of validation DMatrices are not equal"
    assert all(actual_val_matrix.get_label()[i] == expected_val_matrix.get_label()[i] for i in range(len(expected_val_matrix.get_label()))), "Labels of validation DMatrices are not equal"
    
def test_train_model_wrong_model_name():
    
    with patch('builtins.print') as mocked_print:
        model_training.train_model('random_forest','./tests/test_data.parquet',False)
        mocked_print.assert_called_once_with("Error: unknown model. model_name can only be \"isolation_forest\" or \"xgboost\".") 

def test_preprocess_data_predict():
    
    expected = TEST_DATAFRAME
    
    actual = predict.preprocess_data(TEST_DATAFRAME)
    
    assert actual.shape == expected.shape, "Shape mismatch"
    assert (actual.columns == expected.columns).all(), "Columns mismatch"
    assert (actual.dtypes == expected.dtypes).all(), "Data types mismatch"
    assert (actual.values == expected.values).all(), "Values mismatch"
    
def test_load_model_worng_model():
    
    with patch('builtins.print') as mocked_print:
        predict.load_model('random_forest')
        mocked_print.assert_called_once_with("Error: unknown model. model_name can only be \"isolation_forest\" or \"xgboost\".") 

def test_compute_predictions_isf():
    
    expected_preds = [1]
    
    model = ISF.fit(TEST_TRAIN[ISF_VARIABLES])
    actual_preds = predict.compute_predictions(model, TEST_VAL.drop('Exited',axis=1))
    
    print(actual_preds)
    
    assert (expected_preds == actual_preds).all()
    
    
def test_compute_predictions_xgb():
    
    expected_preds = [0]
    
    model = XGB_CLASSIFIER.fit(TEST_TRAIN.drop('Exited',axis=1),TEST_TRAIN['Exited'])
    actual_preds = predict.compute_predictions(model, TEST_VAL.drop('Exited',axis=1))
    
    print(actual_preds)
    
    assert (expected_preds == actual_preds).all()
    
def test_split_data():
    
    expected_train = TEST_TRAIN
    expected_val = TEST_VAL
    
    current_data = pd.read_parquet('./tests/test_data.parquet')
    actual_train, actual_val = monitoring.split_data(current_data,False)
    
    assert actual_train.shape == expected_train.shape, "Training data shape mismatch"
    assert (actual_train.columns == expected_train.columns).all(), "Training data columns mismatch"
    assert (actual_train.dtypes == expected_train.dtypes).all(), "Training data types mismatch"
    assert (actual_train.values == expected_train.values).all(), "Training data values mismatch"
    
    assert actual_val.shape == expected_val.shape, "Validation data shape mismatch"
    assert (actual_val.columns == expected_val.columns).all(), "Validation data columns mismatch"
    assert (actual_val.dtypes == expected_val.dtypes).all(), "Validation data types mismatch"
    assert (actual_val.values == expected_val.values).all(), "Validation data values mismatch"
    
def test_compute_predictions_df():
    
    expected_pred_data = TEST_VAL
    expected_pred_data['Anomalies'] = 1
    expected_pred_data['Predictions'] = 0
    
    isf = ISF.fit(TEST_TRAIN[ISF_VARIABLES])
    xgb_classifier = XGB_CLASSIFIER.fit(TEST_TRAIN.drop('Exited',axis=1),TEST_TRAIN['Exited'])
    actual_pred_data = monitoring.compute_predictions_df(TEST_VAL,isf,xgb_classifier)
    
    assert actual_pred_data.shape == expected_pred_data.shape, "Shape mismatch"
    assert (actual_pred_data.columns == expected_pred_data.columns).all(), "Columns mismatch"
    assert (actual_pred_data.dtypes == expected_pred_data.dtypes).all(), "Types mismatch"
    assert (actual_pred_data.values == expected_pred_data.values).all(), "Values mismatch"
