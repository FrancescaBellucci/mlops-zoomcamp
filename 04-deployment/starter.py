import os
import sys

import pickle
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor


def read_data(filename: str):
    
    categorical = ['PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



def prepare_dictionaries(df: pd.DataFrame):

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    
    return dicts
    



def apply_model(model_file, input_file, output_file, year, month):
    
    print("Loading model")
    
    with open(model_file, 'rb') as f_in:
        	dv, model = pickle.load(f_in)
    
    print("Reading data")
    df = read_data(input_file)
    
    print("Preparing dictionaries")
    dicts = prepare_dictionaries(df)
    
    print("Applying model")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print("The mean predicted duration is ", y_pred.mean())
    
    print("Saving results")
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame({'ride_id':df['ride_id'], 'predicted_duration':y_pred})

    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)
    

def run():
    
    year = int(sys.argv[1]) #2023
    month = int(sys.argv[2]) #3
    taxi_color = sys.argv[3] #'yellow'
    
    model_file = './model.bin'
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_color}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'./output/predictions_{taxi_color}_tripdata_{year:04d}_{month:02d}.parquet'
    
    apply_model(model_file, input_file, output_file, year, month)


if __name__ == "__main__" :
    run()



