import batch

import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    
    categorical = ['PULocationID', 'DOLocationID']

    actual_output = batch.prepare_data(df, categorical)
    
    expected_data = [
        ('-1', '-1',  dt(1, 1), dt(1, 10), 9),
        ('1', '1', dt(1, 2), dt(1, 10),  8),     
    ]
    
    expected_output = pd.DataFrame(expected_data, columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration'])
    
    assert (actual_output['PULocationID'] == expected_output['PULocationID']).all()
    assert (actual_output['DOLocationID'] == expected_output['DOLocationID']).all()
    assert (actual_output['tpep_pickup_datetime'] == expected_output['tpep_pickup_datetime']).all()
    assert (actual_output['tpep_dropoff_datetime'] == expected_output['tpep_dropoff_datetime']).all()
    assert (actual_output['duration'] - expected_output['duration']).abs().sum() < 0.0000001
