import os
import pandas as pd
import datetime
from src.config import RAW_DATA_DIR, TRAIN_DATA_DIR,VAL_DATA_DIR
import json
import pickle


def check_database_day():
    on_time=False

    today=datetime.datetime.now().strftime('%Y-%m-%d')
    db_day=sorted(os.listdir(RAW_DATA_DIR))

    if len(db_day)==0:
        return on_time
    
    db_day=db_day[-1]
    db_day=db_day.split('.')[0]

    if today >= db_day:
        on_time=True

    return on_time
        

def get_last_dataset():
    db_name=os.listdir(RAW_DATA_DIR)[0]

    return db_name


def save_artifact(artifact_path, object, is_json=False):
    dir_path=os.path.dirname(artifact_path)
    os.makedirs(dir_path, exist_ok=True)

    if is_json:
        with open(artifact_path, "w") as file: 
            json.dump(object, file, sort_keys=True, indent=4)
    else:
        with open(artifact_path, 'wb') as file:
            pickle.dump(object, file, protocol=pickle.HIGHEST_PROTOCOL)
        
    