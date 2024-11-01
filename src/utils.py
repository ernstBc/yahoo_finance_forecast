import os
import pandas as pd
import datetime
from src.config import RAW_DATA_DIR, TRAIN_DATA_DIR,VAL_DATA_DIR, MODEL_DIR
import json
import pickle
from tensorflow import keras


def check_database_day():
    on_time=False

    today=datetime.datetime.now().strftime('%Y-%m-%d')
    db_day=sorted(os.listdir(RAW_DATA_DIR))[0]

    if len(db_day)==0:
        return on_time
    
    db_day=db_day[-1]
    db_day=db_day.split('.')[0]

    if today == db_day:
        on_time=True

    return on_time
        

def get_last_dataset():
    db_name=sorted(os.listdir(RAW_DATA_DIR))[-1]

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
        
    
def load_model(path, keras_model=False):
    if keras_model:
        model=keras.models.load_model(str(path)+'.keras')
    else:
        with open(str(path)+'.pkl', 'rb') as file:
            model=pickle.load(file) 

    return model


def get_last_model_path():
    model_name=sorted(os.listdir(MODEL_DIR))[-1]
    model_name=model_name.split('.')[0]

    return os.path.join(MODEL_DIR, model_name)

def percentage_calculator(series):
    
    percentage_change=round((series.iloc[-1] - series.iloc[-2]) /series.iloc[-2] *100, 2)
    return percentage_change, series.iloc[-1]

