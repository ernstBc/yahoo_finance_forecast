import os
import pandas as pd
import datetime
from src.config import RAW_DATA_DIR, TRAIN_DATA_DIR,VAL_DATA_DIR



def check_database_day():
    on_date=False

    today=datetime.datetime.now().strftime('%Y-%m-%d')
    db_day=sorted(os.listdir(RAW_DATA_DIR))

    if len(db_day)==0:
        return on_date
    
    db_day=db_day[-1]
    db_day=db_day.split('.')[0]

    if today >= db_day:
        on_date=True

    return on_date
        

def get_last_dataset():
    db_name=os.listdir(RAW_DATA_DIR)[0]

    return db_name

    