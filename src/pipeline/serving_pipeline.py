import os
import pandas as pd
from src.config import MODEL_DIR, FORECAST_DIR, RAW_DATA_DIR, WINDOW_SIZE
from src.utils import get_last_model_path, get_last_dataset, load_model
from src.components.transformer import prepare_dataset
import datetime

def serving_pipeline(today_date:str, ticker:str, model:str='ARIMA',pred_only_next_day=True):
    if model == 'ARIMA':
        today_date=datetime.datetime.now().strftime('%Y_%m_%d')
        model_path=os.path.join(FORECAST_DIR, today_date)
        forecast_model=load_model(model_path)
        
        prediction=forecast_model.forecast()
        prediction=prediction.iloc[0]
        
    else:
        values=pd.read_csv(os.path.join(RAW_DATA_DIR, f'{today_date.replace('_','-')}.csv')).loc[:, ticker].values[-WINDOW_SIZE:]
        data_pred=prepare_dataset(values, window_size=WINDOW_SIZE, batch_size=WINDOW_SIZE)
        model_path=os.path.join(MODEL_DIR, today_date)

        model=load_model(model_path, keras_model=True)
        prediction=model.predict(data_pred)[0][0]

    return prediction






if __name__=='__main__':
    today_date=datetime.datetime.now().strftime('%Y_%m_%d')
    prediction=serving_pipeline(today_date=today_date,
                                ticker='open_MSFT',
                                model='keras')
    print(prediction)