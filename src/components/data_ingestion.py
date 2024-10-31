import pandas as pd
import os
import yfinance as yf
from src.config import RAW_DATA_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR, TICKERS
from dataclasses import dataclass
import datetime
from src.utils import check_database_day
from src.config import TICKERS

@dataclass
class IngestionConfig:
    date:str=datetime.datetime.now().strftime('%Y-%m-%d')+'.csv'
    raw_data_dir:str=os.path.join(RAW_DATA_DIR, date)
    train_data_dir:str=os.path.join(TRAIN_DATA_DIR, date)
    val_data_dir:str=os.path.join(VAL_DATA_DIR, date)


class DataIngestion:
    def __init__(self):
        self.data_config=IngestionConfig()

    def init_data_ingestion(self, tickers:str, time_delta:str):
        db_on_date=check_database_day()
        print('check data', db_on_date)
        
        if db_on_date == False:
            print('Downloading today data')
            data=get_data(tickers, time_delta)
            data.index=data.index.strftime('%Y-%m-%d')

            train=data.iloc[:int(data.shape[0] * 0.9), :]
            val=data.iloc[-int(data.shape[0] * 0.9):, :]



            data.to_csv(self.data_config.raw_data_dir)
            train.to_csv(self.data_config.train_data_dir)
            val.to_csv(self.data_config.val_data_dir)

        return (self.data_config.raw_data_dir,
                self.data_config.train_data_dir,
                self.data_config.val_data_dir)



def get_data(tickers:str, time_delta:str):
        label_tickers=tickers.split(' ')

        tickers = yf.Tickers(tickers)
        
        data=pd.DataFrame({})
        for idx, tick in enumerate(label_tickers):
            new_df=tickers.tickers[tick.upper()].history(period=time_delta)
            new_df.columns=[f'{col.replace(' ', '_').lower()}_{tick}' for col in new_df.columns]

            if idx==0:
                data=new_df
            elif idx==1:
                data=data.join(new_df, how='left', on='Date')
            else:
                data=data.join(new_df, how='left', on='Date')
                

        return data
    

if __name__=='__main__':
    ingestion=DataIngestion()
    ingestion.init_data_ingestion(TICKERS, '5y')