from src.components.data_ingestion import DataIngestion
from src.components.transformer import DataTransformer
from src.components.trainer import TrainerARIMA
from src.components.trainer_rnn import TrainerNeuralNetwork
import pandas as pd
from src.config import (TICKERS, 
                        MAX_AR,
                        MAX_I,
                        MAX_MA,
                        EPOCHS,
                        WINDOW_SIZE,
                        BATCH_SIZE
)


def pipeline(ticker='open_MSFT', model:str='both'):
    data_ingesition=DataIngestion()
    transformer_data=DataTransformer(window_size=WINDOW_SIZE, batch_size=BATCH_SIZE)
    trainer_arima=TrainerARIMA()
    rnn_trainer=TrainerNeuralNetwork(recurrent_units=64, dense_units=16)

    raw_path,train_path,test_path=data_ingesition.init_data_ingestion(TICKERS, '5y')

    if model.lower()=='rnn':
            train_ds, test_ds=transformer_data.init_transformer(rnn_model=True,
                                                        ticker=ticker,
                                                        train_path=train_path,
                                                        test_path=test_path,
                                                        shuffle_size=100)
            
            rnn_model_path=rnn_trainer.init_model(train_dataset=train_ds, 
                                          val_dataset=test_ds, 
                                          epochs=EPOCHS)
            
    elif model.lower()=='arima':
            
        
        train_df, test_df=transformer_data.init_transformer(rnn_model=False,
                                                        ticker=ticker,
                                                        train_path=train_path,
                                                        test_path=test_path,
                                                        shuffle_size=None)


        arima_model_path, forecast_model_path=trainer_arima.init_trainer(train_dataset=pd.concat([train_df, test_df], axis=0),
                                                 max_ar=MAX_AR,
                                                 max_i=MAX_I, 
                                                 max_ma=MAX_MA)

    else:
        train_df, test_df=transformer_data.init_transformer(rnn_model=False,
                                                        ticker=ticker,
                                                        train_path=train_path,
                                                        test_path=test_path,
                                                        shuffle_size=None)
        arima_model_path, forecast_model_path=trainer_arima.init_trainer(train_dataset=pd.concat([train_df, test_df], axis=0),
                                                 max_ar=MAX_AR,
                                                 max_i=MAX_I, 
                                                 max_ma=MAX_MA)
    
        train_ds, test_ds=transformer_data.init_transformer(rnn_model=True,
                                                        ticker=ticker,
                                                        train_path=train_path,
                                                        test_path=test_path,
                                                        shuffle_size=100)
        rnn_model_path=rnn_trainer.init_model(train_dataset=train_ds, 
                                          val_dataset=test_ds, 
                                          epochs=EPOCHS)



if __name__=='__main__':
    pipeline()