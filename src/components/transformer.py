import tensorflow as tf
import numpy as np
import pandas as pd

from src.config import BATCH_SIZE, WINDOW_SIZE
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    batch_size:int=BATCH_SIZE
    window_size:int=WINDOW_SIZE

class DataTransformer:
    def __init__(self):
        self.config=TransformerConfig()


    def init_transformer(self, rnn_model:bool, ticker:str, train_path:str, test_path:str):
        train_df=pd.read_csv(train_path)
        val_df=pd.read_csv(test_path)

        ticker_serie_train=train_df.loc[:,ticker]
        ticker_serie_val=val_df.loc[:, ticker]

        if rnn_model:
            ticker_serie_train=ticker_serie_train.values
            ticker_serie_val=ticker_serie_val.values

            train_dataset=prepare_dataset(ticker_serie_train,
                                          window_size=self.config.window_size,
                                          batch_size=self.config.batch_size,
                                          train_set=True)
            
            val_dataset=prepare_dataset(ticker_serie_val,
                                          window_size=self.config.window_size,
                                          batch_size=self.config.batch_size,
                                          train_set=False)
            return train_dataset, val_dataset
        return ticker_serie_train, ticker_serie_val
        


def prepare_dataset(values, window_size:int, batch_size:int, shuffle_size:int, train_set:bool):
    dataset=tf.data.Dataset.from_tensor_slices(values)
    dataset=dataset.window(window_size, shift=1, drop_remainder=True)
    dataset=dataset.flat_map(lambda x: x.batch(window_size+1))
    dataset=dataset.map(lambda x:(x[:-1], x[-1]))
    if train_set:
        dataset=dataset.shuffle(buffer_size=shuffle_size)
    dataset=dataset.batch(batch_size=batch_size).prefetch

    return dataset
        