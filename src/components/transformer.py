import tensorflow as tf
import numpy as np
import pandas as pd


class DataTransformer:
    def __init__(self, window_size, batch_size):
        self.window_size=window_size
        self.batch_size=batch_size


    def init_transformer(self, rnn_model:bool, ticker:str, train_path:str, test_path:str, shuffle_size:None):
        train_df=pd.read_csv(train_path)
        val_df=pd.read_csv(test_path)

        ticker_serie_train=train_df.loc[:,ticker]
        ticker_serie_val=val_df.loc[:, ticker]

        if rnn_model:
            ticker_serie_train=ticker_serie_train.values
            ticker_serie_val=ticker_serie_val.values

            train_dataset=prepare_dataset(ticker_serie_train,
                                          window_size=self.window_size,
                                          batch_size=self.batch_size,
                                          shuffle_size=shuffle_size,
                                          train_set=True,
)
            
            val_dataset=prepare_dataset(ticker_serie_val,
                                          window_size=self.window_size,
                                          batch_size=self.batch_size)
            return train_dataset, val_dataset
        return ticker_serie_train, ticker_serie_val
        


def prepare_dataset(values, window_size:int, batch_size:int, shuffle_size:int=None, train_set:bool=False):
    dataset=tf.data.Dataset.from_tensor_slices(values)
    dataset=dataset.window(window_size, shift=1, drop_remainder=True)
    dataset=dataset.flat_map(lambda x: x.batch(window_size+1))
    dataset=dataset.map(lambda x:(x[:-1], x[-1]))
    if train_set:
        dataset=dataset.shuffle(buffer_size=shuffle_size).repeat(1)
    dataset=dataset.batch(batch_size=batch_size).prefetch(1)

    return dataset
        