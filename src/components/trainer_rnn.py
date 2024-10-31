import tensorflow as tf
import datetime

from tensorflow.keras.layers import SimpleRNN, Dense, GRU
from tensorflow.keras import Sequential

from src.config import WINDOW_SIZE, MODEL_DIR
from dataclasses import dataclass

@dataclass
class TrainerConfig:
    window_size:int=WINDOW_SIZE
    model_id=datetime.datetime.now().strftime('%Y-%m-%d')
    model_path=MODEL_DIR
    model_save_path=str(model_path)+str(model_id)


class TrainerNeuralNetwork:
    def __init__(self, recurrent_units, dense_units):
        self.dense_units=dense_units
        self.recurrent_units=recurrent_units
        self.config=TrainerConfig()



    def init_model(self, train_dataset, val_dataset, epochs):
        model=build_nn_models(dense_units=self.dense_units,
                              recurrent_units=self.recurrent_units,
                              window_size=self.config.window_size)
        

        hist=model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        tf.saved_model.save(model, self.config.model_save_path)

        return self.confi.model_save_path, hist
        

def build_nn_models(dense_units, recurrent_units, window_size):
    model=Sequential([
        GRU(units=recurrent_units, input_shape=[window_size, 1]),
        Dense(dense_units, activation='selu'),
        Dense(1)
    ])

    model.compile(loss='huber',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  metrics=['mae'])
    
    return model