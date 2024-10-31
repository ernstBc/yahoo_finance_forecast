import tensorflow as tf
import datetime
from pathlib import Path

from tensorflow.keras.layers import SimpleRNN, Dense, GRU
from tensorflow.keras import Sequential

from src.config import WINDOW_SIZE, MODEL_DIR, CHECKPOINT_DIR
from dataclasses import dataclass

@dataclass
class TrainerConfig:
    window_size:int=WINDOW_SIZE
    model_id=datetime.datetime.now().strftime('%Y_%m_%d')
    model_path=MODEL_DIR
    model_save_path=Path(model_path, model_id)
    checkpoint=Path.joinpath(CHECKPOINT_DIR, 'checkpoint.weights.h5')


class TrainerNeuralNetwork:
    def __init__(self, recurrent_units, dense_units):
        self.dense_units=dense_units
        self.recurrent_units=recurrent_units
        self.config=TrainerConfig()



    def init_model(self, train_dataset, val_dataset, epochs):
        model=build_nn_models(dense_units=self.dense_units,
                              recurrent_units=self.recurrent_units,
                              window_size=self.config.window_size)
        
        callbacks=get_callbacks(self.config.checkpoint)
        

        hist=model.fit(train_dataset,
                        epochs=epochs, 
                        validation_data=val_dataset,
                        callbacks=[callbacks])
        
        model.load_weights(self.config.checkpoint)
        model.save(str(self.config.model_save_path)+'.keras')

        return self.config.model_save_path, hist
        

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

def get_callbacks(filepath):
    callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        )
    return callback


if __name__=='__main__':
    trainer=TrainerNeuralNetwork(1,1)
    print(trainer.config.checkpoint)