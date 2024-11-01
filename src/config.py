from pathlib import Path
import os

# dirs
WORKDIR=Path(os.getcwd())
DATA_DIR=Path(WORKDIR, 'data')
RAW_DATA_DIR=Path.joinpath(DATA_DIR, 'data')
TRAIN_DATA_DIR=Path.joinpath(DATA_DIR, 'train')
VAL_DATA_DIR=Path.joinpath(DATA_DIR, 'val')
MODEL_DIR=Path(WORKDIR, 'models')
CHECKPOINT_DIR=Path.joinpath(WORKDIR, 'tmp', 'model_checkpoints')
FORECAST_DIR=Path.joinpath(WORKDIR, 'artifacts', 'forecast')

# data
TICKERS='msft goog NVDA'.upper()
METRICS=["open","high","low","close","volume","dividends","stock_splits"]

# hyperparams
BATCH_SIZE=32
WINDOW_SIZE=15
EPOCHS=250
MAX_AR=5
MAX_I=1
MAX_MA=5
DENSE_UNITS=16
RECURRENT_UNITS=64


#EPOCHS=250
#MAX_AR=5
#MAX_I=1
#MAX_MA=5