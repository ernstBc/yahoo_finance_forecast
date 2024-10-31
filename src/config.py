from pathlib import Path
import os


WORKDIR=Path(os.getcwd())
DATA_DIR=Path(WORKDIR, 'data')
RAW_DATA_DIR=Path.joinpath(DATA_DIR, 'data')
TRAIN_DATA_DIR=Path.joinpath(DATA_DIR, 'train')
VAL_DATA_DIR=Path.joinpath(DATA_DIR, 'val')
MODEL_DIR=Path(WORKDIR, 'models')


TICKERS='msft goog NVDA'.upper()

BATCH_SIZE=32
WINDOW_SIZE=15

if __name__=='__main__':
    print(WORKDIR)