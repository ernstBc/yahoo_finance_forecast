import os

import streamlit as st
import pandas as pd
from src.config import TICKERS, RAW_DATA_DIR
from src.utils import get_last_dataset

@st.cache_data
def get_data(path):
    return pd.read_csv(path)

df=get_data(os.path.join(RAW_DATA_DIR, get_last_dataset()))

with st.sidebar:
    time_delta=st.slider('Show days before', min_value=1, max_value=df.shape[0])
    tickers=st.multiselect(label='tickers', options=TICKERS.split(), default='MSFT')

    model=st.selectbox('Select a model', ['ARIMA', 'RNN'])

    predict=st.button('Predict next day')


st.header(' '.join(tickers))
st.write('')

col1, col2= st.columns(2)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")

st.write('')
st.line_chart(data=df.iloc[-time_delta:], x='Date', y=[f'open_{tic.upper()}' for tic in tickers], y_label='US dollars')