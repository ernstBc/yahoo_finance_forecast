import os
import streamlit as st
import pandas as pd
import datetime
from src.config import TICKERS, RAW_DATA_DIR, MODEL_DIR, METRICS
from src.utils import get_last_dataset, load_model, get_last_model_path, percentage_calculator
from src.pipeline.train_pipeline import pipeline
from src.pipeline.serving_pipeline import serving_pipeline


@st.cache_data
def get_data(path):
    return pd.read_csv(path)


df=get_data(os.path.join(RAW_DATA_DIR, get_last_dataset()))
today_date=datetime.datetime.now().strftime('%Y_%m_%d')
model_trained='open_MSFT'


with st.sidebar:
    st.subheader('Settings')
    time_delta=st.slider('Days', min_value=1, max_value=df.shape[0])
    tickers=st.multiselect(label='tickers', options=TICKERS.split(), default='MSFT')
    ticker_metric=st.selectbox('Select a metric', METRICS)

    st.subheader('Prediction Settings')
    model=st.selectbox('Select a model to make predictions', ['ARIMA', 'RNN'])
    ticker_user=f'{ticker_metric}_{tickers[0]}'


    predict=st.button('Predict next day')
    if predict:
        st.info(f"The actual model was trained for {model_trained}", icon="ℹ️")
        prediction=serving_pipeline(today_date=today_date, 
                                    ticker=ticker_user, 
                                    model=model,)
        st.write(prediction)

    st.subheader('Re Training Options')
    to_train=st.selectbox('Select a model to train', options=['ARIMA', 'RNN', 'Both'])

    if st.button('Re-train'):
        if len(tickers)>1:
            st.write('Select just one ticker to train')
        else:
            st.warning('This should take about two minutes to finish.', icon="ℹ️")
            with st.spinner("Please wait..."):
                pipeline(ticker=ticker_user, model=to_train)
                st.info('Model Trained', icon="ℹ️")
                model_trained=f'{ticker_metric}_{tickers[0]}'


st.header(' '.join(tickers))

col1,col2,col3,col4, col5= st.columns(5)
for idx, col in enumerate([col1,col2,col3,col4]):
    serie=df.loc[:, f'{METRICS[idx]}_{tickers[0]}']
    pc, last_value=percentage_calculator(series=serie)
    col.metric(METRICS[idx], f"{last_value:.2f}", f"{pc:.2f}%")


if tickers is not None:
    st.line_chart(data=df.iloc[-time_delta:], x='Date', y=[f'open_{tic.upper()}' for tic in tickers], y_label='US dollars')
if predict:
    last_value=df.loc[:, f'{ticker_metric}_{tickers[0]}'].values[-1]
    serie=pd.Series([last_value, prediction])
    pc_, last_value_=percentage_calculator(serie)

    col5.metric(f'Predicted {ticker_user}', f'{round(last_value_,2)}', f'{pc_}%')