import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame, Series, concat

# Fungsi yang sudah ada
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def scale(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    return scaler, train_scaled

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value[0]]  
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2]))) 
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False)
    return model

def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

def toOneDimension(value):
    return np.asarray(value)

def convertDimension(value):
    return np.reshape(value, (value.shape[0], 1, value.shape[0]))

# Fungsi untuk memproses data
def process_data(file):
    series = pd.read_csv(file, usecols=[0], engine='python')
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    train = supervised_values[0:]
    scaler, train_scaled = scale(train)
    lstm_model = fit_lstm(train_scaled, 1, 100, 5)

    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    hasiltraining = lstm_model.predict(train_reshaped, batch_size=1)
    predictions = list()
    tmpPredictions = list()
    for i in range(len(train_scaled)):
        X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        tmpPredictions.append(yhat)
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_values, yhat, len(train_scaled)+1-i)
        predictions.append(yhat)
        expected = raw_values[i+1 ]

    rmse = sqrt(mean_squared_error(raw_values[0:87], predictions))

    futureMonth = 6
    lastPredict = tmpPredictions[-1:]
    lastPredict = toOneDimension(lastPredict)
    lastPredict = convertDimension(lastPredict)
    futureArray = []
    for i in range(futureMonth):
        lastPredict = lstm_model.predict(lastPredict)
        futureArray.append(lastPredict)
        lastPredict = convertDimension(lastPredict)

    newFutureData = np.reshape(futureArray, (-1,1))
    dataHasilPrediksi = []
    for i in range(len(newFutureData)):
        tmpResult = invert_scale(scaler, [0], newFutureData[i])
        tmpResult = inverse_difference(raw_values, tmpResult, len(newFutureData) + 1 - i)
        dataHasilPrediksi.append(tmpResult)

    return {
        'data': series,
        'predictions': predictions,
        'future_predictions': dataHasilPrediksi,
        'rmse': rmse
    }

# Aplikasi Streamlit
st.title("Time Series Forecasting with LSTM")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    with st.spinner("Processing..."):
        result = process_data(uploaded_file)

        st.sidebar.header("Navigation")
        selection = st.sidebar.radio("Select View", ["Dataset", "Forecast"])

        if selection == "Dataset":
            st.subheader("Dataset Overview")
            st.write(result['data'].head(20))
            
            st.subheader("Line Chart of Dataset")
            st.line_chart(result['data'])

        elif selection == "Forecast":
            st.subheader("Forecast Results")
            st.write(f"Test RMSE: {result['rmse']:.3f}")

            st.subheader("Future Predictions")
            st.line_chart(result['future_predictions'])
