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
import tensorflow as tf
import os

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

# Fungsi untuk memproses data dan model
def process_and_save_model(file):
    # Load dataset
    series = pd.read_csv(file, usecols=[0], engine='python')

    # Transform data
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # Split data into train
    train = supervised_values[0:]

    # Transform the scale of the data
    scaler, train_scaled = scale(train)

    # Fit the model
    lstm_model = fit_lstm(train_scaled, 1, 100, 5)

    # Save the model
    model_path = 'lstm_model.h5'
    lstm_model.save(model_path)

    return {
        'data': series,
        'model_path': model_path
    }

# Streamlit app
st.title("Time Series Forecasting with LSTM")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    with st.spinner("Processing and training the model..."):
        result = process_and_save_model(uploaded_file)

        st.sidebar.header("Navigation")
        selection = st.sidebar.radio("Select View", ["Dataset", "Forecast"])

        if selection == "Dataset":
            st.subheader("Dataset Overview")
            st.write(result['data'].head(20))
            
            st.subheader("Line Chart of Dataset")
            st.line_chart(result['data'])

        elif selection == "Forecast":
            st.subheader("Model Saved Successfully")
            st.write(f"The model has been saved to: {result['model_path']}")
            
            st.subheader("Future Predictions")
            # Placeholder for future predictions section
            st.write("To view future predictions, please load the model and make predictions.")

# Note: To run this script, ensure you have a `requirements.txt` file with the necessary libraries.
