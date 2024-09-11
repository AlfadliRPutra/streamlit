import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from pandas import DataFrame, Series, concat
import joblib
import os
from math import sqrt
from sklearn.metrics import mean_squared_error

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

# def invert_scale(scaler, X, value):
#     new_row = [x for x in X] + [value[0]]  
#     array = np.array(new_row)
#     array = array.reshape(1, len(array))
#     inverted = scaler.inverse_transform(array)
#     return inverted[0, -1]

def invert_scale(scaler, X, value):
    # Pastikan value adalah array numpy
    if np.isscalar(value):
        value = np.array([value])
    
    new_row = np.array([x for x in X] + [value[0]])
    new_row = new_row.reshape(1, -1)
    
    inverted = scaler.inverse_transform(new_row)
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

# Path to save/load the model
model_file_path = 'lstm_model.pkl'

# Function to save model to local file
def save_model(model, model_file_path):
    joblib.dump(model, model_file_path)

# Function to load model from local file
def load_model(model_file_path):
    if os.path.exists(model_file_path):
        return joblib.load(model_file_path)
    else:
        return None

# Streamlit app
st.title("Time Series Forecasting with LSTM")

# Handle file upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")

    # Initialize session state if not already done
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

    if not st.session_state.model_trained:
        with st.spinner("Processing and training the model..."):
            # Load and process the data
            series = pd.read_csv(uploaded_file, usecols=[0], engine='python')
            raw_values = series.values
            diff_values = difference(raw_values, 1)
            supervised = timeseries_to_supervised(diff_values, 1)
            supervised_values = supervised.values
            train = supervised_values[0:]
            scaler, train_scaled = scale(train)
            # Train the model
            lstm_model = fit_lstm(train_scaled, 1, 100, 5)
            # Save the model
            save_model(lstm_model, model_file_path)
            # Update session state
            st.session_state.data = series
            st.session_state.model_file_path = model_file_path
            st.session_state.scaler = scaler
            st.session_state.model_trained = True

    st.sidebar.header("Navigation")
    selection = st.sidebar.radio("Select View", ["Dataset", "Forecast"])

    if selection == "Dataset":
        st.subheader("Dataset Overview")
        st.write(st.session_state.data.head(20))
        
        st.subheader("Line Chart of Dataset")
        st.line_chart(st.session_state.data)

    elif selection == "Forecast":
        st.subheader("Forecasting")

        if 'model_trained' in st.session_state and st.session_state.model_trained:
            if st.session_state.model is None:
                st.session_state.model = load_model(st.session_state.model_file_path)
                
            if st.session_state.model is not None:
                st.write("Model loaded successfully.")
                lstm_model = st.session_state.model
                scaler = st.session_state.scaler
                raw_values = st.session_state.data.values.flatten()
                diff_values = difference(raw_values, 1)
                supervised = timeseries_to_supervised(diff_values, 1)
                supervised_values = supervised.values
                train = supervised_values[0:]
                train_scaled = scaler.transform(train)
                
                # Forecast the entire training dataset
                train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
                hasiltraining = lstm_model.predict(train_reshaped, batch_size=1)
                
                # Walk-forward validation
                predictions = []
                for i in range(len(train_scaled)):
                    X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
                    yhat = forecast_lstm(lstm_model, 1, X)
                    yhat = invert_scale(scaler, X, yhat)
                    yhat = inverse_difference(raw_values, yhat, len(train_scaled)+1-i)
                    predictions.append(yhat)
                
                # Ensure predictions align with raw values
                predictions = np.array(predictions)
                if len(predictions) < len(raw_values):
                    padding = [None] * (len(raw_values) - len(predictions))
                    predictions = np.concatenate([predictions, padding])
            
                # Prepare future predictions
                lastPredict = np.array([predictions[-1]])
                lastPredict = toOneDimension(lastPredict)
                lastPredict = convertDimension(lastPredict)
            
                futureMonth = 6
                futureArray = []
                for i in range(futureMonth):
                    try:
                        lastPredict = lstm_model.predict(lastPredict)
                        lastPredict = lastPredict.flatten()
                        futureArray.append(lastPredict)
                        lastPredict = convertDimension(lastPredict)
                    except Exception as e:
                        st.error(f"Error predicting future values: {e}")
                        break
                
                futureArray = np.array(futureArray).flatten()
                
                # Ensure that futureArray has the correct length
                futureIndex = np.arange(len(predictions), len(predictions) + len(futureArray))
                
                # Combine actual, predictions, and future predictions
                full_index = np.arange(len(raw_values) + len(futureArray))
                full_values = np.concatenate([raw_values, futureArray])
                
                data_for_plotting = pd.DataFrame({
                    'Month': full_index,
                    'Values': full_values
                }).set_index('Month')
                
                st.subheader("Actual and Predicted Data")
                st.line_chart(pd.DataFrame({
                    'Actual Data': np.concatenate([raw_values, [None] * len(futureArray)]),
                    'Predicted Data': predictions
                }), use_container_width=True)
                
                st.subheader("Future Predictions")
                st.line_chart(data_for_plotting, use_container_width=True)
            else:
                st.write("Failed to load model.")

        else:
            st.write("Model not available.")
