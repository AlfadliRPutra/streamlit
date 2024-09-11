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
                raw_values = st.session_state.data.values
                diff_values = difference(raw_values, 1)
                supervised = timeseries_to_supervised(diff_values, 1)
                supervised_values = supervised.values
                train = supervised_values[0:]
                train_scaled = scaler.transform(train)
                
                # Forecast the entire training dataset
                train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
                hasiltraining = lstm_model.predict(train_reshaped, batch_size=1)
                
                # Walk-forward validation
                predictions = list()
                tmpPredictions = list()
                for i in range(len(train_scaled)):
                    X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
                    yhat = forecast_lstm(lstm_model, 1, X)
                    tmpPredictions.append(yhat)
                    yhat = invert_scale(scaler, X, yhat)
                    yhat = inverse_difference(raw_values, yhat, len(train_scaled)+1-i)
                    
                    # Ensure yhat is a scalar
                    yhat = yhat if np.issubdtype(type(yhat), np.number) else yhat.item()
                    
                    predictions.append(yhat)
                
                # Flatten arrays before plotting
                raw_values = raw_values.flatten()
                predictions = np.array(predictions).flatten()
                
                # Report performance
                rmse = sqrt(mean_squared_error(raw_values[0:87], predictions))
                st.write(f'Test RMSE: {rmse:.3f}')
                
                # Plotting
                plt.figure(figsize=(15, 7))
                plt.plot(raw_values, label="Actual data")
                plt.plot(np.arange(len(predictions)), predictions, label="Predicted data", linestyle="--")
                plt.xlabel("Month")
                plt.ylabel("Case")
                plt.title("Actual Data vs. Predictions")
                plt.legend()
                st.pyplot(plt)
                
                # Prepare future predictions
                lastPredict = tmpPredictions[-1:]
                lastPredict = toOneDimension(lastPredict)
                lastPredict = convertDimension(lastPredict)
            
                futureMonth = 6  # Predict for 6 months
            
                futureArray = []
                for i in range(futureMonth):
                    lastPredict = lstm_model.predict(lastPredict)
                    futureArray.append(lastPredict)
                    lastPredict = convertDimension(lastPredict)
            
                # Before denormalize
                newFutureData = np.reshape(futureArray, (-1, 1))
                newFuture = np.reshape(newFutureData, (-1, 1))
            
                dataHasilPrediksi = []
                for i in range(len(newFutureData)):
                    tmpResult = invert_scale(scaler, [0], newFutureData[i])
                    tmpResult = inverse_difference(raw_values, tmpResult, len(newFutureData) + 1 - i)
                    dataHasilPrediksi.append(tmpResult)
            
                # Plot future predictions
                plt.figure(figsize=(15, 7))
                plt.plot(raw_values, label="Actual data")
                plt.plot(np.arange(len(predictions)), predictions, label="Predicted data", linestyle="--")
                future_index = np.arange(len(predictions), len(predictions) + len(dataHasilPrediksi))
                plt.plot(future_index, dataHasilPrediksi, label="Future Predictions", linestyle="--", color="orange")
                plt.xlabel("Month")
                plt.ylabel("Case")
                plt.title("Actual Data, Predictions, and Future Predictions")
                plt.legend()
                st.pyplot(plt)
            
            else:
                st.write("Failed to load model.")

        else:
            st.write("Model not available.")
