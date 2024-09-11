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

# Helper functions
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def difference(dataset, interval=1):
    return Series([dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))])

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def scale(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    return scaler, train_scaled

def invert_scale(scaler, X, value):
    new_row = np.array(X + [value]).reshape(1, -1)
    return scaler.inverse_transform(new_row)[0, -1]

def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, :-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False)
    return model

def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    return model.predict(X, batch_size=batch_size)[0, 0]

def to_one_dimension(value):
    return np.asarray(value)

def convert_dimension(value):
    return np.reshape(value, (value.shape[0], 1, value.shape[0]))

# File path to save/load the model
model_file_path = 'lstm_model.pkl'

def save_model(model, model_file_path):
    joblib.dump(model, model_file_path)

def load_model(model_file_path):
    return joblib.load(model_file_path) if os.path.exists(model_file_path) else None

# Streamlit app
st.title("Time Series Forecasting with LSTM")

# Sidebar for file upload and navigation
st.sidebar.header("Upload & Navigation")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")

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
            try:
                # Load and process the data
                series = pd.read_csv(uploaded_file, usecols=[0], engine='python')
                raw_values = series.values
                diff_values = difference(raw_values, 1)
                supervised = timeseries_to_supervised(diff_values, 1)
                supervised_values = supervised.values
                train = supervised_values
                scaler, train_scaled = scale(train)
                lstm_model = fit_lstm(train_scaled, batch_size=1, nb_epoch=100, neurons=5)
                save_model(lstm_model, model_file_path)

                st.session_state.data = series
                st.session_state.model_file_path = model_file_path
                st.session_state.scaler = scaler
                st.session_state.model_trained = True

            except Exception as e:
                st.error(f"Error during processing: {e}")
                st.session_state.model_trained = False

    # Sidebar navigation
    selection = st.sidebar.radio("Select View", ["Dataset Overview", "Forecast"])

    if selection == "Dataset Overview":
        st.subheader("Dataset Overview")
        st.write(st.session_state.data.head(20))
        
        st.subheader("Line Chart of Dataset")
        st.line_chart(st.session_state.data)

    elif selection == "Forecast":
        st.subheader("Forecasting")

        if st.session_state.model_trained:
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
                train = supervised_values
                train_scaled = scaler.transform(train)

                # Forecast the entire training dataset
                train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
                tmp_predictions = [forecast_lstm(lstm_model, 1, X) for X in train_scaled[:, :-1]]
                predictions = [invert_scale(scaler, X, yhat) for X, yhat in zip(train_scaled[:, :-1], tmp_predictions)]
                predictions = [inverse_difference(raw_values, yhat, len(train_scaled) + 1 - i) for i, yhat in enumerate(predictions)]

                # Flatten arrays before plotting
                raw_values = raw_values.flatten()
                predictions = np.array(predictions).flatten()

                # Report performance
                rmse = sqrt(mean_squared_error(raw_values, predictions))
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
                last_predict = to_one_dimension([tmp_predictions[-1]])
                last_predict = convert_dimension(last_predict)

                future_month = 6
                future_array = []
                for _ in range(future_month):
                    last_predict = lstm_model.predict(last_predict)
                    future_array.append(last_predict)
                    last_predict = convert_dimension(last_predict)

                new_future_data = np.reshape(future_array, (-1, 1))
                data_hasil_prediksi = [inverse_difference(raw_values, invert_scale(scaler, [0], future_data), len(new_future_data) + 1 - i) for i, future_data in enumerate(new_future_data)]

                # Plot future predictions
                plt.figure(figsize=(15, 7))
                plt.plot(raw_values, label="Actual data")
                plt.plot(np.arange(len(predictions)), predictions, label="Predicted data", linestyle="--")
                future_index = np.arange(len(predictions), len(predictions) + len(data_hasil_prediksi))
                plt.plot(future_index, data_hasil_prediksi, label="Future Predictions", linestyle="--", color="orange")
                plt.xlabel("Month")
                plt.ylabel("Case")
                plt.title("Actual Data, Predictions, and Future Predictions")
                plt.legend()
                st.pyplot(plt)

            else:
                st.error("Failed to load model.")
        else:
            st.warning("Model not available.")
