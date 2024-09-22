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
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Initialize session state if not already done
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'raw_data' not in st.session_state:  # Menyimpan data asli
        st.session_state.raw_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

    if not st.session_state.model_trained:
        with st.spinner("Processing and training the model..."):
            # Determine the file extension and load the data accordingly
            file_extension = uploaded_file.name.split('.')[-1]

            if file_extension == 'csv':
                # Load all columns for raw data
                st.session_state.raw_data = pd.read_csv(uploaded_file, engine='python')
                # Use only the second column (index 1) for the series
                series = st.session_state.raw_data.iloc[:, 1]
            elif file_extension in ['xlsx', 'xls']:
                # Load all columns for raw data
                st.session_state.raw_data = pd.read_excel(uploaded_file, header=0)
                # Use only the second column (index 1) for the series
                series = st.session_state.raw_data.iloc[:, 1]

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
        st.write(st.session_state.raw_data.head(20))  # Tampilkan data asli
        
        st.subheader("Line Chart of Dataset")
        st.line_chart(st.session_state.raw_data)  # Tampilkan grafik dari data asli

    elif selection == "Forecast":
        st.subheader("Forecasting")
    
        if 'model_trained' in st.session_state and st.session_state.model_trained:
            if st.session_state.model is None:
                st.session_state.model = load_model(st.session_state.model_file_path)
                
            if st.session_state.model is not None:
                lstm_model = st.session_state.model
                scaler = st.session_state.scaler
                raw_values = st.session_state.data.values
                diff_values = difference(raw_values, 1)
                supervised = timeseries_to_supervised(diff_values, 1)
                supervised_values = supervised.values
                train = supervised_values[0:]
                train_scaled = scaler.transform(train)
                
                # Prepare future predictions
                lastPredict = train_scaled[-1, 0].reshape(1, 1, 1)
                future_months = 6  # Predict for 6 months
                future_predictions = []
    
                for _ in range(future_months):
                    yhat = forecast_lstm(lstm_model, 1, lastPredict)
                    future_predictions.append(yhat)
                    lastPredict = convertDimension(np.array([[yhat]]))
                
                # Inverse scaling and differencing
                future_predictions_inverted = []
                for i in range(len(future_predictions)):
                    tmp_result = invert_scale(scaler, [0], future_predictions[i])
                    tmp_result = inverse_difference(raw_values, tmp_result, len(future_predictions) + 1 - i)
                    future_predictions_inverted.append(tmp_result)
    
                # Create a DataFrame for future predictions
                future_index = np.arange(len(raw_values), len(raw_values) + future_months)
                future_df = pd.DataFrame({
                    'Month': future_index,
                    'Future Prediction': future_predictions_inverted
                })
    
                # Display the future predictions DataFrame
                st.subheader("Future Predictions")
                st.dataframe(future_df)
    
                # Plotting
                plt.figure(figsize=(15, 7))
                plt.plot(raw_values, label="Actual data")
                plt.plot(np.arange(len(raw_values), len(raw_values) + future_months), future_predictions_inverted, label="Future Predictions", linestyle="--", color="red")
                plt.xlabel("Month")
                plt.ylabel("Case")
                plt.title("Actual Data and Future Predictions")
                plt.legend()
                st.pyplot(plt)
            
            else:
                st.write("Failed to load model.")
        else:
            st.write("Model not available.")
