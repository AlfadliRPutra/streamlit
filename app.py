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

# File types that can be uploaded
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:

    # Initialize session state if not already done
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

    # Determine the file extension to process the data
    file_extension = uploaded_file.name.split('.')[-1]

    # Load the file based on the extension
    if file_extension == 'csv':
        series = pd.read_csv(uploaded_file, usecols=[0], engine='python')
    elif file_extension in ['xls', 'xlsx']:
        series = pd.read_excel(uploaded_file, usecols=[0])
    
    st.session_state.data = series

    if not st.session_state.model_trained:
        with st.spinner("Processing and training the model..."):
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
    
                # Add user input for future forecast horizon
                st.subheader("Future Forecasting")
                future_months = st.number_input('Select number of months to forecast:', min_value=1, max_value=24, value=6)
                
                # Prepare future predictions
                lastPredict = tmpPredictions[-1:]
                lastPredict = toOneDimension(lastPredict)
                lastPredict = convertDimension(lastPredict)
            
                futureArray = []
                for i in range(future_months):
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
                plt.plot(future_index, dataHasilPrediksi, label="Future Predictions", linestyle="--", color="red")
                plt.xlabel("Month")
                plt.ylabel("Case")
                plt.title(f"Actual Data, Predictions, and {future_months}-Month Future Predictions")
                plt.legend()
                st.pyplot(plt)
            
            else:
                st.write("Failed to load model.")


        else:
            st.write("Model not available.")
