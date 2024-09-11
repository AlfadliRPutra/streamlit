import pandas as pd
import numpy as np
import streamlit as st
from forecast_functions import (
    timeseries_to_supervised, difference, inverse_difference,
    scale, invert_scale, fit_lstm, forecast_lstm,
    toOneDimension, convertDimension
)
from math import sqrt

# Page configuration
st.set_page_config(page_title="Forecast Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")

# ---- FILE UPLOAD ----
st.title(":file_folder: Upload Your Dataset for Forecasting")
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, usecols=[0])
    st.success("File uploaded and data loaded successfully!")

    # Display a sample of the uploaded dataset
    st.write("### Dataset Preview")
    st.write(df.head())

    # Forecasting logic
    series = df
    raw_values = series.values
    diff_values = difference(raw_values, 1)
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    train = supervised_values[0:]
    scaler, train_scaled = scale(train)
    lstm_model = fit_lstm(train_scaled, 1, 100, 5)

    # Forecast the entire training dataset
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    predictions = []
    for i in range(len(train_scaled)):
        X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_values, yhat, len(train_scaled)+1-i)
        predictions.append(yhat)

    # Report performance
    rmse = sqrt(mean_squared_error(raw_values[0:len(predictions)], predictions))
    st.write(f"Test RMSE: {rmse:.3f}")

    # Forecast future months
    futureMonth = st.slider("Select number of future months to predict:", min_value=1, max_value=12, value=6)
    model_predict = lstm_model
    lastPredict = train_scaled[-1, 0:-1]
    futureArray = []
    for i in range(futureMonth):
        lastPredict = model_predict.predict(lastPredict.reshape(1, 1, -1))
        futureArray.append(lastPredict)
        lastPredict = convertDimension(lastPredict)

    # Before denormalize
    newFutureData = np.reshape(futureArray, (-1, 1))
    future_predictions_df = pd.DataFrame(newFutureData, columns=['Future Prediction Result (Before Invert Scaling)'])

    # Change dimension and invert scaling
    newFuture = np.reshape(newFutureData, (-1, 1))
    dataHasilPrediksi = []
    for i in range(len(newFutureData)):
        tmpResult = invert_scale(scaler, [0], newFutureData[i])
        tmpResult = inverse_difference(raw_values, tmpResult, len(newFutureData) + 1 - i)
        dataHasilPrediksi.append(tmpResult)

    # Display future predictions
    st.write("### Future Predictions")
    future_predictions_df = pd.DataFrame(dataHasilPrediksi, columns=['Future Prediction Result (After Invert Scaling)'])
    st.write(future_predictions_df)

    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(raw_values)), raw_values, label='Actual')
    plt.plot(range(len(raw_values), len(raw_values) + len(dataHasilPrediksi)), dataHasilPrediksi, label='Forecasted', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Forecast vs Actual')
    plt.legend()
    st.pyplot(plt)
