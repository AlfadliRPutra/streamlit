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
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math

# Functions for RMSE and MAPE
def calculate_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

def calculate_mape(actual, predicted):
    return mean_absolute_percentage_error(actual, predicted) * 100

# Time series to supervised learning
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# Functions for differencing
def difference(dataset, interval=1):
    diff = [dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))]
    return Series(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# Scale data
def scale(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    return scaler, train_scaled

# Inverse scaling
def invert_scale(scaler, X, value):
    new_row = np.array([x for x in X] + [value])
    new_row = new_row.reshape(1, -1)
    inverted = scaler.inverse_transform(new_row)
    return inverted[0, -1]

# Fit LSTM model
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False)
    return model

# Forecast using LSTM
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

# Convert dimensions
def convertDimension(value):
    return np.reshape(value, (value.shape[0], 1, value.shape[0]))

# File path for model
model_file_path = 'lstm_model.pkl'

# Save and load model functions
def save_model(model, model_file_path):
    joblib.dump(model, model_file_path)

def load_model(model_file_path):
    if os.path.exists(model_file_path):
        return joblib.load(model_file_path)
    else:
        return None

# Streamlit app
st.title("Aplikasi Peramalan")

# File uploader
uploaded_file = st.sidebar.file_uploader("Unggah file CSV atau Excel Anda", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None

    if not st.session_state.model_trained:
        with st.spinner("Memproses dan melatih model..."):
            # Load data
            file_extension = uploaded_file.name.split('.')[-1]
            if file_extension == 'csv':
                series = pd.read_csv(uploaded_file, usecols=[0, 1], engine='python', header=0)
            elif file_extension in ['xlsx', 'xls']:
                series = pd.read_excel(uploaded_file, usecols=[0, 1], header=0)

            # Process data
            series['Tanggal'] = pd.to_datetime(series['Tanggal'], format='%d/%m/%Y')
            series.set_index('Tanggal', inplace=True)

            raw_values = series['PM10'].values.reshape(-1, 1)
            diff_values = difference(raw_values, 1)
            supervised = timeseries_to_supervised(diff_values, 1)
            supervised_values = supervised.values
            train = supervised_values[0:]
            scaler, train_scaled = scale(train)

            # Train model
            lstm_model = fit_lstm(train_scaled, 1, 100, 5)
            save_model(lstm_model, model_file_path)

            # Update session state
            st.session_state.data = series
            st.session_state.scaler = scaler
            st.session_state.model_trained = True

    st.sidebar.header("Navigasi")
    selection = st.sidebar.radio("Pilih Tampilan", ["Dataset", "Peramalan"])

    if selection == "Dataset":
        st.subheader("Tabel")
        st.write(st.session_state.data)
    
        st.subheader("Visualisasi data")
        plt.figure(figsize=(10, 6))
        plt.plot(st.session_state.data.index, st.session_state.data['PM10'], label='PM10')
        plt.xlabel('Tanggal')
        plt.ylabel('Konsentrasi PM10')
        plt.title('Visualisasi Data PM10')
        plt.legend()
        st.pyplot(plt)

    elif selection == "Peramalan":
        st.subheader("Peramalan")

        if st.session_state.model_trained:
            lstm_model = load_model(model_file_path)
            raw_values = st.session_state.data['PM10'].values.reshape(-1, 1)

            # Prepare last prediction
            lastPredict = raw_values[-1:]
            lastPredict = toOneDimension(lastPredict)
            lastPredict = convertDimension(lastPredict)

            # Input for future predictions
            future_days = st.number_input("Pilih jumlah hari untuk diprediksi:", min_value=1, max_value=300)

            if future_days > 0:
                futureArray = []
                for _ in range(future_days):
                    lastPredict = lstm_model.predict(lastPredict)
                    futureArray.append(lastPredict)
                    lastPredict = convertDimension(lastPredict)

                # Before denormalize
                newFutureData = np.reshape(futureArray, (-1, 1))
                dataHasilPrediksi = []

                for i in range(len(newFutureData)):
                    tmpResult = invert_scale(st.session_state.scaler, raw_values[-1], newFutureData[i])
                    tmpResult = inverse_difference(raw_values, tmpResult, len(newFutureData) + 1 - i)
                    dataHasilPrediksi.append(tmpResult)

                # Create future dates
                last_date = st.session_state.data.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

                # Create DataFrame for future predictions
                df_future = pd.DataFrame({
                    'Tanggal': future_dates,
                    'Future Predictions': np.round(dataHasilPrediksi)
                }).set_index('Tanggal')

                # Create a DataFrame for actual data
                df_actual = pd.DataFrame(st.session_state.data['PM10'])

                # Create testing predictions DataFrame (assuming newTestingLine is prepared)
                # Replace this with your actual testing predictions logic
                newTestingLine = []  # This should be defined based on your previous model evaluation
                len_testing_line = len(newTestingLine)
                if len_testing_line > 0:
                    testing_index = df_actual.index[-len_testing_line:]
                    shifted_testing_index = testing_index.insert(0, testing_index[0] - pd.Timedelta(days=1)).drop(testing_index[-1])
                    df_testing = pd.DataFrame({
                        'Tanggal': shifted_testing_index,
                        'Testing Predictions': newTestingLine[-len_testing_line:]
                    }).set_index('Tanggal')
                else:
                    raise ValueError("No predictions available in newTestingLine")

                # Plot actual, testing, and future data
                plt.figure(figsize=(15, 10))

                # Plot actual data
                plt.plot(df_actual.index, df_actual['PM10'], label='Actual Data', color='blue')

                # Plot testing predictions
                plt.plot(df_testing.index, df_testing['Testing Predictions'], label='Testing Predictions', linestyle='--', color='orange')

                # Plot future predictions
                plt.plot(df_future.index, df_future['Future Predictions'], label='Future Predictions', linestyle='--', color='red')

                # Auto-format dates
                plt.gcf().autofmt_xdate()

                # Add labels and title
                plt.xlabel("Tanggal")
                plt.ylabel("PM10")
                plt.title("Actual Data vs Testing Predictions vs Future Predictions")

                # Show legend
                plt.legend()

                # Show plot
                st.pyplot(plt)

                # Display future predictions table
                st.subheader("Tabel Prediksi")
                st.dataframe(df_future)
