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

# Fungsi untuk mengonversi deret waktu menjadi pembelajaran terawasi
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# Fungsi untuk menghitung selisih
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# Fungsi untuk membalikkan selisih
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# Fungsi untuk mengukur skala data
def scale(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    return scaler, train_scaled

# Fungsi untuk membalikkan skala
def invert_scale(scaler, X, value):
    if np.isscalar(value):
        value = np.array([value])
    
    new_row = np.array([x for x in X] + [value[0]])
    new_row = new_row.reshape(1, -1)
    
    inverted = scaler.inverse_transform(new_row)
    return inverted[0, -1]

# Fungsi untuk melatih model LSTM
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False)
    return model

# Fungsi untuk memprediksi menggunakan LSTM
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

# Fungsi untuk mengubah dimensi untuk LSTM
def convertDimension(value):
    return np.reshape(value, (value.shape[0], 1, value.shape[0]))

# Path untuk menyimpan/mengambil model
model_file_path = 'lstm_model.pkl'

# Fungsi untuk menyimpan model ke file lokal
def save_model(model, model_file_path):
    joblib.dump(model, model_file_path)

# Fungsi untuk memuat model dari file lokal
def load_model(model_file_path):
    if os.path.exists(model_file_path):
        return joblib.load(model_file_path)
    else:
        return None

# Aplikasi Streamlit
st.title("Aplikasi Peramalan")

# Menangani unggahan file
uploaded_file = st.sidebar.file_uploader("Unggah file CSV atau Excel Anda", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Inisialisasi state sesi jika belum dilakukan
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
            # Memuat data berdasarkan tipe file
            file_extension = uploaded_file.name.split('.')[-1]

            if file_extension == 'csv':
                series = pd.read_csv(uploaded_file, usecols=[0, 1], engine='python', header=0)
            elif file_extension in ['xlsx', 'xls']:
                series = pd.read_excel(uploaded_file, usecols=[0, 1], header=0)
                
            # Mengonversi kolom 'Tanggal' menjadi datetime dan mengatur sebagai indeks
            series['Tanggal'] = pd.to_datetime(series['Tanggal'], format='%d/%m/%Y')
            series.set_index('Tanggal', inplace=True)

            raw_values = series['PM10'].values.reshape(-1, 1)
            diff_values = difference(raw_values, 1)
            supervised = timeseries_to_supervised(diff_values, 1)
            supervised_values = supervised.values
            train = supervised_values[0:]
            scaler, train_scaled = scale(train)

            # Melatih model
            lstm_model = fit_lstm(train_scaled, 1, 100, 5)
            # Menyimpan model
            save_model(lstm_model, model_file_path)

            # Memperbarui state sesi
            st.session_state.data = series
            st.session_state.scaler = scaler  # Pastikan skala disimpan
            st.session_state.model_trained = True

    st.sidebar.header("Navigasi")
    selection = st.sidebar.radio("Pilih Tampilan", ["Dataset", "Peramalan"])

        if selection == "Dataset":
            st.subheader("Tabel")
            col1, col2 = st.columns(2)  # Membuat dua kolom
    
            with col1:
                st.write(st.session_state.data.head(20))  # Tampilkan tabel di kolom pertama
    
            with col2:
                st.subheader("Visualisasi data")
                st.line_chart(st.session_state.data['PM10'])  # Grafik di kolom kedua
    
        elif selection == "Peramalan":
            st.subheader("Peramalan")
            
            if st.session_state.model_trained:
                lstm_model = load_model(model_file_path)
                raw_values = st.session_state.data['PM10'].values.reshape(-1, 1)
                diff_values = difference(raw_values, 1)
                supervised = timeseries_to_supervised(diff_values, 1)
                supervised_values = supervised.values
                train = supervised_values[0:]
                train_scaled = st.session_state.scaler.transform(train)
        
                # GUI untuk memilih jumlah hari untuk prediksi
                future_days = st.number_input("Pilih jumlah hari untuk diprediksi:", min_value=1, max_value=30, value=7)
        
                # Menyiapkan prediksi masa depan
                lastPredict = train_scaled[-1, 0].reshape(1, 1, 1)
                future_predictions = []
        
                for _ in range(future_days):
                    yhat = forecast_lstm(lstm_model, 1, lastPredict)
                    future_predictions.append(yhat)
                    lastPredict = convertDimension(np.array([[yhat]]))
                
                # Membalikkan skala dan selisih
                future_predictions_inverted = []
                for i in range(len(future_predictions)):
                    tmp_result = invert_scale(st.session_state.scaler, [0], future_predictions[i])
                    tmp_result = inverse_difference(raw_values, tmp_result, len(future_predictions) + 1 - i)
                    future_predictions_inverted.append(tmp_result)
        
                # Membuat DataFrame untuk prediksi masa depan
                last_date = st.session_state.data.index[-1]  # Mendapatkan tanggal terakhir dari dataset
                future_index = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=future_days, freq='D')
                future_df = pd.DataFrame({
                    'Tanggal': future_index,
                    'Prediksi Masa Depan': future_predictions_inverted
                }).set_index('Tanggal')
        
                # Menampilkan DataFrame prediksi masa depan
                st.subheader("Forecast")
                st.dataframe(future_df)
    
                # Grafik
                plt.figure(figsize=(15, 7))
                plt.plot(st.session_state.data.index, st.session_state.data['PM10'], label="Tingkat PM10")
                plt.plot(future_df.index, future_predictions_inverted, label="Hasil Prediksi", linestyle="--", color="red")
                plt.xlabel("Tanggal")
                plt.ylabel("Tingkat PM10")
                plt.title("Tingkat PM10 dan Hasil Prediksi")
                plt.legend()
                
                col1, col2 = st.columns(2)  # Membuat dua kolom untuk layout
    
                with col1:
                    st.pyplot(plt)  # Grafik di kolom pertama
    
                with col2:
                    st.subheader("Prediksi Dataframe")
                    st.dataframe(future_df)  # Tabel di kolom kedua
            else:
                st.write("Model tidak tersedia.")

