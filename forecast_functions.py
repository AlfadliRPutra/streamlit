# forecast_functions.py
import numpy as np
import pandas as pd
import sklearn
from pandas import DataFrame, Series, concat
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    #test = test.reshape(test.shape[0], test.shape[1])
    #test_scaled = scaler.transform(test)
    return scaler, train_scaled

# # inverse scaling for a forecasted value
# def invert_scale(scaler, X, value):
#     new_row = [x for x in X] + [value]
#     array = np.array(new_row)
#     array = array.reshape(1, len(array))
#     inverted = scaler.inverse_transform(array)
#     return inverted[0, -1]

def invert_scale(scaler, X, value):
    # Convert value to scalar using value[0]
    new_row = [x for x in X] + [value[0]]  # Extract scalar value from value (which is a 1-element array)
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]

    # Reshape X to (samples, timesteps, features)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()

    # Stateless LSTM
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))  # input_shape is (timesteps, features)
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False)

    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

#Convert list to one dimension array
def toOneDimension(value):
    return np.asarray(value)

#Convert to multi dimension array
def convertDimension(value):
    return (np.reshape(lastPredict, (lastPredict.shape[0], 1, lastPredict.shape[0])))