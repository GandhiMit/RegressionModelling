from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt


def model_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(
        SimpleRNN(
            hidden_units,
            input_shape=input_shape,
            activation=activation[0]
        )
    )
    model.add(
        Dense(
            units=dense_units,
            activation=activation[0]
         )
    )
    model.compile(
        loss='mean_squared_error',
        optimizer='adam'
    )
    return model


def function():

    print("Running Simple_RNN")
    RNN = model_RNN(2, 1, (3, 1), activation=['linear'])
    wx = RNN.get_weights()[0]
    wh = RNN.get_weights()[1]
    bh = RNN.get_weights()[2]
    wy = RNN.get_weights()[3]
    by = RNN.get_weights()[4]

#     Inputs
    x = np.array([1, 2, 3])
    # Reshape the input to the required sample_size x time_steps x features
    x_input = np.reshape(x, (1, 3, 1))
    y_pred_model = RNN.predict(x_input)

    m = 2
    h0 = np.zeros(m)
    h1 = np.dot(x[0], wx) + h0 + bh
    h2 = np.dot(x[1], wx) + np.dot(h1, wh) + bh
    h3 = np.dot(x[2], wx) + np.dot(h2, wh) + bh
    o3 = np.dot(h3, wy) + by

    print('h1 = ', h1, 'h2 = ', h2, 'h3 = ', h3)

    print("Prediction from network ", y_pred_model)
    print("Prediction from our computation ", o3)

if __name__ == '__main__':
    function()
