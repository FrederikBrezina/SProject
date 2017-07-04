from keras.models import Model
from keras.layers import Input, Dense, LSTM, RepeatVector
import MyCallbacks
import HelpFunctions as hp
import random as rn
import numpy as np
from math import exp
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed

number_of_time_steps = 15
dimensionality_of_first_lstm = 10
number_of_context_layers = 3
model = Sequential()
model.add(TimeDistributed(LSTM(dimensionality_of_first_lstm), input_shape=(number_of_context_layers, number_of_time_steps,dimensionality_of_first_lstm)))
model.add(LSTM(200))
model.add(RepeatVector(number_of_time_steps))
model.add(LSTM(dimensionality_of_first_lstm, return_sequences=True))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=100, batch_size=32, validation_data=(x_test, y_test))
