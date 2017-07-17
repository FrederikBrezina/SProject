import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense

import matplotlib.pyplot as plt



dataset = np.loadtxt("units_of_first_layer.txt", delimiter=" ")


X = dataset[:, 0:1].tolist()
Y = dataset[:, 1:2].tolist()


# Types of layers used
c = ['relu', 'tanh', 'sigmoid']
num_of_ouputs = 1
input = Input(shape=(1,))
output = Dense(num_of_ouputs, activation='linear')(input)
model = Model(inputs=input, outputs=output)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


model.fit(X, Y, epochs=500, batch_size=10, validation_data=(),
          callbacks=[], shuffle=True)
Y_predict = model.predict(X)
Y_diff = []
for point in range(0,len(Y)):
    Y_diff.append(np.sqrt((Y_predict[point] - Y[point])**2))
plt.figure(1)
plt.plot(Y_diff)
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(Y)
plt.subplot(2,1,2)
plt.plot(Y_predict)
plt.show()
