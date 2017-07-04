# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import keras
from keras.utils import plot_model
import MyCallbacks

# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("new.txt", delimiter=" ")
# split into input (X) and output (Y) variables
X = dataset[0:-101,0:2]
Y = dataset[0:-101,2:5]
x_test=dataset[-101:-1,0:2]
y_test=dataset[-101:-1,2:5]

# create model
model = Sequential()
model.add(Dense(30, input_dim=2, activation='relu'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.add(Dense(3, activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=100, batch_size=10,validation_data=(x_test,y_test), callbacks=[MyCallbacks.AccHistory])
# evaluate the model
scores = model.evaluate(X, Y)
plot_model(model, to_file='model.png')
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
