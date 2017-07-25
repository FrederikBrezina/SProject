import numpy as np
np.random.seed(7)
from keras.models import Model, Sequential

from keras.layers import Input, Dense, Conv2D
import MyCallbacks
import random
import HelpFunctions as hp
#import multi_gpu as mg

model = Sequential()
model.add(Dense(32, input_dim=784, activation='relu'))



