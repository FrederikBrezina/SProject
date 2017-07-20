import numpy as np
np.random.seed(7)
from keras.models import Model
from keras.layers import Input, Dense, Conv2D
import MyCallbacks
import random
import HelpFunctions as hp
#import multi_gpu as mg


a = Input(shape=(3, 32, 32))
b = Input(shape=(3, 64, 64))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# Only one input so far, the following will work:
print(conv.input_shape)


