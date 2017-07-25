import numpy as np
np.random.seed(7)
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import MyCallbacks
import random
import HelpFunctions as hp
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def loss_nn_dense(args):
