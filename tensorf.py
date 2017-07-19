import tensorflow as tf
import numpy as np

f = np.random.rand(3,3)
h = np.random.rand(3,3)


f = f[:,0]
h = h[:,0]
o  = [0,0]
o[0] = tf.subtract(f,h) + tf.subtract(h,f)

o[1] = tf.subtract(f,h)
ses = tf.Session()
print(ses.run(o))