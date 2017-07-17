import tensorflow as tf
import numpy as np
f = [4 , 3, 2 ,1 ,0]
g = f[0:4]
h = f[1:5]

k = tf.subtract(g,h) + tf.subtract(h,g)
ses = tf.Session()
print(ses.run(k))