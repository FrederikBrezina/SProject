import tensorflow as tf
for e in tf.train.summary_iterator('events.out.tfevents.1497690555.DESKTOP-92FU084'):
      print(e)