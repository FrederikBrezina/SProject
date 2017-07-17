from multiprocessing import Pool
import numpy as np
import tensorflow as tf
from pathos.multiprocessing import ProcessingPool as Pool
get_bin = lambda x, n: format(x, 'b').zfill(n)
g =''+ get_bin(1,10)
print(g)

