import tensorflow as tf
# Creates a graph.
get_bin = lambda x, n: format(x, 'b').zfill(n)
print(int(get_bin(10,10)[7:],2) + int(get_bin(10,10),2) )