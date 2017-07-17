import dill
import os

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


if __name__ == "__main__":
    def getHessian():
        import tensorflow as tf


        import numpy as np

        # Each time getHessian is called, we create a new graph so that the default graph (which exists a priori) won't be filled with old ops.
        dim = 3
        g = tf.Graph()
        with g.as_default():
            # First create placeholders for inputs: A, b, and c.
            A = tf.placeholder(tf.float32, shape=[dim, dim])
            b = tf.placeholder(tf.float32, shape=[dim, 1])
            c = tf.placeholder(tf.float32, shape=[1])
            # Define our variable
            x = tf.Variable(np.float32(np.repeat(1, dim).reshape(dim, 1)))
            # Construct the computational graph for quadratic function: f(x) = 1/2 * x^t A x + b^t x + c
            fx = 0.5 * tf.matmul(tf.matmul(tf.transpose(x), A), x) + tf.matmul(tf.transpose(b), x) + c

            # Get gradients of fx with repect to x
            dfx = tf.gradients(fx, x)[0]

            init_op = tf.initialize_all_variables()

            with tf.Session() as sess:
                sess.run(init_op)
                # We need to feed actual values into the computational graph that we created above.
                feed_dict = {A: np.float32(np.repeat(2, dim * dim).reshape(dim, dim)),
                             b: np.float32(np.repeat(3, dim).reshape(dim, 1)), c: [1]}
                # sess.run() executes the graph. Here, "hess" will be calculated with the values in "feed_dict".
                print(sess.run(fx, feed_dict))


    from multiprocessing import Pool
    pool = Pool(processes=5)

    jobs = []
    for i in range(10):
        job = apply_async(pool, getHessian, ())
        jobs.append(job)

    for job in jobs:
        print(job.get())
    print()