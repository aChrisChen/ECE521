import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Part 1
#                                                                                X: N1 * D
#                                                                                Z: N2 * D
#                                                             tf.expand_dims(X, 2): N1 * D * 1
#                                                                  tf.transpose(Z): D * N2
#                                               tf.expand_dims(tf.transpose(Z), 0): 1 * D * N2
#                        tf.expand_dims(X, 2) - tf.expand_dims(tf.transpose(Z), 0): N1 * D * N2
# tf.reduce_sum((tf.expand_dims(X, 2) - tf.expand_dims(tf.transpose(Z), 0))**2, 1): N1 * N2
def dist(X, Z):
    return tf.reduce_sum((tf.expand_dims(X, 2) - tf.expand_dims(tf.transpose(Z), 0))**2, 1)

# sess = tf.Session()
# N1 = 2
# N2 = 3
# D = 2
# matrix1 = tf.random_normal([N1, D], mean=0, stddev=1)
# matrix2 = tf.random_normal([N2, D], mean=0, stddev=1)

# m1 = sess.run(matrix1)
# m2 = sess.run(matrix2)
# res = sess.run(dist(m1, m2))
# for i, x in enumerate(m1):
#     for j, z in enumerate(m2):
#         if np.sum(np.square(x - z)) != res[i,j]: print("Error:", res[i, j] - np.sum(np.square(x - z)))