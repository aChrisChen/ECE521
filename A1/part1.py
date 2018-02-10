import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
@parameters: X(N1 x D), Z(N2 x D)
@return: (N1 x N2)
'''
def dist(X, Z):
    return tf.reduce_sum((tf.expand_dims(X, 2) - tf.expand_dims(tf.transpose(Z), 0))**2, 1)