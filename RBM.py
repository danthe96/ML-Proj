import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import pandas as pd


"""
    This file contains the TF implementation of the Restricted Boltzman Machine
"""


def run_ebm(x, W, bx, bh, layer_shape):
    # Returns energy cost
    h = [None] * len(layer_shape)
    h[0] = x

    for i in range(1, len(layer_shape)):
      h[i] = tf.sigmoid(tf.matmul(tf.transpose(W[i]), h[i - 1]) + bh[i])

    encode = 0.5 * tf.reduce_sum(tf.square(x - bx[0])) + tf.reduce_sum(h[-1])

    result = tf.ones((layer_shape[-1], 1))
    for i in range(len(layer_shape) - 2, -1, -1):
      result = tf.sigmoid(tf.matmul(W[i + 1], h[i + 1]) + bx[i]) * tf.matmul(W[i + 1], result)

    reconstruct = result + bx[0]
    cost = tf.reduce_sum(tf.square(x - reconstruct))
    return cost
