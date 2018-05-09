import tensorflow as tf
import numpy as np
import glob

from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

from RBM import run_ebm
import midi_manipulation


"""
    This file contains the TF implementation of the RNN-RBM, as well as the hyperparameters of the model
"""

n_hidden = 50  # The size of the RBM hidden layer
n_hidden_recurrent = 100  # The size of each RNN hidden layer


def rnnrbm(d):
    # This function builds the RNN-RBM and returns the parameters of the model
    n_visible = d
    ebm_layer_shape = [n_visible, 80, 80, n_hidden]

    x = tf.placeholder(tf.float32, [None, n_visible])  # The placeholder variable that holds our data
    lr = tf.placeholder(tf.float32)  # The learning rate. We set and change this value during training.

    size_bt = tf.shape(x)[0]  # the batch size

    # Here we set aside the space for each of the variables.
    # We intialize these variables when we load saved parameters in rnn_rbm_train.py or rnn_rbm_generate.py
    W = tf.Variable(tf.zeros([n_visible, n_hidden]), name="W")
    Wuh = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden]), name="Wuh")
    Wux = tf.Variable(tf.zeros([n_hidden_recurrent, n_visible]), name="Wux")
    Wxu = tf.Variable(tf.zeros([n_visible, n_hidden_recurrent]), name="Wxu")
    Wuu = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuu")
    bh = tf.Variable(tf.zeros([1, n_hidden]), name="bh")
    bx = tf.Variable(tf.zeros([1, n_visible]), name="bx")
    bu = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bu")
    u0 = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="u0")
    BH_t = tf.Variable(tf.zeros([1, n_hidden]), name="BH_t")
    BX_t = tf.Variable(tf.zeros([1, n_visible]), name="BX_t")

    def rnn_recurrence(u_tmin1, sl):
        # Iterate through the data in the batch and generate the values of the RNN hidden nodes
        sl = tf.reshape(sl, [1, n_visible])
        u_t = tf.nn.softplus(bu + tf.matmul(sl, Wxu) + tf.matmul(u_tmin1, Wuu))
        return u_t

    def visible_bias_recurrence(bx_t, u_tm1):
        # Iterate through the values of the RNN hidden nodes and generate the values of the visible bias vectors
        bx_t = tf.add(bx, tf.matmul(u_tm1, Wux))
        return bx_t

    def hidden_bias_recurrence(bh_t, u_tm1):
        # Iterate through the values of the RNN hidden nodes and generate the values of the hidden bias vectors
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))
        return bh_t

    # Reshape our bias matrices to be the same size as the batch.
    tf.assign(BH_t, tf.tile(BH_t, [size_bt, 1]))
    tf.assign(BX_t, tf.tile(BX_t, [size_bt, 1]))
    # Scan through the rnn and generate the value for each hidden node in the batch
    u_t = tf.scan(rnn_recurrence, x, initializer=u0)
    # Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
    BX_t = tf.reshape(tf.scan(visible_bias_recurrence, u_t, tf.zeros([1, n_visible], tf.float32)), [size_bt, n_visible])
    BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, u_t, tf.zeros([1, n_hidden], tf.float32)), [size_bt, n_hidden])

    # Get the free energy cost from each of the RBMs in the batch
    # TODO: Add EBM cost method
    cost = run_ebm(x, W, BX_t, BH_t, ebm_layer_shape)

    return x, cost, W, bh, bx, lr, Wuh, Wuv, Wvu, Wuu, bu, u0, ebm_layer_shape
