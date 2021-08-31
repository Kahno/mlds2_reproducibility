import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, Flatten, Dropout, Softmax,
                                     Add, Activation, Concatenate, Reshape, BatchNormalization)
from tensorflow.keras.regularizers import L1L2, L2, L1
from tensorflow.keras.initializers import GlorotUniform, RandomNormal, Constant
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers.experimental.preprocessing import Hashing
from tensorflow.keras.backend import pow, dot, mean, ndim, sum
from tensorflow.keras.metrics import AUC

from tensorflow_addons.optimizers import LazyAdam


def AFN(num_feats, num_bins, num_factors, hidden_size, hidden_size_2=10, nlayers=0):
    
    inputs = Input((num_feats,), dtype=tf.int32)    
    hashs = Hashing(num_bins=num_bins)(inputs)
    
    # Feature interaction embeddings
    embs = Embedding(num_bins, num_factors, input_length=num_feats)(hashs)

    weights = tf.Variable(GlorotUniform()(shape=(num_feats, hidden_size), dtype=tf.float32), True)
    biases = tf.Variable(tf.zeros((1, 1, hidden_size)), trainable=True)
    
    # Ensure only positive embedding values
    x = tf.math.abs(embs)
    x = tf.transpose(x, perm=[0, 2, 1])
    
    # Avoid zero values due to numerical instability
    x = tf.clip_by_value(x, 1e-7, np.infty)
    
    # Perform logarithmic transformation
    x = tf.math.log(x)
    
    x = BatchNormalization()(x)
    
    x = tf.linalg.matmul(x, weights)
    x = tf.math.add(x, biases)
    
    # Transform back into original space
    x = tf.math.exp(x)
    
    x = BatchNormalization()(x)
    x = Flatten()(x)

    #x = Dropout(0.2)(x)
    for i in range(nlayers):
        x = Dense(hidden_size_2, kernel_initializer="glorot_uniform", activation="relu")(x)
        #x = Dropout(0.2)(x)
    
    interaction_term = Dense(1, kernel_initializer="glorot_uniform")(x)
    
    # Bias term
    bias_term = tf.Variable(GlorotUniform()(shape=(), dtype=tf.float32), True)
    
    # Linear feature embeddings
    lin_embs = Embedding(num_bins, 1, input_length=num_feats)(hashs)
    
    # Linear term
    linear_term = tf.math.reduce_sum(lin_embs, axis=1)

    # Final model output
    outputs = Activation("sigmoid")(interaction_term + linear_term + bias_term)

    model = Model(inputs=inputs, outputs=outputs, name="AFN")
    model.compile(loss="binary_crossentropy", optimizer=LazyAdam(0.003), metrics=[AUC()])
    
    return model