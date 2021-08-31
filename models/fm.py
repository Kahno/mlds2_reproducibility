import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, Flatten, Dropout, Softmax,
                                     Add, Activation, Concatenate, Reshape, BatchNormalization)
from tensorflow.keras.regularizers import L1L2, L2, L1
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers.experimental.preprocessing import Hashing
from tensorflow.keras.backend import pow, dot, mean, ndim, sum
from tensorflow.keras.metrics import AUC

from tensorflow_addons.optimizers import LazyAdam


def FM(num_feats, num_bins, num_factors):
    
    inputs = Input((num_feats,), dtype=tf.int32)    
    hashs = Hashing(num_bins=num_bins)(inputs)
    
    # Feature interaction embeddings
    int_embs = Embedding(num_bins, num_factors, input_length=num_feats)(hashs)
    
    # Linear feature embeddings
    lin_embs = Embedding(num_bins, 1, input_length=num_feats)(hashs)
    
    # Bias term
    bias_term = tf.Variable(GlorotUniform()(shape=(), dtype=tf.float32), True)
    
    # Linear term
    linear_term = tf.math.reduce_sum(lin_embs, axis=1)
    
    # Interaction term for classic FM
    a = pow(tf.math.reduce_sum(int_embs, axis=1), 2)
    b = tf.math.reduce_sum(pow(int_embs, 2), axis=1)
    interaction_term = sum(a - b, 1, keepdims=True) * 0.5
    
    # Final model output
    outputs = Activation("sigmoid")(bias_term + linear_term + interaction_term)
    
    model = Model(inputs=inputs, outputs=outputs, name=f"FM{num_factors}_{num_bins}")
    model.compile(loss="binary_crossentropy", optimizer=LazyAdam(0.003), metrics=[AUC()])
    
    return model