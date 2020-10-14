# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf

def OWNNorm(W):
    """
    Implements the Orthogonalize Weight Normalization
    
    Args:
        W      [in X out], [h, w, in_channels, out_channels]
        
    Returns
        W = VP, where P = UD^{-1/2}U^T.
    
    """
    shape = W.get_shape().as_list()
    V = tf.reshape(W, [-1, shape[-1]])
    Vc = V - tf.reduce_mean(V, axis=0, keepdims=True) # Zero center the vectors
    S = tf.matmul(tf.transpose(Vc), Vc)
    s, u, _ = tf.linalg.svd(S)
    D = tf.linalg.diag(tf.math.rsqrt(s))
    P = tf.linalg.matmul(tf.linalg.matmul(u, D), tf.transpose(u))
    W_hat = tf.matmul(Vc, P)
    return tf.reshape(W_hat, shape)
    
