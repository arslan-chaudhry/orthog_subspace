# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf

def svb(conv, inp_shape, epsilon=5e-2):
  conv_tr = tf.cast(tf.transpose(conv, perm=[2, 3, 0, 1]), tf.complex64)
  conv_shape = conv.get_shape().as_list()
  padding = tf.constant([[0, 0], [0, 0],
                         [0, inp_shape[0] - conv_shape[0]],
                         [0, inp_shape[1] - conv_shape[1]]])
  transform_coeff = tf.fft2d(tf.pad(conv_tr, padding))
  D, U, V = tf.svd(tf.transpose(transform_coeff, perm = [2, 3, 0, 1]))
  #norm = tf.reduce_max(D)
  #D_clipped = tf.cast(tf.minimum(D, clip_to), tf.complex64)
  D_clipped = tf.cast(tf.clip_by_value(D, 1/(1+epsilon), 1+epsilon), tf.complex64)
  clipped_coeff = tf.matmul(U, tf.matmul(tf.linalg.diag(D_clipped),
                                         V, adjoint_b=True))
  clipped_conv_padded = tf.real(tf.ifft2d(
      tf.transpose(clipped_coeff, perm=[2, 3, 0, 1])))
  return tf.slice(tf.transpose(clipped_conv_padded, perm=[2, 3, 0, 1]),
                  [0] * len(conv_shape), conv_shape)
  #return tf.slice(tf.transpose(clipped_conv_padded, perm=[2, 3, 0, 1]),
   #               [0] * len(conv_shape), conv_shape), norm
