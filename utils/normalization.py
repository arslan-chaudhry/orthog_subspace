# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf

class OrthogNormalization(tf.keras.layers.Wrapper):

    def __init__(self, layer, **kwargs):
        super(OrthogNormalization, self).__init__(layer, **kwargs)
        self._track_checkpointable(layer, name='layer')
        #self._track_trackable(layer, name='layer')
        #self._init_critical_section = tf.CriticalSection(name='init_mutex')

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer

        if not hasattr(kernel_layer, 'kernel'):
            raise ValueError('`OrthogNormalization` must wrap a layer that'
                             ' contains a `kernel` for weights')

        kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel.shape[-1])

        self.v = kernel

        """
        self._initialized = self.add_weight(
            name='initialized',
            shape=None,
            initializer='zeros',
            dtype=tf.dtypes.bool,
            trainable=False)
        """
        
        V = tf.reshape(self.v, [-1, kernel.shape[-1]])
        Vc = V - tf.reduce_mean(V, axis=0, keepdims=True) # Zero center the vectors
        S = tf.matmul(tf.transpose(Vc), Vc)
        s, u, _ = tf.linalg.svd(S)
        D = tf.linalg.diag(tf.math.rsqrt(s))
        P = tf.linalg.matmul(tf.linalg.matmul(u, D), tf.transpose(u))
        
        # Replace kernel by VP.
        kernel = tf.reshape(tf.matmul(Vc, P), self.layer.kernel.shape)

        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        with tf.name_scope('compute_weights'):
            V = tf.reshape(self.v, [-1, self.layer.kernel.shape[-1]])
            Vc = V - tf.reduce_mean(V, axis=0, keepdims=True) # Zero center the vectors
            S = tf.matmul(tf.transpose(Vc), Vc)
            s, u, _ = tf.linalg.svd(S)
            D = tf.linalg.diag(tf.math.rsqrt(s))
            P = tf.linalg.matmul(tf.linalg.matmul(u, D), tf.transpose(u))
            # Replace kernel by VP.
            kernel = tf.reshape(tf.matmul(Vc, P), self.layer.kernel.shape)

            self.layer.kernel = kernel
            update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())


    def get_config(self):
        base_config = super(OrthogNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def remove(self):
        V = tf.reshape(self.v, [-1, kernel.shape[-1]])
        Vc = V - tf.reduce_mean(V, axis=0, keepdims=True) # Zero center the vectors
        S = tf.matmul(tf.transpose(Vc), Vc)
        s, u, _ = tf.linalg.svd(S)
        D = tf.linalg.diag(tf.math.rsqrt(s))
        P = tf.linalg.matmul(tf.linalg.matmul(u, D), tf.transpose(u))
        # Replace kernel by VP.
        kernel = tf.Variable(tf.reshape(tf.matmul(Vc, P), self.layer.kernel.shape), 
                             name='kernel')

        self.layer.kernel = kernel

        return self.layer
