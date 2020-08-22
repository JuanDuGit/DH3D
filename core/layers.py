# -*- coding: utf-8 -*-

# Copyright 2017 ComputerGraphics Tuebingen. All Rights Reserved.
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
# ==============================================================================
# Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch
# Modifications copyright (C) 2013 <Technical University of Munich/Juan Du>


import tensorflow as tf

import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from user_ops import flex_convolution as _flex_convolution
from user_ops import flex_pooling as _flex_pooling
from user_ops import knn_bruteforce as _knn_bruteforce
from user_ops import flex_convolution_transpose as _flex_convolution_transpose
from user_ops import convolution_pointset as _convolution_pointset

from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.layers.base import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops

all = ['FlexPooling', 'FlexConvolution', 'FlexConvolutionTranspose', 'ConvolutionPointset'
       'flex_pooling', 'flex_convolution', 'flex_convolution_transpose', 'convolution_pointset',
       'knn_bruteforce', 'KnnBruteforce']


def _remove_dim(x, axis=2):
    return tf.squeeze(x, axis=axis)


class KnnBruteforce(Layer):
    """knn bruteforce layer.
    This layer performs a nearest neighbor lookup for a batch of given positions.

    Arguments:
        K: size of each neighborhood
        data_format: A string, one of `simple` (default) or `expanded`.
          If `simple` the shapes are [B, Din, N], when `expanded` the shapes
          are assumed to be [B, Din, 1, N] to match `channels_first` in trad
          convolutions.
        name: A string, the name of the layer.

    Inputs:
        positions: A `Tensor` of the format [B, Dp, (1), N].

    Outputs:
        neighborhoods: A `Tensor` of the format [B, K, (1), N] (tf.int32).

    """

    def __init__(self,
                 k,
                 data_format='simple',
                 name=None):

        super(KnnBruteforce, self).__init__(name=name)
        assert k > 0
        assert data_format in ['simple', 'expanded']
        self.k = k
        self.data_format = data_format

    def compute_output_shape(self, input_shapes):
        output_shape = input_shapes[0]
        output_shape[1] = self.k
        return output_shape

    def call(self, inputs):

        positions = ops.convert_to_tensor(inputs, dtype=self.dtype)
        if self.data_format == 'expanded':
            positions = _remove_dim(positions, 2)

        NN, distances = _knn_bruteforce(positions, k=self.k)
        NN = tf.transpose(NN, [0, 2, 1])
        distances = tf.transpose(distances, [0, 2, 1])

        if self.data_format == 'expanded':
            NN = tf.expand_dims(NN, axis=2)

        return NN, distances


def knn_bruteforce(positions,
                   k,
                   data_format='simple',
                   name=None):
    layer = KnnBruteforce(k, data_format=data_format, name=name)

    return layer.apply(positions)


class FlexPooling(Layer):
    """flex pooling layer.

    This layer performs a max-pooling operation over elements in arbitrary
    neighborhoods. When `data_format` is 'simple', the input shape should
    have rank 3, otherwise rank 4 and dimension 2 should be 1.

    Remarks:
        In contrast to traditional pooling, this operation has no option for
        sub-sampling.

    Arguments:
        features: A `Tensor` of the format [B, Din, (1), N].
        neighborhoods: A `Tensor` of the format [B, K, (1), N] (tf.int32).
        data_format: A string, one of `simple` (default) or `expanded`.
          If `simple` the shapes are [B, Din, N], when `expanded` the shapes
          are assumed to be [B, Din, 1, N] to match `channels_first` in trad
          convolutions.
        name: A string, the name of the layer.

    Inputs:
        features: A `Tensor` of the format[B, Din, (1), N].
        neighborhoods: A `Tensor` of the format [B, K, (1), N] (tf.int32).

    Outputs:
        features: A `Tensor` of the format[B, Din, (1), N].

    """

    def __init__(self,
                 data_format='simple',
                 name=None):

        super(FlexPooling, self).__init__(name=name)
        assert data_format in ['simple', 'expanded']
        self.data_format = data_format

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A flexconv layer should be called '
                             'on a list of inputs.')

        features = ops.convert_to_tensor(inputs[0], dtype=self.dtype)
        neighborhoods = ops.convert_to_tensor(inputs[1], dtype=tf.int32)

        if self.data_format == 'expanded':
            features = _remove_dim(features, 2)
            neighborhoods = _remove_dim(neighborhoods, 2)

        y, _ = _flex_pooling(features, neighborhoods)

        if self.data_format == 'expanded':
            y = tf.expand_dims(y, axis=2)

        return y


def flex_pooling(features,
                 neighborhoods,
                 data_format='simple',
                 name=None):
    layer = FlexPooling(data_format=data_format, name=name)
    return layer.apply([features, neighborhoods])


class FlexConvolution(Layer):
    """flex convolution layer.

    This layer convolves elements in arbitrary neighborhoods with a kernel to
    produce a tensor of outputs.
    If `use_feature_bias` is True (and a `features_bias_initializer` is provided),
    a bias vector is created and added to the outputs after te convolution.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.
    When `data_format` is 'simple', the input shape should have rank 3,
    otherwise rank 4 and dimension 2 should be 1.

    Remarks:
        In contrast to traditional convolutions, this operation has two
        bias terms:
          - bias term when dynamically computing the weight [Din, Dout]
          - bias term which is added tot the features [Dout]

    Arguments:
        filters: Integer, the dimensionality of the output space (i.e. the number
          of filters in the convolution).
        activation: Activation function. Set it to None to maintain a
          linear activation.
        kernel_initializer: An initializer for the convolution kernel.
        position_bias_initializer: An initializer for the bias vector within
          the convolution. If None, the default initializer will be used.
        features_bias_initializer: An initializer for the bias vector after
          the convolution. If None, the default initializer will be used.
        use_feature_bias: Boolean, whether the layer uses a bias.
        data_format: A string, one of `simple` (default) or `expanded`.
          If `simple` the shapes are [B, Din, N], when `expanded` the shapes
          are assumed to be [B, Din, 1, N] to match `channels_first` in trad
          convolutions.
        trainable: Boolean, if `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: A string, the name of the layer.

    Inputs:
        features: A `Tensor` of the format[B, Din, (1), N].
        positions: A `Tensor` of the format[B, Dp, (1), N].
        neighborhoods: A `Tensor` of the format [B, K, (1), N] (tf.int32).

    Outputs:
        features: A `Tensor` of the format[B, Dout, (1), N].

    """

    def __init__(self,
                 filters,
                 activation=None,
                 kernel_initializer=None,
                 position_bias_initializer=tf.zeros_initializer(),
                 features_bias_initializer=tf.zeros_initializer(),
                 use_feature_bias=True,
                 data_format='simple',
                 trainable=True,
                 name=None):

        super(FlexConvolution, self).__init__(trainable=trainable,
                                              name=name)
        assert data_format in ['simple', 'expanded']
        self.filters = int(filters)
        self.activation = activations.get(activation)
        self.use_feature_bias = use_feature_bias
        self.data_format = data_format
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.position_bias_initializer = initializers.get(position_bias_initializer)
        self.features_bias_initializer = initializers.get(features_bias_initializer)

    def compute_output_shape(self, input_shapes):
        assert isinstance(input_shapes, list)
        output_shape = input_shapes[0]
        output_shape[1] = self.filters
        return output_shape

    def build(self, input_shapes):
        # assert isinstance(input_shapes, list)
        #
        if self.data_format == 'expanded':
            _, Din, _, N = input_shapes[0].as_list()
        else:
            _, Din, N = input_shapes[0].as_list()
        #
        Dp = input_shapes[1].as_list()[1]
        Dout = self.filters

        # _, Din, N = input_shapes[0].as_list()

        self.position_theta = self.add_weight(
            'position_theta',
            shape=[Dp, Din, Dout],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        self.position_bias = self.add_weight(
            'position_bias',
            shape=[Din, Dout],
            initializer=self.position_bias_initializer,
            dtype=self.dtype,
            trainable=True)

        if self.use_feature_bias:
            self.feature_bias = self.add_weight(
                'feature_bias',
                shape=[Dout, 1],
                initializer=self.features_bias_initializer,
                dtype=self.dtype,
                trainable=True)
        else:
            self.feature_bias = None
        self.built = True

    def internal_call(self,
                      features,
                      positions,
                      neighborhoods,
                      theta,
                      bias):
        return _flex_convolution(features, positions, neighborhoods,
                                 theta, bias)

    def call(self, inputs):
        """
        Args:
            inputs[0] (tf.tensor): A `Tensor` of the format [B, Din, (1), N]
              describing the incoming features.
            inputs[1] (tf.tensor): A `Tensor` of the format [B, Dp, (1), N]
              containing the position of the incoming features.
            inputs[2] (tf.tensor): A `Tensor` of the format [B, K, (1), N]
              containting the neighborhood structure (tf.int32).

        Returns:
            tf.tensor: A `Tensor` of the format [B, Dout, (1), N] describing
              the outgoing features.

        """
        if not isinstance(inputs, list):
            raise ValueError('A flexconv layer should be called '
                             'on a list of inputs.')

        features = ops.convert_to_tensor(inputs[0], dtype=self.dtype)
        positions = ops.convert_to_tensor(inputs[1], dtype=self.dtype)
        neighborhoods = ops.convert_to_tensor(inputs[2], dtype=tf.int32)

        if self.data_format == 'expanded':
            features = _remove_dim(features, 2)
            positions = _remove_dim(positions, 2)
            neighborhoods = _remove_dim(neighborhoods, 2)

        y = self.internal_call(features, positions, neighborhoods,
                               self.position_theta, self.position_bias)

        if self.use_feature_bias:
            y = tf.add(y, self.feature_bias)

        if self.activation is not None:
            y = self.activation(y)

        if self.data_format == 'expanded':
            y = tf.expand_dims(y, axis=2)

        return y


class Flex_Avg(Layer):

    def __init__(self,
                 filters,
                 activation=None,
                 kernel_initializer=tf.zeros_initializer(),
                 data_format='simple',
                 trainable=True,
                 name=None):

        super(Flex_Avg, self).__init__(trainable=trainable,
                                       name=name)
        assert data_format in ['simple', 'expanded']
        self.filters = int(filters)
        self.activation = activations.get(activation)
        self.data_format = data_format
        self.kernel_initializer = initializers.get(kernel_initializer)

    def compute_output_shape(self, input_shapes):
        assert isinstance(input_shapes, list)
        output_shape = input_shapes[0]
        output_shape[1] = self.filters
        return output_shape

    def build(self, input_shapes):
        # assert isinstance(input_shapes, list)
        #
        if self.data_format == 'expanded':
            _, Din, _, N = input_shapes[0].as_list()
        else:
            _, Din, N = input_shapes[0].as_list()
        #
        Dp = input_shapes[1].as_list()[1]
        Dout = self.filters

        # _, Din, N = input_shapes[0].as_list()

        self.position_theta = self.add_weight(
            'position_theta',
            shape=[Dp, Din, Dout],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=False)

        self.position_bias = tf.eye(Dout)

        self.built = True

    def internal_call(self,
                      features,
                      positions,
                      neighborhoods,
                      theta,
                      bias):
        return _flex_convolution(features, positions, neighborhoods,
                                 theta, bias)

    def call(self, inputs):
        """
        Args:
            inputs[0] (tf.tensor): A `Tensor` of the format [B, Din, (1), N]
              describing the incoming features.
            inputs[1] (tf.tensor): A `Tensor` of the format [B, Dp, (1), N]
              containing the position of the incoming features.
            inputs[2] (tf.tensor): A `Tensor` of the format [B, K, (1), N]
              containting the neighborhood structure (tf.int32).

        Returns:
            tf.tensor: A `Tensor` of the format [B, Dout, (1), N] describing
              the outgoing features.

        """
        if not isinstance(inputs, list):
            raise ValueError('A flexconv layer should be called '
                             'on a list of inputs.')

        features = ops.convert_to_tensor(inputs[0], dtype=self.dtype)
        positions = ops.convert_to_tensor(inputs[1], dtype=self.dtype)
        neighborhoods = ops.convert_to_tensor(inputs[2], dtype=tf.int32)

        if self.data_format == 'expanded':
            features = _remove_dim(features, 2)
            positions = _remove_dim(positions, 2)
            neighborhoods = _remove_dim(neighborhoods, 2)

        y = self.internal_call(features, positions, neighborhoods,
                               self.position_theta, self.position_bias)

        if self.activation is not None:
            y = self.activation(y)

        if self.data_format == 'expanded':
            y = tf.expand_dims(y, axis=2)

        return y


def flex_convolution(features,
                     positions,
                     neighborhoods,
                     filters,
                     activation=None,
                     kernel_initializer=None,
                     position_bias_initializer=tf.zeros_initializer(),
                     features_bias_initializer=tf.zeros_initializer(),
                     use_feature_bias=True,
                     data_format='simple',
                     trainable=True,
                     name=None):
    layer = FlexConvolution(filters,
                            activation=activation,
                            kernel_initializer=kernel_initializer,
                            position_bias_initializer=position_bias_initializer,
                            features_bias_initializer=features_bias_initializer,
                            use_feature_bias=use_feature_bias,
                            data_format=data_format,
                            trainable=trainable,
                            name=name)

    return layer.apply([features, positions, neighborhoods])


def flex_avg(features,
             positions,
             neighborhoods,
             filters,
             activation=None,
             kernel_initializer=tf.zeros_initializer(),
             data_format='simple',
             trainable=True,
             name=None):
    layer = Flex_Avg(filters,
                     activation=activation,
                     kernel_initializer=kernel_initializer,
                     data_format=data_format,
                     trainable=trainable,
                     name=name)

    return layer.apply([features, positions, neighborhoods])


class FlexConvolutionTranspose(FlexConvolution):
    """flex convolution-transpose layer.

    This layer applies a transpose convolution to elements in arbitrary
    neighborhoods.
    If `use_feature_bias` is True (and a `features_bias_initializer` is provided),
    a bias vector is created and added to the outputs after te convolution.
    Finally, if `activation` is not `None`, it is applied to the outputs as well.
    When `data_format` is 'simple', the input shape should have rank 3,
    otherwise rank 4 and dimension 2 should be 1.

    Remarks:
        In contrast to traditional transposed convolutions, this operation has two
        bias terms:
          - bias term when dynamically computing the weight [Din, Dout]
          - bias term which is added tot the features [Dout]

    Arguments:
        filters: Integer, the dimensionality of the output space (i.e. the number
          of filters in the convolution).
        activation: Activation function. Set it to None to maintain a
          linear activation.
        kernel_initializer: An initializer for the convolution kernel.
        position_bias_initializer: An initializer for the bias vector within
          the convolution. If None, the default initializer will be used.
        features_bias_initializer: An initializer for the bias vector after
          the convolution. If None, the default initializer will be used.
        use_feature_bias: Boolean, whether the layer uses a bias.
        data_format: A string, one of `simple` (default) or `expanded`.
          If `simple` the shapes are [B, Din, N], when `expanded` the shapes
          are assumed to be [B, Din, 1, N] to match `channels_first` in trad
          convolutions.
        trainable: Boolean, if `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: A string, the name of the layer.

    Inputs:
        features: A `Tensor` of the format [B, Din, (1), N].
        positions: A `Tensor` of the format [B, Dp, (1), N].
        neighborhoods: A `Tensor` of the format [B, K, (1), N] (tf.int32).

    Outputs:
        features: A `Tensor` of the format[B, Dout, (1), N].

    """

    def internal_call(self,
                      features,
                      positions,
                      neighborhoods,
                      theta,
                      bias):
        return _flex_convolution_transpose(features, positions, neighborhoods,
                                           theta, bias)


def flex_convolution_transpose(features,
                               positions,
                               neighborhoods,
                               filters,
                               activation=None,
                               kernel_initializer=None,
                               position_bias_initializer=tf.zeros_initializer(),
                               features_bias_initializer=tf.zeros_initializer(),
                               use_feature_bias=True,
                               data_format='simple',
                               trainable=True,
                               name=None):
    layer = FlexConvolutionTranspose(filters,
                                     activation=activation,
                                     kernel_initializer=kernel_initializer,
                                     position_bias_initializer=position_bias_initializer,
                                     features_bias_initializer=features_bias_initializer,
                                     use_feature_bias=use_feature_bias,
                                     data_format=data_format,
                                     trainable=trainable,
                                     name=name)

    return layer.apply([features, positions, neighborhoods])


class ConvolutionPointset(Layer):
    """ConvolutionPointset layer.

    todo

    """

    def __init__(self,
                 filters,
                 activation=None,
                 kernel_initializer=None,
                 position_bias_initializer=tf.zeros_initializer(),
                 features_bias_initializer=tf.zeros_initializer(),
                 use_feature_bias=True,
                 data_format='simple',
                 trainable=True,
                 name=None):

        super(ConvolutionPointset, self).__init__(trainable=trainable,
                                                  name=name)
        assert data_format in ['simple', 'expanded']
        self.filters = int(filters)
        self.activation = activations.get(activation)
        self.use_feature_bias = use_feature_bias
        self.data_format = data_format
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.position_bias_initializer = initializers.get(position_bias_initializer)
        self.features_bias_initializer = initializers.get(features_bias_initializer)

    def compute_output_shape(self, input_shapes):
        assert isinstance(input_shapes, list)
        output_shape = input_shapes[0]
        output_shape[1] = self.filters
        return output_shape

    def build(self, input_shapes):
        # assert isinstance(input_shapes, list)
        #
        if self.data_format == 'expanded':
            _, Din, _, N = input_shapes[0].as_list()
        else:
            _, Din, N = input_shapes[0].as_list()
        #

        Dout = self.filters

        # _, Din, N = input_shapes[0].as_list()

        self.position_theta = self.add_weight(
            'position_theta',
            shape=[Din, Dout],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)

        self.position_bias = self.add_weight(
            'position_bias',
            shape=[Dout],
            initializer=self.position_bias_initializer,
            dtype=self.dtype,
            trainable=True)

        if self.use_feature_bias:
            self.feature_bias = self.add_weight(
                'feature_bias',
                shape=[Dout, 1],
                initializer=self.features_bias_initializer,
                dtype=self.dtype,
                trainable=True)
        else:
            self.feature_bias = None
        self.built = True

    def internal_call(self,
                      features,
                      neighborhoods,
                      theta,
                      bias):
        return _convolution_pointset(features, neighborhoods,
                                     theta, bias)

    def call(self, inputs):
        """
        Args:
            inputs[0] (tf.tensor): A `Tensor` of the format [B, Din, (1), N]
              describing the incoming features.
            inputs[1] (tf.tensor): A `Tensor` of the format [B, Dp, (1), N]
              containing the position of the incoming features.
            inputs[2] (tf.tensor): A `Tensor` of the format [B, K, (1), N]
              containting the neighborhood structure (tf.int32).

        Returns:
            tf.tensor: A `Tensor` of the format [B, Dout, (1), N] describing
              the outgoing features.

        """
        if not isinstance(inputs, list):
            raise ValueError('A convolution1d_pointset layer should be called '
                             'on a list of inputs.')

        features = ops.convert_to_tensor(inputs[0], dtype=self.dtype)
        neighborhoods = ops.convert_to_tensor(inputs[1], dtype=tf.int32)

        if self.data_format == 'expanded':
            features = _remove_dim(features, 2)
            neighborhoods = _remove_dim(neighborhoods, 2)

        y = self.internal_call(features, neighborhoods,
                               self.position_theta, self.position_bias)

        if self.use_feature_bias:
            y = tf.add(y, self.feature_bias)

        if self.activation is not None:
            y = self.activation(y)

        if self.data_format == 'expanded':
            y = tf.expand_dims(y, axis=2)

        return y


def convolution_pointset(features,
                         neighborhoods,
                         filters,
                         activation=None,
                         kernel_initializer=None,
                         position_bias_initializer=tf.zeros_initializer(),
                         features_bias_initializer=tf.zeros_initializer(),
                         use_feature_bias=False,
                         data_format='simple',
                         trainable=True,
                         name=None):
    layer = ConvolutionPointset(filters,
                                activation=activation,
                                kernel_initializer=kernel_initializer,
                                position_bias_initializer=position_bias_initializer,
                                features_bias_initializer=features_bias_initializer,
                                use_feature_bias=use_feature_bias,
                                data_format=data_format,
                                trainable=trainable,
                                name=name)

    return layer.apply([features, neighborhoods])
