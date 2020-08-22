# Copyright 2018 ComputerGraphics Tuebingen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch
# Modifications copyright (C) 2013 <Technical University of Munich/Juan Du>
"""Tensorflow op performing flex convolution operation."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from tensorflow.python.framework import ops
from tensorflow.contrib.util import loader

__all__ = []


def load_op(name, has_grad=False, public=False):
    global __all__
    path = os.path.join(os.path.dirname(__file__), '%s_op.so' % name)
    if os.path.isfile(path):
        _module = loader.load_op_library(path)
        if has_grad:
            if public:
                __all__.append('%s' % name)
                __all__.append('%s_grad' % name)
            return getattr(_module, '%s' % name), getattr(_module, '%s_grad' % name)
        else:
            if public:
                __all__.append('%s' % name)
            return getattr(_module, '%s' % name)
    else:
        print('[WARNING]: %s does not exists' % name)


knn_bruteforce = load_op('knn_bruteforce', has_grad=False, public=True)

_flex_conv, _flex_conv_grad = load_op(
    'flex_conv', has_grad=True, public=False)
_flex_pool, _flex_pool_grad = load_op(
    'flex_pool', has_grad=True, public=False)
_flex_deconv, _flex_deconv_grad = load_op(
    'flex_deconv', has_grad=True, public=False)
_conv_pointset, _conv_pointset_grad = load_op(
    'conv_pointset', has_grad=True, public=False)


# pylint: disable=redefined-builtin
def flex_convolution(features,
                     position,
                     neighborhood,
                     theta,
                     bias,
                     name=None):
    """Flex-Convolution computation.

    Computes a convolution over arbitrary neighborhoods with elements of
    arbitrary positions:

      output(c', l) = sum_{c} sum_{l'}  w(c, l, l') * f(c, l')

    Args:
      features: A `Tensor` of the format [B, Din, N].
      position: A `Tensor` of the format [B, Dp, N].
      neighborhood: A `Tensor` of the format [B, K, N] (tf.int32).
      theta: A `Tensor` of the format [1, Dp, Din, Dout].
      bias: A `Tensor` of the format [Din, Dout].
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the format [B, Dout, N].
    """

    with ops.name_scope(name, "flex_convolution"):
        return _flex_conv(features, theta, bias, neighborhood, position)


__all__.append('flex_convolution')


@ops.RegisterGradient("FlexConv")
def _FlexConvGrad(op, *grads):  # noqa
    features = ops.convert_to_tensor(op.inputs[0])
    theta = ops.convert_to_tensor(op.inputs[1])
    bias = ops.convert_to_tensor(op.inputs[2])
    neighborhood = ops.convert_to_tensor(op.inputs[3], dtype=tf.int32)
    positions = ops.convert_to_tensor(op.inputs[4])
    topdiff = ops.convert_to_tensor(grads[0])

    df, dt, db = _flex_conv_grad(
        features, theta, bias, neighborhood, positions, topdiff)

    df = ops.convert_to_tensor(df, name='gradient_features')
    dt = ops.convert_to_tensor(dt, name='gradient_theta')
    db = ops.convert_to_tensor(db, name='gradient_bias')

    return [df, dt, db, None, None]


# pylint: disable=redefined-builtin
def flex_pooling(features,
                 neighborhood,
                 name=None):
    """Flex-Pooling computation.

    Computes a pooling over arbitrary neighborhoods:

      output(n) = max_l'  f(l')

    Args:
      features: A `Tensor` of the format [B, D, N].
      neighborhood: A `Tensor` of the format [B, K, N] (tf.int32).
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the format [B, D, N] containing the max values.
      A `Tensor` of the format [B, D, N] containing the max indicies.
    """

    with ops.name_scope(name, "flex_pooling"):
        return _flex_pool(features, neighborhood)


__all__.append('flex_pooling')


@ops.RegisterGradient("FlexPool")
def _FlexPoolGrad(op, *grads):  # noqa
    features = ops.convert_to_tensor(op.inputs[0])
    neighborhood = ops.convert_to_tensor(op.inputs[1])
    argmax = ops.convert_to_tensor(op.outputs[1])
    topdiff = ops.convert_to_tensor(grads[0])

    df = _flex_pool_grad(features, neighborhood, topdiff, argmax)
    df = ops.convert_to_tensor(df, name='gradient_features')

    return [df, None]


# pylint: disable=redefined-builtin
def flex_convolution_transpose(features,
                               position,
                               neighborhood,
                               theta,
                               bias,
                               name=None):
    """Flex-Convolution computation.

    Computes a tranposed convolution over arbitrary neighborhoods with elements of
    arbitrary positions.

    Args:
      features: A `Tensor` of the format [B, Din, N].
      position: A `Tensor` of the format [B, Dp, N].
      neighborhood: A `Tensor` of the format [B, K, N] (tf.int32).
      theta: A `Tensor` of the format [1, Dp, Din, Dout].
      bias: A `Tensor` of the format [Din, Dout].
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the format [B, Dout, N].
    """

    with ops.name_scope(name, "flex_convolution_transpose"):
        return _flex_deconv(features, theta, bias, neighborhood, position)


__all__.append('_flex_deconv')


@ops.RegisterGradient("FlexDeconv")
def _FlexDeconvGrad(op, *grads):  # noqa
    features = ops.convert_to_tensor(op.inputs[0])
    theta = ops.convert_to_tensor(op.inputs[1])
    bias = ops.convert_to_tensor(op.inputs[2])
    neighborhood = ops.convert_to_tensor(op.inputs[3], dtype=tf.int32)
    positions = ops.convert_to_tensor(op.inputs[4])
    topdiff = ops.convert_to_tensor(grads[0])

    df, dt, db = _flex_deconv_grad(
        features, theta, bias, neighborhood, positions, topdiff)

    df = ops.convert_to_tensor(df, name='gradient_features')
    dt = ops.convert_to_tensor(dt, name='gradient_theta')
    db = ops.convert_to_tensor(db, name='gradient_bias')

    return [df, dt, db, None, None]



def convolution_pointset(features,
                         neighborhood,
                         theta,
                         bias,
                         name=None):
    """one-by-one convolution on pointset computation.


    Args:
      features: A `Tensor` of the format [B, Din, N].
      neighborhood: A `Tensor` of the format [B, K, N] (tf.int32).
      theta: A `Tensor` of the format [1,  Din, Dout].
      bias: A `Tensor` of the format [Dout].
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the format [B, Dout, N].
    """

    with ops.name_scope(name, "convolution1d_pointset"):
        return _conv_pointset(features, theta, bias, neighborhood)


__all__.append('convolution_pointset')


@ops.RegisterGradient("ConvPointset")
def _ConvPointsetGrad(op, *grads):  # noqa
    features = ops.convert_to_tensor(op.inputs[0])
    theta = ops.convert_to_tensor(op.inputs[1])
    bias = ops.convert_to_tensor(op.inputs[2])
    neighborhood = ops.convert_to_tensor(op.inputs[3], dtype=tf.int32)
    topdiff = ops.convert_to_tensor(grads[0])

    df, dt, db = _conv_pointset_grad(
        features, theta, bias, neighborhood, topdiff)

    df = ops.convert_to_tensor(df, name='gradient_features')
    dt = ops.convert_to_tensor(dt, name='gradient_theta')
    db = ops.convert_to_tensor(db, name='gradient_bias')

    return [df, dt, db, None]
