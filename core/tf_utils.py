# Copyright (C) 2020 Juan Du (Technical University of Munich)
# For more information see <https://vision.in.tum.de/research/vslam/dh3d>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorpack import *
from tensorpack import logger
from tensorpack.tfutils.varreplace import freeze_variables
from contextlib import ExitStack, contextmanager

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from tf_ops.grouping.tf_grouping import group_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from layers import flex_convolution, flex_pooling, knn_bruteforce, flex_avg, convolution_pointset


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def flexconv_withBatchnorm(feats, points, nn, dout, name, ac_func=tf.nn.relu):
    '''
    :param feats: B, dim_in, N
    :param points: B, 3, N
    :param nn: [B, K, N],
    :param dout: dim_out
    :param name:
    :return: [B, dim_out, N]
    '''
    # -> B, dim_out, N
    x = flex_convolution(feats, points, nn, dout, activation=tf.identity, name=name)
    # -> B, dim_out, 1, N
    x = tf.expand_dims(x, 2)
    x = ac_func(BatchNorm('{}_bn'.format(name), x, data_format='NCHW'))
    # -> B, dim_out, N
    x = tf.squeeze(x, 2)
    return x


def convolution_pointset_withBatchnorm(points, nn, dout, name, ac_func=tf.nn.relu):
    '''
    :param feats: B, dim_in, N
    :param points: B, 3, N
    :param nn: [B, K, N],
    :param dout: dim_out
    :param name:
    :return: [B, dim_out, N]
    '''
    # -> B, dim_out, N
    x = convolution_pointset(points, nn, dout, activation=tf.identity, name=name)
    # -> B, dim_out, 1, N
    x = tf.expand_dims(x, 2)
    x = ac_func(BatchNorm('{}_bn'.format(name), x, data_format='NCHW'))
    # -> B, dim_out, N
    x = tf.squeeze(x, 2)
    return x


def subsample(points, feat, targetnum, kp_idx):
    if kp_idx is not None:
        kp_indices = kp_idx
    else:
        kp_indices = farthest_point_sample(targetnum, points)
        kp_indices = tf.expand_dims(kp_indices, 2)
    feat_sampled = group_point(feat, kp_indices)
    feat_sampled = tf.squeeze(feat_sampled, 2)
    xyz_sampled = group_point(points, kp_indices)
    xyz_sampled = tf.squeeze(xyz_sampled, 2)
    return xyz_sampled, feat_sampled, kp_indices


def feature_conv1d_1(feat, dim, name, c_last=True, ac_func=BNReLU):
    # [B,N,K] --> [B,N,K2]
    if not c_last:
        feat = tf.transpose(feat, perm=[0, 2, 1])
    feat = tf.expand_dims(feat, 2)
    with tf.variable_scope(name):
        newfeat = Conv2D('tfconv0', feat, dim,  kernel_shape=1, padding='VALID', activation=ac_func)
        newfeat = tf.squeeze(newfeat, 2)
    if not c_last:
        newfeat = tf.transpose(newfeat, perm=[0, 2, 1])
    return newfeat


def pairwise_dist_cosine(A, B, isl2norm=True):
    ''' Computes pairwise distance

    :param A: (B x N x D) containing descriptors of A
    :param B: (B x N x D) containing descriptors of B
    :return: (B x N x N) tensor. Element[i,j,k] denotes the distance between the jth descriptor in ith model of A,
             and kth descriptor in ith model of B
    '''
    cos_dist_mat = tf.matmul(A, B, transpose_b=True)
    dist_mat = tf.clip_by_value(cos_dist_mat, -1, 1)
    return dist_mat


def pairwise_dist(A, B):
    ''' Computes pairwise distance

    :param A: (B x N x D) containing descriptors of A
    :param B: (B x N x D) containing descriptors of B
    :return: (B x N x N) tensor. Element[i,j,k] denotes the distance between the jth descriptor in ith model of A,
             and kth descriptor in ith model of B
    '''
    A = tf.expand_dims(A, 2)  # b, n , 1, d
    B = tf.expand_dims(B, 1)  # b, 1, n, d
    dist = tf.reduce_sum(tf.squared_difference(A, B), 3)
    return dist


def log_tensor_info(tensors):
    for t in tensors:
        logger.info("name: {}, shape: {}".format(t.name, t.get_shape()))


@contextmanager
def backbone_scope(freeze):
    """
    Args:
        freeze (bool): whether to freeze all the variables under the scope
    """
    with ExitStack() as stack:
        if freeze:
            stack.enter_context(freeze_variables(stop_gradient=False, skip_collection=True))
        yield


def sample_points(xyz, npoint):
    '''
    :param xyz:
    :param npoint:
    :param knn:
    :param use_xyz:
    :return: new_xyz - Cluster centers
    '''

    if npoint <= 0:
        new_xyz = tf.identity(xyz)
    else:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))

    return new_xyz
