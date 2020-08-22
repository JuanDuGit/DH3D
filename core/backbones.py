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


import os
import math
import tensorflow as tf
from tensorpack import *
import tensorflow.contrib.slim as slim

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from layers import flex_convolution, flex_pooling, knn_bruteforce, flex_avg, convolution_pointset
from tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate
from tf_utils import *


def se_bottleneck(l, pool_l, ch_out, name):
    with tf.variable_scope(name):
        pool_l_T = tf.transpose(pool_l, perm=[0, 2, 1])
        # squeeze
        squeeze = feature_conv1d_1(pool_l_T, ch_out // 4, 'f1', ac_func=tf.nn.relu, c_last=True)
        squeeze = feature_conv1d_1(squeeze, ch_out, 'f2', ac_func=tf.nn.sigmoid, c_last=True)

        # excite
        l = l * tf.transpose(squeeze, perm=[0, 2, 1])
    return l


def se_res_bottleneck(l, pool_l, ch_out, name):
    shortcut = l
    with tf.variable_scope(name):
        pool_l_T = tf.transpose(pool_l, perm=[0, 2, 1])
        # squeeze
        squeeze = feature_conv1d_1(pool_l_T, ch_out // 4, 'f1', ac_func=tf.nn.relu, c_last=True)
        squeeze = feature_conv1d_1(squeeze, ch_out, 'f2', ac_func=tf.nn.sigmoid, c_last=True)

        # excite
        l = l * tf.transpose(squeeze, perm=[0, 2, 1])
    return tf.nn.relu(shortcut + l)


def flex_conv_dilate(xyz, feat, dilate, knn, outdims, scope, knn_indices=None, concat=True,
                     add_se='max_pool', upsample=True, **unused):
    num_point = xyz.get_shape()[1]
    npoint = num_point // dilate
    with tf.variable_scope(scope) as sc:
        if dilate > 1:
            points_sampled, feat_sampled, kp_indices = subsample(xyz, feat, npoint, kp_idx=None)
        else:
            points_sampled, feat_sampled = xyz, feat

        feats_T = tf.transpose(feat_sampled, perm=[0, 2, 1])
        points_T = tf.transpose(points_sampled, perm=[0, 2, 1])
        if knn_indices is None:  # B, knn, numpts
            knn_indices, distances = knn_bruteforce(points_T, k=knn)

        x = feats_T
        for i, d in enumerate(outdims):
            x = flexconv_withBatchnorm(x, points_T, knn_indices, d, name='flexconv_{}'.format(i))

        if add_se == 'max_pool':
            x_pool = flex_pooling(x, knn_indices, name='se_maxpool')
            newx = se_res_bottleneck(x, x_pool, outdims[-1], "se")  # l: B, 64, N
        elif add_se == 'avg_pool':
            x_pool = flex_avg(x, points_T, knn_indices, outdims[-1], name='se_avgpool')
            x_pool = x_pool * (1.0 / knn)
            newx = se_res_bottleneck(x, x_pool, outdims[-1], "se")  # l: B, 64, N
        else:
            newx = x

        new_feat = tf.transpose(newx, perm=[0, 2, 1])  # B, N, outdim

        # upsampling
        if upsample and dilate > 1:
            dist, idx = three_nn(xyz, points_sampled)
            dist = tf.maximum(dist, 1e-10)
            norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
            norm = tf.tile(norm, [1, 1, 3])
            weight = (1.0 / dist) / norm
            new_feat = three_interpolate(new_feat, idx, weight)

        if concat:
            new_feat = tf.concat(axis=2, values=[new_feat, feat])
            new_feat = feature_conv1d_1(new_feat, outdims[-1], name='concat_conv1d', c_last=True, ac_func=BNReLU)
        return xyz, new_feat


def backbone_local_dilate(points, featdim, knn_ind, dilate2=8, **unused):
    nn_8 = knn_ind[:, 0:8, :]

    # conv1d
    init_features = convolution_pointset_withBatchnorm(tf.transpose(points, perm=[0, 2, 1]), nn_8, 32, name='initconv')
    init_features = flex_pooling(init_features, nn_8, name='init_pool')
    init_features = tf.transpose(init_features, perm=[0, 2, 1])

    # stage 1
    newpoints1, x1 = flex_conv_dilate(points, init_features, dilate=1, knn=8, outdims=[64, 64], scope='stage1',
                                      knn_indices=nn_8, concat=False, add_se='max_pool')

    # stage 2
    x2 = feature_conv1d_1(x1, 64, name='before_stage2_conv1d', c_last=True, ac_func=BNReLU)
    print(x2)
    newpoints2, x2 = flex_conv_dilate(newpoints1, x2, dilate=dilate2, knn=8, outdims=[128, 128], scope='stage2',
                                      knn_indices=None, concat=True,
                                      add_se='max_pool')
    # combine
    feat = feature_conv1d_1(x1, 128, 'local_stage1_shortcut', c_last=True, ac_func=BNReLU) + x2

    if featdim < 128:
        feat = feature_conv1d_1(feat, featdim, 'final_fc', c_last=True)
    return newpoints2, feat


########################################################### detection:

def detection_block(features, scope='detection_block_reliable', freeze_det=False, conv_dims=[128, 256, 1024],
                    ac_func=BNReLU, use_softplus=False, **unused):
    features = tf.expand_dims(features, 2)

    with backbone_scope(freeze=freeze_det), \
         tf.variable_scope(scope), \
         argscope(Conv2D, kernel_shape=1, padding='VALID'):

        for i, d in enumerate(conv_dims):
            features = Conv2D('detec_conv{}'.format(i), features, d, activation=ac_func)

        logits = Conv2D('detec_conv_fc', features, 1, activation=tf.identity,
                        bias_initializer=tf.constant_initializer(1.0 / 8))
        logits = tf.squeeze(logits, axis=2)
    if use_softplus:
        detect_att = tf.nn.softplus(logits)
        detect_att = tf.identity(detect_att, name='det_att')
    else:
        detect_att = tf.nn.sigmoid(logits, name='det_att')
    return detect_att


########################################################### detection:

def globalatt_block(features, scope, ac_func):
    featdim = features.get_shape()[2]
    features = tf.expand_dims(features, 2)
    if featdim > 256:
        conv_dims = [256, 1024]
    else:
        conv_dims = [1024]

    with tf.variable_scope(scope), \
         argscope(Conv2D, kernel_shape=1, padding='VALID'):

        for i, d in enumerate(conv_dims):
            features = Conv2D('detec_conv{}'.format(i), features, d, activation=ac_func)

        logits = Conv2D('detec_conv_fc', features, 1, activation=tf.identity)
        logits = tf.squeeze(logits, axis=2)
        att = tf.nn.sigmoid(logits)
    return att


########################################################### for global:

def global_before_assemble(points, localdesc, knn_ind=None, knn_num=8, gl_dilate=8, gl_dims=[256, 1024],
                           concat_xyz=False, **unused):
    if concat_xyz:
        localdesc = tf.concat([points, localdesc], -1)
    newpoints, newfeat = flex_conv_dilate(points, localdesc, dilate=gl_dilate, knn=knn_num, outdims=gl_dims,
                                          scope='global_before_assemble',
                                          knn_indices=knn_ind,
                                          concat=False, upsample=True, add_se='')
    return newpoints, newfeat


def global_before_assemble_conv1d(points, localdesc, gl_dims=[256], concat_xyz=False, **unused):
    # conv1d is found to be better than flexconv for global descriptor
    if concat_xyz:
        localdesc = tf.concat([points, localdesc], -1)
    for i, d in enumerate(gl_dims):
        newfeat = feature_conv1d_1(localdesc, d, 'global_before_assemble_conv1{}'.format(i), c_last=True,
                                   ac_func=BNReLU)
    return points, newfeat


########################################################### global:

# Adopted from PCAN  (https://github.com/XLechter/PCAN/blob/master/pcan_cls.py)
def global_netvald_block(xyz, features, att, is_training, cluster_size=64, output_dim=256, add_batch_norm=True,
                         gating=True, **unused_kwargs):
    num_point = features.get_shape()[1].value
    feature_size = features.get_shape()[2].value

    ######################## code from loupe.netvlad
    reshaped_input = tf.reshape(features, [-1, feature_size])  # batch * numpt, feat_in
    reshaped_input = tf.nn.l2_normalize(reshaped_input, 1)

    cluster_weights = tf.get_variable("cluster_weights",
                                      [feature_size, cluster_size],
                                      initializer=tf.random_normal_initializer(
                                          stddev=1 / math.sqrt(feature_size)))  # feat_in x num_cluster

    activation = tf.matmul(reshaped_input, cluster_weights)  # batch * numpt, num_cluster
    if add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn", fused=False)
    else:
        cluster_biases = tf.get_variable("cluster_biases",
                                         [cluster_size],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(feature_size)))
        activation = activation + cluster_biases
    activation = tf.nn.softmax(activation)

    att = tf.reshape(att, [-1, 1], name='globalaggreation_m')

    att = tf.tile(att, [1, cluster_size])

    activation_crn = tf.multiply(activation, att)

    activation = tf.reshape(activation_crn,
                            [-1, num_point, cluster_size])

    a_sum = tf.reduce_sum(activation, -2, keepdims=True)

    cluster_weights2 = tf.get_variable("cluster_weights2",
                                       [1, feature_size, cluster_size],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(feature_size)))

    a = tf.multiply(a_sum, cluster_weights2)
    activation = tf.transpose(activation, perm=[0, 2, 1])

    reshaped_input = tf.reshape(reshaped_input, [-1,
                                                 num_point, feature_size])

    vlad = tf.matmul(activation, reshaped_input)
    vlad = tf.transpose(vlad, perm=[0, 2, 1])  # batch, feature, cluster,
    vlad = tf.subtract(vlad, a)  # batch, feature, cluster

    vlad = tf.nn.l2_normalize(vlad, 1)  # each feature is normalzed

    vlad = tf.reshape(vlad, [-1, cluster_size * feature_size])
    vlad = tf.nn.l2_normalize(vlad, 1)  # whole feature is normalized

    hidden1_weights = tf.get_variable("hidden1_weights",
                                      [cluster_size * feature_size, output_dim],
                                      initializer=tf.random_normal_initializer(
                                          stddev=1 / math.sqrt(cluster_size)))

    vlad = tf.matmul(vlad, hidden1_weights)

    ##Added a batch norm
    vlad = tf.contrib.layers.batch_norm(vlad,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope='bn')

    if gating:
        vlad = context_gating(vlad, add_batch_norm, is_training)
    vlad = tf.identity(vlad, name='final_global')
    return vlad


def context_gating(input_layer, add_batch_norm=True, is_training=True):
    """Context Gating

    Args:
    input_layer: Input layer in the following shape:
    'batch_size' x 'number_of_activation'

    Returns:
    activation: gated layer in the following shape:
    'batch_size' x 'number_of_activation'
    """

    input_dim = input_layer.get_shape().as_list()[1]

    gating_weights = tf.get_variable("gating_weights",
                                     [input_dim, input_dim],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(input_dim)))

    gates = tf.matmul(input_layer, gating_weights)

    if add_batch_norm:
        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            scope="gating_bn")
    else:
        gating_biases = tf.get_variable("gating_biases",
                                        [input_dim],
                                        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
        gates = gates + gating_biases

    gates = tf.sigmoid(gates)

    activation = tf.multiply(input_layer, gates)

    return activation
