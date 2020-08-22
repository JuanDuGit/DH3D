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


import tensorflow as tf
import sys
import os
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from tf_utils import *
from layers import knn_bruteforce


def desc_local_loss(outs_dict, pos_r=0.5, search_r=20, margin=0.8, extra=False, neg_weight=5, **kwargs):
    """
    n-tuple loss
    """

    xyz0, xyz1 = tf.split(outs_dict['xyz_sampled'], 2, axis=0)
    feat0, feat1 = tf.split(outs_dict['feat_sampled'], 2, axis=0)
    rot = outs_dict['R']

    xyz0_warp = tf.matmul(xyz0, rot)
    xyzdist_sqrt = tf.sqrt(pairwise_dist(xyz0_warp, xyz1) + 1e-10)
    is_out_of_safe_radius = tf.greater(xyzdist_sqrt, pos_r * 2)
    is_within_search_radius = tf.less(xyzdist_sqrt, search_r)
    is_neg = tf.cast(tf.logical_and(is_out_of_safe_radius, is_within_search_radius), dtype=tf.float32)
    is_pos = tf.cast(tf.less(xyzdist_sqrt, pos_r), dtype=tf.float32)

    feat_dist = tf.sqrt(pairwise_dist(feat0, feat1) + 1e-10)
    num_pos = tf.count_nonzero(is_pos)
    num_neg = tf.count_nonzero(is_neg)
    num_neg = tf.cast(num_neg, tf.float32, name='num_neg')
    num_pos = tf.cast(num_pos, tf.float32, name='num_pos')

    pos_loss = tf.reduce_sum(is_pos * feat_dist) / (num_pos + 1e-10)
    neg_loss = tf.reduce_sum(is_neg * tf.nn.relu(margin - feat_dist)) / (num_neg + 1e-10)

    pos_loss = tf.identity(pos_loss, name='pos_loss')
    neg_loss = tf.identity(neg_weight * neg_loss, name='neg_loss')
    loss_sum = tf.add(pos_loss, neg_loss, name='loss_sum')

    if extra:
        add_moving_summary(loss_sum)
        add_moving_summary(pos_loss)
        add_moving_summary(neg_loss)
    return loss_sum


def local_detection_loss_nn(outs_dict, ar_th=0.3, det_k=16, ar_nn_k=5, pos_r=0.3, use_hardest_neg=True, **unused):
    xyz0, xyz1 = tf.split(outs_dict['xyz'], 2, axis=0)
    feat0, feat1 = tf.split(outs_dict['feat'], 2, axis=0)
    sample_ind0, sample_ind1 = tf.split(outs_dict['sample_nodes_concat'], 2, axis=0)

    score0, score1 = tf.split(outs_dict['att_sampled'], 2, axis=0)
    xyz_s0, xyz_s1 = tf.split(outs_dict['xyz_sampled'], 2, axis=0)
    feat_s0, feat_s1 = tf.split(outs_dict['feat_sampled'], 2, axis=0)
    rot = outs_dict['R']
    knn1, _ = knn_bruteforce(tf.transpose(xyz1, perm=[0, 2, 1]), k=det_k)

    batchsize = tf.shape(xyz0)[0]
    samplenum = tf.shape(xyz_s0)[1]

    xyz0_warp = tf.matmul(xyz_s0, rot)

    batch_indices = tf.tile(tf.reshape(tf.range(batchsize), (-1, 1, 1)), (1, samplenum, 1))  # N M 1
    indices = tf.concat([batch_indices, sample_ind1], axis=-1)  # Batch, M, 2
    knn1 = tf.transpose(knn1, [0, 2, 1])
    knn_sampled1 = tf.gather_nd(knn1, indices)  # batch, numpts, k ===> batch, M, k

    if use_hardest_neg:
        matching_xyz_dist_all = tf.sqrt(pairwise_dist(xyz0_warp, xyz_s1) + 1e-10)
        is_neg = tf.greater(matching_xyz_dist_all, 1)
        is_neg = tf.cast(is_neg, dtype=tf.float32)


        feat_dist_all = tf.sqrt(pairwise_dist(feat_s0, feat_s1) + 1e-10)
        neg_dist = feat_dist_all + (1 - is_neg) * 100
        hardest_neg_ind1 = tf.cast(tf.argmin(neg_dist, axis=2), tf.int32)  # batch, M
        hardest_neg_ind1 = tf.expand_dims(hardest_neg_ind1, 2)

        hardest_neg_indices = tf.concat([batch_indices, hardest_neg_ind1], axis=-1)  # Batch, M, 2
        knn_sampled_neg1 = tf.gather_nd(knn1, hardest_neg_indices)  # batch, numpts, k ===> batch, M, k
        knn_sampled1 = tf.concat([knn_sampled1, knn_sampled_neg1], -1)
        det_k = det_k * 2

    # gather feat
    batch_indices = tf.tile(tf.reshape(tf.range(batchsize), (-1, 1, 1, 1)), (1, samplenum, det_k, 1))  # N samplenum k,1
    feat1_indices = tf.concat([batch_indices, tf.expand_dims(knn_sampled1, axis=3)], axis=-1)  # N m k 2
    sampled_xyz1 = tf.gather_nd(xyz1, feat1_indices)
    sampled_feat1 = tf.gather_nd(feat1, feat1_indices)

    matching_xyz_dist = tf.sqrt(
        tf.reduce_sum(tf.squared_difference(tf.expand_dims(xyz0_warp, 2), sampled_xyz1), -1))

    # match features
    matching_feat_dist = tf.reduce_sum(tf.squared_difference(tf.expand_dims(feat_s0, 2), sampled_feat1),
                                       -1)
    dists, indices_k_feat = tf.nn.top_k(-matching_feat_dist, k=5)
    batch_indices = tf.tile(tf.reshape(tf.range(batchsize), (-1, 1, 1, 1)),
                            (1, samplenum, ar_nn_k, 1))
    samplenum_indices = tf.tile(tf.reshape(tf.range(samplenum), (1, -1, 1, 1)), (batchsize, 1, ar_nn_k, 1))
    indices_k_feat_select = tf.concat([batch_indices, samplenum_indices, tf.expand_dims(indices_k_feat, axis=3)],
                                      axis=-1)
    # compute ar
    sampled_xyzdist_selected = tf.gather_nd(matching_xyz_dist, indices_k_feat_select)

    is_good = tf.cast(tf.less_equal(sampled_xyzdist_selected, pos_r), tf.float32)
    padones = tf.ones([is_good.get_shape()[0], is_good.get_shape()[1], 1], tf.float32)
    is_good = tf.concat([is_good, padones], -1)
    first = tf.cast(tf.argmax(is_good, axis=-1), tf.float32)

    AR = tf.cast((first + 1e-8) / ar_nn_k, tf.float32)  # ar is between 0 and 1, 0 is the best
    score0 = tf.squeeze(score0, axis=2)
    matchingloss = 1 - (AR * score0 + ar_th * (1 - score0))
    det_loss = tf.reduce_mean(matchingloss, name='det_loss')
    add_moving_summary(det_loss)
    return det_loss


# adopted from PoinNetVLAD (https://github.com/mikacuy/pointnetvlad/blob/master/pointnetvlad_cls.py)
def best_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        # batch = query.get_shape()[0]
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1, int(num_pos), 1])  # shape num_pos x output_dim
        best_pos = tf.reduce_min(tf.reduce_sum(tf.squared_difference(pos_vecs, query_copies), 2), 1)
        # best_pos=tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        return best_pos


def lazy_triplet_loss_impl(q_vec, pos_vecs, neg_vecs, margin, scope="lazy_triplet_loss", **kwargs):
    with tf.name_scope(scope):
        print("margin!!!!!!!!!!!!!!!!!!!!!111", margin)
        best_pos = best_pos_distance(q_vec, pos_vecs)
        num_neg = neg_vecs.get_shape()[1]
        batch = q_vec.get_shape()[0]
        query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
        best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
        m = tf.fill([int(batch), int(num_neg)], margin)
        triplet_loss = tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m, tf.subtract(best_pos, tf.reduce_sum(
                                                                                         tf.squared_difference(neg_vecs, query_copies),
                                                                                         2))),
                                                               tf.zeros([int(batch), int(num_neg)])), 1))
        return triplet_loss


def lazy_triplet_loss(global_descs, batch_size, num_pos, num_neg, global_triplet_margin=0.5, **kwargs):
    outdim = global_descs.get_shape()[-1].value
    qvec, posvec, negvec = tf.split(global_descs, [batch_size, num_pos * batch_size,
                                                   num_neg * batch_size], axis=0)
    q_vec = tf.reshape(qvec, [batch_size, 1, outdim], name='qvec')
    pos_vecs = tf.reshape(posvec, [batch_size, num_pos, outdim], name='posvec')
    neg_vecs = tf.reshape(negvec, [batch_size, num_neg, outdim], name='negvec')
    return lazy_triplet_loss_impl(q_vec, pos_vecs, neg_vecs, global_triplet_margin)


def lazy_quadruplet_loss(global_descs, batch_size, num_pos, num_neg, global_triplet_margin=0.5, global_quadruplet_margin=0.2, **kwargs):
    outdim = global_descs.get_shape()[-1].value
    qvec, posvec, negvec, othernegvec = tf.split(global_descs,
                                                 [batch_size, num_pos * batch_size,
                                                  num_neg * batch_size, batch_size],
                                                 axis=0)
    q_vec = tf.reshape(qvec, [batch_size, 1, outdim], name='qvec')
    pos_vecs = tf.reshape(posvec, [batch_size, num_pos, outdim], name='posvec')
    neg_vecs = tf.reshape(negvec, [batch_size, num_neg, outdim], name='negvec')
    other_neg_vec = tf.reshape(othernegvec, [batch_size, 1, outdim], name='othernegvec')

    trip_loss = lazy_triplet_loss_impl(q_vec, pos_vecs, neg_vecs, global_triplet_margin)

    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg_vec, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], global_quadruplet_margin)

    second_loss = tf.reduce_mean(tf.reduce_max(tf.maximum(
        tf.add(m2, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, other_neg_copies), 2))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = trip_loss + second_loss

    return total_loss
