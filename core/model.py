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


import sys
import os
import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

import losses
import backbones
from layers import knn_bruteforce
from tf_utils import backbone_scope
from tf_ops.grouping.tf_grouping import group_point

class DH3D(ModelDesc):

    def __init__(self, config):
        super(ModelDesc, self).__init__()
        self.config = config
        self.input_knn_indices = True if self.config.num_points > 8192 else False  # # if num_points > 8192 since the maximum
        # number of knn_bruteforce is 8192

        ## local
        self.local_backbone = self.config.local_backbone
        self.detection_block = self.config.detection_block

        ## global
        self.global_backbone = self.config.global_backbone
        self.global_assemble = self.config.global_assemble

        ## loss
        self.local_loss_func = self.config.local_loss
        self.global_loss_func = self.config.global_loss
        self.detection_loss_func = self.config.detection_loss

    @property
    def training(self):
        return bool(get_current_tower_context().is_training)

    def inputs(self):
        # anc, pos, R, ind1, ind2, knn
        # pointclouds
        ret = [tf.TensorSpec((self.config.batch_size, self.config.num_points, 3), tf.float32, 'anchor')]
        if self.config.num_pos > 0:
            ret.append(
                tf.TensorSpec((self.config.batch_size, self.config.num_points * self.config.num_pos, 3), tf.float32,
                              'pos'))
        if self.config.num_neg > 0:
            ret.append(
                tf.TensorSpec((self.config.batch_size, self.config.num_points * self.config.num_neg, 3), tf.float32,
                              'neg'))
        if self.config.other_neg:
            ret.append(tf.TensorSpec((self.config.batch_size, self.config.num_points, 3), tf.float32, 'otherneg'))

        # rotation for local training
        if self.config.input_R:
            ret.append(tf.TensorSpec((self.config.batch_size, 3, 3), tf.float32, 'R'))

        # random indices from local training
        if self.config.sampled_kpnum > 0:
            ret.append(
                tf.TensorSpec((self.config.batch_size, self.config.sampled_kpnum), tf.int32, 'sample_ind_anchor'))
            ret.append(tf.TensorSpec((self.config.batch_size, self.config.sampled_kpnum), tf.int32, 'sample_ind_pos'))

        # knn indices from input 
        if self.config.num_points > 8192:
            ret.append(tf.TensorSpec((self.config.batch_size, self.config.num_points, self.config.knn_num), tf.int32,
                                     'knn_ind_anchor'))
            if self.config.num_pos > 0:
                ret.append(tf.TensorSpec(
                    (self.config.batch_size, self.config.num_points * self.config.num_pos, self.config.knn_num),
                    tf.int32,
                    'knn_ind_pos'))
            if self.config.num_neg > 0:
                ret.append(tf.TensorSpec(
                    (self.config.batch_size, self.config.num_points * self.config.num_neg, self.config.knn_num),
                    tf.int32,
                    'knn_ind_neg'))
        return ret

    def compute_local(self, points, isfreeze=False):
        with backbone_scope(freeze=isfreeze):
            inputs_dict = {
                'points': points,
                'featdim': self.config.featdim,
                'knn_ind': self.knn_indices,
                'dilate': self.config.dilate
            }
            newpoints, localdesc = getattr(backbones, self.local_backbone)(**inputs_dict)
        return newpoints, localdesc



    def compute_global(self, outs, freeze_global=False):
        with backbone_scope(freeze=freeze_global):
            points = outs['xyz']
            localdesc = outs['feat']
            newpoints, forglobal = getattr(backbones, self.global_backbone)(points, localdesc, **self.config)

            ## if sample
            if self.config.global_subsample > 0:
                newpoints, forglobal, kp_indices = backbones.subsample(newpoints, forglobal, self.global_subsample,
                                                                       kp_idx=None)
            # global attention
            global_att = backbones.globalatt_block(forglobal, scope="globalatt", ac_func=BNReLU)

            inputs_dict = {
                'xyz': newpoints,
                'features': forglobal,
                'att': global_att,
                'is_training': self.training,
                'add_batch_norm': self.config.add_batch_norm,
            }
            globaldesc = getattr(backbones, self.global_assemble)(**inputs_dict)
        return globaldesc

    def build_graph(self, *inputs_dict):
        inputs_dict = dict(zip(self.input_names, inputs_dict))

        ####### concat pointclouds
        pcdset = [inputs_dict['anchor']]
        if self.config.num_pos > 0:
            pcdset.append(tf.reshape(inputs_dict['pos'], [-1, self.config.num_points, 3]))
        if self.config.num_neg > 0:
            pcdset.append(tf.reshape(inputs_dict['neg'], [-1, self.config.num_points, 3]))
        if self.config.other_neg:
            pcdset.append(inputs_dict['otherneg'])
        points = tf.concat(pcdset, 0, name='pointclouds')  # query+pos+neg+otherneg, numpts, 3

        if self.input_knn_indices:
            knn_ind_set = [inputs_dict['knn_ind_anchor']]
            if inputs_dict.get('knn_ind_pos'):
                knn_ind_set.append(inputs_dict['knn_ind_pos'])
            if inputs_dict.get('knn_ind_neg'):
                knn_ind_set.append(inputs_dict['knn_ind_neg'])
            knn_inds = tf.concat(knn_ind_set, 0, name='knn_inds')
            self.knn_indices = tf.transpose(knn_inds, perm=[0, 2, 1])  # batch, k. numpts
        else:
            self.knn_indices, distances = knn_bruteforce(tf.transpose(points, perm=[0, 2, 1]), k=self.config.knn_num)

        if self.config.sampled_kpnum > 0:
            sample_nodes_concat = tf.concat([inputs_dict['sample_ind_anchor'], inputs_dict['sample_ind_pos']], 0)
            self.sample_nodes_concat = tf.expand_dims(sample_nodes_concat, 2)
        else:
            self.sample_nodes_concat = None

        freeze_local = self.config.freezebackbone
        freeze_det = self.config.freezedetection
        freeze_global = self.config.freezeglobal

        ####### get local features
        outs = {}
        outs['xyz'] = points
        outs['knn_indices'] = self.knn_indices
        if self.config.input_R:
            outs['R'] = inputs_dict['R']

        newpoints, localdesc = self.compute_local(points, freeze_local)
        localdesc_l2normed = tf.nn.l2_normalize(localdesc, dim=2, epsilon=1e-8, name='feat_l2normed')
        outs['feat'] = localdesc
        outs['local_desc'] = localdesc_l2normed

        saved_tensor_xyz_feat = tf.concat([newpoints, localdesc_l2normed], -1, name='xyz_feat')


        ####### get local attentions
        if self.config.detection:
            detect_att = getattr(backbones, self.detection_block)(localdesc, freeze_det=freeze_det)
            outs['attention'] = detect_att
            saved_tensor_xyz_feat_att = tf.concat([newpoints, localdesc_l2normed, detect_att], -1, name='xyz_feat_att')

        if self.config.sampled_kpnum > 0:
            outs['sample_nodes_concat'] = self.sample_nodes_concat
            localxyzsample, localfeatsample, kp_indices = backbones.subsample(points, localdesc_l2normed,
                                                                                  self.config.sampled_kpnum,
                                                                                  kp_idx=self.sample_nodes_concat)
            outs['feat_sampled'] = localfeatsample
            outs['xyz_sampled'] = localxyzsample
            xyz_feat = tf.concat([localxyzsample, localfeatsample], -1, name='xyz_feat_sampled')
            if self.config.get('detection'):
                att_sampled = tf.squeeze(group_point(detect_att, kp_indices), axis=-1)
                outs['att_sampled'] = att_sampled

        #### get global features
        if self.config.extract_global:
            globaldesc = self.compute_global(outs, freeze_global=freeze_global)
            globaldesc_l2normed = tf.nn.l2_normalize(globaldesc, dim=-1, epsilon=1e-8, name='globaldesc')
            outs['global_desc'] = globaldesc_l2normed

        ### loss
        if self.training:
            return self.compute_loss(outs)

    def compute_loss(self, outs):
        loss = 0.0

        # global loss
        if self.config.extract_global:
            global_loss = getattr(losses, self.global_loss_func)(global_descs=outs['global_desc'], **self.config)
            global_loss = tf.multiply(global_loss, self.config.global_loss_weight, name='globaldesc_loss')
            add_moving_summary(global_loss)
            loss += global_loss

        # local loss
        if self.config.add_local_loss:
            local_loss = getattr(losses, self.local_loss_func)(outs, **self.config)
            local_loss = tf.multiply(local_loss, self.config.local_loss_weight, name='localdesc_loss')
            add_moving_summary(local_loss)
            loss += local_loss

        ## detection loss
        if self.config.detection and self.config.add_det_loss:
            det_loss = getattr(losses, self.detection_loss_func)(outs, **self.config)
            det_loss = tf.multiply(det_loss, self.config.det_loss_weight, name='det_loss')
            add_moving_summary(det_loss)
            loss += det_loss

        loss = tf.identity(loss, name="gl_loc_loss")
        add_moving_summary(loss)

        if self.config.add_weight_decay:
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(self.config.train_weight_decay), name='wd_cost')
        else:
            wd_cost = 0
        total_cost = tf.add(wd_cost, loss, name='total_cost')
        add_moving_summary(total_cost)
        return total_cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=self.config.start_lr,
            global_step=get_global_step_var(),
            decay_steps=self.config.decay_step,
            decay_rate=self.config.decay_rate, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)
