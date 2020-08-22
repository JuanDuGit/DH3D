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


import argparse

log_dir = './'


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ConfigFactory(object):

    def __init__(self, name):
        super(ConfigFactory, self).__init__()
        self.config_name = name

    def basic_config(self):
        config = {

            ###train
            'training_local': True,
            'extract_global': False,
            'detection': False,  # set to False when training local for the first few epochs
            'freezedetection': False,
            'freezebackbone': False,
            'freezeglobal': False,

            'local_backbone': 'backbone_local_dilate',

            'add_batch_norm': True,
            'loadpath': None,

            'start_lr': 5e-4,
            'decay_step': 5 * 2000,
            'decay_rate': 0.5,
            'add_weight_decay': True,
            'train_weight_decay': 1e-5,

            ### model parameters
            'init_feat_dim': 32,
            'featdim': 128,
            'knn_num': 8,

            ###data
            'num_points': 8192,
            'batch_size': 10,
            'num_pos': 1,
            'num_neg': 0,
            'other_neg': False,
            'input_R': True,
            'data_aug': ['Jitter'],
            'sampled_kpnum': 512,
            'data_basedir': '/usr/stud/duj/rbc/fixedgrid_lidarpc',

            ###  loss
            'add_local_loss': True,
            'add_det_loss': False,
            'add_global_loss': False,
            'margin': 1.0,
            'neg_weight': 5.0,
            'local_loss': 'desc_local_loss',
            'pos_r': 0.5,
            'search_r': 20.0,
            'local_loss_weight': 1.0,

            'savemodel_every_k_steps': 200,
        }
        return config

    def detection_config(self):
        cfg = self.basic_config()
        cfg = dotdict(cfg)
        # for detection
        cfg.detection = True
        cfg.detection_block = 'detection_block'
        cfg.loadpath = None
        cfg.add_det_loss = True
        cfg.detection_loss = 'local_detection_loss_nn'
        cfg.ar_th = 0.4
        cfg.det_k = 16
        cfg.ar_nn_k = 5
        cfg.det_loss_weight = 0.2

        return cfg

    def global_config(self):
        cfg = self.basic_config()
        cfg = dotdict(cfg)

        ## for global
        cfg.training_local = False
        cfg.detection = False
        cfg.extract_global = True
        cfg.freezebackbone = True
        cfg.freezedetection = True

        cfg.start_lr = 5e-4
        cfg.decay_step = 20000
        cfg.decay_rate = 0.9

        cfg.global_backbone = 'global_before_assemble'
        cfg.global_assemble = 'global_netvald_block'
        cfg.concat_xyz = False
        cfg.sampled_kpnum = -1
        cfg.global_subsample = -1
        cfg.gl_dilate = 8
        cfg.gl_dims = [256]

        # data
        cfg.batch_size = 2
        cfg.num_pos = 2
        cfg.num_neg = 8
        cfg.other_neg = True
        cfg.input_R = False
        cfg.data_aug = ['Rotate1D', 'Jitter', 'RotateSmall']

        # loss
        cfg.add_local_loss = False
        cfg.add_det_loss = False
        cfg.add_global_loss = True
        cfg.global_loss = 'lazy_quadruplet_loss'  # or lazy_triplet_loss
        cfg.global_triplet_margin = 0.5
        cfg.global_quadruplet_margin = 0.2
        cfg.global_loss_weight = 1

        return cfg

    def getconfig(self):
        cfg = self.basic_config()

        extra_configs = getattr(self, self.config_name)
        extra = extra_configs()
        cfg.update(extra)
        cfg = dotdict(cfg)
        return cfg


def printinfo(pos_r, **unused):
    print(pos_r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    args = parser.parse_args()
    configs = ConfigFactory(args.cfg).getconfig()
    printinfo(**configs)
