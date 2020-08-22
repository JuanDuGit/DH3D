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
import sys
import numpy as np
import os
import json

from tensorpack.predict import PredictConfig, OfflinePredictor
from tensorpack.tfutils import get_model_loader
from tensorpack.dataflow import BatchData

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
# sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
from core.model import DH3D
from core.utils import mkdir_p, single_nms
from core.datasets import Local_test_dataset
from core.configs import dotdict


def get_eval_oxford_data(cfg={}):
    querybatch = cfg.batch_size
    totalbatch = querybatch
    df = Local_test_dataset(basedir='./demo_data', dim=3,
                            numpts=cfg.get('num_points'),
                            knn_require=cfg.get('knn_num'),
                            )
    df = BatchData(df, totalbatch, remainder=True)
    df.reset_state()
    return df, totalbatch


def get_model_config(Model_Path):
    model_base = os.path.dirname(Model_Path)
    model_config_json = os.path.join(model_base, 'config.json')

    assert os.path.exists(model_config_json)
    with open(model_config_json) as f:
        model_config = dotdict(json.load(f))
    return model_config


def get_predictor(model_config, Model_Path):
    model_config.sampled_kpnum = -1
    model_config.extract_global = False
    model_config.num_pos = 0
    model_config.num_neg = 0
    model_config.other_neg = False
    model_config.batch_size = 4

    if model_config.detection:
        output_vas = ['xyz_feat_att']
    else:
        output_vas = ['xyz_feat']

    input_vas = ['pointclouds']
    if model_config.num_points > 8192:
        input_vas.append('knn_inds')
    pred_config = PredictConfig(
        model=DH3D(model_config),
        session_init=get_model_loader(Model_Path),
        input_names=input_vas,
        output_names=output_vas
    )
    predictor = OfflinePredictor(pred_config)
    return predictor


def pred_saveres(eval_config, res, filename, kp_savenum=-1):
    ext_name = 'res.bin'
    if eval_config.save_all:
        save_res = np.float32(res)
        savename = os.path.join(eval_config.save_dir, '{}_{}'.format(filename, ext_name))
        res.tofile(savename)

    elif eval_config.perform_nms:
        xyz = res[:, 0:3]
        attention = 1 - res[:, -1]
        num_keypoints, max_indices = single_nms(xyz, attention, nms_radius=eval_config.nms_rad,
                                                min_response_ratio=eval_config.nms_min_ratio,
                                                max_keypoints=eval_config.nms_max_kp)
        xyzfeatatt_nms = res[max_indices, :]
        savename = os.path.join(eval_config.save_dir, '{}_nms_{}'.format(
            filename, ext_name))
        print('after NMS, shape', xyzfeatatt_nms.shape)
        xyzfeatatt_nms.tofile(savename)
        # print(savename)


def perform_pred(df, totalbatch, predictor, eval_config):
    totalnum = 0
    knn_ind = None

    for k in df:
        pc, name, ori_num = k[0:3]
        if len(k) > 3:
            knn_ind = k[3]

        batch = pc.shape[0]
        if totalbatch > batch:
            numpts, pcddim = pc.shape[1], pc.shape[2]
            padzeros = np.zeros([totalbatch - batch, numpts, pcddim], dtype=np.float32)
            pc = np.vstack([pc, padzeros])
            if knn_ind is not None:
                padzeros = np.zeros([totalbatch - batch, numpts, knn_ind.shape[2]], dtype=np.int32)
                knn_ind = np.vstack([knn_ind, padzeros])

        if knn_ind is not None:
            result = predictor(pc, knn_ind)[0]
        else:
            result = predictor(pc)[0]

        result = result[0:batch, :, :]
        kp_savenum = -1
        for b in range(0, batch):
            filename = name[b]
            res = result[b]
            num = ori_num[b]
            res = res[0:num, :]
            totalnum += 1
            pred_saveres(eval_config, res, filename[:-4], kp_savenum)
    return totalnum


def pred_local_oxford(eval_args):
    mkdir_p(eval_args.save_dir)
    model_config = get_model_config(eval_args.ModelPath)
    ####===============set up graph ===================+##
    if eval_args.dataset == 'oxford_lidar':
        model_config.num_points = 16384
    elif eval_args.dataset == 'oxford_dso':
        model_config.num_points = 9000
    predictor = get_predictor(model_config, eval_args.ModelPath)

    ####===============data ===================+##
    df, totalbatch = get_eval_oxford_data(model_config)

    ####=============== predict ===================+##
    totalnum = perform_pred(df, totalbatch, predictor, eval_args)
    print('Predict {} point clouds'.format(totalnum))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--save_dir', type=str, default='./demo_data/res_local')
    parser.add_argument('--ModelPath', type=str, help='Model to load (for evaluation)',
                        default='../../models/local/localmodel')
    parser.add_argument('--dataset', type=str, help='oxford_lidar or oxford_dso', default='oxford_lidar')
    parser.add_argument('--save_all', action='store_true',
                        help='save dense feature map, which can be helpful when evaluating with other 3D detectors', default=False)
    parser.add_argument('--perform_nms', action='store_true', help='perform nms and save detected descriptors', default=False)
    parser.add_argument('--nms_rad', type=float, default=0.5)
    parser.add_argument('--nms_min_ratio', type=float, default=0.01)
    parser.add_argument('--nms_max_kp', type=int, default=512)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pred_local_oxford(args)
