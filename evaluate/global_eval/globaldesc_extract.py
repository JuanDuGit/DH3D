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
import shutil

from tensorpack.predict import PredictConfig, OfflinePredictor
from tensorpack.tfutils import get_model_loader
from tensorpack.dataflow import BatchData

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(CURRENT_DIR)))
from core.model import DH3D
from core.utils import mkdir_p
from core.datasets import Global_test_dataset
from core.configs import dotdict
from evaluation_retrieval import GlobalDesc_eval


def get_eval_global_testdata(cfg, data_path, ref_gt_file):
    querybatch = cfg.batch_size
    pos = cfg.num_pos
    neg = cfg.num_neg
    other_neg = 1 if cfg.other_neg else 0
    totalbatch = querybatch * (pos + neg + other_neg + 1)

    df = Global_test_dataset(basedir=data_path, test_file=os.path.join(data_path, ref_gt_file))
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


def eval_retrieval(evalargs):
    # Set up evaluation:
    save_dir = evalargs.save_dir
    mkdir_p(evalargs.save_dir)

    model_configs = get_model_config(evalargs.ModelPath)

    # Set up graph:
    pred_config = PredictConfig(
        model=DH3D(model_configs),
        session_init=get_model_loader(evalargs.ModelPath),
        input_names=['pointclouds'],
        output_names=['globaldesc'],  # ['globaldesc'], output_weights
    )
    predictor = OfflinePredictor(pred_config)

    # Data:
    df, totalbatch = get_eval_global_testdata(model_configs, evalargs.data_path, evalargs.ref_gt_file)

    # Predict:
    pcdnum = 0
    for [pcds, names] in df:  # pcds is a list, batchsize x numpts x 3
        batch = pcds.shape[0]
        if totalbatch > batch:
            numpts = pcds.shape[1]
            pcddim = pcds.shape[2]
            padzeros = np.zeros([totalbatch - batch, numpts, pcddim], dtype=np.float32)
            pcds = np.vstack([pcds, padzeros])
        results = predictor(pcds)

        global_feats = results[0]

        for i in range(batch):
            pcdnum += 1
            globaldesc = global_feats[i]
            name = names[i]
            savename = os.path.join(evalargs.save_dir, name)
            basedir = os.path.dirname(savename)
            mkdir_p(basedir)
            globaldesc.tofile(savename)


    print('predicted {} poitnclouds \n'.format(pcdnum))

    # Evaluation recall:
    if evalargs.eval_recall:
        evaluator = GlobalDesc_eval(result_savedir='./', desc_dir=save_dir,
                                    database_file=os.path.join(evalargs.data_path, evalargs.ref_gt_file),
                                    query_file=os.path.join(evalargs.data_path, evalargs.qry_gt_file),
                                    max_num_nn=25)
        evaluator.evaluate()
        print("evaluation finished!\n")

    if evalargs.delete_tmp:
        # delete all the descriptors
        descdirs = [os.path.join(save_dir, f) for f in os.listdir(save_dir)]
        descdirs = [d for d in descdirs if os.path.isdir(d)]
        for d in descdirs:
            shutil.rmtree(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--save_dir', type=str, default='./demo_data/res_global')

    # for evaluation
    parser.add_argument('--ModelPath', type=str, help='Model to load (for evaluation)',
                        default='../../models/global/globalmodel')
    # parser.add_argument('--data_path', type=str, default="../data/oxford_test_global")
    parser.add_argument('--data_path', type=str, default="./demo_data/")
    parser.add_argument('--ref_gt_file', type=str, default='global_ref_demo.pickle')
    parser.add_argument('--qry_gt_file', type=str, default='global_query_demo.pickle')

    parser.add_argument('--eval_recall', action='store_true', default=False)
    parser.add_argument('--delete_tmp', action='store_true', default=False)

    evalargs = parser.parse_args()
    eval_retrieval(evalargs)
