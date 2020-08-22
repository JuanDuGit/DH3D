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


from scipy.spatial import cKDTree
import numpy as np
import os
from tabulate import tabulate
from collections import namedtuple
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(CURRENT_DIR)))
from core.utils import load_descriptor_bin, get_sets_dict


def is_gt_match_2D(queries, ref, distance_thresh=25):
    distances = np.linalg.norm(
        np.expand_dims([queries['northing'], queries['easting']], axis=2)
        - np.expand_dims([ref['northing'], ref['easting']], axis=1), axis=0)

    return distances < distance_thresh


def retrieval(ref_descriptors, query_descriptors, max_num_nn):
    ref_tree = cKDTree(ref_descriptors)
    _, indices = ref_tree.query(query_descriptors, k=max_num_nn)
    return indices


def compute_tp_fp(ref_descriptors, query_descriptors,
                  gt_matches, *arg, **kwarg):
    threshold = max(int(round(len(ref_descriptors) / 100.0)), 1)
    indices = retrieval(ref_descriptors, query_descriptors, *arg, **kwarg)
    tp = gt_matches[np.expand_dims(np.arange(len(indices)), axis=1), indices]
    fp = np.logical_not(tp)
    tp_cum = np.cumsum(tp, axis=1)
    fp_cum = np.cumsum(fp, axis=1)
    valid = np.any(gt_matches, axis=1)
    one_percent_retrieved = np.any(tp[:, 0:threshold], axis=1)
    return tp_cum, fp_cum, valid, one_percent_retrieved


def compute_recall(*arg, **kwarg):
    tp, fp, valid = compute_tp_fp(*arg, **kwarg)
    return np.mean(tp[valid] > 0, axis=0)


class GlobalDesc_eval(object):

    def __init__(self, result_savedir, database_file, query_file, desc_dir, max_num_nn=25, dim=256, *arg, **kwarg):
        super(GlobalDesc_eval, self).__init__()
        assert (os.path.isfile(database_file) and os.path.isfile(query_file))
        self.desc_dir = desc_dir
        self.database_sets = get_sets_dict(database_file)
        self.query_sets = get_sets_dict(query_file)
        self.database_seqnum = len(self.database_sets)
        self.query_seqnum = len(self.query_sets)
        self.max_querynum = max_num_nn
        self.savedir = result_savedir
        self.desc_dim = dim
        if 'database_sequences' in kwarg:
            self.database_sequences = sorted(kwarg.get('database_sequences'))
        else:
            self.database_sequences = sorted(self.database_sets.keys())
        if 'query_sequences' in kwarg:
            self.query_sequences = sorted(kwarg.get('query_sequences'))
        else:
            self.query_sequences = sorted(self.query_sets.keys())

        print("loading databse position info and descriptors")
        self.database_pos, self.database_desc = self.get_database_pos_desc(isquery=False)
        print(" {} databases loaded".format(len(self.database_pos)))

        print("loading query position info and descriptors")
        self.query_pos, self.query_desc = self.get_database_pos_desc(True)
        print("{} queries loaded".format(len(self.query_pos)))

        print(tabulate(
            [[self.database_sequences[i], len(self.database_desc[i])] for i in range(len(self.database_sequences))],
            headers=["databaseseq", "segnum"]))
        print()
        print(tabulate([[self.query_sequences[i], len(self.query_desc[i])] for i in range(len(self.query_sequences))],
                       headers=["queryseq", "segnum"]))

    def get_database_pos_desc(self, isquery):
        pos_sets = []
        desc_sets = []
        descdir = self.desc_dir
        ext = '.bin'
        if isquery:
            usedict = self.query_sets
            useseq = self.query_sequences

        else:
            usedict = self.database_sets
            useseq = self.database_sequences

        for seq in useseq:
            seqinfo = usedict[seq]
            # print("{} has {} pointclouds \n".format(seq, len(seqinfo)))
            pos = {'northing': [], 'easting': []}
            descriptors = []
            for pcd in seqinfo:
                pos['northing'].append(pcd['northing'])
                pos['easting'].append(pcd['easting'])
                pcd_filepath = pcd['query']
                desc = load_descriptor_bin(os.path.join(descdir, pcd_filepath + ext), self.desc_dim)
                descriptors.append(desc)
            pos_sets.append(pos)
            descriptors = np.vstack(descriptors)
            desc_sets.append(descriptors)
        print(len(desc_sets))
        print(desc_sets[0].shape)
        return pos_sets, desc_sets

    def evaluate(self):
        retrieval_Result = namedtuple(
            'RetrievResult', ['refseq', 'queryseq', 'recalls', 'one_percent_retrieved'])

        results = []
        for ref_ind in range(len(self.database_sequences)):
            ref_loc = self.database_pos[ref_ind]
            ref_desc = self.database_desc[ref_ind]

            for query_ind in range(len(self.query_sequences)):
                if self.database_sequences[ref_ind] == self.query_sequences[query_ind]:
                    continue
                query_loc = self.query_pos[query_ind]
                query_desc = self.query_desc[query_ind]
                gt_matches = is_gt_match_2D(query_loc, ref_loc, 25)

                tp, fp, valid, one_percent = compute_tp_fp(ref_desc, query_desc, gt_matches,
                                                           max_num_nn=self.max_querynum)
                recall = np.mean(tp[valid] > 0, axis=0)
                validoneperc = np.mean(one_percent[valid])
                ret = retrieval_Result(self.database_sequences[ref_ind], self.query_sequences[query_ind], recall,
                                       validoneperc)
                results.append(ret)
        print(tabulate(
            [[ret.refseq, ret.queryseq, ret.recalls[0:5], ret.recalls[5:10], ret.recalls[10:15], ret.one_percent_retrieved] for ret in
             results], floatfmt=".4f",
            headers=["refseq", "queryseq", "recalls0-5", "recalls5-10", "recalls10-15", "1%"]))

        recalls = np.vstack([ret.recalls for ret in results])
        one_percent_retrieved = np.hstack([ret.one_percent_retrieved for ret in results])

        avg_recall = np.mean(recalls, axis=0)
        avg_one_percent_retrieved = np.mean(one_percent_retrieved)

        print("\n")
        print("Avg_recall:")
        for i, r in enumerate(avg_recall):
            print("{}: {:.4f}".format(i+1, r))
        print("\n")
        print("Avg_one_percent_retrieved:")
        print("{:.4f}".format(avg_one_percent_retrieved))





if __name__ == '__main__':
    ref_file = "../data/oxford_test_global/oxford_test_global_gt_reference.pickle"
    query_file = "../data/oxford_test_global/oxford_test_global_gt_query.pickle"
    res_dir = "../data/global/globaldesc_results"

    retrieval = GlobalDesc_eval(result_savedir='./', desc_dir=res_dir,
                                database_file=ref_file, query_file=query_file,
                                database_sequences=['2015-03-10-14-18-10'],
                                query_sequences=['2015-11-13-10-28-08'])
    retrieval.evaluate()
