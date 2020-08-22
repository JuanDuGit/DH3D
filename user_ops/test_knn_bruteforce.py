#!/usr/bin/env python
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


from misc import FakePointCloud, VerboseTestCase
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, squareform

from __init__ import knn_bruteforce

case = FakePointCloud(B=2, N=32, K=4, Din=2, Dout=6, Dp=3)
case = FakePointCloud(B=1, N=4, K=2, Din=1, Dout=1, Dp=3)


def python_bruteforce(positions, k):
    # B, Dpos, N
    all_neighbors = []
    all_distances = []
    for batch in positions:
        distances = squareform(pdist(batch.T, 'euclidean'))
        all_neighbors.append(np.argsort(distances, axis=1)[:, :k])
        all_distances.append(np.sort(distances, axis=1)[:, :k])
    return np.array(all_neighbors), np.array(all_distances)


class KnnBruteforceTest(VerboseTestCase):
    def __init__(self, methodName="runTest"):
        super(KnnBruteforceTest, self).__init__(methodName)

    def _forward(self, use_gpu=False):
        case.init_ops()
        expected_nn, expected_dist = python_bruteforce(case.position, k=4)

        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            actual_nn, actual_dist = knn_bruteforce(case.position_op, k=4)
            actual_nn, actual_dist = sess.run([actual_nn, actual_dist])

        self.assertAllClose(expected_dist, actual_dist)
        self.assertAllClose(expected_nn, actual_nn)

    def test_forward_cpu(self):
        self._forward(use_gpu=False)

    def test_forward_gpu(self):
        self._forward(use_gpu=True)


if __name__ == '__main__':
    tf.test.main()
