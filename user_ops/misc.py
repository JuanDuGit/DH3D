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
# Modifications copyright (C) 2013 <Technical University of Munich/Juan Du>


import numpy as np
import tensorflow as tf
from tabulate import tabulate
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)
tf.set_random_seed(42)


class FakePointCloud(object):
  """docstring for FakePointCloud"""

  def __init__(self, B, N, K, Din, Dout, Dp, N2=1, scaling=1):
    super(FakePointCloud, self).__init__()
    assert K < N
    self.B = B
    self.N = N
    self.K = K
    self.Din = Din
    self.Dout = Dout
    self.Dp = Dp
    self.N2 = N2

    dtype = np.float64

    def find_neighbors(positions, K):
      # B, Dpos, N
      all_neighbors = []
      for batch in positions:
        distances = squareform(pdist(batch.T, 'euclidean'))
        all_neighbors.append(np.argsort(distances, axis=1)[:, :K])
      return np.array(all_neighbors).transpose(0, 2, 1)

    def random_values(shape):
      # return (np.random.randn(*shape) * 100).astype(np.int32).astype(np.float32)
      # return (np.random.randn(*shape) * 100).astype(np.int32).astype(np.float32)
      return np.random.randn(*shape).astype(np.float32)

    self.theta = random_values([self.Dp, self.Din, self.Dout]).astype(dtype)
    self.bias = random_values([self.Din, self.Dout]).astype(dtype)
    self.theta_rel = random_values([self.Din, self.Dout]).astype(dtype)
    self.bias_rel = random_values([self.Dout]).astype(dtype)

    self.position = random_values([self.B, self.Dp, self.N]).astype(dtype)
    self.features = random_values([self.B, self.Din, self.N]).astype(dtype)
    self.neighborhood = find_neighbors(
        self.position, self.K).astype(dtype=np.int32)

  def init_ops(self, dtype=np.float32):
    self.theta_op = tf.convert_to_tensor(self.theta.astype(dtype))
    self.bias_op = tf.convert_to_tensor(self.bias.astype(dtype))
    self.theta_rel_op = tf.convert_to_tensor(self.theta_rel.astype(dtype))
    self.bias_rel_op = tf.convert_to_tensor(self.bias_rel.astype(dtype))

    self.features_op = tf.convert_to_tensor(self.features.astype(dtype))
    self.position_op = tf.convert_to_tensor(self.position.astype(dtype))
    self.neighborhood_op = tf.convert_to_tensor(self.neighborhood)

  def expected_feature_shape(self):
    return [self.B, self.Din, self.N]

  def expected_output_shape(self):
    return [self.B, self.Dout, self.N]


class VerboseTestCase(tf.test.TestCase):

  def assertAllClose(self, a, b, rtol=1e-6, atol=1e-6):
    max_outputs = 20

    def max_tol(b):
      return atol + rtol * np.abs(b)

    if not np.allclose(a, b, rtol=rtol, atol=atol):
      cond = np.logical_or(
          np.abs(a - b) > atol + rtol * np.abs(b),
          np.isnan(a) != np.isnan(b))

      lines = []
      if a.ndim:

        shape = a.shape

        a = a.flatten()
        b = b.flatten()

        cond = np.logical_or(
            np.abs(a - b) > atol + rtol * np.abs(b),
            np.isnan(a) != np.isnan(b))

        idxArr = np.arange(a.shape[0])[np.where(cond)]
        xArr = a[np.where(cond)]
        yArr = b[np.where(cond)]
        for idx, x, y in zip(idxArr, xArr, yArr):
          idx = np.unravel_index(idx, shape)
          lines.append((idx, x, y, np.abs(x - y), max_tol(y)))
          max_outputs -= 1
          if max_outputs == 0:
            break
        print(tabulate(lines, headers=["index", "actual", "expected",
                                       "diff", "max-tol"]))
        print("diff (sum): ", np.abs(a - b).sum())
        print("diff (max): ", np.abs(a - b).max())
        print("diff (mean): ", np.abs(a - b).mean())
      else:
        # np.where is broken for scalars
        x, y = a, b
        lines.append((x, y, np.abs(x - y), max_tol(y)))
        print(tabulate(lines, headers=["actual", "expected",
                                       "diff", "max-tol"]))

      assert np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True), "failed"
