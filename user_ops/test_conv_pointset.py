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


from misc import FakePointCloud, VerboseTestCase
import tensorflow as tf
import numpy as np
from __init__ import convolution_pointset

case = FakePointCloud(B=2, N=32, K=4, Din=2, Dout=6, Dp=3)


class ConvPointsetTest(VerboseTestCase):
    def __init__(self, methodName="runTest"):
        super(ConvPointsetTest, self).__init__(methodName)

    def _forward(self, use_gpu=False, force_gpu=False, dtype=np.float32):
        case.init_ops(dtype=dtype)
        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            actual_op = convolution_pointset(case.features_op,
                                             case.neighborhood_op,
                                             case.theta_rel_op, case.bias_rel_op)
            actual = sess.run(actual_op)
        return actual

    def test_forward(self, dtype=np.float32):
        cpu = self._forward(use_gpu=False)
        gpu = self._forward(use_gpu=True)
        self.assertAllClose(cpu, gpu, 1e-4)

    def test_forward_features_gpu_floats(self):
        cpu32 = self._forward(use_gpu=True, dtype=np.float32)
        cpu64 = self._forward(use_gpu=True, dtype=np.float64)
        self.assertAllClose(cpu32, cpu64, 1e-4)

    def _backward_features(self, use_gpu=False, dtype=np.float32, numdiff=True):
        case.init_ops(dtype=dtype)
        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            actual_op = convolution_pointset(case.features_op,
                                             case.neighborhood_op,
                                             case.theta_rel_op, case.bias_rel_op)
            if numdiff:
                return tf.test.compute_gradient(
                    [case.features_op], [case.features.shape], actual_op,
                    case.expected_output_shape())[0]
            else:
                return sess.run(tf.gradients(actual_op, [case.features_op]))[0]

    def _backward_bias(self, use_gpu=False, dtype=np.float32, numdiff=True):
        case.init_ops(dtype=dtype)
        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            actual_op = convolution_pointset(case.features_op,
                                             case.neighborhood_op,
                                             case.theta_rel_op, case.bias_rel_op)

            if numdiff:
                return tf.test.compute_gradient(
                    [case.bias_rel_op], [case.bias_rel.shape], actual_op,
                    case.expected_output_shape())[0]
            else:
                return sess.run(tf.gradients(actual_op, [case.bias_rel_op]))[0]

    def _backward_theta(self, use_gpu=False, dtype=np.float32, numdiff=True):
        case.init_ops(dtype=dtype)
        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            actual_op = convolution_pointset(case.features_op,
                                             case.neighborhood_op,
                                             case.theta_rel_op, case.bias_rel_op)

            if numdiff:
                return tf.test.compute_gradient(
                    [case.theta_rel_op], [case.theta_rel.shape], actual_op,
                    case.expected_output_shape())[0]
            else:
                return sess.run(tf.gradients(actual_op, [case.theta_rel_op]))[0]

    def test_backward_features_cpu_float64(self):
        actual, expected = self._backward_features(use_gpu=False, dtype=np.float64)
        self.assertAllClose(actual, expected)

    def test_backward_bias_cpu_float64(self):
        actual, expected = self._backward_bias(use_gpu=False, dtype=np.float64)
        self.assertAllClose(actual, expected)

    def test_backward_theta_cpu_float64(self):
        actual, expected = self._backward_theta(use_gpu=False, dtype=np.float64)
        self.assertAllClose(actual, expected)

    def test_backward_features_gpu_float64(self):
        actual, expected = self._backward_features(use_gpu=True, dtype=np.float64)
        self.assertAllClose(actual, expected)

    def test_backward_bias_gpu_float64(self):
        actual, expected = self._backward_bias(use_gpu=True, dtype=np.float64)
        self.assertAllClose(actual, expected)

    def test_backward_theta_gpu_float64(self):
        actual, expected = self._backward_theta(use_gpu=True, dtype=np.float64)
        self.assertAllClose(actual, expected)

    # float32 has some numerical instabilities due to summation
    # central difference as derivatives are totally instable
    # hence we just compare cpu and gpu outputs (float64 num-diff tests pass)
    def test_backward_features_gpu_float32(self, dtype=np.float32):
        cpu = self._backward_features(use_gpu=False, dtype=dtype, numdiff=False)
        gpu = self._backward_features(use_gpu=True, dtype=dtype, numdiff=False)
        self.assertAllClose(cpu, gpu, 1e-4)

    def test_backward_bias_gpu_float32(self, dtype=np.float32):
        cpu = self._backward_bias(use_gpu=False, dtype=dtype, numdiff=False)
        gpu = self._backward_bias(use_gpu=True, dtype=dtype, numdiff=False)
        self.assertAllClose(cpu, gpu, 1e-4)

    def test_backward_theta_gpu_float32(self, dtype=np.float32):
        cpu = self._backward_theta(use_gpu=False, dtype=dtype, numdiff=False)
        gpu = self._backward_theta(use_gpu=True, dtype=dtype, numdiff=False)
        self.assertAllClose(cpu, gpu, 1e-4)


if __name__ == '__main__':
    tf.test.main()
