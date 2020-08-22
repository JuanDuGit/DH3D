/* Copyright 2017 ComputerGraphics Tuebingen. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch
// Modifications copyright (C) 2013 <Technical University of Munich/Juan Du>

#include "conv_pointset_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

namespace functor {

template <typename Dtype>
struct ConvPointsetFunctor<CPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& features_,
                  const Tensor& theta_, const Tensor& bias_,
                  const Tensor& neighborhood_,
                  Tensor* output_) {
    const auto features = features_.tensor<Dtype, 3>();
    const auto theta = theta_.tensor<Dtype, 2>();
    const auto bias = bias_.tensor<Dtype, 1>();
    const auto neighborhood = neighborhood_.tensor<int, 3>();


    auto output = output_->tensor<Dtype, 3>();

    // get dimensions
    const int B = neighborhood_.dim_size(0);
    const int K = neighborhood_.dim_size(1);
    const int N = neighborhood_.dim_size(2);
    const int Din = theta_.dim_size(0);
    const int Dout = theta_.dim_size(1);

    output.setZero();


    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
        for (int k_ = 0; k_ < K; ++k_) {
          int k = neighborhood(b, k_, n);

          for (int dout = 0; dout < Dout; ++dout) {
            for (int din = 0; din < Din; ++din) {
              Dtype delta_v = features(b, din, k) -
                              features(b, din, neighborhood(b, 0, n));
              output(b, dout, n) = output(b, dout, n) + theta(din, dout) * delta_v;
            }
            if (!k_) output(b, dout, n) = output(b, dout, n) + bias(dout);
          }
        }
      }
    }
  }
};

template struct ConvPointsetFunctor<CPUDevice, float>;
template struct ConvPointsetFunctor<CPUDevice, double>;

template <typename Dtype>
struct ConvPointsetGrad<CPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& features_,
                  const Tensor& theta_, const Tensor& bias_,
                  const Tensor& neighborhood_,
                  const Tensor& topdiff_, Tensor* grad_features_,
                  Tensor* grad_theta_, Tensor* grad_bias_) {
    const auto features = features_.tensor<Dtype, 3>();
    const auto theta = theta_.tensor<Dtype, 2>();
    const auto bias = bias_.tensor<Dtype, 1>();
    const auto neighborhood = neighborhood_.tensor<int, 3>();
    const auto topdiff = topdiff_.tensor<Dtype, 3>();

    auto grad_features = grad_features_->tensor<Dtype, 3>();
    auto grad_theta = grad_theta_->tensor<Dtype, 2>();
    auto grad_bias = grad_bias_->tensor<Dtype, 1>();

    // get dimensions
    const int B = neighborhood_.dim_size(0);
    const int K = neighborhood_.dim_size(1);
    const int N = neighborhood_.dim_size(2);


    const int Din = theta_.dim_size(0);
    const int Dout = theta_.dim_size(1);

    grad_features.setZero();
    grad_theta.setZero();
    grad_bias.setZero();

    // ========================= bias ==============================
    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
            for (int l = 0; l < Dout; ++l) {
              grad_bias(l) += topdiff(b, l, n);
            }

      }
    }

    // ========================= theta ==============================
    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
        for (int k_ = 0; k_ < K; ++k_) {
          int k = neighborhood(b, k_, n);

          for (int j = 0; j < Din; ++j) {
          const Dtype delta_features = features(b, j, k) -
                              features(b, j, neighborhood(b, 0, n));

            for (int l = 0; l < Dout; ++l) {
                grad_theta(j, l) += delta_features  * topdiff(b, l, n);
              }
          }
        }
      }
    }

    // ========================= features ==============================

    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
        for (int k_ = 0; k_ < K; ++k_) {
          int k = neighborhood(b, k_, n);

          for (int j = 0; j < Din; ++j) {
            for (int l = 0; l < Dout; ++l) {

              grad_features(b, j, k) += theta(j,l) * topdiff(b, l, n);
              grad_features(b, j, neighborhood(b, 0, n)) -= theta(j,l) * topdiff(b, l, n);
            }
          }
        }
      }
    }
  }
};


template struct ConvPointsetGrad<CPUDevice, float>;
template struct ConvPointsetGrad<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
