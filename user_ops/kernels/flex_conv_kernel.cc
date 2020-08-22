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

#include "flex_conv_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

namespace functor {

template <typename Dtype>
struct FlexConvFunctor<CPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& features_,
                  const Tensor& theta_, const Tensor& bias_,
                  const Tensor& neighborhood_, const Tensor& positions_,
                  Tensor* output_) {
    const auto features = features_.tensor<Dtype, 3>();
    const auto theta = theta_.tensor<Dtype, 3>();
    const auto bias = bias_.tensor<Dtype, 2>();
    const auto neighborhood = neighborhood_.tensor<int, 3>();
    const auto positions = positions_.tensor<Dtype, 3>();

    auto output = output_->tensor<Dtype, 3>();

    // get dimensions
    const int B = neighborhood_.dim_size(0);
    const int K = neighborhood_.dim_size(1);
    const int N = neighborhood_.dim_size(2);
    const int Dp = theta_.dim_size(0);
    const int Din = theta_.dim_size(1);
    const int Dout = theta_.dim_size(2);

    output.setZero();

    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
        for (int k_ = 0; k_ < K; ++k_) {
          int k = neighborhood(b, k_, n);

          for (int dout = 0; dout < Dout; ++dout) {
            for (int din = 0; din < Din; ++din) {
              const Dtype v = features(b, din, k);

              Dtype W = bias(din, dout);
              for (int dp = 0; dp < Dp; ++dp) {
                Dtype delta = positions(b, dp, k) -
                              positions(b, dp, neighborhood(b, 0, n));
                W += theta(dp, din, dout) * delta;
              }
              output(b, dout, n) = output(b, dout, n) + W * v;
            }
          }
        }
      }
    }
  }
};

template struct FlexConvFunctor<CPUDevice, float>;
template struct FlexConvFunctor<CPUDevice, double>;

template <typename Dtype>
struct FlexConvGrad<CPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& features_,
                  const Tensor& theta_, const Tensor& bias_,
                  const Tensor& neighborhood_, const Tensor& positions_,
                  const Tensor& topdiff_, Tensor* grad_features_,
                  Tensor* grad_theta_, Tensor* grad_bias_) {
    const auto features = features_.tensor<Dtype, 3>();
    const auto theta = theta_.tensor<Dtype, 3>();
    const auto bias = bias_.tensor<Dtype, 2>();
    const auto neighborhood = neighborhood_.tensor<int, 3>();
    const auto positions = positions_.tensor<Dtype, 3>();
    const auto topdiff = topdiff_.tensor<Dtype, 3>();

    auto grad_features = grad_features_->tensor<Dtype, 3>();
    auto grad_theta = grad_theta_->tensor<Dtype, 3>();
    auto grad_bias = grad_bias_->tensor<Dtype, 2>();

    // get dimensions
    const int B = neighborhood_.dim_size(0);
    const int K = neighborhood_.dim_size(1);
    const int N = neighborhood_.dim_size(2);
    const int Ddegree = theta_.dim_size(0);
    const int Dp = theta_.dim_size(0);
    const int Din = theta_.dim_size(1);
    const int Dout = theta_.dim_size(2);

    grad_features.setZero();
    grad_theta.setZero();
    grad_bias.setZero();

    // ========================= bias ==============================
    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
        for (int k_ = 0; k_ < K; ++k_) {
          int k = neighborhood(b, k_, n);

          for (int j = 0; j < Din; ++j) {
            for (int l = 0; l < Dout; ++l) {
              grad_bias(j, l) += features(b, j, k) * topdiff(b, l, n);
            }
          }
        }
      }
    }

    // ========================= theta ==============================
    for (int b = 0; b < B; ++b) {
      for (int n = 0; n < N; ++n) {
        for (int k_ = 0; k_ < K; ++k_) {
          int k = neighborhood(b, k_, n);

          for (int j = 0; j < Din; ++j) {
            for (int l = 0; l < Dout; ++l) {
              for (int i = 0; i < Dp; ++i) {
                const Dtype delta =
                    positions(b, i, k) - positions(b, i, neighborhood(b, 0, n));

                grad_theta(i, j, l) +=
                    features(b, j, k) * delta * topdiff(b, l, n);
              }
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
              Dtype W = bias(j, l);
              for (int i = 0; i < Dp; ++i) {
                const Dtype delta =
                    positions(b, i, k) - positions(b, i, neighborhood(b, 0, n));

                W += theta(i, j, l) * delta;
              }
              grad_features(b, j, k) += W * topdiff(b, l, n);
            }
          }
        }
      }
    }
  }
};

template struct FlexConvGrad<CPUDevice, float>;
template struct FlexConvGrad<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
