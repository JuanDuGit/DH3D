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

#include "flex_pool_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

namespace functor {

template <typename Dtype>
struct FlexPoolFunctor<CPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& features_,
                  const Tensor& neighborhood_, Tensor* output_,
                  Tensor* argmax_) {
    const auto features = features_.tensor<Dtype, 3>();
    const auto neighborhood = neighborhood_.tensor<int, 3>();

    auto output = output_->tensor<Dtype, 3>();
    auto argmax = argmax_->tensor<int, 3>();

    // get dimensions
    const int B = neighborhood_.dim_size(0);
    const int K = neighborhood_.dim_size(1);
    const int N = neighborhood_.dim_size(2);
    const int D = features_.dim_size(1);

    output.setConstant(Eigen::NumTraits<Dtype>::lowest());
    argmax.setZero();  // stores global id

    for (int b = 0; b < B; ++b) {
      for (int d = 0; d < D; ++d) {
        for (int n = 0; n < N; ++n) {
          // max in neighborhood
          for (int k_ = 0; k_ < K; ++k_) {
            const int other_global_id = neighborhood(b, k_, n);
            if (output(b, d, n) < features(b, d, other_global_id)) {
              argmax(b, d, n) = other_global_id;
              output(b, d, n) = features(b, d, other_global_id);
            }
          }
        }
      }
    }
  }
};

template struct FlexPoolFunctor<CPUDevice, float>;
template struct FlexPoolFunctor<CPUDevice, double>;

template <typename Dtype>
struct FlexPoolGrad<CPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& features_,
                  const Tensor& neighborhood_, const Tensor& topdiff_,
                  const Tensor& argmax_, Tensor* grad_features_) {
    // as only argmax contributes to the output
    // only argmax receives the topdiff

    const auto features = features_.tensor<Dtype, 3>();
    const auto neighborhood = neighborhood_.tensor<int, 3>();
    const auto topdiff = topdiff_.tensor<Dtype, 3>();
    const auto argmax = argmax_.tensor<int, 3>();

    auto grad_features = grad_features_->tensor<Dtype, 3>();

    // get dimensions
    const int B = neighborhood_.dim_size(0);
    const int K = neighborhood_.dim_size(1);
    const int N = neighborhood_.dim_size(2);
    const int D = features_.dim_size(1);
    // printf("B %i K %i N %i D %i\n", B, K ,N, D);

    grad_features.setZero();

    for (int b = 0; b < B; ++b) {
      for (int d = 0; d < D; ++d) {
        for (int n = 0; n < N; ++n) {
          grad_features(b, d, argmax(b, d, n)) += topdiff(b, d, n);
        }
      }
    }
  }
};

template struct FlexPoolGrad<CPUDevice, float>;
template struct FlexPoolGrad<CPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow
