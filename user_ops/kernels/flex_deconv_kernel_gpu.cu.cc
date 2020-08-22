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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cub/cub.cuh>

#include "flex_deconv_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace {
inline int up2(int len, int th) { return (len - 1) / th + 1; }

template <typename Dtype>
__global__ void forward(const int B, const int N, const int K, const int Dp,
                        const int Din, const int Dout, const Dtype* positions,
                        const Dtype* features, const int* neighborhood,
                        const Dtype* theta, const Dtype* bias, Dtype* output) {
  /*
  positions B, Dp, N
  features  B, Din, N
  neighborhood B, K, N
  theta  Dp, Din, Dout
  bias Din, Dout
  output B, Dout, N
  */

  const int b = blockIdx.z;

  for (int n = blockIdx.y * blockDim.y + threadIdx.y; n < N;
       n += blockDim.y * gridDim.y) {
    const int self_k = neighborhood[b * K * N + 0 * N + n];

    for (int k_ = 0; k_ < K; ++k_) {
      const int other_k = neighborhood[b * K * N + k_ * N + n];

      for (int dout = blockIdx.x * blockDim.x + threadIdx.x; dout < Dout;
           dout += blockDim.x * gridDim.x) {
        for (int din = 0; din < Din; ++din) {
          const Dtype v = features[b * Din * N + din * N + self_k];
          Dtype W = bias[din * Dout + dout];

          for (int dp = 0; dp < Dp; ++dp) {
            Dtype delta = positions[b * Dp * N + dp * N + other_k] -
                          positions[b * Dp * N + dp * N + self_k];
            W += theta[dp * Din * Dout + din * Dout + dout] * delta;
          }

          Dtype Wv = W * v;
          tensorflow::CudaAtomicAdd(&output[b * Dout * N + dout * N + other_k],
                                    Wv);
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void backward(const int B, const int N, const int K, const int Dp,
                         const int Din, const int Dout,

                         const Dtype* positions, const Dtype* features,
                         const int* neighborhood,

                         const Dtype* theta, const Dtype* bias,

                         const Dtype* top_diff,

                         Dtype* grad_features, Dtype* grad_theta,
                         Dtype* grad_bias) {
  /*
  B, Dp,  N      positions, grad_positions
  B, Din, N      features, grad_features
  B, K,   N      neighborhood
  Dp, Din, Dout  theta, grad_theta
  Din, Dout      bias, grad_bias
  B, Dout, N     output, top_diff
  */

  const int b = blockIdx.z;

  // Compute
  // ---------------------------------------------------------------

  for (int n = blockIdx.y * blockDim.y + threadIdx.y; n < N;
       n += blockDim.y * gridDim.y) {
    const int self_k = neighborhood[b * K * N + 0 * N + n];

    for (int k_ = 0; k_ < K; ++k_) {
      const int other_k = neighborhood[b * K * N + k_ * N + n];

      for (int dout = blockIdx.x * blockDim.x + threadIdx.x; dout < Dout;
           dout += blockDim.x * gridDim.x) {
        for (int din = 0; din < Din; ++din) {
          const Dtype current_top_diff =
              top_diff[b * Dout * N + dout * N + other_k];
          const Dtype v = features[b * Din * N + din * N + self_k];

          // update bias
          Dtype bias_update = v * current_top_diff;
          tensorflow::CudaAtomicAdd(&grad_bias[din * Dout + dout], bias_update);

          Dtype W = bias[din * Dout + dout];

          // update theta
          for (int dp = 0; dp < Dp; ++dp) {
            Dtype delta = positions[b * Dp * N + dp * N + other_k] -
                          positions[b * Dp * N + dp * N + self_k];
            Dtype theta_update = v * delta * current_top_diff;
            tensorflow::CudaAtomicAdd(
                &grad_theta[dp * Din * Dout + din * Dout + dout], theta_update);

            W += theta[dp * Din * Dout + din * Dout + dout] * delta;
          }

          // update features
          Dtype feature_update = W * current_top_diff;
          tensorflow::CudaAtomicAdd(
              &grad_features[b * Din * N + din * N + self_k], feature_update);
          // tensorflow::CudaAtomicAdd(&grad_features[b * Din * N + din * N +
          // self_k], 1);
        }
      }
    }
  }
}

}  // namespace

namespace tensorflow {
namespace functor {

template <typename Dtype>
struct FlexDeconvFunctor<GPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& features_,
                  const Tensor& theta_, const Tensor& bias_,
                  const Tensor& neighborhood_, const Tensor& positions_,
                  Tensor* output_) {
    // printf("GPU::FlexDeconvFunctor:operator()\n");

    // get dimensions
    const int B = neighborhood_.dim_size(0);
    const int K = neighborhood_.dim_size(1);
    const int N = neighborhood_.dim_size(2);
    const int Dp = theta_.dim_size(0);
    const int Din = theta_.dim_size(1);
    const int Dout = theta_.dim_size(2);

    const int threads = 32;
    dim3 block(threads, threads, 1);
    dim3 grid(up2(Dout, threads), up2(N, threads), B);

    cudaMemset(output_->flat<Dtype>().data(), 0,
               output_->NumElements() * sizeof(Dtype));

    forward<Dtype><<<grid, block>>>(
        B, N, K, Dp, Din, Dout, positions_.flat<Dtype>().data(),
        features_.flat<Dtype>().data(), neighborhood_.flat<int>().data(),
        theta_.flat<Dtype>().data(), bias_.flat<Dtype>().data(),
        output_->flat<Dtype>().data());
  }
};

template struct FlexDeconvFunctor<GPUDevice, float>;
template struct FlexDeconvFunctor<GPUDevice, double>;

template <typename Dtype>
struct FlexDeconvGrad<GPUDevice, Dtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& features_,
                  const Tensor& theta_, const Tensor& bias_,
                  const Tensor& neighborhood_, const Tensor& positions_,
                  const Tensor& topdiff_, Tensor* grad_features_,
                  Tensor* grad_theta_, Tensor* grad_bias_) {
    // get dimensions
    const int B = neighborhood_.dim_size(0);
    const int K = neighborhood_.dim_size(1);
    const int N = neighborhood_.dim_size(2);
    const int Dp = theta_.dim_size(0);
    const int Din = theta_.dim_size(1);
    const int Dout = theta_.dim_size(2);

    const int threads = 32;
    dim3 block(threads, threads, 1);
    dim3 grid(up2(Dout, threads), up2(N, threads), B);

    cudaMemset(grad_features_->flat<Dtype>().data(), 0,
               grad_features_->NumElements() * sizeof(Dtype));
    cudaMemset(grad_theta_->flat<Dtype>().data(), 0,
               grad_theta_->NumElements() * sizeof(Dtype));
    cudaMemset(grad_bias_->flat<Dtype>().data(), 0,
               grad_bias_->NumElements() * sizeof(Dtype));

    backward<Dtype><<<grid, block>>>(
        B, N, K, Dp, Din, Dout,

        positions_.flat<Dtype>().data(), features_.flat<Dtype>().data(),
        neighborhood_.flat<int>().data(),

        theta_.flat<Dtype>().data(), bias_.flat<Dtype>().data(),

        topdiff_.flat<Dtype>().data(),

        grad_features_->flat<Dtype>().data(), grad_theta_->flat<Dtype>().data(),
        grad_bias_->flat<Dtype>().data());
  }
};

template struct FlexDeconvGrad<GPUDevice, float>;
template struct FlexDeconvGrad<GPUDevice, double>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
