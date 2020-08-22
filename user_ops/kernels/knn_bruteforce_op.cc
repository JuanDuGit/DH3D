/* Copyright 2018 ComputerGraphics Tuebingen. All Rights Reserved.

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

#include "knn_bruteforce_op.h"

#include <stdio.h>
#include <type_traits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class KnnBruteforceOp : public OpKernel {
 public:
  explicit KnnBruteforceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &K));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& positions = ctx->input(0);
    const Tensor& neighborhood_in = ctx->input(1);

    const int B = positions.shape().dim_size(0);
    const int N = positions.shape().dim_size(2);

    Tensor* neighborhood_out = nullptr;
    Tensor* distances = nullptr;

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({B, N, K}),
                                             &neighborhood_out));

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({B, N, K}), &distances));

    ::tensorflow::functor::KnnBruteforceFunctor<Device, Dtype, int> knnBFF;
    knnBFF(ctx, positions, neighborhood_out, distances);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(KnnBruteforceOp);

  int K;
};

#define REGISTER_CUSTOM_OP(NAME, DEVICE, T)                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name(#NAME).Device(DEVICE_##DEVICE).TypeConstraint<T>("T"), \
      NAME##Op<DEVICE##Device, T>)

REGISTER_CUSTOM_OP(KnnBruteforce, CPU, float);

#ifdef GOOGLE_CUDA
REGISTER_CUSTOM_OP(KnnBruteforce, GPU, float);
#endif  // GOOGLE_CUDA
#undef REGISTER_CUSTOM_OP

}  // namespace tensorflow
