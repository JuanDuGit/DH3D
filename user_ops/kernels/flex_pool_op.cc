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

#include <stdio.h>
#include <type_traits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class FlexPoolOp : public OpKernel {
 public:
  explicit FlexPoolOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& features_ = ctx->input(0);
    const Tensor& neighborhood_ = ctx->input(1);

    const int B = features_.dim_size(0);
    const int D = features_.dim_size(1);
    const int N = features_.dim_size(2);

    Tensor* output_ = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({B, D, N}), &output_));

    Tensor* argmax_ = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({B, D, N}), &argmax_));

    ::tensorflow::functor::FlexPoolFunctor<Device, Dtype>()(
        ctx, features_, neighborhood_, output_, argmax_);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FlexPoolOp);
};

// Backward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class FlexPoolGradOp : public OpKernel {
 public:
  explicit FlexPoolGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& features_ = ctx->input(0);
    const Tensor& neighborhood_ = ctx->input(1);
    const Tensor& topdiff_ = ctx->input(2);
    const Tensor& argmax_ = ctx->input(3);

    // specify output shape
    Tensor* grad_features_ = nullptr;

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, features_.shape(), &grad_features_));

    ::tensorflow::functor::FlexPoolGrad<Device, Dtype>()(
        ctx, features_, neighborhood_, topdiff_, argmax_, grad_features_);
  }
};

#define REGISTER_CUSTOM_OP(NAME, DEVICE, T)                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name(#NAME).Device(DEVICE_##DEVICE).TypeConstraint<T>("T"), \
      NAME##Op<DEVICE##Device, T>)

REGISTER_CUSTOM_OP(FlexPool, CPU, float);
REGISTER_CUSTOM_OP(FlexPoolGrad, CPU, float);
REGISTER_CUSTOM_OP(FlexPool, CPU, double);
REGISTER_CUSTOM_OP(FlexPoolGrad, CPU, double);

#ifdef GOOGLE_CUDA
REGISTER_CUSTOM_OP(FlexPool, GPU, float);
REGISTER_CUSTOM_OP(FlexPoolGrad, GPU, float);
REGISTER_CUSTOM_OP(FlexPool, GPU, double);
REGISTER_CUSTOM_OP(FlexPoolGrad, GPU, double);
#endif  // GOOGLE_CUDA
#undef REGISTER_CUSTOM_OP

}  // namespace tensorflow
