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

#include "flex_deconv_op.h"

#include <stdio.h>
#include <type_traits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class FlexDeconvOp : public OpKernel {
 public:
  explicit FlexDeconvOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // printf("--> Compute CPU Version <--\n");
    const Tensor& features_ = ctx->input(0);
    const Tensor& theta_ = ctx->input(1);
    const Tensor& bias_ = ctx->input(2);
    const Tensor& neighborhood_ = ctx->input(3);
    const Tensor& positions_ = ctx->input(4);

    const int B = neighborhood_.shape().dim_size(0);
    const int N = neighborhood_.shape().dim_size(2);
    const int Dout = theta_.shape().dim_size(2);

    Tensor* output_ = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({B, Dout, N}), &output_));

    ::tensorflow::functor::FlexDeconvFunctor<Device, Dtype>()(
        ctx, features_, theta_, bias_, neighborhood_, positions_, output_);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FlexDeconvOp);
};

// Backward-Pass (CPU, GPU)
// --------------------------------------------------
template <typename Device, typename Dtype>
class FlexDeconvGradOp : public OpKernel {
 public:
  explicit FlexDeconvGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // printf("--> Compute CPU Version <--\n");
    const Tensor& features_ = ctx->input(0);
    const Tensor& theta_ = ctx->input(1);
    const Tensor& bias_ = ctx->input(2);
    const Tensor& neighborhood_ = ctx->input(3);
    const Tensor& positions_ = ctx->input(4);

    const Tensor& topdiff_ = ctx->input(5);

    // specify output shape
    Tensor* grad_features_ = nullptr;
    Tensor* grad_theta_ = nullptr;
    Tensor* grad_bias_ = nullptr;

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, features_.shape(), &grad_features_));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, theta_.shape(), &grad_theta_));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, bias_.shape(), &grad_bias_));

    ::tensorflow::functor::FlexDeconvGrad<Device, Dtype>()(
        ctx, features_, theta_, bias_, neighborhood_, positions_, topdiff_,
        grad_features_, grad_theta_, grad_bias_);
  }
};

#define REGISTER_CUSTOM_OP(NAME, DEVICE, T)                       \
  REGISTER_KERNEL_BUILDER(                                        \
      Name(#NAME).Device(DEVICE_##DEVICE).TypeConstraint<T>("T"), \
      NAME##Op<DEVICE##Device, T>)

REGISTER_CUSTOM_OP(FlexDeconv, CPU, float);
REGISTER_CUSTOM_OP(FlexDeconvGrad, CPU, float);
REGISTER_CUSTOM_OP(FlexDeconv, CPU, double);
REGISTER_CUSTOM_OP(FlexDeconvGrad, CPU, double);

#ifdef GOOGLE_CUDA
REGISTER_CUSTOM_OP(FlexDeconv, GPU, float);
REGISTER_CUSTOM_OP(FlexDeconvGrad, GPU, float);
REGISTER_CUSTOM_OP(FlexDeconv, GPU, double);
REGISTER_CUSTOM_OP(FlexDeconvGrad, GPU, double);
#endif  // GOOGLE_CUDA
#undef REGISTER_CUSTOM_OP

}  // namespace tensorflow
