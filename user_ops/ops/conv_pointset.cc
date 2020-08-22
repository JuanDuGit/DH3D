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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("ConvPointset")
    .Input("features: T")
    .Input("theta: T")
    .Input("bias: T")
    .Input("neighborhood: int32")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      const auto features = c->input(0);
      const auto theta = c->input(1);
      const auto bias = c->input(2);
      const auto neighborhood = c->input(3);

      ::tensorflow::shape_inference::ShapeHandle shape_hnd;
      TF_RETURN_IF_ERROR(c->WithRank(features, 3, &shape_hnd));  // B x Din x Ng
      TF_RETURN_IF_ERROR(
          c->WithRank(theta, 2, &shape_hnd));  // Din x Dout
      TF_RETURN_IF_ERROR(c->WithRank(bias, 1, &shape_hnd));  // Dout
      TF_RETURN_IF_ERROR(
          c->WithRank(neighborhood, 3, &shape_hnd));             // B x K x N


      shape_inference::DimensionHandle merged;

      // assert B equal
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 0), c->Dim(neighborhood, 0), &merged));


      // assert Ng equal
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 2), c->Dim(neighborhood, 2),
                   &merged));  // TODO(fabi?) fix global access and remove


      // assert Dp equal
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(theta, 1), c->Dim(bias, 0), &merged));

      // assert Din equal
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 1), c->Dim(theta, 0), &merged));


      // specify output-shape
      auto B = c->Dim(features, 0);
      auto Dout = c->Dim(bias, 0);
      auto N = c->Dim(neighborhood, 2);
      c->set_output(0, c->MakeShape({B, Dout, N}));

      return Status::OK();
    })
    .Doc(R"doc(
Apply one-by-one Convolution to inputs.

This applies a convolution to a neighborhood of inputs.

features: each feature description for each point [B, Din, N].
theta: parameters for kernel function [ Din, Dout].
bias: bias for kernel function [Dout].
neighborhood: all K nearest neighbors [B, K, N].
output: each feature description for each point [B, Dout, N].
)doc");


REGISTER_OP("ConvPointsetGrad")
    .Input("features: T")
    .Input("theta: T")
    .Input("bias: T")
    .Input("neighborhood: int32")
    .Input("gradients: T")
    .Output("grad_features: T")
    .Output("grad_theta: T")
    .Output("grad_bias: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));  // features
      c->set_output(1, c->input(1));  // theta
      c->set_output(2, c->input(2));  // bias
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Returns gradients of one-by-one Convolution to inputs.

gradients: topdiff[B, N, Dout].
neighborhood: all K nearest neighbors [B, K, N].
features: each feature description for each point [B, Din, N].
theta: parameters for kernel function [Din, Dout].
bias: bias for kernel function [Din, Dout].
grad_features: gradient to each feature description for each point [B, N, Din].
grad_theta: gradient to parameters for kernel function [1, Dp, Din, Dout].
grad_bias: gradient to bias for kernel function [Din, Dout].
)doc");

}  // namespace tensorflow
