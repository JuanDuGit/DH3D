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
//Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("FlexConv")
    .Input("features: T")
    .Input("theta: T")
    .Input("bias: T")
    .Input("neighborhood: int32")
    .Input("position: T")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      const auto features = c->input(0);
      const auto theta = c->input(1);
      const auto bias = c->input(2);
      const auto neighborhood = c->input(3);
      const auto position = c->input(4);

      // we require the input to have 4 axes
      ::tensorflow::shape_inference::ShapeHandle shape_hnd;
      TF_RETURN_IF_ERROR(c->WithRank(features, 3, &shape_hnd));  // B x Din x Ng
      TF_RETURN_IF_ERROR(
          c->WithRank(theta, 3, &shape_hnd));  // Dp x Din x Dout
      TF_RETURN_IF_ERROR(c->WithRank(bias, 2, &shape_hnd));  // Din x Dout
      TF_RETURN_IF_ERROR(
          c->WithRank(neighborhood, 3, &shape_hnd));             // B x K x N
      TF_RETURN_IF_ERROR(c->WithRank(position, 3, &shape_hnd));  // B x Dp x Ng

      shape_inference::DimensionHandle merged;

      // assert B equal
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 0), c->Dim(neighborhood, 0), &merged));
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 0), c->Dim(position, 0), &merged));

      // assert Ng equal
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 2), c->Dim(neighborhood, 2),
                   &merged));  // TODO(fabi?) fix global access and remove
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 2), c->Dim(position, 2), &merged));

      // assert Dp equal
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(theta, 0), c->Dim(position, 1), &merged));

      // assert Dout equal
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(theta, 2), c->Dim(bias, 1), &merged));

      // assert Din equal
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 1), c->Dim(theta, 1), &merged));
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 1), c->Dim(bias, 0), &merged));

      // specify output-shape
      auto B = c->Dim(features, 0);
      auto Dout = c->Dim(bias, 1);
      auto N = c->Dim(neighborhood, 2);
      c->set_output(0, c->MakeShape({B, Dout, N}));

      return Status::OK();
    })
    .Doc(R"doc(
Apply Sparse Convolution to inputs.

This applies a convolution to a neighborhood of inputs. The formula for computing the output is as follows:

  `output_j = \sum_i w(x_i, x0) * x_i`
  `w(x_i, x0) = ??`

features: each feature description for each point [B, Din, N].
theta: parameters for kernel function [Dp, Din, Dout].
bias: bias for kernel function [Din, Dout].
neighborhood: all K nearest neighbors [B, K, N].
position: each datapoint in 3d space [B, Dp, N].
output: each feature description for each point [B, Dout, N].
)doc");

REGISTER_OP("FlexConvGrad")
    .Input("features: T")
    .Input("theta: T")
    .Input("bias: T")
    .Input("neighborhood: int32")
    .Input("position: T")
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
Returns gradients of Sparse Convolution to inputs.

gradients: topdiff[B, N, Dout].
neighborhood: all K nearest neighbors [B, K, N].
position: each datapoint in 3d space [B, Dp, N].
features: each feature description for each point [B, Din, N].
theta: parameters for kernel function [Dp, Din, Dout].
bias: bias for kernel function [Din, Dout].
grad_features: gradient to each feature description for each point [B, N, Din].
grad_theta: gradient to parameters for kernel function [1, Dp, Din, Dout].
grad_bias: gradient to bias for kernel function [Din, Dout].
)doc");

}  // namespace tensorflow
