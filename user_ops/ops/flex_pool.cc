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

REGISTER_OP("FlexPool")
    .Input("features: T")
    .Input("neighborhood: int32")
    .Output("output: T")
    .Output("argmax: int32")
    .Attr("T: realnumbertype")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      const auto features = c->input(0);
      const auto neighborhood = c->input(1);

      // we require the input to have 3 axes
      ::tensorflow::shape_inference::ShapeHandle shape_hnd;
      TF_RETURN_IF_ERROR(c->WithRank(features, 3, &shape_hnd));  // B x D x N
      TF_RETURN_IF_ERROR(
          c->WithRank(neighborhood, 3, &shape_hnd));  // B x K x N

      shape_inference::DimensionHandle merged;

      // assert B equal
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 0), c->Dim(neighborhood, 0), &merged));

      // assert N equal
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(features, 2), c->Dim(neighborhood, 2), &merged));

      // specify output-shape
      auto B = c->Dim(features, 0);
      auto D = c->Dim(features, 1);
      auto N = c->Dim(features, 2);
      c->set_output(0, c->MakeShape({B, D, N}));
      c->set_output(1, c->MakeShape({B, D, N}));

      return Status::OK();
    })
    .Doc(R"doc(
Apply Sparse Pooling to inputs.

This applies a max-pooling to a neighborhood of inputs. T

features: each feature description for each point [B, D, N].
neighborhood: all K nearest neighbors [B, K, N].
output: each feature description for each point [B, D, N].
argmax: global id in neighborhood who was winning the pooling [B, D, N]. This is needed for gradients.
)doc");

REGISTER_OP("FlexPoolGrad")
    .Input("features: T")
    .Input("neighborhood: int32")
    .Input("gradients: T")
    .Input("argmax: int32")
    .Output("grad_features: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));  // features
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Returns gradients of MaxPool to inputs.

features: each feature description for each point [B, D, N].
neighborhood: all K nearest neighbors [B, K, N].
gradients: topdiff[B, D, N].
argmax: argmax[B, D, N].
grad_features: gradient to each feature description for each point [B, D, N].

)doc");

}  // namespace tensorflow
