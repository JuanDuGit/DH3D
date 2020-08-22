// ComputerGraphics Tuebingen, 2018

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("KnnBruteforce")
    .Input("position: T")  // position:	 each datapoint in nd space [B,Dp, N].
    .Output("neighborhood_out: NBtype")  // neighborhood_out: all K nearest
                                         // neighbors        [B, N, K].
    .Output("distances: T")              // distances: 	 all K nearest distances
                                         // [B, N, K].
    .Attr("k: int")
    .Attr("return_timings: bool = false")
    .Attr("T: realnumbertype")
    .Attr("NBtype: {int32} = DT_INT32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      const auto position = c->input(0);
      const auto neighborhood_out = c->input(1);

      int K;
      c->GetAttr("k", &K);

      auto B = c->Dim(position, 0);
      auto N = c->Dim(position, 2);

      c->set_output(0, c->MakeShape({B, N, K}));
      c->set_output(1, c->MakeShape({B, N, K}));

      return Status::OK();
    });

}  // namespace tensorflow

// doc: K:					number of neighbors.
