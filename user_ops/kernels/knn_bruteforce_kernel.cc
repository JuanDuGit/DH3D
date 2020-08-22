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
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

namespace functor {

template <typename Dtype, typename NBtype>
struct KnnBruteforceFunctor<CPUDevice, Dtype, NBtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& positions_,
                  Tensor* neighborhood_out_, Tensor* distances_) {
    // positions [B, Dp, N]
    // neighborhood_out_ [B, N, K]
    // distances_ [B, N, K]
    const auto positions = positions_.tensor<Dtype, 3>();
    auto neighborhood_out = neighborhood_out_->tensor<NBtype, 3>();
    auto distances_out = distances_->tensor<Dtype, 3>();

    const int B = positions_.dim_size(0);
    const int Dp = positions_.dim_size(1);
    const int N = positions_.dim_size(2);

    const int K = neighborhood_out_->dim_size(2);

    for (int b = 0; b < B; ++b) {
      const Dtype* pc_raw = positions_.flat<Dtype>().data() + b * Dp * N;
      Eigen::Map<const Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
          pc(pc_raw, Dp, N);
      // pc: [Dp, N]

      for (int n = 0; n < N; ++n) {
        const auto query = pc.col(n);
        const auto diffs =
            (pc.colwise() - query).array() * (pc.colwise() - query).array();
        const auto distances = (diffs.colwise().sum()).array().sqrt();

        std::vector<Dtype> vec_dist;
        std::vector<Dtype> vec_ids;

        for (int i = 0; i < N; ++i) {
          vec_dist.push_back(distances(i));
          vec_ids.push_back(i);
        }

        std::sort(std::begin(vec_ids), std::end(vec_ids),
                  [&](int i1, int i2) { return vec_dist[i1] < vec_dist[i2]; });

        for (int k = 0; k < K; ++k) {
          neighborhood_out(b, n, k) = vec_ids[k];
          distances_out(b, n, k) = vec_dist[vec_ids[k]];
        }
      }
    }
  }
};

template struct KnnBruteforceFunctor<CPUDevice, float, int>;
template struct KnnBruteforceFunctor<CPUDevice, double, int>;

}  // namespace functor
}  // namespace tensorflow
