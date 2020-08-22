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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <vector>
#include "cuda_utils.h"

#include <cuda.h>

#include <cub/cub.cuh>

#include <curand.h>
#include <curand_kernel.h>

#include <limits>

#include "knn_bruteforce_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

template <typename Dtype, typename NBtype, int C_THREADS = 256, int C_VPT = 2>
struct BlockBFKernel;

template <typename Dtype, typename NBtype, int C_THREADS, int C_VPT>
__global__ void runBlockBFKernel(
    const BlockBFKernel<Dtype, NBtype, C_THREADS, C_VPT> kernel) {
  kernel();
}

template <typename Dtype, typename NBtype, int C_THREADS, int C_VPT>
struct BlockBFKernel {
  void launch(int B) {
    dim3 block(C_THREADS, 1);
    dim3 grid(N, 1, B);

    size_t shm_size = Dp * sizeof(Dtype);

    runBlockBFKernel<<<grid, block, shm_size>>>((*this));
  }

  __device__ __forceinline__ void operator()() const {
    // extern __shared__ Dtype s_shm[];
    Dtype* s_shm = DynamicSharedMemory<Dtype>();
    Dtype* s_point = (Dtype*)&s_shm[0];

    if (N > C_THREADS * C_VPT) {
      printf(
          "<BlockBFKernel> Critical problem!!!!! Not enough resources spend "
          "for N points!!! %d < %d \n",
          C_THREADS * C_VPT, N);
      return;
    }

    int b = blockIdx.z;
    int y = blockIdx.x;

    int tid = threadIdx.x;
    for (int dpi = tid; dpi < Dp; dpi += blockDim.x) {
      s_point[dpi] = d_data[b * N * Dp + dpi * N + y];  // not aligned with
                                                        // data!
    }

    __syncthreads();

    Dtype thread_dists[C_VPT];  // keys
    NBtype thread_ids[C_VPT];   // values

    typedef cub::BlockRadixSort<Dtype, C_THREADS, C_VPT, NBtype> BlockRadixSort;
    typedef cub::BlockStore<Dtype, C_THREADS, C_VPT,
                            cub::BLOCK_STORE_WARP_TRANSPOSE>
        BlockStoreDists;
    typedef cub::BlockStore<NBtype, C_THREADS, C_VPT,
                            cub::BLOCK_STORE_WARP_TRANSPOSE>
        BlockStoreIds;

    // Allocate shared memory
    __shared__ union {
      typename BlockRadixSort::TempStorage sort;
      typename BlockStoreDists::TempStorage store_dists;
      typename BlockStoreIds::TempStorage store_ids;
    } temp_storage;

    for (int vpt_i = 0; vpt_i < C_VPT; ++vpt_i) {
      int x = vpt_i * C_THREADS + threadIdx.x;

      if (x < N) {
        Dtype sum = 0.f;
        for (int dpi = 0; dpi < Dp; ++dpi) {
          Dtype val = d_data[b * N * Dp + dpi * N + x] - s_point[dpi];
          sum += val * val;
        }
        thread_dists[vpt_i] = sqrt(sum);
        thread_ids[vpt_i] = x;
      } else {
        thread_dists[vpt_i] = std::numeric_limits<Dtype>::max();
        thread_ids[vpt_i] = -1;
      }
    }

    BlockRadixSort(temp_storage.sort).Sort(thread_dists, thread_ids);
    __syncthreads();

    BlockStoreIds(temp_storage.store_ids)
        .Store(&d_knn_ids[b * N * K + y * K], thread_ids, K);
    __syncthreads();

    BlockStoreDists(temp_storage.store_dists)
        .Store(&d_knn_dists[b * N * K + y * K], thread_dists, K);
  }

  const Dtype* d_data;

  Dtype* d_knn_dists;
  NBtype* d_knn_ids;

  int N;
  int Dp;
  int K;
};

template <typename Dtype, typename NBtype>
struct BlockBFKernelAttributesSetter {
  template <typename T>
  void setAttributes(T& kernel) {
    kernel.d_data = d_data;
    kernel.d_knn_dists = d_knn_dists;
    kernel.d_knn_ids = d_knn_ids;

    kernel.N = N;
    kernel.Dp = Dp;
    kernel.K = K;
  }

  const Dtype* d_data;

  Dtype* d_knn_dists;
  NBtype* d_knn_ids;

  int N;
  int Dp;
  int K;
};

namespace tensorflow {
namespace functor {

template <typename Dtype, typename NBtype>
struct KnnBruteforceFunctor<GPUDevice, Dtype, NBtype> {
  void operator()(::tensorflow::OpKernelContext* ctx, const Tensor& positions,
                  Tensor* neighborhood_out, Tensor* distances) {
    const int B = positions.dim_size(0);
    const int D = positions.dim_size(1);
    const int N = positions.dim_size(2);

    const int K = neighborhood_out->dim_size(2);

    BlockBFKernelAttributesSetter<Dtype, NBtype> attr;
    attr.d_data = positions.flat<Dtype>().data();
    attr.d_knn_ids = neighborhood_out->flat<NBtype>().data();
    attr.d_knn_dists = distances->flat<Dtype>().data();

    attr.N = N;
    attr.Dp = D;
    attr.K = K;

    if (N <= 32) {
      BlockBFKernel<Dtype, NBtype, 32, 1> knn;
      attr.setAttributes(knn);
      knn.launch(B);
    } else if (N <= 64) {
      BlockBFKernel<Dtype, NBtype, 64, 1> knn;
      attr.setAttributes(knn);
      knn.launch(B);
    } else if (N <= 128) {
      BlockBFKernel<Dtype, NBtype, 128, 1> knn;
      attr.setAttributes(knn);
      knn.launch(B);
    } else if (N <= 256) {
      BlockBFKernel<Dtype, NBtype, 128, 2> knn;
      attr.setAttributes(knn);
      knn.launch(B);
    } else if (N <= 512) {
      BlockBFKernel<Dtype, NBtype, 128, 4> knn;
      attr.setAttributes(knn);
      knn.launch(B);
    } else if (N <= 1024) {
      BlockBFKernel<Dtype, NBtype, 256, 4> knn;
      attr.setAttributes(knn);
      knn.launch(B);
    } else if (N <= 2048) {
      BlockBFKernel<Dtype, NBtype, 256, 8> knn;
      attr.setAttributes(knn);
      knn.launch(B);
    } else if (N <= 4096) {
      BlockBFKernel<Dtype, NBtype, 512, 8> knn;
      attr.setAttributes(knn);
      knn.launch(B);
    } else if (N <= 1024 * 8) {
      BlockBFKernel<Dtype, NBtype, 1024, 8> knn;
      attr.setAttributes(knn);
      knn.launch(B);
    } else {
      printf(
          "point sets greater then 8k are note yet supported!! Change to "
          "knn_graph operation!! \n");
    }
    cudaDeviceSynchronize();

    if (!ctx->eigen_gpu_device().ok()) {
      ctx->SetStatus(tensorflow::errors::Internal("KNNBF execution failed"));
    }
  }
};

template struct KnnBruteforceFunctor<GPUDevice, float, int>;
// // too much shared memory
// template struct KnnBruteforceFunctor<GPUDevice, double, int>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
