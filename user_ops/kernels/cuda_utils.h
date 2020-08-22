#ifndef LIB_CUDA_UTILS_H_
#define LIB_CUDA_UTILS_H_

template <typename T>
__device__ T* DynamicSharedMemory() {
  extern __shared__ __align__(sizeof(T)) unsigned char s_shm[];
  return reinterpret_cast<T*>(s_shm);
}

#endif  // LIB_CUDA_UTILS_H_
