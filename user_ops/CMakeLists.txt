# University Tuebingen, 2018
# Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch

cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project( FlexConv )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(CUDA REQUIRED)
find_package(TensorFlow REQUIRED)

# include CUDA Toolkit header files
set(CUDA_SAMPLE_INC "$ENV{CUDA_INSTALL_PATH}/samples/common/inc")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_SAMPLE_INC: ${CUDA_SAMPLE_INC}")

# include CUBS
if (DEFINED ENV{CUB_INC})
    message(STATUS "Use Cuda-CUB from " $ENV{CUB_INC})
    set(CUB_INC $ENV{CUB_INC})
else()
    message(FATAL_ERROR "requires 'export CUB_INC=/path/to/cub")
endif()

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp -O3 -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 -Xptxas=-v --expt-relaxed-constexpr -D GOOGLE_CUDA=1 --gpu-architecture=sm_52 -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_ABI}" )

# quick fix for drone-ci
include_directories(SYSTEM "/usr/local/")
include_directories(SYSTEM "/extra/samples/Common/")
# fix cgtuebingen
include_directories(SYSTEM "/graphics/opt/opt_Ubuntu18.04/cuda/toolkit_9.2")

include_directories(SYSTEM "${CUDA_SAMPLE_INC}")
include_directories(SYSTEM "${CUDA_INCLUDE_DIRS}/../../")
include_directories(SYSTEM ${CUB_INC})
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIR})
include_directories(SYSTEM kernels)

add_tensorflow_gpu_operation("flex_conv")
add_tensorflow_gpu_operation("flex_deconv")
add_tensorflow_gpu_operation("flex_pool")
add_tensorflow_gpu_operation("knn_bruteforce")
add_tensorflow_gpu_operation("conv_pointset")
