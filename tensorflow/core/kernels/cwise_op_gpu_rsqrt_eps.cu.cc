/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_ROCM || GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "cwise_op_rsqrt_eps.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_fp16.h"
#else
#include "third_party/gpus/cuda/include/cuda_fp16.hpp"
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void RsqrtEpsKernel(T* out, const T* in, T eps, uint64 elem)
{
  uint64 id = threadIdx.x + blockIdx.x*blockDim.x;
  if(id<elem)
    out[id]=T(1.)/(sqrt(in[id])+eps);
}

template <typename T>
__global__ void RsqrtEpsGradKernel(T* out, const T* grad, const T* in, T eps, uint64 elem)
{
  uint64 id = threadIdx.x + blockIdx.x*blockDim.x;
  if(id<elem) {
    // y' = -0.5/(sqrt(x)+c)^2 sqrt(x)
    T s = sqrt(in[id]);
    out[id]=T(-0.5)/(s*(s+eps)*(s+eps));
  }
}

template <>
__global__ void RsqrtEpsKernel<Eigen::half>(Eigen::half* out, const Eigen::half* in, Eigen::half eps, uint64 elem)
{
  uint64 id = threadIdx.x + blockIdx.x*blockDim.x;
  float _eps = __half2float(half(eps));
  if(id<elem) {
    float s = sqrt(__half2float(half(in[id])));
    out[id]=Eigen::half(1./(s+_eps));
  }
}

template <>
__global__ void RsqrtEpsGradKernel<Eigen::half>(Eigen::half* out, const Eigen::half* grad, const Eigen::half* in, Eigen::half eps, uint64 elem)
{
  uint64 id = threadIdx.x + blockIdx.x*blockDim.x;
  float _eps = __half2float(half(eps));
  if(id<elem) {
    // y' = -0.5/(sqrt(x)+c)^2 sqrt(x)
    float s = sqrt(__half2float(half(in[id])));
    out[id]=Eigen::half(-0.5/(s*(s+_eps)*(s+_eps)));
  }
}

template <typename T>
void LaunchRsqrtEpsOp<GPUDevice, T>::operator()(const GPUDevice& device, 
          typename TTypes<T>::Tensor out,
          typename TTypes<T>::ConstTensor in,
          T eps)
{
  const uint64 kThreadInBlock = 256;
  const uint64 num_elements = out.dimension(0);
  const uint64 num_blocks = (num_elements+kThreadInBlock-1)/kThreadInBlock;
      TF_CHECK_OK(GpuLaunchKernel(
        RsqrtEpsKernel<T>, num_blocks, kThreadInBlock,
        0, device.stream(), out.data(), in.data(), eps, num_elements));
}

template <typename T>
void LaunchRsqrtEpsGradOp<GPUDevice, T>::operator()(const GPUDevice& device, 
          typename TTypes<T>::Tensor out,
          typename TTypes<T>::ConstTensor in,
          typename TTypes<T>::ConstTensor grad,
          T eps)
{
  const uint64 kThreadInBlock = 256;
  const uint64 num_elements = out.dimension(0);
  const uint64 num_blocks = (num_elements+kThreadInBlock-1)/kThreadInBlock;
      TF_CHECK_OK(GpuLaunchKernel(
        RsqrtEpsGradKernel<T>, num_blocks, kThreadInBlock,
        0, device.stream(), out.data(), grad.data(), in.data(), eps, num_elements));
}

/*
#define INSTANTIATE_TYPE(T) \
template void LaunchRsqrtEpsOp<GPUDevice, T>::operator()(const GPUDevice& device, \
          typename TTypes<T>::Tensor out,                                       \
          typename TTypes<T>::ConstTensor in,                                  \
          T eps); \
template void LaunchRsqrtEpsGradOp<GPUDevice, T>::operator()(const GPUDevice& device, \
          typename TTypes<T>::Tensor out, \
          typename TTypes<T>::ConstTensor in, \
          typename TTypes<T>::ConstTensor grad, \
          T eps);
*/
#define INSTANTIATE_TYPE(T) \
  template struct LaunchRsqrtEpsOp<GPUDevice, T>; \
  template struct LaunchRsqrtEpsGradOp<GPUDevice, T>;


INSTANTIATE_TYPE(Eigen::half);
//INSTANTIATE_TYPE(bfloat16);
INSTANTIATE_TYPE(float);
INSTANTIATE_TYPE(double);

};  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
