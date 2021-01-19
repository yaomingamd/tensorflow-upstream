/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"
#include "tensorflow/core/kernels/cwise_ops_gpu_gradients.cu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"


namespace tensorflow {
namespace functor {

#ifdef GOOGLE_CUDA
#ifndef MLIR_GENERATED_GPU_KERNELS_ENABLED
DEFINE_UNARY3(tanh, Eigen::half, float, double);
#endif
#else
template <class T>
__global__ void TanhKernel(const T* in, T* out, int32 count);

template <typename T>
struct UnaryFunctor<GPUDevice, functor::tanh<T> >  {
  typedef typename TTypes<T>::ConstTensor tin_type;
  typedef typename TTypes<T>::Tensor tout_type;
  void operator()(const GPUDevice& d, 
                  typename TTypes<T>::Tensor output,
                  typename TTypes<T>::ConstTensor input) {
    int32 count = input.size();
    if (count == 0) return;
    typedef typename std::conditional<std::is_same<T, Eigen::half>::value, 
                                      __half, T>::type TT;
    constexpr int32 kThreadInBlock = 256;
    TF_CHECK_OK(GpuLaunchKernel(
        TanhKernel<TT>, (count + kThreadInBlock - 1) / kThreadInBlock,
        kThreadInBlock, 0, d.stream(),
        reinterpret_cast<const TT*>(input.data()),
        reinterpret_cast<TT*>(output.data()), count));
  }
};

template struct UnaryFunctor<GPUDevice, functor::tanh<Eigen::half> >;
template struct UnaryFunctor<GPUDevice, functor::tanh<float> >;
template struct UnaryFunctor<GPUDevice, functor::tanh<double> >;
#endif

DEFINE_SIMPLE_BINARY3(tanh_grad, Eigen::half, float, double);
}  // namespace functor


}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
