/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#if TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/dropout_op_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace dropout_kernels {

template <typename T>
__global__ void GenMaskKernel(int nthreads, const T* in0, const T* in1,
                              uint8* out) {
  GPU_1D_KERNEL_LOOP(index, nthreads) { out[index] = in0[index] >= *in1; }
}

template <typename T>
void GenMask(OpKernelContext* ctx, const T* in0, const T* in1,
             uint8* out, unsigned N) {
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(N, d);
  TF_CHECK_OK(GpuLaunchKernel(GenMaskKernel<T>, config.block_count,
                              config.thread_per_block, 0, d.stream(),
                              config.virtual_thread_count, in0, in1, out));
}

template
void GenMask<float>(OpKernelContext* ctx, const float* in0, const float* in1,
             uint8* out, unsigned N);
template
void GenMask<Eigen::half>(OpKernelContext* ctx, const Eigen::half* in0,
             const Eigen::half* in1, uint8* out, unsigned N);

}  // namespace dropout_kernels
}  // namespace tensorflow

#endif
