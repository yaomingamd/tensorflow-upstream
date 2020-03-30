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

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
//#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/cwise_op_rsqrt_eps.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <vector>


namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class RsqrtEpsOp : public OpKernel {
 public:
  explicit RsqrtEpsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& eps = ctx->input(1);
    TensorShape out_shape = input.shape();
    Tensor* output = nullptr;
  	OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    auto num_elements = input.NumElements();
    if(num_elements==0)
      return;
    T epsval = static_cast<T>(eps.scalar<T>()());
    LaunchRsqrtEpsOp<Device,T>()(ctx->eigen_device<Device>(), 
      output->flat<T>(), 
      input.flat<T>(), 
      epsval);
  }
};

template <typename Device, typename T>
class RsqrtEpsGradOp : public OpKernel {
 public:
  explicit RsqrtEpsGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& grad = ctx->input(1);
    const Tensor& eps = ctx->input(2);
    TensorShape out_shape = input.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    auto num_elements = input.NumElements();
    if(num_elements==0)
      return;
    T epsval = static_cast<T>(eps.scalar<T>()());
    LaunchRsqrtEpsGradOp<Device,T>()(ctx->eigen_device<Device>(),
      output->flat<T>(),
      input.flat<T>(),
      grad.flat<T>(),
      epsval);
  }
};


#define REGISTER_CPU_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("RsqrtEps").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      RsqrtEpsOp<CPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("RsqrtEpsGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      RsqrtEpsGradOp<CPUDevice, type>);


REGISTER_CPU_KERNEL(Eigen::half);
//REGISTER_CPU_KERNEL(bfloat16);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_GPU_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("RsqrtEps").Device(DEVICE_GPU).TypeConstraint<type>("T")        \
      .HostMemory("eps"),      \
      RsqrtEpsOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("RsqrtEpsGrad").Device(DEVICE_GPU).TypeConstraint<type>("T")    \
      .HostMemory("eps"),      \
      RsqrtEpsGradOp<GPUDevice, type>);

REGISTER_GPU_KERNEL(Eigen::half);
//REGISTER_GPU_KERNEL(bfloat16);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);

#endif
};