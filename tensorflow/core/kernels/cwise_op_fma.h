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

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OP_FMA_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OP_FMA_H_

namespace tensorflow {

template <typename Device, typename T, int SGN>
class LaunchFusedMulAddOp
{
public:
	void operator()(const Device& device,
		T* out,
		const T* x1, const T* y1, const T* x2,
		uint64 elements, 
		bool broadcast_x1, bool broadcast_y1, 
		bool broadcast_x2);
}; 

template <typename Device, typename T, int SGN>
class LaunchFusedMulAdd2Op
{
public:
	void operator()(const Device& device,
		T* out,
		const T* x1, const T* y1, const T* x2, const T* y2,
		uint64 elements,
		bool broadcast_x1, bool broadcast_y1, 
		bool broadcast_x2, bool broadcast_y2);
}; 


template <typename Device, typename T, int SGN>
class FallbackLaunchFusedMulAddOp
{
public:
	void operator()(const Device& device,
		T* out,
		const T* x1, const T* y1, const T* x2,
		int64 dims[5],
		uint8 broadcast_masks[5]);
}; 

template <typename Device, typename T, int SGN>
class FallbackLaunchFusedMulAdd2Op
{
public:
	void operator()(const Device& device,
		T* out,
		const T* x1, const T* y1, const T* x2, const T* y2,
		int64 dims[5],
		uint8 broadcast_masks[5]);
}; 


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;

template <typename T, int SGN>
class LaunchFusedMulAddOp<GPUDevice, T, SGN>
{
public:
	typedef void exec_fun(const GPUDevice& device, T* out,
		const T* x1, const T* y1, const T* x2,
		uint64 elements);

	template <int N>
	static void execute(const GPUDevice& device, T* out,
		const T* x1, const T* y1, const T* x2,
		uint64 elements);

	void operator()(const GPUDevice& device,
		T* out,
		const T* x1, const T* y1, const T* x2,
		uint64 elements,
		bool broadcast_x1, bool broadcast_y1, 
		bool broadcast_x2);
}; 

template <typename T, int SGN>
class LaunchFusedMulAdd2Op<GPUDevice, T, SGN>
{
public:
	typedef void exec_fun(const GPUDevice& device,
		T* out,
		const T* x1, const T* y1, const T* x2, const T* y2,
		uint64 elements);
	template <int N>
	static void execute(const GPUDevice& device,
		T* out,
		const T* x1, const T* y1, const T* x2, const T* y2,
		uint64 elements);
	void operator()(const GPUDevice& device,
		T* out,
		const T* x1, const T* y1, const T* x2, const T* y2,
		uint64 elements,
		bool broadcast_x1, bool broadcast_y1, 
		bool broadcast_x2, bool broadcast_y2);
};

template <typename T, int SGN>
class FallbackLaunchFusedMulAddOp<GPUDevice, T, SGN>
{
public:
  void operator()(const GPUDevice& device,
    T* out,
    const T* x1, const T* y1, const T* x2,
    int64 dims[5],
    uint8 broadcast_masks[5]);
}; 

template <typename T, int SGN>
class FallbackLaunchFusedMulAdd2Op<GPUDevice, T, SGN>
{
public:
  void operator()(const GPUDevice& device,
    T* out,
    const T* x1, const T* y1, const T* x2, const T* y2,
    int64 dims[5],
    uint8 broadcast_masks[5]);
}; 
#endif



};

#endif

