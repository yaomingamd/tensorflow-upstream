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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"
#include "tensorflow/core/kernels/cwise_op_fma.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, int N, int SGN>
__global__ void CwiseFusedMulAddKernel(GpuLaunchConfig cfg,
	T* out, const T* x1, const T* y1, const T* x2)
{
	constexpr bool broadcast_x1 = (N&1);
	constexpr bool broadcast_y1 = (N&2);
	constexpr bool broadcast_x2 = (N&4);
  	GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
		T m1 = x1[broadcast_x1 ? 0 : i]*y1[broadcast_y1 ? 0 : i];
		T m2 = x2[broadcast_x2 ? 0 : i];
		if(SGN==1)
			out[i] = m1+m2;
		else if(SGN==-1)
			out[i] = m1-m2;
		else
			out[i] = m2-m1;
	}
}

template <typename T, int N, int SGN>
__global__ void CwiseFusedMulAdd2Kernel(GpuLaunchConfig cfg,
	T* out, const T* x1, const T* y1, const T* x2, const T* y2)
{
	constexpr bool broadcast_x1 = (N&1);
	constexpr bool broadcast_y1 = (N&2);
	constexpr bool broadcast_x2 = (N&4);
	constexpr bool broadcast_y2 = (N&8);
  	GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
  		T m1 = x1[broadcast_x1 ? 0 : i]*y1[broadcast_y1 ? 0 : i];
		T m2 = x2[broadcast_x2 ? 0 : i]*y2[broadcast_y2 ? 0 : i];
		out[i] = (SGN>0) ? m1+m2 : m1-m2;
	}
}

template <typename T, int SGN>
template <int N>
void LaunchFusedMulAddOp<GPUDevice, T, SGN>::execute(const GPUDevice& device, T* out,
	const T* x1, const T* y1, const T* x2,
	uint64 elements)
{
	//printf("LaunchFusedMulAddOp<GPU,%d>(%ld)\n", N, elements);
    auto config = GetGpuLaunchConfig(elements, device);
    TF_CHECK_OK(GpuLaunchKernel(
        CwiseFusedMulAddKernel<T,N,SGN>, config.block_count, config.thread_per_block, 0,
        device.stream(), config, out,
        x1, y1, x2));
}

template <typename T, int SGN>
void LaunchFusedMulAddOp<GPUDevice, T, SGN>::operator()(const GPUDevice& device,
	T* out,
	const T* x1, const T* y1, const T* x2,
	uint64 elements,
	bool broadcast_x1, bool broadcast_y1, 
	bool broadcast_x2)
{
    int index = (broadcast_x1?1:0)+(broadcast_y1?2:0)+(broadcast_x2?4:0);
    exec_fun* execs[8]={
		&execute<0>,
		&execute<1>,
		&execute<2>,
		&execute<3>,
		&execute<4>,
		&execute<5>,
		&execute<6>,
		&execute<7>
	};
	execs[index](device, out, x1,y1,x2, elements);
}


template <typename T, int SGN>
template <int N>
void LaunchFusedMulAdd2Op<GPUDevice, T, SGN>::execute(const GPUDevice& device,
	T* out,
	const T* x1, const T* y1, const T* x2, const T* y2,
	uint64 elements)
{
	//printf("LaunchFusedMulAdd2Op<GPU,%d>(%ld)\n", N, elements);
    auto config = GetGpuLaunchConfig(elements, device);
    TF_CHECK_OK(GpuLaunchKernel(
        CwiseFusedMulAdd2Kernel<T,N,SGN>, config.block_count, config.thread_per_block, 0,
        device.stream(), config,  out,
        x1, y1, x2, y2));
}

template <typename T, int SGN>
void LaunchFusedMulAdd2Op<GPUDevice, T, SGN>::operator()(const GPUDevice& device,
	T* out,
	const T* x1, const T* y1, const T* x2, const T* y2,
	uint64 elements,
	bool broadcast_x1, bool broadcast_y1, 
	bool broadcast_x2, bool broadcast_y2)
{
    int index = (broadcast_x1?1:0)+(broadcast_y1?2:0)+(broadcast_x2?4:0)+(broadcast_y2?8:0);
	exec_fun* execs[16]={
		&execute<0>,
		&execute<1>,
		&execute<2>,
		&execute<3>,
		&execute<4>,
		&execute<5>,
		&execute<6>,
		&execute<7>,
		&execute<8>,
		&execute<9>,
		&execute<10>,
		&execute<11>,
		&execute<12>,
		&execute<13>,
		&execute<14>,
		&execute<15>,
	};
	execs[index](device, out, x1,y1,x2,y2, elements);
}

template <int N>
struct Fallback_FMA_Arg {
	int32 shifts[N][4];
	int32 dims[5];  // uint64 has high overhead
	uint32 broadcast_mask;
};

template <typename T, int SGN>
__global__ void FallbackLaunchFusedMulAddKernel(
	T* out,
    const T* x1, const T* y1, const T* x2,
    Fallback_FMA_Arg<4> arg)
{
  int32 u = blockIdx.z;
  int32 v = blockIdx.y;
  int32 z = threadIdx.z + blockIdx.x*blockDim.z;
  if(z>=arg.dims[2])
  	return;
  
  out += arg.shifts[0][1]*z + arg.shifts[0][2]*v + arg.shifts[0][3]*u;
  x1 +=  arg.shifts[1][1]*z + arg.shifts[1][2]*v + arg.shifts[1][3]*u;
  y1 +=  arg.shifts[2][1]*z + arg.shifts[2][2]*v + arg.shifts[2][3]*u;
  x2 +=  arg.shifts[3][1]*z + arg.shifts[3][2]*v + arg.shifts[3][3]*u;

  out += threadIdx.y*arg.shifts[0][0];
  x1  += threadIdx.y*arg.shifts[1][0];
  y1  += threadIdx.y*arg.shifts[2][0];
  x2  += threadIdx.y*arg.shifts[3][0];

  for(int32 y=threadIdx.y; y<arg.dims[1]; y+=blockDim.y) {
    for(int32 x=threadIdx.x; x<arg.dims[0]; x+=blockDim.x) {
    	T m1 = x1[(arg.broadcast_mask&1) ? x : 0] 
			 * y1[(arg.broadcast_mask&2) ? x : 0];
   		T m2 = x2[(arg.broadcast_mask&4) ? x : 0];
   		if(SGN==1)
   			out[x] = m1+m2;
   		else if(SGN==-1)
   			out[x] = m1-m2;
   		else
   			out[x] = m2-m1;
    }
    out += blockDim.y*arg.shifts[0][0];
    x1  += blockDim.y*arg.shifts[1][0];
    y1  += blockDim.y*arg.shifts[2][0];
    x2  += blockDim.y*arg.shifts[3][0];
  }
 }


template <typename T, int SGN>
__global__ void FallbackLaunchFusedMulAddKernel2D(
	T* out,
    const T* x1, const T* y1, const T* x2,
    Fallback_FMA_Arg<4> arg)
{
	out += blockIdx.x*arg.shifts[0][0];
	x1  += blockIdx.x*arg.shifts[1][0];
	y1  += blockIdx.x*arg.shifts[2][0];
	x2  += blockIdx.x*arg.shifts[3][0];

	for(int32 x=threadIdx.x; x<arg.dims[0]; x+=blockDim.x) {
		T m1 = x1[(arg.broadcast_mask&1) ? x : 0] 
		 * y1[(arg.broadcast_mask&2) ? x : 0];
		T m2 = x2[(arg.broadcast_mask&4) ? x : 0];
		if(SGN==1)
			out[x] = m1+m2;
		else if(SGN==-1)
			out[x] = m1-m2;
		else
			out[x] = m2-m1;
	}
}

template <typename T, int SGN>
__global__ void FallbackLaunchFusedMulAdd2Kernel(
	T* out,
    const T* x1, const T* y1, const T* x2, const T* y2,
    Fallback_FMA_Arg<5> arg)
{
  int32 u = blockIdx.z;
  int32 v = blockIdx.y;
  int32 z = threadIdx.z + blockIdx.x*blockDim.z;
  if(z>=arg.dims[2])
  	return;
  
  out += arg.shifts[0][1]*z + arg.shifts[0][2]*v + arg.shifts[0][3]*u;
  x1 +=  arg.shifts[1][1]*z + arg.shifts[1][2]*v + arg.shifts[1][3]*u;
  y1 +=  arg.shifts[2][1]*z + arg.shifts[2][2]*v + arg.shifts[2][3]*u;
  x2 +=  arg.shifts[3][1]*z + arg.shifts[3][2]*v + arg.shifts[3][3]*u;
  y2 +=  arg.shifts[4][1]*z + arg.shifts[4][2]*v + arg.shifts[4][3]*u;

  out += threadIdx.y*arg.shifts[0][0];
  x1  += threadIdx.y*arg.shifts[1][0];
  y1  += threadIdx.y*arg.shifts[2][0];
  x2  += threadIdx.y*arg.shifts[3][0];
  y2  += threadIdx.y*arg.shifts[4][0];

  for(int32 y=threadIdx.y; y<arg.dims[1]; y+=blockDim.y) {
    for(int32 x=threadIdx.x; x<arg.dims[0]; x+=blockDim.x) {
    	T m1 = x1[(arg.broadcast_mask&1) ? x : 0] 
			* y1[(arg.broadcast_mask&2) ? x : 0];
   		T m2 = x2[(arg.broadcast_mask&4) ? x : 0]
   			* y2[(arg.broadcast_mask&8) ? x : 0];
	    out[x] = (SGN>0) ? m1+m2 : m1-m2;
    }
    out += blockDim.y*arg.shifts[0][0];
    x1  += blockDim.y*arg.shifts[1][0];
    y1  += blockDim.y*arg.shifts[2][0];
    x2  += blockDim.y*arg.shifts[3][0];
    y2  += blockDim.y*arg.shifts[4][0];
  }
}

template <typename T, int SGN>
__global__ void FallbackLaunchFusedMulAdd2Kernel2D(
	T* out,
    const T* x1, const T* y1, const T* x2, const T* y2,
    Fallback_FMA_Arg<5> arg)
{
	out += blockIdx.x*arg.shifts[0][0];
	x1  += blockIdx.x*arg.shifts[1][0];
	y1  += blockIdx.x*arg.shifts[2][0];
	x2  += blockIdx.x*arg.shifts[3][0];
	y2  += blockIdx.x*arg.shifts[4][0];

	for(int32 x=threadIdx.x; x<arg.dims[0]; x+=blockDim.x) {
		T m1 = x1[(arg.broadcast_mask&1) ? x : 0] 
		     * y1[(arg.broadcast_mask&2) ? x : 0];
		T m2 = x2[(arg.broadcast_mask&4) ? x : 0]
		     * y2[(arg.broadcast_mask&8) ? x : 0];
		out[x] = (SGN>0) ? m1+m2 : m1-m2;
	}
}

template <typename T, int SGN>
void FallbackLaunchFusedMulAddOp<GPUDevice, T, SGN>::operator()(const GPUDevice& device,
    T* out,
    const T* x1, const T* y1, const T* x2,
    int64 dims[5],
    uint8 broadcast_masks[5])
{
  //printf("FallbackFusedMulAdd %d %d %d\n", broadcast_masks[0], broadcast_masks[1], broadcast_masks[2]);
  //printf("Dims %ld %ld %ld %ld %ld\n", dims[0], dims[1], dims[2], dims[3], dims[4]);

  Fallback_FMA_Arg<4> arg;
  for(int i=0; i<5; i++)
  	arg.dims[i] = (int32)dims[i];
  for(int y=0; y<4; y++)
  {
  	int b0, b1, b2;
  	if(y==0)
  		b0=b1=b2=1;
  	else {
  		b0 = broadcast_masks[0] & (1<<(y-1));
  		b1 = broadcast_masks[1] & (1<<(y-1));
  		b2 = broadcast_masks[2] & (1<<(y-1));
  	}
  	int stride = (b0 ? dims[0] : 1) * (b1 ? dims[1] : 1);
  	arg.shifts[y][1] = b2 ? stride : 0;
  	for(int x=1; x<3; x++) {
  		int b  = (y==0) ? 1 : (broadcast_masks[x+1] & (1<<(y-1)));
  		int bn = (y==0) ? 1 : (broadcast_masks[x+2] & (1<<(y-1)));
  		if(b)
  			stride *= dims[x+1];
  		arg.shifts[y][x+1] = bn ? stride : 0;
  	}
  }

  arg.broadcast_mask = broadcast_masks[0];

  arg.shifts[0][0] = dims[0];
  arg.shifts[1][0] = (broadcast_masks[1] & 1) ? ((broadcast_masks[0]&1) ? dims[0] : 1) : 0;
  arg.shifts[2][0] = (broadcast_masks[1] & 2) ? ((broadcast_masks[0]&2) ? dims[0] : 1) : 0;
  arg.shifts[3][0] = (broadcast_masks[1] & 4) ? ((broadcast_masks[0]&4) ? dims[0] : 1) : 0;

  // TODO: cover the case where dim[0]>>256 but dim[2,3,4] are small 
  // (likely performance issue due to a small number of workgroups)
  if(dims[2]==1 && dims[3]==1 && dims[4]==1) {
	TF_CHECK_OK(GpuLaunchKernel(
		FallbackLaunchFusedMulAddKernel2D<T, SGN>, 
    	dim3(dims[1], 1, 1),
    	dim3(dims[0]>256 ? 256 : dims[0], 1, 1),
    	0,
    	device.stream(), out,
    	x1, y1, x2, 
    	arg
    	));
  } else {
	int block_x = min(256, dims[0]);
	int block_y = min(256/block_x, dims[1]);
	int block_z = min(256/(block_x*block_y), dims[2]);
	int grid_x = (dims[2]+block_z-1)/block_z;
	int grid_y = dims[3];
	int grid_z = dims[4];
	//printf("%d %d %d x %d %d %d\n", block_x, block_y, block_z, grid_x, grid_y, grid_z);

	TF_CHECK_OK(GpuLaunchKernel(
		FallbackLaunchFusedMulAddKernel<T, SGN>, 
    	dim3(grid_x, grid_y, grid_z),
    	dim3(block_x, block_y, block_z),
    	0,
    	device.stream(), out,
    	x1, y1, x2, 
    	arg
    	));
	}

}

template <typename T, int SGN>
void FallbackLaunchFusedMulAdd2Op<GPUDevice, T, SGN>::operator()(const GPUDevice& device,
    T* out,
    const T* x1, const T* y1, const T* x2, const T* y2,
    int64 dims[5],
    uint8 broadcast_masks[5])
{
  //printf("FallbackFusedMulAdd2 %d %d %d\n", broadcast_masks[0], broadcast_masks[1], broadcast_masks[2]);
  int block_x = min(256, dims[0]);
  int block_y = min(256/block_x, dims[1]);
  int block_z = min(256/(block_x*block_y), dims[2]);
  int grid_x = (dims[2]+block_z-1)/block_z;
  int grid_y = dims[3];
  int grid_z = dims[4];

  Fallback_FMA_Arg<5> arg;
  for(int i=0; i<5; i++)
  	arg.dims[i] = (int32)dims[i];
  for(int y=0; y<5; y++)
  {
  	int b0, b1, b2;
  	if(y==0)
  		b0=b1=b2=1;
  	else {
  		b0 = broadcast_masks[0] & (1<<(y-1));
  		b1 = broadcast_masks[1] & (1<<(y-1));
  		b2 = broadcast_masks[2] & (1<<(y-1));
  	}
  	int stride = (b0 ? dims[0] : 1) * (b1 ? dims[1] : 1);
  	arg.shifts[y][1] = b2 ? stride : 0;
  	for(int x=1; x<3; x++) {
  		int b  = (y==0) ? 1 : (broadcast_masks[x+1] & (1<<(y-1)));
  		int bn = (y==0) ? 1 : (broadcast_masks[x+2] & (1<<(y-1)));
  		if(b)
  			stride *= dims[x+1];
  		arg.shifts[y][x+1] = bn ? stride : 0;
  	}
  }

  arg.broadcast_mask = broadcast_masks[0];

  arg.shifts[0][0] = dims[0];
  arg.shifts[1][0] = (broadcast_masks[1] & 1) ? ((broadcast_masks[0]&1) ? dims[0] : 1) : 0;
  arg.shifts[2][0] = (broadcast_masks[1] & 2) ? ((broadcast_masks[0]&2) ? dims[0] : 1) : 0;
  arg.shifts[3][0] = (broadcast_masks[1] & 4) ? ((broadcast_masks[0]&4) ? dims[0] : 1) : 0;
  arg.shifts[4][0] = (broadcast_masks[1] & 8) ? ((broadcast_masks[0]&8) ? dims[0] : 1) : 0;
  TF_CHECK_OK(GpuLaunchKernel(
    FallbackLaunchFusedMulAdd2Kernel<T, SGN>, 
    	dim3(grid_x, grid_y, grid_z),
    	dim3(block_x, block_y, block_z),
    	0,
    	device.stream(), out,
    	x1, y1, x2, y2,
    	arg
    	));
}


#define INSTANTIATE_FMA(X, S) \
template void LaunchFusedMulAddOp<GPUDevice, X, S>::operator()(          \
		const GPUDevice& device, X* out,                                 \
		const X* x1, const X* y1, const X* x2,                           \
		uint64 elements, bool bc1, bool bc2, bool bc3);                  \
template void LaunchFusedMulAdd2Op<GPUDevice, X, S>::operator()(         \
		const GPUDevice& device, X* out,                                 \
		const X* x1, const X* y1, const X* x2, const X* y2,              \
		uint64 elements, bool bc1, bool bc2, bool bc3, bool bc4);        \
template void FallbackLaunchFusedMulAddOp<GPUDevice, X, S>::operator()(  \
	const GPUDevice& device, X* out,                                     \
    const X* x1, const X* y1, const X* x2,                               \
    int64 dims[5], uint8 broadcast_masks[5]);                            \
template void FallbackLaunchFusedMulAdd2Op<GPUDevice, X, S>::operator()( \
	const GPUDevice& device, X* out,                                     \
    const X* x1, const X* y1, const X* x2, const X* y2,                  \
    int64 dims[5], uint8 broadcast_masks[5]);   


INSTANTIATE_FMA(Eigen::half, 1);
INSTANTIATE_FMA(float, 1);
INSTANTIATE_FMA(double, 1);
INSTANTIATE_FMA(Eigen::half, 0);
INSTANTIATE_FMA(float, 0);
INSTANTIATE_FMA(double, 0);
INSTANTIATE_FMA(Eigen::half, -1);
INSTANTIATE_FMA(float, -1);
INSTANTIATE_FMA(double, -1);

};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM





