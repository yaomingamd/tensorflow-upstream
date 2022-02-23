#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <limits>
#include "fp8/hip_float8.h"
namespace stream_executor {
namespace gpu {

__global__ void broadcast_fp32_kernel(float* dst, int dst_stride, int batches,
                                      float* src, int size) {
  dst += blockIdx.y * 4 * dst_stride + blockIdx.z * dst_stride * batches;
  src += blockIdx.z * size;
  float* dst2 = dst + dst_stride;
  float* dst3 = dst + dst_stride*2;
  float* dst4 = dst + dst_stride*3;
  bool b2 = (blockIdx.y*4+1 < batches);
  bool b3 = (blockIdx.y*4+2 < batches);
  bool b4 = (blockIdx.y*4+3 < batches);
  for (int i = threadIdx.x + blockIdx.x * 256; i < size; i += blockDim.x*gridDim.x) 
  {
  	dst[i] = src[i];
  	if(b2)
  		dst2[i] = src[i];
  	if(b3)
  		dst3[i] = src[i];
  	if(b4)
  		dst4[i] = src[i];
  }
}

void broadcast_fp32(void* stream, float* dst, int dst_stride, int batches, int src_batches,
                    float* src, int size) {
  int x_blocks = (size+255)/256;
  hipLaunchKernelGGL(broadcast_fp32_kernel, dim3(x_blocks, (batches+3)/4, src_batches), min(256, (int)size), 0,
                     (hipStream_t)stream, dst, dst_stride, batches, src, size);
}

__device__ bool nonfinite(__half x) { return __hisnan(x) || __hisinf(x); }
__device__ bool nonfinite(float x) { return isnan(x) || isinf(x); }

template <typename T, int we, int wm>
__global__ void Quant8_inplace(T* _p, int32_t count, bool stoch, bool nanoo, uint32_t seed) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= count) return;
  typedef typename std::conditional<sizeof(T)==2, uint16_t, uint32_t>::type IT;
  typedef typename std::conditional<sizeof(T)==2, __half, float>::type FT;
  IT* p = (IT*) _p;
  FT* fp = (FT*) _p;
  IT x = p[i];
///  constexpr bool nanoo=false;
  uint8_t y;

  T fx = fp[i];
  if(!stoch)
  {
    if(nanoo)
      y = hip_f8_impl::cast_to_f8<wm,we,float,true,true>(float(fx), false);
    else
      y = hip_f8_impl::cast_to_f8<wm,we,float,false,true>(float(fx), false);
  }
  else {
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    if(sizeof(x)==4)
      drop_bits ^= x>>16;
    drop_bits = ((drop_bits & 31)<<11) | (drop_bits>>5);
    drop_bits *= 0x7000149;
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (i*229791) ^ seed);
    if(nanoo)
      y = hip_f8_impl::cast_to_f8<wm,we,float,true,true>(fx, true, rng);
    else
      y = hip_f8_impl::cast_to_f8<wm,we,float,false,true>(fx, true, rng);
  }
  float newfp;
  if(nanoo)
    newfp = hip_f8_impl::cast_from_f8<wm,we,float,true>(y);
  else
    newfp = hip_f8_impl::cast_from_f8<wm,we,float,false>(y);
  fp[i]=T(newfp);
}

template __global__ void Quant8_inplace<__half,5,2>(__half* _p, int32_t count, bool stoch, bool nanoo, uint32_t seed);
template __global__ void Quant8_inplace<__half,4,3>(__half* _p, int32_t count, bool stoch, bool nanoo, uint32_t seed);

void Quant8_inplace(__half* _p, int32_t count, uint32_t seed, hipStream_t stream, bool f152, bool stoch, bool nanoo) {
        auto fun = f152 ? Quant8_inplace<__half,5,2> : Quant8_inplace<__half,4,3>;
        uint32_t dim_a = count; 
        uint32_t grid_a = (dim_a+255)/256;
//        bool stoch=false;//true;
        hipLaunchKernelGGL(fun,
           dim3(grid_a,1,1), dim3(256,1,1), 0, stream, _p, dim_a, stoch, nanoo, seed);
}


__global__ void inplace_fp16_to_bf16_kernel(half* dst, int nElements)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  for(int i=id; i<nElements; i+=blockDim.x*gridDim.x)
  {
    float x = __half2float(dst[i]);
    uint32_t v = reinterpret_cast<uint32_t&>(x);
    *(uint16_t*)(dst+i) = v>>16;
  }
}

__global__ void inplace_bf16_to_fp16_kernel(half* dst, int nElements)
{
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  for(int i=id; i<nElements; i+=blockDim.x*gridDim.x)
  {
    uint16_t x = *(uint16_t*)(dst+i);
    uint32_t y = ((uint32_t)x)<<16;
    float fy = reinterpret_cast<float&>(y);
    dst[i] = __float2half(fy);
  }
}

void inplace_fp16_to_bf16(void* data, int nElements, hipStream_t stream)
{
  int blocks = min(1024, (nElements+255)/256);
  hipLaunchKernelGGL(inplace_fp16_to_bf16_kernel, dim3(blocks, 1, 1), dim3(256, 1, 1), 0, stream, (half*)data, nElements);
}

void inplace_bf16_to_fp16(void* data, int nElements, hipStream_t stream)
{
  int blocks = min(1024, (nElements+255)/256);
  hipLaunchKernelGGL(inplace_bf16_to_fp16_kernel, dim3(blocks, 1, 1), dim3(256, 1, 1), 0, stream, (half*)data, nElements);
}


};  // namespace gpu
};  // namespace stream_executor

