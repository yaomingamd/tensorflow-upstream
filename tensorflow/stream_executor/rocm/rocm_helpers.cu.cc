#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <limits>
#include <math.h>
typedef int index_t;
#include "number.hpp"
//#include "amd_xdlops.hpp"
#include "float_type.hpp"

using ck::half4_t;
using ck::float4_t;
using ck::float16_t;

extern "C" __device__ float4_t llvm_intrin_amdgcn_mfma_f32_16x16x16f16(
    half4_t, half4_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x16f16");

extern "C" __device__ float4_t my_llvm_intrin_amdgcn_mfma_f32_4x4x4f16(
    half4_t, half4_t, float4_t, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x4f16");

namespace stream_executor {
namespace gpu {

__global__ void broadcast_fp32_kernel(float* dst, int dst_stride, int batches,
                                      float* src, int size) {
  dst += blockIdx.y * 4 * dst_stride;
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

void broadcast_fp32(void* stream, float* dst, int dst_stride, int batches,
                    float* src, int size) {
  int x_blocks = (size+255)/256;
  hipLaunchKernelGGL(broadcast_fp32_kernel, dim3(x_blocks, (batches+3)/4, 1), min(256, (int)size), 0,
                     (hipStream_t)stream, dst, dst_stride, batches, src, size);
}

};  // namespace gpu
};  // namespace stream_executor


__device__ inline void transpose4(float* c) {

  int blk = threadIdx.x & 60;
  int lane = threadIdx.x & 3;
  float t[4];
  t[0] = __shfl(c[lane], blk);
  t[1] = __shfl(c[lane], blk+1);
  t[2] = __shfl(c[lane], blk+2);
  t[3] = __shfl(c[lane], blk+3);
  c[0] = t[0];
  c[1] = t[1];
  c[2] = t[2];
  c[3] = t[3];
}

__device__ inline void transpose4(half4_t& c) {
  int blk = threadIdx.x & 60;
  int lane = threadIdx.x & 3;
  float t[4];
  t[0] = __shfl(float(c[lane]), blk);
  t[1] = __shfl(float(c[lane]), blk+1);
  t[2] = __shfl(float(c[lane]), blk+2);
  t[3] = __shfl(float(c[lane]), blk+3);
  c[0] = t[0];
  c[1] = t[1];
  c[2] = t[2];
  c[3] = t[3];
}


__device__ inline void manual_intrin_mfma_f32_4x4x4f16(const half4_t& a, const half4_t& b, float4_t* c, int mode) {
  //int blk = threadIdx.x & 60;
  //int lane = threadIdx.x & 3;
  int lane = __lane_id() & 3;
  //half4_t temp;
  //temp[0] = __shfl(float(a[lane]), blk);
  //temp[1] = __shfl(float(a[lane]), blk+1);
  //temp[2] = __shfl(float(a[lane]), blk+2);
  //temp[3] = __shfl(float(a[lane]), blk+3);
  for(int i=0; i<4; i++)  {
    float va = (float)b[i];
    float vb = (float)a[i];
    if(mode)
      (*c)[i] += __shfl_xor(va, lane) * b[0] + __shfl_xor(va, lane^1)*b[1] + __shfl_xor(va, lane^2)*b[2] + __shfl_xor(va, lane^3)*b[3];
    else
      (*c)[i] += a[0] * __shfl_xor(vb, lane) +  a[1] * __shfl_xor(vb, lane^1) +  a[2] * __shfl_xor(vb, lane^2) +  a[3] * __shfl_xor(vb, lane^3);
  }
}

//__global__ void convolve_kernel(const __half* data, const __half* filter, __half* output, int mode)
const int log_strip_width = 6;
const int strip_width = 1 << log_strip_width;


// required that blockDim.y * strip_width >= xo 
// e.g. strip_width=64, blockDim.y=2
// strip_width=32, blockDim.y=4
__launch_bounds__(512) __global__ void convolve_kernel(const __half* data, const __half* filter, __half* output, uint64_t x_block_dim) {
  int wi=230, hi=230, n=256, c=3, x=7, y=7, k=64, u=2, v=2;
  int wo = (wi-x+1)/u, ho=(hi-y+1)/v;

  int b = blockIdx.z;
  int x_block_id = blockIdx.x;

  int w_eff = strip_width - 4;

  int z_block_id = threadIdx.x >> log_strip_width;
  int oc_per_block = blockDim.x >> log_strip_width;

  int px = (threadIdx.x & (strip_width-1)) + blockIdx.y*strip_width;
  int lane = px & (strip_width-1);
  int group = blockIdx.y; //px >> log_strip_width;

  int offset_in = b*c*wi*hi + group*w_eff*u; 
  int offset_out = b*k*wo*ho+ group*w_eff + lane;
  data += offset_in;
  output += offset_out;
  int blk = lane >> 2;
  int tid = lane & 3;
  data += blk*8;
  __shared__ half4_t share[64*3*4 + 3*64*16];
  half4_t* trans_filter = &share[64*3*4];

  if(threadIdx.x < 256)
  {
    int xx = (threadIdx.x) & 3;
    int oc = (threadIdx.x >> 2) & 63;
    for(int ic=0; ic<3; ic++) {      
        half4_t reg_c1={0,0,0,0}, reg_c2={0,0,0,0}, reg_d1={0,0,0,0}, reg_d2={0,0,0,0};
        const __half* pf = filter + ic*x*y + oc*c*x*y; 
        pf += xx*2;
        reg_c1[0] = pf[0];
        reg_c1[1] = pf[7];
        reg_c1[2] = pf[14];
        reg_c1[3] = pf[21];
        reg_c2[0] = pf[28];
        reg_c2[1] = pf[35];
        reg_c2[2] = pf[42];
        if(xx<3) {
          reg_d1 = half4_t{pf[1], pf[8], pf[15], pf[22]};
          reg_d2 = half4_t{pf[29], pf[36], pf[43], 0};
        }
        trans_filter[oc*16*3+xx*12+ic*4]=reg_c1;
        trans_filter[oc*16*3+xx*12+ic*4+1]=reg_c2;
        trans_filter[oc*16*3+xx*12+ic*4+2]=reg_d1;
        trans_filter[oc*16*3+xx*12+ic*4+3]=reg_d2;
    }
  }
  
  __syncthreads();

  for(int py=x_block_id*x_block_dim; py<ho && py<(x_block_id+1)*x_block_dim; py++) 
  {
    __syncthreads();
    
    if(z_block_id<3) {
      int nc = z_block_id;
      const __half* ptr = data + py*u*wi + nc*wi*hi + tid*2;
      half4_t reg_a1={0,0,0,0}, reg_a2={0,0,0,0}, reg_b1={0,0,0,0}, reg_b2={0,0,0,0};
      if(group*w_eff*u+tid*u<wi) 
      {
        reg_a1={ptr[0],     ptr[wi],    ptr[2*wi],   ptr[3*wi]};
        reg_b1={ptr[1],     ptr[wi+1],  ptr[2*wi+1], ptr[3*wi+1]};
        reg_a2={ptr[4*wi+0],ptr[5*wi],  ptr[6*wi],   ptr[7*wi]};
        reg_b2={ptr[4*wi+1],ptr[5*wi+1],ptr[6*wi+1], ptr[7*wi+1]};
      }
      share[lane*4*3 + nc*4 + 0] = reg_a1;
      share[lane*4*3 + nc*4 + 1] = reg_a2;
      share[lane*4*3 + nc*4 + 2] = reg_b1;
      share[lane*4*3 + nc*4 + 3] = reg_b2;
    }
    
    __syncthreads();

    // compiler preloads reg_a1...reg_b2 into registers for all 3 values of nc
    // (they don't change) - this takes 24 VGPRs
    // reg_c1...reg_d2 must be read from LDS every time
    // read 3*4*sizeof(half4) = 3*4*8 = 96 bytes from LDS per output point
    // 205M output points, total traffic 19 GB
    // That seems to be the biggest remaining bottleneck
    // reading 96 bytes takes substantially longer than executing 12 MFMA's per output point
    for(int out_plane=z_block_id; out_plane<k; out_plane+=oc_per_block) {
      float sum = 0;
      const half4_t* pfr = trans_filter+out_plane*16*3+tid*12;
      const half4_t* psh = share+lane*4*3;
      float4_t acc={0,0,0,0};
      for(int nc=0; nc<3; nc++) {
        half4_t reg_a1 = psh[0];
        half4_t reg_a2 = psh[1];
        half4_t reg_b1 = psh[2];
        half4_t reg_b2 = psh[3];
        
        half4_t reg_c1 = pfr[0];
        half4_t reg_c2 = pfr[1];
        half4_t reg_d1 = pfr[2];
        half4_t reg_d2 = pfr[3];
        pfr += 4;
        psh += 4;

        acc = __builtin_amdgcn_mfma_f32_4x4x4f16(reg_c1, reg_a1, acc, 4, 0, 0);
        acc = __builtin_amdgcn_mfma_f32_4x4x4f16(reg_c2, reg_a2, acc, 4, 0, 0);
        acc = __builtin_amdgcn_mfma_f32_4x4x4f16(reg_d1, reg_b1, acc, 4, 0, 0);
        acc = __builtin_amdgcn_mfma_f32_4x4x4f16(reg_d2, reg_b2, acc, 4, 0, 0);
      }
      sum+=acc[0];
      sum+=__shfl_down((float)acc[1], 1);
      sum+=__shfl_down((float)acc[2], 2);
      sum+=__shfl_down((float)acc[3], 3);
      if(group*w_eff+lane<wo && lane<w_eff)
          output[out_plane*wo*ho+py*wo] = __float2half(sum);
    }
  }
}

void doConvolveManual_fw7x7x2(const __half* data, const __half* filter, __half* output, hipStream_t stream)
{
  int wi=230, hi=230, n=256, c=3, x=7, y=7, k=64, u=2, v=2;
  int wo = (wi-x+1)/u, ho=(hi-y+1)/v;
  static int call_count=0;
// printf("%d\n", call_count&7);
  int x_dim = 16;
  hipLaunchKernelGGL(convolve_kernel, dim3((ho+x_dim-1)/x_dim, (ho+strip_width-1)/strip_width, n), dim3(512, 1, 1), 0,
               stream, data, filter, output, x_dim);
  call_count++;

/*
    std::vector<__half> temp, host_data, host_filter;
    temp.resize(wo*ho*k*n);
    host_data.resize(wi*hi*c*n);
    host_filter.resize(x*y*k);
    hipMemcpy(&host_data[0], data, host_data.size()*2, hipMemcpyDeviceToHost);
    hipMemcpy(&host_filter[0], filter, host_filter.size()*2, hipMemcpyDeviceToHost);
    for(int b=0; b<n; b++) {
	for(int j=0; j<k; j++) {
	    for(int py=0; py<ho; py++) {
       for(int px=0; px<wo; px++) {
        float sum=0;
         for(int nc=0; nc<c; nc++)
          for(int dy=0; dy<y; dy++)
              for(int dx=0; dx<x; dx++)
             sum+=float(host_filter[dx+dy*x+nc*x*y+b*x*y*c])*float(host_data[px*u+dx+(py*v+dy)*wi+nc*wi*hi+b*c*wi*hi]);
            temp[px+py*wo+j*wo*ho+b*k*wo*ho] = __float2half(sum);
       }
     }
 }
    }
    hipMemcpy(output, &temp[0], temp.size()*2, hipMemcpyHostToDevice);
    */
}
/*
__device__ void
intrin_mfma_f32_16x16x16f16(const half4_t* reg_a, const half4_t* reg_b, float4_t* reg_c)
{
    reg_c[0] = llvm_intrin_amdgcn_mfma_f32_16x16x16f16(reg_a[0], reg_b[0], reg_c[0], 0, 0, 0);
}
*/

__device__ void testatomicAdd(float* p, float a) 
{ 
//  p[0] += a; 
  atomicAdd(p, a);
}


__launch_bounds__(1024) __global__ void convolve_bwdw_1x1x1_xdlops_v3(int n, int k, int c, int a, const __half* p1, const __half* p2, float* out)
{
  int b = blockIdx.x;
  p1 += a*b*c;
  p2 += a*b*k;

  int c_scaled = c>>5;
  int k_scaled = k>>5;

  int tid = threadIdx.x;
  //int row = tid>>2; // 0..15
  //int px_offset=4*(tid-row*4);
  int row = tid & 15;
  int px_offset = 4*(tid >> 4); // 0..12
/*
  input 1: 64 x 3136 (m,k)
  input 2: 64 x 3136 (k,n)
  each wavefront does  16x3136 by 16x3136
  each 16-strip gets read 4 times
*/
  __shared__ half4_t reg_cache[64*16*2];
  
  bool cached_a = (blockDim.y>1);
  bool cached_b = (blockDim.z>1);

  half4_t* reg_a_cache = &reg_cache[0];
  half4_t* reg_b_cache = &reg_cache[0];//64*blockDim.z*2];
  if(cached_a)
    reg_b_cache+=64*blockDim.z*2;
  for(int j = 0; j < k_scaled; j+=blockDim.z) 
    for(int nc = 0; nc<c_scaled; nc+=blockDim.y) {
      // all threads with the same threadIdx.x and threadIdx.y (up to 16) share the same reg_b and reg_b1
      // all threads with the same threadIdx.x and threadIdx.z share reg_a and reg_a1
      // worst case scenario 64x16x1  blockDim.y=16 need to read 64x16 x2 half4 = 64*16*8*2 = 16 KB
      int true_nc = (nc+threadIdx.y)*32+row;
      int true_j = (j+threadIdx.z)*32+row;
      float4_t reg_c00={0,0,0,0};
      float4_t reg_c01={0,0,0,0};
      float4_t reg_c10={0,0,0,0};
      float4_t reg_c11={0,0,0,0};
      for(int offset=0; offset<a; offset+=16) {
        const __half* pp1 = p1 + true_nc*a;
        const __half* pp2 = p2 + true_j*a;
        const __half* pp3 = p1 + (true_nc+16)*a;
        const __half* pp4 = p2 + (true_j+16)*a;
        half4_t reg_a={0,0,0,0}, reg_b={0,0,0,0}, reg_a1={0,0,0,0}, reg_b1={0,0,0,0};
        /*
        reg_a = *(half4_t*)(pp1 + offset + px_offset);
        reg_b = *(half4_t*)(pp2 + offset + px_offset);
        if(offset+px_offset>a-4) {
          for(int t=0; t<4; t++) {
            if(offset+px_offset+t>=a) {
              reg_a[t]=0;
              reg_b[t]=0;
            }
          }
        }
        */
        __syncthreads();
        if(threadIdx.z==0) {
          if(offset+px_offset+4<=a) 
          {
            reg_b = *(half4_t*)(pp1 + offset + px_offset);
            reg_b1 = *(half4_t*)(pp3 + offset + px_offset);
          }
          if(cached_b) {
            reg_b_cache[threadIdx.y*64*2+threadIdx.x*2+0] = reg_b;
            reg_b_cache[threadIdx.y*64*2+threadIdx.x*2+1] = reg_b1;
          }
        }
        if(threadIdx.y==0) {
          if(offset+px_offset+4<=a) 
          {
            reg_a = *(half4_t*)(pp2 + offset + px_offset);
            reg_a1 = *(half4_t*)(pp4 + offset + px_offset);
            if(cached_a) {
              reg_a_cache[threadIdx.z*64*2+threadIdx.x*2+0] = reg_a;
              reg_a_cache[threadIdx.z*64*2+threadIdx.x*2+1] = reg_a1;
            }
          }
        }
        __syncthreads();
        if(threadIdx.y!=0) {
          reg_a = reg_a_cache[threadIdx.z*64*2+threadIdx.x*2+0];
          reg_a1 = reg_a_cache[threadIdx.z*64*2+threadIdx.x*2+1];
        }
        if(threadIdx.z!=0) {
          reg_b = reg_b_cache[threadIdx.y*64*2+threadIdx.x*2+0];
          reg_b1 = reg_b_cache[threadIdx.y*64*2+threadIdx.x*2+1];
        }
        //reg_a[0]+=1;
        //reg_b[0]+=1;
        reg_c00 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b, reg_c00, 0, 0, 0);
        reg_c01 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a1, reg_b, reg_c01, 0, 0, 0);
        reg_c10 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b1, reg_c10, 0, 0, 0);
        reg_c11 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a1, reg_b1, reg_c11, 0, 0, 0);
      }
      // 64 threads x 4 entries give me 16x16 values, one row per 4 threads
      int out_nc = (nc+threadIdx.y)*32 + row;
      int out_j = (j+threadIdx.z)*32 + px_offset;
      testatomicAdd(out+out_nc*k+(out_j),   reg_c00[0]);
      testatomicAdd(out+out_nc*k+(out_j+1), reg_c00[1]);
      testatomicAdd(out+out_nc*k+(out_j+2), reg_c00[2]);
      testatomicAdd(out+out_nc*k+(out_j+3), reg_c00[3]);
      testatomicAdd(out+out_nc*k+(out_j+16), reg_c01[0]);
      testatomicAdd(out+out_nc*k+(out_j+17), reg_c01[1]);
      testatomicAdd(out+out_nc*k+(out_j+18), reg_c01[2]);
      testatomicAdd(out+out_nc*k+(out_j+19), reg_c01[3]);
      testatomicAdd(out+(out_nc+16)*k+(out_j),   reg_c10[0]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+1), reg_c10[1]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+2), reg_c10[2]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+3), reg_c10[3]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+16), reg_c11[0]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+17), reg_c11[1]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+18), reg_c11[2]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+19), reg_c11[3]);
    }
}

// assumptions:
// c>32, k>32
// blockDim.y=c/32
// blockDim.z=k/32
// therefore 2<=blockDim.y,blockDim.z<=8
__launch_bounds__(1024) __global__ void convolve_bwdw_1x1x1_xdlops(int n, int k, int c, int a, const __half* p1, const __half* p2, float* out)
{
  int b = blockIdx.x;
  p1 += a*b*c;
  p2 += a*b*k;

  int c_scaled = c>>5;
  int k_scaled = k>>5;

  int tid = threadIdx.x;
  //int row = tid>>2; // 0..15
  //int px_offset=4*(tid-row*4);
  int row = tid & 15;
  int px_offset = 4*(tid >> 4); // 0..12
/*
  for 64x64:
  reg_cache 16 rows: 0.785
  8 rows: 0.909 0.787
  4 rows: 0.915 
*/
  __shared__ half4_t reg_cache[64*4*4];
  
  bool cached_a = true;
  bool cached_b = true;

  half4_t* reg_a_cache = &reg_cache[0];
  half4_t* reg_b_cache = &reg_cache[0];//64*blockDim.z*2];
  if(cached_a)
    reg_b_cache+=64*blockDim.z*4;
  int j=0, nc=0;
  {
      int true_nc = (nc+threadIdx.y)*32+row;
      int true_j = (j+threadIdx.z)*32+row;
      float4_t reg_c00={0,0,0,0};
      float4_t reg_c01={0,0,0,0};
      float4_t reg_c10={0,0,0,0};
      float4_t reg_c11={0,0,0,0};
      for(int offset=0; offset<a; offset+=32) {
        const __half* pp1 = p1 + true_nc*a;
        const __half* pp2 = p2 + true_j*a;
        const __half* pp3 = p1 + (true_nc+16)*a;
        const __half* pp4 = p2 + (true_j+16)*a;
        /*
        reg_a = *(half4_t*)(pp1 + offset + px_offset);
        reg_b = *(half4_t*)(pp2 + offset + px_offset);
        if(offset+px_offset>a-4) {
          for(int t=0; t<4; t++) {
            if(offset+px_offset+t>=a) {
              reg_a[t]=0;
              reg_b[t]=0;
            }
          }
        }
        */
        __syncthreads();
        half4_t reg[8];
        if(threadIdx.z==0) {
          if(offset+px_offset+4<=a) 
          {
            reg[0] = *(half4_t*)(pp1 + offset + px_offset);
            reg[1] = *(half4_t*)(pp3 + offset + px_offset);
          }
          else
          {
            reg[0]={0,0,0,0};
            reg[1]={0,0,0,0};
          }
        }
        if(threadIdx.y==0) {
          if(offset+px_offset+4<=a) 
          {
            reg[2] = *(half4_t*)(pp2 + offset + px_offset);
            reg[3] = *(half4_t*)(pp4 + offset + px_offset);
          }
          else
          {
            reg[2]={0,0,0,0};
            reg[3]={0,0,0,0};
          }
        }
        if(threadIdx.z==0) {
          if(offset+px_offset+16+4<=a) 
          {
            reg[4] = *(half4_t*)(pp1 + offset + px_offset+16);
            reg[5] = *(half4_t*)(pp3 + offset + px_offset+16);
          }
          else
          {
            reg[4]={0,0,0,0};
            reg[5]={0,0,0,0};
          }
        }
        if(threadIdx.y==0) {
          if(offset+px_offset+16+4<=a) 
          {
            reg[6] = *(half4_t*)(pp2 + offset + px_offset+16);
            reg[7] = *(half4_t*)(pp4 + offset + px_offset+16);
          }
          else
          {
            reg[6]={0,0,0,0};
            reg[7]={0,0,0,0};
          }
        }
        if(threadIdx.z==0) {
          reg_b_cache[threadIdx.y*64*4+threadIdx.x*4+0] = reg[0];
          reg_b_cache[threadIdx.y*64*4+threadIdx.x*4+1] = reg[1];
        }

        if(threadIdx.y==0) {
          reg_a_cache[threadIdx.z*64*4+threadIdx.x*4+0] = reg[2];
          reg_a_cache[threadIdx.z*64*4+threadIdx.x*4+1] = reg[3];
        }


        __syncthreads();
        half4_t reg_a = reg_a_cache[threadIdx.z*64*4+threadIdx.x*4+0];
        half4_t reg_a1 = reg_a_cache[threadIdx.z*64*4+threadIdx.x*4+1];
        half4_t reg_b = reg_b_cache[threadIdx.y*64*4+threadIdx.x*4+0];
        half4_t reg_b1 = reg_b_cache[threadIdx.y*64*4+threadIdx.x*4+1];
        //reg_a[0]+=1;
        //reg_b[0]+=1;
        reg_c00 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b, reg_c00, 0, 0, 0);
        reg_c01 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a1, reg_b, reg_c01, 0, 0, 0);
        reg_c10 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b1, reg_c10, 0, 0, 0);
        reg_c11 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a1, reg_b1, reg_c11, 0, 0, 0);

        __syncthreads();
        if(threadIdx.z==0) {
          reg_b_cache[threadIdx.y*64*4+threadIdx.x*4+2] = reg[4];
          reg_b_cache[threadIdx.y*64*4+threadIdx.x*4+3] = reg[5];
        }

        if(threadIdx.y==0) {
          reg_a_cache[threadIdx.z*64*4+threadIdx.x*4+2] = reg[6];
          reg_a_cache[threadIdx.z*64*4+threadIdx.x*4+3] = reg[7];
        }
        __syncthreads();
        reg_a = reg_a_cache[threadIdx.z*64*4+threadIdx.x*4+2];
        reg_a1 = reg_a_cache[threadIdx.z*64*4+threadIdx.x*4+3];
        reg_b = reg_b_cache[threadIdx.y*64*4+threadIdx.x*4+2];
        reg_b1 = reg_b_cache[threadIdx.y*64*4+threadIdx.x*4+3];
        //reg_a[0]+=1;
        //reg_b[0]+=1;
        reg_c00 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b, reg_c00, 0, 0, 0);
        reg_c01 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a1, reg_b, reg_c01, 0, 0, 0);
        reg_c10 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b1, reg_c10, 0, 0, 0);
        reg_c11 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a1, reg_b1, reg_c11, 0, 0, 0);
      }
      // 64 threads x 4 entries give me 16x16 values, one row per 4 threads
      int out_nc = (nc+threadIdx.y)*32 + row;
      int out_j = (j+threadIdx.z)*32 + px_offset;
      testatomicAdd(out+out_nc*k+(out_j),   reg_c00[0]);
      testatomicAdd(out+out_nc*k+(out_j+1), reg_c00[1]);
      testatomicAdd(out+out_nc*k+(out_j+2), reg_c00[2]);
      testatomicAdd(out+out_nc*k+(out_j+3), reg_c00[3]);
      testatomicAdd(out+out_nc*k+(out_j+16), reg_c01[0]);
      testatomicAdd(out+out_nc*k+(out_j+17), reg_c01[1]);
      testatomicAdd(out+out_nc*k+(out_j+18), reg_c01[2]);
      testatomicAdd(out+out_nc*k+(out_j+19), reg_c01[3]);
      testatomicAdd(out+(out_nc+16)*k+(out_j),   reg_c10[0]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+1), reg_c10[1]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+2), reg_c10[2]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+3), reg_c10[3]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+16), reg_c11[0]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+17), reg_c11[1]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+18), reg_c11[2]);
      testatomicAdd(out+(out_nc+16)*k+(out_j+19), reg_c11[3]);
    }
}

__global__ void convolve_bwdw_1x1x1(int n, int k, int c, int a, const __half* p1, const __half* p2, float* out)
{
  int b = blockIdx.x;
  p1 += a*b*c;
  p2 += a*b*k;

  for(int nc=threadIdx.y; nc<c; nc+=blockDim.y) {
    for(int j = threadIdx.x; j<k; j+=blockDim.x) {
      float sum=0;
      for(int i=0; i<a; i++)
      {
        float v1=p1[i+nc*a];
        float v2=p2[i+ j*a];
        sum+=v1*v2;
      }
      atomicAdd(out+nc*k+j, sum);
    }
  }
}

__global__ void cvt_fp32_fp16(const float* in, __half* out, int n)
{
  for(int i=threadIdx.x; i<n; i+=blockDim.x)
    out[i]=__float2half(in[i]);
} 

void doConvolveManual(const __half* input1, const __half* input2, __half* output,
  int wi, int hi, int c, int n, int k)
{
  int wo = wi, ho = hi;
  int a = ho*wo;
  /*
  if((k&15) || (c&15) || (a&3)) {
    printf("Skipping\n");
    return;
  }
*/
  static int call_count=0;
 // printf("%d\n", call_count&7);
  //hipLaunchKernelGGL(convolve_kernel, dim3((ho+1)/2, k, n), dim3(256, 1, 1), 0,
  //                   (hipStream_t)0, data, filter, output, call_count & 7);

  //call_count++;

  //std::vector<float> ftemp;
  float* temp;
  hipMalloc(&temp, c*k*sizeof(float));
  //ftemp.resize(c*k)
  //for(int i=0; i<c*k; i++)
  //  ftemp[i]=0;
//  if(!(call_count & 1)) {
//  }
//  else {
//    hipMemcpy(&host_dy[0], input2, host_dy.size()*2, hipMemcpyDeviceToHost);
//    hipMemcpy(&host_x[0], input1, host_x.size()*2, hipMemcpyDeviceToHost);
//  }

/*
nan in host_dy at b=6 nc=0 j=0 px=0 py=4
  -> offset 6*256*14*14+4*14 = 301112
*/
  #if 0
  std::vector<__half> host_dy, host_x;
  //temp.resize(c*k);
  host_dy.resize(wo*ho*c*n);
  host_x.resize(wo*ho*k*n);
hipMemcpy(&host_dy[0], input1, host_dy.size()*2, hipMemcpyDeviceToHost);
hipMemcpy(&host_x[0], input2, host_x.size()*2, hipMemcpyDeviceToHost);
  const __half* p1=&host_dy[0], *p2=&host_x[0];
  int nc=1, j=0;
  float sum=0;
  for(int b=0; b<n; b++)
  {
      for(int i=0; i<a; i++)
      {
        float v1=p1[i+nc*a];
        float v2=p2[i+ j*a];
        //float v1=host_dy[i+nc*a+b*c*a];
        //float v2=host_x [i +j*a+b*k*a];
        sum+=v1*v2;
      }
      p1 += a*c;
      p2 += a*k;
  }
  p1=&host_dy[0];
  p2=&host_x[0];
  nc=0;j=1;
  float sum2=0;
  for(int b=0; b<n; b++)
  {
      for(int i=0; i<a; i++)
      {
        float v1=p1[i+nc*a];
        float v2=p2[i+ j*a];
        //float v1=host_dy[i+nc*a+b*c*a];
        //float v2=host_x [i +j*a+b*k*a];
        sum2+=v1*v2;
      }
      p1 += a*c;
      p2 += a*k;
  }
  printf("Check: %f %f\n", sum, sum2);
#endif
  #if 0
  for(int nc=0; nc<c; nc++)
    for(int j=0; j<k; j++) {
      float sum=0;
      for(int b=0; b<n; b++)
        for(int py=0; py<ho; py++)
          for(int px=0; px<wo; px++)
      {
//        float v1[c], v2[k];
          float v1=host_dy[px+py*wo+nc*wo*ho+b*c*wo*ho];
          float v2=host_x[px+py*wo+j*wo*ho+b*k*wo*ho];
          if(nc==0 && j==0 && (b==0 || b==n-1) && (py==0 || py==ho-1) && (px==0 || px==wo-1))
          {
            printf("%d %d %d   %f  %f\n", b, px, py, v1, v2);
          }
          if(isnan(v1)||isnan(v2))
          {
            printf("%d %d %d %d %d   %f  %f\n", b, nc, j, px, py, v1, v2);
            exit(-1);
          }
          sum+=v1*v2;
        }
       ftemp[nc+j*c]=sum;
       if(nc==0 && j==0)
        printf("Sum[0,0] = %f\n", sum);
    }
    #endif
  //for(int i=0; i<c*k; i++)
  //  temp[i]=__float2half(ftemp[i]);
  hipMemset(temp, 0, c*k*4);
//  printf("Inputs: %f %f\n", __half2float(host_dy[0]), __half2float(host_x[0]));
#if 1
  dim3 block;
  block.x = 64;
  block.y=2;
  block.z=2;
  //block.y = max(1, min(16, c/32));
  //block.z = max(1, min(1024/(block.x*block.y), k/32));
//  __global__ void convolve_bwdw_1x1x1_xdlops(int n, int k, int c, int a, const __half* p1, const __half* p2, float* out)
//  printf("Block %d x %d x %d, n %d, k %d, c %d, a %d\n",
    //block.x, block.y, block.z, n, k, c, a);
  hipLaunchKernelGGL(convolve_bwdw_1x1x1_xdlops, dim3(n,1,1), block, 0,
    0, n, k, c, a, input1, input2, temp);// const __half* p1, const __half* p2, float* out)
#else
  hipLaunchKernelGGL(convolve_bwdw_1x1x1, dim3(n,1,1), dim3(min(1024,k), max(1,min(c, (1024+k-1)/k)), 1), 0,
    0, n, k, c, a, input1, input2, temp);// const __half* p1, const __half* p2, float* out)
#endif  
  hipLaunchKernelGGL(cvt_fp32_fp16, 1, min(1024,k*c), 0,
    0, temp, output, k*c);
  //call_count++;
  //hipMemcpy(output, &temp[0], temp.size()*2, hipMemcpyHostToDevice);
  hipFree(temp);
}
