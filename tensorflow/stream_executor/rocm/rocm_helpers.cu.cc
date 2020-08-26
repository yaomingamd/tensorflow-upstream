#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <limits>
#include <math.h>
typedef int index_t;
#include "number.hpp"
//#include "amd_xdlops.hpp"
#include "float_type.hpp"

using ck::half2_t;
using ck::half4_t;
using ck::float4_t;
using ck::float16_t;

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
#if 0
// does not work (investigate)
  int tid = (threadIdx.x&63) + (threadIdx.z&1)*64 + (threadIdx.y&7)*128;
  __shared__ float temp[4*1024];
  //__syncthreads();
  for(int i=0; i<4; i++)
    temp[tid+i]=c[i];
  for(int i=0; i<4; i++)
    c[i]=temp[(tid&~3)*4+i*4+(tid&3)];
#else  
  int blk = threadIdx.x & 60;
  int lane = threadIdx.x & 3;
  float t[4];
  t[0]=c[0];
  t[1]=c[1];
  t[2]=c[2];
  t[3]=c[3];

  float x = t[0];
  float u[3];
  u[0] = __shfl(x, blk+1);
  u[1] = __shfl(x, blk+2);
  u[2] = __shfl(x, blk+3);
  if(lane==0) {
    c[1]=u[0];
    c[2]=u[1];
    c[3]=u[2];
  }
  x = t[1];
  u[0] = __shfl(x, blk+0);
  u[1] = __shfl(x, blk+2);
  u[2] = __shfl(x, blk+3);
  if(lane==1) {
    c[0]=u[0];
    c[2]=u[1];
    c[3]=u[2];
  }
  x=t[2];
  u[0] = __shfl(x, blk+0);
  u[1] = __shfl(x, blk+1);
  u[2] = __shfl(x, blk+3);
  if(lane==2) {
    c[0]=u[0];
    c[1]=u[1];
    c[3]=u[2];
  }
  x=t[3];
  u[0] = __shfl(x, blk+0);
  u[1] = __shfl(x, blk+1);
  u[2] = __shfl(x, blk+2);
  if(lane==3) {
    c[0]=u[0];
    c[1]=u[1];
    c[2]=u[2];
  }
#endif
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

const int log_strip_width = 6;
const int strip_width = 1 << log_strip_width;

// fwd-conv7x7u2, wi=230, hi=230
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
  int group = blockIdx.y; 

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

void doConvolveManual_fw7x7x2(const __half* data, const __half* filter, __half* output, int n, hipStream_t stream)
{
  int wi=230, hi=230, c=3, x=7, y=7, k=64, u=2, v=2;
  int wo = (wi-x+1)/u, ho=(hi-y+1)/v;
  static int call_count=0;
// printf("%d\n", call_count&7);
  int x_dim = 16;
  hipLaunchKernelGGL(convolve_kernel, dim3((ho+x_dim-1)/x_dim, (ho+strip_width-1)/strip_width, n), dim3(512, 1, 1), 0,
               stream, data, filter, output, x_dim);
  call_count++;
}

__device__ void try_fetch(half4_t& out, const __half* ptr, int offset, int a) 
{
  if(offset+4<=a)
    out = *(const half4_t*)(ptr+offset);
  else
    out={0,0,0,0};
}

__device__ void try_fetch_partial(half4_t& out, const __half* ptr, int offset, int a) 
{
  if(offset+4<=a)
    out = *(const half4_t*)(ptr+offset);
  else {
    for(int i=0; i<4; i++)
      out[i] = (offset+i<a) ? ptr[offset+i] : __half(0.0f);
  }
}


/****
  Instantiated assuming blockDim.x=64, blockDim.y=(1<<Z).
  Calls to global_fetch() with 't' sweeping from 0 to W-1 and with 'z_index' from 0 to 1
   load a 2D region, 32<<Z high by 16*W wide, into registers.
  Calls local_store() store these registers into the block of shared memory passed
   as _data at construction time.
  Calls to local_fetch() pull the data out of the shared block in the right order
   to be passed to mfma (typically, each element will be pulled more than once.)
  Permutations perm1 and perm2 (5!=120 options each) control the read order (which byte
   is read by which thread), and the memory layout. Any choice of permutations is valid
   but some are faster than others.
****/
template <int logW, int Z>
class lds_buffer
{
public:
  static const int W=1<<logW;
  half4_t* data;
  int stride_by, stride_bx, stride_bxh, stride_bt, stride_bh,
        y5, x2, y0, x4, y4;
    __device__ lds_buffer(half4_t* _data, const int* perm1, const int* perm2) : data(_data)
    {
      // This all will hopefully be resolved at compile time
      struct layer {
        int* p;
        int w;
      };

      layer e[5]={{&stride_by,Z},{&stride_bx,4},{&stride_bxh,2},{&stride_bh,1},{&stride_bt,logW}};
      int stride=1;
      for(int i=0; i<5; i++) {
        *(e[perm1[i]].p)=stride;
        stride<<=e[perm1[i]].w;
      }

      int wy0=4, wx2=2, wy4=1, wy5=Z, wx4=logW;
      layer f[5]={{&x2,wx2},{&y5,wy5},{&y0,wy0},{&x4,wx4},{&y4,wy4}};
      int p=0;
      for(int i=0; i<5; i++) {
        *(f[perm2[i]].p)=p;
        p+=f[perm2[i]].w;
      }
    }

    __device__ void global_fetch(half4_t& dst, const __half* p, int a, int a2, int offset, int t, int z_index)
    {
      int yw=Z;
      int mw = (1<<yw)-1;
      int id = (threadIdx.x&63) + (z_index&1)*64 + (threadIdx.y&mw)*128 + t*(128<<yw);

      int index_x = offset        + 4*((id>>x2)&3)+16*((id>>x4)&(W-1));
      int index_y = ((id>>y0)&15) + ((id>>y4)&1)*16 + ((id>>y5)&mw)*32;
      try_fetch(dst, p + index_y*a, index_x, a2);
    }

    __device__ void global_fetch_partial(half4_t& dst, const __half* p, int a, int a2, int offset, int t, int z_index)
    {
      int yw=Z;
      int mw = (1<<yw)-1;
      int id = (threadIdx.x&63) + (z_index&1)*64 + (threadIdx.y&mw)*128 + t*(128<<yw);

      int index_x = offset        + 4*((id>>x2)&3)+16*((id>>x4)&(W-1));
      int index_y = ((id>>y0)&15) + ((id>>y4)&1)*16 + ((id>>y5)&mw)*32;
      //if(threadIdx.y<=mw)
      try_fetch_partial(dst, p + index_y*a, index_x, a2);
      //else
      //  dst={0,0,0,0};
    }

    __device__ void local_store(const half4_t& val, int t, int z_index)
    {
      int yw=Z;
      int mw = (1<<yw)-1;
       int id = (threadIdx.x&63) + (z_index&1)*64 + (threadIdx.y&mw)*128 + t*(128<<yw);
      
      int index = 
          ((id>>y5)&mw)*stride_by 
        + ((id>>x2)&3) *stride_bxh
        + ((id>>y0)&15)*stride_bx 
        + ((id>>x4)&(W-1))*stride_bt 
        + ((id>>y4)&1) * stride_bh;

      data[index] = val;
    }

    __device__ void local_fetch(half4_t& r1, half4_t& r2, int t, int tid)
    {
        int off = stride_by*tid + stride_bx*(threadIdx.x&15) + stride_bxh*((threadIdx.x>>4)&3) + stride_bt*t;
        r1 = data[off];
        r2 = data[off + stride_bh];
    }
};


template <bool transpose>
__device__ void store_results(float* out, int k, int c, float4_t& reg_c00, float4_t& reg_c01, float4_t& reg_c10, float4_t& reg_c11)
{
  if(transpose) {
    int out_nc = threadIdx.z*32 + ((threadIdx.x >> 4)&3)*4 ;
    int out_j  = threadIdx.y*32 + (threadIdx.x & 12) + (threadIdx.x & 3);
    for(int x=0; x<4; x++)
      atomicAdd(&out[out_nc*c+out_j+x*c], reg_c00[x]);
    for(int x=0; x<4; x++)
      atomicAdd(&out[out_nc*c+out_j+x*c+16*c], reg_c01[x]);
    for(int x=0; x<4; x++)
      atomicAdd(&out[out_nc*c+out_j+x*c+16], reg_c10[x]);
    for(int x=0; x<4; x++)
      atomicAdd(&out[out_nc*c+out_j+x*c+16+16*c], reg_c11[x]);
  }
  else {
    int out_nc = threadIdx.y*32 + (threadIdx.x & 12);
    int out_j = threadIdx.z*32 + ((threadIdx.x >> 4)&3)*4 + (threadIdx.x & 3);
    
    transpose4(reinterpret_cast<float*>(&reg_c00));
    transpose4(reinterpret_cast<float*>(&reg_c01));
    transpose4(reinterpret_cast<float*>(&reg_c10));
    transpose4(reinterpret_cast<float*>(&reg_c11));

    for(int x=0; x<4; x++)
      atomicAdd(&out[out_nc*k+out_j+x*k], reg_c00[x]);
    for(int x=0; x<4; x++)
      atomicAdd(&out[out_nc*k+out_j+x*k+16], reg_c01[x]);
    for(int x=0; x<4; x++)
      atomicAdd(&out[out_nc*k+out_j+x*k+16*k], reg_c10[x]);
    for(int x=0; x<4; x++)
      atomicAdd(&out[out_nc*k+out_j+x*k+16*k+16], reg_c11[x]);
  }
}

template <int logC, int logK, bool transpose>
__launch_bounds__(1024) __global__ void convolve_bwdw_1x1x1_xdlops(uint64_t n_, uint64_t k_, uint64_t c_, uint64_t a_, const __half* p1_, const __half* p2_, float* out)
{
  int n=n_, k=k_, c=c_, a=a_;
 
  int tid = threadIdx.x & 63;
  int row = tid & 15;
  int tid_high = tid>>4;
  int px_offset = 4*(tid >> 4); 
  const int logW=(logC==3) ? 2 : 3;
  const int W=1<<logW;
  const int mW = (1<<logC)-1;
  int j=0, nc=0;
  float4_t reg_c00={0,0,0,0};
  float4_t reg_c01={0,0,0,0};
  float4_t reg_c10={0,0,0,0};
  float4_t reg_c11={0,0,0,0};
  int true_nc = (nc+(threadIdx.y&mW))*32+row;
  int true_j = (j+(threadIdx.z&1))*32+row;

  half4_t* __restrict__ reg_a_cache;
  half4_t* __restrict__ reg_b_cache;

  __shared__ half4_t reg_cache[64*W*2*((1<<logC)+(1<<logK))];
  reg_a_cache = &reg_cache[0];
  reg_b_cache = &reg_cache[64*W*2*(1<<logK)];

  int offset=0;  

  const int permute_table[8][5]={
    {1,2,4,3,0},
    {1,4,2,0,3},
    {3,2,1,4,0},
    {0,2,4,3,1},
    {2,1,3,0,4},
    {0,1,4,3,2},
    {3,2,1,0,4},
    {2,1,3,4,0},
  };

  const int *perm1a, *perm1b, *perm2a, *perm2b;
  if(logC==3) {
    perm1a=permute_table[0];
    perm1b=permute_table[1];
    perm2a=permute_table[2];
    perm2b=permute_table[3];
  } else {
    // 0x6F 0x31 0x77 0x57
    perm1a=permute_table[4];
    perm1b=permute_table[5];
    perm2a=permute_table[6];
    perm2b=permute_table[7];
  }

  lds_buffer<logW,logK> abuf(reg_a_cache, perm1a, perm2a);
  lds_buffer<logW,logC> bbuf(reg_b_cache, perm1b, perm2b);
  half4_t rnext[2][W];

  // be very careful messing with subsequent code
  // even trivial changes can lead to issues with register spilling, etc.,
  // dropping kernel performance by 10-20%
  for(int b = blockIdx.x; b<n; b+=gridDim.x) {
    const __half* p1 = p1_ + a*b*c;
     const __half* p2 = p2_ + a*b*k;
    int a2 = a;
    if(gridDim.y>1) {
      a2>>=1;
      p1 += a2*blockIdx.y;
      p2 += a2*blockIdx.y;
    }

    for(int t=0; t<W; t++) {
      half4_t a0;
      bbuf.global_fetch(a0, p1, a, a2, 0, t, threadIdx.z);
      bbuf.local_store(a0, t, threadIdx.z);
    }

    if(threadIdx.y<2) {
      for(int t=0; t<W; t++) {
        half4_t a1;
        abuf.global_fetch(a1, p2, a, a2, 0, t, threadIdx.z);
        abuf.local_store(a1, t, threadIdx.z);
      }
    } 

  for(int offset=0; offset<a2; offset+=16*W) {
    __syncthreads();
      for(int t=0; t<W; t++) {
        bbuf.global_fetch(rnext[0][t], p1, a, a2, offset+16*W, t, threadIdx.z);
      }
      for(int t=0; t<W; t++) {
        if(threadIdx.y<2) {
          abuf.global_fetch(rnext[1][t], p2, a, a2, offset+16*W, t, threadIdx.z);
        }
      }

      for(int t=0; t<W; t++) {
        half4_t reg_a, reg_a1, reg_b, reg_b1;
        abuf.local_fetch(reg_a, reg_a1, t, threadIdx.z);
        bbuf.local_fetch(reg_b, reg_b1, t, threadIdx.y);

        reg_c00 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b, reg_c00, 0, 0, 0);
        reg_c01 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a1, reg_b, reg_c01, 0, 0, 0);
        reg_c10 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a, reg_b1, reg_c10, 0, 0, 0);
        reg_c11 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_a1, reg_b1, reg_c11, 0, 0, 0);
      }

      __syncthreads();
      for(int t=0; t<W; t++) {
        bbuf.local_store(rnext[0][t], t, threadIdx.z);
        if(threadIdx.y<2) {
          abuf.local_store(rnext[1][t], t, threadIdx.z);
        }
      }
    }
  }
  store_results<transpose>(out, k, c, reg_c00, reg_c01, reg_c10, reg_c11);
}

/**
  The version for w=7,h=7,c=512,k=2048 or c=2048,k=512.
  In this mode, we produce 4 MB of data per batch element (512x2048 floats).
  It would take 1 ms just to write them all once at 1 TB/s without any atomics block subdivision.
  We're targeting <1.4 ms.
  -> Neccessary to accumulate across batch in registers or shared memory.
    -> To accumulate across batch, one thread block can't process anywhere close to full batch, since we don't have 4 MB of shared memory.
       We split the output region into 128x128 parts, which requires 2x (64x128) input elements in shared memory (32 KB),
       and accumulate in accVGPRs.
**/
template <int logC, int logK, bool transpose>
__launch_bounds__(1024) __global__ void convolve_bwdw_1x1x1_xdlops_2(uint64_t n_, uint64_t k_, uint64_t c_, uint64_t a_, const __half* p1_, const __half* p2_, float* out)
{
  int n=n_, k=k_, c=c_, a=a_;
 
  int tid = threadIdx.x & 63;
  int row = tid & 15;
  int tid_high = tid>>4;
  int px_offset = 4*(tid >> 4); 
  const int logW=2;
  const int W=1<<logW;
  const int mW = (1<<logC)-1;


  half4_t* __restrict__ reg_a_cache;
  half4_t* __restrict__ reg_b_cache;
  // 32*4*4*(4+4)
  __shared__ half4_t reg_cache[32*W*4*((1<<logC)+(1<<logK))];
  reg_a_cache = &reg_cache[0];
  reg_b_cache = &reg_cache[32*W*4*(1<<logC)];
 
  int offset=0;

  const int perm1a[5] = {3,1,0,4,2};
  const int perm1b[5] = {1,3,2,0,4};
  const int perm2a[5] = {0,2,1,4,3};
  const int perm2b[5] = {0,2,1,3,4};

  // logC=logK=2
  lds_buffer<logW,logC> abuf(reg_a_cache, perm1a, perm2a);
  lds_buffer<logW,logK> bbuf(reg_b_cache, perm1b, perm2b);
  half4_t rnext[2][W];

  int nc = blockIdx.y*128, j=blockIdx.z*128;
  float4_t reg_c00={0,0,0,0};
  float4_t reg_c01={0,0,0,0};
  float4_t reg_c10={0,0,0,0};
  float4_t reg_c11={0,0,0,0};

  int b = blockIdx.x;
  int a2 = a;

    const __half* p1 = p1_ + a*b*c + a*nc;
    const __half* p2 = p2_ + a*b*k + a*j;

    __syncthreads();
    for(int t=0; t<W; t++) {
      if(threadIdx.z<2)
        abuf.global_fetch_partial(rnext[0][t], p1, a, a2, offset, t, threadIdx.z);
      else
        bbuf.global_fetch_partial(rnext[1][t], p2, a, a2, offset, t, threadIdx.z-2);
    }
    for(int t=0; t<W; t++) {
      if(threadIdx.z<2)
        abuf.local_store(rnext[0][t], t, threadIdx.z);
      else
        bbuf.local_store(rnext[1][t], t, threadIdx.z-2);
    }


  for(; b<n; b+=gridDim.x) {
    const __half* p1 = p1_ + a*b*c + a*nc;
    const __half* p2 = p2_ + a*b*k + a*j;
    __syncthreads();
    int offset=0;

      __syncthreads();
//      if(threadIdx.z<2) {
      if(b+gridDim.x<n) {
        for(int t=0; t<W; t++) {
          if(threadIdx.z<2)
            abuf.global_fetch_partial(rnext[0][t], p1+a*c*gridDim.x, a, a2, offset, t, threadIdx.z);
          else
            bbuf.global_fetch_partial(rnext[1][t], p2+a*k*gridDim.x, a, a2, offset, t, threadIdx.z-2);
        }
      }

      for(int t=0; t<W; t++) {
        half4_t reg_a, reg_a1, reg_b, reg_b1;
        abuf.local_fetch(reg_a, reg_a1, t, threadIdx.y);
        bbuf.local_fetch(reg_b, reg_b1, t, threadIdx.z);

        reg_c00 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b, reg_a, reg_c00, 0, 0, 0);
        reg_c01 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b1, reg_a, reg_c01, 0, 0, 0);
        reg_c10 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b, reg_a1, reg_c10, 0, 0, 0);
        reg_c11 = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b1, reg_a1, reg_c11, 0, 0, 0);
      }
      __syncthreads();
      for(int t=0; t<W; t++) {
        if(threadIdx.z<2)
          abuf.local_store(rnext[0][t], t, threadIdx.z);
        else
          bbuf.local_store(rnext[1][t], t, threadIdx.z-2);
      }
  }
  if(!transpose)
    out += j+nc*k;
  else
    out += nc+j*c;
  store_results<transpose>(out, k, c, reg_c00, reg_c01, reg_c10, reg_c11);
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

__global__ void acc_fp16(const __half* in, __half* out, int b, int n)
{
  for(int i=threadIdx.x*2; i<n; i+=blockDim.x*2) {
    float acc[2]={0,0};
    for(int j=0; j<b; j++) {
      half2_t v = *(const half2_t*)(in+i+j*n);
      acc[0] += __half2float(v[0]);
      acc[1] += __half2float(v[1]);
    }
    out[i]=acc[0];
    out[i+1]=acc[1];
  }
}

void doConvolveManual(const __half* input1, const __half* input2, __half* output,
  int wi, int hi, int c, int n, int k, hipStream_t stream, void* scratch, int scratch_size)
{
  int wo = wi, ho = hi;
  int a = ho*wo;

  static int call_count=0;
  bool transpose = false;
  if(c<k) {
    int j=c;
    c=k;
    k=j;
    transpose = true;
    auto temp = input1;
    input1 = input2;
    input2 = temp;
  }


  bool allocated = false;
  float* temp;
  if(scratch==0 || scratch_size<c*k*4) {
    hipMalloc(&temp, c*k*sizeof(float));
    allocated = true;
  }
  else {
    temp = reinterpret_cast<float*>(scratch);
  }
  if(call_count<4) {
    printf("%d (%dx%d%s)   scratch %d / %d, %s allocating\n", 
      call_count, c, k, transpose ? " transposed" : "",
      scratch_size, c*k*4,
      allocated ? "" : "NOT");
  }
  hipMemset(temp, 0, c*k*4);
  if(wi==7 && hi==7) {
    auto cf = transpose ? convolve_bwdw_1x1x1_xdlops_2<2, 2, true> : convolve_bwdw_1x1x1_xdlops_2<2, 2, false>;
    hipLaunchKernelGGL(cf, dim3(1, c/128, k/128), dim3(64, 4, 4), 0, stream,
      n, k, c, a, input1, input2, temp);
  }
  else {
    dim3 block;
    block.x = 64;
    block.y=c>>5;
    block.z=k>>5;

    auto cf = (c==256) ? 
      (transpose ? convolve_bwdw_1x1x1_xdlops<3, 1, true> : convolve_bwdw_1x1x1_xdlops<3, 1, false> )
      : convolve_bwdw_1x1x1_xdlops<1, 1, false>;
    hipLaunchKernelGGL(cf, dim3(n,1,1), block, 0, stream,
      n, k, c, a, input1, input2, temp);
  }
  hipLaunchKernelGGL(cvt_fp32_fp16, 1, min(1024,k*c), 0, stream,
    temp, output, k*c);
  if(allocated)
    hipFree(temp);

  call_count++;
}
