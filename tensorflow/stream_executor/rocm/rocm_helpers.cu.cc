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
using ck::half6_t;
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

__device__ void fetch(half4_t& out, const __half* ptr, int offset) 
{
  out = *(const half4_t*)(ptr+offset);
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

/***

  Helper class to control data transfer between global memory and mfma in the conv_bwdw1x1u1 case.


***/
template <int logW, int Z, int logZ1, int logZ2, int logZ3, int Perm>
class lds_buffer2
{
public:
  static const int W=1<<logW;
  //static const int Z=logZ1+logZ2-1;
  //static const int mw = (1<<(logZ1+logZ2-1))-1;
  static const int mw = (1<<Z)-1;
  static const int overcover = logZ1+logZ2-Z-1;
  //static const int operation_mode = (logZ1+logZ2-Z-1==0) ? 0 : (logZ1+logZ2-Z-1<0 ? 1 : 2);
  int operation_mode;
  int order_selector;
  half4_t* data;
  half4_t* data_store;
  half4_t* data_load;
  unsigned int stride_by, stride_bx, stride_bxm, stride_bxh, stride_bt, stride_bh, stride_z3,
        y5, x2, y0, x4, y2, y4, rej;
//        unsigned int id_base, local_base, local_addr_base,
//        index_x_base, index_y_base, local_fetch_base;
    __device__ lds_buffer2(half4_t* _data) : data(_data)
    {

const int perm120[120][5]=
{
  {4,0,1,2,3},    {4,0,1,3,2},    {4,0,2,1,3},    {4,0,2,3,1},     {4,0,3,1,2},     {4,0,3,2,1},     {4,1,0,2,3},     {4,1,0,3,2}, 
  {4,1,2,0,3},    {4,1,2,3,0},    {4,1,3,0,2},    {4,1,3,2,0},     {4,2,0,1,3},     {4,2,0,3,1},     {4,2,1,0,3},     {4,2,1,3,0}, 
// 0x10
  {4,2,3,0,1},    {4,2,3,1,0},    {4,3,0,1,2},    {4,3,0,2,1},     {4,3,1,0,2},    {4,3,1,2,0},     {4,3,2,0,1},    {4,3,2,1,0},
  {0,4,1,2,3},    {0,4,1,3,2},    {0,4,2,1,3},    {0,4,2,3,1},     {0,4,3,1,2},    {0,4,3,2,1},    {1,4,0,2,3},    {1,4,0,3,2}, 
// 0x20
  {1,4,2,0,3},    {1,4,2,3,0},    {1,4,3,0,2},    {1,4,3,2,0},     {2,4,0,1,3},   {2,4,0,3,1},    {2,4,1,0,3},   {2,4,1,3,0},
  {2,4,3,0,1},    {2,4,3,1,0},    {3,4,0,1,2},    {3,4,0,2,1},    {3,4,1,0,2},   {3,4,1,2,0},    {3,4,2,0,1},   {3,4,2,1,0},
// 0x30
  {0,1,4,2,3},    {0,1,4,3,2},    {0,2,4,1,3},    {0,2,4,3,1},     {0,3,4,1,2},    {0,3,4,2,1},    {1,0,4,2,3},    {1,0,4,3,2}, 
  {1,2,4,0,3},    {1,2,4,3,0},    {1,3,4,0,2},    {1,3,4,2,0},     {2,0,4,1,3},   {2,0,4,3,1},    {2,1,4,0,3},   {2,1,4,3,0},
// 0x40    
  {2,3,4,0,1},    {2,3,4,1,0},    {3,0,4,1,2},    {3,0,4,2,1},    {3,1,4,0,2},    {3,1,4,2,0},    {3,2,4,0,1},    {3,2,4,1,0},
  {0,1,2,4,3},    {0,1,3,4,2},    {0,2,1,4,3},    {0,2,3,4,1},     {0,3,1,4,2},    {0,3,2,4,1},    {1,0,2,4,3},    {1,0,3,4,2}, 
// 0x50    
  {1,2,0,4,3},    {1,2,3,4,0},    {1,3,0,4,2},    {1,3,2,4,0},    {2,0,1,4,3},    {2,0,3,4,1},    {2,1,0,4,3},    {2,1,3,4,0},
  {2,3,0,4,1},    {2,3,1,4,0},    {3,0,1,4,2},    {3,0,2,4,1},    {3,1,0,4,2},    {3,1,2,4,0},    {3,2,0,4,1},    {3,2,1,4,0},
// 0x60
  {0,1,2,3,4},    {0,1,3,2,4},    {0,2,1,3,4},    {0,2,3,1,4},    {0,3,1,2,4},    {0,3,2,1,4},    {1,0,2,3,4},    {1,0,3,2,4}, 
  {1,2,0,3,4},    {1,2,3,0,4},    {1,3,0,2,4},    {1,3,2,0,4},    {2,0,1,3,4},    {2,0,3,1,4},    {2,1,0,3,4},    {2,1,3,0,4},
// 0x70    
  {2,3,0,1,4},    {2,3,1,0,4},    {3,0,1,2,4},    {3,0,2,1,4},    {3,1,0,2,4},    {3,1,2,0,4},    {3,2,0,1,4},    {3,2,1,0,4},
};      
      // This all will hopefully be resolved at compile time
      struct layer {
        unsigned int* p;
        unsigned int* q;
        int w;
      };

      static_assert(logZ1+logZ2+logZ3>=Z+1, "Wrong template parameters");
      static_assert(logZ1+logZ2<Z+1 || logZ3==0, "Wrong template parameters");

      const layer e[6]=
      {
        {&stride_by,  &y5, Z},
        {&stride_bx,  &y0, 2},
        {&stride_bxm,  &y2, 2},
        {&stride_bxh, &x2, 2},
        {&stride_bh,  &y4, 1},
        {&stride_bt,  &x4, logW}
      };

      int n5 = Perm/120;
      int perm1idx = Perm % 120;
      const int* perm = perm120[perm1idx];
      int long_perm[6]={perm[0],perm[1],perm[2],perm[3],perm[4],5};
      int temp = long_perm[n5];
      long_perm[n5] = 5;
      long_perm[5] = temp;

      const layer f[6]={e[long_perm[0]],e[long_perm[1]],e[long_perm[2]],e[long_perm[3]],e[long_perm[4]],e[long_perm[5]]};
      int p=0, q=0;
      for(int i=0; i<6; i++) {
        *(f[i].p)=p;
        *(f[i].q)=q;
        p+=f[i].w;
        q+=f[i].w;
      }
      rej = q;
      q += overcover>=0 ? overcover : 0;

      if(overcover==0) 
        operation_mode=0;
      else if(overcover<0)
        operation_mode=1;
      else
        operation_mode=2;
    }

    __device__ constexpr bool is_in_range(int id) const {
      if(overcover<=0)
        return true;
      return (((id>>rej)&((1<<overcover)-1))==0);
    }
     __device__ int index(int t, int z3) const {
      
      return (threadIdx.x&63) 
        | ((threadIdx.y & ((1<<logZ1)-1))<<6)
        | ((threadIdx.z & ((1<<logZ2)-1))<<(6+logZ1))
        | ((z3 & ((1<<logZ3)-1))<<(6+logZ1+logZ2))
        | (t<<(6+Z+1+(overcover>0?overcover:0)));
    }
    __device__ void get_xy(int id, int& index_x, int& index_y) const {
      index_x = 4*((id>>x2)&3) + 16*((id>>x4)&(W-1));
      index_y = ((id>>y0)&3) 
        + ((id>>y2)&3)*4
        + ((id>>y4)&1)*16 + ((id>>y5)&mw)*32; 
    }
    __device__ int store_address(int id) const {
          return (((id>>y5)&mw)<<stride_by)
          + (((id>>x2)&3) <<stride_bxh)
          + (((id>>y2)&3) <<stride_bxm )
          + (((id>>y0)&3) <<stride_bx )
         // + (((id>>y0)&15)<<stride_bx)
          + (((id>>x4)&(W-1))<<stride_bt) 
          + (((id>>y4)&1) << stride_bh);
      }
      __device__ int load_address(int t, int tid) const {
        return 
          ((threadIdx.x&3)<<stride_bx) 
          | (((threadIdx.x>>2)&3)<<stride_bxm) 
          | (((threadIdx.x>>4)&3)<<stride_bxh) 
          | (tid<<stride_by) 
          | (t<<stride_bt);
      }
    __device__ void local_store(const half4_t& val, unsigned int t, int z1, int z2, unsigned int z3)
    {
      int id = index(t, z3);
      if(is_in_range(id)) {
        data[store_address(id)] = val;
      }
    }

    template <int z3>
    __device__ void local_store2(const half4_t* val, int t) const
    {
      int id = index(t, z3);
      if(is_in_range(id)) {
        data[store_address(id)] = val[z3];
      }
      local_store2<z3-1>(val, t);
    }
    template <>
    __device__ void local_store2<-1>(const half4_t* val, int t) const {}

    template <int check_level>
    __device__ void global_fetch(half4_t& dst, const __half* p, int a, int a2, int offset, unsigned int t, int z1, int z2, unsigned int z3)
    {
      int id = index(t, z3);
      int index_x, index_y;
      get_xy(id, index_x, index_y);
      if(check_level == 2) {
        if(is_in_range(id))
          try_fetch_partial(dst, p + index_y*a, index_x + offset, a2);
      }
      else if(check_level == 1) {
        if(is_in_range(id))
          try_fetch(dst, p + index_y*a, index_x + offset, a2);
      }
      else {
        fetch(dst, p + index_y*a, index_x + offset);
      }
    }

    __device__ void local_fetch(half4_t& r1, half4_t& r2, int t, int tid)
    {
        //int off = (tid<<stride_by) + (t<<stride_bt);
        int off = load_address(t,tid);
        r1 = data[off];
        r2 = data[off + (1<<stride_bh)];
    }
};

/*
  Convolution parameters:
  input buffer p1, n*c*a elements (batch size n, c channels, a=wi*hi elements)
  input buffer p2, n*k*a
  output buffer, c*k elements

  Assuming c & k to be power of 2.

  Launched using block size 64 * (2^logGC) * (2^logGK) and grid size l * c/(32*2^logGC) * k/(32*2^logGK) 
    where 1 <= l <= n.

  Each block processes an output window (32*2^logC) x (32*2^logK).
  A single group of mfma calls in one warp generates a 32x32 block. We get to (32*2^logC) x (32*2^logK) by
  firing up (2^logGC) * (2^logGK) warps and iterating over (2^(logC-logGC)) * (2^(logK-logGK)) subblocks in
  a loop.

  Along the 'a' axis, one group of mfma calls processes 16 elements. We execute an inner loop (index 't') from 
  0 to (2^logW-1) to process 16*2^logW elements, and an outer loop (index 'offset') from 0 to 'a'.

  Setting the parameters:
  * 'Fractional' should be set to true if 'a' is not a multiple of 4 (enables extra logic to handle partial reads)
  * logC, logK, ... logW: any nonnegative values subject to the following constraints:
  logGC <= logC
  logGK <= logK
  32*2^logC <= c
  32*2^logK <= k
  logGC+logGK <= 4 (limited by the number of threads/block)
  logC+logK <= 6 (limited by the number of AGPRs)
  (2^logW)(2^logC+2^logK)<=64 (limited by shared memory)
  16*2^logW <= a

  Mode1, Mode2: controls the order in which data is read from global memory and cached in shared memory;
  allowed values are 0..719 for either, all values are legal but some are faster than others.
*/
template <bool Fractional, int logC, int logK, int logGC, int logGK, int logW, int Mode1, int Mode2>
__launch_bounds__(1024) __global__ void convolve_bwdw_1x1x1_xdlops_2(uint64_t n_, uint64_t k_, uint64_t c_, uint64_t a_, const __half* p1_, const __half* p2_, float* out)
{
  int n=n_, k=k_, c=c_, a=a_;
 
  int tid = threadIdx.x & 63;
  int row = tid & 15;
  int tid_high = tid>>4;
  int px_offset = 4*(tid >> 4); 
  const int W=1<<logW;
  const int mW = (1<<logC)-1;

  half4_t* __restrict__ reg_a_cache;
  half4_t* __restrict__ reg_b_cache;
  __shared__ half4_t reg_cache[32*W*4*((1<<logC)+(1<<logK))];
  reg_a_cache = &reg_cache[0];
  reg_b_cache = &reg_cache[32*W*4*(1<<logC)];

  int offset=0;
 
  const int read_grid_size = 1<<(logGC+logGK);
  const int overcoverC = logC+1-(logGC+logGK);
  const int overcoverK = logK+1-(logGC+logGK);
  const int log_read_repeats_a = overcoverC>0 ? overcoverC : 0;
  const int log_read_repeats_b = overcoverK>0 ? overcoverK : 0;
  const int read_repeats_a = 1<<log_read_repeats_a;
  const int read_repeats_b = 1<<log_read_repeats_b;

  lds_buffer2<logW,logC,logGC,logGK,log_read_repeats_a,Mode1> abuf(reg_a_cache);
  lds_buffer2<logW,logK,logGC,logGK,log_read_repeats_b,Mode2> bbuf(reg_b_cache);

  int nc = blockIdx.y<<(logC+5), j=blockIdx.z<<(logK+5);
  
  const int y_repeats = (1<<(logC-logGC));
  const int z_repeats = (1<<(logK-logGK));

  float4_t acc[4*y_repeats*z_repeats];
  for(int y=0; y<4*y_repeats*z_repeats; y++) {
      acc[y][0]=0;
      acc[y][1]=0;
      acc[y][2]=0;
      acc[y][3]=0;
  }

  int b = blockIdx.x;
  int a2 = a;

  float4_t* pf = ((float4_t*)&acc);

  //for(int b=blockIdx.x; b<n; b+=gridDim.x) 
label:  
  {
    const __half* p1 = p1_ + a*b*c + a*nc;
    const __half* p2 = p2_ + a*b*k + a*j;
    int offset=0;
    __syncthreads();

    half4_t temp_a[W][read_repeats_a];
    half4_t temp_b[W][read_repeats_b];

#pragma unroll
  for(int t=0; t<W; t++) {
      {
        for(int i=0; i<read_repeats_a; i++) {
          abuf.template global_fetch<0>(temp_a[t][i], p1, a, a2, 0, t, threadIdx.y, threadIdx.z, i);
          abuf.local_store(temp_a[t][i], t, threadIdx.y, threadIdx.z, i);
        }
        for(int i=0; i<read_repeats_b; i++) { 
          bbuf.template global_fetch<0>(temp_b[t][i], p2, a, a2, 0, t, threadIdx.y, threadIdx.z, i);
          bbuf.local_store(temp_b[t][i], t, threadIdx.y, threadIdx.z, i);
        }
      }
  }

    //for(offset=16*W*blockIdx.x; offset<a; offset+=16*W*gridDim.x) 
label2:  
    {
      __syncthreads();
      if(offset+16*W<a) 
      {
  #pragma unroll
        for(int t=0; t<W; t++) {
          for(int i=0; i<read_repeats_a; i++) 
            abuf.template global_fetch<Fractional ? 2 : 1>(temp_a[t][i], p1, a, a2, offset+16*W, t, threadIdx.y, threadIdx.z, i);
          for(int i=0; i<read_repeats_b; i++) 
            bbuf.template global_fetch<Fractional ? 2 : 1>(temp_b[t][i], p2, a, a2, offset+16*W, t, threadIdx.y, threadIdx.z, i);
        }
      }
#pragma unroll
      for(int t=0; t<W; t++) {
#pragma unroll
        for(int y=0; y<y_repeats; y++) 
        {
          half4_t reg_a, reg_a1;
          abuf.local_fetch(reg_a, reg_a1, t, threadIdx.y+(y<<logGC));
#pragma unroll
          for(int z=0; z<z_repeats; z++) {
            half4_t reg_b, reg_b1;
            bbuf.local_fetch(reg_b, reg_b1, t, threadIdx.z+(z<<logGK));
            acc[4*z*y_repeats+4*y+0] = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b, reg_a,   acc[4*z*y_repeats+4*y+0], 0, 0, 0);
            acc[4*z*y_repeats+4*y+1] = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b1, reg_a,  acc[4*z*y_repeats+4*y+1], 0, 0, 0);
            acc[4*z*y_repeats+4*y+2] = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b, reg_a1,  acc[4*z*y_repeats+4*y+2], 0, 0, 0);
            acc[4*z*y_repeats+4*y+3] = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b1, reg_a1, acc[4*z*y_repeats+4*y+3], 0, 0, 0);
          }
        }
      }
      __syncthreads();
      if(offset+16*W<a) 
      {
    #pragma unroll
        for(int t=0; t<W; t++) {
        //  abuf.template local_store2<read_repeats_a>(temp_a[t], t);
        //  bbuf.template local_store2<read_repeats_b>(temp_b[t], t);
          for(int i=0; i<read_repeats_a; i++)
            abuf.local_store(temp_a[t][i], t, threadIdx.y, threadIdx.z, i);
          for(int i=0; i<read_repeats_b; i++)
            bbuf.local_store(temp_b[t][i], t, threadIdx.y, threadIdx.z, i);
        }
      }
      
    }
    offset+=16*W;
    if(offset<a)
      goto label2;
  }
  b+=gridDim.x;
  if(b<n)
    goto label;

  for(int z=0; z<z_repeats; z++) {
    for(int y=0; y<y_repeats; y++) {
        float4_t& reg_c00 = pf[4*z*y_repeats+4*y+0];
        float4_t& reg_c01 = pf[4*z*y_repeats+4*y+1];
        float4_t& reg_c10 = pf[4*z*y_repeats+4*y+2];
        float4_t& reg_c11 = pf[4*z*y_repeats+4*y+3];
        
        transpose4(reinterpret_cast<float*>(&reg_c00));
        transpose4(reinterpret_cast<float*>(&reg_c01));
        transpose4(reinterpret_cast<float*>(&reg_c10));
        transpose4(reinterpret_cast<float*>(&reg_c11));
        int out_nc = nc + (threadIdx.y+(y<<logGC))*32 + (threadIdx.x & 12);
        int out_j =  j + (threadIdx.z+(z<<logGK))*32 + ((threadIdx.x >> 4)&3)*4 + (threadIdx.x & 3);
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
}



__device__ half2_t int_as_half2(int x) { return *(half2_t*)&x; }
__device__ int half2_as_int(half2_t x) { return *(int*)&x; }

__device__
inline
void atomicAdd2(__half* address, half2_t val)
{
    unsigned int* uaddr{reinterpret_cast<unsigned int*>(address)};
    unsigned int r{__atomic_load_n(uaddr, __ATOMIC_RELAXED)};

    unsigned int old;
    do {
        old = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
    
        if (r != old) { r = old; continue; }
        half2_t hr = int_as_half2(r);
        r = atomicCAS(uaddr, r, half2_as_int(half2_t{val[0]+hr[0],val[1]+hr[1]}));

        if (r == old) break;
    } while (true);
}


__device__ void generate_perm6(int perm2idx, int& pos0, int& pos1, int& pos2, int& pos3, int& pos4, int& pos5) {
      pos5 = perm2idx / 120;
      perm2idx %= 120;
      pos4 = (perm2idx / 24) + (((perm2idx / 24)>=pos5) ? 1 : 0);
      int mask = (1<<pos5) | (1<<pos4);
      perm2idx %= 24;
      pos3 = (perm2idx / 6);
//      pos3 += (pos3>=pos5) ? 1 : 0;
//      pos3 += (pos3>=pos4) ? 1 : 0;
      if(mask & (1<<pos3))
        pos3++;
      if(mask & (1<<pos3))
        pos3++;
      mask |= 1<<pos3;
      perm2idx %= 6;
      pos2 = (perm2idx / 2);
      if(mask & (1<<pos2))
        pos2++;
      if(mask & (1<<pos2))
        pos2++;
      if(mask & (1<<pos2))
        pos2++;
      mask |= 1<<pos2;
      pos1 = perm2idx % 2;
      if(mask & (1<<pos1))
        pos1++;
      if(mask & (1<<pos1))
        pos1++;
      if(mask & (1<<pos1))
        pos1++;
      if(mask & (1<<pos1))
        pos1++;
      mask |= 1<<pos1;
      pos0 = 1+2+3+4+5-(pos1+pos2+pos3+pos4+pos5);
}

template <int logGC, int logGK, int logC, int logW, int Mode>
class lds_buffer_vert
{
  int x0, y0, y2, y4, y5, s0;
public:
  const int log_read_repeats = 1; 
  const int W = 1<<logW;
  __device__ lds_buffer_vert() {
    x0 = 0;
    y0 = logW;
    y2 = logW+2;
    y4 = logW+4;
    y5 = logW+5;
    s0 = logW+5+logC;
  }
  __device__ void coords(
    int y, 
    int& index_x, int& index_y, bool& active)
  {
    int idx = (threadIdx.x & 63) 
      | ((threadIdx.y & 3)<<6)
      | ((threadIdx.z & 3)<<8)
      | (y<<10); 
    index_x = threadIdx.x & 15; //idx & (W-1);
    index_y = (idx >> logW) & ((1<<(logC+5))-1);
    int selector = idx >> (logW+logC+5);
    active = (selector==0 && index_x*4 + index_y*64 < 49*(1<<(5+logC)));
  }
};

template <int logGridC, int logGridK>
__launch_bounds__(1024) __global__ void convolve_bwdw_1x1x1x7x7_xdlops(uint64_t n_, uint64_t k_, uint64_t c_, uint64_t a_, const __half* p1_, const __half* p2_, __half* out)
{
  const int logC=2, logK=2;
  const int logGC=2, logGK=2;
  int n=n_, k=k_, c=c_, a=a_;
  const int logW = 2;
  const int logBC = 3;
  const int W=1<<logW;
  const int mW = (1<<logC)-1;

  half4_t* __restrict__ reg_a_cache;
  half4_t* __restrict__ reg_b_cache;

  int offset=0;

  const int MODE1=719;
  const int MODE2=719;
 
  int b = blockIdx.x & ((1<<logBC)-1);
  int by = (blockIdx.x >> logBC) & ((1<<(logGridC-logC-5))-1);
  int bz = blockIdx.x >> (logBC+logGridC-logC-5);
  int nc = by<<(logC+5), j=bz<<(logK+5);

  const int y_repeats = (1<<(logC-logGC));
  const int z_repeats = (1<<(logK-logGK));

  float4_t acc[4*y_repeats*z_repeats];
  for(int y=0; y<4*y_repeats*z_repeats; y++) {
      acc[y][0]=0;
      acc[y][1]=0;
      acc[y][2]=0;
      acc[y][3]=0;
  }

  p1_ +=  a*b*c + a*nc;
  p2_ +=  a*b*k + a*j;

  __shared__ half4_t reg_cache[W*128*((1<<logC)+(1<<logK))];
  reg_a_cache = reg_cache;
  reg_b_cache = reg_a_cache+W*128*(1<<logC);

  lds_buffer_vert<2, 2, 2, logW+2, MODE1> abuf;
  lds_buffer_vert<2, 2, 2, logW+2, MODE2> bbuf;

  half4_t temp_a[2][1<<abuf.log_read_repeats], temp_b[2][1<<bbuf.log_read_repeats];
  for(int y=0; y<(1<<abuf.log_read_repeats); y++) {
      int index_x, index_y; 
      bool active;
      abuf.coords(y, index_x, index_y, active);
      if(active) {
        temp_a[0][y] = *(half4_t*)(p1_ + index_x*4 + index_y*64);
        reg_a_cache[index_x + index_y*16] = temp_a[0][y];
      }
  }
  for(int z=0; z<(1<<bbuf.log_read_repeats); z++) {
      int index_x, index_y;
      bool active;
      bbuf.coords(z, index_x, index_y, active);
      if(active) {
        temp_b[0][z] = *(half4_t*)(p2_ + index_x*4 + index_y*64);
        reg_b_cache[index_x + index_y*16] = temp_b[0][z];
      }
  }
  
  uint64_t mask3 = (((threadIdx.x>>4)&3)!=0) ? 0 : 0xffff;

  for(int b_idx=0; b_idx<256; b_idx += 8)
  {
    const __half* p1 = p1_;
    const __half* p2 = p2_;

    __syncthreads();
    half6_t v_reg_a[2][W][y_repeats], v_reg_b[2][W][z_repeats];
    for(int y=0; y<y_repeats; y++) 
      for(int t=0; t<W; t++) {
        const __half* preg = (const __half*) reg_a_cache;
        int index_x = ((threadIdx.x>>4)&3) + 4*t;
        int index_y = (threadIdx.x&15)*2 + (threadIdx.y+(y<<logGC))*32;
        preg += (4*index_x + 49*index_y);
        ((float*)(&v_reg_a[0][t][y]))[0] = *(float*)preg;
        ((float*)(&v_reg_a[0][t][y]))[1] = *(float*)(preg+2);
        ((float*)(&v_reg_a[1][t][y]))[0] = *(float*)(preg+48);
        ((float*)(&v_reg_a[1][t][y]))[1] = *(float*)(preg+48+2);
        ((float*)(&v_reg_a[1][t][y]))[2] = *(float*)(preg+48+4);
      }
    for(int z=0; z<z_repeats; z++) 
      for(int t=0; t<W; t++) {
        const __half* preg = (const __half*) reg_b_cache;
        int index_x = ((threadIdx.x>>4)&3) + 4*t;
        int index_y = (threadIdx.x&15)*2 + (threadIdx.z+(z<<logGC))*32;
        preg += (4*index_x + 49*index_y);
        ((float*)(&v_reg_b[0][t][z]))[0] = *(float*)preg;
        ((float*)(&v_reg_b[0][t][z]))[1] = *(float*)(preg+2);
        ((float*)(&v_reg_b[1][t][z]))[0] = *(float*)(preg+48);
        ((float*)(&v_reg_b[1][t][z]))[1] = *(float*)(preg+48+2);
        ((float*)(&v_reg_b[1][t][z]))[2] = *(float*)(preg+48+4);
      }

    if(b_idx != 256-8) 
    {
      for(int y=0; y<(1<<abuf.log_read_repeats); y++) {
          int index_x, index_y; 
          bool active;
          abuf.coords(y, index_x, index_y, active);
          if(active) {
            temp_a[0][y] = *(half4_t*)(p1_ + (1<<logBC)*a*c + index_x*4 + index_y*64);
          }
      }
      for(int z=0; z<(1<<bbuf.log_read_repeats); z++) {
          int index_x, index_y;
          bool active;
          bbuf.coords(z, index_x, index_y, active);
          if(active) {
            temp_b[0][z] = *(half4_t*)(p2_ + (1<<logBC)*a*k + index_x*4 + index_y*64);
          }
      }
    }

    for(int z=0; z<z_repeats; z++) 
      for(int y=0; y<y_repeats; y++) 
        for(int t=0; t<4; t++) 
        {
          uint64_t* p = (uint64_t*) &v_reg_a[1][t][y];
          uint16_t* q = (uint16_t*)(p+1);
          p[0] = (p[0]>>16) | (uint64_t(q[0])<<48);
          p = (uint64_t*) &v_reg_b[1][t][z];
          q = (uint16_t*)(p+1);
          p[0] = (p[0]>>16) | (uint64_t(q[0])<<48);

          if(t==3) {
            for(int y=0; y<y_repeats; y++) {
              *(uint64_t*)&v_reg_a[0][3][y] &= mask3;
               *(uint64_t*)&v_reg_a[1][3][y] &= mask3;
            }
            for(int z=0; z<z_repeats; z++) {
              *(uint64_t*)&v_reg_b[0][3][z] &= mask3;
              *(uint64_t*)&v_reg_b[1][3][z] &= mask3;
            }
          }

          half4_t& reg_a=*(half4_t*)&v_reg_a[0][t][y];
          half4_t& reg_a1=*(half4_t*)&v_reg_a[1][t][y];
          half4_t& reg_b=*(half4_t*)&v_reg_b[0][t][z];
          half4_t& reg_b1=*(half4_t*)&v_reg_b[1][t][z];

          acc[4*z*y_repeats+4*y+0] = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b, reg_a,   acc[4*z*y_repeats+4*y+0], 0, 0, 0);
          acc[4*z*y_repeats+4*y+1] = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b1, reg_a,  acc[4*z*y_repeats+4*y+1], 0, 0, 0);
          acc[4*z*y_repeats+4*y+2] = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b, reg_a1,  acc[4*z*y_repeats+4*y+2], 0, 0, 0);
          acc[4*z*y_repeats+4*y+3] = __builtin_amdgcn_mfma_f32_16x16x16f16(reg_b1, reg_a1, acc[4*z*y_repeats+4*y+3], 0, 0, 0);
        }
    if(b_idx!=256-8) 
    {
      __syncthreads();
      for(int y=0; y<(1<<abuf.log_read_repeats); y++) {
          int index_x, index_y; 
          bool active;
          abuf.coords(y, index_x, index_y, active);
        if(active)
          reg_a_cache[index_x+index_y*16] = temp_a[0][y];
      }
      for(int z=0; z<(1<<bbuf.log_read_repeats); z++) {
          int index_x, index_y;
          bool active;
          bbuf.coords(z, index_x, index_y, active);
       if(active)
          reg_b_cache[index_x+index_y*16] = temp_b[0][z];
      }
    }
    p1_ += a*c*(1<<logBC);
    p2_ += a*k*(1<<logBC);
    b += 1<<logBC;
  }

  __half* hout = (__half*) out;
  for(int z=0; z<z_repeats; z++) {
    for(int y=0; y<y_repeats; y++) {
        float4_t& reg_c00 = acc[4*z*y_repeats+4*y+0];
        float4_t& reg_c01 = acc[4*z*y_repeats+4*y+1];
        float4_t& reg_c10 = acc[4*z*y_repeats+4*y+2];
        float4_t& reg_c11 = acc[4*z*y_repeats+4*y+3];
       
        transpose4(reinterpret_cast<float*>(&reg_c00));
        transpose4(reinterpret_cast<float*>(&reg_c01));
        transpose4(reinterpret_cast<float*>(&reg_c10));
        transpose4(reinterpret_cast<float*>(&reg_c11));

        // correct output order is 
        // row 0: t0.reg00[0] t0.reg01[0] t1.reg00[0] t1.reg01[0] t2.reg00[0] t2.reg01[0] t3.reg00[0] t3.reg01[0] t16.reg00[0] t16.reg01[0] t17.reg00[0] ...
        // row 1: t0.reg10[0] t0.reg11[0] t1.reg10[0] t1.reg11[0] t2.reg10[0] ....
        // row 2: t0.reg00[1] ...
        //. ...
        // row 8: t4.reg00[0] ...

        // if output is float, it's optimal to write to consecutive locations from consecutive threads: 
        //   out[0] <- block 0 thread 0
        //   out[1] <- block 0 thread 1
        //  etc.
        //   so we have to do an additional permute or suffer substantial performance penalty.
        // 
        // Fortunately, we can use atomicAdd(half2) instead.
        int out_nc = nc + (threadIdx.y+(y<<logGC))*32 + (threadIdx.x & 12)*2;
        int out_j =  j + (threadIdx.z+(z<<logGK))*32 + ((threadIdx.x >> 4)&3)*8 + (threadIdx.x & 3)*2;
        for(int x=0; x<4; x++)
          atomicAdd2(hout+out_nc*k+out_j+2*x*k, half2_t{__half(reg_c00[x]),__half(reg_c01[x])});
        for(int x=0; x<4; x++)
          atomicAdd2(hout+out_nc*k+out_j+2*x*k+k, half2_t{__half(reg_c10[x]),__half(reg_c11[x])});
    }
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
  /*
  if(c<k) {
    int j=c;
    c=k;
    k=j;
    transpose = true;
    auto temp = input1;
    input1 = input2;
    input2 = temp;
  }
*/

  bool allocated = false;
  float* temp;
  if((a!=7*7) && (scratch==0 || scratch_size<c*k*4)) {
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
  if(a!=7*7)
    hipMemset(temp, 0, c*k*4);
  auto cf = convolve_bwdw_1x1x1_xdlops_2<true, 1, 1, 1, 1, 1, 1, 1>;
  dim3 grid, block;

  if(a==7*7 && ((c==512 && k==2048) || (c==2048 && k==512))) {
    auto cf = (c==512) ? convolve_bwdw_1x1x1x7x7_xdlops<9,11> : convolve_bwdw_1x1x1x7x7_xdlops<11,9>;
    const int blocks_a = 8;
    const int log_zone_y = 2;
    const int log_zone_z = 2;
    const int log_blocks_y = 2;
    const int log_blocks_z = 2;
    grid = dim3(blocks_a * ( c/(32<<log_zone_y)) * (k/(32<<log_zone_z)), 1, 1);
    block = dim3(64, 1<<log_blocks_y, 1<<log_blocks_z);
    hipMemsetAsync(output, 0, c*k*2, stream);
    hipLaunchKernelGGL(cf, grid, block, 0, stream,
        n, k, c, a, input1, input2, output);
    call_count++;
    return;
  } else if(a==56*56 && c==64 && k==64) {
    const int Mode1=670, Mode2=455;
    const bool fractional = (a&3);
    const int blocks_a = 256;
    const int log_zone_y = 1;
    const int log_zone_z = 1;
    const int log_blocks_y = 1;
    const int log_blocks_z = 1;
    const int log_W = 1;

    cf = convolve_bwdw_1x1x1_xdlops_2<false, log_zone_y, log_zone_z, log_blocks_y, log_blocks_z, log_W, Mode1, Mode2>;

    grid = dim3(blocks_a, c/(32<<log_zone_y), k/(32<<log_zone_z));
    block = dim3(64, 1<<log_blocks_y, 1<<log_blocks_z);
  } else if(a==56*56 && c==256 && k==64) {
    const int Mode1=430, Mode2=344;
    const bool fractional = (a&3);
    const int blocks_a = 256;
    const int log_zone_y = 3;
    const int log_zone_z = 1;
    const int log_blocks_y = 3;
    const int log_blocks_z = 1;
    const int log_W = 2;

    cf = convolve_bwdw_1x1x1_xdlops_2<false, log_zone_y, log_zone_z, log_blocks_y, log_blocks_z, log_W, Mode1, Mode2>;

    grid = dim3(blocks_a, c/(32<<log_zone_y), k/(32<<log_zone_z));
    block = dim3(64, 1<<log_blocks_y, 1<<log_blocks_z);
  } else if(a==56*56 && c==64 && k==256) {
    const int Mode1=417, Mode2=333;
    const bool fractional = (a&3);
    const int blocks_a = 256;
    const int log_zone_y = 1;
    const int log_zone_z = 3;
    const int log_blocks_y = 1;
    const int log_blocks_z = 3;
    const int log_W = 2;

    cf = convolve_bwdw_1x1x1_xdlops_2<false, log_zone_y, log_zone_z, log_blocks_y, log_blocks_z, log_W, Mode1, Mode2>;

    grid = dim3(blocks_a, c/(32<<log_zone_y), k/(32<<log_zone_z));
    block = dim3(64, 1<<log_blocks_y, 1<<log_blocks_z);
  } else if(a==14*14 && c==256 && k==1024) {
    //[3,2,2,2,1] [561,49] 618k
    const int Mode1=561, Mode2=49;
    const bool fractional = (a&3);
    const int blocks_a = 8;
    const int log_zone_y = 3;
    const int log_zone_z = 2;
    const int log_blocks_y = 2;
    const int log_blocks_z = 2;
    const int log_W = 1;

    cf = convolve_bwdw_1x1x1_xdlops_2<false, log_zone_y, log_zone_z, log_blocks_y, log_blocks_z, log_W, Mode1, Mode2>;

    grid = dim3(blocks_a, c/(32<<log_zone_y), k/(32<<log_zone_z));
    block = dim3(64, 1<<log_blocks_y, 1<<log_blocks_z);
  } else if(a==14*14 && c==1024 && k==256) {
    //  [[2,3,1,3,1],[561,567] 615k
    const int Mode1=561, Mode2=567;
    const bool fractional = (a&3);
    const int blocks_a = 8;
    const int log_zone_y = 2;
    const int log_zone_z = 3;
    const int log_blocks_y = 1;
    const int log_blocks_z = 3;
    const int log_W = 1;

    cf = convolve_bwdw_1x1x1_xdlops_2<false, log_zone_y, log_zone_z, log_blocks_y, log_blocks_z, log_W, Mode1, Mode2>;

    grid = dim3(blocks_a, c/(32<<log_zone_y), k/(32<<log_zone_z));
    block = dim3(64, 1<<log_blocks_y, 1<<log_blocks_z);
  } else if(a==28*28 && c==128 && k==512) {
    //  [[2,3,1,3,1],[561,567] 615k
    const int Mode1=717, Mode2=543;
    const bool fractional = (a&3);
    const int blocks_a = 32;
    const int log_zone_y = 2;
    const int log_zone_z = 2;
    const int log_blocks_y = 2;
    const int log_blocks_z = 1;
    const int log_W = 1;

    cf = convolve_bwdw_1x1x1_xdlops_2<false, log_zone_y, log_zone_z, log_blocks_y, log_blocks_z, log_W, Mode1, Mode2>;

    grid = dim3(blocks_a, c/(32<<log_zone_y), k/(32<<log_zone_z));
    block = dim3(64, 1<<log_blocks_y, 1<<log_blocks_z);
  }
  else {
    printf("ERROR: unsupported scenario %d %d %d\n", a, c, k);
  }

    hipLaunchKernelGGL(cf, grid, block, 0, stream,
        n, k, c, a, input1, input2, temp);

  hipLaunchKernelGGL(cvt_fp32_fp16, 1, min(1024,k*c), 0, stream,
    temp, output, k*c);
  if(allocated)
    hipFree(temp);

  call_count++;
}
