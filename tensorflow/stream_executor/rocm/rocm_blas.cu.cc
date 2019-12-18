#include "rocm/include/hip/hip_runtime.h"

namespace stream_executor {
namespace gpu {


template <class T>
__global__ void SplitComplexKernel(T* b, const T* a, int n, bool conjugate)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
  if(i<n)
  {
    b[i]=a[2*i+0];
    b[i+n]=conjugate ? -a[2*i+1] : a[2*i+1];
  }
}

template <class T>
__global__ void CombineComplexKernel(T* b, const T* a, int n)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
  if(i<n)
  {
    b[2*i+0]=a[i];
    b[2*i+1]=a[i+n];
  }
}


template <class T, int N>
__global__ void AddComplexKernel(T* b, const T* a, int n)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*blockDim.x*gridDim.x;
  if(i<n)
    b[i] += a[i]*N;
}

template <class T>
void SplitComplex(T* b, const T* a, int n, void* stream, bool conjugate)
{
  dim3 grid(min(1024,(n+1023)/1024),  (n+1024*1024-1)/(1024*1024), 1);
  hipLaunchKernelGGL(SplitComplexKernel<T>, grid, 1024, 0, (hipStream_t)stream, b, a, n, conjugate);
}

template <class T>
void CombineComplex(T* b, const T* a, int n, void* stream)
{
  dim3 grid(min(1024,(n+1023)/1024),  (n+1024*1024-1)/(1024*1024), 1);
  hipLaunchKernelGGL(CombineComplexKernel<T>, grid, 1024, 0, (hipStream_t)stream, b, a, n);
}

template <class T, int N>
void AddComplex(T* b, const T* a, int n, void* stream)
{
  dim3 grid(min(1024,(n+1023)/1024),  (n+1024*1024-1)/(1024*1024), 1);
  hipLaunchKernelGGL(AddComplexKernel<T,N>, grid, 1024, 0, (hipStream_t)stream, b, a, n);
}

template void SplitComplex<float>(float* b, const float* a, int n, void* stream, bool);
template void CombineComplex<float>(float* b, const float* a, int n, void* stream);
template void AddComplex<float,1>(float* b, const float* a, int n, void* stream);
template void AddComplex<float,-1>(float* b, const float* a, int n, void* stream);

template void SplitComplex<double>(double* b, const double* a, int n, void* stream, bool);
template void CombineComplex<double>(double* b, const double* a, int n, void* stream);
template void AddComplex<double,1>(double* b, const double* a, int n, void* stream);
template void AddComplex<double,-1>(double* b, const double* a, int n, void* stream);

};
};