#if TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "external/rocprim_archive/hipcub/include/hipcub/hipcub.hpp"
#include "tensorflow/core/kernels/reduce_thunk_op.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
namespace gpuprim = ::hipcub;

namespace tensorflow {

// each block does a grid strided loop and reduces its values locally
// the case of one block is used for low latency small reductions to scalars
__global__ void BlockReduceKernel(const float* in, float* out, int num_elems,
                                  float initVal) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const int gid = bid * blockDim.x + tid;
  const int stride = blockDim.x * gridDim.x;

  float sum = initVal;
  if (gid < num_elems) {
    sum = in[gid];
    for (int pos = gid + stride; pos < num_elems; pos += stride) {
      sum = sum + in[pos];
    }
  }

  typedef gpuprim::BlockReduce<float, 256> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  // only include input values in the reduction
  //
  // elements: -----------------
  // grid:     |====|====|====|====|====|
  const int num_elements_to_reduce = max(min(256 - bid * blockDim.x, 256), 0);

  sum = BlockReduce(temp_storage)
            .Reduce(sum, gpuprim::Sum(), num_elements_to_reduce);

  if (tid == 0) out[bid] = sum;
}

void ReduceKernelLaunch(gpuStream_t gpu_stream,
                        const se::DeviceMemoryBase& input,
                        se::DeviceMemoryBase* output, float init_value,
                        int64 reduction_dimension) {
  VLOG(3) << "ReduceKernelLaunch()";
  VLOG(3) << "input device memory size: " << input.size();
  VLOG(3) << "output device memory size: " << output->size();
  VLOG(3) << "init value: " << init_value;
  VLOG(3) << "reudction dimension: " << reduction_dimension;

  const int num_blocks = 1;
  const int num_threads = 256;
  TF_CHECK_OK(GpuLaunchKernel(BlockReduceKernel, dim3(num_blocks),
                              dim3(num_threads), 0, gpu_stream,
                              reinterpret_cast<const float*>(input.opaque()),
                              reinterpret_cast<float*>(output->opaque()),
                              input.size() / sizeof(float), init_value));
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
