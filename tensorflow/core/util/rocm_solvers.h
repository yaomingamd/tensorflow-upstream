/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_ROCM_SOLVERS_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_ROCM_SOLVERS_H_

// This header declares the class ROCmSolver, which contains wrappers of linear
// algebra solvers in the rocBlas and rocSolverDN libraries for use in TensorFlow
// kernels.

#if TENSORFLOW_USE_ROCM

#include <functional>
#include <vector>

#include "rocm/include/hip/hip_complex.h"
#include "rocm/include/rocblas.h"
#include "rocm/include/rocsolver.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/blas.h"

namespace tensorflow {

// Type traits to get ROCm complex types from std::complex<T>.
template <typename T>
struct ROCmComplexT {
  typedef T type;
};
template <>
struct ROCmComplexT<std::complex<float>> {
  typedef hipComplex type;
};
template <>
struct ROCmComplexT<std::complex<double>> {
  typedef hipDoubleComplex type;
};
// Converts pointers of std::complex<> to pointers of
// ROCmComplex/ROCmDoubleComplex. No type conversion for non-complex types.
template <typename T>
inline const typename ROCmComplexT<T>::type* ROCmComplex(const T* p) {
  return reinterpret_cast<const typename ROCmComplexT<T>::type*>(p);
}
template <typename T>
inline typename ROCmComplexT<T>::type* ROCmComplex(T* p) {
  return reinterpret_cast<typename ROCmComplexT<T>::type*>(p);
}

template <typename Scalar>
class ScratchSpace;

class ROCmSolver {
 public:
  // This object stores a pointer to context, which must outlive it.
  explicit ROCmSolver(OpKernelContext* context);
  virtual ~ROCmSolver();

  // Allocates a temporary tensor that will live for the duration of the
  // ROCmSolver object.
  Status allocate_scoped_tensor(DataType type, const TensorShape& shape,
                                Tensor* scoped_tensor);

  Status forward_input_or_allocate_scoped_tensor(
      gtl::ArraySlice<int> candidate_input_indices, DataType type,
      const TensorShape& shape, Tensor* input_alias_or_new_scoped_tensor);

  OpKernelContext* context() { return context_; }

  template <typename Scalar>
  ScratchSpace<Scalar> GetScratchSpace(const TensorShape& shape,
                                       const std::string& debug_info,
                                       bool on_host);
  template <typename Scalar>
  ScratchSpace<Scalar> GetScratchSpace(int64 size,
                                       const std::string& debug_info,
                                       bool on_host);
  // Returns a DeviceLapackInfo that will live for the duration of the
  // ROCmSolver object.
  inline DeviceLapackInfo GetDeviceLapackInfo(int64 size,
                                              const std::string& debug_info);

  // ====================================================================
  // Wrappers for ROCSolver start here
  //
  // The method names below
  // map to those in ROCSolver, which follow the naming
  // convention in LAPACK see

  // LU factorization.
  // Computes LU factorization with partial pivoting P * A = L * U.
  template <typename Scalar>
  Status getrf(rocblas_handle handle, int m, int n, Scalar* dev_A, int lda, int* dev_pivots,
               int* dev_lapack_info) TF_MUST_USE_RESULT;

  // Uses LU factorization to solve A * X = B.
  template <typename Scalar>
  Status getrs(rocblas_handle handle, int n, int nrhs, const Scalar* A,
               int lda, const int* pivots, Scalar* B, int ldb,
               int* dev_lapack_info) const TF_MUST_USE_RESULT;

  // Computes partially pivoted LU factorizations for a batch of small matrices.
  // Returns Status::OK() if the kernel was launched successfully. See:
  template <typename Scalar>
  Status getrfBatched(rocblas_handle handle, int n, const Scalar* const host_a_dev_ptrs[], int lda,
                      int* dev_pivots, DeviceLapackInfo* dev_lapack_info,
                      int batch_size) TF_MUST_USE_RESULT;

  // Batched linear solver using LU factorization from getrfBatched.
  // Notice that lapack_info is returned on the host, as opposed to
  // most of the other functions that return it on the device. 
  template <typename Scalar>
  Status getrsBatched(rocblas_handle handle, int n, int nrhs,
                      const Scalar* const dev_Aarray[], int lda,
                      const int* devIpiv, const Scalar* const dev_Barray[],
                      int ldb, int* host_lapack_info,
                      int batch_size) TF_MUST_USE_RESULT;

  template <typename Scalar>
  Status Trsm(rocblas_side side, rocblas_fill uplo, rocblas_operation trans,
              rocblas_diagonal diag, int m, int n, const Scalar* alpha,
              const Scalar* A, int lda, Scalar* B, int ldb);

 private:
  OpKernelContext* context_;  // not owned.
  hipStream_t hip_stream_;
  rocblas_handle rocm_blas_handle_;
  std::vector<TensorReference> scratch_tensor_refs_;

  TF_DISALLOW_COPY_AND_ASSIGN(ROCmSolver);
};

// Helper class to allocate scratch memory and keep track of debug info.
// Mostly a thin wrapper around Tensor & allocate_temp.
template <typename Scalar>
class ScratchSpace {
 public:
  ScratchSpace(OpKernelContext* context, int64 size, bool on_host)
      : ScratchSpace(context, TensorShape({size}), "", on_host) {}

  ScratchSpace(OpKernelContext* context, int64 size, const string& debug_info,
               bool on_host)
      : ScratchSpace(context, TensorShape({size}), debug_info, on_host) {}

  ScratchSpace(OpKernelContext* context, const TensorShape& shape,
               const string& debug_info, bool on_host)
      : context_(context), debug_info_(debug_info), on_host_(on_host) {
    AllocatorAttributes alloc_attr;
    if (on_host) {
      // Allocate pinned memory on the host to avoid unnecessary
      // synchronization.
      alloc_attr.set_on_host(true);
      alloc_attr.set_gpu_compatible(true);
    }
    TF_CHECK_OK(context->allocate_temp(DataTypeToEnum<Scalar>::value, shape,
                                       &scratch_tensor_, alloc_attr));
  }

  virtual ~ScratchSpace() {}

  Scalar* mutable_data() {
    return scratch_tensor_.template flat<Scalar>().data();
  }
  const Scalar* data() const {
    return scratch_tensor_.template flat<Scalar>().data();
  }
  Scalar& operator()(int64 i) {
    return scratch_tensor_.template flat<Scalar>()(i);
  }
  const Scalar& operator()(int64 i) const {
    return scratch_tensor_.template flat<Scalar>()(i);
  }
  int64 bytes() const { return scratch_tensor_.TotalBytes(); }
  int64 size() const { return scratch_tensor_.NumElements(); }
  const string& debug_info() const { return debug_info_; }

  Tensor& tensor() { return scratch_tensor_; }
  const Tensor& tensor() const { return scratch_tensor_; }

  // Returns true if this ScratchSpace is in host memory.
  bool on_host() const { return on_host_; }

 protected:
  OpKernelContext* context() const { return context_; }

 private:
  OpKernelContext* context_;  // not owned
  const string debug_info_;
  const bool on_host_;
  Tensor scratch_tensor_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_ROCM_SOLVERS_H_
