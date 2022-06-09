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

#include "rocm/include/rocblas.h"

#include "tensorflow/stream_executor/rocm/rocm_blas.h"

#define EIGEN_USE_GPU
#include <assert.h>

#include <complex>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "rocm/rocm_config.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kRocBlasPlugin);

extern void broadcast_fp32(void* stream, float* dst, int dst_stride,
                           int batches, float* src, int size);

namespace wrap {

#ifdef PLATFORM_GOOGLE
#define STREAM_EXECUTOR_ROCBLAS_WRAP(__name)                       \
  struct WrapperShim__##__name {                                   \
    static const char *kName;                                      \
    template <typename... Args>                                    \
    rocblas_status operator()(GpuExecutor *parent, Args... args) { \
      gpu::ScopedActivateExecutorContext sac{parent};              \
      return ::__name(args...);                                    \
    }                                                              \
  } __name;                                                        \
  const char *WrapperShim__##__name::kName = #__name;

#define STREAM_EXECUTOR_ROCBLAS_V2_WRAP(__name) \
  STREAM_EXECUTOR_ROCBLAS_WRAP(__name)

#else

#define STREAM_EXECUTOR_ROCBLAS_WRAP(__name)                              \
  struct DynLoadShim__##__name {                                          \
    static const char *kName;                                             \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;          \
    static void *GetDsoHandle() {                                         \
      auto s = internal::CachedDsoLoader::GetRocblasDsoHandle();          \
      return s.ValueOrDie();                                              \
    }                                                                     \
    static FuncPtrT LoadOrDie() {                                         \
      void *f;                                                            \
      auto s = port::Env::Default()->GetSymbolFromLibrary(GetDsoHandle(), \
                                                          kName, &f);     \
      CHECK(s.ok()) << "could not find " << kName                         \
                    << " in rocblas DSO; dlerror: " << s.error_message(); \
      return reinterpret_cast<FuncPtrT>(f);                               \
    }                                                                     \
    static FuncPtrT DynLoad() {                                           \
      static FuncPtrT f = LoadOrDie();                                    \
      return f;                                                           \
    }                                                                     \
    template <typename... Args>                                           \
    rocblas_status operator()(GpuExecutor *parent, Args... args) {        \
      gpu::ScopedActivateExecutorContext sac{parent};                     \
      return DynLoad()(args...);                                          \
    }                                                                     \
  } __name;                                                               \
  const char *DynLoadShim__##__name::kName = #__name;

#define STREAM_EXECUTOR_ROCBLAS_V2_WRAP(__name) \
  STREAM_EXECUTOR_ROCBLAS_WRAP(__name)

#endif

// clang-format off
#define ROCBLAS_BLAS_ROUTINE_EACH(__macro)  \
  __macro(rocblas_snrm2)                    \
  __macro(rocblas_dnrm2)                    \
  /*__macro(rocblas_scnrm2)                   \
    __macro(rocblas_dznrm2)                */ \
  __macro(rocblas_sdot)                     \
  __macro(rocblas_ddot)                     \
  /*__macro(rocblas_cdotu)                    \
    __macro(rocblas_cdotc)                    \
    __macro(rocblas_zdotu)                    \
    __macro(rocblas_zdotc)                 */ \
  __macro(rocblas_sscal)                    \
  __macro(rocblas_dscal)                    \
  __macro(rocblas_cscal)                    \
    __macro(rocblas_csscal)                   \
    __macro(rocblas_zscal)                    \
    __macro(rocblas_zdscal)                 \
  __macro(rocblas_saxpy)                    \
  __macro(rocblas_daxpy)                    \
  /*__macro(rocblas_caxpy)                    \
    __macro(rocblas_zaxpy)                 */ \
  __macro(rocblas_scopy)                    \
  __macro(rocblas_dcopy)                    \
  /*__macro(rocblas_ccopy)                    \
    __macro(rocblas_zcopy)                 */ \
  __macro(rocblas_sswap)                    \
  __macro(rocblas_dswap)                    \
  /*__macro(rocblas_cswap)                    \
    __macro(rocblas_zswap)                 */ \
  __macro(rocblas_isamax)                   \
  __macro(rocblas_idamax)                   \
  /*__macro(rocblas_icamax)                   \
    __macro(rocblas_izamax)                */ \
  __macro(rocblas_isamin)                   \
  __macro(rocblas_idamin)                   \
  /*__macro(rocblas_icamin)                   \
    __macro(rocblas_izamin)                */ \
  __macro(rocblas_sasum)                    \
  __macro(rocblas_dasum)                    \
  /*__macro(rocblas_scasum)                   \
    __macro(rocblas_dzasum)                   \
    __macro(rocblas_srot)                     \
    __macro(rocblas_drot)                     \
    __macro(rocblas_crot)                     \
    __macro(rocblas_csrot)                    \
    __macro(rocblas_zrot)                     \
    __macro(rocblas_zdrot)                    \
    __macro(rocblas_srotg)                    \
    __macro(rocblas_drotg)                    \
    __macro(rocblas_Crotg)                    \
    __macro(rocblas_crotg)                    \
    __macro(rocblas_zrotm)                    \
    __macro(rocblas_drotm)                    \
    __macro(rocblas_srotmg)                   \
    __macro(rocblas_drotmg)                */ \
  __macro(rocblas_sgemv)                    \
  __macro(rocblas_dgemv)                    \
  __macro(rocblas_cgemv)                    \
    __macro(rocblas_zgemv)                    \
  /*  __macro(rocblas_sgbmv)                    \
    __macro(rocblas_dgbmv)                    \
    __macro(rocblas_cgbmv)                    \
    __macro(rocblas_zgbmv)                    \
    __macro(rocblas_strmv)                    \
    __macro(rocblas_dtrmv)                    \
    __macro(rocblas_ctrmv)                    \
    __macro(rocblas_ztrmv)                    \
    __macro(rocblas_stbmv)                    \
    __macro(rocblas_dtbmv)                    \
    __macro(rocblas_ctbmv)                    \
    __macro(rocblas_ztbmv)                    \
    __macro(rocblas_stpmv)                    \
    __macro(rocblas_dtpmv)                    \
    __macro(rocblas_ctpmv)                    \
    __macro(rocblas_ztpmv)                    \
    __macro(rocblas_strsv)                    \
    __macro(rocblas_dtrsv)                    \
    __macro(rocblas_ctrsv)                    \
    __macro(rocblas_ztrsv)                    \
    __macro(rocblas_stpsv)                    \
    __macro(rocblas_dtpsv)                    \
    __macro(rocblas_ctpsv)                    \
    __macro(rocblas_ztpsv)                    \
    __macro(rocblas_stbsv)                    \
    __macro(rocblas_dtbsv)                    \
    __macro(rocblas_ctbsv)                    \
    __macro(rocblas_ztbsv)                    \
    __macro(rocblas_ssymv)                    \
    __macro(rocblas_dsymv)                    \
    __macro(rocblas_csymv)                    \
    __macro(rocblas_zsymv)                    \
    __macro(rocblas_chemv)                    \
    __macro(rocblas_zhemv)                    \
    __macro(rocblas_ssbmv)                    \
    __macro(rocblas_dsbmv)                    \
    __macro(rocblas_chbmv)                    \
    __macro(rocblas_zhbmv)                    \
    __macro(rocblas_sspmv)                    \
    __macro(rocblas_dspmv)                    \
    __macro(rocblas_chpmv)                    \
    __macro(rocblas_zhpmv)                 */ \
  __macro(rocblas_sger)                     \
  __macro(rocblas_dger)                     \
  /*__macro(rocblas_cgeru)                    \
    __macro(rocblas_cgerc)                    \
    __macro(rocblas_zgeru)                    \
    __macro(rocblas_zgerc)                 */ \
  __macro(rocblas_ssyr)                     \
  __macro(rocblas_dsyr)                     \
  /*__macro(rocblas_csyr)                     \
    __macro(rocblas_zsyr)                     \
    __macro(rocblas_cher)                     \
    __macro(rocblas_zher)                     \
    __macro(rocblas_sspr)                     \
    __macro(rocblas_dspr)                     \
    __macro(rocblas_chpr)                     \
    __macro(rocblas_zhpr)                     \
    __macro(rocblas_ssyr2)                    \
    __macro(rocblas_dsyr2)                    \
    __macro(rocblas_csyr2)                    \
    __macro(rocblas_zsyr2)                    \
    __macro(rocblas_cher2)                    \
    __macro(rocblas_zher2)                    \
    __macro(rocblas_sspr2)                    \
    __macro(rocblas_dspr2)                    \
    __macro(rocblas_chpr2)                    \
    __macro(rocblas_zhpr2)                 */ \
  __macro(rocblas_sgemm)                    \
  __macro(rocblas_dgemm)                    \
  __macro(rocblas_hgemm)                    \
  __macro(rocblas_cgemm)                    \
    __macro(rocblas_zgemm)                    \
  /*  __macro(rocblas_ssyrk)                    \
    __macro(rocblas_dsyrk)                    \
    __macro(rocblas_csyrk)                    \
    __macro(rocblas_zsyrk)                    \
    __macro(rocblas_cherk)                    \
    __macro(rocblas_zherk)                    \
    __macro(rocblas_ssyr2k)                   \
    __macro(rocblas_dsyr2k)                   \
    __macro(rocblas_csyr2k)                   \
    __macro(rocblas_zsyr2k)                   \
    __macro(rocblas_cher2k)                   \
    __macro(rocblas_zher2k)                   \
    __macro(rocblas_ssyrkx)                   \
    __macro(rocblas_dsyrkx)                   \
    __macro(rocblas_csyrkx)                   \
    __macro(rocblas_zsyrkx)                   \
    __macro(rocblas_cherkx)                   \
    __macro(rocblas_zherkx)                   \
    __macro(rocblas_ssymm)                    \
    __macro(rocblas_dsymm)                    \
    __macro(rocblas_csymm)                    \
    __macro(rocblas_zsymm)                    \
    __macro(rocblas_chemm)                    \
    __macro(rocblas_zhemm)                 */ \
  __macro(rocblas_strsm)                    \
  __macro(rocblas_dtrsm)                    \
  /*__macro(rocblas_ctrsm)                    \
    __macro(rocblas_ztrsm)                    \
    __macro(rocblas_strmm)                    \
    __macro(rocblas_dtrmm)                    \
    __macro(rocblas_ctrmm)                    \
    __macro(rocblas_ztrmm)                 */ \
  __macro(rocblas_sgeam)                    \
  __macro(rocblas_dgeam)                    \
  __macro(rocblas_gemm_ex)                  \
  /*__macro(rocblas_cgeam)                  \
    __macro(rocblas_zgeam)                  \
    __macro(rocblas_sdgmm)                  \
    __macro(rocblas_ddgmm)                  \
    __macro(rocblas_cdgmm)                  \
    __macro(rocblas_zdgmm) */
// clang-format on

STREAM_EXECUTOR_ROCBLAS_V2_WRAP(rocblas_create_handle)
STREAM_EXECUTOR_ROCBLAS_V2_WRAP(rocblas_destroy_handle)
STREAM_EXECUTOR_ROCBLAS_V2_WRAP(rocblas_set_stream)
// STREAM_EXECUTOR_ROCBLAS_V2_WRAP(rocblas_set_pointer_mode)
// STREAM_EXECUTOR_ROCBLAS_V2_WRAP(rocblas_get_pointer_mode)
// STREAM_EXECUTOR_ROCBLAS_WRAP(rocblas_sgemm_batched)
STREAM_EXECUTOR_ROCBLAS_WRAP(rocblas_hgemm_strided_batched)
STREAM_EXECUTOR_ROCBLAS_WRAP(rocblas_sgemm_strided_batched)
// STREAM_EXECUTOR_ROCBLAS_WRAP(rocblas_dgemm_batched)
STREAM_EXECUTOR_ROCBLAS_WRAP(rocblas_dgemm_strided_batched)
STREAM_EXECUTOR_ROCBLAS_WRAP(rocblas_cgemm_strided_batched)
STREAM_EXECUTOR_ROCBLAS_WRAP(rocblas_zgemm_strided_batched)
// STREAM_EXECUTOR_ROCBLAS_WRAP(rocblas_cgemm_batched)
// STREAM_EXECUTOR_ROCBLAS_WRAP(rocblas_zgemm_batched)
ROCBLAS_BLAS_ROUTINE_EACH(STREAM_EXECUTOR_ROCBLAS_V2_WRAP)

}  // namespace wrap

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const DeviceMemory<T> &a) {
  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemory(a));
}
template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const T &a) {
  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(&a);
}
template <class T>
typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    DeviceMemory<T> *a) {
  return reinterpret_cast<
      typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemoryMutable(a));
}

static void blas_log(const char *c) {}


static string ToString(rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    default:
      return absl::StrCat("<invalid rocBLAS status: ", status, ">");
  }
}

bool ROCMBlas::Init() {
  rocblas_status ret = wrap::rocblas_create_handle(parent_, &blas_);
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to create rocBLAS handle: " << ToString(ret);
    return false;
  }

  return true;
}

ROCMBlas::ROCMBlas(gpu::GpuExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr) {}

ROCMBlas::~ROCMBlas() {
  if (blas_ != nullptr) {
    wrap::rocblas_destroy_handle(parent_, blas_);
  }
}

bool ROCMBlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  rocblas_status ret =
      wrap::rocblas_set_stream(parent_, blas_, AsGpuStreamValue(stream));
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to set stream for rocBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming blas arguments into rocBLAS arguments.

rocblas_operation ROCMBlasTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return rocblas_operation_none;
    case blas::Transpose::kTranspose:
      return rocblas_operation_transpose;
    case blas::Transpose::kConjugateTranspose:
      return rocblas_operation_conjugate_transpose;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

rocblas_fill ROCMBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return rocblas_fill_upper;
    case blas::UpperLower::kLower:
      return rocblas_fill_lower;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

rocblas_diagonal ROCMBlasDiagonal(blas::Diagonal diag) {
  switch (diag) {
    case blas::Diagonal::kUnit:
      return rocblas_diagonal_unit;
    case blas::Diagonal::kNonUnit:
      return rocblas_diagonal_non_unit;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

rocblas_side ROCMBlasSide(blas::Side side) {
  switch (side) {
    case blas::Side::kLeft:
      return rocblas_side_left;
    case blas::Side::kRight:
      return rocblas_side_right;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

}  // namespace

template <typename FuncT, typename... Args>
bool ROCMBlas::DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args... args) {
  absl::MutexLock lock{&mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  rocblas_status ret = rocblas_func(parent_, blas_, args...);
  if (err_on_failure && ret != rocblas_status_success) {
    LOG(ERROR) << "failed to run ROCBLAS routine " << rocblas_func.kName << ": "
               << ToString(ret);
  }
  return ret == rocblas_status_success;
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(wrap::rocblas_sasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(wrap::rocblas_dasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the ASUM operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the ASUM operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_saxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_daxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the AXPY operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the AXPY operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_scopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_dcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the COPY operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the COPY operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
  blas_log("DoBlasDot");
  return DoBlasInternal(
      wrap::rocblas_sdot, stream, false /* = pointer_mode_host */, elem_count,
      GpuMemory(x), incx, GpuMemory(y), incy, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
  blas_log("DoBlasDot");
  return DoBlasInternal(
      wrap::rocblas_ddot, stream, false /* = pointer_mode_host */, elem_count,
      GpuMemory(x), incx, GpuMemory(y), incy, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the DOT operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the DOT operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the DOT operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the DOT operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(wrap::rocblas_snrm2, stream,
                        false /* = pointer_mode_host */, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(wrap::rocblas_dnrm2, stream,
                        false /* = pointer_mode_host */, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the NRM2 operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the NRM2 operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROT operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROT operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROT operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROT operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTG operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTG operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTG operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTG operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTM operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTM operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTMG operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTMG operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  blas_log("DoBlasScal<float>");
  return DoBlasInternal(wrap::rocblas_sscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_dscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_csscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_zdscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_cscal, stream,
                        true /* = pointer_mode_host */, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_zscal, stream,
                        true /* = pointer_mode_host */, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_sswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        GpuMemoryMutable(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_dswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        GpuMemoryMutable(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SWAP operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SWAP operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(wrap::rocblas_isamax, stream,
                        false /* = pointer_mode_host */, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(wrap::rocblas_idamax, stream,
                        false /* = pointer_mode_host */, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the AMAX operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the AMAX operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::rocblas_isamin, stream, false /* = pointer_mode_host */, elem_count,
      GpuComplex(GpuMemory(x)), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::rocblas_idamin, stream, false /* = pointer_mode_host */, elem_count,
      GpuComplex(GpuMemory(x)), incx, GpuMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the AMIN operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the AMIN operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GBMV operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GBMV operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GBMV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GBMV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_sgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_dgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_cgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  blas_log("DoBlasGemv\n");
  return DoBlasInternal(
      wrap::rocblas_zgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(
      wrap::rocblas_sger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      GpuMemory(x), incx, GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(
      wrap::rocblas_dger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      GpuMemory(x), incx, GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the GER operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the GER operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the GERU operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the GERU operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HBMV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HBMV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HEMV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HEMV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the HER operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the HER operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the HER2 operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the HER2 operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HPMV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HPMV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the HPR operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the HPR operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the HPR2 operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the HPR2 operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SBMV operation "
             << "for the \"complex<float>\" datatype";

  return false;
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SBMV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SPMV operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SPMV operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the SPR operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the SPR operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the SPR2 operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the SPR2 operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMV operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMV operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(wrap::rocblas_ssyr, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(wrap::rocblas_dsyr, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2 operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2 operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBMV operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBMV operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBMV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBMV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBSV operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBSV operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBSV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBSV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPMV operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPMV operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPMV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPMV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPSV operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPSV operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPSV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPSV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMV operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMV operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSV operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSV operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSV operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSV operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

template <class T, class V, class W>
bool ROCMBlas::DoBlasGemmImpl(Stream *stream, blas::GemmCallContext<T> ctx, V strided_fun, W fun)
{
  if(ctx.output_profile_result)
  {
      LOG(ERROR) << "rocBLAS does not currently support profiling";
      return false;
  }
  const T alpha(ctx.alpha);
  const T beta(ctx.beta);
  typedef typename std::conditional<std::is_same<T, Eigen::half>::value, rocblas_half, T>::type DT;
  // we don't handle CallContext - this would not be called when XDLOPS are available (MI100 and up)
  using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;
  if(ctx.stride_a>=0)
  {
    return DoBlasInternal(
        strided_fun, stream,
        false, /* pointer_mode_host */
        ROCMBlasTranspose(ctx.transa), ROCMBlasTranspose(ctx.transb), ctx.m, ctx.n, ctx.k,
        reinterpret_cast<const MAPPED_T *>(&alpha),
        reinterpret_cast<const MAPPED_T *>(GpuMemory(*ctx.pa)), ctx.lda, ctx.stride_a,
        reinterpret_cast<const MAPPED_T *>(GpuMemory(*ctx.pb)), ctx.ldb, ctx.stride_b,
        reinterpret_cast<const MAPPED_T *>(&beta),
        reinterpret_cast<MAPPED_T *>(GpuMemoryMutable(ctx.c)), ctx.ldc, ctx.stride_c,
        ctx.batch_count);
  }
  else
  {
      return DoBlasInternal(
        fun, stream, /* pointer_mode_host = */ true,
        ROCMBlasTranspose(ctx.transa), ROCMBlasTranspose(ctx.transb), ctx.m, ctx.n, ctx.k,
        reinterpret_cast<const MAPPED_T *>(&alpha),
        reinterpret_cast<const MAPPED_T *>(GpuMemory(*ctx.pa)), ctx.lda, 
        reinterpret_cast<const MAPPED_T *>(GpuMemory(*ctx.pb)), ctx.ldb, 
        reinterpret_cast<const MAPPED_T *>(&beta),
        reinterpret_cast<MAPPED_T *>(GpuMemoryMutable(ctx.c)), ctx.ldc);
  }
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::GemmCallContext<Eigen::half> ctx, blas::ProfileResult *output_profile_result)
{
  bool hasXDLOPS = false, gfx90a = false;
  auto status = GpuDriver::GetMFMASupport(hasXDLOPS, gfx90a);
  bool is_backprop =
      (ctx.context == blas::CallContext::kBackpropInput1) ||
      (ctx.context == blas::CallContext::kBackpropInput2);

  if(hasXDLOPS)
  {
    if(ctx.stride_a>=0)
    {
      LOG(ERROR)<<"ROCMBlas::DoBlasGemm does not support denorm workaround in strided GEMM";
      exit(0);
    }
    uint32_t flags = rocblas_gemm_flags_none;
#if TF_ROCM_VERSION >= 50000
    if(is_backprop && gfx90a)
      flags = rocblas_gemm_flags_fp16_alt_impl;
#endif
    return DoBlasInternal(
      wrap::rocblas_gemm_ex, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(ctx.transa), ROCMBlasTranspose(ctx.transb),
      (rocblas_int)ctx.m, (rocblas_int)ctx.n, (rocblas_int)ctx.k,
      reinterpret_cast<const void*>(&ctx.alpha),
      reinterpret_cast<const void*>(GpuMemory(*ctx.pa)), rocblas_datatype_f16_r, ctx.lda,
      reinterpret_cast<const void*>(GpuMemory(*ctx.pb)), rocblas_datatype_f16_r, ctx.ldb,
      reinterpret_cast<const void*>(&ctx.beta),
      reinterpret_cast<const void*>(GpuMemoryMutable(ctx.c)),
      rocblas_datatype_f16_r, ctx.ldc,
      reinterpret_cast<void*>(GpuMemoryMutable(ctx.c)), rocblas_datatype_f16_r, ctx.ldc,
          rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, flags);
  }

  return DoBlasGemmImpl(stream, ctx, wrap::rocblas_hgemm_strided_batched, wrap::rocblas_hgemm);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::GemmCallContext<float> ctx, blas::ProfileResult *output_profile_result)
{
  return DoBlasGemmImpl(stream, ctx, wrap::rocblas_sgemm_strided_batched, wrap::rocblas_sgemm);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::GemmCallContext<double> ctx, blas::ProfileResult *output_profile_result)
{
  return DoBlasGemmImpl(stream, ctx, wrap::rocblas_dgemm_strided_batched, wrap::rocblas_dgemm);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::GemmCallContext<std::complex<float> > ctx, blas::ProfileResult *output_profile_result)
{
  return DoBlasGemmImpl(stream, ctx, wrap::rocblas_cgemm_strided_batched, wrap::rocblas_cgemm);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::GemmCallContext<std::complex<double> > ctx, blas::ProfileResult *output_profile_result)
{
  return DoBlasGemmImpl(stream, ctx, wrap::rocblas_zgemm_strided_batched, wrap::rocblas_zgemm);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n, float alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &x,
    int incx, float beta, DeviceMemory<float> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n, double alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &x,
    int incx, double beta, DeviceMemory<double> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n,
    std::complex<float> alpha, const DeviceMemory<std::complex<float>> &a,
    int lda, const DeviceMemory<std::complex<float>> &x, int incx,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n,
    std::complex<double> alpha, const DeviceMemory<std::complex<double>> &a,
    int lda, const DeviceMemory<std::complex<double>> &x, int incx,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

template <typename T>
bool ROCMBlas::DoBlasGemvWithProfilingImpl(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n, const T &alpha,
    const DeviceMemory<T> &a, int lda, const DeviceMemory<T> &x, int incx,
    const T &beta, DeviceMemory<T> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  // ROCM TODO: properly implement the interface
  return false;
}

bool ROCMBlas::GetBlasGemmAlgorithms(
    std::vector<blas::AlgorithmType> *out_algorithms) {
  // ROCM TODO: properly implement the interface
  return true;
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::GemmCallContext<int8, int32> ctx, blas::ProfileResult *output_profile_result) {
  LOG(ERROR)
      << "rocBLAS does not currently support the GEMMwithAlgorithm operation "
      << "for the \"int8\" datatype";
  return false;
}

struct memory_copy_op {
    char* src_ptr;
    char* dst_ptr;
    uint64_t size;
    uint64_t count;
    uint64_t dst_stride;
};


static bool fold(memory_copy_op& y, const memory_copy_op& x) 
{
  bool misaligned = (x.size & 3) 
    || (reinterpret_cast<uint64_t>(x.dst_ptr) & 3)
    || (reinterpret_cast<uint64_t>(x.src_ptr) & 3)
    || (reinterpret_cast<uint64_t>(y.dst_ptr) & 3)
    || (reinterpret_cast<uint64_t>(y.src_ptr) & 3);
  int64_t dst_step = reinterpret_cast<int64_t>(x.dst_ptr)-reinterpret_cast<int64_t>(y.dst_ptr);
  if(x.src_ptr==y.src_ptr
    && x.size==y.size
    && (y.count==1 || x.dst_ptr==y.dst_ptr+y.count*y.dst_stride)
    && !misaligned
    && !(dst_step & 3) ) {
    if(y.count==1)
      y.dst_stride=dst_step;
    y.count++;
    return true;
  }
  else if(x.src_ptr==y.src_ptr+y.size && x.dst_ptr==y.dst_ptr+y.size && y.count==1) {
    y.size += x.size;
    return true;
  }
  return false;
}

// This copies from source memory: raw_ptrs[i] to target memory:
// device_memory_ptr at the interval of matrix_byte_size, or vice versa.
// The below algorithm tries to minimize the number of memcpy by consolidating
// neighboring memcpy into a single request
template <typename MAPPED_T>
/*port::Status*/bool ReorganizeMemory(Stream* stream,
      DeviceMemory<MAPPED_T> *device_memory,
        const std::vector<MAPPED_T*>  &raw_ptrs,
    int batch_count, uint64_t batch_stride,
    bool gather)
{
  if(gather==false)
    return false;
    //return port::Status(port::error::INTERNAL, "gather=false unsupported");
  assert(batch_count > 0);
  char *device_memory_ptr = static_cast<char *>(device_memory->opaque());
  char* src_ptr = reinterpret_cast<char*>(raw_ptrs[0]);
  char* dst_ptr = device_memory_ptr;
  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);

  std::vector<memory_copy_op> ops;
  ops.push_back(memory_copy_op{src_ptr, dst_ptr, matrix_byte_size, 1, 0});
  for (int i = 1; i < batch_count; ++i) {
    memory_copy_op& op = ops.back();
    src_ptr = reinterpret_cast<char*>(raw_ptrs[i]);
    dst_ptr = device_memory_ptr + i * matrix_byte_size;

    memory_copy_op x{src_ptr, dst_ptr, matrix_byte_size, 1, 0};
    if(fold(op, x))
      continue;
    if(ops.size()>1 && fold(ops[ops.size()-2], op))
      ops.pop_back();
    ops.push_back(x);
  }
  if(ops.size()>1 && fold(ops[ops.size()-2], ops.back()))
    ops.pop_back();
  for (auto& x : ops) {
    if(x.count>1) {
      broadcast_fp32(AsGpuStreamValue(stream),
                     reinterpret_cast<float*>(x.dst_ptr),
                     x.dst_stride >> 2, x.count,
                     reinterpret_cast<float*>(x.src_ptr),
                     x.size >> 2);
    }
    else {
      DeviceMemoryBase src_mem = DeviceMemoryBase(x.src_ptr, x.size);
      DeviceMemoryBase target_mem = DeviceMemoryBase(x.dst_ptr, x.size);
      bool a_status = stream->ThenMemcpy(&target_mem, src_mem, x.size).ok();
      if (!a_status) {
	return false;
	      /*
        return port::Status(
            port::error::INTERNAL,
          "failed to copy device memory in ROCMBlas::DoBlasGemmBatched");
	  */
      }
    }
  }
  //return port::Status::OK();
  return true;
}

template <typename T, typename U>
port::Status ROCMBlas::AllocateStridedBuffer(
    const std::vector<U*> &raw_ptrs,
    int batch_count, uint64_t batch_stride, ScratchAllocator *scratch_allocator,
    Stream *stream,
    std::unique_ptr<TemporaryDeviceMemory<U> > *temp_memory,
    DeviceMemory<U> *device_memory,
    bool copy_data, bool &reallocated) {
  assert(device_memory != nullptr);

  using MAPPED_T = U;

  bool needs_allocate_strided = false;
  for (int i = 1; i < batch_count; ++i) {
    uint64_t tmp_batch_stride = raw_ptrs[i] - raw_ptrs[i - 1];
    if (tmp_batch_stride != batch_stride) {
      needs_allocate_strided = true;
      break;
    }
  }

  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);
  size_t matrix_batch_byte_size = matrix_byte_size * batch_count;

  // No need to do re-allocation, take the short cut and return
  if (!needs_allocate_strided) {
    *device_memory = DeviceMemory<MAPPED_T>(
        DeviceMemoryBase(raw_ptrs[0], matrix_batch_byte_size));
    reallocated = false;
    return port::Status::OK();
  }

  if (scratch_allocator != nullptr) {
    SE_ASSIGN_OR_RETURN(
        DeviceMemory<uint8> batch_matrix_bytes,
        scratch_allocator->AllocateBytes(matrix_batch_byte_size));
    *device_memory = DeviceMemory<MAPPED_T>(batch_matrix_bytes);
  } else {
    assert(temp_memory != nullptr);
    SE_ASSIGN_OR_RETURN(*temp_memory, stream->AllocateTemporaryArray<MAPPED_T>(
                                          matrix_batch_byte_size));
    *device_memory =
        DeviceMemory<MAPPED_T>(*(*temp_memory)->mutable_device_memory());
  }
  assert(batch_count > 0);
  char* device_memory_ptr = static_cast<char*>(device_memory->opaque());
  char* src_ptr = reinterpret_cast<char*>(raw_ptrs[0]);
  char* dst_ptr = device_memory_ptr;
  uint64_t cur_stride_size = matrix_byte_size;

  reallocated = true;

  if (copy_data)
    if (ReorganizeMemory(stream, device_memory, raw_ptrs, batch_count,
                            batch_stride, true))
      return port::Status::OK();
    else
      return port::Status(port::error::INTERNAL, "failed BLAS call, see log for details"); 

	  /*
    return ReorganizeMemory(stream, device_memory, raw_ptrs, batch_count,
                            batch_stride, true);
			    */
  return port::Status::OK();
}

template <class T, class V>
bool ROCMBlas::DoBlasGemmBatchedImpl(Stream *stream, blas::BatchedGemmCallContext<T> ctx, V rocblas_func) {
  //const T alpha(ctx.alpha), beta(ctx.beta);
  blas::Transpose transa = ctx.transa;
  blas::Transpose transb = ctx.transb;
  uint64 m = ctx.m, n = ctx.n, k = ctx.k;
  int lda = ctx.lda, ldb = ctx.ldb, ldc = ctx.ldc, batch_count = ctx.batch_count;
  const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers = *ctx.pa;
  const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers = *ctx.pb;
  const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers = *ctx.pc;
  ScratchAllocator *scratch_allocator = ctx.scratch_allocator;
  blas::CallContext context = ctx.context;

  //using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;

  // Sanity checks before making any further progress
  uint64_t batch_stride_a = 0;
  uint64_t batch_stride_b = 0;
  uint64_t batch_stride_c = 0;

  assert(ldc >= m);
  batch_stride_c = ldc * n;

  if (ROCMBlasTranspose(transa) == rocblas_operation_none) {
    assert(lda >= m);
    batch_stride_a = lda * k;
  } else {    assert(lda >= k);
    batch_stride_a = lda * m;
  }

  if (ROCMBlasTranspose(transb) == rocblas_operation_none) {
    assert(ldb >= k);
    batch_stride_b = ldb * n;
  } else {
    assert(ldb >= n);
    batch_stride_b = ldb * k;
  }

  // Allocate local vectors to hold device pointers to matrices
  //std::vector<MAPPED_T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  std::vector<T*> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    // static_cast does work when converting Eigen::half* to rocblas_half*,
    // hence the use of reinterpret_cast
    a_raw_ptrs.push_back(reinterpret_cast<T*>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(reinterpret_cast<T*>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(reinterpret_cast<T*>(c_ptrs_to_wrappers[i]->opaque()));
  }

  DeviceMemory<T> a;
  // Make sure the temporary memory are in-scope before the function returns
  std::unique_ptr<TemporaryDeviceMemory<T>> a_temp;
  bool reallocated_a, reallocated_b, reallocated_c;
  port::Status a_allocation_status = AllocateStridedBuffer<T>(
      a_raw_ptrs, batch_count, batch_stride_a, scratch_allocator, stream,
      &a_temp, &a, true, reallocated_a);
  if (a_allocation_status != port::Status::OK()) {
    return false;
  }

  DeviceMemory<T> b;
  std::unique_ptr<TemporaryDeviceMemory<T>> b_temp;
  port::Status b_allocation_status = AllocateStridedBuffer<T>(
      b_raw_ptrs, batch_count, batch_stride_b, scratch_allocator, stream,
      &b_temp, &b, true, reallocated_b);
  if (b_allocation_status != port::Status::OK()) {
    return false;
  }

  DeviceMemory<T> c;
  std::unique_ptr<TemporaryDeviceMemory<T>> c_temp;
  port::Status c_allocation_status = AllocateStridedBuffer<T>(
      c_raw_ptrs, batch_count, batch_stride_c, scratch_allocator, stream,
      &c_temp, &c, true, reallocated_c);  // can disable copy if beta=0
  if (c_allocation_status != port::Status::OK()) {
    return false;
  }

  //MAPPED_T *alpha_ptr = reinterpret_cast<MAPPED_T *>(&alpha);
  //MAPPED_T *beta_ptr = reinterpret_cast<MAPPED_T *>(&beta);

  blas::GemmCallContext<T> newctx{ transa, transb, m, n, k,
   ctx.alpha, ctx.beta, &a, lda, &b, ldb, &c, ldc, context,
   batch_stride_a, batch_stride_b, batch_stride_c, batch_count};

  bool ok = DoBlasGemm(stream, newctx, nullptr);
  if (!ok)
    return false;
    //return port::Status(port::error::INTERNAL,
    //                    "failed BLAS call, see log for details");
 
  if (reallocated_c)
    if (ReorganizeMemory(stream, &c, c_raw_ptrs, batch_count, batch_stride_c, false))
      return true;
    else
      return false;
  /*
    return ReorganizeMemory(stream, &c, c_raw_ptrs, batch_count, batch_stride_c,
                            false);
			    */
  //return port::Status::OK();
  return true;
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCallContext<Eigen::half> ctx) {
  return DoBlasGemmBatchedImpl(stream, ctx, wrap::rocblas_hgemm_strided_batched);
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCallContext<float> ctx) {
  return DoBlasGemmBatchedImpl(stream, ctx, wrap::rocblas_sgemm_strided_batched);
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCallContext<double> ctx) {
  return DoBlasGemmBatchedImpl(stream, ctx, wrap::rocblas_dgemm_strided_batched);
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCallContext<std::complex<float> > ctx) {
  return DoBlasGemmBatchedImpl(stream, ctx, wrap::rocblas_cgemm_strided_batched);
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::BatchedGemmCallContext<std::complex<double> > ctx) {
  return DoBlasGemmBatchedImpl(stream, ctx, wrap::rocblas_zgemm_strided_batched);
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HEMM operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HEMM operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HERK operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HERK operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HER2K operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HER2K operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMM operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMM operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMM operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMM operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYRK operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYRK operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYRK operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYRK operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2K operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2K operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2K operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2K operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMM operation "
             << "for the \"float\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMM operation "
             << "for the \"double\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMM operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMM operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  blas_log("DoBlasTrsm");
  return DoBlasInternal(
      wrap::rocblas_strsm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, &alpha, const_cast<float *>(GpuMemory(a)),
      lda, GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  blas_log("DoBlasTrsm");
  return DoBlasInternal(
      wrap::rocblas_dtrsm, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, &alpha, const_cast<double *>(GpuMemory(a)),
      lda, GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSM operation "
             << "for the \"complex<float>\" datatype";
  return false;
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSM operation "
             << "for the \"complex<double>\" datatype";
  return false;
}

port::Status ROCMBlas::GetVersion(string *version) {
  return port::UnimplementedError("");
}

port::StatusOr<std::unique_ptr<blas::IBlasLtMatmulPlan>>
ROCMBlas::CreateBlasLtMatmulPlan(const blas::BlasLtMatmulPlanParams &p) {
  return port::Status(
      port::error::UNIMPLEMENTED,
      "CreateBlasLtMatmulPlan is not supported with this version of ROCM");
}

port::StatusOr<std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>>>
ROCMBlas::GetBlasLtMatmulAlgorithms(const blas::IBlasLtMatmulPlan *plan,
                                    size_t max_workspace_size,
                                    int max_algorithm_count) {
  return port::Status(
      port::error::UNIMPLEMENTED,
      "GetBlasLtMatmulAlgorithms is not supported with this version of ROCM");
}

bool ROCMBlas::DoBlasLtMatmul(
    Stream *stream, const blas::IBlasLtMatmulPlan *plan,
    const HostOrDeviceScalar<void> &alpha, DeviceMemoryBase a,
    DeviceMemoryBase b, const HostOrDeviceScalar<void> &beta,
    DeviceMemoryBase c, ScratchAllocator *scratch_allocator,
    const blas::IBlasLtMatmulAlgorithm *algorithm, DeviceMemoryBase bias,
    blas::ProfileResult *output_profile_result) {
  return false;
}

}  // namespace gpu

void initialize_rocblas() {
  auto rocBlasAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kBlas, gpu::kRocBlasPlugin);

  if (!rocBlasAlreadyRegistered) {
    port::Status status =
        PluginRegistry::Instance()
            ->RegisterFactory<PluginRegistry::BlasFactory>(
                rocm::kROCmPlatformId, gpu::kRocBlasPlugin, "rocBLAS",
                [](internal::StreamExecutorInterface *parent)
                    -> blas::BlasSupport * {
                  gpu::GpuExecutor *rocm_executor =
                      dynamic_cast<gpu::GpuExecutor *>(parent);
                  if (rocm_executor == nullptr) {
                    LOG(ERROR)
                        << "Attempting to initialize an instance of the "
                           "rocBLAS "
                        << "support library with a non-ROCM StreamExecutor";
                    return nullptr;
                  }

                  gpu::ROCMBlas *blas = new gpu::ROCMBlas(rocm_executor);
                  if (!blas->Init()) {
                    // Note: Init() will log a more specific error.
                    delete blas;
                    return nullptr;
                  }
                  return blas;
                });

    if (!status.ok()) {
      LOG(ERROR) << "Unable to register rocBLAS factory: "
                 << status.error_message();
    }

    PluginRegistry::Instance()->SetDefaultFactory(
        rocm::kROCmPlatformId, PluginKind::kBlas, gpu::kRocBlasPlugin);
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_rocblas,
                            { stream_executor::initialize_rocblas(); });
