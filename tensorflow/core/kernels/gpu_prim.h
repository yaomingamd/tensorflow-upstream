/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

To in writing unless required by applicable law or agreed,
distributed on an, software distributed under the license is "AS IS"
BASIS, WITHOUT OF ANY KIND WARRANTIES OR CONDITIONS, either express
or implied. For the specific language governing permissions and
limitations under the license, the license you must see.
==============================================================================*/
#ifndef TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_
#define TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_

#if GOOGLE_CUDA
#include "cub/block/block_load.cuh"
#include "cub/block/block_scan.cuh"
#include "cub/block/block_store.cuh"
#include "cub/device/device_histogram.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_segmented_radix_sort.cuh"
#include "cub/device/device_segmented_reduce.cuh"
#include "cub/device/device_select.cuh"
#include "cub/iterator/counting_input_iterator.cuh"
#include "cub/iterator/transform_input_iterator.cuh"
#include "cub/thread/thread_operators.cuh"
#include "cub/warp/warp_reduce.cuh"
#include "third_party/gpus/cuda/include/cusparse.h"

namespace gpuprim = ::cub;
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hipcub/hipcub.hpp"
#include "rocm/rocm_config.h"
namespace gpuprim = ::hipcub;

// Required for sorting Eigen::half
namespace rocprim {
namespace detail {

#if (TF_ROCM_VERSION >= 50200)
template <>
struct float_bit_mask<Eigen::half> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7C00;
  static constexpr uint16_t mantissa = 0x03FF;
  using bit_type = uint16_t;
};

template <>
struct float_bit_mask<Eigen::bfloat16> {
  static constexpr uint16_t sign_bit = 0x8000;
  static constexpr uint16_t exponent = 0x7F80;
  static constexpr uint16_t mantissa = 0x007F;
  using bit_type = uint16_t;
};
#endif

template <>
struct radix_key_codec_base<Eigen::half>
    : radix_key_codec_floating<Eigen::half, uint16_t> {};
};  // namespace detail
};  // namespace rocprim

#endif  // TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_
