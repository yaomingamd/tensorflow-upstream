#include <cstdlib>
#include <tuple>

#include "rocm/include/hip/hip_fp16.h"
//#include "rocm/include/hip/hip_runtime.h"
#include "hip/hip_runtime.h"

#include "tensorflow/stream_executor/rocm/utils/rocm_collect_value_distribution.h"

template <typename T>
struct type_info;

// FP32 == float
//
// 1-sign, 8-exponent, 23-mantissa
// number of buckets =
//  +    1  zero                                    0
//      23  "denormal buckets", 2^-149 - 2^-127 ,   1 - 23
//  +  254  "normal buckets",   2^-126 - 2^127  ,  24 - 277
//  +    2  inf, nan,,                             278 = 279
//     ---
//     280

template <>
struct type_info<float> {
  static const uint32_t num_sign_bits = 1;
  static const uint32_t num_exponent_bits = 8;
  static const uint32_t num_mantissa_bits = 23;

  static const uint32_t sign_mask = 0x1;
  static const uint32_t exponent_mask = 0xff;
  static const uint32_t mantissa_mask = 0x7fffff;

  static const int32_t exponent_bias = 127;
  static const int32_t min_positive_denormal_exponent_value = -149;

  static const uint32_t num_denormal_buckets = 23;
  static const uint32_t num_exponent_value_buckets = 280;

  typedef uint32_t bit_pattern_type;
};

// FP16 == half
//
// 1-sign, 5-exponent, 10-mantissa
// number of buckets =
//  +    1  zero                                  0
//      10  "denormal buckets", 2^-24 - 2^-15 ,   1 - 10
//  +   30  "normal buckets",   2^-14 - 2^15  ,  11 - 40
//  +    2  inf, nan,                            41 - 42
//     ---
//      43

template <>
struct type_info<half> {
  static const uint32_t num_sign_bits = 1;
  static const uint32_t num_exponent_bits = 5;
  static const uint32_t num_mantissa_bits = 10;

  static const uint32_t sign_mask = 0x1;
  static const uint32_t exponent_mask = 0x1f;
  static const uint32_t mantissa_mask = 0x3ff;

  static const int32_t exponent_bias = 15;
  static const int32_t min_positive_denormal_exponent_value = -24;

  static const uint32_t num_denormal_buckets = 10;
  static const uint32_t num_exponent_value_buckets = 43;

  typedef uint16_t bit_pattern_type;
};

template <typename T>
struct type_utils {
  using bit_pattern_type = typename type_info<T>::bit_pattern_type;

  static T get_value(uint32_t sign, uint32_t exponent, uint32_t mantissa) {
    bit_pattern_type bit_pattern = 0;
    bit_pattern |=
        (sign & type_info<T>::sign_mask)
        << (type_info<T>::num_exponent_bits + type_info<T>::num_mantissa_bits);
    bit_pattern |= (exponent & type_info<T>::exponent_mask)
                   << type_info<T>::num_mantissa_bits;
    bit_pattern |= (mantissa & type_info<T>::mantissa_mask);
    return *((T*)&bit_pattern);
  }

  __host__ __device__  __forceinline__
  static std::tuple<uint32_t, uint32_t, uint32_t> get_sign_exponent_mantissa(
      T value) {
    bit_pattern_type bit_pattern = *((bit_pattern_type*)&value);
    uint32_t sign = (bit_pattern >> (type_info<T>::num_exponent_bits +
                                     type_info<T>::num_mantissa_bits)) &
                    type_info<T>::sign_mask;
    uint32_t exponent = (bit_pattern >> type_info<T>::num_mantissa_bits) &
                        type_info<T>::exponent_mask;
    uint32_t mantissa = bit_pattern & type_info<T>::mantissa_mask;
    return std::tie(sign, exponent, mantissa);
  }

  __host__ __device__  __forceinline__
  static uint32_t get_bucket_id(uint32_t sign, uint32_t exponent,
                                uint32_t mantissa) {
    uint32_t bucket_id = 0;
    exponent = exponent & type_info<T>::exponent_mask;
    mantissa = mantissa & type_info<T>::mantissa_mask;
    if (exponent == 0) {
      if (mantissa == 0) {  // zero
        bucket_id = 0;
      } else {  // denormals
        while (mantissa > 0) {
          mantissa = mantissa >> 1;
          bucket_id++;
        }
      }
    } else if (exponent == type_info<T>::exponent_mask) {
      if (mantissa == 0) {  // infinity
        bucket_id = type_info<T>::num_exponent_value_buckets - 2;
      } else {  // NaN
        bucket_id = type_info<T>::num_exponent_value_buckets - 1;
      }
    } else {  // normal values
      bucket_id = exponent + type_info<T>::num_denormal_buckets;
    }
    return bucket_id;
  }

  __host__ __device__ 
  static std::tuple<uint32_t, uint32_t, uint32_t> get_sign_exponent_mantissa(
      uint32_t bucket_id) {
    uint32_t sign = std::rand() % 2;
    uint32_t exponent, mantissa;

    if (bucket_id >= type_info<T>::num_exponent_value_buckets) {  // invalid
      exponent = 0;
      mantissa = 0;
    } else if (bucket_id == 0) {  // zero
      exponent = 0;
      mantissa = 0;
    } else if (bucket_id ==
               (type_info<T>::num_exponent_value_buckets - 2)) {  // inf
      exponent = type_info<T>::exponent_mask;
      mantissa = 0;
    } else if (bucket_id ==
               (type_info<T>::num_exponent_value_buckets - 1)) {  // nan
      exponent = type_info<T>::exponent_mask;
      mantissa = std::rand() | 0x1;
    } else if (bucket_id <= type_info<T>::num_denormal_buckets) {  // denormal
      exponent = 0;
      mantissa = 0x1 << (bucket_id - 1);
      mantissa |= std::rand() & (mantissa - 1);
    } else {  // normal
      exponent = bucket_id - type_info<T>::num_denormal_buckets;
      mantissa = std::rand() & type_info<T>::mantissa_mask;
    }

    return std::tie(sign, exponent, mantissa);
  }
};

template <typename T>
CollectValueDistribution<T>::CollectValueDistribution(std::string name) {
  name_ = name;
  num_buckets_ = type_info<T>::num_exponent_value_buckets;
  for (int i = 0; i < num_buckets_; i++) {
    bucket_counts_[i] = 0;
  }
}

template <typename T>
CollectValueDistribution<T>::~CollectValueDistribution() {
  dump_data();
}

template <typename T>
uint32_t CollectValueDistribution<T>::get_num_buckets() {
  return num_buckets_;
}

template <typename T>
uint64_t CollectValueDistribution<T>::get_bucket_count(int bucket_id) {
  uint64_t count = 0;
  if (bucket_id < num_buckets_) {
    count = bucket_counts_[bucket_id];
  }
  return count;
}

template <typename T>
void CollectValueDistribution<T>::process_data(T* data, int num_elems) {
  bucket_counts_mutex_.lock();

  for (int i = 0; i < num_elems; i++) {
    uint32_t sign, exponent, mantissa;
    std::tie(sign, exponent, mantissa) =
        type_utils<T>::get_sign_exponent_mantissa(data[i]);

    uint32_t bucket_id = type_utils<T>::get_bucket_id(sign, exponent, mantissa);
    // std::printf("bucket_id = %03d (%d, %d)\n", bucket_id, exponent,
    // mantissa);

    bucket_counts_[bucket_id]++;
  }

  bucket_counts_mutex_.unlock();
}

template <typename T>
__global__ void exponent_bucket_count_gpu_kernel(T* data, int num_elems, uint32_t* result) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num_elems) {
    uint32_t sign, exponent, mantissa;
    auto tmp = type_utils<T>::get_sign_exponent_mantissa(data[idx]);
    sign = std::get<0>(tmp);
    exponent = std::get<1>(tmp);
    mantissa = std::get<2>(tmp);
    uint32_t bucket_id = type_utils<T>::get_bucket_id(sign, exponent, mantissa);
    atomicAdd(&result[bucket_id], 1);
  }
}

template <typename T>
void CollectValueDistribution<T>::process_data_gpu(hipStream_t stream, void* data, int num_elems) {

  uint32_t* result;
  size_t result_size = sizeof(uint32_t)*get_num_buckets();
  hipMalloc(&result, result_size);
  hipMemsetAsync(result, 0, result_size, stream);

  T* data_T = reinterpret_cast<T*>(data);

  int threads_per_block = 256;
  int num_blocks = (num_elems + threads_per_block -1 ) / threads_per_block;
  hipLaunchKernelGGL(exponent_bucket_count_gpu_kernel, dim3(num_blocks,1,1), dim3(threads_per_block,1,1), 0, stream, data_T, num_elems, result); 
  hipMemcpyAsync(result_host_, result, result_size, hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
  hipFree(result);

  bucket_counts_mutex_.lock();
  for (int i=0;i<get_num_buckets();i++) {
    bucket_counts_[i] += result_host_[i];
  }
  bucket_counts_mutex_.unlock();
}

template <typename T>
void CollectValueDistribution<T>::dump_data() {
  bucket_counts_mutex_.lock();
  std::printf("%s_bucket_counts = [", name_.c_str());
  for (int i = 0; i < num_buckets_; i++) {
    std::printf("%lu,", bucket_counts_[i]);
  }
  std::printf("]\n");
  bucket_counts_mutex_.unlock();
}

template <typename T>
void generate_test_data(T* test_data, int N, uint64_t* bucket_counts) {
  for (int i = 0; i < N; i++) {
    uint32_t sign = 0;
    uint32_t exponent = 0;
    uint32_t mantissa = 0;

    uint32_t bucket_id = std::rand() % type_info<T>::num_exponent_value_buckets;

    std::tie(sign, exponent, mantissa) =
        type_utils<T>::get_sign_exponent_mantissa(bucket_id);

    bucket_counts[bucket_id]++;

    T value = type_utils<T>::get_value(sign, exponent, mantissa);

    // int32_t exponent_value = bucket_id - type_info<T>::num_denormal_buckets -
    //                          type_info<T>::exponent_bias;
    // std::printf("bucket_id = %03d (%03d), value = %a\n", bucket_id,
    //             exponent_value, (float)value);

    test_data[i] = value;
  }
}

template class CollectValueDistribution<float>;

template class CollectValueDistribution<half>;

template void generate_test_data<float>(float* test_data, int N,
                                        uint64_t* expected_bucket_counts);

template void generate_test_data<half>(half* test_data, int N,
                                       uint64_t* expected_bucket_counts);
