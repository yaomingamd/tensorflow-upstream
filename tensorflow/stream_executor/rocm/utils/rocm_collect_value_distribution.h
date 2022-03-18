#pragma once

#include <cstdlib>
#include <mutex>
#include <thread>
//#include "rocm/include/hip/hip_runtime.h"
#include "hip/hip_runtime.h"

template <typename T>
class CollectValueDistribution {
 public:
  CollectValueDistribution(std::string name);

  ~CollectValueDistribution();

  uint32_t get_num_buckets();

  uint64_t get_bucket_count(int bucket_id);

  void process_data(T* data, int num_elems);

  void process_data_gpu(hipStream_t stream, void* data, int num_elems);

  void dump_data();

 private:
  std::string name_;
  uint32_t num_buckets_;
  uint64_t bucket_counts_[512];
  std::mutex bucket_counts_mutex_;
};

template <typename T>
void generate_test_data(T* test_data, int N, uint64_t* expected_bucket_counts);
