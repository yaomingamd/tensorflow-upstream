#include <cstdio>
#include "rocm_collect_value_distribution.h"

#include "hip/hip_fp16.h"

#define NUM_TEST_VALUES 10000
#define BUCKET_ARRAY_SIZE 512

template <typename T>
int test_rocm_collect_value_distribution(bool test_gpu=false) {
  T test_data[NUM_TEST_VALUES];
  uint64_t expected_bucket_counts[BUCKET_ARRAY_SIZE];

  for (int i = 0; i < BUCKET_ARRAY_SIZE; i++) {
    expected_bucket_counts[i] = 0;
  }

  generate_test_data(test_data, NUM_TEST_VALUES, expected_bucket_counts);

  CollectValueDistribution<T> actual_data("cvd");
  uint32_t num_buckets = actual_data.get_num_buckets();

  if (test_gpu) {
    hipStream_t stream;
    hipStreamCreate(&stream);
    T* test_data_gpu;
    hipMalloc(&test_data_gpu, sizeof(T)*NUM_TEST_VALUES);
    hipMemcpyAsync(test_data_gpu, test_data, sizeof(T)*NUM_TEST_VALUES, hipMemcpyHostToDevice, stream);
    actual_data.process_data_gpu(stream, test_data_gpu, NUM_TEST_VALUES);
    hipFree(test_data_gpu);
    hipStreamDestroy(stream);
  }
  else {
    actual_data.process_data(test_data, NUM_TEST_VALUES);
  }

  bool found_mismatch = false;
  for (int i = 0; i < num_buckets; i++) {
    if (actual_data.get_bucket_count(i) != expected_bucket_counts[i]) {
      std::printf("mismatch for %d, %lu != %lu\n", i,
                  actual_data.get_bucket_count(i), expected_bucket_counts[i]);
      found_mismatch = true;
    }
  }

  std::printf("%s %s\n", (test_gpu ? "GPU" : "CPU"),
              (found_mismatch ? "TEST ***** FAILED *****" : "TEST  PASSED"));

  return found_mismatch;
}

int main() {
  int errors = 0;

  std::printf("Testing for fp32\n");

  errors += test_rocm_collect_value_distribution<float>();

  errors += test_rocm_collect_value_distribution<float>(true);

  std::printf("Testing for fp16\n");

  errors += test_rocm_collect_value_distribution<half>();

  errors += test_rocm_collect_value_distribution<half>(true);

  return errors;
}
