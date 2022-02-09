#include <cstdio>
#include "rocm_collect_value_distribution.h"

#include "hip/hip_fp16.h"

#define NUM_TEST_VALUES 10000
#define BUCKET_ARRAY_SIZE 512

int test_fp32() {
  float test_data[NUM_TEST_VALUES];
  int expected_bucket_counts[BUCKET_ARRAY_SIZE];

  for (int i = 0; i < BUCKET_ARRAY_SIZE; i++) {
    expected_bucket_counts[i] = 0;
  }

  generate_test_data(test_data, NUM_TEST_VALUES, expected_bucket_counts);

  CollectValueDistribution<float> actual_data("cvd");
  uint32_t num_buckets = actual_data.get_num_buckets();

  actual_data.process_data(test_data, NUM_TEST_VALUES);

  bool found_mismatch = false;
  for (int i = 0; i < num_buckets; i++) {
    if (actual_data.get_bucket_count(i) != expected_bucket_counts[i]) {
      std::printf("mismatch for %d, %d != %d\n", i,
                  actual_data.get_bucket_count(i), expected_bucket_counts[i]);
      found_mismatch = true;
    }
  }

  std::printf("%s\n",
              (found_mismatch ? "TEST ***** FAILED *****" : "TEST  PASSED"));

  return found_mismatch;
}

int test_fp16() {
  half test_data[NUM_TEST_VALUES];
  int expected_bucket_counts[BUCKET_ARRAY_SIZE];

  for (int i = 0; i < BUCKET_ARRAY_SIZE; i++) {
    expected_bucket_counts[i] = 0;
  }
  generate_test_data(test_data, NUM_TEST_VALUES, expected_bucket_counts);

  CollectValueDistribution<half> actual_data("cvd");
  uint32_t num_buckets = actual_data.get_num_buckets();

  actual_data.process_data(test_data, NUM_TEST_VALUES);

  bool found_mismatch = false;
  for (int i = 0; i < num_buckets; i++) {
    if (actual_data.get_bucket_count(i) != expected_bucket_counts[i]) {
      std::printf("mismatch for %d, %d != %d\n", i,
                  actual_data.get_bucket_count(i), expected_bucket_counts[i]);
      found_mismatch = true;
    }
  }

  std::printf("%s\n",
              (found_mismatch ? "TEST ***** FAILED *****" : "TEST  PASSED"));

  return found_mismatch;
}

int main() {
  int errors = 0;

  errors += test_fp32();

  errors += test_fp16();

  return errors;
}
