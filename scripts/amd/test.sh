#!/usr/bin/env bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
set -e
set -x

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
N_TEST_JOBS=1 # run tests serially

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s)."
echo ""

# First positional argument (if any) specifies the ROCM_INSTALL_DIR
ROCM_INSTALL_DIR=/opt/rocm-4.5.2
if [[ -n $1 ]]; then
    ROCM_INSTALL_DIR=$1
fi

# Run configure.
export PYTHON_BIN_PATH=$(which python3)

export TF_NEED_ROCM=1
export ROCM_PATH=$ROCM_INSTALL_DIR

yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test \
    --config=rocm \
    -k \
    --test_tag_filters=-no_gpu,-no_rocm \
    --jobs=${N_BUILD_JOBS} \
    --local_test_jobs=${N_TEST_JOBS} \
    --test_timeout 600,900,2400,7200 \
    --build_tests_only \
    --test_output=errors \
    --test_sharding_strategy=disabled \
    --test_size_filters=small,medium,large \
    --cache_test_results=no \
    --test_env=TF_PER_DEVICE_MEMORY_LIMIT_MB=2048 \
    -- \
    //tensorflow/python/distribute:collective_all_reduce_strategy_test_xla_2gpu \
    //tensorflow/python/keras/utils:multi_gpu_utils_test_xla_2gpu


//tensorflow/core/nccl:nccl_manager_test_2gpu 
//tensorflow/python/distribute:collective_all_reduce_strategy_test_xla_2gpu \
//tensorflow/python/keras/utils:multi_gpu_utils_test_xla_2gpu \
# //tensorflow/core/kernels:collective_nccl_test_2gpu 
