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

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_BUILD__JOBS} concurrent test job(s)."
echo ""

# First positional argument (if any) specifies the ROCM_INSTALL_DIR
ROCM_INSTALL_DIR=/opt/rocm-5.1.0
if [[ -n $1 ]]; then
    ROCM_INSTALL_DIR=$1
fi

# Run configure.
export PYTHON_BIN_PATH=`which python3`

# Force CPU
export TF_NEED_ROCM=0

yes "" | $PYTHON_BIN_PATH configure.py

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test \
      -k \
      --test_tag_filters=-no_oss,-oss_serial,-gpu,-tpu,-benchmark-test,-v1only,-no_rocm \
      --test_lang_filters=py,cc \
      --jobs=${N_BUILD_JOBS} \
      --local_test_jobs=${N_BUILD_JOBS} \
      --test_timeout 600,900,2400,7200 \
      --build_tests_only \
      --test_output=errors \
      --test_sharding_strategy=disabled \
      --test_size_filters=small,medium,large \
      -- \
      //tensorflow/... \
      -//tensorflow/python/integration_testing/... \
      -//tensorflow/core/tpu/... \
      -//tensorflow/lite/... \
      -//tensorflow/core/platform:ram_file_system_test \
      -//tensorflow/python/data/experimental/kernel_tests/serialization:checkpoint_input_pipeline_hook_test \
      -//tensorflow/python/distribute:parameter_server_strategy_test \
      -//tensorflow/python/keras/distribute:custom_training_loop_models_test \
      -//tensorflow/python/keras/engine:deferred_sequential_test \
      -//tensorflow/python/keras/layers:gru_v2_test \
      -//tensorflow/python/keras/mixed_precision:keras_test \
      -//tensorflow/python/keras/saving:hdf5_format_test \
      -//tensorflow/python/keras/saving:losses_serialization_test \
      -//tensorflow/python/keras/saving:metrics_serialization_test \
      -//tensorflow/python/keras/saving:save_test \
      -//tensorflow/python/keras/saving:saved_model_test \
      -//tensorflow/python/keras/tests:model_architectures_test \
      -//tensorflow/python/keras/tests:model_subclassing_compiled_test \
      -//tensorflow/python/keras/tests:model_subclassing_test 
 

