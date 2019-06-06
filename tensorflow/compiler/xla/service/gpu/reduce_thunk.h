/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCE_THUNK_H_

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

class ReduceThunk : public Thunk {
 public:
  ReduceThunk(const BufferAllocation::Slice& reduce_output_tensor,
              const BufferAllocation::Slice& output,
              const HloInstruction* custom_call_hlo, const HloInstruction* hlo,
              const BufferAllocation::Slice& reduce_input_tensor,
              int64 reduce_dimension, float init_value);
  ReduceThunk(const ReduceThunk&) = delete;
  ReduceThunk& operator=(const ReduceThunk&) = delete;

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const BufferAllocation::Slice reduce_input_;
  const BufferAllocation::Slice reduce_output_;
  const BufferAllocation::Slice output_;
  const HloInstruction* hlo_;
  int64 reduce_dimension_;
  float init_value_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDUCE_THUNK_H_
