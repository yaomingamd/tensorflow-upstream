/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------*/

#ifndef XLA_STREAM_EXECUTOR_NUMERIC_OPTIONS_H_
#define XLA_STREAM_EXECUTOR_NUMERIC_OPTIONS_H_

namespace stream_executor {

// Options that specify the numeric behavior of operations like matrix
// multiplications and convolutions
struct NumericOptions {
  NumericOptions(bool require_determinism, bool allow_tf32, int grad)
      : require_determinism(require_determinism), allow_tf32(allow_tf32), grad_flags(grad | GF_Initialized) {}

  NumericOptions() : require_determinism(false), allow_tf32(true), grad_flags(0) {}

  // If true, the op must be deterministic
  bool require_determinism;
  // If true, float32 inputs can be rounded to TensorFloat-32 precision
  bool allow_tf32;

  enum GradientFlags {
    GF_Gradient1 = 1<<0,
    GF_Gradient2 = 1<<1,
    GF_Initialized = 1<<8
  };

  // bit 0: interpret input 1 as gradient
  // bit 1: interpret input 2 as gradient
  int grad_flags;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_NUMERIC_OPTIONS_H_
