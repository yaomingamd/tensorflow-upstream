/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_TEST_HELPERS_H_
#define XLA_TEST_HELPERS_H_

#include <list>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/regexp.h"
#include "tsl/platform/test.h"

// This module contains a minimal subset of gmock functionality just
// sufficient to execute the currently existing tests.

namespace xla {
template <typename T>
class Array2D;
class Literal;

namespace testing {

namespace internal_status {
inline const Status& GetStatus(const Status& status) { return status; }

template <typename T>
inline const Status& GetStatus(const StatusOr<T>& status) {
  return status.status();
}
}  // namespace internal_status

}  // namespace testing
}  // namespace xla

// The following macros are similar to macros in gmock, but deliberately named
// differently in order to avoid conflicts in files which include both.

// Macros for testing the results of functions that return Status or
// StatusOr<T> (for any type T).
#define EXPECT_IS_OK(expression) \
  EXPECT_EQ(::tsl::OkStatus(),   \
            xla::testing::internal_status::GetStatus(expression))
#define EXPECT_IS_NOT_OK(expression) \
  EXPECT_NE(::tsl::OkStatus(),       \
            xla::testing::internal_status::GetStatus(expression))
#undef ASSERT_IS_OK
#define ASSERT_IS_OK(expression) \
  ASSERT_EQ(::tsl::OkStatus(),   \
            xla::testing::internal_status::GetStatus(expression))
#undef ASSERT_IS_NOT_OK
#define ASSERT_IS_NOT_OK(expression) \
  ASSERT_NE(::tsl::OkStatus(),       \
            xla::testing::internal_status::GetStatus(expression))

#endif  // XLA_TEST_HELPERS_H_
