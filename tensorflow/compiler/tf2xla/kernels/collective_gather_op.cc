/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/math.h"
#include "xla/client/xla_builder.h"
#include "xla/util.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

class CollectiveGatherV2Op : public XlaOpKernel {
 public:
  explicit CollectiveGatherV2Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("communication_hint", &communication_hint_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    int64_t group_key, group_size;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("group_key", &group_key));
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntScalar("group_size", &group_size));
    OP_REQUIRES(ctx,
                communication_hint_ == "nccl" || communication_hint_ == "auto",
                errors::InvalidArgument(
                    "Only compiling NCCL/auto collective is supported, got: ",
                    communication_hint_));

    // Store all traversed collective configurations, and generate channel_id
    // for the collective.
    StatusOr<int64_t> channel_id =
        ctx->xla_context()->RecordCollectiveInfo(group_key, group_size);
    OP_REQUIRES_OK(ctx, channel_id.status());

    VLOG(2) << "Emitting xla::AllGather on channel " << *channel_id
            << " for Op " << ctx->op_kernel().name()
            << " group_size=" << group_size << " group_key=" << group_key;
    xla::ChannelHandle channel_handle;
    channel_handle.set_type(xla::ChannelHandle::DEVICE_TO_DEVICE);
    channel_handle.set_handle(*channel_id);
    std::vector<xla::ReplicaGroup> replica_groups(1);
    for (int64_t i = 0; i < group_size; i++) {
      replica_groups[0].add_replica_ids(i);
    }
    ctx->SetOutput(0, xla::AllGather(ctx->Input(0), 
        /* all_gather_dimension */ 0, 
        /* shard_count */ group_size, 
        replica_groups,
        channel_handle));
  }

 private:
  DataType dtype_ = DT_INVALID;
  string communication_hint_;

  CollectiveGatherV2Op(const CollectiveGatherV2Op&) = delete;
  void operator=(const CollectiveGatherV2Op&) = delete;
};


REGISTER_XLA_OP(Name("CollectiveGatherV2")
                    .CompileTimeConstantInput("group_key")
                    .CompileTimeConstantInput("group_size"),
                CollectiveGatherV2Op);

}  // namespace tensorflow
