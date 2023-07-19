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
==============================================================================*/

#include "tensorflow/dtensor/mlir/expansions/where_spmd_expander.h"

#include <cstddef>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

// Convert the local index (associated with local tensor per device) to the
// global index (associated with the global tensor). The local index must be the
// first result of `op`.
StatusOr<mlir::Operation*> LocalIndexToGlobalIndex(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractLayoutFromOperand(op->getOperand(0)));

  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfter(op);

  // Calculate the index offset using DeviceId, for now, DTensor only supports
  // index conversion when sharding is on the first dimension.
  mlir::Value num_devices_per_dim_0 =
      IntConst(builder, op->getLoc(),
               input_layout->mesh().num_local_devices() /
                   input_layout->num_shards_for_dim(0));
  TF_ASSIGN_OR_RETURN(mlir::Value device_id, DeviceId(op));
  mlir::Value device_id_offset = builder.create<mlir::TF::DivOp>(
      op->getLoc(), device_id, num_devices_per_dim_0);

  TF_ASSIGN_OR_RETURN(const auto& shape,
                      ExtractGlobalInputShape(op->getOpOperand(0)));
  mlir::Value size_per_shard_dim_0 = IntConst(
      builder, op->getLoc(), shape[0] / input_layout->num_shards_for_dim(0));
  mlir::Value index_offset = builder.create<mlir::TF::MulOp>(
      op->getLoc(), size_per_shard_dim_0, device_id_offset);

  // Add index offset to the local index to get the global index.
  mlir::Value index_offset_i64 = builder.create<mlir::TF::CastOp>(
      op->getLoc(),
      mlir::RankedTensorType::get(
          index_offset.getType().cast<mlir::TensorType>().getShape(),
          builder.getIntegerType(64)),
      index_offset);
  mlir::Value global_index = builder.create<mlir::TF::AddV2Op>(
      op->getLoc(), op->getResultTypes(), index_offset_i64, op->getOpResult(0));

  op->getOpResult(0).replaceAllUsesExcept(global_index,
                                          global_index.getDefiningOp());

  return global_index.getDefiningOp();
}

}  // namespace

StatusOr<mlir::Operation*> WhereOpSPMDExpander::ExpandOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractLayoutFromOperand(op->getOperand(0)));
  assert(input_layout);

  // If input is fully replicated, there is no need to manipulate the index
  // calculated by the Where Op, just return directly.
  if (input_layout->IsFullyReplicated()) {
    return op;
  }

  // Only supports sharding on the first dimension.
  if (!input_layout->IsBatchParallel()) {
    return absl::InvalidArgumentError(
        "Where op only supports batch sharding for now.");
  }

  // Where Op returns the indices of the non-zero elements in the input tensor.
  // Convert the local index to global index as the final output.
  return LocalIndexToGlobalIndex(op);
}

StatusOr<llvm::DenseMap<int, Layout>> WhereOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (input_layouts.find(0) == input_layouts.end()) {
    return llvm::DenseMap<int, Layout>();
  }
  // Currently Where op only supports 1D input.
  Layout layout = input_layouts.lookup(0);
  if (layout.rank() != 1) {
    return llvm::DenseMap<int, Layout>();
  }

  // Append an unsharded sharding spec for the index dimension generated by the
  // Where op.
  std::vector<std::string> layout_specs;
  layout_specs.push_back(layout.sharding_spec(0));
  layout_specs.push_back(Layout::kUnshardedDim);
  TF_ASSIGN_OR_RETURN(Layout new_layout,
                      Layout::GetLayout(layout_specs, layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, new_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
WhereOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end()) {
    return llvm::DenseMap<int, Layout>();
  }

  // Remove the unsharded sharding spec generated by the Where op.
  Layout layout = output_layouts.lookup(0);
  std::vector<std::string> layout_specs;
  layout_specs.reserve(layout.rank() - 1);
  for (int i = 0; i < layout.rank() - 1; i++) {
    layout_specs.push_back(layout.sharding_spec(i));
  }
  TF_ASSIGN_OR_RETURN(Layout new_layout,
                      Layout::GetLayout(layout_specs, layout.mesh()));
  return llvm::DenseMap<int, Layout>({{0, new_layout}});
}

}  // namespace dtensor
}  // namespace tensorflow
