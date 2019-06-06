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

#include "tensorflow/compiler/xla/service/gpu/reduce_rewriter.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace gpu {
namespace {

class Visitor : public DfsHloVisitorWithDefault {
 public:
  explicit Visitor(HloComputation* computation) : computation_(computation) {}

  static bool Run(HloComputation* computation) {
    Visitor visitor(computation);
    TF_CHECK_OK(computation->Accept(&visitor));
    return visitor.changed_;
  }

  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleReduce(HloInstruction* reduce) override;

 private:
  bool changed_ = false;
  HloComputation* computation_;
};

Status Visitor::HandleReduce(HloInstruction* reduce) {
  const HloReduceInstruction* reduce_instr = Cast<HloReduceInstruction>(reduce);
  if (reduce_instr) {
    // only rewrite when the reduction results a scalar f32 value
    if (reduce_instr->shape().dimensions_size() == 0 &&
        reduce_instr->shape().element_type() == F32) {
      LOG(INFO) << "[ReduceRewriter] Rewrite: " << reduce_instr->ToString();
      std::vector<HloInstruction*> operands;
      operands.push_back(reduce);
      HloInstruction* new_instr =
          computation_->AddInstruction(HloInstruction::CreateCustomCall(
              reduce->shape(), operands, kReduceCallTarget));
      LOG(INFO) << "[ReduceRewriter] New instruction: "
                << new_instr->ToString();
      if (computation_->root_instruction() == reduce) {
        computation_->set_root_instruction(new_instr);
      }
      changed_ = true;
    } else {
      VLOG(3) << "[ReduceRewriter] Skip: " << reduce_instr->ToString();
    }
  }
  return Status::OK();
}

}  // anonymous namespace

StatusOr<bool> ReduceRewriter::Run(HloModule* module) {
  VLOG(2) << "ReduceRewriter::Run(), before:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (Visitor::Run(comp)) {
      changed = true;
    }
  }

  VLOG(2) << "ReduceRewriter::Run(), after:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
