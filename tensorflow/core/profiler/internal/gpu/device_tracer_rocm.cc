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

#if TENSORFLOW_USE_ROCM

#include <stdlib.h>
#include <memory>
#include <iostream>

#include "absl/container/fixed_array.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/annotation.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/internal/gpu/rocm_tracer.h"
#include "tensorflow/core/profiler/internal/parse_annotation.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/util/env_var.h"

#include "roctracer/roctracer.h"
#include "hip/hip_runtime.h"

namespace tensorflow {
namespace profiler {

// Adapter from RocmTraceCollector to StepStatsCollector: This class convert
// and filter from RocmTracerEvent to tensorflow::NodeExecStats.
// We can not just forward event on the fly because StepStatsCollector have
// a single mutex for all devices, Therefore we will cache events and forward
// only when Flush().
class StepStatsRocmTracerAdaptor : public RocmTraceCollector {
 public:
  StepStatsRocmTracerAdaptor(const RocmTracerCollectorOptions& option,
                              const std::string prefix, int num_gpus,
                              uint64 start_walltime_ns, uint64 start_gpu_ns,
                              StepStatsCollector* trace_collector)
      : RocmTraceCollector(option),
        trace_collector_(trace_collector),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gpu_ns_(start_gpu_ns),
        num_gpus_(num_gpus),
        per_device_adaptor_(num_gpus) {
    for (int i = 0; i < num_gpus; ++i) {  // for each device id.
      per_device_adaptor_[i].stream_device =
          strings::StrCat(prefix, "/device:GPU:", i, "/stream:");
      per_device_adaptor_[i].memcpy_device =
          strings::StrCat(prefix, "/device:GPU:", i, "/memcpy");
      per_device_adaptor_[i].sync_device =
          strings::StrCat(prefix, "/device:GPU:", i, "/sync");
    }
  }

  void AddEvent(RocmTracerEvent&& event) override {
    if (event.device_id >= num_gpus_) return;
    if (event.source == RocmTracerEventSource::ApiCallback) {
      if (num_callback_events_ > options_.max_callback_api_events) {
        OnEventsDropped("trace collector", 1);
        return;
      }
      num_callback_events_++;
    } else {
      if (num_activity_events_ > options_.max_activity_api_events) {
        OnEventsDropped("trace collector", 1);
        return;
      }
      num_activity_events_++;
    }
    per_device_adaptor_[event.device_id].AddEvent(std::move(event));
  }
  void OnEventsDropped(const std::string& reason, uint32 num_events) override {}
  void Flush() override {
    LOG(INFO) << " GpuTracer has collected " << num_callback_events_
              << " callback api events and " << num_activity_events_
              << " activity events.";
    for (int i = 0; i < num_gpus_; ++i) {
      per_device_adaptor_[i].Flush(trace_collector_, start_walltime_ns_,
                                   start_gpu_ns_);
    }
  }

 private:
  StepStatsCollector* trace_collector_;
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64 start_walltime_ns_;
  uint64 start_gpu_ns_;
  int num_gpus_;

  struct CorrelationInfo {
    CorrelationInfo(uint32 t, uint32 e) : thread_id(t), enqueue_time_ns(e) {}
    uint32 thread_id;
    uint64 enqueue_time_ns;
  };
  struct PerDeviceAdaptor {
    void AddEvent(RocmTracerEvent&& event) {
      absl::MutexLock lock(&mutex);
      if (event.source == RocmTracerEventSource::ApiCallback) {
        // Rocm api callcack events were used to populate launch times etc.
        if (event.name == "hipStreamSynchronize") {
          events.emplace_back(std::move(event));
        }
        if (event.correlation_id != RocmTracerEvent::kInvalidCorrelationId) {
          correlation_info.insert(
              {event.correlation_id,
               CorrelationInfo(event.thread_id, event.start_time_ns)});
        }
      } else {
        // Rocm activity events measure device times etc.
        events.emplace_back(std::move(event));
      }
    }
    void Flush(StepStatsCollector* collector, uint64 start_walltime_ns,
               uint64 start_gpu_ns) {
      absl::MutexLock lock(&mutex);
      for (auto& event : events) {
        NodeExecStats* ns = new NodeExecStats;
        ns->set_all_start_micros(
            (start_walltime_ns + (event.start_time_ns - start_gpu_ns)) / 1000);
        ns->set_op_start_rel_micros(0);
        auto elapsed_ns = event.end_time_ns - event.start_time_ns;
        ns->set_op_end_rel_micros(elapsed_ns / 1000);
        ns->set_all_end_rel_micros(elapsed_ns / 1000);

        if (event.source == RocmTracerEventSource::ApiCallback) {
          DCHECK_EQ(event.name, "hipStreamSynchronize");
          ns->set_node_name(event.name);
          ns->set_timeline_label(absl::StrCat("ThreadId ", event.thread_id));
          ns->set_thread_id(event.thread_id);
          collector->Save(sync_device, ns);
        } else {  // RocmTracerEventSource::Activity
          // Get launch information if available.
          if (event.correlation_id != RocmTracerEvent::kInvalidCorrelationId) {
            auto it = correlation_info.find(event.correlation_id);
            if (it != correlation_info.end()) {
              ns->set_scheduled_micros(it->second.enqueue_time_ns / 1000);
              ns->set_thread_id(it->second.thread_id);
            }
          }

          auto annotation_stack = ParseAnnotationStack(event.annotation);
          std::string kernel_name = port::MaybeAbiDemangle(event.name.c_str());
          std::string activity_name =
              !annotation_stack.empty()
                  ? std::string(annotation_stack.back().name)
                  : kernel_name;
          ns->set_node_name(activity_name);
          switch (event.type) {
            case RocmTracerEventType::Kernel: {
              const std::string details = strings::Printf(
                  "grid:%llu,%llu,%llu block:%llu,%llu,%llu",
                  event.kernel_info.grid_x, event.kernel_info.grid_y,
                  event.kernel_info.grid_z, event.kernel_info.block_x,
                  event.kernel_info.block_y, event.kernel_info.block_z);
              ns->set_timeline_label(absl::StrCat(kernel_name, " ", details,
                                                  "@@", event.annotation));
              auto nscopy = new NodeExecStats(*ns);
              collector->Save(absl::StrCat(stream_device, "all"), ns);
              collector->Save(absl::StrCat(stream_device, event.stream_id),
                              nscopy);
              break;
            }
            case RocmTracerEventType::MemcpyH2D:
            case RocmTracerEventType::MemcpyD2H:
            case RocmTracerEventType::MemcpyD2D:
            case RocmTracerEventType::MemcpyP2P: {
              std::string details = absl::StrCat(
                  activity_name, " bytes:", event.memcpy_info.num_bytes);
              if (event.memcpy_info.async) {
                absl::StrAppend(&details, " aync");
              }
              if (event.memcpy_info.destination != event.device_id) {
                absl::StrAppend(&details,
                                " to device:", event.memcpy_info.destination);
              }
              ns->set_timeline_label(std::move(details));
              auto nscopy = new NodeExecStats(*ns);
              collector->Save(memcpy_device, ns);
              collector->Save(
                  absl::StrCat(stream_device, event.stream_id, "<",
                               GetTraceEventTypeName(event.type), ">"),
                  nscopy);
              break;
            }
            default:
              ns->set_timeline_label(activity_name);
              collector->Save(stream_device, ns);
          }
        }
      }
    }

    absl::Mutex mutex;
    std::string stream_device GUARDED_BY(mutex);
    std::string memcpy_device GUARDED_BY(mutex);
    std::string sync_device GUARDED_BY(mutex);
    std::vector<RocmTracerEvent> events GUARDED_BY(mutex);
    absl::flat_hash_map<uint32, CorrelationInfo> correlation_info
        GUARDED_BY(mutex);
  };
  absl::FixedArray<PerDeviceAdaptor> per_device_adaptor_;

  TF_DISALLOW_COPY_AND_ASSIGN(StepStatsRocmTracerAdaptor);
};

// RocmGpuTracer for ROCm GPU.
class RocmGpuTracer : public profiler::ProfilerInterface {
 public:
  RocmGpuTracer(RocmTracer* rocm_tracer)
      : rocm_tracer_(rocm_tracer),
        trace_collector_(&step_stats_) {}
  ~RocmGpuTracer() override {}

  // RocmGpuTracer interface:
  Status Start() override;
  Status Stop() override;
  Status CollectData(RunMetadata* run_metadata) override;
  profiler::DeviceType GetDeviceType() override {
    return profiler::DeviceType::kGpu;
  }

 private:
  Status DoStart();
  Status DoStop();

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  RocmTracer* rocm_tracer_;
  RocmTracerOptions options_;
  StepStats step_stats_;
  StepStatsCollector trace_collector_;
  std::unique_ptr<StepStatsRocmTracerAdaptor> step_stats_rocm_adaptor_;
};

Status RocmGpuTracer::DoStart() {
  if (!rocm_tracer_->IsAvailable()) {
    return errors::Unavailable("Another profile session running.");
  }

  options_.cbids_selected = {
      // KERNEL
      HIP_API_ID_hipModuleLaunchKernel,
      // MEMCPY
      HIP_API_ID_hipMemcpyDtoH,
      HIP_API_ID_hipMemcpyDtoHAsync,
      HIP_API_ID_hipMemcpyHtoD,
      HIP_API_ID_hipMemcpyHtoDAsync,
      HIP_API_ID_hipMemcpyDtoD,
      HIP_API_ID_hipMemcpyDtoDAsync,
      // GENERIC
      HIP_API_ID_hipStreamSynchronize,
  };

  options_.activities_selected.push_back(ACTIVITY_DOMAIN_HIP_API);
  options_.activities_selected.push_back(ACTIVITY_DOMAIN_HCC_OPS);

  RocmTracerCollectorOptions collector_options;
  uint64 start_gputime_ns = RocmTracer::GetTimestamp();
  uint64 start_walltime_ns = tensorflow::EnvTime::Default()->NowNanos();
  int num_gpus = rocm_tracer_->NumGpus();
  step_stats_rocm_adaptor_ = absl::make_unique<StepStatsRocmTracerAdaptor>(
      collector_options, "", num_gpus, start_walltime_ns, start_gputime_ns,
      &trace_collector_);

  tensorflow::tracing::ScopedAnnotation::Enable(true);
  rocm_tracer_->Enable(options_, step_stats_rocm_adaptor_.get());
  return Status::OK();
}

Status RocmGpuTracer::Start() {
  Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return Status::OK();
  } else {
    profiling_state_ = State::kStartedError;
    return status;
  }
}

Status RocmGpuTracer::DoStop() {
  rocm_tracer_->Disable();
  tensorflow::tracing::ScopedAnnotation::Enable(false);
  return Status::OK();
}

Status RocmGpuTracer::Stop() {
  if (profiling_state_ == State::kStartedOk) {
    Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return Status::OK();
}

Status RocmGpuTracer::CollectData(RunMetadata* run_metadata) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(1) << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, xprof failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(1) << "No trace data collected";
      return Status::OK();
    case State::kStoppedOk: {
      // Input run_metadata is shared by profiler interfaces, we need append.
      trace_collector_.Finalize();
      for (auto& dev_stats : *step_stats_.mutable_dev_stats()) {
        run_metadata->mutable_step_stats()->add_dev_stats()->Swap(&dev_stats);
      }
      return Status::OK();
    }
  }
  return errors::Internal("Invalid profiling state: ", profiling_state_);
}

}  // namespace profiler

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const profiler::ProfilerOptions& options) {
  if (options.device_type != profiler::DeviceType::kGpu &&
      options.device_type != profiler::DeviceType::kUnspecified)
    return nullptr;
  profiler::RocmTracer* rocm_tracer =
      profiler::RocmTracer::GetRocmTracerSingleton();
  if (!rocm_tracer->IsAvailable()) {
    return nullptr;
  }
  return absl::make_unique<profiler::RocmGpuTracer>(rocm_tracer);
}

auto register_rocm_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
