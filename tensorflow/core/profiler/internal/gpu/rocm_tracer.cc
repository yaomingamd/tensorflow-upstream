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

#include "tensorflow/core/profiler/internal/gpu/rocm_tracer.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/annotation.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"

#include "roctracer/roctracer.h"
#include "hip/hip_runtime.h"

namespace tensorflow {
namespace profiler {

namespace {

static thread_local bool internalRocmCall;

// Temporary disable cupti api tracing for this thread during the life scope of
// this class. Used for the API calls that initiated by us.
class RocmApiTracingDisabler {
 public:
  RocmApiTracingDisabler() { internalRocmCall = true; }
  ~RocmApiTracingDisabler() { internalRocmCall = false; }
};

Status ToStatus(roctracer_status_t result) {
  if (result == ROCTRACER_STATUS_SUCCESS) {
    return Status::OK();
  }
  const char *str = roctracer_error_string();
  return errors::Unavailable("ROCTRACER error: ", str ? str : "<unknown>");
}

Status ToStatus(hipError_t result) {
  if (result == hipSuccess) {
    return Status::OK();
  }
  const char *str = hipGetErrorString(result);
  return errors::Unavailable("ROCM error: ", str ? str : "<unknown>");
}

inline void LogIfError(const Status &status) {
  if (status.ok()) return;
  LOG(ERROR) << status.error_message();
}

// ROCM TODO: revise with roctracer enum
//// Maps an OverheadKind enum to a const string.
//const char *getActivityOverheadKindString(CUpti_ActivityOverheadKind kind) {
//  switch (kind) {
//    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
//      return "COMPILER";
//    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
//      return "BUFFER_FLUSH";
//    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
//      return "INSTRUMENTATION";
//    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
//      return "RESOURCE";
//    default:
//      break;
//  }
//  return "<UNKNOWN>";
//}

#define RETURN_IF_ROCTRACER_ERROR(expr)                                     \
  do {                                                                      \
    roctracer_status_t status = expr;                                       \
    if (status != ROCTRACER_STATUS_SUCCESS) {                               \
      const char *errstr = roctracer_error_string();                        \
      LOG(ERROR) << "function " << #expr << "failed with error " << errstr; \
      return errors::Internal(absl::StrCat("roctracer call error", errstr));\
    }                                                                       \
  } while (false)

// GetCachedTID() caches the thread ID in thread-local storage (which is a
// userspace construct) to avoid unnecessary system calls. Without this caching,
// it can take roughly 98ns, while it takes roughly 1ns with this caching.
int32 GetCachedTID() {
  static thread_local int32 current_thread_id =
      Env::Default()->GetCurrentThreadId();
  return current_thread_id;
}

// ROCM TODO: revise with roctracer / HIP API?
//std::tuple<size_t /*bytes*/, RocmTracerEventType, bool /*async*/>
//DecodeDriverMemcpy(CUpti_CallbackId cbid, const void *params) {
//  switch (cbid) {
//    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy: {
//      const auto *p = reinterpret_cast<const cuMemcpy_params *>(params);
//      return std::make_tuple(p->ByteCount, RocmTracerEventType::Unsupported,
//                             false);
//    }
//    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync: {
//      const auto *p = reinterpret_cast<const cuMemcpyAsync_params *>(params);
//      return std::make_tuple(p->ByteCount, RocmTracerEventType::Unsupported,
//                             true);
//    }
//    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer: {
//      const cuMemcpyPeer_params *p2p_params =
//          reinterpret_cast<const cuMemcpyPeer_params *>(params);
//      return std::make_tuple(p2p_params->ByteCount,
//                             RocmTracerEventType::MemcpyP2P, false);
//    }
//    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync: {
//      const cuMemcpyPeerAsync_params_st *p2p_params =
//          reinterpret_cast<const cuMemcpyPeerAsync_params_st *>(params);
//      return std::make_tuple(p2p_params->ByteCount,
//                             RocmTracerEventType::MemcpyP2P, true);
//    }
//    default: {
//      LOG(ERROR) << "Unsupported memcpy activity observed: " << cbid;
//      return std::make_tuple(0, RocmTracerEventType::Unsupported, false);
//    }
//  }
//}

// Rocm callback corresponding to a driver or runtime API. This global function
// is invoked twice for each API: at entry and at exit. The cbdata
// parameter is guaranteed by Rocm to be thread-safe. Most invocations are
// dropped to the floor and entry/exit is tracked for the APIs we deem
// performance-relevant.
void ApiCallback(uint32_t domain,
                 uint32_t cbid,
                 const void *cbdata,
                 void *user_data) {
  LOG(INFO) << "ApiCallback\n";
  RocmTracer *tracer = reinterpret_cast<RocmTracer *>(user_data);
  tracer->HandleCallback(domain, cbid, cbdata).IgnoreError();
}

void ActivityCallback(const char *begin, const char *end, void *user_data) {
  LOG(INFO) << "ActivityCallback\n";
  RocmTracer *tracer = reinterpret_cast<RocmTracer *>(user_data);
  tracer->ProcessActivityRecord(begin, end);
}

// ROCM TODO: revise with roctracer API
//void AddKernelEventUponApiExit(RocmTraceCollector *collector, uint32 device_id,
//                               const CUpti_CallbackData *cbdata,
//                               uint64 start_time, uint64 end_time) {
//  RocmTracerEvent event;
//  event.type = RocmTracerEventType::Kernel;
//  event.source = RocmTracerEventSource::DriverCallback;
//  event.name = cbdata->symbolName;
//  event.start_time_ns = start_time;
//  event.end_time_ns = end_time;
//  event.thread_id = GetCachedTID();
//  event.device_id = device_id;
//  event.context_id = cbdata->contextUid;
//  event.correlation_id = cbdata->correlationId;
//  VLOG(3) << "Cuda Kernel Launched: " << event.name;
//  collector->AddEvent(std::move(event));
//}

// ROCM TODO: revise with roctracer API
//// Performs the actual callback for both normal and P2P memcpy operations.
//RocmTracerEvent PopulateMemcpyCallbackEvent(
//    RocmTracerEventType type, const CUpti_CallbackData *cbdata,
//    size_t num_bytes, uint32 src_device, uint32 dst_device, bool async,
//    uint64 start_time, uint64 end_time) {
//  RocmTracerEvent event;
//  event.type = type;
//  event.source = RocmTracerEventSource::DriverCallback;
//  event.start_time_ns = start_time;
//  event.end_time_ns = end_time;
//  event.thread_id = GetCachedTID();
//  event.device_id = src_device;
//  event.context_id = cbdata->contextUid;
//  event.correlation_id = cbdata->correlationId;
//  event.memcpy_info.kind = CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN;
//  event.memcpy_info.num_bytes = num_bytes;
//  event.memcpy_info.destination = dst_device;
//  event.memcpy_info.async = async;
//  return event;
//}

// ROCM TODO: revise with roctracer API
//void AddNormalMemcpyEventUponApiExit(RocmTraceCollector *collector,
//                                     uint32 device_id, CUpti_CallbackId cbid,
//                                     const CUpti_CallbackData *cbdata,
//                                     uint64 start_time, uint64 end_time) {
//  size_t num_bytes;
//  RocmTracerEventType type;
//  bool async;
//  std::tie(num_bytes, type, async) =
//      DecodeDriverMemcpy(cbid, cbdata->functionParams);
//
//  VLOG(3) << "Cuda Memcpy observed :" << num_bytes;
//  RocmTracerEvent event =
//      PopulateMemcpyCallbackEvent(type, cbdata, num_bytes, device_id, device_id,
//                                  async, start_time, end_time);
//  collector->AddEvent(std::move(event));
//}

// ROCM TODO: revise with roctracer API
//void AddP2PMemcpyEventUponApiExit(RocmTraceCollector *collector,
//                                  RocmInterface *cupti_interface,
//                                  uint32 device_id, CUpti_CallbackId cbid,
//                                  const CUpti_CallbackData *cbdata,
//                                  uint64 start_time, uint64 end_time) {
//  size_t num_bytes;
//  RocmTracerEventType type;
//  bool async;
//  std::tie(num_bytes, type, async) =
//      DecodeDriverMemcpy(cbid, cbdata->functionParams);
//
//  uint32 dst_device = -1, src_device = -1;
//  const cuMemcpyPeer_params *p2p_params =
//      reinterpret_cast<const cuMemcpyPeer_params *>(cbdata->functionParams);
//  cupti_interface->GetDeviceId(p2p_params->srcContext, &src_device);
//  cupti_interface->GetDeviceId(p2p_params->dstContext, &dst_device);
//  VLOG(3) << "Cuda P2P Memcpy observed, src: " << src_device
//          << " dst: " << dst_device << " size:" << num_bytes;
//  RocmTracerEvent event =
//      PopulateMemcpyCallbackEvent(type, cbdata, num_bytes, src_device,
//                                  dst_device, async, start_time, end_time);
//  collector->AddEvent(std::move(event));
//}

// ROCM TODO: revise with roctracer API
//void AddCudaMallocEventUponApiExit(RocmTraceCollector *collector,
//                                   uint32 device_id, CUpti_CallbackId cbid,
//                                   const CUpti_CallbackData *cbdata,
//                                   uint64 start_time, uint64 end_time) {
//  const cuMemAlloc_v2_params_st *params =
//      reinterpret_cast<const cuMemAlloc_v2_params_st *>(cbdata->functionParams);
//  RocmTracerEvent event;
//  event.type = RocmTracerEventType::MemoryAlloc;
//  event.source = RocmTracerEventSource::DriverCallback;
//  event.name = cbdata->functionName;
//  event.start_time_ns = start_time;
//  event.end_time_ns = end_time;
//  event.thread_id = GetCachedTID();
//  event.device_id = device_id;
//  event.context_id = cbdata->contextUid;
//  event.correlation_id = cbdata->correlationId;
//  event.memalloc_info.num_bytes = params->bytesize;
//  VLOG(3) << "Cuda Malloc/Free observed: " << params->bytesize;
//  collector->AddEvent(std::move(event));
//}

void AddGenericEventUponApiExit(RocmTraceCollector *collector,
                                uint32 device_id, uint32_t cbid,
                                const void *cbdata,
                                uint64 start_time, uint64 end_time) {
  RocmTracerEvent event;
  event.type = RocmTracerEventType::Generic;
  event.source = RocmTracerEventSource::DriverCallback;
  //event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = GetCachedTID();
  event.device_id = device_id;
  //event.context_id = cbdata->contextUid;
  //event.correlation_id = cbdata->correlationId;
  LOG(INFO) << "collect AddEvent";
  collector->AddEvent(std::move(event));
}

// ROCM TODO: revise with roctracer API
//void AddKernelActivityEvent(RocmTraceCollector *collector,
//                            AnnotationMap *annotation_map,
//                            const CUpti_ActivityKernel4 *kernel) {
//  RocmTracerEvent event;
//  event.type = RocmTracerEventType::Kernel;
//  event.source = RocmTracerEventSource::Activity;
//  event.name = kernel->name;
//  event.start_time_ns = kernel->start;
//  event.end_time_ns = kernel->end;
//  event.device_id = kernel->deviceId;
//  event.context_id = kernel->contextId;
//  event.stream_id = kernel->streamId;
//  event.correlation_id = kernel->correlationId;
//  event.annotation =
//      annotation_map->LookUp(event.device_id, event.correlation_id);
//  event.kernel_info.registers_per_thread = kernel->registersPerThread;
//  event.kernel_info.static_shared_memory_usage = kernel->staticSharedMemory;
//  event.kernel_info.dynamic_shared_memory_usage = kernel->dynamicSharedMemory;
//  event.kernel_info.block_x = kernel->blockX;
//  event.kernel_info.block_y = kernel->blockY;
//  event.kernel_info.block_z = kernel->blockZ;
//  event.kernel_info.grid_x = kernel->gridX;
//  event.kernel_info.grid_y = kernel->gridY;
//  event.kernel_info.grid_z = kernel->gridZ;
//  collector->AddEvent(std::move(event));
//}

// ROCM TODO: revise with roctracer API
//void AddMemcpyActivityEvent(RocmTraceCollector *collector,
//                            AnnotationMap *annotation_map,
//                            const CUpti_ActivityMemcpy *memcpy) {
//  RocmTracerEvent event;
//  switch (memcpy->copyKind) {
//    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
//      event.type = RocmTracerEventType::MemcpyH2D;
//      event.name = "MemcpyH2D";
//      break;
//    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
//      event.type = RocmTracerEventType::MemcpyD2H;
//      event.name = "MemcpyD2H";
//      break;
//    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
//      event.type = RocmTracerEventType::MemcpyD2D;
//      event.name = "MemcpyD2D";
//      break;
//    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
//      event.type = RocmTracerEventType::MemcpyP2P;
//      event.name = "MemcpyP2P";
//      break;
//    default:
//      event.type = RocmTracerEventType::MemcpyOther;
//      event.name = "MemcpyOther";
//      break;
//  }
//  event.source = RocmTracerEventSource::Activity;
//  event.start_time_ns = memcpy->start;
//  event.end_time_ns = memcpy->end;
//  event.device_id = memcpy->deviceId;
//  event.context_id = memcpy->contextId;
//  event.stream_id = memcpy->streamId;
//  event.correlation_id = memcpy->correlationId;
//  event.annotation =
//      annotation_map->LookUp(event.device_id, event.correlation_id);
//  event.memcpy_info.kind = memcpy->copyKind;
//  event.memcpy_info.num_bytes = memcpy->bytes;
//  event.memcpy_info.destination = memcpy->deviceId;
//  event.memcpy_info.async = memcpy->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
//  collector->AddEvent(std::move(event));
//}

// ROCM TODO: revise with roctracer API
//// Invokes callback upon peer-2-peer memcpy between different GPU devices.
//void AddMemcpy2ActivityEvent(RocmTraceCollector *collector,
//                             AnnotationMap *annotation_map,
//                             const CUpti_ActivityMemcpy2 *memcpy2) {
//  RocmTracerEvent event;
//  event.type = RocmTracerEventType::MemcpyP2P;
//  event.name = "MemcpyP2P";
//  event.source = RocmTracerEventSource::Activity;
//  event.start_time_ns = memcpy2->start;
//  event.end_time_ns = memcpy2->end;
//  event.device_id = memcpy2->srcDeviceId;
//  event.context_id = memcpy2->contextId;
//  event.stream_id = memcpy2->streamId;
//  event.correlation_id = memcpy2->correlationId;
//  event.annotation =
//      annotation_map->LookUp(event.device_id, event.correlation_id);
//  event.memcpy_info.kind = CUPTI_ACTIVITY_MEMCPY_KIND_PTOP;
//  event.memcpy_info.num_bytes = memcpy2->bytes;
//  event.memcpy_info.destination = memcpy2->dstDeviceId;
//  event.memcpy_info.async = memcpy2->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
//  collector->AddEvent(std::move(event));
//}

// ROCM TODO: revise with roctracer API
//void AddRocmOverheadActivityEvent(RocmTraceCollector *collector,
//                                   const CUpti_ActivityOverhead *overhead) {
//  RocmTracerEvent event;
//  event.type = RocmTracerEventType::Overhead;
//  event.name = getActivityOverheadKindString(overhead->overheadKind);
//  event.source = RocmTracerEventSource::Activity;
//  event.start_time_ns = overhead->start;
//  event.end_time_ns = overhead->end;
//  // If the overhead is not related to a device, we assign it to device 0.
//  event.device_id = 0;
//  // NOTE: no correlation id.
//  switch (overhead->objectKind) {
//    case CUPTI_ACTIVITY_OBJECT_UNKNOWN:
//      // Don't know how to deal with such activities because of we need either
//      // attribute it to a GPU stream or a CPU thread.
//      return;
//
//    case CUPTI_ACTIVITY_OBJECT_THREAD:
//    case CUPTI_ACTIVITY_OBJECT_PROCESS:
//      event.thread_id = overhead->objectId.pt.threadId;
//      break;
//    case CUPTI_ACTIVITY_OBJECT_STREAM:
//      event.stream_id = overhead->objectId.dcs.streamId;
//      ABSL_FALLTHROUGH_INTENDED;
//    case CUPTI_ACTIVITY_OBJECT_DEVICE:
//    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
//      event.device_id = overhead->objectId.dcs.deviceId;
//      break;
//    default:
//      LOG(ERROR) << "Unexpected object kind: " << overhead->objectKind;
//      return;
//  }
//  collector->AddEvent(std::move(event));
//}

// ROCM TODO: revise with roctracer API
//void AddUnifiedMemoryActivityEvent(
//    RocmTraceCollector *collector,
//    const CUpti_ActivityUnifiedMemoryCounter2 *record) {
//  VLOG(3) << "Cuda Unified Memory Activity, kind: " << record->counterKind
//          << " src: " << record->srcId << " dst: " << record->dstId;
//  RocmTracerEvent event;
//  event.type = RocmTracerEventType::UnifiedMemory;
//  event.name = getActivityUnifiedMemoryKindString(record->counterKind);
//  event.source = RocmTracerEventSource::Activity;
//  event.start_time_ns = record->start;
//  if (record->counterKind ==
//          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT ||
//      record->counterKind ==
//          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING ||
//      record->counterKind ==
//          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP ||
//      record->end <= record->start) {
//    // If the end time is not valid, trim it so that it can be shown on the UI.
//    event.end_time_ns = record->start + 1;
//  } else {
//    event.end_time_ns = record->end;
//  }
//  event.device_id = record->srcId;
//  // NOTE: not context id and correlation id.
//
//  // For visualization purpose, we assign a pseudo stream id for each
//  // record->counterKind of unified memory related events.
//  constexpr int kPseudoStreamId = 0x10000000;
//  event.stream_id = kPseudoStreamId + record->counterKind;
//  event.memcpy_info.kind = CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN;
//  // Check whether the activity is byte transfer.
//  if (record->counterKind ==
//          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD ||
//      record->counterKind ==
//          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH ||
//      record->counterKind ==
//          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD) {
//    event.memcpy_info.num_bytes = record->value;
//  } else {
//    event.memcpy_info.num_bytes = 0;
//  }
//  event.memcpy_info.destination = record->dstId;
//  event.memcpy_info.async = false;
//  collector->AddEvent(std::move(event));
//}

// This hook uses cupti activity api to measure device side activities.
class RocmDriverApiHookWithActivityApi : public RocmDriverApiHook {
 public:
  RocmDriverApiHookWithActivityApi(const RocmTracerOptions &option,
                                   RocmTraceCollector *collector,
                                   AnnotationMap *annotation_map)
      : option_(option),
        collector_(collector),
        annotation_map_(annotation_map) {}

  Status OnDriverApiEnter(int device_id, uint32_t domain,
                          uint32_t cbid,
                          const void *cbdata) override {
    // ROCM TODO: revise this.
    //// Stash away the current Rocm timestamp into cbdata.
    //*cbdata->correlationData =
    //    option_.required_callback_api_events ? RocmTracer::GetTimestamp() : 0;
    return Status::OK();
  }
  Status OnDriverApiExit(int device_id, uint32_t domain,
                         uint32_t cbid,
                         const void *cbdata) override {
    // If we are not collecting CPU events from Callback API, we can return now.
    if (!option_.required_callback_api_events) {
      return Status::OK();
    }

    // Grab timestamp for API exit. API entry timestamp saved in cbdata.
    uint64 end_tsc = RocmTracer::GetTimestamp();
    // ROCM TODO: revise this.
    uint64 start_tsc = 0; //*cbdata->correlationData;
    return AddDriverApiCallbackEvent(collector_, device_id,
                                     start_tsc, end_tsc, domain, cbid, cbdata);
  }
  Status Flush() override { return Status::OK(); }

 private:
  const RocmTracerOptions option_;
  RocmTraceCollector *collector_;
  AnnotationMap *annotation_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(RocmDriverApiHookWithActivityApi);
};

struct KernelRecord {
  const char *kernel_name;
  // TODO(csigg): cuStreamGetCtx introduced in CUDA 9.2 would allow us to only
  // record the stream and infer the context during collection.
  hipCtx_t context;
  hipStream_t stream;
  uint32 correlation_id;
  hipEvent_t start_event;
  hipEvent_t stop_event;
  KernelDetails details;
  uint64 start_timestamp;
};

struct MemcpyRecord {
  RocmTracerEventType type;
  size_t size_bytes;
  hipCtx_t context;
  hipStream_t stream;
  uint32 correlation_id;
  hipEvent_t start_event;
  hipEvent_t stop_event;
  uint64 start_timestamp;
};

// ROCM TODO: revise this with HIP API.
//Status CreateAndRecordEvent(hipEvent_t *event, hipStream_t stream) {
//  RocmApiTracingDisabler disabler;
//  TF_RETURN_IF_ERROR(ToStatus(cuEventCreate(event, CU_EVENT_DEFAULT)));
//  return ToStatus(cuEventRecord(*event, stream));
//}

// Stores a series of kernel and memcpy records.
class HipEventRecorder {
 public:
  HipEventRecorder(RocmTraceCollector *collector, int ordinal)
      : collector_(collector),
        ordinal_(ordinal) {
    device_name_ = absl::StrCat("gpu ", ordinal);  // default.
    // ROCM TODO: revise this with HIP API.
    //hipDevice_t device;
    //if (hipGetDevice(&device, ordinal) == CUDA_SUCCESS) {
    //  char name[100];
    //  if (cuDeviceGetName(name, sizeof(name), device) == CUDA_SUCCESS) {
    //    device_name_ = name;
    //  }
    //}
  }

  // ROCM TODO: enable this at the right timing.
  //// Registers the start of a kernel launch. The returned index should be passed
  //// to StopKernel() after the kernel launch has completed.
  //size_t StartKernel(const char *kernel_name, hipCtx_t context,
  //                   uint32 correlation_id,
  //                   const cuLaunchKernel_params *params) {
  //  hipStream_t stream = params->hStream;
  //  KernelRecord record = {kernel_name, context, stream, correlation_id};
  //  record.details.registers_per_thread = 0;  // unknown.
  //  record.details.static_shared_memory_usage = params->sharedMemBytes;
  //  record.details.dynamic_shared_memory_usage = 0;  // unknown
  //  record.details.block_x = params->blockDimX;
  //  record.details.block_y = params->blockDimY;
  //  record.details.block_z = params->blockDimZ;
  //  record.details.grid_x = params->gridDimX;
  //  record.details.grid_y = params->gridDimY;
  //  record.details.grid_z = params->gridDimZ;
  //  record.start_timestamp = RocmTracer::GetTimestamp();
  //  LogIfError(CreateAndRecordEvent(&record.start_event, stream));
  //  absl::MutexLock lock(&mutex_);
  //  if (stopped_) return -1;
  //  kernel_records_.push_back(record);
  //  return kernel_records_.size() - 1;
  //}
  //uint64 StopKernel(size_t index) {
  //  absl::MutexLock lock(&mutex_);
  //  if (index >= kernel_records_.size()) return 0;
  //  auto &record = kernel_records_[index];
  //  LogIfError(CreateAndRecordEvent(&record.stop_event, record.stream));
  //  return record.start_timestamp;
  //}

  //// Registers the start of a copy operation. The returned index should be
  //// passed to StopMemcpy() after the memcpy has completed.
  //size_t StartMemcpy(RocmTracerEventType type, size_t size_bytes,
  //                   hipCtx_t context, hipStream_t stream,
  //                   uint32 correlation_id) {
  //  MemcpyRecord record = {type, size_bytes, context, stream, correlation_id};
  //  record.start_timestamp = RocmTracer::GetTimestamp();
  //  LogIfError(CreateAndRecordEvent(&record.start_event, stream));
  //  absl::MutexLock lock(&mutex_);
  //  if (stopped_) return -1;
  //  memcpy_records_.push_back(record);
  //  return memcpy_records_.size() - 1;
  //}
  //uint64 StopMemcpy(size_t index) {
  //  absl::MutexLock lock(&mutex_);
  //  if (index >= memcpy_records_.size()) return 0;
  //  auto &record = memcpy_records_[index];
  //  LogIfError(CreateAndRecordEvent(&record.stop_event, record.stream));
  //  return record.start_timestamp;
  //}

  Status Stop() {
    {
      absl::MutexLock lock(&mutex_);
      stopped_ = true;
      LOG(INFO) << "Collecting " << kernel_records_.size()
                << " kernel records, " << memcpy_records_.size()
                << " memcpy records.";

      // Gather all profiled streams and contexts.
      for (const auto &record : kernel_records_) {
        TF_RETURN_IF_ERROR(
            AddStreamInfo(record.context, record.stream, "Kernel"));
      }
      for (const auto &record : memcpy_records_) {
        TF_RETURN_IF_ERROR(AddStreamInfo(record.context, record.stream,
                                         GetTraceEventTypeName(record.type)));
      }
    }

    // Synchronize all contexts, record end events, synchronize again.
    // This scheme is an unreliable measure to associate a event with the wall
    // time. There are chances that other threads might enque kernels which
    // delay the second synchornization.
    TF_RETURN_IF_ERROR(Synchronize());
    // ROCM TODO: re-enable this later.
    //for (auto &pair : context_infos_) {
    //  TF_RETURN_IF_ERROR(ToStatus(cuCtxSetCurrent(pair.first)));
    //  TF_RETURN_IF_ERROR(CreateAndRecordEvent(&pair.second.end_event, nullptr));
    //}

    TF_RETURN_IF_ERROR(Synchronize());
    end_walltime_us_ = Env::Default()->NowMicros();
    return Status::OK();
  }

  Status Flush(AnnotationMap *annotation_map) {
    auto kernel_records = ConsumeKernelRecords();
    auto memcpy_records = ConsumeMemcpyRecords();
    for (const auto &record : kernel_records) {
      TF_RETURN_IF_ERROR(SaveRecord(record, annotation_map));
    }
    for (const auto &record : memcpy_records) {
      TF_RETURN_IF_ERROR(SaveRecord(record, annotation_map));
    }
    return Status::OK();
  }

  std::vector<KernelRecord> ConsumeKernelRecords() {
    absl::MutexLock lock(&mutex_);
    return std::move(kernel_records_);
  }
  std::vector<MemcpyRecord> ConsumeMemcpyRecords() {
    absl::MutexLock lock(&mutex_);
    return std::move(memcpy_records_);
  }

 private:
  struct ContextInfo {
    uint32 context_id = 0;
    int num_streams = 0;
    hipEvent_t end_event;
  };

  struct StreamInfo {
    uint32 stream_id = 0;
    std::string name;
    int index;  // 0 is reserved for null stream.
    const ContextInfo *ctx_info;
  };

  // Synchronizes all contexts.
  Status Synchronize() const {
    RocmApiTracingDisabler disabler;
    // ROCM TODO: re-enable this later.
    //for (const auto &pair : context_infos_) {
    //  TF_RETURN_IF_ERROR(ToStatus(cuCtxSetCurrent(pair.first)));
    //  TF_RETURN_IF_ERROR(ToStatus(cuCtxSynchronize()));
    //}
    return Status::OK();
  }

  // Returns element from context_infos_, adding it if not yet present.
  Status GetContextInfo(hipCtx_t context, ContextInfo **ctx_info_ptr) {
    auto it = context_infos_.find(context);

    if (it == context_infos_.end()) {
      uint32 context_id = 0;
      // ROCM TODO: enable this per HIP API or roctracer API.
      //RETURN_IF_ROCTRACER_ERROR(
      //    cupti_interface_->GetContextId(context, &context_id));
      ContextInfo ctx_info = {context_id};
      it = context_infos_.emplace(context, ctx_info).first;
    }

    *ctx_info_ptr = &it->second;
    return Status::OK();
  }

  // Adds element to stream_infos_ if not yet present. If present, clear name
  // if it doesn't match parameter.
  Status AddStreamInfo(hipCtx_t context, hipStream_t stream,
                       absl::string_view name) {
    StreamKey key(context, stream);
    auto it = stream_infos_.find(key);
    if (it != stream_infos_.end()) {
      if (it->second.name != name) {
        it->second.name.clear();  // Stream with inconsistent names, clear it.
      }
      return Status::OK();
    }

    // ROCM TODO: re-enable this later.
    ContextInfo *ctx_info;
    TF_RETURN_IF_ERROR(GetContextInfo(context, &ctx_info));
    int index = stream ? ++ctx_info->num_streams : 0;
    uint32 stream_id = 0;
    // ROCM TODO: enable this per HIP API or roctrcer API.
//#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM)
//    RETURN_IF_ROCTRACER_ERROR(
//        cupti_interface_->GetStreamIdEx(context, stream, 1, &stream_id));
//#else
//    RETURN_IF_ROCTRACER_ERROR(
//        cupti_interface_->GetStreamIdEx(context, stream, 0, &stream_id));
//#endif

    StreamInfo stream_info = {stream_id, static_cast<std::string>(name), index,
                              ctx_info};
    stream_infos_.emplace(key, stream_info);
    return Status::OK();
  }

  // Returns time in microseconds between events recorded on the GPU.
  static uint64_t GetElapsedTimeUs(hipEvent_t start, hipEvent_t stop) {
    RocmApiTracingDisabler disabler;
    float elapsed_ms = 0.0f;
    // ROCM TODO: re-enable this later.
    //LogIfError(ToStatus(cuEventElapsedTime(&elapsed_ms, start, stop)));
    return static_cast<uint64>(
        std::llroundf(1000 * std::max(elapsed_ms, 0.0f)));
  }

  Status SaveRecord(const KernelRecord &record,
                    AnnotationMap *annotation_map) const {
    if (!record.start_event || !record.stop_event) {
      return Status::OK();
    }
    const auto &stream_info =
        stream_infos_.at(StreamKey(record.context, record.stream));
    auto start_us =
        GetElapsedTimeUs(record.start_event, stream_info.ctx_info->end_event);
    auto elapsed_us = GetElapsedTimeUs(record.start_event, record.stop_event);

    std::string annotation;

    RocmTracerEvent event;
    event.type = RocmTracerEventType::Kernel;
    event.source = RocmTracerEventSource::Activity;  // on gpu device.
    event.name = record.kernel_name;
    event.start_time_ns = (end_walltime_us_ - start_us) * 1000;
    event.end_time_ns = event.start_time_ns + elapsed_us * 1000;
    event.device_id = ordinal_;
    event.context_id = stream_info.ctx_info->context_id;
    event.stream_id = stream_info.stream_id;
    event.correlation_id = record.correlation_id;
    event.annotation =
        annotation_map->LookUp(event.device_id, event.correlation_id);
    event.kernel_info = record.details;
    collector_->AddEvent(std::move(event));
    return Status::OK();
  }

  Status SaveRecord(const MemcpyRecord &record,
                    AnnotationMap *annotation_map) const {
    if (!record.start_event || !record.stop_event) {
      return Status::OK();
    }
    const auto &stream_info =
        stream_infos_.at(StreamKey(record.context, record.stream));
    auto start_us =
        GetElapsedTimeUs(record.start_event, stream_info.ctx_info->end_event);
    auto elapsed_us = GetElapsedTimeUs(record.start_event, record.stop_event);

    RocmTracerEvent event;
    event.type = record.type;
    event.name = GetTraceEventTypeName(event.type);
    event.source = RocmTracerEventSource::Activity;
    event.start_time_ns = (end_walltime_us_ - start_us) * 1000;
    event.end_time_ns = event.start_time_ns + elapsed_us * 1000;
    event.device_id = ordinal_;
    event.context_id = stream_info.ctx_info->context_id;
    event.stream_id = stream_info.stream_id;
    event.correlation_id = record.correlation_id;
    event.annotation =
        annotation_map->LookUp(event.device_id, event.correlation_id);
    event.memcpy_info.num_bytes = record.size_bytes;
    event.memcpy_info.destination = ordinal_;
    // TODO: support differentiate sync and async memcpy.
    event.memcpy_info.async = false;
    collector_->AddEvent(std::move(event));
    return Status::OK();
  }

  absl::Mutex mutex_;
  bool stopped_ GUARDED_BY(mutex_) = false;
  std::vector<KernelRecord> kernel_records_ GUARDED_BY(mutex_);
  std::vector<MemcpyRecord> memcpy_records_ GUARDED_BY(mutex_);

  RocmTraceCollector *collector_;
  const int ordinal_;
  std::string device_name_;
  uint64 end_walltime_us_;
  // Include context in key to distinguish null streams.
  using StreamKey = std::pair<hipCtx_t, hipStream_t>;

  absl::node_hash_map<hipCtx_t, ContextInfo> context_infos_;
  absl::flat_hash_map<StreamKey, StreamInfo, hash<StreamKey>> stream_infos_;
};

// This hook uses cuda events to measure device side activities.
class RocmDriverApiHookWithHipEvent : public RocmDriverApiHook {
 public:
  RocmDriverApiHookWithHipEvent(const RocmTracerOptions &option,
                                  RocmTraceCollector *collector,
                                  AnnotationMap *annotation_map)
      : option_(option),
        annotation_map_(annotation_map),
        collector_(collector) {
    int num_gpus = RocmTracer::NumGpus();
    hip_event_recorders_.reserve(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
      hip_event_recorders_.emplace_back(
          absl::make_unique<HipEventRecorder>(collector, i));
    }
  }

  Status OnDriverApiEnter(int device_id, uint32_t domain,
                          uint32_t cbid,
                          const void *cbdata) override {
    auto *recorder = hip_event_recorders_[device_id].get();
    switch (cbid) {
      // ROCM TODO: revise this.
      //case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: {
      //  DCHECK_NE(cbdata->symbolName, nullptr);
      //  auto params =
      //      static_cast<const cuLaunchKernel_params *>(cbdata->functionParams);
      //  *cbdata->correlationData = recorder->StartKernel(
      //      cbdata->symbolName, cbdata->context, cbdata->correlationId, params);
      //  break;
      //}
      //case CUPTI_DRIVER_TRACE_CBID_cuMemcpy: {
      //  auto params =
      //      static_cast<const cuMemcpy_params *>(cbdata->functionParams);
      //  StartMemcpy<cuMemcpy_params>(GetMemcpyType(params->src, params->dst),
      //                               cbdata, recorder);
      //  break;
      //}
      //case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync: {
      //  auto params =
      //      static_cast<const cuMemcpyAsync_params *>(cbdata->functionParams);
      //  StartMemcpyAsync<cuMemcpyAsync_params>(
      //      GetMemcpyType(params->src, params->dst), cbdata, recorder);
      //  break;
      //}
      default:
        VLOG(1) << "Unexpected callback id: " << cbid;
        break;
    }
    return Status::OK();
  }
  Status OnDriverApiExit(int device_id, uint32_t domain,
                         uint32_t cbid,
                         const void *cbdata) override {
    auto *recorder = hip_event_recorders_[device_id].get();
    // ROCM TODO: revise this per roctracer API.
    //if (*cbdata->correlationData == static_cast<size_t>(-1))
    //  return Status::OK();
    uint64 start_tsc = 0;
    switch (cbid) {
      // ROCM TODO: revise this.
      //case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      //  start_tsc = recorder->StopKernel(*cbdata->correlationData);
      //  break;
      //case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
      //case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
      //  start_tsc = recorder->StopMemcpy(*cbdata->correlationData);
      //  break;
      default:
        VLOG(1) << "Unexpected callback id: " << cbid;
        // TODO: figure out how to get start timestamp in this case.
        return Status::OK();
    }
    // If we are not collecting CPU events from Callback API, we can return now.
    if (!option_.required_callback_api_events) {
      return Status::OK();
    }

    // Grab timestamp for API exit. API entry timestamp saved in cbdata.
    uint64 end_tsc = RocmTracer::GetTimestamp();
    return AddDriverApiCallbackEvent(collector_, device_id,
                                     start_tsc, end_tsc, domain, cbid, cbdata);
  }
  Status Flush() override {
    for (auto &recorder : hip_event_recorders_) {
      TF_RETURN_IF_ERROR(recorder->Stop());
    }
    for (auto &recorder : hip_event_recorders_) {
      TF_RETURN_IF_ERROR(recorder->Flush(annotation_map_));
    }
    return Status::OK();
  }

 private:
  // ROCM TODO: revise this.
  //template <typename T>
  //static void StartMemcpy(RocmTracerEventType type,
  //                        const CUpti_CallbackData *cbdata,
  //                        HipEventRecorder *recorder) {
  //  auto params = static_cast<const T *>(cbdata->functionParams);
  //  *cbdata->correlationData =
  //      recorder->StartMemcpy(type, params->ByteCount, cbdata->context, nullptr,
  //                            cbdata->correlationId);
  //}
  //template <typename T>
  //static void StartMemcpyAsync(RocmTracerEventType type,
  //                             const CUpti_CallbackData *cbdata,
  //                             HipEventRecorder *recorder) {
  //  auto params = static_cast<const T *>(cbdata->functionParams);
  //  *cbdata->correlationData =
  //      recorder->StartMemcpy(type, params->ByteCount, cbdata->context,
  //                            params->hStream, cbdata->correlationId);
  //}

  //static CUmemorytype GetMemoryType(CUdeviceptr ptr) {
  //  RocmApiTracingDisabler disabler;
  //  CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
  //  auto status =
  //      cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr);
  //  if (status == CUDA_ERROR_INVALID_VALUE) {
  //    // Pointer not registered with CUDA, must be host memory.
  //    return CU_MEMORYTYPE_HOST;
  //  }
  //  LogIfError(ToStatus(status));
  //  return mem_type;
  //}

  //static RocmTracerEventType GetMemcpyType(CUdeviceptr src, CUdeviceptr dst) {
  //  CUmemorytype src_type = GetMemoryType(src);
  //  CUmemorytype dst_type = GetMemoryType(dst);
  //  // TODO: handle CU_MEMORYTYPE_ARRAY case
  //  if (src_type == CU_MEMORYTYPE_HOST && dst_type == CU_MEMORYTYPE_DEVICE) {
  //    return RocmTracerEventType::MemcpyH2D;
  //  } else if (src_type == CU_MEMORYTYPE_DEVICE &&
  //             dst_type == CU_MEMORYTYPE_HOST) {
  //    return RocmTracerEventType::MemcpyD2H;
  //  } else if (src_type == CU_MEMORYTYPE_DEVICE &&
  //             dst_type == CU_MEMORYTYPE_DEVICE) {
  //    return RocmTracerEventType::MemcpyD2D;
  //  }
  //  return RocmTracerEventType::MemcpyOther;
  //}

  const RocmTracerOptions option_;
  AnnotationMap *annotation_map_;
  RocmTraceCollector *collector_;
  std::vector<std::unique_ptr<HipEventRecorder>> hip_event_recorders_;
  TF_DISALLOW_COPY_AND_ASSIGN(RocmDriverApiHookWithHipEvent);
};
}  // namespace

/*static*/ Status RocmDriverApiHook::AddDriverApiCallbackEvent(
    RocmTraceCollector *collector,
    int device_id, uint64 start_tsc, uint64 end_tsc,
    uint32_t domain, uint32_t cbid,
    const void *cbdata) {
  switch (cbid) {
    //case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    //  AddKernelEventUponApiExit(collector, device_id, cbdata, start_tsc,
    //                            end_tsc);
    //  break;
    //case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
    //case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
    //  AddNormalMemcpyEventUponApiExit(collector, device_id, cbid, cbdata,
    //                                  start_tsc, end_tsc);
    //  break;
    //case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer:
    //case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync:
    //  AddP2PMemcpyEventUponApiExit(collector, device_id, cbid,
    //                               cbdata, start_tsc, end_tsc);
    //  break;
    //case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2:
    //  AddCudaMallocEventUponApiExit(collector, device_id, cbid, cbdata,
    //                                start_tsc, end_tsc);
    //  break;
    default:
      AddGenericEventUponApiExit(collector, device_id, cbid, cbdata, start_tsc,
                                 end_tsc);
      break;
  }
  return Status::OK();
}

const char *GetTraceEventTypeName(const RocmTracerEventType &type) {
  switch (type) {
    case RocmTracerEventType::MemcpyH2D:
      return "MemcpyH2D";
    case RocmTracerEventType::MemcpyD2H:
      return "MemcpyD2H";
    case RocmTracerEventType::MemcpyD2D:
      return "MemcpyD2D";
    case RocmTracerEventType::MemcpyP2P:
      return "MemcpyP2P";
    case RocmTracerEventType::MemcpyOther:
      return "MemcpyOther";
    case RocmTracerEventType::Kernel:
      return "Compute";
    case RocmTracerEventType::MemoryAlloc:
      return "MemoryAlloc";
    case RocmTracerEventType::Overhead:
      return "Overhead";
    case RocmTracerEventType::UnifiedMemory:
      return "UnifiedMemory";
    case RocmTracerEventType::Generic:
      return "Generic";
    default:
      DCHECK(false);
      return "";
  }
}

void AnnotationMap::Add(uint32 device_id, uint32 correlation_id,
                        const std::string &annotation) {
  if (annotation.empty()) return;
  VLOG(3) << "Add annotation: device_id: " << device_id
          << " correlation_id: " << correlation_id
          << " annotation: " << annotation;
  if (device_id >= per_device_map_.size()) return;
  auto &per_device_map = per_device_map_[device_id];
  absl::MutexLock lock(&per_device_map.mutex);
  if (per_device_map.annotations.size() < max_size_) {
    absl::string_view annotation_str =
        *per_device_map.annotations.insert(annotation).first;
    per_device_map.correlation_map.emplace(correlation_id, annotation_str);
  }
}

absl::string_view AnnotationMap::LookUp(uint32 device_id,
                                        uint32 correlation_id) {
  if (device_id >= per_device_map_.size()) return absl::string_view();
  auto &per_device_map = per_device_map_[device_id];
  absl::MutexLock lock(&per_device_map.mutex);
  auto it = per_device_map.correlation_map.find(correlation_id);
  return it != per_device_map.correlation_map.end() ? it->second
                                                    : absl::string_view();
}

/* static */ RocmTracer *RocmTracer::GetRocmTracerSingleton() {
  static auto *singleton = new RocmTracer();
  return singleton;
}

bool RocmTracer::IsAvailable() const {
  return !activity_tracing_enabled_ && !api_tracing_enabled_;
}

int RocmTracer::NumGpus() {
  static int num_gpus = []() -> int {
    if (hipInit(0) != hipSuccess) {
      return 0;
    }
    int gpu_count;
    if (hipGetDeviceCount(&gpu_count) != hipSuccess) {
      return 0;
    }
    LOG(INFO) << "Profiler found " << gpu_count << " GPUs";
    return gpu_count;
  }();
  return num_gpus;
}

void RocmTracer::Enable(const RocmTracerOptions &option,
                        RocmTraceCollector *collector) {
  option_ = option;
  collector_ = collector;
  annotation_map_.emplace(option.max_annotation_strings, NumGpus());

  if (option_->enable_event_based_activity) {
    option_->enable_activity_api = false;
    roctracer_driver_api_hook_.reset(new RocmDriverApiHookWithHipEvent(
        option, collector, &*annotation_map_));
  } else {
    roctracer_driver_api_hook_.reset(new RocmDriverApiHookWithActivityApi(
        option, collector, &*annotation_map_));
  }

  EnableApiTracing().IgnoreError();
  if (option_->enable_activity_api) {
    EnableActivityTracing().IgnoreError();
  }
}

void RocmTracer::Disable() {
  DisableApiTracing().IgnoreError();
  if (option_->enable_activity_api) {
    DisableActivityTracing().IgnoreError();
  }
  Finalize().IgnoreError();
  roctracer_driver_api_hook_->Flush().IgnoreError();
  collector_->Flush();
  collector_ = nullptr;
  option_.reset();
  roctracer_driver_api_hook_.reset();
  annotation_map_.reset();
}

Status RocmTracer::EnableApiTracing() {
  if (api_tracing_enabled_) return Status::OK();
  api_tracing_enabled_ = true;

  // ROCM TODO: do more selected filtering later on.
  //if (!option_->cbids_selected.empty()) {
  //  for (auto cbid : option_->cbids_selected) {
  //    // ROCM TODO: determine the best domain(s) to monitor.
  //    RETURN_IF_ROCTRACER_ERROR(roctracer_enable_op_callback(
  //        ACTIVITY_DOMAIN_HIP_API, cbid, ApiCallback, this));
  //  }
  //} else {  // select all callback ids.
    RETURN_IF_ROCTRACER_ERROR(roctracer_enable_callback(
        ApiCallback, this));
  //}
  return Status::OK();
}

Status RocmTracer::DisableApiTracing() {
  if (!api_tracing_enabled_) return Status::OK();

  api_tracing_enabled_ = false;

  if (!option_->cbids_selected.empty()) {
    for (auto cbid : option_->cbids_selected) {
      RETURN_IF_ROCTRACER_ERROR(roctracer_disable_op_callback(
          ACTIVITY_DOMAIN_HIP_API, cbid));
    }
  } else {
    RETURN_IF_ROCTRACER_ERROR(roctracer_disable_callback());
  }

  return Status::OK();
}

Status RocmTracer::EnableActivityTracing() {
  if (!option_->activities_selected.empty()) {
    // Initialize callback functions for Rocm Activity API.
    VLOG(1) << "Registering roctracer activity callbacks";

    // Creating tracer pool.
    roctracer_properties_t properties{};
    properties.buffer_size = 0x1000;
    properties.buffer_callback_fun = ActivityCallback;
    properties.buffer_callback_arg = this;
    if (roctracer_default_pool() == NULL)
      RETURN_IF_ROCTRACER_ERROR(roctracer_open_pool(&properties));

    VLOG(1) << "Enabling activity tracing for "
            << option_->activities_selected.size() << " activities";

    for (auto activity : option_->activities_selected) {
      VLOG(1) << "Enabling activity tracing for: " << activity;
      RETURN_IF_ROCTRACER_ERROR(roctracer_enable_domain_activity(activity));
    }
    //RETURN_IF_ROCTRACER_ERROR(roctracer_enable_activity());
  }
  activity_tracing_enabled_ = true;
  return Status::OK();
}

Status RocmTracer::DisableActivityTracing() {
  if (activity_tracing_enabled_) {
    VLOG(1) << "Disabling activity tracing for "
            << option_->activities_selected.size() << " activities";
    for (auto activity : option_->activities_selected) {
      VLOG(1) << "Disabling activity tracing for: " << activity;
      RETURN_IF_ROCTRACER_ERROR(roctracer_disable_domain_activity(activity));
    }
    //RETURN_IF_ROCTRACER_ERROR(roctracer_disable_activity());
    option_->activities_selected.clear();

    VLOG(1) << "Flushing roctracer activity buffer";
    RETURN_IF_ROCTRACER_ERROR(roctracer_flush_activity());
    LOG(INFO) << "roctracer activity buffer flushed";
  }
  activity_tracing_enabled_ = false;
  return Status::OK();
}

Status RocmTracer::Finalize() {
  return Status::OK();
}

/*static*/ uint64 RocmTracer::GetTimestamp() {
  uint64_t tsc;
  // ROCM TODO: revise with HIP or ROCR API
  //RocmInterface *cupti_interface = GetRocmInterface();
  //if (cupti_interface && cupti_interface->GetTimestamp(&tsc) == CUPTI_SUCCESS) {
  //  return tsc;
  //}
  // Return 0 on error. If an activity timestamp is 0, the activity will be
  // dropped during time normalization.
  return 0;
}

Status RocmTracer::HandleCallback(uint32_t domain, uint32_t cbid,
                                  const void *cbdata) {
  if (!api_tracing_enabled_) return Status::OK();  // already unsubscribed.
  if (domain != ACTIVITY_DOMAIN_HIP_API) return Status::OK();
  if (internalRocmCall) return Status::OK();

  // ROCM TODO: revise this.
  //if (cbdata->context == nullptr) {
  //  // API callback is called before any CUDA context is created.
  //  // This is expected to be rare, and we ignore this case.
  //  VLOG(3) << "API callback received before creation of CUDA context\n";
  //  return errors::Internal("cutpi callback without context");
  //}

  // ROCM TODO: revise this.
  // Grab a correct device ID.
  uint32 device_id = -1;
  //RETURN_IF_ROCTRACER_ERROR(
  //    cupti_interface_->GetDeviceId(cbdata->context, &device_id));
  //if (device_id >= num_gpus_) {
  //  return errors::Internal(absl::StrCat("Invalid device id:", device_id));
  //}

  const hip_api_data_t *data = reinterpret_cast<const hip_api_data_t*>(cbdata);
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    TF_RETURN_IF_ERROR(roctracer_driver_api_hook_->OnDriverApiEnter(
        device_id, domain, cbid, cbdata));
  } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
    // Set up the map from correlation id to annotation string.
    const std::string &annotation = tensorflow::Annotation::CurrentAnnotation();
    if (!annotation.empty()) {
      annotation_map_->Add(device_id, data->correlation_id, annotation);
    }

    TF_RETURN_IF_ERROR(roctracer_driver_api_hook_->OnDriverApiExit(
        device_id, domain, cbid, cbdata));
  }
  return Status::OK();
}

Status RocmTracer::ProcessActivityRecord(const char *begin, const char *end) {
  if (!activity_tracing_enabled_) {
    LOG(WARNING) << "roctracer activity buffer is freed after flush.";
    return Status::OK();
  }

  const roctracer_record_t *record = reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t *end_record = reinterpret_cast<const roctracer_record_t*>(end);
  while (record < end_record) {
  //    switch (record->kind) {
  //      case CUPTI_ACTIVITY_KIND_KERNEL:
  //        AddKernelActivityEvent(
  //            collector_, &*annotation_map_,
  //            reinterpret_cast<CUpti_ActivityKernel4 *>(record));
  //        break;
  //      case CUPTI_ACTIVITY_KIND_MEMCPY:
  //        AddMemcpyActivityEvent(
  //            collector_, &*annotation_map_,
  //            reinterpret_cast<CUpti_ActivityMemcpy *>(record));
  //        break;
  //      case CUPTI_ACTIVITY_KIND_OVERHEAD:
  //        AddRocmOverheadActivityEvent(
  //            collector_, reinterpret_cast<CUpti_ActivityOverhead *>(record));
  //        break;
  //      default:
  //        LOG(ERROR) << "Activity type " << record->kind << " not supported.";
  //        break;
  //    }

    RETURN_IF_ROCTRACER_ERROR(static_cast<roctracer_status_t>(roctracer_next_record(record, &record)));
  }
  return Status::OK();
}

}  // namespace profiler
}  // namespace tensorflow
