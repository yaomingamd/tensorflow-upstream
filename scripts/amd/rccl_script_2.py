import tensorflow as tf

from tensorflow.python.distribute import reduce_util, cross_device_utils, collective_util, cross_device_ops
from tensorflow.python.framework import indexed_slices, ops
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.ops import array_ops

CollectiveReplicaLauncher = cross_device_utils.CollectiveReplicaLauncher
CommunicationImplementation = collective_util.CommunicationImplementation
ReduceOp = reduce_util.ReduceOp
IndexedSlicesValue = indexed_slices.IndexedSlicesValue
IndexedSlices = indexed_slices.IndexedSlices


def as_list(value):
    if isinstance(value, ops.Tensor):
        return [value]
    elif isinstance(value, IndexedSlices):
        return [value]
    elif isinstance(value, value_lib.Mirrored):
        return value.values
    else:
        raise ValueError(
            "unwrap: unsupported input type: %s" % type(value))


def make_per_replica_value(value, devices):
    """Creates a `PerReplica` object whose values reside in `devices`.

    Args:
      value: a tensor-convertible value or a `IndexedSlicesValue`, or a callable
        that takes one argument (`device_idx`) and should return the value that is
        going to be created on devices[device_idx].
      devices: a list of device strings to create `PerReplica` values on.

    Returns:
      A `PerReplica` object.
    """
    values = []
    for device_idx, device in enumerate(devices):
        if callable(value):
            v = value(device_idx)
        elif isinstance(value, list):
            v = value[device_idx]
        else:
            v = value
        if isinstance(v, IndexedSlicesValue):
            with ops.device(device):
                values.append(
                    IndexedSlices(
                        values=array_ops.identity(v.values),
                        indices=array_ops.identity(v.indices),
                        dense_shape=array_ops.identity(v.dense_shape)))
        else:
            with ops.device(device):
                values.append(array_ops.identity(v))
    return value_lib.PerReplica(values)


def make_collective(num_processes, gpu_per_process):
    """Returns collectives and other info to be used in tests.

    Args:
      num_processes: an integer indicating the number of processes that
        participate in the collective.
      gpu_per_process: number of GPUs (0 if no GPUs) used by each process.

    Returns:
     A tuple of (collective, devices, pid) where collective is a instance
     of `CollectiveAllReduce`, devices are a list of local devices (str)
     attached to the current process, and pid is the id of this process among
     all participant processes.
    """

    # cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()  # not needed locally
    # task_id=cluster_resolver.task_id
    task_id = 0
    devices = [
        "/job:localhost/replica:0/task:%d/device:CPU:0" % task_id
    ]
    if gpu_per_process > 0:
        devices = [
            "/job:localhost/replica:0/task:%d/device:GPU:%d" %
            (task_id, i) for i in range(gpu_per_process)
        ]
    group_size = num_processes * len(devices)
    collective = cross_device_ops.CollectiveAllReduce(
        devices=devices,
        group_size=group_size,
        options=collective_util.Options())
    return collective, devices, task_id


num_processes = 1
reduce_op = ReduceOp.SUM
communication_options = collective_util.Options(
    implementation=CommunicationImplementation.NCCL)
gpus_per_process = 2

collective, devices, pid = make_collective(num_processes, gpus_per_process)
print(collective, devices, pid)

data_1 = tf.convert_to_tensor([[1.0, 2.0, 3.0, 4.0]])
data_2 = tf.convert_to_tensor([[9.0, 8.0, 7.0, 6.0]])
inputs = [data_1, data_2]

def reduce_fn():
    def value_fn(device_idx): return inputs[pid * len(devices) + device_idx]
    per_replica_value = make_per_replica_value(value_fn, devices)
    reduced_values = collective.reduce(reduce_op, per_replica_value,
                                       per_replica_value,
                                       communication_options)
    reduced_values = as_list(reduced_values)
    print(reduced_values)
    return [ops.convert_to_tensor(v) for v in reduced_values]


ans = reduce_fn()
print(ans)
