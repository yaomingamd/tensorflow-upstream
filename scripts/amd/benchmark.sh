ROOT_DIR=$(pwd)
DEFAULT_LOG_DIR=$ROOT_DIR/$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

# export NCCL_DEBUG=INFO
# # export NCCL_SHM_DISABLE=1
# # export NCCL_IB_HCA=mlx5
# # export NCCL_P2P_LEVEL=5
# export HSA_FORCE_FINE_GRAIN_PCIE=1
# # export NCCL_DEBUG_SUBSYS=ALL
# # export NCCL_SOCKET_IFNAME=ib1

env_vars=""

options=""

options="$options --model=resnet50_v1.5"
options="$options --xla_compile"
options="$options --use_fp16"
options="$options --batch_size=256"
options="$options --print_training_accuracy"
options="$options --num_batches=1"
options="$options --num_gpus=8"
options="$options --variable_update=replicated"
options="$options --all_reduce_spec=nccl"

env_vars="$env_vars NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,INIT"

export $env_vars

cd /dockerx/benchmarks
# python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --help
# export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=$LOG_DIR" 
python3 scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $options 2>&1 | tee $LOG_DIR/tf_cnn_benchmarks.log