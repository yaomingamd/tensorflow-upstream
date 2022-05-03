# HIP LOGS
# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

# TF LOGS
# export TF_CPP_MAX_VLOG_LEVEL=3

# NCCL LOGS
export NCCL_DEBUG=INFO
# export NCCL_SHM_DISABLE=1
# export NCCL_IB_HCA=mlx5
# export NCCL_P2P_LEVEL=5
# export HSA_FORCE_FINE_GRAIN_PCIE=1
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_SOCKET_IFNAME=ib1

# pick at least 2 GPUS
export CUDA_VISIBLE_DEVICES=6,7
export HIP_VISIBLE_DEVICES=6,7

# log dir
ROOT_DIR=$(pwd)
BRANCH_NAME=$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
DEFAULT_LOG_DIR=$ROOT_DIR/log_${BRANCH_NAME}
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
mkdir -p $LOG_DIR/xla
mkdir -p $LOG_DIR/ckpts
chmod -R 777 $LOG_DIR

# run model
# pip3 install tensorflow_datasets ipywidgets
# export XLA_FLAGS="--xla_dump_hlo_as_text"
export XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=$LOG_DIR/xla"
python3 scripts/amd/rccl_xla_script.py --log_dir $LOG_DIR 2>&1 | tee $LOG_DIR/rccl_script.log

echo "NCCL KERNELS:"
cat $LOG_DIR/rccl_script.log | grep nccl

chmod -R 777 $LOG_DIR

# visualize graph
# bash scripts/amd/vis.sh
