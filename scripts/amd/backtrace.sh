ROOT_DIR=$(pwd)
DEFAULT_LOG_DIR=$ROOT_DIR/$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

sudo apt install gdb -y
pip3 install absl-py


gdb -ex "set pagination off" \
    -ex "file python3" \
    -ex "run scripts/amd/rccl_script.py" \
    -ex "backtrace" \
    -ex "set confirm off" \
    -ex "q" \
    2>&1 | tee ${LOG_DIR}/rccl_script.log

# gdb -ex "set pagination off" \
#     -ex "break" \
#     -ex "file python3" \
#     -ex "run scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py ${options}" \
#     2>&1 | tee ${LOG_DIR}/tf_cnn_benchmarks_gdb.log
