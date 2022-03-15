ROOT_DIR=$(pwd)
DEFAULT_LOG_DIR=$ROOT_DIR/$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"

python3 scripts/amd/visualize_graph.py $LOG_DIR/*.pbtxt

tensorboard --logdir $LOG_DIR