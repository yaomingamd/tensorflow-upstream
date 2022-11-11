clear

set -x
set -e

ROOT_DIR=$(pwd)
DEFAULT_LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
# rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

bash scripts/amd/docker_build.sh 2>&1 | tee $LOG_DIR/docker_build.log
bash scripts/amd/docker_run.sh 2>&1 | tee $LOG_DIR/docker_run.log
bash scripts/amd/docker_exec.sh 2>&1 | tee $LOG_DIR/docker_exec.log
