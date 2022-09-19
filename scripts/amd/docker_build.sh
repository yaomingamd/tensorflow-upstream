cd tensorflow/tools/tf_sig_build_dockerfiles
docker build -f Dockerfile.rocm --build-arg ROCM_VERSION=5.2.0 --build-arg PYTHON_VERSION=3.10 -t tf_centos . \
	2>&1
# docker build -f Dockerfile.centos -t tf_centos .  2>&1 | tee $LOG_DIR/centos_build.log
