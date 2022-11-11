set -ex

# configure tensorflow for ROCM
docker exec tf \
	sed -i "s/build:rbe_linux_rocm_base --action_env=TF_ROCM_CONFIG_REPO=\"@ubuntu20.04-gcc9_manylinux2014-rocm_config_rocm\"/build:rbe_linux_rocm_base --action_env=TF_ROCM_CONFIG_REPO=\"@ubuntu20.04-gcc9_manylinux2014-rocm_config_rocm\"\nbuild:rbe_linux_rocm_base --action_env=TF_ROCM_GCC=1/" \
	/tf/tensorflow/.bazelrc

# docker exec tf \
# 	cat /tf/tensorflow/.bazelrc

docker exec tf \
	echo $TF_NEED_ROCM

docker exec tf \
	echo $ROCM_PATH

docker exec tf \
	echo $PYTHON_BIN_PATH

# docker exec tf \
# 	ls /tf/tensorflow

# docker exec tf \
# 	"TF_NEED_ROCM=1 ROCM_TOOLKIT_PATH=${ROCM_PATH} PYTHON_BIN_PATH=${PYTHON_BIN_PATH} bash /tf/tensorflow/configure"

docker exec tf \
	/tf/tensorflow/configure

# build pip package
docker exec tf \
	bazel build --config=opt --config=rocm \
	--action_env TF_ROCM_GCC=1 \
	tensorflow/tools/pip_package:build_pip_package --verbose_failures

# docker exec tf \
# 	./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
# 	/tf/pkg \
# 	--nightly_flag
