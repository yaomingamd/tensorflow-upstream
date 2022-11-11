set -ex

# configure tensorflow for ROCM
docker exec tf \
	sed -i "s/build:rbe_linux_rocm_base --action_env=TF_ROCM_CONFIG_REPO=\"@ubuntu20.04-gcc9_manylinux2014-rocm_config_rocm\"/build:rbe_linux_rocm_base --action_env=TF_ROCM_CONFIG_REPO=\"@ubuntu20.04-gcc9_manylinux2014-rocm_config_rocm\"\nbuild:rbe_linux_rocm_base --action_env=TF_ROCM_GCC=1/" \
	/tf/tensorflow/.bazelrc

docker exec tf \
	/tf/tensorflow/configure

# build tensorflow
docker exec tf \
	bazel build --config=opt --config=rocm \
	--action_env TF_ROCM_GCC=1 \
	tensorflow/tools/pip_package:build_pip_package --verbose_failures

# build tensorflow pip wheel
docker exec tf \
	./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
	/tf/pkg \
	--rocm \
	--project_name tensorflow_rocm

# check wheel
docker exec tf \
	auditwheel show /tf/pkg/tensorflow*.whl
