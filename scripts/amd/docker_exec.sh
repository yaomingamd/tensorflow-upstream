set -ex

docker exec -e REF_LINE='build:rbe_linux_rocm_base --action_env=TF_ROCM_CONFIG_REPO="@ubuntu20.04-gcc9_manylinux2014-rocm_config_rocm"' \
	tf \
	sed -i "s/$REF_LINE/$REF_LINE\nbuild:rbe_linux_rocm_base --action_env=TF_ROCM_GCC=1/" /tf/tensorflow/.bazelrc

docker exec tf \
	./configure

# bazel build --config=opt --action_env TF_ROCM_GCC=1 --config=rocm //tensorflow/tools/pip_package:build_pip_package --verbose_failures
docker exec tf \
	bazel build --config=opt --config=rocm \
	--action_env TF_ROCM_GCC=1 \
	tensorflow/tools/pip_package:build_pip_package --verbose_failures

docker exec tf \
	./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
	/tf/pkg \
	--nightly_flag
