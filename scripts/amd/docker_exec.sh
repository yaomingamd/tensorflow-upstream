set -ex

# configure tensorflow for ROCM
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
	/usertools/rename_and_verify_ROCM_wheels.sh
