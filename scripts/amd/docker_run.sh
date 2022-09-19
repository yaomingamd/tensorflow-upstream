docker stop tf
docker rm tf
docker run --name tf -w /tf/tensorflow -it -d --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "/tmp/packages:/tf/pkg" \
  -v "/home/fpadmin/dockerx/tensorflow-upstream:/tf/tensorflow" \
  -v "/tmp/bazelcache:/tf/cache" \
  tf_centos \
  bash
