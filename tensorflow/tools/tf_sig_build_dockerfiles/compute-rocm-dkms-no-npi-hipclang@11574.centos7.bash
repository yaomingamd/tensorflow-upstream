set -x

rocm_build=${0%%.*}

yum install -y http://artifactory-cdn.amd.com/artifactory/list/amdgpu-rpm/rhel/amd-nonfree-radeon-7-1.noarch.rpm
amdgpu-repo --rocm-build=${rocm_build//@/\/}

