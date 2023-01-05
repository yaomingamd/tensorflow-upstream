#!/usr/bin/env bash
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# setup.rocm.sh: Prepare the ROCM installation on the container.
# Usage: setup.rocm.sh <ROCM_VERSION>
set -x

# # Add the ROCm package repo location
ROCM_VERSION=$1 # e.g. 5.2.0
ROCM_BUILD_NAME=ubuntu
ROCM_BUILD_NUM=main
ROCM_PATH=/opt/rocm-${ROCM_VERSION}
ROCM_VERSION_REPO=$(echo $ROCM_VERSION | grep -o "\w.\w") # e.g 5.2
ROCM_DEB_REPO=http://repo.radeon.com/rocm/apt/$(echo $ROCM_VERSION | grep -o "\w.\w")
#apt-get --allow-unauthenticated update && apt install -y wget software-properties-common
#wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -;
echo -e "deb [arch=amd64] $ROCM_DEB_REPO $ROCM_BUILD_NAME $ROCM_BUILD_NUM"
echo -e "deb [arch=amd64] $ROCM_DEB_REPO $ROCM_BUILD_NAME $ROCM_BUILD_NUM" > /etc/apt/sources.list.d/rocm.list 

# Use devtoolset env
export PATH=/dt-9/root/usr/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH:+:${PATH}}
export MANPATH=/dt-9/root/usr/share/man:${MANPATH}
export INFOPATH=/dt-9/root/usr/share/info${INFOPATH:+:${INFOPATH}}
export PCP_DIR=/dt-9/root
export PERL5LIB=/dt-9/root//usr/lib64/perl5/vendor_perl:/dt-9/root/usr/lib/perl5:/dt-9/root//usr/share/perl5/
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:/usr/local/lib:/opt/rh/devtoolset-10/root$rpmlibdir$rpmlibdir32${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LDFLAGS="-Wl,-rpath=/dt-9/root/usr/lib64 -Wl,-rpath=/dt-9/root/usr/lib"
GPU_DEVICE_TARGETS="gfx900 gfx906 gfx908 gfx90a gfx1030"

echo $ROCM_VERSION
echo $ROCM_REPO
echo $ROCM_PATH

# install rocm
/setup.packages.sh /devel.packages.rocm.ub.txt

# Ensure the ROCm target list is set up
bash -c "echo -e 'gfx900\ngfx906\ngfx908\ngfx90a\ngfx1030' >> $ROCM_PATH/bin/target.lst"
touch ${ROCM_PATH}/.info/version
