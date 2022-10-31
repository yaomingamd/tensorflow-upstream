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

ROCM_VERSION=$1

export AMDGPU_REPO=https://repo.radeon.com/amdgpu/${ROCM_VERSION}/rhel/7.9/main/x86_64
export ROCM_REPO=https://repo.radeon.com/rocm/yum/${ROCM_VERSION}/main
export ROCM_PATH=/opt/rocm-${ROCM_VERSION}.0
export GPU_DEVICE_TARGETS="gfx900 gfx906 gfx908 gfx90a gfx1030"

echo $ROCM_VERSION
echo $ROCM_REPO
echo $ROCM_PATH

#Enable EPEL
yum install -y epel-release

yum install -y \
        wget \
        swig \
        libtool \
        automake \
        autoconf \
        make \
        curl \
        unzip \
        zip \
        pkg-config \
        perl-File-BaseDir \
        perl-URI-Encode \
        libffi-devel \
        xz-devel \
        ncurses-devel \
        readline-devel \
        sqlite3-devel \
        openssl-devel \
        libxml2-devel \
        llvm-toolset-7 \
        rsync \
        tk-devel \
        wget \
        zlib-devel \
        git \
        which \
        rocm-dev  \
        rocm-libs \
        rccl 
        #miopenkernels*


#clang-8
#clang-format-12
#colordiff
#ffmpeg
#gdb
#jq
#less
#libcurl3-dev
#libcurl4-openssl-dev
#libfreetype6-dev
#libhdf5-serial-dev
#libtool
#libzmq3-dev
#mlocate
#moreutils
#openjdk-11-jdk
#openjdk-11-jre-headless
#patchelf
#python3-dev
#python3-setuptools
#software-properties-common
#sudo
#swig
#vim
#zlib1g-dev

# Add target file to help determine which device(s) to build for
printf '%s\n' > ${ROCM_PATH}/bin/target.lst ${GPU_DEVICE_TARGETS}
touch ${ROCM_PATH}/.info/version
