# Tensorflow ROCm port: Basic installation

## Intro

This instruction provides a starting point for TensorFlow ROCm port (mostly via rpm packages).
*Note*: it is recommended to start with a clean RHEL 8.3 system

## Install ROCm

```
# ROCm repo location
# FIXME: update repo location
#export RPM_ROCM_REPO=http://repo.radeon.com/rocm/yum/3.7
export RPM_ROCM_REPO=http://compute-artifactory.amd.com/artifactory/rocm-osdb-centos-8.1/compute-rocm-dkms-no-npi-hipclang-3333

# Enable extra repositories
yum --enablerepo=extras install -y epel-release

# Install required base build and packaging commands for ROCm
yum -y install \
    bc \
    cmake \
    cmake3 \
    dkms \
    dpkg \
    elfutils-libelf-devel \
    expect \
    file \
    gettext \
    gcc-c++ \
    git \
    libgcc \
    ncurses \
    ncurses-base \
    ncurses-libs \
    numactl-devel \
    numactl-libs \
    libssh \
    libunwind-devel \
    libunwind \
    llvm \
    llvm-libs \
    make \
    openssl \
    openssl-libs \
    openssh \
    openssh-clients \
    pciutils \
    pciutils-devel \
    pciutils-libs \
    python36 \
    python36-devel \
    pkgconfig \
    qemu-kvm \
    rpm \
    rpm-build \
    subversion \
    wget

# Add the ROCm package repo location
echo -e "[ROCm]\nname=ROCm\nbaseurl=$RPM_ROCM_REPO\nenabled=1\ngpgcheck=0" >> /etc/yum.repos.d/rocm.repo

# Install the ROCm rpms
sudo yum clean all
sudo yum install -y rocm-dev
sudo yum install -y miopen-hip miopengemm rocblas rocrand rocfft hipblas rocprim hipcub rccl

# Ensure the ROCm target list is set up
bash -c 'echo -e "gfx803\ngfx900\ngfx906\ngfx908" >> $ROCM_PATH/bin/target.lst'

# Install Python & dependencies
pip3.6 install --user \
    cget \
    pyyaml \
                pip \
    setuptools==39.1.0 \
                virtualenv \
                absl-py \
                six==1.10.0 \
                protobuf==3.6.1 \
                numpy==1.18.2 \
                scipy==1.4.1 \
                scikit-learn==0.19.1 \
                pandas==0.19.2 \
                gnureadline \
                bz2file \
                wheel==0.29.0 \
                portpicker \
                werkzeug \
                grpcio \
                astor \
                gast \
                termcolor \
                h5py==2.8.0 \
                keras_preprocessing==1.0.5

# Install ROCm manylinux2010 WHL 
wget <location of WHL file>
pip3.6 ./tensorflow*linux_x86_64.whl

## Install Tensorflow Community Supported Builds 

Link to the upstream Tensorflow CSB doc:
<https://github.com/tensorflow/tensorflow#community-supported-builds>

We provide nightly tensorflow-rocm whl packages for Python 2.7, 3.5, 3.6 and 3.7 based systems.
After downloading the compatible whl package, you can use pip/pip3 to install.

For example, the following commands can be used to download and install the tensorflow-rocm nightly CSB package on an Ubuntu 16.04 system previously configured with ROCm2.8 and Python3.5:
```
wget http://ml-ci.amd.com:21096/job/tensorflow-rocm-nightly/lastSuccessfulBuild/artifact/pip35_test/whl/tensorflow_rocm-2.0.0-cp35-cp35m-manylinux1_x86_64.whl
pip3 install --user tensorflow_rocm-2.0.0-cp35-cp35m-manylinux1_x86_64.whl
```

## Install TensorFlow ROCm release build

Uninstall any previously-installed tensorflow whl packages:  
```
pip list | grep tensorflow && pip uninstall -y tensorflow
```

We maintain `tensorflow-rocm` whl packages on PyPI [here](https://pypi.org/project/tensorflow-rocm).

For Python 2-based systems:
```
# Pip install the whl package from PyPI
pip install --user tensorflow-rocm --upgrade
```

For Python 3-based systems:
```
# Pip3 install the whl package from PyPI
pip3 install --user tensorflow-rocm --upgrade
```
