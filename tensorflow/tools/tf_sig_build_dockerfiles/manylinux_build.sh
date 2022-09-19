set -x

# set environment vars
export BUILD_DIR=/tmp

# set vars
MAYBE_NO_CACHE=--no-cache
TF_REL=$1         # nightly or 2.8
PYTHON_VERSION=$2 # 3.6-3.10
ROCM_REL=$3       # 5.2.0

# create intermediate vars
ROCM_REL_REPO=$(echo $ROCM_REL | grep -o "\w.\w") # 5.2
DOCKER_IMAGE=rocm/tensorflow-private:manylinux2014-rocm${ROCM_REL}-tf${TF_REL}
RPM_ROCM_REPO=http://repo.radeon.com/rocm/yum/${ROCM_REL_REPO}/main
ROCM_PATH=/opt/rocm-${ROCM_REL}
TF_VERSION=$TF_REL

# Enable extra repositories
yum update -y
yum --enablerepo=extras install -y epel-release #devtoolset-10
yum install -y https://repo.ius.io/7/x86_64/packages/i/ius-release-2-1.el7.ius.noarch.rpm
yum install -y openssl-devel libffi-devel centos-release-scl wget hdf5-devel

# PYTHON
if [ "$PYTHON_VERSION" = "3.6" ]; then
    yum install -y rh-python36

    scl enable devtoolset-10 rh-python36 'bash'
    rm -f /usr/bin/ld && ln -s /opt/rh/devtoolset-10/root/usr/bin/ld /usr/bin/ld

    export PYTHON_LIB_PATH=/opt/rh/rh-python36/root/usr/lib/python3.6/site-packages
    export PYTHON_BIN_PATH=/opt/rh/rh-python36/root/usr/bin/python3.6

    ln -sf /opt/rh/rh-python36/root/usr/bin/python3.6 /usr/bin/python3 && ln -sf /opt/rh/rh-python36/root/usr/bin/pip3.6 /usr/bin/pip3

    NUMPY_VERSION=1.18.5
elif [ "$PYTHON_VERSION" = "3.7" ]; then
    wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz && tar xvf Python-3.7.11.tgz && cd Python-3.7*/ && ./configure --enable-optimizations && make altinstall
    ln -sf /usr/local/bin/python3.7 /usr/bin/python3 && ln -sf /usr/local/bin/pip3.7 /usr/bin/pip3

    scl enable devtoolset-10 'bash'
    rm -f /usr/bin/ld && ln -s /opt/rh/devtoolset-10/root/usr/bin/ld /usr/bin/ld

    export PYTHON_LIB_PATH=/usr/local/lib/python3.7/site-packages
    export PYTHON_BIN_PATH=/usr/local/bin/python3.7

    NUMPY_VERSION=1.18.5
elif [ "$PYTHON_VERSION" = "3.8" ]; then
    wget https://www.python.org/ftp/python/3.8.9/Python-3.8.9.tgz && tar xvf Python-3.8.9.tgz && cd Python-3.8*/ && ./configure --enable-optimizations && make altinstall
    ln -sf /usr/local/bin/python3.8 /usr/bin/python3 && ln -sf /usr/local/bin/pip3.8 /usr/bin/pip3

    scl enable devtoolset-10 'bash'
    rm -f /usr/bin/ld && ln -s /opt/rh/devtoolset-10/root/usr/bin/ld /usr/bin/ld

    export PYTHON_LIB_PATH=/usr/local/lib/python3.8/site-packages
    export PYTHON_BIN_PATH=/usr/local/bin/python3.8

    NUMPY_VERSION=1.18.5
elif [ "$PYTHON_VERSION" = "3.9" ]; then
    wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz && tar xvf Python-3.9.7.tgz && cd Python-3.9*/ && ./configure --enable-optimizations && make altinstall
    ln -sf /usr/local/bin/python3.9 /usr/bin/python3 && ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3

    scl enable devtoolset-10 'bash'
    rm -f /usr/bin/ld && ln -s /opt/rh/devtoolset-10/root/usr/bin/ld /usr/bin/ld

    export PYTHON_LIB_PATH=/usr/local/lib/python3.9/site-packages
    export PYTHON_BIN_PATH=/usr/local/bin/python3.9

    NUMPY_VERSION=1.20.3
elif [ "$PYTHON_VERSION" = "3.10" ]; then
    #install openssl1.1.1
    wget --no-check-certificate https://ftp.openssl.org/source/openssl-1.1.1k.tar.gz && tar xvf openssl-1.1.1k.tar.gz && cd openssl-1.1.1k &&
        ./config --prefix=/usr --openssldir=/etc/ssl --libdir=lib no-shared zlib-dynamic && make && make install

    wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz && tar xvf Python-3.10.2.tgz && cd Python-3.10*/ &&
        sed -i 's/PKG_CONFIG openssl /PKG_CONFIG openssl11 /g' configure && ./configure --enable-optimizations && make altinstall
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3

    scl enable devtoolset-10 'bash'
    rm -f /usr/bin/ld && ln -s /opt/rh/devtoolset-10/root/usr/bin/ld /usr/bin/ld

    export PYTHON_LIB_PATH=/usr/local/lib/python3.10/site-packages
    export PYTHON_BIN_PATH=/usr/local/bin/python3.10

    NUMPY_VERSION=1.21.4
else
    printf '%s\n' "Python Version not Supported" >&2
    exit 1
fi

# Pip version required by manylinux2014
pip3 install --upgrade pip
PYTHON_PACKAGES="six numpy==${NUMPY_VERSION} scipy wheel argparse keras_applications keras_preprocessing tqdm Pillow portpicker h5py==2.10.0 wheel scikit-learn packaging requests"
if [ "$TF_VERSION" == "2.10" ]; then
    pip3 --no-cache-dir install ${PYTHON_PACKAGES}
else
    pip3 --no-cache-dir install ${PYTHON_PACKAGES}
fi

# Install required base build and packaging commands for ROCm
yum install -y \
    bc \
    bridge-utils \
    cmake \
    cmake3 \
    devscripts \
    dkms \
    doxygen \
    dpkg \
    dpkg-dev \
    dpkg-perl \
    elfutils-libelf-devel \
    expect \
    file \
    gettext \
    gcc-c++ \
    git \
    libgcc \
    libcxx-devel \
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
    java-11-openjdk-devel \
    pkgconfig \
    pth \
    qemu-kvm \
    re2c \
    rpm \
    rpm-build \
    subversion \
    wget

# Use devtoolset env
export PATH=/opt/rh/devtoolset-10/root/usr/bin:${ROCM_PATH}/llvm/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH:+:${PATH}}
export MANPATH=/opt/rh/devtoolset-10/root/usr/share/man:${MANPATH}
export INFOPATH=/opt/rh/devtoolset-10/root/usr/share/info${INFOPATH:+:${INFOPATH}}
export PCP_DIR=/opt/rh/devtoolset-10/root
export PERL5LIB=/opt/rh/devtoolset-10/root//usr/lib64/perl5/vendor_perl:/opt/rh/devtoolset-10/root/usr/lib/perl5:/opt/rh/devtoolset-10/root//usr/share/perl5/
export LD_LIBRARY_PATH=${ROCM_PATH}/lib:/usr/local/lib:/opt/rh/devtoolset-10/root$rpmlibdir$rpmlibdir32${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LDFLAGS="-Wl,-rpath=/opt/rh/devtoolset-10/root/usr/lib64 -Wl,-rpath=/opt/rh/devtoolset-10/root/usr/lib"

# Add the ROCm package repo location
echo -e "[ROCm]\nname=ROCm\nbaseurl=$RPM_ROCM_REPO\nenabled=1\ngpgcheck=0" >>/etc/yum.repos.d/rocm.repo
echo -e "[amdgpu]\nname=amdgpu\nbaseurl=https://repo.radeon.com/amdgpu/latest/rhel/7.9/main/x86_64/\nenabled=1\ngpgcheck=0" >>/etc/yum.repos.d/amdgpu.repo

# Install the ROCm rpms
yum clean all
yum install -y libdrm-amdgpu
yum install -y rocm-dev
yum install -y miopen-hip miopen-hip-devel miopengemm rocblas rocblas-devel rocsolver-devel rocrand-devel rocfft-devel hipfft-devel hipblas-devel rocprim-devel hipcub-devel rccl-devel hipsparse-devel hipsolver-devel

# Ensure the ROCm target list is set up
bash -c 'echo -e "gfx900\ngfx906\ngfx908\ngfx90a\ngfx1030" >> $ROCM_PATH/bin/target.lst'

#Clone and install Tensorflow with rocm
cd $BUILD_DIR
if [ "$TF_VERSION" = "nightly" ]; then
    export TF_ROCM_GCC=1
    git clone https://github.com/ROCmSoftwarePlatform/tensorflow-upstream tensorflow
    # build:rbe_linux_rocm_base --action_env=TF_ROCM_GCC=1
    REF_LINE='build:rbe_linux_rocm_base --action_env=TF_ROCM_CONFIG_REPO="@ubuntu20.04-gcc9_manylinux2014-rocm_config_rocm"'
    sed -i "s/$REF_LINE/$REF_LINE\nbuild:rbe_linux_rocm_base --action_env=TF_ROCM_GCC=1/" $BUILD_DIR/tensorflow/.bazelrc
    cat $BUILD_DIR/tensorflow/.bazelrc
else
    git clone --branch r${TF_VERSION}-rocm-enhanced https://github.com/ROCmSoftwarePlatform/tensorflow-upstream tensorflow
fi

# Install Bazel
BAZEL_VERSION=$(cat $BUILD_DIR/tensorflow/.bazelversion) # get the right version of bazel
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh &&
    chmod -x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

# export env variables
export ROCM_PATH=$ROCM_PATH
export GCC_HOST_COMPILER_PATH=/opt/rh/devtoolset-10/root/usr/bin/gcc
export TF_PKG_LOC=/tmp/tensorflow_pkg
export TF_CONFIGURE_IOS=0

# First positional argument (if any) specifies the ROCM_INSTALL_DIR
export ROCM_INSTALL_DIR=$ROCM_PATH
export ROCM_TOOLKIT_PATH=$ROCM_INSTALL_DIR

# build wheel
cd $BUILD_DIR/tensorflow && yes "" | TF_NEED_ROCM=1 ROCM_TOOLKIT_PATH=${ROCM_INSTALL_DIR} PYTHON_BIN_PATH=${PYTHON_BIN_PATH} ./configure &&
    bazel build --config=opt --action_env TF_ROCM_GCC=1 --config=rocm //tensorflow/tools/pip_package:build_pip_package --verbose_failures &&
    bazel-bin/tensorflow/tools/pip_package/build_pip_package $TF_PKG_LOC --rocm --project_name tensorflow_rocm

# check that wheels are manylinux compatiable
echo "Checking $TF_WHEEL..."
pip3 install auditwheel
TF_WHEEL=$(ls -Art $TF_PKG_LOC/tensorflow*.whl | tail -n 1)
NEW_TF_WHEEL=${TF_WHEEL/linux/"manylinux2014"}
echo "rename whl to $NEW_TF_WHEEL"
mv $TF_WHEEL $NEW_TF_WHEEL
auditwheel show "$NEW_TF_WHEEL"
