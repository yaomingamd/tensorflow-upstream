#!/bin/bash -eu

VERSION="$1"
shift

mkdir /build
cd /build
wget "https://www.python.org/ftp/python/${VERSION}/Python-${VERSION}.tgz"
tar xvzf "Python-${VERSION}.tgz"
cd "Python-${VERSION}"
./configure --enable-optimizations "$@"
make altinstall

rm -rf /build
