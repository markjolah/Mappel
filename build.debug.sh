#!/bin/bash

INSTALL_PATH=_install
BUILD_PATH=_build/Debug
NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)
ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_TESTING=Off"
ARGS="${ARGS} -DOPT_DEBUG=On"
ARGS="${ARGS} -DOPT_MATLAB=Off"
ARGS="${ARGS} -DOPT_PYTHON=Off"

set -ex
rm -rf $BUILD_PATH $INSTALL_PATH
cmake -H. -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug ${ARGS} $@
VERBOSE=1 cmake --build $BUILD_PATH --target install -- -j${NUM_PROCS}
