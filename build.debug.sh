#!/bin/bash

INSTALL_PATH=_install
BUILD_PATH=_build/Debug
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
COMMON_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DOPT_PYTHON=1 -DOPT_TESTING=1 -DOPT_DEBUG=1"
rm -rf $INSTALL_PATH $BUILD_PATH
set -e

cmake -H. -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug ${COMMON_ARGS} $@
VERBOSE=1 cmake --build $BUILD_PATH --target install -- -j${NUM_PROCS}
