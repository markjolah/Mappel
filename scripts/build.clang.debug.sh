##!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_PATH=${SCRIPT_DIR}/..
INSTALL_PATH=${SRC_DIR}/_install
BUILD_PATH=${SRC_DIR}/_build/ClangDebug
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
COMMON_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
rm -rf $INSTALL_PATH $BUILD_PATH

set -e
CC=/usr/lib64/llvm/7/bin/clang CXX=/usr/lib64/llvm/7/bin/clang++ cmake -H${SRC_DIR} -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug ${COMMON_ARGS}
VERBOSE=1 cmake --build $BUILD_PATH --target install -- -j${NUM_PROCS}
