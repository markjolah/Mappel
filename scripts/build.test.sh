##!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_PATH=${SCRIPT_DIR}/..
BUILD_PATH=${SRC_PATH}/_build/Test
INSTALL_PATH=${BUILD_PATH}/_install #If we need to install some dependencies, do it internally.  We are just testing.
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
ARGS=""
ARGS="${ARGS} -DBUILD_STATIC_LIBS=Off"
ARGS="${ARGS} -DBUILD_TESTING=On"

set -ex
rm -rf $BUILD_PATH
cmake -H${SRC_PATH} -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ${ARGS}
cmake --build $BUILD_PATH/test --target all -- -j${NUM_PROCS}
cmake --build $BUILD_PATH/test --target test -- -j${NUM_PROCS}
