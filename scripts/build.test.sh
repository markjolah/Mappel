##!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_PATH=${SCRIPT_DIR}/..
INSTALL_PATH=${SRC_PATH}/_install
BUILD_PATH=${SRC_PATH}/_build/Debug
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
ARGS=""
ARGS="${ARGS} -DBUILD_TESTING=On"

set -ex
rm -rf $INSTALL_PATH $BUILD_PATH
cmake -H${SRC_PATH} -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH ${ARGS}
cmake --build $BUILD_PATH/Debug --target install -- -j${NUM_PROCS}
ctest ${PWD} $BUILD_PATH/Debug -V -j${NUM_PROCS}
