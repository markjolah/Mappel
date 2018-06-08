##!/bin/bash

INSTALL_PATH=_install
BUILD_PATH=_build/Debug
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`
COMMON_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
# rm -rf $INSTALL_PATH $BUILD_PATH

set -e
if [ ! -d $BUILD_PATH ]; then
    cmake -H. -B$BUILD_PATH -DCMAKE_BUILD_TYPE=Debug ${COMMON_ARGS}
fi
cmake --build $BUILD_PATH --target install -- -j${NUM_PROCS}
cd $BUILD_PATH && ctest -V -j${NUM_PROCS}
