#!/bin/bash

INSTALL_PATH=_install
BUILD_PATH=_build/Debug

#rm -rf $INSTALL_PATH $BUILD_PATH
rm -rf $BUILD_PATH
set -e

cmake -Wdev -H. -B$BUILD_PATH -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH -DCMAKE_BUILD_TYPE=Debug -DOPT_PYTHON=1 $@
VERBOSE=1 cmake -Wdev --build $BUILD_PATH --target install -- -j8
