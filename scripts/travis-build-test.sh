#!/bin/bash
# scripts/travis-build-test.sh <cmake args...>
#
# Continuous integration build and test script.
#
# Environment variables to be set:
#  BUILD_TYPE
#
PACKAGE_NAME=Mappel
if [ -z "$BUILD_TYPE" ]; then
    BUILD_TYPE="Release"
fi
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_PATH=${SCRIPT_DIR}/..
INSTALL_PATH=${SCRIPT_DIR}/../_travis.install
BUILD_PATH=${SCRIPT_DIR}/../_travis.build/${BUILD_TYPE}
NUM_PROCS=`grep -c ^processor /proc/cpuinfo`

ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=OFF"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=Off"
ARGS="${ARGS} -DOPT_AddExternalDependency_DISABLE_TESTING=On" #Disable recursive testing. we are just testing this package.
ARGS="${ARGS} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=On"  # Disable finding packages in the build-tree


set -ex
rm -rf $BUILD_PATH
cmake -H$SRC_PATH -B$BUILD_PATH -DCMAKE_BUILD_TYPE=$BUILD_TYPE $ARGS $@
VERBOSE=1 cmake --build $BUILD_PATH --target install -- -j$NUM_PROCS


if [ "$BUILD_TYPE" == "Debug" ]; then
    SUFFIX=".debug"
fi
#CTest in build directory
cmake --build $BUILD_PATH --target test -- -j$NUM_PROCS
#Run test in install directory
${INSTALL_PATH}/lib/${PACKAGE_NAME}/test/test${PACKAGE_NAME}${SUFFIX}
if [ $?==0 ]; then
    echo "$PACKAGE_NAME: install-tree test success!"
fi
