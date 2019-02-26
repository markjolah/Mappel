
#!/bin/bash
# scripts/matlab-dist-build.sh <INSTALL_DIR> <cmake-args...>
#
# Builds cross-platform (linux,Win64) re-distributable release-only build for Matlab and C++ libraries.
# A top-level startup@PACKAGE_NAME@.m will be created, which can be called in matlab to initialized all required matlab
# and mex code paths and dependencies.
#  * Testing and documentation are enabled.
#  * Creates a .zip and .tar.gz archives.
#  * Has to install each arch to seperate directory before combining because of the possibility of PackageConfig.cmake files
#    from one arch being detected by another.
#
# Args:
#  <INSTALL_DIR> - path to distribution install directory [Default: ${SRC_PATH}/_dist].
#                  The distribution files will be created under this directory with names based on
#                  package and versions.
#  <cmake_args...> - additional cmake arguments.
#
# Optional environment variables:
#  OPT_DEBUG - Enable debugging builds
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_PATH=${SCRIPT_DIR}/..

NAME=$(grep -Po "project\(\K([A-Za-z]+)" ${SRC_PATH}/CMakeLists.txt)
VERSION=$(grep -Po "project\([A-Za-z]+ VERSION \K([0-9.]+)" ${SRC_PATH}/CMakeLists.txt)
if [ -z $NAME ] || [ -z $VERSION ]; then
    echo "Unable to find package name and version from: ${SRC_PATH}/CMakeLists.txt"
    exit 1
fi

DIST_DIR_NAME=${NAME}-${VERSION}
if [ -z $1 ]; then
    INSTALL_PATH=${SRC_PATH}/_matlab_dist/$DIST_DIR_NAME
else
    INSTALL_PATH=$1/$DIST_DIR_NAME
fi

LINUX_FULL_ARCH=x86_64-gcc4_9-linux-gnu
W64_FULL_ARCH=x86_64-w64-mingw32
LINUX_TOOLCHAIN_FILE=${SRC_PATH}/cmake/UncommonCMakeModules/Toolchains/Toolchain-${LINUX_FULL_ARCH}.cmake
W64_TOOLCHAIN_FILE=${SRC_PATH}/cmake/UncommonCMakeModules/Toolchains/Toolchain-MXE-${W64_FULL_ARCH}.cmake

ZIP_FILE=${NAME}-${VERSION}.zip
TAR_FILE=${NAME}-${VERSION}.tbz2

BUILD_PATH=${SRC_PATH}/_build/dist
NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)

ARGS=""
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=Off"
ARGS="${ARGS} -DOPT_FIXUP_DEPENDENCIES=On"
ARGS="${ARGS} -DOPT_FIXUP_DEPENDENCIES_BUILD_TREE=Off"
ARGS="${ARGS} -DOPT_FIXUP_DEPENDENCIES_COPY_GCC_LIBS=Off"
ARGS="${ARGS} -DOPT_MexIFace_INSTALL_DISTRIBUTION_STARTUP=On" #Copy startupPackage.m to root for distribution
ARGS="${ARGS} -DOPT_MATLAB=On"

set -ex
rm -rf $BUILD_PATH
rm -rf $INSTALL_PATH

cmake -H${SRC_PATH} -B$BUILD_PATH/LinuxRelease -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/Linux -DCMAKE_TOOLCHAIN_FILE=$LINUX_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Release -DOPT_DOC=On ${ARGS} ${@:2}
cmake --build $BUILD_PATH/LinuxRelease --target doc -- -j${NUM_PROCS}
cmake --build $BUILD_PATH/LinuxRelease --target pdf -- -j${NUM_PROCS}
cmake --build $BUILD_PATH/LinuxRelease --target install -- -j${NUM_PROCS}

cmake -H${SRC_PATH} -B$BUILD_PATH/W64Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/Win64 -DCMAKE_TOOLCHAIN_FILE=$W64_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Release ${ARGS} ${@:2}
cmake --build $BUILD_PATH/W64Release --target install -- -j${NUM_PROCS}

if [ -n ${OPT_DEBUG} ]; then
    cmake -H${SRC_PATH} -B$BUILD_PATH/LinuxDebug -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/Linux -DCMAKE_TOOLCHAIN_FILE=$LINUX_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Debug ${ARGS} ${@:2}
    cmake --build $BUILD_PATH/LinuxDebug --target install -- -j${NUM_PROCS}

    cmake -H${SRC_PATH} -B$BUILD_PATH/W64Debug -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/Win64 -DCMAKE_TOOLCHAIN_FILE=$W64_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Debug ${ARGS} ${@:2}
    cmake --build $BUILD_PATH/W64Debug --target install -- -j${NUM_PROCS}
fi

cd $INSTALL_PATH
cp -a $INSTALL_PATH/Linux/* .
cp -a $INSTALL_PATH/Win64/* .
rm -rf ./Linux ./Win64
cd ..
zip -rq $ZIP_FILE $DIST_DIR_NAME
tar cjf $TAR_FILE $DIST_DIR_NAME
