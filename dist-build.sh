#!/bin/bash
# dist-build.sh <INSTALL_PREFIX>
# Release-only build for distribution.  Testing is enabled.
# Does not clear INSTALL_PATH for obvious reasons.
#
# Build both linux and windows into the same file for distribution
#
# Args:
#  <INSTALL_PREFIX> - path to distribution install directory [Default: _dist]. Installed at <PkgName>-<PkgVers> subdir.
#                     Interpreted relative to current directory.
#
#
if [ -z $1 ]; then
    INSTALL_DIR=_dist
else
    INSTALL_DIR=$1
fi

LINUX_FULL_ARCH=x86_64-gcc4_9-linux-gnu
W64_FULL_ARCH=x86_64-w64-mingw32
VERSION=0.0.3
NAME=Mappel
INSTALL_DIR_NAME=${NAME}-${VERSION}
ZIP_FILE=${NAME}-${VERSION}.zip
TAR_FILE=${NAME}-${VERSION}.tbz2

LINUX_TOOLCHAIN_FILE=./cmake/UncommonCMakeModules/Toolchains/Toolchain-${LINUX_FULL_ARCH}.cmake
W64_TOOLCHAIN_FILE=./cmake/UncommonCMakeModules/Toolchains/Toolchain-MXE-${W64_FULL_ARCH}.cmake
INSTALL_PATH=${INSTALL_DIR}/$INSTALL_DIR_NAME
BUILD_PATH=_build/dist
NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)

ARGS=""
ARGS="${ARGS} -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH"
ARGS="${ARGS} -DBUILD_STATIC_LIBS=ON"
ARGS="${ARGS} -DBUILD_SHARED_LIBS=ON"
ARGS="${ARGS} -DBUILD_TESTING=On"
ARGS="${ARGS} -DOPT_INSTALL_TESTING=On"
ARGS="${ARGS} -DOPT_EXPORT_BUILD_TREE=Off"
ARGS="${ARGS} -DOPT_FIXUP_DEPENDENCIES=On"
ARGS="${ARGS} -DOPT_FIXUP_DEPENDENCIES_BUILD_TREE=Off"
ARGS="${ARGS} -DOPT_FIXUP_DEPENDENCIES_COPY_GCC_LIBS=Off"
ARGS="${ARGS} -DOPT_MexIFace_INSTALL_DISTRIBUTION_STARTUP=On" #Copy startupPackage.m to root for distribution
ARGS="${ARGS} -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=On" #Don't try to use any build-tree exports.  Install all dependencies.
ARGS="${ARGS} -DOPT_MATLAB=On"
ARGS="${ARGS} -DOPT_PYTHON=On"

set -ex
rm -rf $BUILD_PATH

cmake -H. -B$BUILD_PATH/LinuxDebug -DCMAKE_TOOLCHAIN_FILE=$LINUX_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Debug ${ARGS}
cmake --build $BUILD_PATH/LinuxDebug --target install -- -j${NUM_PROCS}

cmake -H. -B$BUILD_PATH/LinuxRelease -DCMAKE_TOOLCHAIN_FILE=$LINUX_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Release ${ARGS}
cmake --build $BUILD_PATH/LinuxRelease --target install -- -j${NUM_PROCS}

cmake -H. -B$BUILD_PATH/W64Debug -DCMAKE_TOOLCHAIN_FILE=$W64_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Debug ${ARGS}
cmake --build $BUILD_PATH/W64Debug --target install -- -j${NUM_PROCS}

cmake -H. -B$BUILD_PATH/W64Release -DCMAKE_TOOLCHAIN_FILE=$W64_TOOLCHAIN_FILE -DCMAKE_BUILD_TYPE=Release ${ARGS}
cmake --build $BUILD_PATH/W64Release --target install -- -j${NUM_PROCS}

cd $INSTALL_DIR
zip -rq $ZIP_FILE $INSTALL_DIR_NAME
tar cjf $TAR_FILE $INSTALL_DIR_NAME
