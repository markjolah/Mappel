#!/bin/bash
# install-blas-lapack.sh <INSTALL_PREFIX> <CMAKE-ARGS...>
#
# Download, configure and install armadillo.  If install prefix is omitted, defaults to root.
#
# Environment variables:
# BLAS_INT64=1 - for 64-bit integer support
#
if [ -z "$1" ]; then
    INSTALL_PREFIX="/usr"
    SUDO=sudo
else
    if [ ! -d "$1" ]; then
        mkdir -p $1
    fi
    INSTALL_PREFIX=$(cd $1; pwd)
    SUDO=""
fi

#Find fortran compiler
if [ -z "$FC" ]; then
    FC=$(find /usr/bin -name gfortran* -print -quit)
fi

WORK_DIR=_work
PKG_NAME=LAPACK
BUILD_PATH=_build
PKG_URL="https://github.com/Reference-LAPACK/lapack-release.git"
PKG_BRANCH="lapack-3.8.0"
PC_SUFFIX="" #Suffix for pkg-condifg .pc files produced
PKG_CONFIG_PATH=$INSTALL_PREFIX/lib64/pkgconfig #Install location of .pc files
if [ "${BLAS_INT64,,}" == "on" ] || [ "${BLAS_INT64}" -eq 1 ]; then
    PC_SUFFIX="-int64"
    FFLAGS="-fdefault-integer-8 $FFLAGS" #Force 64-bit integer support
fi
NUM_PROCS=$(grep -c ^processor /proc/cpuinfo)
REPOS_DIR=$WORK_DIR/$PKG_NAME

CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
CMAKE_ARGS="${CMAKE_ARGS} -DBUILD_SHARED_LIBS=On -DBUILD_STATIC_LIBS=Off"
CMAKE_ARGS="${CMAKE_ARGS} -DCBLAS=OFF -DLAPACKE=OFF"
CMAKE_ARGS="${CMAKE_ARGS} ${@:2}"

set -ex

rm -rf $REPOS_DIR
mkdir -p $REPOS_DIR
cd $WORK_DIR
git clone $PKG_URL -b $PKG_BRANCH $PKG_NAME --depth 1
cd $PKG_NAME
mkdir -p $BUILD_PATH
cmake . -B$BUILD_PATH -DCMAKE_Fortran_COMPILER="$FC" -DCMAKE_Fortran_FLAGS="${FFLAGS}"  ${CMAKE_ARGS}
cd $BUILD_PATH
make all -j$NUM_PROCS
$SUDO make install
set +x
echo "PKG_CONFIG: $PKG_CONFIG_PATH"
echo "Modified: $($SUDO find $INSTALL_PREFIX/lib/pkgconfig $INSTALL_PREFIX/lib64/pkgconfig $INSTALL_PREFIX/x86_64-linux-gnu/lib/pkgconfig -type f -name blas.pc -print -exec rename blas.pc blas-reference${PC_SUFFIX}.pc {} \; 2> /dev/null)"
echo "Modified: $($SUDO find $INSTALL_PREFIX/lib/pkgconfig $INSTALL_PREFIX/lib64/pkgconfig $INSTALL_PREFIX/x86_64-linux-gnu/lib/pkgconfig -type f -name lapack.pc -print -exec rename lapack.pc lapack-reference${PC_SUFFIX}.pc {} \; 2> /dev/null)"
