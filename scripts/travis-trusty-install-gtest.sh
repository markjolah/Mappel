#!/bin/bash
#
# travis-trusty-install-gtest.sh
#
# Install's gtest shared libraries on
set -ex
sudo apt-get install -y libgtest-dev
sudo chown -R travis /usr/src/gtest
cd /usr/src/gtest
#install shared libs
cmake CMakeLists.txt -DBUILD_SHARED_LIBS=On
make
sudo cp libgtest.so libgtest_main.so /usr/lib/
#install static libs
cmake CMakeLists.txt -DBUILD_SHARED_LIBS=Off
make
sudo cp libgtest.a libgtest_main.a /usr/lib/
