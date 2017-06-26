# Mappel - Main CMakeLists.txt
#
# MAPPEL: [M]aximum [A] [P]osteriori [P]oint [E]mitter [L]ocalization 
# High-performance, parallel, robust, cross-platform localization for EMCCD and SCMOS microscopy.
#
# Mark J. Olah (mjo@cs.unm DOT edu)
# Copyright 2014-2017
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
# See: LICENCE file

cmake_minimum_required( VERSION 3.0 )
project(Mappel VERSION 1.9.0 LANGUAGES CXX)

# Hyperspectral code
option(OPT_HYPERSPECTRAL "Add support for hyperspectral and blinking psf" OFF)

#Modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

#Enable build and install of MexIFace via ExternalProject_Add
include(MexIFace-External)

#Configure MexIFace envrionment
include(MexIFace)
include(MexIFace-configure-boost) #Add support for boost
include(MexIFace-configure-gtest) #Add support for googletest
include(MexIFace-configure-trng)  #Add support for TRNG parallel random number generator
include(MexIFace-Doxygen) #Add Doxygen documentation targets
include(MexIFace-PackageConfig) #Configure Package Config and associated files for exports

add_subdirectory(src/libmappel) #Main library
add_subdirectory(src/Iface) #MATLAB Modules
add_subdirectory(src/test) #Test executables

