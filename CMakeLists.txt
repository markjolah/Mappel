# Mappel - Main CMakeLists.txt
#
# MAPPEL: [M]aximum [A] [P]osteriori [P]oint [E]mitter [L]ocalization 
# High-performance, parallel, robust, cross-platform localization for EMCCD and SCMOS microscopy.
#
# Mark J. Olah (mjo@cs.unm DOT edu)
# Copyright 2014-2019
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
# See: LICENCE file

cmake_minimum_required( VERSION 3.9 )
project(Mappel VERSION 0.0.3 LANGUAGES CXX)

# Enable Matlab Support via MexIFace
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    option(BUILD_TESTING "Build testing framework" ON)
else()
    option(BUILD_TESTING "Build testing framework" OFF)
endif()

option(OPT_MATLAB "Add support for Matlab via MexIFace." OFF)

# Enable Support for Python via Boost::Python
option(OPT_PYTHON "Add support for Python via boost::python" OFF)

# # Hyperspectral code
# option(OPT_HYPERSPECTRAL "Add support for hyperspectral and blinking psf" OFF)

option(OPT_EXTRA_DEBUG "Support extra noisy debugging features" OFF) #Extra debug features (Armadillo)

message(STATUS "OPTION: BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "OPTION: BUILD_STATIC_LIBS: ${BUILD_STATIC_LIBS}")
message(STATUS "OPTION: BUILD_TESTING: ${BUILD_TESTING}")
message(STATUS "OPTION: OPT_EXPORT_BUILD_TREE: ${OPT_EXPORT_BUILD_TREE}")
message(STATUS "OPTION: OPT_MATLAB: ${OPT_MATLAB}")
message(STATUS "OPTION: OPT_PYTHON: ${OPT_PYTHON}")
message(STATUS "OPTION: OPT_DEBUG: ${OPT_DEBUG}")
message(STATUS "OPTION: OPT_EXTRA_DEBUG: ${OPT_EXTRA_DEBUG}") 

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/cmake/UncommonCMakeModules)

# Standard configuration for a numerical computation environment 
#  * Armadillo 
#  * OpenMP
#  * LAPACK
#  * Blas
#  * Pthreads 
include(ConfigureNumericalDependenciesArmadillo)

find_package(Boost REQUIRED COMPONENTS thread chrono)

#Configure standard CFlags and definitions for debug builds
include(ConfigureDebugBuilds) 

#External depenency integration
include(AddExternalDependency)

#BacktraceException provides backtraces for caught exceptions.
set(BacktraceExceptionURL https://github.com/markjolah/BacktraceException.git)
add_external_dependency(NAME BacktraceException URL ${BacktraceExceptionURL} SHARED)

#ParallelRngManager allows for management of parallel rng streams.
set(ParallelRngManagerURL https://github.com/markjolah/ParallelRngManager.git)
add_external_dependency(NAME ParallelRngManager URL ${ParallelRngManagerURL} SHARED)

#PriorHessian allows for hessians of prior distirbutions
set(PriorHessianURL https://github.com/markjolah/PriorHessian.git)
add_external_dependency(NAME PriorHessian URL ${PriorHessianURL} SHARED)
message(STATUS "PATH:${CMAKE_MODULE_PATH}")


#TODO: get this working again
#include(Mappel-Doxygen) #Add Doxygen documentation targets

### PackageConfig Exports from UncommonCMakeModules
include(ExportPackageWizzard)
export_package_wizzard()

#Main library
add_subdirectory(src/libmappel) 

if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(src/test)
endif()

if(OPT_MATLAB)
    message(STATUS "*** Matlab MEX Modules Enabled ***")
    set(MexIFaceURL https://github.com/markjolah/MexIFace.git CACHE STRING "URL of MexIFace library dependency")
    add_external_dependency(NAME MexIFace URL "${MexIFaceURL}" SHARED TESTING)
    add_subdirectory(src/MexIFace) #MATLAB Modules
    mexiface_configure_install(EXPORT_BUILD_TREE "${OPT_EXPORT_BUILD_TREE}") #Matlab code and startupTracker.m configure and install
endif()

if(OPT_PYTHON)
    message(STATUS "*** Python Libraries Enabled ***")
    add_subdirectory(src/python) #C++ interface source code
    add_subdirectory(python) #python source code
endif()
