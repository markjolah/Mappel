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
option(OPT_INSTALL_TESTING "Install testing executables" OFF)
option(OPT_EXPORT_BUILD_TREE "Configure the package so it is usable from the build tree.  Useful for development." OFF)
option(OPT_ARMADILLO_INT64 "Use 64-bit integers for Armadillo, BLAS, and LAPACK. [Must be on if OPT_MATLAB is on]" OFF)
option(OPT_MATLAB "Add support for Matlab via MexIFace." OFF)
option(OPT_PYTHON "Add support for Python via boost::python" OFF) # Enable Support for Python via Boost::Python
option(OPT_EXTRA_DEBUG "Support extra noisy debugging features" OFF) #Extra debug features (Armadillo)
# # Hyperspectral code
# option(OPT_HYPERSPECTRAL "Add support for hyperspectral and blinking psf" OFF)

if(OPT_MATLAB AND NOT OPT_ARMADILLO_INT64)
    set(OPT_ARMADILLO_INT64 True)
    set(OPT_ARMADILLO_INT64 True CACHE BOOL "Use 64-bit integers for Armadillo, BLAS, and LAPACK. [Forced on by OPT_MATLAB]." FORCE)
endif()

message(STATUS "OPTION: BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "OPTION: BUILD_STATIC_LIBS: ${BUILD_STATIC_LIBS}")
message(STATUS "OPTION: BUILD_TESTING: ${BUILD_TESTING}")
message(STATUS "OPTION: OPT_INSTALL_TESTING: ${OPT_INSTALL_TESTING}")
message(STATUS "OPTION: OPT_EXPORT_BUILD_TREE: ${OPT_EXPORT_BUILD_TREE}")
message(STATUS "OPTION: OPT_ARMADILLO_INT64: ${OPT_ARMADILLO_INT64}")
message(STATUS "OPTION: OPT_MATLAB: ${OPT_MATLAB}")
message(STATUS "OPTION: OPT_PYTHON: ${OPT_PYTHON}")
message(STATUS "OPTION: OPT_EXTRA_DEBUG: ${OPT_EXTRA_DEBUG}") 

#Add UcommonCmakeModules git subpreo to path.
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/cmake/UncommonCMakeModules)

### Dependencies
#External depenency integration
include(AddExternalDependency)
set(EXTERNAL_DEPENDENCY_TESTING TESTING PASS_CACHE_VARIABLES OPT_INSTALL_TESTING OPT_ARMADILLO_INT64)
#BacktraceException provides backtraces for caught exceptions.
set(BacktraceExceptionURL https://github.com/markjolah/BacktraceException.git)
add_external_dependency(NAME BacktraceException URL ${BacktraceExceptionURL} SHARED ${EXTERNAL_DEPENDENCY_TESTING})

#ParallelRngManager allows for management of parallel rng streams.
set(ParallelRngManagerURL https://github.com/markjolah/ParallelRngManager.git)
add_external_dependency(NAME ParallelRngManager URL ${ParallelRngManagerURL} SHARED ${EXTERNAL_DEPENDENCY_TESTING})

#PriorHessian allows for hessians of prior distirbutions
set(PriorHessianURL https://github.com/markjolah/PriorHessian.git)
add_external_dependency(NAME PriorHessian URL ${PriorHessianURL} SHARED ${EXTERNAL_DEPENDENCY_TESTING})

#Armadillo, BLAS, and LAPACK, using the UncommonCMakeModules Find<PKG>.cmake modules
if(OPT_ARMADILLO_INT64)
    set(ARMADILLO_INT64_COMPONENT INT64) #Enable INT64 interface for armadillo, Blas and Lapack
else()
    set(ARMADILLO_INT64_COMPONENT)
endif()
find_package(BLAS REQUIRED COMPONENTS ${ARMADILLO_INT64_COMPONENT})
find_package(LAPACK REQUIRED COMPONENTS ${ARMADILLO_INT64_COMPONENT})
find_package(Armadillo REQUIRED COMPONENTS BLAS LAPACK CXX11 ${ARMADILLO_INT64_COMPONENT})
add_compile_definitions(${ARMADILLO_PRIVATE_COMPILE_DEFINITIONS})

find_package(OpenMP REQUIRED)
find_package(Pthread REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread chrono)

#Configure standard CFlags and definitions for debug builds
include(ConfigureDebugBuilds) 

#include(Mappel-Doxygen) #Add Doxygen documentation targets

### PackageConfig Exports from UncommonCMakeModules/ExportPackageWizzard.cmake
#setup build-tree and install-tree exports and PackageConfig files
include(ExportPackageWizzard)
export_package_wizzard()

add_subdirectory(src/libmappel) #Main library

if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(src/test)
endif()

if(OPT_MATLAB)
    message(STATUS "*** Matlab MEX Modules Enabled ***")
    set(MexIFaceURL https://github.com/markjolah/MexIFace.git CACHE STRING "URL of MexIFace library dependency")
    add_external_dependency(NAME MexIFace URL "${MexIFaceURL}" SHARED ${EXTERNAL_DEPENDENCY_TESTING})
    add_subdirectory(src/MexIFace) #MATLAB Modules
    mexiface_configure_install(EXPORT_BUILD_TREE "${OPT_EXPORT_BUILD_TREE}") #Matlab code and startupTracker.m configure and install
endif()

if(OPT_PYTHON)
    message(STATUS "*** Python Libraries Enabled ***")
    add_subdirectory(src/python) #C++ interface source code
    add_subdirectory(python) #python source code
endif()
