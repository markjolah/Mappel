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

cmake_minimum_required( VERSION 3.9 )
project(Mappel VERSION 0.0.2 LANGUAGES CXX)

# Enable Matlab Support via MexIFace
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)

option(OPT_MATLAB "Add support for Matlab via MexIFace." OFF)

# Enable Support for Python via Boost::Python
option(OPT_PYTHON "Add support for Python via boost::python" OFF)

# # Hyperspectral code
# option(OPT_HYPERSPECTRAL "Add support for hyperspectral and blinking psf" OFF)

option(OPT_DEBUG "Support debugging features" OFF)
option(OPT_TESTING "Support testing features" OFF)

message(STATUS "OPTION: BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "OPTION: BUILD_STATIC_LIBS: ${BUILD_STATIC_LIBS}")
message(STATUS "OPTION: OPT_MATLAB: ${OPT_MATLAB}")
message(STATUS "OPTION: OPT_PYTHON: ${OPT_PYTHON}")
message(STATUS "OPTION: OPT_DEBUG: ${OPT_DEBUG}")
message(STATUS "OPTION: OPT_TESTING: ${OPT_TESTING}")

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)


#INTERPROCEDURAL_OPTIMIZATION
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_SUPPORTED_ERROR)

if( IPO_SUPPORTED )
    message(STATUS "IPO / LTO enabled")
    set_property(DIRECTORY PROPERTY INTERPROCEDURAL_OPTIMIZATION ON) 
else()
    message(STATUS "IPO / LTO not supported: <${IPO_SUPPORTED_ERROR}>")
endif()

# Armadillo
find_package(Armadillo REQUIRED)
add_definitions(-DARMA_USE_CXX11)
add_definitions(-DARMA_DONT_USE_WRAPPER)
add_definitions(-DARMA_BLAS_LONG)
add_definitions(-DARMA_DONT_USE_OPENMP) #We want to control the use of openMP at a higher-grained level
add_definitions(-DARMA_DONT_USE_HDF5)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:ARMA_PRINT_ERRORS>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<NOT:$<CONFIG:Debug>>:ARMA_NO_DEBUG>)
# if(OPT_DEBUG)
#     set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:ARMA_EXTRA_DEBUG>)
# endif()

# OpenMP
find_package(OpenMP REQUIRED)

# LAPACK & BLAS
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

#Boost configure
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_LIBS OFF)
if(WIN32)
    find_library(Boost_THREAD_LIBRARY_RELEASE libboost_thread_win32-mt.dll )
endif()
find_package(Boost REQUIRED COMPONENTS system chrono thread iostreams)
add_definitions( -DBOOST_THREAD_USE_LIB )

# Pthreads
if (WIN32)
    find_library(PTHREAD_LIBRARY libwinpthread.dll REQUIRED)
elseif(UNIX)
    find_library(PTHREAD_LIBRARY libpthread.so REQUIRED)
endif()
message(STATUS "Found Pthread Libarary: ${PTHREAD_LIBRARY}")
#Allow inclusion of external dependencies
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
include(ExportPackageWizzard)

# CFLAGS ##
add_compile_options($<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-O0>)
add_compile_options($<$<CONFIG:Debug>:-W>)
add_compile_options($<$<CONFIG:Debug>:-Wall>)
add_compile_options($<$<CONFIG:Debug>:-Wextra>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:DEBUG>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_OPTIONS $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU>>:-fmax-errors=5>)

# Compiler Definitions
if (WIN32)
    add_definitions( -DWIN32 )
elseif(UNIX AND NOT APPLE)
    add_definitions( -DLINUX )
endif()

set(CMAKE_DEBUG_POSTFIX ".debug" CACHE STRING "Debug file extension")

## MAC OS X Config ##
set(CMAKE_MACOSX_RPATH 1) #Enable rpaths on OS X

### PackageConfig Exports

export_package_wizzard()

#include(Mappel-Doxygen) #Add Doxygen documentation targets

add_subdirectory(src/libmappel) #Main library

if(OPT_TESTING)
    include(CTest)
    enable_testing()
#     add_subdirectory(src/test)
endif()

if(OPT_MATLAB)
    #MexIFace allows for cross-building of .mex modules for matlab in linux win64 and macosX
    message(STATUS "*** Matlab Module Building Enabled ***")
    set(MexIFaceURL https://github.com/markjolah/MexIFace.git CACHE STRING URL of MexIFace library dependency)
    AddExternalDependency(MexIFace ${MexIFaceURL} SHARED)
    #Configure MexIFace envrionment
    include(MexIFace)
    add_subdirectory(src/MexIFace) #MATLAB Modules
endif()

if(OPT_PYTHON)
    add_subdirectory(src/python)
    add_subdirectory(python)
endif()
