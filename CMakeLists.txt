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

# cmake_policy(SET CMP0054 NEW) #Don't derrefernce in if() statements
# cmake_policy(SET CMP0022 NEW) #LINK_INTERFACE_LIBRARIES fix

# Enable Matlab Support via MexIFace
option(OPT_MATLAB "Add support for Matlab via MexIFace." OFF)

# Enable Support for Python via Boost::Python
option(OPT_PYTHON "Add support for Python via boost::python" OFF)

# Hyperspectral code
option(OPT_HYPERSPECTRAL "Add support for hyperspectral and blinking psf" OFF)

option(OPT_DEBUG "Support debugging features")

#Modules
set(${PROJECT_NAME}_CMAKE_DIR ${CMAKE_SOURCE_DIR}/cmake)
set(${PROJECT_NAME}_CMAKE_MODULES_DIR ${${PROJECT_NAME}_CMAKE_DIR}/Modules)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${${PROJECT_NAME}_CMAKE_MODULES_DIR})
set(${PROJECT_NAME}_CMAKE_FUNCTIONS_DIR ${${PROJECT_NAME}_CMAKE_DIR}/Functions)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${${PROJECT_NAME}_CMAKE_FUNCTIONS_DIR}) 
set(${PROJECT_NAME}_CMAKE_TEMPLATES_DIR ${${PROJECT_NAME}_CMAKE_DIR}/Templates)
set(${PROJECT_NAME}_CMAKE_TOOLCHAINS_DIR ${MexIFace_CMAKE_DIR}/Toolchains)

#Allow inclusion of external dependencies
include(AddExternalDependency)

#BacktraceException provides backtraces for caught exceptions.
set(BacktraceExceptionURL https://github.com/markjolah/BacktraceException.git)
# CACHE STRING URL of BacktraceException library dependency)
AddExternalDependency(BacktraceException ${BacktraceExceptionURL} SHARED)

#ParallelRngManager allows for management of parallel rng streams.
set(ParallelRngManagerURL https://github.com/markjolah/ParallelRngManager.git)
AddExternalDependency(ParallelRngManager ${ParallelRngManagerURL} SHARED)

#PriorHessian allows for hessians of prior distirbutions
set(PriorHessianURL https://github.com/markjolah/PriorHessian.git)
AddExternalDependency(PriorHessian ${PriorHessianURL} SHARED)

include(Mappel-configure) #Main configuration module
include(Mappel-Doxygen) #Add Doxygen documentation targets
include(Mappel-PackageConfig) #Configure Package Config and associated files for exports

add_subdirectory(src/libmappel) #Main library
# add_subdirectory(src/test) #Test executables

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


