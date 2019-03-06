# Mappel - Main CMakeLists.txt
#
# MAPPEL: [M]aximum [A] [P]osteriori [P]oint [E]mitter [L]ocalization 
# High-performance, parallel, robust, cross-platform localization for EMCCD and SCMOS microscopy.
#
# Mark J. Olah (mjo@cs.unm DOT edu)
# Copyright 2014-2019
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
# See: LICENSE file

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
option(OPT_DOC "Build documentation" OFF)
option(OPT_INSTALL_TESTING "Install testing executables" OFF)
option(OPT_EXPORT_BUILD_TREE "Configure the package so it is usable from the build tree.  Useful for development." OFF)
option(OPT_BLAS_INT64 "Use 64-bit integers for Armadillo, BLAS, and LAPACK. [Must be on if OPT_MATLAB is on]" OFF)
option(OPT_MATLAB "Add support for Matlab via MexIFace." OFF)
option(OPT_PYTHON "Add support for Python via boost::python" OFF) # Enable Support for Python via Boost::Python
option(OPT_EXTRA_DEBUG "Support extra noisy debugging features" OFF) #Extra debug features (Armadillo)
option(OPT_EXAMPLES "Build C++ examples" OFF) #Extra debug features (Armadillo)
option(OPT_PROFILE "Enable performance profiling" OFF) #Extra debug features (Armadillo)
# # Hyperspectral code
# option(OPT_HYPERSPECTRAL "Add support for hyperspectral and blinking psf" OFF)

if(OPT_MATLAB AND NOT OPT_BLAS_INT64)
    set(OPT_BLAS_INT64 True)
    set(OPT_BLAS_INT64 True CACHE BOOL "Use 64-bit integers for Armadillo, BLAS, and LAPACK. [Forced on by OPT_MATLAB]." FORCE)
endif()

message(STATUS "OPTION: BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "OPTION: BUILD_STATIC_LIBS: ${BUILD_STATIC_LIBS}")
message(STATUS "OPTION: BUILD_TESTING: ${BUILD_TESTING}")
message(STATUS "OPTION: OPT_DOC: ${OPT_DOC}")
message(STATUS "OPTION: OPT_INSTALL_TESTING: ${OPT_INSTALL_TESTING}")
message(STATUS "OPTION: OPT_EXPORT_BUILD_TREE: ${OPT_EXPORT_BUILD_TREE}")
message(STATUS "OPTION: OPT_BLAS_INT64: ${OPT_BLAS_INT64}")
message(STATUS "OPTION: OPT_MATLAB: ${OPT_MATLAB}")
message(STATUS "OPTION: OPT_PYTHON: ${OPT_PYTHON}")
message(STATUS "OPTION: OPT_EXTRA_DEBUG: ${OPT_EXTRA_DEBUG}")
message(STATUS "OPTION: OPT_EXAMPLES: ${OPT_EXAMPLES}")
message(STATUS "OPTION: OPT_PROFILE: ${OPT_PROFILE}")

#Add UcommonCmakeModules git subpreo to path.
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/cmake/UncommonCMakeModules)

### Dependencies
#External depenency integration
include(AddExternalDependency)
if(OPT_BLAS_INT64)
    set(BLAS_INT_COMPONENT BLAS_INT64)
else()
    set(BLAS_INT_COMPONENT BLAS_INT32)
endif()
#BacktraceException provides backtraces for caught exceptions.
set(BacktraceExceptionURL https://github.com/markjolah/BacktraceException.git CACHE STRING "URL of BacktraceException library dependency")
add_external_dependency(NAME BacktraceException URL ${BacktraceExceptionURL} VARS OPT_INSTALL_TESTING)

#ParallelRngManager allows for management of parallel rng streams.
set(ParallelRngManagerURL https://github.com/markjolah/ParallelRngManager.git CACHE STRING "URL of ParallelRngManager library dependency")
add_external_dependency(NAME ParallelRngManager URL ${ParallelRngManagerURL} VARS OPT_INSTALL_TESTING)

#PriorHessian allows for computation of Hessians of prior distributions
# - PriorHessian has BLAS/LAPACK dependencies.  Must have the same BLAS_INT_COMPONENT settings as this package.
set(PriorHessianURL https://github.com/markjolah/PriorHessian.git CACHE STRING "URL of PriorHessian library dependency")
add_external_dependency(NAME PriorHessian URL ${PriorHessianURL}
                        VARS OPT_INSTALL_TESTING OPT_BLAS_INT64
                        COMPONENTS ${BLAS_INT_COMPONENT})

if(OPT_MATLAB)
    #MexIFace - A Cross-Platform C++ / MEX Object-based interface wrapper and CMake build tool.
    set(MexIFaceURL https://github.com/markjolah/MexIFace.git CACHE STRING "URL of MexIFace library dependency")
    add_external_dependency(NAME MexIFace URL ${MexIFaceURL} VARS OPT_INSTALL_TESTING)

    #CellFun - A functional programming oriented library for matlab CellArray manipulation.
    set(CellFunRL https://github.com/markjolah/CellFun.git CACHE STRING "URL of CellFun library dependency")
    add_external_dependency(NAME CellFun URL ${CellFunURL})

    #GUIBuilder - A Matlab mixin class, with methods for creating interactive GUIs for objects.
    set(GUIBuilderRL https://github.com/markjolah/GUIBuilder.git CACHE STRING "URL of GUIBuilder library dependency")
    add_external_dependency(NAME GUIBuilder URL ${GUIBuilderURL})
endif()

#Armadillo, BLAS, and LAPACK, using the UncommonCMakeModules Find<PKG>.cmake modules
find_package(BLAS REQUIRED COMPONENTS ${BLAS_INT_COMPONENT})
find_package(LAPACK REQUIRED COMPONENTS ${BLAS_INT_COMPONENT})
find_package(Armadillo REQUIRED COMPONENTS BLAS LAPACK CXX11 ${BLAS_INT_COMPONENT})
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS ${ARMADILLO_PRIVATE_COMPILE_DEFINITIONS})

find_package(OpenMP REQUIRED)
if(WIN32)
    find_package(Boost REQUIRED chrono system)
else()
    find_package(Boost REQUIRED)
endif()

#Configure standard CFlags and definitions for debug builds
include(ConfigureDebugBuilds) 

#include(Mappel-Doxygen) #Add Doxygen documentation targets

### PackageConfig Exports from UncommonCMakeModules/ExportPackageWizzard.cmake
#setup build-tree and install-tree exports and PackageConfig files
include(ExportPackageWizzard)
set(EXPORTED_FIND_MODULES cmake/UncommonCMakeModules/FindArmadillo.cmake
                          cmake/UncommonCMakeModules/FindBLAS.cmake
                          cmake/UncommonCMakeModules/FindLAPACK.cmake
                          cmake/UncommonCMakeModules/MakePkgConfigTarget.cmake)
export_package_wizzard(PROVIDED_COMPONENTS ${BLAS_INT_COMPONENT} #Propagate BLAS integer size dependency
                       FIND_MODULES ${EXPORTED_FIND_MODULES})

### Main C++ library
add_subdirectory(src/libmappel) #Main library

### C++ Testing
if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()

### Matlab MexIFace interface
if(OPT_MATLAB)
    message(STATUS "*** Matlab MEX Modules Enabled ***")
    add_subdirectory(src/MexIFace) #MATLAB Modules
    #Matlab code and startupTracker.m configure and install
    mexiface_configure_install(DEPENDENCY_STARTUP_M_LOCATIONS ${GUIBuilder_MATLAB_STARTUP_M})
endif()

### Python pybind11 interface
if(OPT_PYTHON)
    message(STATUS "*** Python Libraries Enabled ***")
    add_subdirectory(src/python) #C++ interface source code
    add_subdirectory(python) #python source code
endif()

### C++ Examples
if(OPT_EXAMPLES)
    message(STATUS "*** C++ Example Programs Enabled ***")
    add_subdirectory(examples) #C++ example libraries
endif()

### Documentation
if(OPT_DOC)
    add_subdirectory(doc)
endif()
