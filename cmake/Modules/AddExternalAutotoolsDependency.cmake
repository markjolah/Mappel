#
# File: AddExternalAutotoolsDependency.cmake
# Mark J. Olah (mjo AT cs.unm.edu)
# copyright: Licensed under the Apache License, Version 2.0.  See LICENSE file.
# date: 2017
#
# Function: AddExternalAutotoolsDependency
#
# Allows a cmake package dependency to be automatically added as a cmake ExternalProject, then built and installed
# to CMAKE_INSTALL_PREFIX.  All this happens before configure time for the client package, so that the dependency will be
# automatically found through the cmake PackageConfig system and the normal find_package() mechanism.
#
# This approach eliminates the need for an explicit git submodule for the external package, and it allows the client package to
# be quickly built on systems where the ExternalProject is already installed.
#
# useage: AddExternalDependency(<package_name> <package-git-clone-url> [SHARED] [STATIC])
cmake_policy(SET CMP0057 NEW)

set(AddExternalAutotoolsDependency_include_path ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "Path of AddExternalAutotoolsDependency.cmake")

macro(AddExternalAutotoolsDependency)
    set(ExtProjectName ${ARGV0})
    set(ExtProjectInstallPrefix ${CMAKE_INSTALL_PREFIX})
    #override ExtProjectURL passed in with environment variable
    set(ExtProjectURL_ENV $ENV{${ExtProjectName}URL}) 
    if(ExtProjectURL_ENV)
        set(ExtProjectURL $ENV{${ExtProjectName}URL})
    else()
        set(ExtProjectURL ${ARGV1})
    endif()

    set(ExtProject_BUILD_SHARED_LIBS OFF)
    set(ExtProject_BUILD_STATIC_LIBS OFF)
    if(${ARGC} GREATER 2)
        if(${ARGV2} STREQUAL STATIC)
            set(ExtProject_BUILD_STATIC_LIBS ON)
        elseif(${ARGV2} MATCHES SHARED)
            set(ExtProject_BUILD_SHARED_LIBS ON)
        endif()
    endif()
    if(${ARGC} GREATER 3)
        if(${ARGV3} MATCHES STATIC)
            set(ExtProject_BUILD_STATIC_LIBS ON)
        elseif(${ARGV3} MATCHES SHARED)
            set(ExtProject_BUILD_SHARED_LIBS ON)
        endif()
    endif()
    
    find_package(${ExtProjectName})
    if(NOT ${ExtProjectName}_FOUND)
        set(ExtProjectDir ${CMAKE_BINARY_DIR}/ExternalAutotools/${ExtProjectName})
        message(STATUS "[AddExternalAutotoolsDependency] 3rd Party Package Not found: ${ExtProjectName}")
        message(STATUS "[AddExternalAutotoolsDependency] Initializing as ExternalProject using git URL:${ExtProjectURL}")
        message(STATUS "[AddExternalAutotoolsDependency] BUILD_STATIC_LIBS:${ExtProject_BUILD_STATIC_LIBS} BUILD_SHARED_LIBS:${ExtProject_BUILD_SHARED_LIBS}")
        find_file(EXTERNAL_ATUOTOOLS_CMAKELISTS_TEMPLATE NAME ExternalAutotools.CMakeLists.txt.in PATHS ${AddExternalAutotoolsDependency_include_path}/Templates)
        configure_file(${EXTERNAL_ATUOTOOLS_CMAKELISTS_TEMPLATE} ${ExtProjectDir}/CMakeLists.txt @ONLY)
        execute_process(COMMAND ${CMAKE_COMMAND} . WORKING_DIRECTORY ${ExtProjectDir})
        message(STATUS "[AddExternalAutotoolsDependency] Downloading Building and Installing: ${ExtProjectName}")
        execute_process(COMMAND ${CMAKE_COMMAND} --build . WORKING_DIRECTORY ${ExtProjectDir})
        find_package(${ExtProjectName} REQUIRED)
        if(NOT ${ExtProjectName}_FOUND)
            message(FATAL_ERROR "[AddExternalAutotoolsDependency] Install of ${ExtProjectName} failed.")
        endif()
        message(STATUS "[AddExternalAutotoolsDependency] Installed: ${ExtProjectName} Ver:${${ExtProjectName}_VERSION} Location:${ExtProjectInstallPrefix}")
    else()
        message(STATUS "[AddExternalAutotoolsDependency] Found:${ExtProjectName} Ver:${${ExtProjectName}_VERSION} Location:${TRNG_LIBRARIES}")
    endif()
endmacro()
