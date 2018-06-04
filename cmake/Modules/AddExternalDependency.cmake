#
# File: AddExternalDependency.cmake
# Mark J. Olah (mjo AT cs.unm.edu)
# copyright: Licensed under the Apache License, Version 2.0.  See LICENSE file.
# date: 2017
#
# Function: AddExternalDependency
#
# Allows a cmake package dependency to be automatically added as a cmake ExternalProject, then built and installed
# to CMAKE_INSTALL_PREFIX.  All this happens before configure time for the client package, so that the dependency will be
# automatically found through the cmake PackageConfig system and the normal find_package() mechanism.
#
# This approach eliminates the need for an explicit git submodule for the external package, and it allows the client package to
# be quickly built on systems where the ExternalProject is already installed.
#
# useage: AddExternalDependency(<package-name> <package-git-clone-url> [SHARED] [STATIC])
#cmake_policy(SET CMP0057 NEW)

set(AddExternalDependency_include_path ${CMAKE_CURRENT_LIST_DIR} CACHE INTERNAL "Path of AddExternalDependency.cmake")

macro(AddExternalDependency)
    set(ExtProjectName ${ARGV0})
    
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
    
    find_package(${ExtProjectName} CONFIG)
    string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
#     message("${ExtProjectName}_DIR:${${ExtProjectName}_DIR}")
#     message("${ExtProjectName}_CONSIDERED_CONFIGS:${${ExtProjectName}_CONSIDERED_CONFIGS}")
#     message("${ExtProjectName}_CONSIDERED_VERSIONS:${${ExtProjectName}_CONSIDERED_VERSIONS}")
#     message("${ExtProjectName}_BUILD_TYPES:${${ExtProjectName}_BUILD_TYPES}")
#     message("BUILD_TYPE:${BUILD_TYPE}")
    if(NOT ${ExtProjectName}_FOUND OR (${ExtProjectName}_BUILD_TYPES AND (NOT ${BUILD_TYPE} IN_LIST ${ExtProjectName}_BUILD_TYPES )))
        set(ExtProjectDir ${CMAKE_BINARY_DIR}/External/${ExtProjectName})
        message(STATUS "[AddExternalProjectDependency] Not found: ${ExtProjectName}")
        message(STATUS "[AddExternalProjectDependency] Initializing as ExternalProject URL:${ExtProjectURL}")
        message(STATUS "[AddExternalProjectDependency] BUILD_STATIC_LIBS:${ExtProject_BUILD_STATIC_LIBS} BUILD_SHARED_LIBS:${ExtProject_BUILD_SHARED_LIBS}")
        message(STATUS "[AddExternalProjectDependency] ExtProjectBuildTypes:${${ExtProjectName}_BUILD_TYPES}")
        
        find_file(EXTERNAL_CMAKELISTS_TEMPLATE NAME External.CMakeLists.txt.in PATHS ${AddExternalDependency_include_path}/Templates)
        
        configure_file(${EXTERNAL_CMAKELISTS_TEMPLATE} 
                       ${ExtProjectDir}/CMakeLists.txt @ONLY)
        execute_process(COMMAND ${CMAKE_COMMAND} . WORKING_DIRECTORY ${ExtProjectDir})
        message(STATUS "[AddExternalProjectDependency] Downloading Building and Installing: ${ExtProjectName}")
        execute_process(COMMAND ${CMAKE_COMMAND} --build . WORKING_DIRECTORY ${ExtProjectDir})
        find_package(${ExtProjectName} CONFIG PATHS ${CMAKE_INSTALL_PREFIX}/lib/cmake/${ExtProjectName} NO_CMAKE_FIND_ROOT_PATH)
        if(NOT ${ExtProjectName}_FOUND)
            message(FATAL_ERROR "[AddExternalProjectDependency] Install of ${ExtProjectName} failed.")
        endif()
        message(STATUS "[AddExternalProjectDependency] Installed: ${ExtProjectName} Ver:${${ExtProjectName}_VERSION} Location:${CMAKE_INSTALL_PREFIX}")
    else()
        message(STATUS "[AddExternalProjectDependency] Found:${ExtProjectName} Ver:${${ExtProjectName}_VERSION}")
    endif()
endmacro()
