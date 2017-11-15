# MexIFace-External.cmake
#
# This allows MexIFace to be automatically added as an cmake ExternalProject and built and installed
# to CMAKE_INSTALL_PREFIX before configure time.
#
# This file should be copied into "cmake/MexIFace-External.cmake" by any project that wants to 
# allow users to automatically install MexIFace if needed.
#
# This method eliminates the need for an explict submodule for MexIFace and allows MexIFace tools to
# be quickly built on systems where MexIFace is already installed
#
find_package(MexIFace QUIET CONFIG PATHS ${CMAKE_INSTALL_PREFIX}/lib/cmake/MexIFace)
if(NOT MexIFace_FOUND) 
    #Try to configure build and install MexIFace
    set(ExtProjectName MexIFace)
    set(ExtProjectDir ${CMAKE_BINARY_DIR}/External/${ExtProjectName})
    set(ExtProjectURL https://github.com/markjolah/${ExtProjectName}.git)
    #override ExtProjectURL passed in with environment variable
    set(ExtProjectURL_ENV $ENV{${ExtProjectName}URL}) 
    if(ExtProjectURL_ENV)
        set(ExtProjectURL $ENV{${ExtProjectName}URL})
    endif()
    message(STATUS "[MexIFace-External] MexIFace: Not found.")
    message(STATUS "[MexIFace-External] Initializing as ExternalProject URL:${ExtProjectURL}")
    message(STATUS "[MexIFace-External] ExtProjectBuildTypes:${${ExtProjectName}_BUILD_TYPES}")
    configure_file(${Mappel_CMAKE_TEMPLATES_DIR}/MexIFace-External.CMakeLists.txt.in 
                   ${ExtProjectDir}/CMakeLists.txt @ONLY)
    message(STATUS "[MexIFace-External] Configuring External Project: ${ExtProjectName}...")
    execute_process(COMMAND ${CMAKE_COMMAND} . WORKING_DIRECTORY ${ExtProjectDir})
    message(STATUS "[MexIFace-External] Automatically downloading, configuring, building, and installing: ${ExtProjectName}...")
    execute_process(COMMAND ${CMAKE_COMMAND} --build . WORKING_DIRECTORY ${ExtProjectDir})
    find_package(${ExtProjectName} CONFIG PATHS ${CMAKE_INSTALL_PREFIX}/lib/cmake/${ExtProjectName} NO_CMAKE_FIND_ROOT_PATH)
    if(NOT ${ExtProjectName}_FOUND)
        message(FATAL_ERROR "Install of ${ExtProjectName} failed.")
    endif()
    message(STATUS "[MexIFace-External] Installed: ${ExtProjectName} Ver:${${ExtProjectName}_VERSION} Location:${CMAKE_INSTALL_PREFIX}")
elseif()
    message(STATUS "[MexIFace-External] Found:${ExtProjectName} Ver:${${ExtProjectName}_VERSION}")
endif()
