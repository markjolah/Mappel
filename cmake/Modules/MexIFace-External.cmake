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
    message(STATUS "[MexIFace] Not found.")
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/MexIFace-External.CMakeLists.txt.in 
                   ${CMAKE_BINARY_DIR}/MexIFace-External/CMakeLists.txt
                   @ONLY)
    message(STATUS "[MexIFace] Configuring External Project...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} .
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/MexIFace-External
    )
    message(STATUS "[MexIFace] Automatically downloading, configuring, building, and installing MexIFace...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build .
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/MexIFace-External
    )
    find_package(MexIFace CONFIG PATHS ${CMAKE_INSTALL_PREFIX}/lib/cmake/MexIFace NO_CMAKE_FIND_ROOT_PATH)
    if(NOT MexIFace_FOUND)
        message(FATAL_ERROR "Install of MexIFace failed.")
    endif()
    message(STATUS "[MexIFace] Installed Ver:${MexIFace_VERSION} Location:${CMAKE_INSTALL_PREFIX}")
elseif()
    message(STATUS "[MexIFace] Found Ver:${MexIFace_VERSION}")
endif()
