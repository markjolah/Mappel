# Mappel -MAPPEL CMakeLists.txt
# Mark J. Olah [mjo@cs.unm.edu] (08/2014)
#

cmake_minimum_required( VERSION 2.8 )
project(MAPPEL CXX)
set(LOCAL_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake CACHE STRING "Local cmake library")

#This controls addition of hyperspectral code
option(OPT_HYPERSPECTRAL "Add support for hyperspectral and blinking psf" OFF)

include(${LOCAL_CMAKE_DIR}/SMAToolbox-configure.cmake)
include(${LOCAL_CMAKE_DIR}/SMAToolbox-configure-blas.cmake)
include(${LOCAL_CMAKE_DIR}/SMAToolbox-configure-lapack.cmake)
include(${LOCAL_CMAKE_DIR}/SMAToolbox-configure-boost.cmake)
include(${LOCAL_CMAKE_DIR}/SMAToolbox-configure-gtest.cmake)
include(${LOCAL_CMAKE_DIR}/SMAToolbox-configure-trng.cmake)
include(${LOCAL_CMAKE_DIR}/MexIface.cmake)

include_directories(src/Iface src/libmappel src/test)

add_subdirectory(src/Iface)
add_subdirectory(src/libmappel)
add_subdirectory(src/test)

