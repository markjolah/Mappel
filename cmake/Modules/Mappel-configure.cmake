# MexIFace CMake build system
# Mark J. Olah (mjo@cs.unm DOT edu)
# 03-2014
# Common Configuration for Project Libraries and Dependencies
# Configures armadillo and OpenMP for x-platform builds

## Find and Configure Required Libraries ##
message(STATUS "[Mappel]: Configure Libraries")

# Armadillo
find_package(Armadillo REQUIRED)
add_definitions(-DARMA_USE_CXX11 -DARMA_DONT_USE_WRAPPER -DARMA_BLAS_LONG)
add_definitions(-DARMA_DONT_USE_OPENMP) #We want to control the use of openMP at a higher-grained level
add_definitions(-DARMA_DONT_USE_HDF5)
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    add_definitions(-DARMA_PRINT_ERRORS)
else()
    add_definitions(-DARMA_NO_DEBUG)
endif()
# Optionally enable extra debugging from armadillo to log every call.
if( (${CMAKE_BUILD_TYPE} MATCHES Debug) AND ${PROJECT_NAME}_EXTRA_DEBUG)
    add_definitions(-DARMA_EXTRA_DEBUG)
endif()

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

# Compiler Definitions
if (WIN32)
    add_definitions( -DWIN32 )
elseif(UNIX AND NOT APPLE)
    add_definitions( -DLINUX )
endif()

## CFLAGS ##
add_compile_options(-W -Wall -Wextra -Werror -Wno-unused-parameter)
if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    add_definitions(-DDEBUG)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options(-fmax-errors=5) #Limit number of reported errors
    endif()
elseif()
    add_definitions(-DNDEBUG)
endif()
set(CMAKE_DEBUG_POSTFIX ".debug" CACHE STRING "Debug file extension")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-ggdb -O3")

## MAC OS X Config ##
set(CMAKE_MACOSX_RPATH 1) #Enable rpaths on OS X

