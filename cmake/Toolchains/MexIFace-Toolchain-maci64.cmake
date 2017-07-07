# MexIFace CMake build system
# Mark J. Olah (mjo@cs.unm DOT edu)
# 03-2014
#
# Change MXE_ROOT to the cross-dev-environment directory.
# 
# We need to use shared libraries because a matlab mex module is a shared library.
# we can use static libraries if the files are compiled with -fPIC.  Thus we
# are using the MXE shared target
#
set(OSXCROSS_ROOT $ENV{OSXCROSS_ROOT})
set(USER_CROSS_PATH $ENV{MACI64_PATH})
set(TARGET_ARCH x86_64-w64-mingw32)
set(TARGET_MXE_ARCH ${TARGET_ARCH}.shared)
set(MXE_ARCH_ROOT ${MXE_ROOT}/usr/${TARGET_MXE_ARCH})
set(MXE_BIN_DIR ${MXE_ROOT}/usr/bin)
set(MXE_BIN_PFX ${MXE_BIN_DIR}/${TARGET_MXE_ARCH})
#Look here for libraries at install time
# set(LIB_SEARCH_PATHS "${MXE_ROOT}/usr/${TARGET_MXE_ARCH}/lib"
#                      "${MXE_ROOT}/usr/${TARGET_MXE_ARCH}/bin"
#                      "${MXE_ROOT}/usr/bin"
#                      "${MXE_ROOT}/usr/lib"
#                      "${MXE_ROOT}/usr/lib/gcc/x86_64-w64-mingw32.shared/4.9.4/"
#                      "${USER_W64_CROSS_ROOT}/lib")

set(CMAKE_SYSTEM_NAME Windows)

set(CMAKE_SYSTEM_PROGRAM_PATH ${MXE_BIN_DIR})
set(CMAKE_C_COMPILER ${MXE_BIN_PFX}-gcc)
set(CMAKE_CXX_COMPILER ${MXE_BIN_PFX}-g++)
set(CMAKE_RC_COMPILER ${MXE_BIN_PFX}-windres)

set(CMAKE_FIND_ROOT_PATH ${MXE_ROOT} ${USER_W64_CROSS_ROOT} ${MXE_ARCH_ROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

