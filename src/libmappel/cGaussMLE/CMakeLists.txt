# cGaussMLE
set(CGAUSS_SRCS cGaussMLE.cpp GaussLib.cpp matinv.cpp)
set(CGAUSS_PUBLIC_HDRS cGaussMLE.h)
add_library(cGaussMLE OBJECT ${CGAUSS_SRCS})
set_target_properties(cGaussMLE PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
