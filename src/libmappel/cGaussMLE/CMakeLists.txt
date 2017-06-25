# cGaussMLE
set(CGAUSS_SRCS cGaussMLE.cpp GaussLib.cpp matinv.cpp)
add_library(cGaussMLE OBJECT ${CGAUSS_SRCS})
set_target_properties(cGaussMLE PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
