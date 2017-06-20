# cGaussMLE
# Mark J. Olah [mjo@cs.unm.edu] (02/2014)

set(CGAUSS_SRCS cGaussMLE.cpp GaussLib.cpp matinv.cpp)
add_library( cGaussMLE OBJECT ${CGAUSS_SRCS} )
