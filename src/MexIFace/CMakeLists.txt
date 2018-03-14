# Mappel - mex wrapper CMakeLists.txt
#
# Mark J. Olah [mjo@cs.unm DOT edu] (2014)
#
#copyright Mark J. Olah and The Regents of the University of New Mexico (2014).
# This code is free for non-commercial use and modification, provided
# this copyright notice remains unmodified and attached to the code


## Source Files ##
# set(IFACE_TARGETS Gauss2DMLE_Iface Gauss2DMAP_Iface Gauss2DsMLE_Iface Gauss2DsMAP_Iface)

if(OPT_MATLAB)
    set(IFACE_TARGETS Gauss1DMLE_IFace)

    if(OPT_HYPERSPECTRAL)
        set(IFACE_TARGETS ${IFACE_TARGETS} hyperspectral/GaussHSMAP_IFace hyperspectral/GaussHSsMAP_IFace 
                                        hyperspectral/Blink2DsMAP_IFace hyperspectral/BlinkHSsMAP_IFace)
    endif()
    #Create IFace target
    foreach(target ${IFACE_TARGETS})
        make_mex(${target})
        target_link_libraries(${target} Mappel) #Link to Mappel
    endforeach()
endif()
