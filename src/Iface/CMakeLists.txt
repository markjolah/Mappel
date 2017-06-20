# Mappel - mex wrapper CMakeLists.txt
#
# Mark J. Olah [mjo@cs.unm.edu] (2014)
#
#copyright Mark J. Olah and The Regents of the University of New Mexico (2014).
# This code is free for non-commercial use and modification, provided
# this copyright notice remains unmodified and attached to the code


## Source Files ##
# set(IFACE_TARGETS Gauss2DMLE_Iface Gauss2DMAP_Iface Gauss2DsMLE_Iface Gauss2DsMAP_Iface)
set(IFACE_TARGETS Gauss2DMLE_Iface Gauss2DMAP_Iface)

if(OPT_HYPERSPECTRAL)
    set(IFACE_TARGETS ${IFACE_TARGETS} hyperspectral/GaussHSMAP_Iface hyperspectral/GaussHSsMAP_Iface 
                                       hyperspectral/Blink2DsMAP_Iface hyperspectral/BlinkHSsMAP_Iface)
endif()
#Create IFace target
foreach(target  ${IFACE_TARGETS})
    Make_mex( ${target} )
    target_link_libraries( ${target}  mappel${DEBUG_FILE_EXT})
endforeach()
