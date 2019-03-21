# Mappel - mex wrapper CMakeLists.txt
#

## Source Files ##
# set(IFACE_TARGETS Gauss2DMLE_Iface Gauss2DMAP_Iface Gauss2DsMLE_Iface Gauss2DsMAP_Iface)

set(IFACE_TARGETS Gauss1DMAP_IFace Gauss2DMAP_IFace Gauss2DMLE_IFace Gauss2DsMAP_IFace)

# if(OPT_HYPERSPECTRAL)
#     set(IFACE_TARGETS ${IFACE_TARGETS} hyperspectral/GaussHSMAP_IFace hyperspectral/GaussHSsMAP_IFace
#                                     hyperspectral/Blink2DsMAP_IFace hyperspectral/BlinkHSsMAP_IFace)
# endif()

foreach(target_src IN LISTS IFACE_TARGETS)
    get_filename_component(target_name ${target_src} NAME_WE)
    message(STATUS "MakingMexModule: ${target_name}")
    mexiface_make_mex(MEXNAME ${target_name} SOURCES ${target_src}
                        LINK_LIBRARIES Mappel::Mappel)
endforeach()
