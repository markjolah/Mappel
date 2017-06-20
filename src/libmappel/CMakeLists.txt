# Mappel - libmappel CMakeLists.txt
#
# Mark J. Olah [mjo@cs.unm.edu] (02/2014)
#
# Currently we build shared and static libs for each plaform.
# We depend on the cGaussMLE and SFMT object libraries and make a single object
# library mappel-core which is used for both the shared and static builds
#
add_subdirectory(cGaussMLE)

set(MAPPEL_SRCS display.cpp stencil.cpp rng.cpp util.cpp mcmc.cpp numerical.cpp
                PointEmitterModel.cpp 
                ImageFormat1DBase.cpp ImageFormat2DBase.cpp
                PoissonNoise1DObjective.cpp  PoissonNoise2DObjective.cpp
                Gauss1DModel.cpp
                Gauss2DModel.cpp Gauss2DMAP.cpp Gauss2DMLE.cpp
                )

if(OPT_HYPERSPECTRAL)
    set(MAPPEL_SRCS ${MAPPEL_SRCS} hyperspectral/PointEmitterHSModel.cpp 
                                   hyperspectral/BlinkModel.cpp hyperspectral/Blink2DsMAP.cpp
                                   hyperspectral/GaussHSMAP.cpp hyperspectral/GaussHSsMAP.cpp hyperspectral/BlinkHSsMAP.cpp)
endif()

set(LIB_TARGET "mappel${DEBUG_FILE_EXT}")
set(STATIC_LIB_TARGET "mappel-static${DEBUG_FILE_EXT}")
message("Lib target: " ${LIB_TARGET})
message("Static Lib target: " ${STATIC_LIB_TARGET})

add_library( mappel-core OBJECT ${MAPPEL_SRCS} )

add_library( ${LIB_TARGET} SHARED $<TARGET_OBJECTS:mappel-core> $<TARGET_OBJECTS:cGaussMLE>)


add_library( ${STATIC_LIB_TARGET} STATIC $<TARGET_OBJECTS:mappel-core> $<TARGET_OBJECTS:cGaussMLE>)
set_target_properties( ${STATIC_LIB_TARGET} PROPERTIES ARCHIVE_OUTPUT_NAME mappel${DEBUG_FILE_EXT} )
set(MAPPEL_LIBS ${LIB_TARGET} ${STATIC_LIB_TARGET})

foreach(lib ${MAPPEL_LIBS})
    message("Linking lib: " ${lib})
    target_link_libraries( ${lib} ${Boost_THREAD_LIBRARY_RELEASE}  ${Boost_SYSTEM_LIBRARY_RELEASE} )
    target_link_libraries( ${lib} ${Boost_CHRONO_LIBRARY_RELEASE})
    target_link_libraries( ${lib} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} )
    target_link_libraries( ${lib} ${TRNG_LIBRARY} )
    if (WIN32)
        target_link_libraries( ${lib} ${LIBGOMP_LIBRARY})
    endif()
    set_property(TARGET ${lib} PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

if(UNIX)
    install(TARGETS ${MAPPEL_LIBS}
            RUNTIME DESTINATION bin        COMPONENT Runtime
            LIBRARY DESTINATION lib        COMPONENT Runtime
            ARCHIVE DESTINATION lib/mappel COMPONENT Development)
elseif(WIN32)
    # This prevents installation of the .dll.a files which are part of the ARCHIVE set
    install(TARGETS ${LIB_TARGET}
            RUNTIME DESTINATION .  COMPONENT Runtime
            LIBRARY DESTINATION .  COMPONENT Runtime)
endif()

set(FIXUP_LIBRARY ${LIB_TARGET} CACHE STRING "Library to fixup")

include(${LOCAL_CMAKE_DIR}/SMAToolbox-FixupBundle.cmake)
