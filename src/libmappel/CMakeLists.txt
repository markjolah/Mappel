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

file(GLOB MAPPEL_HDRS *.h)
                
if(OPT_HYPERSPECTRAL)
    file(GLOB HS_SRCS hyperspectral/*.cpp)
    file(GLOB HS_HDRS hyperspectral/*.cpp)
    set(MAPPEL_SRCS ${MAPPEL_SRCS} ${HS_SRCS})
    set(MAPPEL_HDRS ${MAPPEL_HDRS} ${HS_HDRS})
endif()


add_library( mappel-core OBJECT ${MAPPEL_SRCS} )
set_target_properties(mappel-core PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

set(LIB_TARGET mappel)
add_library( ${LIB_TARGET} SHARED $<TARGET_OBJECTS:mappel-core> $<TARGET_OBJECTS:cGaussMLE>)
target_link_libraries(${LIB_TARGET} ${Boost_THREAD_LIBRARY_RELEASE}  ${Boost_SYSTEM_LIBRARY_RELEASE} ${Boost_CHRONO_LIBRARY_RELEASE})
target_link_libraries(${LIB_TARGET} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} )
target_link_libraries(${LIB_TARGET} ${TRNG_LIBRARY} ) #Parallel random number generation
target_link_libraries(${LIB_TARGET} ${OPENMP_LIBRARY}) #OpenMP (non-empty for WIN32)
target_link_libraries(${LIB_TARGET} ${PTHREAD_LIBRARY}) #Pthreads
target_link_libraries(${LIB_TARGET} Boost::chrono Boost::thread Boost::system) 
set_target_properties(${LIB_TARGET} PROPERTIES INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${LIB_TARGET} PROPERTIES PUBLIC_HEADER "${MAPPEL_HDRS}")
target_include_directories(${LIB_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cGaussMLE)
target_include_directories(${LIB_TARGET} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)

if(UNIX)
    install(TARGETS ${LIB_TARGET} EXPORT ${EXPORT_TARGETS_NAME}
            RUNTIME DESTINATION bin COMPONENT Runtime
            LIBRARY DESTINATION lib COMPONENT Runtime
            PUBLIC_HEADER DESTINATION include/${PROJECT_NAME} COMPONENT Development)
elseif(WIN32)
    install(TARGETS ${LIB_TARGET} EXPORT ${EXPORT_TARGETS_NAME}
            RUNTIME DESTINATION .  COMPONENT Runtime
            LIBRARY DESTINATION .  COMPONENT Runtime
            ARCHIVE DESTINATION .  COMPONENT Development
            PUBLIC_HEADER DESTINATION include/${PROJECT_NAME} COMPONENT Development)
endif()


