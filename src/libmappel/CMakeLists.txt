# Mappel - libmappel CMakeLists.txt
#
# Mark J. Olah [mjo@cs.unm DOT edu] (02/2014)
#
# Currently we build shared and static libs for each plaform.
# We depend on the cGaussMLE and SFMT object libraries and make a single object
# library mappel-core which is used for both the shared and static builds
#
add_subdirectory(cGaussMLE)

set(MAPPEL_SRCS display.cpp stencil.cpp util.cpp mcmc.cpp numerical.cpp
                PointEmitterModel.cpp 
                ImageFormat1DBase.cpp PoissonNoise1DObjective.cpp  
                ImageFormat2DBase.cpp PoissonNoise2DObjective.cpp  
                Gauss1DModel.cpp Gauss1DMLE.cpp Gauss1DMAP.cpp
                Gauss1DsModel.cpp Gauss1DsMLE.cpp Gauss1DsMAP.cpp
                Gauss2DModel.cpp Gauss2DMLE.cpp Gauss2DMAP.cpp
                Gauss2DsModel.cpp Gauss2DsMLE.cpp Gauss2DsMAP.cpp)

set(PUBLIC_HEADER_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/../include)
                
if(OPT_HYPERSPECTRAL)
    file(GLOB HS_SRCS hyperspectral/*.cpp)
    file(GLOB HS_HDRS hyperspectral/*.cpp)
    set(MAPPEL_SRCS ${MAPPEL_SRCS} ${HS_SRCS})
    set(MAPPEL_HDRS ${MAPPEL_HDRS} ${HS_HDRS})
endif()

include(AddSharedStaticLibraries)
add_shared_static_libraries("${MAPPEL_SRCS};$<TARGET_OBJECTS:cGaussMLE>" ${PROJECT_NAME}) 

foreach(target ${LIB_TARGETS})
    target_link_libraries(${target} PUBLIC BacktraceException::BacktraceException )
    target_link_libraries(${target} PUBLIC ParallelRngManager::ParallelRngManager )
    target_link_libraries(${target} PUBLIC PriorHessian::PriorHessian)
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX) #OpenMP (non-empty for WIN32)
    target_link_libraries(${target} PUBLIC Boost::chrono)
    target_link_libraries(${target} PUBLIC Boost::thread) 
    target_link_libraries(${target} PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} )
    target_link_libraries(${target} PUBLIC ${PTHREAD_LIBRARY}) #Pthreads
    target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${PUBLIC_HEADER_SRC_DIR}> 
                                                $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                                $<INSTALL_INTERFACE:include>)
endforeach()
if(UNIX)
    install(TARGETS ${LIB_TARGETS} EXPORT ${PROJECT_NAME}Targets
            LIBRARY DESTINATION lib COMPONENT Runtime
            ARCHIVE DESTINATION lib  COMPONENT Development)
elseif(WIN32)
    install(TARGETS ${LIB_TARGETS} EXPORT ${PROJECT_NAME}Targets
            RUNTIME DESTINATION lib  COMPONENT Runtime
            ARCHIVE DESTINATION lib  COMPONENT Development)
endif()

install(DIRECTORY ${PUBLIC_HEADER_SRC_DIR}/ DESTINATION include COMPONENT Development)

if(OPT_MATLAB)
    fixup_dependencies(${LIB_TARGETS})
endif()
