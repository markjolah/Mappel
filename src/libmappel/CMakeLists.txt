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
                Gauss2DsModel.cpp Gauss2DsMLE.cpp Gauss2DsMAP.cpp
                )

file(GLOB MAPPEL_HDRS *.h)
                
if(OPT_HYPERSPECTRAL)
    file(GLOB HS_SRCS hyperspectral/*.cpp)
    file(GLOB HS_HDRS hyperspectral/*.cpp)
    set(MAPPEL_SRCS ${MAPPEL_SRCS} ${HS_SRCS})
    set(MAPPEL_HDRS ${MAPPEL_HDRS} ${HS_HDRS})
endif()

set(LIB_TARGET Mappel) #Name for main Mappel library
add_library(${LIB_TARGET} SHARED ${MAPPEL_SRCS} $<TARGET_OBJECTS:cGaussMLE>)
target_link_libraries(${LIB_TARGET} PUBLIC ${BacktraceException_LIBRARIES} )
target_link_libraries(${LIB_TARGET} PUBLIC ${ParallelRngManager_LIBRARIES} )
target_link_libraries(${LIB_TARGET} PUBLIC ${PriorHessian_LIBRARIES})
target_link_libraries(${LIB_TARGET} PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} )
target_link_libraries(${LIB_TARGET} PUBLIC OpenMP::OpenMP_CXX) #OpenMP (non-empty for WIN32)
target_link_libraries(${LIB_TARGET} PUBLIC ${PTHREAD_LIBRARY}) #Pthreads
target_link_libraries(${LIB_TARGET} PUBLIC Boost::chrono Boost::thread) 
set_target_properties(${LIB_TARGET} PROPERTIES PUBLIC_HEADER "${MAPPEL_HDRS}")
# set_target_properties(${LIB_TARGET} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
target_include_directories(${LIB_TARGET} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
#target_include_directories(${LIB_TARGET} PUBLIC $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
if( IPO_SUPPORTED  )
    set_property(TARGET ${LIB_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(UNIX)
    install(TARGETS ${LIB_TARGET} EXPORT ${EXPORT_TARGETS_NAME}
            LIBRARY DESTINATION lib COMPONENT Runtime
            PUBLIC_HEADER DESTINATION include/${PROJECT_NAME} COMPONENT Development)
elseif(WIN32)
    install(TARGETS ${LIB_TARGET} EXPORT ${EXPORT_TARGETS_NAME}
            RUNTIME DESTINATION lib  COMPONENT Runtime
            ARCHIVE DESTINATION lib  COMPONENT Development
            PUBLIC_HEADER DESTINATION include/${PROJECT_NAME} COMPONENT Development)
endif()

if(OPT_MATLAB)
    fixup_dependencies(${LIB_TARGET})
endif()
