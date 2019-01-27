# Mappel - libmappel CMakeLists.txt
#
# Mark J. Olah [mjo@cs.unm DOT edu] (02/2014)
#
# Currently we build shared and static libs for each plaform.
# We depend on the cGaussMLE and SFMT object libraries and make a single object
# library mappel-core which is used for both the shared and static builds
#
add_subdirectory(cGaussMLE)

set(MAPPEL_SRCS display.cpp stencil.cpp util.cpp mcmc.cpp numerical.cpp rng.cpp
                PointEmitterModel.cpp 
                ImageFormat1DBase.cpp PoissonNoise1DObjective.cpp  
                ImageFormat2DBase.cpp PoissonNoise2DObjective.cpp  
                Gauss1DModel.cpp Gauss1DMLE.cpp Gauss1DMAP.cpp
                MCMCAdaptorBase.cpp MCMCAdaptor1D.cpp MCMCAdaptor1Ds.cpp MCMCAdaptor2D.cpp MCMCAdaptor2Ds.cpp
                Gauss1DsModel.cpp Gauss1DsMLE.cpp Gauss1DsMAP.cpp
                Gauss2DModel.cpp Gauss2DMLE.cpp Gauss2DMAP.cpp
                Gauss2DsModel.cpp Gauss2DsMLE.cpp Gauss2DsMAP.cpp)
                
if(OPT_HYPERSPECTRAL)
    file(GLOB HS_SRCS hyperspectral/*.cpp)
    list(APPEND MAPPEL_SRCS ${HS_SRCS})
endif()

include(AddSharedStaticLibraries)
add_shared_static_libraries(SOURCES ${MAPPEL_SRCS} $<TARGET_OBJECTS:cGaussMLE> PUBLIC_HEADER_DIR ${CMAKE_CURRENT_LIST_DIR}/../../include
    LINK_LIBRARIES BacktraceException::BacktraceException ParallelRngManager::ParallelRngManager PriorHessian::PriorHessian
                   OpenMP::OpenMP_CXX Boost::thread Boost::chrono ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${PTHREAD_LIBRARY}
    INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
