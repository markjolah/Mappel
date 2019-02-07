# Mapple/src/libmapple
# Main CMakeLists.txt for Mapple libraries

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
# add_shared_static_libraries()
# * Add shared and static library targets to project namespace
# * Sets target_include_directories
# * Sets target_compile_features and target_link_libraries
# * Installs target libraries and headers
# * returns lsit SHARED_STATIC_LIB_TARGETS of created targets for adjusting additional target properties
add_shared_static_libraries(SOURCES ${MAPPEL_SRCS} $<TARGET_OBJECTS:cGaussMLE>
                            COMPILE_FEATURES cxx_std_14
                            PUBLIC_HEADER_DIR ${CMAKE_SOURCE_DIR}/include
                            INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                            RETURN_TARGETS lib_targets)

#Custom target settings for each lib_target from add_shared_static_libraries
foreach(target IN LISTS lib_targets)
    if(OPT_ARMADILLO_INT64)
        target_link_libraries(${target} INTERFACE LAPACK::LapackInt64 BLAS::BlasInt64)
    else()
        target_link_libraries(${target} INTERFACE LAPACK::Lapack BLAS::Blas)
    endif()

    target_link_libraries(${target} PUBLIC BacktraceException::BacktraceException)
    target_link_libraries(${target} PUBLIC ParallelRngManager::ParallelRngManager)
    target_link_libraries(${target} PUBLIC PriorHessian::PriorHessian)
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(${target} PUBLIC Boost::thread Boost::chrono)
    target_link_libraries(${target} PUBLIC Pthread::Pthread)
    target_link_libraries(${target} INTERFACE Armadillo::Armadillo)
endforeach()
