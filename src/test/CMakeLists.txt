# Mappel - test CMakeLists.txt
# Mark J. Olah [mjo@cs.unm DOT edu] 
# 2014-2019
#
# Build google-test executable
#
# Dependencies
#   * Google Test

find_package(GTest REQUIRED)

set(GTEST_EXE test${PROJECT_NAME})
file(GLOB GTEST_SRCS test_*.cpp)
add_executable(${GTEST_EXE} ${GTEST_SRCS})
target_link_libraries(${GTEST_EXE} ${PROJECT_NAME}::${PROJECT_NAME}) 
target_link_libraries(${GTEST_EXE} GTest::GTest GTest::Main)
add_test(NAME gtest_mappel COMMAND ${GTEST_EXE} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
if(CMAKE_CROSSCOMPILING)
    fixup_dependencies(TARGETS ${GTEST_EXE} EXPORT_BUILD_TREE True EXPORT_INSTALL_TREE False)
endif()

add_executable(estimator_speed estimator_speed.cpp)
target_link_libraries(estimator_speed PUBLIC Mappel::Mappel)

if(OPT_PROFILE)
    find_library(GOOGLE_PROFILER profiler)
    if(GOOGLE_PROFILER)
        message(STATUS "GoogleProfiler found: ${GOOGLE_PROFILER}.  Enabling profiling for speed testing executables.")
        target_link_libraries(estimator_speed PRIVATE ${GOOGLE_PROFILER})
    endif()
endif()

if(CMAKE_CROSSCOMPILING)
    fixup_dependencies(TARGETS estimator_speed EXPORT_BUILD_TREE True EXPORT_INSTALL_TREE False)
endif()


# Reenabled older tests.
add_executable(demo test.cpp)
target_link_libraries(demo PUBLIC Mappel::Mappel)

set(DEMO_CMDS mappel.demo_speed${EXE_POSTFIX}
              mappel.demo_image${EXE_POSTFIX})
#                mappel.test_prior${EXE_POSTFIX}
#                mappel.test_compare_estimators${EXE_POSTFIX}
#                mappel.test_image_compare_estimators${EXE_POSTFIX}
#                mappel.test_point_evaluate_estimators${EXE_POSTFIX})

foreach(cmd ${DEMO_CMDS})
    add_custom_command(TARGET demo POST_BUILD COMMAND ln -sf demo ${cmd} )
#     install(CODE "execute_process(COMMAND ln -sf demo ${cmd} WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/bin)")
    add_test(NAME ${cmd} COMMAND ${cmd})
endforeach()

    #Tests can only be run automatically on Unix caus we are cross-compiling to w64
#     add_test(NAME GTest COMMAND ${GTEST_TARGET})
#     add_test(NAME TestSpeed COMMAND mappel.test_speed${DEBUG_FILE_EXT})
#     add_test(NAME TestImage COMMAND mappel.test_image${DEBUG_FILE_EXT})


if(CMAKE_CROSSCOMPILING)
    fixup_dependencies(TARGETS demo EXPORT_BUILD_TREE True EXPORT_INSTALL_TREE False)
endif()


# add_test(NAME GTest COMMAND ${GTEST_EXE})
# 
# set(TEST_CORE_SRCS )
# set(GTEST_SRCS gtest.cpp)
# 
# set(GTEST_TARGET mappel.gtest)
# set(CTEST_TARGET mappel.ctest)
# 


# add_executable( ${GTEST_TARGET} ${GTEST_SRCS} $<TARGET_OBJECTS:test-core> )
# if(UNIX)
#     install(TARGETS ${GTEST_TARGET} RUNTIME DESTINATION bin COMPONENT Runtime )
# elseif(WIN32)
#     install(TARGETS ${GTEST_TARGET} DESTINATION . )
# endif()

# target_link_libraries( ${GTEST_TARGET} mappel${DEBUG_FILE_EXT})
# target_link_libraries( ${GTEST_TARGET} ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})
# if(CMAKE_BUILD_TYPE MATCHES "Debug")
#     set(EXE_POSTFIX ${CMAKE_DEBUG_POSTFIX})
# else()
#     set(EXE_POSTFIX)
# endif()
# 
# message(STATUS "EXE_POSTFIX:${EXE_POSTFIX}")
# set(ESTIMATORS Newton Huristic CGauss QuasiNewton NewtonRaphson)
# 
# #Ctest is only availible on Unix as it uses symlinks
# if(UNIX)
#     add_executable(${CTEST_TARGET} ${CTEST_SRCS} ${TEST_CORE_SRCS})
#     target_link_libraries(${CTEST_TARGET} Mappel)
#     set_target_properties(${CTEST_TARGET} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
#     set_target_properties(${CTEST_TARGET} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
#     if( IPO_SUPPORTED  )
#         set_property(TARGET ${LIB_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
#     endif()
# 
#     install(TARGETS ${CTEST_TARGET} RUNTIME DESTINATION bin COMPONENT Runtime )
#     #symlinks to enable different test commands
#     set(CTEST_NAME ${CTEST_TARGET}${EXE_POSTFIX})
#     foreach(cmd ${CTEST_CMDS})
#         add_custom_command( TARGET ${CTEST_TARGET} POST_BUILD COMMAND ln -sf ${CTEST_NAME} ${cmd} )
#         install(CODE "execute_process(COMMAND ln -sf ${CTEST_NAME} ${cmd} WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/bin)")
#     endforeach()
#
#     #Tests can only be run automatically on Unix caus we are cross-compiling to w64
# #     add_test(NAME GTest COMMAND ${GTEST_TARGET})
# #     add_test(NAME TestSpeed COMMAND mappel.test_speed${DEBUG_FILE_EXT})
# #     add_test(NAME TestImage COMMAND mappel.test_image${DEBUG_FILE_EXT})
# #     add_test(NAME TestCompareEstimators COMMAND mappel.test_compare_estimators${DEBUG_FILE_EXT})
# #     add_test(NAME TestImageCompareEstimators COMMAND mappel.test_image_compare_estimators${DEBUG_FILE_EXT})
# #     add_test(NAME TestPointEvaluateEstimators  COMMAND mappel.test_point_evaluate_estimators${DEBUG_FILE_EXT})
# #     foreach(estimator ${ESTIMATORS})
# #         add_test(NAME ${estimator} COMMAND mappel.test_evaluate_estimator${DEBUG_FILE_EXT} ${estimator})
# #     endforeach()
# endif()
# 
# # set(FIXUP_BINARY mappel.gtest${DEBUG_FILE_EXT} CACHE STRING "Binary to fixup")
# 
# endif()
