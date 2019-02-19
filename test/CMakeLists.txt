# Mappel - test CMakeLists.txt
# Mark J. Olah [mjo@cs.unm DOT edu] 
# 2014-2019
#
# Build google-test executable
#
# Dependencies
#   * Google Test

find_package(GTest REQUIRED)

#Main google-test C++ unit testing.  set OPT_INSTALL_TESTING to install.
set(TEST_TARGET test${PROJECT_NAME})
file(GLOB GTEST_SRCS test_*.cpp)
add_executable(${TEST_TARGET} ${GTEST_SRCS})
target_link_libraries(${TEST_TARGET} PUBLIC ${PROJECT_NAME}::${PROJECT_NAME})
target_link_libraries(${TEST_TARGET} PUBLIC GTest::GTest)
set_target_properties(${TEST_TARGET} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
add_test(NAME gtest_mappel_cxx COMMAND ${TEST_TARGET})
if(OPT_INSTALL_TESTING)
    if(WIN32)
        set(TESTING_INSTALL_DESTINATION bin)
    elseif(UNIX)
        set(TESTING_INSTALL_DESTINATION lib/${PROJECT_NAME}/test)
        set_target_properties(${TEST_TARGET} PROPERTIES INSTALL_RPATH "\$ORIGIN/../..")
    endif()
    install(TARGETS ${TEST_TARGET} RUNTIME DESTINATION ${TESTING_INSTALL_DESTINATION} COMPONENT Testing)
elseif(CMAKE_CROSSCOMPILING)
    fixup_dependencies(TARGETS ${GTEST_EXE} EXPORT_BUILD_TREE True EXPORT_INSTALL_TREE False)
endif()

### Debugging-only testing.  No installation.

#estimator_speed - debugging speed testing and profiling target.
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


#demo - debugging command-line targets.
add_executable(demo test.cpp)
target_link_libraries(demo PUBLIC Mappel::Mappel)
set(DEMO_CMDS mappel.demo_speed
              mappel.demo_image)
#                mappel.test_prior${EXE_POSTFIX}
#                mappel.test_compare_estimators${EXE_POSTFIX}
#                mappel.test_image_compare_estimators${EXE_POSTFIX}
#                mappel.test_point_evaluate_estimators${EXE_POSTFIX})

foreach(cmd ${DEMO_CMDS})
    add_custom_command(TARGET demo POST_BUILD COMMAND ln -sf demo ${cmd} )
    add_test(NAME ${cmd} COMMAND ${cmd})
endforeach()
