# Mappel - test CMakeLists.txt
# Mark J. Olah [mjo@cs.unm.edu] (02/2014)
#
# Build google-test executable (cross-platform) and 
# test execucatable which is unix command line testing
#
# Dependencies
#   * Google Test
#   * Boost - Math (test)
#

if(UNIX) #Globally disabling this for windows as the test does not work

include(CTest)


set(TEST_CORE_SRCS stats.cpp)
set(GTEST_SRCS gtest.cpp)
set(CTEST_SRCS test.cpp)

set(GTEST_TARGET mappel.gtest${DEBUG_FILE_EXT})
set(CTEST_TARGET mappel.ctest${DEBUG_FILE_EXT})


add_library( test-core OBJECT ${TEST_CORE_SRCS})

# add_executable( ${GTEST_TARGET} ${GTEST_SRCS} $<TARGET_OBJECTS:test-core> )
# if(UNIX)
#     install(TARGETS ${GTEST_TARGET} RUNTIME DESTINATION bin COMPONENT Runtime )
# elseif(WIN32)
#     install(TARGETS ${GTEST_TARGET} DESTINATION . )
# endif()

# target_link_libraries( ${GTEST_TARGET} mappel${DEBUG_FILE_EXT})
# target_link_libraries( ${GTEST_TARGET} ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})

set(CTEST_CMDS mappel.test_speed${DEBUG_FILE_EXT}
               mappel.test_image${DEBUG_FILE_EXT}
               mappel.test_compare_estimators${DEBUG_FILE_EXT}
               mappel.test_image_compare_estimators${DEBUG_FILE_EXT}
               mappel.test_point_evaluate_estimators${DEBUG_FILE_EXT}
               mappel.test_evaluate_estimator${DEBUG_FILE_EXT})
set(ESTIMATORS Newton Huristic CGauss QuasiNewton NewtonRaphson)

if(UNIX)
    add_executable( ${CTEST_TARGET} ${CTEST_SRCS} $<TARGET_OBJECTS:test-core>  )
    install(TARGETS ${CTEST_TARGET} RUNTIME DESTINATION bin COMPONENT Runtime )
    target_link_libraries( ${CTEST_TARGET} mappel${DEBUG_FILE_EXT} )
    #symlinks to enable different test commands
    foreach(cmd ${CTEST_CMDS})
        add_custom_command( TARGET ${CTEST_TARGET} POST_BUILD COMMAND ln -sf ${CTEST_TARGET} ${cmd} )
        install(CODE "execute_process(COMMAND ln -sf ${CTEST_TARGET} ${cmd} WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/bin)")
    endforeach()

    #Tests can only be run automatically on Unix caus we are cross-compiling to w64
#     add_test(NAME GTest COMMAND ${GTEST_TARGET})
#     add_test(NAME TestSpeed COMMAND mappel.test_speed${DEBUG_FILE_EXT})
#     add_test(NAME TestImage COMMAND mappel.test_image${DEBUG_FILE_EXT})
#     add_test(NAME TestCompareEstimators COMMAND mappel.test_compare_estimators${DEBUG_FILE_EXT})
#     add_test(NAME TestImageCompareEstimators COMMAND mappel.test_image_compare_estimators${DEBUG_FILE_EXT})
#     add_test(NAME TestPointEvaluateEstimators  COMMAND mappel.test_point_evaluate_estimators${DEBUG_FILE_EXT})
#     foreach(estimator ${ESTIMATORS})
#         add_test(NAME ${estimator} COMMAND mappel.test_evaluate_estimator${DEBUG_FILE_EXT} ${estimator})
#     endforeach()
endif()

set(FIXUP_BINARY mappel.gtest${DEBUG_FILE_EXT} CACHE STRING "Binary to fixup")

endif()
