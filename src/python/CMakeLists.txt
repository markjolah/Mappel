# Mappel - src/python
#
# Python module building.
# We are using boost::Python
#
message(STATUS "*** Python Module Building Enabled ***")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/pybind11/tools")

add_subdirectory( pybind11 )

#List of all Models to build a python module for.
set(MAPPEL_PYTHON_MODELS Gauss1DMLE)

set(MAPPEL_PYTHON_EXECUTABLE "" CACHE STRING "Override python executable name")

if(NOT MAPPEL_PYTHON_VERSIONS)
    message(STATUS "MAPPEL_PYTHON_VERSIONS unset. Using system python version.")
    if(MAPPEL_PYTHON_EXECUTABLE)
        set(PYTHON_EXECUTABLE ${MAPPEL_PYTHON_EXECUTABLE})
    endif()
    find_package(PythonInterp)
    message(STATUS "Found python version:${PYTHON_VERSION_STRING}")
    string(REGEX MATCH "^[2-3]\.[0-9]+" FOUND_PYTHON_VERSION ${PYTHON_VERSION_STRING})
    set(MAPPEL_PYTHON_VERSIONS ${FOUND_PYTHON_VERSION} CACHE STRING "List of X.Y python version to build for.")
else()
    string(REGEX MATCHALL "[0-9]\.[0-9]" MAPPEL_PYTHON_VERSIONS ${MAPPEL_PYTHON_VERSIONS})
endif()

message(STATUS "MAPPEL_PYTHON_VERSIONS:${MAPPEL_PYTHON_VERSIONS}")

if(NOT MAPPEL_PYTHON_VERSIONS)
    message(FATAL_ERROR "No valid python versions found")
endif()

foreach(vers ${MAPPEL_PYTHON_VERSIONS})
    string(REPLACE "\." "_" vers_ ${vers})  
    unset(PYTHON_EXECUTABLE CACHE)
    unset(PYTHONLIBS_FOUND CACHE)
    unset(PYTHON_LIBRARIES CACHE)
    unset(PYTHON_LIBRARY CACHE)
    unset(PYTHON_INCLUDE_DIRS CACHE)
    unset(PYTHON_MODULE_PREFIX CACHE)
    unset(PYTHON_MODULE_EXTENSION CACHE)
    unset(PYTHON_EXECUTABLE)
    unset(PYTHONLIBS_FOUND)
    unset(PYTHON_LIBRARY)
    unset(PYTHON_LIBRARIES)
    unset(PYTHON_INCLUDE_DIRS)
    unset(PYTHON_MODULE_EXTENSION)
    unset(PYTHON_MODULE_PREFIX)
    if(MAPPEL_PYTHON_EXECUTABLE)
        set(PYTHON_EXECUTABLE ${MAPPEL_PYTHON_EXECUTABLE})
    endif()
    find_package(PythonLibsNew ${vers} REQUIRED)
    
    message(STATUS "Configuring Python Vers:${vers}")
    message(STATUS "PYTHON_VERSION_STRING:${PYTHON_VERSION_STRING}")
    message(STATUS "PYTHON_PREFIX:${PYTHON_PREFIX}")
    message(STATUS "PYTHON_LIBRARIES:${PYTHON_LIBRARIES}")
    message(STATUS "PYTHON_INCLUDE_DIRS:${PYTHON_INCLUDE_DIRS}")
    message(STATUS "PYTHON_MODULE_EXTENSION:${PYTHON_MODULE_EXTENSION}")
    message(STATUS "PYTHON_SITE_PACKAGES:${PYTHON_SITE_PACKAGES}")
    message(STATUS "PYTHON_IS_DEBUG:${PYTHON_IS_DEBUG}")
    
    foreach(model ${MAPPEL_PYTHON_MODELS})
        set(MODULE_NAME _${model}_${vers_})
        set(MODULE_OUTPUT_NAME _${model})
        set(MODULE_SRC ${model}_module.cpp)
        
        pybind11_add_module(${MODULE_NAME} ${MODULE_SRC})
        set_target_properties(${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_NAME ${MODULE_OUTPUT_NAME})
        target_link_libraries(${MODULE_NAME} PRIVATE Mappel)
        target_link_libraries(${MODULE_NAME} PRIVATE -Wl,--no-undefined)
        target_link_libraries(${MODULE_NAME} PUBLIC ${PYTHON_LIBRARY})
        target_include_directories(${MODULE_NAME} PUBLIC ${PYTHON_INCLUDE_DIRS} )
        set_target_properties(${MODULE_NAME} PROPERTIES DEBUG_POSTFIX "")
        set_target_properties(${MODULE_NAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/..:\$ORIGIN/../..")

        install(TARGETS ${MODULE_NAME}
                LIBRARY DESTINATION lib/Mappel/python COMPONENT Runtime)        
    endforeach()
endforeach()




