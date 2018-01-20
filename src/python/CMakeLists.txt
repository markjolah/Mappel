# Mappel - src/python
#
# Python module building.
# We are using boost::Python
#

# add_subdirectory(pybind11)

set(PYTHON_SRCS MappelPythonModule.cpp)
set(MAPPEL_PY_MODULE mappel)

python_add_module(${MAPPEL_PY_MODULE} ${PYTHON_SRCS})
target_link_libraries(${MAPPEL_PY_MODULE} PUBLIC Mappel)

# target_link_libraries(${MAPPEL_PY_MODULE} PUBLIC Boost::python)

target_include_directories(${MAPPEL_PY_MODULE} PUBLIC ${PYTHON_INCLUDE_DIRS} ${BOOST_INCLUDE_DIRS})
target_link_libraries(${MAPPEL_PY_MODULE} PUBLIC ${PYTHON_LIBRARY})
target_link_libraries(${MAPPEL_PY_MODULE} PUBLIC ${Boost_PYTHON_LIBRARIES})

set_target_properties(${MAPPEL_PY_MODULE} PROPERTIES DEBUG_POSTFIX "")
set_target_properties(${MAPPEL_PY_MODULE} PROPERTIES INSTALL_RPATH "\$ORIGIN/..:\$ORIGIN/../..")
install(TARGETS ${MAPPEL_PY_MODULE}
            LIBRARY DESTINATION lib/Mappel/python COMPONENT Runtime)
