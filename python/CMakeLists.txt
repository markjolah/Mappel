#
# Mappel python wrapper build system
# Mark J. Olah
# mjo@cs.unm DOT edu
# 2018
#
# python bindings are only an optional component of Mappel, but the easiest way to manipulate and
# install python extension modules is with python setuptools, providing a setup.py and setup.cfg.
# This enables use of the setup.py script commands or the pip tool.
# 
# The setup.py develop mode and other build tools modify the local directory they operate from,
# we move these modifications to the build tree, in the build_dir/python directory, which becomes
# the central location for the setup.py and associated code.
# 

#Central location (under build directory) where the setup.py build directory will be located
set(PYTHON_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

# setup.py.in: We don't need @'substitution, but also don't want users to run this setup.py.in where it exists in the 
# repository, so we keep the .in suffix
configure_file("setup.py.in" "${PYTHON_BUILD_DIR}/setup.py" COPYONLY) 
configure_file("setup.cfg.in" "${PYTHON_BUILD_DIR}/setup.cfg" @ONLY)
# configure_file("PYTHON_README.md" "PYTHON_README.md" COPY_ONLY)
configure_file("../LICENSE" "${PYTHON_BUILD_DIR}/LICENSE" COPYONLY)

add_custom_target(prepare_python ALL DEPENDS "${PYTHON_BUILD_DIR}/setup.py")

#symlink the mappel python code into build_dir/python
add_custom_command(TARGET prepare_python PRE_BUILD
   COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_CURRENT_SOURCE_DIR}/mappel" "${PYTHON_BUILD_DIR}/mappel" 
   COMMENT "Symlinking mappel python code")

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} setup.py install –prefix=${CMAKE_INSTALL_PREFIX} 
                              WORKING_DIRECTORY ${PYTHON_BUILD_DIR}")
