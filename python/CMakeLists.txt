



set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(SETUP_CFG_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.cfg.in")
set(SETUP_CFG    "${CMAKE_CURRENT_BINARY_DIR}/setup.cfg")
configure_file(${SETUP_PY_IN} ${SETUP_PY} @ONLY)
configure_file(${SETUP_CFG_IN} ${SETUP_CFG} @ONLY)

add_custom_target(prepare_python ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/setup.py")

# install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install –prefix=${CMAKE_INSTALL_PREFIX})")
add_custom_command(TARGET prepare_python PRE_BUILD
   COMMAND ${CMAKE_COMMAND} -E create_symlink "${CMAKE_CURRENT_SOURCE_DIR}/mappel" "${CMAKE_CURRENT_BINARY_DIR}/mappel" 
   COMMENT "Symlinking mappel python code")

