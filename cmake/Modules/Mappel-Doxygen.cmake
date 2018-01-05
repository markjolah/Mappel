# Mappel CMake build system
# Mark J. Olah (mjo@cs.unm DOT edu)
# 01-2018
# Doxygen integration

option(DOC "Generate Doxygen Documentation" OFF)

if(DOC)
    find_package(LATEX)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        #Configure the doxyfile for the build directory
        set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in)
        set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile)
        set(DOXYGEN_PDF_DIR ${CMAKE_CURRENT_BINARY_DIR}/doc/pdf)
        set(DOXYGEN_HTML_DIR ${CMAKE_CURRENT_BINARY_DIR}/doc/html)
        set(DOXYGEN_PDF_NAME refman.pdf)
        set(PDF_DOC_NAME ${PROJECT_NAME}-${PROJECT_VERSION}-reference.pdf)
        set(DOC_INSTALL_DIR share/${PROJECT_NAME}/doc)
        file(GLOB SRC_FILES src/*.h src/*.cpp)
        configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)
        add_custom_target( doc VERBATIM
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
            DEPENDS ${DOXYFILE_OUT} ${SRC_FILES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
            COMMENT "Generate Doxygen Documentation")
        add_custom_target( pdf VERBATIM
            COMMAND make ${DOXYGEN_PDF_NAME}
            COMMAND mv ${DOXYGEN_PDF_NAME} ${PDF_DOC_NAME}
            DEPENDS ${DOXYGEN_PDF_DIR}/refman.tex ${DOXYGEN_PDF_DIR}/Makefile
            WORKING_DIRECTORY ${DOXYGEN_PDF_DIR}
            COMMENT "Build pdf documentation")
        add_dependencies(pdf doc)
        install(FILES ${DOXYGEN_PDF_DIR}/${PDF_DOC_NAME} DESTINATION ${DOC_INSTALL_DIR} COMPONENT Development)
        install(DIRECTORY ${DOXYGEN_HTML_DIR} DESTINATION ${DOC_INSTALL_DIR} COMPONENT Development)
    else()
        message(WARNING "Doxygen not found. Cannot build documentation.")
    endif()
endif()
