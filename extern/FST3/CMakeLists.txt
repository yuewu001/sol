cmake_minimum_required(VERSION 2.8)

project(FST3)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set (ARCHIVE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

if (WIN32)
    if (CMAKE_GENERATOR MATCHES "Win64")
        set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/x64)
    else()
        set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/x86)
    endif()
else()
    set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/linux)
endif()

if (NOT DEFINED BOOST_DIR)
    message(FATAL_ERROR "Boost Dir is not specified!")
endif (NOT DEFINED BOOST_DIR)

FILE(GLOB src_criteria _src_criteria/*.hpp)
source_group("src_criteria" FILES ${src_cirteria})

FILE(GLOB src_dataio _src_dataio/*.hpp)
source_group("src_dataio" FILES ${src_dataio})

FILE(GLOB src_global _src_global/*.hpp)
source_group("src_global" FILES ${src_global})

FILE(GLOB src_search _src_search/*.hpp)
source_group("src_search" FILES ${src_search})

set (libsvm
    libsvm/svm.h
    libsvm/svm.cpp
    )
source_group("libsvm" FILES ${libsvm})

include_directories(
    ${BOOST_DIR}
    _src_criteria
    _src_dataio
    _src_global
    _src_search
    libsvm
    )
link_directories(
    D:/Programs/boost/stage/lib
    )

add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES)

#FILE(GLOB demo_files demo*.cpp)
#FOREACH(demo_file ${demo_files})
#    STRING(REGEX REPLACE "(.*)/(demo[0-9]+[a-z]*)\\.cpp"  "\\2" project_name  ${demo_file})
#    STRING(REGEX REPLACE "(.*)/(demo[0-9]+)([a-z]*)\\.cpp" "\\3" is_thread ${demo_file})
#
#    if (is_thread STREQUAL "")
#        add_executable(${project_name} ${demo_file} ${libsvm})
#
#        install(TARGETS ${project_name}
#            RUNTIME DESTINATION .
#            )
#    endif()
#ENDFOREACH(demo_file ${demo_files})

add_executable(BIF demo31.cpp util.h ${libsvm})
#add_executable(WrapperFloating demo11.cpp util.h ${libsvm})

install(TARGETS BIF 
    #WrapperFloating
    RUNTIME DESTINATION .
    )

