cmake_minimum_required(VERSION 2.8)

#shoud be placed before SOL for work
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Installation Directory")

project(SOL)

set(CMAKE_VERBOSE_MAKEFILE ON) 

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (ARCHIVE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set (CMAKE_MODULE_PATH  ${CMAKE_SOURCE_DIR}/cmake)

#RPATH and library search setting
#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
#SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(src)
#LIST(APPEND all_algo_files ${solm_files} ${ofs_files} ${olm_files} ${om_files} ${algo_files})
LIST(APPEND all_algo_files ${solm_files} ${olm_files} ${om_files} ${algo_files})

#grouping the files
if (WIN32)
    source_group("io" FILES ${io_files})
    source_group("loss" FILES ${loss_files})
    source_group("utils" FILES ${util_files})
    source_group("optimizers" FILES ${opti_files})

    source_group("algorithms" FILES ${algo_files})
    source_group("algorithms\\om" FILES ${om_files})
    source_group("algorithms\\om\\olm" FILES ${olm_files})
    source_group("algorithms\\om\\olm\\ofs" FILES ${ofs_files})
    source_group("algorithms\\om\\olm\\solm" FILES ${solm_files})
ENDIF(WIN32)

set (SRC_LIST
    ${io_files} ${loss_files} ${all_algo_files} ${opti_files} ${util_files}
    )


IF (WIN32 AND MSVC AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set (CMAKE_BUILD_TYPE "Debug|Release")	
ENDIF()

IF (CMAKE_COMPILER_IS_GNUCXX AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

IF(WIN32)
    OPTION(DCMTK_WITH_DEBUG_POSTFIX "Use postfix 'd' for debug libraries (e.g. Release dcmsr.lib, Debug dcmsrd.lib)" ON)
ENDIF()

IF(DCMTK_WITH_DEBUG_POSTFIX)
    SET(CMAKE_DEBUG_POSTFIX  "d")  
ENDIF(DCMTK_WITH_DEBUG_POSTFIX)

add_executable(SOL src/BOC.h src/main.cpp ${SRC_LIST})
IF(UNIX)
    target_link_libraries(SOL pthread)
ENDIF(UNIX)

#add_library(SOLdll SHARED src/SOL_interface.h src/Params.cpp src/Params.h 
#    src/SOL_interface.cpp  ${SRC_LIST}) 
#IF(UNIX)
#    target_link_libraries(SOLdll pthread)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") 
#ENDIF(UNIX)
#
#add_library(SOLstatic STATIC src/SOL_interface.h src/Params.cpp src/Params.h 
#    src/SOL_interface.cpp ${SRC_LIST}) 
#IF(UNIX)
#    target_link_libraries(SOLstatic pthread)
#ENDIF(UNIX)
#
add_executable(data_analysis src/analysis/data_analysis.cpp ${io_files} ${util_files})
add_executable(dtcleaner src/dtcleaner/dtcleaner.cpp ${io_files} ${util_files})
add_executable(Converter ${converter_files} ${io_files} ${util_files})
IF(UNIX)
    target_link_libraries(Converter pthread)
ENDIF(UNIX)


IF(DCMTK_WITH_DEBUG_POSTFIX)  
    set_target_properties(SOL PROPERTIES DEBUG_POSTFIX "d")
    #set_target_properties(SyntheticData PROPERTIES DEBUG_POSTFIX "d")
    set_target_properties(data_analysis PROPERTIES DEBUG_POSTFIX "d")
    set_target_properties(Converter PROPERTIES DEBUG_POSTFIX "d")
    set_target_properties(dtcleaner PROPERTIES DEBUG_POSTFIX "d")
ENDIF(DCMTK_WITH_DEBUG_POSTFIX)

#Organize projects into folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

#add_subdirectory(test)

SET_PROPERTY(TARGET data_analysis PROPERTY FOLDER "Tools")
SET_PROPERTY(TARGET dtcleaner PROPERTY FOLDER "Tools")
SET_PROPERTY(TARGET Converter PROPERTY FOLDER "Tools")
#SET_PROPERTY(TARGET testDll PROPERTY FOLDER "TestCases")
#SET_PROPERTY(TARGET testStatic PROPERTY FOLDER "TestCases")
#SET_PROPERTY(TARGET testMNIST PROPERTY FOLDER "TestCases")

#install(TARGETS SOL analysis dtcleaner Converter SOLdll SOLstatic
install(TARGETS SOL  Converter data_analysis dtcleaner
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib)
#if (WIN32)
#    install(TARGETS SOLdll 
#        RUNTIME DESTINATION test
#        LIBRARY DESTINATION test
#        ARCHIVE DESTINATION lib)
#endif()

#install (FILES 
#    src/io/DataPoint.h
#    src/io/io_interface.h
#    src/io/DataReader.h
#    src/loss/LossFunction.h
#    src/algorithms/Optimizer.h
#    src/SOL_interface.h
#    DESTINATION include)

# ========================== build platform ==========================
message(STATUS "")
message(STATUS "  Platform:")
message(STATUS "    Host:"             ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR})
if(CMAKE_CROSSCOMPILING)
    message(STATUS "    Target:"         ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
endif()
message(STATUS "    CMake:"            ${CMAKE_VERSION})
message(STATUS "    CMake generator:"  ${CMAKE_GENERATOR})
message(STATUS "    CMake build tool:" ${CMAKE_BUILD_TOOL})
if(MSVC)
    message(STATUS "    MSVC:"           ${MSVC_VERSION})
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
    message(STATUS "    Xcode:"          ${XCODE_VERSION})
endif()
if(NOT CMAKE_GENERATOR MATCHES "Xcode")
    message(STATUS "    Configuration:"  ${CMAKE_BUILD_TYPE})
endif()

message(STATUS "")
message(STATUS "  C/C++:")
message(STATUS "    C++ Compiler:"           ${OPENCV_COMPILER_STR})
message(STATUS "    C++ flags: "    ${CMAKE_CXX_FLAGS})
message(STATUS "    C++ flags (Release):"    ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "    C++ flags (Debug):"       ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "    C flags (Release):"       ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "    C flags (Debug):"         ${CMAKE_C_FLAGS_DEBUG})
if(NOT WIN32)  
    message(STATUS "    Linker flags (Release):"  ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
    message(STATUS "    Linker flags (Debug):"    ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})
endif()
