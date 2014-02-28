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

set(io_files 
    src/io/DataPoint.h
    src/io/MPBuffer.h
    src/io/DataReader.h
    src/io/DataHandler.h
    src/io/DataSet.h
    src/io/MPDataSet.h
    src/io/DataSetHelper.h
    src/io/basic_io.cpp
    src/io/basic_io.h
    src/io/comp.h
    #src/io/gzip_io.cpp
    #src/io/gzip_io.h
    #src/io/zlib_io.cpp
    #src/io/zlib_io.h
    src/io/io_interface.h
    src/io/libsvm_binary.h
    src/io/libsvm_io.h
    src/io/csv_io.h
    src/io/sol_io.h
    src/io/parser.h
    src/io/s_array.h)

set(loss_files
    src/loss/sol_loss.h
    src/loss/HingeLoss.h
    src/loss/LogisticLoss.h
    src/loss/LossFunction.h
    src/loss/SquareLoss.h
    src/loss/SquaredHingeLoss.h)

set(algo_files
    src/algorithms/sol_algorithms.h
    src/algorithms/SOFS.h
    src/algorithms/AROW_AdaFS.h
    src/algorithms/Ada_FOBOS.h
    src/algorithms/Ada_RDA.h
    src/algorithms/AROW_DA.h
    src/algorithms/DAROW.h
    src/algorithms/FOBOS.h
    src/algorithms/MaxHeap.h
    src/algorithms/MinHeap.h
    src/algorithms/FOFS.h
    src/algorithms/Optimizer.h
    src/algorithms/RDA_L1.h
    src/algorithms/SCW.h
    src/algorithms/SCW_RDA.h
    src/algorithms/SGD.h
	src/algorithms/PET.h
    src/algorithms/mRMR_OGD.h
    src/algorithms/AROW_TG.h
    src/algorithms/STG.h)

set(util_files 
    src/utils/ezOptionParser.hpp
    src/utils/init_param.h
    src/utils/md5.h
    src/utils/thread_primitive.h
    src/utils/util.h)

set(converter_files 
    src/converter/Converter.cpp
    src/converter/Converter.h
    src/converter/Params.cpp
    src/converter/Params.h)

source_group("io" FILES ${io_files})
source_group("loss" FILES ${loss_files})
source_group("algorithms" FILES ${algo_files})
source_group("utils" FILES ${util_files})

set (SRC_LIST
    ${io_files} ${loss_files} ${algo_files} ${util_files}
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

add_executable(SOL src/Params.cpp src/Params.h 
    src/main.cpp src/SOL_interface.cpp ${SRC_LIST})
IF(UNIX)
    target_link_libraries(SOL pthread)
ENDIF(UNIX)

add_library(SOLdll SHARED src/SOL_interface.h src/Params.cpp src/Params.h 
    src/SOL_interface.cpp  ${SRC_LIST}) 
IF(UNIX)
    target_link_libraries(SOLdll pthread)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") 
ENDIF(UNIX)

add_library(SOLstatic STATIC src/SOL_interface.h src/Params.cpp src/Params.h 
    src/SOL_interface.cpp ${SRC_LIST}) 
IF(UNIX)
    target_link_libraries(SOLstatic pthread)
ENDIF(UNIX)

add_executable(analysis src/analysis/data_analysis.cpp ${io_files})
add_executable(dtcleaner src/dtcleaner/dtcleaner.cpp ${io_files} ${util_files})
add_executable(Converter ${converter_files} ${io_files} ${util_files})
IF(UNIX)
    target_link_libraries(Converter pthread)
ENDIF(UNIX)


IF(DCMTK_WITH_DEBUG_POSTFIX)  
  set_target_properties(SOL PROPERTIES DEBUG_POSTFIX "d")
  #set_target_properties(SyntheticData PROPERTIES DEBUG_POSTFIX "d")
  set_target_properties(analysis PROPERTIES DEBUG_POSTFIX "d")
  set_target_properties(Converter PROPERTIES DEBUG_POSTFIX "d")
ENDIF(DCMTK_WITH_DEBUG_POSTFIX)

#Organize projects into folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(test)

SET_PROPERTY(TARGET analysis PROPERTY FOLDER "Tools")
SET_PROPERTY(TARGET dtcleaner PROPERTY FOLDER "Tools")
SET_PROPERTY(TARGET Converter PROPERTY FOLDER "Tools")
SET_PROPERTY(TARGET testDll PROPERTY FOLDER "TestCases")
SET_PROPERTY(TARGET testStatic PROPERTY FOLDER "TestCases")
SET_PROPERTY(TARGET testMNIST PROPERTY FOLDER "TestCases")

install(TARGETS SOL analysis dtcleaner Converter SOLdll SOLstatic
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib)
if (WIN32)
    install(TARGETS SOLdll 
        RUNTIME DESTINATION test
        LIBRARY DESTINATION test
        ARCHIVE DESTINATION lib)
endif()

install (FILES 
    src/io/DataPoint.h
    src/io/io_interface.h
    src/io/DataReader.h
    src/loss/LossFunction.h
    src/algorithms/Optimizer.h
    src/SOL_interface.h
    DESTINATION include)

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
