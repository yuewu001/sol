cmake_minimum_required(VERSION 2.8)

#shoud be placed before SOL for work
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Installation Directory")

project(SOL)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set (ARCHIVE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set (CMAKE_MODULE_PATH  ${CMAKE_SOURCE_DIR}/cmake)

#RPATH and library search setting
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(io_files 
    src/io/DataPoint.h
    src/io/DataReader.h
    src/io/DataSet.h
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
    src/algorithms/AROW_FS.h
    src/algorithms/Ada_FOBOS.h
    src/algorithms/Ada_RDA.h
    src/algorithms/AROW_DA.h
    src/algorithms/DAROW.h
    src/algorithms/FOBOS.h
    src/algorithms/HeapList.h
    src/algorithms/OFSGD.h
    src/algorithms/Optimizer.h
    src/algorithms/RDA_L1.h
    src/algorithms/SCW.h
    src/algorithms/SCW_RDA.h
    src/algorithms/SGD.h
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

IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -O2 -s")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

add_executable(SOL src/Params.cpp src/Params.h 
    src/main.cpp src/SOL_interface.cpp ${SRC_LIST}) 
IF(UNIX)
    target_link_libraries(SOL pthread)
ENDIF(UNIX)

add_library(SOLdll SHARED src/SOL_interface.h src/Params.cpp src/Params.h 
    src/SOL_interface.cpp src/SOLdll.h ${SRC_LIST}) 
IF(UNIX)
    target_link_libraries(SOLdll pthread)
    set(CMAKE_CXX_FLAGS ${CMAKE_CSS_FLAGS} "-fPIC") 
ENDIF(UNIX)

add_library(SOLstatic STATIC src/SOL_interface.h src/Params.cpp src/Params.h 
    src/SOL_interface.cpp src/SOLstatic.h ${SRC_LIST}) 
IF(UNIX)
    target_link_libraries(SOLstatic pthread)
ENDIF(UNIX)

add_executable(analysis src/analysis/data_analysis.cpp ${io_files})
add_executable(Converter ${converter_files} ${io_files})
IF(UNIX)
    target_link_libraries(Converter pthread)
ENDIF(UNIX)

#Organize projects into folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(test)

SET_PROPERTY(TARGET analysis PROPERTY FOLDER "Tools")
SET_PROPERTY(TARGET Converter PROPERTY FOLDER "Tools")
SET_PROPERTY(TARGET testDll PROPERTY FOLDER "TestCases")
SET_PROPERTY(TARGET testStatic PROPERTY FOLDER "TestCases")
SET_PROPERTY(TARGET testMNIST PROPERTY FOLDER "TestCases")

install(TARGETS SOL analysis Converter SOLdll SOLstatic
    RUNTIME DESTINATION .
    LIBRARY DESTINATION .
    ARCHIVE DESTINATION lib)

install (FILES 
    src/io/DataPoint.h
    src/io/io_interface.h
    src/io/DataReader.h
    src/loss/LossFunction.h
    src/algorithms/Optimizer.h
    DESTINATION include)
