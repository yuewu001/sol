cmake_minimum_required(VERSION 2.8)

project(SOL)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
#ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)


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
    src/io/parser.h
    src/io/s_array.h)

set(loss_files
    src/loss/HingeLoss.h
    src/loss/LogisticLoss.h
    src/loss/LossFunction.h
    src/loss/SquareLoss.h
    src/loss/SquaredHingeLoss.h)

set(algo_files
    src/algorithms/ASAROW.h
    src/algorithms/Ada_FOBOS.h
    src/algorithms/Ada_RDA.h
    src/algorithms/CW_RDA.h
    src/algorithms/DAROW.h
    src/algorithms/FOBOS.h
    src/algorithms/HeapList.h
    src/algorithms/OFSGD.h
    src/algorithms/Optimizer.h
    src/algorithms/RDA_L1.h
    src/algorithms/SCW.h
    src/algorithms/SCW_RDA.h
    src/algorithms/SGD.h
    src/algorithms/SSAROW.h
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
    src/converter/Params.h
    src/io/mnist/MNISTReader.h)

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
    src/main.cpp ${SRC_LIST}) 
IF(UNIX)
    target_link_libraries(SOL pthread)
ENDIF(UNIX)

add_library(SOLdll SHARED src/SOL_interface.h src/Params.cpp src/Params.h 
    src/main.cpp ${SRC_LIST}) 
IF(UNIX)
    target_link_libraries(SOLdll pthread)
ENDIF(UNIX)

add_library(SOLstatic STATIC src/SOL_interface.h src/Params.cpp src/Params.h 
    src/main.cpp ${SRC_LIST}) 
IF(UNIX)
    target_link_libraries(SOLstatic pthread)
ENDIF(UNIX)



add_executable(analysis src/analysis/data_analysis.cpp ${io_files})
add_executable(Converter ${converter_files} ${io_files})
IF(UNIX)
    target_link_libraries(Converter pthread)
ENDIF(UNIX)

install(TARGETS SOL analysis Converter SOLdll SOLstatic
    RUNTIME DESTINATION .
    LIBRARY DESTINATION bin
    ARCHIVE DESTINATION lib)
