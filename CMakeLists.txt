cmake_minimum_required(VERSION 2.8)

project(SOL)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
#ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)


FILE(GLOB data_files src/data/*.h)
FILE(GLOB loss_files src/loss/*.h)
FILE(GLOB opti_files src/optimizer/*.h)
FILE(GLOB comm_files src/common/*.h)

set (data_files ${data_files} 
    src/data/basic_io.cpp 
    )

#set (data_files ${datafiles} 
#    src/data/basic_io.cpp 
#    src/data/zlib_io.cpp 
#    src/data/gzip_io.cpp
#    )
source_group("data" FILES ${data_files})
source_group("loss" FILES ${loss_files})
source_group("optimizer" FILES ${opti_files})
source_group("common" FILES ${comm_files})

set (SRC_LIST
    ${data_files} ${loss_files} ${opti_files} ${comm_files}
    )

IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -O2 -s")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)
    
add_executable(SOL src/Params.cpp src/Params.h 
    src/main.cpp ${SRC_LIST}) 
IF(UNIX)
    target_link_libraries(SOL pthread)
ENDIF(UNIX)

add_executable(analysis src/data/data_analysis.cpp ${data_files})
add_executable(Cacher src/data/Cacher.cpp ${data_files})
IF(UNIX)
    target_link_libraries(Cacher pthread)
ENDIF(UNIX)

add_executable(MNISTConvert src/data/MNISTConvert.cpp)

install(TARGETS SOL analysis Cacher MNISTConvert
    DESTINATION .)
