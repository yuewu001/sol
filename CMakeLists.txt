cmake_minimum_required(VERSION 2.8)

project(SOL)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
#ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)


FILE(GLOB io_files src/io/*.h)
FILE(GLOB loss_files src/loss/*.h)
FILE(GLOB algo_files src/algorithms/*.h)
FILE(GLOB util_files src/utils/*.h src/utils/*.hpp) 

set (io_files ${io_files} 
    src/io/basic_io.cpp 
    )

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

add_executable(analysis src/io/data_analysis.cpp ${io_files})
add_executable(Cacher src/io/Cacher.cpp ${io_files})
IF(UNIX)
    target_link_libraries(Cacher pthread)
ENDIF(UNIX)

add_executable(MNISTConvert src/io/MNISTConvert.cpp)

install(TARGETS SOL analysis Cacher MNISTConvert
    DESTINATION .)
