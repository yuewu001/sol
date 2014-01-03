cmake_minimum_required(VERSION 2.8)


include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/io
    )
link_directories(
    )

add_executable(testDll testDll/testDll.cpp) 
target_link_libraries(testDll SOLdll)

add_executable(testStatic testStatic/testStatic.cpp) 
target_link_libraries(testStatic SOLstatic)

add_executable(testMNIST testMNIST/testMNIST.cpp testMNIST/MNISTReader.h) 
target_link_libraries(testMNIST SOLstatic)

install(TARGETS testDll testStatic testMNIST
    RUNTIME DESTINATION test
    LIBRARY DESTINATION test
    ARCHIVE DESTINATION lib)


