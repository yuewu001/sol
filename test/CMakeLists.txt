cmake_minimum_required(VERSION 2.8)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/io
    )
link_directories(
    )

add_executable(testDll testDll/testDll.cpp) 
IF ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug") 
	target_link_libraries(testDll SOLdlld)
ELSE()
	target_link_libraries(testDll SOLdll)
ENDIF()


add_executable(testStatic testStatic/testStatic.cpp)
IF ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug") 
	target_link_libraries(testStatic SOLstaticd)
ELSE()
	target_link_libraries(testStatic SOLstatic)
ENDIF()

add_executable(testMNIST testMNIST/testMNIST.cpp testMNIST/MNISTReader.h) 
IF ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug") 
	target_link_libraries(testMNIST SOLstaticd)
ELSE()
	target_link_libraries(testMNIST SOLstatic)
ENDIF()


install(TARGETS testDll testStatic testMNIST
    RUNTIME DESTINATION test
    LIBRARY DESTINATION test
    ARCHIVE DESTINATION lib)


